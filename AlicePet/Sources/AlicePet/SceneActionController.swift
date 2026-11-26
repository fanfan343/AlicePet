import Foundation
import simd

/// Orchestrates multi-step autonomous Picture-World behavior chains (e.g. walk with ARDY to a sofa,
/// align & turn outward, sit down smoothly via SLERP crossfade, and hold seated Idle until the next
/// user or LLM command).
@MainActor
final class SceneActionController {
    enum Phase: Equatable {
        case idle
        case walkingToAnchor(SceneAnchor)
        case aligningToAnchor(anchor: SceneAnchor, elapsed: TimeInterval, duration: TimeInterval, startPos: SIMD3<Float>, startYaw: Float)
        case seatedIdle(SceneAnchor)
    }

    private(set) var phase: Phase = .idle
    private var walkElapsed: TimeInterval = 0
    private var lastWalkPosXZ: SIMD2<Float>?
    private var stallElapsed: TimeInterval = 0

    var onPhaseChanged: (() -> Void)?

    var activeTaskIdentifier: String? {
        switch phase {
        case .idle: return nil
        case .walkingToAnchor(let a): return "walking_to:\(a.id)"
        case .aligningToAnchor(let a, _, _, _, _): return "aligning_to:\(a.id)"
        case .seatedIdle(let a): return "seated_idle:\(a.id)"
        }
    }

    var seatedAnchor: SceneAnchor? {
        if case .seatedIdle(let a) = phase { return a }
        return nil
    }

    /// Remaining waypoints of the A* route to the current anchor (empty = walk straight at it).
    private var walkPath: [SIMD2<Float>] = []

    var activeWalkGoalXZ: SIMD2<Float>? {
        if case .walkingToAnchor(let a) = phase {
            if let next = walkPath.first { return next }
            let p = a.approachPos
            return SIMD2<Float>(p.x, p.z)
        }
        return nil
    }

    /// Starts the autonomous chain: ARDY walks to `anchor.approachPosition` -> aligns onto seat ->
    /// sits down and holds seated Idle until the next command.
    @discardableResult
    func startWalkAndSit(anchor: SceneAnchor, on controller: PetController) -> Bool {
        guard controller.isPictureWorldActive else { return false }

        // If already seated at another spot, stand up onto its approach floor first
        if case .seatedIdle(let prev) = phase {
            if prev.id == anchor.id { return true }
            controller.pictureWorldRenderer.avatarWorldPosition = prev.approachPos
            controller.setPoseMode(.stand)
        }

        walkElapsed = 0
        stallElapsed = 0
        let startXZ = SIMD2<Float>(
            controller.pictureWorldRenderer.avatarWorldPosition.x,
            controller.pictureWorldRenderer.avatarWorldPosition.z
        )
        lastWalkPosXZ = startXZ

        // Route around furniture instead of walking through it. Falls back to a straight
        // line when the scene has no walkable grid (hand-authored scenes).
        let goalXZ = SIMD2<Float>(anchor.approachPos.x, anchor.approachPos.z)
        if let grid = controller.pictureWorldRenderer.roomGrid,
           let path = grid.findPath(from: startXZ, to: goalXZ) {
            walkPath = path
            AppLogger.shared.info(
                "SceneAction",
                "A* route to '\(anchor.label)': \(path.count) leg(s) \(path.map { "(\(String(format: "%.2f", $0.x)),\(String(format: "%.2f", $0.y)))" }.joined(separator: " -> "))"
            )
        } else {
            walkPath = []
        }

        // Start or redirect ARDY with a gentle forward-walking prompt
        let walkPrompt = "a petite gentle young girl walks slowly and gracefully forward with small light steps"
        controller.localArdyService.currentPrompt = walkPrompt
        if controller.localArdyActive {
            controller.localArdyService.sendLivePrompt(walkPrompt)
        } else {
            guard controller.startArdyMotionTest() else { return false }
        }

        phase = .walkingToAnchor(anchor)
        AppLogger.shared.info("SceneAction", "Started walk-and-sit chain -> '\(anchor.label)' (id=\(anchor.id), approach=\(anchor.approachPos), seat=\(anchor.targetPos))")
        onPhaseChanged?()
        return true
    }

    /// Exits seated Idle (standing up cleanly onto the floor in front of the seat) or cancels active walk.
    func standUpOrCancel(on controller: PetController, stopArdy: Bool = false) {
        let prevPhase = phase
        switch prevPhase {
        case .idle:
            return
        case .seatedIdle(let anchor):
            controller.pictureWorldRenderer.avatarWorldPosition = anchor.approachPos
            controller.setPoseMode(.stand)
            AppLogger.shared.info("SceneAction", "Stood up from '\(anchor.label)' -> floor \(anchor.approachPos)")
        case .walkingToAnchor(let anchor), .aligningToAnchor(let anchor, _, _, _, _):
            if stopArdy {
                controller.stopArdyMotionTest()
            }
            AppLogger.shared.info("SceneAction", "Cancelled walk-and-sit toward '\(anchor.label)'")
        }
        phase = .idle
        walkElapsed = 0
        stallElapsed = 0
        lastWalkPosXZ = nil
        walkPath.removeAll()
        onPhaseChanged?()
    }

    func reset() {
        phase = .idle
        walkElapsed = 0
        stallElapsed = 0
        lastWalkPosXZ = nil
        walkPath.removeAll()
    }

    /// Evaluates one frame of the active behavior chain inside `PetController.draw(in:)`.
    func tick(dt: TimeInterval, on controller: PetController, viewportAspect: Float) {
        guard controller.isPictureWorldActive else {
            if phase != .idle {
                reset()
                onPhaseChanged?()
            }
            return
        }

        let clampedDt = min(max(dt, 0.0), 0.1)

        switch phase {
        case .idle, .seatedIdle:
            return

        case .walkingToAnchor(let anchor):
            walkElapsed += clampedDt
            let curPos = controller.pictureWorldRenderer.avatarWorldPosition
            let curXZ = SIMD2<Float>(curPos.x, curPos.z)
            let finalGoalXZ = SIMD2<Float>(anchor.approachPos.x, anchor.approachPos.z)
            // Advance along the A* route: a waypoint counts as reached once she is on top of it.
            while let next = walkPath.first, simd_length(next - curXZ) <= 0.20 {
                walkPath.removeFirst()
            }
            let goalXZ = walkPath.first ?? finalGoalXZ
            let dist = simd_length(goalXZ - curXZ)

            // Fallback gentle locomotion assist only if ARDY takes >1.5s to start moving forward
            if let last = lastWalkPosXZ {
                let moved = simd_length(curXZ - last)
                if moved < 0.0005 {
                    stallElapsed += clampedDt
                } else {
                    stallElapsed = 0
                }
            }
            lastWalkPosXZ = curXZ

            if stallElapsed > 1.8 && controller.animationMixer.activeBodySource == .ardy && dist > 0.16 {
                // Keep natural slow girl walking pace (~0.48 m/s) along ARDY's stepping gait if ARDY cycles in place
                let dir = (goalXZ - curXZ) / max(dist, 1e-4)
                let step = min(dist, Float(clampedDt) * 0.48)
                controller.pictureWorldRenderer.avatarWorldPosition = SIMD3<Float>(curXZ.x + dir.x * step, 0.0, curXZ.y + dir.y * step)
                let desiredYaw = atan2(dir.x, dir.y)
                controller.pictureWorldRenderer.avatarWorldYaw = Self.slerpAngle(
                    controller.pictureWorldRenderer.avatarWorldYaw,
                    desiredYaw,
                    t: Float(min(1.0, clampedDt * 5.0))
                )
            }

            let updatedPos = controller.pictureWorldRenderer.avatarWorldPosition
            let updatedDist = simd_length(finalGoalXZ - SIMD2<Float>(updatedPos.x, updatedPos.z))
            if (walkPath.isEmpty && updatedDist <= 0.18) || walkElapsed > 20.0 {
                // Arrived right in front of the sofa -> transition to aligning & turning outward to sit
                phase = .aligningToAnchor(
                    anchor: anchor,
                    elapsed: 0,
                    duration: 0.70,
                    startPos: updatedPos,
                    startYaw: controller.pictureWorldRenderer.avatarWorldYaw
                )
                AppLogger.shared.info("SceneAction", "Reached '\(anchor.label)' approach (dist=\(String(format: "%.2fm", updatedDist))) -> aligning & sitting down")
                onPhaseChanged?()
            }

        case .aligningToAnchor(let anchor, let elapsed, let duration, let startPos, let startYaw):
            let nextElapsed = elapsed + clampedDt
            let rawT = Float(min(nextElapsed / max(duration, 0.05), 1.0))
            let smoothT = rawT * rawT * (3.0 - 2.0 * rawT)

            // Seated Y elevation, so Alice's thighs rest on the measured cushion. Measured
            // against the local floor rather than world Y=0, because the floor surface is
            // fitted (and slightly curved) — the root transform adds that back.
            let seatYOffset: Float = {
                guard let seatH = anchor.seatHeight else { return 0.0 }
                let localGround = controller.pictureWorldRenderer.groundHeight(
                    atX: anchor.targetPos.x, z: anchor.targetPos.z)
                // Author `sit` pose places the seated thigh underside ~0.42 m above the root.
                return min(max(seatH - localGround - 0.42, -0.15), 0.25)
            }()
            let finalSeatPos = SIMD3<Float>(anchor.targetPos.x, seatYOffset, anchor.targetPos.z)
            controller.pictureWorldRenderer.avatarWorldPosition = simd_mix(startPos, finalSeatPos, SIMD3<Float>(repeating: smoothT))
            controller.pictureWorldRenderer.avatarWorldYaw = Self.slerpAngle(startYaw, anchor.facingYawRad, t: smoothT)

            if rawT >= 0.55 && controller.poseMode != .sit {
                // Switch base Layer 0 to `.sit` and stop ARDY so AnimationMixer's 0.35s SLERP smoothly sits Alice down!
                controller.setPoseMode(.sit)
                controller.stopArdyMotionTest()
            }

            if rawT >= 1.0 {
                controller.pictureWorldRenderer.avatarWorldPosition = finalSeatPos
                controller.pictureWorldRenderer.avatarWorldYaw = anchor.facingYawRad
                if controller.poseMode != .sit {
                    controller.setPoseMode(.sit)
                }
                if controller.localArdyActive {
                    controller.stopArdyMotionTest()
                }
                phase = .seatedIdle(anchor)
                controller.applySeatedUprightPostureIfNeeded()
                controller.applyPictureWorldRootTransform()
                controller.settleSpringBonePhysics(warmupSteps: 30)
                AppLogger.shared.info("SceneAction", "Seated Idle at '\(anchor.label)' (pos=\(finalSeatPos), yaw=\(String(format: "%.1f°", anchor.facingYawRad * 180 / .pi))) — holding until next command")
                onPhaseChanged?()
            } else {
                phase = .aligningToAnchor(
                    anchor: anchor,
                    elapsed: nextElapsed,
                    duration: duration,
                    startPos: startPos,
                    startYaw: startYaw
                )
            }
        }
    }

    private static func slerpAngle(_ from: Float, _ to: Float, t: Float) -> Float {
        var diff = to - from
        while diff > .pi { diff -= 2 * .pi }
        while diff < -.pi { diff += 2 * .pi }
        var res = from + diff * t
        while res > .pi { res -= 2 * .pi }
        while res < -.pi { res += 2 * .pi }
        return res
    }
}
