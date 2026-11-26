import Foundation
import simd

extension PictureWorldRenderer {
    // MARK: - ARDY Room Locomotion & Anti-Stick Boundary Steering

    func resetARDYRootTracking() {
        lastARDYRootXZ = nil
    }

    /// World Y of the floor at a room position. Zero for hand-authored scenes without a
    /// fitted ground surface; the analyser's quadratic otherwise.
    func groundHeight(atX x: Float, z: Float) -> Float {
        roomGrid?.groundHeight(x: x, z: z) ?? 0
    }

    /// Integrates continuous ARDY root XZ displacement (meters) onto the Picture-World floor while
    /// enforcing walkBounds, unzoomed frustum visibility, and walkableGrid furniture footprints.
    func applyARDYRootMotion(
        worldRootXZ: SIMD2<Float>,
        viewportAspect: Float,
        goalTargetXZ: SIMD2<Float>? = nil
    ) {
        guard isActive else { return }
        guard let prev = lastARDYRootXZ else {
            lastARDYRootXZ = worldRootXZ
            return
        }
        let rawDelta = worldRootXZ - prev
        lastARDYRootXZ = worldRootXZ

        let stepLen = simd_length(rawDelta)
        guard stepLen > 0.0001 && stepLen < 0.35 else { return }

        func rotateStep(_ yaw: Float) -> SIMD2<Float> {
            let cy = cos(yaw)
            let sy = sin(yaw)
            return SIMD2<Float>(rawDelta.x * cy + rawDelta.y * sy, -rawDelta.x * sy + rawDelta.y * cy)
        }

        if let goal = goalTargetXZ {
            let toGoal = goal - SIMD2<Float>(avatarWorldPosition.x, avatarWorldPosition.z)
            let distToGoal = simd_length(toGoal)
            if distToGoal > 0.03 {
                let desiredDir = toGoal / distToGoal
                let currentStep = rotateStep(avatarWorldYaw)
                let stepDir = currentStep / max(stepLen, 1e-5)
                let curAngle = atan2(stepDir.x, stepDir.y)
                let targetAngle = atan2(desiredDir.x, desiredDir.y)
                var angleDiff = targetAngle - curAngle
                while angleDiff > .pi { angleDiff -= 2 * .pi }
                while angleDiff < -.pi { angleDiff += 2 * .pi }

                let maxTurn = min(0.28, stepLen * 6.5 + 0.05)
                let turnStep = min(max(angleDiff, -maxTurn), maxTurn)
                avatarWorldYaw += turnStep
                if avatarWorldYaw > .pi { avatarWorldYaw -= 2 * .pi }
                if avatarWorldYaw < -.pi { avatarWorldYaw += 2 * .pi }
            }
        }

        var worldStep = rotateStep(avatarWorldYaw)
        if let goal = goalTargetXZ {
            let toGoal = goal - SIMD2<Float>(avatarWorldPosition.x, avatarWorldPosition.z)
            let dist = simd_length(toGoal)
            if dist < stepLen && dist > 1e-4 {
                worldStep = toGoal
            }
        }
        let candidate = SIMD3<Float>(avatarWorldPosition.x + worldStep.x, 0.0, avatarWorldPosition.z + worldStep.y)
        let eval = clampAvatarPositionWithContacts(candidate, viewportAspect: viewportAspect)

        if goalTargetXZ == nil, (!eval.contacts.isEmpty || eval.isNearBoundary), simd_length(eval.inwardNormal) > 0.01 {
            let stepDir = worldStep / max(stepLen, 1e-5)
            let inwardDot = simd_dot(stepDir, eval.inwardNormal)
            if !eval.contacts.isEmpty || inwardDot < 0.25 {
                let desiredDir = simd_normalize(eval.inwardNormal)
                let curAngle = atan2(stepDir.x, stepDir.y)
                let targetAngle = atan2(desiredDir.x, desiredDir.y)
                var angleDiff = targetAngle - curAngle
                while angleDiff > .pi { angleDiff -= 2 * .pi }
                while angleDiff < -.pi { angleDiff += 2 * .pi }

                let urgency: Float = eval.contacts.isEmpty ? 0.65 : 1.0
                let maxTurn = min(0.22, (stepLen * 5.2 + 0.03) * urgency)
                let turnStep = min(max(angleDiff, -maxTurn), maxTurn)
                avatarWorldYaw += turnStep
                if avatarWorldYaw > .pi { avatarWorldYaw -= 2 * .pi }
                if avatarWorldYaw < -.pi { avatarWorldYaw += 2 * .pi }

                worldStep = rotateStep(avatarWorldYaw)
                let steeredCandidate = SIMD3<Float>(avatarWorldPosition.x + worldStep.x, 0.0, avatarWorldPosition.z + worldStep.y)
                let steeredEval = clampAvatarPositionWithContacts(steeredCandidate, viewportAspect: viewportAspect)
                avatarWorldPosition = steeredEval.position
                lastBoundaryContacts = steeredEval.contacts
                lastIsNearBoundary = steeredEval.isNearBoundary
                return
            }
        }

        avatarWorldPosition = eval.position
        lastBoundaryContacts = eval.contacts
        lastIsNearBoundary = eval.isNearBoundary
    }

    /// Evaluates room bounds + unzoomed camera frustum guard + walkable grid footprints.
    func clampAvatarPositionWithContacts(
        _ pos: SIMD3<Float>,
        viewportAspect: Float
    ) -> (position: SIMD3<Float>, contacts: [String], isNearBoundary: Bool, inwardNormal: SIMD2<Float>) {
        guard let cfg = config else {
            return (pos, [], false, .zero)
        }
        var p = SIMD3<Float>(pos.x, 0.0, pos.z)
        var contacts = Set<String>()
        var isNear = false
        var inward = SIMD2<Float>.zero
        let nearMargin: Float = 0.22

        // 1. Scene floor walk bounds
        if let minB = cfg.walkBoundsMin, minB.count == 2,
           let maxB = cfg.walkBoundsMax, maxB.count == 2 {
            if p.x <= minB[0] + 0.005 {
                p.x = minB[0]
                contacts.insert("left")
                inward.x += 1.0
            } else if p.x < minB[0] + nearMargin {
                isNear = true
                inward.x += 0.6
            }
            if p.x >= maxB[0] - 0.005 {
                p.x = maxB[0]
                contacts.insert("right")
                inward.x -= 1.0
            } else if p.x > maxB[0] - nearMargin {
                isNear = true
                inward.x -= 0.6
            }
            if p.z <= minB[1] + 0.005 {
                p.z = minB[1]
                contacts.insert("back")
                inward.y += 1.0
            } else if p.z < minB[1] + nearMargin {
                isNear = true
                inward.y += 0.6
            }
            if p.z >= maxB[1] - 0.005 {
                p.z = maxB[1]
                contacts.insert("front")
                inward.y -= 1.0
            } else if p.z > maxB[1] - nearMargin {
                isNear = true
                inward.y -= 0.6
            }
        }

        // 2. Unzoomed base camera frustum guard with exact orthonormal floor basis
        let baseUVScale = baseAspectFillUVScale(viewportAspect: viewportAspect)
        let imgTanHalfY = tan((cfg.fovYDegrees * .pi / 180.0) * 0.5)
        let imgAspect = Float(cfg.imageWidth) / max(Float(cfg.imageHeight), 1.0)
        let baseTanHalfX = imgTanHalfY * imgAspect * baseUVScale.x

        let camPos = cfg.camPos
        let fwd3 = simd_normalize(cfg.camTarget - camPos)
        let fwdH = simd_normalize(SIMD2<Float>(fwd3.x, fwd3.z))
        let rightH = SIMD2<Float>(-fwdH.y, fwdH.x)

        let relH = SIMD2<Float>(p.x - camPos.x, p.z - camPos.z)
        var depth = simd_dot(relH, fwdH)
        var lateral = simd_dot(relH, rightH)
        var frustumClamped = false

        let minSafeDepth: Float = 1.55
        if depth <= minSafeDepth + 0.005 {
            depth = minSafeDepth
            frustumClamped = true
            contacts.insert("front")
            inward += fwdH
        } else if depth < minSafeDepth + nearMargin {
            isNear = true
            inward += fwdH * 0.6
        }

        let maxLateral = max(0.36, depth * baseTanHalfX * 0.72)
        if lateral <= -maxLateral + 0.005 {
            lateral = -maxLateral
            frustumClamped = true
            contacts.insert("left")
            inward += rightH
        } else if lateral < -maxLateral + nearMargin {
            isNear = true
            inward += rightH * 0.6
        }
        if lateral >= maxLateral - 0.005 {
            lateral = maxLateral
            frustumClamped = true
            contacts.insert("right")
            inward -= rightH
        } else if lateral > maxLateral - nearMargin {
            isNear = true
            inward -= rightH * 0.6
        }

        if frustumClamped {
            let clampedH = SIMD2<Float>(camPos.x, camPos.z) + fwdH * depth + rightH * lateral
            p.x = clampedH.x
            p.z = clampedH.y
        }

        if let minB = cfg.walkBoundsMin, minB.count == 2,
           let maxB = cfg.walkBoundsMax, maxB.count == 2 {
            p.x = min(max(p.x, minB[0]), maxB[0])
            p.z = min(max(p.z, minB[1]), maxB[1])
        }

        // 3. Walkable floor grid (furniture footprints)
        if let grid = roomGrid, !grid.isEmpty, !grid.isWalkable(x: p.x, z: p.z) {
            if let snapped = grid.nearestWalkable(x: p.x, z: p.z, maxRadius: 12) {
                let push = SIMD2<Float>(snapped.x - p.x, snapped.y - p.z)
                if simd_length(push) > 1e-3 {
                    inward += simd_normalize(push) * 0.9
                }
                p.x = snapped.x
                p.z = snapped.y
                contacts.insert("furniture")
                isNear = true
            }
        }

        let toCenter = SIMD2<Float>(cfg.spawnPos.x - p.x, cfg.spawnPos.z - p.z)
        if simd_length(toCenter) > 0.05 {
            inward += simd_normalize(toCenter) * 0.45
        }
        let inwardNorm = simd_length(inward) > 1e-4 ? simd_normalize(inward) : .zero
        return (p, contacts.sorted(), isNear || !contacts.isEmpty, inwardNorm)
    }
}
