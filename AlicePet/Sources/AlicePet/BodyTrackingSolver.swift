import Foundation
import VRMMetalKit
import simd

struct TrackedUpperBodyFrame: Sendable {
    let torsoYaw: Float
    let torsoPitch: Float
    let torsoRoll: Float
    let leftUpperArmDir: SIMD3<Float>?
    let leftForearmDir: SIMD3<Float>?
    let rightUpperArmDir: SIMD3<Float>?
    let rightForearmDir: SIMD3<Float>?
}

/// Smooths stable 2D-derived upper-body tracking data via low-cutoff 1€ filters + deadzones,
/// and applies it deterministically on top of the active base pose (zero per-frame feedback jitter).
@MainActor
final class BodyTrackingSolver {
    private var yawFilter = OneEuroFilterScalar(minCutoff: 0.45, beta: 0.03, dCutoff: 1.0)
    private var pitchFilter = OneEuroFilterScalar(minCutoff: 0.45, beta: 0.03, dCutoff: 1.0)
    private var rollFilter = OneEuroFilterScalar(minCutoff: 0.45, beta: 0.03, dCutoff: 1.0)

    private var leftUpperArmFilter = OneEuroFilterQuaternion(minCutoff: 0.65, beta: 0.04, dCutoff: 1.0)
    private var leftForearmFilter = OneEuroFilterQuaternion(minCutoff: 0.75, beta: 0.05, dCutoff: 1.0)
    private var rightUpperArmFilter = OneEuroFilterQuaternion(minCutoff: 0.65, beta: 0.04, dCutoff: 1.0)
    private var rightForearmFilter = OneEuroFilterQuaternion(minCutoff: 0.75, beta: 0.05, dCutoff: 1.0)

    private(set) var hasActiveBody = false
    private var smoothedTorsoYaw: Float = 0
    private var smoothedTorsoPitch: Float = 0
    private var smoothedTorsoRoll: Float = 0

    private var leftUpperArmRot = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var leftForearmRot = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var rightUpperArmRot = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var rightForearmRot = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

    private var leftArmVisibleStreak: Int = 0
    private var rightArmVisibleStreak: Int = 0
    private var leftArmWeight: Float = 0
    private var rightArmWeight: Float = 0
    private var lastUpdateTime: TimeInterval = 0

    func reset() {
        hasActiveBody = false
        yawFilter.reset()
        pitchFilter.reset()
        rollFilter.reset()
        leftUpperArmFilter.reset()
        leftForearmFilter.reset()
        rightUpperArmFilter.reset()
        rightForearmFilter.reset()
        smoothedTorsoYaw = 0
        smoothedTorsoPitch = 0
        smoothedTorsoRoll = 0
        leftArmVisibleStreak = 0
        rightArmVisibleStreak = 0
        leftArmWeight = 0
        rightArmWeight = 0
        lastUpdateTime = 0
    }

    func update(with frame: TrackedUpperBodyFrame, timestamp: TimeInterval) {
        hasActiveBody = true
        lastUpdateTime = timestamp

        smoothedTorsoYaw = deadzone(yawFilter.filter(frame.torsoYaw, timestamp: timestamp), threshold: 0.012)
        smoothedTorsoPitch = deadzone(pitchFilter.filter(frame.torsoPitch, timestamp: timestamp), threshold: 0.012)
        smoothedTorsoRoll = deadzone(rollFilter.filter(frame.torsoRoll, timestamp: timestamp), threshold: 0.012)

        let torsoQ = torsoRotation(fraction: 1.0)
        let invTorsoQ = simd_inverse(torsoQ)

        // Require 5 consecutive visible frames before fading an arm in to prevent flicker
        if let rawUpper = frame.leftUpperArmDir {
            leftArmVisibleStreak += 1
            let localUpper = simd_normalize(invTorsoQ.act(rawUpper))
            let qUpper = clampArmRotation(fromTo(SIMD3<Float>(1, 0, 0), localUpper))
            leftUpperArmRot = leftUpperArmFilter.filter(qUpper, timestamp: timestamp)

            if let rawForearm = frame.leftForearmDir {
                let localForearm = simd_normalize(simd_inverse(qUpper).act(simd_normalize(invTorsoQ.act(rawForearm))))
                let qForearm = clampForearmRotation(fromTo(SIMD3<Float>(1, 0, 0), localForearm), isLeft: true)
                leftForearmRot = leftForearmFilter.filter(qForearm, timestamp: timestamp)
            }
            if leftArmVisibleStreak >= 5 {
                leftArmWeight = min(1.0, leftArmWeight + 0.12)
            }
        } else {
            leftArmVisibleStreak = 0
            leftArmWeight = max(0.0, leftArmWeight - 0.08)
        }

        if let rawUpper = frame.rightUpperArmDir {
            rightArmVisibleStreak += 1
            let localUpper = simd_normalize(invTorsoQ.act(rawUpper))
            let qUpper = clampArmRotation(fromTo(SIMD3<Float>(-1, 0, 0), localUpper))
            rightUpperArmRot = rightUpperArmFilter.filter(qUpper, timestamp: timestamp)

            if let rawForearm = frame.rightForearmDir {
                let localForearm = simd_normalize(simd_inverse(qUpper).act(simd_normalize(invTorsoQ.act(rawForearm))))
                let qForearm = clampForearmRotation(fromTo(SIMD3<Float>(-1, 0, 0), localForearm), isLeft: false)
                rightForearmRot = rightForearmFilter.filter(qForearm, timestamp: timestamp)
            }
            if rightArmVisibleStreak >= 5 {
                rightArmWeight = min(1.0, rightArmWeight + 0.12)
            }
        } else {
            rightArmVisibleStreak = 0
            rightArmWeight = max(0.0, rightArmWeight - 0.08)
        }
    }

    /// Applies smoothed torso & arm tracking cleanly on top of the current frame's base pose.
    /// Caller MUST reset/apply the base pose before calling this so SLERP never feeds back into itself.
    func apply(to model: VRMModel, restTransforms: [Int: NodeRestTransform], now: TimeInterval) {
        guard hasActiveBody else { return }

        let staleFactor: Float
        if now - lastUpdateTime > 0.7 {
            staleFactor = max(0, 1.0 - Float((now - lastUpdateTime - 0.7) / 0.7))
        } else {
            staleFactor = 1.0
        }
        guard staleFactor > 0.001 else { return }

        // 1. Spine (40%) + Chest (60%) additive delta over current base pose
        if let spineIdx = model.humanoid?.getBoneNode(.spine), spineIdx < model.nodes.count {
            let delta = simd_slerp(simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), torsoRotation(fraction: 0.40), staleFactor)
            model.nodes[spineIdx].rotation = simd_normalize(delta * model.nodes[spineIdx].rotation)
            model.nodes[spineIdx].updateLocalMatrix()
        }
        if let chestIdx = model.humanoid?.getBoneNode(.chest), chestIdx < model.nodes.count {
            let delta = simd_slerp(simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), torsoRotation(fraction: 0.60), staleFactor)
            model.nodes[chestIdx].rotation = simd_normalize(delta * model.nodes[chestIdx].rotation)
            model.nodes[chestIdx].updateLocalMatrix()
        }

        // 2. Blend visible arms cleanly from the fresh base-pose rotation toward the tracked arm target
        let lWeight = leftArmWeight * staleFactor
        if lWeight > 0.02 {
            blendBoneFromCurrentBase(.leftUpperArm, target: leftUpperArmRot, weight: lWeight, model: model, restTransforms: restTransforms)
            blendBoneFromCurrentBase(.leftLowerArm, target: leftForearmRot, weight: lWeight, model: model, restTransforms: restTransforms)
        }
        let rWeight = rightArmWeight * staleFactor
        if rWeight > 0.02 {
            blendBoneFromCurrentBase(.rightUpperArm, target: rightUpperArmRot, weight: rWeight, model: model, restTransforms: restTransforms)
            blendBoneFromCurrentBase(.rightLowerArm, target: rightForearmRot, weight: rWeight, model: model, restTransforms: restTransforms)
        }
    }

    private func torsoRotation(fraction: Float) -> simd_quatf {
        let qYaw = simd_quatf(angle: smoothedTorsoYaw * fraction, axis: SIMD3<Float>(0, 1, 0))
        let qPitch = simd_quatf(angle: smoothedTorsoPitch * fraction, axis: SIMD3<Float>(1, 0, 0))
        let qRoll = simd_quatf(angle: smoothedTorsoRoll * fraction, axis: SIMD3<Float>(0, 0, 1))
        return simd_normalize(qYaw * qPitch * qRoll)
    }

    private func blendBoneFromCurrentBase(
        _ bone: VRMHumanoidBone,
        target: simd_quatf,
        weight: Float,
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        guard let idx = model.humanoid?.getBoneNode(bone),
              idx < model.nodes.count,
              let rest = restTransforms[idx] else { return }
        let baseRot = model.nodes[idx].rotation
        var goal = simd_normalize(target * rest.rot)
        if simd_dot(baseRot, goal) < 0 {
            goal = -goal
        }
        model.nodes[idx].rotation = simd_normalize(simd_slerp(baseRot, goal, weight))
        model.nodes[idx].updateLocalMatrix()
    }

    private func deadzone(_ value: Float, threshold: Float) -> Float {
        if abs(value) <= threshold { return 0 }
        return value > 0 ? (value - threshold) : (value + threshold)
    }

    private func fromTo(_ from: SIMD3<Float>, _ to: SIMD3<Float>) -> simd_quatf {
        let f = simd_normalize(from)
        let t = simd_normalize(to)
        let d = max(-1.0, min(1.0, simd_dot(f, t)))
        if d > 0.9999 {
            return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        }
        if d < -0.9999 {
            return simd_quatf(angle: .pi, axis: SIMD3<Float>(0, 1, 0))
        }
        let axis = simd_normalize(simd_cross(f, t))
        return simd_quatf(angle: acos(d), axis: axis)
    }

    private func clampArmRotation(_ q: simd_quatf) -> simd_quatf {
        let angle = q.angle
        guard angle > 0.001 else { return q }
        let maxAngle: Float = 115.0 * .pi / 180.0
        if angle <= maxAngle { return simd_normalize(q) }
        return simd_quatf(angle: maxAngle, axis: q.axis)
    }

    private func clampForearmRotation(_ q: simd_quatf, isLeft: Bool) -> simd_quatf {
        let dir = q.act(isLeft ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(-1, 0, 0))
        let clampedZ = max(-0.08, dir.z)
        let safeDir = simd_normalize(SIMD3<Float>(dir.x, dir.y * 0.65, clampedZ))
        let target = fromTo(isLeft ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(-1, 0, 0), safeDir)
        let maxBend: Float = 125.0 * .pi / 180.0
        if target.angle > maxBend {
            return simd_quatf(angle: maxBend, axis: target.axis)
        }
        return simd_normalize(target)
    }
}
