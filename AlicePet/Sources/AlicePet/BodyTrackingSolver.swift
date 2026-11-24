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

/// Smooths native Vision 3D upper-body tracking data via 1€ scalar & quaternion filters
/// and retargets spine/chest lean + twist and visible arms onto any VRM Humanoid skeleton.
@MainActor
final class BodyTrackingSolver {
    private var yawFilter = OneEuroFilterScalar(minCutoff: 1.1, beta: 0.08, dCutoff: 1.0)
    private var pitchFilter = OneEuroFilterScalar(minCutoff: 1.1, beta: 0.08, dCutoff: 1.0)
    private var rollFilter = OneEuroFilterScalar(minCutoff: 1.1, beta: 0.08, dCutoff: 1.0)

    private var leftUpperArmFilter = OneEuroFilterQuaternion(minCutoff: 1.4, beta: 0.12, dCutoff: 1.0)
    private var leftForearmFilter = OneEuroFilterQuaternion(minCutoff: 1.6, beta: 0.14, dCutoff: 1.0)
    private var rightUpperArmFilter = OneEuroFilterQuaternion(minCutoff: 1.4, beta: 0.12, dCutoff: 1.0)
    private var rightForearmFilter = OneEuroFilterQuaternion(minCutoff: 1.6, beta: 0.14, dCutoff: 1.0)

    private(set) var hasActiveBody = false
    private var smoothedTorsoYaw: Float = 0
    private var smoothedTorsoPitch: Float = 0
    private var smoothedTorsoRoll: Float = 0

    private var leftUpperArmRot = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var leftForearmRot = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var rightUpperArmRot = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var rightForearmRot = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

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
        leftArmWeight = 0
        rightArmWeight = 0
        lastUpdateTime = 0
    }

    func update(with frame: TrackedUpperBodyFrame, timestamp: TimeInterval) {
        hasActiveBody = true
        lastUpdateTime = timestamp

        smoothedTorsoYaw = yawFilter.filter(frame.torsoYaw, timestamp: timestamp)
        smoothedTorsoPitch = pitchFilter.filter(frame.torsoPitch, timestamp: timestamp)
        smoothedTorsoRoll = rollFilter.filter(frame.torsoRoll, timestamp: timestamp)

        let torsoQ = torsoRotation(fraction: 1.0)
        let invTorsoQ = simd_inverse(torsoQ)

        // Left arm (VRM rest direction = +X)
        if let rawUpper = frame.leftUpperArmDir {
            let localUpper = simd_normalize(invTorsoQ.act(rawUpper))
            let qUpper = clampArmRotation(fromTo(SIMD3<Float>(1, 0, 0), localUpper))
            leftUpperArmRot = leftUpperArmFilter.filter(qUpper, timestamp: timestamp)
            leftArmWeight = min(1.0, leftArmWeight + 0.18)

            if let rawForearm = frame.leftForearmDir {
                let localForearm = simd_normalize(simd_inverse(qUpper).act(simd_normalize(invTorsoQ.act(rawForearm))))
                let qForearm = clampForearmRotation(fromTo(SIMD3<Float>(1, 0, 0), localForearm), isLeft: true)
                leftForearmRot = leftForearmFilter.filter(qForearm, timestamp: timestamp)
            }
        } else {
            leftArmWeight = max(0.0, leftArmWeight - 0.08)
        }

        // Right arm (VRM rest direction = -X)
        if let rawUpper = frame.rightUpperArmDir {
            let localUpper = simd_normalize(invTorsoQ.act(rawUpper))
            let qUpper = clampArmRotation(fromTo(SIMD3<Float>(-1, 0, 0), localUpper))
            rightUpperArmRot = rightUpperArmFilter.filter(qUpper, timestamp: timestamp)
            rightArmWeight = min(1.0, rightArmWeight + 0.18)

            if let rawForearm = frame.rightForearmDir {
                let localForearm = simd_normalize(simd_inverse(qUpper).act(simd_normalize(invTorsoQ.act(rawForearm))))
                let qForearm = clampForearmRotation(fromTo(SIMD3<Float>(-1, 0, 0), localForearm), isLeft: false)
                rightForearmRot = rightForearmFilter.filter(qForearm, timestamp: timestamp)
            }
        } else {
            rightArmWeight = max(0.0, rightArmWeight - 0.08)
        }
    }

    func apply(to model: VRMModel, restTransforms: [Int: NodeRestTransform], now: TimeInterval) {
        guard hasActiveBody else { return }

        // Smoothly decay if body tracking temporarily drops out (>0.8s)
        let staleFactor: Float
        if now - lastUpdateTime > 0.8 {
            staleFactor = max(0, 1.0 - Float((now - lastUpdateTime - 0.8) / 0.8))
        } else {
            staleFactor = 1.0
        }
        guard staleFactor > 0.001 else { return }

        // 1. Spine (40%) + Chest (60%) torso lean & twist
        if let spineIdx = model.humanoid?.getBoneNode(.spine),
           let rest = restTransforms[spineIdx] {
            let q = simd_slerp(simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), torsoRotation(fraction: 0.40), staleFactor)
            model.nodes[spineIdx].rotation = q * rest.rot
            model.nodes[spineIdx].updateLocalMatrix()
        }
        if let chestIdx = model.humanoid?.getBoneNode(.chest),
           let rest = restTransforms[chestIdx] {
            let q = simd_slerp(simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), torsoRotation(fraction: 0.60), staleFactor)
            model.nodes[chestIdx].rotation = q * rest.rot
            model.nodes[chestIdx].updateLocalMatrix()
        }

        // 2. Visible arms blended smoothly over the active pose's resting arms
        let lWeight = leftArmWeight * staleFactor
        if lWeight > 0.02 {
            blendBone(.leftUpperArm, target: leftUpperArmRot, weight: lWeight, model: model, restTransforms: restTransforms)
            blendBone(.leftLowerArm, target: leftForearmRot, weight: lWeight, model: model, restTransforms: restTransforms)
        }
        let rWeight = rightArmWeight * staleFactor
        if rWeight > 0.02 {
            blendBone(.rightUpperArm, target: rightUpperArmRot, weight: rWeight, model: model, restTransforms: restTransforms)
            blendBone(.rightLowerArm, target: rightForearmRot, weight: rWeight, model: model, restTransforms: restTransforms)
        }
    }

    private func torsoRotation(fraction: Float) -> simd_quatf {
        let qYaw = simd_quatf(angle: smoothedTorsoYaw * fraction, axis: SIMD3<Float>(0, 1, 0))
        let qPitch = simd_quatf(angle: smoothedTorsoPitch * fraction, axis: SIMD3<Float>(1, 0, 0))
        let qRoll = simd_quatf(angle: smoothedTorsoRoll * fraction, axis: SIMD3<Float>(0, 0, 1))
        return simd_normalize(qYaw * qPitch * qRoll)
    }

    private func blendBone(
        _ bone: VRMHumanoidBone,
        target: simd_quatf,
        weight: Float,
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        guard let idx = model.humanoid?.getBoneNode(bone),
              idx < model.nodes.count,
              let rest = restTransforms[idx] else { return }
        let current = model.nodes[idx].rotation
        var goal = simd_normalize(target * rest.rot)
        if simd_dot(current, goal) < 0 {
            goal = -goal
        }
        model.nodes[idx].rotation = simd_normalize(simd_slerp(current, goal, weight))
        model.nodes[idx].updateLocalMatrix()
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
        let maxAngle: Float = 125.0 * .pi / 180.0
        if angle <= maxAngle { return simd_normalize(q) }
        return simd_quatf(angle: maxAngle, axis: q.axis)
    }

    private func clampForearmRotation(_ q: simd_quatf, isLeft: Bool) -> simd_quatf {
        // Forearm bends primarily forward around Y (elbow hinge); clamp extreme backward hyperextension
        let dir = q.act(isLeft ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(-1, 0, 0))
        let clampedZ = max(-0.12, dir.z) // prevent elbows bending backward
        let safeDir = simd_normalize(SIMD3<Float>(dir.x, dir.y * 0.65, clampedZ))
        let target = fromTo(isLeft ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(-1, 0, 0), safeDir)
        let maxBend: Float = 130.0 * .pi / 180.0
        if target.angle > maxBend {
            return simd_quatf(angle: maxBend, axis: target.axis)
        }
        return simd_normalize(target)
    }
}
