import Foundation
import VRMMetalKit
import simd

/// Decoded author animation clip exported from Unity `.anim` Humanoid Muscle curves.
struct AuthorAnimationBundle: Codable {
    let version: Int
    let source: String
    let clips: [String: AuthorAnimationClip]
}

struct AuthorAnimationClip: Codable {
    let duration: Float
    let fps: Float
    let loop: Bool
    let sourceAnim: String?
    let frames: [AuthorAnimationFrame]
}

struct AuthorAnimationFrame: Codable {
    let t: Float
    let rootT: [Float]
    let rootQ: [Float]
    let m: [String: Float]
}

struct SampledAuthorPose {
    let rootPositionUnity: SIMD3<Float>
    let rootRotationUnity: simd_quatf
    let muscles: [String: Float]
}

/// Converts Unity Humanoid Muscle values (`[-1, 1]`) and left-handed Unity RootT/RootQ
/// into VRM right-handed runtime coordinates (`Left = +X, Up = +Y, Forward = +Z`).
enum UnityMuscleRetargeter {
    private static let deg2rad: Float = .pi / 180.0

    static func unityQuatToVRM(_ q: simd_quatf) -> simd_quatf {
        // Mirror across YZ plane (Unity +X=Right -> VRM +X=Left): negate y and z imaginary parts
        simd_normalize(simd_quatf(ix: q.imag.x, iy: -q.imag.y, iz: -q.imag.z, r: q.real))
    }

    static func apply(
        pose: SampledAuthorPose,
        standReferenceRootT: SIMD3<Float>,
        clipInitialRootZ: Float,
        verticalBias: Float,
        to model: VRMModel,
        restTransforms: [Int: NodeRestTransform],
        applyFingers: Bool
    ) {
        let m = pose.muscles

        // 1. Hips root translation & rotation
        if let hipsIdx = model.humanoid?.getBoneNode(.hips),
           hipsIdx < model.nodes.count,
           let rest = restTransforms[hipsIdx] {
            let standY = max(standReferenceRootT.y, 0.5)
            let scale = rest.pos.y / standY
            let dx = -(pose.rootPositionUnity.x - standReferenceRootT.x) * scale
            let dy = (pose.rootPositionUnity.y - standReferenceRootT.y) * scale + verticalBias
            // Keep Z centered relative to clip start so desktop pet stays inside floating window
            let dz = (pose.rootPositionUnity.z - clipInitialRootZ) * scale

            model.nodes[hipsIdx].translation = rest.pos + SIMD3<Float>(dx, dy, dz)
            model.nodes[hipsIdx].rotation = unityQuatToVRM(pose.rootRotationUnity) * rest.rot
            model.nodes[hipsIdx].updateLocalMatrix()
        }

        // 2. Torso & Head chain
        applyTorsoBone(.spine, prefix: "Spine", maxDeg: 40, muscles: m, model: model, restTransforms: restTransforms)
        applyTorsoBone(.chest, prefix: "Chest", maxDeg: 40, muscles: m, model: model, restTransforms: restTransforms)
        applyTorsoBone(.upperChest, prefix: "UpperChest", maxDeg: 25, muscles: m, model: model, restTransforms: restTransforms)
        applyNeckOrHead(.neck, prefix: "Neck", maxDeg: 40, muscles: m, model: model, restTransforms: restTransforms)
        applyNeckOrHead(.head, prefix: "Head", maxDeg: 40, muscles: m, model: model, restTransforms: restTransforms)

        // 3. Shoulders, Arms, Forearms, Hands
        applyShoulder(isLeft: true, muscles: m, model: model, restTransforms: restTransforms)
        applyShoulder(isLeft: false, muscles: m, model: model, restTransforms: restTransforms)
        applyUpperArm(isLeft: true, muscles: m, model: model, restTransforms: restTransforms)
        applyUpperArm(isLeft: false, muscles: m, model: model, restTransforms: restTransforms)
        applyLowerArm(isLeft: true, muscles: m, model: model, restTransforms: restTransforms)
        applyLowerArm(isLeft: false, muscles: m, model: model, restTransforms: restTransforms)
        applyHand(isLeft: true, muscles: m, model: model, restTransforms: restTransforms)
        applyHand(isLeft: false, muscles: m, model: model, restTransforms: restTransforms)

        // 4. Legs & Feet
        applyUpperLeg(isLeft: true, muscles: m, model: model, restTransforms: restTransforms)
        applyUpperLeg(isLeft: false, muscles: m, model: model, restTransforms: restTransforms)
        applyLowerLeg(isLeft: true, muscles: m, model: model, restTransforms: restTransforms)
        applyLowerLeg(isLeft: false, muscles: m, model: model, restTransforms: restTransforms)
        applyFoot(isLeft: true, muscles: m, model: model, restTransforms: restTransforms)
        applyFoot(isLeft: false, muscles: m, model: model, restTransforms: restTransforms)
        applyToes(isLeft: true, muscles: m, model: model, restTransforms: restTransforms)
        applyToes(isLeft: false, muscles: m, model: model, restTransforms: restTransforms)

        // 5. Optional finger muscles embedded in the author clip
        if applyFingers {
            applyFingersFromClip(isLeft: true, muscles: m, model: model, restTransforms: restTransforms)
            applyFingersFromClip(isLeft: false, muscles: m, model: model, restTransforms: restTransforms)
        }
    }

    // MARK: - Torso & Head

    private static func applyTorsoBone(
        _ bone: VRMHumanoidBone,
        prefix: String,
        maxDeg: Float,
        muscles: [String: Float],
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        let fb = muscles["\(prefix) Front-Back"] ?? 0
        let lr = muscles["\(prefix) Left-Right"] ?? 0
        let tw = muscles["\(prefix) Twist Left-Right"] ?? 0
        let pitch = -fb * maxDeg * deg2rad
        let roll = -lr * maxDeg * deg2rad
        let yaw = -tw * maxDeg * deg2rad
        let q = simd_quatf(angle: yaw, axis: SIMD3<Float>(0, 1, 0))
            * simd_quatf(angle: pitch, axis: SIMD3<Float>(1, 0, 0))
            * simd_quatf(angle: roll, axis: SIMD3<Float>(0, 0, 1))
        setBoneRotation(bone, q, model: model, restTransforms: restTransforms)
    }

    private static func applyNeckOrHead(
        _ bone: VRMHumanoidBone,
        prefix: String,
        maxDeg: Float,
        muscles: [String: Float],
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        let nod = muscles["\(prefix) Nod Down-Up"] ?? 0
        let tilt = muscles["\(prefix) Tilt Left-Right"] ?? 0
        let turn = muscles["\(prefix) Turn Left-Right"] ?? 0
        let pitch = -nod * maxDeg * deg2rad
        let roll = -tilt * maxDeg * deg2rad
        let yaw = -turn * maxDeg * deg2rad
        let q = simd_quatf(angle: yaw, axis: SIMD3<Float>(0, 1, 0))
            * simd_quatf(angle: pitch, axis: SIMD3<Float>(1, 0, 0))
            * simd_quatf(angle: roll, axis: SIMD3<Float>(0, 0, 1))
        setBoneRotation(bone, q, model: model, restTransforms: restTransforms)
    }

    // MARK: - Arms

    private static func applyShoulder(
        isLeft: Bool,
        muscles: [String: Float],
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        let side = isLeft ? "Left" : "Right"
        let bone: VRMHumanoidBone = isLeft ? .leftShoulder : .rightShoulder
        let du = muscles["\(side) Shoulder Down-Up"] ?? 0
        let fb = muscles["\(side) Shoulder Front-Back"] ?? 0
        let s: Float = isLeft ? 1.0 : -1.0
        let q = simd_quatf(angle: -s * fb * 15.0 * deg2rad, axis: SIMD3<Float>(0, 1, 0))
            * simd_quatf(angle: s * du * 15.0 * deg2rad, axis: SIMD3<Float>(0, 0, 1))
        setBoneRotation(bone, q, model: model, restTransforms: restTransforms)
    }

    private static func applyUpperArm(
        isLeft: Bool,
        muscles: [String: Float],
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        let side = isLeft ? "Left" : "Right"
        let bone: VRMHumanoidBone = isLeft ? .leftUpperArm : .rightUpperArm
        let du = muscles["\(side) Arm Down-Up"] ?? -0.55
        let fb = muscles["\(side) Arm Front-Back"] ?? 0
        let tw = muscles["\(side) Arm Twist In-Out"] ?? 0

        let s: Float = isLeft ? 1.0 : -1.0
        // Unity Arm Down-Up: T-pose is ~+0.25, -0.60 is natural standing arm down (~-68 deg)
        // Unity Arm Front-Back: negative = swing forward (+Z across chest), positive = swing back (-Z)
        let downUpRad = (du * 80.0 - 20.0) * deg2rad
        let frontBackRad = fb * 85.0 * deg2rad
        let twistRad = tw * 65.0 * deg2rad

        let qFrontBack = simd_quatf(angle: s * frontBackRad, axis: SIMD3<Float>(0, 1, 0))
        let qDownUp = simd_quatf(angle: s * downUpRad, axis: SIMD3<Float>(0, 0, 1))
        // X-axis axial roll is mirror-symmetric across YZ without flipping sign between left and right arms
        let qTwist = simd_quatf(angle: -twistRad, axis: SIMD3<Float>(1, 0, 0))
        setBoneRotation(bone, qFrontBack * qDownUp * qTwist, model: model, restTransforms: restTransforms)
    }

    private static func applyLowerArm(
        isLeft: Bool,
        muscles: [String: Float],
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        let side = isLeft ? "Left" : "Right"
        let bone: VRMHumanoidBone = isLeft ? .leftLowerArm : .rightLowerArm
        let stretch = muscles["\(side) Forearm Stretch"] ?? 0.85
        let tw = muscles["\(side) Forearm Twist In-Out"] ?? 0

        let s: Float = isLeft ? 1.0 : -1.0
        let bendRad = max(0, 1.0 - stretch) * 72.5 * deg2rad
        let twistRad = tw * 60.0 * deg2rad

        let qBend = simd_quatf(angle: -s * bendRad, axis: SIMD3<Float>(0, 1, 0))
        let qTwist = simd_quatf(angle: -twistRad, axis: SIMD3<Float>(1, 0, 0))
        setBoneRotation(bone, qBend * qTwist, model: model, restTransforms: restTransforms)
    }

    private static func applyHand(
        isLeft: Bool,
        muscles: [String: Float],
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        let side = isLeft ? "Left" : "Right"
        let bone: VRMHumanoidBone = isLeft ? .leftHand : .rightHand
        let du = muscles["\(side) Hand Down-Up"] ?? 0
        let io = muscles["\(side) Hand In-Out"] ?? 0
        let s: Float = isLeft ? 1.0 : -1.0

        let q = simd_quatf(angle: s * io * 30.0 * deg2rad, axis: SIMD3<Float>(0, 1, 0))
            * simd_quatf(angle: s * du * 50.0 * deg2rad, axis: SIMD3<Float>(0, 0, 1))
        setBoneRotation(bone, q, model: model, restTransforms: restTransforms)
    }

    // MARK: - Legs & Feet

    private static func applyUpperLeg(
        isLeft: Bool,
        muscles: [String: Float],
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        let side = isLeft ? "Left" : "Right"
        let bone: VRMHumanoidBone = isLeft ? .leftUpperLeg : .rightUpperLeg
        let fb = muscles["\(side) Upper Leg Front-Back"] ?? 0.62
        let io = muscles["\(side) Upper Leg In-Out"] ?? 0
        let tw = muscles["\(side) Upper Leg Twist In-Out"] ?? 0

        let s: Float = isLeft ? 1.0 : -1.0
        // Unity Upper Leg Front-Back: +0.62 = straight down, -0.56 = seated (~-82 deg forward)
        let pitchRad = (fb - 0.62) * 70.0 * deg2rad
        let rollRad = s * io * 45.0 * deg2rad
        let yawRad = s * tw * 40.0 * deg2rad

        let q = simd_quatf(angle: pitchRad, axis: SIMD3<Float>(1, 0, 0))
            * simd_quatf(angle: rollRad, axis: SIMD3<Float>(0, 0, 1))
            * simd_quatf(angle: yawRad, axis: SIMD3<Float>(0, 1, 0))
        setBoneRotation(bone, q, model: model, restTransforms: restTransforms)
    }

    private static func applyLowerLeg(
        isLeft: Bool,
        muscles: [String: Float],
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        let side = isLeft ? "Left" : "Right"
        let bone: VRMHumanoidBone = isLeft ? .leftLowerLeg : .rightLowerLeg
        let stretch = muscles["\(side) Lower Leg Stretch"] ?? 0.98
        let tw = muscles["\(side) Lower Leg Twist In-Out"] ?? 0

        let s: Float = isLeft ? 1.0 : -1.0
        let kneeRad = max(0, 1.0 - stretch) * 80.0 * deg2rad
        let yawRad = s * tw * 25.0 * deg2rad

        let q = simd_quatf(angle: kneeRad, axis: SIMD3<Float>(1, 0, 0))
            * simd_quatf(angle: yawRad, axis: SIMD3<Float>(0, 1, 0))
        setBoneRotation(bone, q, model: model, restTransforms: restTransforms)
    }

    private static func applyFoot(
        isLeft: Bool,
        muscles: [String: Float],
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        let side = isLeft ? "Left" : "Right"
        let bone: VRMHumanoidBone = isLeft ? .leftFoot : .rightFoot
        let ud = muscles["\(side) Foot Up-Down"] ?? 0
        let tw = muscles["\(side) Foot Twist In-Out"] ?? 0
        let s: Float = isLeft ? 1.0 : -1.0

        let q = simd_quatf(angle: -ud * 45.0 * deg2rad, axis: SIMD3<Float>(1, 0, 0))
            * simd_quatf(angle: s * tw * 25.0 * deg2rad, axis: SIMD3<Float>(0, 0, 1))
        setBoneRotation(bone, q, model: model, restTransforms: restTransforms)
    }

    private static func applyToes(
        isLeft: Bool,
        muscles: [String: Float],
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        let side = isLeft ? "Left" : "Right"
        let bone: VRMHumanoidBone = isLeft ? .leftToes : .rightToes
        let ud = muscles["\(side) Toes Up-Down"] ?? 0
        let q = simd_quatf(angle: -ud * 30.0 * deg2rad, axis: SIMD3<Float>(1, 0, 0))
        setBoneRotation(bone, q, model: model, restTransforms: restTransforms)
    }

    // MARK: - Fingers

    private static func applyFingersFromClip(
        isLeft: Bool,
        muscles: [String: Float],
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        let prefix = isLeft ? "LeftHand." : "RightHand."
        guard muscles.keys.contains(where: { $0.hasPrefix(prefix) }) else { return }

        // In VRMMetalKit runtime coordinates, Left arm extends along +X, Right arm along -X.
        // Curling fingers down into palm: Left rotates around -Z, Right rotates around +Z.
        let curlSign: Float = isLeft ? -1.0 : 1.0
        let spreadSign: Float = isLeft ? -1.0 : 1.0

        func curlAngle(_ v: Float?) -> Float {
            let val = v ?? 0.65
            return (val >= 0 ? -(val * 14.0) : (-val * 75.0)) * deg2rad
        }

        let fingers: [(String, [VRMHumanoidBone])] = [
            ("Index", isLeft ? [.leftIndexProximal, .leftIndexIntermediate, .leftIndexDistal] : [.rightIndexProximal, .rightIndexIntermediate, .rightIndexDistal]),
            ("Middle", isLeft ? [.leftMiddleProximal, .leftMiddleIntermediate, .leftMiddleDistal] : [.rightMiddleProximal, .rightMiddleIntermediate, .rightMiddleDistal]),
            ("Ring", isLeft ? [.leftRingProximal, .leftRingIntermediate, .leftRingDistal] : [.rightRingProximal, .rightRingIntermediate, .rightRingDistal]),
            ("Little", isLeft ? [.leftLittleProximal, .leftLittleIntermediate, .leftLittleDistal] : [.rightLittleProximal, .rightLittleIntermediate, .rightLittleDistal]),
        ]

        for (fname, bones) in fingers {
            let spread = (muscles["\(prefix)\(fname).Spread"] ?? 0) * 14.0 * deg2rad * spreadSign
            for (segIdx, bone) in bones.enumerated() {
                let c = curlAngle(muscles["\(prefix)\(fname).\(segIdx + 1) Stretched"])
                let qCurl = simd_quatf(angle: c * curlSign, axis: SIMD3<Float>(0, 0, 1))
                let q = (segIdx == 0)
                    ? simd_quatf(angle: spread, axis: SIMD3<Float>(0, 1, 0)) * qCurl
                    : qCurl
                setBoneRotation(bone, q, model: model, restTransforms: restTransforms)
            }
        }

        let t1 = muscles["\(prefix)Thumb.1 Stretched"] ?? 0.5
        let t2 = muscles["\(prefix)Thumb.2 Stretched"] ?? 0.5
        let t3 = muscles["\(prefix)Thumb.3 Stretched"] ?? 0.5
        let tSpread = muscles["\(prefix)Thumb.Spread"] ?? 0.0
        let thumbBones: [VRMHumanoidBone] = isLeft
            ? [.leftThumbMetacarpal, .leftThumbProximal, .leftThumbDistal]
            : [.rightThumbMetacarpal, .rightThumbProximal, .rightThumbDistal]

        let q0 = simd_quatf(angle: (isLeft ? 1 : -1) * max(0, -t1) * 38.0 * deg2rad, axis: SIMD3<Float>(0, 1, 0))
            * simd_quatf(angle: (isLeft ? -1 : 1) * tSpread * 20.0 * deg2rad, axis: SIMD3<Float>(0, 0, 1))
        setBoneRotation(thumbBones[0], q0, model: model, restTransforms: restTransforms)
        setBoneRotation(thumbBones[1], simd_quatf(angle: (isLeft ? 1 : -1) * max(0, -t2) * 48.0 * deg2rad, axis: SIMD3<Float>(0, 1, 0)), model: model, restTransforms: restTransforms)
        setBoneRotation(thumbBones[2], simd_quatf(angle: (isLeft ? 1 : -1) * max(0, -t3) * 52.0 * deg2rad, axis: SIMD3<Float>(0, 1, 0)), model: model, restTransforms: restTransforms)
    }

    private static func setBoneRotation(
        _ bone: VRMHumanoidBone,
        _ delta: simd_quatf,
        model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        guard let idx = model.humanoid?.getBoneNode(bone),
              idx < model.nodes.count,
              let rest = restTransforms[idx] else { return }
        model.nodes[idx].rotation = delta * rest.rot
        model.nodes[idx].updateLocalMatrix()
    }
}
