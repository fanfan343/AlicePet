import Foundation
import VRMMetalKit
import simd

enum HandSide {
    case left
    case right
    case both
}

/// Unified per-frame skeletal motion output interface.
/// Future VRMA playback and out-of-process Kimodo / ARDY retargeting streams will conform to this protocol.
@MainActor
protocol MotionSource {
    func applyBasePose(to model: VRMModel, restTransforms: [Int: NodeRestTransform])
    func applyProceduralFrame(
        to model: VRMModel,
        restTransforms: [Int: NodeRestTransform],
        elapsed: Float,
        idleBreathingEnabled: Bool,
        cameraHeadAngles: SIMD3<Float>?
    )
}

struct NodeRestTransform {
    let pos: SIMD3<Float>
    let rot: simd_quatf
    let scale: SIMD3<Float>
}

/// Manages body poses, hand gestures, hair style bone adjustments, and per-frame procedural motion.
@MainActor
final class PoseController: MotionSource {
    private(set) var poseMode: PetPoseMode = .stand
    private(set) var leftHandGesture: HandGesturePreset = .natural
    private(set) var rightHandGesture: HandGesturePreset = .natural

    private(set) var restNodeTransforms: [Int: NodeRestTransform] = [:]
    private(set) var hairNodeIndices: [String: Int] = [:]

    private let afkBookChestPitchDeg: Float = 10.0
    private let afkBookChestYawDeg: Float = -6.0
    private let afkTeaChestPitchDeg: Float = 4.0
    private let afkTeaChestYawDeg: Float = -5.0

    func captureRestPose(from model: VRMModel) {
        restNodeTransforms.removeAll()
        hairNodeIndices.removeAll()
        for (idx, node) in model.nodes.enumerated() {
            restNodeTransforms[idx] = NodeRestTransform(
                pos: node.translation,
                rot: node.rotation,
                scale: node.scale
            )
            if let name = node.name, name.lowercased().contains("hair") {
                hairNodeIndices[name] = idx
            }
        }
    }

    func setPoseMode(_ mode: PetPoseMode) {
        poseMode = mode
    }

    func setHandGesture(_ preset: HandGesturePreset, side: HandSide) {
        switch side {
        case .left:
            leftHandGesture = preset
        case .right:
            rightHandGesture = preset
        case .both:
            leftHandGesture = preset
            rightHandGesture = preset
        }
    }

    func applyAll(to model: VRMModel, recipeStore: AssetRecipeStore, profile: ModelProfile, hairStyle: HairStyleOption) {
        applyBasePose(to: model, restTransforms: restNodeTransforms)
        applyHandGestures(to: model, recipeStore: recipeStore)
        applyHairStyle(hairStyle, profile: profile, to: model)
        model.updateNodeTransforms()
    }

    // MARK: - MotionSource

    func applyBasePose(to model: VRMModel, restTransforms: [Int: NodeRestTransform]) {
        let bodyBones: [VRMHumanoidBone] = [
            .hips, .spine, .chest, .upperChest, .neck, .head,
            .leftShoulder, .leftUpperArm, .leftLowerArm, .leftHand,
            .rightShoulder, .rightUpperArm, .rightLowerArm, .rightHand,
            .leftUpperLeg, .leftLowerLeg, .leftFoot, .leftToes,
            .rightUpperLeg, .rightLowerLeg, .rightFoot, .rightToes
        ]
        resetHumanoidBones(bodyBones, on: model, restTransforms: restTransforms)

        switch poseMode {
        case .stand:
            rotateBone(.leftUpperArm, on: model, restTransforms: restTransforms, roll: -62)
            rotateBone(.rightUpperArm, on: model, restTransforms: restTransforms, roll: 62)
            rotateBone(.leftLowerArm, on: model, restTransforms: restTransforms, yaw: -12)
            rotateBone(.rightLowerArm, on: model, restTransforms: restTransforms, yaw: 12)

        case .sit:
            translateBone(.hips, on: model, restTransforms: restTransforms, offset: SIMD3<Float>(0, -0.18, 0))
            rotateBone(.hips, on: model, restTransforms: restTransforms, pitch: -4)
            rotateBone(.spine, on: model, restTransforms: restTransforms, pitch: 5)
            rotateBone(.leftUpperLeg, on: model, restTransforms: restTransforms, yaw: 6, pitch: -82, roll: -4)
            rotateBone(.rightUpperLeg, on: model, restTransforms: restTransforms, yaw: -6, pitch: -82, roll: 4)
            rotateBone(.leftLowerLeg, on: model, restTransforms: restTransforms, pitch: 82)
            rotateBone(.rightLowerLeg, on: model, restTransforms: restTransforms, pitch: 82)
            rotateBone(.leftFoot, on: model, restTransforms: restTransforms, pitch: 12)
            rotateBone(.rightFoot, on: model, restTransforms: restTransforms, pitch: 12)

            rotateBone(.leftUpperArm, on: model, restTransforms: restTransforms, yaw: -16, pitch: -12, roll: -58)
            rotateBone(.rightUpperArm, on: model, restTransforms: restTransforms, yaw: 16, pitch: -12, roll: 58)
            rotateBone(.leftLowerArm, on: model, restTransforms: restTransforms, yaw: -42, pitch: -14)
            rotateBone(.rightLowerArm, on: model, restTransforms: restTransforms, yaw: 42, pitch: -14)

        case .afkBook:
            translateBone(.hips, on: model, restTransforms: restTransforms, offset: SIMD3<Float>(0, 0.04, 0))
            rotateBone(.hips, on: model, restTransforms: restTransforms, yaw: -6, pitch: -8, roll: 3)
            rotateBone(.spine, on: model, restTransforms: restTransforms, yaw: 4, pitch: 8)
            rotateBone(.chest, on: model, restTransforms: restTransforms, yaw: afkBookChestYawDeg, pitch: afkBookChestPitchDeg)
            rotateBone(.neck, on: model, restTransforms: restTransforms, yaw: 4, pitch: 12)
            rotateBone(.head, on: model, restTransforms: restTransforms, yaw: 0, pitch: 8)

            // Gentle floating legs tuned to stay completely clear of skirt mesh
            rotateBone(.leftUpperLeg, on: model, restTransforms: restTransforms, yaw: 3, pitch: -15, roll: -3)
            rotateBone(.leftLowerLeg, on: model, restTransforms: restTransforms, pitch: 34)
            rotateBone(.leftFoot, on: model, restTransforms: restTransforms, pitch: 18)
            rotateBone(.rightUpperLeg, on: model, restTransforms: restTransforms, yaw: -3, pitch: -21, roll: 3)
            rotateBone(.rightLowerLeg, on: model, restTransforms: restTransforms, pitch: 44)
            rotateBone(.rightFoot, on: model, restTransforms: restTransforms, pitch: 22)

            // Arms brought forward toward open book
            rotateBone(.leftUpperArm, on: model, restTransforms: restTransforms, yaw: -36, pitch: -22, roll: -52)
            rotateBone(.leftLowerArm, on: model, restTransforms: restTransforms, yaw: -52, pitch: -16)
            rotateBone(.leftHand, on: model, restTransforms: restTransforms, yaw: 10, pitch: 14, roll: 14)

            rotateBone(.rightUpperArm, on: model, restTransforms: restTransforms, yaw: 36, pitch: -22, roll: 52)
            rotateBone(.rightLowerArm, on: model, restTransforms: restTransforms, yaw: 52, pitch: -16)
            rotateBone(.rightHand, on: model, restTransforms: restTransforms, yaw: -10, pitch: 14, roll: -14)

        case .afkTea:
            translateBone(.hips, on: model, restTransforms: restTransforms, offset: SIMD3<Float>(0, 0.02, 0))
            rotateBone(.hips, on: model, restTransforms: restTransforms, yaw: 6, pitch: -5, roll: -2)
            rotateBone(.spine, on: model, restTransforms: restTransforms, pitch: 4)
            rotateBone(.chest, on: model, restTransforms: restTransforms, yaw: afkTeaChestYawDeg, pitch: afkTeaChestPitchDeg)
            rotateBone(.neck, on: model, restTransforms: restTransforms, yaw: -4, pitch: 4)

            // Gentle floating legs tuned to avoid skirt clipping
            rotateBone(.leftUpperLeg, on: model, restTransforms: restTransforms, yaw: 3, pitch: -19, roll: -3)
            rotateBone(.leftLowerLeg, on: model, restTransforms: restTransforms, pitch: 40)
            rotateBone(.leftFoot, on: model, restTransforms: restTransforms, pitch: 18)
            rotateBone(.rightUpperLeg, on: model, restTransforms: restTransforms, yaw: -3, pitch: -13, roll: 3)
            rotateBone(.rightLowerLeg, on: model, restTransforms: restTransforms, pitch: 26)
            rotateBone(.rightFoot, on: model, restTransforms: restTransforms, pitch: 16)

            rotateBone(.leftUpperArm, on: model, restTransforms: restTransforms, yaw: -24, pitch: -16, roll: -50)
            rotateBone(.leftLowerArm, on: model, restTransforms: restTransforms, yaw: -44, pitch: -14)
            rotateBone(.rightUpperArm, on: model, restTransforms: restTransforms, yaw: 16, pitch: -8, roll: 58)
            rotateBone(.rightLowerArm, on: model, restTransforms: restTransforms, yaw: 24, pitch: -8)
        }
    }

    func applyProceduralFrame(
        to model: VRMModel,
        restTransforms: [Int: NodeRestTransform],
        elapsed t: Float,
        idleBreathingEnabled: Bool,
        cameraHeadAngles: SIMD3<Float>?
    ) {
        let breath = sin(t * 1.7)

        if idleBreathingEnabled {
            if let chestIdx = model.humanoid?.getBoneNode(.chest),
               let base = restTransforms[chestIdx]?.rot {
                let pitchDeg: Float = (poseMode == .afkBook) ? afkBookChestPitchDeg : ((poseMode == .afkTea) ? afkTeaChestPitchDeg : 0.0)
                let yawDeg: Float = (poseMode == .afkBook) ? afkBookChestYawDeg : ((poseMode == .afkTea) ? afkTeaChestYawDeg : 0.0)
                let pitch = simd_quatf(angle: pitchDeg * .pi / 180.0 + breath * 0.014, axis: SIMD3<Float>(1, 0, 0))
                let yaw = simd_quatf(angle: yawDeg * .pi / 180.0, axis: SIMD3<Float>(0, 1, 0))
                model.nodes[chestIdx].rotation = yaw * pitch * base
                model.nodes[chestIdx].updateLocalMatrix()
            }

            if poseMode == .afkBook || poseMode == .afkTea {
                if let hipsIdx = model.humanoid?.getBoneNode(.hips),
                   let rest = restTransforms[hipsIdx] {
                    let baseHover: Float = (poseMode == .afkBook) ? 0.04 : 0.02
                    model.nodes[hipsIdx].translation = rest.pos + SIMD3<Float>(0, baseHover + sin(t * 1.35) * 0.018, 0)
                    model.nodes[hipsIdx].updateLocalMatrix()
                }
            }
        }

        if let headAngles = cameraHeadAngles {
            let yaw = headAngles.x
            let pitch = headAngles.y
            let roll = headAngles.z
            if let neckIdx = model.humanoid?.getBoneNode(.neck), let base = restTransforms[neckIdx]?.rot {
                let qYaw = simd_quatf(angle: yaw * 0.35, axis: SIMD3<Float>(0, 1, 0))
                let qPitch = simd_quatf(angle: pitch * 0.35, axis: SIMD3<Float>(1, 0, 0))
                let qRoll = simd_quatf(angle: roll * 0.35, axis: SIMD3<Float>(0, 0, 1))
                model.nodes[neckIdx].rotation = base * qYaw * qPitch * qRoll
                model.nodes[neckIdx].updateLocalMatrix()
            }
            if let headIdx = model.humanoid?.getBoneNode(.head), let base = restTransforms[headIdx]?.rot {
                let qYaw = simd_quatf(angle: yaw * 0.65, axis: SIMD3<Float>(0, 1, 0))
                let qPitch = simd_quatf(angle: pitch * 0.65, axis: SIMD3<Float>(1, 0, 0))
                let qRoll = simd_quatf(angle: roll * 0.65, axis: SIMD3<Float>(0, 0, 1))
                model.nodes[headIdx].rotation = base * qYaw * qPitch * qRoll
                model.nodes[headIdx].updateLocalMatrix()
            }
            return
        }

        if poseMode != .afkBook && idleBreathingEnabled {
            let sway = sin(t * 0.75)
            if let headIdx = model.humanoid?.getBoneNode(.head), let base = restTransforms[headIdx]?.rot {
                let roll = simd_quatf(angle: sway * 0.018, axis: SIMD3<Float>(0, 0, 1))
                let pitch = simd_quatf(angle: breath * 0.01, axis: SIMD3<Float>(1, 0, 0))
                model.nodes[headIdx].rotation = base * roll * pitch
                model.nodes[headIdx].updateLocalMatrix()
            }
        }
    }

    func computeBookWorldMatrix(time: Float, profile: ModelProfile) -> simd_float4x4 {
        let bob = (poseMode == .afkBook || poseMode == .afkTea) ? sin(time * 1.4) * 0.015 : 0.0
        let basePos = profile.bookBasePosition
        let pos = SIMD3<Float>(basePos.x, basePos.y + bob, basePos.z)
        let tilt = simd_quatf(angle: profile.bookBaseTilt, axis: SIMD3<Float>(1, 0, 0))
        let yaw = simd_quatf(angle: sin(time * 0.6) * 0.05, axis: SIMD3<Float>(0, 1, 0))
        let rot = float4x4(yaw * tilt)
        var t = matrix_identity_float4x4
        t.columns.3 = SIMD4<Float>(pos.x, pos.y, pos.z, 1.0)
        return t * rot
    }

    // MARK: - Hands & Hair

    func applyHandGestures(to model: VRMModel, recipeStore: AssetRecipeStore) {
        let allFingers: [VRMHumanoidBone] = [
            .leftThumbMetacarpal, .leftThumbProximal, .leftThumbDistal,
            .leftIndexProximal, .leftIndexIntermediate, .leftIndexDistal,
            .leftMiddleProximal, .leftMiddleIntermediate, .leftMiddleDistal,
            .leftRingProximal, .leftRingIntermediate, .leftRingDistal,
            .leftLittleProximal, .leftLittleIntermediate, .leftLittleDistal,
            .rightThumbMetacarpal, .rightThumbProximal, .rightThumbDistal,
            .rightIndexProximal, .rightIndexIntermediate, .rightIndexDistal,
            .rightMiddleProximal, .rightMiddleIntermediate, .rightMiddleDistal,
            .rightRingProximal, .rightRingIntermediate, .rightRingDistal,
            .rightLittleProximal, .rightLittleIntermediate, .rightLittleDistal
        ]
        resetHumanoidBones(allFingers, on: model, restTransforms: restNodeTransforms)

        for (bone, q) in recipeStore.fingerRotations(for: leftHandGesture, isLeft: true) {
            if let idx = model.humanoid?.getBoneNode(bone), idx < model.nodes.count, let rest = restNodeTransforms[idx] {
                model.nodes[idx].rotation = rest.rot * q
                model.nodes[idx].updateLocalMatrix()
            }
        }
        for (bone, q) in recipeStore.fingerRotations(for: rightHandGesture, isLeft: false) {
            if let idx = model.humanoid?.getBoneNode(bone), idx < model.nodes.count, let rest = restNodeTransforms[idx] {
                model.nodes[idx].rotation = rest.rot * q
                model.nodes[idx].updateLocalMatrix()
            }
        }
    }

    func applyHairStyle(_ style: HairStyleOption, profile: ModelProfile, to model: VRMModel) {
        for (_, idx) in hairNodeIndices where idx < model.nodes.count {
            if let rest = restNodeTransforms[idx] {
                model.nodes[idx].rotation = rest.rot
                model.nodes[idx].updateLocalMatrix()
            }
        }

        guard let adjustments = profile.hairStyles[style.rawValue] else { return }
        for adj in adjustments {
            if let idx = hairNodeIndices[adj.nodeName], let rest = restNodeTransforms[idx] {
                model.nodes[idx].rotation = simd_quatf(angle: adj.angleRad, axis: adj.normalizedAxis) * rest.rot
                model.nodes[idx].updateLocalMatrix()
            }
        }
    }

    // MARK: - Helpers

    private func resetHumanoidBones(
        _ bones: [VRMHumanoidBone],
        on model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        for bone in bones {
            if let idx = model.humanoid?.getBoneNode(bone),
               idx < model.nodes.count,
               let rest = restTransforms[idx] {
                model.nodes[idx].translation = rest.pos
                model.nodes[idx].rotation = rest.rot
                model.nodes[idx].scale = rest.scale
                model.nodes[idx].updateLocalMatrix()
            }
        }
    }

    private func rotateBone(
        _ bone: VRMHumanoidBone,
        on model: VRMModel,
        restTransforms: [Int: NodeRestTransform],
        yaw: Float = 0,
        pitch: Float = 0,
        roll: Float = 0
    ) {
        guard let idx = model.humanoid?.getBoneNode(bone),
              idx < model.nodes.count,
              let rest = restTransforms[idx] else { return }
        let qY = simd_quatf(angle: yaw * .pi / 180.0, axis: SIMD3<Float>(0, 1, 0))
        let qX = simd_quatf(angle: pitch * .pi / 180.0, axis: SIMD3<Float>(1, 0, 0))
        let qZ = simd_quatf(angle: roll * .pi / 180.0, axis: SIMD3<Float>(0, 0, 1))
        model.nodes[idx].rotation = qY * qX * qZ * rest.rot
        model.nodes[idx].updateLocalMatrix()
    }

    private func translateBone(
        _ bone: VRMHumanoidBone,
        on model: VRMModel,
        restTransforms: [Int: NodeRestTransform],
        offset: SIMD3<Float>
    ) {
        guard let idx = model.humanoid?.getBoneNode(bone),
              idx < model.nodes.count,
              let rest = restTransforms[idx] else { return }
        model.nodes[idx].translation = rest.pos + offset
        model.nodes[idx].updateLocalMatrix()
    }
}
