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
        idleBreathingEnabled: Bool
    )
}

struct NodeRestTransform {
    let pos: SIMD3<Float>
    let rot: simd_quatf
    let scale: SIMD3<Float>
}

/// Manages authentic author-authored Unity muscle clips (with procedural fallback for generic VRMs),
/// hand gesture overrides, hair style bone adjustments, and per-frame animation playback.
@MainActor
final class PoseController: MotionSource {
    let animationPlayer = AuthorAnimationPlayer()

    private(set) var poseMode: PetPoseMode = .stand
    private(set) var leftHandGesture: HandGesturePreset = .natural
    private(set) var rightHandGesture: HandGesturePreset = .natural

    private(set) var restNodeTransforms: [Int: NodeRestTransform] = [:]
    private(set) var hairNodeIndices: [String: Int] = [:]
    private var poseStartTime: Float = 0
    private var playAFKIntro: Bool = false

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

    func loadAuthorAnimations(from folderURL: URL?) {
        animationPlayer.load(from: folderURL)
    }

    func setPoseMode(_ mode: PetPoseMode, at elapsed: Float = 0, playIntro: Bool = true) {
        let wasAFK = (poseMode == .afkBook || poseMode == .afkTea)
        let willAFK = (mode == .afkBook || mode == .afkTea)
        poseMode = mode
        poseStartTime = elapsed
        playAFKIntro = playIntro && willAFK && !wasAFK && (animationPlayer.clip(named: "afk_in") != nil)
    }

    func setHandGesture(_ preset: HandGesturePreset, side: HandSide) {
        switch side {
        case .left: leftHandGesture = preset
        case .right: rightHandGesture = preset
        case .both:
            leftHandGesture = preset
            rightHandGesture = preset
        }
    }

    func applyAll(
        to model: VRMModel,
        recipeStore: AssetRecipeStore,
        profile: ModelProfile,
        hairStyle: HairStyleOption,
        at animTime: Float = 0
    ) {
        applyAuthorOrFallbackPose(to: model, restTransforms: restNodeTransforms, clipTime: animTime)
        applyHandGestureOverrides(to: model, recipeStore: recipeStore)
        applyHairStyle(hairStyle, profile: profile, to: model)
        model.updateNodeTransforms()
    }

    // MARK: - MotionSource

    func applyBasePose(to model: VRMModel, restTransforms: [Int: NodeRestTransform]) {
        applyAuthorOrFallbackPose(to: model, restTransforms: restTransforms, clipTime: 0)
    }

    func applyProceduralFrame(
        to model: VRMModel,
        restTransforms: [Int: NodeRestTransform],
        elapsed t: Float,
        idleBreathingEnabled: Bool
    ) {
        let localTime = max(0, t - poseStartTime)

        // If entering AFK interactively, play Alice_AFK_In (5s) first then seamlessly transition to Alice_AFK_Loop
        if playAFKIntro, let introClip = animationPlayer.clip(named: "afk_in") {
            if localTime < introClip.duration {
                applyAuthorClip(introClip, at: localTime, to: model, restTransforms: restTransforms)
            } else {
                playAFKIntro = false
                poseStartTime = t
                if let loopClip = activeAuthorClip() {
                    applyAuthorClip(loopClip, at: 0, to: model, restTransforms: restTransforms)
                }
            }
        } else if let clip = activeAuthorClip(), clip.duration > 0.05 {
            applyAuthorClip(clip, at: localTime, to: model, restTransforms: restTransforms)
        }

        let breath = sin(t * 1.7)
        if idleBreathingEnabled {
            // Add subtle breathing modulation on top of the current chest pose without overwriting it
            if let chestIdx = model.humanoid?.getBoneNode(.chest), chestIdx < model.nodes.count {
                if activeAuthorClip()?.duration ?? 0 <= 0.05 {
                    // Static clip: recompute base chest rotation first so breathing doesn't accumulate
                    applyAuthorOrFallbackPose(to: model, restTransforms: restTransforms, clipTime: 0)
                }
                let breathQ = simd_quatf(angle: breath * 0.012, axis: SIMD3<Float>(1, 0, 0))
                model.nodes[chestIdx].rotation = model.nodes[chestIdx].rotation * breathQ
                model.nodes[chestIdx].updateLocalMatrix()
            }
        }

        if poseMode != .afkBook && poseMode != .afkTea && idleBreathingEnabled {
            let sway = sin(t * 0.75)
            if let headIdx = model.humanoid?.getBoneNode(.head), headIdx < model.nodes.count {
                let swayQ = simd_quatf(angle: sway * 0.016, axis: SIMD3<Float>(0, 0, 1))
                    * simd_quatf(angle: breath * 0.008, axis: SIMD3<Float>(1, 0, 0))
                model.nodes[headIdx].rotation = model.nodes[headIdx].rotation * swayQ
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

    // MARK: - Author Clip Playback vs Generic Fallback

    private func activeAuthorClip() -> AuthorAnimationClip? {
        if let exact = animationPlayer.clip(named: poseMode.authorClipID) {
            return exact
        }
        if poseMode == .sit {
            return animationPlayer.clip(named: "sit")
        }
        return nil
    }

    private func applyAuthorOrFallbackPose(to model: VRMModel, restTransforms: [Int: NodeRestTransform], clipTime: Float) {
        if let clip = activeAuthorClip() {
            applyAuthorClip(clip, at: clipTime, to: model, restTransforms: restTransforms)
            return
        }
        applyGenericFallbackPose(to: model, restTransforms: restTransforms)
    }

    private func applyAuthorClip(
        _ clip: AuthorAnimationClip,
        at time: Float,
        to model: VRMModel,
        restTransforms: [Int: NodeRestTransform]
    ) {
        guard let sampled = animationPlayer.sample(clip: clip, at: time) else { return }
        let verticalBias: Float = switch poseMode {
        case .afkBook, .afkTea: 0.14
        case .prone: 0.34
        case .crouch: 0.12
        default: 0.0
        }
        let clipHasOnlyNaturalHands = (leftHandGesture == .natural && rightHandGesture == .natural)
        UnityMuscleRetargeter.apply(
            pose: sampled,
            standReferenceRootT: animationPlayer.standReferenceRootT,
            clipInitialRootZ: animationPlayer.initialRootZ(of: clip),
            verticalBias: verticalBias,
            to: model,
            restTransforms: restTransforms,
            applyFingers: clipHasOnlyNaturalHands
        )
    }

    private func applyGenericFallbackPose(to model: VRMModel, restTransforms: [Int: NodeRestTransform]) {
        let bodyBones: [VRMHumanoidBone] = [
            .hips, .spine, .chest, .upperChest, .neck, .head,
            .leftShoulder, .leftUpperArm, .leftLowerArm, .leftHand,
            .rightShoulder, .rightUpperArm, .rightLowerArm, .rightHand,
            .leftUpperLeg, .leftLowerLeg, .leftFoot, .leftToes,
            .rightUpperLeg, .rightLowerLeg, .rightFoot, .rightToes
        ]
        resetHumanoidBones(bodyBones, on: model, restTransforms: restTransforms)
        rotateBone(.leftUpperArm, on: model, restTransforms: restTransforms, roll: -64)
        rotateBone(.rightUpperArm, on: model, restTransforms: restTransforms, roll: 64)
        rotateBone(.leftLowerArm, on: model, restTransforms: restTransforms, yaw: -14)
        rotateBone(.rightLowerArm, on: model, restTransforms: restTransforms, yaw: 14)
    }

    // MARK: - Hand Overrides & Hair

    func applyHandGestureOverrides(
        to model: VRMModel,
        recipeStore: AssetRecipeStore,
        applyRelaxedNatural: Bool = false
    ) {
        let leftFingers: [VRMHumanoidBone] = [
            .leftThumbMetacarpal, .leftThumbProximal, .leftThumbDistal,
            .leftIndexProximal, .leftIndexIntermediate, .leftIndexDistal,
            .leftMiddleProximal, .leftMiddleIntermediate, .leftMiddleDistal,
            .leftRingProximal, .leftRingIntermediate, .leftRingDistal,
            .leftLittleProximal, .leftLittleIntermediate, .leftLittleDistal,
        ]
        if leftHandGesture != .natural || applyRelaxedNatural {
            resetHumanoidBones(leftFingers, on: model, restTransforms: restNodeTransforms)
            let rots = (leftHandGesture != .natural)
                ? recipeStore.fingerRotations(for: leftHandGesture, isLeft: true)
                : AssetRecipeStore.relaxedNaturalFingerRotations(isLeft: true)
            for (bone, q) in rots {
                if let idx = model.humanoid?.getBoneNode(bone), idx < model.nodes.count, let rest = restNodeTransforms[idx] {
                    model.nodes[idx].rotation = rest.rot * q
                    model.nodes[idx].updateLocalMatrix()
                }
            }
        }
        let rightFingers: [VRMHumanoidBone] = [
            .rightThumbMetacarpal, .rightThumbProximal, .rightThumbDistal,
            .rightIndexProximal, .rightIndexIntermediate, .rightIndexDistal,
            .rightMiddleProximal, .rightMiddleIntermediate, .rightMiddleDistal,
            .rightRingProximal, .rightRingIntermediate, .rightRingDistal,
            .rightLittleProximal, .rightLittleIntermediate, .rightLittleDistal,
        ]
        if rightHandGesture != .natural || applyRelaxedNatural {
            resetHumanoidBones(rightFingers, on: model, restTransforms: restNodeTransforms)
            let rots = (rightHandGesture != .natural)
                ? recipeStore.fingerRotations(for: rightHandGesture, isLeft: false)
                : AssetRecipeStore.relaxedNaturalFingerRotations(isLeft: false)
            for (bone, q) in rots {
                if let idx = model.humanoid?.getBoneNode(bone), idx < model.nodes.count, let rest = restNodeTransforms[idx] {
                    model.nodes[idx].rotation = rest.rot * q
                    model.nodes[idx].updateLocalMatrix()
                }
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

    private func resetHumanoidBones(_ bones: [VRMHumanoidBone], on model: VRMModel, restTransforms: [Int: NodeRestTransform]) {
        for bone in bones {
            if let idx = model.humanoid?.getBoneNode(bone), idx < model.nodes.count, let rest = restTransforms[idx] {
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
        guard let idx = model.humanoid?.getBoneNode(bone), idx < model.nodes.count, let rest = restTransforms[idx] else { return }
        let q = simd_quatf(angle: yaw * .pi / 180.0, axis: SIMD3<Float>(0, 1, 0))
            * simd_quatf(angle: pitch * .pi / 180.0, axis: SIMD3<Float>(1, 0, 0))
            * simd_quatf(angle: roll * .pi / 180.0, axis: SIMD3<Float>(0, 0, 1))
        model.nodes[idx].rotation = q * rest.rot
        model.nodes[idx].updateLocalMatrix()
    }
}
