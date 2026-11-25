import AppKit
import Foundation
import VRMMetalKit
import simd

/// Layered real-time skeletal & facial animation mixer for AlicePet.
///
/// Replaces single-port flat VMC overwrites with isolated per-engine UDP channels and
/// three deterministic compositing layers evaluated every frame:
///   - Layer 0 (Base Full-Body): Exclusive winner among ARDY / External VMC / Author Pose
///     with smoothstep 0.35s SLERP crossfading on every source switch.
///   - Layer 1 (Additive Speech Micro-Motion): Composes LAM Audio2Expression Neck/Head nods additively
///     on top of ANY active full-body base layer without fighting or overwriting body orientation.
///   - Layer 2 (Facial BlendShapes & Gaze): Routes 52-channel ARKit PerfectSync from LAM or External VMC
///     independently from body motion, keeping Auto-Blink & Mouse LookAt alive during body-only AI streams.
@MainActor
final class AnimationMixer {
    enum BodyLayerSource: String {
        case authorPose = "AuthorPose"
        case externalVMC = "ExternalVMC"
        case ardy = "ARDY"

        var displayLabel: String {
            switch self {
            case .authorPose: return "作者姿态"
            case .externalVMC: return "外部/精捕 VMC"
            case .ardy: return "ARDY Mini"
            }
        }
    }

    enum FaceLayerSource: String {
        case procedural = "Procedural"
        case externalVMC = "ExternalVMC"
        case audioFace = "Audio2Face"
    }

    /// Isolated UDP channels so concurrent AI engines never contend over a single VMC port.
    let externalChannel = VMCTrackingController()
    let ardyChannel = VMCTrackingController()
    let audioFaceChannel = VMCTrackingController()

    /// Additive blend weight for LAM speech head nods when layered over an AI/VMC full-body base.
    var additiveHeadWeightOverFullBody: Float = 0.65
    /// Smoothstep SLERP transition duration when Layer 0 switches between body sources.
    var crossfadeDuration: TimeInterval = 0.35

    /// Preferred full-body source when multiple body channels happen to have fresh frames simultaneously.
    var preferredFullBodySource: BodyLayerSource?

    private(set) var activeBodySource: BodyLayerSource = .authorPose
    private(set) var activeFaceSource: FaceLayerSource = .procedural

    var onMixerStateChanged: (() -> Void)?

    private var lastBaseBodyRotations: [VRMHumanoidBone: simd_quatf] = [:]
    private var lastBaseBodyHipsPos: SIMD3<Float>?
    private var crossfadeFromRotations: [VRMHumanoidBone: simd_quatf] = [:]
    private var crossfadeFromHipsPos: SIMD3<Float>?
    private var crossfadeElapsed: TimeInterval = 0

    /// True whenever an external or LAM stream is actively driving facial blendshapes (pauses procedural Auto-Blink).
    var isFaceCaptured: Bool {
        audioFaceChannel.hasFreshFrame
            || (externalChannel.hasFreshFrame && externalChannel.receivedBlendShapeCount > 0)
    }

    /// True whenever any internal or external VMC channel is actively receiving frames.
    var isAnyStreamReceiving: Bool {
        externalChannel.hasFreshFrame
            || ardyChannel.hasFreshFrame
            || audioFaceChannel.hasFreshFrame
    }

    init() {
        // Internal AI body engines keep hips anchored by default so Alice never walks off the desktop window
        ardyChannel.appliesHipsPosition = false
        audioFaceChannel.appliesHipsPosition = false
    }

    func resetCrossfadeState() {
        lastBaseBodyRotations.removeAll()
        lastBaseBodyHipsPos = nil
        crossfadeFromRotations.removeAll()
        crossfadeFromHipsPos = nil
        crossfadeElapsed = 0
        activeBodySource = .authorPose
        activeFaceSource = .procedural
    }

    func stopAllChannels(renderer: VRMRenderer? = nil, expressionMixer: ExpressionMixer? = nil) {
        externalChannel.stop(renderer: renderer, expressionMixer: expressionMixer)
        ardyChannel.stop()
        audioFaceChannel.stop(renderer: renderer, expressionMixer: expressionMixer)
        resetCrossfadeState()
    }

    /// Evaluates all 3 animation layers onto `model` and `renderer` for the current frame.
    func evaluate(
        model: VRMModel,
        renderer: VRMRenderer,
        poseController: PoseController,
        recipeStore: AssetRecipeStore,
        expressionMixer: ExpressionMixer,
        elapsed: Float,
        dt: TimeInterval,
        idleBreathingEnabled: Bool,
        poseMode: PetPoseMode,
        window: NSWindow?
    ) {
        // MARK: Layer 0 — Base Full-Body Arbitration + SLERP Crossfade
        let nextBody = resolveActiveBodySource()
        if nextBody != activeBodySource {
            if !lastBaseBodyRotations.isEmpty {
                crossfadeFromRotations = lastBaseBodyRotations
                crossfadeFromHipsPos = lastBaseBodyHipsPos
                crossfadeElapsed = 0
            }
            AppLogger.shared.info("Mixer", "Layer 0 Body: \(activeBodySource.rawValue) -> \(nextBody.rawValue) (SLERP \(String(format: "%.2fs", crossfadeDuration)))")
            activeBodySource = nextBody
            onMixerStateChanged?()
        }

        switch activeBodySource {
        case .authorPose:
            poseController.applyProceduralFrame(
                to: model,
                restTransforms: poseController.restNodeTransforms,
                elapsed: elapsed,
                idleBreathingEnabled: idleBreathingEnabled
            )
        case .externalVMC:
            poseController.applyBasePose(to: model, restTransforms: poseController.restNodeTransforms)
            poseController.applyHandGestureOverrides(to: model, recipeStore: recipeStore)
            _ = externalChannel.applyBonesOnly(to: model)
        case .ardy:
            poseController.applyBasePose(to: model, restTransforms: poseController.restNodeTransforms)
            poseController.applyHandGestureOverrides(to: model, recipeStore: recipeStore)
            _ = ardyChannel.applyBonesOnly(to: model)
        }

        // Smoothstep SLERP crossfade between previous Layer 0 pose and incoming Layer 0 pose
        if !crossfadeFromRotations.isEmpty, let humanoid = model.humanoid {
            crossfadeElapsed += min(max(dt, 0.0), 0.1)
            let rawT = Float(min(crossfadeElapsed / max(crossfadeDuration, 0.01), 1.0))
            let alpha = rawT * rawT * (3.0 - 2.0 * rawT)

            for (bone, fromQ) in crossfadeFromRotations {
                guard let idx = humanoid.getBoneNode(bone), idx < model.nodes.count else { continue }
                let targetQ = model.nodes[idx].rotation
                model.nodes[idx].rotation = simd_normalize(simd_slerp(fromQ, targetQ, alpha))
                model.nodes[idx].updateLocalMatrix()
            }
            if let fromHips = crossfadeFromHipsPos,
               let hipsIdx = humanoid.getBoneNode(.hips), hipsIdx < model.nodes.count {
                let targetHips = model.nodes[hipsIdx].translation
                model.nodes[hipsIdx].translation = simd_mix(fromHips, targetHips, SIMD3<Float>(repeating: alpha))
                model.nodes[hipsIdx].updateLocalMatrix()
            }
            if rawT >= 1.0 {
                crossfadeFromRotations.removeAll()
                crossfadeFromHipsPos = nil
            }
        }

        // Snapshot pure Layer 0 pose (before Layer 1 additive nods) for future crossfades
        captureBaseBodyPose(from: model)

        // MARK: Layer 1 — Additive Speech Micro-Motion (Neck + Head)
        if audioFaceChannel.hasFreshFrame {
            let additiveWeight: Float = (activeBodySource == .authorPose) ? 1.0 : additiveHeadWeightOverFullBody
            _ = audioFaceChannel.applyBonesAdditive(
                to: model,
                bones: [.neck, .head],
                weight: additiveWeight
            )
        }

        model.updateNodeTransforms()

        // MARK: Layer 2 — Facial BlendShapes & LookAt / Auto-Blink Coordination
        let nextFace: FaceLayerSource
        if audioFaceChannel.hasFreshFrame {
            externalChannel.clearBlendShapesIfNeeded(on: renderer, expressionMixer: expressionMixer)
            _ = audioFaceChannel.applyBlendShapesOnly(on: renderer, expressionMixer: expressionMixer)
            if renderer.lookAtController?.enabled == true {
                renderer.lookAtController?.enabled = false
            }
            nextFace = .audioFace
        } else if externalChannel.hasFreshFrame && externalChannel.receivedBlendShapeCount > 0 {
            audioFaceChannel.clearBlendShapesIfNeeded(on: renderer, expressionMixer: expressionMixer)
            _ = externalChannel.applyBlendShapesOnly(on: renderer, expressionMixer: expressionMixer)
            if renderer.lookAtController?.enabled == true {
                renderer.lookAtController?.enabled = false
            }
            nextFace = .externalVMC
        } else {
            audioFaceChannel.clearBlendShapesIfNeeded(on: renderer, expressionMixer: expressionMixer)
            externalChannel.clearBlendShapesIfNeeded(on: renderer, expressionMixer: expressionMixer)
            if renderer.lookAtController?.enabled == false {
                renderer.lookAtController?.enabled = true
            }
            expressionMixer.updateLookAt(on: renderer, poseMode: poseMode, window: window)
            nextFace = .procedural
        }

        if nextFace != activeFaceSource {
            AppLogger.shared.info("Mixer", "Layer 2 Face: \(activeFaceSource.rawValue) -> \(nextFace.rawValue)")
            activeFaceSource = nextFace
            onMixerStateChanged?()
        }
    }

    private func resolveActiveBodySource() -> BodyLayerSource {
        let hasArdy = ardyChannel.hasFreshFrame && ardyChannel.receivedBoneCount > 0
        let hasExt = externalChannel.hasFreshFrame && externalChannel.receivedBoneCount > 0

        if let preferred = preferredFullBodySource {
            switch preferred {
            case .ardy where hasArdy: return .ardy
            case .externalVMC where hasExt: return .externalVMC
            default: break
            }
        }

        if hasArdy { return .ardy }
        if hasExt { return .externalVMC }
        return .authorPose
    }

    private func captureBaseBodyPose(from model: VRMModel) {
        guard let humanoid = model.humanoid else { return }
        for (bone, ref) in humanoid.humanBones where ref.node < model.nodes.count {
            lastBaseBodyRotations[bone] = model.nodes[ref.node].rotation
        }
        if let hipsIdx = humanoid.getBoneNode(.hips), hipsIdx < model.nodes.count {
            lastBaseBodyHipsPos = model.nodes[hipsIdx].translation
        }
    }
}
