import Foundation
import VRMMetalKit
import simd

/// Zero-token procedural Layer 1 animation overlays composed additively on top of `.authorPose`:
/// 1. **Emotional Posture Modulation**: reads `EmotionStateMachine` continuous axes so Alice's
///    spine, chest, shoulders, and head subtly reflect her lingering mood (joy lift, shy/sad droop,
///    curious head tilt) without any extra LLM tokens.
/// 2. **Procedural Talking-Hands Beats**: when speech audio is driving mouth movement (or during
///    headless verification), converts instantaneous mouth/jaw energy into smooth, organic anime
///    upper-body & arm/wrist gesture beats safely clamped away from body/skirt clipping.
@MainActor
final class SpeechGestureLayer {
    var enabled: Bool = true

    /// Optional deterministic override (`0...1`) used by headless CLI verification (`--test-speech-gesture`).
    var debugSpeechEnergyOverride: Float?

    private var smoothedSpeechEnergy: Float = 0.0
    private var beatPhase: Float = 0.0
    private var smoothedPitchLift: Float = 0.0
    private var smoothedHeadTilt: Float = 0.0
    private var smoothedShoulderDrop: Float = 0.0

    func reset() {
        smoothedSpeechEnergy = 0.0
        beatPhase = 0.0
        smoothedPitchLift = 0.0
        smoothedHeadTilt = 0.0
        smoothedShoulderDrop = 0.0
    }

    /// Applies emotional posture + talking-hands beats additively onto `model`.
    /// Strictly activates on `.authorPose` so it never fights ARDY, External VMC, or `.vrma` clips.
    func apply(
        to model: VRMModel,
        activeBodySource: AnimationMixer.BodyLayerSource,
        rawSpeechEnergy: Float,
        emotionState: EmotionStateMachine?,
        elapsed: Float,
        dt: TimeInterval
    ) {
        guard enabled, activeBodySource == .authorPose, let humanoid = model.humanoid else {
            // Smoothly decay envelopes while another full-body driver owns the rig
            smoothedSpeechEnergy *= 0.85
            return
        }

        let clampedDt = Float(min(max(dt, 0.0), 0.1))

        // 1. Emotional Posture Modulation (continuous exponential smoothing)
        var targetLift: Float = 0.0      // + = chest/head up (joy/pride/excitement), - = droop (sad/shy)
        var targetHeadTilt: Float = 0.0  // Z roll tilt (curiosity/amusement/affection)
        var targetShoulder: Float = 0.0  // subtle shoulder raise/drop

        if let emotionState {
            var axisByName: [String: Float] = [:]
            for axis in emotionState.axes {
                axisByName[axis.positive] = max(0, axis.value)
                axisByName[axis.negative] = max(0, -axis.value)
            }
            let joy = axisByName["joy"] ?? 0
            let excitement = axisByName["excitement"] ?? 0
            let pride = axisByName["pride"] ?? 0
            let affection = axisByName["affection"] ?? 0
            let curiosity = axisByName["curiosity"] ?? 0
            let amusement = axisByName["amusement"] ?? 0
            let sadness = axisByName["sadness"] ?? 0
            let embarrassment = axisByName["embarrassment"] ?? 0
            let anxiety = axisByName["anxiety"] ?? 0
            let boredom = axisByName["boredom"] ?? 0
            let anger = axisByName["anger"] ?? 0

            targetLift = (joy * 0.6 + excitement * 0.8 + pride * 0.7)
                - (sadness * 0.9 + embarrassment * 0.7 + anxiety * 0.5 + boredom * 0.4)
            targetHeadTilt = (curiosity * 1.0 + amusement * 0.55 + affection * 0.45)
                - (embarrassment * 0.45 + anger * 0.25)
            targetShoulder = (excitement * 0.5 + anxiety * 0.6 + anger * 0.4)
                - (sadness * 0.5 + boredom * 0.4)
        }

        let moodAlpha: Float = clampedDt > 0 ? (1.0 - exp(-clampedDt / 0.28)) : 1.0
        smoothedPitchLift += (min(max(targetLift, -1.0), 1.0) - smoothedPitchLift) * moodAlpha
        smoothedHeadTilt += (min(max(targetHeadTilt, -1.0), 1.0) - smoothedHeadTilt) * moodAlpha
        smoothedShoulderDrop += (min(max(targetShoulder, -1.0), 1.0) - smoothedShoulderDrop) * moodAlpha

        let moodMag = abs(smoothedPitchLift) + abs(smoothedHeadTilt) + abs(smoothedShoulderDrop)
        if moodMag > 0.015 {
            // Subtle, tasteful anime posture adjustments (max ~3.5° torso, ~6.5° head)
            applyAdditiveEuler(
                to: model, humanoid: humanoid, bone: .chest,
                pitchX: -smoothedPitchLift * 0.045,
                yawY: 0,
                rollZ: smoothedHeadTilt * 0.025
            )
            applyAdditiveEuler(
                to: model, humanoid: humanoid, bone: .head,
                pitchX: -smoothedPitchLift * 0.065,
                yawY: smoothedHeadTilt * 0.035,
                rollZ: smoothedHeadTilt * 0.105
            )
            if abs(smoothedShoulderDrop) > 0.05 {
                let shZ = smoothedShoulderDrop * 0.04
                applyAdditiveEuler(to: model, humanoid: humanoid, bone: .leftShoulder, pitchX: 0, yawY: 0, rollZ: shZ)
                applyAdditiveEuler(to: model, humanoid: humanoid, bone: .rightShoulder, pitchX: 0, yawY: 0, rollZ: -shZ)
            }
        }

        // 2. Procedural Talking-Hands Beats (attack-release speech envelope)
        let inputEnergy = min(max(debugSpeechEnergyOverride ?? rawSpeechEnergy, 0.0), 1.0)
        if clampedDt > 0 {
            let tau: Float = inputEnergy > smoothedSpeechEnergy ? 0.055 : 0.24
            let alpha = 1.0 - exp(-clampedDt / tau)
            smoothedSpeechEnergy += (inputEnergy - smoothedSpeechEnergy) * alpha
            beatPhase += clampedDt * (2.6 + smoothedSpeechEnergy * 2.2)
        } else {
            smoothedSpeechEnergy = inputEnergy
            beatPhase = elapsed * 3.4
        }

        guard smoothedSpeechEnergy > 0.025 else { return }

        let env = min(max((smoothedSpeechEnergy - 0.025) / 0.975, 0.0), 1.0)
        let smoothEnv = env * env * (3.0 - 2.0 * env)

        // Asymmetric organic beat waves (right hand leads slightly, left hand counter-phrases)
        let beatR = 0.55 + 0.45 * sin(beatPhase * 1.15 + 0.35 * sin(beatPhase * 0.47))
        let beatL = 0.50 + 0.50 * sin(beatPhase * 0.92 + 1.85 + 0.30 * cos(beatPhase * 0.39))
        let sway = sin(beatPhase * 0.65)

        // Gentle torso rhythm & chest emphasis
        applyAdditiveEuler(
            to: model, humanoid: humanoid, bone: .spine,
            pitchX: 0.020 * smoothEnv * (0.5 + 0.5 * sin(beatPhase)),
            yawY: 0.030 * smoothEnv * sway,
            rollZ: 0.018 * smoothEnv * sway
        )
        applyAdditiveEuler(
            to: model, humanoid: humanoid, bone: .chest,
            pitchX: -0.025 * smoothEnv * beatR,
            yawY: 0.025 * smoothEnv * sway,
            rollZ: 0
        )

        // Expressive upper-arm lift in parent/character space (swings arms forward & slightly outward, away from skirt)
        applyAdditiveEuler(
            to: model, humanoid: humanoid, bone: .rightUpperArm,
            pitchX: -0.22 * smoothEnv * (0.55 + 0.45 * beatR),
            yawY: 0.14 * smoothEnv * beatR,
            rollZ: -0.18 * smoothEnv * (0.60 + 0.40 * beatR),
            preMultiply: true
        )
        applyAdditiveEuler(
            to: model, humanoid: humanoid, bone: .leftUpperArm,
            pitchX: -0.18 * smoothEnv * (0.50 + 0.50 * beatL),
            yawY: -0.12 * smoothEnv * beatL,
            rollZ: 0.16 * smoothEnv * (0.60 + 0.40 * beatL),
            preMultiply: true
        )

        // Forearm/elbow gesture phrasing (bends elbows naturally forward in front of chest/waist)
        applyAdditiveEuler(
            to: model, humanoid: humanoid, bone: .rightLowerArm,
            pitchX: 0.08 * smoothEnv * beatR,
            yawY: 0.78 * smoothEnv * (0.55 + 0.45 * beatR),
            rollZ: -0.12 * smoothEnv * sin(beatPhase * 1.3)
        )
        applyAdditiveEuler(
            to: model, humanoid: humanoid, bone: .leftLowerArm,
            pitchX: 0.06 * smoothEnv * beatL,
            yawY: -0.64 * smoothEnv * (0.50 + 0.50 * beatL),
            rollZ: 0.10 * smoothEnv * cos(beatPhase * 1.1)
        )

        // Soft wrist accent flicks
        applyAdditiveEuler(
            to: model, humanoid: humanoid, bone: .rightHand,
            pitchX: 0.14 * smoothEnv * sin(beatPhase * 1.7),
            yawY: 0.12 * smoothEnv * beatR,
            rollZ: -0.14 * smoothEnv * beatR
        )
        applyAdditiveEuler(
            to: model, humanoid: humanoid, bone: .leftHand,
            pitchX: 0.12 * smoothEnv * cos(beatPhase * 1.5),
            yawY: -0.10 * smoothEnv * beatL,
            rollZ: 0.12 * smoothEnv * beatL
        )
    }

    private func applyAdditiveEuler(
        to model: VRMModel,
        humanoid: VRMHumanoid,
        bone: VRMHumanoidBone,
        pitchX: Float,
        yawY: Float,
        rollZ: Float,
        preMultiply: Bool = false
    ) {
        guard let idx = humanoid.getBoneNode(bone), idx < model.nodes.count else { return }
        let qx = simd_quatf(angle: pitchX, axis: SIMD3<Float>(1, 0, 0))
        let qy = simd_quatf(angle: yawY, axis: SIMD3<Float>(0, 1, 0))
        let qz = simd_quatf(angle: rollZ, axis: SIMD3<Float>(0, 0, 1))
        let delta = simd_normalize(qy * qx * qz)
        let node = model.nodes[idx]
        node.rotation = simd_normalize(preMultiply ? (delta * node.rotation) : (node.rotation * delta))
        node.updateLocalMatrix()
    }
}
