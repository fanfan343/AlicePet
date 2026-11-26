import Foundation
import VRMMetalKit

/// Continuous opposing-axis emotion state machine inspired by Plutchik's emotion wheel,
/// Google GoEmotions, and Animama's 24-emotion inertia model.
///
/// Each axis is a continuous value in `[-1, +1]` where the positive and negative poles are
/// opposing emotions (raising one naturally suppresses its opposite). All axes decay smoothly
/// back toward neutral over time so mood carries realistic conversational inertia.
@MainActor
final class EmotionStateMachine {
    struct Axis {
        let positive: String
        let negative: String
        var value: Float = 0.0 // [-1, +1]
    }

    struct Snapshot: Codable, Sendable {
        let dominantEmotion: String
        let dominantIntensity: Float
        let axes: [String: Float]
        let a2fEmotionPreset: String
    }

    private(set) var axes: [Axis] = [
        Axis(positive: "joy", negative: "sadness"),
        Axis(positive: "affection", negative: "rejection"),
        Axis(positive: "excitement", negative: "calm"),
        Axis(positive: "curiosity", negative: "boredom"),
        Axis(positive: "pride", negative: "embarrassment"),
        Axis(positive: "amusement", negative: "annoyance"),
        Axis(positive: "gratitude", negative: "anger"),
        Axis(positive: "hope", negative: "anxiety"),
    ]

    /// Half-life in seconds for passive exponential decay back toward neutral.
    var halfLifeSeconds: Float = 180.0

    /// Maps common synonyms / GoEmotions / VRM mood labels onto our 8 opposing axes.
    private static let aliasMap: [String: (axisIndex: Int, sign: Float)] = [
        // Axis 0: joy <-> sadness
        "joy": (0, +1), "happy": (0, +1), "cheerful": (0, +1), "delight": (0, +1),
        "sadness": (0, -1), "sad": (0, -1), "grief": (0, -1), "disappointment": (0, -1),
        // Axis 1: affection <-> rejection
        "affection": (1, +1), "love": (1, +1), "caring": (1, +1), "warm": (1, +1),
        "rejection": (1, -1), "disgust": (1, -1), "cold": (1, -1),
        // Axis 2: excitement <-> calm
        "excitement": (2, +1), "excited": (2, +1), "energetic": (2, +1),
        "calm": (2, -1), "relaxed": (2, -1), "peaceful": (2, -1), "serene": (2, -1),
        // Axis 3: curiosity <-> boredom
        "curiosity": (3, +1), "curious": (3, +1), "surprise": (3, +1), "surprised": (3, +1),
        "boredom": (3, -1), "bored": (3, -1), "tired": (3, -1),
        // Axis 4: pride <-> embarrassment
        "pride": (4, +1), "proud": (4, +1), "confident": (4, +1),
        "embarrassment": (4, -1), "embarrassed": (4, -1), "shy": (4, -1), "flustered": (4, -1),
        // Axis 5: amusement <-> annoyance
        "amusement": (5, +1), "amused": (5, +1), "playful": (5, +1), "fun": (5, +1),
        "annoyance": (5, -1), "annoyed": (5, -1), "frustrated": (5, -1),
        // Axis 6: gratitude <-> anger
        "gratitude": (6, +1), "grateful": (6, +1), "admiration": (6, +1),
        "anger": (6, -1), "angry": (6, -1), "mad": (6, -1),
        // Axis 7: hope <-> anxiety
        "hope": (7, +1), "optimism": (7, +1), "relief": (7, +1),
        "anxiety": (7, -1), "anxious": (7, -1), "fear": (7, -1), "nervous": (7, -1), "worried": (7, -1),
    ]

    /// Applies an emotional impulse. `intensity` accepts 0...1 (or 1...4 discrete levels, auto-normalized).
    @discardableResult
    func applyImpulse(emotion rawEmotion: String, intensity rawIntensity: Float) -> Snapshot {
        let key = rawEmotion.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let normIntensity: Float = (rawIntensity > 1.0)
            ? min(max(rawIntensity / 4.0, 0.0), 1.0)
            : min(max(rawIntensity, 0.0), 1.0)

        if key == "neutral" || key == "reset" {
            for i in axes.indices {
                axes[i].value *= 0.25
            }
            let snap = snapshot()
            AppLogger.shared.info("Emotion", "Soft reset toward neutral -> dominant=\(snap.dominantEmotion)(\(String(format: "%.2f", snap.dominantIntensity)))")
            return snap
        }

        if let mapping = Self.aliasMap[key] {
            let delta = mapping.sign * normIntensity * 0.75
            axes[mapping.axisIndex].value = min(max(axes[mapping.axisIndex].value + delta, -1.0), 1.0)
            // Mildly dampen unrelated axes so the new emotional shift feels focused
            for i in axes.indices where i != mapping.axisIndex {
                axes[i].value *= 0.88
            }
            let snap = snapshot()
            AppLogger.shared.info(
                "Emotion",
                "Impulse '\(key)' (\(String(format: "%.2f", normIntensity))) -> dominant=\(snap.dominantEmotion)(\(String(format: "%.2f", snap.dominantIntensity))) a2f=\(snap.a2fEmotionPreset)"
            )
            return snap
        }

        AppLogger.shared.warn("Emotion", "Unknown emotion label '\(rawEmotion)'; keeping current state")
        return snapshot()
    }

    /// Exponentially decays all emotion axes toward neutral over elapsed `dt` seconds.
    func tick(dt: TimeInterval) {
        guard dt > 0, halfLifeSeconds > 1 else { return }
        let decay = pow(0.5, Float(dt) / halfLifeSeconds)
        for i in axes.indices {
            axes[i].value *= decay
            if abs(axes[i].value) < 0.005 {
                axes[i].value = 0.0
            }
        }
    }

    var dominant: (name: String, intensity: Float) {
        var bestName = "neutral"
        var bestMag: Float = 0.0
        for axis in axes {
            let mag = abs(axis.value)
            if mag > bestMag {
                bestMag = mag
                bestName = axis.value >= 0 ? axis.positive : axis.negative
            }
        }
        if bestMag < 0.12 {
            return ("neutral", 0.0)
        }
        return (bestName, bestMag)
    }

    /// Maps dominant emotion onto VRM standard expression presets for idle/non-speaking posture.
    var recommendedVRMExpression: VRMExpressionPreset? {
        let (name, mag) = dominant
        guard mag >= 0.22 else { return nil }
        switch name {
        case "joy", "affection", "excitement", "amusement", "gratitude", "pride", "hope":
            return .happy
        case "sadness", "rejection", "embarrassment", "anxiety":
            return .sad
        case "anger", "annoyance":
            return .angry
        case "calm", "boredom":
            return .relaxed
        default:
            return nil
        }
    }

    /// Maps dominant emotion onto Audio2Face-3D Claire's native 10-emotion preset names.
    var recommendedA2FEmotion: String {
        let (name, mag) = dominant
        guard mag >= 0.20 else { return "neutral" }
        switch name {
        case "joy", "affection", "gratitude", "hope", "pride": return "joy"
        case "excitement", "amusement": return "amusement"
        case "curiosity": return "amazement"
        case "sadness", "boredom": return "sadness"
        case "anger", "annoyance": return "anger"
        case "rejection": return "disgust"
        case "anxiety", "embarrassment": return "fear"
        default: return "neutral"
        }
    }

    func snapshot() -> Snapshot {
        let (name, mag) = dominant
        var map: [String: Float] = [:]
        for axis in axes where abs(axis.value) >= 0.01 {
            let key = axis.value >= 0 ? axis.positive : axis.negative
            map[key] = (abs(axis.value) * 100).rounded() / 100
        }
        return Snapshot(
            dominantEmotion: name,
            dominantIntensity: (mag * 100).rounded() / 100,
            axes: map,
            a2fEmotionPreset: recommendedA2FEmotion
        )
    }
}
