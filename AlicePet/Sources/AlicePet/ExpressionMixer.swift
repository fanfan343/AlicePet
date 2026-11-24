import AppKit
import Foundation
import VRMMetalKit
import simd

/// Layered expression controller keeping Outfit morphs, Mood/Recipe presets, ARKit face tracking,
/// and Auto-Blink/LookAt cleanly separated without fighting over weights.
@MainActor
final class ExpressionMixer {
    let faceDriver: ARKitFaceDriver

    private(set) var activePresetMood: VRMExpressionPreset?
    private(set) var activeCustomExpression: String?

    var autoBlinkEnabled: Bool = true
    var eyeTrackingEnabled: Bool = true

    private var blinkTimer: Timer?

    init() {
        var smoothing = SmoothingConfig.default
        smoothing.perExpression[ARKitFaceBlendShapes.eyeBlinkLeft] = .ema(alpha: 0.55)
        smoothing.perExpression[ARKitFaceBlendShapes.eyeBlinkRight] = .ema(alpha: 0.55)
        smoothing.perExpression[ARKitFaceBlendShapes.jawOpen] = .ema(alpha: 0.5)
        self.faceDriver = ARKitFaceDriver(mapper: .default, smoothing: smoothing)
    }

    func initializeForModel(_ model: VRMModel, renderer: VRMRenderer) {
        activePresetMood = nil
        activeCustomExpression = nil
        faceDriver.initializeForModel(model)
        if let lookAt = renderer.lookAtController {
            lookAt.enabled = true
            lookAt.smoothing = 0.15
            lookAt.saccadeEnabled = true
            lookAt.target = .camera
        }
    }

    func startAutoBlink(rendererProvider: @escaping @MainActor () -> VRMRenderer?, isCameraActive: @escaping @MainActor () -> Bool) {
        scheduleNextBlink(rendererProvider: rendererProvider, isCameraActive: isCameraActive)
    }

    private func scheduleNextBlink(
        rendererProvider: @escaping @MainActor () -> VRMRenderer?,
        isCameraActive: @escaping @MainActor () -> Bool
    ) {
        blinkTimer?.invalidate()
        let interval = Double.random(in: 2.6...5.4)
        blinkTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                if self.autoBlinkEnabled && !isCameraActive() {
                    rendererProvider()?.expressionController?.blink(duration: 0.11)
                }
                self.scheduleNextBlink(rendererProvider: rendererProvider, isCameraActive: isCameraActive)
            }
        }
    }

    // MARK: - Mood & Recipe Expressions (Layer 1)

    func setPresetMood(_ preset: VRMExpressionPreset?, on renderer: VRMRenderer) {
        clearCustomExpression(on: renderer)
        activePresetMood = preset
        if let preset {
            renderer.expressionController?.setMood(preset, intensity: 1.0)
        } else {
            renderer.expressionController?.setMood(.neutral, intensity: 0.0)
        }
    }

    func setCustomExpression(_ name: String, weight: Float = 1.0, on renderer: VRMRenderer) {
        clearCustomExpression(on: renderer)
        activePresetMood = nil
        renderer.expressionController?.setMood(.neutral, intensity: 0.0)
        renderer.setExpression(name, weight: weight)
        activeCustomExpression = name
    }

    /// Resolves standard VRM presets (`happy`, `angry`, `sad`, `relaxed`, `surprised`, `neutral`) first,
    /// then curated recipes (case-insensitive), and finally raw custom expression names.
    func applyExpression(named query: String, weight: Float = 1.0, on renderer: VRMRenderer, recipeStore: AssetRecipeStore) {
        let lower = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let presetMap: [String: VRMExpressionPreset?] = [
            "neutral": nil,
            "happy": .happy,
            "joy": .happy,
            "angry": .angry,
            "sad": .sad,
            "sorrow": .sad,
            "relaxed": .relaxed,
            "fun": .relaxed,
            "surprised": .surprised,
        ]
        if let matched = presetMap[lower] {
            setPresetMood(matched, on: renderer)
            return
        }
        if let entry = recipeStore.availableFaceRecipes.first(where: {
            $0.id.lowercased().contains(lower) || $0.title.lowercased().contains(lower)
        }) {
            setCustomExpression(entry.id, weight: weight, on: renderer)
            return
        }
        setCustomExpression(query, weight: weight, on: renderer)
    }

    func clearCustomExpression(on renderer: VRMRenderer) {
        if let prev = activeCustomExpression {
            renderer.setExpression(prev, weight: 0.0)
            activeCustomExpression = nil
        }
    }

    // MARK: - VMC / ARKit Tracking (Layer 2)

    func prepareForTrackingStart(on renderer: VRMRenderer) {
        clearCustomExpression(on: renderer)
        activePresetMood = nil
        renderer.expressionController?.setMood(.neutral, intensity: 0.0)
    }

    func resetAfterTrackingStop(on renderer: VRMRenderer) {
        if let exprCtrl = renderer.expressionController {
            let zeros = Dictionary(uniqueKeysWithValues: ARKitFaceBlendShapes.allKeys.map { ($0, Float(0)) })
            faceDriver.update(
                blendShapes: ARKitFaceBlendShapes(timestamp: Date().timeIntervalSinceReferenceDate, shapes: zeros),
                controller: exprCtrl,
                maxAge: 1.0
            )
        }
        if !eyeTrackingEnabled {
            renderer.lookAtController?.target = .forward
        }
    }

    // MARK: - LookAt (Layer 3)

    func updateLookAt(
        on renderer: VRMRenderer,
        poseMode: PetPoseMode,
        window: NSWindow?
    ) {
        guard let lookAt = renderer.lookAtController else { return }

        if poseMode == .afkBook {
            lookAt.target = .headLocalPoint(SIMD3<Float>(0, -0.35, 1.0))
            return
        }
        if eyeTrackingEnabled, let window {
            let mouseScreen = NSEvent.mouseLocation
            let winFrame = window.frame
            let normX = Float(((mouseScreen.x - winFrame.midX) / max(winFrame.width, 1)).clamped(to: -1.5...1.5))
            let normY = Float(((mouseScreen.y - winFrame.midY) / max(winFrame.height, 1)).clamped(to: -1.5...1.5))
            lookAt.target = .headLocalPoint(SIMD3<Float>(normX * 0.85, normY * 0.6, 1.5))
        }
    }
}

private extension Comparable {
    func clamped(to limits: ClosedRange<Self>) -> Self {
        min(max(self, limits.lowerBound), limits.upperBound)
    }
}
