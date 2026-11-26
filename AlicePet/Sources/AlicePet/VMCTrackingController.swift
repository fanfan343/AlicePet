import Darwin
import Foundation
import VRMMetalKit
import simd

/// Channel mask controlling which ARKit facial regions a VMC stream is allowed to drive.
/// Keeps speech lip-sync decoupled from eye LookAt and procedural Auto-Blink.
struct VMCFaceChannelMask: OptionSet, Sendable {
    let rawValue: UInt8

    static let lowerFace = VMCFaceChannelMask(rawValue: 1 << 0)    // jaw*, mouth*, tongueOut, noseSneer*
    static let upperFace = VMCFaceChannelMask(rawValue: 1 << 1)    // brow*, cheek*, eyeSquint*, eyeWide*
    static let eyesAndBlink = VMCFaceChannelMask(rawValue: 1 << 2) // eyeBlink*, eyeLook*

    static let speechAudio: VMCFaceChannelMask = [.lowerFace, .upperFace]
    static let all: VMCFaceChannelMask = [.lowerFace, .upperFace, .eyesAndBlink]

    func allows(arkitKey: String) -> Bool {
        if self == .all { return true }
        if arkitKey.hasPrefix("eyeBlink") || arkitKey.hasPrefix("eyeLook") {
            return contains(.eyesAndBlink)
        }
        if arkitKey.hasPrefix("brow") || arkitKey.hasPrefix("cheek")
            || arkitKey.hasPrefix("eyeSquint") || arkitKey.hasPrefix("eyeWide") {
            return contains(.upperFace)
        }
        return contains(.lowerFace)
    }
}

/// Receives VMC / OSC UDP packets (e.g. from iPhone TrueDepth apps like Waidayo / iFacialMocap
/// or desktop trackers like XR Animator / VSeeFace) and drives bones + 52 ARKit PerfectSync /
/// VRM preset expressions cleanly without fighting idle animations or outfit morphs.
@MainActor
final class VMCTrackingController {
    static let defaultPort: UInt16 = 39539
    static let secondaryPort: UInt16 = 39540
    static let defaultMaxAge: TimeInterval = 1.0

    let driver: VMCDriver
    private var receiver: VMCReceiver?
    private var statusTimer: Timer?

    private(set) var isListening: Bool = false
    private(set) var port: UInt16 = VMCTrackingController.defaultPort
    private(set) var boundPort: UInt16?
    private(set) var isReceivingFrames: Bool = false

    var appliesHipsPosition: Bool {
        get { driver.appliesHipsPosition }
        set { driver.appliesHipsPosition = newValue }
    }
    /// Enables proportional 2-Bone Arm IK & torso/head anti-clipping whenever incoming VMC packets
    /// include non-zero FK joint positions (e.g. ARDY Mini); automatically no-ops on rotation-only streams.
    var enablesProportionalArmIK: Bool = true

    var packetCount: Int { receiver?.packetCount ?? 0 }
    var decodeErrorCount: Int { receiver?.decodeErrorCount ?? 0 }
    var receivedBoneCount: Int { driver.frame.bones.count }
    var receivedBlendShapeCount: Int { driver.frame.blendShapes.count }

    var onStatusChanged: (() -> Void)?
    var onError: ((String) -> Void)?

    private var appliedVMCPresets: Set<VRMExpressionPreset> = []
    private var appliedVMCCustomKeys: Set<String> = []
    private var appliedAnimeBridgeWeights: [String: Float] = [:]
    private var hasAppliedARKitShapes: Bool = false
    private var wasDrivingStream: Bool = false

    /// Latest continuous world-space root XZ position (VRM meters) received via `/VMC/Ext/Root/Pos`,
    /// used by Picture-World mode to walk Alice naturally across the 3D room floor.
    var latestWorldRootXZ: SIMD2<Float>? {
        guard isListening, hasFreshFrame, let root = driver.frame.root else { return nil }
        let p = driver.convention.convert(position: root.position)
        return SIMD2<Float>(p.x, p.z)
    }

    /// Instantaneous speech energy in `[0, 1]` derived from live mouth/jaw blendshapes
    /// (used by `SpeechGestureLayer` to drive procedural talking-hands beats).
    var speechEnergy: Float {
        guard isListening, hasFreshFrame else { return 0 }
        let shapes = driver.frame.blendShapes
        guard !shapes.isEmpty else { return 0 }
        var peak: Float = 0
        for (rawKey, val) in shapes {
            let k = rawKey.lowercased()
            if k == "jawopen" || k == "aa" || k == "a" || k == "oh" || k == "o" || k == "mouthopen" {
                peak = max(peak, val)
            }
        }
        return min(max(peak, 0), 1)
    }
    private(set) var lastArmIKDiagnostics: [ArmIKRetargeter.ArmDiagnostics] = []
    private var restWorldCacheModelID: ObjectIdentifier?
    private var restWorldCache: [VRMHumanoidBone: simd_quatf] = [:]

    /// Case-insensitive + `_L`/`_R` suffix lookup table mapping sender blendshape names
    /// (e.g. `EyeBlinkLeft`, `eyeBlinkLeft`, `eyeBlink_L`) to canonical `ARKitFaceBlendShapes` keys.
    private static let arkitKeyLookup: [String: String] = {
        var map: [String: String] = [:]
        for key in ARKitFaceBlendShapes.allKeys {
            map[key.lowercased()] = key
            if key.hasSuffix("Left") {
                let stem = String(key.dropLast(4)).lowercased()
                map["\(stem)_l"] = key
            } else if key.hasSuffix("Right") {
                let stem = String(key.dropLast(5)).lowercased()
                map["\(stem)_r"] = key
            }
        }
        return map
    }()

    /// VRM standard presets that overlap directly with ARKit 52 facial muscles.
    /// Skipped when the sender provides ARKit 52 channels on a full-PerfectSync model to avoid double-morphing.
    private static let arkitOverlappingPresets: Set<VRMExpressionPreset> = [
        .aa, .ih, .ou, .ee, .oh,
        .blink, .blinkLeft, .blinkRight,
        .lookUp, .lookDown, .lookLeft, .lookRight,
    ]

    init() {
        let driver = VMCDriver(convention: .flipX)
        driver.appliesHipsPosition = false
        driver.maxAge = Self.defaultMaxAge
        self.driver = driver
    }

    /// Starts UDP listening on the requested port (`0` selects an ephemeral port for testing).
    /// Returns `true` if the UDP socket bound successfully.
    @discardableResult
    func start(port: UInt16 = VMCTrackingController.defaultPort, waitReadyTimeout: TimeInterval = 1.5) -> Bool {
        stop()

        self.port = port
        driver.reset()
        driver.maxAge = Self.defaultMaxAge

        let receiver = VMCReceiver(port: port, driver: driver)
        receiver.onError = { [weak self] error in
            let message = error.localizedDescription
            Task { @MainActor [weak self] in
                guard let self, self.isListening else { return }
                AppLogger.shared.error("VMC", "UDP receiver error: \(message)")
                // Only tear down and alert on fatal listener errors (not isolated malformed datagrams)
                if case VMCReceiver.ReceiverError.listenerFailed = error {
                    self.stop()
                    self.onError?(message)
                }
            }
        }

        do {
            try receiver.start()
        } catch {
            let msg = "无法启动 VMC 监听 (UDP \(port))：\(error.localizedDescription)"
            AppLogger.shared.error("VMC", msg)
            onError?(msg)
            return false
        }

        let ready = receiver.waitUntilReady(timeout: waitReadyTimeout)
        guard ready, let actualPort = receiver.boundPort else {
            receiver.stop()
            let msg = "VMC UDP 端口 \(port) 绑定超时或被其他应用占用，请检查是否已有 VMC 接收端在运行或授予本地网络权限。"
            AppLogger.shared.error("VMC", msg)
            onError?(msg)
            return false
        }

        self.receiver = receiver
        self.boundPort = actualPort
        self.port = actualPort
        self.isListening = true
        self.isReceivingFrames = false
        self.wasDrivingStream = false

        startStatusTimer()
        let lanIP = Self.primaryLocalIPv4() ?? "127.0.0.1"
        AppLogger.shared.info("VMC", "Listening for VMC/OSC on UDP \(lanIP):\(actualPort) (appliesHipsPosition=\(appliesHipsPosition))")
        onStatusChanged?()
        return true
    }

    func stop(renderer: VRMRenderer? = nil, expressionMixer: ExpressionMixer? = nil) {
        let wasListening = isListening
        statusTimer?.invalidate()
        statusTimer = nil

        receiver?.stop()
        receiver = nil
        driver.reset()

        isListening = false
        boundPort = nil
        isReceivingFrames = false
        wasDrivingStream = false

        if let renderer, let expressionMixer {
            clearAppliedExpressions(on: renderer, expressionMixer: expressionMixer)
            renderer.lookAtController?.enabled = true
        } else {
            appliedVMCPresets.removeAll()
            appliedVMCCustomKeys.removeAll()
            hasAppliedARKitShapes = false
        }

        if wasListening {
            AppLogger.shared.info("VMC", "Stopped VMC/OSC UDP receiver")
            onStatusChanged?()
        }
    }

    /// Returns `true` when a fresh VMC packet arrived within `driver.maxAge`.
    var hasFreshFrame: Bool {
        guard isListening else { return false }
        let ts = driver.frame.receivedAt
        guard ts > 0 else { return false }
        let maxAge = driver.maxAge ?? Self.defaultMaxAge
        return (Date().timeIntervalSinceReferenceDate - ts) <= maxAge
    }

    /// Applies the latest VMC bones and normalized ARKit / VRM expressions to the avatar.
    /// Returns `true` if a fresh VMC frame was actively applied.
    @discardableResult
    func apply(to model: VRMModel, renderer: VRMRenderer, expressionMixer: ExpressionMixer) -> Bool {
        guard isListening else { return false }

        // Apply bones via VRMMetalKit's VMCDriver (pass controller: nil so we handle ARKit 52 normalization below)
        let applied = driver.apply(to: model, controller: nil)
        guard applied else {
            if wasDrivingStream {
                wasDrivingStream = false
                clearAppliedExpressions(on: renderer, expressionMixer: expressionMixer)
                renderer.lookAtController?.enabled = true
            }
            return false
        }

        wasDrivingStream = true
        let snapshot = driver.frame
        if enablesProportionalArmIK {
            lastArmIKDiagnostics = ArmIKRetargeter.apply(to: model, frame: snapshot, convention: driver.convention)
        } else {
            lastArmIKDiagnostics = []
        }

        // Disable procedural LookAt while VMC is driving eyes/head so it never overwrites VMC eye bones
        if renderer.lookAtController?.enabled == true {
            renderer.lookAtController?.enabled = false
        }

        if let exprCtrl = renderer.expressionController, !snapshot.blendShapes.isEmpty {
            applyBlendShapes(snapshot.blendShapes, on: exprCtrl, expressionMixer: expressionMixer)
        }

        if !isReceivingFrames {
            isReceivingFrames = true
            onStatusChanged?()
        }
        return true
    }

    /// Applies ONLY humanoid bones from this VMC channel (used by AnimationMixer Layer 0).
    @discardableResult
    func applyBonesOnly(to model: VRMModel) -> Bool {
        guard isListening else { return false }
        let applied = driver.apply(to: model, controller: nil)
        if applied {
            if enablesProportionalArmIK {
                lastArmIKDiagnostics = ArmIKRetargeter.apply(to: model, frame: driver.frame, convention: driver.convention)
            } else {
                lastArmIKDiagnostics = []
            }
            if !isReceivingFrames {
                isReceivingFrames = true
                onStatusChanged?()
            }
        }
        return applied
    }

    /// Applies selected humanoid bones ADDITIVELY on top of the model's current pose
    /// (used by AnimationMixer Layer 1 so speech head nods compose cleanly over full-body AI motion).
    @discardableResult
    func applyBonesAdditive(
        to model: VRMModel,
        bones: Set<VRMHumanoidBone>,
        weight: Float
    ) -> Bool {
        guard isListening, hasFreshFrame, weight > 0.0001, let humanoid = model.humanoid else {
            return false
        }
        ensureRestWorldCache(for: model, humanoid: humanoid)

        let snapshot = driver.frame
        let clampedWeight = min(max(weight, 0.0), 1.5)
        let identity = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        var touched = false

        for bone in bones {
            guard let transform = snapshot.bones[bone],
                  let index = humanoid.getBoneNode(bone), index < model.nodes.count,
                  let restWorld = restWorldCache[bone] else {
                continue
            }
            let node = model.nodes[index]
            let delta = driver.convention.convert(rotation: transform.rotation)
            let deltaLocal = simd_normalize(restWorld.inverse * delta * restWorld)
            let scaledDelta = simd_slerp(identity, deltaLocal, min(clampedWeight, 1.0))
            node.rotation = simd_normalize(node.rotation * scaledDelta)
            node.updateLocalMatrix()
            touched = true
        }

        if touched {
            for node in model.nodes where node.parent == nil {
                node.updateWorldTransform()
            }
            if !isReceivingFrames {
                isReceivingFrames = true
                onStatusChanged?()
            }
        }
        return touched
    }

    /// Applies ONLY blendshapes (52 ARKit PerfectSync / VRM presets) from this VMC channel
    /// (used by AnimationMixer Layer 2).
    @discardableResult
    func applyBlendShapesOnly(
        on renderer: VRMRenderer,
        expressionMixer: ExpressionMixer,
        mask: VMCFaceChannelMask = .all
    ) -> Bool {
        guard isListening, hasFreshFrame else {
            clearBlendShapesIfNeeded(on: renderer, expressionMixer: expressionMixer)
            return false
        }
        let snapshot = driver.frame
        guard !snapshot.blendShapes.isEmpty, let exprCtrl = renderer.expressionController else {
            clearBlendShapesIfNeeded(on: renderer, expressionMixer: expressionMixer)
            return false
        }
        wasDrivingStream = true
        applyBlendShapes(snapshot.blendShapes, on: exprCtrl, expressionMixer: expressionMixer, mask: mask)
        if !isReceivingFrames {
            isReceivingFrames = true
            onStatusChanged?()
        }
        return true
    }

    /// Zeroes out any blendshapes previously written by this channel when another face layer takes over or stream stops.
    func clearBlendShapesIfNeeded(on renderer: VRMRenderer, expressionMixer: ExpressionMixer) {
        guard hasAppliedARKitShapes || !appliedVMCPresets.isEmpty || !appliedVMCCustomKeys.isEmpty || wasDrivingStream else {
            return
        }
        wasDrivingStream = false
        clearAppliedExpressions(on: renderer, expressionMixer: expressionMixer)
    }

    private func ensureRestWorldCache(for model: VRMModel, humanoid: VRMHumanoid) {
        let modelID = ObjectIdentifier(model)
        if restWorldCacheModelID == modelID { return }
        restWorldCacheModelID = modelID
        restWorldCache.removeAll()
        for (bone, ref) in humanoid.humanBones where ref.node < model.nodes.count {
            var rot = model.nodes[ref.node].initialRotation
            var cursor = model.nodes[ref.node].parent
            while let parent = cursor {
                rot = parent.initialRotation * rot
                cursor = parent.parent
            }
            restWorldCache[bone] = simd_normalize(rot)
        }
    }

    private func applyBlendShapes(
        _ rawShapes: [String: Float],
        on exprCtrl: VRMExpressionController,
        expressionMixer: ExpressionMixer,
        mask: VMCFaceChannelMask = .all
    ) {
        var arkitShapes: [String: Float] = [:]
        var vrmPresets: [VRMExpressionPreset: Float] = [:]
        var customShapes: [String: Float] = [:]

        for (rawName, rawValue) in rawShapes {
            let value = min(max(rawValue, 0.0), 1.0)
            let lower = rawName.lowercased()
            if let canonicalARKit = Self.arkitKeyLookup[lower] {
                if mask.allows(arkitKey: canonicalARKit) {
                    arkitShapes[canonicalARKit] = value
                }
            } else if let preset = VMCDriver.expressionPreset(blendShapeName: rawName) {
                if mask == .all || !Self.arkitOverlappingPresets.contains(preset) {
                    vrmPresets[preset] = value
                }
            } else if !rawName.hasPrefix(AssetRecipeStore.outfitMorphInternalPrefix) {
                customShapes[rawName] = value
            }
        }

        let useARKitPerfectSync = !arkitShapes.isEmpty
        if useARKitPerfectSync {
            sanitizeARKitShapes(&arkitShapes, exprCtrl: exprCtrl, mask: mask)
            let payload = ARKitFaceBlendShapes(
                timestamp: Date().timeIntervalSinceReferenceDate,
                shapes: arkitShapes
            )
            expressionMixer.faceDriver.update(blendShapes: payload, controller: exprCtrl, maxAge: 1.0)
            hasAppliedARKitShapes = true
            applyAnimeUpperFaceBridge(from: arkitShapes, on: exprCtrl, mask: mask)
        }

        let fullPerfectSyncActive = useARKitPerfectSync && arkitShapes.count >= 4
            && expressionMixer.faceDriver.perfectSyncCapability == .full

        // Zero any previously set VMC presets not present in this frame
        let nextPresetKeys = Set(vrmPresets.keys.filter { !fullPerfectSyncActive || !Self.arkitOverlappingPresets.contains($0) })
        for stalePreset in appliedVMCPresets.subtracting(nextPresetKeys) {
            exprCtrl.setExpressionWeight(stalePreset, weight: 0.0)
        }
        for preset in nextPresetKeys {
            exprCtrl.setExpressionWeight(preset, weight: vrmPresets[preset] ?? 0.0)
        }
        appliedVMCPresets = nextPresetKeys

        // Zero any previously set VMC custom expressions not present in this frame
        let nextCustomKeys = Set(customShapes.keys)
        for staleCustom in appliedVMCCustomKeys.subtracting(nextCustomKeys) {
            exprCtrl.setCustomExpressionWeight(staleCustom, weight: 0.0)
        }
        for (name, value) in customShapes {
            exprCtrl.setCustomExpressionWeight(name, weight: value)
        }
        appliedVMCCustomKeys = nextCustomKeys
    }

    /// Enforces biomechanical invariants across 52-channel ARKit PerfectSync weights before they hit
    /// additive GPU morph targets:
    /// 1. `mouthClose <= jawOpen` (on anime VRMs `mouthClose` is baked as `-jawOpen`, so exceeding
    ///    `jawOpen` shoves the lower jaw upward into the upper lip/nose).
    /// 2. Yields `eyeWide*` / `eyeSquint*` smoothly to procedural Auto-Blink (`cos = -0.78` on eyelids).
    /// 3. Resolves direct opposing muscle pairs (`mouthSmile` vs `mouthFrown`, `browDown` vs `brow*Up`).
    private func sanitizeARKitShapes(
        _ shapes: inout [String: Float],
        exprCtrl: VRMExpressionController,
        mask: VMCFaceChannelMask
    ) {
        // 1. Jaw/lip closure invariant
        let jawOpen = shapes[ARKitFaceBlendShapes.jawOpen] ?? 0.0
        if let close = shapes[ARKitFaceBlendShapes.mouthClose], close > 0 {
            let capFactor: Float = (mask == .speechAudio) ? 0.35 : 0.95
            shapes[ARKitFaceBlendShapes.mouthClose] = min(close, jawOpen * capFactor)
        }

        // 2. Yield upper-eyelid opening/squinting to native Auto-Blink during speech-only streams
        if !mask.contains(.eyesAndBlink) {
            let blinkAll = exprCtrl.weight(for: .blink)
            let blinkL = min(max(max(blinkAll, exprCtrl.weight(for: .blinkLeft)), 0.0), 1.0)
            let blinkR = min(max(max(blinkAll, exprCtrl.weight(for: .blinkRight)), 0.0), 1.0)
            if blinkL > 0.001 {
                if let w = shapes[ARKitFaceBlendShapes.eyeWideLeft] {
                    shapes[ARKitFaceBlendShapes.eyeWideLeft] = w * (1.0 - blinkL)
                }
                if let s = shapes[ARKitFaceBlendShapes.eyeSquintLeft] {
                    shapes[ARKitFaceBlendShapes.eyeSquintLeft] = s * (1.0 - blinkL * 0.75)
                }
            }
            if blinkR > 0.001 {
                if let w = shapes[ARKitFaceBlendShapes.eyeWideRight] {
                    shapes[ARKitFaceBlendShapes.eyeWideRight] = w * (1.0 - blinkR)
                }
                if let s = shapes[ARKitFaceBlendShapes.eyeSquintRight] {
                    shapes[ARKitFaceBlendShapes.eyeSquintRight] = s * (1.0 - blinkR * 0.75)
                }
            }
        }

        // 3. Softly resolve directly opposing muscle pairs so vertices don't pull against themselves
        func resolveOpposition(_ aKey: String, _ bKey: String) {
            let a = shapes[aKey] ?? 0.0
            let b = shapes[bKey] ?? 0.0
            guard a > 0.01 && b > 0.01 else { return }
            let overlap = min(a, b) * 0.75
            shapes[aKey] = max(0.0, a - overlap)
            shapes[bKey] = max(0.0, b - overlap)
        }
        resolveOpposition(ARKitFaceBlendShapes.mouthSmileLeft, ARKitFaceBlendShapes.mouthFrownLeft)
        resolveOpposition(ARKitFaceBlendShapes.mouthSmileRight, ARKitFaceBlendShapes.mouthFrownRight)
        resolveOpposition(ARKitFaceBlendShapes.browDownLeft, ARKitFaceBlendShapes.browOuterUpLeft)
        resolveOpposition(ARKitFaceBlendShapes.browDownRight, ARKitFaceBlendShapes.browOuterUpRight)
        resolveOpposition(ARKitFaceBlendShapes.browDownLeft, ARKitFaceBlendShapes.browInnerUp)
        resolveOpposition(ARKitFaceBlendShapes.browDownRight, ARKitFaceBlendShapes.browInnerUp)
    }

    /// Bridges live ARKit upper-face & emotion signals onto high-visibility sculpted anime brow/eye
    /// custom expressions (e.g. `eyebrow up` 15mm, `eyebrow angry` 13mm, `eyebrow perplex` 11mm)
    /// when present on the VRM. On anime models converted by HANA_Tool, ARKit `brow*` morphs only move
    /// ~1.4mm (invisible under bangs), whereas native anime brow/eye groups move 8–15mm cleanly.
    /// Safely no-ops on generic VRMs without these custom expression names.
    private func applyAnimeUpperFaceBridge(
        from shapes: [String: Float],
        on exprCtrl: VRMExpressionController,
        mask: VMCFaceChannelMask
    ) {
        guard mask.contains(.upperFace) else { return }

        let browOuter = 0.5 * ((shapes[ARKitFaceBlendShapes.browOuterUpLeft] ?? 0) + (shapes[ARKitFaceBlendShapes.browOuterUpRight] ?? 0))
        let browInner = shapes[ARKitFaceBlendShapes.browInnerUp] ?? 0
        let browDown = 0.5 * ((shapes[ARKitFaceBlendShapes.browDownLeft] ?? 0) + (shapes[ARKitFaceBlendShapes.browDownRight] ?? 0))
        let smile = 0.5 * ((shapes[ARKitFaceBlendShapes.mouthSmileLeft] ?? 0) + (shapes[ARKitFaceBlendShapes.mouthSmileRight] ?? 0))
        let frown = 0.5 * ((shapes[ARKitFaceBlendShapes.mouthFrownLeft] ?? 0) + (shapes[ARKitFaceBlendShapes.mouthFrownRight] ?? 0))
        let jawOpen = shapes[ARKitFaceBlendShapes.jawOpen] ?? 0

        let blinkMax = min(max(max(exprCtrl.weight(for: .blink), max(exprCtrl.weight(for: .blinkLeft), exprCtrl.weight(for: .blinkRight))), 0.0), 1.0)
        let eyeYield = 1.0 - blinkMax

        func clamp01(_ v: Float) -> Float { min(max(v, 0.0), 1.0) }

        // Distinct, non-fighting anime eyebrow & eye targets driven purely from live ARKit weights:
        // - Joy / Amazement -> lifted bright brows (`eyebrow up`) + warm mouth corner boost (`mouth smile2`)
        // - Anger -> fierce lowered brows (`eyebrow angry`) + sharp eyes (`eyes angry`, yielded to blink)
        // - Sadness -> troubled tilted brows (`eyebrow perplex`) + gentle down-cast eye mood (`eyes scornful`)
        // - Cheekiness -> playful `eyebrow perplex2` (`八`) + subtle teasing `eyes sarcastic`
        // - Speech prosody -> subtle eyebrow bounce tied to syllable jaw energy even in neutral!
        let angerScore = clamp01((browDown - browInner * 0.55) * 1.45)
        let sadScore = clamp01(browInner * 1.45 + frown * 0.70 - smile * 0.55 - angerScore * 0.4)
        let cheekyScore = clamp01(browInner * 1.15 * clamp01(smile * 2.2 + 0.35) - frown * 0.9 - angerScore * 0.6)
        let upScore = clamp01(browOuter * 1.10 + jawOpen * 0.22 - angerScore * 0.85 - sadScore * 0.45)

        // Note: `eyes angry` already binds `怒り` (100), so split `怒り` cleanly without double-stacking!
        let eyesAngryW = clamp01(angerScore * 0.72) * eyeYield
        let browAngryW = clamp01(angerScore * 0.92 - eyesAngryW * 0.90)
        let eyesSarcasticW = clamp01(cheekyScore * 0.62) * eyeYield
        let browPerplexW = clamp01(sadScore * 0.95 - eyesSarcasticW * 0.45)
        let browPerplex2W = clamp01(cheekyScore * 0.80)
        let browUpW = clamp01(upScore * 0.92)
        let smileBoostW = clamp01((smile - 0.28) * 0.55)

        let targets: [String: Float] = [
            "eyebrow up": browUpW,
            "eyebrow angry": browAngryW,
            "eyebrow perplex": browPerplexW,
            "eyebrow perplex2": browPerplex2W,
            "eyes angry": eyesAngryW,
            "eyes sarcastic": eyesSarcasticW,
            "mouth smile2": smileBoostW,
        ]

        // Smooth transitions cleanly so emotion switches & speech prosody feel organic at 60fps
        let alpha: Float = appliedAnimeBridgeWeights.isEmpty ? 1.0 : 0.35
        for (name, target) in targets {
            let prev = appliedAnimeBridgeWeights[name] ?? 0.0
            let next = prev + (target - prev) * alpha
            let cleaned = next < 0.003 ? 0.0 : next
            appliedAnimeBridgeWeights[name] = cleaned
            exprCtrl.setCustomExpressionWeight(name, weight: cleaned)
        }
    }

    private func clearAppliedExpressions(on renderer: VRMRenderer, expressionMixer: ExpressionMixer) {
        guard let exprCtrl = renderer.expressionController else {
            appliedVMCPresets.removeAll()
            appliedVMCCustomKeys.removeAll()
            appliedAnimeBridgeWeights.removeAll()
            hasAppliedARKitShapes = false
            return
        }
        if hasAppliedARKitShapes {
            let zeros = Dictionary(uniqueKeysWithValues: ARKitFaceBlendShapes.allKeys.map { ($0, Float(0)) })
            expressionMixer.faceDriver.update(
                blendShapes: ARKitFaceBlendShapes(timestamp: Date().timeIntervalSinceReferenceDate, shapes: zeros),
                controller: exprCtrl,
                maxAge: 1.0
            )
            hasAppliedARKitShapes = false
        }
        for (name, _) in appliedAnimeBridgeWeights {
            exprCtrl.setCustomExpressionWeight(name, weight: 0.0)
        }
        appliedAnimeBridgeWeights.removeAll()
        for preset in appliedVMCPresets {
            exprCtrl.setExpressionWeight(preset, weight: 0.0)
        }
        appliedVMCPresets.removeAll()

        for name in appliedVMCCustomKeys {
            exprCtrl.setCustomExpressionWeight(name, weight: 0.0)
        }
        appliedVMCCustomKeys.removeAll()
    }

    private func startStatusTimer() {
        statusTimer?.invalidate()
        statusTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.isListening else { return }
                let fresh = self.hasFreshFrame
                if fresh != self.isReceivingFrames {
                    self.isReceivingFrames = fresh
                    AppLogger.shared.info("VMC", fresh
                        ? "Receiving VMC stream (packets=\(self.packetCount), bones=\(self.receivedBoneCount), blendShapes=\(self.receivedBlendShapeCount))"
                        : "VMC stream idle (waiting for packets on UDP \(self.port))")
                    self.onStatusChanged?()
                }
            }
        }
    }

    // MARK: - Local Network IPv4 Discovery

    /// Returns non-loopback local IPv4 addresses sorted with Wi-Fi / Ethernet (`en*`) first.
    static func localIPv4Addresses() -> [String] {
        var ifaddrPointer: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddrPointer) == 0, let first = ifaddrPointer else { return [] }
        defer { freeifaddrs(ifaddrPointer) }

        var preferred: [(name: String, ip: String)] = []
        var fallback: [(name: String, ip: String)] = []

        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        while let ptr = cursor {
            let iface = ptr.pointee
            cursor = iface.ifa_next

            let flags = Int32(iface.ifa_flags)
            guard (flags & IFF_UP) != 0, (flags & IFF_LOOPBACK) == 0,
                  let addr = iface.ifa_addr, addr.pointee.sa_family == UInt8(AF_INET) else {
                continue
            }

            var hostBuffer = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let result = getnameinfo(
                addr,
                socklen_t(addr.pointee.sa_len),
                &hostBuffer,
                socklen_t(hostBuffer.count),
                nil,
                0,
                NI_NUMERICHOST
            )
            guard result == 0 else { continue }
            let ipLen = hostBuffer.firstIndex(of: 0) ?? hostBuffer.count
            let ip = String(decoding: hostBuffer.prefix(ipLen).map { UInt8(bitPattern: $0) }, as: UTF8.self)
            guard !ip.hasPrefix("127."), !ip.hasPrefix("169.254.") else { continue }

            let name = String(cString: iface.ifa_name)
            if name.hasPrefix("en") {
                preferred.append((name, ip))
            } else if !name.hasPrefix("utun") && !name.hasPrefix("awdl") && !name.hasPrefix("llw") {
                fallback.append((name, ip))
            }
        }

        preferred.sort { $0.name < $1.name }
        fallback.sort { $0.name < $1.name }

        var seen = Set<String>()
        var ordered: [String] = []
        for entry in preferred + fallback {
            if seen.insert(entry.ip).inserted {
                ordered.append(entry.ip)
            }
        }
        return ordered
    }

    static func primaryLocalIPv4() -> String? {
        localIPv4Addresses().first
    }
}
