import Darwin
import Foundation
import VRMMetalKit
import simd

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

    var packetCount: Int { receiver?.packetCount ?? 0 }
    var decodeErrorCount: Int { receiver?.decodeErrorCount ?? 0 }
    var receivedBoneCount: Int { driver.frame.bones.count }
    var receivedBlendShapeCount: Int { driver.frame.blendShapes.count }

    var onStatusChanged: (() -> Void)?
    var onError: ((String) -> Void)?

    private var appliedVMCPresets: Set<VRMExpressionPreset> = []
    private var appliedVMCCustomKeys: Set<String> = []
    private var hasAppliedARKitShapes: Bool = false
    private var wasDrivingStream: Bool = false

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

    private func applyBlendShapes(
        _ rawShapes: [String: Float],
        on exprCtrl: VRMExpressionController,
        expressionMixer: ExpressionMixer
    ) {
        var arkitShapes: [String: Float] = [:]
        var vrmPresets: [VRMExpressionPreset: Float] = [:]
        var customShapes: [String: Float] = [:]

        for (rawName, rawValue) in rawShapes {
            let value = min(max(rawValue, 0.0), 1.0)
            let lower = rawName.lowercased()
            if let canonicalARKit = Self.arkitKeyLookup[lower] {
                arkitShapes[canonicalARKit] = value
            } else if let preset = VMCDriver.expressionPreset(blendShapeName: rawName) {
                vrmPresets[preset] = value
            } else if !rawName.hasPrefix(AssetRecipeStore.outfitMorphInternalPrefix) {
                customShapes[rawName] = value
            }
        }

        let useARKitPerfectSync = !arkitShapes.isEmpty
        if useARKitPerfectSync {
            let payload = ARKitFaceBlendShapes(
                timestamp: Date().timeIntervalSinceReferenceDate,
                shapes: arkitShapes
            )
            expressionMixer.faceDriver.update(blendShapes: payload, controller: exprCtrl, maxAge: 1.0)
            hasAppliedARKitShapes = true
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

    private func clearAppliedExpressions(on renderer: VRMRenderer, expressionMixer: ExpressionMixer) {
        guard let exprCtrl = renderer.expressionController else {
            appliedVMCPresets.removeAll()
            appliedVMCCustomKeys.removeAll()
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
