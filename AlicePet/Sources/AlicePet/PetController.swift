import AppKit
import CoreGraphics
import Foundation
import Metal
import MetalKit
import UniformTypeIdentifiers
import VRMMetalKit
import simd

@MainActor
final class PetController: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let renderer: VRMRenderer

    private(set) var model: VRMModel?
    private(set) var loadedModelURL: URL?

    var camera = OrbitCamera()
    weak var hostView: MTKView?

    var springBoneEnabled: Bool = true {
        didSet { renderer.enableSpringBone = springBoneEnabled }
    }
    var autoBlinkEnabled: Bool = true
    var eyeTrackingEnabled: Bool = true {
        didSet {
            if !eyeTrackingEnabled && !cameraTrackingActive {
                renderer.lookAtController?.target = .forward
            }
        }
    }
    var idleBreathingEnabled: Bool = true

    // ARKit face driver & camera capture
    let faceDriver: ARKitFaceDriver
    let cameraTracker = CameraFaceTracker()
    private(set) var activeCameraID: String?
    var cameraTrackingActive: Bool { activeCameraID != nil }
    var onCameraError: ((String) -> Void)?
    var onCameraDevicesChanged: (() -> Void)?
    var onCameraFaceDetectedChanged: ((Bool) -> Void)?

    private var smoothedHeadYaw: Float = 0
    private var smoothedHeadPitch: Float = 0
    private var smoothedHeadRoll: Float = 0
    private var latestCameraGaze: SIMD2<Float> = .zero

    private var activeCustomExpression: String?
    private var blinkTimer: Timer?
    private var startTime: TimeInterval = CACurrentMediaTime()
    private var restRotations: [Int: simd_quatf] = [:]

    init(sampleCount: Int = 4) throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            throw NSError(domain: "AlicePet", code: 1, userInfo: [NSLocalizedDescriptionKey: "Metal device unavailable"])
        }
        self.device = device
        self.commandQueue = commandQueue

        var config = RendererConfig()
        config.sampleCount = sampleCount
        config.strict = .off
        self.renderer = VRMRenderer(device: device, config: config)

        var smoothing = SmoothingConfig.default
        smoothing.perExpression[ARKitFaceBlendShapes.eyeBlinkLeft] = .ema(alpha: 0.55)
        smoothing.perExpression[ARKitFaceBlendShapes.eyeBlinkRight] = .ema(alpha: 0.55)
        smoothing.perExpression[ARKitFaceBlendShapes.jawOpen] = .ema(alpha: 0.5)
        self.faceDriver = ARKitFaceDriver(mapper: .default, smoothing: smoothing)

        super.init()

        renderer.setup3PointLighting(keyIntensity: 1.0, fillIntensity: 0.65, rimIntensity: 0.35)
        renderer.setAmbientColor(SIMD3<Float>(0.22, 0.21, 0.23))
        renderer.setLightNormalizationMode(.radiometric)
        renderer.outlineScale = 0.0
        renderer.enableSpringBone = springBoneEnabled

        cameraTracker.onFrame = { [weak self] frame in
            self?.handleTrackedFaceFrame(frame)
        }
        cameraTracker.onTrackingStateChanged = { [weak self] detected in
            self?.onCameraFaceDetectedChanged?(detected)
        }
        cameraTracker.onError = { [weak self] msg in
            self?.stopCameraCapture()
            self?.onCameraError?(msg)
        }
        cameraTracker.onDevicesChanged = { [weak self] in
            self?.onCameraDevicesChanged?()
        }
    }

    func loadModel(from url: URL, synchronousPhysics: Bool = false) async throws {
        if synchronousPhysics {
            renderer.config.synchronousSpringBone = true
        }
        let loaded = try await VRMModel.load(from: url, device: device)
        self.model = loaded
        self.loadedModelURL = url
        self.activeCustomExpression = nil
        self.restRotations.removeAll()

        applyNaturalStandingPose(to: loaded)

        let (minBounds, maxBounds) = loaded.calculateBoundingBox()
        camera.fit(minBounds: minBounds, maxBounds: maxBounds)

        for bone in [VRMHumanoidBone.spine, .chest, .neck, .head] {
            if let idx = loaded.humanoid?.getBoneNode(bone), idx >= 0, idx < loaded.nodes.count {
                restRotations[idx] = loaded.nodes[idx].rotation
            }
        }

        renderer.loadModel(loaded)
        faceDriver.initializeForModel(loaded)
        renderer.enableSpringBone = springBoneEnabled
        if let lookAt = renderer.lookAtController {
            lookAt.enabled = true
            lookAt.smoothing = 0.15
            lookAt.saccadeEnabled = true
            lookAt.target = .camera
        }

        if let view = hostView, view.drawableSize.width > 0, view.drawableSize.height > 0 {
            let aspect = Float(view.drawableSize.width / view.drawableSize.height)
            renderer.projectionMatrix = camera.projectionMatrix(aspect: aspect)
        } else {
            renderer.projectionMatrix = camera.projectionMatrix(aspect: 460.0 / 680.0)
        }
        renderer.viewMatrix = camera.viewMatrix

        AppLogger.shared.info("Model", "Loaded VRM '\(loaded.meta.name ?? url.lastPathComponent)' from \(url.path)")
        AppLogger.shared.info("Model", "PerfectSync=\(faceDriver.perfectSyncCapability), customExpressions=\(loaded.expressions?.custom.count ?? 0), bounds=(\(minBounds), \(maxBounds))")
    }

    private func applyNaturalStandingPose(to model: VRMModel) {
        let armDown: Float = 62.0 * .pi / 180.0
        let elbowBend: Float = 12.0 * .pi / 180.0

        if let idx = model.humanoid?.getBoneNode(.leftUpperArm), idx < model.nodes.count {
            model.nodes[idx].rotation = simd_quatf(angle: -armDown, axis: SIMD3<Float>(0, 0, 1)) * model.nodes[idx].rotation
            model.nodes[idx].updateLocalMatrix()
        }
        if let idx = model.humanoid?.getBoneNode(.rightUpperArm), idx < model.nodes.count {
            model.nodes[idx].rotation = simd_quatf(angle: armDown, axis: SIMD3<Float>(0, 0, 1)) * model.nodes[idx].rotation
            model.nodes[idx].updateLocalMatrix()
        }
        if let idx = model.humanoid?.getBoneNode(.leftLowerArm), idx < model.nodes.count {
            model.nodes[idx].rotation = simd_quatf(angle: -elbowBend, axis: SIMD3<Float>(0, 1, 0)) * model.nodes[idx].rotation
            model.nodes[idx].updateLocalMatrix()
        }
        if let idx = model.humanoid?.getBoneNode(.rightLowerArm), idx < model.nodes.count {
            model.nodes[idx].rotation = simd_quatf(angle: elbowBend, axis: SIMD3<Float>(0, 1, 0)) * model.nodes[idx].rotation
            model.nodes[idx].updateLocalMatrix()
        }
        model.updateNodeTransforms()
    }

    func startTimers() {
        scheduleNextBlink()
    }

    private func scheduleNextBlink() {
        blinkTimer?.invalidate()
        let interval = Double.random(in: 2.6...5.4)
        blinkTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                if self.autoBlinkEnabled && !self.cameraTrackingActive {
                    self.renderer.expressionController?.blink(duration: 0.11)
                }
                self.scheduleNextBlink()
            }
        }
    }

    // MARK: - Camera Capture Control

    func startCameraCapture(deviceID: String) {
        AppLogger.shared.info("Camera", "Requesting camera capture start: \(deviceID)")
        clearCustomExpression()
        renderer.expressionController?.setMood(.neutral, intensity: 0.0)
        activeCameraID = deviceID
        smoothedHeadYaw = 0
        smoothedHeadPitch = 0
        smoothedHeadRoll = 0
        latestCameraGaze = .zero
        cameraTracker.start(deviceID: deviceID)
    }

    func stopCameraCapture() {
        if activeCameraID != nil {
            AppLogger.shared.info("Camera", "Stopping camera capture")
        }
        cameraTracker.stop()
        activeCameraID = nil
        smoothedHeadYaw = 0
        smoothedHeadPitch = 0
        smoothedHeadRoll = 0
        latestCameraGaze = .zero

        // Reset ARKit expressions & head pose back to neutral rest state
        if let exprCtrl = renderer.expressionController {
            let zeros = Dictionary(uniqueKeysWithValues: ARKitFaceBlendShapes.allKeys.map { ($0, Float(0)) })
            faceDriver.update(
                blendShapes: ARKitFaceBlendShapes(timestamp: Date().timeIntervalSinceReferenceDate, shapes: zeros),
                controller: exprCtrl,
                maxAge: 1.0
            )
        }
        if let model {
            for bone in [VRMHumanoidBone.neck, .head] {
                if let idx = model.humanoid?.getBoneNode(bone), let base = restRotations[idx] {
                    model.nodes[idx].rotation = base
                    model.nodes[idx].updateLocalMatrix()
                }
            }
        }
        if !eyeTrackingEnabled {
            renderer.lookAtController?.target = .forward
        }
    }

    private func handleTrackedFaceFrame(_ frame: TrackedFaceFrame) {
        guard cameraTrackingActive, let exprCtrl = renderer.expressionController else { return }
        faceDriver.update(
            blendShapes: frame.blendShapes,
            controller: exprCtrl,
            maxAge: 0.5
        )
        let alpha: Float = 0.28
        smoothedHeadYaw = smoothedHeadYaw * (1 - alpha) + frame.headYaw * alpha
        smoothedHeadPitch = smoothedHeadPitch * (1 - alpha) + frame.headPitch * alpha
        smoothedHeadRoll = smoothedHeadRoll * (1 - alpha) + frame.headRoll * alpha
        latestCameraGaze = frame.gazeOffset
    }

    // MARK: - Expression Test Controls

    func setPresetMood(_ preset: VRMExpressionPreset?) {
        clearCustomExpression()
        if let preset {
            renderer.expressionController?.setMood(preset, intensity: 1.0)
        } else {
            renderer.expressionController?.setMood(.neutral, intensity: 0.0)
        }
    }

    func setCustomExpression(_ name: String, weight: Float = 1.0) {
        clearCustomExpression()
        renderer.expressionController?.setMood(.neutral, intensity: 0.0)
        renderer.setExpression(name, weight: weight)
        activeCustomExpression = name
    }

    func clearCustomExpression() {
        if let prev = activeCustomExpression {
            renderer.setExpression(prev, weight: 0.0)
            activeCustomExpression = nil
        }
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        guard size.width > 0, size.height > 0 else { return }
        let aspect = Float(size.width / size.height)
        renderer.projectionMatrix = camera.projectionMatrix(aspect: aspect)
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }

        descriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

        let aspect = (view.drawableSize.height > 0)
            ? Float(view.drawableSize.width / view.drawableSize.height)
            : Float(view.bounds.width / max(view.bounds.height, 1))
        renderer.projectionMatrix = camera.projectionMatrix(aspect: aspect)
        renderer.viewMatrix = camera.viewMatrix

        updateProceduralMotion(in: view)

        renderer.draw(in: view, commandBuffer: commandBuffer, renderPassDescriptor: descriptor)
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    private func updateProceduralMotion(in view: MTKView) {
        guard let model else { return }

        let t = Float(CACurrentMediaTime() - startTime)
        let breath = sin(t * 1.7)

        if idleBreathingEnabled {
            if let chestIdx = model.humanoid?.getBoneNode(.chest),
               let base = restRotations[chestIdx] {
                let pitch = simd_quatf(angle: breath * 0.014, axis: SIMD3<Float>(1, 0, 0))
                model.nodes[chestIdx].rotation = base * pitch
                model.nodes[chestIdx].updateLocalMatrix()
            }
        }

        if cameraTrackingActive {
            if let neckIdx = model.humanoid?.getBoneNode(.neck),
               let base = restRotations[neckIdx] {
                let qYaw = simd_quatf(angle: smoothedHeadYaw * 0.35, axis: SIMD3<Float>(0, 1, 0))
                let qPitch = simd_quatf(angle: smoothedHeadPitch * 0.35, axis: SIMD3<Float>(1, 0, 0))
                let qRoll = simd_quatf(angle: smoothedHeadRoll * 0.35, axis: SIMD3<Float>(0, 0, 1))
                model.nodes[neckIdx].rotation = base * qYaw * qPitch * qRoll
                model.nodes[neckIdx].updateLocalMatrix()
            }
            if let headIdx = model.humanoid?.getBoneNode(.head),
               let base = restRotations[headIdx] {
                let qYaw = simd_quatf(angle: smoothedHeadYaw * 0.65, axis: SIMD3<Float>(0, 1, 0))
                let qPitch = simd_quatf(angle: smoothedHeadPitch * 0.65, axis: SIMD3<Float>(1, 0, 0))
                let qRoll = simd_quatf(angle: smoothedHeadRoll * 0.65, axis: SIMD3<Float>(0, 0, 1))
                model.nodes[headIdx].rotation = base * qYaw * qPitch * qRoll
                model.nodes[headIdx].updateLocalMatrix()
            }
            if let lookAt = renderer.lookAtController {
                lookAt.target = .headLocalPoint(SIMD3<Float>(latestCameraGaze.x * 0.55, latestCameraGaze.y * 0.35, 1.5))
            }
            return
        }

        if idleBreathingEnabled {
            let sway = sin(t * 0.75)
            if let headIdx = model.humanoid?.getBoneNode(.head),
               let base = restRotations[headIdx] {
                let roll = simd_quatf(angle: sway * 0.018, axis: SIMD3<Float>(0, 0, 1))
                let pitch = simd_quatf(angle: breath * 0.01, axis: SIMD3<Float>(1, 0, 0))
                model.nodes[headIdx].rotation = base * roll * pitch
                model.nodes[headIdx].updateLocalMatrix()
            }
        }

        if eyeTrackingEnabled, let lookAt = renderer.lookAtController, let window = view.window {
            let mouseScreen = NSEvent.mouseLocation
            let winFrame = window.frame
            let normX = Float(((mouseScreen.x - winFrame.midX) / max(winFrame.width, 1)).clamped(to: -1.5...1.5))
            let normY = Float(((mouseScreen.y - winFrame.midY) / max(winFrame.height, 1)).clamped(to: -1.5...1.5))
            lookAt.target = .headLocalPoint(SIMD3<Float>(normX * 0.85, normY * 0.6, 1.5))
        }
    }

    // MARK: - Offscreen Snapshot (CLI verification)

    func renderSnapshot(to outputURL: URL, width: Int = 720, height: Int = 1024) async throws {
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        colorDesc.usage = [.renderTarget, .shaderRead]
        colorDesc.storageMode = .shared

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        depthDesc.usage = .renderTarget
        depthDesc.storageMode = .shared

        guard let colorTex = device.makeTexture(descriptor: colorDesc),
              let depthTex = device.makeTexture(descriptor: depthDesc) else {
            throw NSError(domain: "AlicePet", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate offscreen textures"])
        }

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = colorTex
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

        rpd.depthAttachment.texture = depthTex
        rpd.depthAttachment.loadAction = .clear
        rpd.depthAttachment.storeAction = .dontCare
        rpd.depthAttachment.clearDepth = 1.0

        renderer.projectionMatrix = camera.projectionMatrix(aspect: Float(width) / Float(height))
        renderer.viewMatrix = camera.viewMatrix

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw NSError(domain: "AlicePet", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer"])
        }

        renderer.drawOffscreenHeadless(
            to: colorTex,
            depth: depthTex,
            commandBuffer: commandBuffer,
            renderPassDescriptor: rpd
        )

        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }

        try exportBGRA8Texture(colorTex, to: outputURL)
    }

    private func exportBGRA8Texture(_ texture: MTLTexture, to url: URL) throws {
        let width = texture.width
        let height = texture.height
        let bytesPerRow = width * 4
        var data = Data(count: bytesPerRow * height)
        data.withUnsafeMutableBytes { buf in
            guard let ptr = buf.baseAddress else { return }
            texture.getBytes(ptr, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        guard let provider = CGDataProvider(data: data as CFData) else {
            throw NSError(domain: "AlicePet", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGDataProvider"])
        }
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw NSError(domain: "AlicePet", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGImage"])
        }
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw NSError(domain: "AlicePet", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to create image destination"])
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw NSError(domain: "AlicePet", code: 7, userInfo: [NSLocalizedDescriptionKey: "Failed to write PNG"])
        }
    }
}

private extension Comparable {
    func clamped(to limits: ClosedRange<Self>) -> Self {
        min(max(self, limits.lowerBound), limits.upperBound)
    }
}
