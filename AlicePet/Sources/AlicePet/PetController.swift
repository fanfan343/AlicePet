import AppKit
import CoreGraphics
import Darwin
import Foundation
import Metal
import MetalKit
import UniformTypeIdentifiers
import VRMMetalKit
import simd

enum OutfitToggleKind {
    case dress
    case pantyhose
    case shoes
    case hairBand
    case chestRibbon
    case hairSide
}

@MainActor
final class PetController: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let renderer: VRMRenderer
    let propRenderer: PropSceneRenderer
    let recipeStore = AssetRecipeStore()
    let poseController = PoseController()
    let expressionMixer = ExpressionMixer()
    let vmcController = VMCTrackingController()
    let localMotionService = LocalMotionService()
    let localAudioFaceService = LocalAudioFaceService()

    private(set) var model: VRMModel?
    private(set) var loadedModelURL: URL?
    private(set) var activeProfile: ModelProfile = .generic

    var camera = OrbitCamera()
    weak var hostView: MTKView?

    var springBoneEnabled: Bool = true {
        didSet { renderer.enableSpringBone = springBoneEnabled }
    }
    var idleBreathingEnabled: Bool = true

    var autoBlinkEnabled: Bool {
        get { expressionMixer.autoBlinkEnabled }
        set { expressionMixer.autoBlinkEnabled = newValue }
    }
    var eyeTrackingEnabled: Bool {
        get { expressionMixer.eyeTrackingEnabled }
        set {
            expressionMixer.eyeTrackingEnabled = newValue
            if !newValue && !vmcReceiving {
                renderer.lookAtController?.target = .forward
            }
        }
    }

    var poseMode: PetPoseMode { poseController.poseMode }
    var leftHandGesture: HandGesturePreset { poseController.leftHandGesture }
    var rightHandGesture: HandGesturePreset { poseController.rightHandGesture }
    var activePresetMood: VRMExpressionPreset? { expressionMixer.activePresetMood }
    var activeCustomExpression: String? { expressionMixer.activeCustomExpression }
    var faceDriver: ARKitFaceDriver { expressionMixer.faceDriver }

    var outfitState = OutfitState()
    var manualPropOverride: PropSceneRenderer.ActivePropMode? {
        didSet { syncPropRendererMode() }
    }
    private(set) var lightingPreset: LightingPreset = .animeSoftDay
    private(set) var outlineStyle: OutlineStyle = .off

    var vmcActive: Bool { vmcController.isListening }
    var vmcReceiving: Bool { vmcController.isReceivingFrames }
    var localMotionActive: Bool { localMotionService.isRunning }
    var localAudioFaceActive: Bool { localAudioFaceService.isRunning }
    var onVMCStatusChanged: (() -> Void)?
    var onVMCError: ((String) -> Void)?

    private var startTime: TimeInterval = CACurrentMediaTime()
    private var outfitNodeMeshes: [String: (nodeIndex: Int, meshIndex: Int)] = [:]

    // Lightweight periodic performance baseline sampler
    private var perfWindowStart: TimeInterval = CACurrentMediaTime()
    private var perfFrameCount: Int = 0
    private var perfEncodeMsSum: Double = 0
    private let perfLogIntervalSeconds: TimeInterval = 60.0

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
        self.propRenderer = try PropSceneRenderer(device: device, sampleCount: sampleCount)

        super.init()

        applyLightingAndOutline()
        renderer.enableSpringBone = springBoneEnabled

        vmcController.onStatusChanged = { [weak self] in
            self?.onVMCStatusChanged?()
        }
        vmcController.onError = { [weak self] msg in
            self?.localMotionService.stop()
            self?.onVMCError?(msg)
        }
        localMotionService.onStateChanged = { [weak self] in
            self?.onVMCStatusChanged?()
        }
        localMotionService.onError = { [weak self] msg in
            self?.onVMCError?(msg)
        }
        localAudioFaceService.onStateChanged = { [weak self] in
            self?.onVMCStatusChanged?()
        }
        localAudioFaceService.onFinishedNaturally = { [weak self] in
            guard let self else { return }
            if !self.localMotionActive {
                self.stopVMC()
            }
            self.onVMCStatusChanged?()
        }
        localAudioFaceService.onError = { [weak self] msg in
            self?.stopAudioFaceTest()
            self?.onVMCError?(msg)
        }
    }

    func loadModel(from url: URL, synchronousPhysics: Bool = false) async throws {
        if synchronousPhysics {
            renderer.config.synchronousSpringBone = true
        }
        let loaded = try await VRMModel.load(from: url, device: device)
        let profile = ModelProfile.resolve(forModelName: loaded.meta.name, fileURL: url)

        self.model = loaded
        self.loadedModelURL = url
        self.activeProfile = profile
        self.outfitNodeMeshes.removeAll()
        self.outfitState = OutfitState()

        poseController.captureRestPose(from: loaded)
        let outfitNodeSet = Set(profile.outfitNodeNames)
        for (idx, node) in loaded.nodes.enumerated() {
            if let name = node.name, outfitNodeSet.contains(name), let meshIdx = node.mesh {
                outfitNodeMeshes[name] = (idx, meshIdx)
            }
        }

        let (minBounds, maxBounds) = loaded.calculateBoundingBox()
        camera.fit(minBounds: minBounds, maxBounds: maxBounds)

        renderer.loadModel(loaded)
        applyLightingAndOutline()
        expressionMixer.initializeForModel(loaded, renderer: renderer)
        recipeStore.loadRecipes(near: url, profile: profile)
        recipeStore.registerExpressions(on: loaded, controller: renderer.expressionController, profile: profile)
        poseController.loadAuthorAnimations(from: recipeStore.assetsFolderURL)

        if let folder = recipeStore.assetsFolderURL {
            await propRenderer.loadProps(from: folder)
        }
        syncPropRendererMode()

        applyAllBodyAndOutfitState(settlePhysics: true)
        renderer.enableSpringBone = springBoneEnabled

        if let view = hostView, view.drawableSize.width > 0, view.drawableSize.height > 0 {
            let aspect = Float(view.drawableSize.width / view.drawableSize.height)
            renderer.projectionMatrix = camera.projectionMatrix(aspect: aspect)
        } else {
            renderer.projectionMatrix = camera.projectionMatrix(aspect: 460.0 / 680.0)
        }
        renderer.viewMatrix = camera.viewMatrix

        AppLogger.shared.info("Model", "Loaded VRM '\(loaded.meta.name ?? url.lastPathComponent)' (profile=\(profile.id)) from \(url.path)")
        AppLogger.shared.info("Model", "PerfectSync=\(faceDriver.perfectSyncCapability), customExpressions=\(loaded.expressions?.custom.count ?? 0), outfitNodes=\(Array(outfitNodeMeshes.keys).sorted())")
    }

    // MARK: - Pose, Props, Hand Gestures & Outfits

    func setLightingPreset(_ preset: LightingPreset) {
        lightingPreset = preset
        applyLightingAndOutline()
        AppLogger.shared.info("LookDev", "Switched lightingPreset=\(preset.rawValue)")
    }

    func setOutlineStyle(_ style: OutlineStyle) {
        outlineStyle = style
        applyLightingAndOutline()
        AppLogger.shared.info("LookDev", "Switched outlineStyle=\(style.rawValue)")
    }

    private func applyLightingAndOutline() {
        LightingConfigurator.apply(preset: lightingPreset, outline: outlineStyle, to: renderer)
    }

    func setPoseMode(_ mode: PetPoseMode) {
        let elapsed = Float(CACurrentMediaTime() - startTime)
        poseController.setPoseMode(mode, at: elapsed)
        syncPropRendererMode()
        applyAllBodyAndOutfitState(settlePhysics: true)
        AppLogger.shared.info("Pose", "Switched poseMode=\(mode.rawValue), propMode=\(propRenderer.activePropMode.rawValue)")
    }

    func setHandGesture(_ preset: HandGesturePreset, side: HandSide = .both) {
        poseController.setHandGesture(preset, side: side)
        applyAllBodyAndOutfitState(settlePhysics: false)
        AppLogger.shared.info("Gesture", "Hand gesture updated: L=\(leftHandGesture.rawValue), R=\(rightHandGesture.rawValue)")
    }

    var hasAnyOutfitControls: Bool {
        activeProfile.hasCustomOutfits && (!outfitNodeMeshes.isEmpty || !recipeStore.availableOutfitMorphKeys.isEmpty)
    }

    var hasHairStyleControls: Bool {
        activeProfile.hairStyles.values.contains { !$0.isEmpty }
    }

    func isOutfitToggleSupported(_ kind: OutfitToggleKind) -> Bool {
        let morphs = recipeStore.availableOutfitMorphKeys
        switch kind {
        case .dress: return outfitNodeMeshes["Clothes"] != nil
        case .pantyhose: return outfitNodeMeshes["Pantyhose"] != nil
        case .shoes: return outfitNodeMeshes["Shoes"] != nil
        case .hairBand: return morphs.contains("ClothesPart-HairBandOff")
        case .chestRibbon: return morphs.contains("ClothesPart-ChestRibbonOff")
        case .hairSide: return morphs.contains("HairPart-SideOff")
        }
    }

    func applyOutfitState() {
        guard let model else { return }

        var changedVisibility = false
        let nodeMap: [(String, Bool)] = [
            ("Clothes", outfitState.showDress),
            ("Pantyhose", outfitState.showPantyhose),
            ("Shoes", outfitState.showShoes),
        ]
        for (name, visible) in nodeMap {
            if let entry = outfitNodeMeshes[name], entry.nodeIndex < model.nodes.count {
                let expectedMesh: Int? = visible ? entry.meshIndex : nil
                if model.nodes[entry.nodeIndex].mesh != expectedMesh {
                    model.nodes[entry.nodeIndex].mesh = expectedMesh
                    changedVisibility = true
                }
            }
        }
        if changedVisibility {
            let prev = renderer.cameraMode
            renderer.cameraMode = (prev == .thirdPerson) ? .firstPerson : .thirdPerson
            renderer.cameraMode = prev
        }

        if let exprCtrl = renderer.expressionController {
            func setOutfitMorph(_ morphName: String, weight: Float) {
                guard recipeStore.availableOutfitMorphKeys.contains(morphName) else { return }
                exprCtrl.setCustomExpressionWeight(AssetRecipeStore.outfitMorphInternalPrefix + morphName, weight: weight)
            }
            setOutfitMorph("Body2 ShapeKey-Dress", weight: outfitState.showDress ? 1.0 : 0.0)
            setOutfitMorph("Pantyhose ShapeKey-Dress", weight: outfitState.showDress ? 1.0 : 0.0)
            setOutfitMorph("Body2 ShapeKey-Pantyhose", weight: outfitState.showPantyhose ? 1.0 : 0.0)
            setOutfitMorph("Body2 ShapeKey-Shoes", weight: outfitState.showShoes ? 1.0 : 0.0)
            setOutfitMorph("Pantyhose ShapeKey-Shoes", weight: outfitState.showShoes ? 1.0 : 0.0)
            setOutfitMorph("ClothesPart-HairBandOff", weight: outfitState.showHairBand ? 0.0 : 1.0)
            setOutfitMorph("ClothesPart-ChestRibbonOff", weight: outfitState.showChestRibbon ? 0.0 : 1.0)
            setOutfitMorph("HairPart-SideOff", weight: outfitState.showHairSide ? 0.0 : 1.0)
        }

        poseController.applyHairStyle(outfitState.hairStyle, profile: activeProfile, to: model)
        model.updateNodeTransforms()
        AppLogger.shared.info("Outfit", "Applied outfit: dress=\(outfitState.showDress), pantyhose=\(outfitState.showPantyhose), shoes=\(outfitState.showShoes), band=\(outfitState.showHairBand), ribbon=\(outfitState.showChestRibbon), side=\(outfitState.showHairSide), hair=\(outfitState.hairStyle.rawValue)")
    }

    private func syncPropRendererMode() {
        if let manualPropOverride {
            propRenderer.activePropMode = manualPropOverride
            return
        }
        switch poseMode {
        case .stand, .sit, .crouch, .prone:
            propRenderer.activePropMode = .none
        case .afkBook:
            propRenderer.activePropMode = .bookAndTea
        case .afkTea:
            propRenderer.activePropMode = .teaSetOnly
        }
    }

    private func applyAllBodyAndOutfitState(settlePhysics: Bool, animTime: Float = 0) {
        guard let model else { return }
        poseController.applyAll(
            to: model,
            recipeStore: recipeStore,
            profile: activeProfile,
            hairStyle: outfitState.hairStyle,
            at: animTime
        )
        applyOutfitState()
        if settlePhysics {
            settleSpringBonePhysics()
        }
    }

    /// Resets and warms up SpringBone chains after instantaneous pose teleports (e.g. stand -> sit)
    /// so hair and ribbons settle naturally instead of whipping upward from single-frame velocity spikes.
    func settleSpringBonePhysics(warmupSteps: Int = 24) {
        guard springBoneEnabled, model != nil else { return }
        renderer.resetPhysics()
        renderer.warmupPhysics(steps: warmupSteps)
    }

    // MARK: - Timers, VMC / Local MediaPipe Tracking & Expressions

    func startTimers() {
        expressionMixer.startAutoBlink(
            rendererProvider: { [weak self] in self?.renderer },
            isCameraActive: { [weak self] in self?.vmcReceiving ?? false }
        )
    }

    @discardableResult
    func startVMC(port: UInt16 = VMCTrackingController.defaultPort) -> Bool {
        expressionMixer.prepareForTrackingStart(on: renderer)
        return vmcController.start(port: port)
    }

    func stopVMC() {
        localMotionService.stop()
        localAudioFaceService.stop()
        let wasActive = vmcActive
        vmcController.stop(renderer: renderer, expressionMixer: expressionMixer)
        if wasActive {
            expressionMixer.resetAfterTrackingStop(on: renderer)
            applyAllBodyAndOutfitState(settlePhysics: false)
        }
    }

    @discardableResult
    func startLocalMotionCapture(port: UInt16 = VMCTrackingController.defaultPort) -> Bool {
        localAudioFaceService.stop()
        if !vmcActive || vmcController.port != port {
            guard startVMC(port: port) else { return false }
        }
        let ok = localMotionService.start(port: vmcController.port)
        if !ok {
            stopVMC()
        }
        return ok
    }

    func stopLocalMotionCapture() {
        stopVMC()
    }

    @discardableResult
    func startAudioFaceTest(audioURL: URL? = nil, port: UInt16 = VMCTrackingController.defaultPort, maxSeconds: Double = 0) -> Bool {
        localMotionService.stop()
        if !vmcActive || vmcController.port != port {
            guard startVMC(port: port) else { return false }
        }
        let ok = localAudioFaceService.start(audioURL: audioURL, port: vmcController.port, maxSeconds: maxSeconds)
        if !ok {
            stopVMC()
        }
        return ok
    }

    func stopAudioFaceTest() {
        stopVMC()
    }

    func stopAllTrackingSync() {
        localMotionService.stopSync()
        localAudioFaceService.stopSync()
        vmcController.stop()
    }

    func setPresetMood(_ preset: VRMExpressionPreset?) {
        expressionMixer.setPresetMood(preset, on: renderer)
    }

    func setCustomExpression(_ name: String, weight: Float = 1.0) {
        expressionMixer.setCustomExpression(name, weight: weight, on: renderer)
    }

    func applyExpression(named query: String, weight: Float = 1.0) {
        expressionMixer.applyExpression(named: query, weight: weight, on: renderer, recipeStore: recipeStore)
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        guard size.width > 0, size.height > 0 else { return }
        let aspect = Float(size.width / size.height)
        renderer.projectionMatrix = camera.projectionMatrix(aspect: aspect)
    }

    func draw(in view: MTKView) {
        let frameStart = CACurrentMediaTime()
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

        let elapsed = Float(frameStart - startTime)
        if let model {
            let vmcDriving = vmcController.isListening && vmcController.hasFreshFrame
            if vmcDriving {
                // Start from clean base pose so bones omitted by partial VMC senders (e.g. iPhone head+face only) stay natural
                poseController.applyBasePose(to: model, restTransforms: poseController.restNodeTransforms)
                poseController.applyHandGestureOverrides(to: model, recipeStore: recipeStore)
                vmcController.apply(to: model, renderer: renderer, expressionMixer: expressionMixer)
            } else {
                if vmcController.isListening {
                    // Handles clean transition when an active VMC stream goes stale
                    vmcController.apply(to: model, renderer: renderer, expressionMixer: expressionMixer)
                }
                poseController.applyProceduralFrame(
                    to: model,
                    restTransforms: poseController.restNodeTransforms,
                    elapsed: elapsed,
                    idleBreathingEnabled: idleBreathingEnabled
                )
                expressionMixer.updateLookAt(
                    on: renderer,
                    poseMode: poseMode,
                    window: view.window
                )
            }
        }

        let drawProps = propRenderer.hasActiveProps
        if drawProps {
            if descriptor.colorAttachments[0].resolveTexture != nil {
                descriptor.colorAttachments[0].storeAction = .storeAndMultisampleResolve
            } else {
                descriptor.colorAttachments[0].storeAction = .store
            }
            descriptor.depthAttachment.storeAction = .store
        }

        renderer.draw(in: view, commandBuffer: commandBuffer, renderPassDescriptor: descriptor)

        if drawProps,
           let colorTex = descriptor.colorAttachments[0].texture,
           let depthTex = descriptor.depthAttachment.texture {
            propRenderer.drawProps(
                commandBuffer: commandBuffer,
                colorTexture: colorTex,
                resolveTexture: descriptor.colorAttachments[0].resolveTexture,
                depthTexture: depthTex,
                viewProjection: renderer.projectionMatrix * renderer.viewMatrix,
                cameraPosition: camera.eyePosition,
                bookWorldMatrix: poseController.computeBookWorldMatrix(time: elapsed, profile: activeProfile),
                time: elapsed,
                lightingPreset: lightingPreset
            )
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
        recordPerfSample(frameStart: frameStart)
    }

    private func recordPerfSample(frameStart: TimeInterval) {
        let now = CACurrentMediaTime()
        perfFrameCount += 1
        perfEncodeMsSum += (now - frameStart) * 1000.0
        let window = now - perfWindowStart
        guard window >= perfLogIntervalSeconds, perfFrameCount > 0 else { return }

        let fps = Double(perfFrameCount) / window
        let avgEncodeMs = perfEncodeMsSum / Double(perfFrameCount)
        let memMB = Self.currentMemoryFootprintMB()
        let motionFlag = localMotionActive ? "local" : "off"
        let vmcFlag = vmcReceiving ? "rx" : (vmcActive ? "listen" : "off")
        AppLogger.shared.info(
            "Perf",
            String(
                format: "fps=%.1f avgEncodeMs=%.2f memMB=%.1f pose=%@ props=%@ motion=%@ vmc=%@",
                fps, avgEncodeMs, memMB, poseMode.rawValue, propRenderer.activePropMode.rawValue, motionFlag, vmcFlag
            )
        )
        perfWindowStart = now
        perfFrameCount = 0
        perfEncodeMsSum = 0
    }

    private static func currentMemoryFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return -1.0 }
        return Double(info.phys_footprint) / (1024.0 * 1024.0)
    }

    // MARK: - Offscreen Snapshot (CLI verification)

    func renderSnapshot(
        to outputURL: URL,
        width: Int = 720,
        height: Int = 1024,
        animTime: Float = 0.8,
        applyBaseState: Bool = true
    ) async throws {
        if applyBaseState {
            applyAllBodyAndOutfitState(settlePhysics: true, animTime: animTime)
        }

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
        depthDesc.usage = [.renderTarget]
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
        rpd.depthAttachment.storeAction = propRenderer.hasActiveProps ? .store : .dontCare
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

        if propRenderer.hasActiveProps {
            propRenderer.drawProps(
                commandBuffer: commandBuffer,
                colorTexture: colorTex,
                resolveTexture: nil,
                depthTexture: depthTex,
                viewProjection: renderer.projectionMatrix * renderer.viewMatrix,
                cameraPosition: camera.eyePosition,
                bookWorldMatrix: poseController.computeBookWorldMatrix(time: animTime, profile: activeProfile),
                time: animTime,
                lightingPreset: lightingPreset
            )
        }

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
