import AppKit
import CoreGraphics
import Darwin
import Foundation
import Metal
import MetalKit
import VRMMetalKit
import simd

@MainActor
final class PetController: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let renderer: VRMRenderer
    let propRenderer: PropSceneRenderer
    let pictureWorldRenderer: PictureWorldRenderer
    let sceneActionController = SceneActionController()
    let recipeStore = AssetRecipeStore()
    let poseController = PoseController()
    let expressionMixer = ExpressionMixer()
    let emotionStateMachine = EmotionStateMachine()
    let animationMixer = AnimationMixer()
    var vmcController: VMCTrackingController { animationMixer.externalChannel }
    let localMotionService = LocalMotionService()
    let localAudioFaceService = LocalAudioFaceService()
    let localArdyService = LocalArdyService()

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
    var lightingPreset: LightingPreset = .animeSoftDay
    var outlineStyle: OutlineStyle = .off

    var vmcActive: Bool { animationMixer.externalChannel.isListening }
    var vmcReceiving: Bool { animationMixer.externalChannel.isReceivingFrames }
    var localMotionActive: Bool { localMotionService.isRunning }
    var localAudioFaceActive: Bool { localAudioFaceService.isRunning }
    var localArdyActive: Bool { localArdyService.isRunning }
    var onVMCStatusChanged: (() -> Void)?
    var onVMCError: ((String) -> Void)?
    var onPictureWorldChanged: (() -> Void)?
    var isPictureWorldActive: Bool { pictureWorldRenderer.isActive }
    /// Live spatial & perceptual state snapshot exposed for future LLM / agentic tool calls.
    var latestSceneState: AvatarSceneState?
    var onSceneStateUpdated: ((AvatarSceneState) -> Void)?

    var startTime: TimeInterval = CACurrentMediaTime()
    private var lastFrameTime: TimeInterval = CACurrentMediaTime()
    var lastSceneStateSampleTime: TimeInterval = 0
    var outfitNodeMeshes: [String: (nodeIndex: Int, meshIndex: Int)] = [:]

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
        self.pictureWorldRenderer = try PictureWorldRenderer(device: device, sampleCount: sampleCount)

        super.init()

        applyLightingAndOutline()
        renderer.enableSpringBone = springBoneEnabled
        bindServiceCallbacks()
    }

    private func bindServiceCallbacks() {
        animationMixer.onMixerStateChanged = { [weak self] in self?.onVMCStatusChanged?() }
        animationMixer.externalChannel.onStatusChanged = { [weak self] in self?.onVMCStatusChanged?() }
        animationMixer.externalChannel.onError = { [weak self] msg in
            self?.localMotionService.stop()
            self?.onVMCError?(msg)
        }
        animationMixer.ardyChannel.onStatusChanged = { [weak self] in self?.onVMCStatusChanged?() }
        animationMixer.audioFaceChannel.onStatusChanged = { [weak self] in self?.onVMCStatusChanged?() }
        localMotionService.onStateChanged = { [weak self] in self?.onVMCStatusChanged?() }
        localMotionService.onError = { [weak self] msg in self?.onVMCError?(msg) }
        localAudioFaceService.onStateChanged = { [weak self] in self?.onVMCStatusChanged?() }
        localAudioFaceService.onFinishedNaturally = { [weak self] in
            guard let self else { return }
            self.stopAudioFaceTest()
            self.onVMCStatusChanged?()
        }
        localAudioFaceService.onError = { [weak self] msg in
            self?.stopAudioFaceTest()
            self?.onVMCError?(msg)
        }
        localArdyService.onStateChanged = { [weak self] in self?.onVMCStatusChanged?() }
        localArdyService.onError = { [weak self] msg in
            self?.stopArdyMotionTest()
            self?.onVMCError?(msg)
        }
        sceneActionController.onPhaseChanged = { [weak self] in self?.onPictureWorldChanged?() }
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

        // Filter out VRMMetalKit's synthetic hand palm spheres & forearm capsules so hand
        // movements near hips never hook underneath and flip flared skirts upward.
        if var sb = loaded.springBone, let humanoid = loaded.humanoid, !sb.syntheticColliders.isEmpty {
            let excludedArmNodes = Set(
                [VRMHumanoidBone.leftHand, .rightHand, .leftLowerArm, .rightLowerArm]
                    .compactMap { humanoid.getBoneNode($0) }
            )
            let beforeCount = sb.syntheticColliders.count
            sb.syntheticColliders.removeAll { excludedArmNodes.contains($0.node) }
            loaded.springBone = sb
            let removed = beforeCount - sb.syntheticColliders.count
            if removed > 0 {
                AppLogger.shared.info("Physics", "Filtered \(removed) synthetic hand/forearm SpringBone collider(s) (kept \(sb.syntheticColliders.count) leg/body/head colliders)")
            }
        }

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
        if let pwMats = pictureWorldRenderer.cameraMatrices(viewportAspect: aspect) {
            renderer.projectionMatrix = pwMats.projection
            renderer.viewMatrix = pwMats.view
        } else {
            renderer.projectionMatrix = camera.projectionMatrix(aspect: aspect)
            renderer.viewMatrix = camera.viewMatrix
        }

        let dt = max(0.0, min(frameStart - lastFrameTime, 0.1))
        lastFrameTime = frameStart
        emotionStateMachine.tick(dt: dt)
        let elapsed = Float(frameStart - startTime)
        if let model {
            animationMixer.evaluate(
                model: model,
                renderer: renderer,
                poseController: poseController,
                recipeStore: recipeStore,
                expressionMixer: expressionMixer,
                elapsed: elapsed,
                dt: dt,
                idleBreathingEnabled: idleBreathingEnabled,
                poseMode: poseMode,
                window: view.window
            )
            if pictureWorldRenderer.isActive {
                if animationMixer.activeBodySource == .ardy,
                   let rootXZ = animationMixer.ardyChannel.latestWorldRootXZ {
                    pictureWorldRenderer.applyARDYRootMotion(
                        worldRootXZ: rootXZ,
                        viewportAspect: aspect,
                        goalTargetXZ: sceneActionController.activeWalkGoalXZ
                    )
                } else {
                    pictureWorldRenderer.resetARDYRootTracking()
                }
                sceneActionController.tick(dt: dt, on: self, viewportAspect: aspect)
                applySeatedUprightPostureIfNeeded()
                applyPictureWorldRootTransform()
            }
            _ = refreshSceneState(aspect: aspect)
        }

        if pictureWorldRenderer.isActive,
           let colorTex = descriptor.colorAttachments[0].texture,
           let depthTex = descriptor.depthAttachment.texture {
            pictureWorldRenderer.encodePrepass(
                commandBuffer: commandBuffer,
                colorTexture: colorTex,
                depthTexture: depthTex,
                viewportAspect: aspect
            )
            descriptor.colorAttachments[0].loadAction = .load
            descriptor.depthAttachment.loadAction = .load
        }

        let drawProps = propRenderer.hasActiveProps
        if drawProps {
            descriptor.colorAttachments[0].storeAction = (descriptor.colorAttachments[0].resolveTexture != nil)
                ? .storeAndMultisampleResolve
                : .store
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
}
