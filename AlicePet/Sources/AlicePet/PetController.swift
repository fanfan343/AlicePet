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
    let pictureWorldRenderer: PictureWorldRenderer
    let sceneActionController = SceneActionController()
    let recipeStore = AssetRecipeStore()
    let poseController = PoseController()
    let expressionMixer = ExpressionMixer()
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
    private(set) var lightingPreset: LightingPreset = .animeSoftDay
    private(set) var outlineStyle: OutlineStyle = .off

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
    private(set) var latestSceneState: AvatarSceneState?
    var onSceneStateUpdated: ((AvatarSceneState) -> Void)?

    private var startTime: TimeInterval = CACurrentMediaTime()
    private var lastFrameTime: TimeInterval = CACurrentMediaTime()
    private var lastSceneStateSampleTime: TimeInterval = 0
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
        self.pictureWorldRenderer = try PictureWorldRenderer(device: device, sampleCount: sampleCount)

        super.init()

        applyLightingAndOutline()
        renderer.enableSpringBone = springBoneEnabled

        animationMixer.onMixerStateChanged = { [weak self] in
            self?.onVMCStatusChanged?()
        }
        animationMixer.externalChannel.onStatusChanged = { [weak self] in
            self?.onVMCStatusChanged?()
        }
        animationMixer.externalChannel.onError = { [weak self] msg in
            self?.localMotionService.stop()
            self?.onVMCError?(msg)
        }
        animationMixer.ardyChannel.onStatusChanged = { [weak self] in
            self?.onVMCStatusChanged?()
        }
        animationMixer.audioFaceChannel.onStatusChanged = { [weak self] in
            self?.onVMCStatusChanged?()
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
            self.stopAudioFaceTest()
            self.onVMCStatusChanged?()
        }
        localAudioFaceService.onError = { [weak self] msg in
            self?.stopAudioFaceTest()
            self?.onVMCError?(msg)
        }
        localArdyService.onStateChanged = { [weak self] in
            self?.onVMCStatusChanged?()
        }
        localArdyService.onError = { [weak self] msg in
            self?.stopArdyMotionTest()
            self?.onVMCError?(msg)
        }
        sceneActionController.onPhaseChanged = { [weak self] in
            self?.onPictureWorldChanged?()
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

        // Filter out VRMMetalKit's synthetic 14cm hand palm spheres & forearm capsules so hand
        // movements near hips never hook underneath and flip flared skirts upward, while keeping
        // all authored colliders + synthetic leg/head/torso/shoulder/breast colliders intact.
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
                AppLogger.shared.info("Physics", "Filtered \(removed) synthetic hand/forearm SpringBone collider(s) to prevent skirt lifting (kept \(sb.syntheticColliders.count) leg/body/head colliders)")
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

    // MARK: - Picture-World (Mode B: Single-Image Metric Depth 3D Scene)

    func loadPictureWorld(from folderURL: URL) throws {
        sceneActionController.reset()
        try pictureWorldRenderer.loadScene(from: folderURL)
        if let lightRaw = pictureWorldRenderer.config?.lightingPreset,
           let preset = LightingPreset(rawValue: lightRaw) {
            setLightingPreset(preset)
        }
        applyPictureWorldRootTransform()
        settleSpringBonePhysics(warmupSteps: 16)
        onPictureWorldChanged?()
    }

    func exitPictureWorld() {
        guard pictureWorldRenderer.isActive else { return }
        sceneActionController.standUpOrCancel(on: self, stopArdy: false)
        pictureWorldRenderer.unloadScene()
        restoreRestRootTransform()
        camera.reset()
        settleSpringBonePhysics(warmupSteps: 12)
        onPictureWorldChanged?()
    }

    /// Autonomous Picture-World chain: ARDY walks to `anchorId`, aligns, sits down, and holds seated Idle.
    @discardableResult
    func walkToSceneAnchorAndSit(anchorId: String) -> Bool {
        guard let anchor = pictureWorldRenderer.config?.anchors?.first(where: { $0.id == anchorId }) else {
            return false
        }
        return sceneActionController.startWalkAndSit(anchor: anchor, on: self)
    }

    func standUpFromSceneSeat(stopArdy: Bool = true) {
        sceneActionController.standUpOrCancel(on: self, stopArdy: stopArdy)
    }

    func panPictureWorldCanvas(deltaX: Float, deltaY: Float) {
        guard pictureWorldRenderer.isActive else { return }
        let size = hostView?.bounds.size ?? CGSize(width: 820, height: 480)
        pictureWorldRenderer.panCanvas(deltaX: deltaX, deltaY: deltaY, viewportSize: size)
    }

    func zoomPictureWorldCanvas(deltaY: Float) {
        guard pictureWorldRenderer.isActive else { return }
        let aspect = currentViewportAspect()
        pictureWorldRenderer.zoomCanvas(deltaY: deltaY, viewportAspect: aspect)
    }

    func resetPictureWorldViewAndAvatar() {
        guard pictureWorldRenderer.isActive else { return }
        sceneActionController.standUpOrCancel(on: self, stopArdy: false)
        pictureWorldRenderer.resetCanvasAndAvatarToOrigin()
        if poseMode == .sit {
            setPoseMode(.stand)
        }
        applyPictureWorldRootTransform()
        settleSpringBonePhysics(warmupSteps: 8)
        _ = refreshSceneState(aspect: currentViewportAspect(), force: true)
    }

    private func currentViewportAspect() -> Float {
        guard let view = hostView, view.bounds.height > 0 else { return 820.0 / 480.0 }
        return Float(view.bounds.width / view.bounds.height)
    }

    /// Builds an on-demand structured `AvatarSceneState` snapshot (location, depth occlusion,
    /// boundary contact, active AI drivers) for future LLM / agentic tool integration.
    func currentAvatarSceneState(aspect: Float? = nil) -> AvatarSceneState {
        let resolvedAspect = aspect ?? currentViewportAspect()
        let now = Date().timeIntervalSince1970
        let activePrompt: String? = localArdyActive ? localArdyService.currentPrompt : nil

        if pictureWorldRenderer.isActive {
            let spatial = pictureWorldRenderer.evaluateSpatialSnapshot(viewportAspect: resolvedAspect)
            let p = pictureWorldRenderer.avatarWorldPosition
            return AvatarSceneState(
                timestamp: now,
                mode: "picture_world",
                sceneName: pictureWorldRenderer.config?.name,
                positionMeters: [p.x, p.y, p.z],
                normalizedRoomPosition: spatial.normalizedRoomPosition,
                facingYawDegrees: pictureWorldRenderer.avatarWorldYaw * 180.0 / .pi,
                distanceToCameraMeters: spatial.distanceToCameraMeters,
                screenNDC: spatial.screenNDC,
                isOccluded: spatial.isOccluded,
                occlusionRatio: spatial.occlusionRatio,
                occludedRegions: spatial.occludedRegions,
                boundaryContacts: spatial.boundaryContacts,
                isNearBoundary: spatial.isNearBoundary,
                activeBodyEngine: animationMixer.activeBodySource.rawValue,
                activeFaceEngine: animationMixer.activeFaceSource.rawValue,
                activeMotionPrompt: activePrompt,
                activeSceneTask: sceneActionController.activeTaskIdentifier,
                seatedAnchorId: sceneActionController.seatedAnchor?.id,
                canvasZoom: pictureWorldRenderer.canvasZoom
            )
        } else {
            return AvatarSceneState(
                timestamp: now,
                mode: "desktop_pet",
                sceneName: nil,
                positionMeters: [0, 0, 0],
                normalizedRoomPosition: [0, 0],
                facingYawDegrees: camera.yaw * 180.0 / .pi,
                distanceToCameraMeters: camera.distance,
                screenNDC: [0, 0],
                isOccluded: false,
                occlusionRatio: 0,
                occludedRegions: [],
                boundaryContacts: [],
                isNearBoundary: false,
                activeBodyEngine: animationMixer.activeBodySource.rawValue,
                activeFaceEngine: animationMixer.activeFaceSource.rawValue,
                activeMotionPrompt: activePrompt,
                activeSceneTask: nil,
                seatedAnchorId: nil,
                canvasZoom: 1.0
            )
        }
    }

    func currentAvatarSceneStateJSON(pretty: Bool = false) -> String? {
        currentAvatarSceneState().toJSONString(pretty: pretty)
    }

    @discardableResult
    private func refreshSceneState(aspect: Float, force: Bool = false) -> AvatarSceneState? {
        let now = CACurrentMediaTime()
        if !force && (now - lastSceneStateSampleTime) < 0.20 {
            return latestSceneState
        }
        lastSceneStateSampleTime = now
        let prev = latestSceneState
        let next = currentAvatarSceneState(aspect: aspect)
        latestSceneState = next
        if prev?.isOccluded != next.isOccluded || prev?.boundaryContacts != next.boundaryContacts {
            AppLogger.shared.info(
                "SceneState",
                String(
                    format: "mode=%@ pos=(%.2f,%.2f) occluded=%@(%.0f%%:%@) boundary=%@",
                    next.mode,
                    next.positionMeters[0],
                    next.positionMeters[2],
                    next.isOccluded ? "yes" : "no",
                    next.occlusionRatio * 100.0,
                    next.occludedRegions.joined(separator: ","),
                    next.boundaryContacts.isEmpty ? "none" : next.boundaryContacts.joined(separator: ",")
                )
            )
        }
        onSceneStateUpdated?(next)
        return next
    }

    func applyPictureWorldRootTransform() {
        guard pictureWorldRenderer.isActive, let model else { return }
        var offset = pictureWorldRenderer.avatarWorldPosition
        // avatarWorldPosition.y is the *local* elevation (0 when standing, the sit offset when
        // seated); the floor itself is not the Y=0 plane, so lift her onto the real surface.
        offset.y += pictureWorldRenderer.groundHeight(atX: offset.x, z: offset.z)
        let yawQ = simd_quatf(angle: pictureWorldRenderer.avatarWorldYaw, axis: SIMD3<Float>(0, 1, 0))
        for (idx, node) in model.nodes.enumerated() where node.parent == nil {
            guard let rest = poseController.restNodeTransforms[idx] else { continue }
            node.translation = rest.pos + offset
            node.rotation = simd_normalize(yawQ * rest.rot)
            node.updateLocalMatrix()
        }
        model.updateNodeTransforms()
    }

    private func restoreRestRootTransform() {
        guard let model else { return }
        for (idx, node) in model.nodes.enumerated() where node.parent == nil {
            guard let rest = poseController.restNodeTransforms[idx] else { continue }
            node.translation = rest.pos
            node.rotation = rest.rot
            node.updateLocalMatrix()
        }
        model.updateNodeTransforms()
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
        if pictureWorldRenderer.isActive {
            applySeatedUprightPostureIfNeeded()
            applyPictureWorldRootTransform()
        }
        if settlePhysics {
            settleSpringBonePhysics()
        }
    }

    /// Gently lifts torso & head upright when seated against a sofa/chair backrest
    /// (Author `sit_action` leans ~30° forward for a low stool; uprighting keeps posture graceful & skirt draped).
    func applySeatedUprightPostureIfNeeded() {
        guard sceneActionController.seatedAnchor != nil, let model, let humanoid = model.humanoid else { return }
        let uprightAdjust: [(VRMHumanoidBone, Float)] = [
            (.spine, -0.26),
            (.chest, -0.14),
            (.head, -0.12),
        ]
        for (bone, pitchRad) in uprightAdjust {
            if let idx = humanoid.getBoneNode(bone), idx < model.nodes.count {
                let liftQ = simd_quatf(angle: pitchRad, axis: SIMD3<Float>(1, 0, 0))
                model.nodes[idx].rotation = simd_normalize(model.nodes[idx].rotation * liftQ)
                model.nodes[idx].updateLocalMatrix()
            }
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
            isCameraActive: { [weak self] in self?.animationMixer.isEyeAndBlinkCaptured ?? false }
        )
    }

    @discardableResult
    func startVMC(port: UInt16 = VMCTrackingController.defaultPort) -> Bool {
        expressionMixer.prepareForTrackingStart(on: renderer)
        animationMixer.preferredFullBodySource = .externalVMC
        return animationMixer.externalChannel.start(port: port)
    }

    func stopVMC() {
        localMotionService.stop()
        let wasActive = animationMixer.externalChannel.isListening
        animationMixer.externalChannel.stop(renderer: renderer, expressionMixer: expressionMixer)
        if wasActive && !animationMixer.isFaceCaptured {
            expressionMixer.resetAfterTrackingStop(on: renderer)
        }
        if wasActive && !localArdyActive {
            applyAllBodyAndOutfitState(settlePhysics: false)
        }
    }

    @discardableResult
    func startLocalMotionCapture(port: UInt16 = VMCTrackingController.defaultPort) -> Bool {
        if !vmcActive || vmcController.port != port {
            guard startVMC(port: port) else { return false }
        } else {
            animationMixer.preferredFullBodySource = .externalVMC
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
    func startAudioFaceTest(audioURL: URL? = nil, maxSeconds: Double = 0) -> Bool {
        if !animationMixer.audioFaceChannel.isListening {
            guard animationMixer.audioFaceChannel.start(port: 0) else { return false }
        }
        guard let boundPort = animationMixer.audioFaceChannel.boundPort else { return false }
        // Clear preset moods / custom face recipes when full-face Audio2Face-3D drives emotions
        // so VRM preset morphs never stack additively onto 52-channel ARKit emotion morphs.
        if localAudioFaceService.params.engine == .a2fClaire {
            expressionMixer.prepareForTrackingStart(on: renderer)
        }
        let ok = localAudioFaceService.start(audioURL: audioURL, port: boundPort, maxSeconds: maxSeconds)
        if !ok {
            animationMixer.audioFaceChannel.stop(renderer: renderer, expressionMixer: expressionMixer)
        }
        return ok
    }

    func stopAudioFaceTest() {
        localAudioFaceService.stop()
        let wasListening = animationMixer.audioFaceChannel.isListening
        animationMixer.audioFaceChannel.stop(renderer: renderer, expressionMixer: expressionMixer)
        if wasListening && !animationMixer.isFaceCaptured {
            expressionMixer.resetAfterTrackingStop(on: renderer)
        }
    }

    @discardableResult
    func startArdyMotionTest() -> Bool {
        if !animationMixer.ardyChannel.isListening {
            guard animationMixer.ardyChannel.start(port: 0) else { return false }
        }
        guard let boundPort = animationMixer.ardyChannel.boundPort else { return false }
        animationMixer.preferredFullBodySource = .ardy
        let ok = localArdyService.start(port: boundPort)
        if !ok {
            animationMixer.ardyChannel.stop()
        }
        return ok
    }

    func stopArdyMotionTest() {
        localArdyService.stop()
        animationMixer.ardyChannel.stop()
    }

    func stopAllTrackingSync() {
        localMotionService.stopSync()
        localAudioFaceService.stopSync()
        localArdyService.stopSync()
        animationMixer.stopAllChannels()
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
        if let pwMats = pictureWorldRenderer.cameraMatrices(viewportAspect: aspect) {
            renderer.projectionMatrix = pwMats.projection
            renderer.viewMatrix = pwMats.view
        } else {
            renderer.projectionMatrix = camera.projectionMatrix(aspect: aspect)
            renderer.viewMatrix = camera.viewMatrix
        }

        let dt = max(0.0, min(frameStart - lastFrameTime, 0.1))
        lastFrameTime = frameStart
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
        if pictureWorldRenderer.isActive {
            applyPictureWorldRootTransform()
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

        let aspect = Float(width) / Float(height)
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = colorTex
        rpd.colorAttachments[0].loadAction = pictureWorldRenderer.isActive ? .load : .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

        rpd.depthAttachment.texture = depthTex
        rpd.depthAttachment.loadAction = pictureWorldRenderer.isActive ? .load : .clear
        rpd.depthAttachment.storeAction = propRenderer.hasActiveProps ? .store : .dontCare
        rpd.depthAttachment.clearDepth = 1.0

        if let pwMats = pictureWorldRenderer.cameraMatrices(viewportAspect: aspect) {
            renderer.projectionMatrix = pwMats.projection
            renderer.viewMatrix = pwMats.view
        } else {
            renderer.projectionMatrix = camera.projectionMatrix(aspect: aspect)
            renderer.viewMatrix = camera.viewMatrix
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw NSError(domain: "AlicePet", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer"])
        }

        if pictureWorldRenderer.isActive {
            pictureWorldRenderer.encodePrepass(
                commandBuffer: commandBuffer,
                colorTexture: colorTex,
                depthTexture: depthTex,
                viewportAspect: aspect
            )
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
