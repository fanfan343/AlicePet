import AppKit
import CoreGraphics
import Foundation
import Metal
import MetalKit
import UniformTypeIdentifiers
import VRMMetalKit
import simd

enum HandSide {
    case left
    case right
    case both
}

@MainActor
final class PetController: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let renderer: VRMRenderer
    let propRenderer: PropSceneRenderer
    let recipeStore = AssetRecipeStore()

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

    // State: Pose, AFK Props, Hand Gestures, Outfits
    private(set) var poseMode: PetPoseMode = .stand
    private(set) var leftHandGesture: HandGesturePreset = .natural
    private(set) var rightHandGesture: HandGesturePreset = .natural
    var outfitState = OutfitState()
    var manualPropOverride: PropSceneRenderer.ActivePropMode? {
        didSet { syncPropRendererMode() }
    }

    // ARKit face driver & camera capture (smoothed via 1€ Filter)
    let faceDriver: ARKitFaceDriver
    let cameraTracker = CameraFaceTracker()
    private(set) var activeCameraID: String?
    var cameraTrackingActive: Bool { activeCameraID != nil }
    var onCameraError: ((String) -> Void)?
    var onCameraDevicesChanged: (() -> Void)?
    var onCameraFaceDetectedChanged: ((Bool) -> Void)?

    private var yawFilter = OneEuroFilterScalar(minCutoff: 1.5, beta: 0.14, dCutoff: 1.0)
    private var pitchFilter = OneEuroFilterScalar(minCutoff: 1.5, beta: 0.14, dCutoff: 1.0)
    private var rollFilter = OneEuroFilterScalar(minCutoff: 1.5, beta: 0.14, dCutoff: 1.0)
    private var gazeXFilter = OneEuroFilterScalar(minCutoff: 2.2, beta: 0.10, dCutoff: 1.0)
    private var gazeYFilter = OneEuroFilterScalar(minCutoff: 2.2, beta: 0.10, dCutoff: 1.0)

    private var smoothedHeadYaw: Float = 0
    private var smoothedHeadPitch: Float = 0
    private var smoothedHeadRoll: Float = 0
    private var latestCameraGaze: SIMD2<Float> = .zero

    private var activeCustomExpression: String?
    private var blinkTimer: Timer?
    private var startTime: TimeInterval = CACurrentMediaTime()

    private var restNodeTransforms: [Int: (pos: SIMD3<Float>, rot: simd_quatf, scale: SIMD3<Float>)] = [:]
    private var outfitNodeMeshes: [String: (nodeIndex: Int, meshIndex: Int)] = [:]
    private var hairNodeIndices: [String: Int] = [:]

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
        self.restNodeTransforms.removeAll()
        self.outfitNodeMeshes.removeAll()
        self.hairNodeIndices.removeAll()

        for (idx, node) in loaded.nodes.enumerated() {
            restNodeTransforms[idx] = (node.translation, node.rotation, node.scale)
            if let name = node.name {
                if name.lowercased().contains("hair") {
                    hairNodeIndices[name] = idx
                }
                if ["Clothes", "Pantyhose", "Shoes", "HairShadow"].contains(name), let meshIdx = node.mesh {
                    outfitNodeMeshes[name] = (idx, meshIdx)
                }
            }
        }

        let (minBounds, maxBounds) = loaded.calculateBoundingBox()
        camera.fit(minBounds: minBounds, maxBounds: maxBounds)

        renderer.loadModel(loaded)
        faceDriver.initializeForModel(loaded)
        recipeStore.loadRecipes(near: url)
        recipeStore.registerExpressions(on: loaded, controller: renderer.expressionController)
        if let folder = recipeStore.assetsFolderURL {
            await propRenderer.loadProps(from: folder)
        }

        applyAllBodyAndOutfitState()
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
        AppLogger.shared.info("Model", "PerfectSync=\(faceDriver.perfectSyncCapability), customExpressions=\(loaded.expressions?.custom.count ?? 0), outfitNodes=\(Array(outfitNodeMeshes.keys).sorted())")
    }

    // MARK: - Pose, Props, Hand Gestures & Outfits

    func setPoseMode(_ mode: PetPoseMode) {
        poseMode = mode
        syncPropRendererMode()
        applyAllBodyAndOutfitState()
        AppLogger.shared.info("Pose", "Switched poseMode=\(mode.rawValue), propMode=\(propRenderer.activePropMode.rawValue)")
    }

    func setHandGesture(_ preset: HandGesturePreset, side: HandSide = .both) {
        switch side {
        case .left: leftHandGesture = preset
        case .right: rightHandGesture = preset
        case .both:
            leftHandGesture = preset
            rightHandGesture = preset
        }
        applyAllBodyAndOutfitState()
        AppLogger.shared.info("Gesture", "Hand gesture updated: L=\(leftHandGesture.rawValue), R=\(rightHandGesture.rawValue)")
    }

    func applyOutfitState() {
        guard let model else { return }

        // 1. Toggle node meshes + invalidate VRMRenderer cache if visibility changed
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

        // 2. Apply outfit blendshapes through registered custom expressions
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

        // 3. Apply hair style rotations
        applyHairStyle(outfitState.hairStyle, to: model)
        model.updateNodeTransforms()
        AppLogger.shared.info("Outfit", "Applied outfit: dress=\(outfitState.showDress), pantyhose=\(outfitState.showPantyhose), shoes=\(outfitState.showShoes), band=\(outfitState.showHairBand), ribbon=\(outfitState.showChestRibbon), side=\(outfitState.showHairSide), hair=\(outfitState.hairStyle.rawValue)")
    }

    private func syncPropRendererMode() {
        if let manualPropOverride {
            propRenderer.activePropMode = manualPropOverride
            return
        }
        switch poseMode {
        case .stand, .sit:
            propRenderer.activePropMode = .none
        case .afkBook:
            propRenderer.activePropMode = .bookOnly
        case .afkTea:
            propRenderer.activePropMode = .teaSetOnly
        }
    }

    private func applyAllBodyAndOutfitState() {
        guard let model else { return }
        applyBodyPose(poseMode, to: model)
        applyHandGestures(to: model)
        applyOutfitState()
    }

    private func resetHumanoidBones(_ bones: [VRMHumanoidBone], on model: VRMModel) {
        for bone in bones {
            if let idx = model.humanoid?.getBoneNode(bone),
               idx < model.nodes.count,
               let rest = restNodeTransforms[idx] {
                model.nodes[idx].translation = rest.pos
                model.nodes[idx].rotation = rest.rot
                model.nodes[idx].scale = rest.scale
                model.nodes[idx].updateLocalMatrix()
            }
        }
    }

    private func rotateBone(_ bone: VRMHumanoidBone, on model: VRMModel, yaw: Float = 0, pitch: Float = 0, roll: Float = 0) {
        guard let idx = model.humanoid?.getBoneNode(bone),
              idx < model.nodes.count,
              let rest = restNodeTransforms[idx] else { return }
        let qY = simd_quatf(angle: yaw * .pi / 180.0, axis: SIMD3<Float>(0, 1, 0))
        let qX = simd_quatf(angle: pitch * .pi / 180.0, axis: SIMD3<Float>(1, 0, 0))
        let qZ = simd_quatf(angle: roll * .pi / 180.0, axis: SIMD3<Float>(0, 0, 1))
        model.nodes[idx].rotation = qY * qX * qZ * rest.rot
        model.nodes[idx].updateLocalMatrix()
    }

    private func translateBone(_ bone: VRMHumanoidBone, on model: VRMModel, offset: SIMD3<Float>) {
        guard let idx = model.humanoid?.getBoneNode(bone),
              idx < model.nodes.count,
              let rest = restNodeTransforms[idx] else { return }
        model.nodes[idx].translation = rest.pos + offset
        model.nodes[idx].updateLocalMatrix()
    }

    private func applyBodyPose(_ mode: PetPoseMode, to model: VRMModel) {
        let bodyBones: [VRMHumanoidBone] = [
            .hips, .spine, .chest, .upperChest, .neck, .head,
            .leftShoulder, .leftUpperArm, .leftLowerArm, .leftHand,
            .rightShoulder, .rightUpperArm, .rightLowerArm, .rightHand,
            .leftUpperLeg, .leftLowerLeg, .leftFoot, .leftToes,
            .rightUpperLeg, .rightLowerLeg, .rightFoot, .rightToes
        ]
        resetHumanoidBones(bodyBones, on: model)

        switch mode {
        case .stand:
            // Natural arms down at sides
            rotateBone(.leftUpperArm, on: model, roll: -62)
            rotateBone(.rightUpperArm, on: model, roll: 62)
            rotateBone(.leftLowerArm, on: model, yaw: -12)
            rotateBone(.rightLowerArm, on: model, yaw: 12)

        case .sit:
            // Seated posture lowered slightly into frame
            translateBone(.hips, on: model, offset: SIMD3<Float>(0, -0.18, 0))
            rotateBone(.hips, on: model, pitch: -4)
            rotateBone(.spine, on: model, pitch: 5)
            rotateBone(.leftUpperLeg, on: model, yaw: 6, pitch: -82, roll: -4)
            rotateBone(.rightUpperLeg, on: model, yaw: -6, pitch: -82, roll: 4)
            rotateBone(.leftLowerLeg, on: model, pitch: 82)
            rotateBone(.rightLowerLeg, on: model, pitch: 82)
            rotateBone(.leftFoot, on: model, pitch: 12)
            rotateBone(.rightFoot, on: model, pitch: 12)

            rotateBone(.leftUpperArm, on: model, yaw: -16, pitch: -12, roll: -58)
            rotateBone(.rightUpperArm, on: model, yaw: 16, pitch: -12, roll: 58)
            rotateBone(.leftLowerArm, on: model, yaw: -42, pitch: -14)
            rotateBone(.rightLowerArm, on: model, yaw: 42, pitch: -14)

        case .afkBook:
            // Levitating/curled AFK reading pose based on Alice_AFK_Loop muscles
            translateBone(.hips, on: model, offset: SIMD3<Float>(0, 0.04, 0))
            rotateBone(.hips, on: model, yaw: -6, pitch: -10, roll: 4)
            rotateBone(.spine, on: model, yaw: 4, pitch: 10)
            rotateBone(.chest, on: model, yaw: -6, pitch: 10)
            rotateBone(.neck, on: model, yaw: 4, pitch: 12)
            rotateBone(.head, on: model, yaw: 0, pitch: 8)

            // Relaxed floating legs (left tucked, right slightly extended)
            rotateBone(.leftUpperLeg, on: model, yaw: 8, pitch: -34, roll: -10)
            rotateBone(.leftLowerLeg, on: model, pitch: 44)
            rotateBone(.leftFoot, on: model, pitch: 22)
            rotateBone(.rightUpperLeg, on: model, yaw: -10, pitch: -52, roll: 6)
            rotateBone(.rightLowerLeg, on: model, pitch: 66)
            rotateBone(.rightFoot, on: model, pitch: 26)

            // Arms brought forward toward open book (+Z: leftUpperArm -yaw, rightUpperArm +yaw)
            rotateBone(.leftUpperArm, on: model, yaw: -38, pitch: -24, roll: -50)
            rotateBone(.leftLowerArm, on: model, yaw: -54, pitch: -16)
            rotateBone(.leftHand, on: model, yaw: 10, pitch: 14, roll: 14)

            rotateBone(.rightUpperArm, on: model, yaw: 38, pitch: -24, roll: 50)
            rotateBone(.rightLowerArm, on: model, yaw: 54, pitch: -16)
            rotateBone(.rightHand, on: model, yaw: -10, pitch: 14, roll: -14)

        case .afkTea:
            // Afternoon Tea levitating idle pose surrounded by tea-set items
            translateBone(.hips, on: model, offset: SIMD3<Float>(0, 0.02, 0))
            rotateBone(.hips, on: model, yaw: 8, pitch: -6, roll: -3)
            rotateBone(.spine, on: model, pitch: 4)
            rotateBone(.chest, on: model, yaw: -5, pitch: 4)
            rotateBone(.neck, on: model, yaw: -4, pitch: 4)

            rotateBone(.leftUpperLeg, on: model, yaw: 6, pitch: -58, roll: -8)
            rotateBone(.leftLowerLeg, on: model, pitch: 70)
            rotateBone(.leftFoot, on: model, pitch: 22)
            rotateBone(.rightUpperLeg, on: model, yaw: -8, pitch: -28, roll: 8)
            rotateBone(.rightLowerLeg, on: model, pitch: 38)
            rotateBone(.rightFoot, on: model, pitch: 18)

            // Left hand poised gently forward, right arm relaxed
            rotateBone(.leftUpperArm, on: model, yaw: -24, pitch: -16, roll: -50)
            rotateBone(.leftLowerArm, on: model, yaw: -44, pitch: -14)
            rotateBone(.rightUpperArm, on: model, yaw: 16, pitch: -8, roll: 58)
            rotateBone(.rightLowerArm, on: model, yaw: 24, pitch: -8)
        }
        model.updateNodeTransforms()
    }

    private func applyHandGestures(to model: VRMModel) {
        let allFingers: [VRMHumanoidBone] = [
            .leftThumbMetacarpal, .leftThumbProximal, .leftThumbDistal,
            .leftIndexProximal, .leftIndexIntermediate, .leftIndexDistal,
            .leftMiddleProximal, .leftMiddleIntermediate, .leftMiddleDistal,
            .leftRingProximal, .leftRingIntermediate, .leftRingDistal,
            .leftLittleProximal, .leftLittleIntermediate, .leftLittleDistal,
            .rightThumbMetacarpal, .rightThumbProximal, .rightThumbDistal,
            .rightIndexProximal, .rightIndexIntermediate, .rightIndexDistal,
            .rightMiddleProximal, .rightMiddleIntermediate, .rightMiddleDistal,
            .rightRingProximal, .rightRingIntermediate, .rightRingDistal,
            .rightLittleProximal, .rightLittleIntermediate, .rightLittleDistal
        ]
        resetHumanoidBones(allFingers, on: model)

        for (bone, q) in recipeStore.fingerRotations(for: leftHandGesture, isLeft: true) {
            if let idx = model.humanoid?.getBoneNode(bone), idx < model.nodes.count, let rest = restNodeTransforms[idx] {
                model.nodes[idx].rotation = rest.rot * q
                model.nodes[idx].updateLocalMatrix()
            }
        }
        for (bone, q) in recipeStore.fingerRotations(for: rightHandGesture, isLeft: false) {
            if let idx = model.humanoid?.getBoneNode(bone), idx < model.nodes.count, let rest = restNodeTransforms[idx] {
                model.nodes[idx].rotation = rest.rot * q
                model.nodes[idx].updateLocalMatrix()
            }
        }
        model.updateNodeTransforms()
    }

    private func applyHairStyle(_ style: HairStyleOption, to model: VRMModel) {
        // Restore hair rest transforms first
        for (name, idx) in hairNodeIndices where idx < model.nodes.count {
            if let rest = restNodeTransforms[idx] {
                model.nodes[idx].rotation = rest.rot
                model.nodes[idx].updateLocalMatrix()
            }
            _ = name
        }

        // Style A = author default; Style B / B2 tuck & flare side/back locks subtly relative to rest
        switch style {
        case .styleA:
            break
        case .styleB:
            if let lIdx = hairNodeIndices["Hair_Side_01.L.001"], let rest = restNodeTransforms[lIdx] {
                model.nodes[lIdx].rotation = simd_quatf(angle: 0.28, axis: SIMD3<Float>(0, 0, 1)) * rest.rot
                model.nodes[lIdx].updateLocalMatrix()
            }
            if let rIdx = hairNodeIndices["Hair_Side_01.R.001"], let rest = restNodeTransforms[rIdx] {
                model.nodes[rIdx].rotation = simd_quatf(angle: -0.28, axis: SIMD3<Float>(0, 0, 1)) * rest.rot
                model.nodes[rIdx].updateLocalMatrix()
            }
            if let bIdx = hairNodeIndices["HairBack_Root"], let rest = restNodeTransforms[bIdx] {
                model.nodes[bIdx].rotation = simd_quatf(angle: 0.22, axis: SIMD3<Float>(1, 0, 0)) * rest.rot
                model.nodes[bIdx].updateLocalMatrix()
            }
        case .styleB2:
            if let bIdx = hairNodeIndices["HairBack_Root"], let rest = restNodeTransforms[bIdx] {
                model.nodes[bIdx].rotation = simd_quatf(angle: -0.18, axis: SIMD3<Float>(1, 0, 0)) * rest.rot
                model.nodes[bIdx].updateLocalMatrix()
            }
            if let lBack = hairNodeIndices["Hair_Back_02.L.001"], let rest = restNodeTransforms[lBack] {
                model.nodes[lBack].rotation = simd_quatf(angle: 0.32, axis: SIMD3<Float>(0, 0, 1)) * rest.rot
                model.nodes[lBack].updateLocalMatrix()
            }
            if let rBack = hairNodeIndices["Hair_Back_02.R.001"], let rest = restNodeTransforms[rBack] {
                model.nodes[rBack].rotation = simd_quatf(angle: -0.32, axis: SIMD3<Float>(0, 0, 1)) * rest.rot
                model.nodes[rBack].updateLocalMatrix()
            }
        }
    }

    // MARK: - Timers & Camera Tracking

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

    func startCameraCapture(deviceID: String) {
        AppLogger.shared.info("Camera", "Requesting camera capture start: \(deviceID)")
        clearCustomExpression()
        renderer.expressionController?.setMood(.neutral, intensity: 0.0)
        activeCameraID = deviceID
        yawFilter.reset()
        pitchFilter.reset()
        rollFilter.reset()
        gazeXFilter.reset()
        gazeYFilter.reset()
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

        if let exprCtrl = renderer.expressionController {
            let zeros = Dictionary(uniqueKeysWithValues: ARKitFaceBlendShapes.allKeys.map { ($0, Float(0)) })
            faceDriver.update(
                blendShapes: ARKitFaceBlendShapes(timestamp: Date().timeIntervalSinceReferenceDate, shapes: zeros),
                controller: exprCtrl,
                maxAge: 1.0
            )
        }
        applyAllBodyAndOutfitState()
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
        let now = CACurrentMediaTime()
        smoothedHeadYaw = yawFilter.filter(frame.headYaw, timestamp: now)
        smoothedHeadPitch = pitchFilter.filter(frame.headPitch, timestamp: now)
        smoothedHeadRoll = rollFilter.filter(frame.headRoll, timestamp: now)
        latestCameraGaze = SIMD2<Float>(
            gazeXFilter.filter(frame.gazeOffset.x, timestamp: now),
            gazeYFilter.filter(frame.gazeOffset.y, timestamp: now)
        )
    }

    // MARK: - Expression Controls

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

        let elapsed = Float(CACurrentMediaTime() - startTime)
        updateProceduralMotion(in: view, elapsed: elapsed)

        let drawProps = propRenderer.hasActiveProps
        if drawProps {
            if let resolveTex = descriptor.colorAttachments[0].resolveTexture {
                descriptor.colorAttachments[0].storeAction = .storeAndMultisampleResolve
                _ = resolveTex
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
                bookWorldMatrix: computeBookWorldMatrix(time: elapsed),
                time: elapsed
            )
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    private func computeBookWorldMatrix(time: Float) -> simd_float4x4 {
        let bob = (poseMode == .afkBook || poseMode == .afkTea) ? sin(time * 1.4) * 0.015 : 0.0
        // Position the open book right in front of Alice's hands/chest (+Z)
        let pos = SIMD3<Float>(0.0, 0.97 + bob, 0.30)
        // Tilt book slightly open toward Alice & camera so inner pages are visible
        let tilt = simd_quatf(angle: 0.42, axis: SIMD3<Float>(1, 0, 0))
        let yaw = simd_quatf(angle: sin(time * 0.6) * 0.05, axis: SIMD3<Float>(0, 1, 0))
        let rot = float4x4(yaw * tilt)
        var t = matrix_identity_float4x4
        t.columns.3 = SIMD4<Float>(pos.x, pos.y, pos.z, 1.0)
        return t * rot
    }

    private func updateProceduralMotion(in view: MTKView, elapsed t: Float) {
        guard let model else { return }
        let breath = sin(t * 1.7)

        if idleBreathingEnabled {
            let baseChest = restNodeTransforms[model.humanoid?.getBoneNode(.chest) ?? -1]?.rot
            if let chestIdx = model.humanoid?.getBoneNode(.chest), let base = baseChest {
                let poseOffset: Float = (poseMode == .afkBook) ? (12.0 * .pi / 180.0) : ((poseMode == .afkTea) ? (4.0 * .pi / 180.0) : 0.0)
                let pitch = simd_quatf(angle: poseOffset + breath * 0.014, axis: SIMD3<Float>(1, 0, 0))
                let yawOffset: Float = (poseMode == .afkBook) ? (-6.0 * .pi / 180.0) : ((poseMode == .afkTea) ? (-5.0 * .pi / 180.0) : 0.0)
                let yaw = simd_quatf(angle: yawOffset, axis: SIMD3<Float>(0, 1, 0))
                model.nodes[chestIdx].rotation = yaw * pitch * base
                model.nodes[chestIdx].updateLocalMatrix()
            }

            // Gentle levitation hover in AFK modes
            if poseMode == .afkBook || poseMode == .afkTea {
                if let hipsIdx = model.humanoid?.getBoneNode(.hips), let rest = restNodeTransforms[hipsIdx] {
                    let baseHover: Float = (poseMode == .afkBook) ? 0.04 : 0.02
                    model.nodes[hipsIdx].translation = rest.pos + SIMD3<Float>(0, baseHover + sin(t * 1.35) * 0.018, 0)
                    model.nodes[hipsIdx].updateLocalMatrix()
                }
            }
        }

        if cameraTrackingActive {
            if let neckIdx = model.humanoid?.getBoneNode(.neck), let base = restNodeTransforms[neckIdx]?.rot {
                let qYaw = simd_quatf(angle: smoothedHeadYaw * 0.35, axis: SIMD3<Float>(0, 1, 0))
                let qPitch = simd_quatf(angle: smoothedHeadPitch * 0.35, axis: SIMD3<Float>(1, 0, 0))
                let qRoll = simd_quatf(angle: smoothedHeadRoll * 0.35, axis: SIMD3<Float>(0, 0, 1))
                model.nodes[neckIdx].rotation = base * qYaw * qPitch * qRoll
                model.nodes[neckIdx].updateLocalMatrix()
            }
            if let headIdx = model.humanoid?.getBoneNode(.head), let base = restNodeTransforms[headIdx]?.rot {
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

        if poseMode == .afkBook {
            renderer.lookAtController?.target = .headLocalPoint(SIMD3<Float>(0, -0.35, 1.0))
        } else if idleBreathingEnabled {
            let sway = sin(t * 0.75)
            if let headIdx = model.humanoid?.getBoneNode(.head), let base = restNodeTransforms[headIdx]?.rot {
                let basePitch: Float = 0.0
                let roll = simd_quatf(angle: sway * 0.018, axis: SIMD3<Float>(0, 0, 1))
                let pitch = simd_quatf(angle: basePitch + breath * 0.01, axis: SIMD3<Float>(1, 0, 0))
                model.nodes[headIdx].rotation = base * roll * pitch
                model.nodes[headIdx].updateLocalMatrix()
            }
        }

        if eyeTrackingEnabled, poseMode != .afkBook, let lookAt = renderer.lookAtController, let window = view.window {
            let mouseScreen = NSEvent.mouseLocation
            let winFrame = window.frame
            let normX = Float(((mouseScreen.x - winFrame.midX) / max(winFrame.width, 1)).clamped(to: -1.5...1.5))
            let normY = Float(((mouseScreen.y - winFrame.midY) / max(winFrame.height, 1)).clamped(to: -1.5...1.5))
            lookAt.target = .headLocalPoint(SIMD3<Float>(normX * 0.85, normY * 0.6, 1.5))
        }
    }

    // MARK: - Offscreen Snapshot (CLI verification)

    func renderSnapshot(to outputURL: URL, width: Int = 720, height: Int = 1024, animTime: Float = 0.8) async throws {
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
                bookWorldMatrix: computeBookWorldMatrix(time: animTime),
                time: animTime
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

private extension Comparable {
    func clamped(to limits: ClosedRange<Self>) -> Self {
        min(max(self, limits.lowerBound), limits.upperBound)
    }
}
