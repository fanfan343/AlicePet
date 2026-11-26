import Foundation
import VRMMetalKit
import simd

/// Headless CLI `--snapshot` mode used by `tools/verify_visuals.sh` and Picture-World regression tests.
@MainActor
func runHeadlessSnapshot(modelURL: URL, outputPath: String) -> Never {
    runMainActorCLI("Snapshot") {
        let controller = try PetController(sampleCount: 1)
        try await controller.loadModel(from: modelURL, synchronousPhysics: true)

        if let viewRaw = parseCLIArg("--view"), let preset = CameraViewPreset(rawValue: viewRaw) {
            controller.camera.apply(preset: preset)
        }
        if let yawStr = parseCLIArg("--yaw"), let yawDeg = Float(yawStr) {
            controller.camera.yaw = yawDeg * .pi / 180.0
        }
        if let pitchStr = parseCLIArg("--pitch"), let pitchDeg = Float(pitchStr) {
            controller.camera.pitch = pitchDeg * .pi / 180.0
        }
        if let lightRaw = parseCLIArg("--lighting"), let preset = LightingPreset(rawValue: lightRaw) {
            controller.setLightingPreset(preset)
        }
        if let outlineRaw = parseCLIArg("--outline"), let style = OutlineStyle(rawValue: outlineRaw) {
            controller.setOutlineStyle(style)
        }
        if let poseRaw = parseCLIArg("--pose"), let pose = PetPoseMode(rawValue: poseRaw) {
            controller.setPoseMode(pose)
        }
        if let propRaw = parseCLIArg("--prop"), let prop = PropSceneRenderer.ActivePropMode(rawValue: propRaw) {
            controller.manualPropOverride = prop
        }
        if let handRaw = parseCLIArg("--hand"), let hand = HandGesturePreset(rawValue: handRaw) {
            controller.setHandGesture(hand, side: .both)
        }
        if let offList = parseCLIArg("--outfit-off") {
            for key in offList.split(separator: ",").map({ $0.trimmingCharacters(in: .whitespaces).lowercased() }) {
                switch key {
                case "dress": controller.outfitState.showDress = false
                case "pantyhose": controller.outfitState.showPantyhose = false
                case "shoes": controller.outfitState.showShoes = false
                case "hairband": controller.outfitState.showHairBand = false
                case "ribbon": controller.outfitState.showChestRibbon = false
                case "hairside": controller.outfitState.showHairSide = false
                default: break
                }
            }
            controller.applyOutfitState()
        }
        if let hairRaw = parseCLIArg("--hair") {
            switch hairRaw.uppercased() {
            case "B": controller.outfitState.hairStyle = .styleB
            case "B2": controller.outfitState.hairStyle = .styleB2
            default: controller.outfitState.hairStyle = .styleA
            }
            controller.applyOutfitState()
        }

        if let expr = parseCLIArg("--expression") {
            controller.applyExpression(named: expr, weight: 1.0)
        } else if CommandLine.arguments.contains("--test-arkit") {
            if let exprCtrl = controller.renderer.expressionController {
                let shapes = ARKitFaceBlendShapes(
                    timestamp: Date().timeIntervalSinceReferenceDate,
                    shapes: [
                        ARKitFaceBlendShapes.jawOpen: 0.85,
                        ARKitFaceBlendShapes.eyeBlinkLeft: 1.0,
                        ARKitFaceBlendShapes.mouthSmileLeft: 0.8,
                        ARKitFaceBlendShapes.mouthSmileRight: 0.8,
                        ARKitFaceBlendShapes.browInnerUp: 0.7,
                    ]
                )
                controller.faceDriver.update(blendShapes: shapes, controller: exprCtrl, maxAge: 1.0)
                print("ARKit stats:", controller.faceDriver.getStatistics())
            }
        }

        if let emoName = parseCLIArg("--emotion") {
            let intensity = Float(parseCLIArg("--emotion-strength") ?? "0.85") ?? 0.85
            controller.emotionStateMachine.applyImpulse(emotion: emoName, intensity: intensity)
            if parseCLIArg("--expression") == nil {
                controller.setPresetMood(controller.emotionStateMachine.recommendedVRMExpression)
            }
        }
        let testSpeechGesture = CommandLine.arguments.contains("--test-speech-gesture")
        if testSpeechGesture {
            controller.animationMixer.speechGestureLayer.debugSpeechEnergyOverride = 0.82
        }

        if let vrmaPath = parseCLIArg("--vrma"), let model = controller.model {
            let vrmaURL = URL(fileURLWithPath: vrmaPath)
            _ = try controller.playVRMA(from: vrmaURL, loop: true)
            let seekTime = Float(parseCLIArg("--vrma-time") ?? "0.6") ?? 0.6
            controller.poseController.applyBasePose(to: model, restTransforms: controller.poseController.restNodeTransforms)
            controller.animationMixer.vrmaPlayer.seekAndApply(time: seekTime, model: model, renderer: controller.renderer)
            model.updateNodeTransforms()
            controller.settleSpringBonePhysics(warmupSteps: 20)
        } else if testSpeechGesture || parseCLIArg("--emotion") != nil, let model = controller.model {
            controller.animationMixer.evaluate(
                model: model, renderer: controller.renderer, poseController: controller.poseController,
                recipeStore: controller.recipeStore, expressionMixer: controller.expressionMixer,
                emotionStateMachine: controller.emotionStateMachine,
                elapsed: 1.2, dt: 0.0, idleBreathingEnabled: true, poseMode: controller.poseMode, window: nil
            )
            controller.settleSpringBonePhysics(warmupSteps: 16)
        }

        if let sceneDir = parseCLIArg("--scene") {
            if try await applyPictureWorldCLIOptions(
                controller: controller,
                sceneDir: sceneDir,
                outputPath: outputPath
            ) {
                return
            }
        }

        var snapW = 720
        var snapH = 1024
        if let sizeRaw = parseCLIArg("--size") {
            let parts = sizeRaw.lowercased().split(separator: "x").compactMap { Int($0) }
            if parts.count == 2, parts[0] > 64, parts[1] > 64 {
                snapW = parts[0]
                snapH = parts[1]
            }
        } else if controller.isPictureWorldActive, let cfg = controller.pictureWorldRenderer.config {
            let scale = min(1.0, 1280.0 / Double(max(cfg.imageWidth, cfg.imageHeight)))
            snapW = max(320, Int((Double(cfg.imageWidth) * scale).rounded()))
            snapH = max(240, Int((Double(cfg.imageHeight) * scale).rounded()))
        }

        let skipBase = (parseCLIArg("--vrma") != nil) || testSpeechGesture || (parseCLIArg("--emotion") != nil)
        try await controller.renderSnapshot(to: URL(fileURLWithPath: outputPath), width: snapW, height: snapH, applyBaseState: !skipBase)
        print("Snapshot saved to \(outputPath)")
    }
}

/// Returns `true` if a sub-mode (such as `--test-walk-and-sit`) already wrote the final snapshot.
@MainActor
private func applyPictureWorldCLIOptions(
    controller: PetController,
    sceneDir: String,
    outputPath: String
) async throws -> Bool {
    let lowerScene = sceneDir.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if lowerScene == "empty" || lowerScene == "transparent" || lowerScene == "empty_transparent" {
        controller.loadEmptyTransparentPictureWorld()
    } else {
        try controller.loadPictureWorld(from: URL(fileURLWithPath: sceneDir))
    }
    if let posRaw = parseCLIArg("--avatar-pos") {
        let parts = posRaw.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
        if parts.count == 2 {
            controller.pictureWorldRenderer.avatarWorldPosition = SIMD3<Float>(parts[0], 0, parts[1])
        } else if parts.count == 3 {
            controller.pictureWorldRenderer.avatarWorldPosition = SIMD3<Float>(parts[0], parts[1], parts[2])
        }
    }
    if let yawDegStr = parseCLIArg("--avatar-yaw"), let yawDeg = Float(yawDegStr) {
        controller.pictureWorldRenderer.avatarWorldYaw = yawDeg * .pi / 180.0
    }
    let snapAspect: Float = {
        if let cfg = controller.pictureWorldRenderer.config {
            return Float(cfg.imageWidth) / max(Float(cfg.imageHeight), 1.0)
        }
        return 820.0 / 480.0
    }()
    if let zoomStr = parseCLIArg("--pw-zoom"), let z = Float(zoomStr), z > 1.0 {
        for _ in 0..<24 where controller.pictureWorldRenderer.canvasZoom < z {
            controller.pictureWorldRenderer.zoomCanvas(deltaY: 2.0, viewportAspect: snapAspect)
        }
    }
    if let panRaw = parseCLIArg("--pw-pan") {
        let parts = panRaw.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
        if parts.count == 2 {
            controller.pictureWorldRenderer.panCanvas(deltaX: parts[0], deltaY: parts[1], viewportSize: CGSize(width: 1280, height: 848))
        }
    }
    if CommandLine.arguments.contains("--pw-test-wall-walk") {
        try runWallWalkCheck(controller: controller, aspect: snapAspect)
    }
    if let seatId = parseCLIArg("--test-walk-and-sit") {
        try await runWalkAndSitStoryboard(
            controller: controller,
            seatId: seatId,
            aspect: snapAspect,
            outputPath: outputPath
        )
        return true
    }
    if let json = controller.currentAvatarSceneState(aspect: snapAspect).toJSONString(pretty: false) {
        print("AvatarSceneState:", json)
    }
    return false
}

@MainActor
private func runWallWalkCheck(controller: PetController, aspect: Float) throws {
    controller.pictureWorldRenderer.resetCanvasAndAvatarToOrigin()
    for step in 0...120 {
        let jitterX = sin(Float(step) * 0.25) * 0.004
        let jitterZ = cos(Float(step) * 0.25) * 0.004
        controller.pictureWorldRenderer.applyARDYRootMotion(worldRootXZ: SIMD2<Float>(jitterX, jitterZ), viewportAspect: aspect)
    }
    let standPos = controller.pictureWorldRenderer.avatarWorldPosition
    let standDrift = simd_length(SIMD2<Float>(standPos.x, standPos.z))
    print(String(format: "StandingDriftCheck (120 frames): pos=(%.4f,%.4f) drift=%.4fm (must be <0.02m)", standPos.x, standPos.z, standDrift))
    guard standDrift < 0.02 else {
        throw NSError(domain: "AlicePet.PictureWorld", code: 99, userInfo: [NSLocalizedDescriptionKey: "Phantom floor drift detected: \(standDrift)m"])
    }

    controller.pictureWorldRenderer.resetCanvasAndAvatarToOrigin()
    for step in 0...80 {
        controller.pictureWorldRenderer.applyARDYRootMotion(worldRootXZ: SIMD2<Float>(Float(step) * 0.05, 0), viewportAspect: aspect)
        if step == 20 || step == 40 || step == 60 || step == 80 {
            let p = controller.pictureWorldRenderer.avatarWorldPosition
            let yawD = controller.pictureWorldRenderer.avatarWorldYaw * 180.0 / .pi
            print(String(format: "WallWalk step=%d pos=(%.2f,%.2f) yaw=%.1fdeg contacts=%@", step, p.x, p.z, yawD, controller.pictureWorldRenderer.lastBoundaryContacts.joined(separator: ",")))
        }
    }
}

@MainActor
private func runWalkAndSitStoryboard(
    controller: PetController,
    seatId: String,
    aspect: Float,
    outputPath: String
) async throws {
    guard controller.walkToSceneAnchorAndSit(anchorId: seatId) else {
        throw NSError(domain: "AlicePet.WalkSit", code: 80, userInfo: [NSLocalizedDescriptionKey: "Failed to start walk-and-sit for anchor '\(seatId)'"])
    }
    let outPrefix = outputPath.replacingOccurrences(of: ".png", with: "")
    var capturedMidWalk = false
    var capturedAlign = false
    let dtStep: TimeInterval = 1.0 / 30.0
    let simDeadline = Date().addingTimeInterval(25.0)
    var elapsedSim: Float = 0

    while Date() < simDeadline {
        if let model = controller.model {
            controller.animationMixer.evaluate(
                model: model, renderer: controller.renderer, poseController: controller.poseController,
                recipeStore: controller.recipeStore, expressionMixer: controller.expressionMixer,
                elapsed: elapsedSim, dt: dtStep, idleBreathingEnabled: true, poseMode: controller.poseMode, window: nil
            )
            if controller.animationMixer.activeBodySource == .ardy,
               let rootXZ = controller.animationMixer.ardyChannel.latestWorldRootXZ {
                controller.pictureWorldRenderer.applyARDYRootMotion(
                    worldRootXZ: rootXZ, viewportAspect: aspect, goalTargetXZ: controller.sceneActionController.activeWalkGoalXZ
                )
            } else {
                controller.pictureWorldRenderer.resetARDYRootTracking()
            }
            controller.sceneActionController.tick(dt: dtStep, on: controller, viewportAspect: aspect)
            controller.applyPictureWorldRootTransform()
        }

        if !capturedMidWalk, case .walkingToAnchor = controller.sceneActionController.phase,
           simd_length(SIMD2<Float>(controller.pictureWorldRenderer.avatarWorldPosition.x, controller.pictureWorldRenderer.avatarWorldPosition.z)) > 0.30 {
            capturedMidWalk = true
            controller.settleSpringBonePhysics(warmupSteps: 8)
            try await controller.renderSnapshot(to: URL(fileURLWithPath: "\(outPrefix)_1_walk.png"), width: 1280, height: 848, applyBaseState: false)
        }
        if !capturedAlign, case .aligningToAnchor = controller.sceneActionController.phase {
            capturedAlign = true
            controller.settleSpringBonePhysics(warmupSteps: 8)
            try await controller.renderSnapshot(to: URL(fileURLWithPath: "\(outPrefix)_2_arrive.png"), width: 1280, height: 848, applyBaseState: false)
        }
        if case .seatedIdle = controller.sceneActionController.phase {
            for _ in 0..<18 {
                elapsedSim += Float(dtStep)
                if let model = controller.model {
                    controller.animationMixer.evaluate(
                        model: model, renderer: controller.renderer, poseController: controller.poseController,
                        recipeStore: controller.recipeStore, expressionMixer: controller.expressionMixer,
                        elapsed: elapsedSim, dt: dtStep, idleBreathingEnabled: true, poseMode: controller.poseMode, window: nil
                    )
                    controller.applySeatedUprightPostureIfNeeded()
                    controller.applyPictureWorldRootTransform()
                }
            }
            controller.settleSpringBonePhysics(warmupSteps: 32)
            if let json = controller.currentAvatarSceneState(aspect: aspect).toJSONString(pretty: false) {
                print("WalkSit Stage 3 (Seated Idle) AvatarSceneState:", json)
            }
            try await controller.renderSnapshot(to: URL(fileURLWithPath: outputPath), width: 1280, height: 848, applyBaseState: false)
            print("WalkSit Stage 3 (Seated Idle) saved to \(outputPath)")
            controller.stopAllTrackingSync()
            return
        }
        elapsedSim += Float(dtStep)
        try await Task.sleep(nanoseconds: 33_000_000)
    }
    controller.stopAllTrackingSync()
    throw NSError(domain: "AlicePet.WalkSit", code: 81, userInfo: [NSLocalizedDescriptionKey: "Walk-and-sit timed out in phase \(controller.sceneActionController.phase)"])
}
