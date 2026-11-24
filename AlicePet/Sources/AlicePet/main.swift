import AppKit
import Foundation
import VRMMetalKit

let defaultModelPath = "/Users/funnyliu/Downloads/VRM项目/降采样模型/Alicephil VRM (LightShapekey) (TexLite).vrm"

func parseArg(_ flag: String) -> String? {
    let args = CommandLine.arguments
    guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else { return nil }
    return args[idx + 1]
}

func resolveInitialModelURL(isHeadlessSnapshot: Bool) -> URL {
    if let explicit = parseArg("--model") {
        return URL(fileURLWithPath: explicit)
    }
    if !isHeadlessSnapshot,
       let saved = UserDefaults.standard.string(forKey: AppDelegate.lastModelPathKey),
       FileManager.default.fileExists(atPath: saved) {
        return URL(fileURLWithPath: saved)
    }
    return URL(fileURLWithPath: defaultModelPath)
}

let isSnapshotMode = (parseArg("--snapshot") != nil)
let modelURL = resolveInitialModelURL(isHeadlessSnapshot: isSnapshotMode)

if let snapshotPath = parseArg("--snapshot") {
    var done = false
    var exitCode: Int32 = 0
    Task { @MainActor in
        do {
            let controller = try PetController(sampleCount: 1)
            try await controller.loadModel(from: modelURL, synchronousPhysics: true)

            if let lightRaw = parseArg("--lighting"), let preset = LightingPreset(rawValue: lightRaw) {
                controller.setLightingPreset(preset)
            }
            if let outlineRaw = parseArg("--outline"), let style = OutlineStyle(rawValue: outlineRaw) {
                controller.setOutlineStyle(style)
            }
            if let poseRaw = parseArg("--pose"), let pose = PetPoseMode(rawValue: poseRaw) {
                controller.setPoseMode(pose)
            }
            if let propRaw = parseArg("--prop"), let prop = PropSceneRenderer.ActivePropMode(rawValue: propRaw) {
                controller.manualPropOverride = prop
            }
            if let handRaw = parseArg("--hand"), let hand = HandGesturePreset(rawValue: handRaw) {
                controller.setHandGesture(hand, side: .both)
            }
            if let offList = parseArg("--outfit-off") {
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
            if let hairRaw = parseArg("--hair") {
                switch hairRaw.uppercased() {
                case "B": controller.outfitState.hairStyle = .styleB
                case "B2": controller.outfitState.hairStyle = .styleB2
                default: controller.outfitState.hairStyle = .styleA
                }
                controller.applyOutfitState()
            }

            if let expr = parseArg("--expression") {
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
                            ARKitFaceBlendShapes.browInnerUp: 0.7
                        ]
                    )
                    controller.faceDriver.update(blendShapes: shapes, controller: exprCtrl, maxAge: 1.0)
                    print("ARKit stats:", controller.faceDriver.getStatistics())
                }
            }

            try await controller.renderSnapshot(to: URL(fileURLWithPath: snapshotPath))
            print("Snapshot saved to \(snapshotPath)")
        } catch {
            fputs("Snapshot error: \(error)\n", stderr)
            exitCode = 1
        }
        done = true
    }
    while !done {
        RunLoop.main.run(mode: .default, before: Date(timeIntervalSinceNow: 0.01))
    }
    exit(exitCode)
}

@MainActor
func runDesktopApp() {
    let app = NSApplication.shared
    let delegate = AppDelegate(modelURL: modelURL)
    app.delegate = delegate
    app.run()
}

runDesktopApp()
