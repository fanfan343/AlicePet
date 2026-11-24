import AppKit
import Foundation
import VRMMetalKit

let defaultModelPath = "/Users/funnyliu/Downloads/VRM项目/降采样模型/Alicephil VRM (LightShapekey) (TexLite).vrm"

func parseArg(_ flag: String) -> String? {
    let args = CommandLine.arguments
    guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else { return nil }
    return args[idx + 1]
}

let modelURL = URL(fileURLWithPath: parseArg("--model") ?? defaultModelPath)

if let snapshotPath = parseArg("--snapshot") {
    var done = false
    var exitCode: Int32 = 0
    Task { @MainActor in
        do {
            let controller = try PetController(sampleCount: 1)
            try await controller.loadModel(from: modelURL, synchronousPhysics: true)
            if let expr = parseArg("--expression") {
                controller.setCustomExpression(expr, weight: 1.0)
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
