import AppKit
import Foundation

let preferredDefaultModelPaths = [
    "/Users/funnyliu/Downloads/VRM项目/降采样模型/Alicephil VRM (TexLite).vrm",
    "/Users/funnyliu/Downloads/VRM项目/降采样模型/Alicephil VRM (LightShapekey) (TexLite).vrm",
]
let defaultModelPath = preferredDefaultModelPaths.first(where: { FileManager.default.fileExists(atPath: $0) })
    ?? preferredDefaultModelPaths[0]

func parseCLIArg(_ flag: String) -> String? {
    let args = CommandLine.arguments
    guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else { return nil }
    return args[idx + 1]
}

@MainActor
func runMainActorCLI(_ label: String, _ work: @escaping @MainActor () async throws -> Void) -> Never {
    var done = false
    var exitCode: Int32 = 0
    Task { @MainActor in
        do {
            try await work()
        } catch {
            fputs("\(label) error: \(error)\n", stderr)
            exitCode = 1
        }
        done = true
    }
    while !done {
        RunLoop.main.run(mode: .default, before: Date(timeIntervalSinceNow: 0.01))
    }
    exit(exitCode)
}

private func resolveInitialModelURL(isHeadlessSnapshot: Bool) -> URL {
    if let explicit = parseCLIArg("--model") {
        return URL(fileURLWithPath: explicit)
    }
    if !isHeadlessSnapshot,
       let saved = UserDefaults.standard.string(forKey: AppDelegate.lastModelPathKey),
       FileManager.default.fileExists(atPath: saved) {
        return URL(fileURLWithPath: saved)
    }
    return URL(fileURLWithPath: defaultModelPath)
}

private let vmcLoopbackPath = parseCLIArg("--test-vmc-loopback")
private let motionServiceTestPath = parseCLIArg("--test-motion-service")
private let audioFaceTestPath = parseCLIArg("--test-audio2face")
private let a2f3dTestPath = parseCLIArg("--test-a2f3d")
private let ardyTestPath = parseCLIArg("--test-ardy")
private let mixerTestPath = parseCLIArg("--test-mixer")
private let snapshotPath = parseCLIArg("--snapshot")

private let isHeadlessMode = [
    snapshotPath, vmcLoopbackPath, motionServiceTestPath,
    audioFaceTestPath, a2f3dTestPath, ardyTestPath, mixerTestPath,
].contains { $0 != nil }

private let modelURL = resolveInitialModelURL(isHeadlessSnapshot: isHeadlessMode)

if let out = vmcLoopbackPath { runVMCLoopbackTest(modelURL: modelURL, outputPath: out) }
if let out = motionServiceTestPath { runMotionServiceTest(modelURL: modelURL, outputPath: out) }
if let out = audioFaceTestPath { runAudioFaceTest(modelURL: modelURL, outputPath: out) }
if let out = a2f3dTestPath { runA2F3DTest(modelURL: modelURL, outputPath: out) }
if let out = ardyTestPath { runArdyTest(modelURL: modelURL, outputPath: out) }
if let out = mixerTestPath { runMixerTest(modelURL: modelURL, outputPath: out) }
if let out = snapshotPath { runHeadlessSnapshot(modelURL: modelURL, outputPath: out) }

@MainActor
private func runDesktopApp() {
    let app = NSApplication.shared
    let delegate = AppDelegate(modelURL: modelURL)
    app.delegate = delegate
    app.run()
}

runDesktopApp()
