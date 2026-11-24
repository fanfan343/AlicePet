import AppKit
import Darwin
import Foundation
import VRMMetalKit
import simd

let preferredDefaultModelPaths = [
    "/Users/funnyliu/Downloads/VRM项目/降采样模型/Alicephil VRM (TexLite).vrm",
    "/Users/funnyliu/Downloads/VRM项目/降采样模型/Alicephil VRM (LightShapekey) (TexLite).vrm",
]
let defaultModelPath = preferredDefaultModelPaths.first(where: { FileManager.default.fileExists(atPath: $0) })
    ?? preferredDefaultModelPaths[0]

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

let vmcLoopbackPath = parseArg("--test-vmc-loopback")
let motionServiceTestPath = parseArg("--test-motion-service")
let isSnapshotMode = (parseArg("--snapshot") != nil) || (vmcLoopbackPath != nil) || (motionServiceTestPath != nil)
let modelURL = resolveInitialModelURL(isHeadlessSnapshot: isSnapshotMode)

func sendLoopbackUDP(_ data: Data, toPort port: UInt16) throws {
    let sock = Darwin.socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
    guard sock >= 0 else {
        throw NSError(domain: "AlicePet.VMC", code: Int(errno), userInfo: [NSLocalizedDescriptionKey: "socket() failed"])
    }
    defer { Darwin.close(sock) }

    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = port.bigEndian
    addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))

    let sent = data.withUnsafeBytes { rawBuf -> ssize_t in
        guard let base = rawBuf.baseAddress else { return -1 }
        return withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { saPtr in
                Darwin.sendto(sock, base, data.count, 0, saPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
    }
    guard sent == data.count else {
        throw NSError(domain: "AlicePet.VMC", code: Int(errno), userInfo: [NSLocalizedDescriptionKey: "sendto() sent \(sent)/\(data.count) bytes"])
    }
}

if let loopbackOutputPath = vmcLoopbackPath {
    var done = false
    var exitCode: Int32 = 0
    Task { @MainActor in
        do {
            let controller = try PetController(sampleCount: 1)
            try await controller.loadModel(from: modelURL, synchronousPhysics: true)

            if let viewRaw = parseArg("--view"), let preset = CameraViewPreset(rawValue: viewRaw) {
                controller.camera.apply(preset: preset)
            } else {
                controller.camera.apply(preset: .upperBody)
            }

            guard controller.startVMC(port: 0), let boundPort = controller.vmcController.boundPort else {
                throw NSError(domain: "AlicePet.VMC", code: 10, userInfo: [NSLocalizedDescriptionKey: "Failed to bind VMC UDP loopback port"])
            }

            // Build a real Unity-coordinate VMC OSC bundle (head tilt + waving left arm + mixed-case ARKit 52 wink/smile/open mouth)
            func boneMsg(_ name: String, q: simd_quatf) -> OSCPacket {
                .message(OSCMessage(
                    "/VMC/Ext/Bone/Pos",
                    .string(name),
                    .float32(0), .float32(0), .float32(0),
                    .float32(q.imag.x), .float32(q.imag.y), .float32(q.imag.z), .float32(q.real)
                ))
            }
            func blendMsg(_ name: String, _ weight: Float) -> OSCPacket {
                .message(OSCMessage("/VMC/Ext/Blend/Val", .string(name), .float32(weight)))
            }

            let qHead = simd_quatf(angle: 0.22, axis: SIMD3<Float>(0, 0, 1)) * simd_quatf(angle: -0.18, axis: SIMD3<Float>(0, 1, 0))
            let qNeck = simd_quatf(angle: 0.08, axis: SIMD3<Float>(0, 0, 1))
            let qLeftUpperArm = simd_quatf(angle: -0.55, axis: SIMD3<Float>(0, 0, 1)) * simd_quatf(angle: 0.35, axis: SIMD3<Float>(1, 0, 0))
            let qLeftLowerArm = simd_quatf(angle: -1.15, axis: SIMD3<Float>(0, 0, 1))
            let qRightUpperArm = simd_quatf(angle: -1.15, axis: SIMD3<Float>(0, 0, 1))

            let bundle = OSCPacket.bundle(OSCBundle(elements: [
                .message(OSCMessage("/VMC/Ext/OK", .int32(1))),
                boneMsg("Head", q: qHead),
                boneMsg("Neck", q: qNeck),
                boneMsg("LeftUpperArm", q: qLeftUpperArm),
                boneMsg("LeftLowerArm", q: qLeftLowerArm),
                boneMsg("RightUpperArm", q: qRightUpperArm),
                blendMsg("EyeBlinkLeft", 1.0),
                blendMsg("jawOpen", 0.72),
                blendMsg("MouthSmileLeft", 0.85),
                blendMsg("mouthSmileRight", 0.85),
                blendMsg("BrowInnerUp", 0.65),
                blendMsg("cheekSquint_L", 0.45),
                blendMsg("A", 0.90),
                .message(OSCMessage("/VMC/Ext/Blend/Apply")),
            ]))

            let datagram = bundle.encode()
            try sendLoopbackUDP(datagram, toPort: boundPort)

            let deadline = Date().addingTimeInterval(1.0)
            while controller.vmcController.packetCount == 0 || controller.vmcController.receivedBlendShapeCount < 6 {
                if Date() > deadline { break }
                try await Task.sleep(nanoseconds: 10_000_000)
            }

            guard controller.vmcController.packetCount > 0,
                  let model = controller.model else {
                throw NSError(domain: "AlicePet.VMC", code: 11, userInfo: [NSLocalizedDescriptionKey: "VMC UDP loopback timed out without receiving packets"])
            }

            controller.poseController.applyBasePose(to: model, restTransforms: controller.poseController.restNodeTransforms)
            controller.poseController.applyHandGestureOverrides(to: model, recipeStore: controller.recipeStore)
            let applied = controller.vmcController.apply(
                to: model,
                renderer: controller.renderer,
                expressionMixer: controller.expressionMixer
            )
            controller.settleSpringBonePhysics(warmupSteps: 16)

            print("VMC loopback OK: port=\(boundPort) bytes=\(datagram.count) packets=\(controller.vmcController.packetCount) bones=\(controller.vmcController.receivedBoneCount) blendShapes=\(controller.vmcController.receivedBlendShapeCount) applied=\(applied) perfectSync=\(controller.faceDriver.perfectSyncCapability)")

            try await controller.renderSnapshot(to: URL(fileURLWithPath: loopbackOutputPath), applyBaseState: false)
            print("VMC loopback snapshot saved to \(loopbackOutputPath)")
            controller.stopVMC()
        } catch {
            fputs("VMC loopback error: \(error)\n", stderr)
            exitCode = 1
        }
        done = true
    }
    while !done {
        RunLoop.main.run(mode: .default, before: Date(timeIntervalSinceNow: 0.01))
    }
    exit(exitCode)
}

if let motionOutputPath = motionServiceTestPath {
    var done = false
    var exitCode: Int32 = 0
    Task { @MainActor in
        do {
            let controller = try PetController(sampleCount: 1)
            try await controller.loadModel(from: modelURL, synchronousPhysics: true)
            controller.camera.apply(preset: .upperBody)

            guard controller.startVMC(port: 0), let boundPort = controller.vmcController.boundPort else {
                throw NSError(domain: "AlicePet.Motion", code: 20, userInfo: [NSLocalizedDescriptionKey: "Failed to bind VMC port"])
            }
            guard let serviceDir = LocalMotionService.resolveServiceDirectory() else {
                throw NSError(domain: "AlicePet.Motion", code: 21, userInfo: [NSLocalizedDescriptionKey: "services/motion directory not found"])
            }
            let proc = Process()
            proc.executableURL = serviceDir.appendingPathComponent(".venv/bin/python")
            proc.currentDirectoryURL = serviceDir
            proc.arguments = ["tracker.py", "--self-test", "--host", "127.0.0.1", "--port", String(boundPort)]
            var env = ProcessInfo.processInfo.environment
            env["GLOG_minloglevel"] = "2"
            env["TF_CPP_MIN_LOG_LEVEL"] = "2"
            proc.environment = env
            try proc.run()

            let deadline = Date().addingTimeInterval(15.0)
            while (controller.vmcController.packetCount == 0 || proc.isRunning) && Date() < deadline {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            proc.waitUntilExit()
            guard proc.terminationStatus == 0, controller.vmcController.packetCount > 0, let model = controller.model else {
                throw NSError(domain: "AlicePet.Motion", code: 22, userInfo: [NSLocalizedDescriptionKey: "Motion service self-test failed (status=\(proc.terminationStatus), packets=\(controller.vmcController.packetCount))"])
            }

            controller.poseController.applyBasePose(to: model, restTransforms: controller.poseController.restNodeTransforms)
            let applied = controller.vmcController.apply(
                to: model,
                renderer: controller.renderer,
                expressionMixer: controller.expressionMixer
            )
            controller.settleSpringBonePhysics(warmupSteps: 16)
            print("MotionService E2E OK: port=\(boundPort) packets=\(controller.vmcController.packetCount) bones=\(controller.vmcController.receivedBoneCount) blendShapes=\(controller.vmcController.receivedBlendShapeCount) applied=\(applied)")

            try await controller.renderSnapshot(to: URL(fileURLWithPath: motionOutputPath), applyBaseState: false)
            print("MotionService E2E snapshot saved to \(motionOutputPath)")
            controller.stopAllTrackingSync()
        } catch {
            fputs("MotionService E2E error: \(error)\n", stderr)
            exitCode = 1
        }
        done = true
    }
    while !done {
        RunLoop.main.run(mode: .default, before: Date(timeIntervalSinceNow: 0.01))
    }
    exit(exitCode)
}

if let snapshotPath = parseArg("--snapshot") {
    var done = false
    var exitCode: Int32 = 0
    Task { @MainActor in
        do {
            let controller = try PetController(sampleCount: 1)
            try await controller.loadModel(from: modelURL, synchronousPhysics: true)

            if let viewRaw = parseArg("--view"), let preset = CameraViewPreset(rawValue: viewRaw) {
                controller.camera.apply(preset: preset)
            }
            if let yawStr = parseArg("--yaw"), let yawDeg = Float(yawStr) {
                controller.camera.yaw = yawDeg * .pi / 180.0
            }
            if let pitchStr = parseArg("--pitch"), let pitchDeg = Float(pitchStr) {
                controller.camera.pitch = pitchDeg * .pi / 180.0
            }
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
