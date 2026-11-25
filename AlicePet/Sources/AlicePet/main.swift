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
let audioFaceTestPath = parseArg("--test-audio2face")
let a2f3dTestPath = parseArg("--test-a2f3d")
let ardyTestPath = parseArg("--test-ardy")
let mixerTestPath = parseArg("--test-mixer")
let isSnapshotMode = (parseArg("--snapshot") != nil) || (vmcLoopbackPath != nil) || (motionServiceTestPath != nil) || (audioFaceTestPath != nil) || (a2f3dTestPath != nil) || (ardyTestPath != nil) || (mixerTestPath != nil)
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
            proc.standardInput = FileHandle.nullDevice
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

if let audioOutputPath = audioFaceTestPath {
    var done = false
    var exitCode: Int32 = 0
    Task { @MainActor in
        do {
            let controller = try PetController(sampleCount: 1)
            try await controller.loadModel(from: modelURL, synchronousPhysics: true)
            controller.camera.apply(preset: .faceCloseUp)

            guard controller.startVMC(port: 0), let boundPort = controller.vmcController.boundPort else {
                throw NSError(domain: "AlicePet.Audio2Face", code: 30, userInfo: [NSLocalizedDescriptionKey: "Failed to bind VMC port"])
            }
            guard let serviceDir = LocalAudioFaceService.resolveServiceDirectory() else {
                throw NSError(domain: "AlicePet.Audio2Face", code: 31, userInfo: [NSLocalizedDescriptionKey: "services/tts directory not found"])
            }
            let proc = Process()
            proc.executableURL = serviceDir.appendingPathComponent(".venv/bin/python")
            proc.currentDirectoryURL = serviceDir
            proc.arguments = ["audio2face_vmc.py", "--self-test", "--host", "127.0.0.1", "--port", String(boundPort)]
            var env = ProcessInfo.processInfo.environment
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + (env["PATH"] ?? "")
            proc.environment = env
            proc.standardInput = FileHandle.nullDevice
            try proc.run()

            let deadline = Date().addingTimeInterval(15.0)
            while (controller.vmcController.packetCount == 0 || proc.isRunning) && Date() < deadline {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            proc.waitUntilExit()
            guard proc.terminationStatus == 0, controller.vmcController.packetCount > 0, let model = controller.model else {
                throw NSError(domain: "AlicePet.Audio2Face", code: 32, userInfo: [NSLocalizedDescriptionKey: "Audio2Face self-test failed (status=\(proc.terminationStatus), packets=\(controller.vmcController.packetCount))"])
            }

            controller.poseController.applyBasePose(to: model, restTransforms: controller.poseController.restNodeTransforms)
            let applied = controller.vmcController.apply(
                to: model,
                renderer: controller.renderer,
                expressionMixer: controller.expressionMixer
            )
            controller.settleSpringBonePhysics(warmupSteps: 16)
            print("Audio2Face E2E OK: port=\(boundPort) packets=\(controller.vmcController.packetCount) bones=\(controller.vmcController.receivedBoneCount) blendShapes=\(controller.vmcController.receivedBlendShapeCount) applied=\(applied)")

            try await controller.renderSnapshot(to: URL(fileURLWithPath: audioOutputPath), applyBaseState: false)
            print("Audio2Face E2E snapshot saved to \(audioOutputPath)")
            controller.stopAllTrackingSync()
        } catch {
            fputs("Audio2Face E2E error: \(error)\n", stderr)
            exitCode = 1
        }
        done = true
    }
    while !done {
        RunLoop.main.run(mode: .default, before: Date(timeIntervalSinceNow: 0.01))
    }
    exit(exitCode)
}

if let a2fOutputPath = a2f3dTestPath {
    var done = false
    var exitCode: Int32 = 0
    Task { @MainActor in
        do {
            let controller = try PetController(sampleCount: 1)
            try await controller.loadModel(from: modelURL, synchronousPhysics: true)
            controller.camera.apply(preset: .faceCloseUp)

            guard controller.animationMixer.audioFaceChannel.start(port: 0),
                  let boundPort = controller.animationMixer.audioFaceChannel.boundPort else {
                throw NSError(domain: "AlicePet.A2F3D", code: 35, userInfo: [NSLocalizedDescriptionKey: "Failed to bind isolated audioFace port"])
            }
            controller.animationMixer.audioFaceChannel.driver.maxAge = 20.0
            guard let serviceDir = LocalAudioFaceService.resolveServiceDirectory() else {
                throw NSError(domain: "AlicePet.A2F3D", code: 36, userInfo: [NSLocalizedDescriptionKey: "services/tts directory not found"])
            }
            let emotion = parseArg("--emotion") ?? "joy"
            let strength = parseArg("--emotion-strength") ?? "0.85"
            let proc = Process()
            proc.executableURL = serviceDir.appendingPathComponent(".venv/bin/python")
            proc.currentDirectoryURL = serviceDir
            proc.arguments = [
                "audio2face3d_vmc.py", "--self-test",
                "--emotion", emotion, "--emotion-strength", strength,
                "--host", "127.0.0.1", "--port", String(boundPort),
            ]
            var env = ProcessInfo.processInfo.environment
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + (env["PATH"] ?? "")
            proc.environment = env
            proc.standardInput = FileHandle.nullDevice
            try proc.run()

            let deadline = Date().addingTimeInterval(30.0)
            while (controller.animationMixer.audioFaceChannel.packetCount == 0 || proc.isRunning) && Date() < deadline {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            proc.waitUntilExit()
            guard proc.terminationStatus == 0,
                  controller.animationMixer.audioFaceChannel.packetCount > 0,
                  let model = controller.model else {
                throw NSError(domain: "AlicePet.A2F3D", code: 37, userInfo: [NSLocalizedDescriptionKey: "A2F-3D self-test failed (status=\(proc.terminationStatus), packets=\(controller.animationMixer.audioFaceChannel.packetCount))"])
            }

            controller.animationMixer.evaluate(
                model: model,
                renderer: controller.renderer,
                poseController: controller.poseController,
                recipeStore: controller.recipeStore,
                expressionMixer: controller.expressionMixer,
                elapsed: 0.5,
                dt: 0.5,
                idleBreathingEnabled: true,
                poseMode: controller.poseMode,
                window: nil
            )
            controller.settleSpringBonePhysics(warmupSteps: 16)
            print("A2F-3D E2E OK: emo=\(emotion) lookAtEnabled=\(controller.renderer.lookAtController?.enabled ?? false) shapes=\(controller.animationMixer.audioFaceChannel.receivedBlendShapeCount)")

            try await controller.renderSnapshot(to: URL(fileURLWithPath: a2fOutputPath), applyBaseState: false)
            print("A2F-3D E2E snapshot saved to \(a2fOutputPath)")
            controller.stopAllTrackingSync()
        } catch {
            fputs("A2F-3D E2E error: \(error)\n", stderr)
            exitCode = 1
        }
        done = true
    }
    while !done {
        RunLoop.main.run(mode: .default, before: Date(timeIntervalSinceNow: 0.01))
    }
    exit(exitCode)
}

if let ardyOutputPath = ardyTestPath {
    var done = false
    var exitCode: Int32 = 0
    Task { @MainActor in
        do {
            let controller = try PetController(sampleCount: 1)
            try await controller.loadModel(from: modelURL, synchronousPhysics: true)
            controller.camera.apply(preset: .fullBody)

            guard controller.startVMC(port: 0), let boundPort = controller.vmcController.boundPort else {
                throw NSError(domain: "AlicePet.ARDY", code: 40, userInfo: [NSLocalizedDescriptionKey: "Failed to bind VMC port"])
            }
            guard let serviceDir = LocalArdyService.resolveServiceDirectory() else {
                throw NSError(domain: "AlicePet.ARDY", code: 41, userInfo: [NSLocalizedDescriptionKey: "services/ardy directory not found"])
            }
            let prompt = parseArg("--prompt") ?? "a person stands and waves cheerfully with their right hand"
            let proc = Process()
            proc.executableURL = serviceDir.appendingPathComponent(".venv/bin/python")
            proc.currentDirectoryURL = serviceDir
            proc.arguments = ["ardy_vmc.py", "--self-test", "--prompt", prompt, "--host", "127.0.0.1", "--port", String(boundPort)]
            proc.standardInput = FileHandle.nullDevice
            try proc.run()

            let deadline = Date().addingTimeInterval(60.0)
            while (controller.vmcController.packetCount == 0 || proc.isRunning) && Date() < deadline {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            proc.waitUntilExit()
            guard proc.terminationStatus == 0, controller.vmcController.packetCount > 0, let model = controller.model else {
                throw NSError(domain: "AlicePet.ARDY", code: 42, userInfo: [NSLocalizedDescriptionKey: "ARDY self-test failed (status=\(proc.terminationStatus), packets=\(controller.vmcController.packetCount))"])
            }

            controller.poseController.applyBasePose(to: model, restTransforms: controller.poseController.restNodeTransforms)
            let applied = controller.vmcController.apply(
                to: model,
                renderer: controller.renderer,
                expressionMixer: controller.expressionMixer
            )
            controller.settleSpringBonePhysics(warmupSteps: 16)
            let ikSummary = controller.vmcController.lastArmIKDiagnostics
                .map { "\($0.side):shift=\(String(format: "%.1f", $0.targetShiftCm))cm,clip=\($0.clippedAdjusted)" }
                .joined(separator: ",")
            print("ARDY E2E OK: port=\(boundPort) packets=\(controller.vmcController.packetCount) bones=\(controller.vmcController.receivedBoneCount) applied=\(applied) armIK=[\(ikSummary)]")

            try await controller.renderSnapshot(to: URL(fileURLWithPath: ardyOutputPath), applyBaseState: false)
            print("ARDY E2E snapshot saved to \(ardyOutputPath)")
            controller.stopAllTrackingSync()
        } catch {
            fputs("ARDY E2E error: \(error)\n", stderr)
            exitCode = 1
        }
        done = true
    }
    while !done {
        RunLoop.main.run(mode: .default, before: Date(timeIntervalSinceNow: 0.01))
    }
    exit(exitCode)
}

if let mixerOutputPath = mixerTestPath {
    var done = false
    var exitCode: Int32 = 0
    Task { @MainActor in
        do {
            let controller = try PetController(sampleCount: 1)
            try await controller.loadModel(from: modelURL, synchronousPhysics: true)
            controller.camera.apply(preset: .upperBody)

            guard controller.animationMixer.ardyChannel.start(port: 0),
                  let ardyPort = controller.animationMixer.ardyChannel.boundPort,
                  controller.animationMixer.audioFaceChannel.start(port: 0),
                  let facePort = controller.animationMixer.audioFaceChannel.boundPort else {
                throw NSError(domain: "AlicePet.Mixer", code: 60, userInfo: [NSLocalizedDescriptionKey: "Failed to bind isolated mixer ports"])
            }
            controller.animationMixer.ardyChannel.driver.maxAge = 15.0
            controller.animationMixer.audioFaceChannel.driver.maxAge = 15.0
            guard let ardyDir = LocalArdyService.resolveServiceDirectory(),
                  let ttsDir = LocalAudioFaceService.resolveServiceDirectory() else {
                throw NSError(domain: "AlicePet.Mixer", code: 61, userInfo: [NSLocalizedDescriptionKey: "Service directories not found"])
            }

            let prompt = parseArg("--prompt") ?? "a person claps hands enthusiastically"
            let ardyProc = Process()
            ardyProc.executableURL = ardyDir.appendingPathComponent(".venv/bin/python")
            ardyProc.currentDirectoryURL = ardyDir
            ardyProc.arguments = ["ardy_vmc.py", "--self-test", "--prompt", prompt, "--host", "127.0.0.1", "--port", String(ardyPort)]
            ardyProc.standardInput = FileHandle.nullDevice

            let faceProc = Process()
            faceProc.executableURL = ttsDir.appendingPathComponent(".venv/bin/python")
            faceProc.currentDirectoryURL = ttsDir
            faceProc.arguments = ["audio2face_vmc.py", "--self-test", "--host", "127.0.0.1", "--port", String(facePort)]
            var env = ProcessInfo.processInfo.environment
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + (env["PATH"] ?? "")
            faceProc.environment = env
            faceProc.standardInput = FileHandle.nullDevice

            try ardyProc.run()
            try faceProc.run()

            let deadline = Date().addingTimeInterval(60.0)
            while (ardyProc.isRunning || faceProc.isRunning) && Date() < deadline {
                try await Task.sleep(nanoseconds: 50_000_000)
            }
            ardyProc.waitUntilExit()
            faceProc.waitUntilExit()

            guard ardyProc.terminationStatus == 0,
                  faceProc.terminationStatus == 0,
                  controller.animationMixer.ardyChannel.packetCount > 0,
                  controller.animationMixer.audioFaceChannel.packetCount > 0,
                  let model = controller.model else {
                throw NSError(domain: "AlicePet.Mixer", code: 62, userInfo: [NSLocalizedDescriptionKey: "Mixer concurrent self-test failed (ardy=\(ardyProc.terminationStatus), face=\(faceProc.terminationStatus))"])
            }

            controller.animationMixer.preferredFullBodySource = .ardy
            controller.animationMixer.evaluate(
                model: model,
                renderer: controller.renderer,
                poseController: controller.poseController,
                recipeStore: controller.recipeStore,
                expressionMixer: controller.expressionMixer,
                elapsed: 0.5,
                dt: 0.5,
                idleBreathingEnabled: true,
                poseMode: controller.poseMode,
                window: nil
            )
            controller.settleSpringBonePhysics(warmupSteps: 24)
            let ikSummary = controller.animationMixer.ardyChannel.lastArmIKDiagnostics
                .map { "\($0.side):shift=\(String(format: "%.1f", $0.targetShiftCm))cm,clip=\($0.clippedAdjusted)" }
                .joined(separator: ",")
            print("Mixer E2E OK: bodyPort=\(ardyPort)(bones=\(controller.animationMixer.ardyChannel.receivedBoneCount)) facePort=\(facePort)(bones=\(controller.animationMixer.audioFaceChannel.receivedBoneCount), shapes=\(controller.animationMixer.audioFaceChannel.receivedBlendShapeCount)) activeBody=\(controller.animationMixer.activeBodySource.rawValue) activeFace=\(controller.animationMixer.activeFaceSource.rawValue) armIK=[\(ikSummary)]")

            try await controller.renderSnapshot(to: URL(fileURLWithPath: mixerOutputPath), applyBaseState: false)
            print("Mixer E2E snapshot saved to \(mixerOutputPath)")
            controller.stopAllTrackingSync()
        } catch {
            fputs("Mixer E2E error: \(error)\n", stderr)
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

            if let sceneDir = parseArg("--scene") {
                try controller.loadPictureWorld(from: URL(fileURLWithPath: sceneDir))
                if let posRaw = parseArg("--avatar-pos") {
                    let parts = posRaw.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
                    if parts.count == 2 {
                        controller.pictureWorldRenderer.avatarWorldPosition = SIMD3<Float>(parts[0], 0, parts[1])
                    } else if parts.count == 3 {
                        controller.pictureWorldRenderer.avatarWorldPosition = SIMD3<Float>(parts[0], parts[1], parts[2])
                    }
                }
                if let yawDegStr = parseArg("--avatar-yaw"), let yawDeg = Float(yawDegStr) {
                    controller.pictureWorldRenderer.avatarWorldYaw = yawDeg * .pi / 180.0
                }
            }

            var snapW = 720
            var snapH = 1024
            if let sizeRaw = parseArg("--size") {
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

            try await controller.renderSnapshot(to: URL(fileURLWithPath: snapshotPath), width: snapW, height: snapH)
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
