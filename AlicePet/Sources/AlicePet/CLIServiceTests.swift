import Darwin
import Foundation
import VRMMetalKit
import simd

/// Headless CLI end-to-end self-test runners for VMC loopback, MediaPipe, Audio2Face, A2F-3D, ARDY, and Mixer.

private func sendLoopbackUDP(_ data: Data, toPort port: UInt16) throws {
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

@MainActor
func runVMCLoopbackTest(modelURL: URL, outputPath: String) -> Never {
    runMainActorCLI("VMC loopback") {
        let controller = try PetController(sampleCount: 1)
        try await controller.loadModel(from: modelURL, synchronousPhysics: true)
        if let viewRaw = parseCLIArg("--view"), let preset = CameraViewPreset(rawValue: viewRaw) {
            controller.camera.apply(preset: preset)
        } else {
            controller.camera.apply(preset: .upperBody)
        }

        guard controller.startVMC(port: 0), let boundPort = controller.vmcController.boundPort else {
            throw NSError(domain: "AlicePet.VMC", code: 10, userInfo: [NSLocalizedDescriptionKey: "Failed to bind VMC UDP loopback port"])
        }

        func boneMsg(_ name: String, q: simd_quatf) -> OSCPacket {
            .message(OSCMessage("/VMC/Ext/Bone/Pos", .string(name), .float32(0), .float32(0), .float32(0),
                                .float32(q.imag.x), .float32(q.imag.y), .float32(q.imag.z), .float32(q.real)))
        }
        func blendMsg(_ name: String, _ weight: Float) -> OSCPacket {
            .message(OSCMessage("/VMC/Ext/Blend/Val", .string(name), .float32(weight)))
        }

        let bundle = OSCPacket.bundle(OSCBundle(elements: [
            .message(OSCMessage("/VMC/Ext/OK", .int32(1))),
            boneMsg("Head", q: simd_quatf(angle: 0.22, axis: SIMD3<Float>(0, 0, 1)) * simd_quatf(angle: -0.18, axis: SIMD3<Float>(0, 1, 0))),
            boneMsg("Neck", q: simd_quatf(angle: 0.08, axis: SIMD3<Float>(0, 0, 1))),
            boneMsg("LeftUpperArm", q: simd_quatf(angle: -0.55, axis: SIMD3<Float>(0, 0, 1)) * simd_quatf(angle: 0.35, axis: SIMD3<Float>(1, 0, 0))),
            boneMsg("LeftLowerArm", q: simd_quatf(angle: -1.15, axis: SIMD3<Float>(0, 0, 1))),
            boneMsg("RightUpperArm", q: simd_quatf(angle: -1.15, axis: SIMD3<Float>(0, 0, 1))),
            blendMsg("EyeBlinkLeft", 1.0), blendMsg("jawOpen", 0.72),
            blendMsg("MouthSmileLeft", 0.85), blendMsg("mouthSmileRight", 0.85),
            blendMsg("BrowInnerUp", 0.65), blendMsg("cheekSquint_L", 0.45), blendMsg("A", 0.90),
            .message(OSCMessage("/VMC/Ext/Blend/Apply")),
        ]))

        let datagram = bundle.encode()
        try sendLoopbackUDP(datagram, toPort: boundPort)

        let deadline = Date().addingTimeInterval(1.0)
        while (controller.vmcController.packetCount == 0 || controller.vmcController.receivedBlendShapeCount < 6) && Date() < deadline {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        guard controller.vmcController.packetCount > 0, let model = controller.model else {
            throw NSError(domain: "AlicePet.VMC", code: 11, userInfo: [NSLocalizedDescriptionKey: "VMC UDP loopback timed out"])
        }

        controller.poseController.applyBasePose(to: model, restTransforms: controller.poseController.restNodeTransforms)
        controller.poseController.applyHandGestureOverrides(to: model, recipeStore: controller.recipeStore)
        let applied = controller.vmcController.apply(to: model, renderer: controller.renderer, expressionMixer: controller.expressionMixer)
        controller.settleSpringBonePhysics(warmupSteps: 16)
        print("VMC loopback OK: port=\(boundPort) bytes=\(datagram.count) packets=\(controller.vmcController.packetCount) applied=\(applied)")

        try await controller.renderSnapshot(to: URL(fileURLWithPath: outputPath), applyBaseState: false)
        controller.stopVMC()
    }
}

@MainActor
func runMotionServiceTest(modelURL: URL, outputPath: String) -> Never {
    runPythonSelfTest(
        label: "MotionService E2E", modelURL: modelURL, outputPath: outputPath,
        cameraPreset: .upperBody, serviceDirProvider: { LocalMotionService.resolveServiceDirectory() },
        script: "tracker.py", extraArgs: [], timeoutSec: 15.0,
        envOverrides: ["GLOG_minloglevel": "2", "TF_CPP_MIN_LOG_LEVEL": "2"]
    )
}

@MainActor
func runAudioFaceTest(modelURL: URL, outputPath: String) -> Never {
    let basePath = ProcessInfo.processInfo.environment["PATH"] ?? ""
    runPythonSelfTest(
        label: "Audio2Face E2E", modelURL: modelURL, outputPath: outputPath,
        cameraPreset: .faceCloseUp, serviceDirProvider: { LocalAudioFaceService.resolveServiceDirectory() },
        script: "audio2face_vmc.py", extraArgs: [], timeoutSec: 15.0,
        envOverrides: ["PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + basePath]
    )
}

@MainActor
func runArdyTest(modelURL: URL, outputPath: String) -> Never {
    let prompt = parseCLIArg("--prompt") ?? "a person stands and waves cheerfully with their right hand"
    runPythonSelfTest(
        label: "ARDY E2E", modelURL: modelURL, outputPath: outputPath,
        cameraPreset: .fullBody, serviceDirProvider: { LocalArdyService.resolveServiceDirectory() },
        script: "ardy_vmc.py", extraArgs: ["--prompt", prompt], timeoutSec: 60.0,
        envOverrides: [:]
    )
}

@MainActor
private func runPythonSelfTest(
    label: String,
    modelURL: URL,
    outputPath: String,
    cameraPreset: CameraViewPreset,
    serviceDirProvider: @escaping @MainActor () -> URL?,
    script: String,
    extraArgs: [String],
    timeoutSec: TimeInterval,
    envOverrides: [String: String]
) -> Never {
    runMainActorCLI(label) {
        let controller = try PetController(sampleCount: 1)
        try await controller.loadModel(from: modelURL, synchronousPhysics: true)
        controller.camera.apply(preset: cameraPreset)

        guard controller.startVMC(port: 0), let boundPort = controller.vmcController.boundPort else {
            throw NSError(domain: "AlicePet.CLI", code: 20, userInfo: [NSLocalizedDescriptionKey: "Failed to bind VMC port"])
        }
        guard let serviceDir = serviceDirProvider() else {
            throw NSError(domain: "AlicePet.CLI", code: 21, userInfo: [NSLocalizedDescriptionKey: "Service directory not found for \(script)"])
        }

        let proc = Process()
        proc.executableURL = serviceDir.appendingPathComponent(".venv/bin/python")
        proc.currentDirectoryURL = serviceDir
        proc.arguments = [script, "--self-test"] + extraArgs + ["--host", "127.0.0.1", "--port", String(boundPort)]
        if !envOverrides.isEmpty {
            var env = ProcessInfo.processInfo.environment
            for (k, v) in envOverrides { env[k] = v }
            proc.environment = env
        }
        proc.standardInput = FileHandle.nullDevice
        try proc.run()

        let deadline = Date().addingTimeInterval(timeoutSec)
        while (controller.vmcController.packetCount == 0 || proc.isRunning) && Date() < deadline {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        proc.waitUntilExit()
        guard proc.terminationStatus == 0, controller.vmcController.packetCount > 0, let model = controller.model else {
            throw NSError(domain: "AlicePet.CLI", code: 22, userInfo: [NSLocalizedDescriptionKey: "\(label) failed (status=\(proc.terminationStatus), packets=\(controller.vmcController.packetCount))"])
        }

        controller.poseController.applyBasePose(to: model, restTransforms: controller.poseController.restNodeTransforms)
        let applied = controller.vmcController.apply(to: model, renderer: controller.renderer, expressionMixer: controller.expressionMixer)
        controller.poseController.applyHandGestureOverrides(to: model, recipeStore: controller.recipeStore, applyRelaxedNatural: true)
        model.updateNodeTransforms()
        controller.settleSpringBonePhysics(warmupSteps: 24)
        print("\(label) OK: port=\(boundPort) packets=\(controller.vmcController.packetCount) bones=\(controller.vmcController.receivedBoneCount) shapes=\(controller.vmcController.receivedBlendShapeCount) applied=\(applied)")

        try await controller.renderSnapshot(to: URL(fileURLWithPath: outputPath), applyBaseState: false)
        controller.stopAllTrackingSync()
    }
}

@MainActor
func runA2F3DTest(modelURL: URL, outputPath: String) -> Never {
    runMainActorCLI("A2F-3D E2E") {
        let controller = try PetController(sampleCount: 1)
        try await controller.loadModel(from: modelURL, synchronousPhysics: true)
        controller.camera.apply(preset: .faceCloseUp)

        guard controller.animationMixer.audioFaceChannel.start(port: 0),
              let boundPort = controller.animationMixer.audioFaceChannel.boundPort else {
            throw NSError(domain: "AlicePet.A2F3D", code: 35, userInfo: [NSLocalizedDescriptionKey: "Failed to bind audioFace port"])
        }
        controller.animationMixer.audioFaceChannel.driver.maxAge = 20.0
        guard let serviceDir = LocalAudioFaceService.resolveServiceDirectory() else {
            throw NSError(domain: "AlicePet.A2F3D", code: 36, userInfo: [NSLocalizedDescriptionKey: "services/tts directory not found"])
        }

        let emotion = parseCLIArg("--emotion") ?? "joy"
        let strength = parseCLIArg("--emotion-strength") ?? "0.85"
        let proc = Process()
        proc.executableURL = serviceDir.appendingPathComponent(".venv/bin/python")
        proc.currentDirectoryURL = serviceDir
        proc.arguments = ["audio2face3d_vmc.py", "--self-test", "--emotion", emotion, "--emotion-strength", strength, "--host", "127.0.0.1", "--port", String(boundPort)]
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
        guard proc.terminationStatus == 0, controller.animationMixer.audioFaceChannel.packetCount > 0, let model = controller.model else {
            throw NSError(domain: "AlicePet.A2F3D", code: 37, userInfo: [NSLocalizedDescriptionKey: "A2F-3D self-test failed"])
        }

        let forcedBlink = Float(parseCLIArg("--blink") ?? "0") ?? 0
        if forcedBlink > 0 {
            controller.renderer.expressionController?.setExpressionWeight(.blink, weight: forcedBlink)
        }
        controller.animationMixer.evaluate(
            model: model, renderer: controller.renderer, poseController: controller.poseController,
            recipeStore: controller.recipeStore, expressionMixer: controller.expressionMixer,
            elapsed: 0.5, dt: 0.5, idleBreathingEnabled: true, poseMode: controller.poseMode, window: nil
        )
        controller.settleSpringBonePhysics(warmupSteps: 16)
        print("A2F-3D E2E OK: emo=\(emotion) blink=\(forcedBlink) shapes=\(controller.animationMixer.audioFaceChannel.receivedBlendShapeCount)")

        try await controller.renderSnapshot(to: URL(fileURLWithPath: outputPath), applyBaseState: false)
        controller.stopAllTrackingSync()
    }
}

@MainActor
func runMixerTest(modelURL: URL, outputPath: String) -> Never {
    runMainActorCLI("Mixer E2E") {
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

        let prompt = parseCLIArg("--prompt") ?? "a person claps hands enthusiastically"
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

        guard ardyProc.terminationStatus == 0, faceProc.terminationStatus == 0,
              controller.animationMixer.ardyChannel.packetCount > 0,
              controller.animationMixer.audioFaceChannel.packetCount > 0,
              let model = controller.model else {
            throw NSError(domain: "AlicePet.Mixer", code: 62, userInfo: [NSLocalizedDescriptionKey: "Mixer concurrent self-test failed"])
        }

        controller.animationMixer.preferredFullBodySource = .ardy
        controller.animationMixer.evaluate(
            model: model, renderer: controller.renderer, poseController: controller.poseController,
            recipeStore: controller.recipeStore, expressionMixer: controller.expressionMixer,
            elapsed: 0.5, dt: 0.5, idleBreathingEnabled: true, poseMode: controller.poseMode, window: nil
        )
        controller.settleSpringBonePhysics(warmupSteps: 24)
        print("Mixer E2E OK: activeBody=\(controller.animationMixer.activeBodySource.rawValue) activeFace=\(controller.animationMixer.activeFaceSource.rawValue)")

        try await controller.renderSnapshot(to: URL(fileURLWithPath: outputPath), applyBaseState: false)
        controller.stopAllTrackingSync()
    }
}
