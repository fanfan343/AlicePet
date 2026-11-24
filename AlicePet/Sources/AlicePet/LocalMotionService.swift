import AVFoundation
import Darwin
import Foundation

struct LocalCameraChoice {
    let index: Int
    let name: String
    let isContinuity: Bool

    var displayTitle: String {
        isContinuity ? "\(name)（连续互通·仅后置）" : name
    }
}

enum MotionPoseModelTier: String, CaseIterable {
    case lite = "lite"
    case full = "full"

    var displayTitle: String {
        switch self {
        case .lite: return "姿态精度：轻量流畅 (Lite · 推荐)"
        case .full: return "姿态精度：高精度 (Full)"
        }
    }
}

/// Manages the out-of-process headless MediaPipe -> VMC/OSC tracker in `services/motion/`.
/// Stopping or quitting terminates the child process directly, guaranteeing 100% camera & memory release.
@MainActor
final class LocalMotionService {
    var cameraIndex: Int = 0
    var enableBody: Bool = true
    var enableHands: Bool = true
    var mirrorMode: Bool = true
    var poseModel: MotionPoseModelTier = .lite

    private(set) var isRunning: Bool = false
    private var process: Process?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var launchGeneration: UInt64 = 0

    var onStateChanged: (() -> Void)?
    var onError: ((String) -> Void)?

    static func availableCameras() -> [LocalCameraChoice] {
        var types: [AVCaptureDevice.DeviceType] = [.builtInWideAngleCamera, .external]
        if #available(macOS 14.0, *) {
            types.append(.continuityCamera)
        }
        let session = AVCaptureDevice.DiscoverySession(
            deviceTypes: types,
            mediaType: .video,
            position: .unspecified
        )
        return session.devices.enumerated().map { idx, dev in
            let isCont: Bool
            if #available(macOS 14.0, *) {
                isCont = (dev.deviceType == .continuityCamera) || dev.localizedName.localizedCaseInsensitiveContains("iPhone")
            } else {
                isCont = dev.localizedName.localizedCaseInsensitiveContains("iPhone")
            }
            return LocalCameraChoice(index: idx, name: dev.localizedName, isContinuity: isCont)
        }
    }

    static func resolveServiceDirectory() -> URL? {
        let fm = FileManager.default
        let candidates: [URL] = [
            // Relative to AlicePet.app inside repo root
            Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("services/motion", isDirectory: true),
            // Relative to SwiftPM debug/release binary
            URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("../../../services/motion", isDirectory: true).standardizedFileURL,
            // Repo default path
            URL(fileURLWithPath: "/Users/funnyliu/Downloads/VRM项目/services/motion", isDirectory: true),
        ]
        for dir in candidates {
            let tracker = dir.appendingPathComponent("tracker.py")
            if fm.fileExists(atPath: tracker.path) {
                return dir
            }
        }
        return nil
    }

    @discardableResult
    func start(port: UInt16) -> Bool {
        stop()

        guard let serviceDir = Self.resolveServiceDirectory() else {
            let msg = "未找到本地捕捉服务目录 services/motion/tracker.py。"
            AppLogger.shared.error("Motion", msg)
            onError?(msg)
            return false
        }

        let pythonURL = serviceDir.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: pythonURL.path) else {
            let msg = "本地捕捉虚拟环境未初始化，请在终端执行：cd services/motion && uv sync --python 3.12"
            AppLogger.shared.error("Motion", msg)
            onError?(msg)
            return false
        }

        let faceModel = serviceDir.appendingPathComponent("models/face_landmarker.task")
        guard FileManager.default.fileExists(atPath: faceModel.path) else {
            let msg = "缺少 MediaPipe 模型文件：\(faceModel.path)"
            AppLogger.shared.error("Motion", msg)
            onError?(msg)
            return false
        }

        let proc = Process()
        proc.executableURL = pythonURL
        proc.currentDirectoryURL = serviceDir
        proc.arguments = [
            "tracker.py",
            "--host", "127.0.0.1",
            "--port", String(port),
            "--camera", String(max(0, cameraIndex)),
            "--fps", "30",
            "--body", enableBody ? "1" : "0",
            "--hands", enableHands ? "1" : "0",
            "--mirror", mirrorMode ? "1" : "0",
            "--pose-model", poseModel.rawValue,
        ]
        var env = ProcessInfo.processInfo.environment
        env["GLOG_minloglevel"] = "2"
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardInput = FileHandle.nullDevice
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let line = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !line.isEmpty else { return }
            Task { @MainActor in
                AppLogger.shared.info("Motion", line)
            }
        }
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            let importantLines = text
                .split(separator: "\n")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { $0.contains("[MotionService]") || $0.contains("Traceback") || $0.contains("ERROR") }
            guard !importantLines.isEmpty else { return }
            let joined = importantLines.joined(separator: " | ")
            Task { @MainActor in
                AppLogger.shared.error("Motion", joined)
            }
        }

        launchGeneration &+= 1
        let gen = launchGeneration

        proc.terminationHandler = { [weak self] finished in
            let code = finished.terminationStatus
            let reason = finished.terminationReason
            Task { @MainActor [weak self] in
                guard let self, self.launchGeneration == gen else { return }
                self.cleanupHandles()
                let wasRunning = self.isRunning
                self.isRunning = false
                self.process = nil
                self.onStateChanged?()
                if wasRunning && reason == .exit && code != 0 {
                    let msg: String
                    if code == 3 {
                        msg = "无法打开所选摄像头（索引 \(self.cameraIndex)），请检查系统摄像头权限或切换摄像头。"
                    } else {
                        msg = "本机 MediaPipe 捕捉服务异常退出（code=\(code)），详情见 alicepet.log。"
                    }
                    self.onError?(msg)
                }
            }
        }

        do {
            try proc.run()
            self.process = proc
            self.stdoutPipe = outPipe
            self.stderrPipe = errPipe
            self.isRunning = true
            AppLogger.shared.info(
                "Motion",
                "Launched local MediaPipe tracker (pid=\(proc.processIdentifier), port=\(port), camera=\(cameraIndex), body=\(enableBody), hands=\(enableHands), mirror=\(mirrorMode), pose=\(poseModel.rawValue))"
            )
            onStateChanged?()
            return true
        } catch {
            cleanupHandles()
            let msg = "启动本机捕捉服务失败：\(error.localizedDescription)"
            AppLogger.shared.error("Motion", msg)
            onError?(msg)
            return false
        }
    }

    func stop() {
        launchGeneration &+= 1
        let proc = process
        process = nil
        cleanupHandles()
        let wasRunning = isRunning
        isRunning = false

        if let proc, proc.isRunning {
            let pid = proc.processIdentifier
            proc.terminate()
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 0.8) {
                if proc.isRunning {
                    Darwin.kill(pid, SIGKILL)
                }
            }
            AppLogger.shared.info("Motion", "Terminated local MediaPipe tracker (pid=\(pid))")
        }
        if wasRunning {
            onStateChanged?()
        }
    }

    func stopSync() {
        launchGeneration &+= 1
        let proc = process
        process = nil
        cleanupHandles()
        isRunning = false

        guard let proc, proc.isRunning else { return }
        let pid = proc.processIdentifier
        proc.terminate()
        let deadline = Date().addingTimeInterval(0.6)
        while proc.isRunning && Date() < deadline {
            usleep(20_000)
        }
        if proc.isRunning {
            Darwin.kill(pid, SIGKILL)
        }
        AppLogger.shared.info("Motion", "Synchronously stopped local MediaPipe tracker (pid=\(pid))")
    }

    private func cleanupHandles() {
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        stdoutPipe = nil
        stderrPipe = nil
    }
}
