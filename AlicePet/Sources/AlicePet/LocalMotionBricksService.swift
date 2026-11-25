import Darwin
import Foundation

/// Presets exposed by the local NVIDIA MotionBricks SOMA (ONNX Runtime) locomotion & style engine.
enum MotionBricksPreset: String, CaseIterable {
    case idle = "idle"
    case walk = "walk"
    case run = "run"
    case strafeLeft = "strafe_left"
    case strafeRight = "strafe_right"
    case crouch = "crouch"
    case skipping = "skipping"
    case vogue = "vogue"
    case combat = "combat"
    case zombie = "zombie"
    case injuredTorso = "injured_torso"
    case injuredLeg = "injured_leg"

    var displayTitle: String {
        switch self {
        case .idle: return "放松待机"
        case .walk: return "自然行走"
        case .run: return "轻快小跑"
        case .strafeLeft: return "向左横移"
        case .strafeRight: return "向右横移"
        case .crouch: return "蹲伏潜行"
        case .skipping: return "欢快蹦跳"
        case .vogue: return "时尚台步"
        case .combat: return "战斗警戒"
        case .zombie: return "僵尸漫步"
        case .injuredTorso: return "护胸缓行"
        case .injuredLeg: return "瘸腿缓行"
        }
    }
}

/// Manages the experimental `services/motionbricks/motionbricks_vmc.py` subprocess
/// (NVIDIA MotionBricks SOMA fp16 ONNX -> 60 FPS SLERP VMC/OSC full-body locomotion & style transitions).
@MainActor
final class LocalMotionBricksService {
    var currentPreset: MotionBricksPreset = .idle
    var faceCamera: Bool = true

    private(set) var isRunning: Bool = false
    private(set) var lastStatusLine: String = "未启动"

    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var launchGeneration: UInt64 = 0

    var onStateChanged: (() -> Void)?
    var onStatusLine: ((String) -> Void)?
    var onError: ((String) -> Void)?

    static func resolveServiceDirectory() -> URL? {
        let fm = FileManager.default
        let candidates: [URL] = [
            Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("services/motionbricks", isDirectory: true),
            URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("../../../services/motionbricks", isDirectory: true).standardizedFileURL,
            URL(fileURLWithPath: "/Users/funnyliu/Downloads/VRM项目/services/motionbricks", isDirectory: true),
        ]
        for dir in candidates {
            if fm.fileExists(atPath: dir.appendingPathComponent("motionbricks_vmc.py").path) {
                return dir
            }
        }
        return nil
    }

    @discardableResult
    func start(port: UInt16) -> Bool {
        stop()

        guard let serviceDir = Self.resolveServiceDirectory() else {
            let msg = "未找到 MotionBricks 服务目录 services/motionbricks/motionbricks_vmc.py。"
            AppLogger.shared.error("MotionBricks", msg)
            onError?(msg)
            return false
        }

        let pythonURL = serviceDir.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: pythonURL.path) else {
            let msg = "MotionBricks 虚拟环境未初始化，请在终端执行：cd services/motionbricks && python3 setup_assets.py && uv sync --python 3.12"
            AppLogger.shared.error("MotionBricks", msg)
            onError?(msg)
            return false
        }

        let onnxURL = serviceDir.appendingPathComponent("upstream/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx")
        guard FileManager.default.fileExists(atPath: onnxURL.path) else {
            let msg = "缺少 MotionBricks SOMA ONNX 模型：\(onnxURL.path)"
            AppLogger.shared.error("MotionBricks", msg)
            onError?(msg)
            return false
        }

        let proc = Process()
        proc.executableURL = pythonURL
        proc.currentDirectoryURL = serviceDir
        proc.arguments = [
            "motionbricks_vmc.py",
            "--preset", currentPreset.rawValue,
            "--host", "127.0.0.1",
            "--port", String(port),
            "--face-camera", faceCamera ? "1" : "0",
            "--live-commands",
        ]
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env

        let inPipe = Pipe()
        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardInput = inPipe
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty,
                  let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !text.isEmpty else { return }
            let lastLine = text.split(separator: "\n").last.map(String.init) ?? text
            Task { @MainActor [weak self] in
                AppLogger.shared.info("MotionBricks", text)
                self?.lastStatusLine = lastLine
                self?.onStatusLine?(lastLine)
            }
        }
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty,
                  let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !text.isEmpty else { return }
            Task { @MainActor in
                AppLogger.shared.error("MotionBricks", text)
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
                self.lastStatusLine = "已停止"
                self.onStateChanged?()
                if wasRunning && reason == .exit && code != 0 {
                    self.onError?("MotionBricks 动作生成进程异常退出（code=\(code)），详情见 alicepet.log。")
                }
            }
        }

        do {
            try proc.run()
            self.process = proc
            self.stdinPipe = inPipe
            self.stdoutPipe = outPipe
            self.stderrPipe = errPipe
            self.isRunning = true
            self.lastStatusLine = "正在加载 NVIDIA SOMA 模型与 Warp 内核…"
            AppLogger.shared.info("MotionBricks", "Started SOMA engine (pid=\(proc.processIdentifier), port=\(port), preset=\(currentPreset.rawValue))")
            onStateChanged?()
            return true
        } catch {
            cleanupHandles()
            let msg = "启动 MotionBricks 失败：\(error.localizedDescription)"
            AppLogger.shared.error("MotionBricks", msg)
            onError?(msg)
            return false
        }
    }

    func sendPreset(_ preset: MotionBricksPreset) {
        currentPreset = preset
        sendCommand(["preset": preset.rawValue, "face_camera": faceCamera])
    }

    func setFaceCamera(_ enabled: Bool) {
        faceCamera = enabled
        sendCommand(["preset": currentPreset.rawValue, "face_camera": enabled])
    }

    private func sendCommand(_ payload: [String: Any]) {
        guard isRunning, let handle = stdinPipe?.fileHandleForWriting else { return }
        guard var data = try? JSONSerialization.data(withJSONObject: payload) else { return }
        data.append(0x0A)
        try? handle.write(contentsOf: data)
    }

    func stop() {
        launchGeneration &+= 1
        let proc = process
        process = nil
        cleanupHandles()
        let wasRunning = isRunning
        isRunning = false
        lastStatusLine = "未启动"

        if let proc, proc.isRunning {
            let pid = proc.processIdentifier
            proc.terminate()
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 0.6) {
                if proc.isRunning {
                    Darwin.kill(pid, SIGKILL)
                }
            }
            AppLogger.shared.info("MotionBricks", "Stopped MotionBricks SOMA engine (pid=\(pid))")
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
    }

    private func cleanupHandles() {
        try? stdinPipe?.fileHandleForWriting.close()
        stdinPipe = nil
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        stdoutPipe = nil
        stderrPipe = nil
    }
}
