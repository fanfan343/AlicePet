import Darwin
import Foundation

/// Manages the experimental `services/ardy/ardy_vmc.py` subprocess
/// (ARDY Mini distilled MiniLM + Core40 ONNX -> 60 FPS SLERP VMC/OSC motion stream).
@MainActor
final class LocalArdyService {
    var currentPrompt: String = "a petite gentle young girl stands gracefully and waves softly with her right hand"
    var cfgWeight: Double = 2.2

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
            Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("services/ardy", isDirectory: true),
            URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("../../../services/ardy", isDirectory: true).standardizedFileURL,
            URL(fileURLWithPath: "/Users/funnyliu/Downloads/VRM项目/services/ardy", isDirectory: true),
        ]
        for dir in candidates {
            if fm.fileExists(atPath: dir.appendingPathComponent("ardy_vmc.py").path) {
                return dir
            }
        }
        return nil
    }

    @discardableResult
    func start(port: UInt16 = VMCTrackingController.defaultPort) -> Bool {
        stop()

        guard let serviceDir = Self.resolveServiceDirectory() else {
            let msg = "未找到 ARDY Mini 服务目录 services/ardy/ardy_vmc.py。"
            AppLogger.shared.error("ARDY", msg)
            onError?(msg)
            return false
        }

        let pythonURL = serviceDir.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: pythonURL.path) else {
            let msg = "ARDY Mini 虚拟环境未初始化，请在终端执行：cd services/ardy && uv sync --python 3.12"
            AppLogger.shared.error("ARDY", msg)
            onError?(msg)
            return false
        }

        let denoiserURL = serviceDir.appendingPathComponent("models/denoiser.onnx")
        guard FileManager.default.fileExists(atPath: denoiserURL.path) else {
            let msg = "缺少 ARDY Mini ONNX 模型：\(denoiserURL.path)"
            AppLogger.shared.error("ARDY", msg)
            onError?(msg)
            return false
        }

        let proc = Process()
        proc.executableURL = pythonURL
        proc.currentDirectoryURL = serviceDir
        proc.arguments = [
            "ardy_vmc.py",
            "--models-dir", serviceDir.appendingPathComponent("models").path,
            "--prompt", currentPrompt,
            "--cfg", String(format: "%.2f", cfgWeight),
            "--host", "127.0.0.1",
            "--port", String(port),
            "--live-prompts",
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
                AppLogger.shared.info("ARDY", text)
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
                AppLogger.shared.error("ARDY", text)
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
                    self.onError?("ARDY Mini 动作生成进程异常退出（code=\(code)），详情见 alicepet.log。")
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
            self.lastStatusLine = "模型加载中，正在生成首个动作窗口…"
            AppLogger.shared.info("ARDY", "Started ARDY Mini streaming engine (pid=\(proc.processIdentifier), port=\(port), prompt='\(currentPrompt)')")
            onStateChanged?()
            return true
        } catch {
            cleanupHandles()
            let msg = "启动 ARDY Mini 失败：\(error.localizedDescription)"
            AppLogger.shared.error("ARDY", msg)
            onError?(msg)
            return false
        }
    }

    func sendLivePrompt(_ prompt: String, cfg: Double? = nil) {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            currentPrompt = trimmed
        }
        if let cfg {
            cfgWeight = cfg
        }
        guard isRunning, let handle = stdinPipe?.fileHandleForWriting else { return }
        let payload: [String: Any] = ["prompt": currentPrompt, "cfg": cfgWeight]
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
            AppLogger.shared.info("ARDY", "Stopped ARDY Mini streaming engine (pid=\(pid))")
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
