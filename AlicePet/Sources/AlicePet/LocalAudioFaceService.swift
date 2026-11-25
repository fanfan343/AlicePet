import Darwin
import Foundation

/// Manages the experimental `services/tts/audio2face_vmc.py` subprocess
/// (LAM_Audio2Expression INT8 ONNX -> 52-channel ARKit VMC/OSC synchronized audio playback).
@MainActor
final class LocalAudioFaceService {
    var params: AudioLipSyncParams = AudioLipSyncParams.loadFromDefaults()
    private(set) var isRunning: Bool = false
    private(set) var activeAudioName: String?

    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var launchGeneration: UInt64 = 0

    var onStateChanged: (() -> Void)?
    var onFinishedNaturally: (() -> Void)?
    var onError: ((String) -> Void)?

    static func resolveServiceDirectory() -> URL? {
        let fm = FileManager.default
        let candidates: [URL] = [
            Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("services/tts", isDirectory: true),
            URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("../../../services/tts", isDirectory: true).standardizedFileURL,
            URL(fileURLWithPath: "/Users/funnyliu/Downloads/VRM项目/services/tts", isDirectory: true),
        ]
        for dir in candidates {
            if fm.fileExists(atPath: dir.appendingPathComponent("audio2face_vmc.py").path) {
                return dir
            }
        }
        return nil
    }

    static func defaultSampleAudioURL() -> URL? {
        guard let dir = resolveServiceDirectory() else { return nil }
        let sample = dir.appendingPathComponent("samples/vtuber_10s.wav")
        return FileManager.default.fileExists(atPath: sample.path) ? sample : nil
    }

    @discardableResult
    func start(audioURL: URL? = nil, port: UInt16 = VMCTrackingController.defaultPort, maxSeconds: Double = 0) -> Bool {
        stop()

        guard let serviceDir = Self.resolveServiceDirectory() else {
            let msg = "未找到音频表情测试服务目录 services/tts/audio2face_vmc.py。"
            AppLogger.shared.error("Audio2Face", msg)
            onError?(msg)
            return false
        }

        let pythonURL = serviceDir.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: pythonURL.path) else {
            let msg = "音频表情服务虚拟环境未初始化，请在终端执行：cd services/tts && uv sync --python 3.12"
            AppLogger.shared.error("Audio2Face", msg)
            onError?(msg)
            return false
        }

        let a2fScript = serviceDir.appendingPathComponent("audio2face3d_vmc.py")
        let a2fModel = serviceDir.appendingPathComponent("models/a2f_claire/network.onnx")
        let useA2F = (params.engine == .a2fClaire)
            && FileManager.default.fileExists(atPath: a2fScript.path)
            && FileManager.default.fileExists(atPath: a2fModel.path)

        let lamModelURL = serviceDir.appendingPathComponent("models/lam-int8.onnx")
        if !useA2F && !FileManager.default.fileExists(atPath: lamModelURL.path) {
            let msg = "缺少 LAM_Audio2Expression ONNX 模型：\(lamModelURL.path)"
            AppLogger.shared.error("Audio2Face", msg)
            onError?(msg)
            return false
        }

        guard let targetAudio = audioURL ?? Self.defaultSampleAudioURL() else {
            let msg = "未找到测试音频文件（默认 samples/vtuber_10s.wav）。"
            AppLogger.shared.error("Audio2Face", msg)
            onError?(msg)
            return false
        }

        let proc = Process()
        proc.executableURL = pythonURL
        proc.currentDirectoryURL = serviceDir
        var args: [String]
        if useA2F {
            args = [
                "audio2face3d_vmc.py",
                "--audio", targetAudio.path,
                "--models-dir", serviceDir.appendingPathComponent("models/a2f_claire").path,
                "--host", "127.0.0.1",
                "--port", String(port),
                "--emotion", params.emotion.rawValue,
                "--emotion-strength", String(format: "%.2f", params.emotionStrength),
                "--jaw-gain", String(format: "%.3f", params.jawGain),
                "--vowel-gain", String(format: "%.3f", params.vowelGain),
                "--lip-close", String(format: "%.3f", params.lipCloseStrength),
                "--lip-reveal", String(format: "%.3f", params.lipRevealGain),
                "--smoothing", String(format: "%.3f", params.smoothing),
                "--upper-face", String(format: "%.3f", params.upperFaceGain),
                "--head-motion", String(format: "%.3f", params.headMotionGain),
                "--sync-offset-ms", String(format: "%.1f", params.syncOffsetMs),
                "--loop", params.loopPlayback ? "1" : "0",
                "--live-params",
            ]
        } else {
            args = [
                "audio2face_vmc.py",
                "--audio", targetAudio.path,
                "--model", lamModelURL.path,
                "--host", "127.0.0.1",
                "--port", String(port),
                "--jaw-gain", String(format: "%.3f", params.jawGain),
                "--vowel-gain", String(format: "%.3f", params.vowelGain),
                "--lip-close", String(format: "%.3f", params.lipCloseStrength),
                "--lip-reveal", String(format: "%.3f", params.lipRevealGain),
                "--smoothing", String(format: "%.3f", params.smoothing),
                "--upper-face", String(format: "%.3f", params.upperFaceGain),
                "--head-motion", String(format: "%.3f", params.headMotionGain),
                "--sync-offset-ms", String(format: "%.1f", params.syncOffsetMs),
                "--natural-blinks", params.naturalBlinks ? "1" : "0",
                "--loop", params.loopPlayback ? "1" : "0",
                "--live-params",
            ]
        }
        if maxSeconds > 0 {
            args += ["--max-seconds", String(format: "%.1f", maxSeconds)]
        }
        proc.arguments = args

        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + (env["PATH"] ?? "")
        proc.environment = env

        let inPipe = Pipe()
        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardInput = inPipe
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty,
                  let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !text.isEmpty else { return }
            Task { @MainActor in
                AppLogger.shared.info("Audio2Face", text)
            }
        }
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty,
                  let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !text.isEmpty else { return }
            Task { @MainActor in
                AppLogger.shared.error("Audio2Face", text)
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
                self.activeAudioName = nil
                self.process = nil
                self.onStateChanged?()
                if wasRunning && reason == .exit {
                    if code == 0 {
                        self.onFinishedNaturally?()
                    } else {
                        self.onError?("音频对口型测试进程异常退出（code=\(code)），详情见 alicepet.log。")
                    }
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
            self.activeAudioName = targetAudio.lastPathComponent
            AppLogger.shared.info("Audio2Face", "Started LAM 52-channel audio test: \(targetAudio.lastPathComponent) (pid=\(proc.processIdentifier), port=\(port), \(params.prettyRecipeSummary()))")
            onStateChanged?()
            return true
        } catch {
            cleanupHandles()
            let msg = "启动音频对口型测试失败：\(error.localizedDescription)"
            AppLogger.shared.error("Audio2Face", msg)
            onError?(msg)
            return false
        }
    }

    func pushLiveParams(_ newParams: AudioLipSyncParams) {
        self.params = newParams
        guard isRunning, let handle = stdinPipe?.fileHandleForWriting,
              let line = newParams.toLiveJSONLine() else { return }
        try? handle.write(contentsOf: line)
    }

    func stop() {
        launchGeneration &+= 1
        let proc = process
        process = nil
        cleanupHandles()
        let wasRunning = isRunning
        isRunning = false
        activeAudioName = nil

        if let proc, proc.isRunning {
            let pid = proc.processIdentifier
            proc.terminate()
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 0.6) {
                if proc.isRunning {
                    Darwin.kill(pid, SIGKILL)
                }
            }
            AppLogger.shared.info("Audio2Face", "Stopped audio lip-sync test (pid=\(pid))")
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
        activeAudioName = nil

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
