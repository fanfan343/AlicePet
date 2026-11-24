import Foundation
import os

enum LogLevel: String {
    case debug = "DEBUG"
    case info = "INFO"
    case warn = "WARN"
    case error = "ERROR"
}

final class AppLogger: @unchecked Sendable {
    static let shared = AppLogger()

    private let osLog = Logger(subsystem: "com.funnyliu.AlicePet", category: "App")
    private let queue = DispatchQueue(label: "com.funnyliu.AlicePet.logger", qos: .utility)
    private let maxFileBytes: UInt64 = 2 * 1024 * 1024
    private let dateFormatter: DateFormatter

    let logDirectoryURL: URL
    let logFileURL: URL

    private init() {
        let fm = FileManager.default
        let libraryLogs = fm.urls(for: .libraryDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        self.logDirectoryURL = libraryLogs.appendingPathComponent("Logs/AlicePet", isDirectory: true)
        self.logFileURL = logDirectoryURL.appendingPathComponent("alicepet.log")

        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        df.locale = Locale(identifier: "en_US_POSIX")
        self.dateFormatter = df

        try? fm.createDirectory(at: logDirectoryURL, withIntermediateDirectories: true)
    }

    func info(_ category: String, _ message: String) {
        log(.info, category, message)
    }

    func warn(_ category: String, _ message: String) {
        log(.warn, category, message)
    }

    func error(_ category: String, _ message: String) {
        log(.error, category, message)
    }

    func debug(_ category: String, _ message: String) {
        log(.debug, category, message)
    }

    private func log(_ level: LogLevel, _ category: String, _ message: String) {
        let formatted = "[\(dateFormatter.string(from: Date()))] [\(level.rawValue)] [\(category)] \(message)"
        switch level {
        case .debug:
            osLog.debug("\(formatted, privacy: .public)")
        case .info:
            osLog.info("\(formatted, privacy: .public)")
        case .warn:
            osLog.warning("\(formatted, privacy: .public)")
        case .error:
            osLog.error("\(formatted, privacy: .public)")
        }

        queue.async { [logFileURL, logDirectoryURL, maxFileBytes] in
            Self.appendLine(formatted, to: logFileURL, dir: logDirectoryURL, maxBytes: maxFileBytes)
        }
    }

    private static func appendLine(_ line: String, to fileURL: URL, dir: URL, maxBytes: UInt64) {
        let fm = FileManager.default
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }

        if let attrs = try? fm.attributesOfItem(atPath: fileURL.path),
           let size = attrs[.size] as? UInt64,
           size >= maxBytes {
            let rotated = dir.appendingPathComponent("alicepet.log.1")
            try? fm.removeItem(at: rotated)
            try? fm.moveItem(at: fileURL, to: rotated)
        }

        guard let data = (line + "\n").data(using: .utf8) else { return }
        if fm.fileExists(atPath: fileURL.path),
           let handle = try? FileHandle(forWritingTo: fileURL) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        } else {
            try? data.write(to: fileURL, options: .atomic)
        }
    }
}
