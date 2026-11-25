import AppKit
import Foundation

struct AudioLipSyncParams: Codable, Equatable {
    var jawGain: Double = 3.00
    var vowelGain: Double = 2.50
    var lipCloseStrength: Double = 1.50
    var lipRevealGain: Double = 1.50
    var smoothing: Double = 0.00
    var upperFaceGain: Double = 1.50
    var headMotionGain: Double = 1.40
    var syncOffsetMs: Double = 0.0
    var naturalBlinks: Bool = true
    var loopPlayback: Bool = true

    private static let defaultsKey = "labAudioLipSyncParamsV2"

    static func loadFromDefaults() -> AudioLipSyncParams {
        guard let data = UserDefaults.standard.data(forKey: defaultsKey),
              let decoded = try? JSONDecoder().decode(AudioLipSyncParams.self, from: data) else {
            return AudioLipSyncParams()
        }
        return decoded
    }

    func saveToDefaults() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.defaultsKey)
        }
    }

    func toLiveJSONLine() -> Data? {
        let dict: [String: Any] = [
            "jaw_gain": jawGain,
            "vowel_gain": vowelGain,
            "lip_close_strength": lipCloseStrength,
            "lip_reveal_gain": lipRevealGain,
            "smoothing": smoothing,
            "upper_face_gain": upperFaceGain,
            "head_motion_gain": headMotionGain,
            "sync_offset_ms": syncOffsetMs,
            "natural_blinks": naturalBlinks,
            "loop": loopPlayback,
        ]
        guard var data = try? JSONSerialization.data(withJSONObject: dict) else { return nil }
        data.append(0x0A) // newline delimiter
        return data
    }

    func prettyRecipeSummary() -> String {
        String(
            format: "jaw=%.2f, vowel=%.2f, close=%.2f, reveal=%.2f, smooth=%.2f, upper=%.2f, head=%.2f, offset=%.0fms, blink=%@",
            jawGain, vowelGain, lipCloseStrength, lipRevealGain, smoothing, upperFaceGain, headMotionGain, syncOffsetMs,
            naturalBlinks ? "on" : "off"
        )
    }
}

enum AudioLipSyncPreset: String, CaseIterable {
    case animeStandard = "anime"
    case energetic = "energetic"
    case gentleSoft = "gentle"
    case rawRealistic = "raw"

    var title: String {
        switch self {
        case .animeStandard: return "二次元标准"
        case .energetic: return "元气活泼"
        case .gentleSoft: return "轻声温柔"
        case .rawRealistic: return "原始写实"
        }
    }

    func apply(preservingLoop loop: Bool, offsetMs: Double) -> AudioLipSyncParams {
        switch self {
        case .animeStandard:
            return AudioLipSyncParams(
                jawGain: 3.00, vowelGain: 2.50, lipCloseStrength: 1.50, lipRevealGain: 1.50,
                smoothing: 0.00, upperFaceGain: 1.50, headMotionGain: 1.40,
                syncOffsetMs: offsetMs, naturalBlinks: true, loopPlayback: loop
            )
        case .energetic:
            return AudioLipSyncParams(
                jawGain: 3.60, vowelGain: 2.05, lipCloseStrength: 0.45, lipRevealGain: 0.78,
                smoothing: 0.20, upperFaceGain: 1.35, headMotionGain: 1.40,
                syncOffsetMs: offsetMs, naturalBlinks: true, loopPlayback: loop
            )
        case .gentleSoft:
            return AudioLipSyncParams(
                jawGain: 1.95, vowelGain: 1.20, lipCloseStrength: 0.72, lipRevealGain: 0.40,
                smoothing: 0.50, upperFaceGain: 0.75, headMotionGain: 0.65,
                syncOffsetMs: offsetMs, naturalBlinks: true, loopPlayback: loop
            )
        case .rawRealistic:
            return AudioLipSyncParams(
                jawGain: 1.00, vowelGain: 1.00, lipCloseStrength: 0.00, lipRevealGain: 1.00,
                smoothing: 0.15, upperFaceGain: 1.00, headMotionGain: 0.00,
                syncOffsetMs: offsetMs, naturalBlinks: false, loopPlayback: loop
            )
        }
    }
}

/// Compact native macOS floating HUD panel for real-time tuning of LAM_Audio2Expression lip-sync parameters.
@MainActor
final class AudioLipSyncTuningPanel: NSObject {
    private let panel: NSPanel
    private var params: AudioLipSyncParams

    private struct SliderRow {
        let key: String
        let slider: NSSlider
        let valueLabel: NSTextField
        let formatter: (Double) -> String
    }

    private var rows: [SliderRow] = []
    private let loopCheckbox = NSButton(checkboxWithTitle: "循环播放（边听边调，实时生效）", target: nil, action: nil)
    private let blinkCheckbox = NSButton(checkboxWithTitle: "自然周期眨眼叠加", target: nil, action: nil)
    private let playButton = NSButton(title: "播放 10s 测试", target: nil, action: nil)
    private let pickAudioButton = NSButton(title: "选音频…", target: nil, action: nil)
    private let copyButton = NSButton(title: "复制参数", target: nil, action: nil)
    private let resetButton = NSButton(title: "重置", target: nil, action: nil)
    private let statusLabel = NSTextField(labelWithString: "拖动滑块即可实时改变口型，无需重新推理")

    var onParamsChanged: ((AudioLipSyncParams) -> Void)?
    var onTogglePlay: (() -> Void)?
    var onPickAudio: (() -> Void)?

    init(initialParams: AudioLipSyncParams) {
        self.params = initialParams
        let size = NSSize(width: 400, height: 520)
        self.panel = NSPanel(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.titled, .closable, .utilityWindow],
            backing: .buffered,
            defer: false
        )
        super.init()

        panel.title = "Lab · 口型与表情实时调参 (LAM 52通道)"
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.isReleasedWhenClosed = false
        panel.center()

        buildUI()
        syncControlsFromParams()
    }

    func show() {
        panel.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func setPlayingState(_ playing: Bool, audioName: String?) {
        playButton.title = playing ? "停止播放" : "播放 10s 测试"
        if playing {
            statusLabel.stringValue = "正在播放：\(audioName ?? "vtuber_10s.wav") · 拖动滑块实时生效"
        } else {
            statusLabel.stringValue = "拖动滑块即可实时改变口型，无需重新推理"
        }
    }

    private func buildUI() {
        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 10
        root.edgeInsets = NSEdgeInsets(top: 14, left: 16, bottom: 14, right: 16)
        root.translatesAutoresizingMaskIntoConstraints = false

        // 1. Preset row
        let presetLabel = NSTextField(labelWithString: "一键预设风格")
        presetLabel.font = .systemFont(ofSize: 12, weight: .semibold)
        presetLabel.textColor = .secondaryLabelColor
        root.addArrangedSubview(presetLabel)

        let presetBar = NSStackView()
        presetBar.orientation = .horizontal
        presetBar.distribution = .fillEqually
        presetBar.spacing = 6
        for (idx, preset) in AudioLipSyncPreset.allCases.enumerated() {
            let btn = NSButton(title: preset.title, target: self, action: #selector(applyPreset(_:)))
            btn.bezelStyle = .rounded
            btn.tag = idx
            presetBar.addArrangedSubview(btn)
        }
        root.addArrangedSubview(presetBar)
        presetBar.widthAnchor.constraint(equalTo: root.widthAnchor, constant: -32).isActive = true

        // 2. 8 Live Sliders
        let specs: [(key: String, title: String, hint: String, min: Double, max: Double, fmt: (Double) -> String)] = [
            ("jaw", "张嘴幅度", "a/o 大开口倍率", 0.5, 4.5, { String(format: "%.2f×", $0) }),
            ("vowel", "元音口型夸张度", "u/o/i/e 嘟嘴与咧嘴", 0.5, 2.5, { String(format: "%.2f×", $0) }),
            ("close", "闭唇紧密度", "b/p/m 与停顿收口", 0.0, 1.5, { String(format: "%.2f", $0) }),
            ("reveal", "上下唇牵拉 / 露齿", "上下唇翻开幅度", 0.0, 1.5, { String(format: "%.2f×", $0) }),
            ("smooth", "平滑度", "低=干脆灵敏 / 高=柔滑", 0.0, 0.85, { String(format: "%.0f%%", $0 * 100) }),
            ("upper", "眉眼微动", "说话挑眉与脸颊联动", 0.0, 1.5, { String(format: "%.2f×", $0) }),
            ("head", "说话点头幅度", "随音量节奏微点头 (0=关)", 0.0, 2.0, { String(format: "%.2f×", $0) }),
            ("offset", "音画同步偏移", "负=嘴型提前 / 正=嘴型延后", -150.0, 150.0, { String(format: "%+.0f ms", $0) }),
        ]

        for spec in specs {
            let rowStack = NSStackView()
            rowStack.orientation = .vertical
            rowStack.alignment = .leading
            rowStack.spacing = 2

            let header = NSStackView()
            header.orientation = .horizontal
            header.distribution = .fill

            let titleLbl = NSTextField(labelWithString: "\(spec.title)  ")
            titleLbl.font = .systemFont(ofSize: 12, weight: .medium)
            let hintLbl = NSTextField(labelWithString: spec.hint)
            hintLbl.font = .systemFont(ofSize: 11)
            hintLbl.textColor = .secondaryLabelColor
            let valLbl = NSTextField(labelWithString: "")
            valLbl.font = .monospacedDigitSystemFont(ofSize: 12, weight: .semibold)
            valLbl.alignment = .right

            let spacer = NSView()
            spacer.setContentHuggingPriority(.defaultLow, for: .horizontal)

            header.addArrangedSubview(titleLbl)
            header.addArrangedSubview(hintLbl)
            header.addArrangedSubview(spacer)
            header.addArrangedSubview(valLbl)

            let slider = NSSlider(value: spec.min, minValue: spec.min, maxValue: spec.max, target: self, action: #selector(sliderChanged(_:)))
            slider.isContinuous = true
            slider.tag = rows.count

            rowStack.addArrangedSubview(header)
            rowStack.addArrangedSubview(slider)
            root.addArrangedSubview(rowStack)

            rowStack.widthAnchor.constraint(equalTo: root.widthAnchor, constant: -32).isActive = true
            header.widthAnchor.constraint(equalTo: rowStack.widthAnchor).isActive = true
            slider.widthAnchor.constraint(equalTo: rowStack.widthAnchor).isActive = true

            rows.append(SliderRow(key: spec.key, slider: slider, valueLabel: valLbl, formatter: spec.fmt))
        }

        // 3. Checkboxes
        loopCheckbox.target = self
        loopCheckbox.action = #selector(checkboxChanged)
        blinkCheckbox.target = self
        blinkCheckbox.action = #selector(checkboxChanged)

        let checkRow = NSStackView(views: [loopCheckbox, blinkCheckbox])
        checkRow.orientation = .horizontal
        checkRow.spacing = 12
        root.addArrangedSubview(checkRow)

        // 4. Status + Action Buttons
        statusLabel.font = .systemFont(ofSize: 11)
        statusLabel.textColor = .secondaryLabelColor
        root.addArrangedSubview(statusLabel)

        playButton.bezelStyle = .rounded
        playButton.keyEquivalent = "\r"
        playButton.target = self
        playButton.action = #selector(handleTogglePlay)

        pickAudioButton.bezelStyle = .rounded
        pickAudioButton.target = self
        pickAudioButton.action = #selector(handlePickAudio)

        copyButton.bezelStyle = .rounded
        copyButton.target = self
        copyButton.action = #selector(handleCopyParams)

        resetButton.bezelStyle = .rounded
        resetButton.target = self
        resetButton.action = #selector(handleReset)

        let actionBar = NSStackView(views: [playButton, pickAudioButton, copyButton, resetButton])
        actionBar.orientation = .horizontal
        actionBar.distribution = .fillProportionally
        actionBar.spacing = 8
        root.addArrangedSubview(actionBar)
        actionBar.widthAnchor.constraint(equalTo: root.widthAnchor, constant: -32).isActive = true

        guard let contentView = panel.contentView else { return }
        contentView.addSubview(root)
        NSLayoutConstraint.activate([
            root.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            root.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            root.topAnchor.constraint(equalTo: contentView.topAnchor),
            root.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
        ])
    }

    private func valueFor(key: String) -> Double {
        switch key {
        case "jaw": return params.jawGain
        case "vowel": return params.vowelGain
        case "close": return params.lipCloseStrength
        case "reveal": return params.lipRevealGain
        case "smooth": return params.smoothing
        case "upper": return params.upperFaceGain
        case "head": return params.headMotionGain
        case "offset": return params.syncOffsetMs
        default: return 0
        }
    }

    private func setValue(_ v: Double, for key: String) {
        switch key {
        case "jaw": params.jawGain = v
        case "vowel": params.vowelGain = v
        case "close": params.lipCloseStrength = v
        case "reveal": params.lipRevealGain = v
        case "smooth": params.smoothing = v
        case "upper": params.upperFaceGain = v
        case "head": params.headMotionGain = v
        case "offset": params.syncOffsetMs = v.rounded()
        default: break
        }
    }

    private func syncControlsFromParams() {
        for row in rows {
            let v = valueFor(key: row.key)
            row.slider.doubleValue = v
            row.valueLabel.stringValue = row.formatter(v)
        }
        loopCheckbox.state = params.loopPlayback ? .on : .off
        blinkCheckbox.state = params.naturalBlinks ? .on : .off
    }

    private func emitChanged() {
        params.saveToDefaults()
        onParamsChanged?(params)
    }

    @objc private func sliderChanged(_ sender: NSSlider) {
        guard sender.tag >= 0, sender.tag < rows.count else { return }
        let row = rows[sender.tag]
        let v = (row.key == "offset") ? sender.doubleValue.rounded() : sender.doubleValue
        setValue(v, for: row.key)
        row.valueLabel.stringValue = row.formatter(v)
        emitChanged()
    }

    @objc private func checkboxChanged() {
        params.loopPlayback = (loopCheckbox.state == .on)
        params.naturalBlinks = (blinkCheckbox.state == .on)
        emitChanged()
    }

    @objc private func applyPreset(_ sender: NSButton) {
        let all = AudioLipSyncPreset.allCases
        guard sender.tag >= 0, sender.tag < all.count else { return }
        params = all[sender.tag].apply(preservingLoop: params.loopPlayback, offsetMs: params.syncOffsetMs)
        syncControlsFromParams()
        emitChanged()
    }

    @objc private func handleTogglePlay() {
        onTogglePlay?()
    }

    @objc private func handlePickAudio() {
        onPickAudio?()
    }

    @objc private func handleCopyParams() {
        let summary = params.prettyRecipeSummary()
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(summary, forType: .string)
        statusLabel.stringValue = "已复制参数：\(summary)"
        AppLogger.shared.info("Audio2Face", "Copied tuning params: \(summary)")
    }

    @objc private func handleReset() {
        params = AudioLipSyncParams()
        syncControlsFromParams()
        emitChanged()
    }
}
