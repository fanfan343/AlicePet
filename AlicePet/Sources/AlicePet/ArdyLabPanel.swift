import AppKit
import Foundation

/// Floating HUD panel in 「测试功能 (Lab)」 for typing English motion prompts
/// to drive Alice in real time via ARDY Mini (distilled MiniLM + ARDY-Core40 ONNX).
@MainActor
final class ArdyLabPanel: NSObject, NSTextFieldDelegate {
    private let panel: NSPanel
    private let promptField = NSTextField()
    private let sendButton = NSButton(title: "切换动作", target: nil, action: nil)
    private let toggleButton = NSButton(title: "启动 ARDY 实时生成", target: nil, action: nil)
    private let cfgSlider = NSSlider(value: 2.5, minValue: 1.0, maxValue: 5.0, target: nil, action: nil)
    private let cfgValueLabel = NSTextField(labelWithString: "2.5")
    private let statusLabel = NSTextField(labelWithString: "状态：未启动 · 输入英文动作指令，ARDY 实时自回归续写")

    private let quickPrompts: [(title: String, prompt: String)] = [
        ("站立放松", "a person stands relaxed and shifts weight gently"),
        ("开心挥手", "a person stands and waves cheerfully with their right hand"),
        ("双手招手", "a person waves happily with both hands raised high"),
        ("优雅鞠躬", "a person bows politely forward and stands back up"),
        ("开心转圈", "a person spins around happily in place"),
        ("伸个懒腰", "a person stretches both arms up high and relaxes"),
        ("拍手鼓掌", "a person claps hands enthusiastically"),
        ("原地漫步", "a person walks casually forward and looks around"),
    ]

    var onToggleStartStop: (() -> Void)?
    var onSendPrompt: ((String, Double) -> Void)?

    init(initialPrompt: String, initialCfg: Double) {
        let size = NSSize(width: 440, height: 320)
        self.panel = NSPanel(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.titled, .closable, .utilityWindow],
            backing: .buffered,
            defer: false
        )
        super.init()

        panel.title = "Lab · ARDY Mini 文本生成全身动作 (20fps)"
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.isReleasedWhenClosed = false
        panel.center()

        promptField.stringValue = initialPrompt
        cfgSlider.doubleValue = initialCfg
        cfgValueLabel.stringValue = String(format: "%.1f", initialCfg)

        buildUI()
    }

    func show() {
        panel.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func setRunningState(_ running: Bool, status: String? = nil) {
        toggleButton.title = running ? "停止 ARDY 实时生成" : "启动 ARDY 实时生成"
        if let status, !status.isEmpty {
            statusLabel.stringValue = status
        } else if running {
            statusLabel.stringValue = "状态：运行中 · 输入新英文指令回车即可无缝切动作"
        } else {
            statusLabel.stringValue = "状态：未启动 · 输入英文动作指令，ARDY 实时自回归续写"
        }
    }

    private func buildUI() {
        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 10
        root.edgeInsets = NSEdgeInsets(top: 14, left: 16, bottom: 14, right: 16)
        root.translatesAutoresizingMaskIntoConstraints = false

        let inputTitle = NSTextField(labelWithString: "英文动作指令（支持随时无缝切换）")
        inputTitle.font = .systemFont(ofSize: 12, weight: .semibold)
        inputTitle.textColor = .secondaryLabelColor
        root.addArrangedSubview(inputTitle)

        promptField.placeholderString = "e.g. a person waves cheerfully with both hands"
        promptField.delegate = self
        promptField.target = self
        promptField.action = #selector(handleSendPrompt)

        sendButton.bezelStyle = .rounded
        sendButton.target = self
        sendButton.action = #selector(handleSendPrompt)

        let inputRow = NSStackView(views: [promptField, sendButton])
        inputRow.orientation = .horizontal
        inputRow.spacing = 8
        root.addArrangedSubview(inputRow)
        inputRow.widthAnchor.constraint(equalTo: root.widthAnchor, constant: -32).isActive = true

        // Quick preset buttons (2 rows of 4)
        let quickTitle = NSTextField(labelWithString: "快捷测试动作")
        quickTitle.font = .systemFont(ofSize: 12, weight: .semibold)
        quickTitle.textColor = .secondaryLabelColor
        root.addArrangedSubview(quickTitle)

        for rowIdx in 0..<2 {
            let row = NSStackView()
            row.orientation = .horizontal
            row.distribution = .fillEqually
            row.spacing = 6
            for colIdx in 0..<4 {
                let idx = rowIdx * 4 + colIdx
                let item = quickPrompts[idx]
                let btn = NSButton(title: item.title, target: self, action: #selector(handleQuickPrompt(_:)))
                btn.bezelStyle = .rounded
                btn.tag = idx
                row.addArrangedSubview(btn)
            }
            root.addArrangedSubview(row)
            row.widthAnchor.constraint(equalTo: root.widthAnchor, constant: -32).isActive = true
        }

        // CFG slider
        let cfgTitle = NSTextField(labelWithString: "指令引导强度 (CFG)")
        cfgTitle.font = .systemFont(ofSize: 12, weight: .medium)
        cfgSlider.isContinuous = true
        cfgSlider.target = self
        cfgSlider.action = #selector(handleCfgChanged)
        cfgValueLabel.font = .monospacedDigitSystemFont(ofSize: 12, weight: .semibold)
        cfgValueLabel.alignment = .right
        cfgValueLabel.widthAnchor.constraint(equalToConstant: 36).isActive = true

        let cfgRow = NSStackView(views: [cfgTitle, cfgSlider, cfgValueLabel])
        cfgRow.orientation = .horizontal
        cfgRow.spacing = 8
        root.addArrangedSubview(cfgRow)
        cfgRow.widthAnchor.constraint(equalTo: root.widthAnchor, constant: -32).isActive = true

        // Status + Toggle
        statusLabel.font = .systemFont(ofSize: 11)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.lineBreakMode = .byTruncatingTail
        root.addArrangedSubview(statusLabel)
        statusLabel.widthAnchor.constraint(equalTo: root.widthAnchor, constant: -32).isActive = true

        toggleButton.bezelStyle = .rounded
        toggleButton.target = self
        toggleButton.action = #selector(handleToggleStartStop)
        root.addArrangedSubview(toggleButton)
        toggleButton.widthAnchor.constraint(equalTo: root.widthAnchor, constant: -32).isActive = true

        guard let contentView = panel.contentView else { return }
        contentView.addSubview(root)
        NSLayoutConstraint.activate([
            root.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            root.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            root.topAnchor.constraint(equalTo: contentView.topAnchor),
            root.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
        ])
    }

    @objc private func handleToggleStartStop() {
        onToggleStartStop?()
    }

    @objc private func handleSendPrompt() {
        let text = promptField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        onSendPrompt?(text, cfgSlider.doubleValue)
    }

    @objc private func handleQuickPrompt(_ sender: NSButton) {
        guard sender.tag >= 0, sender.tag < quickPrompts.count else { return }
        let p = quickPrompts[sender.tag].prompt
        promptField.stringValue = p
        onSendPrompt?(p, cfgSlider.doubleValue)
    }

    @objc private func handleCfgChanged() {
        cfgValueLabel.stringValue = String(format: "%.1f", cfgSlider.doubleValue)
        let text = promptField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            onSendPrompt?(text, cfgSlider.doubleValue)
        }
    }
}
