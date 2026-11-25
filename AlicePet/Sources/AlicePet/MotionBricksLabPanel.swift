import AppKit
import Foundation

/// Floating HUD panel in 「测试功能 (Lab)」 for driving Alice with NVIDIA MotionBricks SOMA
/// (real-time generative locomotion, styles, and seamless in-between transitions over an isolated mixer channel).
@MainActor
final class MotionBricksLabPanel: NSObject {
    private let panel: NSPanel
    private let toggleButton = NSButton(title: "启动 MotionBricks 实时引擎", target: nil, action: nil)
    private let faceCameraCheckbox = NSButton(checkboxWithTitle: "保持朝向镜头（消除原地转向，适合桌宠展示）", target: nil, action: nil)
    private let statusLabel = NSTextField(labelWithString: "状态：未启动 · NVIDIA MotionBricks SOMA (~30ms 生成 · 独立混合通道)")
    private var presetButtons: [MotionBricksPreset: NSButton] = [:]

    private var selectedPreset: MotionBricksPreset

    var onToggleStartStop: (() -> Void)?
    var onSelectPreset: ((MotionBricksPreset) -> Void)?
    var onToggleFaceCamera: ((Bool) -> Void)?

    init(initialPreset: MotionBricksPreset, faceCamera: Bool) {
        self.selectedPreset = initialPreset
        let size = NSSize(width: 450, height: 340)
        self.panel = NSPanel(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.titled, .closable, .utilityWindow],
            backing: .buffered,
            defer: false
        )
        super.init()

        panel.title = "Lab · NVIDIA MotionBricks 实时动作与风格过渡"
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.isReleasedWhenClosed = false
        panel.center()

        faceCameraCheckbox.state = faceCamera ? .on : .off
        buildUI()
        highlightSelectedPreset()
    }

    func show() {
        panel.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func setRunningState(_ running: Bool, preset: MotionBricksPreset, status: String? = nil) {
        selectedPreset = preset
        highlightSelectedPreset()
        toggleButton.title = running ? "停止 MotionBricks 引擎" : "启动 MotionBricks 实时引擎"
        if let status, !status.isEmpty {
            statusLabel.stringValue = status
        } else if running {
            statusLabel.stringValue = "状态：运行中 [\(preset.displayTitle)] · 点击任意动作/风格实时平滑过渡"
        } else {
            statusLabel.stringValue = "状态：未启动 · NVIDIA MotionBricks SOMA (~30ms 生成 · 独立混合通道)"
        }
    }

    private func buildUI() {
        let root = NSStackView()
        root.orientation = .vertical
        root.alignment = .leading
        root.spacing = 10
        root.edgeInsets = NSEdgeInsets(top: 14, left: 16, bottom: 14, right: 16)
        root.translatesAutoresizingMaskIntoConstraints = false

        let locoTitle = NSTextField(labelWithString: "基础运动状态（实时生成过渡，不瞬移）")
        locoTitle.font = .systemFont(ofSize: 12, weight: .semibold)
        locoTitle.textColor = .secondaryLabelColor
        root.addArrangedSubview(locoTitle)

        let locoRows: [[MotionBricksPreset]] = [
            [.idle, .walk, .run],
            [.strafeLeft, .strafeRight, .crouch],
        ]
        for rowItems in locoRows {
            root.addArrangedSubview(makePresetRow(rowItems, parentWidthAnchor: root.widthAnchor))
        }

        let styleTitle = NSTextField(labelWithString: "6 种官方风格变体 (Style Variations)")
        styleTitle.font = .systemFont(ofSize: 12, weight: .semibold)
        styleTitle.textColor = .secondaryLabelColor
        root.addArrangedSubview(styleTitle)

        let styleRows: [[MotionBricksPreset]] = [
            [.skipping, .vogue, .combat],
            [.zombie, .injuredTorso, .injuredLeg],
        ]
        for rowItems in styleRows {
            root.addArrangedSubview(makePresetRow(rowItems, parentWidthAnchor: root.widthAnchor))
        }

        faceCameraCheckbox.font = .systemFont(ofSize: 11.5)
        faceCameraCheckbox.target = self
        faceCameraCheckbox.action = #selector(handleFaceCameraToggle)
        root.addArrangedSubview(faceCameraCheckbox)

        let mixerHint = NSTextField(labelWithString: "分层混合：Body 由 MotionBricks 驱动，可与 LAM 语音对口型 + 头部叠加点头同时开启")
        mixerHint.font = .systemFont(ofSize: 10.5)
        mixerHint.textColor = .tertiaryLabelColor
        root.addArrangedSubview(mixerHint)

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

    private func makePresetRow(_ presets: [MotionBricksPreset], parentWidthAnchor: NSLayoutDimension) -> NSStackView {
        let row = NSStackView()
        row.orientation = .horizontal
        row.distribution = .fillEqually
        row.spacing = 6
        for preset in presets {
            let btn = NSButton(title: preset.displayTitle, target: self, action: #selector(handlePresetTap(_:)))
            btn.bezelStyle = .rounded
            btn.identifier = NSUserInterfaceItemIdentifier(preset.rawValue)
            presetButtons[preset] = btn
            row.addArrangedSubview(btn)
        }
        row.widthAnchor.constraint(equalTo: parentWidthAnchor, constant: -32).isActive = true
        return row
    }

    private func highlightSelectedPreset() {
        for (preset, btn) in presetButtons {
            btn.contentTintColor = (preset == selectedPreset) ? .controlAccentColor : nil
        }
    }

    @objc private func handlePresetTap(_ sender: NSButton) {
        guard let raw = sender.identifier?.rawValue,
              let preset = MotionBricksPreset(rawValue: raw) else { return }
        selectedPreset = preset
        highlightSelectedPreset()
        onSelectPreset?(preset)
    }

    @objc private func handleFaceCameraToggle() {
        onToggleFaceCamera?(faceCameraCheckbox.state == .on)
    }

    @objc private func handleToggleStartStop() {
        onToggleStartStop?()
    }
}
