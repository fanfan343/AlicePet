import AppKit
import MetalKit
import UniformTypeIdentifiers
import VRMMetalKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var mtkView: PetMetalView!
    private var controller: PetController!
    private var statusItem: NSStatusItem!

    private var isClickThrough: Bool = false
    private var isAlwaysOnTop: Bool = true
    private let autoConnectIPhoneKey = "autoConnectIPhoneCamera"

    private var autoConnectIPhoneCamera: Bool {
        get {
            if UserDefaults.standard.object(forKey: autoConnectIPhoneKey) == nil {
                return true
            }
            return UserDefaults.standard.bool(forKey: autoConnectIPhoneKey)
        }
        set {
            UserDefaults.standard.set(newValue, forKey: autoConnectIPhoneKey)
        }
    }

    private let initialModelURL: URL

    init(modelURL: URL) {
        self.initialModelURL = modelURL
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        AppLogger.shared.info("App", "AlicePet launching (pid=\(ProcessInfo.processInfo.processIdentifier))")

        do {
            controller = try PetController(sampleCount: 4)
        } catch {
            AppLogger.shared.error("App", "Failed to initialize Metal/VRMMetalKit: \(error)")
            NSApp.terminate(nil)
            return
        }

        let size = NSSize(width: 460, height: 680)
        let screenFrame = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let origin = NSPoint(
            x: screenFrame.maxX - size.width - 40,
            y: screenFrame.minY + 24
        )

        window = NSWindow(
            contentRect: NSRect(origin: origin, size: size),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.level = .floating
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.isMovableByWindowBackground = false

        mtkView = PetMetalView(frame: NSRect(origin: .zero, size: size), device: controller.device)
        mtkView.autoresizingMask = [.width, .height]
        mtkView.wantsLayer = true
        mtkView.layer?.isOpaque = false
        mtkView.layer?.backgroundColor = NSColor.clear.cgColor
        mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.sampleCount = 4
        mtkView.preferredFramesPerSecond = 60
        mtkView.delegate = controller

        mtkView.onOrbitDrag = { [weak self] dx, dy in
            self?.controller.camera.orbit(deltaX: dx, deltaY: dy)
        }
        mtkView.onPanDrag = { [weak self] dx, dy in
            self?.controller.camera.pan(deltaX: dx, deltaY: dy)
        }
        mtkView.onZoom = { [weak self] dy in
            self?.controller.camera.zoom(deltaY: dy)
        }
        mtkView.onDoubleClick = { [weak self] in
            self?.controller.camera.reset()
        }
        mtkView.contextMenuProvider = { [weak self] in
            self?.buildMenu() ?? NSMenu()
        }

        controller.onCameraError = { [weak self] message in
            self?.refreshMenus()
            let alert = NSAlert()
            alert.messageText = "摄像头捕捉提醒"
            alert.informativeText = message
            alert.alertStyle = .warning
            alert.runModal()
        }
        controller.onCameraDevicesChanged = { [weak self] in
            self?.refreshMenus()
        }
        controller.onCameraFaceDetectedChanged = { [weak self] detected in
            guard let self, self.controller.cameraTrackingActive else { return }
            self.statusItem.button?.title = detected ? "AlicePet · 📹" : "AlicePet · 📹(未见人脸)"
        }

        controller.hostView = mtkView
        window.contentView = mtkView
        window.makeKeyAndOrderFront(nil)

        setupStatusBar()
        controller.startTimers()

        Task { @MainActor in
            do {
                try await controller.loadModel(from: initialModelURL)
                let cameras = CameraFaceTracker.availableCameras()
                AppLogger.shared.info("Camera", "Detected \(cameras.count) camera(s): \(cameras.map(\.name).joined(separator: ", "))")
                if autoConnectIPhoneCamera, let iphoneCam = cameras.first(where: { $0.isContinuityCamera }) {
                    AppLogger.shared.info("Camera", "Auto-connecting iPhone Continuity Camera: \(iphoneCam.name)")
                    controller.startCameraCapture(deviceID: iphoneCam.id)
                    statusItem.button?.title = "AlicePet · 📹"
                    refreshMenus()
                }
            } catch {
                AppLogger.shared.error("Model", "Failed to load VRM model: \(error)")
            }
        }
    }

    private func setupStatusBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "AlicePet"
        statusItem.menu = buildMenu()
    }

    private func refreshMenus() {
        statusItem.menu = buildMenu()
    }

    private func buildMenu() -> NSMenu {
        let menu = NSMenu()

        let titleItem = NSMenuItem(title: "AlicePet · VRMMetalKit", action: nil, keyEquivalent: "")
        titleItem.isEnabled = false
        menu.addItem(titleItem)
        menu.addItem(.separator())

        // Camera face capture submenu (Mac / iPhone Continuity Camera)
        let camMenu = NSMenu()
        let stopCamItem = NSMenuItem(
            title: "关闭摄像头捕捉",
            action: #selector(stopCameraCapture),
            keyEquivalent: ""
        )
        stopCamItem.target = self
        stopCamItem.state = controller.cameraTrackingActive ? .off : .on
        camMenu.addItem(stopCamItem)
        camMenu.addItem(.separator())

        let cameras = CameraFaceTracker.availableCameras()
        if cameras.isEmpty {
            let emptyItem = NSMenuItem(title: "未检测到可用摄像头", action: nil, keyEquivalent: "")
            emptyItem.isEnabled = false
            camMenu.addItem(emptyItem)
        } else {
            for cam in cameras {
                let item = NSMenuItem(
                    title: cam.displayTitle,
                    action: #selector(selectCameraDevice(_:)),
                    keyEquivalent: ""
                )
                item.target = self
                item.representedObject = cam.id
                item.state = (controller.activeCameraID == cam.id) ? .on : .off
                camMenu.addItem(item)
            }
        }

        camMenu.addItem(.separator())
        let autoIPhoneItem = NSMenuItem(
            title: "启动时自动连接 iPhone 相机",
            action: #selector(toggleAutoConnectIPhone),
            keyEquivalent: ""
        )
        autoIPhoneItem.target = self
        autoIPhoneItem.state = autoConnectIPhoneCamera ? .on : .off
        camMenu.addItem(autoIPhoneItem)

        let mirrorItem = NSMenuItem(
            title: "镜像左右表情 (Mirror)",
            action: #selector(toggleCameraMirror),
            keyEquivalent: ""
        )
        mirrorItem.target = self
        mirrorItem.state = controller.cameraTracker.mirrorMode ? .on : .off
        camMenu.addItem(mirrorItem)

        let refreshCamItem = NSMenuItem(
            title: "刷新摄像头列表",
            action: #selector(refreshCameraList),
            keyEquivalent: ""
        )
        refreshCamItem.target = self
        camMenu.addItem(refreshCamItem)

        let camRoot = NSMenuItem(
            title: controller.cameraTrackingActive ? "摄像头捕捉 (运行中)" : "摄像头捕捉 (Mac / iPhone)",
            action: nil,
            keyEquivalent: ""
        )
        camRoot.submenu = camMenu
        menu.addItem(camRoot)

        // Expression test submenu
        let exprMenu = NSMenu()
        let presets: [(String, VRMExpressionPreset?)] = [
            ("Neutral (默认)", nil),
            ("Joy / Happy (开心)", .happy),
            ("Angry (生气)", .angry),
            ("Sorrow / Sad (悲伤)", .sad),
            ("Fun / Relaxed (放松)", .relaxed),
        ]
        for (label, preset) in presets {
            let item = NSMenuItem(title: label, action: #selector(selectPresetMood(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = preset?.rawValue ?? "__neutral__"
            exprMenu.addItem(item)
        }
        exprMenu.addItem(.separator())
        let customSamples = [
            "eyes winkL",
            "eyes winkR",
            "eyes sarcastic",
            "eyes shock",
            "mouth smile",
            "mouth ω",
            "nihihi",
            "Surprised",
            "JawOpen",
        ]
        for name in customSamples {
            let item = NSMenuItem(title: name, action: #selector(selectCustomExpression(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = name
            exprMenu.addItem(item)
        }
        let exprRoot = NSMenuItem(title: "表情测试", action: nil, keyEquivalent: "")
        exprRoot.submenu = exprMenu
        menu.addItem(exprRoot)

        menu.addItem(.separator())

        // Toggles
        let clickItem = NSMenuItem(title: "鼠标穿透 (Click-Through)", action: #selector(toggleClickThrough), keyEquivalent: "")
        clickItem.target = self
        clickItem.state = isClickThrough ? .on : .off
        menu.addItem(clickItem)

        let topItem = NSMenuItem(title: "窗口置顶", action: #selector(toggleAlwaysOnTop), keyEquivalent: "")
        topItem.target = self
        topItem.state = isAlwaysOnTop ? .on : .off
        menu.addItem(topItem)

        let springItem = NSMenuItem(title: "SpringBone 物理", action: #selector(toggleSpringBone), keyEquivalent: "")
        springItem.target = self
        springItem.state = controller.springBoneEnabled ? .on : .off
        menu.addItem(springItem)

        let eyeItem = NSMenuItem(title: "视线跟随鼠标", action: #selector(toggleEyeTracking), keyEquivalent: "")
        eyeItem.target = self
        eyeItem.state = controller.eyeTrackingEnabled ? .on : .off
        menu.addItem(eyeItem)

        let blinkItem = NSMenuItem(title: "自动眨眼", action: #selector(toggleAutoBlink), keyEquivalent: "")
        blinkItem.target = self
        blinkItem.state = controller.autoBlinkEnabled ? .on : .off
        menu.addItem(blinkItem)

        menu.addItem(.separator())

        let resetCamItem = NSMenuItem(title: "重置视角与平移 (或双击角色)", action: #selector(resetCamera), keyEquivalent: "")
        resetCamItem.target = self
        menu.addItem(resetCamItem)

        let openItem = NSMenuItem(title: "更换 VRM 模型…", action: #selector(openModelPanel), keyEquivalent: "o")
        openItem.target = self
        menu.addItem(openItem)

        let logItem = NSMenuItem(title: "打开日志文件夹", action: #selector(openLogFolder), keyEquivalent: "l")
        logItem.target = self
        menu.addItem(logItem)

        menu.addItem(.separator())

        let quitItem = NSMenuItem(title: "退出 AlicePet", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        return menu
    }

    @objc private func selectCameraDevice(_ sender: NSMenuItem) {
        guard let deviceID = sender.representedObject as? String else { return }
        controller.startCameraCapture(deviceID: deviceID)
        statusItem.button?.title = "AlicePet · 📹"
        refreshMenus()
    }

    @objc private func stopCameraCapture() {
        controller.stopCameraCapture()
        statusItem.button?.title = "AlicePet"
        refreshMenus()
    }

    @objc private func toggleCameraMirror() {
        controller.cameraTracker.mirrorMode.toggle()
        refreshMenus()
    }

    @objc private func toggleAutoConnectIPhone() {
        autoConnectIPhoneCamera.toggle()
        AppLogger.shared.info("App", "autoConnectIPhoneCamera=\(autoConnectIPhoneCamera)")
        refreshMenus()
    }

    @objc private func openLogFolder() {
        NSWorkspace.shared.activateFileViewerSelecting([AppLogger.shared.logFileURL])
    }

    @objc private func refreshCameraList() {
        refreshMenus()
    }

    @objc private func selectPresetMood(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String else { return }
        if raw == "__neutral__" {
            controller.setPresetMood(nil)
        } else if let preset = VRMExpressionPreset(rawValue: raw) {
            controller.setPresetMood(preset)
        }
    }

    @objc private func selectCustomExpression(_ sender: NSMenuItem) {
        guard let name = sender.representedObject as? String else { return }
        controller.setCustomExpression(name, weight: 1.0)
    }

    @objc private func toggleClickThrough() {
        isClickThrough.toggle()
        window.ignoresMouseEvents = isClickThrough
        refreshMenus()
    }

    @objc private func toggleAlwaysOnTop() {
        isAlwaysOnTop.toggle()
        window.level = isAlwaysOnTop ? .floating : .normal
        refreshMenus()
    }

    @objc private func toggleSpringBone() {
        controller.springBoneEnabled.toggle()
        refreshMenus()
    }

    @objc private func toggleEyeTracking() {
        controller.eyeTrackingEnabled.toggle()
        refreshMenus()
    }

    @objc private func toggleAutoBlink() {
        controller.autoBlinkEnabled.toggle()
        refreshMenus()
    }

    @objc private func resetCamera() {
        controller.camera.reset()
    }

    @objc private func openModelPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "vrm") ?? .data, .data]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.directoryURL = initialModelURL.deletingLastPathComponent()
        if panel.runModal() == .OK, let url = panel.url {
            Task { @MainActor in
                try? await controller.loadModel(from: url)
            }
        }
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }
}
