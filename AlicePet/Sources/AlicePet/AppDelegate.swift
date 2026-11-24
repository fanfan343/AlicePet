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
    private let upperBodyTrackingKey = "cameraUpperBodyTracking"
    private let lightingPresetKey = "mtoonLightingPreset"
    private let outlineStyleKey = "mtoonOutlineStyle"
    nonisolated static let lastModelPathKey = "lastOpenedVRMModelPath"

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
            if UserDefaults.standard.object(forKey: upperBodyTrackingKey) != nil {
                controller.cameraTracker.upperBodyTrackingEnabled = UserDefaults.standard.bool(forKey: upperBodyTrackingKey)
            }
            if let savedLight = UserDefaults.standard.string(forKey: lightingPresetKey),
               let preset = LightingPreset(rawValue: savedLight) {
                controller.setLightingPreset(preset)
            }
            if let savedOutline = UserDefaults.standard.string(forKey: outlineStyleKey),
               let style = OutlineStyle(rawValue: savedOutline) {
                controller.setOutlineStyle(style)
            }
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
        mtkView.depthStencilStorageMode = .private
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
                refreshMenus()
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

        // 1. Camera face capture submenu (Mac / iPhone Continuity Camera)
        let camMenu = NSMenu()
        let stopCamItem = NSMenuItem(title: "关闭摄像头捕捉", action: #selector(stopCameraCapture), keyEquivalent: "")
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
                let item = NSMenuItem(title: cam.displayTitle, action: #selector(selectCameraDevice(_:)), keyEquivalent: "")
                item.target = self
                item.representedObject = cam.id
                item.state = (controller.activeCameraID == cam.id) ? .on : .off
                camMenu.addItem(item)
            }
        }

        camMenu.addItem(.separator())
        let autoIPhoneItem = NSMenuItem(title: "启动时自动连接 iPhone 相机", action: #selector(toggleAutoConnectIPhone), keyEquivalent: "")
        autoIPhoneItem.target = self
        autoIPhoneItem.state = autoConnectIPhoneCamera ? .on : .off
        camMenu.addItem(autoIPhoneItem)

        let upperBodyItem = NSMenuItem(title: "联动上半身与双臂 (3D 骨架)", action: #selector(toggleUpperBodyTracking), keyEquivalent: "")
        upperBodyItem.target = self
        upperBodyItem.state = controller.cameraTracker.upperBodyTrackingEnabled ? .on : .off
        camMenu.addItem(upperBodyItem)

        let mirrorItem = NSMenuItem(title: "镜像左右表情 (Mirror)", action: #selector(toggleCameraMirror), keyEquivalent: "")
        mirrorItem.target = self
        mirrorItem.state = controller.cameraTracker.mirrorMode ? .on : .off
        camMenu.addItem(mirrorItem)

        let calibrateItem = NSMenuItem(title: "校准中性表情与坐姿", action: #selector(calibrateCameraNeutral), keyEquivalent: "")
        calibrateItem.target = self
        calibrateItem.isEnabled = controller.cameraTrackingActive
        camMenu.addItem(calibrateItem)

        let refreshCamItem = NSMenuItem(title: "刷新摄像头列表", action: #selector(refreshCameraList), keyEquivalent: "")
        refreshCamItem.target = self
        camMenu.addItem(refreshCamItem)

        let camRoot = NSMenuItem(title: controller.cameraTrackingActive ? "摄像头捕捉 (运行中)" : "摄像头捕捉 (Mac / iPhone)", action: nil, keyEquivalent: "")
        camRoot.submenu = camMenu
        menu.addItem(camRoot)

        // 2. Pose & AFK Mode Submenu
        let poseMenu = NSMenu()
        for mode in PetPoseMode.allCases {
            let item = NSMenuItem(title: mode.displayTitle, action: #selector(selectPoseMode(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = mode.rawValue
            item.state = (controller.poseMode == mode) ? .on : .off
            poseMenu.addItem(item)
        }
        poseMenu.addItem(.separator())
        let autoPropItem = NSMenuItem(title: "道具显示：随姿态自动", action: #selector(selectPropOverride(_:)), keyEquivalent: "")
        autoPropItem.target = self
        autoPropItem.representedObject = "__auto__"
        autoPropItem.state = (controller.manualPropOverride == nil) ? .on : .off
        poseMenu.addItem(autoPropItem)

        let propChoices: [(String, PropSceneRenderer.ActivePropMode)] = [
            ("单独显示：魔法书 (Book)", .bookOnly),
            ("单独显示：下午茶套装 (Tea Set)", .teaSetOnly),
            ("全部道具同时显示", .bookAndTea),
            ("隐藏全部独立道具", .none),
        ]
        for (title, mode) in propChoices {
            let item = NSMenuItem(title: title, action: #selector(selectPropOverride(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = mode.rawValue
            item.state = (controller.manualPropOverride == mode) ? .on : .off
            poseMenu.addItem(item)
        }
        let poseRoot = NSMenuItem(title: "姿态与 AFK 状态 (\(controller.poseMode.displayTitle))", action: nil, keyEquivalent: "")
        poseRoot.submenu = poseMenu
        menu.addItem(poseRoot)

        // 3. Hand Gesture Submenu
        let handMenu = NSMenu()
        let bothMenu = NSMenu()
        let leftMenu = NSMenu()
        let rightMenu = NSMenu()
        for preset in HandGesturePreset.allCases {
            let bItem = NSMenuItem(title: preset.displayTitle, action: #selector(selectBothHandGesture(_:)), keyEquivalent: "")
            bItem.target = self
            bItem.representedObject = preset.rawValue
            bItem.state = (controller.leftHandGesture == preset && controller.rightHandGesture == preset) ? .on : .off
            bothMenu.addItem(bItem)

            let lItem = NSMenuItem(title: preset.displayTitle, action: #selector(selectLeftHandGesture(_:)), keyEquivalent: "")
            lItem.target = self
            lItem.representedObject = preset.rawValue
            lItem.state = (controller.leftHandGesture == preset) ? .on : .off
            leftMenu.addItem(lItem)

            let rItem = NSMenuItem(title: preset.displayTitle, action: #selector(selectRightHandGesture(_:)), keyEquivalent: "")
            rItem.target = self
            rItem.representedObject = preset.rawValue
            rItem.state = (controller.rightHandGesture == preset) ? .on : .off
            rightMenu.addItem(rItem)
        }
        let bothRoot = NSMenuItem(title: "双手同步", action: nil, keyEquivalent: "")
        bothRoot.submenu = bothMenu
        let leftRoot = NSMenuItem(title: "左手 (\(controller.leftHandGesture.rawValue))", action: nil, keyEquivalent: "")
        leftRoot.submenu = leftMenu
        let rightRoot = NSMenuItem(title: "右手 (\(controller.rightHandGesture.rawValue))", action: nil, keyEquivalent: "")
        rightRoot.submenu = rightMenu
        handMenu.addItem(bothRoot)
        handMenu.addItem(leftRoot)
        handMenu.addItem(rightRoot)
        let handRoot = NSMenuItem(title: "手势配方 (Hands)", action: nil, keyEquivalent: "")
        handRoot.submenu = handMenu
        menu.addItem(handRoot)

        // 4. Outfit & Hair Submenu (adapts dynamically to the loaded VRM profile & morphs)
        if controller.hasAnyOutfitControls || controller.hasHairStyleControls {
            let outfitMenu = NSMenu()
            let outfitToggles: [(String, OutfitToggleKind, Selector, Bool)] = [
                ("连衣裙 (Dress)", .dress, #selector(toggleOutfitDress), controller.outfitState.showDress),
                ("丝袜 (Pantyhose)", .pantyhose, #selector(toggleOutfitPantyhose), controller.outfitState.showPantyhose),
                ("鞋子 (Shoes)", .shoes, #selector(toggleOutfitShoes), controller.outfitState.showShoes),
                ("发箍 (HairBand)", .hairBand, #selector(toggleOutfitHairBand), controller.outfitState.showHairBand),
                ("胸前蝴蝶结 (ChestRibbon)", .chestRibbon, #selector(toggleOutfitChestRibbon), controller.outfitState.showChestRibbon),
                ("侧发 (HairSide)", .hairSide, #selector(toggleOutfitHairSide), controller.outfitState.showHairSide),
            ]
            for (title, kind, sel, active) in outfitToggles {
                let supported = controller.isOutfitToggleSupported(kind)
                let item = NSMenuItem(
                    title: supported ? title : "\(title) · 当前模型不含",
                    action: supported ? sel : nil,
                    keyEquivalent: ""
                )
                item.target = self
                item.isEnabled = supported
                item.state = (supported && active) ? .on : .off
                outfitMenu.addItem(item)
            }
            if controller.hasHairStyleControls {
                outfitMenu.addItem(.separator())
                for style in HairStyleOption.allCases {
                    let item = NSMenuItem(title: style.displayTitle, action: #selector(selectHairStyle(_:)), keyEquivalent: "")
                    item.target = self
                    item.representedObject = style.rawValue
                    item.state = (controller.outfitState.hairStyle == style) ? .on : .off
                    outfitMenu.addItem(item)
                }
            }
            let outfitRoot = NSMenuItem(title: "换装与发型", action: nil, keyEquivalent: "")
            outfitRoot.submenu = outfitMenu
            menu.addItem(outfitRoot)
        }

        // 5. Expression Presets Submenu
        let exprMenu = NSMenu()
        let presets: [(String, VRMExpressionPreset?)] = [
            ("Neutral (重置默认)", nil),
            ("Happy (开心)", .happy),
            ("Angry (生气)", .angry),
            ("Sad (委屈)", .sad),
            ("Relaxed (放松)", .relaxed),
        ]
        for (label, preset) in presets {
            let item = NSMenuItem(title: label, action: #selector(selectPresetMood(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = preset?.rawValue ?? "__neutral__"
            let isSelected = (controller.activeCustomExpression == nil) && (controller.activePresetMood == preset)
            item.state = isSelected ? .on : .off
            exprMenu.addItem(item)
        }
        if !controller.recipeStore.availableFaceRecipes.isEmpty {
            exprMenu.addItem(.separator())
            for entry in controller.recipeStore.availableFaceRecipes {
                let item = NSMenuItem(title: entry.title, action: #selector(selectCustomExpression(_:)), keyEquivalent: "")
                item.target = self
                item.representedObject = entry.id
                item.state = (controller.activeCustomExpression == entry.id) ? .on : .off
                exprMenu.addItem(item)
            }
        }
        let exprRoot = NSMenuItem(title: "表情配方 (\(controller.recipeStore.availableFaceRecipes.count) 套)", action: nil, keyEquivalent: "")
        exprRoot.submenu = exprMenu
        menu.addItem(exprRoot)

        // 5b. MToon Lighting & Outline Submenu
        let lookDevMenu = NSMenu()
        for preset in LightingPreset.allCases {
            let item = NSMenuItem(title: preset.displayTitle, action: #selector(selectLightingPreset(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = preset.rawValue
            item.state = (controller.lightingPreset == preset) ? .on : .off
            lookDevMenu.addItem(item)
        }
        lookDevMenu.addItem(.separator())
        for style in OutlineStyle.allCases {
            let item = NSMenuItem(title: style.displayTitle, action: #selector(selectOutlineStyle(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = style.rawValue
            item.state = (controller.outlineStyle == style) ? .on : .off
            lookDevMenu.addItem(item)
        }
        let lookDevRoot = NSMenuItem(title: "光影与描边 (MToon)", action: nil, keyEquivalent: "")
        lookDevRoot.submenu = lookDevMenu
        menu.addItem(lookDevRoot)

        menu.addItem(.separator())

        // 6. Window & Renderer toggles
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

        let viewMenu = NSMenu()
        for preset in CameraViewPreset.allCases {
            let item = NSMenuItem(title: preset.displayTitle, action: #selector(selectCameraViewPreset(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = preset.rawValue
            viewMenu.addItem(item)
        }
        viewMenu.addItem(.separator())
        let resetCamItem = NSMenuItem(title: "重置视角与平移 (或双击角色)", action: #selector(resetCamera), keyEquivalent: "")
        resetCamItem.target = self
        viewMenu.addItem(resetCamItem)
        let viewRoot = NSMenuItem(title: "视角机位 (Camera)", action: nil, keyEquivalent: "")
        viewRoot.submenu = viewMenu
        menu.addItem(viewRoot)

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

    // MARK: - Actions

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

    @objc private func toggleUpperBodyTracking() {
        controller.cameraTracker.upperBodyTrackingEnabled.toggle()
        let enabled = controller.cameraTracker.upperBodyTrackingEnabled
        UserDefaults.standard.set(enabled, forKey: upperBodyTrackingKey)
        if !enabled {
            controller.bodySolver.reset()
        }
        AppLogger.shared.info("Camera", "upperBodyTrackingEnabled=\(enabled)")
        refreshMenus()
    }

    @objc private func calibrateCameraNeutral() {
        controller.calibrateCameraNeutral()
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

    @objc private func selectPoseMode(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let mode = PetPoseMode(rawValue: raw) else { return }
        controller.setPoseMode(mode)
        refreshMenus()
    }

    @objc private func selectPropOverride(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String else { return }
        if raw == "__auto__" {
            controller.manualPropOverride = nil
        } else if let mode = PropSceneRenderer.ActivePropMode(rawValue: raw) {
            controller.manualPropOverride = mode
        }
        refreshMenus()
    }

    @objc private func selectBothHandGesture(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let preset = HandGesturePreset(rawValue: raw) else { return }
        controller.setHandGesture(preset, side: .both)
        refreshMenus()
    }

    @objc private func selectLeftHandGesture(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let preset = HandGesturePreset(rawValue: raw) else { return }
        controller.setHandGesture(preset, side: .left)
        refreshMenus()
    }

    @objc private func selectRightHandGesture(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let preset = HandGesturePreset(rawValue: raw) else { return }
        controller.setHandGesture(preset, side: .right)
        refreshMenus()
    }

    @objc private func toggleOutfitDress() {
        controller.outfitState.showDress.toggle()
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc private func toggleOutfitPantyhose() {
        controller.outfitState.showPantyhose.toggle()
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc private func toggleOutfitShoes() {
        controller.outfitState.showShoes.toggle()
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc private func toggleOutfitHairBand() {
        controller.outfitState.showHairBand.toggle()
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc private func toggleOutfitChestRibbon() {
        controller.outfitState.showChestRibbon.toggle()
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc private func toggleOutfitHairSide() {
        controller.outfitState.showHairSide.toggle()
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc private func selectHairStyle(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let style = HairStyleOption(rawValue: raw) else { return }
        controller.outfitState.hairStyle = style
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc private func selectLightingPreset(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let preset = LightingPreset(rawValue: raw) else { return }
        UserDefaults.standard.set(preset.rawValue, forKey: lightingPresetKey)
        controller.setLightingPreset(preset)
        refreshMenus()
    }

    @objc private func selectOutlineStyle(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let style = OutlineStyle(rawValue: raw) else { return }
        UserDefaults.standard.set(style.rawValue, forKey: outlineStyleKey)
        controller.setOutlineStyle(style)
        refreshMenus()
    }

    @objc private func selectPresetMood(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String else { return }
        if raw == "__neutral__" {
            controller.setPresetMood(nil)
        } else if let preset = VRMExpressionPreset(rawValue: raw) {
            controller.setPresetMood(preset)
        }
        refreshMenus()
    }

    @objc private func selectCustomExpression(_ sender: NSMenuItem) {
        guard let name = sender.representedObject as? String else { return }
        controller.setCustomExpression(name, weight: 1.0)
        refreshMenus()
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

    @objc private func selectCameraViewPreset(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let preset = CameraViewPreset(rawValue: raw) else { return }
        controller.camera.apply(preset: preset)
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
            UserDefaults.standard.set(url.path, forKey: Self.lastModelPathKey)
            Task { @MainActor in
                try? await controller.loadModel(from: url)
                refreshMenus()
            }
        }
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }
}
