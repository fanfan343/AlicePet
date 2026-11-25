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
    private var lipSyncTuningPanel: AudioLipSyncTuningPanel?

    private var isClickThrough: Bool = false
    private var isAlwaysOnTop: Bool = true
    private let lightingPresetKey = "mtoonLightingPreset"
    private let outlineStyleKey = "mtoonOutlineStyle"
    private let vmcPortKey = "vmcListenPort"
    private let vmcAppliesHipsPositionKey = "vmcAppliesHipsPosition"
    private let motionBodyKey = "localMotionEnableBody"
    private let motionHandsKey = "localMotionEnableHands"
    private let motionMirrorKey = "localMotionMirrorMode"
    private let motionPoseTierKey = "localMotionPoseTier"
    private let motionCameraIndexKey = "localMotionCameraIndex"
    nonisolated static let lastModelPathKey = "lastOpenedVRMModelPath"

    private var preferredVMCPort: UInt16 {
        get {
            let saved = UserDefaults.standard.integer(forKey: vmcPortKey)
            if saved > 0, saved <= Int(UInt16.max) {
                return UInt16(saved)
            }
            return VMCTrackingController.defaultPort
        }
        set {
            UserDefaults.standard.set(Int(newValue), forKey: vmcPortKey)
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
            let defaults = UserDefaults.standard
            if let savedLight = defaults.string(forKey: lightingPresetKey),
               let preset = LightingPreset(rawValue: savedLight) {
                controller.setLightingPreset(preset)
            }
            if let savedOutline = defaults.string(forKey: outlineStyleKey),
               let style = OutlineStyle(rawValue: savedOutline) {
                controller.setOutlineStyle(style)
            }
            if defaults.object(forKey: vmcAppliesHipsPositionKey) != nil {
                controller.vmcController.appliesHipsPosition = defaults.bool(forKey: vmcAppliesHipsPositionKey)
            }
            if defaults.object(forKey: motionBodyKey) != nil {
                controller.localMotionService.enableBody = defaults.bool(forKey: motionBodyKey)
            }
            if defaults.object(forKey: motionHandsKey) != nil {
                controller.localMotionService.enableHands = defaults.bool(forKey: motionHandsKey)
            }
            if defaults.object(forKey: motionMirrorKey) != nil {
                controller.localMotionService.mirrorMode = defaults.bool(forKey: motionMirrorKey)
            }
            if let tierRaw = defaults.string(forKey: motionPoseTierKey),
               let tier = MotionPoseModelTier(rawValue: tierRaw) {
                controller.localMotionService.poseModel = tier
            }
            if defaults.object(forKey: motionCameraIndexKey) != nil {
                controller.localMotionService.cameraIndex = max(0, defaults.integer(forKey: motionCameraIndexKey))
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

        controller.onVMCStatusChanged = { [weak self] in
            guard let self else { return }
            self.updateStatusBarTitle()
            self.lipSyncTuningPanel?.setPlayingState(
                self.controller.localAudioFaceActive,
                audioName: self.controller.localAudioFaceService.activeAudioName
            )
            self.refreshMenus()
        }
        controller.onVMCError = { [weak self] message in
            self?.updateStatusBarTitle()
            self?.refreshMenus()
            let alert = NSAlert()
            alert.messageText = "动作与面部捕捉提醒"
            alert.informativeText = message
            alert.alertStyle = .warning
            alert.runModal()
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
                let cameras = LocalMotionService.availableCameras()
                AppLogger.shared.info("Motion", "Detected \(cameras.count) camera(s): \(cameras.map(\.name).joined(separator: ", "))")
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

    private func updateStatusBarTitle() {
        if controller.localAudioFaceActive {
            statusItem.button?.title = controller.vmcReceiving ? "AlicePet · 🗣️" : "AlicePet · 🗣️(推理中)"
        } else if controller.localMotionActive {
            statusItem.button?.title = controller.vmcReceiving ? "AlicePet · 🎭" : "AlicePet · 🎭(启动中)"
        } else if controller.vmcActive {
            statusItem.button?.title = controller.vmcReceiving ? "AlicePet · 📡" : "AlicePet · 📡(等待)"
        } else {
            statusItem.button?.title = "AlicePet"
        }
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

        // 1. Unified Motion & Face Capture Submenu (Local MediaPipe Service + External VMC/OSC)
        let captureMenu = NSMenu()
        let vmcPort = controller.vmcActive ? controller.vmcController.port : preferredVMCPort

        let localToggleTitle = controller.localMotionActive
            ? "关闭本机 MediaPipe 精捕（立即释放摄像头）"
            : "开启本机 MediaPipe 精捕（52表情 + 上半身 + 十指）"
        let localToggleItem = NSMenuItem(title: localToggleTitle, action: #selector(toggleLocalMotionCapture), keyEquivalent: "")
        localToggleItem.target = self
        localToggleItem.state = controller.localMotionActive ? .on : .off
        captureMenu.addItem(localToggleItem)

        let bodyToggle = NSMenuItem(title: "联动上半身与双臂", action: #selector(toggleLocalMotionBody), keyEquivalent: "")
        bodyToggle.target = self
        bodyToggle.state = controller.localMotionService.enableBody ? .on : .off
        captureMenu.addItem(bodyToggle)

        let handsToggle = NSMenuItem(title: "联动双手与十指 (30 关节)", action: #selector(toggleLocalMotionHands), keyEquivalent: "")
        handsToggle.target = self
        handsToggle.state = controller.localMotionService.enableHands ? .on : .off
        captureMenu.addItem(handsToggle)

        let mirrorToggle = NSMenuItem(title: "镜像左右动作与表情 (Mirror)", action: #selector(toggleLocalMotionMirror), keyEquivalent: "")
        mirrorToggle.target = self
        mirrorToggle.state = controller.localMotionService.mirrorMode ? .on : .off
        captureMenu.addItem(mirrorToggle)

        // Camera & Pose model quality nested submenu
        let deviceMenu = NSMenu()
        let cameras = LocalMotionService.availableCameras()
        if cameras.isEmpty {
            let emptyItem = NSMenuItem(title: "未检测到摄像头（默认索引 0）", action: nil, keyEquivalent: "")
            emptyItem.isEnabled = false
            deviceMenu.addItem(emptyItem)
        } else {
            for cam in cameras {
                let item = NSMenuItem(title: "[\(cam.index)] \(cam.displayTitle)", action: #selector(selectLocalMotionCamera(_:)), keyEquivalent: "")
                item.target = self
                item.representedObject = cam.index
                item.state = (controller.localMotionService.cameraIndex == cam.index) ? .on : .off
                deviceMenu.addItem(item)
            }
        }
        deviceMenu.addItem(.separator())
        for tier in MotionPoseModelTier.allCases {
            let item = NSMenuItem(title: tier.displayTitle, action: #selector(selectLocalMotionPoseTier(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = tier.rawValue
            item.state = (controller.localMotionService.poseModel == tier) ? .on : .off
            deviceMenu.addItem(item)
        }
        let deviceRoot = NSMenuItem(title: "摄像头与精度设置", action: nil, keyEquivalent: "")
        deviceRoot.submenu = deviceMenu
        captureMenu.addItem(deviceRoot)

        captureMenu.addItem(.separator())

        // External VMC/OSC Receiver section
        let extVMCTitle = (controller.vmcActive && !controller.localMotionActive)
            ? "关闭外部 VMC / OSC 接收 (UDP \(vmcPort))"
            : "仅开启外部 VMC / OSC 接收 (UDP \(vmcPort))"
        let extVMCItem = NSMenuItem(title: extVMCTitle, action: #selector(toggleExternalVMCReceiver), keyEquivalent: "")
        extVMCItem.target = self
        extVMCItem.state = (controller.vmcActive && !controller.localMotionActive) ? .on : .off
        captureMenu.addItem(extVMCItem)

        let vmcStatusText: String
        if !controller.vmcActive {
            vmcStatusText = "状态：未开启"
        } else if controller.vmcReceiving {
            let modeTag = controller.localMotionActive ? "本机精捕" : "外部 VMC"
            vmcStatusText = "状态：\(modeTag)接收中（\(controller.vmcController.receivedBoneCount) 骨骼 / \(controller.vmcController.receivedBlendShapeCount) 表情 · \(controller.vmcController.packetCount) 包）"
        } else if controller.localMotionActive {
            vmcStatusText = "状态：本机 MediaPipe 模型加载中…"
        } else {
            vmcStatusText = "状态：监听中 (UDP \(vmcPort))，等待发包…"
        }
        let vmcStatusItem = NSMenuItem(title: vmcStatusText, action: nil, keyEquivalent: "")
        vmcStatusItem.isEnabled = false
        captureMenu.addItem(vmcStatusItem)

        let localIP = VMCTrackingController.primaryLocalIPv4() ?? "127.0.0.1"
        let copyIPItem = NSMenuItem(
            title: "复制本机 IP：\(localIP)（端口 \(vmcPort)）",
            action: #selector(copyLocalVMCAddress(_:)),
            keyEquivalent: ""
        )
        copyIPItem.target = self
        copyIPItem.representedObject = localIP
        captureMenu.addItem(copyIPItem)

        let portMenu = NSMenu()
        let portChoices: [(String, UInt16)] = [
            ("39539（默认 · Waidayo / XR Animator）", VMCTrackingController.defaultPort),
            ("39540（Performer / 备用）", VMCTrackingController.secondaryPort),
        ]
        for (title, portNum) in portChoices {
            let item = NSMenuItem(title: title, action: #selector(selectVMCPort(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = Int(portNum)
            item.state = (vmcPort == portNum) ? .on : .off
            portMenu.addItem(item)
        }
        let portRoot = NSMenuItem(title: "VMC 监听端口 (\(vmcPort))", action: nil, keyEquivalent: "")
        portRoot.submenu = portMenu
        captureMenu.addItem(portRoot)

        let hipsPosItem = NSMenuItem(title: "应用 Hips 根位移（关闭可防跑出窗口）", action: #selector(toggleVMCAppliesHipsPosition), keyEquivalent: "")
        hipsPosItem.target = self
        hipsPosItem.state = controller.vmcController.appliesHipsPosition ? .on : .off
        captureMenu.addItem(hipsPosItem)

        let captureRootTitle: String
        if controller.localMotionActive {
            captureRootTitle = controller.vmcReceiving ? "动作与面部捕捉 (本机精捕运行中)" : "动作与面部捕捉 (本机精捕启动中…)"
        } else if controller.vmcReceiving {
            captureRootTitle = "动作与面部捕捉 (VMC 接收中 · \(vmcPort))"
        } else if controller.vmcActive {
            captureRootTitle = "动作与面部捕捉 (VMC 监听中 · \(vmcPort))"
        } else {
            captureRootTitle = "动作与面部捕捉 (VMC)"
        }
        let captureRoot = NSMenuItem(title: captureRootTitle, action: nil, keyEquivalent: "")
        captureRoot.submenu = captureMenu
        menu.addItem(captureRoot)

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

        // Experimental / Lab Features Submenu (tests live here until promoted to main menu)
        let labMenu = NSMenu()
        let openTuningItem = NSMenuItem(title: "打开口型与表情实时调参面板…", action: #selector(openLabLipSyncTuningPanel), keyEquivalent: "")
        openTuningItem.target = self
        labMenu.addItem(openTuningItem)
        labMenu.addItem(.separator())

        let audioTestTitle = controller.localAudioFaceActive
            ? "停止音频对口型测试 (\(controller.localAudioFaceService.activeAudioName ?? "播放中"))"
            : "播放 10s VTuber 语音对口型测试 (LAM 52通道)"
        let audioTestItem = NSMenuItem(title: audioTestTitle, action: #selector(toggleLabAudioFaceTest), keyEquivalent: "")
        audioTestItem.target = self
        audioTestItem.state = controller.localAudioFaceActive ? .on : .off
        labMenu.addItem(audioTestItem)

        let customAudioItem = NSMenuItem(title: "选择任意音频文件对口型测试…", action: #selector(pickCustomAudioForLabTest), keyEquivalent: "")
        customAudioItem.target = self
        labMenu.addItem(customAudioItem)

        labMenu.addItem(.separator())
        let labInfoItem = NSMenuItem(title: "引擎：LAM_Audio2Expression (INT8 ONNX · 52 通道)", action: nil, keyEquivalent: "")
        labInfoItem.isEnabled = false
        labMenu.addItem(labInfoItem)

        let labRootTitle = controller.localAudioFaceActive ? "测试功能 (Lab · 运行中)" : "测试功能 (Lab)"
        let labRoot = NSMenuItem(title: labRootTitle, action: nil, keyEquivalent: "")
        labRoot.submenu = labMenu
        menu.addItem(labRoot)

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

    @objc private func toggleLocalMotionCapture() {
        if controller.localMotionActive {
            controller.stopLocalMotionCapture()
        } else {
            _ = controller.startLocalMotionCapture(port: preferredVMCPort)
        }
        updateStatusBarTitle()
        refreshMenus()
    }

    @objc private func toggleExternalVMCReceiver() {
        if controller.vmcActive && !controller.localMotionActive {
            controller.stopVMC()
        } else {
            controller.localMotionService.stop()
            _ = controller.startVMC(port: preferredVMCPort)
        }
        updateStatusBarTitle()
        refreshMenus()
    }

    private func restartLocalMotionIfRunning() {
        if controller.localMotionActive {
            _ = controller.startLocalMotionCapture(port: preferredVMCPort)
        }
    }

    @objc private func toggleLocalMotionBody() {
        controller.localMotionService.enableBody.toggle()
        UserDefaults.standard.set(controller.localMotionService.enableBody, forKey: motionBodyKey)
        restartLocalMotionIfRunning()
        refreshMenus()
    }

    @objc private func toggleLocalMotionHands() {
        controller.localMotionService.enableHands.toggle()
        UserDefaults.standard.set(controller.localMotionService.enableHands, forKey: motionHandsKey)
        restartLocalMotionIfRunning()
        refreshMenus()
    }

    @objc private func toggleLocalMotionMirror() {
        controller.localMotionService.mirrorMode.toggle()
        UserDefaults.standard.set(controller.localMotionService.mirrorMode, forKey: motionMirrorKey)
        restartLocalMotionIfRunning()
        refreshMenus()
    }

    @objc private func selectLocalMotionCamera(_ sender: NSMenuItem) {
        guard let idx = sender.representedObject as? Int else { return }
        controller.localMotionService.cameraIndex = max(0, idx)
        UserDefaults.standard.set(controller.localMotionService.cameraIndex, forKey: motionCameraIndexKey)
        restartLocalMotionIfRunning()
        refreshMenus()
    }

    @objc private func selectLocalMotionPoseTier(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let tier = MotionPoseModelTier(rawValue: raw) else { return }
        controller.localMotionService.poseModel = tier
        UserDefaults.standard.set(tier.rawValue, forKey: motionPoseTierKey)
        restartLocalMotionIfRunning()
        refreshMenus()
    }

    @objc private func selectVMCPort(_ sender: NSMenuItem) {
        guard let portInt = sender.representedObject as? Int,
              portInt > 0, portInt <= Int(UInt16.max) else { return }
        let newPort = UInt16(portInt)
        preferredVMCPort = newPort
        if controller.localMotionActive {
            _ = controller.startLocalMotionCapture(port: newPort)
        } else if controller.vmcActive {
            _ = controller.startVMC(port: newPort)
        }
        updateStatusBarTitle()
        refreshMenus()
    }

    @objc private func copyLocalVMCAddress(_ sender: NSMenuItem) {
        guard let ip = sender.representedObject as? String else { return }
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(ip, forType: .string)
        AppLogger.shared.info("VMC", "Copied local IPv4 '\(ip)' to clipboard")
    }

    @objc private func toggleVMCAppliesHipsPosition() {
        controller.vmcController.appliesHipsPosition.toggle()
        let enabled = controller.vmcController.appliesHipsPosition
        UserDefaults.standard.set(enabled, forKey: vmcAppliesHipsPositionKey)
        AppLogger.shared.info("VMC", "appliesHipsPosition=\(enabled)")
        refreshMenus()
    }

    @objc private func openLogFolder() {
        NSWorkspace.shared.activateFileViewerSelecting([AppLogger.shared.logFileURL])
    }

    @objc private func openLabLipSyncTuningPanel() {
        if lipSyncTuningPanel == nil {
            let panel = AudioLipSyncTuningPanel(initialParams: controller.localAudioFaceService.params)
            panel.onParamsChanged = { [weak self] newParams in
                self?.controller.localAudioFaceService.pushLiveParams(newParams)
            }
            panel.onTogglePlay = { [weak self] in
                self?.toggleLabAudioFaceTest()
            }
            panel.onPickAudio = { [weak self] in
                self?.pickCustomAudioForLabTest()
            }
            lipSyncTuningPanel = panel
        }
        lipSyncTuningPanel?.setPlayingState(
            controller.localAudioFaceActive,
            audioName: controller.localAudioFaceService.activeAudioName
        )
        lipSyncTuningPanel?.show()
    }

    @objc private func toggleLabAudioFaceTest() {
        if controller.localAudioFaceActive {
            controller.stopAudioFaceTest()
        } else {
            _ = controller.startAudioFaceTest(audioURL: nil, port: preferredVMCPort)
        }
        updateStatusBarTitle()
        lipSyncTuningPanel?.setPlayingState(
            controller.localAudioFaceActive,
            audioName: controller.localAudioFaceService.activeAudioName
        )
        refreshMenus()
    }

    @objc private func pickCustomAudioForLabTest() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .wav, .mpeg4Audio, .mp3]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        if let dir = LocalAudioFaceService.resolveServiceDirectory() {
            panel.directoryURL = dir.appendingPathComponent("samples")
        }
        if panel.runModal() == .OK, let url = panel.url {
            _ = controller.startAudioFaceTest(audioURL: url, port: preferredVMCPort, maxSeconds: 30.0)
            updateStatusBarTitle()
            lipSyncTuningPanel?.setPlayingState(
                controller.localAudioFaceActive,
                audioName: controller.localAudioFaceService.activeAudioName
            )
            refreshMenus()
        }
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

    func applicationWillTerminate(_ notification: Notification) {
        controller?.stopAllTrackingSync()
    }

    @objc private func quitApp() {
        controller.stopAllTrackingSync()
        NSApp.terminate(nil)
    }
}
