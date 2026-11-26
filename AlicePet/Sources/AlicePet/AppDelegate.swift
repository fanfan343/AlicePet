import AppKit
import MetalKit
import VRMMetalKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var mtkView: PetMetalView!
    var controller: PetController!
    var statusItem: NSStatusItem!
    var lipSyncTuningPanel: AudioLipSyncTuningPanel?
    var ardyLabPanel: ArdyLabPanel?
    let controlServer = LocalControlServer()

    var isClickThrough: Bool = false
    /// True while the scene service is analysing an image, so the menu can say so.
    var sceneBuildInProgress: Bool = false
    var isAlwaysOnTop: Bool = true

    let lightingPresetKey = "mtoonLightingPreset"
    let outlineStyleKey = "mtoonOutlineStyle"
    let vmcPortKey = "vmcListenPort"
    let vmcAppliesHipsPositionKey = "vmcAppliesHipsPosition"
    let motionBodyKey = "localMotionEnableBody"
    let motionHandsKey = "localMotionEnableHands"
    let motionMirrorKey = "localMotionMirrorMode"
    let motionPoseTierKey = "localMotionPoseTier"
    let motionCameraUniqueIDKey = "localMotionCameraUniqueID"
    nonisolated static let lastModelPathKey = "lastOpenedVRMModelPath"

    var preferredVMCPort: UInt16 {
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

    let initialModelURL: URL

    init(modelURL: URL) {
        self.initialModelURL = modelURL
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        AppLogger.shared.info("App", "AlicePet launching (pid=\(ProcessInfo.processInfo.processIdentifier))")

        do {
            controller = try PetController(sampleCount: 4)
            restoreSavedPreferences()
        } catch {
            AppLogger.shared.error("App", "Failed to initialize Metal/VRMMetalKit: \(error)")
            NSApp.terminate(nil)
            return
        }

        setupMainWindowAndMetalView()
        bindControllerCallbacks()
        setupStatusBar()
        controller.startTimers()
        let controlPort = UInt16(ProcessInfo.processInfo.environment["ALICEPET_CONTROL_PORT"] ?? "") ?? LocalControlServer.defaultPort
        controlServer.start(port: controlPort, controller: controller, appDelegate: self)

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

    private func restoreSavedPreferences() {
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
        if let savedCamUID = defaults.string(forKey: motionCameraUniqueIDKey), !savedCamUID.isEmpty {
            controller.localMotionService.selectedCameraUniqueID = savedCamUID
        }
    }

    private func setupMainWindowAndMetalView() {
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
            guard let self, !self.controller.isPictureWorldActive else { return }
            self.controller.camera.orbit(deltaX: dx, deltaY: dy)
        }
        mtkView.onPanDrag = { [weak self] dx, dy in
            guard let self else { return }
            if self.controller.isPictureWorldActive {
                self.controller.panPictureWorldCanvas(deltaX: dx, deltaY: dy)
            } else {
                self.controller.camera.pan(deltaX: dx, deltaY: dy)
            }
        }
        mtkView.onZoom = { [weak self] dy in
            guard let self else { return }
            if self.controller.isPictureWorldActive {
                self.controller.zoomPictureWorldCanvas(deltaY: dy)
            } else {
                self.controller.camera.zoom(deltaY: dy)
            }
        }
        mtkView.onDoubleClick = { [weak self] in
            guard let self else { return }
            if self.controller.isPictureWorldActive {
                self.controller.resetPictureWorldViewAndAvatar()
            } else {
                self.controller.camera.reset()
            }
        }
        mtkView.contextMenuProvider = { [weak self] in
            self?.buildMenu() ?? NSMenu()
        }

        controller.hostView = mtkView
        window.contentView = mtkView
        window.makeKeyAndOrderFront(nil)
    }

    private func bindControllerCallbacks() {
        controller.onPictureWorldChanged = { [weak self] in
            self?.syncWindowForPictureWorld()
            self?.refreshMenus()
        }
        controller.onVMCStatusChanged = { [weak self] in
            guard let self else { return }
            self.updateStatusBarTitle()
            self.lipSyncTuningPanel?.setPlayingState(
                self.controller.localAudioFaceActive,
                audioName: self.controller.localAudioFaceService.activeAudioName
            )
            self.ardyLabPanel?.setRunningState(
                self.controller.localArdyActive,
                status: self.controller.localArdyService.lastStatusLine
            )
            self.refreshMenus()
        }
        controller.localArdyService.onStatusLine = { [weak self] line in
            guard let self else { return }
            self.ardyLabPanel?.setRunningState(self.controller.localArdyActive, status: line)
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
    }

    private func setupStatusBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "AlicePet"
        statusItem.menu = buildMenu()
    }

    func updateStatusBarTitle() {
        var badges: [String] = []
        if controller.localArdyActive {
            badges.append(controller.animationMixer.ardyChannel.hasFreshFrame ? "💃" : "💃(生成中)")
        } else if controller.isVRMAActive {
            badges.append("🎬")
        } else if controller.localMotionActive {
            badges.append(controller.vmcReceiving ? "🎭" : "🎭(启动中)")
        } else if controller.vmcActive {
            badges.append(controller.vmcReceiving ? "📡" : "📡(等待)")
        }
        if controller.localAudioFaceActive {
            badges.append(controller.animationMixer.audioFaceChannel.hasFreshFrame ? "🗣️" : "🗣️(推理中)")
        }
        statusItem.button?.title = badges.isEmpty ? "AlicePet" : "AlicePet · " + badges.joined(separator: "")
    }

    func refreshMenus() {
        statusItem.menu = buildMenu()
    }

    func syncWindowForPictureWorld() {
        guard let window, let mtkView else { return }
        if controller.isPictureWorldActive {
            let isTransparent = controller.pictureWorldRenderer.isEmptyTransparentMode
            let targetSize = isTransparent ? NSSize(width: 820, height: 520) : NSSize(width: 820, height: 480)
            var frame = window.frame
            frame.origin.x -= (targetSize.width - frame.size.width) * 0.5
            frame.size = targetSize
            window.setFrame(frame, display: true, animate: true)
            window.isOpaque = false
            window.backgroundColor = .clear
            window.hasShadow = !isTransparent
            mtkView.layer?.cornerRadius = isTransparent ? 0 : 14
            mtkView.layer?.masksToBounds = !isTransparent
        } else {
            let targetSize = NSSize(width: 460, height: 680)
            var frame = window.frame
            frame.origin.x += (frame.size.width - targetSize.width) * 0.5
            frame.size = targetSize
            window.setFrame(frame, display: true, animate: true)
            window.hasShadow = false
            mtkView.layer?.cornerRadius = 0
            mtkView.layer?.masksToBounds = false
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        controlServer.stop()
        controller?.stopAllTrackingSync()
    }

    @objc func quitApp() {
        controlServer.stop()
        controller.stopAllTrackingSync()
        NSApp.terminate(nil)
    }
}
