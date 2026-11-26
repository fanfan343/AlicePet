import AppKit
import UniformTypeIdentifiers

extension AppDelegate {
    // MARK: - Motion Capture & VMC Actions

    @objc func toggleLocalMotionCapture() {
        if controller.localMotionActive {
            controller.stopLocalMotionCapture()
        } else {
            _ = controller.startLocalMotionCapture(port: preferredVMCPort)
        }
        updateStatusBarTitle()
        refreshMenus()
    }

    @objc func toggleExternalVMCReceiver() {
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

    @objc func toggleLocalMotionBody() {
        controller.localMotionService.enableBody.toggle()
        UserDefaults.standard.set(controller.localMotionService.enableBody, forKey: motionBodyKey)
        restartLocalMotionIfRunning()
        refreshMenus()
    }

    @objc func toggleLocalMotionHands() {
        controller.localMotionService.enableHands.toggle()
        UserDefaults.standard.set(controller.localMotionService.enableHands, forKey: motionHandsKey)
        restartLocalMotionIfRunning()
        refreshMenus()
    }

    @objc func toggleLocalMotionMirror() {
        controller.localMotionService.mirrorMode.toggle()
        UserDefaults.standard.set(controller.localMotionService.mirrorMode, forKey: motionMirrorKey)
        restartLocalMotionIfRunning()
        refreshMenus()
    }

    @objc func selectLocalMotionCamera(_ sender: NSMenuItem) {
        guard let uid = sender.representedObject as? String else { return }
        controller.localMotionService.selectedCameraUniqueID = uid
        UserDefaults.standard.set(uid, forKey: motionCameraUniqueIDKey)
        restartLocalMotionIfRunning()
        refreshMenus()
    }

    @objc func selectLocalMotionPoseTier(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let tier = MotionPoseModelTier(rawValue: raw) else { return }
        controller.localMotionService.poseModel = tier
        UserDefaults.standard.set(tier.rawValue, forKey: motionPoseTierKey)
        restartLocalMotionIfRunning()
        refreshMenus()
    }

    @objc func selectVMCPort(_ sender: NSMenuItem) {
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

    @objc func copyLocalVMCAddress(_ sender: NSMenuItem) {
        guard let ip = sender.representedObject as? String else { return }
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(ip, forType: .string)
        AppLogger.shared.info("VMC", "Copied local IPv4 '\(ip)' to clipboard")
    }

    @objc func toggleVMCAppliesHipsPosition() {
        controller.vmcController.appliesHipsPosition.toggle()
        let enabled = controller.vmcController.appliesHipsPosition
        UserDefaults.standard.set(enabled, forKey: vmcAppliesHipsPositionKey)
        AppLogger.shared.info("VMC", "appliesHipsPosition=\(enabled)")
        refreshMenus()
    }

    // MARK: - Dev / Lab Panels & Logs

    @objc func openLogFolder() {
        NSWorkspace.shared.activateFileViewerSelecting([AppLogger.shared.logFileURL])
    }

    @objc func openLabArdyPanel() {
        if ardyLabPanel == nil {
            let panel = ArdyLabPanel(
                initialPrompt: controller.localArdyService.currentPrompt,
                initialCfg: controller.localArdyService.cfgWeight
            )
            panel.onToggleStartStop = { [weak self] in
                guard let self else { return }
                if self.controller.localArdyActive {
                    self.controller.stopArdyMotionTest()
                } else {
                    _ = self.controller.startArdyMotionTest()
                }
                self.updateStatusBarTitle()
                self.ardyLabPanel?.setRunningState(
                    self.controller.localArdyActive,
                    status: self.controller.localArdyService.lastStatusLine
                )
                self.refreshMenus()
            }
            panel.onSendPrompt = { [weak self] prompt, cfg in
                guard let self else { return }
                if self.controller.sceneActionController.phase != .idle {
                    self.controller.standUpFromSceneSeat(stopArdy: false)
                }
                self.controller.localArdyService.sendLivePrompt(prompt, cfg: cfg)
                if !self.controller.localArdyActive {
                    _ = self.controller.startArdyMotionTest()
                    self.updateStatusBarTitle()
                    self.refreshMenus()
                }
                self.ardyLabPanel?.setRunningState(
                    self.controller.localArdyActive,
                    status: "已下发动作指令：\(prompt)"
                )
            }
            ardyLabPanel = panel
        }
        ardyLabPanel?.setRunningState(
            controller.localArdyActive,
            status: controller.localArdyService.lastStatusLine
        )
        ardyLabPanel?.show()
    }

    @objc func openLabLipSyncTuningPanel() {
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

    @objc func toggleLabAudioFaceTest() {
        if controller.localAudioFaceActive {
            controller.stopAudioFaceTest()
        } else {
            _ = controller.startAudioFaceTest(audioURL: nil)
        }
        updateStatusBarTitle()
        lipSyncTuningPanel?.setPlayingState(
            controller.localAudioFaceActive,
            audioName: controller.localAudioFaceService.activeAudioName
        )
        refreshMenus()
    }

    @objc func pickCustomAudioForLabTest() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .wav, .mpeg4Audio, .mp3]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        if let dir = LocalAudioFaceService.resolveServiceDirectory() {
            panel.directoryURL = dir.appendingPathComponent("samples")
        }
        if panel.runModal() == .OK, let url = panel.url {
            _ = controller.startAudioFaceTest(audioURL: url, maxSeconds: 30.0)
            updateStatusBarTitle()
            lipSyncTuningPanel?.setPlayingState(
                controller.localAudioFaceActive,
                audioName: controller.localAudioFaceService.activeAudioName
            )
            refreshMenus()
        }
    }
}
