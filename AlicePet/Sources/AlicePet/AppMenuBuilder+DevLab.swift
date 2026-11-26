import AppKit

extension AppDelegate {
    // MARK: - 5. 开发与测试 (Lab)

    func buildDevAndLabRootItem() -> NSMenuItem {
        let labMenu = NSMenu()

        // 5a. Motion & Face Capture (MediaPipe / External VMC)
        labMenu.addItem(buildCaptureSubmenuItem())
        labMenu.addItem(.separator())

        // 5b. ARDY & Audio LipSync panels
        let openArdyItem = makeToggleItem(
            controller.localArdyActive ? "ARDY 文本动作面板… (运行中)" : "ARDY 文本动作面板… (20fps)",
            action: #selector(openLabArdyPanel),
            isOn: controller.localArdyActive
        )
        labMenu.addItem(openArdyItem)

        let openTuningItem = NSMenuItem(title: "口型与表情实时调参面板…", action: #selector(openLabLipSyncTuningPanel), keyEquivalent: "")
        openTuningItem.target = self
        labMenu.addItem(openTuningItem)

        let audioTestTitle = controller.localAudioFaceActive
            ? "停止音频对口型测试 (\(controller.localAudioFaceService.activeAudioName ?? "播放中"))"
            : "播放 10s 语音对口型测试"
        labMenu.addItem(makeToggleItem(audioTestTitle, action: #selector(toggleLabAudioFaceTest), isOn: controller.localAudioFaceActive))

        let customAudioItem = NSMenuItem(title: "选择任意音频文件对口型测试…", action: #selector(pickCustomAudioForLabTest), keyEquivalent: "")
        customAudioItem.target = self
        labMenu.addItem(customAudioItem)

        labMenu.addItem(.separator())
        let mixerStatus = "混合器：Body=\(controller.animationMixer.activeBodySource.displayLabel) · Face=\(controller.animationMixer.activeFaceSource.rawValue) · Head叠加"
        let labInfoItem = NSMenuItem(title: mixerStatus, action: nil, keyEquivalent: "")
        labInfoItem.isEnabled = false
        labMenu.addItem(labInfoItem)

        let logItem = NSMenuItem(title: "打开日志文件夹", action: #selector(openLogFolder), keyEquivalent: "l")
        logItem.target = self
        labMenu.addItem(logItem)

        let anyDevRunning = controller.localAudioFaceActive
            || controller.localArdyActive
            || controller.localMotionActive
            || controller.vmcActive
        return makeSubmenuRoot(anyDevRunning ? "开发与测试 (Lab · 运行中)" : "开发与测试 (Lab)", submenu: labMenu)
    }

    private func buildCaptureSubmenuItem() -> NSMenuItem {
        let captureMenu = NSMenu()
        let vmcPort = controller.vmcActive ? controller.vmcController.port : preferredVMCPort

        let localToggleTitle = controller.localMotionActive
            ? "关闭本机 MediaPipe 精捕（立即释放摄像头）"
            : "开启本机 MediaPipe 精捕（52表情 + 上半身 + 十指）"
        captureMenu.addItem(makeToggleItem(localToggleTitle, action: #selector(toggleLocalMotionCapture), isOn: controller.localMotionActive))
        captureMenu.addItem(makeToggleItem("联动上半身与双臂", action: #selector(toggleLocalMotionBody), isOn: controller.localMotionService.enableBody))
        captureMenu.addItem(makeToggleItem("联动双手与十指 (30 关节)", action: #selector(toggleLocalMotionHands), isOn: controller.localMotionService.enableHands))
        captureMenu.addItem(makeToggleItem("镜像左右动作与表情 (Mirror)", action: #selector(toggleLocalMotionMirror), isOn: controller.localMotionService.mirrorMode))

        let deviceMenu = NSMenu()
        let cameras = LocalMotionService.availableCameras()
        let resolvedIdx = controller.localMotionService.resolvedCameraIndex()
        if cameras.isEmpty {
            let emptyItem = NSMenuItem(title: "未检测到摄像头（默认索引 0）", action: nil, keyEquivalent: "")
            emptyItem.isEnabled = false
            deviceMenu.addItem(emptyItem)
        } else {
            for cam in cameras {
                let item = NSMenuItem(title: "[\(cam.index)] \(cam.displayTitle)", action: #selector(selectLocalMotionCamera(_:)), keyEquivalent: "")
                item.target = self
                item.representedObject = cam.uniqueID
                item.state = (resolvedIdx == cam.index) ? .on : .off
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
        captureMenu.addItem(makeSubmenuRoot("摄像头与精度设置", submenu: deviceMenu))

        captureMenu.addItem(.separator())

        let extActive = controller.vmcActive && !controller.localMotionActive
        let extVMCTitle = extActive
            ? "关闭外部 VMC / OSC 接收 (UDP \(vmcPort))"
            : "仅开启外部 VMC / OSC 接收 (UDP \(vmcPort))"
        captureMenu.addItem(makeToggleItem(extVMCTitle, action: #selector(toggleExternalVMCReceiver), isOn: extActive))

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
        captureMenu.addItem(makeSubmenuRoot("VMC 监听端口 (\(vmcPort))", submenu: portMenu))
        captureMenu.addItem(makeToggleItem(
            "应用 Hips 根位移（关闭可防跑出窗口）",
            action: #selector(toggleVMCAppliesHipsPosition),
            isOn: controller.vmcController.appliesHipsPosition
        ))

        let rootTitle: String
        if controller.localMotionActive {
            rootTitle = controller.vmcReceiving ? "动作与面部捕捉 (本机精捕运行中)" : "动作与面部捕捉 (本机精捕启动中…)"
        } else if controller.vmcReceiving {
            rootTitle = "动作与面部捕捉 (VMC 接收中 · \(vmcPort))"
        } else if controller.vmcActive {
            rootTitle = "动作与面部捕捉 (VMC 监听中 · \(vmcPort))"
        } else {
            rootTitle = "动作与面部捕捉 (MediaPipe / VMC)"
        }
        return makeSubmenuRoot(rootTitle, submenu: captureMenu)
    }
}
