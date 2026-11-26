import AppKit
import UniformTypeIdentifiers
import VRMMetalKit

extension AppDelegate {
    // MARK: - Pose, Props, Hand Gestures, Outfits & LookDev

    @objc func selectPoseMode(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let mode = PetPoseMode(rawValue: raw) else { return }
        controller.setPoseMode(mode)
        refreshMenus()
    }

    @objc func selectPropOverride(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String else { return }
        if raw == "__auto__" {
            controller.manualPropOverride = nil
        } else if let mode = PropSceneRenderer.ActivePropMode(rawValue: raw) {
            controller.manualPropOverride = mode
        }
        refreshMenus()
    }

    @objc func selectBothHandGesture(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let preset = HandGesturePreset(rawValue: raw) else { return }
        controller.setHandGesture(preset, side: .both)
        refreshMenus()
    }

    @objc func selectLeftHandGesture(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let preset = HandGesturePreset(rawValue: raw) else { return }
        controller.setHandGesture(preset, side: .left)
        refreshMenus()
    }

    @objc func selectRightHandGesture(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let preset = HandGesturePreset(rawValue: raw) else { return }
        controller.setHandGesture(preset, side: .right)
        refreshMenus()
    }

    @objc func toggleOutfitDress() {
        controller.outfitState.showDress.toggle()
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc func toggleOutfitPantyhose() {
        controller.outfitState.showPantyhose.toggle()
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc func toggleOutfitShoes() {
        controller.outfitState.showShoes.toggle()
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc func toggleOutfitHairBand() {
        controller.outfitState.showHairBand.toggle()
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc func toggleOutfitChestRibbon() {
        controller.outfitState.showChestRibbon.toggle()
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc func toggleOutfitHairSide() {
        controller.outfitState.showHairSide.toggle()
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc func selectHairStyle(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let style = HairStyleOption(rawValue: raw) else { return }
        controller.outfitState.hairStyle = style
        controller.applyOutfitState()
        refreshMenus()
    }

    @objc func selectLightingPreset(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let preset = LightingPreset(rawValue: raw) else { return }
        UserDefaults.standard.set(preset.rawValue, forKey: lightingPresetKey)
        controller.setLightingPreset(preset)
        refreshMenus()
    }

    @objc func selectOutlineStyle(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let style = OutlineStyle(rawValue: raw) else { return }
        UserDefaults.standard.set(style.rawValue, forKey: outlineStyleKey)
        controller.setOutlineStyle(style)
        refreshMenus()
    }

    @objc func selectPresetMood(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String else { return }
        if raw == "__neutral__" {
            controller.setPresetMood(nil)
        } else if let preset = VRMExpressionPreset(rawValue: raw) {
            controller.setPresetMood(preset)
        }
        refreshMenus()
    }

    @objc func selectCustomExpression(_ sender: NSMenuItem) {
        guard let name = sender.representedObject as? String else { return }
        controller.setCustomExpression(name, weight: 1.0)
        refreshMenus()
    }

    // MARK: - Window, Behavior & Camera

    @objc func toggleClickThrough() {
        isClickThrough.toggle()
        window.ignoresMouseEvents = isClickThrough
        refreshMenus()
    }

    @objc func toggleAlwaysOnTop() {
        isAlwaysOnTop.toggle()
        window.level = isAlwaysOnTop ? .floating : .normal
        refreshMenus()
    }

    @objc func toggleSpringBone() {
        controller.springBoneEnabled.toggle()
        refreshMenus()
    }

    @objc func toggleEyeTracking() {
        controller.eyeTrackingEnabled.toggle()
        refreshMenus()
    }

    @objc func toggleAutoBlink() {
        controller.autoBlinkEnabled.toggle()
        refreshMenus()
    }

    @objc func toggleSpeechGestures() {
        controller.animationMixer.speechGestureLayer.enabled.toggle()
        refreshMenus()
    }

    @objc func openVRMAPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "vrma") ?? .data, .data]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.prompt = "播放 .vrma 动作"
        if panel.runModal() == .OK, let url = panel.url {
            do {
                _ = try controller.playVRMA(from: url, loop: true)
                refreshMenus()
            } catch {
                AppLogger.shared.error("VRMA", "Failed to play VRMA '\(url.lastPathComponent)': \(error)")
            }
        }
    }

    @objc func stopVRMAPlayback() {
        controller.stopVRMA()
        refreshMenus()
    }

    @objc func enterEmptyTransparentPictureWorld() {
        controller.loadEmptyTransparentPictureWorld()
    }

    @objc func selectCameraViewPreset(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String,
              let preset = CameraViewPreset(rawValue: raw) else { return }
        controller.camera.apply(preset: preset)
    }

    @objc func resetCamera() {
        if controller.isPictureWorldActive {
            controller.resetPictureWorldViewAndAvatar()
        } else {
            controller.camera.reset()
        }
    }

    @objc func openModelPanel() {
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

    // MARK: - Picture-World Scene Management

    func discoverPictureWorldScenes() -> [URL] {
        let fm = FileManager.default
        let candidates = [
            Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("scenes", isDirectory: true),
            URL(fileURLWithPath: "/Users/funnyliu/Downloads/VRM项目/scenes", isDirectory: true),
        ]
        var found: [URL] = []
        var seen: Set<String> = []
        for root in candidates {
            guard let items = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey]) else { continue }
            for dir in items.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                let json = dir.appendingPathComponent("scene.json")
                if fm.fileExists(atPath: json.path), seen.insert(dir.standardizedFileURL.path).inserted {
                    found.append(dir)
                }
            }
        }
        return found
    }

    @objc func exitPictureWorldMode() {
        controller.exitPictureWorld()
    }

    @objc func walkToSceneSeat(_ sender: NSMenuItem) {
        guard let anchorId = sender.representedObject as? String else { return }
        _ = controller.walkToSceneAnchorAndSit(anchorId: anchorId)
    }

    @objc func standUpFromSceneSeat() {
        controller.standUpFromSceneSeat(stopArdy: true)
    }

    @objc func selectPictureWorldScene(_ sender: NSMenuItem) {
        guard let path = sender.representedObject as? String else { return }
        do {
            try controller.loadPictureWorld(from: URL(fileURLWithPath: path))
        } catch {
            AppLogger.shared.error("PictureWorld", "Failed to load scene: \(error)")
        }
    }

    /// Builds a new Picture-World room from one photo/illustration (MoGe-2 native depth + fitted
    /// ground surface + walkable floor grid) via `services/scene/estimate_scene.py`, then loads it.
    @objc func createPictureWorldFromImage() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.image]
        panel.prompt = "选择一张房间照片或插画"
        guard panel.runModal() == .OK, let url = panel.url else { return }

        let projectRoot = URL(fileURLWithPath: "/Users/funnyliu/Downloads/VRM项目")
        let python = projectRoot.appendingPathComponent("services/scene/.venv/bin/python")
        let script = projectRoot.appendingPathComponent("services/scene/estimate_scene.py")
        let scenesRoot = projectRoot.appendingPathComponent("scenes", isDirectory: true)

        let fm = FileManager.default
        guard fm.fileExists(atPath: python.path), fm.fileExists(atPath: script.path) else {
            presentSceneBuildFailure("找不到场景分析服务：\(python.path)")
            return
        }

        let base = url.deletingPathExtension().lastPathComponent
            .replacingOccurrences(of: " ", with: "_")
        var slug = base
        var suffix = 2
        while fm.fileExists(atPath: scenesRoot.appendingPathComponent(slug).path) {
            slug = "\(base)_\(suffix)"
            suffix += 1
        }
        let outDir = scenesRoot.appendingPathComponent(slug, isDirectory: true)

        sceneBuildInProgress = true
        refreshMenus()
        AppLogger.shared.info("PictureWorld", "Building room from '\(url.lastPathComponent)' -> scenes/\(slug)")

        DispatchQueue.global(qos: .userInitiated).async {
            let process = Process()
            process.executableURL = python
            process.arguments = [
                script.path,
                "--image", url.path,
                "--out-dir", outDir.path,
                "--name", url.deletingPathExtension().lastPathComponent,
            ]
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            var output = ""
            do {
                try process.run()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                output = String(data: data, encoding: .utf8) ?? ""
            } catch {
                DispatchQueue.main.async {
                    self.sceneBuildInProgress = false
                    self.refreshMenus()
                    self.presentSceneBuildFailure("无法启动场景分析：\(error.localizedDescription)")
                }
                return
            }

            let status = process.terminationStatus
            DispatchQueue.main.async {
                self.sceneBuildInProgress = false
                self.refreshMenus()
                if status == 0 {
                    let summary = output
                        .split(separator: "\n")
                        .filter { $0.contains("[Scene]") && !$0.contains("Loading") }
                        .suffix(4)
                        .joined(separator: "\n")
                    AppLogger.shared.info("PictureWorld", "Room built:\n\(summary)")
                    do {
                        try self.controller.loadPictureWorld(from: outDir)
                        self.syncWindowForPictureWorld()
                    } catch {
                        self.presentSceneBuildFailure("房间已生成，但加载失败：\(error)")
                    }
                } else {
                    let tail = output.split(separator: "\n").suffix(6).joined(separator: "\n")
                    self.presentSceneBuildFailure("场景分析失败（退出码 \(status)）：\n\(tail)")
                }
            }
        }
    }

    private func presentSceneBuildFailure(_ message: String) {
        AppLogger.shared.error("PictureWorld", message)
        let alert = NSAlert()
        alert.messageText = "无法生成交互房间"
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "好")
        alert.runModal()
    }

    @objc func pickPictureWorldScene() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.prompt = "选择含 scene.json 的画中世界目录"
        if panel.runModal() == .OK, let url = panel.url {
            do {
                try controller.loadPictureWorld(from: url)
            } catch {
                AppLogger.shared.error("PictureWorld", "Failed to load scene: \(error)")
            }
        }
    }
}
