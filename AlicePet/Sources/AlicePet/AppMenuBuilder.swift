import AppKit
import VRMMetalKit

extension AppDelegate {
    /// Categorized status-bar / context menu:
    /// 1. 场景与房间  2. 姿态与表情  3. 外观与模型  4. 窗口与视角  5. 开发与测试 (Lab)
    func buildMenu() -> NSMenu {
        let menu = NSMenu()

        let titleItem = NSMenuItem(title: "AlicePet · VRMMetalKit", action: nil, keyEquivalent: "")
        titleItem.isEnabled = false
        menu.addItem(titleItem)
        menu.addItem(.separator())

        // 1. Scene & Room Mode (Mode A desktop pet <-> Mode B Picture-World)
        menu.addItem(buildSceneRootItem())

        // 2. Character Pose, Props, Hand Gestures & Expressions
        menu.addItem(buildPoseAndExpressionRootItem())

        // 3. Appearance & Model (Load VRM, Outfit/Hair, MToon Lighting)
        menu.addItem(buildAppearanceRootItem())

        menu.addItem(.separator())

        // 4. High-frequency Window Toggles + Camera & Basic Switches
        menu.addItem(makeToggleItem("鼠标穿透 (Click-Through)", action: #selector(toggleClickThrough), isOn: isClickThrough))
        menu.addItem(makeToggleItem("窗口置顶", action: #selector(toggleAlwaysOnTop), isOn: isAlwaysOnTop))
        menu.addItem(buildViewAndBehaviorRootItem())

        menu.addItem(.separator())

        // 5. Unified Dev & Lab Tools (Capture/VMC, ARDY, Audio LipSync, Mixer Status, Logs)
        menu.addItem(buildDevAndLabRootItem())

        menu.addItem(.separator())

        let quitItem = NSMenuItem(title: "退出 AlicePet", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        return menu
    }

    // MARK: - 1. 场景与房间

    private func buildSceneRootItem() -> NSMenuItem {
        let sceneMenu = NSMenu()

        sceneMenu.addItem(makeToggleItem(
            "透明桌宠模式（形态 A · 0 场景显存）",
            action: #selector(exitPictureWorldMode),
            isOn: !controller.isPictureWorldActive
        ))
        sceneMenu.addItem(makeToggleItem(
            "画中模式 · 空透明房间（观察 ARDY 自由动作）",
            action: #selector(enterEmptyTransparentPictureWorld),
            isOn: controller.pictureWorldRenderer.isEmptyTransparentMode
        ))

        let discoveredScenes = discoverPictureWorldScenes()
        if !discoveredScenes.isEmpty {
            sceneMenu.addItem(.separator())
            for folder in discoveredScenes {
                let item = NSMenuItem(
                    title: "进入：\(folder.lastPathComponent)",
                    action: #selector(selectPictureWorldScene(_:)),
                    keyEquivalent: ""
                )
                item.target = self
                item.representedObject = folder.path
                item.state = (controller.pictureWorldRenderer.sceneFolderURL?.standardizedFileURL == folder.standardizedFileURL) ? .on : .off
                sceneMenu.addItem(item)
            }
        }

        sceneMenu.addItem(.separator())
        let createSceneTitle = sceneBuildInProgress
            ? "正在分析图片，生成交互房间…"
            : "从任意图片新建交互房间… (深度 + 可走网格)"
        let createSceneItem = NSMenuItem(title: createSceneTitle, action: #selector(createPictureWorldFromImage), keyEquivalent: "")
        createSceneItem.target = self
        createSceneItem.isEnabled = !sceneBuildInProgress
        sceneMenu.addItem(createSceneItem)

        let openSceneItem = NSMenuItem(title: "打开自定义画中世界目录…", action: #selector(pickPictureWorldScene), keyEquivalent: "")
        openSceneItem.target = self
        sceneMenu.addItem(openSceneItem)

        if controller.isPictureWorldActive {
            if let anchors = controller.pictureWorldRenderer.config?.anchors, !anchors.isEmpty {
                sceneMenu.addItem(.separator())
                for anchor in anchors where anchor.kind == "seat" {
                    let isSeatedHere = (controller.sceneActionController.seatedAnchor?.id == anchor.id)
                    let isWalkingHere = (controller.sceneActionController.activeTaskIdentifier == "walking_to:\(anchor.id)")
                    let title: String
                    if isSeatedHere {
                        title = "已就座保持待机 · \(anchor.label)"
                    } else if isWalkingHere {
                        title = "正在走去坐下 · \(anchor.label)…"
                    } else {
                        title = "走过去坐下 → \(anchor.label)"
                    }
                    let seatItem = NSMenuItem(title: title, action: #selector(walkToSceneSeat(_:)), keyEquivalent: "")
                    seatItem.target = self
                    seatItem.representedObject = anchor.id
                    seatItem.state = (isSeatedHere || isWalkingHere) ? .on : .off
                    sceneMenu.addItem(seatItem)
                }
                if controller.sceneActionController.phase != .idle {
                    let standItem = NSMenuItem(title: "起身 / 取消当前移动", action: #selector(standUpFromSceneSeat), keyEquivalent: "")
                    standItem.target = self
                    sceneMenu.addItem(standItem)
                }
            }
            sceneMenu.addItem(.separator())
            let hintItem = NSMenuItem(title: "提示：滚轮缩放、中键平移画布，双击回到原点", action: nil, keyEquivalent: "")
            hintItem.isEnabled = false
            sceneMenu.addItem(hintItem)
        }

        let title = controller.isPictureWorldActive
            ? "场景与房间 (画中世界 · \(controller.pictureWorldRenderer.config?.name ?? ""))"
            : "场景与房间 (透明桌宠)"
        return makeSubmenuRoot(title, submenu: sceneMenu)
    }

    // MARK: - 2. 姿态与表情

    private func buildPoseAndExpressionRootItem() -> NSMenuItem {
        let rootMenu = NSMenu()

        // 2a. Pose & Props
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
        rootMenu.addItem(makeSubmenuRoot("姿态与 AFK 状态 (\(controller.poseMode.displayTitle))", submenu: poseMenu))

        // 2b. Hand Gestures
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
        handMenu.addItem(makeSubmenuRoot("双手同步", submenu: bothMenu))
        handMenu.addItem(makeSubmenuRoot("左手 (\(controller.leftHandGesture.rawValue))", submenu: leftMenu))
        handMenu.addItem(makeSubmenuRoot("右手 (\(controller.rightHandGesture.rawValue))", submenu: rightMenu))
        rootMenu.addItem(makeSubmenuRoot("手势配方", submenu: handMenu))

        // 2c. Expressions
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
        rootMenu.addItem(makeSubmenuRoot("表情配方 (\(controller.recipeStore.availableFaceRecipes.count) 套)", submenu: exprMenu))

        // 2d. Standard .vrma Animation Playback
        rootMenu.addItem(.separator())
        let playVRMAItem = NSMenuItem(title: "播放 .vrma 标准动画…", action: #selector(openVRMAPanel), keyEquivalent: "")
        playVRMAItem.target = self
        rootMenu.addItem(playVRMAItem)
        if controller.isVRMAActive {
            let stopVRMAItem = NSMenuItem(
                title: "停止 .vrma 动画 (\(controller.activeVRMAName ?? "播放中"))",
                action: #selector(stopVRMAPlayback),
                keyEquivalent: ""
            )
            stopVRMAItem.target = self
            stopVRMAItem.state = .on
            rootMenu.addItem(stopVRMAItem)
        }

        return makeSubmenuRoot("姿态与表情", submenu: rootMenu)
    }

    // MARK: - 3. 外观与模型

    private func buildAppearanceRootItem() -> NSMenuItem {
        let appearanceMenu = NSMenu()

        let openItem = NSMenuItem(title: "更换 VRM 模型…", action: #selector(openModelPanel), keyEquivalent: "o")
        openItem.target = self
        appearanceMenu.addItem(openItem)

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
            appearanceMenu.addItem(makeSubmenuRoot("换装与发型", submenu: outfitMenu))
        }

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
        appearanceMenu.addItem(makeSubmenuRoot("光影与描边 (MToon)", submenu: lookDevMenu))

        return makeSubmenuRoot("外观与模型", submenu: appearanceMenu)
    }

    // MARK: - 4. 视角与基础开关

    private func buildViewAndBehaviorRootItem() -> NSMenuItem {
        let menu = NSMenu()
        for preset in CameraViewPreset.allCases {
            let item = NSMenuItem(title: preset.displayTitle, action: #selector(selectCameraViewPreset(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = preset.rawValue
            menu.addItem(item)
        }
        let resetCamItem = NSMenuItem(title: "重置视角与平移 (或双击角色)", action: #selector(resetCamera), keyEquivalent: "")
        resetCamItem.target = self
        menu.addItem(resetCamItem)

        menu.addItem(.separator())
        menu.addItem(makeToggleItem("SpringBone 物理", action: #selector(toggleSpringBone), isOn: controller.springBoneEnabled))
        menu.addItem(makeToggleItem("视线跟随鼠标", action: #selector(toggleEyeTracking), isOn: controller.eyeTrackingEnabled))
        menu.addItem(makeToggleItem("自动眨眼", action: #selector(toggleAutoBlink), isOn: controller.autoBlinkEnabled))
        menu.addItem(makeToggleItem(
            "说话手势节拍与情绪姿态",
            action: #selector(toggleSpeechGestures),
            isOn: controller.animationMixer.speechGestureLayer.enabled
        ))

        return makeSubmenuRoot("视角与基础开关", submenu: menu)
    }

    // MARK: - Helpers

    func makeSubmenuRoot(_ title: String, submenu: NSMenu) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        item.submenu = submenu
        return item
    }

    func makeToggleItem(_ title: String, action: Selector, isOn: Bool) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: "")
        item.target = self
        item.state = isOn ? .on : .off
        return item
    }
}
