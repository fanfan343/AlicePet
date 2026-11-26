import Foundation
import QuartzCore
import VRMMetalKit
import simd

enum OutfitToggleKind {
    case dress
    case pantyhose
    case shoes
    case hairBand
    case chestRibbon
    case hairSide
}

extension PetController {
    // MARK: - Lighting, Pose, Props, Hand Gestures & Outfits

    func setLightingPreset(_ preset: LightingPreset) {
        lightingPreset = preset
        applyLightingAndOutline()
        AppLogger.shared.info("LookDev", "Switched lightingPreset=\(preset.rawValue)")
    }

    func setOutlineStyle(_ style: OutlineStyle) {
        outlineStyle = style
        applyLightingAndOutline()
        AppLogger.shared.info("LookDev", "Switched outlineStyle=\(style.rawValue)")
    }

    func applyLightingAndOutline() {
        LightingConfigurator.apply(preset: lightingPreset, outline: outlineStyle, to: renderer)
    }

    func setPoseMode(_ mode: PetPoseMode) {
        let elapsed = Float(CACurrentMediaTime() - startTime)
        poseController.setPoseMode(mode, at: elapsed)
        syncPropRendererMode()
        applyAllBodyAndOutfitState(settlePhysics: true)
        AppLogger.shared.info("Pose", "Switched poseMode=\(mode.rawValue), propMode=\(propRenderer.activePropMode.rawValue)")
    }

    func setHandGesture(_ preset: HandGesturePreset, side: HandSide = .both) {
        poseController.setHandGesture(preset, side: side)
        applyAllBodyAndOutfitState(settlePhysics: false)
        AppLogger.shared.info("Gesture", "Hand gesture updated: L=\(leftHandGesture.rawValue), R=\(rightHandGesture.rawValue)")
    }

    var hasAnyOutfitControls: Bool {
        activeProfile.hasCustomOutfits && (!outfitNodeMeshes.isEmpty || !recipeStore.availableOutfitMorphKeys.isEmpty)
    }

    var hasHairStyleControls: Bool {
        activeProfile.hairStyles.values.contains { !$0.isEmpty }
    }

    func isOutfitToggleSupported(_ kind: OutfitToggleKind) -> Bool {
        let morphs = recipeStore.availableOutfitMorphKeys
        switch kind {
        case .dress: return outfitNodeMeshes["Clothes"] != nil
        case .pantyhose: return outfitNodeMeshes["Pantyhose"] != nil
        case .shoes: return outfitNodeMeshes["Shoes"] != nil
        case .hairBand: return morphs.contains("ClothesPart-HairBandOff")
        case .chestRibbon: return morphs.contains("ClothesPart-ChestRibbonOff")
        case .hairSide: return morphs.contains("HairPart-SideOff")
        }
    }

    func applyOutfitState() {
        guard let model else { return }

        var changedVisibility = false
        let nodeMap: [(String, Bool)] = [
            ("Clothes", outfitState.showDress),
            ("Pantyhose", outfitState.showPantyhose),
            ("Shoes", outfitState.showShoes),
        ]
        for (name, visible) in nodeMap {
            if let entry = outfitNodeMeshes[name], entry.nodeIndex < model.nodes.count {
                let expectedMesh: Int? = visible ? entry.meshIndex : nil
                if model.nodes[entry.nodeIndex].mesh != expectedMesh {
                    model.nodes[entry.nodeIndex].mesh = expectedMesh
                    changedVisibility = true
                }
            }
        }
        if changedVisibility {
            let prev = renderer.cameraMode
            renderer.cameraMode = (prev == .thirdPerson) ? .firstPerson : .thirdPerson
            renderer.cameraMode = prev
        }

        if let exprCtrl = renderer.expressionController {
            func setOutfitMorph(_ morphName: String, weight: Float) {
                guard recipeStore.availableOutfitMorphKeys.contains(morphName) else { return }
                exprCtrl.setCustomExpressionWeight(AssetRecipeStore.outfitMorphInternalPrefix + morphName, weight: weight)
            }
            setOutfitMorph("Body2 ShapeKey-Dress", weight: outfitState.showDress ? 1.0 : 0.0)
            setOutfitMorph("Pantyhose ShapeKey-Dress", weight: outfitState.showDress ? 1.0 : 0.0)
            setOutfitMorph("Body2 ShapeKey-Pantyhose", weight: outfitState.showPantyhose ? 1.0 : 0.0)
            setOutfitMorph("Body2 ShapeKey-Shoes", weight: outfitState.showShoes ? 1.0 : 0.0)
            setOutfitMorph("Pantyhose ShapeKey-Shoes", weight: outfitState.showShoes ? 1.0 : 0.0)
            setOutfitMorph("ClothesPart-HairBandOff", weight: outfitState.showHairBand ? 0.0 : 1.0)
            setOutfitMorph("ClothesPart-ChestRibbonOff", weight: outfitState.showChestRibbon ? 0.0 : 1.0)
            setOutfitMorph("HairPart-SideOff", weight: outfitState.showHairSide ? 0.0 : 1.0)
        }

        poseController.applyHairStyle(outfitState.hairStyle, profile: activeProfile, to: model)
        model.updateNodeTransforms()
        AppLogger.shared.info("Outfit", "Applied outfit: dress=\(outfitState.showDress), pantyhose=\(outfitState.showPantyhose), shoes=\(outfitState.showShoes), band=\(outfitState.showHairBand), ribbon=\(outfitState.showChestRibbon), side=\(outfitState.showHairSide), hair=\(outfitState.hairStyle.rawValue)")
    }

    func syncPropRendererMode() {
        if let manualPropOverride {
            propRenderer.activePropMode = manualPropOverride
            return
        }
        switch poseMode {
        case .stand, .sit, .crouch, .prone:
            propRenderer.activePropMode = .none
        case .afkBook:
            propRenderer.activePropMode = .bookAndTea
        case .afkTea:
            propRenderer.activePropMode = .teaSetOnly
        }
    }

    func applyAllBodyAndOutfitState(settlePhysics: Bool, animTime: Float = 0) {
        guard let model else { return }
        poseController.applyAll(
            to: model,
            recipeStore: recipeStore,
            profile: activeProfile,
            hairStyle: outfitState.hairStyle,
            at: animTime
        )
        applyOutfitState()
        if pictureWorldRenderer.isActive {
            applySeatedUprightPostureIfNeeded()
            applyPictureWorldRootTransform()
        }
        if settlePhysics {
            settleSpringBonePhysics()
        }
    }

    /// Gently lifts torso & head upright when seated against a sofa/chair backrest.
    func applySeatedUprightPostureIfNeeded() {
        guard sceneActionController.seatedAnchor != nil, let model, let humanoid = model.humanoid else { return }
        let uprightAdjust: [(VRMHumanoidBone, Float)] = [
            (.spine, -0.26),
            (.chest, -0.14),
            (.head, -0.12),
        ]
        for (bone, pitchRad) in uprightAdjust {
            if let idx = humanoid.getBoneNode(bone), idx < model.nodes.count {
                let liftQ = simd_quatf(angle: pitchRad, axis: SIMD3<Float>(1, 0, 0))
                model.nodes[idx].rotation = simd_normalize(model.nodes[idx].rotation * liftQ)
                model.nodes[idx].updateLocalMatrix()
            }
        }
    }

    /// Resets and warms up SpringBone chains after instantaneous pose teleports (e.g. stand -> sit).
    func settleSpringBonePhysics(warmupSteps: Int = 24) {
        guard springBoneEnabled, model != nil else { return }
        renderer.resetPhysics()
        renderer.warmupPhysics(steps: warmupSteps)
    }

    // MARK: - Expressions

    func setPresetMood(_ preset: VRMExpressionPreset?) {
        expressionMixer.setPresetMood(preset, on: renderer)
    }

    func setCustomExpression(_ name: String, weight: Float = 1.0) {
        expressionMixer.setCustomExpression(name, weight: weight, on: renderer)
    }

    func applyExpression(named query: String, weight: Float = 1.0) {
        expressionMixer.applyExpression(named: query, weight: weight, on: renderer, recipeStore: recipeStore)
    }
}
