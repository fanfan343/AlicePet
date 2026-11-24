import Foundation
import VRMMetalKit
import simd

// MARK: - JSON Recipe Models

struct AliceAssetRecipes: Codable {
    let source: String
    let expressions: [ExpressionRecipe]
    let hands: [HandRecipe]
    let outfits: [String: OutfitRecipe]
    let poses: [String: PoseRecipe]
}

struct ExpressionRecipe: Codable {
    let id: String
    let label: String
    let morphs: [String: Float]
}

struct HandRecipe: Codable {
    let id: String
    let label: String
    let left: [String: Float]
    let right: [String: Float]
}

struct OutfitRecipe: Codable {
    let morphs: [String: Float]
    let nodes: [String: Bool]
    let boneEulers: [String: [Float]]
}

struct PoseRecipe: Codable {
    let muscles: [String: Float]
    let propPositions: [String: [Float]]
    let propEulers: [String: [Float]]
}

// MARK: - High-level Pet State Enums

enum PetPoseMode: String, CaseIterable {
    case stand = "stand"
    case sit = "sit"
    case crouch = "crouch"
    case prone = "prone"
    case afkBook = "afk_book"
    case afkTea = "afk_tea"

    var displayTitle: String {
        switch self {
        case .stand: return "原版站立 (Stand)"
        case .sit: return "原版坐姿 (Sit)"
        case .crouch: return "原版蹲姿 (Crouch)"
        case .prone: return "原版趴卧 (Prone)"
        case .afkBook: return "原版 AFK 浮空阅读 (Book + Tea)"
        case .afkTea: return "原版 AFK 茶会 (Tea Set)"
        }
    }

    var authorClipID: String {
        switch self {
        case .stand: return "stand"
        case .sit: return "sit_action"
        case .crouch: return "crouch"
        case .prone: return "prone"
        case .afkBook, .afkTea: return "afk_loop"
        }
    }
}

enum HandGesturePreset: String, CaseIterable {
    case natural = "natural"
    case idle2 = "idle2"
    case open = "open"
    case fist = "fist"
    case peace = "peace"
    case point = "point"
    case rock = "rock"
    case thumbsUp = "thumbs_up"
    case gun = "gun"

    var displayTitle: String {
        switch self {
        case .natural: return "自然放松 (Natural)"
        case .idle2: return "微蜷待机 (Idle)"
        case .open: return "张开手掌 (Open)"
        case .fist: return "握拳 (Fist)"
        case .peace: return "剪刀手 / 比耶 (Peace)"
        case .point: return "食指指向 (Point)"
        case .rock: return "摇滚手势 (Rock)"
        case .thumbsUp: return "竖大拇指 (Thumbs Up)"
        case .gun: return "比枪手势 (Gun)"
        }
    }
}

enum HairStyleOption: String, CaseIterable {
    case styleA = "HairStyleA"
    case styleB = "HairStyleB"
    case styleB2 = "HairStyleB2"

    var displayTitle: String {
        switch self {
        case .styleA: return "发型 A (默认长发)"
        case .styleB: return "发型 B (盘发/收束 A)"
        case .styleB2: return "发型 B2 (盘发/收束 B)"
        }
    }
}

struct OutfitState: Equatable {
    var showDress: Bool = true
    var showPantyhose: Bool = true
    var showShoes: Bool = true
    var showHairBand: Bool = true
    var showChestRibbon: Bool = true
    var showHairSide: Bool = true
    var hairStyle: HairStyleOption = .styleA
}

// MARK: - Asset Recipe Store

@MainActor
final class AssetRecipeStore {
    private(set) var recipes: AliceAssetRecipes?
    private(set) var assetsFolderURL: URL?

    // Dynamically resolved expression names registered on the active VRMModel
    private(set) var availableFaceRecipes: [(id: String, title: String)] = []
    private(set) var availableOutfitMorphKeys: Set<String> = []

    /// Curated expression presets bridging both Unity .anim face recipes & VRM -prefab expressions
    static let curatedExpressionCatalog: [(animID: String, prefabFallback: String?, title: String)] = [
        ("-01prefab 1joy ＞＜", "-1prefab 1joy  -><", "开心眯眼 (＞＜)"),
        ("-02prefab 1joy wink", "-2prefab 1joy wink", "Wink 眨眼"),
        ("-03prefab 1joy tongueOut", "-3prefab 1joy tongueOut", "俏皮吐舌"),
        ("-04prefab 1joy happy", "-4prefab 1joy happy", "灿烂开心"),
        ("-05prefab 1joy complacent", "-5prefab 1joy complacent", "得意洋洋"),
        ("-06prefab 1joy mesugaki", "-6prefab 1joy mesugaki", "小恶魔坏笑"),
        ("-08prefab 1joy stareyes", "-8prefab 1joy stareyes", "星星眼"),
        ("-09prefab 2Angry Disgusted", "-9prefab 2Angry Disgusted", "嫌弃"),
        ("-11prefab 2Angry angry1", "-11prefab 2Angry angry1", "生气鼓脸"),
        ("-12prefab 3sad cry1", "-12prefab 3sad cry1", "委屈哭哭"),
        ("-14prefab 3sad sad1", "-14prefab 3sad sad1", "低落难过"),
        ("-15prefab 4fear fear1", "-15prefab 4fear fear1", "害怕发抖"),
        ("-17prefab 4fear guruguru", "-17prefab 4fear guruguru", "蚊香眼晕眩"),
        ("-18prefab 5other shy", "-18prefab 5other shy", "害羞脸红"),
        ("-19prefab 5other drunk", "-19prefab 5other drunk", "微醺迷糊"),
        ("-20prefab 5other shock", "-20prefab 5other shock", "大受震惊"),
        ("-23prefab 6other wiredsmile", "-23prefab 6other wiredsmile", "尴尬假笑"),
        ("-25prefab 6other Tiger Tooth", "-25prefab 6other Tiger Tooth", "小虎牙"),
        ("eyes sarcastic", nil, "傲娇半眼 (Sarcastic)"),
        ("eyes scornful", nil, "鄙视死鱼眼 (Scornful)"),
        ("eyes wink l", nil, "左眼 Wink"),
        ("eyes wink r", nil, "右眼 Wink"),
        ("mouth ω", nil, "猫猫嘴 (ω)"),
        ("mouth cheekpuff l", nil, "鼓腮帮子"),
        ("other hearteyes", nil, "爱心眼 (HeartEyes)"),
    ]

    static let outfitMorphInternalPrefix = "__alice_outfit_"
    static let expressionRecipePrefix = "__alice_expr_"

    func locateAssetsFolder(near modelURL: URL?, relativeName: String?) -> URL? {
        guard let relativeName, !relativeName.isEmpty else {
            self.assetsFolderURL = nil
            return nil
        }
        let fm = FileManager.default
        var candidates: [URL] = []
        if let modelURL {
            let dir = modelURL.deletingLastPathComponent()
            candidates.append(dir.appendingPathComponent(relativeName))
            candidates.append(dir.deletingLastPathComponent().appendingPathComponent("降采样模型/\(relativeName)"))
        }
        let bundleParent = Bundle.main.bundleURL.deletingLastPathComponent()
        candidates.append(bundleParent.appendingPathComponent("降采样模型/\(relativeName)"))
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        candidates.append(cwd.appendingPathComponent("降采样模型/\(relativeName)"))
        candidates.append(cwd.deletingLastPathComponent().appendingPathComponent("降采样模型/\(relativeName)"))

        for url in candidates {
            if fm.fileExists(atPath: url.appendingPathComponent("recipes.json").path) {
                self.assetsFolderURL = url
                return url
            }
        }
        self.assetsFolderURL = nil
        return nil
    }

    func loadRecipes(near modelURL: URL?, profile: ModelProfile) {
        self.recipes = nil
        guard let folder = locateAssetsFolder(near: modelURL, relativeName: profile.assetsFolderRelative) else {
            if profile.assetsFolderRelative != nil {
                AppLogger.shared.warn("Assets", "Could not locate \(profile.assetsFolderRelative ?? "") folder")
            }
            return
        }
        let jsonURL = folder.appendingPathComponent("recipes.json")
        do {
            let data = try Data(contentsOf: jsonURL)
            self.recipes = try JSONDecoder().decode(AliceAssetRecipes.self, from: data)
            AppLogger.shared.info("Assets", "Loaded recipes.json (\(recipes?.expressions.count ?? 0) exprs, \(recipes?.hands.count ?? 0) hands, \(recipes?.outfits.count ?? 0) outfits)")
        } catch {
            AppLogger.shared.error("Assets", "Failed to decode recipes.json: \(error)")
        }
    }

    // MARK: - Register Custom Morph Expressions on VRMModel

    /// Scans all meshes' morph target names (`targetNames`) and registers dynamic VRMExpressions
    /// for outfit morphs and Unity expression recipes so they work on both LightShapekey & Full VRM.
    func registerExpressions(on model: VRMModel, controller: VRMExpressionController?, profile: ModelProfile) {
        availableFaceRecipes.removeAll()
        availableOutfitMorphKeys.removeAll()
        guard let controller else { return }

        var meshToNode: [Int: Int] = [:]
        for (nIdx, node) in model.nodes.enumerated() {
            if let mIdx = node.mesh, meshToNode[mIdx] == nil {
                meshToNode[mIdx] = nIdx
            }
        }

        var morphLookup: [String: (nodeIdx: Int, meshIdx: Int, targetIdx: Int)] = [:]
        var bodyMorphNames: Set<String> = []
        let gltfMeshes = model.gltf.meshes ?? []
        for (mIdx, mesh) in model.meshes.enumerated() {
            let nodeIdx = meshToNode[mIdx] ?? 0
            let targetNames: [String] = {
                if mIdx < gltfMeshes.count {
                    return gltfMeshes[mIdx].extras?.targetNames ?? []
                }
                return []
            }()
            for (pIdx, prim) in mesh.primitives.enumerated() {
                if pIdx == 0 {
                    for (tIdx, mt) in prim.morphTargets.enumerated() {
                        let tname = (tIdx < targetNames.count) ? targetNames[tIdx] : mt.name
                        if !tname.isEmpty {
                            morphLookup[tname] = (nodeIdx, mIdx, tIdx)
                            if mIdx == 0 { bodyMorphNames.insert(tname) }
                        }
                    }
                }
            }
        }

        // Register profile-declared outfit morphs only when present on this VRM
        for name in profile.outfitMorphNames {
            if let hit = morphLookup[name] {
                var expr = VRMExpression(name: Self.outfitMorphInternalPrefix + name)
                expr.morphTargetBinds = [VRMMorphTargetBind(node: hit.nodeIdx, index: hit.targetIdx, weight: 1.0, meshIndex: hit.meshIdx)]
                controller.registerCustomExpression(expr, name: Self.outfitMorphInternalPrefix + name)
                availableOutfitMorphKeys.insert(name)
            }
        }

        // Collect existing VRM custom expression names
        let existingCustomNames = Set(model.expressions?.custom.keys.map { $0 } ?? [])
        let recipeMap = Dictionary(uniqueKeysWithValues: (recipes?.expressions ?? []).map { ($0.id, $0) })

        for entry in Self.curatedExpressionCatalog {
            // 1. If the VRM already ships this -prefab expression, use it directly
            if let prefabName = entry.prefabFallback, existingCustomNames.contains(prefabName) {
                availableFaceRecipes.append((id: prefabName, title: entry.title))
                continue
            }
            // 2. Otherwise synthesize a VRMExpression from Unity .anim blendshape weights
            guard let animRecipe = recipeMap[entry.animID] else { continue }
            var binds: [VRMMorphTargetBind] = []
            for (rawBS, weight) in animRecipe.morphs {
                if let hit = resolveMorphTarget(rawName: rawBS, lookup: morphLookup) {
                    binds.append(VRMMorphTargetBind(node: hit.nodeIdx, index: hit.targetIdx, weight: max(0, min(1, weight)), meshIndex: hit.meshIdx))
                }
            }
            if !binds.isEmpty {
                let exprKey = Self.expressionRecipePrefix + entry.animID
                var expr = VRMExpression(name: exprKey)
                expr.morphTargetBinds = binds
                controller.registerCustomExpression(expr, name: exprKey)
                availableFaceRecipes.append((id: exprKey, title: entry.title))
            }
        }

        AppLogger.shared.info("Assets", "Registered \(availableOutfitMorphKeys.count) outfit morphs and \(availableFaceRecipes.count) curated face recipes (Body morphs=\(bodyMorphNames.count))")
    }

    private func resolveMorphTarget(rawName: String, lookup: [String: (nodeIdx: Int, meshIdx: Int, targetIdx: Int)]) -> (nodeIdx: Int, meshIdx: Int, targetIdx: Int)? {
        if let exact = lookup[rawName] { return exact }
        if rawName.hasSuffix(".VRC") {
            let stripped = String(rawName.dropLast(4)).trimmingCharacters(in: .whitespaces)
            if let hit = lookup[stripped] { return hit }
        }
        // Fallback Japanese/alias mappings for LightShapekey
        let aliases: [String: String] = [
            "wink": "ウインク",
            "wink.VRC": "ウインク",
            "ウィンク": "ウインク",
            "ウィンク.VRC": "ウインク",
            "ウィンク右": "ウインク右",
            "ウィンク右.VRC": "ウインク右",
            "ふむ.VRC": "ふむ",
            "じと目.VRC": "じと目",
            "怒り.VRC": "怒り",
            "笑い.VRC": "笑い",
            "ω.VRC": "ω",
            "まばたき.VRC": "まばたき",
        ]
        if let alias = aliases[rawName], let hit = lookup[alias] {
            return hit
        }
        return nil
    }

    // MARK: - Unity Muscle -> VRM Finger Bone Rotations

    func fingerRotations(for preset: HandGesturePreset, isLeft: Bool) -> [VRMHumanoidBone: simd_quatf] {
        guard preset != .natural,
              let recipe = recipes?.hands.first(where: { $0.id == preset.rawValue }) else {
            return [:]
        }
        let muscles = isLeft ? recipe.left : recipe.right
        var out: [VRMHumanoidBone: simd_quatf] = [:]

        // In VRM T-pose (palms down):
        // Left fingers extend along -X, curl inward around -Z; Right fingers extend along +X, curl inward around +Z.
        let curlSign: Float = isLeft ? -1.0 : 1.0
        let spreadSign: Float = isLeft ? -1.0 : 1.0

        func curlAngle(_ muscleValue: Float?, maxFlexDeg: Float = 78.0, maxExtDeg: Float = 15.0) -> Float {
            let v = muscleValue ?? 0.65
            // Unity Stretched: +1 = straight/extended, -1 = maximally flexed into palm
            if v >= 0 {
                return -(v * maxExtDeg) * .pi / 180.0
            } else {
                return (-v * maxFlexDeg) * .pi / 180.0
            }
        }

        let fingers: [(name: String, bones: [VRMHumanoidBone])] = [
            ("Index", isLeft
                ? [.leftIndexProximal, .leftIndexIntermediate, .leftIndexDistal]
                : [.rightIndexProximal, .rightIndexIntermediate, .rightIndexDistal]),
            ("Middle", isLeft
                ? [.leftMiddleProximal, .leftMiddleIntermediate, .leftMiddleDistal]
                : [.rightMiddleProximal, .rightMiddleIntermediate, .rightMiddleDistal]),
            ("Ring", isLeft
                ? [.leftRingProximal, .leftRingIntermediate, .leftRingDistal]
                : [.rightRingProximal, .rightRingIntermediate, .rightRingDistal]),
            ("Little", isLeft
                ? [.leftLittleProximal, .leftLittleIntermediate, .leftLittleDistal]
                : [.rightLittleProximal, .rightLittleIntermediate, .rightLittleDistal]),
        ]

        for (fname, bones) in fingers {
            let spread = (muscles["\(fname).Spread"] ?? 0) * 14.0 * (.pi / 180.0) * spreadSign
            for (segIdx, bone) in bones.enumerated() {
                let c = curlAngle(muscles["\(fname).\(segIdx + 1) Stretched"])
                let qCurl = simd_quatf(angle: c * curlSign, axis: SIMD3<Float>(0, 0, 1))
                if segIdx == 0 {
                    let splayBias: Float = fname == "Index" ? -0.05 : (fname == "Little" ? 0.08 : 0)
                    let qSpread = simd_quatf(angle: spread + splayBias * spreadSign, axis: SIMD3<Float>(0, 1, 0))
                    out[bone] = qSpread * qCurl
                } else {
                    out[bone] = qCurl
                }
            }
        }

        // Thumb: Metacarpal + Proximal + Distal
        let t1 = muscles["Thumb.1 Stretched"] ?? 0.5
        let t2 = muscles["Thumb.2 Stretched"] ?? 0.5
        let t3 = muscles["Thumb.3 Stretched"] ?? 0.5
        let tSpread = muscles["Thumb.Spread"] ?? 0.0

        let thumbBones: [VRMHumanoidBone] = isLeft
            ? [.leftThumbMetacarpal, .leftThumbProximal, .leftThumbDistal]
            : [.rightThumbMetacarpal, .rightThumbProximal, .rightThumbDistal]

        // Thumb bends across palm (mix of Y and Z)
        let flex1 = max(0, -t1) * 40.0 * .pi / 180.0
        let splay1 = tSpread * 22.0 * .pi / 180.0
        out[thumbBones[0]] =
            simd_quatf(angle: (isLeft ? 1 : -1) * flex1, axis: SIMD3<Float>(0, 1, 0)) *
            simd_quatf(angle: (isLeft ? -1 : 1) * splay1, axis: SIMD3<Float>(0, 0, 1))
        out[thumbBones[1]] = simd_quatf(angle: (isLeft ? 1 : -1) * max(0, -t2) * 52.0 * .pi / 180.0, axis: SIMD3<Float>(0, 1, 0))
        out[thumbBones[2]] = simd_quatf(angle: (isLeft ? 1 : -1) * max(0, -t3) * 55.0 * .pi / 180.0, axis: SIMD3<Float>(0, 1, 0))

        return out
    }
}
