import CoreGraphics
import Foundation
import QuartzCore
import simd

extension PetController {
    // MARK: - Picture-World (Mode B: Single-Image Metric Depth 3D Scene)

    func loadPictureWorld(from folderURL: URL) throws {
        sceneActionController.reset()
        try pictureWorldRenderer.loadScene(from: folderURL)
        if let lightRaw = pictureWorldRenderer.config?.lightingPreset,
           let preset = LightingPreset(rawValue: lightRaw) {
            setLightingPreset(preset)
        }
        applyPictureWorldRootTransform()
        settleSpringBonePhysics(warmupSteps: 16)
        onPictureWorldChanged?()
    }

    func exitPictureWorld() {
        guard pictureWorldRenderer.isActive else { return }
        sceneActionController.standUpOrCancel(on: self, stopArdy: false)
        pictureWorldRenderer.unloadScene()
        restoreRestRootTransform()
        camera.reset()
        settleSpringBonePhysics(warmupSteps: 12)
        onPictureWorldChanged?()
    }

    /// Autonomous Picture-World chain: ARDY walks to `anchorId`, aligns, sits down, and holds seated Idle.
    @discardableResult
    func walkToSceneAnchorAndSit(anchorId: String) -> Bool {
        guard let anchor = pictureWorldRenderer.config?.anchors?.first(where: { $0.id == anchorId }) else {
            return false
        }
        return sceneActionController.startWalkAndSit(anchor: anchor, on: self)
    }

    func standUpFromSceneSeat(stopArdy: Bool = true) {
        sceneActionController.standUpOrCancel(on: self, stopArdy: stopArdy)
    }

    func panPictureWorldCanvas(deltaX: Float, deltaY: Float) {
        guard pictureWorldRenderer.isActive else { return }
        let size = hostView?.bounds.size ?? CGSize(width: 820, height: 480)
        pictureWorldRenderer.panCanvas(deltaX: deltaX, deltaY: deltaY, viewportSize: size)
    }

    func zoomPictureWorldCanvas(deltaY: Float) {
        guard pictureWorldRenderer.isActive else { return }
        pictureWorldRenderer.zoomCanvas(deltaY: deltaY, viewportAspect: currentViewportAspect())
    }

    func resetPictureWorldViewAndAvatar() {
        guard pictureWorldRenderer.isActive else { return }
        sceneActionController.standUpOrCancel(on: self, stopArdy: false)
        pictureWorldRenderer.resetCanvasAndAvatarToOrigin()
        if poseMode == .sit {
            setPoseMode(.stand)
        }
        applyPictureWorldRootTransform()
        settleSpringBonePhysics(warmupSteps: 8)
        _ = refreshSceneState(aspect: currentViewportAspect(), force: true)
    }

    func currentViewportAspect() -> Float {
        guard let view = hostView, view.bounds.height > 0 else { return 820.0 / 480.0 }
        return Float(view.bounds.width / view.bounds.height)
    }

    /// Builds an on-demand structured `AvatarSceneState` snapshot (location, depth occlusion,
    /// boundary contact, active AI drivers) for future LLM / agentic tool integration.
    func currentAvatarSceneState(aspect: Float? = nil) -> AvatarSceneState {
        let resolvedAspect = aspect ?? currentViewportAspect()
        let now = Date().timeIntervalSince1970
        let activePrompt: String? = localArdyActive ? localArdyService.currentPrompt : nil

        if pictureWorldRenderer.isActive {
            let spatial = pictureWorldRenderer.evaluateSpatialSnapshot(viewportAspect: resolvedAspect)
            let p = pictureWorldRenderer.avatarWorldPosition
            return AvatarSceneState(
                timestamp: now,
                mode: "picture_world",
                sceneName: pictureWorldRenderer.config?.name,
                positionMeters: [p.x, p.y, p.z],
                normalizedRoomPosition: spatial.normalizedRoomPosition,
                facingYawDegrees: pictureWorldRenderer.avatarWorldYaw * 180.0 / .pi,
                distanceToCameraMeters: spatial.distanceToCameraMeters,
                screenNDC: spatial.screenNDC,
                isOccluded: spatial.isOccluded,
                occlusionRatio: spatial.occlusionRatio,
                occludedRegions: spatial.occludedRegions,
                boundaryContacts: spatial.boundaryContacts,
                isNearBoundary: spatial.isNearBoundary,
                activeBodyEngine: animationMixer.activeBodySource.rawValue,
                activeFaceEngine: animationMixer.activeFaceSource.rawValue,
                activeMotionPrompt: activePrompt,
                activeSceneTask: sceneActionController.activeTaskIdentifier,
                seatedAnchorId: sceneActionController.seatedAnchor?.id,
                canvasZoom: pictureWorldRenderer.canvasZoom
            )
        } else {
            return AvatarSceneState(
                timestamp: now,
                mode: "desktop_pet",
                sceneName: nil,
                positionMeters: [0, 0, 0],
                normalizedRoomPosition: [0, 0],
                facingYawDegrees: camera.yaw * 180.0 / .pi,
                distanceToCameraMeters: camera.distance,
                screenNDC: [0, 0],
                isOccluded: false,
                occlusionRatio: 0,
                occludedRegions: [],
                boundaryContacts: [],
                isNearBoundary: false,
                activeBodyEngine: animationMixer.activeBodySource.rawValue,
                activeFaceEngine: animationMixer.activeFaceSource.rawValue,
                activeMotionPrompt: activePrompt,
                activeSceneTask: nil,
                seatedAnchorId: nil,
                canvasZoom: 1.0
            )
        }
    }

    func currentAvatarSceneStateJSON(pretty: Bool = false) -> String? {
        currentAvatarSceneState().toJSONString(pretty: pretty)
    }

    @discardableResult
    func refreshSceneState(aspect: Float, force: Bool = false) -> AvatarSceneState? {
        let now = CACurrentMediaTime()
        if !force && (now - lastSceneStateSampleTime) < 0.20 {
            return latestSceneState
        }
        lastSceneStateSampleTime = now
        let prev = latestSceneState
        let next = currentAvatarSceneState(aspect: aspect)
        latestSceneState = next
        if prev?.isOccluded != next.isOccluded || prev?.boundaryContacts != next.boundaryContacts {
            AppLogger.shared.info(
                "SceneState",
                String(
                    format: "mode=%@ pos=(%.2f,%.2f) occluded=%@(%.0f%%:%@) boundary=%@",
                    next.mode,
                    next.positionMeters[0],
                    next.positionMeters[2],
                    next.isOccluded ? "yes" : "no",
                    next.occlusionRatio * 100.0,
                    next.occludedRegions.joined(separator: ","),
                    next.boundaryContacts.isEmpty ? "none" : next.boundaryContacts.joined(separator: ",")
                )
            )
        }
        onSceneStateUpdated?(next)
        return next
    }

    func applyPictureWorldRootTransform() {
        guard pictureWorldRenderer.isActive, let model else { return }
        var offset = pictureWorldRenderer.avatarWorldPosition
        offset.y += pictureWorldRenderer.groundHeight(atX: offset.x, z: offset.z)
        let yawQ = simd_quatf(angle: pictureWorldRenderer.avatarWorldYaw, axis: SIMD3<Float>(0, 1, 0))
        for (idx, node) in model.nodes.enumerated() where node.parent == nil {
            guard let rest = poseController.restNodeTransforms[idx] else { continue }
            node.translation = rest.pos + offset
            node.rotation = simd_normalize(yawQ * rest.rot)
            node.updateLocalMatrix()
        }
        model.updateNodeTransforms()
    }

    func restoreRestRootTransform() {
        guard let model else { return }
        for (idx, node) in model.nodes.enumerated() where node.parent == nil {
            guard let rest = poseController.restNodeTransforms[idx] else { continue }
            node.translation = rest.pos
            node.rotation = rest.rot
            node.updateLocalMatrix()
        }
        model.updateNodeTransforms()
    }

    struct UVAnchorSpec {
        let id: String
        let label: String
        let kind: String
        let u: Float
        let v: Float
        let facingYawDeg: Float?
    }

    /// Unprojects VLM-provided 2D normalized image coordinates `(u, v)` into 3D scene anchors
    /// using the active MoGe-2 depth map, snaps approach positions onto `walkableGrid`, writes
    /// them into `scene.json`, and reloads the scene anchors in place.
    func registerSceneAnchorsFromUV(_ specs: [UVAnchorSpec]) throws -> [SceneAnchor] {
        guard let folder = pictureWorldRenderer.sceneFolderURL,
              let cfg = pictureWorldRenderer.config else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 30, userInfo: [NSLocalizedDescriptionKey: "No active Picture-World scene"])
        }
        var built: [SceneAnchor] = []
        for spec in specs {
            guard let worldPt = pictureWorldRenderer.unprojectImageUV(u: spec.u, v: spec.v) else { continue }
            let groundY = pictureWorldRenderer.groundHeight(atX: worldPt.x, z: worldPt.z)
            let relH = min(max(worldPt.y - groundY, 0.32), 0.62)
            let facingYaw: Float = {
                if let deg = spec.facingYawDeg { return deg * .pi / 180.0 }
                // Default: face toward the camera/room center so Alice looks into the room
                let dx = cfg.camPos.x - worldPt.x
                let dz = cfg.camPos.z - worldPt.z
                return atan2(dx, dz)
            }()
            let fwd = SIMD2<Float>(sin(facingYaw), cos(facingYaw))
            let rawApproach = SIMD2<Float>(worldPt.x + fwd.x * 0.55, worldPt.z + fwd.y * 0.55)
            let approachXZ = pictureWorldRenderer.roomGrid?.nearestWalkable(x: rawApproach.x, z: rawApproach.y, maxRadius: 20)
                ?? rawApproach
            let approachGroundY = pictureWorldRenderer.groundHeight(atX: approachXZ.x, z: approachXZ.y)

            built.append(SceneAnchor(
                id: spec.id,
                label: spec.label,
                kind: spec.kind,
                approachPosition: [approachXZ.x, approachGroundY, approachXZ.y],
                targetPosition: [worldPt.x, groundY + relH, worldPt.z],
                seatHeight: spec.kind == "seat" ? (groundY + relH) : nil,
                facingYawRad: facingYaw
            ))
        }
        guard !built.isEmpty else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 31, userInfo: [NSLocalizedDescriptionKey: "No valid 3D points could be unprojected from UVs"])
        }

        let jsonURL = folder.appendingPathComponent("scene.json")
        let data = try Data(contentsOf: jsonURL)
        guard var rootObj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 32, userInfo: [NSLocalizedDescriptionKey: "Invalid scene.json format"])
        }
        let encodedAnchors = try JSONEncoder().encode(built)
        rootObj["anchors"] = try JSONSerialization.jsonObject(with: encodedAnchors)
        let outData = try JSONSerialization.data(withJSONObject: rootObj, options: [.prettyPrinted, .sortedKeys])
        try outData.write(to: jsonURL, options: .atomic)

        try pictureWorldRenderer.loadScene(from: folder)
        AppLogger.shared.info("PictureWorld", "Registered \(built.count) VLM UV anchor(s) into \(jsonURL.lastPathComponent)")
        return built
    }
}
