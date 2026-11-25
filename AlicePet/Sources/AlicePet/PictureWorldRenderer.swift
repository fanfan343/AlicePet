import AppKit
import CoreGraphics
import Foundation
import Metal
import MetalKit
import simd

struct SceneAnchor: Codable, Equatable, Sendable {
    let id: String
    let label: String
    let kind: String // "seat" | "stand_spot"
    let approachPosition: [Float]
    let targetPosition: [Float]
    let seatHeight: Float?
    let facingYawRad: Float

    var approachPos: SIMD3<Float> {
        approachPosition.count == 3
            ? SIMD3<Float>(approachPosition[0], approachPosition[1], approachPosition[2])
            : .zero
    }
    var targetPos: SIMD3<Float> {
        targetPosition.count == 3
            ? SIMD3<Float>(targetPosition[0], targetPosition[1], targetPosition[2])
            : approachPos
    }
}

struct PictureWorldSceneConfig: Codable {
    let version: Int
    let name: String
    let imageWidth: Int
    let imageHeight: Int
    let depthWidth: Int
    let depthHeight: Int
    let fovYDegrees: Float
    let cameraPosition: [Float]
    let cameraTarget: [Float]
    let cameraUp: [Float]
    let avatarSpawnPosition: [Float]
    let avatarSpawnYawRad: Float
    let avatarScale: Float?
    let walkBoundsMin: [Float]?
    let walkBoundsMax: [Float]?
    let nearClip: Float?
    let farClip: Float?
    let lightingPreset: String?
    let anchors: [SceneAnchor]?

    var camPos: SIMD3<Float> { vec3(cameraPosition, fallback: SIMD3<Float>(0, 1.35, 2.8)) }
    var camTarget: SIMD3<Float> { vec3(cameraTarget, fallback: SIMD3<Float>(0, 1.0, 0)) }
    var camUp: SIMD3<Float> { vec3(cameraUp, fallback: SIMD3<Float>(0, 1, 0)) }
    var spawnPos: SIMD3<Float> { vec3(avatarSpawnPosition, fallback: .zero) }
    var nearZ: Float { nearClip ?? 0.05 }
    var farZ: Float { farClip ?? 50.0 }

    private func vec3(_ arr: [Float], fallback: SIMD3<Float>) -> SIMD3<Float> {
        guard arr.count == 3 else { return fallback }
        return SIMD3<Float>(arr[0], arr[1], arr[2])
    }
}

/// Native Metal "Picture-World" renderer (Mode B):
/// Draws a single 2D illustration/photo (`bg.png`) and writes its metric depth (`depth.bin` from MoGe-2)
/// directly into the hardware `depth32Float` buffer before `VRMRenderer`, giving Alice 100% native
/// 3D occlusion behind foreground furniture + floor contact shadow with near-zero GPU/RAM overhead.
@MainActor
final class PictureWorldRenderer {
    private struct Uniforms {
        var uvScale: SIMD2<Float>
        var uvOffset: SIMD2<Float>
        var nearZ: Float
        var farZ: Float
        var imgTanHalfFovX: Float
        var imgTanHalfFovY: Float
        var camPos: SIMD4<Float>
        var camRight: SIMD4<Float>
        var camUp: SIMD4<Float>
        var camFwd: SIMD4<Float>
        var avatarFootPos: SIMD4<Float>
        var shadowParams: SIMD4<Float> // x=radius, y=strength, z=yTolerance, w=enabled
    }

    struct SpatialSnapshot {
        let normalizedRoomPosition: [Float]
        let distanceToCameraMeters: Float
        let screenNDC: [Float]
        let isOccluded: Bool
        let occlusionRatio: Float
        let occludedRegions: [String]
        let boundaryContacts: [String]
        let isNearBoundary: Bool
    }

    private let device: MTLDevice
    private let pipelineMSAA: MTLRenderPipelineState
    private let pipelineSingle: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let linearSampler: MTLSamplerState
    private let nearestSampler: MTLSamplerState

    private(set) var config: PictureWorldSceneConfig?
    private(set) var sceneFolderURL: URL?
    private var bgTexture: MTLTexture?
    private var depthTexture: MTLTexture?
    /// CPU-side copy of MoGe-2 metric depth (`depthWidth * depthHeight`, ~1.5 MB) for O(1) LLM occlusion queries.
    private var depthMeters: [Float] = []

    /// Live avatar world offset on the room floor (Y=0 is the fitted floor plane).
    var avatarWorldPosition: SIMD3<Float> = .zero
    var avatarWorldYaw: Float = 0.0
    var depthBiasMeters: Float = 0.0

    /// 2D canvas view zoom (1.0 = unzoomed fit, up to 3.0x) and pan in image UV space (clamped inside photo).
    private(set) var canvasZoom: Float = 1.0
    private(set) var canvasPanUV: SIMD2<Float> = .zero

    /// Anchor & boundary telemetry for ARDY continuous 3D root locomotion inside the room.
    private var lastARDYRootXZ: SIMD2<Float>?
    private(set) var lastBoundaryContacts: [String] = []
    private(set) var lastIsNearBoundary: Bool = false

    var isActive: Bool {
        config != nil && bgTexture != nil && depthTexture != nil
    }

    init(device: MTLDevice, colorFormat: MTLPixelFormat = .bgra8Unorm, depthFormat: MTLPixelFormat = .depth32Float, sampleCount: Int = 4) throws {
        self.device = device

        let library = try device.makeLibrary(source: Self.metalShaderSource, options: nil)
        guard let vs = library.makeFunction(name: "picture_world_vs"),
              let fs = library.makeFunction(name: "picture_world_fs") else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to compile PictureWorld Metal shaders"])
        }

        func makePSO(samples: Int) throws -> MTLRenderPipelineState {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = vs
            desc.fragmentFunction = fs
            desc.colorAttachments[0].pixelFormat = colorFormat
            desc.depthAttachmentPixelFormat = depthFormat
            desc.rasterSampleCount = max(samples, 1)
            return try device.makeRenderPipelineState(descriptor: desc)
        }

        self.pipelineMSAA = try makePSO(samples: sampleCount)
        self.pipelineSingle = try makePSO(samples: 1)

        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .always
        depthDesc.isDepthWriteEnabled = true
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create PictureWorld depth state"])
        }
        self.depthState = ds

        let linDesc = MTLSamplerDescriptor()
        linDesc.minFilter = .linear
        linDesc.magFilter = .linear
        linDesc.sAddressMode = .clampToEdge
        linDesc.tAddressMode = .clampToEdge
        guard let lin = device.makeSamplerState(descriptor: linDesc) else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create linear sampler"])
        }
        self.linearSampler = lin

        let nearDesc = MTLSamplerDescriptor()
        nearDesc.minFilter = .nearest
        nearDesc.magFilter = .nearest
        nearDesc.sAddressMode = .clampToEdge
        nearDesc.tAddressMode = .clampToEdge
        guard let near = device.makeSamplerState(descriptor: nearDesc) else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create nearest sampler"])
        }
        self.nearestSampler = near
    }

    func loadScene(from folderURL: URL) throws {
        let jsonURL = folderURL.appendingPathComponent("scene.json")
        let bgURL = folderURL.appendingPathComponent("bg.png")
        let depthURL = folderURL.appendingPathComponent("depth.bin")

        let jsonData = try Data(contentsOf: jsonURL)
        let cfg = try JSONDecoder().decode(PictureWorldSceneConfig.self, from: jsonData)

        let bgTex = try loadBGRA8Texture(from: bgURL)
        let (depthTex, depthCPU) = try loadDepthFloat32Texture(from: depthURL, width: cfg.depthWidth, height: cfg.depthHeight)

        self.config = cfg
        self.sceneFolderURL = folderURL
        self.bgTexture = bgTex
        self.depthTexture = depthTex
        self.depthMeters = depthCPU
        self.avatarWorldPosition = cfg.spawnPos
        self.avatarWorldYaw = cfg.avatarSpawnYawRad
        self.canvasZoom = 1.0
        self.canvasPanUV = .zero
        self.lastARDYRootXZ = nil
        self.lastBoundaryContacts = []
        self.lastIsNearBoundary = false

        AppLogger.shared.info(
            "PictureWorld",
            "Loaded scene '\(cfg.name)' (\(cfg.imageWidth)x\(cfg.imageHeight), depth=\(cfg.depthWidth)x\(cfg.depthHeight), fovY=\(String(format: "%.1f", cfg.fovYDegrees))°, spawn=\(cfg.spawnPos))"
        )
    }

    func unloadScene() {
        config = nil
        sceneFolderURL = nil
        bgTexture = nil
        depthTexture = nil
        depthMeters.removeAll(keepingCapacity: false)
        avatarWorldPosition = .zero
        avatarWorldYaw = 0.0
        canvasZoom = 1.0
        canvasPanUV = .zero
        lastARDYRootXZ = nil
        lastBoundaryContacts = []
        lastIsNearBoundary = false
        AppLogger.shared.info("PictureWorld", "Unloaded Picture-World scene (0 scene VRAM)")
    }

    // MARK: - Unified Canvas Pan / Zoom / Double-Click Reset

    func zoomCanvas(deltaY: Float, viewportAspect: Float) {
        guard isActive else { return }
        canvasZoom = min(max(canvasZoom * (1.0 + deltaY * 0.045), 1.0), 3.0)
        clampCanvasPan(viewportAspect: viewportAspect)
    }

    func panCanvas(deltaX: Float, deltaY: Float, viewportSize: CGSize) {
        guard isActive, viewportSize.width > 1, viewportSize.height > 1 else { return }
        let aspect = Float(viewportSize.width / viewportSize.height)
        let uvScale = currentUVWindow(viewportAspect: aspect).uvScale
        // Natural grab-and-drag: dragging right reveals left of image (panU decreases); dragging down reveals top (panV decreases)
        canvasPanUV.x -= (deltaX / Float(viewportSize.width)) * uvScale.x
        canvasPanUV.y -= (deltaY / Float(viewportSize.height)) * uvScale.y
        clampCanvasPan(viewportAspect: aspect)
    }

    func resetCanvasAndAvatarToOrigin() {
        canvasZoom = 1.0
        canvasPanUV = .zero
        if let cfg = config {
            avatarWorldPosition = cfg.spawnPos
            avatarWorldYaw = cfg.avatarSpawnYawRad
        } else {
            avatarWorldPosition = .zero
            avatarWorldYaw = 0.0
        }
        lastARDYRootXZ = nil
        lastBoundaryContacts = []
        lastIsNearBoundary = false
    }

    private func clampCanvasPan(viewportAspect: Float) {
        let baseScale = baseAspectFillUVScale(viewportAspect: viewportAspect)
        let zoomedScale = baseScale / max(canvasZoom, 1.0)
        let maxPanU = max(0.0, (1.0 - zoomedScale.x) * 0.5)
        let maxPanV = max(0.0, (1.0 - zoomedScale.y) * 0.5)
        canvasPanUV.x = min(max(canvasPanUV.x, -maxPanU), maxPanU)
        canvasPanUV.y = min(max(canvasPanUV.y, -maxPanV), maxPanV)
    }

    private func baseAspectFillUVScale(viewportAspect: Float) -> SIMD2<Float> {
        guard let cfg = config else { return SIMD2<Float>(1, 1) }
        let safeAspect = max(viewportAspect, 0.1)
        let imgAspect = Float(cfg.imageWidth) / max(Float(cfg.imageHeight), 1.0)
        if safeAspect >= imgAspect {
            return SIMD2<Float>(1.0, imgAspect / safeAspect)
        } else {
            return SIMD2<Float>(safeAspect / imgAspect, 1.0)
        }
    }

    private func currentUVWindow(viewportAspect: Float) -> (uvScale: SIMD2<Float>, uvOffset: SIMD2<Float>) {
        let baseScale = baseAspectFillUVScale(viewportAspect: viewportAspect)
        let uvScale = baseScale / max(canvasZoom, 1.0)
        let maxPanU = max(0.0, (1.0 - uvScale.x) * 0.5)
        let maxPanV = max(0.0, (1.0 - uvScale.y) * 0.5)
        let panU = min(max(canvasPanUV.x, -maxPanU), maxPanU)
        let panV = min(max(canvasPanUV.y, -maxPanV), maxPanV)
        let uvOffset = SIMD2<Float>(
            0.5 - 0.5 * uvScale.x + panU,
            0.5 - 0.5 * uvScale.y + panV
        )
        return (uvScale, uvOffset)
    }

    // MARK: - ARDY Room Locomotion & Anti-Stick Boundary Steering

    func resetARDYRootTracking() {
        lastARDYRootXZ = nil
    }

    /// Integrates continuous ARDY root XZ displacement (meters) onto the Picture-World floor while
    /// enforcing BOTH the scene `walkBounds` and the unzoomed base camera-frustum visibility guard.
    /// When `goalTargetXZ` is supplied (goal-directed navigation), smoothly steers `avatarWorldYaw`
    /// toward the destination so ARDY footsteps walk Alice directly there. Otherwise, when Alice
    /// touches a boundary while walking outward, gently steers inward back into the room.
    func applyARDYRootMotion(
        worldRootXZ: SIMD2<Float>,
        viewportAspect: Float,
        goalTargetXZ: SIMD2<Float>? = nil
    ) {
        guard isActive else { return }
        guard let prev = lastARDYRootXZ else {
            lastARDYRootXZ = worldRootXZ
            return
        }
        let rawDelta = worldRootXZ - prev
        lastARDYRootXZ = worldRootXZ

        // Guard against single-frame stream restart teleports (>35cm in one frame)
        let stepLen = simd_length(rawDelta)
        guard stepLen > 0.0001 && stepLen < 0.35 else { return }

        // Rotate ARDY delta by current world yaw so footfalls match Alice's world facing direction
        func rotateStep(_ yaw: Float) -> SIMD2<Float> {
            let cy = cos(yaw)
            let sy = sin(yaw)
            return SIMD2<Float>(rawDelta.x * cy + rawDelta.y * sy, -rawDelta.x * sy + rawDelta.y * cy)
        }

        // Goal-directed steering: rotate world yaw smoothly so ARDY's walking step heads straight to goalTargetXZ
        if let goal = goalTargetXZ {
            let toGoal = goal - SIMD2<Float>(avatarWorldPosition.x, avatarWorldPosition.z)
            let distToGoal = simd_length(toGoal)
            if distToGoal > 0.03 {
                let desiredDir = toGoal / distToGoal
                let currentStep = rotateStep(avatarWorldYaw)
                let stepDir = currentStep / max(stepLen, 1e-5)
                let curAngle = atan2(stepDir.x, stepDir.y)
                let targetAngle = atan2(desiredDir.x, desiredDir.y)
                var angleDiff = targetAngle - curAngle
                while angleDiff > .pi { angleDiff -= 2 * .pi }
                while angleDiff < -.pi { angleDiff += 2 * .pi }

                let maxTurn = min(0.28, stepLen * 6.5 + 0.05)
                let turnStep = min(max(angleDiff, -maxTurn), maxTurn)
                avatarWorldYaw += turnStep
                if avatarWorldYaw > .pi { avatarWorldYaw -= 2 * .pi }
                if avatarWorldYaw < -.pi { avatarWorldYaw += 2 * .pi }
            }
        }

        var worldStep = rotateStep(avatarWorldYaw)
        if let goal = goalTargetXZ {
            // Avoid overshooting the approach spot on the final step
            let toGoal = goal - SIMD2<Float>(avatarWorldPosition.x, avatarWorldPosition.z)
            let dist = simd_length(toGoal)
            if dist < stepLen && dist > 1e-4 {
                worldStep = toGoal
            }
        }
        let candidate = SIMD3<Float>(avatarWorldPosition.x + worldStep.x, 0.0, avatarWorldPosition.z + worldStep.y)
        let eval = clampAvatarPositionWithContacts(candidate, viewportAspect: viewportAspect)

        // Smooth inward turn-back steering in free-roam mode whenever Alice hits or approaches a wall
        if goalTargetXZ == nil, (!eval.contacts.isEmpty || eval.isNearBoundary), simd_length(eval.inwardNormal) > 0.01 {
            let stepDir = worldStep / max(stepLen, 1e-5)
            let inwardDot = simd_dot(stepDir, eval.inwardNormal)
            if !eval.contacts.isEmpty || inwardDot < 0.25 {
                let desiredDir = simd_normalize(eval.inwardNormal)
                // Compute signed angle from current movement direction (stepDir) to desiredDir in XZ plane
                // Note: rotating worldStep by +dYaw rotates (x, z) -> (x cos + z sin, -x sin + z cos)
                let curAngle = atan2(stepDir.x, stepDir.y)
                let targetAngle = atan2(desiredDir.x, desiredDir.y)
                var angleDiff = targetAngle - curAngle
                while angleDiff > .pi { angleDiff -= 2 * .pi }
                while angleDiff < -.pi { angleDiff += 2 * .pi }

                let urgency: Float = eval.contacts.isEmpty ? 0.65 : 1.0
                let maxTurn = min(0.22, (stepLen * 5.2 + 0.03) * urgency)
                let turnStep = min(max(angleDiff, -maxTurn), maxTurn)
                avatarWorldYaw += turnStep
                if avatarWorldYaw > .pi { avatarWorldYaw -= 2 * .pi }
                if avatarWorldYaw < -.pi { avatarWorldYaw += 2 * .pi }

                // Re-project step with the steered yaw so she curves cleanly away from the wall immediately
                worldStep = rotateStep(avatarWorldYaw)
                let steeredCandidate = SIMD3<Float>(avatarWorldPosition.x + worldStep.x, 0.0, avatarWorldPosition.z + worldStep.y)
                let steeredEval = clampAvatarPositionWithContacts(steeredCandidate, viewportAspect: viewportAspect)
                avatarWorldPosition = steeredEval.position
                lastBoundaryContacts = steeredEval.contacts
                lastIsNearBoundary = steeredEval.isNearBoundary
                return
            }
        }

        avatarWorldPosition = eval.position
        lastBoundaryContacts = eval.contacts
        lastIsNearBoundary = eval.isNearBoundary
    }

    /// Evaluates room bounds + unzoomed camera frustum guard and reports active wall contacts & inward normal.
    func clampAvatarPositionWithContacts(
        _ pos: SIMD3<Float>,
        viewportAspect: Float
    ) -> (position: SIMD3<Float>, contacts: [String], isNearBoundary: Bool, inwardNormal: SIMD2<Float>) {
        guard let cfg = config else {
            return (pos, [], false, .zero)
        }
        var p = SIMD3<Float>(pos.x, 0.0, pos.z)
        var contacts = Set<String>()
        var isNear = false
        var inward = SIMD2<Float>.zero
        let nearMargin: Float = 0.22

        // 1. Scene floor walk bounds
        if let minB = cfg.walkBoundsMin, minB.count == 2,
           let maxB = cfg.walkBoundsMax, maxB.count == 2 {
            if p.x <= minB[0] + 0.005 {
                p.x = minB[0]
                contacts.insert("left")
                inward.x += 1.0
            } else if p.x < minB[0] + nearMargin {
                isNear = true
                inward.x += 0.6
            }
            if p.x >= maxB[0] - 0.005 {
                p.x = maxB[0]
                contacts.insert("right")
                inward.x -= 1.0
            } else if p.x > maxB[0] - nearMargin {
                isNear = true
                inward.x -= 0.6
            }
            if p.z <= minB[1] + 0.005 {
                p.z = minB[1]
                contacts.insert("back")
                inward.y += 1.0
            } else if p.z < minB[1] + nearMargin {
                isNear = true
                inward.y += 0.6
            }
            if p.z >= maxB[1] - 0.005 {
                p.z = maxB[1]
                contacts.insert("front")
                inward.y -= 1.0
            } else if p.z > maxB[1] - nearMargin {
                isNear = true
                inward.y -= 0.6
            }
        }

        // 2. Unzoomed base camera frustum guard (never shrinks walkable room when user zooms in!)
        // Use an EXACT orthonormal 2D floor basis (`rightH = perp(fwdH)`) so projection/reconstruction
        // has ZERO cross-axis drift even when the camera has slight roll (`camUp.x != 0`).
        let baseUVScale = baseAspectFillUVScale(viewportAspect: viewportAspect)
        let imgTanHalfY = tan((cfg.fovYDegrees * .pi / 180.0) * 0.5)
        let imgAspect = Float(cfg.imageWidth) / max(Float(cfg.imageHeight), 1.0)
        let baseTanHalfX = imgTanHalfY * imgAspect * baseUVScale.x

        let camPos = cfg.camPos
        let fwd3 = simd_normalize(cfg.camTarget - camPos)
        let fwdH = simd_normalize(SIMD2<Float>(fwd3.x, fwd3.z))
        // Right-handed Y-up cross((fwdH.x, 0, fwdH.y), (0, 1, 0)) = (-fwdH.y, 0, fwdH.x) -> exact dot == 0
        let rightH = SIMD2<Float>(-fwdH.y, fwdH.x)

        let relH = SIMD2<Float>(p.x - camPos.x, p.z - camPos.z)
        var depth = simd_dot(relH, fwdH)
        var lateral = simd_dot(relH, rightH)
        var frustumClamped = false

        let minSafeDepth: Float = 1.55
        if depth <= minSafeDepth + 0.005 {
            depth = minSafeDepth
            frustumClamped = true
            contacts.insert("front")
            inward += fwdH
        } else if depth < minSafeDepth + nearMargin {
            isNear = true
            inward += fwdH * 0.6
        }

        let maxLateral = max(0.36, depth * baseTanHalfX * 0.72)
        if lateral <= -maxLateral + 0.005 {
            lateral = -maxLateral
            frustumClamped = true
            contacts.insert("left")
            inward += rightH
        } else if lateral < -maxLateral + nearMargin {
            isNear = true
            inward += rightH * 0.6
        }
        if lateral >= maxLateral - 0.005 {
            lateral = maxLateral
            frustumClamped = true
            contacts.insert("right")
            inward -= rightH
        } else if lateral > maxLateral - nearMargin {
            isNear = true
            inward -= rightH * 0.6
        }

        if frustumClamped {
            let clampedH = SIMD2<Float>(camPos.x, camPos.z) + fwdH * depth + rightH * lateral
            p.x = clampedH.x
            p.z = clampedH.y
        }

        if let minB = cfg.walkBoundsMin, minB.count == 2,
           let maxB = cfg.walkBoundsMax, maxB.count == 2 {
            p.x = min(max(p.x, minB[0]), maxB[0])
            p.z = min(max(p.z, minB[1]), maxB[1])
        }

        // Blend room-center vector so inward steering always points cleanly back into open floor
        let toCenter = SIMD2<Float>(cfg.spawnPos.x - p.x, cfg.spawnPos.z - p.z)
        if simd_length(toCenter) > 0.05 {
            inward += simd_normalize(toCenter) * 0.45
        }
        let inwardNorm = simd_length(inward) > 1e-4 ? simd_normalize(inward) : .zero
        return (p, contacts.sorted(), isNear || !contacts.isEmpty, inwardNorm)
    }

    // MARK: - LLM Spatial & Occlusion Telemetry

    /// Samples MoGe-2 metric depth across 5 vertical body points on Alice (`feet`, `knees`, `waist`,
    /// `chest`, `head`) to give future LLM systems exact position, occlusion, and boundary awareness.
    func evaluateSpatialSnapshot(viewportAspect: Float) -> SpatialSnapshot {
        guard let cfg = config, let mats = cameraMatrices(viewportAspect: viewportAspect) else {
            return SpatialSnapshot(
                normalizedRoomPosition: [0, 0],
                distanceToCameraMeters: 2.4,
                screenNDC: [0, 0],
                isOccluded: false,
                occlusionRatio: 0,
                occludedRegions: [],
                boundaryContacts: [],
                isNearBoundary: false
            )
        }

        // Normalized room coords in [-1, 1]
        var nx: Float = 0
        var nz: Float = 0
        if let minB = cfg.walkBoundsMin, minB.count == 2,
           let maxB = cfg.walkBoundsMax, maxB.count == 2 {
            let cx = 0.5 * (minB[0] + maxB[0])
            let cz = 0.5 * (minB[1] + maxB[1])
            let hx = max(0.1, 0.5 * (maxB[0] - minB[0]))
            let hz = max(0.1, 0.5 * (maxB[1] - minB[1]))
            nx = min(max((avatarWorldPosition.x - cx) / hx, -1), 1)
            nz = min(max((avatarWorldPosition.z - cz) / hz, -1), 1)
        }

        let camPos = cfg.camPos
        let fwd = simd_normalize(cfg.camTarget - camPos)
        let right = simd_normalize(simd_cross(fwd, cfg.camUp))
        let up = simd_normalize(simd_cross(right, fwd))
        let imgTanHalfY = tan((cfg.fovYDegrees * .pi / 180.0) * 0.5)
        let imgAspect = Float(cfg.imageWidth) / max(Float(cfg.imageHeight), 1.0)
        let imgTanHalfX = imgTanHalfY * imgAspect

        let torsoWorld = avatarWorldPosition + SIMD3<Float>(0, 0.95, 0)
        let distToCam = simd_length(torsoWorld - camPos)
        let vp = mats.projection * mats.view
        let clipTorso = vp * SIMD4<Float>(torsoWorld, 1.0)
        let ndcX = clipTorso.w > 1e-4 ? (clipTorso.x / clipTorso.w) : 0
        let ndcY = clipTorso.w > 1e-4 ? (clipTorso.y / clipTorso.w) : 0

        let sampleHeights: [(name: String, y: Float)] = [
            ("feet", 0.12),
            ("knees", 0.45),
            ("waist", 0.82),
            ("chest", 1.15),
            ("head", 1.42),
        ]

        var occludedNames: [String] = []
        if depthMeters.count >= cfg.depthWidth * cfg.depthHeight {
            for sample in sampleHeights {
                let pt = avatarWorldPosition + SIMD3<Float>(0, sample.y, 0)
                let rel = pt - camPos
                let ptDepth = simd_dot(rel, fwd)
                guard ptDepth > cfg.nearZ else { continue }
                let rx = simd_dot(rel, right) / ptDepth
                let ry = simd_dot(rel, up) / ptDepth
                let u = 0.5 * (rx / imgTanHalfX + 1.0)
                let v = 0.5 * (1.0 - ry / imgTanHalfY)
                if u >= 0 && u <= 1 && v >= 0 && v <= 1 {
                    let px = min(cfg.depthWidth - 1, max(0, Int(u * Float(cfg.depthWidth))))
                    let py = min(cfg.depthHeight - 1, max(0, Int(v * Float(cfg.depthHeight))))
                    let sceneDepth = depthMeters[py * cfg.depthWidth + px] + depthBiasMeters
                    if sceneDepth > 0.05 && sceneDepth < ptDepth - 0.06 {
                        occludedNames.append(sample.name)
                    }
                }
            }
        }

        let ratio = Float(occludedNames.count) / Float(sampleHeights.count)
        let contactEval = clampAvatarPositionWithContacts(avatarWorldPosition, viewportAspect: viewportAspect)
        let mergedContacts = Array(Set(lastBoundaryContacts).union(contactEval.contacts)).sorted()

        return SpatialSnapshot(
            normalizedRoomPosition: [nx, nz],
            distanceToCameraMeters: distToCam,
            screenNDC: [ndcX, ndcY],
            isOccluded: !occludedNames.isEmpty,
            occlusionRatio: ratio,
            occludedRegions: occludedNames,
            boundaryContacts: mergedContacts,
            isNearBoundary: lastIsNearBoundary || contactEval.isNearBoundary
        )
    }

    /// Computes exact off-axis perspective projection + view matrices locked sub-pixel to `bg.png` + `depth.bin`
    /// under any viewport aspect ratio, canvas zoom (`1.0...3.0`), and canvas pan.
    func cameraMatrices(viewportAspect: Float) -> (view: matrix_float4x4, projection: matrix_float4x4, uvScale: SIMD2<Float>, uvOffset: SIMD2<Float>, imgTanHalfFov: SIMD2<Float>)? {
        guard let cfg = config else { return nil }
        let imgAspect = Float(cfg.imageWidth) / max(Float(cfg.imageHeight), 1.0)
        let imgTanHalfY = tan((cfg.fovYDegrees * .pi / 180.0) * 0.5)
        let imgTanHalfX = imgTanHalfY * imgAspect

        let (uvScale, uvOffset) = currentUVWindow(viewportAspect: viewportAspect)
        let u0 = uvOffset.x
        let u1 = uvOffset.x + uvScale.x
        let v0 = uvOffset.y
        let v1 = uvOffset.y + uvScale.y

        // Off-axis Metal [0, 1] frustum matching visible image UV rectangle [u0, u1] x [v0, v1]
        let left = (2.0 * u0 - 1.0) * imgTanHalfX
        let right = (2.0 * u1 - 1.0) * imgTanHalfX
        let top = (1.0 - 2.0 * v0) * imgTanHalfY
        let bottom = (1.0 - 2.0 * v1) * imgTanHalfY

        let near = cfg.nearZ
        let far = cfg.farZ
        let zs = far / (near - far)
        var proj = matrix_float4x4()
        proj.columns.0 = SIMD4<Float>(2.0 / max(right - left, 1e-5), 0, 0, 0)
        proj.columns.1 = SIMD4<Float>(0, 2.0 / max(top - bottom, 1e-5), 0, 0)
        proj.columns.2 = SIMD4<Float>((right + left) / max(right - left, 1e-5), (top + bottom) / max(top - bottom, 1e-5), zs, -1)
        proj.columns.3 = SIMD4<Float>(0, 0, zs * near, 0)

        let view = OrbitCamera.lookAt(eye: cfg.camPos, center: cfg.camTarget, up: cfg.camUp)
        return (view, proj, uvScale, uvOffset, SIMD2<Float>(imgTanHalfX, imgTanHalfY))
    }

    /// Encodes the background color + metric hardware depth prepass before `VRMRenderer.draw`.
    func encodePrepass(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        depthTexture targetDepth: MTLTexture,
        viewportAspect: Float
    ) {
        guard let cfg = config,
              let bgTex = bgTexture,
              let depTex = depthTexture,
              let mats = cameraMatrices(viewportAspect: viewportAspect) else { return }

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = colorTexture
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        rpd.depthAttachment.texture = targetDepth
        rpd.depthAttachment.loadAction = .clear
        rpd.depthAttachment.storeAction = .store
        rpd.depthAttachment.clearDepth = 1.0

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) else { return }

        let fwd = simd_normalize(cfg.camTarget - cfg.camPos)
        let right = simd_normalize(simd_cross(fwd, cfg.camUp))
        let up = simd_normalize(simd_cross(right, fwd))

        var uniforms = Uniforms(
            uvScale: mats.uvScale,
            uvOffset: mats.uvOffset,
            nearZ: cfg.nearZ,
            farZ: cfg.farZ,
            imgTanHalfFovX: mats.imgTanHalfFov.x,
            imgTanHalfFovY: mats.imgTanHalfFov.y,
            camPos: SIMD4<Float>(cfg.camPos, depthBiasMeters),
            camRight: SIMD4<Float>(right, 0),
            camUp: SIMD4<Float>(up, 0),
            camFwd: SIMD4<Float>(fwd, 0),
            avatarFootPos: SIMD4<Float>(avatarWorldPosition, 1),
            shadowParams: SIMD4<Float>(0.32, 0.42, 0.20, 1.0)
        )

        let pso = (colorTexture.sampleCount > 1) ? pipelineMSAA : pipelineSingle
        encoder.setRenderPipelineState(pso)
        encoder.setDepthStencilState(depthState)
        encoder.setFragmentTexture(bgTex, index: 0)
        encoder.setFragmentTexture(depTex, index: 1)
        encoder.setFragmentSamplerState(linearSampler, index: 0)
        encoder.setFragmentSamplerState(nearestSampler, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    // MARK: - Texture Loaders

    private func loadBGRA8Texture(from url: URL) throws -> MTLTexture {
        guard let nsImage = NSImage(contentsOf: url),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 10, userInfo: [NSLocalizedDescriptionKey: "Failed to decode \(url.path)"])
        }
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerRow = width * 4
        var raw = Data(count: bytesPerRow * height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        let ok = raw.withUnsafeMutableBytes { buf -> Bool in
            guard let ptr = buf.baseAddress,
                  let ctx = CGContext(
                      data: ptr,
                      width: width,
                      height: height,
                      bitsPerComponent: 8,
                      bytesPerRow: bytesPerRow,
                      space: colorSpace,
                      bitmapInfo: bitmapInfo
                  ) else { return false }
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard ok else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 11, userInfo: [NSLocalizedDescriptionKey: "Failed to rasterize \(url.path)"])
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        guard let tex = device.makeTexture(descriptor: desc) else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 12, userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal bg texture"])
        }
        raw.withUnsafeBytes { buf in
            guard let ptr = buf.baseAddress else { return }
            tex.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: ptr,
                bytesPerRow: bytesPerRow
            )
        }
        return tex
    }

    private func loadDepthFloat32Texture(from url: URL, width: Int, height: Int) throws -> (MTLTexture, [Float]) {
        let data = try Data(contentsOf: url)
        let expectedCount = width * height
        let expectedBytes = expectedCount * MemoryLayout<Float>.stride
        guard data.count >= expectedBytes else {
            throw NSError(
                domain: "AlicePet.PictureWorld",
                code: 13,
                userInfo: [NSLocalizedDescriptionKey: "depth.bin size mismatch: got \(data.count), expected \(expectedBytes)"]
            )
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead]
        guard let tex = device.makeTexture(descriptor: desc) else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 14, userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal depth texture"])
        }
        var cpuDepth = [Float](repeating: 0, count: expectedCount)
        data.withUnsafeBytes { buf in
            guard let ptr = buf.baseAddress else { return }
            tex.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: ptr,
                bytesPerRow: width * MemoryLayout<Float>.stride
            )
            let floatPtr = ptr.bindMemory(to: Float.self, capacity: expectedCount)
            cpuDepth.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress?.update(from: floatPtr, count: expectedCount)
            }
        }
        return (tex, cpuDepth)
    }

    // MARK: - Inline Metal Shaders

    private static let metalShaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct PictureWorldUniforms {
        float2 uvScale;
        float2 uvOffset;
        float nearZ;
        float farZ;
        float imgTanHalfFovX;
        float imgTanHalfFovY;
        float4 camPos;       // xyz = camPos, w = depthBiasMeters
        float4 camRight;
        float4 camUp;
        float4 camFwd;
        float4 avatarFootPos;
        float4 shadowParams; // x=radius, y=strength, z=yTol, w=enabled
    };

    struct VSOut {
        float4 position [[position]];
        float2 screenUV;
    };

    struct FSOut {
        float4 color [[color(0)]];
        float depth [[depth(any)]];
    };

    vertex VSOut picture_world_vs(uint vid [[vertex_id]]) {
        float2 pos[3] = {
            float2(-1.0, -1.0),
            float2( 3.0, -1.0),
            float2(-1.0,  3.0)
        };
        float2 p = pos[vid];
        VSOut out;
        out.position = float4(p, 0.0, 1.0);
        out.screenUV = float2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
        return out;
    }

    fragment FSOut picture_world_fs(
        VSOut in [[stage_in]],
        texture2d<float> bgTex [[texture(0)]],
        texture2d<float> depthTex [[texture(1)]],
        sampler linSampler [[sampler(0)]],
        sampler nearSampler [[sampler(1)]],
        constant PictureWorldUniforms& u [[buffer(0)]]
    ) {
        float2 uv = in.screenUV * u.uvScale + u.uvOffset;
        float4 col = bgTex.sample(linSampler, uv);
        float d = depthTex.sample(nearSampler, uv).r + u.camPos.w;
        d = clamp(d, u.nearZ + 0.001, u.farZ - 0.001);

        // Exact Metal [0, 1] perspective depth matching our off-axis projection matrix
        float zNdc = (u.farZ / (u.farZ - u.nearZ)) * (1.0 - (u.nearZ / d));

        // Soft floor contact shadow grounded at avatarFootPos using exact image-UV ray reconstruction
        if (u.shadowParams.w > 0.5) {
            float rayX = (uv.x * 2.0 - 1.0) * u.imgTanHalfFovX;
            float rayY = (1.0 - uv.y * 2.0) * u.imgTanHalfFovY;
            float3 rayDir = u.camFwd.xyz
                + u.camRight.xyz * rayX
                + u.camUp.xyz * rayY;
            float3 worldP = u.camPos.xyz + rayDir * d;
            float dy = abs(worldP.y - u.avatarFootPos.y);
            if (dy < u.shadowParams.z) {
                float2 dXZ = worldP.xz - u.avatarFootPos.xz;
                float r2 = dot(dXZ, dXZ) / max(u.shadowParams.x * u.shadowParams.x, 1e-4);
                float yAtten = 1.0 - smoothstep(0.0, u.shadowParams.z, dy);
                float shadow = exp(-r2 * 2.4) * u.shadowParams.y * yAtten;
                col.rgb *= (1.0 - shadow);
            }
        }

        FSOut out;
        out.color = float4(col.rgb, 1.0);
        out.depth = clamp(zNdc, 0.0, 0.99999);
        return out;
    }
    """
}
