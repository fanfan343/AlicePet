import AppKit
import CoreGraphics
import Foundation
import Metal
import MetalKit
import simd

/// Native Metal "Picture-World" renderer (Mode B):
/// Draws a single 2D illustration/photo (`bg.png`) and writes its metric depth (`depth.bin` from MoGe-2)
/// directly into the hardware `depth32Float` buffer before `VRMRenderer`, giving Alice native
/// 3D occlusion behind foreground furniture + floor contact shadow with near-zero GPU/RAM overhead.
@MainActor
final class PictureWorldRenderer {
    typealias SpatialSnapshot = PictureWorldSpatialSnapshot

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

    private let device: MTLDevice
    private let pipelineMSAA: MTLRenderPipelineState
    private let pipelineSingle: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let linearSampler: MTLSamplerState
    private let nearestSampler: MTLSamplerState

    private(set) var config: PictureWorldSceneConfig?
    private(set) var sceneFolderURL: URL?
    /// True when Picture-World is running in empty transparent backdrop mode (no 2D photo/depth map).
    private(set) var isEmptyTransparentMode: Bool = false
    /// Parsed walkable floor grid for the active scene (nil for hand-authored scenes without one).
    private(set) var roomGrid: WalkableGrid.ParsedGrid?
    private var bgTexture: MTLTexture?
    private var depthTexture: MTLTexture?
    /// CPU-side copy of MoGe-2 metric depth for O(1) occlusion & ray unprojection queries.
    var depthMeters: [Float] = []

    /// Live avatar world offset on the room floor (Y=0 is the fitted floor plane).
    var avatarWorldPosition: SIMD3<Float> = .zero
    var avatarWorldYaw: Float = 0.0
    var depthBiasMeters: Float = 0.0

    /// 2D canvas view zoom (1.0 = unzoomed fit, up to 3.0x) and pan in image UV space (clamped inside photo).
    private(set) var canvasZoom: Float = 1.0
    private(set) var canvasPanUV: SIMD2<Float> = .zero

    /// Anchor & boundary telemetry for ARDY continuous 3D root locomotion inside the room.
    var lastARDYRootXZ: SIMD2<Float>?
    var lastBoundaryContacts: [String] = []
    var lastIsNearBoundary: Bool = false

    var isActive: Bool {
        config != nil && (isEmptyTransparentMode || (bgTexture != nil && depthTexture != nil))
    }

    var hasOpaqueBackground: Bool {
        config != nil && !isEmptyTransparentMode && bgTexture != nil && depthTexture != nil
    }

    init(device: MTLDevice, colorFormat: MTLPixelFormat = .bgra8Unorm, depthFormat: MTLPixelFormat = .depth32Float, sampleCount: Int = 4) throws {
        self.device = device

        let library = try device.makeLibrary(source: PictureWorldShaders.metalSource, options: nil)
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
        self.isEmptyTransparentMode = false
        self.roomGrid = cfg.walkableGrid.map { WalkableGrid.ParsedGrid(grid: $0) }
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

    /// Activates Picture-World Mode 2 without a 2D image background (transparent desktop backdrop)
    /// so the user can observe ARDY free-roaming locomotion & gestures on an empty floor.
    func loadEmptyTransparentRoom() {
        let cfg = PictureWorldSceneConfig.emptyTransparentRoom()
        self.config = cfg
        self.sceneFolderURL = nil
        self.isEmptyTransparentMode = true
        self.roomGrid = nil
        self.bgTexture = nil
        self.depthTexture = nil
        self.depthMeters.removeAll(keepingCapacity: false)
        self.avatarWorldPosition = cfg.spawnPos
        self.avatarWorldYaw = cfg.avatarSpawnYawRad
        self.canvasZoom = 1.0
        self.canvasPanUV = .zero
        self.lastARDYRootXZ = nil
        self.lastBoundaryContacts = []
        self.lastIsNearBoundary = false
        AppLogger.shared.info("PictureWorld", "Entered empty transparent Picture-World room (0 scene VRAM)")
    }

    func unloadScene() {
        config = nil
        sceneFolderURL = nil
        isEmptyTransparentMode = false
        roomGrid = nil
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

    func baseAspectFillUVScale(viewportAspect: Float) -> SIMD2<Float> {
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

    // MARK: - Projection & Prepass

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
                      data: ptr, width: width, height: height, bitsPerComponent: 8,
                      bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo
                  ) else { return false }
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard ok else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 11, userInfo: [NSLocalizedDescriptionKey: "Failed to rasterize \(url.path)"])
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead]
        guard let tex = device.makeTexture(descriptor: desc) else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 12, userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal bg texture"])
        }
        raw.withUnsafeBytes { buf in
            guard let ptr = buf.baseAddress else { return }
            tex.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: ptr, bytesPerRow: bytesPerRow)
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
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: width, height: height, mipmapped: false)
        desc.usage = [.shaderRead]
        guard let tex = device.makeTexture(descriptor: desc) else {
            throw NSError(domain: "AlicePet.PictureWorld", code: 14, userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal depth texture"])
        }
        var cpuDepth = [Float](repeating: 0, count: expectedCount)
        data.withUnsafeBytes { buf in
            guard let ptr = buf.baseAddress else { return }
            tex.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: ptr, bytesPerRow: width * MemoryLayout<Float>.stride)
            let floatPtr = ptr.bindMemory(to: Float.self, capacity: expectedCount)
            cpuDepth.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress?.update(from: floatPtr, count: expectedCount)
            }
        }
        return (tex, cpuDepth)
    }
}
