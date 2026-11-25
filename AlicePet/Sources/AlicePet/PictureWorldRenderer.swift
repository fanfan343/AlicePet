import AppKit
import CoreGraphics
import Foundation
import Metal
import MetalKit
import simd

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
        var tanHalfFovX: Float
        var tanHalfFovY: Float
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
    private var bgTexture: MTLTexture?
    private var depthTexture: MTLTexture?

    /// Live avatar world offset on the room floor (Y=0 is the fitted floor plane).
    var avatarWorldPosition: SIMD3<Float> = .zero
    var avatarWorldYaw: Float = 0.0
    var depthBiasMeters: Float = 0.0

    /// Anchor tracking for ARDY continuous 3D root locomotion inside the room.
    private var lastARDYRootXZ: SIMD2<Float>?

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
        let depthTex = try loadDepthFloat32Texture(from: depthURL, width: cfg.depthWidth, height: cfg.depthHeight)

        self.config = cfg
        self.sceneFolderURL = folderURL
        self.bgTexture = bgTex
        self.depthTexture = depthTex
        self.avatarWorldPosition = cfg.spawnPos
        self.avatarWorldYaw = cfg.avatarSpawnYawRad
        self.lastARDYRootXZ = nil

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
        avatarWorldPosition = .zero
        avatarWorldYaw = 0.0
        lastARDYRootXZ = nil
        AppLogger.shared.info("PictureWorld", "Unloaded Picture-World scene (0 scene VRAM)")
    }

    func resetARDYRootTracking() {
        lastARDYRootXZ = nil
    }

    /// Integrates continuous ARDY root XZ displacement (meters) onto the Picture-World floor while
    /// enforcing BOTH the scene `walkBounds` and live camera-frustum visibility guard so Alice
    /// genuinely walks around the 3D room without ever leaving the frame or clipping the camera lens.
    func applyARDYRootMotion(worldRootXZ: SIMD2<Float>, viewportAspect: Float) {
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

        // Rotate ARDY delta by spawn yaw so footfalls match Alice's world facing direction
        let cy = cos(avatarWorldYaw)
        let sy = sin(avatarWorldYaw)
        let dx = rawDelta.x * cy + rawDelta.y * sy
        let dz = -rawDelta.x * sy + rawDelta.y * cy

        let candidate = SIMD3<Float>(avatarWorldPosition.x + dx, 0.0, avatarWorldPosition.z + dz)
        avatarWorldPosition = clampAvatarPositionToSafeRoom(candidate, viewportAspect: viewportAspect)
    }

    /// Clamps any world floor coordinate inside `walkBounds` AND inside the camera's visible frustum
    /// (keeping a comfortable horizontal margin and >=1.55m clearance in front of the lens).
    func clampAvatarPositionToSafeRoom(_ pos: SIMD3<Float>, viewportAspect: Float) -> SIMD3<Float> {
        guard let cfg = config else { return pos }
        var p = SIMD3<Float>(pos.x, 0.0, pos.z)

        // 1. Scene floor walk bounds
        if let minB = cfg.walkBoundsMin, minB.count == 2,
           let maxB = cfg.walkBoundsMax, maxB.count == 2 {
            p.x = min(max(p.x, minB[0]), maxB[0])
            p.z = min(max(p.z, minB[1]), maxB[1])
        }

        // 2. Live camera frustum guard (projected onto floor plane)
        guard let mats = cameraMatrices(viewportAspect: viewportAspect) else { return p }
        let camPos = cfg.camPos
        let fwd3 = simd_normalize(cfg.camTarget - camPos)
        let right3 = simd_normalize(simd_cross(fwd3, cfg.camUp))
        let fwdH = simd_normalize(SIMD2<Float>(fwd3.x, fwd3.z))
        let rightH = simd_normalize(SIMD2<Float>(right3.x, right3.z))

        let relH = SIMD2<Float>(p.x - camPos.x, p.z - camPos.z)
        var depth = simd_dot(relH, fwdH)
        var lateral = simd_dot(relH, rightH)

        // Never walk closer than 1.55m to the camera lens (avoids head clipping off top of frame)
        let minSafeDepth: Float = 1.55
        if depth < minSafeDepth {
            depth = minSafeDepth
        }

        // Keep full body well inside left/right screen edges (68% of half-frustum width at current depth)
        let tanHalfY = tan(mats.effectiveFovYRad * 0.5)
        let tanHalfX = tanHalfY * max(viewportAspect, 0.25)
        let maxLateral = max(0.32, depth * tanHalfX * 0.68)
        lateral = min(max(lateral, -maxLateral), maxLateral)

        let clampedH = SIMD2<Float>(camPos.x, camPos.z) + fwdH * depth + rightH * lateral
        p.x = clampedH.x
        p.z = clampedH.y

        // Re-apply room walk bounds after frustum projection
        if let minB = cfg.walkBoundsMin, minB.count == 2,
           let maxB = cfg.walkBoundsMax, maxB.count == 2 {
            p.x = min(max(p.x, minB[0]), maxB[0])
            p.z = min(max(p.z, minB[1]), maxB[1])
        }
        return p
    }

    /// Computes the exact camera view + projection matrices locked to `bg.png` + `depth.bin`
    /// under aspect-fill cropping for any viewport aspect ratio.
    func cameraMatrices(viewportAspect: Float) -> (view: matrix_float4x4, projection: matrix_float4x4, uvScale: SIMD2<Float>, uvOffset: SIMD2<Float>, effectiveFovYRad: Float)? {
        guard let cfg = config else { return nil }
        let safeAspect = max(viewportAspect, 0.1)
        let imgAspect = Float(cfg.imageWidth) / max(Float(cfg.imageHeight), 1.0)

        let baseFovYRad = cfg.fovYDegrees * .pi / 180.0
        let baseTanHalfY = tan(baseFovYRad * 0.5)

        let uvScale: SIMD2<Float>
        let uvOffset: SIMD2<Float>
        let effectiveTanHalfY: Float

        if safeAspect >= imgAspect {
            // Viewport is wider than image: crop top/bottom equally
            let visibleVFrac = imgAspect / safeAspect
            uvScale = SIMD2<Float>(1.0, visibleVFrac)
            uvOffset = SIMD2<Float>(0.0, 0.5 * (1.0 - visibleVFrac))
            effectiveTanHalfY = baseTanHalfY * visibleVFrac
        } else {
            // Viewport is narrower than image: crop left/right equally, vertical FOV unchanged
            let visibleUFrac = safeAspect / imgAspect
            uvScale = SIMD2<Float>(visibleUFrac, 1.0)
            uvOffset = SIMD2<Float>(0.5 * (1.0 - visibleUFrac), 0.0)
            effectiveTanHalfY = baseTanHalfY
        }

        let effectiveFovYRad = 2.0 * atan(effectiveTanHalfY)
        let view = OrbitCamera.lookAt(eye: cfg.camPos, center: cfg.camTarget, up: cfg.camUp)
        let proj = OrbitCamera.perspectiveMetal(
            fovRadians: effectiveFovYRad,
            aspect: safeAspect,
            near: cfg.nearZ,
            far: cfg.farZ
        )
        return (view, proj, uvScale, uvOffset, effectiveFovYRad)
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
        let tanHalfY = tan(mats.effectiveFovYRad * 0.5)
        let tanHalfX = tanHalfY * max(viewportAspect, 0.1)

        var uniforms = Uniforms(
            uvScale: mats.uvScale,
            uvOffset: mats.uvOffset,
            nearZ: cfg.nearZ,
            farZ: cfg.farZ,
            tanHalfFovX: tanHalfX,
            tanHalfFovY: tanHalfY,
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

    private func loadDepthFloat32Texture(from url: URL, width: Int, height: Int) throws -> MTLTexture {
        let data = try Data(contentsOf: url)
        let expectedBytes = width * height * MemoryLayout<Float>.stride
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
        data.withUnsafeBytes { buf in
            guard let ptr = buf.baseAddress else { return }
            tex.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: ptr,
                bytesPerRow: width * MemoryLayout<Float>.stride
            )
        }
        return tex
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
        float tanHalfFovX;
        float tanHalfFovY;
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

        // Exact Metal [0, 1] perspective depth matching OrbitCamera.perspectiveMetal
        float zNdc = (u.farZ / (u.farZ - u.nearZ)) * (1.0 - (u.nearZ / d));

        // Soft floor contact shadow grounded at avatarFootPos using reconstructed world point
        if (u.shadowParams.w > 0.5) {
            float ndcX = in.screenUV.x * 2.0 - 1.0;
            float ndcY = 1.0 - in.screenUV.y * 2.0;
            float3 rayDir = u.camFwd.xyz
                + u.camRight.xyz * (ndcX * u.tanHalfFovX)
                + u.camUp.xyz * (ndcY * u.tanHalfFovY);
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
