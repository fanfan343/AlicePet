import Foundation
import Metal
import MetalKit
import GLTFMetalKit
import simd

/// Renders standalone glTF 2.0 (.glb) props (Book & Afternoon Tea Items) via GLTFMetalKit
/// into the active MTKView / offscreen framebuffer, preserving hardware depth occlusion with Alice.
@MainActor
final class PropSceneRenderer {
    enum ActivePropMode: String, CaseIterable {
        case none
        case bookOnly
        case teaSetOnly
        case bookAndTea
    }

    private let device: MTLDevice
    private let gltfRenderer: GLTFRenderer
    private let pipelineStatesMSAA: GLTFRenderer.PipelineStates
    private let pipelineStatesSingle: GLTFRenderer.PipelineStates
    private let depthState: MTLDepthStencilState

    private(set) var bookAsset: GLTFAsset?
    private(set) var teaItemsAsset: GLTFAsset?

    var activePropMode: ActivePropMode = .none

    init(device: MTLDevice, colorFormat: MTLPixelFormat = .bgra8Unorm, depthFormat: MTLPixelFormat = .depth32Float, sampleCount: Int = 4) throws {
        self.device = device
        self.gltfRenderer = try GLTFRenderer(device: device)
        self.pipelineStatesMSAA = try gltfRenderer.makePipelineStates(
            colorFormat: colorFormat,
            depthFormat: depthFormat,
            sampleCount: max(sampleCount, 1)
        )
        self.pipelineStatesSingle = try gltfRenderer.makePipelineStates(
            colorFormat: colorFormat,
            depthFormat: depthFormat,
            sampleCount: 1
        )

        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .lessEqual
        depthDesc.isDepthWriteEnabled = true
        guard let ds = device.makeDepthStencilState(descriptor: depthDesc) else {
            throw NSError(domain: "AlicePet", code: 20, userInfo: [NSLocalizedDescriptionKey: "Failed to create GLTF depth state"])
        }
        self.depthState = ds
    }

    func loadProps(from folderURL: URL) async {
        let loader = GLTFAssetLoader()
        let bookURL = folderURL.appendingPathComponent("Alice_Item_Book.glb")
        let itemsURL = folderURL.appendingPathComponent("Alice_Items.glb")

        if FileManager.default.fileExists(atPath: bookURL.path) {
            do {
                let asset = try await loader.load(from: bookURL, device: device)
                self.bookAsset = asset
                AppLogger.shared.info("Props", "Loaded Alice_Item_Book.glb (drawCalls=\(asset.drawCalls.count), bounds=\(asset.worldBounds))")
            } catch {
                AppLogger.shared.error("Props", "Failed to load Alice_Item_Book.glb: \(error)")
            }
        }
        if FileManager.default.fileExists(atPath: itemsURL.path) {
            do {
                let asset = try await loader.load(from: itemsURL, device: device)
                self.teaItemsAsset = asset
                AppLogger.shared.info("Props", "Loaded Alice_Items.glb (drawCalls=\(asset.drawCalls.count), anims=\(asset.animations.count), bounds=\(asset.worldBounds))")
            } catch {
                AppLogger.shared.error("Props", "Failed to load Alice_Items.glb: \(error)")
            }
        }
    }

    var hasActiveProps: Bool {
        switch activePropMode {
        case .none: return false
        case .bookOnly: return bookAsset != nil
        case .teaSetOnly: return teaItemsAsset != nil
        case .bookAndTea: return bookAsset != nil || teaItemsAsset != nil
        }
    }

    /// Encodes a second pass on top of VRMRenderer output, loading existing color & depth.
    func drawProps(
        commandBuffer: MTLCommandBuffer,
        colorTexture: MTLTexture,
        resolveTexture: MTLTexture?,
        depthTexture: MTLTexture,
        viewProjection: simd_float4x4,
        cameraPosition: SIMD3<Float>,
        bookWorldMatrix: simd_float4x4,
        time: Float,
        lightingPreset: LightingPreset = .animeSoftDay
    ) {
        guard hasActiveProps else { return }

        var drawCalls: [GLTFDrawCall] = []

        if activePropMode == .bookOnly || activePropMode == .bookAndTea, let bookAsset {
            for call in bookAsset.drawCalls {
                drawCalls.append(GLTFDrawCall(
                    mesh: call.mesh,
                    material: call.material,
                    modelMatrix: bookWorldMatrix * call.modelMatrix,
                    skinPalette: call.skinPalette
                ))
            }
        }

        if activePropMode == .teaSetOnly || activePropMode == .bookAndTea, let teaItemsAsset {
            let calls: [GLTFDrawCall]
            if let clip = teaItemsAsset.animations.first, clip.duration > 0 {
                let t = fmodf(max(0, time), clip.duration)
                calls = teaItemsAsset.drawCalls(animationIndex: 0, time: t)
            } else {
                calls = teaItemsAsset.drawCalls
            }
            drawCalls.append(contentsOf: calls)
        }

        guard !drawCalls.isEmpty else { return }

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = colorTexture
        rpd.colorAttachments[0].loadAction = .load
        if let resolveTexture {
            rpd.colorAttachments[0].resolveTexture = resolveTexture
            rpd.colorAttachments[0].storeAction = .multisampleResolve
        } else {
            rpd.colorAttachments[0].storeAction = .store
        }

        rpd.depthAttachment.texture = depthTexture
        rpd.depthAttachment.loadAction = .load
        rpd.depthAttachment.storeAction = .dontCare

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) else { return }
        let propLights = LightingConfigurator.propLights(for: lightingPreset)
        let scene = GLTFSceneState(
            viewProjection: viewProjection,
            cameraPosition: cameraPosition,
            lightDirection: normalize(SIMD3<Float>(-0.25, -0.55, -0.80)),
            lightColor: SIMD3<Float>(2.2, 2.15, 2.10),
            lights: propLights
        )
        let pso = (colorTexture.sampleCount > 1) ? pipelineStatesMSAA : pipelineStatesSingle
        gltfRenderer.encodeOpaqueDrawCalls(
            drawCalls,
            scene: scene,
            pipelineStates: pso,
            depthState: depthState,
            encoder: encoder
        )
        encoder.endEncoding()
    }
}
