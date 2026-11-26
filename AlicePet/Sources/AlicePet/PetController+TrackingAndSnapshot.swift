import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers
import VRMMetalKit
import simd

extension PetController {
    // MARK: - Timers & VMC / Local Subprocess Drivers

    func startTimers() {
        expressionMixer.startAutoBlink(
            rendererProvider: { [weak self] in self?.renderer },
            isCameraActive: { [weak self] in self?.animationMixer.isEyeAndBlinkCaptured ?? false }
        )
    }

    @discardableResult
    func startVMC(port: UInt16 = VMCTrackingController.defaultPort) -> Bool {
        expressionMixer.prepareForTrackingStart(on: renderer)
        animationMixer.preferredFullBodySource = .externalVMC
        return animationMixer.externalChannel.start(port: port)
    }

    func stopVMC() {
        localMotionService.stop()
        let wasActive = animationMixer.externalChannel.isListening
        animationMixer.externalChannel.stop(renderer: renderer, expressionMixer: expressionMixer)
        if wasActive && !animationMixer.isFaceCaptured {
            expressionMixer.resetAfterTrackingStop(on: renderer)
        }
        if wasActive && !localArdyActive {
            applyAllBodyAndOutfitState(settlePhysics: false)
        }
    }

    @discardableResult
    func startLocalMotionCapture(port: UInt16 = VMCTrackingController.defaultPort) -> Bool {
        if !vmcActive || vmcController.port != port {
            guard startVMC(port: port) else { return false }
        } else {
            animationMixer.preferredFullBodySource = .externalVMC
        }
        let ok = localMotionService.start(port: vmcController.port)
        if !ok {
            stopVMC()
        }
        return ok
    }

    func stopLocalMotionCapture() {
        stopVMC()
    }

    @discardableResult
    func startAudioFaceTest(audioURL: URL? = nil, maxSeconds: Double = 0) -> Bool {
        if !animationMixer.audioFaceChannel.isListening {
            guard animationMixer.audioFaceChannel.start(port: 0) else { return false }
        }
        guard let boundPort = animationMixer.audioFaceChannel.boundPort else { return false }
        if localAudioFaceService.params.engine == .a2fClaire {
            expressionMixer.prepareForTrackingStart(on: renderer)
        }
        let ok = localAudioFaceService.start(audioURL: audioURL, port: boundPort, maxSeconds: maxSeconds)
        if !ok {
            animationMixer.audioFaceChannel.stop(renderer: renderer, expressionMixer: expressionMixer)
        }
        return ok
    }

    func stopAudioFaceTest() {
        localAudioFaceService.stop()
        let wasListening = animationMixer.audioFaceChannel.isListening
        animationMixer.audioFaceChannel.stop(renderer: renderer, expressionMixer: expressionMixer)
        if wasListening && !animationMixer.isFaceCaptured {
            expressionMixer.resetAfterTrackingStop(on: renderer)
        }
    }

    @discardableResult
    func startArdyMotionTest() -> Bool {
        if !animationMixer.ardyChannel.isListening {
            guard animationMixer.ardyChannel.start(port: 0) else { return false }
        }
        guard let boundPort = animationMixer.ardyChannel.boundPort else { return false }
        animationMixer.preferredFullBodySource = .ardy
        let ok = localArdyService.start(port: boundPort)
        if !ok {
            animationMixer.ardyChannel.stop()
        }
        return ok
    }

    func stopArdyMotionTest() {
        localArdyService.stop()
        animationMixer.ardyChannel.stop()
    }

    func stopAllTrackingSync() {
        localMotionService.stopSync()
        localAudioFaceService.stopSync()
        localArdyService.stopSync()
        animationMixer.stopAllChannels()
    }

    // MARK: - Offscreen Snapshot (CLI verification)

    func renderSnapshot(
        to outputURL: URL,
        width: Int = 720,
        height: Int = 1024,
        animTime: Float = 0.8,
        applyBaseState: Bool = true
    ) async throws {
        if applyBaseState {
            applyAllBodyAndOutfitState(settlePhysics: true, animTime: animTime)
        }
        if pictureWorldRenderer.isActive {
            applyPictureWorldRootTransform()
        }

        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        colorDesc.usage = [.renderTarget, .shaderRead]
        colorDesc.storageMode = .shared

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        depthDesc.usage = [.renderTarget]
        depthDesc.storageMode = .shared

        guard let colorTex = device.makeTexture(descriptor: colorDesc),
              let depthTex = device.makeTexture(descriptor: depthDesc) else {
            throw NSError(domain: "AlicePet", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to allocate offscreen textures"])
        }

        let aspect = Float(width) / Float(height)
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = colorTex
        rpd.colorAttachments[0].loadAction = pictureWorldRenderer.isActive ? .load : .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

        rpd.depthAttachment.texture = depthTex
        rpd.depthAttachment.loadAction = pictureWorldRenderer.isActive ? .load : .clear
        rpd.depthAttachment.storeAction = propRenderer.hasActiveProps ? .store : .dontCare
        rpd.depthAttachment.clearDepth = 1.0

        if let pwMats = pictureWorldRenderer.cameraMatrices(viewportAspect: aspect) {
            renderer.projectionMatrix = pwMats.projection
            renderer.viewMatrix = pwMats.view
        } else {
            renderer.projectionMatrix = camera.projectionMatrix(aspect: aspect)
            renderer.viewMatrix = camera.viewMatrix
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw NSError(domain: "AlicePet", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer"])
        }

        if pictureWorldRenderer.isActive {
            pictureWorldRenderer.encodePrepass(
                commandBuffer: commandBuffer,
                colorTexture: colorTex,
                depthTexture: depthTex,
                viewportAspect: aspect
            )
        }

        renderer.drawOffscreenHeadless(
            to: colorTex,
            depth: depthTex,
            commandBuffer: commandBuffer,
            renderPassDescriptor: rpd
        )

        if propRenderer.hasActiveProps {
            propRenderer.drawProps(
                commandBuffer: commandBuffer,
                colorTexture: colorTex,
                resolveTexture: nil,
                depthTexture: depthTex,
                viewProjection: renderer.projectionMatrix * renderer.viewMatrix,
                cameraPosition: camera.eyePosition,
                bookWorldMatrix: poseController.computeBookWorldMatrix(time: animTime, profile: activeProfile),
                time: animTime,
                lightingPreset: lightingPreset
            )
        }

        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }

        try exportBGRA8Texture(colorTex, to: outputURL)
    }

    private func exportBGRA8Texture(_ texture: MTLTexture, to url: URL) throws {
        let width = texture.width
        let height = texture.height
        let bytesPerRow = width * 4
        var data = Data(count: bytesPerRow * height)
        data.withUnsafeMutableBytes { buf in
            guard let ptr = buf.baseAddress else { return }
            texture.getBytes(ptr, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        guard let provider = CGDataProvider(data: data as CFData) else {
            throw NSError(domain: "AlicePet", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGDataProvider"])
        }
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw NSError(domain: "AlicePet", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGImage"])
        }
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw NSError(domain: "AlicePet", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to create image destination"])
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw NSError(domain: "AlicePet", code: 7, userInfo: [NSLocalizedDescriptionKey: "Failed to write PNG"])
        }
    }
}
