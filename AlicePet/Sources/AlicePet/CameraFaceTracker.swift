@preconcurrency import AVFoundation
import CoreMedia
import Foundation
import ImageIO
import QuartzCore
import VRMMetalKit
import Vision
import simd

struct CameraDeviceInfo: Identifiable, Sendable {
    let id: String
    let name: String
    let isContinuityCamera: Bool
    let isBuiltIn: Bool

    var displayTitle: String {
        if isContinuityCamera {
            return "📱 \(name)（连续互通相机）"
        } else if isBuiltIn {
            return "💻 \(name)"
        } else {
            return "📷 \(name)"
        }
    }
}

struct TrackedFaceFrame: Sendable {
    let blendShapes: ARKitFaceBlendShapes
    let headYaw: Float
    let headPitch: Float
    let headRoll: Float
    let gazeOffset: SIMD2<Float>
}

final class CameraFaceTracker: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let session = AVCaptureSession()
    private let captureQueue = DispatchQueue(label: "com.funnyliu.AlicePet.cameraCapture", qos: .userInteractive)
    private let videoOutput = AVCaptureVideoDataOutput()

    private var currentDeviceID: String?
    private var isProcessingFrame = false
    private var baselineEyebrowRatio: Float?
    private var baselineMouthWidthRatio: Float?
    private var preferredOrientation: CGImagePropertyOrientation = .up
    private var frameCounter: Int = 0
    private var consecutiveNoFaceFrames: Int = 0

    var mirrorMode: Bool = true
    var onFrame: (@MainActor @Sendable (TrackedFaceFrame) -> Void)?
    var onTrackingStateChanged: (@MainActor @Sendable (Bool) -> Void)?
    var onError: (@MainActor @Sendable (String) -> Void)?
    var onDevicesChanged: (@MainActor @Sendable () -> Void)?

    override init() {
        super.init()
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleDeviceNotification),
            name: AVCaptureDevice.wasConnectedNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleDeviceNotification),
            name: AVCaptureDevice.wasDisconnectedNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleSessionRuntimeError),
            name: AVCaptureSession.runtimeErrorNotification,
            object: session
        )
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        session.stopRunning()
    }

    @objc private func handleDeviceNotification(_ notification: Notification) {
        let callback = onDevicesChanged
        Task { @MainActor in
            callback?()
        }
    }

    @objc private func handleSessionRuntimeError(_ notification: Notification) {
        let err = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError
        let msg = err?.localizedDescription ?? "摄像头会话意外中断"
        AppLogger.shared.error("Camera", "Session runtime error: \(msg)")
        let errCb = onError
        Task { @MainActor in
            errCb?("摄像头运行时异常：\(msg)")
        }
    }

    static func availableCameras() -> [CameraDeviceInfo] {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .continuityCamera, .external],
            mediaType: .video,
            position: .unspecified
        )
        return discovery.devices.map { dev in
            let isIPhone = dev.deviceType == .continuityCamera
                || dev.localizedName.localizedCaseInsensitiveContains("iPhone")
                || dev.localizedName.localizedCaseInsensitiveContains("iPad")
            return CameraDeviceInfo(
                id: dev.uniqueID,
                name: dev.localizedName,
                isContinuityCamera: isIPhone,
                isBuiltIn: dev.deviceType == .builtInWideAngleCamera
            )
        }
    }

    func start(deviceID: String) {
        Task { [weak self] in
            guard let self else { return }
            let status = AVCaptureDevice.authorizationStatus(for: .video)
            let granted: Bool
            switch status {
            case .authorized:
                granted = true
            case .notDetermined:
                granted = await AVCaptureDevice.requestAccess(for: .video)
            default:
                granted = false
            }

            guard granted else {
                AppLogger.shared.warn("Camera", "Camera permission denied (status=\(status.rawValue))")
                let errCb = self.onError
                await MainActor.run {
                    errCb?("未获得摄像头权限，请在「系统设置 → 隐私与安全性 → 相机」中允许 AlicePet 访问摄像头。")
                }
                return
            }

            AppLogger.shared.info("Camera", "Starting capture session for deviceID=\(deviceID)")
            self.captureQueue.async { [weak self] in
                self?.configureAndStartSession(deviceID: deviceID)
            }
        }
    }

    func stop() {
        captureQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
            }
            self.currentDeviceID = nil
            self.baselineEyebrowRatio = nil
            self.baselineMouthWidthRatio = nil
            self.frameCounter = 0
            self.consecutiveNoFaceFrames = 0
        }
    }

    private func configureAndStartSession(deviceID: String) {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .continuityCamera, .external],
            mediaType: .video,
            position: .unspecified
        )
        guard let device = discovery.devices.first(where: { $0.uniqueID == deviceID }) else {
            let errCb = onError
            Task { @MainActor in
                errCb?("找不到所选摄像头（可能已断开连接）。")
            }
            return
        }

        if session.isRunning {
            session.stopRunning()
        }

        session.beginConfiguration()

        for input in session.inputs {
            session.removeInput(input)
        }
        for output in session.outputs {
            session.removeOutput(output)
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
            }
        } catch {
            session.commitConfiguration()
            let errCb = onError
            let msg = error.localizedDescription
            Task { @MainActor in
                errCb?("无法打开摄像头 \(device.localizedName): \(msg)")
            }
            return
        }

        // Pick a preset supported by both built-in Mac cameras and iPhone Continuity Camera
        let preferredPresets: [AVCaptureSession.Preset] = [.hd1280x720, .vga640x480, .medium, .high]
        if let supported = preferredPresets.first(where: { session.canSetSessionPreset($0) }) {
            session.sessionPreset = supported
        }

        // Leave pixel format native so Vision processes Continuity Camera YUV buffers directly with zero conversion issues
        videoOutput.videoSettings = nil
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: captureQueue)
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

        session.commitConfiguration()
        baselineEyebrowRatio = nil
        baselineMouthWidthRatio = nil
        preferredOrientation = .up
        frameCounter = 0
        consecutiveNoFaceFrames = 0
        currentDeviceID = deviceID
        session.startRunning()
        AppLogger.shared.info("Camera", "Started session on '\(device.localizedName)' preset=\(session.sessionPreset.rawValue) running=\(session.isRunning)")
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard !isProcessingFrame else { return }
        isProcessingFrame = true
        defer { isProcessingFrame = false }

        frameCounter += 1
        if frameCounter == 1, let pb = CMSampleBufferGetImageBuffer(sampleBuffer) {
            let w = CVPixelBufferGetWidth(pb)
            let h = CVPixelBufferGetHeight(pb)
            AppLogger.shared.info("Camera", "First frame received: \(w)x\(h)")
        }

        guard let face = detectFaceWithAutoOrientation(in: sampleBuffer),
              let landmarks = face.landmarks else {
            consecutiveNoFaceFrames += 1
            if consecutiveNoFaceFrames == 15 {
                let stateCb = onTrackingStateChanged
                Task { @MainActor in stateCb?(false) }
            }
            return
        }

        if consecutiveNoFaceFrames >= 15 || frameCounter <= 2 {
            let stateCb = onTrackingStateChanged
            Task { @MainActor in stateCb?(true) }
        }
        consecutiveNoFaceFrames = 0

        let mirror = mirrorMode
        let rawYaw = face.yaw?.floatValue ?? 0
        let rawPitch = face.pitch?.floatValue ?? 0
        let rawRoll = face.roll?.floatValue ?? 0

        let headYaw = (mirror ? -rawYaw : rawYaw).clamped(to: -0.75...0.75)
        let headPitch = rawPitch.clamped(to: -0.55...0.55)
        let headRoll = (mirror ? -rawRoll : rawRoll).clamped(to: -0.55...0.55)

        var shapes: [String: Float] = [:]

        // 1. Eye blinks & wide (Left/Right swap in mirror mode)
        let rawLeftEAR = eyeAspectRatio(landmarks.leftEye)
        let rawRightEAR = eyeAspectRatio(landmarks.rightEye)
        let leftEAR = mirror ? rawRightEAR : rawLeftEAR
        let rightEAR = mirror ? rawLeftEAR : rawRightEAR

        if let ear = leftEAR {
            let blink = ((0.29 - ear) / 0.13).clamped(to: 0...1)
            let wide = ((ear - 0.36) / 0.10).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.eyeBlinkLeft] = blink
            shapes[ARKitFaceBlendShapes.eyeWideLeft] = wide
        }
        if let ear = rightEAR {
            let blink = ((0.29 - ear) / 0.13).clamped(to: 0...1)
            let wide = ((ear - 0.36) / 0.10).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.eyeBlinkRight] = blink
            shapes[ARKitFaceBlendShapes.eyeWideRight] = wide
        }

        // Inter-ocular scale reference
        let leftEyeCenter = centerPoint(landmarks.leftEye)
        let rightEyeCenter = centerPoint(landmarks.rightEye)
        let interOcular: Float
        if let l = leftEyeCenter, let r = rightEyeCenter {
            interOcular = max(Float(hypot(r.x - l.x, r.y - l.y)), 0.08)
        } else {
            interOcular = 0.32
        }

        // 2. Mouth opening, smile, pucker
        if let outer = landmarks.outerLips?.normalizedPoints, outer.count >= 6 {
            let minX = outer.map { Float($0.x) }.min() ?? 0
            let maxX = outer.map { Float($0.x) }.max() ?? 1
            let minY = outer.map { Float($0.y) }.min() ?? 0
            let maxY = outer.map { Float($0.y) }.max() ?? 1
            let mouthW = max(maxX - minX, 0.01)
            let widthRatio = mouthW / interOcular

            if baselineMouthWidthRatio == nil {
                baselineMouthWidthRatio = widthRatio
            } else if let base = baselineMouthWidthRatio {
                baselineMouthWidthRatio = base * 0.995 + widthRatio * 0.005
            }

            let innerPoints = landmarks.innerLips?.normalizedPoints ?? outer
            let inMinY = innerPoints.map { Float($0.y) }.min() ?? minY
            let inMaxY = innerPoints.map { Float($0.y) }.max() ?? maxY
            let mouthOpenH = max(inMaxY - inMinY, 0) / interOcular

            let jawOpen = ((mouthOpenH - 0.04) / 0.28).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.jawOpen] = jawOpen

            let leftCorner = outer.min(by: { $0.x < $1.x })!
            let rightCorner = outer.max(by: { $0.x < $1.x })!
            let midLipY = (inMinY + inMaxY) * 0.5
            let leftLift = (Float(leftCorner.y) - midLipY) / interOcular
            let rightLift = (Float(rightCorner.y) - midLipY) / interOcular
            let smileL = ((leftLift - 0.015) / 0.08).clamped(to: 0...1)
            let smileR = ((rightLift - 0.015) / 0.08).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.mouthSmileLeft] = mirror ? smileR : smileL
            shapes[ARKitFaceBlendShapes.mouthSmileRight] = mirror ? smileL : smileR

            let baseW = baselineMouthWidthRatio ?? 0.95
            let narrow = ((baseW - widthRatio) / 0.25).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.mouthPucker] = narrow * (1.0 - jawOpen * 0.6)
            shapes[ARKitFaceBlendShapes.mouthFunnel] = narrow * jawOpen
        }

        // 3. Eyebrows
        if let lBrow = centerPoint(landmarks.leftEyebrow),
           let rBrow = centerPoint(landmarks.rightEyebrow),
           let lEye = leftEyeCenter,
           let rEye = rightEyeCenter {
            let lGap = Float(lBrow.y - lEye.y) / interOcular
            let rGap = Float(rBrow.y - rEye.y) / interOcular
            let avgGap = (lGap + rGap) * 0.5
            if baselineEyebrowRatio == nil {
                baselineEyebrowRatio = avgGap
            }
            let baseGap = baselineEyebrowRatio ?? avgGap
            let deltaL = (mirror ? rGap : lGap) - baseGap
            let deltaR = (mirror ? lGap : rGap) - baseGap
            let avgDelta = (deltaL + deltaR) * 0.5

            shapes[ARKitFaceBlendShapes.browInnerUp] = (avgDelta / 0.08).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.browOuterUpLeft] = (deltaL / 0.08).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.browOuterUpRight] = (deltaR / 0.08).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.browDownLeft] = ((-deltaL) / 0.07).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.browDownRight] = ((-deltaR) / 0.07).clamped(to: 0...1)
        }

        // 4. Pupil gaze offset
        var gazeOffset = SIMD2<Float>(0, 0)
        if let gx = pupilNormalizedOffset(eye: landmarks.leftEye, pupil: landmarks.leftPupil) {
            gazeOffset.x = mirror ? -gx.x : gx.x
            gazeOffset.y = gx.y
        }

        // IMPORTANT: ARKitFaceDriver checks `Date().timeIntervalSinceReferenceDate - blendShapes.timestamp <= maxAge`
        let frame = TrackedFaceFrame(
            blendShapes: ARKitFaceBlendShapes(
                timestamp: Date().timeIntervalSinceReferenceDate,
                shapes: shapes
            ),
            headYaw: headYaw,
            headPitch: headPitch,
            headRoll: headRoll,
            gazeOffset: gazeOffset
        )

        let frameCallback = onFrame
        Task { @MainActor in
            frameCallback?(frame)
        }
    }

    private func detectFaceWithAutoOrientation(in sampleBuffer: CMSampleBuffer) -> VNFaceObservation? {
        if let obs = performFaceDetection(in: sampleBuffer, orientation: preferredOrientation) {
            return obs
        }
        // If no face found in preferredOrientation, probe other orientations (handles portrait/rotated iPhone)
        if frameCounter % 5 == 0 || frameCounter <= 5 {
            let candidates: [CGImagePropertyOrientation] = [.up, .right, .left, .down]
            for orient in candidates where orient != preferredOrientation {
                if let obs = performFaceDetection(in: sampleBuffer, orientation: orient) {
                    preferredOrientation = orient
                    AppLogger.shared.info("Camera", "Locked face orientation to CGImagePropertyOrientation(\(orient.rawValue))")
                    return obs
                }
            }
        }
        return nil
    }

    private func performFaceDetection(in sampleBuffer: CMSampleBuffer, orientation: CGImagePropertyOrientation) -> VNFaceObservation? {
        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: orientation, options: [:])
        do {
            try handler.perform([request])
            return request.results?.first
        } catch {
            return nil
        }
    }

    private func centerPoint(_ region: VNFaceLandmarkRegion2D?) -> CGPoint? {
        guard let pts = region?.normalizedPoints, !pts.isEmpty else { return nil }
        let sum = pts.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
        return CGPoint(x: sum.x / CGFloat(pts.count), y: sum.y / CGFloat(pts.count))
    }

    private func eyeAspectRatio(_ eye: VNFaceLandmarkRegion2D?) -> Float? {
        guard let pts = eye?.normalizedPoints, pts.count >= 4 else { return nil }
        let minX = pts.map { Float($0.x) }.min() ?? 0
        let maxX = pts.map { Float($0.x) }.max() ?? 0
        let minY = pts.map { Float($0.y) }.min() ?? 0
        let maxY = pts.map { Float($0.y) }.max() ?? 0
        let w = max(maxX - minX, 1e-4)
        let h = max(maxY - minY, 0)
        return h / w
    }

    private func pupilNormalizedOffset(eye: VNFaceLandmarkRegion2D?, pupil: VNFaceLandmarkRegion2D?) -> SIMD2<Float>? {
        guard let eyePts = eye?.normalizedPoints, eyePts.count >= 4,
              let pCenter = centerPoint(pupil) else { return nil }
        let minX = eyePts.map { Float($0.x) }.min() ?? 0
        let maxX = eyePts.map { Float($0.x) }.max() ?? 1
        let minY = eyePts.map { Float($0.y) }.min() ?? 0
        let maxY = eyePts.map { Float($0.y) }.max() ?? 1
        let cx = (minX + maxX) * 0.5
        let cy = (minY + maxY) * 0.5
        let halfW = max((maxX - minX) * 0.5, 1e-4)
        let halfH = max((maxY - minY) * 0.5, 1e-4)
        let nx = ((Float(pCenter.x) - cx) / halfW).clamped(to: -1...1)
        let ny = ((Float(pCenter.y) - cy) / halfH).clamped(to: -1...1)
        return SIMD2<Float>(nx, ny)
    }
}

private extension Comparable {
    func clamped(to limits: ClosedRange<Self>) -> Self {
        min(max(self, limits.lowerBound), limits.upperBound)
    }
}
