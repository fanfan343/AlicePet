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
            return "📱 \(name)（连续互通·系统仅支持后置镜头）"
        } else if isBuiltIn {
            return "💻 \(name)（前置镜头）"
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

    // Atomic generation lock prevents stale async start() calls from reopening the camera after stop()
    private let stateLock = NSLock()
    private var sessionGeneration: Int = 0
    private var isStoppedOrStopping: Bool = true

    private var currentDeviceID: String?
    private var isProcessingFrame = false
    private var lastProcessedTimestamp: TimeInterval = 0
    private let minInferenceInterval: TimeInterval = 1.0 / 20.0 // 20Hz Vision inference -> smooth & low power

    private var baselineEyebrowRatio: Float?
    private var baselineMouthWidthRatio: Float?
    private var baselineShoulderWidth: Float?
    private var baselineShoulderCenterX: Float?
    private var baselineShoulderTilt: Float?
    private var baselineHeadPitch: Float?
    private var preferredOrientation: CGImagePropertyOrientation = .up
    private var frameCounter: Int = 0
    private var consecutiveNoFaceFrames: Int = 0

    // Per-channel 1€ smoothing on raw Vision measurements to eliminate high-frequency twitching
    private var earLeftFilter = OneEuroFilterScalar(minCutoff: 1.6, beta: 0.12, dCutoff: 1.0)
    private var earRightFilter = OneEuroFilterScalar(minCutoff: 1.6, beta: 0.12, dCutoff: 1.0)
    private var jawFilter = OneEuroFilterScalar(minCutoff: 1.1, beta: 0.08, dCutoff: 1.0)
    private var smileLeftFilter = OneEuroFilterScalar(minCutoff: 0.9, beta: 0.05, dCutoff: 1.0)
    private var smileRightFilter = OneEuroFilterScalar(minCutoff: 0.9, beta: 0.05, dCutoff: 1.0)
    private var puckerFilter = OneEuroFilterScalar(minCutoff: 0.9, beta: 0.05, dCutoff: 1.0)
    private var browLeftFilter = OneEuroFilterScalar(minCutoff: 0.6, beta: 0.03, dCutoff: 1.0)
    private var browRightFilter = OneEuroFilterScalar(minCutoff: 0.6, beta: 0.03, dCutoff: 1.0)

    var mirrorMode: Bool = true
    var upperBodyTrackingEnabled: Bool = true

    var onFrame: (@MainActor @Sendable (TrackedFaceFrame) -> Void)?
    var onBodyFrame: (@MainActor @Sendable (TrackedUpperBodyFrame) -> Void)?
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
        teardownSessionNow()
    }

    private func nextGeneration(active: Bool) -> Int {
        stateLock.lock()
        defer { stateLock.unlock() }
        sessionGeneration += 1
        isStoppedOrStopping = !active
        return sessionGeneration
    }

    private func isGenerationValid(_ gen: Int) -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return gen == sessionGeneration && !isStoppedOrStopping
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
        AppLogger.shared.error("Camera", "AVCaptureSession runtime error: \(msg)")
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

    func calibrateNeutral() {
        captureQueue.async { [weak self] in
            guard let self else { return }
            self.resetBaselinesAndFilters()
            AppLogger.shared.info("Camera", "Neutral face & posture baselines cleared for recalibration")
        }
    }

    func start(deviceID: String) {
        let gen = nextGeneration(active: true)
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

            guard self.isGenerationValid(gen) else {
                AppLogger.shared.info("Camera", "Cancelled pending start (gen=\(gen) superseded)")
                return
            }

            guard granted else {
                _ = self.nextGeneration(active: false)
                let errCb = self.onError
                await MainActor.run {
                    errCb?("未获得摄像头权限，请在「系统设置 → 隐私与安全性 → 相机」中允许 AlicePet 访问摄像头。")
                }
                return
            }

            self.captureQueue.async { [weak self] in
                self?.configureAndStartSession(deviceID: deviceID, generation: gen)
            }
        }
    }

    func stop() {
        _ = nextGeneration(active: false)
        captureQueue.async { [weak self] in
            self?.teardownSessionNow()
        }
    }

    func stopSync() {
        _ = nextGeneration(active: false)
        captureQueue.sync {
            self.teardownSessionNow()
        }
    }

    private func teardownSessionNow() {
        videoOutput.setSampleBufferDelegate(nil, queue: nil)
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
        session.commitConfiguration()

        currentDeviceID = nil
        resetBaselinesAndFilters()
        frameCounter = 0
        consecutiveNoFaceFrames = 0
        lastProcessedTimestamp = 0
        AppLogger.shared.info("Camera", "Session stopped and all inputs/outputs released")
    }

    private func resetBaselinesAndFilters() {
        baselineEyebrowRatio = nil
        baselineMouthWidthRatio = nil
        baselineShoulderWidth = nil
        baselineShoulderCenterX = nil
        baselineShoulderTilt = nil
        baselineHeadPitch = nil
        earLeftFilter.reset()
        earRightFilter.reset()
        jawFilter.reset()
        smileLeftFilter.reset()
        smileRightFilter.reset()
        puckerFilter.reset()
        browLeftFilter.reset()
        browRightFilter.reset()
    }

    private func configureAndStartSession(deviceID: String, generation: Int) {
        guard isGenerationValid(generation) else {
            AppLogger.shared.info("Camera", "Skipped stale start before configure (gen=\(generation))")
            return
        }

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

        teardownSessionNow()
        guard isGenerationValid(generation) else { return }

        session.beginConfiguration()
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

        let preferredPresets: [AVCaptureSession.Preset] = [.hd1280x720, .vga640x480, .medium, .high]
        if let supported = preferredPresets.first(where: { session.canSetSessionPreset($0) }) {
            session.sessionPreset = supported
        }

        videoOutput.videoSettings = nil
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: captureQueue)
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        session.commitConfiguration()

        guard isGenerationValid(generation) else {
            teardownSessionNow()
            AppLogger.shared.info("Camera", "Aborted start right after configure because stop() was called (gen=\(generation))")
            return
        }

        resetBaselinesAndFilters()
        preferredOrientation = .up
        frameCounter = 0
        consecutiveNoFaceFrames = 0
        currentDeviceID = deviceID
        session.startRunning()

        // Double-check in case stop() arrived during startRunning() blocking call
        if !isGenerationValid(generation) {
            teardownSessionNow()
            AppLogger.shared.info("Camera", "Immediately tore down session because stop() arrived during startRunning (gen=\(generation))")
            return
        }
        AppLogger.shared.info("Camera", "Started session on \(device.localizedName) (running=\(session.isRunning), upperBody=\(upperBodyTrackingEnabled))")
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        stateLock.lock()
        let stopped = isStoppedOrStopping
        stateLock.unlock()
        if stopped { return }

        let now = CACurrentMediaTime()
        guard !isProcessingFrame, (now - lastProcessedTimestamp) >= minInferenceInterval else { return }
        isProcessingFrame = true
        lastProcessedTimestamp = now
        defer { isProcessingFrame = false }

        frameCounter += 1
        if frameCounter == 1, let pb = CMSampleBufferGetImageBuffer(sampleBuffer) {
            let w = CVPixelBufferGetWidth(pb)
            let h = CVPixelBufferGetHeight(pb)
            AppLogger.shared.info("Camera", "First frame received: \(w)x\(h)")
        }

        let mirror = mirrorMode
        let runBody = upperBodyTrackingEnabled
        let (faceObs, bodyObs) = detectFaceAndBody2D(in: sampleBuffer, detectBody: runBody)

        guard let face = faceObs, let landmarks = face.landmarks else {
            consecutiveNoFaceFrames += 1
            if consecutiveNoFaceFrames == 12 {
                let stateCb = onTrackingStateChanged
                Task { @MainActor in stateCb?(false) }
            }
            return
        }

        if consecutiveNoFaceFrames >= 12 || frameCounter <= 2 {
            let stateCb = onTrackingStateChanged
            Task { @MainActor in stateCb?(true) }
        }
        consecutiveNoFaceFrames = 0

        let rawYaw = face.yaw?.floatValue ?? 0
        let rawPitch = face.pitch?.floatValue ?? 0
        let rawRoll = face.roll?.floatValue ?? 0

        if baselineHeadPitch == nil {
            baselineHeadPitch = rawPitch * 0.45
        }
        let headYaw = (mirror ? -rawYaw : rawYaw).clamped(to: -0.72...0.72)
        let headPitch = (rawPitch - (baselineHeadPitch ?? 0)).clamped(to: -0.50...0.50)
        let headRoll = (mirror ? -rawRoll : rawRoll).clamped(to: -0.50...0.50)

        // Emit rock-solid 2D-geometry upper-body frame (immune to monocular 3D Z-depth jitter)
        if let bodyObs, let bodyFrame = extractStableUpperBody(from: bodyObs, headYaw: headYaw, mirror: mirror) {
            let bodyCb = onBodyFrame
            Task { @MainActor in
                bodyCb?(bodyFrame)
            }
        }

        var shapes: [String: Float] = [:]

        // 1. Filtered rotation-invariant EAR with strict open-eye deadzone (zero eyelid flutter)
        let rawLeftEAR = rotationInvariantEAR(landmarks.leftEye)
        let rawRightEAR = rotationInvariantEAR(landmarks.rightEye)
        let mappedLeftEAR = mirror ? rawRightEAR : rawLeftEAR
        let mappedRightEAR = mirror ? rawLeftEAR : rawRightEAR

        if let earRaw = mappedLeftEAR {
            let ear = earLeftFilter.filter(earRaw, timestamp: now)
            let blink = ear >= 0.23 ? 0.0 : ((0.23 - ear) / 0.11).clamped(to: 0...1)
            let wide = ear <= 0.36 ? 0.0 : ((ear - 0.36) / 0.11).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.eyeBlinkLeft] = blink
            shapes[ARKitFaceBlendShapes.eyeWideLeft] = wide
        }
        if let earRaw = mappedRightEAR {
            let ear = earRightFilter.filter(earRaw, timestamp: now)
            let blink = ear >= 0.23 ? 0.0 : ((0.23 - ear) / 0.11).clamped(to: 0...1)
            let wide = ear <= 0.36 ? 0.0 : ((ear - 0.36) / 0.11).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.eyeBlinkRight] = blink
            shapes[ARKitFaceBlendShapes.eyeWideRight] = wide
        }

        let leftEyeCenter = centerPoint(landmarks.leftEye)
        let rightEyeCenter = centerPoint(landmarks.rightEye)
        let interOcular: Float
        if let l = leftEyeCenter, let r = rightEyeCenter {
            interOcular = max(Float(hypot(r.x - l.x, r.y - l.y)), 0.08)
        } else {
            interOcular = 0.32
        }

        // 2. Filtered mouth shapes with deadzone so closed mouth stays completely calm
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
                baselineMouthWidthRatio = base * 0.997 + widthRatio * 0.003
            }

            let innerPoints = landmarks.innerLips?.normalizedPoints ?? outer
            let inMinY = innerPoints.map { Float($0.y) }.min() ?? minY
            let inMaxY = innerPoints.map { Float($0.y) }.max() ?? maxY
            let mouthOpenH = max(inMaxY - inMinY, 0) / interOcular

            let rawJaw = ((mouthOpenH - 0.055) / 0.27).clamped(to: 0...1)
            let jawOpen = applyDeadzone(jawFilter.filter(rawJaw, timestamp: now), threshold: 0.06)
            shapes[ARKitFaceBlendShapes.jawOpen] = jawOpen

            let leftCorner = outer.min(by: { $0.x < $1.x })!
            let rightCorner = outer.max(by: { $0.x < $1.x })!
            let midLipY = (inMinY + inMaxY) * 0.5
            let leftLift = (Float(leftCorner.y) - midLipY) / interOcular
            let rightLift = (Float(rightCorner.y) - midLipY) / interOcular
            let rawSmileL = ((leftLift - 0.020) / 0.075).clamped(to: 0...1)
            let rawSmileR = ((rightLift - 0.020) / 0.075).clamped(to: 0...1)

            let outSmileL = applyDeadzone(smileLeftFilter.filter(mirror ? rawSmileR : rawSmileL, timestamp: now), threshold: 0.07)
            let outSmileR = applyDeadzone(smileRightFilter.filter(mirror ? rawSmileL : rawSmileR, timestamp: now), threshold: 0.07)
            shapes[ARKitFaceBlendShapes.mouthSmileLeft] = outSmileL
            shapes[ARKitFaceBlendShapes.mouthSmileRight] = outSmileR
            shapes[ARKitFaceBlendShapes.cheekSquintLeft] = outSmileL * 0.40
            shapes[ARKitFaceBlendShapes.cheekSquintRight] = outSmileR * 0.40

            let baseW = baselineMouthWidthRatio ?? 0.95
            let rawNarrow = ((baseW - widthRatio) / 0.25).clamped(to: 0...1)
            let narrow = applyDeadzone(puckerFilter.filter(rawNarrow, timestamp: now), threshold: 0.08)
            shapes[ARKitFaceBlendShapes.mouthPucker] = narrow * (1.0 - jawOpen * 0.6)
            shapes[ARKitFaceBlendShapes.mouthFunnel] = narrow * jawOpen
        }

        // 3. Heavily smoothed eyebrows with deadzone
        if let lBrow = centerPoint(landmarks.leftEyebrow),
           let rBrow = centerPoint(landmarks.rightEyebrow),
           let lEye = leftEyeCenter,
           let rEye = rightEyeCenter {
            let lGap = Float(lBrow.y - lEye.y) / interOcular
            let rGap = Float(rBrow.y - rEye.y) / interOcular
            let avgGap = (lGap + rGap) * 0.5
            if baselineEyebrowRatio == nil {
                baselineEyebrowRatio = avgGap
            } else if let base = baselineEyebrowRatio {
                baselineEyebrowRatio = base * 0.998 + avgGap * 0.002
            }
            let baseGap = baselineEyebrowRatio ?? avgGap
            let dL = browLeftFilter.filter((mirror ? rGap : lGap) - baseGap, timestamp: now)
            let dR = browRightFilter.filter((mirror ? lGap : rGap) - baseGap, timestamp: now)
            let avgDelta = (dL + dR) * 0.5

            shapes[ARKitFaceBlendShapes.browInnerUp] = applyDeadzone((avgDelta / 0.08).clamped(to: 0...1), threshold: 0.10)
            shapes[ARKitFaceBlendShapes.browOuterUpLeft] = applyDeadzone((dL / 0.08).clamped(to: 0...1), threshold: 0.10)
            shapes[ARKitFaceBlendShapes.browOuterUpRight] = applyDeadzone((dR / 0.08).clamped(to: 0...1), threshold: 0.10)
            shapes[ARKitFaceBlendShapes.browDownLeft] = applyDeadzone(((-dL) / 0.07).clamped(to: 0...1), threshold: 0.10)
            shapes[ARKitFaceBlendShapes.browDownRight] = applyDeadzone(((-dR) / 0.07).clamped(to: 0...1), threshold: 0.10)
        }

        // 4. Binocular averaged pupil gaze
        var gazeOffset = SIMD2<Float>(0, 0)
        let leftGaze = pupilNormalizedOffset(eye: landmarks.leftEye, pupil: landmarks.leftPupil)
        let rightGaze = pupilNormalizedOffset(eye: landmarks.rightEye, pupil: landmarks.rightPupil)
        if let lg = leftGaze, let rg = rightGaze {
            let avg = (lg + rg) * 0.5
            gazeOffset = SIMD2<Float>(mirror ? -avg.x : avg.x, avg.y)
        } else if let single = leftGaze ?? rightGaze {
            gazeOffset = SIMD2<Float>(mirror ? -single.x : single.x, single.y)
        }

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

    // MARK: - Combined Vision Face + Stable 2D Upper-Body Detection

    private func detectFaceAndBody2D(
        in sampleBuffer: CMSampleBuffer,
        detectBody: Bool
    ) -> (VNFaceObservation?, VNHumanBodyPoseObservation?) {
        let (face, body) = performVisionRequests(in: sampleBuffer, orientation: preferredOrientation, detectBody: detectBody)
        if face != nil || body != nil {
            return (face, body)
        }
        if frameCounter % 6 == 0 || frameCounter <= 5 {
            let candidates: [CGImagePropertyOrientation] = [.up, .right, .left, .down]
            for orient in candidates where orient != preferredOrientation {
                let (f2, b2) = performVisionRequests(in: sampleBuffer, orientation: orient, detectBody: detectBody)
                if f2 != nil || b2 != nil {
                    preferredOrientation = orient
                    AppLogger.shared.info("Camera", "Locked vision orientation to \(orient.rawValue)")
                    return (f2, b2)
                }
            }
        }
        return (nil, nil)
    }

    private func performVisionRequests(
        in sampleBuffer: CMSampleBuffer,
        orientation: CGImagePropertyOrientation,
        detectBody: Bool
    ) -> (VNFaceObservation?, VNHumanBodyPoseObservation?) {
        let faceReq = VNDetectFaceLandmarksRequest()
        var requests: [VNRequest] = [faceReq]
        var bodyReq: VNDetectHumanBodyPoseRequest?
        if detectBody {
            let req = VNDetectHumanBodyPoseRequest()
            bodyReq = req
            requests.append(req)
        }
        let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: orientation, options: [:])
        do {
            try handler.perform(requests)
            return (faceReq.results?.first, bodyReq?.results?.first)
        } catch {
            return (nil, nil)
        }
    }

    /// Computes rock-solid upper-body lean, twist, and visible arm vectors from high-confidence 2D Vision keypoints,
    /// avoiding noisy monocular Z-depth spikes that cause twitching on laptop webcams.
    private func extractStableUpperBody(
        from obs: VNHumanBodyPoseObservation,
        headYaw: Float,
        mirror: Bool
    ) -> TrackedUpperBodyFrame? {
        func pt(_ joint: VNHumanBodyPoseObservation.JointName, minConf: Float = 0.45) -> SIMD2<Float>? {
            guard let p = try? obs.recognizedPoint(joint), p.confidence >= minConf else { return nil }
            let x = Float(p.location.x)
            let y = Float(p.location.y)
            guard x >= 0.02 && x <= 0.98 && y >= 0.02 && y <= 0.98 else { return nil }
            return SIMD2<Float>(x, y)
        }

        guard let lSh = pt(.leftShoulder, minConf: 0.40),
              let rSh = pt(.rightShoulder, minConf: 0.40) else {
            return nil
        }

        // Vision 2D normalized coordinates: x=0 left -> x=1 right, y=0 bottom -> y=1 top
        let shCenter = (lSh + rSh) * 0.5
        let shVec = lSh - rSh // points from user's right (viewer-left) to user's left (viewer-right)
        let shWidth = max(simd_length(shVec), 0.05)
        let shTilt = atan2(shVec.y, max(abs(shVec.x), 0.05))

        if baselineShoulderWidth == nil {
            baselineShoulderWidth = shWidth
            baselineShoulderCenterX = shCenter.x
            baselineShoulderTilt = shTilt
        }

        let baseW = baselineShoulderWidth ?? shWidth
        let baseX = baselineShoulderCenterX ?? shCenter.x
        let baseTilt = baselineShoulderTilt ?? shTilt

        // 1. Torso Roll: smooth blend of shoulder tilt + lateral shift relative to calibrated center
        let lateralShift = (shCenter.x - baseX) * 1.1
        let tiltDelta = (shTilt - baseTilt) * 0.75
        let rawRoll = -(lateralShift + tiltDelta)
        let torsoRoll = (mirror ? rawRoll : -rawRoll).clamped(to: -0.30...0.30)

        // 2. Torso Pitch: leaning closer to camera expands apparent shoulder width smoothly
        let scaleDelta = (shWidth - baseW) / max(baseW, 0.08)
        let torsoPitch = (scaleDelta * 0.65).clamped(to: -0.25...0.35)

        // 3. Torso Yaw: gentle natural follow-through with head yaw (zero Z-depth snap!)
        let torsoYaw = (headYaw * 0.28).clamped(to: -0.35...0.35)

        // 4. Visible Arms (strict confidence >= 0.55 & inside frame so off-screen arms stay gracefully resting)
        func armDir2D(from a: SIMD2<Float>?, to b: SIMD2<Float>?, forwardBias: Float) -> SIMD3<Float>? {
            guard let pa = a, let pb = b else { return nil }
            let d2 = pb - pa
            guard simd_length(d2) > 0.04 else { return nil }
            let dx = mirror ? d2.x : -d2.x
            return simd_normalize(SIMD3<Float>(dx, d2.y, forwardBias))
        }

        let lElb = pt(.leftElbow, minConf: 0.55)
        let lWri = pt(.leftWrist, minConf: 0.55)
        let rElb = pt(.rightElbow, minConf: 0.55)
        let rWri = pt(.rightWrist, minConf: 0.55)

        let uLeftUpper = armDir2D(from: lSh, to: lElb, forwardBias: 0.20)
        let uLeftFore = armDir2D(from: lElb, to: lWri, forwardBias: 0.35)
        let uRightUpper = armDir2D(from: rSh, to: rElb, forwardBias: 0.20)
        let uRightFore = armDir2D(from: rElb, to: rWri, forwardBias: 0.35)

        return TrackedUpperBodyFrame(
            torsoYaw: torsoYaw,
            torsoPitch: torsoPitch,
            torsoRoll: torsoRoll,
            leftUpperArmDir: mirror ? uLeftUpper : uRightUpper,
            leftForearmDir: mirror ? uLeftFore : uRightFore,
            rightUpperArmDir: mirror ? uRightUpper : uLeftUpper,
            rightForearmDir: mirror ? uRightFore : uLeftFore
        )
    }

    // MARK: - Face Geometry Helpers

    private func applyDeadzone(_ value: Float, threshold: Float) -> Float {
        guard value > threshold else { return 0 }
        return ((value - threshold) / max(1.0 - threshold, 0.01)).clamped(to: 0...1)
    }

    private func centerPoint(_ region: VNFaceLandmarkRegion2D?) -> CGPoint? {
        guard let pts = region?.normalizedPoints, !pts.isEmpty else { return nil }
        let sum = pts.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
        return CGPoint(x: sum.x / CGFloat(pts.count), y: sum.y / CGFloat(pts.count))
    }

    private func rotationInvariantEAR(_ eye: VNFaceLandmarkRegion2D?) -> Float? {
        guard let pts = eye?.normalizedPoints, pts.count >= 4 else { return nil }
        let pLeft = pts.min(by: { $0.x < $1.x })!
        let pRight = pts.max(by: { $0.x < $1.x })!
        let dx = Float(pRight.x - pLeft.x)
        let dy = Float(pRight.y - pLeft.y)
        let width = hypot(dx, dy)
        guard width > 1e-4 else { return nil }

        let ux = dx / width
        let uy = dy / width
        let vx = -uy
        let vy = ux

        var minV: Float = .greatestFiniteMagnitude
        var maxV: Float = -.greatestFiniteMagnitude
        for pt in pts {
            let rx = Float(pt.x - pLeft.x)
            let ry = Float(pt.y - pLeft.y)
            let proj = rx * vx + ry * vy
            if proj < minV { minV = proj }
            if proj > maxV { maxV = proj }
        }
        return max(maxV - minV, 0) / width
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
