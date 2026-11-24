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
    private var baselineTorsoPitch: Float?
    private var baselineTorsoRoll: Float?
    private var baselineHeadPitch: Float?
    private var preferredOrientation: CGImagePropertyOrientation = .up
    private var frameCounter: Int = 0
    private var consecutiveNoFaceFrames: Int = 0

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

    /// Resets neutral baselines so the user's current relaxed face & seated posture becomes zero.
    func calibrateNeutral() {
        captureQueue.async { [weak self] in
            guard let self else { return }
            self.baselineEyebrowRatio = nil
            self.baselineMouthWidthRatio = nil
            self.baselineTorsoPitch = nil
            self.baselineTorsoRoll = nil
            self.baselineHeadPitch = nil
            AppLogger.shared.info("Camera", "Neutral face & posture baselines cleared for recalibration")
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
                let errCb = self.onError
                await MainActor.run {
                    errCb?("未获得摄像头权限，请在「系统设置 → 隐私与安全性 → 相机」中允许 AlicePet 访问摄像头。")
                }
                return
            }

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
            self.baselineTorsoPitch = nil
            self.baselineTorsoRoll = nil
            self.baselineHeadPitch = nil
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
        baselineEyebrowRatio = nil
        baselineMouthWidthRatio = nil
        baselineTorsoPitch = nil
        baselineTorsoRoll = nil
        baselineHeadPitch = nil
        preferredOrientation = .up
        frameCounter = 0
        consecutiveNoFaceFrames = 0
        currentDeviceID = deviceID
        session.startRunning()
        AppLogger.shared.info("Camera", "Started session on \(device.localizedName) (running=\(session.isRunning), upperBody=\(upperBodyTrackingEnabled))")
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

        let mirror = mirrorMode
        let runBody = upperBodyTrackingEnabled
        let (faceObs, bodyObs) = detectFaceAndBody(in: sampleBuffer, detectBody: runBody)

        // 1. Emit 3D upper-body tracking frame if detected
        if let bodyObs, let bodyFrame = extractUpperBodyFrame(from: bodyObs, mirror: mirror) {
            let bodyCb = onBodyFrame
            Task { @MainActor in
                bodyCb?(bodyFrame)
            }
        }

        // 2. Emit upgraded face tracking frame
        guard let face = faceObs, let landmarks = face.landmarks else {
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

        let rawYaw = face.yaw?.floatValue ?? 0
        let rawPitch = face.pitch?.floatValue ?? 0
        let rawRoll = face.roll?.floatValue ?? 0

        if baselineHeadPitch == nil {
            baselineHeadPitch = rawPitch * 0.45
        }
        let headYaw = (mirror ? -rawYaw : rawYaw).clamped(to: -0.78...0.78)
        let headPitch = (rawPitch - (baselineHeadPitch ?? 0)).clamped(to: -0.55...0.55)
        let headRoll = (mirror ? -rawRoll : rawRoll).clamped(to: -0.55...0.55)

        var shapes: [String: Float] = [:]

        // 2a. Rotation-invariant Eye Aspect Ratio (EAR) so head tilts don't suppress blinks
        let rawLeftEAR = rotationInvariantEAR(landmarks.leftEye)
        let rawRightEAR = rotationInvariantEAR(landmarks.rightEye)
        let leftEAR = mirror ? rawRightEAR : rawLeftEAR
        let rightEAR = mirror ? rawLeftEAR : rawRightEAR

        if let ear = leftEAR {
            let blink = ((0.27 - ear) / 0.13).clamped(to: 0...1)
            let wide = ((ear - 0.35) / 0.11).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.eyeBlinkLeft] = blink
            shapes[ARKitFaceBlendShapes.eyeWideLeft] = wide
        }
        if let ear = rightEAR {
            let blink = ((0.27 - ear) / 0.13).clamped(to: 0...1)
            let wide = ((ear - 0.35) / 0.11).clamped(to: 0...1)
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

        // 2b. Mouth opening, smile, frown, pucker, funnel + linked cheek squint
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
                baselineMouthWidthRatio = base * 0.996 + widthRatio * 0.004
            }

            let innerPoints = landmarks.innerLips?.normalizedPoints ?? outer
            let inMinY = innerPoints.map { Float($0.y) }.min() ?? minY
            let inMaxY = innerPoints.map { Float($0.y) }.max() ?? maxY
            let mouthOpenH = max(inMaxY - inMinY, 0) / interOcular

            let jawOpen = ((mouthOpenH - 0.035) / 0.27).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.jawOpen] = jawOpen

            let leftCorner = outer.min(by: { $0.x < $1.x })!
            let rightCorner = outer.max(by: { $0.x < $1.x })!
            let midLipY = (inMinY + inMaxY) * 0.5
            let leftLift = (Float(leftCorner.y) - midLipY) / interOcular
            let rightLift = (Float(rightCorner.y) - midLipY) / interOcular
            let smileL = ((leftLift - 0.012) / 0.075).clamped(to: 0...1)
            let smileR = ((rightLift - 0.012) / 0.075).clamped(to: 0...1)
            let frownL = ((-leftLift - 0.025) / 0.07).clamped(to: 0...1)
            let frownR = ((-rightLift - 0.025) / 0.07).clamped(to: 0...1)

            let outSmileL = mirror ? smileR : smileL
            let outSmileR = mirror ? smileL : smileR
            shapes[ARKitFaceBlendShapes.mouthSmileLeft] = outSmileL
            shapes[ARKitFaceBlendShapes.mouthSmileRight] = outSmileR
            shapes[ARKitFaceBlendShapes.mouthFrownLeft] = mirror ? frownR : frownL
            shapes[ARKitFaceBlendShapes.mouthFrownRight] = mirror ? frownL : frownR
            shapes[ARKitFaceBlendShapes.cheekSquintLeft] = outSmileL * 0.45
            shapes[ARKitFaceBlendShapes.cheekSquintRight] = outSmileR * 0.45

            let baseW = baselineMouthWidthRatio ?? 0.95
            let narrow = ((baseW - widthRatio) / 0.24).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.mouthPucker] = narrow * (1.0 - jawOpen * 0.6)
            shapes[ARKitFaceBlendShapes.mouthFunnel] = narrow * jawOpen
        }

        // 2c. Eyebrows (adaptive slow drift + neutral calibration)
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
                baselineEyebrowRatio = base * 0.997 + avgGap * 0.003
            }
            let baseGap = baselineEyebrowRatio ?? avgGap
            let deltaL = (mirror ? rGap : lGap) - baseGap
            let deltaR = (mirror ? lGap : rGap) - baseGap
            let avgDelta = (deltaL + deltaR) * 0.5

            shapes[ARKitFaceBlendShapes.browInnerUp] = (avgDelta / 0.075).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.browOuterUpLeft] = (deltaL / 0.075).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.browOuterUpRight] = (deltaR / 0.075).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.browDownLeft] = ((-deltaL) / 0.065).clamped(to: 0...1)
            shapes[ARKitFaceBlendShapes.browDownRight] = ((-deltaR) / 0.065).clamped(to: 0...1)
        }

        // 2d. Binocular averaged pupil gaze offset (more stable than single-eye)
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

    // MARK: - Combined Vision Face + 3D Upper-Body Detection

    private func detectFaceAndBody(
        in sampleBuffer: CMSampleBuffer,
        detectBody: Bool
    ) -> (VNFaceObservation?, VNHumanBodyPose3DObservation?) {
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
    ) -> (VNFaceObservation?, VNHumanBodyPose3DObservation?) {
        let faceReq = VNDetectFaceLandmarksRequest()
        var requests: [VNRequest] = [faceReq]
        var bodyReq: VNDetectHumanBodyPose3DRequest?
        if detectBody {
            let req = VNDetectHumanBodyPose3DRequest()
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

    // MARK: - 3D Upper-Body Extraction (Vision Camera Space -> VRM Space)

    private func extractUpperBodyFrame(
        from obs: VNHumanBodyPose3DObservation,
        mirror: Bool
    ) -> TrackedUpperBodyFrame? {
        func pos3D(_ joint: VNHumanBodyPose3DObservation.JointName) -> SIMD3<Float>? {
            guard let pt = try? obs.recognizedPoint(joint) else { return nil }
            let c = pt.position.columns.3
            return SIMD3<Float>(c.x, c.y, c.z)
        }
        func isVisibleInFrame(_ joint: VNHumanBodyPose3DObservation.JointName) -> Bool {
            guard let imgPt = try? obs.pointInImage(joint) else { return false }
            return imgPt.x >= 0.03 && imgPt.x <= 0.97 && imgPt.y >= 0.03 && imgPt.y <= 0.97
        }

        guard let spinePos = pos3D(.spine) ?? pos3D(.root),
              let centerShoulder = pos3D(.centerShoulder),
              let lShoulder = pos3D(.leftShoulder),
              let rShoulder = pos3D(.rightShoulder) else {
            return nil
        }

        // Vision 3D camera space: +X = viewer right (user's left), +Y = up, +Z = toward camera
        let torsoUp = simd_normalize(centerShoulder - spinePos)
        let shoulderLine = simd_normalize(lShoulder - rShoulder) // points toward viewer-right (+X)

        // Torso pitch (+ = leaning forward toward camera) & roll (+ = leaning toward viewer-left)
        let rawPitch = atan2(torsoUp.z, max(torsoUp.y, 0.25))
        let rawRoll = -atan2(torsoUp.x, max(torsoUp.y, 0.25))
        // Torso yaw from shoulder depth difference
        let rawYaw = atan2(-shoulderLine.z, max(abs(shoulderLine.x), 0.25))

        if baselineTorsoPitch == nil {
            baselineTorsoPitch = rawPitch
            baselineTorsoRoll = rawRoll
        }

        let pitch = ((rawPitch - (baselineTorsoPitch ?? 0)) * 0.85).clamped(to: -0.40...0.45)
        let roll = (((mirror ? rawRoll : -rawRoll) - (mirror ? (baselineTorsoRoll ?? 0) : -(baselineTorsoRoll ?? 0))) * 0.80).clamped(to: -0.35...0.35)
        let yaw = ((mirror ? rawYaw : -rawYaw) * 0.75).clamped(to: -0.50...0.50)

        // Extract visible arms: in mirror mode, Vision's viewer-right (+X, user's left) drives VRM Left (+X)
        // and viewer-left (-X, user's right) drives VRM Right (-X). In non-mirror mode, swap and negate X.
        func armDir(from a: VNHumanBodyPose3DObservation.JointName, to b: VNHumanBodyPose3DObservation.JointName) -> SIMD3<Float>? {
            guard isVisibleInFrame(a), isVisibleInFrame(b),
                  let pa = pos3D(a), let pb = pos3D(b) else { return nil }
            let d = pb - pa
            guard simd_length(d) > 0.05 else { return nil }
            let n = simd_normalize(d)
            return mirror ? n : SIMD3<Float>(-n.x, n.y, n.z)
        }

        let userLeftUpper = armDir(from: .leftShoulder, to: .leftElbow)
        let userLeftFore = armDir(from: .leftElbow, to: .leftWrist)
        let userRightUpper = armDir(from: .rightShoulder, to: .rightElbow)
        let userRightFore = armDir(from: .rightElbow, to: .rightWrist)

        return TrackedUpperBodyFrame(
            torsoYaw: yaw,
            torsoPitch: pitch,
            torsoRoll: roll,
            leftUpperArmDir: mirror ? userLeftUpper : userRightUpper,
            leftForearmDir: mirror ? userLeftFore : userRightFore,
            rightUpperArmDir: mirror ? userRightUpper : userLeftUpper,
            rightForearmDir: mirror ? userRightFore : userLeftFore
        )
    }

    // MARK: - Upgraded Face Geometry Helpers

    private func centerPoint(_ region: VNFaceLandmarkRegion2D?) -> CGPoint? {
        guard let pts = region?.normalizedPoints, !pts.isEmpty else { return nil }
        let sum = pts.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
        return CGPoint(x: sum.x / CGFloat(pts.count), y: sum.y / CGFloat(pts.count))
    }

    /// Projects eye landmarks onto the eye's own corner-to-corner axis so head roll never distorts blink EAR.
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
        let height = max(maxV - minV, 0)
        return height / width
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
