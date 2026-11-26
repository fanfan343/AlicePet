import Foundation
import Network
import QuartzCore
import VRMMetalKit
import simd

/// Ultra-lightweight native macOS loopback HTTP JSON bridge (`127.0.0.1:39550`) built on Apple's
/// `Network.framework` (zero third-party dependencies). Exposes atomic body/perception controls
/// for ZeroClaw, MCP bridges, or external gateways, with unified `[Control]` logging.
@MainActor
final class LocalControlServer {
    static let defaultPort: UInt16 = 39550

    private let queue = DispatchQueue(label: "com.funnyliu.AlicePet.controlServer", qos: .userInitiated)
    private var listener: NWListener?
    private weak var controller: PetController?
    private weak var appDelegate: AppDelegate?

    private(set) var port: UInt16 = defaultPort
    private(set) var isRunning: Bool = false

    func start(port: UInt16 = defaultPort, controller: PetController, appDelegate: AppDelegate) {
        stop()
        self.controller = controller
        self.appDelegate = appDelegate
        self.port = port

        do {
            let params = NWParameters.tcp
            params.allowLocalEndpointReuse = true
            params.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: NWEndpoint.Port(rawValue: port)!)
            let newListener = try NWListener(using: params)
            newListener.stateUpdateHandler = { [weak self] state in
                Task { @MainActor in
                    guard let self else { return }
                    switch state {
                    case .ready:
                        self.isRunning = true
                        AppLogger.shared.info("Control", "LocalControlServer listening on http://127.0.0.1:\(self.port)")
                        self.appDelegate?.refreshMenus()
                    case .failed(let err):
                        self.isRunning = false
                        AppLogger.shared.error("Control", "LocalControlServer failed: \(err)")
                    case .cancelled:
                        self.isRunning = false
                    default:
                        break
                    }
                }
            }
            newListener.newConnectionHandler = { [weak self] conn in
                self?.handleConnection(conn)
            }
            newListener.start(queue: queue)
            self.listener = newListener
        } catch {
            AppLogger.shared.error("Control", "Failed to start LocalControlServer on port \(port): \(error)")
        }
    }

    func stop() {
        listener?.cancel()
        listener = nil
        isRunning = false
    }

    // MARK: - HTTP/1.1 Loopback Framing

    private nonisolated func handleConnection(_ conn: NWConnection) {
        conn.start(queue: queue)
        receiveLoop(conn: conn, buffer: Data())
    }

    private nonisolated func receiveLoop(conn: NWConnection, buffer: Data) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let self else { conn.cancel(); return }
            if let error {
                AppLogger.shared.debug("Control", "Connection receive error: \(error)")
                conn.cancel()
                return
            }
            var acc = buffer
            if let data { acc.append(data) }

            if let req = Self.parseHTTPRequest(acc) {
                Task { @MainActor in
                    let t0 = CACurrentMediaTime()
                    let (status, payload, summary) = await self.dispatch(method: req.method, path: req.path, body: req.body)
                    let ms = (CACurrentMediaTime() - t0) * 1000.0
                    AppLogger.shared.info("Control", String(format: "%@ %@ -> %d (%.1fms) %@", req.method, req.path, status, ms, summary))
                    Self.sendJSON(conn: conn, status: status, payload: payload)
                }
                return
            }

            if isComplete || acc.count > 2_000_000 {
                Self.sendJSON(conn: conn, status: 400, payload: ["ok": false, "error": "Malformed or oversized HTTP request"])
                return
            }
            self.receiveLoop(conn: conn, buffer: acc)
        }
    }

    private struct HTTPRequest {
        let method: String
        let path: String
        let body: Data
    }

    private nonisolated static func parseHTTPRequest(_ data: Data) -> HTTPRequest? {
        let sep = Data([0x0D, 0x0A, 0x0D, 0x0A]) // \r\n\r\n
        guard let headerRange = data.range(of: sep) else { return nil }
        guard let headerText = String(data: data[..<headerRange.lowerBound], encoding: .utf8) else { return nil }
        let lines = headerText.components(separatedBy: "\r\n")
        guard let first = lines.first else { return nil }
        let parts = first.split(separator: " ")
        guard parts.count >= 2 else { return nil }

        var contentLength = 0
        for line in lines.dropFirst() {
            if line.lowercased().hasPrefix("content-length:"),
               let val = Int(line.split(separator: ":", maxSplits: 1).last?.trimmingCharacters(in: .whitespaces) ?? "") {
                contentLength = max(0, val)
            }
        }
        let bodyStart = headerRange.upperBound
        guard data.count >= bodyStart + contentLength else { return nil }
        let body = data[bodyStart..<(bodyStart + contentLength)]
        let rawPath = String(parts[1]).split(separator: "?", maxSplits: 1).first.map(String.init) ?? String(parts[1])
        return HTTPRequest(method: String(parts[0]).uppercased(), path: rawPath, body: Data(body))
    }

    private nonisolated static func sendJSON(conn: NWConnection, status: Int, payload: [String: Any]) {
        let jsonData = (try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])) ?? Data("{}".utf8)
        let reason: String = switch status {
        case 200: "OK"
        case 400: "Bad Request"
        case 404: "Not Found"
        default: "Internal Server Error"
        }
        let header = "HTTP/1.1 \(status) \(reason)\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: \(jsonData.count)\r\nConnection: close\r\n\r\n"
        var response = Data(header.utf8)
        response.append(jsonData)
        conn.send(content: response, completion: .contentProcessed { _ in
            conn.cancel()
        })
    }

    // MARK: - Route Dispatcher

    private func dispatch(method: String, path: String, body: Data) async -> (Int, [String: Any], String) {
        guard let controller else {
            return (500, ["ok": false, "error": "Controller unavailable"], "no_controller")
        }
        let json = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] ?? [:]

        switch (method, path) {
        case ("GET", "/health"):
            return (200, ["ok": true, "app": "AlicePet", "port": Int(port)], "ok")

        case ("GET", "/state"):
            return (200, buildStatePayload(controller: controller), "mode=\(controller.isPictureWorldActive ? "picture_world" : "desktop_pet")")

        case ("POST", "/observe"):
            return await handleObserve(controller: controller, json: json)

        case ("POST", "/look"):
            return handleLook(controller: controller, json: json)

        case ("POST", "/ardy"):
            return handleArdy(controller: controller, json: json)

        case ("POST", "/scene/load"):
            return handleSceneLoad(controller: controller, json: json)

        case ("POST", "/scene/exit"):
            controller.exitPictureWorld()
            return (200, ["ok": true, "mode": "desktop_pet"], "exited_picture_world")

        case ("POST", "/scene/walk-sit"):
            guard let anchorId = json["anchorId"] as? String, !anchorId.isEmpty else {
                return (400, ["ok": false, "error": "Missing anchorId"], "missing_anchorId")
            }
            let ok = controller.walkToSceneAnchorAndSit(anchorId: anchorId)
            return (ok ? 200 : 400, ["ok": ok, "anchorId": anchorId], "walk_sit=\(anchorId) ok=\(ok)")

        case ("POST", "/scene/stand-up"):
            let stopArdy = (json["stopArdy"] as? Bool) ?? true
            controller.standUpFromSceneSeat(stopArdy: stopArdy)
            return (200, ["ok": true], "stand_up stopArdy=\(stopArdy)")

        case ("POST", "/scene/anchors"):
            return handleRegisterAnchors(controller: controller, json: json)

        default:
            return (404, ["ok": false, "error": "Unknown route \(method) \(path)"], "not_found")
        }
    }

    // MARK: - Handlers

    private func buildStatePayload(controller: PetController) -> [String: Any] {
        let sceneState = controller.currentAvatarSceneState()
        let stateDict: [String: Any] = (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(sceneState))) as? [String: Any] ?? [:]
        let emotionSnap = controller.emotionStateMachine.snapshot()
        let emotionDict: [String: Any] = (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(emotionSnap))) as? [String: Any] ?? [:]

        let anchors: [[String: Any]] = (controller.pictureWorldRenderer.config?.anchors ?? []).map { a in
            [
                "id": a.id,
                "label": a.label,
                "kind": a.kind,
                "seatHeight": a.seatHeight ?? 0,
                "facingYawDeg": round(Double(a.facingYawRad * 180 / .pi)),
            ]
        }
        let scenes = (appDelegate?.discoverPictureWorldScenes() ?? []).map { $0.lastPathComponent }

        return [
            "ok": true,
            "isPictureWorldActive": controller.isPictureWorldActive,
            "poseMode": controller.poseMode.rawValue,
            "sceneName": controller.pictureWorldRenderer.config?.name ?? "",
            "sceneFolder": controller.pictureWorldRenderer.sceneFolderURL?.path ?? "",
            "sceneState": stateDict,
            "emotion": emotionDict,
            "anchors": anchors,
            "availableScenes": scenes,
        ]
    }

    private func handleObserve(controller: PetController, json: [String: Any]) async -> (Int, [String: Any], String) {
        let outPath = (json["outputPath"] as? String) ?? "/tmp/alicepet_observe.png"
        let outURL = URL(fileURLWithPath: outPath)
        let snapW: Int
        let snapH: Int
        if controller.isPictureWorldActive, let cfg = controller.pictureWorldRenderer.config {
            let scale = min(1.0, 1024.0 / Double(max(cfg.imageWidth, cfg.imageHeight)))
            snapW = max(320, Int((Double(cfg.imageWidth) * scale).rounded()))
            snapH = max(240, Int((Double(cfg.imageHeight) * scale).rounded()))
        } else {
            snapW = 640
            snapH = 900
        }
        do {
            try await controller.renderSnapshot(to: outURL, width: snapW, height: snapH, applyBaseState: false)
            var payload = buildStatePayload(controller: controller)
            payload["liveImagePath"] = outURL.path
            if let folder = controller.pictureWorldRenderer.sceneFolderURL {
                payload["roomBackgroundPath"] = folder.appendingPathComponent("bg.png").path
            }
            return (200, payload, "snapshot=\(outURL.path) (\(snapW)x\(snapH))")
        } catch {
            return (500, ["ok": false, "error": error.localizedDescription], "observe_failed")
        }
    }

    private func handleLook(controller: PetController, json: [String: Any]) -> (Int, [String: Any], String) {
        var applied: [String] = []
        if let poseRaw = json["pose"] as? String, let mode = PetPoseMode(rawValue: poseRaw) {
            controller.setPoseMode(mode)
            applied.append("pose=\(poseRaw)")
        }
        if let handRaw = json["hand"] as? String, let preset = HandGesturePreset(rawValue: handRaw) {
            controller.setHandGesture(preset, side: .both)
            applied.append("hand=\(handRaw)")
        }
        if let emo = json["emotion"] as? String {
            let intensity = (json["intensity"] as? NSNumber)?.floatValue ?? 0.7
            let snap = controller.emotionStateMachine.applyImpulse(emotion: emo, intensity: intensity)
            if json["expression"] == nil {
                controller.setPresetMood(controller.emotionStateMachine.recommendedVRMExpression)
            }
            applied.append("emotion=\(snap.dominantEmotion)(\(snap.dominantIntensity))")
        }
        if let expr = json["expression"] as? String {
            if expr.lowercased() == "neutral" {
                controller.setPresetMood(nil)
            } else {
                controller.applyExpression(named: expr, weight: 1.0)
            }
            applied.append("expr=\(expr)")
        }
        appDelegate?.refreshMenus()
        let emotionSnap = controller.emotionStateMachine.snapshot()
        let emoDict: [String: Any] = (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(emotionSnap))) as? [String: Any] ?? [:]
        return (200, ["ok": true, "applied": applied, "emotion": emoDict], applied.joined(separator: " "))
    }

    private func handleArdy(controller: PetController, json: [String: Any]) -> (Int, [String: Any], String) {
        let action = (json["action"] as? String)?.lowercased() ?? "prompt"
        if action == "stop" {
            controller.stopArdyMotionTest()
            appDelegate?.updateStatusBarTitle()
            appDelegate?.refreshMenus()
            return (200, ["ok": true, "ardyActive": false], "ardy=stop")
        }

        let prompt = (json["prompt"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
        let cfg = (json["cfg"] as? NSNumber)?.doubleValue ?? controller.localArdyService.cfgWeight
        if controller.sceneActionController.phase != .idle {
            controller.standUpFromSceneSeat(stopArdy: false)
        }
        if let prompt, !prompt.isEmpty {
            controller.localArdyService.sendLivePrompt(prompt, cfg: cfg)
        }
        var started = controller.localArdyActive
        if !started {
            started = controller.startArdyMotionTest()
        }
        appDelegate?.updateStatusBarTitle()
        appDelegate?.refreshMenus()
        return (started ? 200 : 500, ["ok": started, "ardyActive": started, "prompt": controller.localArdyService.currentPrompt], "ardy=\(action) ok=\(started)")
    }

    private func handleSceneLoad(controller: PetController, json: [String: Any]) -> (Int, [String: Any], String) {
        guard let target = json["scene"] as? String, !target.isEmpty else {
            return (400, ["ok": false, "error": "Missing scene"], "missing_scene")
        }
        let candidateURL: URL
        if target.hasPrefix("/") {
            candidateURL = URL(fileURLWithPath: target)
        } else if let match = appDelegate?.discoverPictureWorldScenes().first(where: { $0.lastPathComponent == target }) {
            candidateURL = match
        } else {
            candidateURL = URL(fileURLWithPath: "/Users/funnyliu/Downloads/VRM项目/scenes/\(target)")
        }
        do {
            try controller.loadPictureWorld(from: candidateURL)
            return (200, buildStatePayload(controller: controller), "loaded=\(candidateURL.lastPathComponent)")
        } catch {
            return (400, ["ok": false, "error": error.localizedDescription], "load_failed")
        }
    }

    private func handleRegisterAnchors(controller: PetController, json: [String: Any]) -> (Int, [String: Any], String) {
        guard controller.isPictureWorldActive else {
            return (400, ["ok": false, "error": "Picture-World is not active"], "no_scene")
        }
        guard let rawItems = json["anchors"] as? [[String: Any]], !rawItems.isEmpty else {
            return (400, ["ok": false, "error": "Missing non-empty anchors array"], "missing_anchors")
        }
        let specs: [PetController.UVAnchorSpec] = rawItems.compactMap { item in
            guard let id = item["id"] as? String,
                  let label = item["label"] as? String,
                  let u = (item["u"] as? NSNumber)?.floatValue,
                  let v = (item["v"] as? NSNumber)?.floatValue else { return nil }
            return PetController.UVAnchorSpec(
                id: id,
                label: label,
                kind: (item["kind"] as? String) ?? "seat",
                u: u,
                v: v,
                facingYawDeg: (item["facingYawDeg"] as? NSNumber)?.floatValue
            )
        }
        do {
            let saved = try controller.registerSceneAnchorsFromUV(specs)
            appDelegate?.refreshMenus()
            return (200, ["ok": true, "savedCount": saved.count, "anchors": saved.map { ["id": $0.id, "label": $0.label, "seatHeight": $0.seatHeight ?? 0] }], "registered=\(saved.count)")
        } catch {
            return (400, ["ok": false, "error": error.localizedDescription], "register_failed")
        }
    }
}
