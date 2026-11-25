import Foundation

/// Structured, JSON-serializable real-time spatial & perceptual state snapshot of Alice
/// designed for future LLM / agentic tool-calling and contextual awareness.
///
/// Covers where Alice is in the 3D scene, whether she is occluded by foreground furniture
/// (measured directly against MoGe-2 metric depth), whether she is touching/near a room
/// boundary, and which body/face engines are currently active.
struct AvatarSceneState: Codable, Equatable, Sendable {
    let timestamp: TimeInterval
    /// `"picture_world"` when inside a 2D→3D depth scene, or `"desktop_pet"` in floating transparent mode.
    let mode: String
    /// Loaded Picture-World scene title, if any.
    let sceneName: String?
    /// World floor position `[x, y, z]` in meters (`y == 0` on the fitted floor plane).
    let positionMeters: [Float]
    /// Normalized position `[nx, nz]` in `[-1, +1]` within the walkable room bounds.
    let normalizedRoomPosition: [Float]
    /// World facing yaw in degrees (`0°` = facing camera / `+Z`).
    let facingYawDegrees: Float
    /// Distance from the scene camera lens in meters.
    let distanceToCameraMeters: Float
    /// Projected torso position `[ndcX, ndcY]` in `[-1, +1]` within the current viewport.
    let screenNDC: [Float]
    /// `true` when foreground scene geometry is closer to the camera than Alice across sampled body regions.
    let isOccluded: Bool
    /// Fraction of sampled body regions (`feet / knees / waist / chest / head`) occluded by scene depth (`0.0 ... 1.0`).
    let occlusionRatio: Float
    /// Named body regions currently occluded (e.g. `["feet", "knees", "waist"]`).
    let occludedRegions: [String]
    /// Active boundary contacts (`"left"`, `"right"`, `"front"`, `"back"`), empty when freely inside room.
    let boundaryContacts: [String]
    /// `true` when touching or within 0.20m of any walkable/frustum boundary.
    let isNearBoundary: Bool
    /// Active Layer 0 body driver (`"AuthorPose"`, `"ARDY"`, `"ExternalVMC"`).
    let activeBodyEngine: String
    /// Active Layer 2 face driver (`"Procedural"`, `"Audio2Face"`, `"ExternalVMC"`).
    let activeFaceEngine: String
    /// Current ARDY text prompt when ARDY is running, if any.
    let activeMotionPrompt: String?
    /// Current Picture-World canvas zoom factor (`1.0` = default unzoomed view).
    let canvasZoom: Float

    func toJSONString(pretty: Bool = false) -> String? {
        let encoder = JSONEncoder()
        if pretty {
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        }
        guard let data = try? encoder.encode(self) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
