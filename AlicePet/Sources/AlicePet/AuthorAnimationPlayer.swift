import Foundation
import simd

/// Loads and interpolates exported author Unity `.anim` clips (`animations.json`).
@MainActor
final class AuthorAnimationPlayer {
    private(set) var bundle: AuthorAnimationBundle?
    private(set) var standReferenceRootT = SIMD3<Float>(0, 0.978, 0)

    var hasClips: Bool { !(bundle?.clips.isEmpty ?? true) }

    func load(from folderURL: URL?) {
        bundle = nil
        guard let folderURL else { return }
        let fileURL = folderURL.appendingPathComponent("animations.json")
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        do {
            let data = try Data(contentsOf: fileURL)
            let decoded = try JSONDecoder().decode(AuthorAnimationBundle.self, from: data)
            self.bundle = decoded
            if let standFrame = decoded.clips["stand"]?.frames.first, standFrame.rootT.count == 3 {
                standReferenceRootT = SIMD3<Float>(standFrame.rootT[0], standFrame.rootT[1], standFrame.rootT[2])
            }
            AppLogger.shared.info("Animation", "Loaded animations.json (\(decoded.clips.count) author clips: \(decoded.clips.keys.sorted().joined(separator: ", ")))")
        } catch {
            AppLogger.shared.error("Animation", "Failed to decode animations.json: \(error)")
        }
    }

    func clip(named id: String) -> AuthorAnimationClip? {
        bundle?.clips[id]
    }

    func initialRootZ(of clip: AuthorAnimationClip) -> Float {
        guard let f0 = clip.frames.first, f0.rootT.count == 3 else { return 0 }
        return f0.rootT[2]
    }

    func sample(clip: AuthorAnimationClip, at rawTime: Float) -> SampledAuthorPose? {
        let frames = clip.frames
        guard !frames.isEmpty else { return nil }
        if frames.count == 1 || clip.duration <= 1e-4 {
            return makePose(from: frames[0])
        }

        let t: Float
        if clip.loop {
            let mod = fmodf(rawTime, clip.duration)
            t = mod >= 0 ? mod : (mod + clip.duration)
        } else {
            t = max(0, min(clip.duration, rawTime))
        }

        // Uniformly resampled frames -> O(1) index lookup
        let norm = (t / clip.duration) * Float(frames.count - 1)
        let idx0 = max(0, min(frames.count - 1, Int(floor(norm))))
        let idx1 = min(frames.count - 1, idx0 + 1)
        let u = max(0, min(1, norm - Float(idx0)))

        if idx0 == idx1 || u <= 1e-4 {
            return makePose(from: frames[idx0])
        }

        let f0 = frames[idx0]
        let f1 = frames[idx1]
        let t0 = vec3(f0.rootT)
        let t1 = vec3(f1.rootT)
        let q0 = quat(f0.rootQ)
        let q1 = quat(f1.rootQ)

        var q1Aligned = q1
        if simd_dot(q0, q1Aligned) < 0 {
            q1Aligned = -q1Aligned
        }

        var blendedMuscles: [String: Float] = [:]
        blendedMuscles.reserveCapacity(max(f0.m.count, f1.m.count))
        for (k, v0) in f0.m {
            let v1 = f1.m[k] ?? 0
            blendedMuscles[k] = v0 + (v1 - v0) * u
        }
        for (k, v1) in f1.m where blendedMuscles[k] == nil {
            blendedMuscles[k] = v1 * u
        }

        return SampledAuthorPose(
            rootPositionUnity: simd_mix(t0, t1, SIMD3<Float>(repeating: u)),
            rootRotationUnity: simd_normalize(simd_slerp(q0, q1Aligned, u)),
            muscles: blendedMuscles
        )
    }

    private func makePose(from frame: AuthorAnimationFrame) -> SampledAuthorPose {
        SampledAuthorPose(
            rootPositionUnity: vec3(frame.rootT),
            rootRotationUnity: quat(frame.rootQ),
            muscles: frame.m
        )
    }

    private func vec3(_ arr: [Float]) -> SIMD3<Float> {
        guard arr.count == 3 else { return .zero }
        return SIMD3<Float>(arr[0], arr[1], arr[2])
    }

    private func quat(_ arr: [Float]) -> simd_quatf {
        guard arr.count == 4 else { return simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) }
        return simd_normalize(simd_quatf(ix: arr[0], iy: arr[1], iz: arr[2], r: arr[3]))
    }
}
