import Foundation
import simd

extension PictureWorldRenderer {
    // MARK: - LLM Spatial & Occlusion Telemetry

    /// Samples MoGe-2 metric depth across 5 vertical body points on Alice (`feet`, `knees`, `waist`,
    /// `chest`, `head`) to give future LLM systems exact position, occlusion, and boundary awareness.
    func evaluateSpatialSnapshot(viewportAspect: Float) -> PictureWorldSpatialSnapshot {
        guard let cfg = config, let mats = cameraMatrices(viewportAspect: viewportAspect) else {
            return PictureWorldSpatialSnapshot(
                normalizedRoomPosition: [0, 0],
                distanceToCameraMeters: 2.4,
                screenNDC: [0, 0],
                isOccluded: false,
                occlusionRatio: 0,
                occludedRegions: [],
                boundaryContacts: [],
                isNearBoundary: false
            )
        }

        var nx: Float = 0
        var nz: Float = 0
        if let minB = cfg.walkBoundsMin, minB.count == 2,
           let maxB = cfg.walkBoundsMax, maxB.count == 2 {
            let cx = 0.5 * (minB[0] + maxB[0])
            let cz = 0.5 * (minB[1] + maxB[1])
            let hx = max(0.1, 0.5 * (maxB[0] - minB[0]))
            let hz = max(0.1, 0.5 * (maxB[1] - minB[1]))
            nx = min(max((avatarWorldPosition.x - cx) / hx, -1), 1)
            nz = min(max((avatarWorldPosition.z - cz) / hz, -1), 1)
        }

        let camPos = cfg.camPos
        let fwd = simd_normalize(cfg.camTarget - camPos)
        let right = simd_normalize(simd_cross(fwd, cfg.camUp))
        let up = simd_normalize(simd_cross(right, fwd))
        let imgTanHalfY = tan((cfg.fovYDegrees * .pi / 180.0) * 0.5)
        let imgAspect = Float(cfg.imageWidth) / max(Float(cfg.imageHeight), 1.0)
        let imgTanHalfX = imgTanHalfY * imgAspect

        let torsoWorld = avatarWorldPosition + SIMD3<Float>(0, 0.95, 0)
        let distToCam = simd_length(torsoWorld - camPos)
        let vp = mats.projection * mats.view
        let clipTorso = vp * SIMD4<Float>(torsoWorld, 1.0)
        let ndcX = clipTorso.w > 1e-4 ? (clipTorso.x / clipTorso.w) : 0
        let ndcY = clipTorso.w > 1e-4 ? (clipTorso.y / clipTorso.w) : 0

        let sampleHeights: [(name: String, y: Float)] = [
            ("feet", 0.12),
            ("knees", 0.45),
            ("waist", 0.82),
            ("chest", 1.15),
            ("head", 1.42),
        ]

        var occludedNames: [String] = []
        if depthMeters.count >= cfg.depthWidth * cfg.depthHeight {
            for sample in sampleHeights {
                let pt = avatarWorldPosition + SIMD3<Float>(0, sample.y, 0)
                let rel = pt - camPos
                let ptDepth = simd_dot(rel, fwd)
                guard ptDepth > cfg.nearZ else { continue }
                let rx = simd_dot(rel, right) / ptDepth
                let ry = simd_dot(rel, up) / ptDepth
                let u = 0.5 * (rx / imgTanHalfX + 1.0)
                let v = 0.5 * (1.0 - ry / imgTanHalfY)
                if u >= 0 && u <= 1 && v >= 0 && v <= 1 {
                    let px = min(cfg.depthWidth - 1, max(0, Int(u * Float(cfg.depthWidth))))
                    let py = min(cfg.depthHeight - 1, max(0, Int(v * Float(cfg.depthHeight))))
                    let sceneDepth = depthMeters[py * cfg.depthWidth + px] + depthBiasMeters
                    if sceneDepth > 0.05 && sceneDepth < ptDepth - 0.06 {
                        occludedNames.append(sample.name)
                    }
                }
            }
        }

        let ratio = Float(occludedNames.count) / Float(sampleHeights.count)
        let contactEval = clampAvatarPositionWithContacts(avatarWorldPosition, viewportAspect: viewportAspect)
        let mergedContacts = Array(Set(lastBoundaryContacts).union(contactEval.contacts)).sorted()

        return PictureWorldSpatialSnapshot(
            normalizedRoomPosition: [nx, nz],
            distanceToCameraMeters: distToCam,
            screenNDC: [ndcX, ndcY],
            isOccluded: !occludedNames.isEmpty,
            occlusionRatio: ratio,
            occludedRegions: occludedNames,
            boundaryContacts: mergedContacts,
            isNearBoundary: lastIsNearBoundary || contactEval.isNearBoundary
        )
    }

    /// Unprojects a normalized 2D image coordinate `(u, v)` in `[0, 1]` into 3D scene world space
    /// using the active MoGe-2 metric `depthMeters` buffer (for VLM-authored scene anchors).
    func unprojectImageUV(u: Float, v: Float) -> SIMD3<Float>? {
        guard let cfg = config,
              depthMeters.count >= cfg.depthWidth * cfg.depthHeight,
              u >= 0, u <= 1, v >= 0, v <= 1 else {
            return nil
        }
        let px = min(cfg.depthWidth - 1, max(0, Int(u * Float(cfg.depthWidth))))
        let py = min(cfg.depthHeight - 1, max(0, Int(v * Float(cfg.depthHeight))))
        // Take a 3x3 median depth sample to ignore single-pixel noise on furniture edges
        var samples: [Float] = []
        for dy in -1...1 {
            for dx in -1...1 {
                let sx = min(cfg.depthWidth - 1, max(0, px + dx))
                let sy = min(cfg.depthHeight - 1, max(0, py + dy))
                let d = depthMeters[sy * cfg.depthWidth + sx] + depthBiasMeters
                if d.isFinite && d > 0.15 && d < 30.0 {
                    samples.append(d)
                }
            }
        }
        guard !samples.isEmpty else { return nil }
        samples.sort()
        let d = samples[samples.count / 2]

        let camPos = cfg.camPos
        let fwd = simd_normalize(cfg.camTarget - camPos)
        let right = simd_normalize(simd_cross(fwd, cfg.camUp))
        let up = simd_normalize(simd_cross(right, fwd))
        let imgTanHalfY = tan((cfg.fovYDegrees * .pi / 180.0) * 0.5)
        let imgAspect = Float(cfg.imageWidth) / max(Float(cfg.imageHeight), 1.0)
        let imgTanHalfX = imgTanHalfY * imgAspect

        let rayDir = fwd
            + right * ((u * 2.0 - 1.0) * imgTanHalfX)
            + up * ((1.0 - v * 2.0) * imgTanHalfY)
        return camPos + rayDir * d
    }
}
