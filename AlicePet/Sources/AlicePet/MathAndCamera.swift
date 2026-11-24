import AppKit
import Foundation
import simd

enum CameraViewPreset: String, CaseIterable {
    case fullBody = "full"
    case upperBody = "bust"
    case faceCloseUp = "face"
    case threeQuarter = "three_quarter"

    var displayTitle: String {
        switch self {
        case .fullBody: return "全身标准视角"
        case .upperBody: return "半身桌宠视角"
        case .faceCloseUp: return "脸部高清特写"
        case .threeQuarter: return "斜 45° 立体视角"
        }
    }
}

struct OrbitCamera {
    var target: SIMD3<Float> = SIMD3<Float>(0, 0.85, 0)
    var distance: Float = 2.4
    var yaw: Float = 0.0        // radians, 0 = front (+Z)
    var pitch: Float = 0.05     // radians, slight eye-level tilt
    var fovDegrees: Float = 32.0

    var defaultTarget: SIMD3<Float> = SIMD3<Float>(0, 0.85, 0)
    var defaultDistance: Float = 2.4
    private(set) var modelMinBounds = SIMD3<Float>(-0.3, 0, -0.2)
    private(set) var modelMaxBounds = SIMD3<Float>(0.3, 1.5, 0.2)

    mutating func fit(minBounds: SIMD3<Float>, maxBounds: SIMD3<Float>) {
        modelMinBounds = minBounds
        modelMaxBounds = maxBounds
        let size = maxBounds - minBounds
        let center = (minBounds + maxBounds) * 0.5
        let height = max(size.y, 0.5)
        defaultTarget = SIMD3<Float>(center.x, center.y + height * 0.03, center.z)
        let halfFov = (fovDegrees * .pi / 180.0) * 0.5
        defaultDistance = max((height * 0.56) / tan(halfFov), 1.0)
        reset()
    }

    mutating func reset() {
        target = defaultTarget
        distance = defaultDistance
        yaw = 0.0
        pitch = 0.04
    }

    mutating func apply(preset: CameraViewPreset) {
        let size = modelMaxBounds - modelMinBounds
        let center = (modelMinBounds + modelMaxBounds) * 0.5
        let height = max(size.y, 0.5)
        let halfFov = (fovDegrees * .pi / 180.0) * 0.5

        switch preset {
        case .fullBody:
            reset()
        case .upperBody:
            target = SIMD3<Float>(center.x, modelMinBounds.y + height * 0.67, center.z)
            distance = max((height * 0.30) / tan(halfFov), 0.72)
            yaw = 0.0
            pitch = 0.02
        case .faceCloseUp:
            target = SIMD3<Float>(center.x, modelMinBounds.y + height * 0.80, center.z + 0.02)
            distance = max((height * 0.145) / tan(halfFov), 0.42)
            yaw = 0.0
            pitch = 0.0
        case .threeQuarter:
            target = defaultTarget
            distance = defaultDistance * 0.96
            yaw = 35.0 * .pi / 180.0
            pitch = 0.08
        }
    }

    mutating func orbit(deltaX: Float, deltaY: Float) {
        yaw -= deltaX * 0.008
        pitch = max(-1.1, min(1.1, pitch + deltaY * 0.008))
    }

    mutating func pan(deltaX: Float, deltaY: Float) {
        let forward = normalize(target - eyePosition)
        let right = normalize(cross(forward, SIMD3<Float>(0, 1, 0)))
        let up = cross(right, forward)
        let scale = distance * 0.0018
        target -= right * (deltaX * scale)
        target += up * (deltaY * scale)
    }

    mutating func zoom(deltaY: Float) {
        distance = max(0.32, min(6.0, distance * (1.0 - deltaY * 0.04)))
    }

    var eyePosition: SIMD3<Float> {
        let cosPitch = cos(pitch)
        let x = target.x + distance * cosPitch * sin(yaw)
        let y = target.y + distance * sin(pitch)
        let z = target.z + distance * cosPitch * cos(yaw)
        return SIMD3<Float>(x, y, z)
    }

    var viewMatrix: matrix_float4x4 {
        Self.lookAt(eye: eyePosition, center: target, up: SIMD3<Float>(0, 1, 0))
    }

    func projectionMatrix(aspect: Float) -> matrix_float4x4 {
        let safeAspect = max(aspect, 0.1)
        return Self.perspectiveMetal(
            fovRadians: fovDegrees * .pi / 180.0,
            aspect: safeAspect,
            near: 0.05,
            far: 50.0
        )
    }

    static func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> matrix_float4x4 {
        let f = normalize(center - eye)
        let s = normalize(cross(f, up))
        let u = cross(s, f)

        var result = matrix_float4x4(diagonal: SIMD4<Float>(1, 1, 1, 1))
        result.columns.0 = SIMD4<Float>(s.x, u.x, -f.x, 0)
        result.columns.1 = SIMD4<Float>(s.y, u.y, -f.y, 0)
        result.columns.2 = SIMD4<Float>(s.z, u.z, -f.z, 0)
        result.columns.3 = SIMD4<Float>(-dot(s, eye), -dot(u, eye), dot(f, eye), 1)
        return result
    }

    /// Right-handed perspective projection mapped to Metal's `[0, 1]` NDC depth range
    /// (matches `VRMRenderer.makeProjectionMatrix` and prevents near-plane depth clipping/z-fighting).
    static func perspectiveMetal(fovRadians: Float, aspect: Float, near: Float, far: Float) -> matrix_float4x4 {
        let ys = 1.0 / tan(fovRadians * 0.5)
        let xs = ys / aspect
        let zs = far / (near - far)
        var result = matrix_float4x4()
        result.columns.0 = SIMD4<Float>(xs, 0, 0, 0)
        result.columns.1 = SIMD4<Float>(0, ys, 0, 0)
        result.columns.2 = SIMD4<Float>(0, 0, zs, -1)
        result.columns.3 = SIMD4<Float>(0, 0, zs * near, 0)
        return result
    }
}
