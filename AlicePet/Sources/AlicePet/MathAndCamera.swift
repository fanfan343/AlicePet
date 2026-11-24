import Foundation
import AppKit
import simd

struct OrbitCamera {
    var target: SIMD3<Float> = SIMD3<Float>(0, 0.85, 0)
    var distance: Float = 2.1
    var yaw: Float = 0.0        // radians, 0 = front (+Z)
    var pitch: Float = 0.08     // radians, slight eye-level downward tilt
    var fovDegrees: Float = 38.0

    var defaultTarget: SIMD3<Float> = SIMD3<Float>(0, 0.85, 0)
    var defaultDistance: Float = 2.1

    mutating func fit(minBounds: SIMD3<Float>, maxBounds: SIMD3<Float>) {
        let size = maxBounds - minBounds
        let center = (minBounds + maxBounds) * 0.5
        let height = max(size.y, 0.5)
        target = SIMD3<Float>(center.x, center.y + height * 0.03, center.z)
        let halfFov = (fovDegrees * .pi / 180.0) * 0.5
        distance = max((height * 0.56) / tan(halfFov), 0.9)
        yaw = 0.0
        pitch = 0.05
        defaultTarget = target
        defaultDistance = distance
    }

    mutating func reset() {
        target = defaultTarget
        distance = defaultDistance
        yaw = 0.0
        pitch = 0.05
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
        distance = max(0.45, min(6.0, distance * (1.0 - deltaY * 0.04)))
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
        return Self.perspective(
            fovRadians: fovDegrees * .pi / 180.0,
            aspect: safeAspect,
            near: 0.05,
            far: 100.0
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

    static func perspective(fovRadians: Float, aspect: Float, near: Float, far: Float) -> matrix_float4x4 {
        let tanHalfFov = tan(fovRadians / 2.0)
        var result = matrix_float4x4()
        result.columns.0 = SIMD4<Float>(1.0 / (aspect * tanHalfFov), 0, 0, 0)
        result.columns.1 = SIMD4<Float>(0, 1.0 / tanHalfFov, 0, 0)
        result.columns.2 = SIMD4<Float>(0, 0, -(far + near) / (far - near), -1)
        result.columns.3 = SIMD4<Float>(0, 0, -(2.0 * far * near) / (far - near), 0)
        return result
    }
}
