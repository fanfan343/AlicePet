import Foundation
import simd

/// Adaptive 1€ low-pass filter (scalar), ported from XR Animator's VRPN OneEuroFilter.
/// Low cutoff when stationary (eliminates micro-jitter); high cutoff during fast motion (low lag).
struct OneEuroFilterScalar {
    var minCutoff: Float
    var beta: Float
    var dCutoff: Float

    private var initialized = false
    private var prevRaw: Float = 0
    private var prevFiltered: Float = 0
    private var prevDeriv: Float = 0
    private var lastTime: TimeInterval = 0

    init(minCutoff: Float = 1.2, beta: Float = 0.08, dCutoff: Float = 1.0) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    mutating func reset() {
        initialized = false
        prevRaw = 0
        prevFiltered = 0
        prevDeriv = 0
        lastTime = 0
    }

    mutating func filter(_ value: Float, timestamp: TimeInterval) -> Float {
        if !initialized {
            initialized = true
            prevRaw = value
            prevFiltered = value
            prevDeriv = 0
            lastTime = timestamp
            return value
        }

        let dt = Float(max(timestamp - lastTime, 1.0 / 240.0))
        lastTime = timestamp

        let dx = (value - prevRaw) / dt
        prevRaw = value

        let alphaD = alpha(cutoff: dCutoff, dt: dt)
        let deriv = prevDeriv + alphaD * (dx - prevDeriv)
        prevDeriv = deriv

        let cutoff = minCutoff + beta * abs(deriv)
        let a = alpha(cutoff: cutoff, dt: dt)
        let filtered = prevFiltered + a * (value - prevFiltered)
        prevFiltered = filtered
        return filtered
    }

    private func alpha(cutoff: Float, dt: Float) -> Float {
        let tau = 1.0 / (2.0 * Float.pi * max(cutoff, 0.001))
        return 1.0 / (1.0 + tau / dt)
    }
}

/// Adaptive 1€ low-pass filter for unit quaternions using geodesic SLERP.
struct OneEuroFilterQuaternion {
    var minCutoff: Float
    var beta: Float
    var dCutoff: Float

    private var initialized = false
    private var prevFiltered = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var prevRaw = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var prevSpeed: Float = 0
    private var lastTime: TimeInterval = 0

    init(minCutoff: Float = 1.6, beta: Float = 0.12, dCutoff: Float = 1.0) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    mutating func reset() {
        initialized = false
        prevSpeed = 0
        lastTime = 0
    }

    mutating func filter(_ value: simd_quatf, timestamp: TimeInterval) -> simd_quatf {
        var q = simd_normalize(value)
        if !initialized {
            initialized = true
            prevRaw = q
            prevFiltered = q
            prevSpeed = 0
            lastTime = timestamp
            return q
        }

        if simd_dot(q, prevFiltered) < 0 {
            q = -q
        }

        let dt = Float(max(timestamp - lastTime, 1.0 / 240.0))
        lastTime = timestamp

        let dotRaw = max(-1.0, min(1.0, abs(simd_dot(prevRaw, q))))
        let angularSpeed = (2.0 * acos(dotRaw)) / dt
        prevRaw = q

        let alphaD = alpha(cutoff: dCutoff, dt: dt)
        prevSpeed = prevSpeed + alphaD * (angularSpeed - prevSpeed)

        let cutoff = minCutoff + beta * prevSpeed
        let a = alpha(cutoff: cutoff, dt: dt)
        let filtered = simd_normalize(simd_slerp(prevFiltered, q, a))
        prevFiltered = filtered
        return filtered
    }

    private func alpha(cutoff: Float, dt: Float) -> Float {
        let tau = 1.0 / (2.0 * Float.pi * max(cutoff, 0.001))
        return 1.0 / (1.0 + tau / dt)
    }
}
