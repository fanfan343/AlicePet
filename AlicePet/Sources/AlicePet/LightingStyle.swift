import Foundation
import GLTFMetalKit
import VRMMetalKit
import simd

enum LightingPreset: String, CaseIterable {
    case animeSoftDay = "soft_day"
    case crispCel = "crisp_cel"
    case warmDeskLamp = "warm_lamp"
    case nightRim = "night_rim"

    var displayTitle: String {
        switch self {
        case .animeSoftDay: return "日光通透 (默认)"
        case .crispCel: return "赛璐璐立体 (明暗利落)"
        case .warmDeskLamp: return "暖光午后 (柔和氛围)"
        case .nightRim: return "暗夜轮廓光 (适配深色壁纸)"
        }
    }
}

enum OutlineStyle: String, CaseIterable {
    case off = "off"
    case subtleWarm = "subtle"
    case celSoft = "cel"

    var displayTitle: String {
        switch self {
        case .off: return "描边：关闭 (默认 · 最干净)"
        case .subtleWarm: return "描边：极细暖棕"
        case .celSoft: return "描边：柔和赛璐璐"
        }
    }
}

/// Centralized MToon + GLTF prop look-dev presets.
/// Preserves author-crafted face shade colors, emission maps, and balanced anime lighting.
@MainActor
struct LightingConfigurator {
    static func apply(preset: LightingPreset, outline: OutlineStyle, to renderer: VRMRenderer) {
        // Preserve author-authored MToon shadeColor, toony/shift, and _EmissionMap on face & skin
        renderer.disableAutoMaterialOverrides = true
        renderer.setLightNormalizationMode(.radiometric)

        // Default directional rim off unless a night/rim preset enables it
        renderer.additiveDirectionalRimEnabled = false
        renderer.additiveDirectionalRimPower = 7.0

        switch preset {
        case .animeSoftDay:
            // Luminous fair anime skin + clean cel folds on dress & hair without blowing out highlights
            renderer.setLight(0, direction: SIMD3<Float>(-0.26, -0.24, -0.93), color: SIMD3<Float>(1.00, 0.98, 0.96), intensity: 1.08)
            renderer.setLight(1, direction: SIMD3<Float>(0.48, -0.06, -0.87), color: SIMD3<Float>(0.90, 0.94, 1.00), intensity: 0.46)
            renderer.setLight(2, direction: SIMD3<Float>(0.15, -0.42, 0.89), color: SIMD3<Float>(0.96, 0.98, 1.00), intensity: 0.38)
            renderer.setHemisphereAmbient(
                sky: SIMD3<Float>(0.25, 0.25, 0.27),
                ground: SIMD3<Float>(0.21, 0.18, 0.19)
            )

        case .crispCel:
            // Stronger angled key vs cool sky fill for crisp anime shadow boundaries
            renderer.setLight(0, direction: SIMD3<Float>(-0.42, -0.34, -0.84), color: SIMD3<Float>(1.00, 0.99, 0.97), intensity: 1.18)
            renderer.setLight(1, direction: SIMD3<Float>(0.52, -0.05, -0.85), color: SIMD3<Float>(0.84, 0.90, 1.00), intensity: 0.28)
            renderer.setLight(2, direction: SIMD3<Float>(0.05, -0.40, 0.91), color: SIMD3<Float>(1.00, 1.00, 1.00), intensity: 0.36)
            renderer.setHemisphereAmbient(
                sky: SIMD3<Float>(0.20, 0.21, 0.24),
                ground: SIMD3<Float>(0.14, 0.13, 0.15)
            )

        case .warmDeskLamp:
            // Warm golden afternoon / desk lamp key with soft lavender shadow fill
            renderer.setLight(0, direction: SIMD3<Float>(-0.40, -0.32, -0.86), color: SIMD3<Float>(1.00, 0.91, 0.80), intensity: 1.10)
            renderer.setLight(1, direction: SIMD3<Float>(0.48, -0.08, -0.87), color: SIMD3<Float>(0.84, 0.86, 0.98), intensity: 0.38)
            renderer.setLight(2, direction: SIMD3<Float>(0.18, -0.45, 0.87), color: SIMD3<Float>(1.00, 0.92, 0.80), intensity: 0.40)
            renderer.setHemisphereAmbient(
                sky: SIMD3<Float>(0.23, 0.20, 0.22),
                ground: SIMD3<Float>(0.20, 0.16, 0.14)
            )

        case .nightRim:
            // Calm evening front fill + crisp moonlight rim suited for dark desktop wallpapers
            renderer.setLight(0, direction: SIMD3<Float>(-0.25, -0.26, -0.93), color: SIMD3<Float>(0.88, 0.92, 1.00), intensity: 0.78)
            renderer.setLight(1, direction: SIMD3<Float>(0.44, -0.08, -0.89), color: SIMD3<Float>(0.74, 0.80, 0.96), intensity: 0.28)
            renderer.setLight(2, direction: SIMD3<Float>(-0.15, -0.38, 0.91), color: SIMD3<Float>(0.92, 0.96, 1.00), intensity: 0.55)
            renderer.setHemisphereAmbient(
                sky: SIMD3<Float>(0.16, 0.17, 0.22),
                ground: SIMD3<Float>(0.12, 0.12, 0.16)
            )
            renderer.additiveDirectionalRimEnabled = true
            renderer.additiveDirectionalRimPower = 7.0
        }

        // Keep outlineMinimumPixels at 0 so small lip/eyelid/ruffle details never extrude dark blobs
        renderer.outlineMinimumPixels = 0.0
        switch outline {
        case .off:
            renderer.outlineScale = 0.0
            renderer.outlineColor = SIMD3<Float>(0, 0, 0)
        case .subtleWarm:
            renderer.outlineScale = 0.22
            renderer.outlineColor = SIMD3<Float>(0.38, 0.27, 0.24)
        case .celSoft:
            renderer.outlineScale = 0.40
            renderer.outlineColor = SIMD3<Float>(0.32, 0.23, 0.22)
        }
    }

    /// Balanced front-key + opposite-fill + backface-bounce directional lights for glTF PBR props
    /// so double-sided book pages and tea-set items are evenly lit across both halves.
    static func propLights(for preset: LightingPreset) -> [GLTFPunctualLightUniform] {
        let keyColor: SIMD3<Float>
        let fillColor: SIMD3<Float>
        let keyIntensity: Float
        let fillIntensity: Float
        let bounceIntensity: Float

        switch preset {
        case .animeSoftDay:
            keyColor = SIMD3<Float>(1.00, 0.98, 0.95)
            fillColor = SIMD3<Float>(0.92, 0.94, 1.00)
            keyIntensity = 2.2
            fillIntensity = 1.5
            bounceIntensity = 1.6
        case .crispCel:
            keyColor = SIMD3<Float>(1.00, 0.99, 0.97)
            fillColor = SIMD3<Float>(0.90, 0.93, 1.00)
            keyIntensity = 2.4
            fillIntensity = 1.4
            bounceIntensity = 1.5
        case .warmDeskLamp:
            keyColor = SIMD3<Float>(1.00, 0.91, 0.79)
            fillColor = SIMD3<Float>(0.86, 0.88, 0.98)
            keyIntensity = 2.2
            fillIntensity = 1.4
            bounceIntensity = 1.5
        case .nightRim:
            keyColor = SIMD3<Float>(0.88, 0.92, 1.00)
            fillColor = SIMD3<Float>(0.76, 0.82, 0.96)
            keyIntensity = 1.6
            fillIntensity = 1.15
            bounceIntensity = 1.2
        }

        return [
            GLTFPunctualLightUniform(
                type: .directional,
                color: keyColor,
                intensity: keyIntensity,
                direction: normalize(SIMD3<Float>(-0.30, -0.55, -0.78))
            ),
            GLTFPunctualLightUniform(
                type: .directional,
                color: fillColor,
                intensity: fillIntensity,
                direction: normalize(SIMD3<Float>(0.42, -0.45, -0.79))
            ),
            GLTFPunctualLightUniform(
                type: .directional,
                color: keyColor,
                intensity: bounceIntensity,
                direction: normalize(SIMD3<Float>(0.0, 0.65, 0.76))
            ),
        ]
    }
}
