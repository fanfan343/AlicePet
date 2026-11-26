import Foundation
import simd

struct SceneAnchor: Codable, Equatable, Sendable {
    let id: String
    let label: String
    let kind: String // "seat" | "stand_spot"
    let approachPosition: [Float]
    let targetPosition: [Float]
    let seatHeight: Float?
    let facingYawRad: Float

    var approachPos: SIMD3<Float> {
        approachPosition.count == 3
            ? SIMD3<Float>(approachPosition[0], approachPosition[1], approachPosition[2])
            : .zero
    }
    var targetPos: SIMD3<Float> {
        targetPosition.count == 3
            ? SIMD3<Float>(targetPosition[0], targetPosition[1], targetPosition[2])
            : approachPos
    }
}

struct PictureWorldSceneConfig: Codable {
    let version: Int
    let name: String
    let imageWidth: Int
    let imageHeight: Int
    let depthWidth: Int
    let depthHeight: Int
    let fovYDegrees: Float
    let cameraPosition: [Float]
    let cameraTarget: [Float]
    let cameraUp: [Float]
    let avatarSpawnPosition: [Float]
    let avatarSpawnYawRad: Float
    let avatarScale: Float?
    let walkBoundsMin: [Float]?
    let walkBoundsMax: [Float]?
    let nearClip: Float?
    let farClip: Float?
    let lightingPreset: String?
    let anchors: [SceneAnchor]?
    let walkableGrid: WalkableGrid?

    var camPos: SIMD3<Float> { vec3(cameraPosition, fallback: SIMD3<Float>(0, 1.35, 2.8)) }
    var camTarget: SIMD3<Float> { vec3(cameraTarget, fallback: SIMD3<Float>(0, 1.0, 0)) }
    var camUp: SIMD3<Float> { vec3(cameraUp, fallback: SIMD3<Float>(0, 1, 0)) }
    var spawnPos: SIMD3<Float> { vec3(avatarSpawnPosition, fallback: .zero) }
    var nearZ: Float { nearClip ?? 0.05 }
    var farZ: Float { farClip ?? 50.0 }

    private func vec3(_ arr: [Float], fallback: SIMD3<Float>) -> SIMD3<Float> {
        guard arr.count == 3 else { return fallback }
        return SIMD3<Float>(arr[0], arr[1], arr[2])
    }
}

struct PictureWorldSpatialSnapshot {
    let normalizedRoomPosition: [Float]
    let distanceToCameraMeters: Float
    let screenNDC: [Float]
    let isOccluded: Bool
    let occlusionRatio: Float
    let occludedRegions: [String]
    let boundaryContacts: [String]
    let isNearBoundary: Bool
}

enum PictureWorldShaders {
    static let metalSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct PictureWorldUniforms {
        float2 uvScale;
        float2 uvOffset;
        float nearZ;
        float farZ;
        float imgTanHalfFovX;
        float imgTanHalfFovY;
        float4 camPos;       // xyz = camPos, w = depthBiasMeters
        float4 camRight;
        float4 camUp;
        float4 camFwd;
        float4 avatarFootPos;
        float4 shadowParams; // x=radius, y=strength, z=yTol, w=enabled
    };

    struct VSOut {
        float4 position [[position]];
        float2 screenUV;
    };

    struct FSOut {
        float4 color [[color(0)]];
        float depth [[depth(any)]];
    };

    vertex VSOut picture_world_vs(uint vid [[vertex_id]]) {
        float2 pos[3] = {
            float2(-1.0, -1.0),
            float2( 3.0, -1.0),
            float2(-1.0,  3.0)
        };
        float2 p = pos[vid];
        VSOut out;
        out.position = float4(p, 0.0, 1.0);
        out.screenUV = float2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
        return out;
    }

    fragment FSOut picture_world_fs(
        VSOut in [[stage_in]],
        texture2d<float> bgTex [[texture(0)]],
        texture2d<float> depthTex [[texture(1)]],
        sampler linSampler [[sampler(0)]],
        sampler nearSampler [[sampler(1)]],
        constant PictureWorldUniforms& u [[buffer(0)]]
    ) {
        float2 uv = in.screenUV * u.uvScale + u.uvOffset;
        float4 col = bgTex.sample(linSampler, uv);
        float d = depthTex.sample(nearSampler, uv).r + u.camPos.w;
        d = clamp(d, u.nearZ + 0.001, u.farZ - 0.001);

        // Exact Metal [0, 1] perspective depth matching our off-axis projection matrix
        float zNdc = (u.farZ / (u.farZ - u.nearZ)) * (1.0 - (u.nearZ / d));

        // Soft floor contact shadow grounded at avatarFootPos using exact image-UV ray reconstruction
        if (u.shadowParams.w > 0.5) {
            float rayX = (uv.x * 2.0 - 1.0) * u.imgTanHalfFovX;
            float rayY = (1.0 - uv.y * 2.0) * u.imgTanHalfFovY;
            float3 rayDir = u.camFwd.xyz
                + u.camRight.xyz * rayX
                + u.camUp.xyz * rayY;
            float3 worldP = u.camPos.xyz + rayDir * d;
            float dy = abs(worldP.y - u.avatarFootPos.y);
            if (dy < u.shadowParams.z) {
                float2 dXZ = worldP.xz - u.avatarFootPos.xz;
                float r2 = dot(dXZ, dXZ) / max(u.shadowParams.x * u.shadowParams.x, 1e-4);
                float yAtten = 1.0 - smoothstep(0.0, u.shadowParams.z, dy);
                float shadow = exp(-r2 * 2.4) * u.shadowParams.y * yAtten;
                col.rgb *= (1.0 - shadow);
            }
        }

        FSOut out;
        out.color = float4(col.rgb, 1.0);
        out.depth = clamp(zNdc, 0.0, 0.99999);
        return out;
    }
    """
}
