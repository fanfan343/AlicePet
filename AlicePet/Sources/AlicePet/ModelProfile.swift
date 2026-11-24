import Foundation
import simd

struct HairBoneAdjustment: Codable {
    let nodeName: String
    let axis: [Float]
    let angleRad: Float

    var normalizedAxis: SIMD3<Float> {
        guard axis.count == 3 else { return SIMD3<Float>(0, 0, 1) }
        let v = SIMD3<Float>(axis[0], axis[1], axis[2])
        let len = simd_length(v)
        return len > 1e-5 ? (v / len) : SIMD3<Float>(0, 0, 1)
    }
}

/// Externalized per-model adaptation profile (loaded from `profiles/*.json`).
/// Keeps Alice-specific node names / morph rules out of Swift code so any generic VRM loads cleanly.
struct ModelProfile: Codable {
    let id: String
    let displayName: String
    let modelNameHints: [String]
    let assetsFolderRelative: String?
    let outfitNodeNames: [String]
    let outfitMorphNames: [String]
    let hairStyles: [String: [HairBoneAdjustment]]
    let bookPropOffset: [Float]?
    let bookTiltRad: Float?

    var hasCustomOutfits: Bool {
        !outfitNodeNames.isEmpty || !outfitMorphNames.isEmpty
    }

    var bookBasePosition: SIMD3<Float> {
        guard let arr = bookPropOffset, arr.count == 3 else {
            return SIMD3<Float>(0.0, 0.90, 0.31)
        }
        return SIMD3<Float>(arr[0], arr[1], arr[2])
    }

    var bookBaseTilt: Float {
        bookTiltRad ?? 0.46
    }

    static let generic = ModelProfile(
        id: "generic",
        displayName: "Generic VRM",
        modelNameHints: [],
        assetsFolderRelative: nil,
        outfitNodeNames: [],
        outfitMorphNames: [],
        hairStyles: [:],
        bookPropOffset: nil,
        bookTiltRad: nil
    )

    static func resolve(forModelName modelName: String?, fileURL: URL) -> ModelProfile {
        let candidates = locateProfileDirectories(near: fileURL)
        let haystack = "\(modelName ?? "") \(fileURL.lastPathComponent)".lowercased()
        let fm = FileManager.default

        for dir in candidates {
            guard let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else {
                continue
            }
            for file in files where file.pathExtension.lowercased() == "json" {
                guard let data = try? Data(contentsOf: file),
                      let profile = try? JSONDecoder().decode(ModelProfile.self, from: data) else {
                    continue
                }
                if profile.modelNameHints.contains(where: { haystack.contains($0.lowercased()) }) {
                    AppLogger.shared.info("Profile", "Matched model profile '\(profile.id)' from \(file.path)")
                    return profile
                }
            }
        }

        AppLogger.shared.info("Profile", "Using generic VRM profile for '\(modelName ?? fileURL.lastPathComponent)'")
        return .generic
    }

    private static func locateProfileDirectories(near modelURL: URL) -> [URL] {
        var dirs: [URL] = []
        let modelParent = modelURL.deletingLastPathComponent()
        dirs.append(modelParent.appendingPathComponent("profiles"))
        dirs.append(modelParent.deletingLastPathComponent().appendingPathComponent("profiles"))
        dirs.append(Bundle.main.bundleURL.deletingLastPathComponent().appendingPathComponent("profiles"))
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        dirs.append(cwd.appendingPathComponent("profiles"))
        dirs.append(cwd.deletingLastPathComponent().appendingPathComponent("profiles"))
        return dirs
    }
}
