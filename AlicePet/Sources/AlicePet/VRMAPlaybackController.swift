import Foundation
import VRMMetalKit

/// Thin native controller wrapping `VRMMetalKit`'s built-in `VRMAnimationLoader` and `AnimationPlayer`
/// for standard `.vrma` (`VRMC_vrm_animation`) playback inside `AnimationMixer` Layer 0.
@MainActor
final class VRMAPlaybackController {
    private let player = AnimationPlayer()
    private var currentClip: AnimationClip?

    private(set) var activeURL: URL?
    private(set) var activeClipName: String?
    private(set) var hasAppliedMorphs: Bool = false

    var onStateChanged: (() -> Void)?

    var isLooping: Bool {
        get { player.isLooping }
        set { player.isLooping = newValue }
    }

    var speed: Float {
        get { player.speed }
        set { player.speed = newValue }
    }

    var duration: Float {
        currentClip?.duration ?? 0
    }

    var currentTime: Float {
        player.time
    }

    var isActive: Bool {
        guard currentClip != nil else { return false }
        return !player.isFinished
    }

    var hasExpressionTracks: Bool {
        guard let clip = currentClip else { return false }
        return !clip.morphTracks.isEmpty || !clip.expressionTracks.isEmpty
    }

    /// Loads a standard `.vrma` file with `VRMMetalKit.VRMAnimationLoader` (VRM 0.0 / 1.0 rest-pose
    /// retargeting handled natively) and starts playback immediately.
    func loadAndPlay(
        url: URL,
        model: VRMModel,
        loop: Bool = false,
        speed: Float = 1.0,
        applyRootMotion: Bool = false
    ) throws {
        let clip = try VRMAnimationLoader.loadVRMA(from: url, model: model)
        self.currentClip = clip
        self.activeURL = url
        self.activeClipName = url.deletingPathExtension().lastPathComponent
        self.player.isLooping = loop
        self.player.speed = speed
        self.player.applyRootMotion = applyRootMotion
        self.player.load(clip)
        AppLogger.shared.info(
            "VRMA",
            String(
                format: "Loaded '%@' (duration=%.2fs joints=%d morphs=%d loop=%@)",
                url.lastPathComponent,
                clip.duration,
                clip.jointTracks.count,
                clip.morphTracks.count + clip.expressionTracks.count,
                loop ? "yes" : "no"
            )
        )
        onStateChanged?()
    }

    func stop(expressionMixer: ExpressionMixer? = nil, renderer: VRMRenderer? = nil) {
        let wasActive = currentClip != nil
        player.stop()
        currentClip = nil
        activeURL = nil
        activeClipName = nil
        if hasAppliedMorphs, let renderer, let expressionMixer {
            expressionMixer.resetAfterTrackingStop(on: renderer)
        }
        hasAppliedMorphs = false
        if wasActive {
            AppLogger.shared.info("VRMA", "Stopped .vrma playback")
            onStateChanged?()
        }
    }

    /// Advances playback by `dt` seconds and writes retargeted humanoid bones onto `model`.
    /// Returns `true` while actively playing; auto-clears once a non-looping clip finishes.
    @discardableResult
    func advanceAndApply(dt: Float, model: VRMModel) -> Bool {
        guard currentClip != nil else { return false }
        player.update(deltaTime: max(0, dt), model: model)
        if !player.isLooping && player.isFinished {
            AppLogger.shared.info("VRMA", "Finished one-shot clip '\(activeClipName ?? "")'")
            currentClip = nil
            activeURL = nil
            activeClipName = nil
            onStateChanged?()
            return false
        }
        return true
    }

    /// Applies any VRMA facial expression/morph tracks cleanly onto the renderer's expression controller.
    func applyExpressionsIfPresent(to renderer: VRMRenderer) {
        guard currentClip != nil, hasExpressionTracks, let exprCtrl = renderer.expressionController else {
            return
        }
        player.applyMorphWeights(to: exprCtrl)
        hasAppliedMorphs = true
    }

    func clearAppliedExpressionsIfNeeded(on renderer: VRMRenderer, expressionMixer: ExpressionMixer) {
        guard hasAppliedMorphs else { return }
        hasAppliedMorphs = false
        expressionMixer.resetAfterTrackingStop(on: renderer)
    }

    /// Deterministic seek + evaluate helper used by headless `--vrma` visual regression snapshots.
    func seekAndApply(time: Float, model: VRMModel, renderer: VRMRenderer) {
        guard currentClip != nil else { return }
        player.seek(to: max(0, time))
        player.update(deltaTime: 0, model: model)
        applyExpressionsIfPresent(to: renderer)
    }
}
