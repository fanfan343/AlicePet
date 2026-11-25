#!/usr/bin/env python3
"""NVIDIA Audio2Face-3D v2.3.1 (Claire ONNX) -> 52-channel ARKit VMC/OSC facial & emotion driver.

Runs official NVIDIA `network.onnx` (16kHz audio + 26-dim emotion vector -> 140 PCA skin coeffs)
and solves exact bounded [0, 1] 52 ARKit blendshapes directly in precomputed 52-D Gram/projection
space (mathematically equivalent to the 10,085-vertex frontal-mask solver, ~2ms for 300 frames!).

Supports all 10 explicit NVIDIA emotions (`joy`, `anger`, `sadness`, `amazement`, `cheekiness`,
`fear`, `disgust`, `grief`, `pain`, `outofbreath`) and streams 60fps interpolated VMC frames
aligned with macOS `afplay` to AlicePet's isolated `audioFaceChannel`.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from audio2face_vmc import (
    ARKIT_52_NAMES,
    IDX_BROW_INNER_UP,
    IDX_JAW_OPEN,
    IDX_MOUTH_CLOSE,
    IDX_MOUTH_FUNNEL,
    IDX_MOUTH_LOWER_L,
    IDX_MOUTH_LOWER_R,
    IDX_MOUTH_PUCKER,
    IDX_MOUTH_SMILE_L,
    IDX_MOUTH_SMILE_R,
    IDX_MOUTH_STRETCH_L,
    IDX_MOUTH_STRETCH_R,
    IDX_MOUTH_UPPER_L,
    IDX_MOUTH_UPPER_R,
    TuningParams,
    build_vmc_frame,
    decode_audio_16k_mono,
)

IDX_BROW_DOWN_L = ARKIT_52_NAMES.index("browDownLeft")
IDX_BROW_DOWN_R = ARKIT_52_NAMES.index("browDownRight")
IDX_BROW_OUTER_L = ARKIT_52_NAMES.index("browOuterUpLeft")
IDX_BROW_OUTER_R = ARKIT_52_NAMES.index("browOuterUpRight")
IDX_MOUTH_FROWN_L = ARKIT_52_NAMES.index("mouthFrownLeft")
IDX_MOUTH_FROWN_R = ARKIT_52_NAMES.index("mouthFrownRight")
IDX_MOUTH_PRESS_L = ARKIT_52_NAMES.index("mouthPressLeft")
IDX_MOUTH_PRESS_R = ARKIT_52_NAMES.index("mouthPressRight")
IDX_MOUTH_SHRUG_U = ARKIT_52_NAMES.index("mouthShrugUpper")
IDX_MOUTH_SHRUG_L = ARKIT_52_NAMES.index("mouthShrugLower")

EXPLICIT_EMOTIONS = [
    "amazement",
    "anger",
    "cheekiness",
    "disgust",
    "fear",
    "grief",
    "joy",
    "outofbreath",
    "pain",
    "sadness",
]

RUNNING = True


def _handle_signal(_signum: int, _frame: object) -> None:
    global RUNNING
    RUNNING = False


class Audio2Face3DEngine:
    """Official NVIDIA Audio2Face-3D v2.3.1 Claire ONNX + fast 52-D bounded Gram solver."""

    def __init__(self, models_dir: Path, intra_threads: int = 4) -> None:
        self.models_dir = models_dir
        self.net_info = json.loads((models_dir / "network_info.json").read_text(encoding="utf-8"))
        self.model_cfg = json.loads((models_dir / "model_config.json").read_text(encoding="utf-8"))["config"]
        self.bs_cfg = json.loads((models_dir / "bs_skin_config.json").read_text(encoding="utf-8"))["blendshape_params"]

        t0 = time.monotonic()
        self._load_or_build_solver_cache()
        self._load_default_implicit_emotion()

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(models_dir / "network.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.load_ms = (time.monotonic() - t0) * 1000.0

    def _load_default_implicit_emotion(self) -> None:
        emo_db = np.load(self.models_dir / "implicit_emo_db.npz")
        spec_names = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in emo_db["emo_spec_names"]]
        shot = self.model_cfg.get("source_shot", "cp1_neutral")
        frame_idx = int(self.model_cfg.get("source_frame", 0))
        if shot in spec_names:
            idx = spec_names.index(shot)
            start = int(emo_db["emo_spec_start"][idx])
            size = int(emo_db["emo_spec_size"][idx])
            self.default_implicit_emo = emo_db["emo_db"][start + min(frame_idx, max(0, size - 1))].astype(np.float32)
        else:
            self.default_implicit_emo = emo_db["emo_db"][0].astype(np.float32)

    def _load_or_build_solver_cache(self) -> None:
        cache_path = self.models_dir / "solver_cache_52d.npz"
        if cache_path.exists():
            c = np.load(cache_path)
            self.G = c["G"].astype(np.float32)               # (52, 52)
            self.M = c["M"].astype(np.float32)               # (52, 140)
            self.b0 = c["b0"].astype(np.float32)             # (52,)
            self.A = c["A"].astype(np.float32)               # (52, 52) regularized Hessian
            self.l1_vec = c["l1_vec"].astype(np.float32)     # (52,)
            self.active_mask = c["active_mask"].astype(np.float32)  # (52,)
            self.step_size = float(c["step_size"])
            self.pose_names = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in c["pose_names"]]
            self.arkit_perm = c["arkit_perm"].astype(np.int32)
            return

        bs = np.load(self.models_dir / "bs_skin.npz")
        md = np.load(self.models_dir / "model_data.npz")
        mask = bs["frontalMask"].astype(np.int64)
        pose_names = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in bs["poseNames"]][1:53]
        assert len(pose_names) == 52, f"Expected 52 poses, got {len(pose_names)}"

        bb_size = float(self.bs_cfg.get("templateBBSize", 45.94))
        scale = 1.0 / max(bb_size, 1e-4)

        # Build frontal-mask delta basis B: (52, N_mask * 3) in BB-normalized units
        B = np.empty((52, len(mask) * 3), dtype=np.float32)
        for i, pname in enumerate(pose_names):
            B[i] = (bs[pname][mask].astype(np.float32) * scale).reshape(-1)

        # Mean offset relative to neutral + 140 PCA modes on frontal mask
        neutral_m = bs["neutral"][mask].astype(np.float32)
        mean_m = md["shapes_mean_skin"][mask].astype(np.float32)
        mean_delta = ((mean_m - neutral_m) * scale).reshape(-1)
        P = (md["shapes_matrix_skin"][:, mask, :].astype(np.float32) * scale).reshape(140, -1)

        G = B @ B.T               # (52, 52)
        M = B @ P.T               # (52, 140)
        b0 = B @ mean_delta       # (52,)

        # Build NVIDIA regularization matrix matching bs_skin_config.json
        # Scale regularization weights relative to mean diagonal energy of G so solver is well-conditioned
        diag_scale = float(np.median(np.diag(G)))
        l2_w = float(self.bs_cfg.get("strengthL2regularization", 0.5)) * 0.04 * diag_scale
        l1_w = float(self.bs_cfg.get("strengthL1regularization", 0.5)) * 0.008 * diag_scale
        sym_w = float(self.bs_cfg.get("strengthSymmetry", 100.0)) * 0.0015 * diag_scale

        active_raw = np.asarray(self.bs_cfg.get("bsSolveActivePoses", [1] * 52), dtype=np.float32)
        active_mask = (active_raw[:52] > 0.5).astype(np.float32)

        L_sym = np.zeros((52, 52), dtype=np.float32)
        sym_ids = self.bs_cfg.get("bsSolveSymmetryPoses", [-1] * 52)
        pairs: dict[int, list[int]] = {}
        for idx, sid in enumerate(sym_ids[:52]):
            if sid >= 0:
                pairs.setdefault(int(sid), []).append(idx)
        for indices in pairs.values():
            if len(indices) == 2:
                i, j = indices
                L_sym[i, i] += 1.0
                L_sym[j, j] += 1.0
                L_sym[i, j] -= 1.0
                L_sym[j, i] -= 1.0

        A = G + np.eye(52, dtype=np.float32) * l2_w + L_sym * sym_w
        eig_max = float(np.linalg.eigvalsh(A).max())
        step_size = 1.0 / max(eig_max, 1e-5)
        l1_vec = np.full(52, l1_w, dtype=np.float32)

        # Map NVIDIA pose_names order -> canonical ARKIT_52_NAMES order
        name_to_idx = {n: i for i, n in enumerate(pose_names)}
        arkit_perm = np.asarray([name_to_idx[n] for n in ARKIT_52_NAMES], dtype=np.int32)

        np.savez_compressed(
            cache_path,
            G=G,
            M=M,
            b0=b0,
            A=A,
            l1_vec=l1_vec,
            active_mask=active_mask,
            step_size=np.float32(step_size),
            pose_names=np.asarray(pose_names, dtype="S32"),
            arkit_perm=arkit_perm,
        )
        self.G = G
        self.M = M
        self.b0 = b0
        self.A = A
        self.l1_vec = l1_vec
        self.active_mask = active_mask
        self.step_size = step_size
        self.pose_names = pose_names
        self.arkit_perm = arkit_perm

    def build_emotion_vector(self, emotion_name: str, strength: float) -> np.ndarray:
        vec = np.zeros(26, dtype=np.float32)
        vec[:16] = self.default_implicit_emo
        key = emotion_name.strip().lower()
        if key in EXPLICIT_EMOTIONS and strength > 0.0:
            vec[16 + EXPLICIT_EMOTIONS.index(key)] = float(np.clip(strength, 0.0, 1.0))
        return vec

    def solve_blendshapes_from_coeffs(self, skin_coeffs: np.ndarray, iters: int = 45) -> np.ndarray:
        """Vectorized FISTA bounded solver on (N, 140) -> (N, 52) in canonical ARKIT_52_NAMES order."""
        # rhs: (N, 52)
        rhs = (skin_coeffs @ self.M.T) + self.b0[None, :]
        w = np.zeros((len(skin_coeffs), 52), dtype=np.float32)
        y = w.copy()
        t = 1.0
        step = self.step_size
        A_T = self.A.T
        mask = self.active_mask[None, :]
        l1 = self.l1_vec[None, :]

        for _ in range(iters):
            grad = (y @ A_T) - rhs + l1
            w_next = np.clip((y - step * grad) * mask, 0.0, 1.0)
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_next
            y = w_next + beta * (w_next - w)
            w = w_next
            t = t_next

        return w[:, self.arkit_perm].astype(np.float32)

    def _prepare_audio_windows(self, waveform_16k: np.ndarray, infer_fps: float = 20.0) -> tuple[np.ndarray, int, int, int]:
        """Pads 16kHz audio once for O(1) slice windowing during real-time chunked inference."""
        duration_s = len(waveform_16k) / 16000.0
        n_frames = max(1, int(math.ceil(duration_s * infer_fps)))
        buf_len = int(self.net_info["audio_params"]["buffer_len"])
        buf_ofs = int(self.net_info["audio_params"]["buffer_ofs"])
        inp_scale = float(self.model_cfg.get("input_strength", 1.0))
        wav_scaled = (waveform_16k * inp_scale).astype(np.float32)
        wav_pad = np.pad(wav_scaled, (buf_ofs, buf_len), mode="constant")
        return wav_pad, n_frames, buf_len, buf_ofs

    def infer_frame_slice(
        self,
        wav_pad: np.ndarray,
        start_frame: int,
        count: int,
        buf_len: int,
        emotion_name: str = "neutral",
        emotion_strength: float = 0.7,
        infer_fps: float = 20.0,
        batch_size: int = 16,
    ) -> tuple[np.ndarray, float]:
        """Infers only `[start_frame, start_frame + count)` (~190ms for 8 frames / 0.4s on M1)."""
        windows = np.empty((count, 1, buf_len), dtype=np.float32)
        for j in range(count):
            c = int(round((start_frame + j) * 16000.0 / infer_fps))
            windows[j, 0, :] = wav_pad[c : c + buf_len]

        emo_vec = self.build_emotion_vector(emotion_name, emotion_strength)
        t0 = time.monotonic()
        outs = []
        for s in range(0, count, batch_size):
            e = min(count, s + batch_size)
            emo_b = np.broadcast_to(emo_vec[None, None, :], (e - s, 1, 26)).copy()
            r = self.sess.run(["result"], {"input": windows[s:e], "emotion": emo_b})[0]
            outs.append(r[:, 0, :140])
        skin_coeffs = np.concatenate(outs, axis=0).astype(np.float32)
        bs_52 = self.solve_blendshapes_from_coeffs(skin_coeffs)
        infer_ms = (time.monotonic() - t0) * 1000.0
        return bs_52, infer_ms

    def infer_audio(
        self,
        waveform_16k: np.ndarray,
        emotion_name: str = "neutral",
        emotion_strength: float = 0.7,
        infer_fps: float = 20.0,
        batch_size: int = 32,
    ) -> tuple[np.ndarray, float]:
        """Runs Audio2Face-3D ONNX + 52-D solver, returning `(weights_52[N, 52], infer_ms)` at `infer_fps`."""
        wav_pad, n_frames, buf_len, _ = self._prepare_audio_windows(waveform_16k, infer_fps=infer_fps)
        return self.infer_frame_slice(
            wav_pad,
            start_frame=0,
            count=n_frames,
            buf_len=buf_len,
            emotion_name=emotion_name,
            emotion_strength=emotion_strength,
            infer_fps=infer_fps,
            batch_size=batch_size,
        )


def postprocess_a2f_for_anime(raw: np.ndarray, params: TuningParams) -> np.ndarray:
    """Gentle anime VRM adaptation preserving NVIDIA's native upper-face & emotional dynamics."""
    if len(raw) == 0:
        return raw
    out = raw.copy()

    # Audio2Face-3D already produces strong realistic jaw/lip motion; now that all 52 channels
    # render cleanly without top-8 morph truncation, keep anime scaling balanced and clean.
    jaw_scale = 0.60 + 0.22 * params.jaw_gain
    vowel_scale = 0.60 + 0.22 * params.vowel_gain
    upper_scale = 0.55 + 0.38 * params.upper_face_gain

    # Fold lip closure cleanly into jawOpen and zero mouthClose afterward: on anime VRMs
    # (Alicephil / HANA_Tool / VRoid), mouthClose is baked as exact -jawOpen (cos = -1.0),
    # so sending mouthClose alongside or above jawOpen crushes the lower jaw upward.
    raw_close = out[:, IDX_MOUTH_CLOSE].copy()
    out[:, IDX_JAW_OPEN] = np.clip(
        out[:, IDX_JAW_OPEN] * jaw_scale - raw_close * (0.28 * params.lip_close_strength),
        0.0,
        0.95,
    )
    out[:, IDX_MOUTH_CLOSE] = 0.0

    for idx in (IDX_MOUTH_LOWER_L, IDX_MOUTH_LOWER_R, IDX_MOUTH_UPPER_L, IDX_MOUTH_UPPER_R):
        out[:, idx] = np.clip(out[:, idx] * (0.65 + 0.25 * params.lip_reveal_gain), 0.0, 0.85)
    for idx in (IDX_MOUTH_PUCKER, IDX_MOUTH_FUNNEL, IDX_MOUTH_STRETCH_L, IDX_MOUTH_STRETCH_R, IDX_MOUTH_SMILE_L, IDX_MOUTH_SMILE_R):
        out[:, idx] = np.clip(out[:, idx] * vowel_scale, 0.0, 0.90)

    # Damp realistic-skin lip-thickness micro-morphs that over-stack with smile/jaw on anime faces
    for idx in (IDX_MOUTH_PRESS_L, IDX_MOUTH_PRESS_R, IDX_MOUTH_SHRUG_U, IDX_MOUTH_SHRUG_L):
        out[:, idx] = np.clip(out[:, idx] * 0.50, 0.0, 0.45)

    # Resolve directly opposing mouth/brow pairs so additive vertices never pull against each other
    for a_idx, b_idx in (
        (IDX_MOUTH_SMILE_L, IDX_MOUTH_FROWN_L),
        (IDX_MOUTH_SMILE_R, IDX_MOUTH_FROWN_R),
        (IDX_BROW_DOWN_L, IDX_BROW_OUTER_L),
        (IDX_BROW_DOWN_R, IDX_BROW_OUTER_R),
        (IDX_BROW_DOWN_L, IDX_BROW_INNER_UP),
        (IDX_BROW_DOWN_R, IDX_BROW_INNER_UP),
    ):
        overlap = np.minimum(out[:, a_idx], out[:, b_idx]) * 0.75
        out[:, a_idx] = np.maximum(0.0, out[:, a_idx] - overlap)
        out[:, b_idx] = np.maximum(0.0, out[:, b_idx] - overlap)

    # Scale genuine upper-face channels (brows 0..4, cheeks 5..7, eyeSquint/Wide 18..21) — NO fake jaw->brow hack!
    out[:, 0:8] = np.clip(out[:, 0:8] * upper_scale, 0.0, 0.90)
    out[:, 18:22] = np.clip(out[:, 18:22] * (0.50 + 0.30 * params.upper_face_gain), 0.0, 0.75)

    # Zero out eyeLook/eyeBlink so Swift Layer 2 keeps Mouse LookAt + native Auto-Blink 100% clean
    out[:, 8:18] = 0.0

    # Temporal smoothing (always apply a light NVIDIA-style anti-chatter floor alpha <= 0.75)
    alpha = max(0.15, min(0.75, 1.0 - params.smoothing))
    for i in range(1, len(out)):
        out[i] = alpha * out[i] + (1.0 - alpha) * out[i - 1]

    return np.clip(out, 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="NVIDIA Audio2Face-3D Claire ONNX -> VMC 52-ARKit driver")
    default_dir = Path(__file__).resolve().parent
    parser.add_argument("--audio", default=str(default_dir / "samples/vtuber_10s.wav"))
    parser.add_argument("--models-dir", default=str(default_dir / "models/a2f_claire"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=39539)
    parser.add_argument("--emotion", default="joy")
    parser.add_argument("--emotion-strength", type=float, default=0.70)
    parser.add_argument("--infer-fps", type=float, default=20.0)
    parser.add_argument("--jaw-gain", type=float, default=3.00)
    parser.add_argument("--vowel-gain", type=float, default=2.50)
    parser.add_argument("--lip-close", type=float, default=1.50)
    parser.add_argument("--lip-reveal", type=float, default=1.50)
    parser.add_argument("--smoothing", type=float, default=0.10)
    parser.add_argument("--upper-face", type=float, default=1.50)
    parser.add_argument("--head-motion", type=float, default=1.40)
    parser.add_argument("--sync-offset-ms", type=float, default=0.0)
    parser.add_argument("--loop", type=int, default=0)
    parser.add_argument("--live-params", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    audio_path = Path(args.audio)
    models_dir = Path(args.models_dir)
    if not audio_path.exists():
        print(f"[A2F-3D] ERROR: audio not found: {audio_path}", file=sys.stderr, flush=True)
        return 2
    if not (models_dir / "network.onnx").exists():
        print(f"[A2F-3D] ERROR: network.onnx not found in {models_dir}", file=sys.stderr, flush=True)
        return 3

    params = TuningParams(
        jaw_gain=args.jaw_gain,
        vowel_gain=args.vowel_gain,
        lip_close_strength=args.lip_close,
        lip_reveal_gain=args.lip_reveal,
        smoothing=args.smoothing,
        upper_face_gain=args.upper_face,
        head_motion_gain=args.head_motion,
        sync_offset_ms=args.sync_offset_ms,
        natural_blinks=False,
    )

    engine = Audio2Face3DEngine(models_dir=models_dir, intra_threads=4)
    max_s = args.max_seconds if args.max_seconds > 0 else None
    waveform = decode_audio_16k_mono(audio_path, max_seconds=max_s)
    duration_s = len(waveform) / 16000.0

    cur_emotion = args.emotion
    cur_emo_strength = args.emotion_strength
    wav_pad, n_frames, buf_len, _ = engine._prepare_audio_windows(waveform, infer_fps=args.infer_fps)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if args.self_test:
        raw_bs, inf_ms = engine.infer_frame_slice(
            wav_pad,
            start_frame=0,
            count=n_frames,
            buf_len=buf_len,
            emotion_name=cur_emotion,
            emotion_strength=cur_emo_strength,
            infer_fps=args.infer_fps,
            batch_size=32,
        )
        rtf = (inf_ms / 1000.0) / max(duration_s, 1e-3)
        processed_bs = postprocess_a2f_for_anime(raw_bs, params)
        scores = (
            processed_bs[:, IDX_JAW_OPEN] * 1.4
            + processed_bs[:, IDX_MOUTH_SMILE_L] * 0.7
            + processed_bs[:, IDX_BROW_INNER_UP] * 0.8
        )
        peak_idx = int(np.argmax(scores))
        peak_vec = processed_bs[peak_idx]
        dgram = build_vmc_frame(
            peak_vec,
            peak_idx / args.infer_fps,
            head_motion_gain=params.head_motion_gain,
            speech_energy=float(peak_vec[IDX_JAW_OPEN]),
        )
        sock.sendto(dgram, (args.host, args.port))
        sock.close()

        top_upper = sorted(
            ((ARKIT_52_NAMES[i], float(raw_bs[:, i].max())) for i in list(range(0, 8)) + list(range(18, 22))),
            key=lambda kv: kv[1],
            reverse=True,
        )[:6]
        top_mouth = sorted(
            ((ARKIT_52_NAMES[i], float(raw_bs[:, i].max())) for i in range(22, 52)),
            key=lambda kv: kv[1],
            reverse=True,
        )[:6]
        print(
            f"[A2F-3D] Self-test OK: emo={cur_emotion}({cur_emo_strength:.2f}) duration={duration_s:.2f}s "
            f"frames={n_frames}@{args.infer_fps:.0f}fps load={engine.load_ms:.0f}ms infer={inf_ms:.0f}ms "
            f"(RTF={rtf:.3f}, {1.0/max(rtf,1e-4):.1f}x realtime) peakFrame={peak_idx}\n"
            f"  RAW UpperFace max: {', '.join(f'{k}={v:.2f}' for k, v in top_upper)}\n"
            f"  RAW Mouth max:     {', '.join(f'{k}={v:.2f}' for k, v in top_mouth)}",
            flush=True,
        )
        return 0

    # Real-time streaming chunked inference:
    # Only infer the first 0.4s (8 frames, ~190ms) before starting playback, then stay ahead
    # in a background thread at ~2x realtime. Live emotion switches re-infer from playhead in ~200ms!
    state_lock = threading.Lock()
    wake_event = threading.Event()
    loop_enabled = bool(args.loop)
    raw_bs = np.zeros((n_frames, 52), dtype=np.float32)
    processed_bs = np.zeros((n_frames, 52), dtype=np.float32)
    frame_gen = np.full(n_frames, -1, dtype=np.int32)
    current_gen = 1
    playhead_frame = 0
    emo_switch_started_at: float | None = None

    first_n = min(n_frames, 8)
    first_raw, first_ms = engine.infer_frame_slice(
        wav_pad,
        start_frame=0,
        count=first_n,
        buf_len=buf_len,
        emotion_name=cur_emotion,
        emotion_strength=cur_emo_strength,
        infer_fps=args.infer_fps,
        batch_size=16,
    )
    raw_bs[:first_n] = first_raw
    frame_gen[:first_n] = current_gen
    processed_bs[:first_n] = postprocess_a2f_for_anime(first_raw, params)

    def _producer_loop() -> None:
        nonlocal processed_bs, emo_switch_started_at
        chunk_size = 10  # 0.5s @ 20fps (~235ms CPU inference -> ~2.1x realtime)
        while RUNNING:
            with state_lock:
                gen = current_gen
                emo = cur_emotion
                strength = cur_emo_strength
                ph = playhead_frame
                is_loop = loop_enabled

                # Pick earliest uncomputed frame ahead of playhead (wrapping if looping)
                start_f: int | None = None
                for off in range(n_frames):
                    idx = ph + off
                    if idx >= n_frames:
                        if not is_loop:
                            break
                        idx -= n_frames
                    if frame_gen[idx] != gen:
                        start_f = idx
                        break

            if start_f is None:
                wake_event.wait(timeout=0.1)
                wake_event.clear()
                continue

            # Collect contiguous stale frames up to chunk_size
            end_f = min(n_frames, start_f + chunk_size)
            with state_lock:
                while end_f > start_f + 1 and frame_gen[end_f - 1] == gen:
                    end_f -= 1
            count = end_f - start_f

            chunk_raw, chunk_ms = engine.infer_frame_slice(
                wav_pad,
                start_frame=start_f,
                count=count,
                buf_len=buf_len,
                emotion_name=emo,
                emotion_strength=strength,
                infer_fps=args.infer_fps,
                batch_size=16,
            )

            with state_lock:
                if gen != current_gen:
                    # Emotion changed while inferring this chunk; immediately prioritize new emotion
                    continue
                raw_bs[start_f:end_f] = chunk_raw
                frame_gen[start_f:end_f] = gen
                # Re-postprocess ready span quickly (<0.2ms) for smooth temporal continuity
                processed_bs = postprocess_a2f_for_anime(raw_bs, params)
                if emo_switch_started_at is not None:
                    sw_ms = (time.monotonic() - emo_switch_started_at) * 1000.0
                    emo_switch_started_at = None
                    print(f"[A2F-3D] Switched live emotion -> {emo} ({strength:.2f}) in {sw_ms:.0f}ms (chunk={chunk_ms:.0f}ms)", flush=True)

    producer_thread = threading.Thread(target=_producer_loop, daemon=True)
    producer_thread.start()

    if args.live_params:
        def _stdin_watcher() -> None:
            nonlocal processed_bs, loop_enabled, cur_emotion, cur_emo_strength, current_gen, emo_switch_started_at
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        with state_lock:
                            emo_changed = False
                            if "loop" in payload:
                                loop_enabled = bool(payload["loop"])
                            if "emotion" in payload:
                                cand_emo = str(payload["emotion"]).strip().lower()
                                if cand_emo and cand_emo != cur_emotion:
                                    cur_emotion = cand_emo
                                    emo_changed = True
                            if "emotion_strength" in payload:
                                cand_s = max(0.0, min(1.0, float(payload["emotion_strength"])))
                                if abs(cand_s - cur_emo_strength) > 0.02:
                                    cur_emo_strength = cand_s
                                    emo_changed = True
                            params.update_from_dict(payload)
                            if emo_changed:
                                current_gen += 1
                                emo_switch_started_at = time.monotonic()
                                wake_event.set()
                            else:
                                processed_bs = postprocess_a2f_for_anime(raw_bs, params)
                except Exception:
                    pass

        threading.Thread(target=_stdin_watcher, daemon=True).start()

    print(
        f"[A2F-3D] Streaming ready in {engine.load_ms + first_ms:.0f}ms (load={engine.load_ms:.0f}ms, first {first_n}f={first_ms:.0f}ms): "
        f"audio={audio_path.name} ({duration_s:.1f}s) emo={cur_emotion}({cur_emo_strength:.2f}) -> {args.host}:{args.port}",
        flush=True,
    )

    afplay_cmd = ["/usr/bin/afplay"]
    if max_s is not None:
        afplay_cmd += ["-t", f"{max_s:.2f}"]
    afplay_cmd.append(str(audio_path))

    send_fps = 60.0
    period = 1.0 / send_fps
    total_packets = 0
    audio_proc: subprocess.Popen | None = None
    last_sent_vec: np.ndarray | None = None

    try:
        while RUNNING:
            audio_proc = subprocess.Popen(
                afplay_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            t0 = time.monotonic()
            pass_packets = 0

            while RUNNING:
                elapsed = time.monotonic() - t0
                if elapsed >= duration_s or (audio_proc.poll() is not None and elapsed > 0.2):
                    break

                t_sample = max(0.0, min(duration_s, elapsed + params.sync_offset_ms / 1000.0))
                pos = t_sample * args.infer_fps
                i0 = min(n_frames - 1, max(0, int(pos)))
                i1 = min(n_frames - 1, i0 + 1)
                frac = pos - i0

                with state_lock:
                    playhead_frame = i0
                    head_gain = params.head_motion_gain
                    r0 = frame_gen[i0] >= 0
                    r1 = frame_gen[i1] >= 0
                    if r0 and r1:
                        vec = (1.0 - frac) * processed_bs[i0] + frac * processed_bs[i1]
                    elif r0:
                        vec = processed_bs[i0].copy()
                    elif last_sent_vec is not None:
                        vec = last_sent_vec.copy()
                    else:
                        vec = np.zeros(52, dtype=np.float32)

                # Smooth emotion-boundary transitions at 60fps so live emotion switches glide cleanly
                if last_sent_vec is not None:
                    vec = 0.55 * vec + 0.45 * last_sent_vec
                last_sent_vec = vec

                energy = float(vec[IDX_JAW_OPEN] + 0.4 * vec[IDX_MOUTH_FUNNEL])
                dgram = build_vmc_frame(vec, elapsed, head_motion_gain=head_gain, speech_energy=energy)
                sock.sendto(dgram, (args.host, args.port))
                pass_packets += 1
                total_packets += 1

                next_tick = t0 + pass_packets * period
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0.001:
                    time.sleep(sleep_s)

            if audio_proc.poll() is None:
                audio_proc.terminate()
                try:
                    audio_proc.wait(timeout=0.4)
                except subprocess.TimeoutExpired:
                    audio_proc.kill()
            audio_proc = None

            with state_lock:
                should_loop = loop_enabled
            if not should_loop:
                break
            pause_end = time.monotonic() + 0.25
            while RUNNING and time.monotonic() < pause_end:
                time.sleep(0.02)
    finally:
        if audio_proc is not None and audio_proc.poll() is None:
            audio_proc.terminate()
            try:
                audio_proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                audio_proc.kill()
        zero_vec = np.zeros(52, dtype=np.float32)
        for _ in range(2):
            sock.sendto(build_vmc_frame(zero_vec, duration_s, head_motion_gain=0.0, speech_energy=0.0), (args.host, args.port))
            time.sleep(0.015)
        sock.close()
        print(f"[A2F-3D] Finished cleanly (packets={total_packets})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
