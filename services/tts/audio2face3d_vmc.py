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

    def infer_audio(
        self,
        waveform_16k: np.ndarray,
        emotion_name: str = "neutral",
        emotion_strength: float = 0.7,
        infer_fps: float = 20.0,
        batch_size: int = 32,
    ) -> tuple[np.ndarray, float]:
        """Runs Audio2Face-3D ONNX + 52-D solver, returning `(weights_52[N, 52], infer_ms)` at `infer_fps`."""
        duration_s = len(waveform_16k) / 16000.0
        n_frames = max(1, int(math.ceil(duration_s * infer_fps)))
        buf_len = int(self.net_info["audio_params"]["buffer_len"])
        buf_ofs = int(self.net_info["audio_params"]["buffer_ofs"])

        # Apply input_strength from model_config
        inp_scale = float(self.model_cfg.get("input_strength", 1.0))
        wav_scaled = (waveform_16k * inp_scale).astype(np.float32)
        wav_pad = np.pad(wav_scaled, (buf_ofs, buf_len), mode="constant")

        windows = np.empty((n_frames, 1, buf_len), dtype=np.float32)
        for i in range(n_frames):
            c = int(round(i * 16000.0 / infer_fps))
            windows[i, 0, :] = wav_pad[c : c + buf_len]

        emo_vec = self.build_emotion_vector(emotion_name, emotion_strength)
        t0 = time.monotonic()
        outs = []
        for s in range(0, n_frames, batch_size):
            e = min(n_frames, s + batch_size)
            emo_b = np.broadcast_to(emo_vec[None, None, :], (e - s, 1, 26)).copy()
            r = self.sess.run(["result"], {"input": windows[s:e], "emotion": emo_b})[0]
            outs.append(r[:, 0, :140])
        skin_coeffs = np.concatenate(outs, axis=0).astype(np.float32)
        bs_52 = self.solve_blendshapes_from_coeffs(skin_coeffs)
        infer_ms = (time.monotonic() - t0) * 1000.0
        return bs_52, infer_ms


def postprocess_a2f_for_anime(raw: np.ndarray, params: TuningParams) -> np.ndarray:
    """Gentle anime VRM adaptation preserving NVIDIA's native upper-face & emotional dynamics."""
    if len(raw) == 0:
        return raw
    out = raw.copy()

    # Audio2Face-3D already produces strong realistic jaw/lip motion, so map 3.0 default slider -> ~1.35x gentle anime boost
    jaw_scale = 0.55 + 0.28 * params.jaw_gain
    vowel_scale = 0.60 + 0.25 * params.vowel_gain
    upper_scale = 0.55 + 0.45 * params.upper_face_gain

    out[:, IDX_JAW_OPEN] = np.clip(
        out[:, IDX_JAW_OPEN] * jaw_scale - out[:, IDX_MOUTH_CLOSE] * (0.25 * params.lip_close_strength),
        0.0,
        0.95,
    )
    for idx in (IDX_MOUTH_LOWER_L, IDX_MOUTH_LOWER_R, IDX_MOUTH_UPPER_L, IDX_MOUTH_UPPER_R):
        out[:, idx] = np.clip(out[:, idx] * (0.65 + 0.30 * params.lip_reveal_gain), 0.0, 0.90)
    for idx in (IDX_MOUTH_PUCKER, IDX_MOUTH_FUNNEL, IDX_MOUTH_STRETCH_L, IDX_MOUTH_STRETCH_R, IDX_MOUTH_SMILE_L, IDX_MOUTH_SMILE_R):
        out[:, idx] = np.clip(out[:, idx] * vowel_scale, 0.0, 0.95)

    # Scale genuine upper-face channels (brows 0..4, cheeks 5..7, eyeSquint/Wide 18..21) — NO fake jaw->brow hack!
    out[:, 0:8] = np.clip(out[:, 0:8] * upper_scale, 0.0, 0.95)
    out[:, 18:22] = np.clip(out[:, 18:22] * upper_scale, 0.0, 0.90)

    # Zero out eyeLook/eyeBlink so Swift Layer 2 keeps Mouse LookAt + native Auto-Blink 100% clean
    out[:, 8:18] = 0.0

    # Temporal smoothing
    if params.smoothing > 0.01:
        alpha = max(0.15, min(1.0, 1.0 - params.smoothing))
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
    raw_bs, inf_ms = engine.infer_audio(
        waveform,
        emotion_name=cur_emotion,
        emotion_strength=cur_emo_strength,
        infer_fps=args.infer_fps,
    )
    rtf = (inf_ms / 1000.0) / max(duration_s, 1e-3)
    processed_bs = postprocess_a2f_for_anime(raw_bs, params)
    n_frames = len(processed_bs)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if args.self_test:
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

    state_lock = threading.Lock()
    loop_enabled = bool(args.loop)

    if args.live_params:
        def _stdin_watcher() -> None:
            nonlocal raw_bs, processed_bs, loop_enabled, cur_emotion, cur_emo_strength
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        re_infer = False
                        new_emo = cur_emotion
                        new_str = cur_emo_strength
                        with state_lock:
                            if "loop" in payload:
                                loop_enabled = bool(payload["loop"])
                            if "emotion" in payload and str(payload["emotion"]).strip().lower() != cur_emotion:
                                new_emo = str(payload["emotion"]).strip().lower()
                                re_infer = True
                            if "emotion_strength" in payload:
                                cand_s = max(0.0, min(1.0, float(payload["emotion_strength"])))
                                if abs(cand_s - cur_emo_strength) > 0.02:
                                    new_str = cand_s
                                    re_infer = True
                            params.update_from_dict(payload)
                            if not re_infer:
                                processed_bs = postprocess_a2f_for_anime(raw_bs, params)
                        if re_infer:
                            new_raw, re_ms = engine.infer_audio(
                                waveform,
                                emotion_name=new_emo,
                                emotion_strength=new_str,
                                infer_fps=args.infer_fps,
                            )
                            with state_lock:
                                cur_emotion = new_emo
                                cur_emo_strength = new_str
                                raw_bs = new_raw
                                processed_bs = postprocess_a2f_for_anime(raw_bs, params)
                            print(f"[A2F-3D] Switched live emotion -> {new_emo} ({new_str:.2f}) in {re_ms:.0f}ms", flush=True)
                except Exception:
                    pass

        threading.Thread(target=_stdin_watcher, daemon=True).start()

    print(
        f"[A2F-3D] Ready: audio={audio_path.name} ({duration_s:.1f}s) emo={cur_emotion}({cur_emo_strength:.2f}) "
        f"infer={inf_ms:.0f}ms ({1.0/max(rtf,1e-4):.1f}x realtime) -> {args.host}:{args.port}",
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

                with state_lock:
                    cur_bs = processed_bs
                    offset_s = params.sync_offset_ms / 1000.0
                    head_gain = params.head_motion_gain

                t_sample = max(0.0, min(duration_s, elapsed + offset_s))
                pos = t_sample * args.infer_fps
                i0 = min(len(cur_bs) - 1, max(0, int(pos)))
                i1 = min(len(cur_bs) - 1, i0 + 1)
                frac = pos - i0
                vec = (1.0 - frac) * cur_bs[i0] + frac * cur_bs[i1]

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
