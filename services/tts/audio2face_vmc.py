#!/usr/bin/env python3
"""LAM_Audio2Expression (lam-int8.onnx) -> real-time 52-channel ARKit VMC/OSC audio player.

Pre-computes 30fps 52-channel ARKit facial blendshapes from any speech audio file using
ONNX Runtime (CPU, ~15-20x faster than real-time on Apple Silicon), then plays the audio via
macOS `afplay` while streaming tightly synchronized 60fps interpolated VMC OSC frames to AlicePet.

Supports `--loop` and `--live-params` (reads JSON parameter updates line-by-line from stdin
and re-applies post-processing in <0.3ms without re-running ONNX inference).
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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder

# Exact canonical 52 ARKit blendshape order from myned-ai/wav2arkit_cpu config.json
ARKIT_52_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]

IDX_JAW_OPEN = 24
IDX_BLINK_L = 8
IDX_BLINK_R = 9
IDX_BROW_INNER_UP = 2
IDX_CHEEK_SQUINT_L = 6
IDX_CHEEK_SQUINT_R = 7
IDX_MOUTH_CLOSE = 26
IDX_MOUTH_FUNNEL = 31
IDX_MOUTH_LOWER_L = 33
IDX_MOUTH_LOWER_R = 34
IDX_MOUTH_PUCKER = 37
IDX_MOUTH_SMILE_L = 43
IDX_MOUTH_SMILE_R = 44
IDX_MOUTH_STRETCH_L = 45
IDX_MOUTH_STRETCH_R = 46
IDX_MOUTH_UPPER_L = 47
IDX_MOUTH_UPPER_R = 48

RUNNING = True


def _handle_signal(_signum: int, _frame: object) -> None:
    global RUNNING
    RUNNING = False


@dataclass
class TuningParams:
    jaw_gain: float = 2.85          # 0.5 .. 4.5 (main mouth open scale)
    vowel_gain: float = 1.60        # 0.5 .. 2.5 (pucker / funnel / stretch / smile)
    lip_close_strength: float = 0.55  # 0.0 .. 1.5 (mouthClose suppression on jawOpen)
    lip_reveal_gain: float = 0.60   # 0.0 .. 1.5 (mouthLowerDown / mouthUpperUp)
    smoothing: float = 0.32         # 0.0 .. 0.85 (0=snappy, 0.85=silky smooth)
    upper_face_gain: float = 1.00   # 0.0 .. 1.5 (brows + cheeks micro-expression)
    head_motion_gain: float = 1.00  # 0.0 .. 2.0 (0=head still)
    sync_offset_ms: float = 0.0     # -150 .. +150 ms (audio-lip alignment)
    natural_blinks: bool = True

    def update_from_dict(self, data: dict) -> None:
        def _clamp(val: object, lo: float, hi: float, cur: float) -> float:
            try:
                return max(lo, min(hi, float(val)))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return cur

        if "jaw_gain" in data:
            self.jaw_gain = _clamp(data["jaw_gain"], 0.4, 5.0, self.jaw_gain)
        if "vowel_gain" in data:
            self.vowel_gain = _clamp(data["vowel_gain"], 0.3, 3.0, self.vowel_gain)
        if "lip_close_strength" in data:
            self.lip_close_strength = _clamp(data["lip_close_strength"], 0.0, 2.0, self.lip_close_strength)
        if "lip_reveal_gain" in data:
            self.lip_reveal_gain = _clamp(data["lip_reveal_gain"], 0.0, 2.0, self.lip_reveal_gain)
        if "smoothing" in data:
            self.smoothing = _clamp(data["smoothing"], 0.0, 0.88, self.smoothing)
        if "upper_face_gain" in data:
            self.upper_face_gain = _clamp(data["upper_face_gain"], 0.0, 2.0, self.upper_face_gain)
        if "head_motion_gain" in data:
            self.head_motion_gain = _clamp(data["head_motion_gain"], 0.0, 2.5, self.head_motion_gain)
        if "sync_offset_ms" in data:
            self.sync_offset_ms = _clamp(data["sync_offset_ms"], -200.0, 200.0, self.sync_offset_ms)
        if "natural_blinks" in data:
            self.natural_blinks = bool(data["natural_blinks"])


def decode_audio_16k_mono(audio_path: Path, max_seconds: float | None = None) -> np.ndarray:
    """Decodes any audio file (wav/m4a/mp3/aac) into 16kHz mono float32 waveform in [-1, 1]."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(audio_path)]
    if max_seconds is not None and max_seconds > 0:
        cmd += ["-t", f"{max_seconds:.2f}"]
    cmd += ["-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]

    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0 or not res.stdout:
        raise RuntimeError(f"ffmpeg decode failed for {audio_path}: {res.stderr.decode('utf-8', 'ignore')}")
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def infer_blendshapes(
    session: ort.InferenceSession,
    waveform_16k: np.ndarray,
    chunk_seconds: float = 6.0,
    overlap_seconds: float = 0.5,
) -> np.ndarray:
    """Runs LAM_Audio2Expression ONNX inference and returns shape (N_frames, 52) float32 at 30fps."""
    sr = 16000
    n_samples = len(waveform_16k)
    if n_samples == 0:
        return np.zeros((1, 52), dtype=np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    if n_samples <= int(12.5 * sr):
        out = session.run([output_name], {input_name: waveform_16k[None, :]})[0]
        return np.clip(out[0], 0.0, 1.0).astype(np.float32)

    chunk_n = int(chunk_seconds * sr)
    overlap_n = int(overlap_seconds * sr)
    hop_n = chunk_n - overlap_n
    total_frames = int(math.ceil(30.0 * n_samples / sr))
    acc = np.zeros((total_frames, 52), dtype=np.float32)
    weights = np.zeros((total_frames, 1), dtype=np.float32)

    start = 0
    while start < n_samples:
        end = min(n_samples, start + chunk_n)
        seg = waveform_16k[start:end]
        if len(seg) < sr // 4:
            seg = np.pad(seg, (0, sr // 4 - len(seg)))
        out = session.run([output_name], {input_name: seg[None, :]})[0][0]
        f_start = int(round(30.0 * start / sr))
        f_len = min(len(out), total_frames - f_start)
        if f_len <= 0:
            break
        w = np.ones((f_len, 1), dtype=np.float32)
        fade = min(int(overlap_seconds * 30), f_len // 3)
        if start > 0 and fade > 1:
            w[:fade, 0] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if end < n_samples and fade > 1:
            w[-fade:, 0] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        acc[f_start : f_start + f_len] += out[:f_len] * w
        weights[f_start : f_start + f_len] += w
        if end >= n_samples:
            break
        start += hop_n

    weights = np.maximum(weights, 1e-6)
    return np.clip(acc / weights, 0.0, 1.0).astype(np.float32)


def postprocess_blendshapes(raw: np.ndarray, params: TuningParams) -> np.ndarray:
    """Applies live-tunable anime VRM retargeting, smoothing, and natural blinks (<0.3ms for 315 frames)."""
    if len(raw) == 0:
        return raw
    out = raw.copy()

    # 1. Jaw open & lip-close balance
    lower_avg = 0.5 * (out[:, IDX_MOUTH_LOWER_L] + out[:, IDX_MOUTH_LOWER_R])
    close_w = out[:, IDX_MOUTH_CLOSE]
    lower_couple = 0.11 * params.jaw_gain  # gentle natural lift from lower lip drop
    out[:, IDX_JAW_OPEN] = np.clip(
        out[:, IDX_JAW_OPEN] * params.jaw_gain
        + lower_avg * lower_couple
        - close_w * params.lip_close_strength,
        0.0,
        0.95,
    )

    # 2. Lip reveal (teeth / upper & lower lip pull)
    out[:, IDX_MOUTH_LOWER_L] = np.clip(out[:, IDX_MOUTH_LOWER_L] * params.lip_reveal_gain, 0.0, 0.90)
    out[:, IDX_MOUTH_LOWER_R] = np.clip(out[:, IDX_MOUTH_LOWER_R] * params.lip_reveal_gain, 0.0, 0.90)
    out[:, IDX_MOUTH_UPPER_L] = np.clip(out[:, IDX_MOUTH_UPPER_L] * (params.lip_reveal_gain * 0.90), 0.0, 0.85)
    out[:, IDX_MOUTH_UPPER_R] = np.clip(out[:, IDX_MOUTH_UPPER_R] * (params.lip_reveal_gain * 0.90), 0.0, 0.85)

    # 3. Vowel shape expressiveness (O/U pucker/funnel, I/E stretch/smile)
    vg = params.vowel_gain
    for idx, rel in (
        (IDX_MOUTH_PUCKER, 1.05),
        (IDX_MOUTH_FUNNEL, 1.05),
        (IDX_MOUTH_STRETCH_L, 0.95),
        (IDX_MOUTH_STRETCH_R, 0.95),
        (IDX_MOUTH_SMILE_L, 0.90),
        (IDX_MOUTH_SMILE_R, 0.90),
    ):
        out[:, idx] = np.clip(out[:, idx] * (vg * rel), 0.0, 0.90)

    # 4. Upper-face micro-expressions (eyebrows + cheek squint)
    ug = params.upper_face_gain
    out[:, 0:8] = np.clip(out[:, 0:8] * ug, 0.0, 0.90)
    if ug > 0.01:
        speech_emp = np.clip(out[:, IDX_JAW_OPEN] * 0.42 * ug, 0.0, 0.45)
        out[:, IDX_BROW_INNER_UP] = np.maximum(out[:, IDX_BROW_INNER_UP], speech_emp)
        out[:, IDX_CHEEK_SQUINT_L] = np.maximum(out[:, IDX_CHEEK_SQUINT_L], out[:, IDX_MOUTH_SMILE_L] * 0.35 * ug)
        out[:, IDX_CHEEK_SQUINT_R] = np.maximum(out[:, IDX_CHEEK_SQUINT_R], out[:, IDX_MOUTH_SMILE_R] * 0.35 * ug)

    # 5. Temporal smoothing (smoothing=0 -> alpha=1.0 snappy; smoothing=0.8 -> alpha=0.20 silky)
    mouth_alpha = max(0.12, min(1.0, 1.0 - params.smoothing))
    upper_alpha = max(0.10, min(1.0, mouth_alpha * 0.72))
    alpha = np.full(52, upper_alpha, dtype=np.float32)
    alpha[22:49] = mouth_alpha
    alpha[IDX_BLINK_L] = 0.78
    alpha[IDX_BLINK_R] = 0.78

    if params.smoothing > 0.01:
        for i in range(1, len(out)):
            out[i] = alpha * out[i] + (1.0 - alpha) * out[i - 1]

    # 6. Natural periodic blinks (~every 3.4s) if LAM doesn't blink strongly on its own
    if params.natural_blinks and len(out) > 45:
        blink_period = 102
        for center in range(42, len(out) - 6, blink_period):
            if float(np.max(out[center - 4 : center + 5, IDX_BLINK_L])) < 0.35:
                profile = (0.35, 0.85, 1.0, 0.75, 0.25)
                for offset, val in enumerate(profile, start=-2):
                    idx = center + offset
                    if 0 <= idx < len(out):
                        out[idx, IDX_BLINK_L] = max(out[idx, IDX_BLINK_L], val)
                        out[idx, IDX_BLINK_R] = max(out[idx, IDX_BLINK_R], val)

    return np.clip(out, 0.0, 1.0)


def _quat_from_yxz_unity(yaw: float, pitch: float, roll: float) -> tuple[float, float, float, float]:
    """Builds right-handed VRM YXZ quaternion and converts to Unity VMC space (flip Y/Z)."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * sp * cr + sy * cp * sr
    y = sy * cp * cr - cy * sp * sr
    z = cy * cp * sr - sy * sp * cr
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return (x, -y, -z, w)


def build_vmc_frame(
    weights_52: np.ndarray,
    timestamp_s: float,
    head_motion_gain: float = 1.0,
    speech_energy: float = 0.0,
) -> bytes:
    bundle = OscBundleBuilder(IMMEDIATELY)

    ok_msg = OscMessageBuilder(address="/VMC/Ext/OK")
    ok_msg.add_arg(1)
    bundle.add_content(ok_msg.build())

    t_msg = OscMessageBuilder(address="/VMC/Ext/T")
    t_msg.add_arg(float(timestamp_s))
    bundle.add_content(t_msg.build())

    if head_motion_gain > 0.01:
        amp = (0.35 + 0.65 * min(1.0, speech_energy * 2.2)) * head_motion_gain
        yaw = math.sin(timestamp_s * 1.15) * math.radians(4.5) * amp
        pitch = (math.sin(timestamp_s * 2.7) * math.radians(2.8) + speech_energy * math.radians(3.2)) * amp
        roll = math.sin(timestamp_s * 0.85 + 0.6) * math.radians(3.4) * amp

        for bone_name, frac in (("Neck", 0.35), ("Head", 0.65)):
            qx, qy, qz, qw = _quat_from_yxz_unity(yaw * frac, pitch * frac, roll * frac)
            bmsg = OscMessageBuilder(address="/VMC/Ext/Bone/Pos")
            bmsg.add_arg(bone_name)
            bmsg.add_arg(0.0)
            bmsg.add_arg(0.0)
            bmsg.add_arg(0.0)
            bmsg.add_arg(float(qx))
            bmsg.add_arg(float(qy))
            bmsg.add_arg(float(qz))
            bmsg.add_arg(float(qw))
            bundle.add_content(bmsg.build())

    for idx, name in enumerate(ARKIT_52_NAMES):
        msg = OscMessageBuilder(address="/VMC/Ext/Blend/Val")
        msg.add_arg(name)
        msg.add_arg(float(weights_52[idx]))
        bundle.add_content(msg.build())

    apply_msg = OscMessageBuilder(address="/VMC/Ext/Blend/Apply")
    bundle.add_content(apply_msg.build())
    return bundle.build().dgram


def main() -> int:
    parser = argparse.ArgumentParser(description="LAM_Audio2Expression ONNX -> VMC 52-ARKit player")
    default_dir = Path(__file__).resolve().parent
    parser.add_argument("--audio", default=str(default_dir / "samples/vtuber_10s.wav"))
    parser.add_argument("--model", default=str(default_dir / "models/lam-int8.onnx"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=39539)
    parser.add_argument("--jaw-gain", type=float, default=2.85)
    parser.add_argument("--vowel-gain", type=float, default=1.60)
    parser.add_argument("--lip-close", type=float, default=0.55)
    parser.add_argument("--lip-reveal", type=float, default=0.60)
    parser.add_argument("--smoothing", type=float, default=0.32)
    parser.add_argument("--upper-face", type=float, default=1.00)
    parser.add_argument("--head-motion", type=float, default=1.00)
    parser.add_argument("--sync-offset-ms", type=float, default=0.0)
    parser.add_argument("--natural-blinks", type=int, default=1)
    parser.add_argument("--loop", type=int, default=0, help="1=loop playback until SIGTERM")
    parser.add_argument("--live-params", action="store_true", help="Listen for JSON param updates on stdin")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    audio_path = Path(args.audio)
    model_path = Path(args.model)
    if not audio_path.exists():
        print(f"[Audio2Face] ERROR: audio file not found: {audio_path}", file=sys.stderr, flush=True)
        return 2
    if not model_path.exists():
        print(f"[Audio2Face] ERROR: ONNX model not found: {model_path}", file=sys.stderr, flush=True)
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
        natural_blinks=bool(args.natural_blinks),
    )

    t_load0 = time.monotonic()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    load_ms = (time.monotonic() - t_load0) * 1000.0

    max_s = args.max_seconds if args.max_seconds > 0 else None
    waveform = decode_audio_16k_mono(audio_path, max_seconds=max_s)
    duration_s = len(waveform) / 16000.0

    t_inf0 = time.monotonic()
    raw_bs = infer_blendshapes(session, waveform)
    inf_ms = (time.monotonic() - t_inf0) * 1000.0
    rtf = (inf_ms / 1000.0) / max(duration_s, 1e-3)

    state_lock = threading.Lock()
    processed_bs = postprocess_blendshapes(raw_bs, params)
    n_frames = len(processed_bs)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if args.self_test:
        scores = (
            processed_bs[:, IDX_JAW_OPEN] * 1.4
            + processed_bs[:, IDX_MOUTH_FUNNEL] * 0.6
            + processed_bs[:, IDX_MOUTH_SMILE_L] * 0.5
            + processed_bs[:, IDX_BROW_INNER_UP] * 0.5
        )
        peak_idx = int(np.argmax(scores))
        peak_vec = processed_bs[peak_idx]
        dgram = build_vmc_frame(
            peak_vec,
            peak_idx / 30.0,
            head_motion_gain=params.head_motion_gain,
            speech_energy=float(peak_vec[IDX_JAW_OPEN]),
        )
        sock.sendto(dgram, (args.host, args.port))
        sock.close()

        top5 = sorted(
            ((ARKIT_52_NAMES[i], float(peak_vec[i])) for i in range(52)),
            key=lambda kv: kv[1],
            reverse=True,
        )[:6]
        top_str = ", ".join(f"{k}={v:.2f}" for k, v in top5)
        print(
            f"[Audio2Face] Self-test OK: duration={duration_s:.2f}s frames={n_frames} load={load_ms:.0f}ms "
            f"infer={inf_ms:.0f}ms (RTF={rtf:.3f}, {1.0/max(rtf,1e-4):.1f}x realtime) "
            f"peakFrame={peak_idx} bytes={len(dgram)} top=[{top_str}]",
            flush=True,
        )
        return 0

    loop_enabled = bool(args.loop)

    if args.live_params:
        def _stdin_watcher() -> None:
            nonlocal processed_bs, loop_enabled
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        with state_lock:
                            if "loop" in payload:
                                loop_enabled = bool(payload["loop"])
                            params.update_from_dict(payload)
                            processed_bs = postprocess_blendshapes(raw_bs, params)
                except Exception:
                    pass

        threading.Thread(target=_stdin_watcher, daemon=True).start()

    print(
        f"[Audio2Face] Ready: audio={audio_path.name} ({duration_s:.1f}s, {n_frames} frames) "
        f"infer={inf_ms:.0f}ms ({1.0/max(rtf,1e-4):.1f}x realtime) loop={loop_enabled} -> {args.host}:{args.port}",
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
                pos = t_sample * 30.0
                i0 = min(n_frames - 1, max(0, int(pos)))
                i1 = min(n_frames - 1, i0 + 1)
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
            # Brief 0.25s pause between loops so sentence boundary feels natural
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
        print(f"[Audio2Face] Finished cleanly (packets={total_packets}, params={json.dumps(asdict(params))})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
