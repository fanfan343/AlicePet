#!/usr/bin/env python3
"""ARDY Mini (distilled all-MiniLM-L6-v2 + ARDY-Core40 ONNX) -> real-time VMC/OSC motion engine.

Faithfully implements NVIDIA ARDY Core40 autoregressive streaming (40-frame history, 40-frame
generation at 20 FPS, 10-step DDIM with CFG, FSQ requantization, and cskel27 -> VRM 1.0 retargeting)
using pure `onnxruntime` + `tokenizers` + `numpy`.

Streams smooth 60 FPS SLERP-interpolated VMC `/VMC/Ext/Bone/Pos` packets to AlicePet (127.0.0.1:39539)
and accepts live English prompt changes over stdin (`--live-prompts`).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import queue
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder
from tokenizers import Tokenizer

RUNNING = True


def _handle_signal(_signum: int, _frame: object) -> None:
    global RUNNING
    RUNNING = False


# Core27 -> VRM 1.0 Unity Humanoid bindings (Spine1+Spine2+Spine3 cleanly folded into Chest
# so models without optional `upperChest` like Alice preserve 100% exact shoulder/neck world orientations!)
CSKEL27_TO_VMC_BINDINGS: list[tuple[str, str, str | None]] = [
    # (source_joint_name, vmc_unity_bone_name, parent_source_joint_name_or_None)
    ("Hips", "Hips", None),
    ("Spine", "Spine", "Hips"),
    ("Spine3", "Chest", "Spine"),
    ("Neck", "Neck", "Spine3"),
    ("Head", "Head", "Neck"),
    ("LeftShoulder", "LeftShoulder", "Spine3"),
    ("LeftArm", "LeftUpperArm", "LeftShoulder"),
    ("LeftForeArm", "LeftLowerArm", "LeftArm"),
    ("LeftHand", "LeftHand", "LeftForeArm"),
    ("RightShoulder", "RightShoulder", "Spine3"),
    ("RightArm", "RightUpperArm", "RightShoulder"),
    ("RightForeArm", "RightLowerArm", "RightArm"),
    ("RightHand", "RightHand", "RightForeArm"),
    ("LeftUpLeg", "LeftUpperLeg", "Hips"),
    ("LeftLeg", "LeftLowerLeg", "LeftUpLeg"),
    ("LeftFoot", "LeftFoot", "LeftLeg"),
    ("LeftToeBase", "LeftToes", "LeftFoot"),
    ("RightUpLeg", "RightUpperLeg", "Hips"),
    ("RightLeg", "RightLowerLeg", "RightUpLeg"),
    ("RightFoot", "RightFoot", "RightLeg"),
    ("RightToeBase", "RightToes", "RightFoot"),
]

CORE27_REST_HIPS_HEIGHT = 0.9544128252334833

# Positive, concrete HumanML3D-friendly motion-style cues that pull ARDY's MiniLM embedding
# toward gentle, graceful, compact anime-girl movements rather than wide/heavy realistic mocap.
ANIME_GIRL_STYLE_SUFFIX = "gently and gracefully with light, soft, modest movements and a relaxed feminine posture"


def stylize_prompt_for_anime_girl(raw_prompt: str, enabled: bool = True) -> str:
    text = raw_prompt.strip()
    if not text or not enabled:
        return text
    lower = text.lower()
    # Replace generic "a person" subject with a gentle young girl subject so ARDY adopts compact stride/posture
    for prefix in ("a person ", "person ", "someone ", "a human "):
        if lower.startswith(prefix):
            text = "a petite gentle young girl " + text[len(prefix) :]
            lower = text.lower()
            break
    if not any(k in lower for k in ("girl", "woman", "lady", "she ")):
        text = f"a petite gentle young girl {text}"
    if "gracefully" not in lower and "gently" not in lower and "softly" not in lower:
        text = f"{text.rstrip('. ')}, {ANIME_GIRL_STYLE_SUFFIX}"
    return text


# MARK: - Quaternion Math (xyzw, right-handed VRM 1.0)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_inv(q: np.ndarray) -> np.ndarray:
    qn = quat_normalize(q)
    return np.array([-qn[0], -qn[1], -qn[2], qn[3]], dtype=np.float32)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return quat_normalize(
        np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float32,
        )
    )


def quat_slerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    t = max(0.0, min(1.0, float(alpha)))
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return quat_normalize((1.0 - t) * a + t * b)
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_t = math.sin(theta)
    w1 = math.sin((1.0 - t) * theta) / sin_t
    w2 = math.sin(t * theta) / sin_t
    return quat_normalize(w1 * a + w2 * b)


def mat3_to_quat(m: np.ndarray) -> np.ndarray:
    """Converts a 3x3 rotation matrix into a unit quaternion (x, y, z, w)."""
    m00, m01, m02 = float(m[0, 0]), float(m[0, 1]), float(m[0, 2])
    m10, m11, m12 = float(m[1, 0]), float(m[1, 1]), float(m[1, 2])
    m20, m21, m22 = float(m[2, 0]), float(m[2, 1]), float(m[2, 2])
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return quat_normalize(np.array([x, y, z, w], dtype=np.float32))


def vrm_to_unity_quat(q: np.ndarray) -> tuple[float, float, float, float]:
    qn = quat_normalize(q)
    return (float(qn[0]), float(-qn[1]), float(-qn[2]), float(qn[3]))


# MARK: - ARDY Mini ONNX Streaming Session


class ArdyMiniEngine:
    """Pure ONNX Runtime implementation of ARDY Mini Core40 autoregressive streaming."""

    def __init__(self, models_dir: Path, intra_threads: int = 4, anime_girl_style: bool = True) -> None:
        self.models_dir = models_dir
        self.anime_girl_style = anime_girl_style
        manifest_path = models_dir / "model.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing ARDY Mini manifest: {manifest_path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.dims = self.manifest["dimensions"]
        self.recenter_cfg = self.manifest["recenter"]
        self.quant_cfg = self.manifest["latent_quantization"]
        self.diff_cfg = self.manifest["diffusion"]
        self.graphs = self.manifest["graphs"]

        self.joint_names: list[str] = self.manifest["skeleton"]["joint_names"]
        self.joint_parents: list[int] = list(self.manifest["skeleton"]["parents"])
        self.neutral_joints = np.asarray(self.manifest["skeleton"]["neutral_joints"], dtype=np.float32)
        self.bone_offsets = np.zeros_like(self.neutral_joints)
        for j, p in enumerate(self.joint_parents):
            if p >= 0:
                self.bone_offsets[j] = self.neutral_joints[j] - self.neutral_joints[p]
        self.joint_index = {name: i for i, name in enumerate(self.joint_names)}
        self.retarget_plan: list[tuple[int, str, int | None]] = []
        for src, vmc_name, parent_src in CSKEL27_TO_VMC_BINDINGS:
            if src in self.joint_index:
                p_idx = self.joint_index[parent_src] if (parent_src and parent_src in self.joint_index) else None
                self.retarget_plan.append((self.joint_index[src], vmc_name, p_idx))

        self.tokenizer = Tokenizer.from_file(str(models_dir / "tokenizer/tokenizer.json"))
        self.max_len = int(self.manifest["tokenizer"].get("max_length", 128))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]

        t0 = time.monotonic()
        self.text_sess = ort.InferenceSession(str(models_dir / "text_encoder.onnx"), sess_options=opts, providers=providers)
        self.denoiser_sess = ort.InferenceSession(str(models_dir / "denoiser.onnx"), sess_options=opts, providers=providers)
        self.decoder_sess = ort.InferenceSession(str(models_dir / "decoder.onnx"), sess_options=opts, providers=providers)
        self.load_ms = (time.monotonic() - t0) * 1000.0

        # Precompute numpy arrays for fast recenter & FSQ requantization
        self._root_mean = np.asarray(self.recenter_cfg["root_mean"], dtype=np.float32)
        self._root_std = np.asarray(self.recenter_cfg["root_std"], dtype=np.float32)
        self._q_mean = np.asarray(self.quant_cfg["mean"], dtype=np.float32)
        self._q_std = np.asarray(self.quant_cfg["std"], dtype=np.float32)
        self._q_half = (np.asarray(self.quant_cfg["levels"], dtype=np.int32) // 2).astype(np.float32)

        # Autoregressive session state
        self.rng = np.random.default_rng(42)
        self.global_hybrid = np.zeros((0, self.dims["hybrid_dim"]), dtype=np.float32)
        self.frame_count = 0
        self.prompt_cache: dict[str, np.ndarray] = {}

    def reset(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.global_hybrid = np.zeros((0, self.dims["hybrid_dim"]), dtype=np.float32)
        self.frame_count = 0

    def encode_prompt(self, prompt: str) -> tuple[np.ndarray, float]:
        styled = stylize_prompt_for_anime_girl(prompt, enabled=self.anime_girl_style)
        key = styled
        if key in self.prompt_cache:
            return self.prompt_cache[key], 0.0
        t0 = time.monotonic()
        enc = self.tokenizer.encode(key, add_special_tokens=True)
        ids = list(enc.ids)
        mask = list(enc.attention_mask)
        tids = list(enc.type_ids) if enc.type_ids else [0] * len(ids)
        if len(ids) > self.max_len:
            ids = ids[: self.max_len - 1] + [ids[-1]]
            mask = mask[: self.max_len - 1] + [mask[-1]]
            tids = tids[: self.max_len - 1] + [tids[-1]]

        inp = self.graphs["text_encoder"]["inputs"]
        out_name = self.graphs["text_encoder"]["outputs"]["textConditions"]
        feeds = {
            inp["inputIds"]: np.asarray([ids], dtype=np.int64),
            inp["attentionMask"]: np.asarray([mask], dtype=np.int64),
            inp["tokenTypeIds"]: np.asarray([tids], dtype=np.int64),
        }
        cond = self.text_sess.run([out_name], feeds)[0].astype(np.float32)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self.prompt_cache[key] = cond
        return cond, elapsed_ms

    def _prepare_history(self) -> tuple[np.ndarray | None, int, np.ndarray, float]:
        dims = self.dims
        fpt = dims["num_frames_per_token"]
        avail_tokens = min(self.frame_count // fpt, len(self.global_hybrid))
        req_tokens = min(dims["history_tokens"], avail_tokens)
        if req_tokens == 0:
            return None, 0, np.zeros(3, dtype=np.float32), 0.0

        hist = self.global_hybrid[avail_tokens - req_tokens : avail_tokens].copy()  # (T_h, hybrid_dim)
        root_dim = dims["root_features_per_frame"]
        roots = hist[:, : dims["nframe_root_dim"]].reshape(req_tokens * fpt, root_dim)
        raw_roots = roots * self._root_std + self._root_mean

        px, py, pz = self.recenter_cfg["position_indices"]
        hc, hs = self.recenter_cfg["heading_indices"]
        center_x = float(raw_roots[-1, px])
        center_z = float(raw_roots[-1, pz])
        first_heading = math.atan2(float(raw_roots[0, hs]), float(raw_roots[0, hc]))

        raw_roots[:, px] -= center_x
        raw_roots[:, pz] -= center_z
        norm_roots = (raw_roots - self._root_mean) / self._root_std
        hist[:, : dims["nframe_root_dim"]] = norm_roots.reshape(req_tokens, dims["nframe_root_dim"])
        return hist, req_tokens, np.array([center_x, 0.0, center_z], dtype=np.float32), first_heading

    def _recenter_and_requantize(
        self,
        decoder_hybrid: np.ndarray,
        valid_tokens: int,
        global_trans: np.ndarray,
    ) -> np.ndarray:
        dims = self.dims
        fpt = dims["num_frames_per_token"]
        root_dim = dims["root_features_per_frame"]
        valid_frames = valid_tokens * fpt

        roots = decoder_hybrid[:valid_tokens, : dims["nframe_root_dim"]].reshape(valid_frames, root_dim)
        raw_roots = roots * self._root_std + self._root_mean
        px, _, pz = self.recenter_cfg["position_indices"]
        cx = float(raw_roots[-1, px])
        cz = float(raw_roots[-1, pz])

        raw_roots[:, px] -= cx
        raw_roots[:, pz] -= cz
        decoder_hybrid[:valid_tokens, : dims["nframe_root_dim"]] = (
            (raw_roots - self._root_mean) / self._root_std
        ).reshape(valid_tokens, dims["nframe_root_dim"])

        # FSQ body latent requantization (ties-to-even via np.rint)
        lat_start = dims["nframe_root_dim"]
        lat = decoder_hybrid[:valid_tokens, lat_start:]
        raw_lat = lat * self._q_std + self._q_mean
        clamped = np.clip(raw_lat, -1.0, 1.0)
        discrete = np.rint(clamped * self._q_half) / self._q_half
        decoder_hybrid[:valid_tokens, lat_start:] = ((discrete - self._q_mean) / self._q_std).astype(np.float32)

        return np.array([global_trans[0] + cx, global_trans[1], global_trans[2] + cz], dtype=np.float32)

    def generate_window(self, prompt: str, cfg_weight: float = 2.5) -> tuple[list[dict[str, tuple[np.ndarray, np.ndarray]]], dict[str, float]]:
        """Generates one 40-frame (2.0s at 20 FPS) autoregressive chunk and returns retargeted VRM bone quaternions."""
        dims = self.dims
        fpt = dims["num_frames_per_token"]
        max_tokens = dims["max_tokens"]
        max_frames = dims["max_frames"]
        hybrid_dim = dims["hybrid_dim"]
        gen_tokens = dims["generation_tokens"]
        gen_frames = dims["generation_frames"]

        text_cond, text_ms = self.encode_prompt(prompt)
        hist, hist_tokens, global_trans, first_heading = self._prepare_history()
        hist_frames = hist_tokens * fpt

        x = np.zeros((1, max_tokens, hybrid_dim), dtype=np.float32)
        if hist is not None:
            x[0, :hist_tokens] = hist
        x[0, hist_tokens : hist_tokens + gen_tokens] = self.rng.standard_normal((gen_tokens, hybrid_dim)).astype(np.float32)

        hist_mask = np.zeros((1, max_frames), dtype=np.float32)
        gen_mask = np.zeros((1, max_frames), dtype=np.float32)
        hist_tmask = np.zeros((1, max_tokens), dtype=np.float32)
        gen_tmask = np.zeros((1, max_tokens), dtype=np.float32)
        hist_mask[0, :hist_frames] = 1.0
        gen_mask[0, hist_frames : hist_frames + gen_frames] = 1.0
        hist_tmask[0, :hist_tokens] = 1.0
        gen_tmask[0, hist_tokens : hist_tokens + gen_tokens] = 1.0

        d_inp = self.graphs["denoiser"]["inputs"]
        d_out = self.graphs["denoiser"]["outputs"]["predX0"]
        timesteps = self.diff_cfg["timesteps"]
        alphas = self.diff_cfg["alphas_cumprod"]
        alphas_prev = self.diff_cfg["alphas_cumprod_prev"]

        t_den0 = time.monotonic()
        for ts in timesteps:
            feeds = {
                d_inp["cfgWeight"]: np.asarray([cfg_weight], dtype=np.float32),
                d_inp["x"]: x,
                d_inp["historyLength"]: np.asarray([hist_frames], dtype=np.int64),
                d_inp["generationLength"]: np.asarray([gen_frames], dtype=np.int64),
                d_inp["historyMask"]: hist_mask,
                d_inp["generationMask"]: gen_mask,
                d_inp["historyTokenMask"]: hist_tmask,
                d_inp["generationTokenMask"]: gen_tmask,
                d_inp["textConditions"]: text_cond,
                d_inp["timestep"]: np.asarray([ts], dtype=np.int64),
                d_inp["firstHeadingAngle"]: np.asarray([first_heading], dtype=np.float32),
            }
            pred_x0 = self.denoiser_sess.run([d_out], feeds)[0]

            alpha = float(alphas[ts])
            alpha_prev = float(alphas_prev[ts])
            sqrt_a = math.sqrt(alpha)
            sqrt_recip_m1 = math.sqrt(max(0.0, 1.0 / alpha - 1.0))
            sqrt_prev = math.sqrt(alpha_prev)
            sqrt_1m_prev = math.sqrt(max(0.0, 1.0 - alpha_prev))

            sl = slice(hist_tokens, hist_tokens + gen_tokens)
            pred_slice = pred_x0[0, sl]
            cur_slice = x[0, sl]
            if sqrt_recip_m1 == 0.0:
                eps = np.zeros_like(cur_slice)
            else:
                eps = (cur_slice / sqrt_a - pred_slice) / sqrt_recip_m1
            x[0, sl] = (pred_slice * sqrt_prev + eps * sqrt_1m_prev).astype(np.float32)
        denoise_ms = (time.monotonic() - t_den0) * 1000.0

        valid_tokens = hist_tokens + gen_tokens
        decoder_hybrid = np.zeros((max_tokens, hybrid_dim), dtype=np.float32)
        decoder_hybrid[:valid_tokens] = x[0, :valid_tokens]
        recentered_trans = self._recenter_and_requantize(decoder_hybrid, valid_tokens, global_trans)

        pad_mask = np.zeros((1, max_frames), dtype=np.float32)
        pad_mask[0, : valid_tokens * fpt] = 1.0
        dec_inp = self.graphs["decoder"]["inputs"]
        dec_outs = self.graphs["decoder"]["outputs"]

        t_dec0 = time.monotonic()
        dec_res = self.decoder_sess.run(
            [dec_outs["normalizedMotion"], dec_outs["globalRotations"]],
            {
                dec_inp["hybridTokens"]: decoder_hybrid[None, :, :],
                dec_inp["motionPadMask"]: pad_mask,
                dec_inp["globalTranslation"]: recentered_trans[None, :],
            },
        )
        decode_ms = (time.monotonic() - t_dec0) * 1000.0

        decoded_motion = dec_res[0][0]   # (max_frames, 330)
        global_rots = dec_res[1][0]      # (max_frames, 27, 3, 3)

        # Build global hybrid tokens for seamless next-window continuation
        new_tokens = np.zeros((gen_tokens, hybrid_dim), dtype=np.float32)
        r_dim = dims["root_features_per_frame"]
        for tk in range(gen_tokens):
            for fi in range(fpt):
                f_idx = hist_frames + tk * fpt + fi
                new_tokens[tk, fi * r_dim : (fi + 1) * r_dim] = decoded_motion[f_idx, :r_dim]
            new_tokens[tk, dims["nframe_root_dim"] :] = decoder_hybrid[hist_tokens + tk, dims["nframe_root_dim"] :]

        # Keep bounded history buffer (only need last 20 tokens max)
        self.global_hybrid = np.concatenate([self.global_hybrid[-dims["history_tokens"] :], new_tokens], axis=0)
        self.frame_count += gen_frames

        # Retarget 40 generated frames from cskel27 global 3x3 rotations -> VRM 1.0 local bone quaternions
        # PLUS source 3D world joint positions (Hips-relative, meters) for proportional 2-Bone Arm IK,
        # AND continuous global root XZ translation (meters) for Picture-World room locomotion.
        vrm_frames: list[tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]] = []
        n_joints = len(self.joint_names)
        px_i, py_i, pz_i = self.recenter_cfg["position_indices"]
        for f_off in range(gen_frames):
            f_idx = hist_frames + f_off
            raw_root = decoded_motion[f_idx, :r_dim] * self._root_std + self._root_mean
            root_world = np.array([float(raw_root[px_i]), 0.0, float(raw_root[pz_i])], dtype=np.float32)
            mats = global_rots[f_idx]  # (27, 3, 3)
            g_quats = [mat3_to_quat(mats[j]) for j in range(n_joints)]
            # Forward kinematics in cskel27 world space (Hips anchored at origin)
            g_pos = np.zeros((n_joints, 3), dtype=np.float32)
            for j in range(n_joints):
                p = self.joint_parents[j]
                if p >= 0:
                    g_pos[j] = g_pos[p] + (mats[p] @ self.bone_offsets[j])
            frame_bones: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for src_idx, vmc_name, parent_idx in self.retarget_plan:
                g_q = g_quats[src_idx]
                if parent_idx is None:
                    local_q = g_q
                else:
                    local_q = quat_mul(quat_inv(g_quats[parent_idx]), g_q)
                frame_bones[vmc_name] = (local_q, g_pos[src_idx].copy())
            vrm_frames.append((frame_bones, root_world))

        timings = {
            "text_ms": text_ms,
            "denoise_ms": denoise_ms,
            "decode_ms": decode_ms,
            "total_ms": text_ms + denoise_ms + decode_ms,
        }
        return vrm_frames, timings


def build_vmc_pose_bundle(
    bones_vrm: dict[str, tuple[np.ndarray, np.ndarray]],
    timestamp_s: float,
    root_world_vrm: np.ndarray | None = None,
) -> bytes:
    bundle = OscBundleBuilder(IMMEDIATELY)

    ok_msg = OscMessageBuilder(address="/VMC/Ext/OK")
    ok_msg.add_arg(1)
    bundle.add_content(ok_msg.build())

    t_msg = OscMessageBuilder(address="/VMC/Ext/T")
    t_msg.add_arg(float(timestamp_s))
    bundle.add_content(t_msg.build())

    if root_world_vrm is not None:
        r_msg = OscMessageBuilder(address="/VMC/Ext/Root/Pos")
        r_msg.add_arg("root")
        # Convert VRM 1.0 right-handed position (+X Left) -> Unity VMC (.flipX inverse: -X, +Y, +Z)
        r_msg.add_arg(float(-root_world_vrm[0]))
        r_msg.add_arg(float(root_world_vrm[1]))
        r_msg.add_arg(float(root_world_vrm[2]))
        r_msg.add_arg(0.0)
        r_msg.add_arg(0.0)
        r_msg.add_arg(0.0)
        r_msg.add_arg(1.0)
        bundle.add_content(r_msg.build())

    for bone_name, (q_vrm, p_vrm) in bones_vrm.items():
        qx, qy, qz, qw = vrm_to_unity_quat(q_vrm)
        # Convert VRM 1.0 right-handed position (+X Left) -> Unity VMC (.flipX inverse: -X, +Y, +Z)
        px, py, pz = float(-p_vrm[0]), float(p_vrm[1]), float(p_vrm[2])
        msg = OscMessageBuilder(address="/VMC/Ext/Bone/Pos")
        msg.add_arg(bone_name)
        msg.add_arg(px)
        msg.add_arg(py)
        msg.add_arg(pz)
        msg.add_arg(qx)
        msg.add_arg(qy)
        msg.add_arg(qz)
        msg.add_arg(qw)
        bundle.add_content(msg.build())

    return bundle.build().dgram


def main() -> int:
    parser = argparse.ArgumentParser(description="ARDY Mini real-time streaming text-to-motion VMC engine")
    default_dir = Path(__file__).resolve().parent / "models"
    parser.add_argument("--models-dir", default=str(default_dir))
    parser.add_argument("--prompt", default="a petite gentle young girl stands gracefully and waves softly with her right hand")
    parser.add_argument("--cfg", type=float, default=2.2)
    parser.add_argument("--style-mode", choices=["anime_girl", "raw"], default="anime_girl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=39539)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--live-prompts", action="store_true", help="Read JSON prompt updates from stdin")
    parser.add_argument("--self-test", action="store_true", help="Generate 1 real chunk, send expressive frame via VMC, and exit")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    models_dir = Path(args.models_dir)
    engine = ArdyMiniEngine(models_dir=models_dir, intra_threads=args.threads, anime_girl_style=(args.style_mode == "anime_girl"))
    engine.reset(seed=args.seed)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if args.self_test:
        frames, timings = engine.generate_window(args.prompt, cfg_weight=args.cfg)
        # Pick frame ~22 (mid-wave / mid-action) for visual snapshot inspection
        pick_idx = min(len(frames) - 1, 22)
        pick_bones, pick_root = frames[pick_idx]
        dgram = build_vmc_pose_bundle(pick_bones, pick_idx / 20.0, root_world_vrm=pick_root)
        sock.sendto(dgram, (args.host, args.port))
        sock.close()
        print(
            f"[ARDY] Self-test OK: load={engine.load_ms:.0f}ms text={timings['text_ms']:.0f}ms "
            f"denoise={timings['denoise_ms']:.0f}ms decode={timings['decode_ms']:.0f}ms "
            f"total={timings['total_ms']:.0f}ms frames={len(frames)} bones={len(pick_bones)} "
            f"root=({pick_root[0]:+.2f},{pick_root[2]:+.2f}) bytes={len(dgram)} prompt='{args.prompt}'",
            flush=True,
        )
        return 0

    prompt_lock = threading.Lock()
    current_prompt = args.prompt.strip()
    current_cfg = float(args.cfg)

    if args.live_prompts:
        def _stdin_loop() -> None:
            nonlocal current_prompt, current_cfg
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        with prompt_lock:
                            if "prompt" in data and str(data["prompt"]).strip():
                                current_prompt = str(data["prompt"]).strip()
                                print(f"[ARDY] Prompt queued: '{current_prompt}'", flush=True)
                            if "cfg" in data:
                                current_cfg = max(0.5, min(6.0, float(data["cfg"])))
                except Exception:
                    pass

        threading.Thread(target=_stdin_loop, daemon=True).start()

    print(
        f"[ARDY] Loaded in {engine.load_ms:.0f}ms. Streaming to {args.host}:{args.port} (initial='{current_prompt}')",
        flush=True,
    )

    frame_queue: collections.deque[tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]] = collections.deque()
    queue_lock = threading.Lock()

    def _generator_loop() -> None:
        chunk_id = 0
        while RUNNING:
            with queue_lock:
                buffered = len(frame_queue)
            # Keep ~2–4 seconds (40–80 frames at 20 FPS) buffered ahead so playback never starves
            if buffered > 55:
                time.sleep(0.05)
                continue
            with prompt_lock:
                p = current_prompt
                cfg = current_cfg
            frames, timings = engine.generate_window(p, cfg_weight=cfg)
            chunk_id += 1
            with queue_lock:
                frame_queue.extend(frames)
            print(
                f"[ARDY] Chunk #{chunk_id} ready ({timings['total_ms']:.0f}ms: text={timings['text_ms']:.0f} "
                f"denoise={timings['denoise_ms']:.0f} decode={timings['decode_ms']:.0f}) prompt='{p}'",
                flush=True,
            )

    gen_thread = threading.Thread(target=_generator_loop, daemon=True)
    gen_thread.start()

    # Wait for first chunk before starting 60 FPS VMC playback
    while RUNNING:
        with queue_lock:
            if len(frame_queue) >= 2:
                break
        time.sleep(0.02)

    t_start = time.monotonic()
    send_fps = 60.0
    period = 1.0 / send_fps
    substeps = 3  # 20 FPS source -> 60 FPS SLERP playback
    cur_frame: tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray] | None = None
    next_frame: tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray] | None = None
    sub_idx = 0
    packets_sent = 0

    try:
        while RUNNING:
            if cur_frame is None or next_frame is None or sub_idx >= substeps:
                with queue_lock:
                    if cur_frame is None and frame_queue:
                        cur_frame = frame_queue.popleft()
                    if next_frame is None and frame_queue:
                        next_frame = frame_queue.popleft()
                    elif sub_idx >= substeps and frame_queue:
                        cur_frame = next_frame
                        next_frame = frame_queue.popleft()
                        sub_idx = 0

            if cur_frame is not None:
                cur_bones, cur_root = cur_frame
                if next_frame is not None:
                    next_bones, next_root = next_frame
                    alpha = min(1.0, sub_idx / float(substeps))
                    blended_bones = {
                        k: (
                            quat_slerp(cur_bones[k][0], next_bones[k][0], alpha),
                            ((1.0 - alpha) * cur_bones[k][1] + alpha * next_bones[k][1]).astype(np.float32),
                        )
                        for k in cur_bones
                        if k in next_bones
                    }
                    blended_root = ((1.0 - alpha) * cur_root + alpha * next_root).astype(np.float32)
                else:
                    blended_bones, blended_root = cur_bones, cur_root

                elapsed = time.monotonic() - t_start
                sock.sendto(build_vmc_pose_bundle(blended_bones, elapsed, root_world_vrm=blended_root), (args.host, args.port))
                packets_sent += 1
                sub_idx += 1

            next_tick = t_start + packets_sent * period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.001:
                time.sleep(sleep_s)
    finally:
        sock.close()
        print(f"[ARDY] Stopped cleanly (packets={packets_sent})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
