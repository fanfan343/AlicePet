#!/usr/bin/env python3
"""NVIDIA MotionBricks SOMA (fp16 ONNX Runtime CPU) -> 60 FPS VMC/OSC full-body motion service.

Runs NVIDIA's official SOMA character state machine + SmartPrimitiveAssetPool (walk, run, strafe,
crouch, 6 style variations, seamless generative in-between transitions) with ONNX Runtime CPU
(~28-30 ms per ~1.0-1.9s clip) and streams SLERP-smoothed 60 FPS VMC `/VMC/Ext/Bone/Pos` frames
to AlicePet over a dedicated isolated loopback UDP port.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import json
import math
import queue
import signal
import socket
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import onnxruntime as ort
from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder

# MARK: - Import-time shims for NVIDIA MotionBricks headless runtime (no TensorRT/OpenGL viewer needed)

SERVICE_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = SERVICE_DIR / "upstream"
SOURCE_ONNX = UPSTREAM_DIR / "motionbricks" / "assets" / "onnx" / "soma" / "mmm_v6_soma_fp16.onnx"
DEFAULT_SCENE = UPSTREAM_DIR / "assets" / "scenes" / "soma_gym.scene.json"

sys.modules.setdefault("tensorrt", MagicMock(name="tensorrt"))


class _ViewerStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    PACKAGES = ("newton", "pyglet", "imgui_bundle", "trimesh", "OpenGL")

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
        if name.split(".")[0] in self.PACKAGES:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec: Any) -> Any:
        mod = MagicMock(name=spec.name)
        mod.__path__, mod.__spec__ = [], spec
        return mod

    def exec_module(self, module: Any) -> None:
        pass


if not any(isinstance(f, _ViewerStubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _ViewerStubFinder())
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

import warp as wp  # noqa: E402
from motionbricks.io.usd import load_skeleton_from_usd  # noqa: E402
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file  # noqa: E402
from motionbricks.model.contract import ModelRig  # noqa: E402
from motionbricks.model.soma.engine import MotionBricksSOMAEngine  # noqa: E402
from motionbricks.utils.pose_utils import compute_world_poses  # noqa: E402
from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter  # noqa: E402
from app.runtime.common.clip_registry import ClipRegistry  # noqa: E402
from app.runtime.common.io.scene_document import load_scene_document  # noqa: E402
from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool  # noqa: E402
from app.runtime.common.motion import LocomotionCommand, LocomotionSettings  # noqa: E402
from app.runtime.soma.character import SOMACharacter  # noqa: E402


RUNNING = True


def _handle_signal(_signum: int, _frame: object) -> None:
    global RUNNING
    RUNNING = False


# SOMA 77-joint skeleton -> VRM 1.0 Unity Humanoid bindings.
# Spine2+Chest fold cleanly into VRM Chest, and Neck1+Neck2 fold into VRM Neck so models without
# optional `upperChest` (like Alice) preserve exact world-space shoulder and head orientations.
SOMA_TO_VMC_BINDINGS: list[tuple[str, str, str | None]] = [
    ("Hips", "Hips", None),
    ("Spine1", "Spine", "Hips"),
    ("Chest", "Chest", "Spine1"),
    ("Neck2", "Neck", "Chest"),
    ("Head", "Head", "Neck2"),
    ("LeftShoulder", "LeftShoulder", "Chest"),
    ("LeftArm", "LeftUpperArm", "LeftShoulder"),
    ("LeftForeArm", "LeftLowerArm", "LeftArm"),
    ("LeftHand", "LeftHand", "LeftForeArm"),
    ("RightShoulder", "RightShoulder", "Chest"),
    ("RightArm", "RightUpperArm", "RightShoulder"),
    ("RightForeArm", "RightLowerArm", "RightArm"),
    ("RightHand", "RightHand", "RightForeArm"),
    ("LeftLeg", "LeftUpperLeg", "Hips"),
    ("LeftShin", "LeftLowerLeg", "LeftLeg"),
    ("LeftFoot", "LeftFoot", "LeftShin"),
    ("LeftToeBase", "LeftToes", "LeftFoot"),
    ("RightLeg", "RightUpperLeg", "Hips"),
    ("RightShin", "RightLowerLeg", "RightLeg"),
    ("RightFoot", "RightFoot", "RightShin"),
    ("RightToeBase", "RightToes", "RightFoot"),
]

# Responsive locomotion tuning so UI preset clicks feel snappy while preserving generative physics
RESPONSIVE_SETTINGS = LocomotionSettings(
    facing_halflife=0.08,
    velocity_direction_halflife=0.10,
    speed_acceleration_halflife=0.22,
    speed_deceleration_halflife=0.18,
)

# Preset map: (move_xy, facing_xy, run, requested_mode)
PRESETS: dict[str, tuple[tuple[float, float], tuple[float, float], bool, str | None]] = {
    "idle": ((0.0, 0.0), (0.0, 1.0), False, "linear_1d"),
    "walk": ((0.0, 1.0), (0.0, 1.0), False, "linear_1d"),
    "run": ((0.0, 1.0), (0.0, 1.0), True, "linear_1d"),
    "strafe_left": ((-1.0, 0.0), (0.0, 1.0), False, "directional_2d"),
    "strafe_right": ((1.0, 0.0), (0.0, 1.0), False, "directional_2d"),
    "crouch": ((0.0, 1.0), (0.0, 1.0), False, "crouch_directional_2d"),
    "skipping": ((0.0, 1.0), (0.0, 1.0), False, "skipping_1d"),
    "vogue": ((0.0, 1.0), (0.0, 1.0), False, "vogue_1d"),
    "combat": ((0.0, 1.0), (0.0, 1.0), False, "combat_1d"),
    "zombie": ((0.0, 1.0), (0.0, 1.0), False, "zombie_1d"),
    "injured_torso": ((0.0, 1.0), (0.0, 1.0), False, "injured_torso_1d"),
    "injured_leg": ((0.0, 1.0), (0.0, 1.0), False, "injured_leg_1d"),
}


# MARK: - Quaternion Math (xyzw, right-handed Y-up VRM 1.0)


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


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    qvec = np.array([qx, qy, qz], dtype=np.float32)
    t = 2.0 * np.cross(qvec, v)
    return (v + qw * t + np.cross(qvec, t)).astype(np.float32)


def quat_from_y_angle(radians: float) -> np.ndarray:
    half = 0.5 * radians
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float32)


def vrm1_to_vmc_unity_quat(q: np.ndarray) -> tuple[float, float, float, float]:
    """Convert right-handed VRM 1.0 quaternion (xyzw) -> Unity left-handed VMC (.flipX inverse)."""
    qn = quat_normalize(q)
    return (float(qn[0]), float(-qn[1]), float(-qn[2]), float(qn[3]))


# MARK: - ONNX Runtime SOMA Engine (drop-in replacement for NVIDIA TensorRT engine)


class _WallTimer:
    def __init__(self) -> None:
        self.milliseconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        yield
        self.milliseconds[name] = (time.perf_counter() - t0) * 1000.0

    def read(self) -> None:
        pass


class ONNXSOMAEngine(MotionBricksSOMAEngine):
    """NVIDIA MotionBricksSOMAEngine backed by ONNX Runtime CPU on Apple Silicon (~28-30ms/clip)."""

    def __init__(self, onnx_path: Path, *, model_to_runtime: Any) -> None:
        self._closed = False
        self._stream = None
        self._cuda_graph = None
        self._use_cuda_graph = False
        specs = self.contract.tensor_specs
        self._tensor_specs = {s.name: s for s in specs}
        self._buffers = {
            s.name: wp.zeros(s.shape, dtype=wp.dtype_from_numpy(s.dtype), device="cpu")
            for s in specs
        }
        self._inputs = [s.name for s in specs if s.mode.value == "input"]

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._model_to_runtime = wp.transform(model_to_runtime)
        self._runtime_to_model = wp.transform_inverse(self._model_to_runtime)
        self.timer = _WallTimer()
        self._alignment = wp.empty(1, dtype=wp.transform, device="cpu")
        self._generated_frame_count = wp.empty(1, dtype=wp.int32, device="cpu")
        self.last_infer_ms = 0.0
        self.gen_count = 0
        self._initialize_pose_layout()

    @property
    def device(self) -> Any:
        return wp.get_device("cpu")

    @property
    def stream(self) -> None:
        return None

    def enqueue_async(self) -> None:
        t0 = time.perf_counter()
        feeds = {n: self._buffers[n].numpy() for n in self._inputs}
        motion, tokens = self._session.run(["pred_global_motions", "pred_num_tokens"], feeds)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        self.gen_count += 1
        self._buffers["pred_global_motions"].assign(np.asarray(motion, dtype=np.float32).reshape(1, 64, 366))
        self._buffers["pred_num_tokens"].assign(np.asarray(tokens, dtype=np.int32).reshape(1))

    def synchronize(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True


# MARK: - MotionBricks SOMA -> VMC Retargeting Runner


class SOMAVMCRunner:
    def __init__(self, onnx_path: Path) -> None:
        wp.init()
        wp.set_device("cpu")

        self.skeleton = load_skeleton_from_usd(
            get_asset_file("human", "soma_base_skel_minimal_no_root.usd"),
            SOMA_SKELETON_PRIM_PATH,
        )
        self.to_runtime = SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())
        self.from_runtime = wp.transform_inverse(self.to_runtime)
        self.pool = SmartPrimitiveAssetPool(
            ClipRegistry(ModelRig.SOMA, skeleton=self.skeleton),
            self.skeleton.reference_local_transforms,
            self.to_runtime,
        )
        # Load gym scene smart primitives for full locomotion & style coverage
        scene, _, _ = load_scene_document(DEFAULT_SCENE)
        flat_scene = scene.flatten()
        for sp_def in flat_scene.smart_primitives:
            self.pool.load(sp_def.reference)

        self.engine = ONNXSOMAEngine(onnx_path, model_to_runtime=self.to_runtime)
        self.character = SOMACharacter(
            self.engine,
            asset_pool=self.pool,
            initial_transform=wp.transform_identity(),
        )
        self._apply_responsive_settings()

        self.joint_names = [self.skeleton.joint_name(i) for i in range(self.skeleton.num_joints)]
        self.name_to_idx = {name: i for i, name in enumerate(self.joint_names)}

        # Precompute SOMA Y-up reference world rotations (in VRM 1.0 aligned space: +X Left, +Y Up, +Z Forward)
        ref_world = wp.empty_like(self.skeleton.reference_local_transforms)
        compute_world_poses(
            self.skeleton.reference_local_transforms.reshape((1, -1)),
            self.skeleton.parent_indices,
            ref_world.reshape((1, -1)),
            wp.transform_identity(),
        )
        ref_np = ref_world.numpy()
        self.rest_world_quats = np.array(
            [quat_normalize(ref_np[i, 3:7]) for i in range(self.skeleton.num_joints)],
            dtype=np.float32,
        )
        self.rest_world_inv = np.array(
            [quat_inv(self.rest_world_quats[i]) for i in range(self.skeleton.num_joints)],
            dtype=np.float32,
        )
        self._world_buf = wp.empty_like(self.character.current_pose)

        # Pre-resolve binding indices
        self.bindings: list[tuple[int, str, int | None]] = []
        for src, vmc_name, parent_src in SOMA_TO_VMC_BINDINGS:
            if src not in self.name_to_idx:
                continue
            p_idx = self.name_to_idx[parent_src] if (parent_src is not None and parent_src in self.name_to_idx) else None
            self.bindings.append((self.name_to_idx[src], vmc_name, p_idx))

    def _apply_responsive_settings(self) -> None:
        self.character.movement.settings = RESPONSIVE_SETTINGS
        for state in vars(self.character.state_machine).values():
            for st in (state if isinstance(state, (list, tuple)) else (state,)):
                if hasattr(st, "settings") and isinstance(getattr(st, "settings"), LocomotionSettings):
                    st.settings = RESPONSIVE_SETTINGS

    def step_and_extract_vmc_bones(
        self,
        dt: float,
        preset_name: str,
        request_mode_now: bool,
        face_camera: bool,
    ) -> dict[str, np.ndarray]:
        move_xy, facing_xy, run_flag, mode_name = PRESETS.get(preset_name, PRESETS["idle"])
        mv = np.array([move_xy[0], move_xy[1], 0.0], dtype=np.float32)
        fc = np.array([facing_xy[0], facing_xy[1], 0.0], dtype=np.float32)
        fc /= max(float(np.linalg.norm(fc)), 1e-6)

        cmd = LocomotionCommand(
            facing_direction=fc,
            movement=mv if float(np.linalg.norm(mv)) > 1e-3 else None,
            use_max_speed=run_flag,
            requested_mode=mode_name if request_mode_now else None,
        )
        self.character.step(min(max(dt, 1e-4), 0.1), cmd)

        # Compute world poses back in Y-up SOMA model space (+X Left, +Y Up, +Z Forward)
        compute_world_poses(
            self.character.current_pose.reshape((1, -1)),
            self.skeleton.parent_indices,
            self._world_buf.reshape((1, -1)),
            self.from_runtime,
        )
        world_np = self._world_buf.numpy()

        # World rotation deltas relative to SOMA T-pose rest pose
        deltas: dict[int, np.ndarray] = {}
        needed_indices = {idx for idx, _, _ in self.bindings} | {p for _, _, p in self.bindings if p is not None}
        for j in needed_indices:
            w_q = quat_normalize(world_np[j, 3:7])
            deltas[j] = quat_mul(w_q, self.rest_world_inv[j])

        # Optionally strip global root yaw so Alice keeps facing the user on desktop instead of turning away
        hips_idx = self.name_to_idx["Hips"]
        yaw_fix = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if face_camera and hips_idx in deltas:
            fwd = quat_rotate_vec(deltas[hips_idx], np.array([0.0, 0.0, 1.0], dtype=np.float32))
            yaw = math.atan2(float(fwd[0]), float(fwd[2]))
            yaw_fix = quat_from_y_angle(-yaw)

        out: dict[str, np.ndarray] = {}
        for j_idx, vmc_name, p_idx in self.bindings:
            if p_idx is None:
                out[vmc_name] = quat_mul(yaw_fix, deltas[j_idx])
            else:
                out[vmc_name] = quat_mul(quat_inv(deltas[p_idx]), deltas[j_idx])
        return out


def build_vmc_bundle(bones_vrm1: dict[str, np.ndarray], timestamp: float) -> bytes:
    builder = OscBundleBuilder(IMMEDIATELY)

    ok_msg = OscMessageBuilder("/VMC/Ext/OK")
    ok_msg.add_arg(1)
    builder.add_content(ok_msg.build())

    t_msg = OscMessageBuilder("/VMC/Ext/T")
    t_msg.add_arg(float(timestamp))
    builder.add_content(t_msg.build())

    for bone_name, q_vrm1 in bones_vrm1.items():
        qx, qy, qz, qw = vrm1_to_vmc_unity_quat(q_vrm1)
        m = OscMessageBuilder("/VMC/Ext/Bone/Pos")
        m.add_arg(bone_name)
        m.add_arg(0.0)
        m.add_arg(0.0)
        m.add_arg(0.0)
        m.add_arg(qx)
        m.add_arg(qy)
        m.add_arg(qz)
        m.add_arg(qw)
        builder.add_content(m.build())

    return builder.build().dgram


def start_stdin_reader(cmd_queue: "queue.Queue[dict[str, Any]]") -> None:
    def _reader() -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    cmd_queue.put(obj)
            except json.JSONDecodeError:
                pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def main() -> int:
    parser = argparse.ArgumentParser(description="NVIDIA MotionBricks SOMA -> 60 FPS VMC service")
    parser.add_argument("--onnx", type=Path, default=SOURCE_ONNX)
    parser.add_argument("--preset", type=str, default="idle")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=39539)
    parser.add_argument("--face-camera", type=int, default=1)
    parser.add_argument("--live-commands", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    t0 = time.perf_counter()
    runner = SOMAVMCRunner(args.onnx)
    load_s = time.perf_counter() - t0
    print(f"[MotionBricks] SOMA ONNX + Warp loaded in {load_s:.2f}s (joints={len(runner.bindings)})", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)
    preset = args.preset if args.preset in PRESETS else "idle"
    face_camera = bool(args.face_camera)

    if args.self_test:
        # Warm up for 2.0 seconds (60 frames at 30 Hz) so MotionBricks actively generates & transitions into the preset
        dt = 1.0 / 30.0
        last_bones: dict[str, np.ndarray] = {}
        for i in range(60):
            last_bones = runner.step_and_extract_vmc_bones(
                dt=dt,
                preset_name=preset,
                request_mode_now=(i == 0 or i == 5),
                face_camera=face_camera,
            )
        # Send 5 packets so the UDP receiver cleanly registers fresh stream state
        for k in range(5):
            sock.sendto(build_vmc_bundle(last_bones, time.monotonic()), addr)
            time.sleep(0.015)
        print(
            f"[MotionBricks] Self-test OK: preset={preset} gens={runner.engine.gen_count} "
            f"last_infer_ms={runner.engine.last_infer_ms:.1f} bones={len(last_bones)}",
            flush=True,
        )
        return 0

    cmd_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
    if args.live_commands:
        start_stdin_reader(cmd_queue)

    # Stream 60 FPS VMC by stepping SOMA at 30 Hz and SLERP-interpolating midpoint frames
    sim_dt = 1.0 / 30.0
    out_dt = 1.0 / 60.0
    pending_mode_request = True
    prev_bones = runner.step_and_extract_vmc_bones(sim_dt, preset, True, face_camera)
    curr_bones = runner.step_and_extract_vmc_bones(sim_dt, preset, False, face_camera)
    pending_mode_request = False

    print(
        f"[MotionBricks] 运行中 · 当前模式={preset} · 最近推理={runner.engine.last_infer_ms:.1f}ms (60fps VMC)",
        flush=True,
    )

    last_gen_logged = runner.engine.gen_count
    substep = 0
    next_tick = time.perf_counter()

    while RUNNING:
        # Drain live commands from Swift panel
        while not cmd_queue.empty():
            cmd = cmd_queue.get_nowait()
            new_preset = str(cmd.get("preset", preset))
            if new_preset in PRESETS and new_preset != preset:
                preset = new_preset
                pending_mode_request = True
                print(f"[MotionBricks] 平滑过渡至模式：{preset}", flush=True)
            if "face_camera" in cmd:
                face_camera = bool(cmd["face_camera"])

        if substep == 0:
            prev_bones = curr_bones
            curr_bones = runner.step_and_extract_vmc_bones(
                sim_dt,
                preset,
                pending_mode_request,
                face_camera,
            )
            pending_mode_request = False
            if runner.engine.gen_count != last_gen_logged:
                last_gen_logged = runner.engine.gen_count
                print(
                    f"[MotionBricks] 运行中 [{preset}] · 第 {last_gen_logged} 次生成 {runner.engine.last_infer_ms:.1f}ms",
                    flush=True,
                )
            frame_bones = {
                k: quat_slerp(prev_bones[k], curr_bones[k], 0.5)
                for k in curr_bones
            }
            substep = 1
        else:
            frame_bones = curr_bones
            substep = 0

        sock.sendto(build_vmc_bundle(frame_bones, time.monotonic()), addr)

        next_tick += out_dt
        now = time.perf_counter()
        sleep_s = next_tick - now
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        elif sleep_s < -0.2:
            next_tick = now

    sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
