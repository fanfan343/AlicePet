#!/usr/bin/env python3
"""Headless MediaPipe (52 ARKit face + upper body + 10-finger hands) -> VMC/OSC UDP tracker.

Launched on demand by AlicePet (or standalone CLI). Sends one atomic OSC bundle per frame
to 127.0.0.1:39539 and releases the webcam immediately on SIGTERM/SIGINT.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder

from rig_solver import MediaPipeRigSolver

RUNNING = True


def _handle_signal(signum: int, _frame: object) -> None:
    global RUNNING
    RUNNING = False


def build_vmc_bundle(
    bones: dict[str, tuple[float, float, float, float]],
    blendshapes: dict[str, float],
    timestamp_s: float,
) -> bytes:
    bundle = OscBundleBuilder(IMMEDIATELY)

    ok_msg = OscMessageBuilder(address="/VMC/Ext/OK")
    ok_msg.add_arg(1)
    bundle.add_content(ok_msg.build())

    t_msg = OscMessageBuilder(address="/VMC/Ext/T")
    t_msg.add_arg(float(timestamp_s))
    bundle.add_content(t_msg.build())

    for bone_name, (qx, qy, qz, qw) in bones.items():
        msg = OscMessageBuilder(address="/VMC/Ext/Bone/Pos")
        msg.add_arg(bone_name)
        msg.add_arg(0.0)
        msg.add_arg(0.0)
        msg.add_arg(0.0)
        msg.add_arg(float(qx))
        msg.add_arg(float(qy))
        msg.add_arg(float(qz))
        msg.add_arg(float(qw))
        bundle.add_content(msg.build())

    if blendshapes:
        for name, weight in blendshapes.items():
            msg = OscMessageBuilder(address="/VMC/Ext/Blend/Val")
            msg.add_arg(name)
            msg.add_arg(float(weight))
            bundle.add_content(msg.build())
        apply_msg = OscMessageBuilder(address="/VMC/Ext/Blend/Apply")
        bundle.add_content(apply_msg.build())

    return bundle.build().dgram


def run_self_test(host: str, port: int, models_dir: Path) -> int:
    """Verifies model loading, synthetic Face+Pose+Hand rig solving, and UDP VMC packet serialization."""
    for fname in ("face_landmarker.task", "pose_landmarker_lite.task", "hand_landmarker.task"):
        p = models_dir / fname
        if not p.exists():
            print(f"[MotionService] Missing model: {p}", file=sys.stderr)
            return 2

    # Instantiate all 3 MediaPipe Task runners to prove models load cleanly on this machine
    face_lm = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(models_dir / "face_landmarker.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
    )
    pose_lm = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(models_dir / "pose_landmarker_lite.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )
    hand_lm = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(models_dir / "hand_landmarker.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_lm.close()
    pose_lm.close()
    hand_lm.close()

    # Synthetic Pose + Face + Hand check
    def lm(x: float, y: float, z: float, v: float = 0.99) -> SimpleNamespace:
        return SimpleNamespace(x=x, y=y, z=z, visibility=v)

    pose_world = [lm(0, 0, 0)] * 33
    pose_world[11] = lm(0.18, -0.45, 0.0)   # Left shoulder
    pose_world[12] = lm(-0.18, -0.45, 0.0)  # Right shoulder
    pose_world[13] = lm(0.36, -0.58, -0.08) # Left elbow raised
    pose_world[15] = lm(0.42, -0.82, -0.15) # Left wrist waving up
    pose_world[14] = lm(-0.26, -0.22, 0.0)  # Right elbow relaxed down
    pose_world[16] = lm(-0.28, 0.02, -0.05) # Right wrist down
    pose_world[23] = lm(0.12, 0.0, 0.0)
    pose_world[24] = lm(-0.12, 0.0, 0.0)

    # Synthetic open Left hand (21 points in MP world space: +X left, -Y up, -Z forward)
    h = [lm(0, 0, 0)] * 21
    h[0] = lm(0.0, 0.0, 0.0)
    # Thumb
    h[1], h[2], h[3], h[4] = lm(0.02, 0.0, -0.03), lm(0.04, 0.0, -0.05), lm(0.06, 0.0, -0.07), lm(0.08, 0.0, -0.08)
    # Index..Little spread slightly forward
    for base, z_off in ((5, -0.03), (9, 0.0), (13, 0.02), (17, 0.04)):
        h[base] = lm(0.08, 0.0, z_off)
        h[base + 1] = lm(0.11, 0.01, z_off)
        h[base + 2] = lm(0.13, 0.025, z_off)
        h[base + 3] = lm(0.15, 0.04, z_off)

    # 12° head roll + 10° yaw
    angle_y = np.radians(10.0)
    R_head = np.eye(4, dtype=np.float64)
    R_head[0, 0], R_head[0, 2] = np.cos(angle_y), np.sin(angle_y)
    R_head[2, 0], R_head[2, 2] = -np.sin(angle_y), np.cos(angle_y)

    face_res = SimpleNamespace(
        face_blendshapes=[[
            SimpleNamespace(category_name="eyeBlinkLeft", score=0.92),
            SimpleNamespace(category_name="jawOpen", score=0.65),
            SimpleNamespace(category_name="mouthSmileLeft", score=0.80),
            SimpleNamespace(category_name="mouthSmileRight", score=0.80),
        ]],
        facial_transformation_matrixes=[R_head],
    )
    pose_res = SimpleNamespace(pose_world_landmarks=[pose_world], pose_landmarks=[])
    hand_res = SimpleNamespace(
        hand_world_landmarks=[h],
        hand_landmarks=[],
        handedness=[[SimpleNamespace(category_name="Left")]],
    )

    solver = MediaPipeRigSolver()
    bones, blends = solver.solve_frame(
        t=0.1,
        face_result=face_res,
        pose_result=pose_res,
        hand_result=hand_res,
        enable_body=True,
        enable_hands=True,
    )
    dgram = build_vmc_bundle(bones, blends, 0.1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(dgram, (host, port))
    sock.close()

    print(
        f"[MotionService] Self-test OK: models=3 bones={len(bones)} blendshapes={len(blends)} "
        f"bytes={len(dgram)} target={host}:{port}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AlicePet Headless MediaPipe -> VMC/OSC Motion Tracker")
    parser.add_argument("--host", default="127.0.0.1", help="VMC target IP (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=39539, help="VMC target UDP port (default 39539)")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0)")
    parser.add_argument("--fps", type=int, default=30, help="Target capture FPS (default 30)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default 480)")
    parser.add_argument("--pose-model", choices=("lite", "full"), default="lite", help="Pose model tier")
    parser.add_argument("--mirror", type=int, default=1, help="1=mirror selfie (default), 0=unmirrored")
    parser.add_argument("--body", type=int, default=1, help="1=enable upper body tracking")
    parser.add_argument("--hands", type=int, default=1, help="1=enable 10-finger hand tracking")
    parser.add_argument("--models-dir", default=str(Path(__file__).resolve().parent / "models"))
    parser.add_argument("--self-test", action="store_true", help="Run headless verification and exit")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if args.self_test:
        return run_self_test(args.host, args.port, models_dir)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    face_model = models_dir / "face_landmarker.task"
    pose_model = models_dir / f"pose_landmarker_{args.pose_model}.task"
    hand_model = models_dir / "hand_landmarker.task"

    for required in (face_model, pose_model, hand_model):
        if not required.exists():
            print(f"[MotionService] ERROR: missing model file {required}", file=sys.stderr, flush=True)
            return 2

    face_lm = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(face_model)),
            running_mode=mp_vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
    )
    pose_lm = (
        mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
        )
        if args.body
        else None
    )
    hand_lm = (
        mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.45,
                min_hand_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
        )
        if args.hands
        else None
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[MotionService] ERROR: cannot open camera index {args.camera}", file=sys.stderr, flush=True)
        face_lm.close()
        if pose_lm:
            pose_lm.close()
        if hand_lm:
            hand_lm.close()
        return 3

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    solver = MediaPipeRigSolver()

    target_period = 1.0 / max(10, min(60, args.fps))
    start_mono = time.monotonic()
    last_ts_ms = -1
    frame_count = 0

    print(
        f"[MotionService] Started (pid={os.getpid()}, camera={args.camera}, target={args.host}:{args.port}, "
        f"body={bool(args.body)}, hands={bool(args.hands)}, mirror={bool(args.mirror)}, pose={args.pose_model})",
        flush=True,
    )

    try:
        while RUNNING:
            loop_start = time.monotonic()
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                time.sleep(0.01)
                continue

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            now_s = time.monotonic() - start_mono
            ts_ms = max(last_ts_ms + 1, int(now_s * 1000.0))
            last_ts_ms = ts_ms

            face_res = face_lm.detect_for_video(mp_image, ts_ms)
            pose_res = pose_lm.detect_for_video(mp_image, ts_ms) if pose_lm else None
            hand_res = hand_lm.detect_for_video(mp_image, ts_ms) if hand_lm else None

            bones, blends = solver.solve_frame(
                t=now_s,
                face_result=face_res,
                pose_result=pose_res,
                hand_result=hand_res,
                enable_body=bool(args.body),
                enable_hands=bool(args.hands),
            )

            if bones or blends:
                dgram = build_vmc_bundle(bones, blends, now_s)
                sock.sendto(dgram, (args.host, args.port))
                frame_count += 1

            elapsed = time.monotonic() - loop_start
            sleep_s = target_period - elapsed
            if sleep_s > 0.001:
                time.sleep(sleep_s)
    finally:
        cap.release()
        sock.close()
        face_lm.close()
        if pose_lm:
            pose_lm.close()
        if hand_lm:
            hand_lm.close()
        print(f"[MotionService] Stopped cleanly (frames_sent={frame_count})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
