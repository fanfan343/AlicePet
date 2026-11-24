"""MediaPipe Face (52 ARKit) + Pose (Upper Body) + Hands (10 Fingers) -> VMC bone/blendshape solver.

All internal math operates in right-handed VRM 1.0 model space:
    +X = Avatar Left (viewer's right when facing the avatar)
    +Y = Up
    +Z = Forward (toward camera)
Before emitting VMC packets, local bone quaternions are converted to Unity VMC space
via `vrm_to_unity_quat` (flipping Y/Z signs), which `VRMMetalKit.VMCDriver(.flipX)`
maps back 1-to-1 with zero handedness ambiguity.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from filters import (
    OneEuroQuat,
    OneEuroScalar,
    quat_from_axis_angle,
    quat_from_rotmat,
    quat_from_two_vectors,
    quat_identity,
    quat_inv,
    quat_mul,
    quat_normalize,
    quat_rotate_vec,
    quat_slerp,
    rotmat_from_quat,
    vrm_to_unity_quat,
)

# MediaPipe Pose Landmark indices
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24

# Snappy ARKit keys (eyelids & jaw react faster than brows/cheeks)
FAST_BLENDSHAPE_KEYS = {
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
}

FINGER_CHAINS = [
    ("Index", (5, 6, 7, 8)),
    ("Middle", (9, 10, 11, 12)),
    ("Ring", (13, 14, 15, 16)),
    ("Little", (17, 18, 19, 20)),
]

# Natural resting A-pose arms when wrists/elbows go off-screen below the desk
REST_LEFT_UPPER_ARM = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), -math.radians(66.0))
REST_RIGHT_UPPER_ARM = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(66.0))
REST_LEFT_LOWER_ARM = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), math.radians(10.0))
REST_RIGHT_LOWER_ARM = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), -math.radians(10.0))


def _mp_world_to_vrm(lm: Any) -> np.ndarray:
    """Converts MediaPipe world landmark (+X right, +Y down, +Z into screen)
    to right-handed VRM model space (+X right/Avatar Left, +Y up, +Z toward camera)."""
    return np.array([float(lm.x), -float(lm.y), -float(lm.z)], dtype=np.float64)


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return fallback.copy()
    return v / n


def _clamp_angle_quat(q: np.ndarray, max_rad: float) -> np.ndarray:
    q = quat_normalize(q)
    w = max(-1.0, min(1.0, float(q[3])))
    if w < 0.0:
        q = -q
        w = -w
    angle = 2.0 * math.acos(w)
    if angle <= max_rad or angle < 1e-6:
        return q
    axis = q[:3] / math.sin(0.5 * angle)
    return quat_from_axis_angle(axis, max_rad)


def _euler_yxz_from_rotmat(R: np.ndarray) -> tuple[float, float, float]:
    """Extracts (yaw_y, pitch_x, roll_z) in radians from a right-handed 3x3 rotation matrix."""
    pitch = math.asin(max(-1.0, min(1.0, -float(R[1, 2]))))
    yaw = math.atan2(float(R[0, 2]), float(R[2, 2]))
    roll = math.atan2(float(R[1, 0]), float(R[1, 1]))
    return yaw, pitch, roll


def _quat_from_yxz(yaw: float, pitch: float, roll: float) -> np.ndarray:
    qy = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), yaw)
    qx = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), pitch)
    qz = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), roll)
    return quat_normalize(quat_mul(quat_mul(qy, qx), qz))


class MediaPipeRigSolver:
    """Converts MediaPipe Face / Pose / Hand outputs into smoothed VMC bone quaternions and 52 ARKit blendshapes."""

    def __init__(self) -> None:
        self._bone_filters: dict[str, OneEuroQuat] = {}
        self._blend_filters: dict[str, OneEuroScalar] = {}
        self._neutral_head_inv: np.ndarray | None = None
        self._last_world_bones: dict[str, np.ndarray] = {}

    def reset_calibration(self) -> None:
        self._neutral_head_inv = None

    def _filter_bone(self, name: str, q_vrm: np.ndarray, t: float, is_finger: bool = False) -> np.ndarray:
        flt = self._bone_filters.get(name)
        if flt is None:
            if is_finger:
                flt = OneEuroQuat(min_cutoff=2.4, beta=0.55, d_cutoff=1.2)
            elif name in ("Spine", "Chest"):
                flt = OneEuroQuat(min_cutoff=1.1, beta=0.22, d_cutoff=1.0)
            elif name in ("Head", "Neck"):
                flt = OneEuroQuat(min_cutoff=1.8, beta=0.40, d_cutoff=1.0)
            else:
                flt = OneEuroQuat(min_cutoff=1.5, beta=0.35, d_cutoff=1.0)
            self._bone_filters[name] = flt
        return flt.filter(q_vrm, t)

    def _filter_blend(self, name: str, value: float, t: float) -> float:
        flt = self._blend_filters.get(name)
        if flt is None:
            if name in FAST_BLENDSHAPE_KEYS:
                flt = OneEuroScalar(min_cutoff=2.6, beta=0.55, d_cutoff=1.2, deadzone=0.008)
            else:
                flt = OneEuroScalar(min_cutoff=1.5, beta=0.25, d_cutoff=1.0, deadzone=0.012)
            self._blend_filters[name] = flt
        return max(0.0, min(1.0, flt.filter(value, t)))

    def solve_frame(
        self,
        t: float,
        face_result: Any | None = None,
        pose_result: Any | None = None,
        hand_result: Any | None = None,
        enable_body: bool = True,
        enable_hands: bool = True,
    ) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, float]]:
        """Returns `(unity_bones_xyzw, arkit_blendshapes)` ready for VMC OSC transmission."""
        local_vrm: dict[str, np.ndarray] = {}
        world_vrm: dict[str, np.ndarray] = {
            "Hips": quat_identity(),
            "Spine": quat_identity(),
            "Chest": quat_identity(),
            "Neck": quat_identity(),
            "Head": quat_identity(),
        }

        # 1. Upper Body Pose (Spine / Chest / Left & Right Arms)
        pose_2d = None
        if enable_body and pose_result is not None and pose_result.pose_world_landmarks:
            world_lms = pose_result.pose_world_landmarks[0]
            pose_2d = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
            self._solve_upper_body(world_lms, t, local_vrm, world_vrm)

        # 2. Face 52 ARKit BlendShapes + Head/Neck 3D Rotation
        blendshapes: dict[str, float] = {}
        if face_result is not None:
            if face_result.face_blendshapes:
                self._solve_blendshapes(face_result.face_blendshapes[0], t, blendshapes)
            if face_result.facial_transformation_matrixes:
                self._solve_head(face_result.facial_transformation_matrixes[0], t, local_vrm, world_vrm)

        # 3. Left & Right Hands + 10 Fingers (30 phalanges)
        if enable_hands and hand_result is not None and hand_result.hand_world_landmarks:
            self._solve_hands(hand_result, pose_2d, t, local_vrm, world_vrm)

        self._last_world_bones = world_vrm
        unity_bones = {name: vrm_to_unity_quat(q) for name, q in local_vrm.items()}
        return unity_bones, blendshapes

    # MARK: - Face 52 BlendShapes & Head Matrix

    def _solve_blendshapes(self, categories: Any, t: float, out: dict[str, float]) -> None:
        for cat in categories:
            name = cat.category_name
            if not name or name == "_neutral":
                continue
            raw = float(cat.score)
            # Gentle blink boost so full physical blink cleanly reaches 1.0
            if name in ("eyeBlinkLeft", "eyeBlinkRight"):
                raw = min(1.0, max(0.0, (raw - 0.04) * 1.28))
            elif name == "jawOpen":
                raw = min(1.0, max(0.0, (raw - 0.03) * 1.15))
            out[name] = self._filter_blend(name, raw, t)

    def _solve_head(
        self,
        matrix_4x4: np.ndarray,
        t: float,
        local_vrm: dict[str, np.ndarray],
        world_vrm: dict[str, np.ndarray],
    ) -> None:
        R = np.asarray(matrix_4x4[:3, :3], dtype=np.float64)
        # Orthonormalize via SVD to guard against numerical scale drift
        u, _, vt = np.linalg.svd(R)
        R = u @ vt
        if np.linalg.det(R) < 0:
            u[:, -1] *= -1
            R = u @ vt

        q_raw = quat_from_rotmat(R)
        if self._neutral_head_inv is None:
            # Softly capture initial pitch bias (~5° laptop camera angle) while keeping yaw/roll centered
            _, init_pitch, _ = _euler_yxz_from_rotmat(R)
            bias_pitch = max(-0.28, min(0.28, init_pitch))
            self._neutral_head_inv = quat_inv(
                quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), bias_pitch)
            )

        q_calibrated = quat_mul(self._neutral_head_inv, q_raw)
        yaw, pitch, roll = _euler_yxz_from_rotmat(rotmat_from_quat(q_calibrated))
        yaw = max(-math.radians(52.0), min(math.radians(52.0), yaw))
        pitch = max(-math.radians(38.0), min(math.radians(38.0), pitch))
        roll = max(-math.radians(34.0), min(math.radians(34.0), roll))
        q_head_world = _quat_from_yxz(yaw, pitch, roll)

        q_chest_world = world_vrm.get("Chest", quat_identity())
        q_rel = quat_mul(quat_inv(q_chest_world), q_head_world)
        q_rel = _clamp_angle_quat(q_rel, math.radians(65.0))

        q_neck_local = self._filter_bone("Neck", quat_slerp(quat_identity(), q_rel, 0.38), t)
        q_head_local = self._filter_bone("Head", quat_slerp(quat_identity(), q_rel, 0.62), t)

        local_vrm["Neck"] = q_neck_local
        local_vrm["Head"] = q_head_local
        world_vrm["Neck"] = quat_normalize(quat_mul(q_chest_world, q_neck_local))
        world_vrm["Head"] = quat_normalize(quat_mul(world_vrm["Neck"], q_head_local))

    # MARK: - Upper Body (Spine, Chest, Arms)

    def _solve_upper_body(
        self,
        lms: Any,
        t: float,
        local_vrm: dict[str, np.ndarray],
        world_vrm: dict[str, np.ndarray],
    ) -> None:
        ls = lms[MP_LEFT_SHOULDER]
        rs = lms[MP_RIGHT_SHOULDER]
        if getattr(ls, "visibility", 1.0) < 0.45 or getattr(rs, "visibility", 1.0) < 0.45:
            return

        p_ls = _mp_world_to_vrm(ls)
        p_rs = _mp_world_to_vrm(rs)
        shoulder_mid = 0.5 * (p_ls + p_rs)

        # Spine / Chest orientation from shoulder line + hip midpoint (virtual hip fallback if seated close)
        lh = lms[MP_LEFT_HIP]
        rh = lms[MP_RIGHT_HIP]
        if getattr(lh, "visibility", 0.0) >= 0.45 and getattr(rh, "visibility", 0.0) >= 0.45:
            hip_mid = 0.5 * (_mp_world_to_vrm(lh) + _mp_world_to_vrm(rh))
            raw_up = _normalize(shoulder_mid - hip_mid, np.array([0.0, 1.0, 0.0]))
        else:
            raw_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # Dampen raw torso lean so seated desk posture looks graceful
        torso_up = _normalize(0.55 * raw_up + 0.45 * np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        shoulder_left = _normalize(p_ls - p_rs, np.array([1.0, 0.0, 0.0]))
        # Dampen noisy monocular Z shoulder yaw
        shoulder_left[2] *= 0.65
        shoulder_left = _normalize(shoulder_left, np.array([1.0, 0.0, 0.0]))

        torso_fwd = _normalize(np.cross(shoulder_left, torso_up), np.array([0.0, 0.0, 1.0]))
        torso_left = _normalize(np.cross(torso_up, torso_fwd), np.array([1.0, 0.0, 0.0]))
        R_torso = np.column_stack((torso_left, torso_up, torso_fwd))
        q_torso = _clamp_angle_quat(quat_from_rotmat(R_torso), math.radians(26.0))

        q_spine_local = self._filter_bone("Spine", quat_slerp(quat_identity(), q_torso, 0.45), t)
        q_chest_local = self._filter_bone("Chest", quat_slerp(quat_identity(), q_torso, 0.55), t)
        local_vrm["Spine"] = q_spine_local
        local_vrm["Chest"] = q_chest_local
        world_vrm["Spine"] = q_spine_local
        world_vrm["Chest"] = quat_normalize(quat_mul(q_spine_local, q_chest_local))

        # Left & Right Arms
        self._solve_arm(
            side="Left",
            p_shoulder=p_ls,
            lm_elbow=lms[MP_LEFT_ELBOW],
            lm_wrist=lms[MP_LEFT_WRIST],
            q_chest_world=world_vrm["Chest"],
            t=t,
            local_vrm=local_vrm,
            world_vrm=world_vrm,
        )
        self._solve_arm(
            side="Right",
            p_shoulder=p_rs,
            lm_elbow=lms[MP_RIGHT_ELBOW],
            lm_wrist=lms[MP_RIGHT_WRIST],
            q_chest_world=world_vrm["Chest"],
            t=t,
            local_vrm=local_vrm,
            world_vrm=world_vrm,
        )

    def _solve_arm(
        self,
        side: str,
        p_shoulder: np.ndarray,
        lm_elbow: Any,
        lm_wrist: Any,
        q_chest_world: np.ndarray,
        t: float,
        local_vrm: dict[str, np.ndarray],
        world_vrm: dict[str, np.ndarray],
    ) -> None:
        is_left = side == "Left"
        rest_dir = np.array([1.0 if is_left else -1.0, 0.0, 0.0], dtype=np.float64)
        upper_name = f"{side}UpperArm"
        lower_name = f"{side}LowerArm"

        elbow_vis = getattr(lm_elbow, "visibility", 1.0)
        wrist_vis = getattr(lm_wrist, "visibility", 1.0)

        if elbow_vis < 0.50:
            # Arm out of frame: glide gracefully to relaxed A-pose
            q_upper_local = self._filter_bone(
                upper_name,
                REST_LEFT_UPPER_ARM if is_left else REST_RIGHT_UPPER_ARM,
                t,
            )
            q_lower_local = self._filter_bone(
                lower_name,
                REST_LEFT_LOWER_ARM if is_left else REST_RIGHT_LOWER_ARM,
                t,
            )
            local_vrm[upper_name] = q_upper_local
            local_vrm[lower_name] = q_lower_local
            world_vrm[upper_name] = quat_normalize(quat_mul(q_chest_world, q_upper_local))
            world_vrm[lower_name] = quat_normalize(quat_mul(world_vrm[upper_name], q_lower_local))
            return

        p_elbow = _mp_world_to_vrm(lm_elbow)
        u_dir = p_elbow - p_shoulder
        # Slightly dampen noisy monocular Z depth on upper arm
        u_dir[2] *= 0.78
        u_dir = _normalize(u_dir, rest_dir)

        # Compute minimal swing from chest-aligned rest arm direction to observed upper arm direction
        chest_rest_dir = quat_rotate_vec(q_chest_world, rest_dir)
        q_upper_swing = quat_from_two_vectors(chest_rest_dir, u_dir)
        q_upper_world = quat_normalize(quat_mul(q_upper_swing, q_chest_world))

        q_upper_local = quat_mul(quat_inv(q_chest_world), q_upper_world)
        q_upper_local = self._filter_bone(upper_name, q_upper_local, t)
        q_upper_world = quat_normalize(quat_mul(q_chest_world, q_upper_local))

        if wrist_vis >= 0.45:
            p_wrist = _mp_world_to_vrm(lm_wrist)
            f_dir = p_wrist - p_elbow
            f_dir[2] *= 0.80
            f_dir = _normalize(f_dir, u_dir)

            # Rotate lower arm relative to upper arm via shortest geodesic arc
            upper_fwd_dir = quat_rotate_vec(q_upper_world, rest_dir)
            q_lower_swing = quat_from_two_vectors(upper_fwd_dir, f_dir)
            q_lower_world = quat_normalize(quat_mul(q_lower_swing, q_upper_world))
            q_lower_local = quat_mul(quat_inv(q_upper_world), q_lower_world)
            q_lower_local = _clamp_angle_quat(q_lower_local, math.radians(145.0))
        else:
            q_lower_local = REST_LEFT_LOWER_ARM if is_left else REST_RIGHT_LOWER_ARM

        q_lower_local = self._filter_bone(lower_name, q_lower_local, t)
        q_lower_world = quat_normalize(quat_mul(q_upper_world, q_lower_local))

        local_vrm[upper_name] = q_upper_local
        local_vrm[lower_name] = q_lower_local
        world_vrm[upper_name] = q_upper_world
        world_vrm[lower_name] = q_lower_world

    # MARK: - Hands & 10 Fingers (30 Phalanges)

    def _solve_hands(
        self,
        hand_result: Any,
        pose_2d: Any | None,
        t: float,
        local_vrm: dict[str, np.ndarray],
        world_vrm: dict[str, np.ndarray],
    ) -> None:
        # Match each detected hand to "Left" or "Right" using Pose wrist proximity when available
        assigned: dict[str, int] = {}
        lw_2d = pose_2d[MP_LEFT_WRIST] if pose_2d is not None else None
        rw_2d = pose_2d[MP_RIGHT_WRIST] if pose_2d is not None else None
        have_pose_wrists = (
            lw_2d is not None
            and rw_2d is not None
            and getattr(lw_2d, "visibility", 0.0) >= 0.45
            and getattr(rw_2d, "visibility", 0.0) >= 0.45
        )

        for idx, world_lms in enumerate(hand_result.hand_world_landmarks):
            side: str | None = None
            if have_pose_wrists and idx < len(hand_result.hand_landmarks):
                hw = hand_result.hand_landmarks[idx][0]
                d_left = math.hypot(hw.x - lw_2d.x, hw.y - lw_2d.y)
                d_right = math.hypot(hw.x - rw_2d.x, hw.y - rw_2d.y)
                if min(d_left, d_right) < 0.22:
                    side = "Left" if d_left <= d_right else "Right"
            if side is None and idx < len(hand_result.handedness) and hand_result.handedness[idx]:
                label = hand_result.handedness[idx][0].category_name
                # MediaPipe handedness assumes mirrored selfie frame: "Left" -> person's Left (viewer's right)
                side = "Left" if label == "Left" else "Right"
            if side is not None and side not in assigned:
                assigned[side] = idx

        for side, idx in assigned.items():
            pts = [_mp_world_to_vrm(lm) for lm in hand_result.hand_world_landmarks[idx]]
            self._solve_single_hand(side, pts, t, local_vrm, world_vrm)

    def _solve_single_hand(
        self,
        side: str,
        pts: list[np.ndarray],
        t: float,
        local_vrm: dict[str, np.ndarray],
        world_vrm: dict[str, np.ndarray],
    ) -> None:
        is_left = side == "Left"
        s = 1.0 if is_left else -1.0  # +1 for Left (+X fingers), -1 for Right (-X fingers)

        # Build right-handed palm frame matching T-pose (fingers along s*X, back of hand +Y, thumb side +Z)
        hand_dir = _normalize(pts[9] - pts[0], np.array([s, 0.0, 0.0]))
        across = _normalize(pts[5] - pts[17], np.array([0.0, 0.0, 1.0]))
        if is_left:
            up = _normalize(np.cross(across, hand_dir), np.array([0.0, 1.0, 0.0]))
            across = _normalize(np.cross(hand_dir, up), np.array([0.0, 0.0, 1.0]))
            R_hand = np.column_stack((hand_dir, up, across))
        else:
            up = _normalize(np.cross(hand_dir, across), np.array([0.0, 1.0, 0.0]))
            across = _normalize(np.cross(up, hand_dir), np.array([0.0, 0.0, 1.0]))
            R_hand = np.column_stack((-hand_dir, up, across))

        q_hand_world = quat_from_rotmat(R_hand)
        lower_name = f"{side}LowerArm"
        q_lower_world = world_vrm.get(lower_name, quat_identity())

        # Local wrist rotation relative to forearm (only if forearm is tracked)
        if lower_name in world_vrm:
            q_hand_local = _clamp_angle_quat(
                quat_mul(quat_inv(q_lower_world), q_hand_world),
                math.radians(78.0),
            )
            q_hand_local = self._filter_bone(f"{side}Hand", q_hand_local, t, is_finger=True)
            local_vrm[f"{side}Hand"] = q_hand_local
            world_vrm[f"{side}Hand"] = quat_normalize(quat_mul(q_lower_world, q_hand_local))

        # Transform all 21 hand landmarks into the hand's local frame for clean anatomical finger angles
        R_hand_inv = R_hand.T
        origin = pts[0]
        local_pts = [R_hand_inv @ (p - origin) for p in pts]

        # 4 main fingers: Index, Middle, Ring, Little
        for fname, (i0, i1, i2, i3) in FINGER_CHAINS:
            v0 = _normalize(local_pts[i1] - local_pts[i0], np.array([s, 0.0, 0.0]))
            v1 = _normalize(local_pts[i2] - local_pts[i1], np.array([s, 0.0, 0.0]))
            v2 = _normalize(local_pts[i3] - local_pts[i2], np.array([s, 0.0, 0.0]))

            # Proximal curl toward palm (-Y) + mild splay (spread along Z)
            proj_x = max(1e-4, s * float(v0[0]))
            prox_curl = math.atan2(-float(v0[1]), proj_x)
            prox_curl = max(-math.radians(15.0), min(math.radians(92.0), prox_curl))
            prox_splay = math.atan2(float(v0[2]), proj_x)
            prox_splay = max(-math.radians(18.0), min(math.radians(18.0), prox_splay))

            # Curl toward -Y is rotation by (-s * curl) around +Z
            q_prox = quat_mul(
                quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), -s * prox_splay),
                quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), -s * prox_curl),
            )

            # Intermediate & Distal hinge curls from consecutive bone segment angles
            inter_curl = math.acos(max(-1.0, min(1.0, float(np.dot(v0, v1)))))
            inter_curl = max(0.0, min(math.radians(110.0), inter_curl))
            q_inter = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), -s * inter_curl)

            dist_curl = math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)))))
            dist_curl = max(0.0, min(math.radians(90.0), dist_curl))
            q_dist = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), -s * dist_curl)

            local_vrm[f"{side}{fname}Proximal"] = self._filter_bone(f"{side}{fname}Proximal", q_prox, t, is_finger=True)
            local_vrm[f"{side}{fname}Intermediate"] = self._filter_bone(f"{side}{fname}Intermediate", q_inter, t, is_finger=True)
            local_vrm[f"{side}{fname}Distal"] = self._filter_bone(f"{side}{fname}Distal", q_dist, t, is_finger=True)

        # Thumb (Unity HumanBodyBones names: Proximal/Intermediate/Distal -> auto-shifted by VMCDriver for VRM 1.0)
        t0 = _normalize(local_pts[2] - local_pts[1], np.array([s * 0.7, 0.0, 0.7]))
        t1 = _normalize(local_pts[3] - local_pts[2], np.array([s * 0.7, 0.0, 0.7]))
        t2 = _normalize(local_pts[4] - local_pts[3], np.array([s * 0.7, 0.0, 0.7]))

        thumb_opp = max(-math.radians(25.0), min(math.radians(50.0), math.atan2(-float(t0[1]), max(1e-4, float(t0[2])))))
        thumb_flex1 = max(0.0, min(math.radians(65.0), math.acos(max(-1.0, min(1.0, float(np.dot(t0, t1)))))))
        thumb_flex2 = max(0.0, min(math.radians(75.0), math.acos(max(-1.0, min(1.0, float(np.dot(t1, t2)))))))

        q_t0 = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), s * thumb_opp * 0.65)
        q_t1 = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), s * thumb_flex1 * 0.85)
        q_t2 = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), s * thumb_flex2 * 0.85)

        local_vrm[f"{side}ThumbProximal"] = self._filter_bone(f"{side}ThumbProximal", q_t0, t, is_finger=True)
        local_vrm[f"{side}ThumbIntermediate"] = self._filter_bone(f"{side}ThumbIntermediate", q_t1, t, is_finger=True)
        local_vrm[f"{side}ThumbDistal"] = self._filter_bone(f"{side}ThumbDistal", q_t2, t, is_finger=True)
