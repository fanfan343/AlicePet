"""Low-latency 1-Euro filters and quaternion math for jitter-free VMC tracking."""

from __future__ import annotations

import math
import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
    return 1.0 / (1.0 + tau / max(dt, 1e-4))


class OneEuroScalar:
    """Adaptive low-pass filter: steady at rest, responsive during quick motion."""

    def __init__(
        self,
        min_cutoff: float = 1.5,
        beta: float = 0.25,
        d_cutoff: float = 1.0,
        deadzone: float = 0.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.deadzone = deadzone
        self._x: float | None = None
        self._dx: float = 0.0
        self._t: float | None = None

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._t = None

    def filter(self, value: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x = value
            self._dx = 0.0
            self._t = t
            return value

        dt = max(1e-4, min(t - self._t, 0.2))
        self._t = t

        if self.deadzone > 0.0 and abs(value - self._x) < self.deadzone:
            value = self._x

        dx_raw = (value - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx_raw + (1.0 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * value + (1.0 - a) * self._x
        return self._x


class OneEuroQuat:
    """1-Euro filter for unit quaternions (x, y, z, w) with antipodal sign alignment."""

    def __init__(
        self,
        min_cutoff: float = 1.6,
        beta: float = 0.35,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._q: np.ndarray | None = None
        self._dq: np.ndarray = np.zeros(4, dtype=np.float64)
        self._t: float | None = None

    def reset(self) -> None:
        self._q = None
        self._dq = np.zeros(4, dtype=np.float64)
        self._t = None

    def filter(self, q: np.ndarray, t: float) -> np.ndarray:
        q = quat_normalize(q)
        if self._q is None or self._t is None:
            self._q = q.copy()
            self._dq = np.zeros(4, dtype=np.float64)
            self._t = t
            return self._q

        # Keep quaternion on the same hemisphere as previous filtered value
        if float(np.dot(q, self._q)) < 0.0:
            q = -q

        dt = max(1e-4, min(t - self._t, 0.2))
        self._t = t

        dq_raw = (q - self._q) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dq = a_d * dq_raw + (1.0 - a_d) * self._dq

        speed = float(np.linalg.norm(self._dq))
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)
        blended = a * q + (1.0 - a) * self._q
        self._q = quat_normalize(blended)
        return self._q


# MARK: - Quaternion Utilities (xyzw convention)


def quat_identity() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return quat_identity()
    return q / n


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = max(0.0, min(1.0, float(t)))
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


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    n = float(np.linalg.norm(axis))
    if n < 1e-8 or abs(angle) < 1e-8:
        return quat_identity()
    u = axis / n
    half = 0.5 * angle
    s = math.sin(half)
    return np.array([u[0] * s, u[1] * s, u[2] * s, math.cos(half)], dtype=np.float64)


def quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    nf = float(np.linalg.norm(v_from))
    nt = float(np.linalg.norm(v_to))
    if nf < 1e-8 or nt < 1e-8:
        return quat_identity()
    u = v_from / nf
    v = v_to / nt
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if dot > 0.99999:
        return quat_identity()
    if dot < -0.99999:
        # 180° fallback perpendicular axis
        orth = np.cross(np.array([0.0, 0.0, 1.0]), u)
        if float(np.linalg.norm(orth)) < 1e-4:
            orth = np.cross(np.array([0.0, 1.0, 0.0]), u)
        return quat_from_axis_angle(orth, math.pi)
    c = np.cross(u, v)
    q = np.array([c[0], c[1], c[2], 1.0 + dot], dtype=np.float64)
    return quat_normalize(q)


def quat_from_rotmat(R: np.ndarray) -> np.ndarray:
    """Converts a 3x3 proper rotation matrix into a unit quaternion (x, y, z, w)."""
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([x, y, z, w], dtype=np.float64))


def rotmat_from_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return rotmat_from_quat(q) @ v


def vrm_to_unity_quat(q: np.ndarray) -> tuple[float, float, float, float]:
    """Converts a right-handed VRM 1.0 quaternion (+X Left, +Y Up, +Z Forward)
    into Unity VMC space so VRMMetalKit's VMCDriver(.flipX) recovers it exactly."""
    qn = quat_normalize(q)
    return (float(qn[0]), float(-qn[1]), float(-qn[2]), float(qn[3]))
