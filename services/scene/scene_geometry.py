#!/usr/bin/env python3
"""Room geometry for the AlicePet Picture-World: point cloud and fitted ground surface.

The reconstructed floor from a single image is not an exact Y=0 plane: monocular depth
on flat-shaded illustrations drifts 15-30 cm across a room. Fitting a smooth quadratic
ground surface lets walkability and avatar grounding stay relative to the actual floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

GROUND_FIT_BAND = 0.80    # points this close to the plane seed the ground-surface fit
GROUND_TOL = 0.10         # a point this close above local ground IS floor
CELL_M = 0.06             # walkable grid resolution
OPENNESS_RADIUS_M = 0.45  # standing room measured around a candidate spawn spot


@dataclass
class RoomCloud:
    """The scene's metric point cloud measured against its own fitted floor."""

    pts: np.ndarray        # (N, 3) world position, valid points only
    normals: np.ndarray    # (N, 3) world normals
    height: np.ndarray     # (N,) height above the local ground
    ground: np.ndarray     # (6,) quadratic ground coefficients

    @classmethod
    def build(cls, points_world: np.ndarray, normals_world: np.ndarray | None,
              valid: np.ndarray) -> "RoomCloud":
        pts = points_world[valid].astype(np.float64)
        normals = (normals_world[valid].astype(np.float64) if normals_world is not None
                   else np.tile(np.array([0.0, 1.0, 0.0]), (len(pts), 1)))
        ground = _fit_ground_surface(pts, normals)
        height = pts[:, 1] - _eval_ground(ground, pts[:, 0], pts[:, 2])
        return cls(pts=pts, normals=normals, height=height, ground=ground)

    def ground_at(self, x: float, z: float) -> float:
        return float(_eval_ground(self.ground, np.array([x]), np.array([z]))[0])


def camera_to_world(points_cam: np.ndarray, normals_cam: np.ndarray | None,
                    spawn_cam: np.ndarray, R_c2w: np.ndarray):
    """Camera space -> scene world space, matching `estimate_scene.build_world_camera`:
    `p_world = R_c2w @ (p_cam - spawn_cam)`, i.e. `(p_cam - spawn_cam) @ R_c2w.T` on rows."""
    rt = R_c2w.T
    return (points_cam - spawn_cam) @ rt, (
        None if normals_cam is None else normals_cam @ rt)


def _eval_ground(coef: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """y = c0 + cx.x + cz.z + cxx.x^2 + cxz.x.z + czz.z^2 (quadratic, so tilt and curl fit)."""
    return (coef[0] + coef[1] * x + coef[2] * z
            + coef[3] * x * x + coef[4] * x * z + coef[5] * z * z)


def _fit_ground_surface(pts: np.ndarray, normals: np.ndarray,
                        iterations: int = 3) -> np.ndarray:
    """Robust quadratic fit of the visible floor.

    The initial guess is the Y=0 plane from `estimate_scene`; each round keeps only points
    that look like floor (upward-facing and near the current surface) and refits, so table
    tops and low shelves drop out after the first pass instead of dragging the fit up.
    """
    y = pts[:, 1]
    up = normals[:, 1] > 0.5
    seed = up & (np.abs(y) <= GROUND_FIT_BAND)
    if int(seed.sum()) < 500:
        return np.zeros(6)

    idx = np.nonzero(seed)[0]
    if len(idx) > 200_000:
        idx = idx[np.random.default_rng(7).choice(len(idx), 200_000, replace=False)]
    x, z, yy = pts[idx, 0], pts[idx, 2], y[idx]

    coef = np.zeros(6)
    for _ in range(iterations):
        resid = yy - _eval_ground(coef, x, z)
        keep = np.abs(resid) <= max(0.06, 2.0 * float(np.std(resid)))
        if int(keep.sum()) < 500:
            break
        A = np.c_[np.ones(int(keep.sum())), x[keep], z[keep],
                  x[keep] ** 2, x[keep] * z[keep], z[keep] ** 2]
        coef, *_ = np.linalg.lstsq(A, yy[keep], rcond=None)

    resid = yy - _eval_ground(coef, x, z)
    if float(np.sqrt(np.mean(resid ** 2))) > 0.35:
        A = np.c_[np.ones(len(x)), x, z]
        plane, *_ = np.linalg.lstsq(A, yy, rcond=None)
        coef = np.array([plane[0], plane[1], plane[2], 0.0, 0.0, 0.0])
    return coef


def refine_depth_edges(
    depth: np.ndarray,
    rgb: np.ndarray,
    grad_rel_thresh: float = 0.04,
    radius: int = 8,
    band_dilate: int = 4,
) -> np.ndarray:
    """Snaps blurry monocular depth discontinuities onto crisp RGB object silhouettes.

    Smooth interior surfaces stay bit-exact untouched. On narrow depth-jump bands, each pixel
    adopts the depth of the clean neighbor with the closest Lab color (removing flying-pixel
    halos around leaves, chairs, and props), while high-gradient outline pixels whose color
    matches neither side prefer the nearer occluder.
    """
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

    k3 = np.ones((3, 3), np.uint8)
    d_min = cv2.erode(depth, k3)
    rel_jump = (cv2.dilate(depth, k3) - d_min) / np.maximum(d_min, 0.1)
    edge_mask = rel_jump > grad_rel_thresh

    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_dilate + 1, 2 * band_dilate + 1))
    uncertain = cv2.dilate(edge_mask.astype(np.uint8), kd) > 0
    clean = ~uncertain
    n_unc = int(uncertain.sum())
    if n_unc == 0:
        return depth

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    is_outline = np.hypot(gx, gy) > 80.0

    uy, ux = np.nonzero(uncertain)
    target_lab = lab[uy, ux]
    target_outline = is_outline[uy, ux]

    best_cost = np.full(n_unc, np.inf, dtype=np.float32)
    best_depth = depth[uy, ux].copy()
    near_depth = np.full(n_unc, np.inf, dtype=np.float32)

    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r2 = dx * dx + dy * dy
            if r2 == 0 or r2 > radius * radius:
                continue
            if r2 > 16 and ((dx & 1) != 0 or (dy & 1) != 0):
                continue
            offsets.append((dy, dx, float(r2)))

    for dy, dx, r2 in offsets:
        ny = np.clip(uy + dy, 0, h - 1)
        nx = np.clip(ux + dx, 0, w - 1)
        valid_n = clean[ny, nx]
        if not valid_n.any():
            continue
        nd = depth[ny, nx]
        diff = lab[ny, nx] - target_lab
        cost = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2] + 0.25 * r2
        better = valid_n & (cost < best_cost)
        best_cost[better] = cost[better]
        best_depth[better] = nd[better]
        if r2 <= 9.0:
            near_better = valid_n & (nd < near_depth)
            near_depth[near_better] = nd[near_better]

    outline_fix = target_outline & (best_cost > 400.0) & np.isfinite(near_depth)
    best_depth[outline_fix] = near_depth[outline_fix]

    out = depth.copy()
    out[uy, ux] = best_depth
    return out
