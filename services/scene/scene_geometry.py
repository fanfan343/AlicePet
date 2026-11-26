#!/usr/bin/env python3
"""Room geometry for the AlicePet Picture-World: point cloud, ground surface, spatial index.

The one idea everything else builds on: the reconstructed floor is *not* the Y=0 plane.
Depth from a flat-shaded illustration drifts 15-30 cm across a room, so every "is this the
floor / how high is this cushion" question is answered relative to a fitted ground surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

GROUND_FIT_BAND = 0.80    # points this close to the plane seed the ground-surface fit
GROUND_TOL = 0.10         # a point this close above local ground IS floor
CELL_M = 0.06             # walkable grid resolution
OPENNESS_RADIUS_M = 0.45  # standing room measured around a candidate approach spot


# --- Data model -----------------------------------------------------------------------

@dataclass
class RoomCloud:
    """The scene's metric point cloud, indexed and measured against its own floor."""

    pts: np.ndarray                     # (N, 3) world position, valid points only
    normals: np.ndarray                 # (N, 3) world normals
    height: np.ndarray                  # (N,) height above the *local* ground
    ground: np.ndarray                  # (6,) quadratic ground coefficients
    valid: np.ndarray = field(repr=False, default=None)   # (H, W) source pixel mask
    index: "PointIndex" = field(repr=False, default=None)

    @classmethod
    def build(cls, points_world: np.ndarray, normals_world: np.ndarray | None,
              valid: np.ndarray) -> "RoomCloud":
        pts = points_world[valid].astype(np.float64)
        normals = (normals_world[valid].astype(np.float64) if normals_world is not None
                   else np.tile(np.array([0.0, 1.0, 0.0]), (len(pts), 1)))
        ground = _fit_ground_surface(pts, normals)
        height = pts[:, 1] - _eval_ground(ground, pts[:, 0], pts[:, 2])
        return cls(pts=pts, normals=normals, height=height, ground=ground,
                   valid=valid, index=PointIndex(pts))

    def pixel_patch(self, mask_np: np.ndarray, erode: int = 2) -> np.ndarray:
        """Instance mask at image resolution -> boolean selector in this cloud's point order."""
        if mask_np.shape == self.valid.shape:
            small = mask_np & self.valid          # native depth: no resampling needed at all
        else:
            h, w = self.valid.shape
            ys = np.clip((np.arange(h) + 0.5) * (mask_np.shape[0] / h), 0,
                         mask_np.shape[0] - 1).astype(np.int32)
            xs = np.clip((np.arange(w) + 0.5) * (mask_np.shape[1] / w), 0,
                         mask_np.shape[1] - 1).astype(np.int32)
            small = mask_np[np.ix_(ys, xs)] & self.valid
        if erode:
            small = ndimage.binary_erosion(small, structure=np.ones((3, 3)), iterations=erode)
        return small[self.valid]

    def ground_at(self, x: float, z: float) -> float:
        return float(_eval_ground(self.ground, np.array([x]), np.array([z]))[0])

    def subset(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.pts[mask], self.normals[mask], self.height[mask]


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

    # Subsample: the fit is smooth, and 3M points buy nothing over ~200k.
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

    # A wild fit is worse than no fit: fall back to the plane if it cannot explain the floor.
    resid = yy - _eval_ground(coef, x, z)
    if float(np.sqrt(np.mean(resid ** 2))) > 0.35:
        A = np.c_[np.ones(len(x)), x, z]
        plane, *_ = np.linalg.lstsq(A, yy, rcond=None)
        coef = np.array([plane[0], plane[1], plane[2], 0.0, 0.0, 0.0])
    return coef



class PointIndex:
    """Uniform XZ bucket index for local 3D queries.

    The seat scorer asks 'what stands behind / above this pad?' dozens of times per
    candidate. Scanning the full cloud each time meant copying ~80 MB per call; bucketing
    once and touching only the few hundred nearby points makes that free.
    """

    CELL = 0.25

    def __init__(self, pts: np.ndarray) -> None:
        self.pts = pts
        if len(pts) == 0:
            self.ox = self.oz = 0.0
            self.cols = 1
            self.starts = self.counts = np.zeros(0, dtype=np.int64)
            self.slot: dict[int, int] = {}
            return
        self.ox = float(pts[:, 0].min())
        self.oz = float(pts[:, 2].min())
        ii = ((pts[:, 0] - self.ox) / self.CELL).astype(np.int64)
        jj = ((pts[:, 2] - self.oz) / self.CELL).astype(np.int64)
        self.cols = int(ii.max()) + 2
        key = jj * self.cols + ii
        order = np.argsort(key, kind="stable")
        self.pts = pts[order]
        uniq, starts, counts = np.unique(key[order], return_index=True, return_counts=True)
        self.starts = starts.astype(np.int64)
        self.counts = counts.astype(np.int64)
        self.slot = {int(k): i for i, k in enumerate(uniq)}

    def query(self, x: float, z: float, radius: float) -> np.ndarray:
        """All points whose XZ position lies inside the given disc."""
        if len(self.pts) == 0:
            return self.pts
        ci = int((x - self.ox) // self.CELL)
        cj = int((z - self.oz) // self.CELL)
        span = max(1, int(math.ceil(radius / self.CELL)))
        chunks = []
        for j in range(cj - span, cj + span + 1):
            for i in range(ci - span, ci + span + 1):
                slot = self.slot.get(j * self.cols + i) if j >= 0 and i >= 0 else None
                if slot is None:
                    continue
                s = int(self.starts[slot])
                chunks.append(self.pts[s:s + int(self.counts[slot])])
        if not chunks:
            return self.pts[:0]
        near = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        d2 = (near[:, 0] - x) ** 2 + (near[:, 2] - z) ** 2
        return near[d2 <= radius * radius]

