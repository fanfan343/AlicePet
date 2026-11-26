#!/usr/bin/env python3
"""Top-down occupancy grid for a Picture-World room: floor, furniture footprint, reachable area."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from scene_geometry import CELL_M, GROUND_TOL, OPENNESS_RADIUS_M, RoomCloud

OBSTACLE_MIN_H = 0.16     # anything standing this high above the ground is furniture
OBSTACLE_MAX_H = 1.85     # above this: ceiling / lamps must not project down onto the floor
CLEARANCE_M = 0.09        # dilated obstacle margin the avatar's hips need


@dataclass
class WalkableGrid:
    """Top-down occupancy grid. Cell (i, j) covers x in [ox + i*cell, +cell),
    z in [oz + j*cell, +cell). `walkable[j, i]` is True where Alice may stand."""

    origin_x: float
    origin_z: float
    cell: float
    width: int
    height: int
    walkable: np.ndarray                                   # (h, w) bool
    floor: np.ndarray = field(repr=False, default=None)     # visible floor
    obstacle: np.ndarray = field(repr=False, default=None)  # furniture footprint
    # Fraction of standable floor within OPENNESS_RADIUS_M of each cell, convolved once so
    # direction probing is a table lookup instead of thousands of Python iterations.
    openness: np.ndarray = field(repr=False, default=None)
    # Where the avatar actually starts: the requested spot snapped onto real standing room.
    spawn: tuple[float, float] = (0.0, 0.0)

    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        return (int(math.floor((x - self.origin_x) / self.cell)),
                int(math.floor((z - self.origin_z) / self.cell)))

    def is_walkable(self, x: float, z: float) -> bool:
        i, j = self.world_to_cell(x, z)
        if i < 0 or j < 0 or i >= self.width or j >= self.height:
            return False
        return bool(self.walkable[j, i])

    def bounds(self) -> tuple[list[float], list[float]]:
        """Tight XZ bbox of the reachable floor."""
        if not self.walkable.any():
            return ([-1.2, -1.2], [1.2, 1.2])
        js, is_ = np.nonzero(self.walkable)
        xs = self.origin_x + (is_ + 0.5) * self.cell
        zs = self.origin_z + (js + 0.5) * self.cell
        return ([float(xs.min()), float(zs.min())], [float(xs.max()), float(zs.max())])

    def to_json(self, ground: np.ndarray | None = None, inset_m: float = 0.06) -> dict:
        """Serialises to compact ASCII rows (`rows[j][i] == '1'` means walkable)."""
        rows = ["".join("1" if v else "0" for v in self.walkable[j])
                for j in range(self.height)]
        (mn_x, mn_z), (mx_x, mx_z) = self.bounds()
        out = {
            "origin": [round(self.origin_x, 4), round(self.origin_z, 4)],
            "cellSize": round(self.cell, 4),
            "width": int(self.width),
            "height": int(self.height),
            "walkableCount": int(self.walkable.sum()),
            "boundsMin": [round(mn_x + inset_m, 3), round(mn_z + inset_m, 3)],
            "boundsMax": [round(mx_x - inset_m, 3), round(mx_z - inset_m, 3)],
            "rows": rows,
        }
        if ground is not None:
            out["groundSurface"] = [round(float(c), 5) for c in ground]
        return out


# --- Walkable grid --------------------------------------------------------------------

def _rasterize(points_xz: np.ndarray, origin_x: float, origin_z: float, cell: float,
               width: int, height: int) -> np.ndarray:
    """Counts points per grid cell."""
    if len(points_xz) == 0:
        return np.zeros((height, width), dtype=np.int32)
    ii = np.floor((points_xz[:, 0] - origin_x) / cell).astype(np.int64)
    jj = np.floor((points_xz[:, 1] - origin_z) / cell).astype(np.int64)
    ok = (ii >= 0) & (ii < width) & (jj >= 0) & (jj < height)
    flat = jj[ok] * width + ii[ok]
    return np.bincount(flat, minlength=width * height).astype(np.int32).reshape(height, width)


def _openness_map(walkable: np.ndarray, cell: float,
                  radius: float = OPENNESS_RADIUS_M) -> np.ndarray:
    """Fraction of standable floor inside a disc of `radius` around every cell."""
    r = max(1, int(round(radius / cell)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = ((xx * xx + yy * yy) <= r * r).astype(np.float32)
    return ndimage.convolve(walkable.astype(np.float32), kernel,
                            mode="constant", cval=0.0) / float(kernel.sum())


def build_walkable_grid(cloud: RoomCloud, spawn: np.ndarray | None = None,
                        cell: float = CELL_M) -> WalkableGrid:
    """Top-down floor / furniture footprint / reachable area from the point cloud.

    Floor = points within GROUND_TOL *of the fitted ground surface* (not of Y=0 — that is
    what used to shred illustrations into diagonal strips). Everything standing
    OBSTACLE_MIN_H..OBSTACLE_MAX_H above that surface projects straight down into a
    furniture footprint, so a table you can see under is still walked around, not through.
    """
    pts, h = cloud.pts, cloud.height
    floor_pts = pts[h <= GROUND_TOL]
    obstacle_pts = pts[(h >= OBSTACLE_MIN_H) & (h <= OBSTACLE_MAX_H)]

    if len(floor_pts) < 200:
        n = 41
        walkable = np.zeros((n, n), dtype=bool)
        walkable[n // 2 - 12:n // 2 + 13, n // 2 - 12:n // 2 + 13] = True
        return WalkableGrid(-1.2, -1.2, cell, n, n, walkable,
                            floor=walkable, obstacle=~walkable,
                            openness=_openness_map(walkable, cell))

    pad = 0.55
    min_x = float(floor_pts[:, 0].min()) - pad
    min_z = float(floor_pts[:, 2].min()) - pad
    max_x = min(float(floor_pts[:, 0].max()) + pad, min_x + 24.0)
    max_z = min(float(floor_pts[:, 2].max()) + pad, min_z + 24.0)
    width = max(8, int(math.ceil((max_x - min_x) / cell)))
    height = max(8, int(math.ceil((max_z - min_z) / cell)))

    floor = _rasterize(floor_pts[:, [0, 2]], min_x, min_z, cell, width, height) >= 2
    floor = ndimage.binary_closing(floor, structure=np.ones((3, 3)))

    # A single stray point must not declare a cell furniture: noisy depth speckles the
    # floor, and after the clearance dilation that shatters the room into a maze.
    obstacle = _rasterize(obstacle_pts[:, [0, 2]], min_x, min_z, cell, width, height) >= 3
    obstacle = ndimage.binary_opening(obstacle, structure=np.ones((3, 3)))
    obstacle = ndimage.binary_dilation(obstacle, structure=np.ones((3, 3)),
                                       iterations=max(1, int(round(CLEARANCE_M / cell))))

    free = _reachable_from(floor & ~obstacle, spawn, min_x, min_z, cell)
    openness = _openness_map(free, cell)
    return WalkableGrid(min_x, min_z, cell, width, height, free,
                        floor=floor, obstacle=obstacle, openness=openness,
                        spawn=_pick_standing_spot(free, openness, spawn,
                                                  min_x, min_z, cell))


def _reachable_from(free: np.ndarray, spawn: np.ndarray | None,
                    min_x: float, min_z: float, cell: float) -> np.ndarray:
    """Only the area connected to the spawn may become a goal — no islands behind sofas."""
    if not free.any():
        return free
    labels, _ = ndimage.label(free, structure=np.ones((3, 3)))
    si, sj = _cell_of(spawn, min_x, min_z, cell, free.shape)
    if free[sj, si]:
        return labels == labels[sj, si]
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _cell_of(point, min_x: float, min_z: float, cell: float,
             shape: tuple[int, int]) -> tuple[int, int]:
    p = np.zeros(3) if point is None else np.asarray(point, dtype=np.float64)
    i = int(np.clip(math.floor((p[0] - min_x) / cell), 0, shape[1] - 1))
    j = int(np.clip(math.floor((p[2] - min_z) / cell), 0, shape[0] - 1))
    return i, j


def _pick_standing_spot(free: np.ndarray, openness: np.ndarray, request,
                        min_x: float, min_z: float,
                        cell: float) -> tuple[float, float]:
    """Where Alice actually starts: the open floor nearest the requested spot.

    The requested point comes from the camera geometry (a floor point 2-4 m ahead) and can
    easily land under a coffee table. Starting there used to strand her inside furniture
    and, worse, disconnect the flood fill so the room collapsed into a narrow side corridor.
    """
    if not free.any():
        return (0.0, 0.0)
    ci, cj = _cell_of(request, min_x, min_z, cell, free.shape)
    js, is_ = np.nonzero(free)
    good = openness[js, is_] >= 0.45
    if not good.any():
        good = np.ones(len(is_), dtype=bool)
    d2 = (is_ - ci) ** 2 + (js - cj) ** 2 + 40.0 * (1.0 - openness[js, is_])
    k = int(np.argmin(np.where(good, d2, np.inf)))
    if not np.isfinite(d2[k]):
        k = int(np.argmin(d2))
    return (float(min_x + (is_[k] + 0.5) * cell), float(min_z + (js[k] + 0.5) * cell))

