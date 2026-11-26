#!/usr/bin/env python3
"""Automatic room semantics for the AlicePet Picture-World (Route B).

Takes the metric point cloud that `estimate_scene.py` already reconstructs with MoGe-2 and
derives, with zero extra models and zero network access, everything the app needs so that an
*arbitrary* room photo/illustration becomes an interactive 3D room:

  1. `walkableGrid`  — top-down occupancy grid (A*-ready, 6 cm cells, flood-filled from spawn)
  2. `anchors`       — seat anchors: cushion plane height + backrest support + open approach
                       direction, expressed exactly like the hand-tuned `demo_room` anchors
  3. room bounds     — tight bbox of the reachable floor instead of a hard-coded +/-1.2 m box
  4. QC overlay      — top-down debug render so a bad detection is visible at a glance

All coordinates are world space of the scene bundle: Y up, Y=0 is the fitted floor plane,
world origin is the avatar spawn, +Z points toward the camera (out of the picture).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# --- Tunables (metres) ----------------------------------------------------------------
FLOOR_TOL = 0.10          # |y| below this counts as floor
OBSTACLE_MIN_Y = 0.16     # anything standing this high above the floor is furniture
OBSTACLE_MAX_Y = 1.85     # above this: ceiling / hanging lamps -> ignored, they must not
                          # project down onto the floor and block the whole room
CLEARANCE_M = 0.09        # dilated obstacle margin the avatar's hips need
CELL_M = 0.06             # walkable grid resolution
SEAT_MIN_Y = 0.26         # cushion height band
SEAT_MAX_Y = 0.68
SEAT_MIN_AREA = 0.06      # m^2, allows for partial occlusion
SEAT_MAX_AREA = 2.60      # m^2, rejects floor/rug planes caught in the band


@dataclass
class WalkableGrid:
    """Top-down occupancy grid. Cell (i, j) covers x in [ox + i*cell, +cell),
    z in [oz + j*cell, +cell). `walkable[j, i]` is True where Alice may stand."""

    origin_x: float
    origin_z: float
    cell: float
    width: int
    height: int
    walkable: np.ndarray            # (height, width) bool
    floor: np.ndarray = field(repr=False, default=None)      # visible floor
    obstacle: np.ndarray = field(repr=False, default=None)    # furniture footprint
    reachable: np.ndarray = field(repr=False, default=None)   # connected to spawn

    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        return (int(math.floor((x - self.origin_x) / self.cell)),
                int(math.floor((z - self.origin_z) / self.cell)))

    def cell_to_world(self, i: int, j: int) -> tuple[float, float]:
        return (self.origin_x + (i + 0.5) * self.cell,
                self.origin_z + (j + 0.5) * self.cell)

    def is_walkable(self, x: float, z: float) -> bool:
        i, j = self.world_to_cell(x, z)
        if i < 0 or j < 0 or i >= self.width or j >= self.height:
            return False
        return bool(self.walkable[j, i])

    def bounds(self) -> tuple[list[float], list[float]]:
        """Tight XZ bbox of the reachable floor, in the [min, max] shape scene.json wants."""
        if not self.walkable.any():
            return ([-1.2, -1.2], [1.2, 1.2])
        js, is_ = np.nonzero(self.walkable)
        xs = self.origin_x + (is_ + 0.5) * self.cell
        zs = self.origin_z + (js + 0.5) * self.cell
        return ([float(xs.min()), float(zs.min())], [float(xs.max()), float(zs.max())])

    def to_json(self, inset_m: float = 0.06) -> dict:
        """Serialises to compact ASCII rows (`rows[j][i] == '1'` means walkable)."""
        rows = []
        for j in range(self.height):
            rows.append("".join("1" if v else "0" for v in self.walkable[j]))
        (mn_x, mn_z), (mx_x, mx_z) = self.bounds()
        return {
            "origin": [round(self.origin_x, 4), round(self.origin_z, 4)],
            "cellSize": round(self.cell, 4),
            "width": int(self.width),
            "height": int(self.height),
            "walkableCount": int(self.walkable.sum()),
            "boundsMin": [round(mn_x + inset_m, 3), round(mn_z + inset_m, 3)],
            "boundsMax": [round(mx_x - inset_m, 3), round(mx_z - inset_m, 3)],
            "rows": rows,
        }


@dataclass
class SeatCandidate:
    x: float
    z: float
    height: float
    area: float
    extent_x: float
    extent_z: float
    score: float = 0.0
    facing_deg: int = 0
    facing: tuple[float, float] = (0.0, 1.0)
    approach: tuple[float, float] = (0.0, 0.0)
    source: str = "geometry"                 # "vision" | "geometry"
    furniture: str = "seat"                  # chair / couch / bench / seat
    detector_score: float = 0.0
    facing_hint: tuple[float, float] | None = None
    overhead: float = 0.0                    # geometry above the cushion (table pushed in)


SEAT_CLASSES = {
    "chair": "椅子",
    "couch": "沙发",
    "bench": "长凳",
    "armchair": "扶手椅",
}
SEAT_DETECTOR_WEIGHTS = "maskrcnn_resnet50_fpn_v2"
SEAT_DETECTOR_MIN_SCORE = 0.75


# --- Point cloud helpers ---------------------------------------------------------------

def camera_to_world(
    points_cam: np.ndarray,
    normals_cam: np.ndarray | None,
    spawn_cam: np.ndarray,
    R_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Maps camera-space points/normals into the scene bundle's world space.

    Matches `estimate_scene.build_world_camera`: `p_world = R_c2w @ (p_cam - spawn_cam)`,
    so with row vectors this is `(p_cam - spawn_cam) @ R_c2w.T`.
    """
    rt = R_c2w.T
    pts = (points_cam - spawn_cam) @ rt
    nrm = None if normals_cam is None else normals_cam @ rt
    return pts, nrm


def _rasterize(points_xz: np.ndarray, origin_x: float, origin_z: float, cell: float,
               width: int, height: int) -> np.ndarray:
    """Counts points per grid cell."""
    counts = np.zeros((height, width), dtype=np.int32)
    if len(points_xz) == 0:
        return counts
    ii = np.floor((points_xz[:, 0] - origin_x) / cell).astype(np.int64)
    jj = np.floor((points_xz[:, 1] - origin_z) / cell).astype(np.int64)
    ok = (ii >= 0) & (ii < width) & (jj >= 0) & (jj < height)
    flat = jj[ok] * width + ii[ok]
    np.add.at(counts.reshape(-1), flat, 1)
    return counts


def build_walkable_grid(
    points_world: np.ndarray,
    valid: np.ndarray,
    spawn_world: np.ndarray | None = None,
    cell: float = CELL_M,
) -> WalkableGrid:
    """Builds the A*-ready top-down walkable grid from the metric point cloud.

    Floor = points on the fitted plane. Everything standing 0.16..1.85 m above it is
    projected straight down into a furniture footprint (this is why a table you can see
    under is still treated as solid — Alice walks around it, not through it). Points above
    1.85 m (ceiling, lamps, upper walls) are ignored so they cannot blanket the room.
    The result is flood-filled from the spawn so unreachable pockets never become goals.
    """
    pts = points_world[valid]
    y = pts[:, 1]

    floor_pts = pts[np.abs(y) <= FLOOR_TOL]
    obstacle_pts = pts[(y >= OBSTACLE_MIN_Y) & (y <= OBSTACLE_MAX_Y)]
    if len(floor_pts) < 200:
        # No usable floor: fall back to a small safe box around the spawn.
        grid_n = 41
        walkable = np.zeros((grid_n, grid_n), dtype=bool)
        walkable[grid_n // 2 - 12:grid_n // 2 + 13, grid_n // 2 - 12:grid_n // 2 + 13] = True
        return WalkableGrid(-1.2, -1.2, cell, grid_n, grid_n, walkable,
                            floor=walkable, obstacle=~walkable, reachable=walkable)

    # Grid bounds: the visible floor, padded so nothing clips at the picture edge.
    pad = 0.55
    min_x = float(floor_pts[:, 0].min()) - pad
    max_x = float(floor_pts[:, 0].max()) + pad
    min_z = float(floor_pts[:, 2].min()) - pad
    max_z = float(floor_pts[:, 2].max()) + pad
    # Hard cap: a hallucinated far-away floor fragment must not create a 100 m grid.
    max_x = min(max_x, min_x + 24.0)
    max_z = min(max_z, min_z + 24.0)

    width = max(8, int(math.ceil((max_x - min_x) / cell)))
    height = max(8, int(math.ceil((max_z - min_z) / cell)))

    floor_counts = _rasterize(floor_pts[:, [0, 2]], min_x, min_z, cell, width, height)
    obstacle_counts = _rasterize(obstacle_pts[:, [0, 2]], min_x, min_z, cell, width, height)

    floor = floor_counts >= 2
    obstacle = obstacle_counts >= 1

    # Close the floor (fills pinholes from missing depth), then keep it tidy.
    floor = ndimage.binary_closing(floor, structure=np.ones((3, 3)), iterations=1)
    floor = ndimage.binary_opening(floor, structure=np.ones((2, 2)), iterations=1)

    # Furniture footprint + hip clearance.
    clearance_cells = max(1, int(round(CLEARANCE_M / cell)))
    obstacle_solid = ndimage.binary_dilation(
        obstacle, structure=np.ones((3, 3)), iterations=clearance_cells)

    free = floor & ~obstacle_solid

    # Flood fill from spawn: only the region Alice can actually reach may be a goal.
    spawn = np.zeros(3) if spawn_world is None else np.asarray(spawn_world, dtype=np.float64)
    si = int(math.floor((spawn[0] - min_x) / cell))
    sj = int(math.floor((spawn[2] - min_z) / cell))
    si = int(np.clip(si, 0, width - 1))
    sj = int(np.clip(sj, 0, height - 1))
    if not free[sj, si]:
        # Snap the seed to the closest free cell (spawn may land on a rug seam).
        fj, fi = np.nonzero(free)
        if len(fi) == 0:
            reachable = free
        else:
            d2 = (fi - si) ** 2 + (fj - sj) ** 2
            k = int(np.argmin(d2))
            si, sj = int(fi[k]), int(fj[k])
            reachable = None
    else:
        reachable = None

    if reachable is None:
        labels, _ = ndimage.label(free, structure=np.ones((3, 3)))
        seed_label = labels[sj, si]
        reachable = labels == seed_label

    return WalkableGrid(min_x, min_z, cell, width, height, reachable,
                        floor=floor, obstacle=obstacle_solid, reachable=reachable)


# --- Seat detection --------------------------------------------------------------------

def _cushion_from_points(
    pts: np.ndarray,
    nrm: np.ndarray,
    cell: float = CELL_M,
) -> SeatCandidate | None:
    """Extracts the cushion plane from the 3D points belonging to one piece of furniture.

    Height layers are scored by *floor area covered*, not by point count: a sofa 1.5 m from
    the camera has several times the point density of a chair 3.5 m away, and raw counts
    would always pick whichever seat is closest. Among the well-covered layers the lowest
    one wins, because the seat pad is always below the armrests and the backrest top.
    """
    if len(pts) < 60:
        return None
    up = nrm[:, 1] > 0.55
    y = pts[:, 1]
    band = up & (y >= SEAT_MIN_Y) & (y <= SEAT_MAX_Y)
    if int(band.sum()) < 40:
        return None

    bp = pts[band]
    ox, oz = float(bp[:, 0].min()), float(bp[:, 2].min())
    ii = ((bp[:, 0] - ox) / cell).astype(np.int64)
    jj = ((bp[:, 2] - oz) / cell).astype(np.int64)
    keys = ii * 1_000_003 + jj
    by = y[band]

    bin_h = 0.04
    edges = np.arange(SEAT_MIN_Y, SEAT_MAX_Y + bin_h * 0.5, bin_h)
    coverage = []
    for lo in edges[:-1]:
        sel = (by >= lo) & (by < lo + bin_h)
        coverage.append(int(np.unique(keys[sel]).size) if sel.any() else 0)
    if not coverage or max(coverage) < 12:
        return None

    peak = max(coverage)
    # Lowest layer that still covers most of the pad (skips armrests and backrest tops).
    k = next((i for i, c in enumerate(coverage) if c >= 0.55 * peak and c >= 28), None)
    if k is None:
        return None
    lo = float(edges[k]) - 0.012
    hi = float(edges[k + 1]) + 0.012
    if lo < 0.32:
        # Below this a "seat" is really a stretcher, a shelf, or a depth artefact; sitting
        # on it would drop the avatar a visible 10+ cm below the animation's rest height.
        return None
    core = band & (y >= lo) & (y <= hi)
    if int(core.sum()) < 20:
        return None

    cp = pts[core]
    ext_x = float(cp[:, 0].max() - cp[:, 0].min())
    ext_z = float(cp[:, 2].max() - cp[:, 2].min())
    if max(ext_x, ext_z) < 0.18:
        return None
    return _cell_centroid_candidate(cp, ox, oz, cell)


def _cell_centroid_candidate(cp: np.ndarray, ox: float, oz: float,
                             cell: float) -> SeatCandidate:
    """Area-weighted centroid of a surface patch.

    Collapsing each occupied 6 cm cell to one sample before averaging keeps the result
    unbiased: a sofa 1.5 m from the camera has several times the point density of the
    same surface 4 m away, and a raw point median would drag the estimated seat centre
    toward whichever part of the furniture happens to be closest.
    """
    ci = ((cp[:, 0] - ox) / cell).astype(np.int64)
    cj = ((cp[:, 2] - oz) / cell).astype(np.int64)
    pairs, inverse = np.unique(np.stack([ci, cj], axis=1), axis=0, return_inverse=True)
    n_cells = len(pairs)
    heights = np.zeros(n_cells)
    counts = np.bincount(inverse, minlength=n_cells).astype(np.float64)
    np.add.at(heights, inverse, cp[:, 1])
    heights /= np.maximum(counts, 1.0)
    return SeatCandidate(
        x=float(ox + (pairs[:, 0].mean() + 0.5) * cell),
        z=float(oz + (pairs[:, 1].mean() + 0.5) * cell),
        height=float(np.median(heights)),
        area=float(n_cells) * cell * cell,
        extent_x=float(cp[:, 0].max() - cp[:, 0].min()),
        extent_z=float(cp[:, 2].max() - cp[:, 2].min()),
    )


def _overhead_ratio(points_world: np.ndarray, valid: np.ndarray,
                    x: float, z: float, height: float) -> float:
    """Geometry sitting directly above a cushion — a tabletop means the chair is pushed in.

    Sitting there would push the avatar through the table, so this is used as a penalty.
    """
    pts = points_world[valid]
    near = ((pts[:, 0] - x) ** 2 + (pts[:, 2] - z) ** 2) < 0.18 * 0.18
    if int(near.sum()) < 20:
        return 0.0
    above = near & (pts[:, 1] >= height + 0.24) & (pts[:, 1] <= height + 0.80)
    pad = near & (np.abs(pts[:, 1] - height) <= 0.08)
    n_above = int(above.sum())
    n_pad = int(pad.sum())
    if n_above + n_pad < 20:
        return 0.0
    return n_above / float(n_above + n_pad)


def _facing_from_backrest(pts: np.ndarray, height: float) -> tuple[float, float] | None:
    """Facing = from the upright backrest toward the seat pad, in the XZ plane.

    Deriving it from where the tall geometry sits makes this independent of the chair's
    orientation in the room — a chair facing sideways or away from the camera still works.
    """
    if len(pts) < 80:
        return None
    tall = (pts[:, 1] >= height + 0.20) & (pts[:, 1] <= height + 0.80)
    if int(tall.sum()) < 40:
        return None
    bx = float(np.median(pts[tall, 0]))
    bz = float(np.median(pts[tall, 2]))
    pad = np.abs(pts[:, 1] - height) <= 0.08
    if int(pad.sum()) < 20:
        return None
    px = float(np.median(pts[pad, 0]))
    pz = float(np.median(pts[pad, 2]))
    vx, vz = px - bx, pz - bz
    n = math.hypot(vx, vz)
    if n < 0.06:
        return None
    return (vx / n, vz / n)


def detect_seats_vision(
    image_path,
    points_world: np.ndarray,
    normals_world: np.ndarray | None,
    valid: np.ndarray,
    cell: float = CELL_M,
    min_score: float = SEAT_DETECTOR_MIN_SCORE,
    max_instances: int = 12,
) -> list[SeatCandidate]:
    """Finds real chairs/sofas with a pretrained COCO instance segmenter, then measures them in 3D.

    The detector answers the question pure geometry cannot ("is this patch a seat or a
    fireplace ledge?"); the metric point cloud answers what the detector cannot ("how high
    is this cushion and which way does it face?"). Returns [] if torchvision or the
    weights are unavailable, so scene builds still work offline.
    """
    try:
        import torch
        import torchvision  # noqa: F401
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )
        from torchvision.transforms.functional import pil_to_tensor
        from PIL import Image
    except Exception as exc:
        print(f"[Semantics] Vision seat detector unavailable ({type(exc).__name__}), "
              f"using geometry only", flush=True)
        return []

    try:
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = maskrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
        categories = weights.meta["categories"]

        image = Image.open(image_path).convert("RGB")
        tensor = pil_to_tensor(image).float().to(device) / 255.0
        with torch.no_grad():
            pred = model([tensor])[0]
    except Exception as exc:
        print(f"[Semantics] Vision seat detector failed ({exc}), using geometry only", flush=True)
        return []

    if normals_world is None:
        return []
    img_w, img_h = image.size
    depth_h, depth_w = valid.shape
    if points_world.shape[:2] != valid.shape:
        return []

    normals_big = normals_world
    found: list[tuple[float, SeatCandidate]] = []
    for box, label, score, mask in zip(pred.get("boxes", []), pred["labels"],
                                       pred["scores"], pred.get("masks", [])):
        score = float(score)
        if score < min_score:
            continue
        name = categories[int(label)]
        if name not in SEAT_CLASSES:
            continue

        mask_np = (mask[0].detach().cpu().numpy() > 0.5)
        # Resample the instance mask onto the compact depth grid.
        ys = np.clip((np.arange(depth_h) + 0.5) * (mask_np.shape[0] / depth_h), 0,
                     mask_np.shape[0] - 1).astype(np.int32)
        xs = np.clip((np.arange(depth_w) + 0.5) * (mask_np.shape[1] / depth_w), 0,
                     mask_np.shape[1] - 1).astype(np.int32)
        mask_small = mask_np[np.ix_(ys, xs)] & valid
        # Erode one pixel so the silhouette boundary (which mixes in floor/wall depth) drops out.
        from scipy import ndimage as _ndi
        mask_small = _ndi.binary_erosion(mask_small, structure=np.ones((3, 3)), iterations=2)
        if int(mask_small.sum()) < 200:
            continue

        pts = points_world[mask_small]
        nrm = normals_big[mask_small]
        cand = _cushion_from_points(pts, nrm, cell)
        if cand is None:
            continue
        cand.source = "vision"
        cand.furniture = name
        cand.detector_score = score
        cand.facing_hint = _facing_from_backrest(pts, cand.height)
        cand.overhead = _overhead_ratio(points_world, valid, cand.x, cand.z, cand.height)
        found.append((score, cand, mask_small))

    # The same piece of furniture arrives twice: "couch" plus a "chair" for one section of
    # it. Drop any weaker instance whose mask is mostly inside a stronger one.
    found.sort(key=lambda t: t[0], reverse=True)
    kept: list[tuple[float, SeatCandidate, np.ndarray]] = []
    for score, cand, mask_small in found:
        m_area = float(mask_small.sum())
        duplicate = False
        for _, _, kmask in kept:
            overlap = float((mask_small & kmask).sum())
            if m_area > 0 and overlap / max(1.0, min(m_area, float(kmask.sum()))) > 0.55:
                duplicate = True
                break
        if not duplicate:
            kept.append((score, cand, mask_small))
    return [cand for _, cand, _ in kept][:max_instances]


def detect_seat_surfaces(
    points_world: np.ndarray,
    normals_world: np.ndarray | None,
    valid: np.ndarray,
    cell: float = CELL_M,
    height_bin: float = 0.05,
    max_candidates: int = 24,
) -> list[SeatCandidate]:
    """Finds cushion-shaped upward-facing planes in the seat height band.

    Pure geometry on purpose: it works the same on photos and on flat-shaded anime
    illustrations, where a pretrained detector has no idea what it is looking at.

    Connected components run *per 5 cm height layer*, not over the whole band: in a
    top-down projection a sofa at 0.39 m sits right next to a tabletop at 0.50 m, and
    merging them makes one oversized blob that has to be thrown away. Layering keeps
    every distinct surface its own patch; neighbouring layers of the same physical
    cushion are merged back together at the end.
    """
    pts = points_world[valid]
    y = pts[:, 1]
    band = (y >= SEAT_MIN_Y) & (y <= SEAT_MAX_Y)
    if normals_world is not None:
        band &= normals_world[valid][:, 1] > 0.55
    band_pts = pts[band]
    if len(band_pts) < 80:
        return []

    min_x = float(band_pts[:, 0].min())
    min_z = float(band_pts[:, 2].min())
    max_x = float(band_pts[:, 0].max())
    max_z = float(band_pts[:, 2].max())
    width = max(8, int(math.ceil((max_x - min_x) / cell)) + 1)
    height = max(8, int(math.ceil((max_z - min_z) / cell)) + 1)

    band_y = band_pts[:, 1]
    ii = np.clip(((band_pts[:, 0] - min_x) / cell).astype(np.int64), 0, width - 1)
    jj = np.clip(((band_pts[:, 2] - min_z) / cell).astype(np.int64), 0, height - 1)
    point_cell = jj * width + ii

    candidates: list[SeatCandidate] = []
    layer = SEAT_MIN_Y
    while layer < SEAT_MAX_Y - 1e-6:
        sel = (band_y >= layer) & (band_y < layer + height_bin)
        layer += height_bin
        if int(sel.sum()) < 40:
            continue

        counts = np.zeros(height * width, dtype=np.int32)
        np.add.at(counts, point_cell[sel], 1)
        labels, n_labels = ndimage.label(counts.reshape(height, width) > 0,
                                         structure=np.ones((3, 3)))
        if n_labels == 0:
            continue

        lab_of_pt = labels.reshape(-1)[point_cell]
        sel_idx = np.nonzero(sel)[0]
        for label_id in range(1, n_labels + 1):
            comp_mask = np.zeros(len(band_pts), dtype=bool)
            comp_mask[sel_idx] = lab_of_pt[sel_idx] == label_id
            n_pts = int(comp_mask.sum())
            if n_pts < 25:
                continue
            cp = band_pts[comp_mask]
            area = n_pts * cell * cell
            if area < SEAT_MIN_AREA or area > SEAT_MAX_AREA:
                continue

            ext_x = float(cp[:, 0].max() - cp[:, 0].min())
            ext_z = float(cp[:, 2].max() - cp[:, 2].min())
            if max(ext_x, ext_z) < 0.16 or min(ext_x, ext_z) < 0.07:
                continue
            # A seat is a compact pad; a big dining/coffee tabletop is not.
            if max(ext_x, ext_z) > 1.75 or min(ext_x, ext_z) > 1.05:
                continue

            candidates.append(SeatCandidate(
                x=float(np.median(cp[:, 0])), z=float(np.median(cp[:, 2])),
                height=float(np.median(cp[:, 1])), area=area,
                extent_x=ext_x, extent_z=ext_z,
            ))

    # Merge the same physical cushion found in two adjacent height layers.
    merged: list[SeatCandidate] = []
    for cand in sorted(candidates, key=lambda c: c.area, reverse=True):
        dup = next((m for m in merged
                    if math.hypot(m.x - cand.x, m.z - cand.z) < 0.30
                    and abs(m.height - cand.height) <= height_bin + 0.02), None)
        if dup is None:
            merged.append(cand)
        else:
            total = dup.area + cand.area
            dup.x = (dup.x * dup.area + cand.x * cand.area) / total
            dup.z = (dup.z * dup.area + cand.z * cand.area) / total
            dup.height = (dup.height * dup.area + cand.height * cand.area) / total
            dup.area = total
            dup.extent_x = max(dup.extent_x, cand.extent_x)
            dup.extent_z = max(dup.extent_z, cand.extent_z)
    for cand in merged:
        cand.overhead = _overhead_ratio(points_world, valid, cand.x, cand.z, cand.height)
    return merged[:max_candidates]


def _direction(deg: int) -> tuple[float, float]:
    rad = math.radians(deg)
    return (math.sin(rad), math.cos(rad))


APPROACH_MAX_M = 1.05
APPROACH_IDEAL_M = 0.55


def _local_openness(grid: WalkableGrid, x: float, z: float, radius: float = 0.45) -> float:
    """Fraction of the floor around a spot Alice could actually stand on.

    Judging this with a straight ray is too harsh in a furnished room: a sofa in a corner
    passes only if the walkway continues in a perfectly straight line, while in reality she
    just needs a small patch of open floor to stand and turn on.
    """
    hits = total = 0
    steps = np.arange(-radius, radius + 1e-6, grid.cell)
    for dz in steps:
        for dx in steps:
            if dx * dx + dz * dz > radius * radius:
                continue
            total += 1
            if grid.is_walkable(x + dx, z + dz):
                hits += 1
    return (hits / total) if total else 0.0


def _probe_front(grid: WalkableGrid, x: float, z: float, dx: float, dz: float,
                 step: float = 0.06) -> tuple[float | None, float]:
    """Walks outward from the cushion and reports (distance to first standable cell, openness).

    A fixed probe ring does not work in real rooms: a seat can sit fully outside the
    reconstructed floor region (a sofa whose foot area is blocked by a foreground table),
    so the first reachable cell along the ray wins, and what is scored is how much open
    floor she has around that standing spot.
    """
    for dist in np.arange(0.28, APPROACH_MAX_M + 1e-6, step):
        px, pz = x + dx * dist, z + dz * dist
        if grid.is_walkable(px, pz):
            return float(dist), _local_openness(grid, px, pz)
    return None, 0.0


def _approach_spot(grid: WalkableGrid, x: float, z: float, dx: float, dz: float) -> tuple[float, float]:
    """First standable cell in front of the cushion."""
    for dist in np.arange(0.28, APPROACH_MAX_M + 1e-6, 0.04):
        px, pz = x + dx * dist, z + dz * dist
        if grid.is_walkable(px, pz):
            return (px, pz)
    return (x + dx * APPROACH_IDEAL_M, z + dz * APPROACH_IDEAL_M)


def _backrest_ratio(points_world: np.ndarray, valid: np.ndarray,
                    x: float, z: float, dx: float, dz: float, height: float) -> float:
    """Fraction of the geometry right behind the cushion that stands upright.

    This is the single cue that separates "a chair/sofa you can sit in" from "a flat
    tabletop you must not sit on". It is a ratio, not a raw count, so it does not drift
    with distance from the camera or with depth-map point density.
    """
    bx, bz = x - dx * 0.20, z - dz * 0.20
    pts = points_world[valid]
    near = ((pts[:, 0] - bx) ** 2 + (pts[:, 2] - bz) ** 2) < 0.22 * 0.22
    back = near & (pts[:, 1] >= height + 0.12) & (pts[:, 1] <= height + 0.62)
    pad = near & (np.abs(pts[:, 1] - height) <= 0.10)
    n_back = int(back.sum())
    n_pad = int(pad.sum())
    if n_back + n_pad < 30:
        return 0.0
    return n_back / float(n_back + n_pad)


def solve_seat_anchors(
    candidates: list[SeatCandidate],
    grid: WalkableGrid,
    points_world: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    min_separation: float = 0.55,
) -> list[dict]:
    """Picks each cushion's outward facing + approach spot and emits scene.json anchors."""
    (room_x, room_z) = (lambda b: ((b[0][0] + b[1][0]) * 0.5, (b[0][1] + b[1][1]) * 0.5))(grid.bounds())

    anchors: list[dict] = []
    for cand in candidates:
        if cand.overhead > 0.40:
            continue
        # A detected chair already knows which way it faces (from its own backrest);
        # only refine that locally. Pure-geometry candidates must search all directions.
        if cand.facing_hint is not None:
            hint_deg = math.degrees(math.atan2(cand.facing_hint[0], cand.facing_hint[1]))
            probe_degs = [int(round(hint_deg + d)) % 360 for d in (-26, -13, 0, 13, 26)]
        else:
            probe_degs = list(range(0, 360, 15))

        best = None
        for deg in probe_degs:
            dx, dz = _direction(deg)
            reach, front = _probe_front(grid, cand.x, cand.z, dx, dz)
            if reach is None or front < (0.30 if cand.source == "vision" else 0.38):
                continue
            back = 0.0
            if points_world is not None and valid is not None:
                back = _backrest_ratio(points_world, valid, cand.x, cand.z, dx, dz, cand.height)
            # Prefer facing into the room (toward the walkable centre of mass).
            vx, vz = room_x - cand.x, room_z - cand.z
            vn = math.hypot(vx, vz)
            center = (dx * vx + dz * vz) / vn if vn > 1e-4 else 0.0
            back_norm = min(1.0, back / 0.32)
            score = 0.46 * front + 0.42 * back_norm + 0.12 * max(0.0, center)
            if cand.facing_hint is not None:
                # Reward staying close to the backrest-derived direction.
                off = abs(((deg - hint_deg + 180) % 360) - 180)
                score += 0.35 * max(0.0, 1.0 - off / 30.0)
            score += 0.25 * cand.detector_score
            # A chair pushed under a table must not become a sitting target.
            score -= 0.9 * min(1.0, cand.overhead / 0.20)
            # A long walk-up is worse than a seat you can simply step onto.
            score -= 0.30 * max(0.0, reach - 0.65)
            if best is None or score > best[0]:
                best = (score, deg, dx, dz, front, back, reach)

        if best is None:
            continue
        # No upright geometry behind it -> a table/ledge, not a seat.
        if best[5] < 0.10 and cand.source != "vision":
            continue
        score, deg, dx, dz, front, back, reach = best
        approach = _approach_spot(grid, cand.x, cand.z, dx, dz)

        # Slide the sitting spot toward the approach so she never shimmies more than ~0.5 m
        # onto the cushion (a deep sofa would otherwise drag her half a metre sideways).
        push = 0.12 + max(0.0, reach - 0.60)
        cand.score = score
        cand.facing_deg = deg
        cand.facing = (dx, dz)
        cand.approach = approach
        anchors.append({
            "_x": cand.x, "_z": cand.z, "_score": score,
            "kind": "seat",
            "_furniture": cand.furniture,
            "approachPosition": [round(approach[0], 3), 0.0, round(approach[1], 3)],
            "targetPosition": [round(cand.x + dx * push, 3), 0.0, round(cand.z + dz * push, 3)],
            "seatHeight": round(cand.height, 3),
            "facingYawRad": round(math.atan2(dx, dz), 3),
            "_front": front, "_back": back, "_area": cand.area,
            "_source": cand.source, "_reach": reach,
        })

    # Drop anchors that compete for the same seat, keeping the best-supported one.
    anchors.sort(key=lambda a: a["_score"], reverse=True)
    kept: list[dict] = []
    for a in anchors:
        if all(math.hypot(a["_x"] - k["_x"], a["_z"] - k["_z"]) >= min_separation for k in kept):
            kept.append(a)

    kept.sort(key=lambda a: (a["_z"], a["_x"]))
    for idx, a in enumerate(kept):
        base = SEAT_CLASSES.get(a.get("_furniture", ""), "座位")
        a["id"] = f"seat_{idx + 1}"
        a["label"] = f"{_seat_label(a['_x'], a['_z'], grid)}{base}"
    return kept


def _seat_label(x: float, z: float, grid: WalkableGrid) -> str:
    """Human-readable place in the room, meant to be read as "<place>的<furniture>"."""
    (mn_x, mn_z), (mx_x, mx_z) = grid.bounds()
    mid_x = (mn_x + mx_x) * 0.5
    mid_z = (mn_z + mx_z) * 0.5
    span_x = max(1e-3, mx_x - mn_x)
    span_z = max(1e-3, mx_z - mn_z)
    nx = (x - mid_x) / span_x
    nz = (z - mid_z) / span_z
    horizontal = "左侧" if nx < -0.12 else ("右侧" if nx > 0.12 else "")
    depth = "靠里" if nz < -0.12 else ("靠前" if nz > 0.12 else "")
    if not horizontal and not depth:
        return "房间中央"
    return f"{horizontal}{depth}"


def detect_seats(
    image_path,
    points_world: np.ndarray,
    normals_world: np.ndarray | None,
    valid: np.ndarray,
    mode: str = "auto",
    cell: float = CELL_M,
) -> tuple[list[SeatCandidate], str]:
    """Seat discovery: COCO instance segmentation when possible, geometry as the fallback.

    Returns `(candidates, source_label)`. An anime illustration usually yields no
    detections at all, which is exactly when the geometric pass takes over.
    """
    if mode in ("auto", "vision"):
        vision = detect_seats_vision(image_path, points_world, normals_world, valid, cell)
        if vision:
            return vision, "vision"
        if mode == "vision":
            return [], "vision"
    if mode == "off":
        return [], "off"
    return detect_seat_surfaces(points_world, normals_world, valid, cell), "geometry"


# --- QC overlay -------------------------------------------------------------------------

def project_world_to_pixel(
    point: tuple[float, float, float],
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    cam_up: np.ndarray,
    fov_y_deg: float,
    image_size: tuple[int, int],
) -> tuple[float, float] | None:
    """Projects a world point back onto the source image (pixel x, y). None if behind."""
    pos = np.asarray(cam_pos, dtype=np.float64)
    fwd = np.asarray(cam_target, dtype=np.float64) - pos
    fwd /= max(1e-9, float(np.linalg.norm(fwd)))
    right = np.cross(fwd, np.asarray(cam_up, dtype=np.float64))
    right /= max(1e-9, float(np.linalg.norm(right)))
    up = np.cross(right, fwd)

    rel = np.asarray(point, dtype=np.float64) - pos
    z = float(np.dot(rel, fwd))
    if z <= 0.02:
        return None
    img_w, img_h = image_size
    tan_y = math.tan(math.radians(fov_y_deg) * 0.5)
    tan_x = tan_y * (img_w / float(img_h))
    u = 0.5 + (float(np.dot(rel, right)) / z) / (2.0 * tan_x)
    v = 0.5 - (float(np.dot(rel, up)) / z) / (2.0 * tan_y)
    return (u * img_w, v * img_h)


def render_image_overlay(
    scene_cfg: dict,
    bg_path,
    candidates: list[SeatCandidate],
    anchors: list[dict],
    out_path,
) -> None:
    """Draws every detected surface and accepted anchor back onto the room photo."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(bg_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = None
    for candidate in ("/System/Library/Fonts/PingFang.ttc",
                      "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            font = ImageFont.truetype(candidate, 28)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    cam_pos = scene_cfg["cameraPosition"]
    cam_target = scene_cfg["cameraTarget"]
    cam_up = scene_cfg["cameraUp"]
    fov = scene_cfg["fovYDegrees"]
    size = (scene_cfg["imageWidth"], scene_cfg["imageHeight"])

    def proj(x, y, z):
        return project_world_to_pixel((x, y, z), cam_pos, cam_target, cam_up, fov, size)

    accepted = {(round(a["_x"], 2), round(a["_z"], 2)) for a in anchors}
    for cand in candidates:
        px = proj(cand.x, cand.height, cand.z)
        if px is None:
            continue
        is_anchor = (round(cand.x, 2), round(cand.z, 2)) in accepted or any(
            math.hypot(cand.x - a["_x"], cand.z - a["_z"]) < 0.05 for a in anchors)
        color = (255, 72, 64) if is_anchor else (90, 170, 255)
        r = 14 if is_anchor else 9
        draw.ellipse([px[0] - r, px[1] - r, px[0] + r, px[1] + r], outline=color, width=3)
        draw.text((px[0] + 12, px[1] - 30), f"{cand.height:.2f}m", fill=color, font=font)

    for a in anchors:
        tgt = proj(a["_x"], a["seatHeight"], a["_z"])
        app = proj(a["approachPosition"][0], 0.02, a["approachPosition"][2])
        if tgt and app:
            draw.line([tgt, app], fill=(255, 210, 90), width=4)
        if tgt:
            draw.text((tgt[0] - 10, tgt[1] + 14), a["id"], fill=(255, 210, 90), font=font)

    img.save(out_path)

def render_qc_overlay(
    grid: WalkableGrid,
    anchors: list[dict],
    out_path,
    scale: int = 7,
) -> None:
    """Top-down debug map: floor, furniture footprint, reachable region, anchors + facings."""
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for candidate in ("/System/Library/Fonts/PingFang.ttc",
                      "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            font = ImageFont.truetype(candidate, 13)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    width = grid.width * scale
    height = grid.height * scale
    img = Image.new("RGB", (width, height), (18, 20, 26))
    draw = ImageDraw.Draw(img)

    for j in range(grid.height):
        for i in range(grid.width):
            x0, y0 = i * scale, j * scale
            if grid.walkable[j, i]:
                color = (74, 148, 226)
            elif grid.floor is not None and grid.floor[j, i]:
                color = (58, 62, 74)
            elif grid.obstacle is not None and grid.obstacle[j, i]:
                color = (96, 62, 48)
            else:
                continue
            draw.rectangle([x0, y0, x0 + scale - 1, y0 + scale - 1], fill=color)

    def to_px(x: float, z: float) -> tuple[int, int]:
        return (int((x - grid.origin_x) / grid.cell * scale),
                int((z - grid.origin_z) / grid.cell * scale))

    # Spawn
    sx, sy = to_px(0.0, 0.0)
    draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], outline=(120, 240, 150), width=2)
    draw.text((sx + 8, sy - 7), "spawn", fill=(120, 240, 150), font=font)

    for a in anchors:
        ax, ay = to_px(a["_x"], a["_z"])
        px, py = to_px(a["approachPosition"][0], a["approachPosition"][2])
        dx, dz = math.sin(a["facingYawRad"]), math.cos(a["facingYawRad"])
        draw.line([ax, ay, ax + dx * 34, ay + dz * 34], fill=(255, 196, 92), width=2)
        draw.ellipse([ax - 6, ay - 6, ax + 6, ay + 6], fill=(232, 96, 88))
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(120, 240, 150))
        draw.text((ax + 9, ay + 4), a["label"], fill=(240, 244, 250), font=font)

    img.save(out_path)
