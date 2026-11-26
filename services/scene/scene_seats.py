#!/usr/bin/env python3
"""Seat discovery and anchor solving for a Picture-World room.

The detector answers what geometry cannot ("is this patch a seat or a fireplace ledge?");
the metric point cloud answers what the detector cannot ("how high is this cushion, which
way does it face, and where can Alice stand to sit down?").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from scene_geometry import CELL_M, GROUND_TOL, PointIndex, RoomCloud
from scene_grid import WalkableGrid


SEAT_MIN_H = 0.26         # any horizontal surface in this band could belong to a chair
SEAT_MAX_H = 0.68
SEAT_PAD_MIN_H = 0.34     # where a sit-able pad actually lives (human furniture standard)
SEAT_PAD_MAX_H = 0.56
SEAT_MIN_AREA = 0.06      # m^2, tolerates partial occlusion
SEAT_MAX_AREA = 2.60      # m^2, rejects floor/rug planes caught in the band

SEAT_CLASSES = {"chair": "椅子", "couch": "沙发", "bench": "长凳", "armchair": "扶手椅"}
SEAT_DETECTOR_MIN_SCORE = 0.75

APPROACH_MAX_M = 1.05
APPROACH_IDEAL_M = 0.55


@dataclass
class SeatCandidate:
    """A sit-able pad as measured in 3D. `height` is absolute world Y (what the app needs
    to place the avatar's root); `rel_height` is above the local ground (what detection
    reasons about)."""

    x: float
    z: float
    height: float
    rel_height: float
    area: float
    extent_x: float
    extent_z: float
    source: str = "geometry"                 # "vision" | "geometry"
    furniture: str = "seat"
    detector_score: float = 0.0
    facing_hint: tuple[float, float] | None = None
    overhead: float = 0.0                    # upright geometry above the pad (table pushed in)
    score: float = 0.0


# --- Seat detection -------------------------------------------------------------------

def _cell_centroid_candidate(cp: np.ndarray, ch: np.ndarray, cell: float) -> SeatCandidate:
    """Area-weighted centroid of a surface patch.

    Collapsing each occupied cell to one sample before averaging keeps the result unbiased:
    a sofa 1.5 m from the camera has several times the point density of the same surface
    4 m away, and a raw point median would drag the centre toward the nearest corner.
    """
    ox, oz = float(cp[:, 0].min()), float(cp[:, 2].min())
    ci = ((cp[:, 0] - ox) / cell).astype(np.int64)
    cj = ((cp[:, 2] - oz) / cell).astype(np.int64)
    pairs, inverse = np.unique(np.stack([ci, cj], axis=1), axis=0, return_inverse=True)
    heights = np.bincount(inverse, weights=ch, minlength=len(pairs)) / np.maximum(
        np.bincount(inverse, minlength=len(pairs)), 1)
    cx = float(ox + (pairs[:, 0].mean() + 0.5) * cell)
    cz = float(oz + (pairs[:, 1].mean() + 0.5) * cell)
    return SeatCandidate(
        x=cx, z=cz,
        height=float(np.median(ch)) + 0.0,   # caller re-bases to world Y
        rel_height=float(np.median(heights)),
        area=float(len(pairs)) * cell * cell,
        extent_x=float(cp[:, 0].max() - cp[:, 0].min()),
        extent_z=float(cp[:, 2].max() - cp[:, 2].min()),
    )


def _cushion_from_patch(pts: np.ndarray, normals: np.ndarray, height: np.ndarray,
                        cell: float = CELL_M) -> SeatCandidate | None:
    """Finds the seat pad inside the 3D points of one piece of furniture.

    Height layers are scored by *area covered*, not point count, and the winner is the
    best-covered layer inside SEAT_PAD_MIN/MAX_H rather than the lowest substantial one:
    on a drawn chair the footrail and the shaded underside of the seat also produce real
    horizontal coverage, and 'lowest plausible' was landing 12 cm below the pad.
    """
    if len(pts) < 60:
        return None
    band = (normals[:, 1] > 0.55) & (height >= SEAT_MIN_H) & (height <= SEAT_MAX_H)
    if int(band.sum()) < 40:
        return None

    bp, bh = pts[band], height[band]
    ox, oz = float(bp[:, 0].min()), float(bp[:, 2].min())
    keys = (((bp[:, 0] - ox) / cell).astype(np.int64) * 1_000_003
            + ((bp[:, 2] - oz) / cell).astype(np.int64))

    step = 0.04
    edges = np.arange(SEAT_MIN_H, SEAT_MAX_H + step * 0.5, step)
    coverage = np.zeros(len(edges) - 1, dtype=np.int64)
    for k, lo in enumerate(edges[:-1]):
        sel = (bh >= lo) & (bh < lo + step)
        if sel.any():
            coverage[k] = np.unique(keys[sel]).size

    lo_k = max(0, int(np.searchsorted(edges, SEAT_PAD_MIN_H)) - 1)
    hi_k = min(len(coverage), int(np.searchsorted(edges, SEAT_PAD_MAX_H)))
    if hi_k <= lo_k or int(coverage[lo_k:hi_k].max(initial=0)) < 25:
        return None
    k = lo_k + int(np.argmax(coverage[lo_k:hi_k]))

    core = band & (height >= float(edges[k]) - 0.012) & (height <= float(edges[k + 1]) + 0.012)
    if int(core.sum()) < 20:
        return None
    cp, ch = pts[core], height[core]
    if max(float(cp[:, 0].max() - cp[:, 0].min()),
           float(cp[:, 2].max() - cp[:, 2].min())) < 0.18:
        return None
    return _cell_centroid_candidate(cp, ch, cell)


def _pad_from_height_band(pts: np.ndarray, height: np.ndarray) -> SeatCandidate | None:
    """Fallback pad for a detected chair whose seat surface is not visible.

    A café chair seen from behind or from the side shows only a sliver of seat: there is no
    horizontal patch to find, and insisting on one loses every chair in the room. The
    detector has already told us this *is* a chair, so the body of the chair between 30 and
    55 cm is where a person sits — take its footprint and its median height.
    """
    sel = (height >= 0.30) & (height <= 0.55)
    if int(sel.sum()) < 120:
        return None
    cp, ch = pts[sel], height[sel]
    ext_x = float(cp[:, 0].max() - cp[:, 0].min())
    ext_z = float(cp[:, 2].max() - cp[:, 2].min())
    if max(ext_x, ext_z) > 1.4 or max(ext_x, ext_z) < 0.20:
        return None
    return SeatCandidate(
        x=float((cp[:, 0].min() + cp[:, 0].max()) * 0.5),
        z=float((cp[:, 2].min() + cp[:, 2].max()) * 0.5),
        height=0.0, rel_height=float(np.median(ch)),
        area=float(max(ext_x, 0.05) * max(ext_z, 0.05)),
        extent_x=ext_x, extent_z=ext_z,
    )


def _facing_from_backrest(pts: np.ndarray, height: np.ndarray,
                          pad_h: float) -> tuple[float, float] | None:
    """Facing = from the upright backrest toward the seat pad, in the XZ plane.

    Deriving the direction from where the tall geometry sits keeps this independent of the
    chair's orientation in the room — one facing sideways still works.
    """
    if len(pts) < 80:
        return None
    tall = (height >= pad_h + 0.20) & (height <= pad_h + 0.80)
    pad = np.abs(height - pad_h) <= 0.08
    if int(tall.sum()) < 40 or int(pad.sum()) < 20:
        return None
    vx = float(np.median(pts[pad, 0]) - np.median(pts[tall, 0]))
    vz = float(np.median(pts[pad, 2]) - np.median(pts[tall, 2]))
    n = math.hypot(vx, vz)
    return None if n < 0.06 else (vx / n, vz / n)


def _overhead_ratio(cloud: RoomCloud, x: float, z: float, rel_height: float) -> float:
    """Geometry at lap height just above a pad — reported, not scored.

    Every chair that belongs to a desk or a table has that table's edge 10-30 cm above the
    cushion; that is normal and sitable, so this number cannot decide anything. What
    actually separates "chair you can use" from "chair buried under a table" is whether
    there is standable floor in front of it, which the walkable grid answers.
    """
    near = cloud.index.query(x, z, 0.18)
    if len(near) < 20:
        return 0.0
    h = near[:, 1] - cloud.ground_at(x, z)
    above = int(((h >= rel_height + 0.02) & (h <= rel_height + 0.30)).sum())
    pad = int((np.abs(h - rel_height) <= 0.08).sum())
    return above / float(above + pad) if above + pad >= 20 else 0.0


def detect_seats_vision(image_path, cloud: RoomCloud, cell: float = CELL_M,
                        min_score: float = SEAT_DETECTOR_MIN_SCORE,
                        max_instances: int = 12) -> list[SeatCandidate]:
    """Finds real chairs/sofas with a pretrained COCO instance segmenter, then measures them.

    The detector answers what geometry alone cannot ("is this patch a seat, or a fireplace
    ledge / cabinet top?"); the metric point cloud answers what the detector cannot ("how
    high is this cushion and which way does it face"). Returns [] when torchvision or the
    weights are unavailable, so scene builds still work offline.
    """
    try:
        import torch
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2)
        from torchvision.transforms.functional import pil_to_tensor
        from PIL import Image
    except Exception as exc:
        print(f"[Semantics] Vision seat detector unavailable ({type(exc).__name__}); "
              f"using geometry only", flush=True)
        return []

    try:
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = maskrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
        # torchvision upscales the short side to 800 px by default. Furniture is large and
        # unambiguous, so 512 px finds the same sofas ~26% faster (measured 1518 -> 1128 ms
        # on M1, identical detections across all test rooms).
        model.transform.min_size = (512,)
        model.transform.max_size = 900
        categories = weights.meta["categories"]
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            pred = model([pil_to_tensor(image).float().to(device) / 255.0])[0]
    except Exception as exc:
        print(f"[Semantics] Vision seat detector failed ({exc}); using geometry only",
              flush=True)
        return []

    found: list[tuple[float, SeatCandidate, np.ndarray]] = []
    for label, score, mask in zip(pred["labels"], pred["scores"], pred.get("masks", [])):
        score = float(score)
        if score < min_score:
            continue
        name = categories[int(label)]
        if name not in SEAT_CLASSES:
            continue

        patch = cloud.pixel_patch(mask[0].detach().cpu().numpy() > 0.5)
        if int(patch.sum()) < 200:
            continue

        pts, nrm, h = cloud.subset(patch)
        cand = _cushion_from_patch(pts, nrm, h, cell)
        if cand is not None and cand.area < 0.15:
            cand = None            # a 0.1 m² "pad" is a table edge or a backrest, not a seat
        if cand is None:
            cand = _pad_from_height_band(pts, h)
            if cand is None:
                continue
            score *= 0.8           # estimated pad ranks below a measured one
        cand.source = "vision"
        cand.furniture = name
        cand.detector_score = score
        cand.height = cloud.ground_at(cand.x, cand.z) + cand.rel_height   # world Y for the app
        cand.facing_hint = _facing_from_backrest(pts, h, cand.rel_height)
        cand.overhead = _overhead_ratio(cloud, cand.x, cand.z, cand.rel_height)
        found.append((score, cand, patch))

    # Same furniture detected twice ("couch" plus a "chair" for one of its sections).
    found.sort(key=lambda t: t[0], reverse=True)
    kept: list[tuple[float, SeatCandidate, np.ndarray]] = []
    for score, cand, patch in found:
        area = float(patch.sum())
        if any(float((patch & k_patch).sum()) / max(1.0, min(area, float(k_patch.sum()))) > 0.55
               for _, _, k_patch in kept):
            continue
        kept.append((score, cand, patch))
    return [cand for _, cand, _ in kept][:max_instances]


def detect_seats_geometry(cloud: RoomCloud, cell: float = CELL_M,
                          height_bin: float = 0.05,
                          max_candidates: int = 24) -> list[SeatCandidate]:
    """Model-free fallback: upward-facing pads in the seat band, no semantic class.

    Used for drawings the COCO detector does not recognise, and whenever torchvision is
    unavailable. Connected components run *per 5 cm height layer*: in a top-down
    projection a sofa pad at 0.39 m sits right next to a tabletop at 0.50 m, and merging
    them makes one oversized blob that has to be thrown away.
    """
    band_sel = ((cloud.normals[:, 1] > 0.55)
                & (cloud.height >= SEAT_MIN_H) & (cloud.height <= SEAT_MAX_H))
    bp, bh = cloud.pts[band_sel], cloud.height[band_sel]
    if len(bp) < 80:
        return []

    min_x, min_z = float(bp[:, 0].min()), float(bp[:, 2].min())
    width = max(8, int(math.ceil((bp[:, 0].max() - min_x) / cell)) + 1)
    height_n = max(8, int(math.ceil((bp[:, 2].max() - min_z) / cell)) + 1)
    point_cell = (np.clip(((bp[:, 2] - min_z) / cell).astype(np.int64), 0, height_n - 1) * width
                  + np.clip(((bp[:, 0] - min_x) / cell).astype(np.int64), 0, width - 1))

    candidates: list[SeatCandidate] = []
    layer = SEAT_MIN_H
    while layer < SEAT_MAX_H - 1e-6:
        sel = (bh >= layer) & (bh < layer + height_bin)
        layer += height_bin
        if int(sel.sum()) < 40:
            continue
        counts = np.bincount(point_cell[sel], minlength=width * height_n)
        labels, n_labels = ndimage.label(counts.reshape(height_n, width) > 0,
                                         structure=np.ones((3, 3)))
        if n_labels == 0:
            continue

        sel_idx = np.nonzero(sel)[0]
        point_label = labels.reshape(-1)[point_cell[sel_idx]]
        per_label = np.bincount(point_label, minlength=n_labels + 1)
        for label_id in range(1, n_labels + 1):
            if per_label[label_id] < 25:
                continue
            cp_cells = sel_idx[point_label == label_id]
            cp, ch = bp[cp_cells], bh[cp_cells]
            area = float(per_label[label_id]) * cell * cell
            ext_x = float(cp[:, 0].max() - cp[:, 0].min())
            ext_z = float(cp[:, 2].max() - cp[:, 2].min())
            # A seat is a compact pad: too small is a shelf, too large is a tabletop.
            if not (SEAT_MIN_AREA <= area <= SEAT_MAX_AREA):
                continue
            if max(ext_x, ext_z) < 0.16 or min(ext_x, ext_z) < 0.07:
                continue
            if max(ext_x, ext_z) > 1.75 or min(ext_x, ext_z) > 1.05:
                continue
            cand = _cell_centroid_candidate(cp, ch, cell)
            cand.height = cloud.ground_at(cand.x, cand.z) + cand.rel_height
            candidates.append(cand)

    # Same physical pad found in two adjacent height layers.
    merged: list[SeatCandidate] = []
    for cand in sorted(candidates, key=lambda c: c.area, reverse=True):
        dup = next((m for m in merged
                    if math.hypot(m.x - cand.x, m.z - cand.z) < 0.30
                    and abs(m.rel_height - cand.rel_height) <= height_bin + 0.02), None)
        if dup is None:
            merged.append(cand)
            continue
        total = dup.area + cand.area
        dup.x = (dup.x * dup.area + cand.x * cand.area) / total
        dup.z = (dup.z * dup.area + cand.z * cand.area) / total
        dup.rel_height = (dup.rel_height * dup.area + cand.rel_height * cand.area) / total
        dup.height = cloud.ground_at(dup.x, dup.z) + dup.rel_height
        dup.area = total
        dup.extent_x = max(dup.extent_x, cand.extent_x)
        dup.extent_z = max(dup.extent_z, cand.extent_z)

    for cand in merged:
        cand.overhead = _overhead_ratio(cloud, cand.x, cand.z, cand.rel_height)
    return merged[:max_candidates]


def detect_seats(image_path, cloud: RoomCloud, mode: str = "auto",
                 cell: float = CELL_M) -> tuple[list[SeatCandidate], str]:
    """Seat discovery: COCO instance segmentation first, geometry as the fallback."""
    if mode in ("auto", "vision"):
        vision = detect_seats_vision(image_path, cloud, cell)
        if vision or mode == "vision":
            return vision, "vision"
    if mode == "off":
        return [], "off"
    return detect_seats_geometry(cloud, cell), "geometry"


# --- Anchors --------------------------------------------------------------------------

def _direction(deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    return (math.sin(rad), math.cos(rad))


def _probe_front(grid: WalkableGrid, x: float, z: float, dx: float, dz: float,
                 step: float = 0.06) -> tuple[float | None, float]:
    """Walks outward from the pad: (distance to first standable cell, standing room there).

    A fixed probe ring does not work in real rooms — a seat can sit fully outside the
    reconstructed floor region — so the first reachable cell along the ray wins and what
    is scored is how much open floor she has around that standing spot.
    """
    for dist in np.arange(0.28, APPROACH_MAX_M + 1e-6, step):
        px, pz = x + dx * dist, z + dz * dist
        if grid.is_walkable(px, pz):
            return float(dist), _lookup_openness(grid, px, pz)
    return None, 0.0


def _lookup_openness(grid: WalkableGrid, x: float, z: float) -> float:
    if grid.openness is None:
        return 1.0
    i, j = grid.world_to_cell(x, z)
    if not (0 <= i < grid.width and 0 <= j < grid.height):
        return 0.0
    return float(grid.openness[j, i])


def _approach_spot(grid: WalkableGrid, x: float, z: float, dx: float,
                   dz: float) -> tuple[float, float]:
    for dist in np.arange(0.28, APPROACH_MAX_M + 1e-6, 0.04):
        px, pz = x + dx * dist, z + dz * dist
        if grid.is_walkable(px, pz):
            return (px, pz)
    return (x + dx * APPROACH_IDEAL_M, z + dz * APPROACH_IDEAL_M)


def _backrest_ratio(cloud: RoomCloud, x: float, z: float, dx: float, dz: float,
                    rel_height: float) -> float:
    """Fraction of the geometry right behind a pad that stands upright.

    The one cue that separates "a chair you sit in" from "a tabletop you must not sit on".
    A ratio, not a count, so it does not drift with distance or point density.
    """
    near = cloud.index.query(x - dx * 0.20, z - dz * 0.20, 0.22)
    if len(near) < 30:
        return 0.0
    h = near[:, 1] - cloud.ground_at(x, z)
    back = int(((h >= rel_height + 0.12) & (h <= rel_height + 0.62)).sum())
    pad = int((np.abs(h - rel_height) <= 0.10).sum())
    return back / float(back + pad) if back + pad >= 30 else 0.0


def solve_seat_anchors(candidates: list[SeatCandidate], grid: WalkableGrid,
                       cloud: RoomCloud, min_separation: float = 0.55) -> list[dict]:
    """Picks each pad's outward facing + approach spot and emits scene.json anchors."""
    (mn_x, mn_z), (mx_x, mx_z) = grid.bounds()
    room_x, room_z = (mn_x + mx_x) * 0.5, (mn_z + mx_z) * 0.5

    anchors: list[dict] = []
    for cand in candidates:
        # Facing is fixed by the furniture, but the approach may come from a wide arc: a
        # chair facing a desk is sat down on from the side, not head-on.
        if cand.facing_hint is not None:
            hint_deg = math.degrees(math.atan2(*cand.facing_hint))
            probe_degs = [hint_deg + d for d in (-60, -45, -30, -15, 0, 15, 30, 45, 60)]
        else:
            hint_deg = None
            probe_degs = [float(d) for d in range(0, 360, 15)]

        best = None
        for deg in probe_degs:
            dx, dz = _direction(deg)
            reach, open_floor = _probe_front(grid, cand.x, cand.z, dx, dz)
            if reach is None or open_floor < (0.30 if cand.source == "vision" else 0.38):
                continue
            back = _backrest_ratio(cloud, cand.x, cand.z, dx, dz, cand.rel_height)
            vx, vz = room_x - cand.x, room_z - cand.z
            vn = math.hypot(vx, vz)
            inward = (dx * vx + dz * vz) / vn if vn > 1e-4 else 0.0
            score = (0.46 * open_floor + 0.42 * min(1.0, back / 0.32)
                     + 0.12 * max(0.0, inward))
            if hint_deg is not None:
                off = abs(((deg - hint_deg + 180) % 360) - 180)
                score += 0.30 * max(0.0, 1.0 - off / 70.0)
            score += 0.25 * cand.detector_score
            # A long walk-up means she has to squeeze past something — the same signal that
            # catches a chair buried under a table, without guessing at what is overhead.
            score -= 0.35 * max(0.0, reach - 0.65)
            if best is None or score > best[0]:
                best = (score, dx, dz, open_floor, back, reach)

        if best is None:
            continue
        score, dx, dz, open_floor, back, reach = best
        if back < 0.10 and cand.source != "vision":
            continue
        approach = _approach_spot(grid, cand.x, cand.z, dx, dz)
        # Sit on the part of the pad she can actually reach, and never shimmy more than
        # ~0.5 m onto it; a chair pulled up to a desk also gets pushed to its front edge.
        push = 0.16 + max(0.0, reach - 0.60)
        cand.score = score
        anchors.append({
            "_x": cand.x, "_z": cand.z, "_score": score, "_rel": cand.rel_height,
            "kind": "seat",
            "_furniture": cand.furniture,
            "approachPosition": [round(approach[0], 3), 0.0, round(approach[1], 3)],
            "targetPosition": [round(cand.x + dx * push, 3), 0.0,
                               round(cand.z + dz * push, 3)],
            "seatHeight": round(cand.height, 3),
            "facingYawRad": round(math.atan2(dx, dz), 3),
        })

    anchors.sort(key=lambda a: a["_score"], reverse=True)
    kept: list[dict] = []
    for a in anchors:
        if all(math.hypot(a["_x"] - k["_x"], a["_z"] - k["_z"]) >= min_separation
               for k in kept):
            kept.append(a)
    kept.sort(key=lambda a: (a["_z"], a["_x"]))
    for idx, a in enumerate(kept):
        base = SEAT_CLASSES.get(a["_furniture"], "座位")
        a["id"] = f"seat_{idx + 1}"
        a["label"] = f"{_seat_label(a['_x'], a['_z'], grid)}{base}"
    return kept


def _seat_label(x: float, z: float, grid: WalkableGrid) -> str:
    """Place in the room, meant to be read as "<place>的<furniture>"."""
    (mn_x, mn_z), (mx_x, mx_z) = grid.bounds()
    nx = (x - (mn_x + mx_x) * 0.5) / max(1e-3, mx_x - mn_x)
    nz = (z - (mn_z + mx_z) * 0.5) / max(1e-3, mx_z - mn_z)
    horizontal = "左侧" if nx < -0.12 else ("右侧" if nx > 0.12 else "")
    depth = "靠里" if nz < -0.12 else ("靠前" if nz > 0.12 else "")
    return f"{horizontal}{depth}" or "房间中央"

