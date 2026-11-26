#!/usr/bin/env python3
"""One-shot single-image metric depth + camera ground-plane + walkable grid estimator (MoGe-2)
for AlicePet Picture-World.

Takes any 2D room illustration or photo, runs Microsoft MoGe-2 (ViT-S/B with metric scale & normals),
fits the floor plane + curved ground surface, builds a top-down walkable grid, and writes:
  - bg.png        (original RGB image)
  - depth.bin     (float32 metric depth map at native resolution, positive camera-forward Z in meters)
  - depth_vis.png (false-color depth preview for inspection)
  - scene.json    (camera extrinsics/FOV, fitted ground surface, walkableGrid, and any preserved anchors)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import scene_geometry
import scene_grid
import scene_qc


def fit_floor_plane_ransac(
    points_cam: np.ndarray,
    normals_cam: np.ndarray | None,
    valid_mask: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fits the lowest horizontal floor plane `n . p + d = 0` in right-handed camera space
    (+X right, +Y up, -Z forward), ignoring elevated horizontal surfaces like tabletops/desks.

    Returns `(normal_unit, d, floor_mask)` with `normal_unit.y > 0` and camera height `h = d > 0`.
    """
    h, w = valid_mask.shape
    ys, _ = np.mgrid[0:h, 0:w]

    z_dist = -points_cam[..., 2]
    cand = valid_mask & (ys > int(h * 0.42)) & (z_dist > 0.6) & (z_dist < 12.0)

    if normals_cam is not None:
        up_cand = cand & (normals_cam[..., 1] > 0.72)
        if int(up_cand.sum()) >= 300:
            n_med = np.median(normals_cam[up_cand], axis=0).astype(np.float64)
            n_norm = float(np.linalg.norm(n_med))
            if n_norm > 1e-4:
                n_up = n_med / n_norm
                if n_up[1] < 0.0:
                    n_up = -n_up
                up_pts = points_cam[up_cand]
                heights_below_cam = -(up_pts @ n_up)
                valid_h = heights_below_cam[(heights_below_cam > 0.5) & (heights_below_cam < 3.8)]
                if len(valid_h) >= 100:
                    h_floor_hi = float(np.percentile(valid_h, 92))
                    floor_band = valid_h[(valid_h >= h_floor_hi - 0.18) & (valid_h <= h_floor_hi + 0.12)]
                    d_floor = float(np.median(floor_band)) if len(floor_band) >= 30 else h_floor_hi

                    full_h = -(points_cam @ n_up)
                    floor_mask = up_cand & (np.abs(full_h - d_floor) < 0.12)
                    if int(floor_mask.sum()) >= 200:
                        f_pts = points_cam[floor_mask]
                        rng = np.random.default_rng(seed)
                        if len(f_pts) > 6000:
                            f_pts = f_pts[rng.choice(len(f_pts), size=6000, replace=False)]
                        centroid = f_pts.mean(axis=0)
                        _, _, vt = np.linalg.svd(f_pts - centroid, full_matrices=False)
                        n_ref = vt[-1]
                        if n_ref[1] < 0.0:
                            n_ref = -n_ref
                        if n_ref[1] >= 0.75 and float(np.dot(n_ref, n_up)) > 0.90:
                            n_up = n_ref / float(np.linalg.norm(n_ref))
                            d_floor = -float(np.dot(n_up, centroid))
                    floor_mask = valid_mask & (np.abs(points_cam @ n_up + d_floor) < 0.10) & (ys > int(h * 0.40))
                    return n_up, d_floor, floor_mask

    n_fallback = np.array([0.0, 0.94, 0.34], dtype=np.float64)
    n_fallback /= float(np.linalg.norm(n_fallback))
    return n_fallback, 1.30, cand


def pick_spawn_point(
    points_cam: np.ndarray,
    floor_mask: np.ndarray,
    normal_cam: np.ndarray,
    plane_d: float,
) -> np.ndarray:
    """Picks a clean central floor spawn point (around 2.0m..3.8m from camera) in camera space."""
    h, w = floor_mask.shape
    ys, xs = np.mgrid[0:h, 0:w]
    center_band = floor_mask & (xs > int(w * 0.30)) & (xs < int(w * 0.70))
    pts = points_cam[center_band] if int(center_band.sum()) >= 50 else points_cam[floor_mask]

    if len(pts) >= 20:
        z_dists = -pts[:, 2]
        target_z = float(np.clip(np.percentile(z_dists, 45), 2.0, 3.8))
        scores = np.abs(pts[:, 0]) * 1.5 + np.abs(z_dists - target_z)
        chosen = pts[int(np.argmin(scores))].astype(np.float64)
        dist = float(np.dot(normal_cam, chosen) + plane_d)
        return chosen - dist * normal_cam

    ray = np.array([0.0, -0.35, -1.0], dtype=np.float64)
    denom = float(np.dot(normal_cam, ray))
    t = -plane_d / denom if abs(denom) > 1e-4 else 2.6
    t = float(np.clip(t, 1.8, 4.0))
    return ray * t


def build_world_camera(
    normal_cam: np.ndarray,
    spawn_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds exact world-space (camPos, camTarget, camUp, R_c2w) where world origin (0,0,0) = spawn on floor Y=0."""
    up_cam = normal_cam / float(np.linalg.norm(normal_cam))
    cam_fwd_in_cam = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    fwd_horiz_cam = cam_fwd_in_cam - float(np.dot(cam_fwd_in_cam, up_cam)) * up_cam
    fwd_norm = float(np.linalg.norm(fwd_horiz_cam))
    if fwd_norm < 1e-4:
        fwd_horiz_cam = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        fwd_horiz_cam = fwd_horiz_cam / fwd_norm

    right_cam = np.cross(fwd_horiz_cam, up_cam)
    right_cam = right_cam / float(np.linalg.norm(right_cam))
    z_world_in_cam = -fwd_horiz_cam

    R_w2c = np.column_stack((right_cam, up_cam, z_world_in_cam))
    R_c2w = R_w2c.T

    cam_pos_world = R_c2w @ (-spawn_cam)
    cam_fwd_world = R_c2w @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    cam_up_world = R_c2w @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cam_target_world = cam_pos_world + cam_fwd_world * 2.5

    return cam_pos_world, cam_target_world, cam_up_world, R_c2w


def main() -> int:
    parser = argparse.ArgumentParser(description="MoGe-2 single-image scene depth & walkable grid estimator for AlicePet")
    parser.add_argument("--image", required=True, help="Input 2D room/scene image path")
    parser.add_argument("--out-dir", required=True, help="Output scene directory")
    parser.add_argument("--model", default="Ruicheng/moge-2-vits-normal", help="MoGe-2 HuggingFace model ID")
    parser.add_argument("--name", default="", help="Display name for scene.json")
    parser.add_argument("--depth-max-dim", type=int, default=0,
                        help="Max dimension for stored depth.bin; 0 keeps MoGe's native resolution.")
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--grid-cell", type=float, default=scene_geometry.CELL_M,
                        help="Walkable grid cell size in metres (default 0.06)")
    parser.add_argument("--no-qc", action="store_true", help="Skip the top-down and image QC overlays")
    args = parser.parse_args()

    img_path = Path(args.image).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pil_img = Image.open(img_path).convert("RGB")
    img_w, img_h = pil_img.size
    rgb = np.asarray(pil_img, dtype=np.uint8)

    bg_out = out_dir / "bg.png"
    pil_img.save(bg_out)

    from moge.model.v2 import MoGeModel

    t0 = time.monotonic()
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[Scene] Loading {args.model} on {device}...", flush=True)
    model = MoGeModel.from_pretrained(args.model).to(device).eval()
    load_ms = (time.monotonic() - t0) * 1000.0

    img_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
    t1 = time.monotonic()
    try:
        with torch.no_grad():
            output = model.infer(img_tensor, use_fp16=False)
    except Exception as e:
        if device.type == "mps":
            print(f"[Scene] MPS infer fallback to CPU ({e})", flush=True)
            device = torch.device("cpu")
            model = model.to(device)
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                output = model.infer(img_tensor, use_fp16=False)
        else:
            raise
    infer_ms = (time.monotonic() - t1) * 1000.0

    points_cv = output["points"].detach().cpu().numpy().astype(np.float64)
    depth_raw = output["depth"].detach().cpu().numpy().astype(np.float32)
    mask = output["mask"].detach().cpu().numpy().astype(bool)
    intrinsics = output["intrinsics"].detach().cpu().numpy().astype(np.float64)
    normals_cv = output["normal"].detach().cpu().numpy().astype(np.float64) if "normal" in output else None

    fy = float(intrinsics[1, 1])
    fov_y_deg = float(np.clip(np.degrees(2.0 * math.atan(0.5 / max(fy, 1e-4))), 20.0, 90.0))

    points_cam = np.stack([points_cv[..., 0], -points_cv[..., 1], -points_cv[..., 2]], axis=-1)
    normals_cam = None
    if normals_cv is not None:
        normals_cam = np.stack([normals_cv[..., 0], -normals_cv[..., 1], -normals_cv[..., 2]], axis=-1)
        if np.median(normals_cam[mask, 1][normals_cam[mask, 1] != 0]) < -0.5:
            normals_cam = -normals_cam

    valid = mask & np.isfinite(depth_raw) & (depth_raw > 0.1) & (depth_raw < 49.0)
    floor_n, floor_d, floor_mask = fit_floor_plane_ransac(points_cam, normals_cam, valid)
    spawn_cam = pick_spawn_point(points_cam, floor_mask, floor_n, floor_d)
    cam_pos, cam_target, cam_up, R_c2w = build_world_camera(floor_n, spawn_cam)

    depth_clean = np.where(valid, np.clip(depth_raw, 0.1, 48.0), 48.0).astype(np.float32)
    depth_clean = scene_geometry.refine_depth_edges(depth_clean, rgb)

    max_dim = args.depth_max_dim if args.depth_max_dim > 0 else max(img_w, img_h)
    scale = min(1.0, max_dim / float(max(img_w, img_h)))
    dep_w = max(64, int(round(img_w * scale)))
    dep_h = max(64, int(round(img_h * scale)))
    depth_resized = (depth_clean if (dep_w, dep_h) == (img_w, img_h)
                     else cv2.resize(depth_clean, (dep_w, dep_h),
                                     interpolation=cv2.INTER_NEAREST)).astype(np.float32)
    depth_resized.tofile(out_dir / "depth.bin")

    d_vis = np.clip(depth_resized, 0.5, 10.0)
    d_norm = ((1.0 - (d_vis - 0.5) / 9.5) * 255.0).astype(np.uint8)
    cv2.imwrite(str(out_dir / "depth_vis.png"), cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO))

    t2 = time.monotonic()
    points_world, normals_world = scene_geometry.camera_to_world(
        points_cam, normals_cam, spawn_cam, R_c2w)
    cloud = scene_geometry.RoomCloud.build(points_world, normals_world, valid)
    grid = scene_grid.build_walkable_grid(cloud, cell=args.grid_cell)
    grid_json = grid.to_json(ground=cloud.ground)
    grid_ms = (time.monotonic() - t2) * 1000.0
    walk_min, walk_max = grid_json["boundsMin"], grid_json["boundsMax"]

    scene_cfg = {
        "version": 1,
        "name": args.name or img_path.stem,
        "imageWidth": img_w,
        "imageHeight": img_h,
        "depthWidth": dep_w,
        "depthHeight": dep_h,
        "fovYDegrees": round(fov_y_deg, 3),
        "cameraPosition": [round(float(x), 4) for x in cam_pos],
        "cameraTarget": [round(float(x), 4) for x in cam_target],
        "cameraUp": [round(float(x), 4) for x in cam_up],
        "avatarSpawnPosition": [round(grid.spawn[0], 3), 0.0, round(grid.spawn[1], 3)],
        "avatarSpawnYawRad": 0.0,
        "avatarScale": 1.0,
        "walkBoundsMin": walk_min,
        "walkBoundsMax": walk_max,
        "nearClip": 0.05,
        "farClip": 50.0,
        "lightingPreset": "warm_lamp",
        "walkableGrid": grid_json,
    }

    # Preserve any hand-authored or VLM-authored anchors already present in scene.json.
    existing_anchors: list[dict] = []
    scene_json_path = out_dir / "scene.json"
    if scene_json_path.exists():
        try:
            loaded = json.loads(scene_json_path.read_text(encoding="utf-8")).get("anchors")
            if isinstance(loaded, list) and loaded:
                existing_anchors = loaded
                scene_cfg["anchors"] = existing_anchors
        except (json.JSONDecodeError, OSError):
            pass

    scene_json_path.write_text(json.dumps(scene_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_qc:
        try:
            scene_qc.render_qc_overlay(grid, existing_anchors, out_dir / "qc_topdown.png")
            scene_qc.render_grid_on_image(scene_cfg, bg_out, grid, out_dir / "qc_grid_on_image.png")
        except Exception as exc:
            print(f"[Scene] QC overlay skipped ({exc})", flush=True)

    walkable_cells = int(grid_json["walkableCount"])
    print(
        f"[Scene] OK! load={load_ms:.0f}ms infer={infer_ms:.0f}ms grid={grid_ms:.0f}ms ({device}) "
        f"fovY={fov_y_deg:.1f}° camHeight={float(cam_pos[1]):.2f}m spawnDist={float(np.linalg.norm(cam_pos)):.2f}m "
        f"floorInliers={int(floor_mask.sum())} -> {out_dir}",
        flush=True,
    )
    print(
        f"[Scene] Walkable: {walkable_cells} cells @ {args.grid_cell:.2f}m "
        f"({walkable_cells * args.grid_cell * args.grid_cell:.2f} m^2), "
        f"bounds X[{walk_min[0]:+.2f},{walk_max[0]:+.2f}] Z[{walk_min[1]:+.2f},{walk_max[1]:+.2f}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
