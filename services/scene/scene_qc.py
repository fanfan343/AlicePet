#!/usr/bin/env python3
"""Debug renders for the scene analyser: walkable grid drawn top-down and on the room image."""

from __future__ import annotations

import math

import numpy as np

from scene_geometry import GROUND_TOL
from scene_grid import WalkableGrid


def _load_font(size: int):
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/PingFang.ttc",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default(size=size)


def render_qc_overlay(grid: WalkableGrid, anchors: list[dict], out_path,
                      scale: int = 7) -> None:
    """Top-down debug map: floor, furniture footprint, reachable region, and any anchors."""
    from PIL import Image, ImageDraw

    font = _load_font(13)
    img = Image.new("RGB", (grid.width * scale, grid.height * scale), (18, 20, 26))
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

    sx, sy = to_px(grid.spawn[0], grid.spawn[1])
    draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], outline=(120, 240, 150), width=2)
    draw.text((sx + 8, sy - 7), "spawn", fill=(120, 240, 150), font=font)

    for a in anchors:
        tgt = a.get("targetPosition")
        app = a.get("approachPosition")
        if not tgt or len(tgt) < 3:
            continue
        ax, ay = to_px(float(tgt[0]), float(tgt[2]))
        if app and len(app) >= 3:
            px, py = to_px(float(app[0]), float(app[2]))
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(120, 240, 150))
        yaw = float(a.get("facingYawRad", 0.0))
        dx, dz = math.sin(yaw), math.cos(yaw)
        draw.line([ax, ay, ax + dx * 34, ay + dz * 34], fill=(255, 196, 92), width=2)
        draw.ellipse([ax - 6, ay - 6, ax + 6, ay + 6], fill=(232, 96, 88))
        draw.text((ax + 9, ay + 4), str(a.get("label", a.get("id", ""))), fill=(240, 244, 250), font=font)
    img.save(out_path)


def render_grid_on_image(scene_cfg: dict, bg_path, grid: WalkableGrid, out_path,
                         alpha: float = 0.45) -> None:
    """Tints the room image with the walkable floor decision (green = walkable, grey = cut off)."""
    from PIL import Image

    img = Image.open(bg_path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32)
    h, w = rgb.shape[:2]
    cfg_w, cfg_h = scene_cfg["depthWidth"], scene_cfg["depthHeight"]
    depth = np.fromfile(str(bg_path).replace("bg.png", "depth.bin"),
                        dtype=np.float32).reshape(cfg_h, cfg_w)

    cam = np.asarray(scene_cfg["cameraPosition"], dtype=np.float64)
    fwd = np.asarray(scene_cfg["cameraTarget"], dtype=np.float64) - cam
    fwd /= max(1e-9, float(np.linalg.norm(fwd)))
    right = np.cross(fwd, np.asarray(scene_cfg["cameraUp"], dtype=np.float64))
    right /= max(1e-9, float(np.linalg.norm(right)))
    up = np.cross(right, fwd)
    tan_y = math.tan(math.radians(scene_cfg["fovYDegrees"]) * 0.5)
    tan_x = tan_y * (scene_cfg["imageWidth"] / float(scene_cfg["imageHeight"]))

    u = ((np.arange(cfg_w) + 0.5) / cfg_w) * 2.0 - 1.0
    v = 1.0 - ((np.arange(cfg_h) + 0.5) / cfg_h) * 2.0
    ray = (fwd[None, None, :]
           + right[None, None, :] * (u[None, :, None] * tan_x)
           + up[None, None, :] * (v[:, None, None] * tan_y))
    world = cam[None, None, :] + ray * depth[:, :, None]

    ii = ((world[..., 0] - grid.origin_x) / grid.cell).astype(np.int64)
    jj = ((world[..., 2] - grid.origin_z) / grid.cell).astype(np.int64)
    inside = (ii >= 0) & (ii < grid.width) & (jj >= 0) & (jj < grid.height)
    ii_c = np.clip(ii, 0, grid.width - 1)
    jj_c = np.clip(jj, 0, grid.height - 1)

    g = np.asarray(scene_cfg.get("walkableGrid", {}).get("groundSurface", [0.0] * 6))
    wx, wy, wz = world[..., 0], world[..., 1], world[..., 2]
    height = wy - (g[0] + g[1] * wx + g[2] * wz + g[3] * wx ** 2 + g[4] * wx * wz
                   + g[5] * wz ** 2)
    on_ground = height <= GROUND_TOL

    tint = np.zeros((cfg_h, cfg_w, 3), dtype=np.float32)
    walk = grid.walkable[jj_c, ii_c] & inside & on_ground
    floor = ((grid.floor[jj_c, ii_c] if grid.floor is not None else walk)
             & inside & on_ground)
    tint[walk] = (60, 220, 120)
    tint[floor & ~walk] = (200, 200, 210)
    mask = inside & on_ground & (walk | floor)

    tint_img = Image.fromarray(np.clip(tint, 0, 255).astype(np.uint8)).resize(
        (w, h), Image.Resampling.NEAREST)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (w, h), Image.Resampling.NEAREST)

    blended = rgb.copy()
    m = np.asarray(mask_img, dtype=np.float32) / 255.0 * alpha
    blended = blended * (1.0 - m[..., None]) + np.asarray(tint_img, dtype=np.float32) * m[..., None]
    Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).save(out_path)
