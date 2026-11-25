#!/usr/bin/env python3
"""Download and SHA-256 verify NVIDIA MotionBricks SOMA LFS assets without needing git-lfs.

Fetches only the SOMA fp16 ONNX model (~380 MB), SOMA USD skeleton (~2 MB), SOMA BVH motion clips,
and SOMA rig metadata from the pinned NVlabs/motionbricks commit into `upstream/`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

UPSTREAM_REPO = "https://github.com/NVlabs/motionbricks.git"
UPSTREAM_COMMIT = "54fc563aeaaf819a753d72cc9fdc6713989cdd43"
MEDIA_BASE = f"https://media.githubusercontent.com/media/NVlabs/motionbricks/{UPSTREAM_COMMIT}"

REQUIRED_PATHS = [
    "motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx",
    "motionbricks/assets/human/soma_base_skel_minimal_no_root.usd",
    "tools/onnx/assets/soma_model_rig.npz",
]
REQUIRED_DIRS = [
    "assets/motions/soma",
]


def parse_lfs_pointer(path: Path) -> tuple[str, int] | None:
    if not path.is_file() or path.stat().st_size > 1024:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    oid = None
    size = None
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split(":", 1)[1].strip()
        elif line.startswith("size "):
            size = int(line.split(" ", 1)[1].strip())
    if oid and size is not None:
        return oid, size
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_upstream_checkout(upstream: Path) -> None:
    if (upstream / ".git").exists():
        return
    upstream.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(
        ["git", "clone", "--depth", "1", UPSTREAM_REPO, str(upstream)],
        check=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "-C", str(upstream), "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT],
        check=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "-C", str(upstream), "checkout", UPSTREAM_COMMIT],
        check=True,
        env=env,
        stdin=subprocess.DEVNULL,
    )


def download_lfs_file(rel_path: str, dest: Path, expected_sha: str, expected_size: int) -> None:
    url = f"{MEDIA_BASE}/{urllib.parse.quote(rel_path, safe='/')}"
    tmp = dest.with_name(dest.name + ".part")
    mb = expected_size / (1024 * 1024)
    print(f"[MotionBricks Setup] Downloading {rel_path} ({mb:.1f} MB)...", flush=True)
    try:
        subprocess.run(
            ["curl", "-fsSL", "--retry", "3", "--connect-timeout", "30", "-o", str(tmp), url],
            check=True,
            stdin=subprocess.DEVNULL,
        )
        actual_size = tmp.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"Size mismatch for {rel_path}: got {actual_size}, expected {expected_size}")
        actual_sha = sha256_file(tmp)
        if actual_sha != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch for {rel_path}: got {actual_sha}, expected {expected_sha}")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    root = Path(__file__).resolve().parent
    upstream = root / "upstream"
    ensure_upstream_checkout(upstream)

    targets: list[str] = list(REQUIRED_PATHS)
    for rel_dir in REQUIRED_DIRS:
        base = upstream / rel_dir
        if base.is_dir():
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    targets.append(p.relative_to(upstream).as_posix())

    downloaded = 0
    already_ok = 0
    for rel_path in targets:
        dest = upstream / rel_path
        if not dest.exists():
            print(f"[MotionBricks Setup] Warning: missing expected path {rel_path}", file=sys.stderr)
            continue
        pointer = parse_lfs_pointer(dest)
        if pointer is None:
            already_ok += 1
            continue
        oid, size = pointer
        download_lfs_file(rel_path, dest, oid, size)
        downloaded += 1

    print(
        f"[MotionBricks Setup] Ready! downloaded={downloaded}, already_hydrated={already_ok}, total={len(targets)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
