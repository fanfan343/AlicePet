#!/usr/bin/env python3
"""
Extract authentic author-authored Unity Humanoid Muscle animation clips
(RootT, RootQ, body muscles, finger muscles) from AlicePhilVRC.unitypackage
into /Users/funnyliu/Downloads/VRM项目/降采样模型/独立道具与配方/animations.json.
"""

import json
import math
import os
import re
import tarfile

UNITY_PKG = "/Users/funnyliu/Downloads/AlicePhil(VRC&Other) 1.05/VRCUnitypackage/AlicePhilVRC.unitypackage"
OUT_PATH = "/Users/funnyliu/Downloads/VRM项目/降采样模型/独立道具与配方/animations.json"

TARGET_CLIPS = {
    "stand": ("Pose/Alice_Stand_still.anim", False, 15.0),
    "sit": ("Pose/Alice_Sit.anim", False, 15.0),
    "sit_action": ("Pose/Alice_Sit (Action).anim", False, 15.0),
    "crouch": ("Pose/Alice_Crouch_still.anim", False, 15.0),
    "prone": ("Pose/Alice_Prone_still.anim", False, 15.0),
    "afk_in": ("Pose/AFK/Alice_AFK_In.anim", False, 15.0),
    "afk_loop": ("Pose/AFK/Alice_AFK_Loop.anim", True, 15.0),
    "afk_out": ("Pose/AFK/Alice_AFK_Out.anim", False, 15.0),
}


def parse_unity_float_curves(yaml_text):
    """Parse m_FloatCurves (where Unity stores RootT/RootQ + Humanoid Muscle curves)."""
    sr_m = re.search(r"m_SampleRate:\s*([0-9.]+)", yaml_text)
    st_m = re.search(r"m_StopTime:\s*([0-9.]+)", yaml_text)
    loop_m = re.search(r"m_LoopTime:\s*(\d+)", yaml_text)
    duration = float(st_m.group(1)) if st_m else 0.0
    sample_rate = float(sr_m.group(1)) if sr_m else 30.0
    loop = (loop_m.group(1) == "1") if loop_m else False

    # Restrict strictly to m_FloatCurves section (before m_PPtrCurves / m_SampleRate)
    fc_start = yaml_text.find("m_FloatCurves:")
    if fc_start < 0:
        return duration, sample_rate, loop, {}
    fc_end = yaml_text.find("m_PPtrCurves:", fc_start)
    if fc_end < 0:
        fc_end = yaml_text.find("m_SampleRate:", fc_start)
    section = yaml_text[fc_start:fc_end] if fc_end > fc_start else yaml_text[fc_start:]

    curves = {}
    blocks = re.split(r"\r?\n  - (?:serializedVersion: \d+\r?\n    )?curve:\r?\n", section)[1:]
    key_re = re.compile(
        r"time:[ \t]*([-\d.eE]+)\r?\n[ \t]*value:[ \t]*([-\d.eE]+)"
    )
    attr_re = re.compile(r"\n    attribute:[ \t]*([^\r\n]+)")
    path_re = re.compile(r"\n    path:[ \t]*([^\r\n]*)")
    class_re = re.compile(r"\n    classID:[ \t]*(\d+)")

    for block in blocks:
        am = attr_re.search(block)
        if not am:
            continue
        attr = am.group(1).strip()
        pm = path_re.search(block)
        path = pm.group(1).strip() if pm else ""
        cm = class_re.search(block)
        class_id = int(cm.group(1)) if cm else 0

        # Humanoid RootT/RootQ and Muscles live on Animator (classID 95) with empty path ""
        if path != "" or class_id != 95:
            continue
        # Skip IK goal curves (LeftFootT/Q, RightFootT/Q, LeftHandT/Q, RightHandT/Q)
        if any(attr.startswith(p) for p in ("LeftFoot", "RightFoot", "LeftHandT", "LeftHandQ", "RightHandT", "RightHandQ")):
            continue
        if attr.startswith("RootT.") or attr.startswith("RootQ.") or attr.startswith("LeftHand.") or attr.startswith("RightHand.") or any(
            kw in attr for kw in (
                "Spine", "Chest", "Neck", "Head", "Shoulder", "Arm", "Forearm", "Hand",
                "Leg", "Foot", "Toes"
            )
        ):
            keys = [(float(t), float(v)) for t, v in key_re.findall(block)]
            if keys:
                curves[attr] = keys

    return duration, sample_rate, loop, curves


def eval_curve(keys, t):
    if not keys:
        return 0.0
    if len(keys) == 1 or t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    # Binary search
    lo, hi = 0, len(keys) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if keys[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, v0 = keys[lo]
    t1, v1 = keys[hi]
    if t1 <= t0 + 1e-8:
        return v0
    u = (t - t0) / (t1 - t0)
    return v0 + (v1 - v0) * u


def normalize_quat(q):
    ln = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if ln < 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    return [round(c / ln, 5) for c in q]


def resample_clip(duration, loop, curves, target_fps):
    if duration <= 1e-4:
        times = [0.0]
    else:
        steps = max(2, int(round(duration * target_fps)))
        times = [duration * i / steps for i in range(steps + 1)]

    frames = []
    muscle_keys = sorted(k for k in curves.keys() if not k.startswith("Root"))
    for t in times:
        root_t = [
            round(eval_curve(curves.get("RootT.x", []), t), 5),
            round(eval_curve(curves.get("RootT.y", []), t), 5),
            round(eval_curve(curves.get("RootT.z", []), t), 5),
        ]
        root_q = normalize_quat([
            eval_curve(curves.get("RootQ.x", []), t),
            eval_curve(curves.get("RootQ.y", []), t),
            eval_curve(curves.get("RootQ.z", []), t),
            eval_curve(curves.get("RootQ.w", [(0.0, 1.0)]), t),
        ])
        muscles = {}
        for mk in muscle_keys:
            val = eval_curve(curves[mk], t)
            if abs(val) > 1e-4:
                muscles[mk] = round(val, 4)
        frames.append({
            "t": round(t, 4),
            "rootT": root_t,
            "rootQ": root_q,
            "m": muscles,
        })

    return {
        "duration": round(duration, 4),
        "fps": target_fps,
        "loop": loop,
        "frames": frames,
    }


def main():
    entries = {}
    with tarfile.open(UNITY_PKG, "r:gz") as tf:
        for m in tf.getmembers():
            parts = m.name.split("/")
            if len(parts) == 2:
                entries.setdefault(parts[0], {})[parts[1]] = m

        clips_out = {}
        for clip_id, (suffix, force_loop, target_fps) in TARGET_CLIPS.items():
            matched_asset = None
            matched_path = None
            for guid, d in entries.items():
                if "pathname" in d and "asset" in d:
                    p = tf.extractfile(d["pathname"]).read().decode("utf-8", errors="replace").splitlines()[0].strip()
                    if p.endswith(suffix):
                        matched_asset = d["asset"]
                        matched_path = p
                        break
            if not matched_asset:
                print(f"[WARN] Could not find {suffix}")
                continue
            yaml_text = tf.extractfile(matched_asset).read().decode("utf-8", errors="replace")
            duration, sr, loop_flag, curves = parse_unity_float_curves(yaml_text)
            clip_data = resample_clip(duration, force_loop or loop_flag, curves, target_fps)
            clip_data["sourceAnim"] = matched_path
            clips_out[clip_id] = clip_data
            print(f"[OK] {clip_id:12s} dur={clip_data['duration']:6.2f}s frames={len(clip_data['frames']):3d} curves={len(curves):3d} ({matched_path.split('/')[-1]})")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    payload = {
        "version": 1,
        "source": "AlicePhil(VRC&Other) 1.05 / AlicePhilVRC.unitypackage",
        "clips": clips_out,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = os.path.getsize(OUT_PATH) / 1024.0
    print(f"Wrote {OUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
