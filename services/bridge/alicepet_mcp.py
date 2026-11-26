#!/usr/bin/env python3
"""Minimal zero-dependency stdio MCP server bridging any MCP harness (ZeroClaw, etc.)
to AlicePet's native loopback LocalControlServer (http://127.0.0.1:39550).

- Pure Python standard library (no pip/venv required).
- Supports both newline-delimited JSON-RPC (MCP 2024-11-05+) and Content-Length framing.
- Logs every tool invocation and latency into ~/Library/Logs/AlicePet/alicepet.log ([MCP]).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

CONTROL_BASE = os.environ.get("ALICEPET_CONTROL_URL", "http://127.0.0.1:39550").rstrip("/")
LOG_PATH = Path.home() / "Library/Logs/AlicePet/alicepet.log"


def log_line(level: str, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{ts}] [{level}] [MCP] {msg}\n"
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def http_json(method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 20.0) -> dict[str, Any]:
    url = f"{CONTROL_BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"ok": False, "error": raw}
        parsed.setdefault("httpStatus", e.code)
        return parsed
    except Exception as e:
        return {"ok": False, "error": f"AlicePet control bridge unreachable at {url}: {e}"}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_scene_state",
        "description": (
            "Read AlicePet's live state: mode (desktop_pet or picture_world), 3D position, "
            "depth occlusion status, boundary contacts, continuous emotion inertia state, "
            "active pose/motion engine, and available room anchors."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "observe_scene",
        "description": (
            "Capture a live visual snapshot of Alice in her current room (plus room background path "
            "and spatial telemetry) so you can visually inspect her surroundings, occlusion, and furniture."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_image": {
                    "type": "boolean",
                    "description": "Attach the captured PNG image inline for visual inspection (default true).",
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "set_look",
        "description": (
            "Update Alice's pose, hand gesture, facial expression, and/or continuous emotion state. "
            "Emotions use opposing-axis inertia (joy/sadness, affection/rejection, excitement/calm, "
            "curiosity/boredom, pride/embarrassment, amusement/annoyance, gratitude/anger, hope/anxiety)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pose": {"type": "string", "enum": ["stand", "sit", "crouch", "prone", "afk_book", "afk_tea"]},
                "hand": {"type": "string", "enum": ["natural", "open", "fist", "peace", "point", "thumbs_up", "ok"]},
                "expression": {"type": "string", "description": "VRM mood or custom face recipe name (e.g. happy, wink, neutral)."},
                "emotion": {"type": "string", "description": "Emotion impulse label (e.g. joy, curiosity, affection, calm, shy)."},
                "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Emotion impulse strength (0..1)."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "ardy_action",
        "description": (
            "Drive Alice's full-body 20fps generative motion via ARDY Mini, or stop it. "
            "Keep prompts gentle, natural, and anime-girl appropriate."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "prompt", "stop"], "description": "Control action (default 'prompt')."},
                "prompt": {"type": "string", "description": "English motion description for ARDY (e.g. 'a gentle anime girl waves cheerfully')."},
                "cfg": {"type": "number", "minimum": 1.0, "maximum": 6.0, "description": "Optional classifier-free guidance weight."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "walk_and_sit",
        "description": "Autonomously walk Alice with A* furniture avoidance to a room seat anchor and sit down.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "anchor_id": {"type": "string", "description": "Anchor ID from get_scene_state (e.g. 'sofa_left')."},
            },
            "required": ["anchor_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "stand_up",
        "description": "Stand Alice up from her current seat or cancel an active walk.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "stop_ardy": {"type": "boolean", "description": "Also stop ARDY stream (default true)."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "load_scene",
        "description": "Enter a Picture-World room by folder name (e.g. 'demo_room') or pass 'exit' to return to transparent desktop pet mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scene": {"type": "string", "description": "Room folder name, absolute path, or 'exit'."},
            },
            "required": ["scene"],
            "additionalProperties": False,
        },
    },
    {
        "name": "register_anchors",
        "description": (
            "Register room interactive anchors identified visually from the room image using normalized 2D coordinates "
            "u (0=left..1=right) and v (0=top..1=bottom). AlicePet unprojects them via MoGe-2 depth into 3D scene.json."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "anchors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string"},
                            "kind": {"type": "string", "enum": ["seat", "stand_spot"]},
                            "u": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "v": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "facingYawDeg": {"type": "number"},
                        },
                        "required": ["id", "label", "u", "v"],
                    },
                }
            },
            "required": ["anchors"],
            "additionalProperties": False,
        },
    },
]


def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    t0 = time.monotonic()
    content: list[dict[str, Any]] = []
    if name == "get_scene_state":
        res = http_json("GET", "/state")
    elif name == "observe_scene":
        res = http_json("POST", "/observe", {}, timeout=25.0)
        if res.get("ok") and args.get("include_image", True):
            img_path = Path(res.get("liveImagePath", ""))
            if img_path.is_file():
                b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
                content.append({"type": "image", "data": b64, "mimeType": "image/png"})
    elif name == "set_look":
        res = http_json("POST", "/look", args)
    elif name == "ardy_action":
        res = http_json("POST", "/ardy", args)
    elif name == "walk_and_sit":
        res = http_json("POST", "/scene/walk-sit", {"anchorId": args.get("anchor_id", "")})
    elif name == "stand_up":
        res = http_json("POST", "/scene/stand-up", {"stopArdy": args.get("stop_ardy", True)})
    elif name == "load_scene":
        scene = str(args.get("scene", "")).strip()
        res = http_json("POST", "/scene/exit", {}) if scene.lower() == "exit" else http_json("POST", "/scene/load", {"scene": scene})
    elif name == "register_anchors":
        res = http_json("POST", "/scene/anchors", {"anchors": args.get("anchors", [])})
    else:
        res = {"ok": False, "error": f"Unknown tool: {name}"}

    ms = (time.monotonic() - t0) * 1000.0
    log_line("INFO" if res.get("ok") else "WARN", f"tool={name} ok={res.get('ok')} ({ms:.1f}ms)")
    content.insert(0, {"type": "text", "text": json.dumps(res, ensure_ascii=False)})
    return {"content": content, "isError": not bool(res.get("ok"))}


def read_message() -> tuple[dict[str, Any] | None, bool]:
    """Reads one JSON-RPC message; returns (message, used_content_length_framing)."""
    stdin = sys.stdin.buffer
    first = stdin.readline()
    if not first:
        return None, False
    stripped = first.strip()
    if not stripped:
        return {}, False
    if stripped.startswith(b"{"):
        return json.loads(stripped.decode("utf-8")), False

    headers: dict[str, str] = {}
    line = first
    while line and line.strip():
        if b":" in line:
            k, v = line.decode("utf-8", errors="replace").split(":", 1)
            headers[k.strip().lower()] = v.strip()
        line = stdin.readline()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return {}, True
    body = stdin.read(length)
    return json.loads(body.decode("utf-8")), True


def write_message(msg: dict[str, Any], framed: bool) -> None:
    payload = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    out = sys.stdout.buffer
    if framed:
        out.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
        out.write(payload)
    else:
        out.write(payload + b"\n")
    out.flush()


def main() -> int:
    log_line("INFO", f"alicepet_mcp starting (target={CONTROL_BASE})")
    while True:
        try:
            msg, framed = read_message()
        except Exception as e:
            log_line("ERROR", f"stdin decode failed: {e}")
            break
        if msg is None:
            break
        if not msg:
            continue

        req_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}

        if method == "initialize":
            proto = params.get("protocolVersion", "2024-11-05")
            write_message({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": proto,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "alicepet-mcp", "version": "0.1.0"},
                },
            }, framed)
        elif method == "notifications/initialized":
            continue
        elif method == "ping":
            write_message({"jsonrpc": "2.0", "id": req_id, "result": {}}, framed)
        elif method == "tools/list":
            write_message({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}, framed)
        elif method == "tools/call":
            result = call_tool(str(params.get("name", "")), params.get("arguments") or {})
            write_message({"jsonrpc": "2.0", "id": req_id, "result": result}, framed)
        elif req_id is not None:
            write_message({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }, framed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
