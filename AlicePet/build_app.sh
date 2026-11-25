#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$REPO_ROOT/AlicePet.app"

CONFIG="${1:-release}"
case "$CONFIG" in
    debug|release) ;;
    *)
        echo "Usage: $0 [release|debug]" >&2
        exit 1
        ;;
esac

cd "$SCRIPT_DIR"

# Ensure VRMMetalKit has our PerfectSync 64-active-morph & SpringBone warmup patches applied cleanly
PATCH_MORPH="$SCRIPT_DIR/patches/VRMMetalKit-perfectsync-morphs.patch"
PATCH_WARMUP="$SCRIPT_DIR/patches/VRMMetalKit-springbone-warmup-binddirs.patch"
EDIT_DIR="$SCRIPT_DIR/Packages/VRMMetalKit"
if [ -f "$PATCH_MORPH" ] || [ -f "$PATCH_WARMUP" ]; then
    if [ ! -d "$EDIT_DIR/Sources" ] || ! grep -q '"name" : "edited"' "$SCRIPT_DIR/.build/workspace-state.json" 2>/dev/null; then
        rm -rf "$EDIT_DIR"
        echo "[AlicePet] Checking out editable VRMMetalKit for patches..."
        swift package edit VRMMetalKit
    fi
    if [ -f "$PATCH_MORPH" ] && ! grep -q "public static let maxActiveMorphs: Int = 64" "$EDIT_DIR/Sources/VRMMetalKit/Core/VRMConstants.swift"; then
        echo "[AlicePet] Applying VRMMetalKit PerfectSync morph limits patch..."
        git -C "$EDIT_DIR" apply "$PATCH_MORPH"
    fi
    if [ -f "$PATCH_WARMUP" ] && ! grep -q "Step 1.5: Refresh world-space bind directions" "$EDIT_DIR/Sources/VRMMetalKit/SpringBoneComputeSystem.swift"; then
        echo "[AlicePet] Applying VRMMetalKit SpringBone warmup world bind directions patch..."
        git -C "$EDIT_DIR" apply "$PATCH_WARMUP"
    fi
fi

echo "[AlicePet] Building ($CONFIG)..."
swift build -c "$CONFIG"

BIN_DIR="$(swift build -c "$CONFIG" --show-bin-path)"
BIN_PATH="$BIN_DIR/AlicePet"

if [ ! -f "$BIN_PATH" ]; then
    echo "[AlicePet] Error: binary not found at $BIN_PATH" >&2
    exit 1
fi

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

cp "$BIN_PATH" "$APP_DIR/Contents/MacOS/AlicePet"
cp "$SCRIPT_DIR/Info.plist" "$APP_DIR/Contents/Info.plist"

for b in "$BIN_DIR"/VRMMetalKit_*.bundle; do
    if [ -d "$b" ]; then
        cp -R "$b" "$APP_DIR/Contents/Resources/"
    fi
done

codesign --force --deep --sign - "$APP_DIR"
echo "[AlicePet] Built $APP_DIR ($CONFIG)"
