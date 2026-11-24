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
