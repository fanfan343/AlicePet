#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="/Users/funnyliu/Downloads/VRM项目/AlicePet"
APP_DIR="/Users/funnyliu/Downloads/VRM项目/AlicePet.app"

cd "$PKG_DIR"
swift build

BIN_PATH=".build/debug/AlicePet"
BUNDLE_SRC=$(find .build -name "VRMMetalKit_VRMMetalKit.bundle" -type d | head -n 1)

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

cp "$BIN_PATH" "$APP_DIR/Contents/MacOS/AlicePet"
cp "$PKG_DIR/Info.plist" "$APP_DIR/Contents/Info.plist"
if [ -n "${BUNDLE_SRC:-}" ]; then
    cp -R "$BUNDLE_SRC" "$APP_DIR/Contents/Resources/"
fi

codesign --force --deep --sign - "$APP_DIR"
echo "Built $APP_DIR"
