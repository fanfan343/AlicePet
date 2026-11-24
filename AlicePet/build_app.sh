#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="/Users/funnyliu/Downloads/VRM项目/AlicePet"
APP_DIR="/Users/funnyliu/Downloads/VRM项目/AlicePet.app"

cd "$PKG_DIR"
swift build

BIN_PATH=".build/debug/AlicePet"
BIN_DIR="$(dirname "$(find .build/debug -maxdepth 1 -name AlicePet -type f 2>/dev/null || echo .build/debug/AlicePet)")"

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

cp "$BIN_PATH" "$APP_DIR/Contents/MacOS/AlicePet"
cp "$PKG_DIR/Info.plist" "$APP_DIR/Contents/Info.plist"
for b in $(find .build -name "VRMMetalKit_*.bundle" -type d); do
    cp -R "$b" "$APP_DIR/Contents/Resources/"
done

codesign --force --deep --sign - "$APP_DIR"
echo "Built $APP_DIR"
