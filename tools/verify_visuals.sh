#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$REPO_ROOT/AlicePet"
OUT_DIR="$REPO_ROOT/snapshots/latest"
BASELINE_DIR="$REPO_ROOT/snapshots/baseline"

CONFIG="release"
SKIP_BUILD=0
UPDATE_BASELINE=0
MODEL_ARG=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="${2:-release}"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --update-baseline)
            UPDATE_BASELINE=1
            shift
            ;;
        --model)
            MODEL_ARG=(--model "$2")
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--config release|debug] [--skip-build] [--update-baseline] [--model <path.vrm>]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$OUT_DIR" "$BASELINE_DIR"

cd "$PKG_DIR"
if [[ "$SKIP_BUILD" -eq 0 ]]; then
    echo "[Verify] Building AlicePet ($CONFIG)..."
    swift build -c "$CONFIG"
fi

BIN_DIR="$(swift build -c "$CONFIG" --show-bin-path)"
BIN_PATH="$BIN_DIR/AlicePet"
if [[ ! -x "$BIN_PATH" ]]; then
    echo "[Verify] Error: executable not found at $BIN_PATH" >&2
    exit 1
fi

run_case() {
    local filename="$1"
    shift
    local out_path="$OUT_DIR/$filename"
    rm -f "$out_path"
    echo "[Verify] Rendering $filename ..."
    "$BIN_PATH" ${MODEL_ARG[@]+"${MODEL_ARG[@]}"} --snapshot "$out_path" "$@"
    if [[ ! -s "$out_path" ]]; then
        echo "[Verify] FAIL: $filename was not created or is empty" >&2
        exit 1
    fi
}

run_case "01_stand_default.png" --pose stand
run_case "02_sit_peace.png" --pose sit --hand peace --expression happy
run_case "03_afk_book.png" --pose afk_book
run_case "04_afk_tea.png" --pose afk_tea
run_case "05_outfit_hairB.png" --pose stand --outfit-off hairband,ribbon --hair B --expression wink
run_case "06_arkit_face.png" --pose stand --test-arkit

SHEET_PATH="$OUT_DIR/contact_sheet.png"
echo "[Verify] Stitching contact sheet..."
swift "$SCRIPT_DIR/make_contact_sheet.swift" "$SHEET_PATH" \
    "1. Stand (Default)=$OUT_DIR/01_stand_default.png" \
    "2. Sit + Peace + Happy=$OUT_DIR/02_sit_peace.png" \
    "3. AFK Book=$OUT_DIR/03_afk_book.png" \
    "4. AFK Tea=$OUT_DIR/04_afk_tea.png" \
    "5. Outfit Off + Hair B + Wink=$OUT_DIR/05_outfit_hairB.png" \
    "6. ARKit Face Test=$OUT_DIR/06_arkit_face.png"

if [[ "$UPDATE_BASELINE" -eq 1 ]]; then
    cp "$OUT_DIR"/*.png "$BASELINE_DIR/"
    echo "[Verify] Updated baseline snapshots in $BASELINE_DIR"
fi

echo "[Verify] All 6 visual regression scenes passed!"
ls -lh "$OUT_DIR"/*.png
