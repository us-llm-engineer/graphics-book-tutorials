#!/bin/bash
# Opens the interactive viewer.   scripts/view.sh [scene] [megakernel|wavefront]
# Without a display (no WSLg/X11/Wayland) it renders one frame with the overlay to /tmp/wfviewer-frame.png instead.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR="${WF_BUILD_DIR:-build}"
scene="${1:-divergence}"
mode="${2:-wavefront}"
if [[ ! -x "$BUILD_DIR/wfviewer" ]]; then
    echo "Error: $BUILD_DIR/wfviewer not found (cmake --build $BUILD_DIR --target wfviewer, or set WF_BUILD_DIR)" >&2
    exit 1
fi
if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    echo "No display found: writing one frame to /tmp/wfviewer-frame.png instead of opening a window." >&2
    exec "$BUILD_DIR/wfviewer" --selftest 24 /tmp/wfviewer-frame.png --scene "$scene" --mode "$mode"
fi
exec "$BUILD_DIR/wfviewer" --scene "$scene" --mode "$mode"
