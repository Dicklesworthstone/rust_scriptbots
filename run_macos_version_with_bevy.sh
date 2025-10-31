#!/usr/bin/env bash
set -euo pipefail

# Detect macOS architecture to select the correct target triple
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
  TARGET="aarch64-apple-darwin"
else
  TARGET="x86_64-apple-darwin"
fi

# Isolate build artifacts per-platform/arch
export CARGO_TARGET_DIR="target-macos-$ARCH"

# Ensure no lingering cross-compilation flags leak in
unset CARGO_BUILD_TARGET || true
unset CARGO_ENCODED_RUSTFLAGS || true
unset RUSTFLAGS || true
unset RUSTC_LINKER || true

# Prefer the Metal backend on macOS for wgpu/Bevy
export WGPU_BACKEND="${WGPU_BACKEND:-metal}"
export WGPU_POWER_PREFERENCE="${WGPU_POWER_PREFERENCE:-high_performance}"

# Default Bevy presentation hints; tweak as needed for retina/high-DPI
export SB_WGPU_PRESENT_MODE=${SB_WGPU_PRESENT_MODE:-full}
export SB_WGPU_RES_SCALE=${SB_WGPU_RES_SCALE:-1.0}
export SB_WGPU_MAX_FPS=${SB_WGPU_MAX_FPS:-60}
export SB_WGPU_BLOOM=${SB_WGPU_BLOOM:-1}
export SB_WGPU_TONEMAP=${SB_WGPU_TONEMAP:-1}
export SB_WGPU_FOG=${SB_WGPU_FOG:-low}

# Thread budget (mirror Windows default of 8 unless overridden)
JOBS="$( (sysctl -n hw.ncpu 2>/dev/null) || (getconf _NPROCESSORS_ONLN 2>/dev/null) || echo 8 )"
if [[ -z "${THREADS:-}" ]]; then
  THREADS=8
else
  THREADS="${THREADS}"
fi

# Encourage Bevy renderer selection even if auto mode is used elsewhere
export SCRIPTBOTS_MODE="${SCRIPTBOTS_MODE:-bevy}"

# Launch ScriptBots with the Bevy renderer
cargo run -p scriptbots-app --bin scriptbots-app \
  --release --target "$TARGET" -j "$JOBS" \
  --features bevy_render \
  -- --mode bevy --threads "$THREADS"
