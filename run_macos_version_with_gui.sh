#!/usr/bin/env bash
set -euo pipefail

# Auto-detect macOS architecture and set target triple
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
	TARGET="aarch64-apple-darwin"
else
	TARGET="x86_64-apple-darwin"
fi

# Isolate build artifacts per-platform/arch
export CARGO_TARGET_DIR="target-macos-$ARCH"

# Ensure we don't inherit any cross-compile or custom linker flags
unset CARGO_BUILD_TARGET || true
unset CARGO_ENCODED_RUSTFLAGS || true
unset RUSTFLAGS || true
unset RUSTC_LINKER || true

# Prefer the Metal backend on macOS for wgpu
export WGPU_BACKEND=metal
export SB_WGPU_PRESENT_MODE=full

# Use all CPU cores for faster builds
JOBS="$( (sysctl -n hw.ncpu 2>/dev/null) || (getconf _NPROCESSORS_ONLN 2>/dev/null) || echo 8 )"

# Launch ScriptBots GUI in release with tuned threads
cargo run -p scriptbots-app --bin scriptbots-app --release --target "$TARGET" -j "$JOBS" --features gui -- --mode gui --threads 8
