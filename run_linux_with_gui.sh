#!/usr/bin/env bash
set -euo pipefail

# Determine repository root (directory of this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine thread count with fallbacks; cap default to 8 for smoother GUI
if [[ -z "${THREADS:-}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    THREADS="$(nproc)"
  elif command -v getconf >/dev/null 2>&1; then
    THREADS="$(getconf _NPROCESSORS_ONLN)"
  else
    THREADS="8"
  fi
  if [[ "$THREADS" =~ ^[0-9]+$ ]] && (( THREADS > 8 )); then
    THREADS=8
  fi
fi

# Prefer high-performance adapter and Vulkan backend on Linux (fallback to GL)
export WGPU_POWER_PREFERENCE="${WGPU_POWER_PREFERENCE:-high_performance}"
if [[ -z "${WGPU_BACKEND:-}" ]]; then
  if command -v vulkaninfo >/dev/null 2>&1 || [[ -d "/usr/share/vulkan/icd.d" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    export WGPU_BACKEND=vulkan
  else
    export WGPU_BACKEND=gl
  fi
fi

# Hint application to choose GUI path even in auto-detect scenarios
export SCRIPTBOTS_FORCE_GUI="${SCRIPTBOTS_FORCE_GUI:-1}"
# Sane wgpu defaults
export SB_WGPU_PRESENT_MODE=${SB_WGPU_PRESENT_MODE:-full}
export SB_WGPU_RES_SCALE=${SB_WGPU_RES_SCALE:-1.0}
export SB_WGPU_MAX_FPS=${SB_WGPU_MAX_FPS:-60}
export SB_WGPU_BLOOM=${SB_WGPU_BLOOM:-1}
export SB_WGPU_TONEMAP=${SB_WGPU_TONEMAP:-1}
export SB_WGPU_FOG=${SB_WGPU_FOG:-low}

# Optimize for local CPU
export RUSTFLAGS="-C target-cpu=native"

# Build and run GUI
cargo run -j "${THREADS}" --manifest-path "${REPO_ROOT}/Cargo.toml" -p scriptbots-app --bin scriptbots-app --release --features gui -- --mode gui --threads "${THREADS}"


