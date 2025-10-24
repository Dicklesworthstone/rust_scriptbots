#!/usr/bin/env bash
set -euo pipefail

# Determine repository root (directory of this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine thread count with fallbacks (respects pre-set THREADS)
if [[ -z "${THREADS:-}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    THREADS="$(nproc)"
  elif command -v getconf >/dev/null 2>&1; then
    THREADS="$(getconf _NPROCESSORS_ONLN)"
  else
    THREADS="4"
  fi
fi

# Compile for native CPU and run in terminal (text) rendering mode
RUSTFLAGS="-C target-cpu=native" \
SCRIPTBOTS_MODE=terminal \
cargo run -j "${THREADS}" --manifest-path "${REPO_ROOT}/Cargo.toml" -p scriptbots-app --bin scriptbots-app --release -- --threads "${THREADS}"


