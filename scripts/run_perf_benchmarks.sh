#!/usr/bin/env bash
#
# Usage:
#   scripts/run_perf_benchmarks.sh [--renderer gui|bevy] [--scenario default|dense_agents|storm_event]
#                                   [--threads N] [--duration SECONDS]
#                                   [--output logs/perf/<scenario>_<renderer>.log]
#
# Requires: GPU-capable host (Metal/Vulkan/D3D12), latest cargo toolchain, and
# the corresponding scenario config files under docs/rendering_reference/configs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/perf"
SCENARIO="default"
RENDERER="gui"
THREADS="${THREADS:-8}"
DURATION=600
OUTPUT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --renderer)
      RENDERER="$2"
      shift 2
      ;;
    --scenario)
      SCENARIO="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOG_DIR}"

case "${SCENARIO}" in
  default)
    CLI_CONFIG=()
    ;;
  dense_agents|storm_event)
    CONFIG_PATH="${REPO_ROOT}/docs/rendering_reference/configs/${SCENARIO}.toml"
    if [[ ! -f "${CONFIG_PATH}" ]]; then
      echo "Scenario config not found: ${CONFIG_PATH}" >&2
      exit 1
    fi
    CLI_CONFIG=(--config "${CONFIG_PATH}")
    ;;
  *)
    echo "Unsupported scenario: ${SCENARIO}" >&2
    exit 1
    ;;
esac

case "${RENDERER}" in
  gui)
    FEATURES=(--features gui)
    MODE_ARGS=(--mode gui)
    ENV_HINTS=(
      SCRIPTBOTS_FORCE_GUI=1
    )
    ;;
  bevy)
    FEATURES=(--features bevy_render)
    MODE_ARGS=(--mode bevy)
    ENV_HINTS=(
      SCRIPTBOTS_MODE=bevy
    )
    ;;
  *)
    echo "Unsupported renderer: ${RENDERER}" >&2
    exit 1
    ;;
endcase

LOG_PATH="${OUTPUT:-${LOG_DIR}/${SCENARIO}_${RENDERER}.log}"

echo "============================================================"
echo "Scenario : ${SCENARIO}"
echo "Renderer : ${RENDERER}"
echo "Threads  : ${THREADS}"
echo "Duration : ${DURATION}s"
echo "Log file : ${LOG_PATH}"
echo "============================================================"

env \
  SB_DIAGNOSTICS=1 \
  RUST_LOG=info,scriptbots::bevy::diagnostics=info \
  SCRIPTBOTS_MAX_THREADS="${THREADS}" \
  "${ENV_HINTS[@]}" \
  timeout "${DURATION}" cargo run \
    --manifest-path "${REPO_ROOT}/Cargo.toml" \
    -p scriptbots-app \
    --bin scriptbots-app \
    --release \
    "${FEATURES[@]}" \
    -- \
    "${MODE_ARGS[@]}" \
    --threads "${THREADS}" \
    --rng-seed 424242 \
    "${CLI_CONFIG[@]}" \
    "${EXTRA_ARGS[@]}" \
    | tee "${LOG_PATH}"
