#!/usr/bin/env bash
#
# Usage:
#   scripts/run_perf_benchmarks.sh [--renderer gui|bevy] [--scenario default|dense_agents|storm_event]
#                                   [--threads N] [--duration SECONDS]
#                                   [--output logs/perf/<scenario>_<renderer>.log]
#                                   [--dry-run] [-- extra application arguments]
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
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "${option} requires a value" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --renderer)
      require_value "$1" "${2:-}"
      RENDERER="$2"
      shift 2
      ;;
    --scenario)
      require_value "$1" "${2:-}"
      SCENARIO="$2"
      shift 2
      ;;
    --threads)
      require_value "$1" "${2:-}"
      THREADS="$2"
      shift 2
      ;;
    --duration)
      require_value "$1" "${2:-}"
      DURATION="$2"
      shift 2
      ;;
    --output)
      require_value "$1" "${2:-}"
      OUTPUT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "${THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--threads must be a positive integer, got: ${THREADS}" >&2
  exit 2
fi

if [[ ! "${DURATION}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--duration must be a positive integer, got: ${DURATION}" >&2
  exit 2
fi

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
    exit 2
    ;;
esac

case "${RENDERER}" in
  gui)
    FEATURES=(--features gui)
    MODE_ARGS=(--mode gui)
    LOG_FILTER="info,scriptbots_render=info"
    ENV_HINTS=(
      SCRIPTBOTS_FORCE_GUI=1
    )
    ;;
  bevy)
    FEATURES=(--features bevy_render)
    MODE_ARGS=(--mode bevy)
    LOG_FILTER="info,scriptbots_bevy=info"
    ENV_HINTS=(
      SB_DIAGNOSTICS=1
    )
    ;;
  *)
    echo "Unsupported renderer: ${RENDERER}" >&2
    exit 2
    ;;
esac

if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_BIN="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN="$(command -v gtimeout)"
else
  echo "GNU timeout is required (install coreutils for gtimeout on macOS)" >&2
  exit 2
fi

LOG_PATH="${OUTPUT:-${LOG_DIR}/${SCENARIO}_${RENDERER}.log}"

ENVIRONMENT=(
  "RUST_LOG=${LOG_FILTER}"
  "SCRIPTBOTS_MAX_THREADS=${THREADS}"
  "${ENV_HINTS[@]}"
)

COMMAND=(
  "${TIMEOUT_BIN}"
  --signal=INT
  --kill-after=10s
  "${DURATION}"
  cargo run
  --manifest-path "${REPO_ROOT}/Cargo.toml"
  -p scriptbots-app
  --bin scriptbots-app
  --release
  --locked
  "${FEATURES[@]}"
  --
  "${MODE_ARGS[@]}"
  --threads "${THREADS}"
  --rng-seed 424242
  "${CLI_CONFIG[@]}"
  "${EXTRA_ARGS[@]}"
)

echo "============================================================"
echo "Scenario : ${SCENARIO}"
echo "Renderer : ${RENDERER}"
echo "Threads  : ${THREADS}"
echo "Duration : ${DURATION}s"
echo "Log file : ${LOG_PATH}"
echo "============================================================"

if (( DRY_RUN )); then
  printf 'env'
  printf ' %q' "${ENVIRONMENT[@]}" "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$(dirname "${LOG_PATH}")"

set +e
env "${ENVIRONMENT[@]}" "${COMMAND[@]}" 2>&1 | tee "${LOG_PATH}"
PIPELINE_STATUS=("${PIPESTATUS[@]}")
set -e

if (( PIPELINE_STATUS[1] != 0 )); then
  echo "Failed to write benchmark log: ${LOG_PATH}" >&2
  exit "${PIPELINE_STATUS[1]}"
fi

case "${PIPELINE_STATUS[0]}" in
  0)
    ;;
  124)
    echo "Benchmark duration reached cleanly after ${DURATION}s."
    ;;
  *)
    echo "Renderer benchmark failed with status ${PIPELINE_STATUS[0]}" >&2
    exit "${PIPELINE_STATUS[0]}"
    ;;
esac
