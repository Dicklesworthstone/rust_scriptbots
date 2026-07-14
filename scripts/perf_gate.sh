#!/usr/bin/env bash
# Run the deterministic ScriptBots performance budget harness.
#
# Usage:
#   scripts/perf_gate.sh [--mode short|full] [--output-dir PATH]
#                        [--baseline PATH] [--ticks N]
#                        [--record-baseline --justification TEXT]
#                        [--synthetic-sleep-us N] [--self-test]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="short"
OUTPUT_DIR=""
BASELINE="${REPO_ROOT}/ci/fixtures/perf_baseline.json"
BASELINE_EXPLICIT=0
TICKS=""
RECORD_BASELINE=0
JUSTIFICATION=""
SYNTHETIC_SLEEP_US=""
SELF_TEST=0

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
    --mode)
      require_value "$1" "${2:-}"
      MODE="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$1" "${2:-}"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --baseline)
      require_value "$1" "${2:-}"
      BASELINE="$2"
      BASELINE_EXPLICIT=1
      shift 2
      ;;
    --ticks)
      require_value "$1" "${2:-}"
      TICKS="$2"
      shift 2
      ;;
    --record-baseline)
      RECORD_BASELINE=1
      shift
      ;;
    --justification)
      require_value "$1" "${2:-}"
      JUSTIFICATION="$2"
      shift 2
      ;;
    --synthetic-sleep-us)
      require_value "$1" "${2:-}"
      SYNTHETIC_SLEEP_US="$2"
      shift 2
      ;;
    --self-test)
      SELF_TEST=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${MODE}" in
  short|full) ;;
  *)
    echo "--mode must be short or full, got: ${MODE}" >&2
    exit 2
    ;;
esac

if [[ -n "${TICKS}" && ! "${TICKS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--ticks must be a positive integer, got: ${TICKS}" >&2
  exit 2
fi

if [[ -n "${SYNTHETIC_SLEEP_US}" && ! "${SYNTHETIC_SLEEP_US}" =~ ^[0-9]+$ ]]; then
  echo "--synthetic-sleep-us must be a non-negative integer, got: ${SYNTHETIC_SLEEP_US}" >&2
  exit 2
fi

if (( RECORD_BASELINE == 1 )); then
  if [[ "${MODE}" != "full" ]]; then
    echo "--record-baseline requires --mode full" >&2
    exit 2
  fi
  if [[ -z "${JUSTIFICATION//[[:space:]]/}" ]]; then
    echo "--record-baseline requires a non-empty --justification" >&2
    exit 2
  fi
  if [[ -n "${SYNTHETIC_SLEEP_US}" && "${SYNTHETIC_SLEEP_US}" != "0" ]]; then
    echo "A synthetic delay can never be recorded as a baseline" >&2
    exit 2
  fi
fi

if (( RECORD_BASELINE == 0 )) && [[ -n "${JUSTIFICATION}" ]]; then
  echo "--justification is only valid with --record-baseline" >&2
  exit 2
fi

if (( RECORD_BASELINE == 1 && BASELINE_EXPLICIT == 1 )); then
  echo "--record-baseline cannot be combined with --baseline" >&2
  exit 2
fi

if (( SELF_TEST == 1 )) && {
  (( RECORD_BASELINE == 1 || BASELINE_EXPLICIT == 1 )) \
    || [[ -n "${TICKS}" || -n "${SYNTHETIC_SLEEP_US}" || -n "${JUSTIFICATION}" ]];
}; then
  echo "--self-test cannot be combined with baseline, tick, justification, or synthetic-delay options" >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/target/perf-gate/$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi

: "${SCRIPTBOTS_MAX_THREADS:=4}"
: "${RAYON_NUM_THREADS:=${SCRIPTBOTS_MAX_THREADS}}"
if [[ ! "${SCRIPTBOTS_MAX_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SCRIPTBOTS_MAX_THREADS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${RAYON_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RAYON_NUM_THREADS must be a positive integer" >&2
  exit 2
fi
export SCRIPTBOTS_MAX_THREADS RAYON_NUM_THREADS

if [[ "${SCRIPTBOTS_MAX_THREADS}" != "${RAYON_NUM_THREADS}" ]]; then
  echo "SCRIPTBOTS_MAX_THREADS and RAYON_NUM_THREADS must match for an exact machine class" >&2
  exit 2
fi

HOST_TARGET="${SCRIPTBOTS_PERF_BUILD_TARGET:-$(rustc -vV | sed -n 's/^host: //p')}"
if [[ -z "${HOST_TARGET}" ]]; then
  echo "Unable to determine the executable host target from rustc -vV" >&2
  exit 2
fi
export SCRIPTBOTS_PERF_BUILD_TARGET="${HOST_TARGET}"

HARNESS_ARGS=(
  --perf-gate
  --mode "${MODE}"
  --output-dir "${OUTPUT_DIR}"
)

if (( SELF_TEST == 1 )); then
  HARNESS_ARGS+=(--self-test)
elif (( RECORD_BASELINE == 1 )); then
  HARNESS_ARGS+=(--record-baseline --justification "${JUSTIFICATION}")
else
  HARNESS_ARGS+=(--baseline "${BASELINE}")
fi

if [[ -n "${TICKS}" ]]; then
  HARNESS_ARGS+=(--ticks "${TICKS}")
fi
if [[ -n "${SYNTHETIC_SLEEP_US}" ]]; then
  HARNESS_ARGS+=(--synthetic-sleep-us "${SYNTHETIC_SLEEP_US}")
fi

echo "Performance artifacts: ${OUTPUT_DIR}"
cd "${REPO_ROOT}"
cargo bench --locked --target "${HOST_TARGET}" -p scriptbots-core --bench world_bench -- "${HARNESS_ARGS[@]}"
