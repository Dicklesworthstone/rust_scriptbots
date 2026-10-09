#!/usr/bin/env bash
# e2e_real_process_control.sh — real-process control E2E acceptance probe (bd-0n87).
#
# Gates on the real-process control-plane lifecycle against the shipped binary:
#  1. Real child process spawned with `--mode server --storage memory`
#  2. OS-assigned REST and FastMCP ports announced on stderr
#  3. REST /api/status live state read (status 200)
#  4. Two-axis command envelope reaching applied state for pause, step, resume
#  5. Step advances tick by exactly +1; pause freezes ticks
#  6. REST /api/screenshot/ascii refuses with 409 in unpresented server mode
#  7. FastMCP HTTP endpoint: initialize, notifications/initialized, tools/list,
#     tools/call get_status, map_generate, map_apply, and unknown tool error
#  8. Graceful ordered shutdown and successful child reaping
#  9. Real CLI map generate/apply roundtrip (JSON and Postcard)
# 10. Source-bound process observations and durable bundle readback in pinned DSR

set -euo pipefail

fail() {
  printf 'real-process-control-e2e: %s\n' "$1" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

if [[ ${1:-} == --verify-evidence ]]; then
  [[ $# == 3 ]] || fail "usage: --verify-evidence DIRECTORY EXPECTED_COMMIT"
  exec bash scripts/dsr_verify.sh --verify-evidence "$2" "$3" connectivity
fi
[[ $# == 2 && $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || fail "usage: PROFILE UNIQUE_VERSION"
[[ ${DSR_CONFIG_DIR:-} = /* ]] || fail "pinned DSR configuration required"
profile="$DSR_CONFIG_DIR/repos.d/$1.yaml"
[[ -f "$profile" && $(yq -r '.env.SCRIPTBOTS_VERIFY_LANE' "$profile") == connectivity ]] || fail "profile must run the complete connectivity lane"
exec bash scripts/dsr_verify.sh --run "$1" "$2"
