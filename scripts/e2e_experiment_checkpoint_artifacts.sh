#!/usr/bin/env bash
# e2e_experiment_checkpoint_artifacts.sh — real-process experiment, checkpoint, and artifact E2E acceptance probe (bd-2z0.12.2).
#
# Gates on the real-process experiment, checkpoint, and artifact lifecycle against the shipped binary:
#  1. Real child process spawned with `--mode server --storage file`
#  2. OS-assigned REST and FastMCP ports announced on stderr
#  3. REST /api/v1/version and /api/v1/schema discovery
#  4. FastMCP initialize, notifications/initialized, and tools/list (all 30 tools)
#  5. Fault injection: empty variants (400), invalid brain family (400), non-existent experiment (404),
#     path traversal artifact download (400/404), missing artifact download (404)
#  6. Experiment lifecycle: create, status, cancel, resume, list
#  7. Checkpoint and artifact lifecycle: create, get, list, download raw bytes
#  8. Byte-level checksum verification (SHA-256 and BLAKE3) of downloaded artifact
#  9. Graceful ordered shutdown, durable SQL readback and separate CLI bundle verification
# 10. Source-bound process observations within the complete pinned DSR connectivity lane

set -euo pipefail

fail() {
  printf 'e2e-experiment-checkpoint-artifacts: %s\n' "$1" >&2
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
