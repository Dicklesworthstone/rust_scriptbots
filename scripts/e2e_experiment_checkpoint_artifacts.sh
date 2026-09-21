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
#  9. Clean server shutdown and child termination/cleanup
# 10. Emits and gates on scriptbots.e2e-experiment-checkpoint-artifact.v1 JSONL evidence

set -euo pipefail

fail() {
  printf 'e2e-experiment-checkpoint-artifacts: %s\n' "$1" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

if [[ "${CI:-}" == "true" ]]; then
  cargo_runner=(cargo)
else
  command -v rch >/dev/null 2>&1 ||
    fail "rch is required outside CI; local Cargo fallback is forbidden"
  cargo_runner=(rch exec -- cargo)
fi

export SCRIPTBOTS_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

printf 'e2e-experiment-checkpoint-artifacts: running real_process_experiments_checkpoints_artifacts_e2e (commit=%s)\n' "$SCRIPTBOTS_GIT_COMMIT"
e2e_log="$(mktemp)"
trap 'rm -f "$e2e_log"' EXIT

"${cargo_runner[@]}" test \
  --locked \
  -p scriptbots-app \
  --test real_process_control_e2e \
  -- real_process_experiments_checkpoints_artifacts_e2e \
  --nocapture 2>&1 | tee "$e2e_log"

grep -q -E '(1|2|3) passed; 0 failed;' "$e2e_log" ||
  fail "real_process_experiments_checkpoints_artifacts_e2e test suite did not pass with 0 failed"

evidence="$(grep -o '{"schema":"scriptbots.e2e-experiment-checkpoint-artifact.v1".*}' "$e2e_log" | tail -1)"
[ -n "$evidence" ] || fail "no scriptbots.e2e-experiment-checkpoint-artifact.v1 evidence line was emitted"

# Field presence check
for field in \
  '"schema":' '"status":' '"binary":' '"mode":' '"storage":' \
  '"tools_count":' '"experiments_count":' '"checkpoints_count":' \
  '"artifacts_count":' '"injected_faults_handled":' '"checksums_verified":' \
  '"cleanup_verified":' '"source_commit":'; do
  case "$evidence" in
    *"$field"*) ;;
    *) fail "evidence line is missing $field" ;;
  esac
done

# Contract assertions
status="$(printf '%s' "$evidence" | jq -r '.status')"
mode="$(printf '%s' "$evidence" | jq -r '.mode')"
storage="$(printf '%s' "$evidence" | jq -r '.storage')"
tools_count="$(printf '%s' "$evidence" | jq -r '.tools_count')"
experiments_count="$(printf '%s' "$evidence" | jq -r '.experiments_count')"
checkpoints_count="$(printf '%s' "$evidence" | jq -r '.checkpoints_count')"
artifacts_count="$(printf '%s' "$evidence" | jq -r '.artifacts_count')"
injected_faults="$(printf '%s' "$evidence" | jq -r '.injected_faults_handled')"
checksums_verified="$(printf '%s' "$evidence" | jq -r '.checksums_verified')"
cleanup_verified="$(printf '%s' "$evidence" | jq -r '.cleanup_verified')"
source_commit="$(printf '%s' "$evidence" | jq -r '.source_commit')"

[ "$status" = "pass" ] || fail "expected status=pass, got $status"
[ "$mode" = "server" ] || fail "expected mode=server, got $mode"
[ "$storage" = "file" ] || fail "expected storage=file, got $storage"
[ "$tools_count" -eq 30 ] || fail "expected tools_count=30, got $tools_count"
[ "$experiments_count" -ge 1 ] || fail "expected experiments_count>=1, got $experiments_count"
[ "$checkpoints_count" -ge 1 ] || fail "expected checkpoints_count>=1, got $checkpoints_count"
[ "$artifacts_count" -ge 1 ] || fail "expected artifacts_count>=1, got $artifacts_count"
[ "$injected_faults" -ge 5 ] || fail "expected injected_faults_handled>=5, got $injected_faults"
[ "$checksums_verified" = "true" ] || fail "expected checksums_verified=true, got $checksums_verified"
[ "$cleanup_verified" = "true" ] || fail "expected cleanup_verified=true, got $cleanup_verified"

if [ -z "$source_commit" ] || [ "$source_commit" = "unknown" ]; then
  source_commit="${SCRIPTBOTS_GIT_COMMIT:-unknown}"
fi
[ -n "$source_commit" ] || fail "source_commit must not be empty"

printf 'e2e-experiment-checkpoint-artifacts: PASS (status=%s, tools=%s, faults=%s, checksums_verified=%s, cleanup_verified=%s)\n' \
  "$status" "$tools_count" "$injected_faults" "$checksums_verified" "$cleanup_verified"
printf '%s\n' "$evidence"
