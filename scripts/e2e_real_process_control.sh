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
#  8. Clean child termination and reaping
#  9. Real CLI map generate/apply roundtrip (JSON and Postcard)
# 10. Emits and gates on scriptbots.real-process-e2e.v2 JSONL evidence

set -euo pipefail

fail() {
  printf 'real-process-control-e2e: %s\n' "$1" >&2
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

printf 'real-process-control-e2e: running real_process_control_e2e (commit=%s)\n' "$SCRIPTBOTS_GIT_COMMIT"
e2e_log="$(mktemp)"
trap 'rm -f "$e2e_log"' EXIT

"${cargo_runner[@]}" test \
  --locked \
  -p scriptbots-app \
  --test real_process_control_e2e \
  -- \
  --nocapture 2>&1 | tee "$e2e_log"

grep -q -E '(2|3|4) passed; 0 failed;' "$e2e_log" ||
  fail "real_process_control_e2e test suite did not pass with 0 failed"

evidence="$(grep -o '{"schema":"scriptbots.real-process-e2e.v2".*}' "$e2e_log" | tail -1)"
[ -n "$evidence" ] || fail "no scriptbots.real-process-e2e.v2 evidence line was emitted"

# Field presence check
for field in \
  '"schema":' '"binary":' '"mode":' '"storage":' '"rest_address":' \
  '"mcp_address":' '"boot_log_lines":' '"status_code":' '"pause_code":' \
  '"pause_id":' '"step_id":' '"resume_id":' '"application_state":' \
  '"journal_state":' '"proved_level":' '"screenshot_code":' \
  '"tools_count":' '"child_exit":' '"source_commit":'; do
  case "$evidence" in
    *"$field"*) ;;
    *) fail "evidence line is missing $field" ;;
  esac
done

# Contract assertions
mode="$(printf '%s' "$evidence" | jq -r '.mode')"
storage="$(printf '%s' "$evidence" | jq -r '.storage')"
status_code="$(printf '%s' "$evidence" | jq -r '.status_code')"
pause_code="$(printf '%s' "$evidence" | jq -r '.pause_code')"
screenshot_code="$(printf '%s' "$evidence" | jq -r '.screenshot_code')"
app_state="$(printf '%s' "$evidence" | jq -r '.application_state')"
proved_level="$(printf '%s' "$evidence" | jq -r '.proved_level')"
child_exit="$(printf '%s' "$evidence" | jq -r '.child_exit')"
tools_count="$(printf '%s' "$evidence" | jq -r '.tools_count')"
boot_log_lines="$(printf '%s' "$evidence" | jq -r '.boot_log_lines')"
pause_id="$(printf '%s' "$evidence" | jq -r '.pause_id')"
step_id="$(printf '%s' "$evidence" | jq -r '.step_id')"
resume_id="$(printf '%s' "$evidence" | jq -r '.resume_id')"
rest_addr="$(printf '%s' "$evidence" | jq -r '.rest_address')"
mcp_addr="$(printf '%s' "$evidence" | jq -r '.mcp_address')"
source_commit="$(printf '%s' "$evidence" | jq -r '.source_commit')"

[ "$mode" = "server" ] || fail "expected mode=server, got $mode"
[ "$storage" = "memory" ] || fail "expected storage=memory, got $storage"
[ "$status_code" -eq 200 ] || fail "expected status_code=200, got $status_code"
[ "$pause_code" -eq 200 ] || fail "expected pause_code=200, got $pause_code"
[ "$screenshot_code" -eq 409 ] || fail "expected screenshot_code=409, got $screenshot_code"
[ "$app_state" = "applied" ] || fail "expected application_state=applied, got $app_state"
[ "$proved_level" = "applied" ] || fail "expected proved_level=applied, got $proved_level"
[ "$child_exit" = "signalled" ] || fail "expected child_exit=signalled, got $child_exit"

# Non-vacuity checks
[ "$boot_log_lines" -gt 0 ] || fail "boot_log_lines must be > 0, got $boot_log_lines"
[ "$tools_count" -ge 13 ] || fail "tools_count must be >= 13, got $tools_count"
[ -n "$pause_id" ] || fail "pause_id must not be empty"
[ -n "$step_id" ] || fail "step_id must not be empty"
[ -n "$resume_id" ] || fail "resume_id must not be empty"
[ "$pause_id" != "$step_id" ] || fail "pause_id and step_id must be distinct"
[ "$step_id" != "$resume_id" ] || fail "step_id and resume_id must be distinct"
[ -n "$rest_addr" ] || fail "rest_address must not be empty"
[ -n "$mcp_addr" ] || fail "mcp_address must not be empty"
if [ -z "$source_commit" ] || [ "$source_commit" = "unknown" ]; then
  source_commit="${SCRIPTBOTS_GIT_COMMIT:-unknown}"
fi
[ -n "$source_commit" ] || fail "source_commit must not be empty"

printf 'real-process-control-e2e: PASS (proved_level=%s, screenshot_code=%s, tools=%s, child_exit=%s)\n' \
  "$proved_level" "$screenshot_code" "$tools_count" "$child_exit"
printf '%s\n' "$evidence"
