#!/usr/bin/env bash
# Browser DOM controls (bd-2z0.12.6) and real package smoke (bd-azi3).
# Execute with already provisioned runtime tools inside the pinned DSR profile.
#
# Proves:
#   1. Real browser DOM execution in headless Chromium.
#   2. Consumes production camelCase snapshot contract after the 500ms stats boundary.
#   3. Asserts population, average energy, average health, and TPS text on production DOM IDs.
#   4. Accumulates >= 2 tick/time windows so TPS is proven derived from deltas.
#   5. Snake_case negative control proves unhandled exception and loop freeze on legacy contract.
#   6. Scheduling-path negative control proves stats do not fire before 500ms.
#   7. Retains browser console logs and emits structured JSON evidence.
#   8. Loads actual generated package glue/WASM and refuses corrupt WASM at bootstrap.

set -euo pipefail

fail() {
  printf 'wasm-browser-stats-dom-e2e: %s\n' "$1" >&2
  exit 1
}

command -v bun >/dev/null 2>&1 || fail "bun is required to run the playwright browser harness"
command -v jq >/dev/null 2>&1 || fail "jq is required to inspect browser observations"
[[ -n "${SCRIPTBOTS_WASM_PACKAGE_DIR:-}" ]] || fail "SCRIPTBOTS_WASM_PACKAGE_DIR must select the freshly produced package"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

printf 'wasm-browser-stats-dom-e2e: executing real-browser DOM stats refresh proof...\n'

if [[ -n "${SCRIPTBOTS_WASM_PROOF_DIR:-}" ]]; then
  [[ -d "$SCRIPTBOTS_WASM_PROOF_DIR" ]] || fail "SCRIPTBOTS_WASM_PROOF_DIR must be an existing proof directory"
  log_file="$(mktemp "$SCRIPTBOTS_WASM_PROOF_DIR/browser-stats.XXXXXX.log")"
else
  log_file="$(mktemp -t scriptbots_dom_stats_XXXXXX.log)"
fi
printf 'wasm-browser-stats-dom-e2e: retained raw log: %s\n' "$log_file"

bun crates/scriptbots-web/web/tests/stats_dom.test.js 2>&1 | tee "$log_file"

evidence="$(grep -o '{"schema":"scriptbots.browser-stats-dom.v1".*}' "$log_file" | tail -1)"
[ -n "$evidence" ] || fail "no scriptbots.browser-stats-dom.v1 structured evidence line was emitted"

# Check required fields in the structured evidence
for field in \
  '"schema":"scriptbots.browser-stats-dom.v1"' \
  '"status":"pass"' \
  '"cases_failed":0' \
  '"camelCasePositive":' \
  '"snakeCaseNegative":' \
  '"schedulingNegative":' \
  '"realPackage":' \
  '"deltaTicksWindow2":' \
  '"observations":'; do
  case "$evidence" in
    *"$field"*) ;;
    *) fail "evidence line is missing required token: $field" ;;
  esac
done

jq -e '
  .schema == "scriptbots.browser-stats-dom.v1" and .status == "pass" and .cases_failed == 0
  and ([.observations.camelCasePositive, .observations.snakeCaseNegative,
        .observations.schedulingNegative, .observations.realPackage] as $cases
       | .cases_passed == ($cases | length) and all($cases[]; .passed == true))
  and .observations.camelCasePositive.deltaTicksWindow2 > 0
  and (.observations.realPackage.loadedFiles | length == 2 and all(.[]; .status == 200 and .bytes > 0))
  and .observations.realPackage.corruptedWasmNegative.passed == true
  and .observations.realPackage.corruptedWasmNegative.requests == 1
' <<< "$evidence" >/dev/null || fail "browser observations are incomplete"

printf 'wasm-browser-stats-dom-e2e: SUCCESS: all required DOM controls and real package case passed.\n'
