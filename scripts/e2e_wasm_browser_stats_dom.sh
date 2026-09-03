#!/usr/bin/env bash
# E2E regression proof for WASM browser stats refresh through the real DOM (bd-2z0.12.6).
#
# Proves:
#   1. Real browser DOM execution in headless Chromium.
#   2. Consumes production camelCase snapshot contract after the 500ms stats boundary.
#   3. Asserts population, average energy, average health, and TPS text on production DOM IDs.
#   4. Accumulates >= 2 tick/time windows so TPS is proven derived from deltas.
#   5. Snake_case negative control proves unhandled exception and loop freeze on legacy contract.
#   6. Scheduling-path negative control proves stats do not fire before 500ms.
#   7. Retains browser console logs and emits structured JSON evidence.

set -euo pipefail

fail() {
  printf 'wasm-browser-stats-dom-e2e: %s\n' "$1" >&2
  exit 1
}

command -v bun >/dev/null 2>&1 || fail "bun is required to run the playwright browser harness"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

printf 'wasm-browser-stats-dom-e2e: executing real-browser DOM stats refresh proof...\n'

log_file="$(mktemp -t scriptbots_dom_stats_XXXXXX.log)"
trap 'rm -f "$log_file"' EXIT

bun crates/scriptbots-web/web/tests/stats_dom.test.js 2>&1 | tee "$log_file"

evidence="$(grep -o '{"schema":"scriptbots.browser-stats-dom.v1".*}' "$log_file" | tail -1)"
[ -n "$evidence" ] || fail "no scriptbots.browser-stats-dom.v1 structured evidence line was emitted"

# Check required fields in the structured evidence
for field in \
  '"schema":"scriptbots.browser-stats-dom.v1"' \
  '"status":"pass"' \
  '"cases_passed":3' \
  '"cases_failed":0' \
  '"camelCasePositive":' \
  '"snakeCaseNegative":' \
  '"schedulingNegative":' \
  '"deltaTicksWindow2":' \
  '"observations":'; do
  case "$evidence" in
    *"$field"*) ;;
    *) fail "evidence line is missing required token: $field" ;;
  esac
done

printf 'wasm-browser-stats-dom-e2e: SUCCESS: real browser DOM stats refresh verified with 3/3 passed cases.\n'
