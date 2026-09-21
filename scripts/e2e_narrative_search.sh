#!/usr/bin/env bash
# End-to-end narrative search cross-surface parity acceptance probe (bd-16g.2.7).
#
# Verifies:
# 1. Direct storage reader facade (`execute_narrative_search` / `execute_narrative_around`).
# 2. REST endpoints `GET /api/narrative/search` and `GET /api/narrative/around/{tick}`.
# 3. FastMCP tool execution for `narrative_search` and `narrative_around`.
# 4. Control CLI commands (`control_cli narrative search` / `control_cli narrative around` with `--db` and `--base-url`).
# 5. Bit-identical parity across all surfaces for identical queries.
# 6. Input validation, bounds checking, pagination caps, and structured error envelopes.
# 7. Validates structured JSONL evidence schemas.

set -euo pipefail

fail() {
  printf 'narrative-search-e2e: %s\n' "$1" >&2
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

printf 'narrative-search-e2e: running narrative_search_e2e test suite (commit=%s)\n' "$SCRIPTBOTS_GIT_COMMIT"
e2e_log="$(mktemp)"
trap 'rm -f "$e2e_log"' EXIT

"${cargo_runner[@]}" test \
  --locked \
  -p scriptbots-app \
  --test narrative_search_e2e \
  -- \
  --nocapture \
  --test-threads=1 2>&1 | tee "$e2e_log"

grep -q 'test result: ok\. 5 passed; 0 failed;' "$e2e_log" ||
  fail "narrative_search_e2e test suite did not pass with 5 passed and 0 failed"

printf 'narrative-search-e2e: parsing and validating structured proof evidence...\n'

python3 -c '
import json, re, sys

log_path = sys.argv[1]
seen_phases = set()
with open(log_path) as f:
    for line in f:
        m = re.search(r"E2E_EVIDENCE:\s*(\{.*\})", line)
        if m:
            data = json.loads(m.group(1))
            if data.get("schema") == "scriptbots.narrative-search.e2e-evidence.v1":
                phase = data.get("phase")
                seen_phases.add(phase)
                surfaces = data.get("surfaces")
                print(f"Verified narrative proof evidence: phase={phase} surfaces={surfaces}")

required_phases = {"cross_surface_parity_confirmed", "around_cross_surface_parity_confirmed"}
missing = required_phases - seen_phases
if missing:
    print(f"Missing required proof phases: {missing}", file=sys.stderr)
    sys.exit(1)
' "$e2e_log"

printf 'narrative-search-e2e: PASS; full cross-surface parity verified across Direct, REST, MCP, and CLI\n'
