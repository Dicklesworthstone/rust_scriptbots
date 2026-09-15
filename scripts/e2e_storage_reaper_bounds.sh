#!/usr/bin/env bash
# Mock-free storage reaper bounds, held timeout handoff, and eventual recovery proof for bd-2z0.5.14.

set -euo pipefail

fail() {
  printf 'storage-reaper-bounds-e2e: %s\n' "$1" >&2
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

test_name="reaper_bounds_mock_free_timeout_handoff_and_eventual_recovery_e2e"
printf 'storage-reaper-bounds-e2e: running exact proof %s\n' "$test_name"

test_log="$(mktemp)"
"${cargo_runner[@]}" test \
  --locked \
  -p scriptbots-storage \
  --test reaper_bounds \
  "$test_name" \
  -- \
  --exact \
  --nocapture | tee "$test_log"

python3 -c '
import json, re, sys

log_path = sys.argv[1]
seen_phases = set()
with open(log_path) as f:
    for line in f:
        m = re.search(r"\{\"schema\":\"scriptbots\.storage-reaper\.evidence\.v1\".*\}", line)
        if m:
            data = json.loads(m.group(0))
            phase = data.get("phase")
            seen_phases.add(phase)
            print(f"Verified reaper proof evidence: phase={phase} data={data}")

required_phases = {"start", "held_worker", "duplicate_coalesced", "cross_path_progress", "recovery_and_reopen"}
missing = required_phases - seen_phases
if missing:
    print(f"Missing required proof phases: {missing}", file=sys.stderr)
    sys.exit(1)
' "$test_log"
rm -f "$test_log"

printf 'storage-reaper-bounds-e2e: PASS; all supervisor reaping invariants verified\n'
