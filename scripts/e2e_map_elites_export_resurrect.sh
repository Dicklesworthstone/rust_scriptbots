#!/usr/bin/env bash
# Real E2E MAP-Elites export, reload, diff, resurrect, and replay pipeline proof (bd-16g.6.3).

set -euo pipefail

fail() {
  printf 'e2e_map_elites: %s\n' "$1" >&2
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

test_name="test_map_elites_export_reload_diff_resurrect_replay_e2e"
printf 'e2e_map_elites: executing proof %s\n' "$test_name"

test_log="$(mktemp)"
"${cargo_runner[@]}" test \
  -p scriptbots-core \
  --features economy-faults \
  --test map_elites \
  "$test_name" \
  -- \
  --exact \
  --nocapture 2>&1 | tee "$test_log"

python3 -c '
import json, re, sys

log_path = sys.argv[1]
seen_phases = set()
with open(log_path) as f:
    for line in f:
        m = re.search(r"\{\"schema\":\"scriptbots\.qd-archive\.e2e\.v1\".*\}", line)
        if m:
            data = json.loads(m.group(0))
            phase = data.get("phase")
            seen_phases.add(phase)
            print(f"Verified MAP-Elites E2E proof evidence: phase={phase} data={data}")

required_phases = {
    "start",
    "archive_populated",
    "archive_exported_and_reloaded",
    "archive_diff_verified",
    "resurrect_applied",
    "replay_digest_verified",
    "completed",
}
missing = required_phases - seen_phases
if missing:
    print(f"Missing required proof phases: {missing}", file=sys.stderr)
    sys.exit(1)
' "$test_log"
rm -f "$test_log"

printf 'e2e_map_elites: PASS; all MAP-Elites export, reload, diff, resurrect, and replay invariants verified\n'
