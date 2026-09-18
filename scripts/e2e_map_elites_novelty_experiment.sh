#!/usr/bin/env bash
# MAP-Elites Matched-Seed Fitness vs Novelty Selection Experiment E2E Proof (bd-16g.6.2, bd-16g.6).
#
# Verifies:
# 1. Matched-seed fitness versus novelty evolution runs with MAP-Elites archive observation.
# 2. Novelty selection computes non-zero novelty scores across population and archive.
# 3. Evolution trajectories diverge between fitness and novelty selection modes.
# 4. Archive metrics (coverage, occupied cells, QD-score) computed and compared.
# 5. Archive export to CSV and independent diff across selection modes.
# 6. Elite resurrection into live world increases population without corruption.

set -euo pipefail

fail() {
  printf 'e2e_map_elites_novelty_experiment: %s\n' "$1" >&2
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

test_name="test_map_elites_matched_seed_fitness_vs_novelty_experiment"
printf 'e2e_map_elites_novelty_experiment: running matched-seed experiment %s...\n' "$test_name"

log_file="$(mktemp)"
"${cargo_runner[@]}" test \
  -p scriptbots-core \
  --features economy-faults \
  --test map_elites \
  "$test_name" \
  -- \
  --exact \
  --nocapture 2>&1 | tee "$log_file"

printf 'e2e_map_elites_novelty_experiment: verifying experiment evidence and metrics...\n'

python3 -c '
import sys, re, json

log = open(sys.argv[1]).read()

# 1. Assert test passed
if not re.search(r"test result: ok\.\s+1 passed;\s+0 failed", log):
    print("Matched-seed experiment test failed", file=sys.stderr)
    sys.exit(1)

# 2. Extract and assert required phases
required_phases = {
    "start",
    "study_executed",
    "archive_diff",
    "resurrection_verified",
    "completed",
}
seen_phases = set()
metrics_data = None

for line in log.splitlines():
    m = re.search(r"\{\"schema\":\"scriptbots\.qd-experiment\.e2e\.v1\".*\}", line)
    if m:
        data = json.loads(m.group(0))
        phase = data.get("phase")
        seen_phases.add(phase)
        if phase == "study_executed":
            metrics_data = data

missing = required_phases - seen_phases
if missing:
    print(f"Missing required phases: {missing}", file=sys.stderr)
    sys.exit(1)

assert metrics_data is not None, "Missing study_executed metrics payload"
assert metrics_data["fitness_coverage"] > 0.0, "Fitness coverage must be positive"
assert metrics_data["novelty_coverage"] > 0.0, "Novelty coverage must be positive"
assert metrics_data["fitness_occupied"] > 0, "Fitness archive must have occupied cells"
assert metrics_data["novelty_occupied"] > 0, "Novelty archive must have occupied cells"

print("e2e_map_elites_novelty_experiment: all matched-seed QD experiment criteria verified.")
' "$log_file"

rm -f "$log_file"
printf 'e2e_map_elites_novelty_experiment: PASS\n'
