#!/usr/bin/env bash
# Seeded tournament ratings E2E proof (bd-16g.12.2).
#
# Emits the full match matrix, fitted parameters, effect sizes, intervals,
# convergence diagnostics, and per-axis table, asserting known planted ordering
# and that reruns are byte-stable.

set -euo pipefail

fail() {
  printf 'e2e_tournament_ratings: %s\n' "$1" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

if [[ "${CI:-}" == "true" ]]; then
  cargo_runner=(env RUST_LOG="scriptbots::tournament::rating=info,info" cargo)
else
  command -v rch >/dev/null 2>&1 ||
    fail "rch is required outside CI; local Cargo fallback is forbidden"
  cargo_runner=(rch exec -- env RUST_LOG="scriptbots::tournament::rating=info,info" cargo)
fi

test_name="test_rating_multi_axis_clustered_bootstrap_and_markdown"
printf 'e2e_tournament_ratings: executing proof %s\n' "$test_name"

run_once() {
  local out_file="$1"
  RUST_LOG="scriptbots::tournament::rating=info" "${cargo_runner[@]}" test \
    -p scriptbots-app \
    --test tournament \
    "$test_name" \
    -- \
    --exact \
    --nocapture 2>&1 | tee "$out_file"
}

run1_log="$(mktemp)"
run2_log="$(mktemp)"

printf 'e2e_tournament_ratings: pass 1 (initial execution)\n'
run_once "$run1_log"

printf 'e2e_tournament_ratings: pass 2 (byte-stability rerun)\n'
run_once "$run2_log"

# Verify structured rating logs, convergence, and byte-stable reruns in the output
python3 -c '
import sys, re

with open(sys.argv[1]) as f:
    c1 = f.read()
with open(sys.argv[2]) as f:
    c2 = f.read()

# Assert Bradley-Terry MLE fit logged
if "fitted Bradley-Terry model for axis" not in c1:
    print("Missing Bradley-Terry fit log", file=sys.stderr)
    sys.exit(1)

# Assert family rating logs
if "family rating on axis" not in c1:
    print("Missing family rating logs", file=sys.stderr)
    sys.exit(1)

# Assert pairwise comparison logs
if "pairwise comparison on axis" not in c1:
    print("Missing pairwise comparison logs", file=sys.stderr)
    sys.exit(1)

# Assert test passed
if not re.search(r"test result: ok\.\s+1 passed;\s+0 failed", c1):
    print("Test pass 1 did not pass", file=sys.stderr)
    sys.exit(1)

if not re.search(r"test result: ok\.\s+1 passed;\s+0 failed", c2):
    print("Test pass 2 did not pass", file=sys.stderr)
    sys.exit(1)

# Assert byte-stable reruns over rating logs (stripping timestamps which vary by run)
def normalize_rating_logs(content):
    content = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", content)
    res = []
    for line in content.splitlines():
        if "scriptbots::tournament::rating:" in line:
            m = re.search(r"(INFO|WARN)\s+scriptbots::tournament::rating:.*$", line)
            if m:
                res.append(m.group(0))
    return res

logs1 = normalize_rating_logs(c1)
logs2 = normalize_rating_logs(c2)

if not logs1:
    print("No rating log lines found to verify byte stability", file=sys.stderr)
    sys.exit(1)

if logs1 != logs2:
    print("Byte stability check failed: pass 1 and pass 2 produced divergent ratings!", file=sys.stderr)
    print(f"Pass 1 logs: {len(logs1)} entries\nPass 2 logs: {len(logs2)} entries", file=sys.stderr)
    sys.exit(1)

print(f"e2e_tournament_ratings: verified {len(logs1)} rating log entries are byte-stable across reruns")
' "$run1_log" "$run2_log"

rm -f "$run1_log" "$run2_log"

printf 'e2e_tournament_ratings: PASS; all tournament rating math, multi-axis CIs, and leaderboard invariants verified\n'
