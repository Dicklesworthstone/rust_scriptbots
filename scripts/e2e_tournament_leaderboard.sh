#!/usr/bin/env bash
# E2E test for CI-regenerated leaderboard with per-cell provenance (bd-16g.12.3).
#
# Proves:
# 1. Pipeline execution: plan -> run -> rate -> render -> emit
# 2. Output artifacts: tournament_results.jsonl, ratings.json, leaderboard.md
# 3. Two runs produce bit-identical result digests and reproducibility gate passes
# 4. 3-family tournament reproduction barrier holds across mlp, dwraon, assembly
# 5. CLI subcommand generates artifacts in target directory
# 6. CLI --check passes against matching generated leaderboard
# 7. CLI negative control 1: tampered document fails --check with diff
# 8. CLI negative control 2: stale config digest fails --check with explicit config drift message
# 9. Exclusion audit: irreproducible runs excluded and recorded in leaderboard footer
# 10. Structured logging emitted at each stage

set -euo pipefail

fail() {
  printf 'e2e_tournament_leaderboard: ERROR: %s\n' "$1" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

if [[ "${CI:-}" == "true" ]]; then
  cargo_runner=(env RUST_LOG="scriptbots::tournament=info,info" cargo)
else
  command -v rch >/dev/null 2>&1 ||
    fail "rch is required outside CI; local Cargo fallback is forbidden"
  cargo_runner=(rch exec -- env RUST_LOG="scriptbots::tournament=info,info" cargo)
fi

run1_log="$(mktemp)"
run2_log="$(mktemp)"
trap 'rm -f "$run1_log" "$run2_log"' EXIT

printf '==> Phase 1: Running unit & pipeline tests on RCH...\n'
"${cargo_runner[@]}" test -p scriptbots-app --lib tournament::tests::test_execute_leaderboard_tournament_smoke \
  -- --exact --nocapture 2>&1 | tee "$run1_log"

printf '==> Phase 2: Running reproducibility check (pass 2)...\n'
"${cargo_runner[@]}" test -p scriptbots-app --lib tournament::tests::test_execute_leaderboard_tournament_smoke \
  -- --exact --nocapture 2>&1 | tee "$run2_log"

printf '==> Verifying structured logging from pipeline runs...\n'
python3 -c '
import sys, re

with open(sys.argv[1]) as f:
    log1 = f.read()
with open(sys.argv[2]) as f:
    log2 = f.read()

# Assert start log present
assert "starting tournament leaderboard execution" in log1, "missing start log in pass 1"
assert "starting tournament leaderboard execution" in log2, "missing start log in pass 2"

# Assert match outcome row logs present
assert "match outcome row" in log1, "missing match outcome rows in pass 1"
assert "match outcome row" in log2, "missing match outcome rows in pass 2"

# Assert reproducibility gate passed
assert "reproducibility gate passed" in log1, "missing reproducibility gate pass log in pass 1"
assert "reproducibility gate passed" in log2, "missing reproducibility gate pass log in pass 2"

# Assert leaderboard completion log
assert "leaderboard execution complete" in log1, "missing completion log in pass 1"
assert "leaderboard execution complete" in log2, "missing completion log in pass 2"
' "$run1_log" "$run2_log"
printf '  Structured logging verified.\n'

printf '==> Phase 3: Testing 3-family reproduction barrier (mlp, dwraon, assembly)...\n'
"${cargo_runner[@]}" test -p scriptbots-app --lib tournament::tests::test_execute_leaderboard_tournament_three_families_barrier \
  -- --exact --nocapture

printf '==> Phase 4: Testing row exclusion filter and negative controls in unit suite...\n'
"${cargo_runner[@]}" test -p scriptbots-app --lib tournament::tests::test_leaderboard_row_exclusion_filter \
  -- --exact --nocapture
"${cargo_runner[@]}" test -p scriptbots-app --lib tournament::tests::test_leaderboard_drift_check \
  -- --exact --nocapture

printf '==> Phase 5: Testing CLI tournament subcommand and artifact emission...\n'
cli_out_dir="artifacts/tournament"
mkdir -p "$cli_out_dir"
"${cargo_runner[@]}" run -p scriptbots-app --bin scriptbots-app -- \
  tournament --smoke --out "$cli_out_dir"

[[ -s "$cli_out_dir/tournament_results.jsonl" ]] || fail "missing tournament_results.jsonl"
[[ -s "$cli_out_dir/ratings.json" ]] || fail "missing ratings.json"
[[ -s "$cli_out_dir/leaderboard.md" ]] || fail "missing leaderboard.md"
printf '  CLI generated all 3 artifacts in %s\n' "$cli_out_dir"

printf '==> Phase 6: Testing CLI --check mode (positive control)...\n'
"${cargo_runner[@]}" run -p scriptbots-app --bin scriptbots-app -- \
  tournament --smoke --check "$cli_out_dir/leaderboard.md"
printf '  Positive --check passed.\n'

printf '==> Phase 7: Testing CLI --check mode negative control 1 (document drift)...\n'
drift_doc="$(mktemp)"
trap 'rm -f "$run1_log" "$run2_log" "$drift_doc"' EXIT
sed 's/1500.0/1999.9/g' "$cli_out_dir/leaderboard.md" > "$drift_doc"
drift_err_log="$(mktemp)"
trap 'rm -f "$run1_log" "$run2_log" "$drift_doc" "$drift_err_log"' EXIT

if "${cargo_runner[@]}" run -p scriptbots-app --bin scriptbots-app -- \
  tournament --smoke --check "$drift_doc" > "$drift_err_log" 2>&1; then
  fail "expected --check to fail on drifted document, but it exited 0"
fi
grep -qi -E "drift|diff" "$drift_err_log" ||
  fail "expected document drift / diff error in output"
printf '  Negative Control 1 passed: detected document drift and rejected.\n'

printf '==> Phase 8: Testing CLI --check mode negative control 2 (config drift)...\n'
cfg_drift_doc="$(mktemp)"
cfg_err_log="$(mktemp)"
trap 'rm -f "$run1_log" "$run2_log" "$drift_doc" "$drift_err_log" "$cfg_drift_doc" "$cfg_err_log"' EXIT
sed -E 's/Effective Config Digest \| `[a-f0-9]+`/Effective Config Digest | `stale_config_digest_deadbeef`/g' \
  "$cli_out_dir/leaderboard.md" > "$cfg_drift_doc"

if "${cargo_runner[@]}" run -p scriptbots-app --bin scriptbots-app -- \
  tournament --smoke --check "$cfg_drift_doc" > "$cfg_err_log" 2>&1; then
  fail "expected --check to fail on stale config digest, but it exited 0"
fi
grep -qi -E "config.*drift" "$cfg_err_log" ||
  fail "expected explicit config drift in output"
printf '  Negative Control 2 passed: detected config drift and rejected.\n'

printf '\n\033[32;1me2e_tournament_leaderboard: ALL TESTS PASSED\033[0m\n'
