#!/usr/bin/env bash
# Default meadow acceptance E2E verification script (bd-2z0.10.5).
#
# Proves:
# 1. Full declared meadow cohort execution (seeds: [42, 137, 20260717], 300 ticks).
# 2. Envelope satisfaction on every seed.
# 3. GUI/TUI scientific parity (bit-exact WorldDigestV1 match).
# 4. Balanced resource ledger (0 breaches, evaluate_conservation pass).
# 5. Injected negative controls.
# 6. Structured JSON evidence schema validation.

set -euo pipefail

fail() {
  printf 'meadow-acceptance-e2e: %s\n' "$1" >&2
  exit 1
}

command -v rch >/dev/null 2>&1 || fail "rch is required; local Cargo fallback is forbidden"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

printf 'meadow-acceptance-e2e: running the default meadow acceptance proof through rch\n'
e2e_log="$(mktemp)"
trap 'rm -f "$e2e_log"' EXIT

rch exec -- cargo test -p scriptbots-app --test meadow_acceptance -- --nocapture 2>&1 |
  tee "$e2e_log"

grep -q 'test result: ok\. 1 passed; 0 failed;' "$e2e_log" ||
  fail "the meadow acceptance test did not pass"

evidence="$(sed -n '/^EVIDENCE_START$/,/^EVIDENCE_END$/{ /^EVIDENCE_START$/d; /^EVIDENCE_END$/d; p; }' "$e2e_log" | tail -1)"
[ -n "$evidence" ] || fail "no evidence JSON was emitted between EVIDENCE_START and EVIDENCE_END"

# Validate required JSON schema and fields
printf '%s' "$evidence" | grep -q '"schema":"scriptbots.meadow-acceptance.v1"' ||
  fail "evidence JSON missing schema scriptbots.meadow-acceptance.v1"

printf '%s' "$evidence" | grep -q '"scenario":"meadow"' ||
  fail "evidence JSON missing scenario meadow"

printf '%s' "$evidence" | grep -q '"parity_verified":true' ||
  fail "evidence JSON did not verify GUI/TUI parity"

printf '%s' "$evidence" | grep -q '"ledger_balanced":true' ||
  fail "evidence JSON did not verify balanced ledger"

printf '%s' "$evidence" | grep -q '"total_breaches":0' ||
  fail "evidence JSON reported non-zero ledger breaches"

printf '%s' "$evidence" | grep -q '"envelope_verified":true' ||
  fail "evidence JSON did not verify scenario envelope"

printf '%s' "$evidence" | grep -q '"negative_controls_verified":true' ||
  fail "evidence JSON did not verify negative controls"

# Check each seed in the cohort
for seed in 42 137 20260717; do
  printf '%s' "$evidence" | grep -q "\"seed\":$seed" ||
    fail "cohort seed $seed is missing from evidence"
done

printf 'meadow-acceptance-e2e: all meadow acceptance criteria verified successfully!\n'
