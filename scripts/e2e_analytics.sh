#!/usr/bin/env bash
# End-to-end scientific analytics pipeline acceptance probe (`bd-2z0.11.9`).
#
# Validates the complete scientific analysis journey:
# 1. Seeded simulation with planted regime change (population crash at tick 100) & stationary null control.
# 2. Execution of the complete report suite (all 12 built-in reports via CLI & Registry).
# 3. Ground-truth invariant assertions across reports (FDR certification of planted shift,
#    rejection of null noise, lineage founder component conservation, descendant birth accounting).
# 4. FrankenPandas Parquet export with exact SQL row count equality & round-trip verification.
# 5. Graph exports (lineage, dynasty, interaction in GraphML and Edge-List).
# 6. FTS5 narrative event search verification.
# 7. Structured MANIFEST.json artifact emission and verification.
#
# RE-PINNING PROCEDURE:
# If future simulation physics changes alter demographic reproduction dynamics, founder cohorts,
# or narrative thresholds:
# 1. Inspect the deterministic trajectory in `crates/scriptbots-analytics/tests/e2e_pipeline.rs`.
# 2. Update the planted event window (default 70..=130) or pre/post expectations in `EventRecord`.
# 3. Re-verify false-positive control (`null_noise_not_significant` must remain rejected under FDR).
# 4. Re-run `bash scripts/e2e_analytics.sh` locally through RCH and confirm all stages and invariants pass.
# 5. Never widen tolerance thresholds silently without documenting the biological rationale in the git commit.

set -euo pipefail

fail() {
  printf 'e2e_analytics: ERROR: %s\n' "$1" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

out_dir="${1:-${SCRIPTBOTS_PROOF_ROOT:-./analytics_e2e_proof}}"
mkdir -p "$out_dir"
manifest_out="$out_dir/MANIFEST.json"

if [[ "${CI:-}" == "true" ]]; then
  cargo_runner=(cargo)
else
  command -v rch >/dev/null 2>&1 ||
    fail "rch is required outside CI; local Cargo fallback is forbidden"
  cargo_runner=(rch exec -- cargo)
fi

export SCRIPTBOTS_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
printf 'e2e_analytics: running analytics e2e pipeline test (commit=%s)...\n' "$SCRIPTBOTS_GIT_COMMIT"

e2e_log="$(mktemp)"
trap 'rm -f "$e2e_log"' EXIT

"${cargo_runner[@]}" test \
  --locked \
  -p scriptbots-analytics \
  --test e2e_pipeline \
  -- \
  --nocapture 2>&1 | tee "$e2e_log"

grep -q 'test result: ok\. 1 passed; 0 failed;' "$e2e_log" ||
  fail "e2e_pipeline test suite did not pass with 1 passed and 0 failed"

printf 'e2e_analytics: extracting and verifying E2E_ANALYTICS_MANIFEST...\n'

manifest_json="$(grep -o '{"schema":"scriptbots.analytics.e2e-manifest.v1".*}' "$e2e_log" | tail -1)"
[ -n "$manifest_json" ] || fail "no E2E_ANALYTICS_MANIFEST line was emitted by e2e_pipeline"

printf '%s\n' "$manifest_json" > "$manifest_out"

python3 -c '
import json, sys

manifest_path = sys.argv[1]
with open(manifest_path) as f:
    m = json.load(f)

schema = m.get("schema")
assert schema == "scriptbots.analytics.e2e-manifest.v1", f"bad schema: {schema}"
verdict = m.get("verdict")
assert verdict == "pass", f"verdict is not pass: {verdict}"

stages = m.get("stages", [])
assert len(stages) >= 6, f"expected at least 6 stages, got {len(stages)}"
for s in stages:
    name = s.get("stage")
    status = s.get("status")
    dur = s.get("duration_ms")
    print(f"  Stage {name}: {status} ({dur} ms)")
    assert status == "pass", f"stage {name} failed: {status}"

invariants = m.get("invariants", [])
assert len(invariants) >= 8, f"expected at least 8 invariants, got {len(invariants)}"
for inv in invariants:
    name = inv.get("invariant")
    status = inv.get("status")
    print(f"  Invariant {name}: {status}")
    assert status == "pass", f"invariant {name} failed: {status}"

print(f"All {len(stages)} stages and {len(invariants)} invariants verified PASS.")
' "$manifest_out"

printf 'e2e_analytics: PASS; MANIFEST written to %s\n' "$manifest_out"
