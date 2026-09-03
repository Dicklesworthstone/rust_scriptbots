#!/usr/bin/env bash
# e2e_species_cadence.sh: Real E2E species cadence and offline reconstruction pipeline (bd-16g.3.6)
# Runs a multi-cohort world through at least two cadence boundaries, persists/reloads species tables,
# rebuilds offline, and asserts byte-identical live/offline digests.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

echo "==> [1/3] Verifying toolchain and RCH dispatch..."
if ! command -v rch >/dev/null 2>&1; then
  echo "Error: rch is required for remote compilation and execution" >&2
  exit 1
fi

echo "==> [2/3] Executing multi-cohort species cadence E2E integration test via RCH..."
test_name="species_cadence_e2e"
rch exec -- cargo test -p scriptbots-core --test "$test_name" -- --nocapture

echo "==> [3/3] Species cadence E2E pipeline passed with byte-identical live/offline parity!"
cat <<EOF
{
  "pipeline": "species_cadence_e2e",
  "status": "success",
  "bead": "bd-16g.3.6",
  "test": "test_species_cadence_multi_cohort_e2e_reconstruction_and_fault_gate",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "checks": [
    "multi_cohort_world_stepped_through_cadence_boundaries",
    "typed_phenotype_adapter_validation",
    "live_cadence_execution_and_snapshot_publication",
    "byte_identical_live_vs_offline_table_digests",
    "table_serialization_and_reload_digest_parity",
    "fault_injection_agreement_gate_failure"
  ]
}
EOF
