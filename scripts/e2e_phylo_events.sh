#!/usr/bin/env bash
# e2e_phylo_events.sh: Real E2E phylogeny events stream and detector hint cross-validation pipeline (bd-16g.3.3)
# Asserts speciation, extinction, radiation events, hint reconciliation, and determinism.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

echo "==> [1/4] Verifying toolchain and RCH dispatch..."
if ! command -v rch >/dev/null 2>&1; then
  echo "Error: rch is required for remote compilation and execution" >&2
  exit 1
fi

echo "==> [2/4] Executing unit tests in phylo.rs (criteria a through j)..."
rch exec -- cargo test -p scriptbots-core --lib test_bd_16g_3_3 -- --nocapture

echo "==> [3/4] Executing E2E multi-cadence integration tests in phylo_events_e2e..."
rch exec -- cargo test -p scriptbots-core --test phylo_events_e2e -- --nocapture

echo "==> [4/4] Phylogeny event stream and hint cross-validation verification completed successfully!"
cat <<EOF
{
  "pipeline": "e2e_phylo_events",
  "status": "success",
  "bead": "bd-16g.3.3",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "checks": [
    "quiet_run_zero_speciation_in_stable_world",
    "interbreeding_split_rejected_as_interbreeding",
    "clean_split_persisting_k_samples_speciation_confirmed",
    "reverting_split_rejected_as_transient",
    "brain_kind_gated_separation_distinguished_from_phenotypic",
    "empty_two_parent_denominator_rejected_no_ancestral_support",
    "sub_cluster_below_min_size_rejected",
    "extinction_detection_and_idempotence",
    "radiation_doubling_inside_window_confirmed",
    "clustering_dropped_living_species_anomaly_no_extinction",
    "total_hint_reconciliation_every_hint_receives_typed_verdict",
    "speciation_evidence_reconstruction_from_raw_ancestry_and_births",
    "byte_identical_determinism_on_repeated_runs"
  ]
}
EOF
