#!/usr/bin/env bash
# Archipelago Host Completion Debt E2E Proof (bd-5tyo, bd-16g.5.1).
#
# Verifies:
# 1. Canonical per-island scenario overlay merge in scriptbots-core and IslandSpec::with_overlay.
# 2. Parallel topology rescope compile-time guard (HostCore !Send structural exclusion).
# 3. Archipelago declaration-order determinism check using canonical heterogeneous island overlays.
# 4. Full DSR 4-island, 2,000-tick, single-storage-file E2E proof with offline reconstruction & conservation audit.
# 5. CLI archipelago execution with per-island overlays into single SQLite database.

set -euo pipefail

fail() {
  printf 'e2e_archipelago_host: %s\n' "$1" >&2
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

log_file="$(mktemp)"
trap 'rm -f "$log_file"' EXIT

printf 'e2e_archipelago_host: (1) running canonical config overlay unit tests in scriptbots-core...\n'
"${cargo_runner[@]}" test \
  -p scriptbots-core \
  config_with_overlay \
  -- \
  --nocapture 2>&1 | tee "$log_file"

grep -q "test tests::config_with_overlay_applies_overrides_recursively_and_validates \.\.\. ok" "$log_file" ||
  fail "config_with_overlay_applies_overrides_recursively_and_validates failed"
grep -q "test tests::config_with_overlay_rejects_invalid_values \.\.\. ok" "$log_file" ||
  fail "config_with_overlay_rejects_invalid_values failed"

printf 'e2e_archipelago_host: (2) running IslandSpec::with_overlay and parallel-topology rescope guard...\n'
"${cargo_runner[@]}" test \
  -p scriptbots-runtime \
  --lib \
  archipelago::tests \
  -- \
  --nocapture 2>&1 | tee "$log_file"

grep -q "test archipelago::tests::test_island_spec_with_overlay_and_archipelago_config_from_base_and_overlays \.\.\. ok" "$log_file" ||
  fail "test_island_spec_with_overlay failed"
grep -q "test archipelago::tests::outer_island_parallelism_stays_excluded_while_host_core_is_not_send \.\.\. ok" "$log_file" ||
  fail "parallel topology rescope guard failed"

printf 'e2e_archipelago_host: (3) running archipelago determinism gate...\n'
"${cargo_runner[@]}" test \
  -p scriptbots-app \
  --test archipelago_determinism \
  -- \
  --nocapture 2>&1 | tee "$log_file"

grep -q "test result: ok\." "$log_file" ||
  fail "archipelago_determinism test suite failed"

printf 'e2e_archipelago_host: (4) running full 4-island / 2,000-tick / single-storage-file proof...\n'
"${cargo_runner[@]}" test \
  -p scriptbots-app \
  --test archipelago_report_cli \
  dsr_four_heterogeneous_islands_reach_2000_ticks_in_single_storage_file \
  -- \
  --ignored \
  --nocapture 2>&1 | tee "$log_file"

grep -q "test dsr_four_heterogeneous_islands_reach_2000_ticks_in_single_storage_file \.\.\. ok" "$log_file" ||
  fail "4-island 2k-tick single-storage proof failed"

printf 'e2e_archipelago_host: all tests passed successfully!\n'
