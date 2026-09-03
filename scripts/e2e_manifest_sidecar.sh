#!/usr/bin/env bash
# Mock-free real-binary E2E manifest sidecar and masked byte identity proof for bd-38us.
#
# Verifies:
# 1. An interrupted run retains its initial parseable sidecar and database with matching run_id.
# 2. A success pair produces byte-identical whole-document canonical JSON after removing only
#    the authorized non-reproducible block ("identity").
# 3. Injected negatives (tampering root_seed and unauthorized mask expansion) fail closed.
# 4. Validates structured evidence logs naming run_id, path, phase, digest, reproducible flag,
#    warning count, completion count, and first failure.

set -euo pipefail

fail() {
  printf 'e2e-manifest-sidecar: %s\n' "$1" >&2
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

printf 'e2e-manifest-sidecar: running mock-free real-binary proof suite via %s\n' "${cargo_runner[0]}"
proof_log="$(mktemp)"
trap 'rm -f "$proof_log"' EXIT

"${cargo_runner[@]}" test \
  --locked \
  -p scriptbots-app \
  --test run_manifest_emitted \
  -- \
  --nocapture \
  --test-threads=1 2>&1 | tee "$proof_log"

grep -q 'test result: ok\. 11 passed; 0 failed;' "$proof_log" ||
  fail "the mock-free manifest test suite did not pass"

printf 'e2e-manifest-sidecar: verifying structured evidence logs\n'

# Verify interrupted evidence
interrupted_evidence="$(grep 'E2E_EVIDENCE: ' "$proof_log" | grep '"phase":"interrupted"' | head -1 | sed 's/.*E2E_EVIDENCE: //')"
[ -n "$interrupted_evidence" ] || fail "no interrupted run evidence line was emitted"

for field in \
  '"schema":"scriptbots.run-manifest.e2e-evidence.v1"' \
  '"phase":"interrupted"' \
  '"run_id":' \
  '"path":' \
  '"database_hash":' \
  '"sidecar_hash":' \
  '"digest":' \
  '"reproducible":' \
  '"warning_count":' \
  '"completion_count":0' \
  '"first_failure":' \
  '"database_matches_sidecar":true'; do
  case "$interrupted_evidence" in
    *"$field"*) ;;
    *) fail "interrupted evidence line is missing required field $field" ;;
  esac
done

# Verify finalized pair evidence
pair_evidence="$(grep 'E2E_EVIDENCE: ' "$proof_log" | grep '"phase":"finalized_pair"' | head -1 | sed 's/.*E2E_EVIDENCE: //')"
[ -n "$pair_evidence" ] || fail "no finalized pair evidence line was emitted"

for field in \
  '"schema":"scriptbots.run-manifest.e2e-evidence.v1"' \
  '"phase":"finalized_pair"' \
  '"run_id_a":' \
  '"run_id_b":' \
  '"path_a":' \
  '"path_b":' \
  '"database_hash_a":' \
  '"database_hash_b":' \
  '"sidecar_hash_a":' \
  '"sidecar_hash_b":' \
  '"masked_hash_a":' \
  '"masked_hash_b":' \
  '"digest":' \
  '"reproducible":' \
  '"warning_count":' \
  '"completion_count":' \
  '"masked_bytes_identical":true'; do
  case "$pair_evidence" in
    *"$field"*) ;;
    *) fail "finalized pair evidence line is missing required field $field" ;;
  esac
done

printf 'e2e-manifest-sidecar: PASS; all contracts, masked byte identities, and interrupted run sidecars verified\n'
