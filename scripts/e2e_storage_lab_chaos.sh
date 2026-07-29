#!/usr/bin/env bash
# Bounded deterministic LabRuntime persistence chaos lane for bd-2z0.8.9.15.

set -euo pipefail

fail() {
  printf 'storage-lab-chaos: %s\n' "$1" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

trace_dir="${SCRIPTBOTS_LAB_TRACE_DIR:-target/lab-runtime-chaos}"
export SCRIPTBOTS_LAB_TRACE_DIR="$trace_dir"

if [[ "${CI:-}" == "true" ]]; then
  cargo_runner=(cargo)
else
  command -v rch >/dev/null 2>&1 ||
    fail "rch is required outside CI; local Cargo fallback is forbidden"
  cargo_runner=(rch exec -- cargo)
fi

printf 'storage-lab-chaos: trace_dir=%s max_steps=512 dpor_runs=8 fixed_seeds=2 repetitions=50\n' \
  "$trace_dir"
"${cargo_runner[@]}" test \
  --locked \
  -p scriptbots-storage \
  tests::lab_runtime_chaos:: \
  -- \
  --nocapture \
  --test-threads=1

artifact_glob="$trace_dir"/*.json
compgen -G "$artifact_glob" >/dev/null ||
  fail "the passing lane emitted no structured trace artifact"
grep -R -Fq '"schema":"scriptbots.persistence-lab-chaos.v1"' "$trace_dir" ||
  fail "trace artifacts do not contain the stable persistence chaos schema"

printf 'storage-lab-chaos: PASS; replayable traces retained under %s\n' "$trace_dir"
