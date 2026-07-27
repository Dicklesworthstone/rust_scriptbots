#!/usr/bin/env bash
# Mock-free persistence preparation deadline and exact-retry proof for bd-2z0.5.8.

set -euo pipefail

fail() {
  printf 'persistence-preparation-e2e: %s\n' "$1" >&2
  exit 1
}

command -v rch >/dev/null 2>&1 || fail "rch is required; local Cargo fallback is forbidden"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

test_name="tests::preparation_deadline_file_pipeline_e2e_preserves_exact_retry"
printf 'persistence-preparation-e2e: running exact proof %s through rch\n' "$test_name"
RCH_VISIBILITY=summary \
  rch exec -- cargo test -p scriptbots-storage --lib "$test_name" -- --exact --nocapture 2>&1 |
  awk -v test_name="$test_name" '
    { print }
    /^\[RCH\] remote [^ ]+ \(/ { witnessed_remote_completion = 1 }
    /\/data\/projects\/rust_scriptbots\// { witnessed_remote_source = 1 }
    /test result: ok\. 1 passed; 0 failed;/ { witnessed_test = 1 }
    /"schema":"scriptbots.persistence-preparation.evidence.v1"/ {
      is_refusal = index($0, "\"phase\":\"refusal\"") &&
        index($0, "\"stage\":\"materialize\"") &&
        index($0, "\"deadline\":\"10ns\"") &&
        index($0, "\"scientific_bytes\":") &&
        index($0, "\"identity_state\":\"unassigned\"") &&
        index($0, "\"disposition\":\"not_admitted\"") &&
        index($0, "\"receipt\":null") &&
        index($0, "\"batch_id\":null") &&
        index($0, "\"tick\":91") &&
        index($0, "\"durable_tick_count\":0")
      if (is_refusal) {
        split($0, byte_parts, "\"scientific_bytes\":")
        split(byte_parts[2], byte_value, /[^0-9]/)
        refusal_bytes = byte_value[1] + 0
        witnessed_refusal = 1
      }
      is_retry = index($0, "\"phase\":\"retry\"") &&
        index($0, "\"stage\":\"outbox_admission\"") &&
        index($0, "\"deadline\":\"10ns\"") &&
        index($0, "\"scientific_bytes\":") &&
        index($0, "\"identity_state\":\"assigned\"") &&
        index($0, "\"disposition\":\"admitted\"") &&
        index($0, "\"receipt\":\"durable\"") &&
        index($0, "\"batch_id\":1") &&
        index($0, "\"tick\":91") &&
        index($0, "\"durable_tick_count\":1")
      if (is_retry) {
        split($0, byte_parts, "\"scientific_bytes\":")
        split(byte_parts[2], byte_value, /[^0-9]/)
        retry_bytes = byte_value[1] + 0
        witnessed_retry = 1
      }
    }
    END {
      failed = 0
      if (!witnessed_remote_completion) {
        printf "persistence-preparation-e2e: proof %s emitted no RCH remote completion marker\n", test_name > "/dev/stderr"
        failed = 1
      }
      if (!witnessed_remote_source) {
        printf "persistence-preparation-e2e: proof %s emitted no remote project path\n", test_name > "/dev/stderr"
        failed = 1
      }
      if (!witnessed_test) {
        printf "persistence-preparation-e2e: exact proof %s did not report one passing test\n", test_name > "/dev/stderr"
        failed = 1
      }
      if (!witnessed_refusal) {
        printf "persistence-preparation-e2e: proof %s emitted no exact refusal evidence\n", test_name > "/dev/stderr"
        failed = 1
      }
      if (!witnessed_retry) {
        printf "persistence-preparation-e2e: proof %s emitted no exact retry evidence\n", test_name > "/dev/stderr"
        failed = 1
      }
      if (witnessed_refusal && witnessed_retry && (refusal_bytes <= 0 || refusal_bytes != retry_bytes)) {
        printf "persistence-preparation-e2e: refusal/retry evidence is not bound to one positive payload estimate\n" > "/dev/stderr"
        failed = 1
      }
      exit failed
    }
  '
