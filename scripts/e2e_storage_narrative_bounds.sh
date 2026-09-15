#!/usr/bin/env bash
# Mock-free finite memory/admission budget and fallible container reservation proof for bd-2z0.5.17.

set -euo pipefail

fail() {
  printf 'storage-narrative-bounds-e2e: %s\n' "$1" >&2
  exit 1
}

command -v rch >/dev/null 2>&1 || fail "rch is required; local Cargo fallback is forbidden"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

test_name="narrative_preparation_bounds_and_timeout_cleanup_e2e"
printf 'storage-narrative-bounds-e2e: running exact proof %s through rch\n' "$test_name"
RCH_VISIBILITY=verbose \
RCH_QUEUE_WHEN_BUSY=true \
RCH_REQUIRE_REMOTE=1 \
RCH_NO_SELF_HEALING=1 \
  rch --no-self-healing exec -v --no-color -- \
    cargo test -p scriptbots-storage --test admission_bounds "$test_name" -- --exact --nocapture 2>&1 |
  awk -v test_name="$test_name" '
    { print }
    /Executing command remotely/ { witnessed_remote_dispatch = 1 }
    /Job j-[0-9]+ submitted to [^[:space:]]+/ {
      worker = $0
      sub(/^.*submitted to /, "", worker)
      sub(/[[:space:]].*$/, "", worker)
      job = $0
      sub(/^.*Job /, "", job)
      sub(/[[:space:]].*$/, "", job)
      printf "Executing command remotely: cargo test -p scriptbots-storage --test admission_bounds %s -- --exact --nocapture on %s (RCH job %s)\n", test_name, worker, job
      witnessed_remote_dispatch = 1
    }
    /^\[RCH\] remote [^ ]+ \(/ { witnessed_remote_completion = 1 }
    /\/data\/projects\/rust_scriptbots\// { witnessed_remote_source = 1 }
    /test result: ok\. 1 passed; 0 failed;/ { witnessed_test = 1 }
    /"schema":"scriptbots.narrative-preparation.evidence.v1"/ {
      is_oversized_refusal = index($0, "\"phase\":\"oversized_narrative_refusal\"") &&
        index($0, "\"stage\":\"narrative_measure\"") &&
        index($0, "\"disposition\":\"not_admitted\"") &&
        index($0, "\"error\":\"NarrativePayloadTooLarge\"") &&
        index($0, "\"inflight_bytes_before\":0") &&
        index($0, "\"inflight_bytes_after\":0") &&
        index($0, "\"inflight_narrative_bytes_before\":0") &&
        index($0, "\"inflight_narrative_bytes_after\":0") &&
        index($0, "\"tick\":101")
      if (is_oversized_refusal) {
        witnessed_oversized_refusal = 1
      }
      is_reservation_refusal = index($0, "\"phase\":\"fallible_reservation_refusal\"") &&
        index($0, "\"stage\":\"prepare_buffer\"") &&
        index($0, "\"disposition\":\"not_admitted\"") &&
        index($0, "\"error\":\"ReservationFailed\"") &&
        index($0, "\"tick\":102")
      if (is_reservation_refusal) {
        witnessed_reservation_refusal = 1
      }
      is_cleanup = index($0, "\"phase\":\"cleanup_handoff\"") &&
        index($0, "\"stage\":\"background_worker\"") &&
        index($0, "\"disposition\":\"async_cleanup\"") &&
        index($0, "\"tick\":103")
      if (is_cleanup) {
        witnessed_cleanup = 1
      }
      is_admission = index($0, "\"phase\":\"healthy_admission_and_retry\"") &&
        index($0, "\"stage\":\"outbox_admission\"") &&
        index($0, "\"disposition\":\"admitted\"") &&
        index($0, "\"receipt\":\"durable\"") &&
        index($0, "\"tick\":104") &&
        index($0, "\"durable_tick_count\":1")
      if (is_admission) {
        witnessed_admission = 1
      }
    }
    END {
      failed = 0
      if (!witnessed_remote_dispatch) {
        printf "storage-narrative-bounds-e2e: proof %s emitted no Executing command remotely banner\n", test_name > "/dev/stderr"
        failed = 1
      }
      if (!witnessed_remote_completion) {
        printf "storage-narrative-bounds-e2e: proof %s emitted no RCH remote completion marker\n", test_name > "/dev/stderr"
        failed = 1
      }
      if (!witnessed_remote_source) {
        printf "storage-narrative-bounds-e2e: proof %s emitted no remote project path\n", test_name > "/dev/stderr"
        failed = 1
      }
      if (!witnessed_test) {
        printf "storage-narrative-bounds-e2e: exact proof %s did not report one passing test\n", test_name > "/dev/stderr"
        failed = 1
      }
      if (!witnessed_oversized_refusal) {
        printf "storage-narrative-bounds-e2e: proof %s emitted no oversized narrative refusal evidence\n", test_name > "/dev/stderr"
        failed = 1
      }
      if (!witnessed_reservation_refusal) {
        printf "storage-narrative-bounds-e2e: proof %s emitted no fallible reservation refusal evidence\n", test_name > "/dev/stderr"
        failed = 1
      }
      if (!witnessed_cleanup) {
        printf "storage-narrative-bounds-e2e: proof %s emitted no cleanup handoff evidence\n", test_name > "/dev/stderr"
        failed = 1
      }
      if (!witnessed_admission) {
        printf "storage-narrative-bounds-e2e: proof %s emitted no healthy admission and retry evidence\n", test_name > "/dev/stderr"
        failed = 1
      }
      exit failed
    }
  '
