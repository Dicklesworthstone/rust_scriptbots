#!/usr/bin/env bash
# Mock-free durable CommandId eviction/restart proof for bd-2z0.5.12.

set -euo pipefail

fail() {
  printf 'command-authority-e2e: %s\n' "$1" >&2
  exit 1
}

command -v rch >/dev/null 2>&1 || fail "rch is required; local Cargo fallback is forbidden"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

run_exact_test() {
  local test_name="$1"
  shift
  printf 'command-authority-e2e: running exact proof %s through rch\n' "$test_name"
  rch exec -- cargo test -p scriptbots-storage "$@" "$test_name" -- --exact --nocapture 2>&1 |
    awk -v test_name="$test_name" '
      { print }
      /test result: ok\. 1 passed; 0 failed;/ { witnessed = 1 }
      END {
        if (!witnessed) {
          printf "command-authority-e2e: exact proof %s did not report one passing test\n", test_name > "/dev/stderr"
          exit 1
        }
      }
    '
}

run_exact_test \
  file_command_authority_survives_cache_eviction_and_restart \
  --test storage_journal
run_exact_test \
  file_channel_concurrent_exact_duplicate_clients_apply_once_and_persist_authority \
  --test storage_journal

crash_window_tests=(
  host_journal_receive_and_admission_transaction_fault_matrix_rolls_back_exactly
  host_journal_scientific_table_transaction_fault_matrix_recovers_exactly_once
  host_journal_lost_rollback_ack_is_indeterminate_but_reopen_safe
  host_journal_post_archive_fault_recovers_and_applies_exactly_once
  host_journal_post_commit_pre_receipt_fault_reopens_without_duplicate_effects
  host_journal_durable_marker_and_publication_faults_fail_closed
  host_journal_flush_fault_recovers_the_final_shutdown_persistence_tail
  host_journal_analytics_publication_fault_keeps_the_durable_event_exactly_once
  host_journal_shutdown_checkpoint_close_fault_is_typed_and_reopen_safe
  host_journal_reopen_scan_fault_releases_the_writer_without_mutation
)
for test_name in "${crash_window_tests[@]}"; do
  run_exact_test "$test_name" --lib
done
