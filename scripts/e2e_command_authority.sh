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

printf 'command-authority-e2e: running real eviction, restart, and concurrent-client proofs through rch\n'
rch exec -- cargo test -p scriptbots-storage \
  --test storage_journal \
  authority \
  -- --nocapture

printf 'command-authority-e2e: running real journal crash-window and recovery proofs through rch\n'
rch exec -- cargo test -p scriptbots-storage \
  --lib \
  host_journal_ \
  -- --nocapture
