#!/usr/bin/env bash
# Mock-free production Bevy HostClient snapshot and command E2E proof for bd-2z0.7.2.
#
# Gates on the acceptance criteria:
#   1. Production Bevy consumes host snapshots via HostClient and snapshot subscriptions.
#   2. Zero, one, and multiple Bevy windows over one host produce identical WorldDigest
#      and tick sequences; bounded snapshots coalesce safely and reconnect from revision gaps.
#   3. Selection and control actions return typed receipts, and disconnect, renderer failure,
#      shutdown, and reconnect cannot duplicate or own simulation time.
#   4. Emits scriptbots.bevy-host-client-e2e.v1 with client, revision, command id, receipt,
#      tick, digest, coalescing, lifecycle, and shutdown.

set -euo pipefail

fail() {
  printf 'bevy-host-client-e2e: %s\n' "$1" >&2
  exit 1
}

command -v rch >/dev/null 2>&1 || fail "rch is required; local Cargo fallback is forbidden"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

printf 'bevy-host-client-e2e: running the mock-free Bevy HostClient proof through rch\n'
e2e_log="$(mktemp)"
trap 'rm -f "$e2e_log"' EXIT

rch exec -- cargo test -p scriptbots-bevy --test bevy_host_client_e2e -- --nocapture 2>&1 |
  tee "$e2e_log"

grep -q 'test result: ok\. 4 passed; 0 failed;' "$e2e_log" ||
  fail "the Bevy HostClient E2E proof did not report 4 passed tests"

evidence="$(grep -o '{.*"schema":"scriptbots.bevy-host-client-e2e.v1".*}' "$e2e_log" | tail -1)"
[ -n "$evidence" ] || fail "no scriptbots.bevy-host-client-e2e.v1 line was emitted"

# Every field the acceptance criterion names must be present.
for field in \
  '"client":' '"revision":' '"command_id":' '"receipt":' \
  '"tick":' '"digest":' '"coalescing":' '"lifecycle":' \
  '"shutdown":' '"source_commit":'; do
  case "$evidence" in
    *"$field"*) ;;
    *) fail "evidence line is missing $field" ;;
  esac
done

case "$evidence" in
  *'"client":"bevy-host-client-primary"'*) ;;
  *) fail "the evidence line did not identify client as bevy-host-client-primary" ;;
esac

case "$evidence" in
  *'"clean":true'*) ;;
  *) fail "shutdown evidence did not report clean shutdown" ;;
esac

case "$evidence" in
  *'"outcome":"stopped"'*) ;;
  *) fail "shutdown outcome was not stopped" ;;
esac

# Non-vacuity assertions
tick="$(printf '%s' "$evidence" | sed -n 's/.*"tick":\([0-9]*\).*/\1/p')"
[ "${tick:-0}" -ge 2 ] || fail "tick was not advanced: got ${tick:-0}"

digest="$(printf '%s' "$evidence" | grep -o '"digest":"[0-9a-f]*"' | sed 's/.*:"//;s/"//')"
[ -n "$digest" ] || fail "no digest in the evidence line"
[ "${#digest}" -ge 16 ] || fail "digest $digest is too short"

printf 'bevy-host-client-e2e: Bevy HostClient proof OK — tick %s, digest %s\n' "$tick" "$digest"
printf 'bevy-host-client-e2e: PASS\n'
printf '%s\n' "$evidence"
