#!/usr/bin/env bash
# Mock-free headless + REST TUI evidence proof for bd-2z0.14.2.6.
#
# Gates on the two halves of the acceptance criterion that live at different
# levels, and runs both rather than citing either:
#
#   1. END TO END: a real world, the real terminal renderer drawing real frames,
#      a real control runtime with a real bound REST listener, and a real HTTP
#      GET over a real socket. Proves the frame the renderer PUBLISHED is the
#      frame the server SERVED. Emits scriptbots.tui-evidence.v1 with the seed,
#      viewport, capability profile, endpoint status, report path, byte counts,
#      full-frame digest, every per-region digest, and the source commit.
#
#   2. LOCALIZATION: the deliberately-broken-widget negative control. It lives at
#      the buffer level, and that is not a shortcut — nothing OUTSIDE the binary
#      can make a widget draw wrongly. Every external lever (theme, palette,
#      capability, reduced motion, viewport) changes the frame legitimately, so
#      an end-to-end "broken widget" would need a test-only breakage hook wired
#      into production paint code. The buffer-level control blanks one panel's
#      rectangle and requires EXACTLY ONE region hash to move, which is a
#      stronger claim than an end-to-end variant could make anyway.

set -euo pipefail

fail() {
  printf 'tui-evidence-e2e: %s\n' "$1" >&2
  exit 1
}

command -v rch >/dev/null 2>&1 || fail "rch is required; local Cargo fallback is forbidden"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

printf 'tui-evidence-e2e: running the mock-free headless+REST proof through rch\n'
e2e_log="$(mktemp)"
trap 'rm -f "$e2e_log"' EXIT

rch exec -- cargo test -p scriptbots-app --test terminal_end_to_end -- \
  tui_evidence_e2e_serves_the_frame_the_renderer_published --exact --nocapture 2>&1 |
  tee "$e2e_log"

grep -q 'test result: ok\. 1 passed; 0 failed;' "$e2e_log" ||
  fail "the end-to-end proof did not pass"

evidence="$(grep -o '{"schema":"scriptbots.tui-evidence.v1".*}' "$e2e_log" | tail -1)"
[ -n "$evidence" ] || fail "no scriptbots.tui-evidence.v1 line was emitted"

# Every field the acceptance criterion names must be present. A run that emitted
# a partial line is a run whose evidence cannot be audited without re-running it.
for field in \
  '"seed":' '"frames":' '"viewport":' '"capability_profile":' '"backend":' \
  '"endpoint":' '"endpoint_status":' '"report_path":' '"served_bytes":' \
  '"served_lines":' '"full_cell_fnv1a64":' '"region_count":' '"regions":' \
  '"source_commit":'; do
  case "$evidence" in
    *"$field"*) ;;
    *) fail "evidence line is missing $field" ;;
  esac
done

case "$evidence" in
  *'"endpoint_status":200'*) ;;
  *) fail "the endpoint did not answer 200" ;;
esac

# Non-vacuity: an evidence line reporting zero regions or zero bytes would satisfy
# every presence check above while proving nothing was rendered or served.
region_count="$(printf '%s' "$evidence" | sed -n 's/.*"region_count":\([0-9]*\).*/\1/p')"
served_bytes="$(printf '%s' "$evidence" | sed -n 's/.*"served_bytes":\([0-9]*\).*/\1/p')"
[ "${region_count:-0}" -ge 8 ] || fail "expected at least 8 hashed regions, got ${region_count:-0}"
[ "${served_bytes:-0}" -gt 500 ] || fail "served body is implausibly small: ${served_bytes:-0} bytes"

# Every region digest must be a full FNV-1a64 and they must be DISTINCT. Identical
# digests would mean the hashes are not per-region, and the localization control
# below would be pinning one value under ten names.
digests="$(printf '%s' "$evidence" | grep -o '"hash":"[0-9a-f]*"' | sed 's/.*:"//;s/"//')"
[ -n "$digests" ] || fail "no region digests in the evidence line"
while read -r digest; do
  [ "${#digest}" -eq 16 ] || fail "region digest $digest is not a 16-hex FNV-1a64"
done <<< "$digests"
total="$(printf '%s\n' "$digests" | wc -l | tr -d ' ')"
unique="$(printf '%s\n' "$digests" | sort -u | wc -l | tr -d ' ')"
[ "$total" = "$unique" ] || fail "region digests are not distinct ($unique unique of $total)"

printf 'tui-evidence-e2e: end-to-end proof OK — %s regions, %s served bytes\n' \
  "$region_count" "$served_bytes"

printf 'tui-evidence-e2e: running the buffer-level localization negative control\n'
control_log="$(mktemp)"
trap 'rm -f "$e2e_log" "$control_log"' EXIT

rch exec -- cargo test -p scriptbots-app --lib -- \
  terminal::tests::a_broken_widget_fails_its_own_region_and_no_others --exact --nocapture 2>&1 |
  tee "$control_log"

grep -q 'test result: ok\. 1 passed; 0 failed;' "$control_log" ||
  fail "the broken-widget localization control did not pass"

printf 'tui-evidence-e2e: localization control OK\n'
printf 'tui-evidence-e2e: PASS\n'
printf '%s\n' "$evidence"
