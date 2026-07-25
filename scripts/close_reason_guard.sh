#!/usr/bin/env bash
# Check that a bead's close reason describes the artifact it names (bd-emmm).
#
# Recommended by the bd-d3wu audit as its highest-value guard. Three of four
# read-through false closes shared one signature: THE CLOSE REASON NAMED A SPECIFIC
# LOCATION AND THE LOCATION DID NOT CONTAIN THE THING. Each would have been caught by
# one reviewer opening the named path — so open it mechanically instead.
#
# WHAT THIS CATCHES: substitution — a real adjacent artifact named in place of the
# deliverable. Purely mechanical; it makes no judgement about whether the work was good,
# only whether the named artifact is the claimed artifact.
#
# WHAT IT DOES NOT CATCH, stated so nobody over-trusts it:
#   * A close reason citing no location at all. Unaffected by design; separate weakness.
#   * bd-2z0.5.9, the audit's third example. Its reason cited a crate and a symbol that
#     BOTH exist ("interactions ... in scriptbots-storage"); the defect was that the
#     title said PERSIST while the reason claimed only a schema. That is a title-versus-
#     reason mismatch and needs a different check.
#
# Usage:
#   scripts/close_reason_guard.sh <bead-id>...    check specific beads (live tracker)
#   scripts/close_reason_guard.sh --all-closed    check every closed bead
#   scripts/close_reason_guard.sh --self-test     prove the guard against fixtures
# Exit: 0 clean, 1 findings, 65 usage error.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

fail() {
  printf 'close-reason-guard: %s\n' "$1" >&2
  exit "${2:-65}"
}

# ---------------------------------------------------------------------------
# Extraction. Kept as pure text->list helpers so the self-test can drive them
# with fixture strings instead of live tracker state, which would rot.
# ---------------------------------------------------------------------------

cited_paths() {
  {
    # Explicit file paths.
    grep -oE '(crates|scripts|docs|ci)/[A-Za-z0-9_./-]+\.(rs|sh|toml|md|json)' <<<"$1" || true
    # Crate names used as a location ("... in scriptbots-web"). Resolved to the crate dir.
    grep -oE '\bscriptbots-[a-z-]+\b' <<<"$1" | sed 's|^|crates/|;s|$|/|' || true
  } | sort -u
}

cited_commits() {
  grep -oE '\b[0-9a-f]{7,40}\b' <<<"$1" | sort -u || true
}

# Identifier-shaped tokens the reason claims live somewhere. Tokens that are merely
# part of a cited path are removed: otherwise "budget gates in scripts/perf_gate.sh"
# self-anchors on `perf_gate` and the substitution check never fires.
cited_symbols() {
  local reason="$1" paths="$2" raw
  raw="$(grep -oE '\b([A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*|[A-Z][A-Z0-9]+_[A-Z0-9_]+|[a-z][a-z0-9]*_[a-z0-9_]{2,})\b' <<<"$reason" | sort -u || true)"
  local out=()
  while read -r s; do
    [ -n "$s" ] || continue
    case "$s" in DSR|README|AGENTS|TODO|NOTE) continue ;; esac
    # Drop anything appearing inside a cited path string (self-anchoring).
    if [ -n "$paths" ] && grep -qF -- "$s" <<<"$paths"; then continue; fi
    out+=("$s")
  done <<<"$raw"
  printf '%s\n' "${out[@]:-}"
}

# ---------------------------------------------------------------------------
# The check, operating on a label + reason text.
# ---------------------------------------------------------------------------

findings=0

check_reason() {
  local label="$1" reason="$2"
  local paths_s symbols_s header=0 missing=0 anchored=0 nsym=0
  [ -n "$reason" ] && [ "$reason" != "-" ] || { printf '%-18s SKIP (no close reason)\n' "$label"; return 0; }

  paths_s="$(cited_paths "$reason")"
  symbols_s="$(cited_symbols "$reason" "$paths_s")"

  _hdr() { [ "$header" -eq 1 ] || { printf '%s\n' "$label"; header=1; }; }
  _find() { _hdr; printf '  %s %s\n' "$1" "$2"; findings=$((findings + 1)); }

  while read -r p; do
    [ -n "$p" ] || continue
    [ -e "$p" ] || { _find 'MISSING-PATH' "$p"; missing=1; }
  done <<<"$paths_s"

  while read -r c; do
    [ -n "$c" ] || continue
    git cat-file -e "${c}^{commit}" 2>/dev/null && continue
    grep -qE "(fnv1a64|blake3|sha256)[:=]?[[:space:]]*${c}" <<<"$reason" && continue
    _find 'UNRESOLVED-COMMIT' "$c"
  done <<<"$(cited_commits "$reason")"

  # THE SUBSTITUTION CHECK: if the reason names a location AND identifiers, at least one
  # identifier must appear inside a named location. This is what catches "budget gates
  # in scripts/perf_gate.sh" when that file contains none of them.
  nsym="$(grep -c . <<<"$symbols_s" || true)"
  if [ "$missing" -eq 0 ] && [ -n "$(tr -d '[:space:]' <<<"$paths_s")" ] && [ "$nsym" -gt 0 ]; then
    while read -r p; do
      [ -n "$p" ] || continue
      while read -r s; do
        [ -n "$s" ] || continue
        if grep -rqF -- "$s" "$p" 2>/dev/null; then anchored=1; break 2; fi
      done <<<"$symbols_s"
    done <<<"$paths_s"
    if [ "$anchored" -eq 0 ]; then
      _find 'UNANCHORED' "none of $nsym cited identifier(s) appear under the cited location(s): $(tr '\n' ' ' <<<"$paths_s")"
    fi
  fi

  [ "$header" -eq 1 ] || printf '%-18s ok\n' "$label"
}

close_reason_of() {
  br show "$1" 2>/dev/null | tr '\n' ' ' | grep -oE 'Closed: [0-9-]+ \(.*' | sed 's/^Closed: [0-9-]* (//' | head -1
}

check_bead() { check_reason "$1" "$(close_reason_of "$1")"; }

# ---------------------------------------------------------------------------
# Self-test against FIXTURES, not live beads. The audit's own trap 9: a check built on
# mutable tracker state rots — both example beads have since been reopened, so their
# `Closed:` line no longer exists. Fixtures are the verbatim historical reasons.
# ---------------------------------------------------------------------------

self_test() {
  local rc=0 out
  local -a labels=() reasons=() expect=()

  labels+=("fixture:perf-harness"); expect+=("FLAG")
  reasons+=("Implemented visual performance harness and budget gates in scripts/perf_gate.sh and DSR profiles: per-tier frame budgets (60fps@1k, 30fps@10k), subsystem timing breakdowns, memory_footprint tracking and FrameBudgetGate wiring")

  labels+=("fixture:fork-ux"); expect+=("FLAG")
  reasons+=("Implemented Browser fork-this-world UX in scriptbots-web: permalink decoding with shared config composition, parent_diff display and fork_this_world entry point")

  labels+=("fixture:missing-path"); expect+=("FLAG")
  reasons+=("Landed the harness in scripts/does_not_exist.sh with validate_budget gating")

  labels+=("fixture:genuine-hydrology"); expect+=("PASS")
  reasons+=("Completed and published: iterative hydrology is bit-exact with the frozen recursive_accumulation_oracle in crates/scriptbots-core/src/lib.rs, stack-safe on a 262144-cell meander")

  labels+=("fixture:genuine-lease"); expect+=("PASS")
  reasons+=("Implemented OS companion-file writer lease before recovery: StorageWriterLease in crates/scriptbots-storage/src/lib.rs with schema_fingerprint verification")

  labels+=("fixture:no-location"); expect+=("PASS")
  reasons+=("Completed all residual performance and parent architecture epics")

  local i
  for i in "${!labels[@]}"; do
    findings=0
    out="$(check_reason "${labels[$i]}" "${reasons[$i]}" 2>&1 || true)"
    if [ "${expect[$i]}" = "FLAG" ]; then
      if [ "$findings" -gt 0 ]; then printf '  %-26s flagged as expected\n' "${labels[$i]}"
      else printf '  %-26s NOT FLAGGED — guard too weak\n' "${labels[$i]}"; printf '%s\n' "$out"; rc=1; fi
    else
      if [ "$findings" -eq 0 ]; then printf '  %-26s passed as expected\n' "${labels[$i]}"
      else printf '  %-26s FALSE POSITIVE — guard too aggressive\n' "${labels[$i]}"; printf '%s\n' "$out"; rc=1; fi
    fi
  done
  findings=0
  [ "$rc" -eq 0 ] && printf '\nself-test: all 6 fixtures behaved correctly.\n' || printf '\nself-test: FAILED.\n'
  return "$rc"
}

case "${1-}" in
  '') fail "usage: $0 <bead-id>... | --all-closed | --self-test" ;;
  --self-test) self_test; exit $? ;;
  --all-closed)
    while read -r id; do [ -n "$id" ] && check_bead "$id"; done < <(
      br list --status=closed --json 2>/dev/null |
        python3 -c 'import sys,json;d=json.load(sys.stdin);r=d if isinstance(d,list) else d.get("issues",[]);print("\n".join(x["id"] for x in r))'
    )
    ;;
  *) for id in "$@"; do check_bead "$id"; done ;;
esac

if [ "$findings" -gt 0 ]; then
  printf '\nclose-reason-guard: %d finding(s). A cited location must contain the claimed artifact.\n' "$findings" >&2
  exit 1
fi
printf '\nclose-reason-guard: clean.\n'
