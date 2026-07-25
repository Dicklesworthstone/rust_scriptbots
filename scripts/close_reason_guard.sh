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
# MEASURED PRECISION. Over the 363 closed beads in .beads/issues.jsonl this flags 8, of
# which 4 are confirmed real (bd-16g.12.1, bd-2z0.14.1.4, bd-2z0.14.1.10, bd-ahkx — each
# verified by hand: the cited file exists and contains none of the claimed identifiers).
# The other 4 share one shape: a DECISION or EVALUATION bead whose reason names a crate as
# its subject rather than as the home of an artifact ("Completed evaluation of asupersync
# BrowserRuntime for scriptbots-web"). That class is deliberately NOT exempted — keying off
# words like "evaluation" would hand every future close a one-word bypass. Add the artifact
# name to the reason, or use `git commit --no-verify` and say why.
#
# Usage:
#   scripts/close_reason_guard.sh <bead-id>...    check specific beads (live tracker)
#   scripts/close_reason_guard.sh --all-closed    check every closed bead in the export
#   scripts/close_reason_guard.sh --staged        check beads this commit newly closes
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

# Commit-shaped tokens. At least one hex LETTER is required: without it every decimal
# number of the right length is a candidate, and close reasons are full of them — unix-ms
# run stamps like 1784156735, agent counts like 43648807. That was 11 of the first 31 hits.
cited_commits() {
  grep -oE '\b[0-9a-f]{7,40}\b' <<<"$1" | grep -E '[a-f]' | sort -u || true
}

# True if the cited location contains the claimed identifier, as file CONTENT or as a
# path NAME under it. The name arm matters: "parity test in
# scriptbots-storage/tests/persistence_integration.rs" cites a file that exists, but that
# filename appears inside no file's text, so a content-only search calls a real close a lie.
anchors_under() {
  local sym="$1" loc="$2"
  grep -rqF -- "$sym" "$loc" 2>/dev/null && return 0
  [ -n "$(find "$loc" -name "*${sym}*" -print -quit 2>/dev/null)" ]
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

# Prints its findings and RETURNS THE COUNT as its exit status. The count has to travel
# out-of-band: the self-test captures stdout in a command substitution, and a global
# incremented in that subshell is discarded on exit. That is exactly what made an earlier
# revision report every fixture as unflagged while printing the findings it had just made.
check_reason() {
  local label="$1" reason="$2"
  local paths_s symbols_s header=0 missing=0 anchored=0 nsym=0 n=0
  [ -n "$reason" ] && [ "$reason" != "-" ] || { printf '%-18s SKIP (no close reason)\n' "$label"; return 0; }

  paths_s="$(cited_paths "$reason")"
  symbols_s="$(cited_symbols "$reason" "$paths_s")"

  _hdr() { [ "$header" -eq 1 ] || { printf '%s\n' "$label"; header=1; }; }
  _find() { _hdr; printf '  %s %s\n' "$1" "$2"; n=$((n + 1)); }
  # Advisory: printed, never counted, so it cannot block a commit.
  _note() { _hdr; printf '  note: %s %s\n' "$1" "$2"; }

  while read -r p; do
    [ -n "$p" ] || continue
    [ -e "$p" ] || { _find 'MISSING-PATH' "$p"; missing=1; }
  done <<<"$paths_s"

  # ADVISORY ONLY. A commit-shaped token that does not resolve here has too many innocent
  # causes to block on: pinned revisions of upstream repos (the fsqlite pin e536d7f is not
  # our object), rewritten history, and 64-bit digests that are simply 16 hex characters.
  # It is still worth surfacing, because a fabricated commit citation looks exactly like this.
  while read -r c; do
    [ -n "$c" ] || continue
    git cat-file -e "${c}^{commit}" 2>/dev/null && continue
    grep -qE "(fnv1a64|blake3|sha256)[:=]?[[:space:]]*${c}" <<<"$reason" && continue
    _note 'unresolved-commit' "$c"
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
        if anchors_under "$s" "$p"; then anchored=1; break 2; fi
      done <<<"$symbols_s"
    done <<<"$paths_s"
    if [ "$anchored" -eq 0 ]; then
      _find 'UNANCHORED' "none of $nsym cited identifier(s) appear under the cited location(s): $(tr '\n' ' ' <<<"$paths_s")"
    fi
  fi

  [ "$header" -eq 1 ] || printf '%-18s ok\n' "$label"
  # Exit status is a byte: clamp rather than wrap 256 findings around to a false clean.
  if [ "$n" -gt 255 ]; then n=255; fi
  return "$n"
}

close_reason_of() {
  br show "$1" 2>/dev/null | tr '\n' ' ' | grep -oE 'Closed: [0-9-]+ \(.*' | sed 's/^Closed: [0-9-]* (//' | head -1
}

check_bead() {
  local n=0
  check_reason "$1" "$(close_reason_of "$1")" || n=$?
  findings=$((findings + n))
}

# Bulk modes read `.beads/issues.jsonl` — AGENTS.md names it the authoritative tracked
# export, it is what a commit actually publishes, and one read beats 363 `br show` calls.
# Emits `id<TAB>close_reason` (tabs/newlines in the reason flattened to spaces), then a
# sentinel line. $1 is a PATH to the prior export whose already-closed beads are skipped,
# or "" for none — a path, never the content: passing the ~1MB export as an argument blew
# ARG_MAX, and because a dead producer just looks like an empty stream, the gate reported
# CLEAN while checking nothing.
reasons_from_jsonl() {
  python3 -c '
import json, sys
prior = set()
if sys.argv[1]:
    with open(sys.argv[1]) as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get("status") == "closed":
                    prior.add(d["id"])
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    if d.get("status") != "closed" or d["id"] in prior:
        continue
    reason = (d.get("close_reason") or "").replace("\t", " ").replace("\n", " ")
    print(d["id"] + "\t" + reason)
print("__END__\t")
' "$@"
}

# Callers MUST feed this by redirection, never by pipe: the right-hand side of a pipeline
# is a subshell, and `findings` accumulated there is discarded, so the guard reports clean
# while printing findings.
check_jsonl_stream() {
  local id reason n saw_end=0
  # The sentinel takes the ID field, not the reason field: a tab is IFS whitespace, so a
  # leading empty field is collapsed away and `\t__END__` reads back as the id.
  while IFS=$'\t' read -r id reason; do
    [ "$id" = '__END__' ] && { saw_end=1; continue; }
    [ -n "$id" ] || continue
    n=0
    check_reason "$id" "$reason" || n=$?
    findings=$((findings + n))
  done
  # No sentinel means the producer died mid-stream. Refuse rather than call it clean.
  [ "$saw_end" -eq 1 ] || fail 'bead export ended without the sentinel — the reader failed; refusing to report a result' 70
}

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

  # Verbatim bd-16g.2.3, the false positive that forced the name arm of anchors_under: the
  # cited test file is real, but its name occurs in no file's text. Depends on that test
  # continuing to exist — if it is ever renamed, fix the fixture, do not weaken the check.
  labels+=("fixture:named-file"); expect+=("PASS")
  reasons+=("Completed Part 1 online/offline narrative event parity test in scriptbots-storage/tests/persistence_integration.rs alongside existing Part 2 false-positive budget suite.")

  local i n
  for i in "${!labels[@]}"; do
    n=0
    out="$(check_reason "${labels[$i]}" "${reasons[$i]}" 2>&1)" || n=$?
    if [ "${expect[$i]}" = "FLAG" ]; then
      if [ "$n" -gt 0 ]; then printf '  %-26s flagged as expected (%d)\n' "${labels[$i]}" "$n"
      else printf '  %-26s NOT FLAGGED — guard too weak\n' "${labels[$i]}"; printf '%s\n' "$out"; rc=1; fi
    else
      if [ "$n" -eq 0 ]; then printf '  %-26s passed as expected\n' "${labels[$i]}"
      else printf '  %-26s FALSE POSITIVE — guard too aggressive\n' "${labels[$i]}"; printf '%s\n' "$out"; rc=1; fi
    fi
  done
  [ "$rc" -eq 0 ] && printf '\nself-test: all %d fixtures behaved correctly.\n' "${#labels[@]}" || printf '\nself-test: FAILED.\n'
  return "$rc"
}

JSONL='.beads/issues.jsonl'

# Installs into the mcp-agent-mail pre-commit chain-runner's plugin directory rather than
# overwriting .git/hooks/pre-commit, so this composes with the agent-mail guard instead of
# displacing it. The plugin is a two-line shim; the logic stays in this tracked file.
install_hook() {
  local dir='.git/hooks/hooks.d/pre-commit' plugin
  [ -d "$(dirname "$dir")" ] || fail "no .git/hooks/hooks.d — is the agent-mail chain-runner installed?"
  mkdir -p "$dir"
  plugin="$dir/50-close-reason-guard"
  cat >"$plugin" <<'PLUGIN'
#!/usr/bin/env bash
# Installed by scripts/close_reason_guard.sh --install-hook (bd-emmm). Edit the guard, not this.
exec "$(git rev-parse --show-toplevel)/scripts/close_reason_guard.sh" --staged
PLUGIN
  chmod +x "$plugin"
  printf 'close-reason-guard: installed %s\n' "$plugin"
}

case "${1-}" in
  '') fail "usage: $0 <bead-id>... | --all-closed | --staged | --self-test | --install-hook" ;;
  --self-test) self_test; exit $? ;;
  --install-hook) install_hook; exit 0 ;;
  --all-closed)
    [ -f "$JSONL" ] || fail "$JSONL not found"
    check_jsonl_stream < <(reasons_from_jsonl '' < "$JSONL")
    ;;
  --staged)
    # Gate only what this commit newly closes: beads closed in the staged export that were
    # not already closed at HEAD. A commit touching no beads exits clean without work.
    git diff --cached --quiet -- "$JSONL" && { printf 'close-reason-guard: no staged bead changes.\n'; exit 0; }
    # HEAD's export arrives as a /dev/fd path, so nothing large crosses the argument list.
    check_jsonl_stream < <(git show ":$JSONL" | reasons_from_jsonl <(git show "HEAD:$JSONL" 2>/dev/null || true))
    ;;
  *) for id in "$@"; do check_bead "$id"; done ;;
esac

if [ "$findings" -gt 0 ]; then
  printf '\nclose-reason-guard: %d finding(s). A cited location must contain the claimed artifact.\n' "$findings" >&2
  if [ "${1-}" = '--staged' ]; then
    cat >&2 <<'EOF'
Commit refused. Either the close reason names the wrong artifact, or it names the right one
by a name that is not in the tree. Fix the reason so it cites what actually landed. If the
guard is wrong here, commit with --no-verify and say so in the commit message.
EOF
  fi
  exit 1
fi
printf '\nclose-reason-guard: clean.\n'
