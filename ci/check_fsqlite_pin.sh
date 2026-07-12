#!/usr/bin/env bash
# ci/check_fsqlite_pin.sh — bd-2z0.8.9.14 (program bd-2js6)
#
# The bd-2z0.8 program rests on "exact-revision" reproducibility for the
# FrankenSQLite pin. This guard fails any PR where the three operative records
# disagree on the fsqlite git revision:
#   1. Cargo.toml      (the declared pin)
#   2. Cargo.lock      (what actually builds)
#   3. AGENTS.md       (the storage-contract record agents follow)
# README.md is checked as a soft fourth source (warning, not failure — prose
# may legitimately discuss multiple revisions; AGENTS.md may not).
#
# History this guard exists to prevent from recurring: e04543d advanced the
# pin cd9990bb -> 1eec0d2 in Cargo.toml+Cargo.lock but narrative docs lagged
# for a day and the bead epic text lagged longer — a three-way record drift
# that made "the pinned revision" ambiguous (full story: UPGRADE_LOG.md
# 2026-07-12 entry, bd-2z0.8.9.14).
#
# Usage:
#   ci/check_fsqlite_pin.sh              # check the repo
#   ci/check_fsqlite_pin.sh --self-test  # doctored-fixture proof
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

extract_manifest_rev() { # $1 = Cargo.toml path
  grep -E '^fsqlite *=' "$1" | grep -oE 'rev = "[a-f0-9]{40}"' | grep -oE '[a-f0-9]{40}' | head -1
}
extract_lock_rev() { # $1 = Cargo.lock path
  grep -oE 'frankensqlite\?rev=[a-f0-9]{40}' "$1" | grep -oE '[a-f0-9]{40}' | sort -u
}
extract_agents_revs() { # $1 = AGENTS.md path
  grep -oE '\b[a-f0-9]{40}\b' "$1" | sort -u
}

check() {
  local manifest="$1" lock="$2" agents="$3" readme="${4:-}"
  echo "== fsqlite pin-consistency guard =="

  local m_rev l_revs a_revs
  m_rev="$(extract_manifest_rev "$manifest" || true)"
  l_revs="$(extract_lock_rev "$lock" || true)"
  a_revs="$(extract_agents_revs "$agents" || true)"

  echo "  Cargo.toml rev : ${m_rev:-<none found>}"
  echo "  Cargo.lock rev(s): ${l_revs:-<none found>}"

  if [[ -z "$m_rev" ]]; then
    echo "::error::no fsqlite rev found in $manifest — pin removed or reformatted; update this guard's extractor"
    return 1
  fi
  local l_count
  l_count="$(printf '%s' "$l_revs" | grep -c . || true)"
  if (( l_count != 1 )); then
    echo "::error::expected exactly 1 frankensqlite rev in $lock, found $l_count:"
    printf '    %s\n' $l_revs
    return 1
  fi
  if [[ "$m_rev" != "$l_revs" ]]; then
    echo "::error::Cargo.toml pins $m_rev but Cargo.lock resolves $l_revs — manifest/lock drift (the bd-2z0.8.9.14 failure class). Refresh the lock through the bd-2z0.8 lane."
    return 1
  fi
  # AGENTS.md must contain the pinned rev and MUST NOT contain any other
  # 40-hex string that looks like a competing fsqlite rev claim. We can't
  # know which 40-hex strings are fsqlite refs vs other tools', so the rule
  # is scoped: every 40-hex within 3 lines of the word 'fsqlite' must equal
  # the pin.
  local a_near
  a_near="$(grep -iE -A3 -B3 'fsqlite|frankensqlite' "$agents" | grep -oE '\b[a-f0-9]{40}\b' | sort -u || true)"
  echo "  AGENTS.md fsqlite-adjacent rev(s): ${a_near:-<none found>}"
  if [[ -z "$a_near" ]]; then
    echo "::error::AGENTS.md contains no fsqlite revision record — the storage contract must state the exact pin"
    return 1
  fi
  local bad=0
  while IFS= read -r rev; do
    [[ -z "$rev" ]] && continue
    if [[ "$rev" != "$m_rev" ]]; then
      echo "::error::AGENTS.md cites fsqlite-adjacent rev $rev but the pin is $m_rev"
      bad=1
    fi
  done <<< "$a_near"
  (( bad == 0 )) || return 1

  if [[ -n "$readme" && -f "$readme" ]]; then
    local r_near
    r_near="$(grep -iE -A3 -B3 'fsqlite|frankensqlite' "$readme" | grep -oE '\b[a-f0-9]{40}\b' | sort -u || true)"
    while IFS= read -r rev; do
      [[ -z "$rev" ]] && continue
      if [[ "$rev" != "$m_rev" ]]; then
        echo "::warning::README.md cites fsqlite-adjacent rev $rev != pin $m_rev (soft check — fix when convenient)"
      fi
    done <<< "$r_near"
  fi

  echo "  OK: manifest == lock == AGENTS.md ($m_rev)"
  return 0
}

self_test() {
  local tmp; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
  local REV_A="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  local REV_B="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  printf 'fsqlite = { version = "=0.1.16", rev = "%s" }\n' "$REV_A" > "$tmp/Cargo.toml"
  printf 'source = "git+https://github.com/x/frankensqlite?rev=%s#%s"\n' "$REV_A" "$REV_A" > "$tmp/Cargo.lock"
  printf 'fsqlite pinned at revision %s per contract\n' "$REV_A" > "$tmp/AGENTS.md"

  echo "== self-test 1: consistent fixture must PASS =="
  check "$tmp/Cargo.toml" "$tmp/Cargo.lock" "$tmp/AGENTS.md" >/dev/null 2>&1 || {
    echo "::error::self-test FAILED — consistent fixture rejected"; return 1; }
  echo "  PASS"

  echo "== self-test 2: doctored AGENTS.md must FAIL =="
  printf 'fsqlite pinned at revision %s per contract\n' "$REV_B" > "$tmp/AGENTS.md"
  if check "$tmp/Cargo.toml" "$tmp/Cargo.lock" "$tmp/AGENTS.md" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — AGENTS.md drift not caught"; return 1
  fi
  echo "  PASS"

  echo "== self-test 3: doctored lock must FAIL =="
  printf 'fsqlite pinned at revision %s per contract\n' "$REV_A" > "$tmp/AGENTS.md"
  printf 'source = "git+https://github.com/x/frankensqlite?rev=%s#%s"\n' "$REV_B" "$REV_B" > "$tmp/Cargo.lock"
  if check "$tmp/Cargo.toml" "$tmp/Cargo.lock" "$tmp/AGENTS.md" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — manifest/lock drift not caught"; return 1
  fi
  echo "  PASS"
  echo "== self-test PASSED =="
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
else
  check "$REPO_ROOT/Cargo.toml" "$REPO_ROOT/Cargo.lock" "$REPO_ROOT/AGENTS.md" "$REPO_ROOT/README.md"
fi
