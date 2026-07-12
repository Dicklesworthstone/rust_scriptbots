#!/usr/bin/env bash
# ci/check_franken_licenses.sh — bd-2z0.8.15 (program bd-2js6)
#
# Enforces the franken-family license audit: every franken-family package
# present in Cargo.lock must be documented in docs/licenses.md (§2 component
# table). This makes it impossible to admit a franken crate without updating
# the license record in the same PR.
#
# Design notes:
# - Pure bash/grep on Cargo.lock: no cargo invocation, no network, safe to run
#   even while the manifest/lock are mid-reconciliation (bd-2z0.8.9.14).
# - Member crates roll up to a family token (fsqlite-core -> "fsqlite") so the
#   audit table stays readable instead of listing dozens of workspace members.
# - Verbose by design: prints every detected crate and its documentation
#   status; failures print remediation steps. The error message is the UX.
#
# Usage:
#   ci/check_franken_licenses.sh              # check the real repo
#   ci/check_franken_licenses.sh --self-test  # negative-fixture proof
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="${REPO_ROOT}/Cargo.lock"
DOC="${REPO_ROOT}/docs/licenses.md"

# Family detection patterns (anchored) and the token that must appear in the
# audit doc for the family to count as documented.
#   pattern|token
FAMILIES=(
  '^fsqlite|fsqlite'
  '^asupersync|asupersync'
  '^franken-kernel$|franken-kernel'
  '^franken-evidence$|franken-evidence'
  '^franken-decision$|franken-decision'
  '^frankenpandas$|frankenpandas'
  '^ftui|ftui'
  '^fnx-|fnx-'
  '^fsci-|fsci-'
  '^fp-|fp-'
  '^ft-|ft-'
  '^fnp-|fnp-'
)

check() {
  local lock_file="$1" doc_file="$2"
  local detected=0 undocumented=0
  local names
  names="$(grep -E '^name = "' "$lock_file" | sed -E 's/^name = "([^"]+)"$/\1/' | sort -u)"

  echo "== franken license guard =="
  echo "lock: $lock_file"
  echo "doc:  $doc_file"

  local missing=()
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    local token=""
    for entry in "${FAMILIES[@]}"; do
      local pat="${entry%%|*}" tok="${entry##*|}"
      if [[ "$name" =~ $pat ]]; then token="$tok"; break; fi
    done
    # Catch-all tier: ANY crate whose name starts with "franken" that no
    # explicit family covered must be documented under its own exact name.
    # This is the clause that catches brand-new franken crates entering the
    # tree before anyone teaches this script about them.
    if [[ -z "$token" && "$name" == franken* ]]; then
      token="$name"
    fi
    [[ -z "$token" ]] && continue
    detected=$((detected + 1))
    if grep -q -- "$token" "$doc_file"; then
      echo "  documented   : $name (token: $token)"
    else
      echo "  UNDOCUMENTED : $name (token: $token)"
      missing+=("$name -> token '$token' not found in $doc_file")
      undocumented=$((undocumented + 1))
    fi
  done <<< "$names"

  echo "-- summary: $detected franken-family package(s) detected, $undocumented undocumented"
  if (( undocumented > 0 )); then
    echo "::error::franken-family crate(s) present in Cargo.lock but absent from docs/licenses.md"
    printf '  %s\n' "${missing[@]}"
    cat <<'REMEDY'
Remediation (bd-2z0.8.15 policy):
  1. Verify the upstream LICENSE sha against the family sha recorded in docs/licenses.md §1.
  2. Add a component row to docs/licenses.md §2 in THIS PR (door, pin, license, wasm, notes).
  3. If this is a brand-new family, extend FAMILIES in ci/check_franken_licenses.sh
     and the wasm denylist in ci/check_wasm_graph.sh (bd-2z0.8.16).
REMEDY
    return 1
  fi
  if (( detected == 0 )); then
    echo "::warning::no franken-family packages detected — if fsqlite was removed this guard may need retiring"
  fi
  return 0
}

self_test() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  cat > "$tmp/Cargo.lock" <<'FIXTURE'
[[package]]
name = "serde"
version = "1.0.210"

[[package]]
name = "franken-bogus"
version = "0.0.1"

[[package]]
name = "asupersync"
version = "0.3.6"
FIXTURE
  cat > "$tmp/licenses.md" <<'FIXTURE'
This fixture documents asupersync only.
FIXTURE
  echo "== self-test 1: unknown crate franken-bogus must FAIL via catch-all =="
  # NB: capture output first — with pipefail, `check | grep -q` would take
  # check's intended non-zero status even when grep matches.
  local out
  out="$(check "$tmp/Cargo.lock" "$tmp/licenses.md" 2>&1 || true)"
  if grep -q "franken-bogus -> token" <<< "$out" \
     && ! check "$tmp/Cargo.lock" "$tmp/licenses.md" >/dev/null 2>&1; then
    echo "  PASS: catch-all flagged franken-bogus and check failed as required"
  else
    echo "::error::self-test FAILED — unknown franken-bogus was not caught"
    printf '%s\n' "$out"
    return 1
  fi
  echo "== self-test 2: fully documented fixture must PASS =="
  cat > "$tmp/licenses2.md" <<'FIXTURE'
Documented: asupersync, franken-bogus.
FIXTURE
  if check "$tmp/Cargo.lock" "$tmp/licenses2.md" >/dev/null 2>&1; then
    echo "  PASS: documented fixture accepted"
  else
    echo "::error::self-test FAILED — documented fixture rejected"
    return 1
  fi
  echo "== self-test PASSED =="
  return 0
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
else
  check "$LOCK" "$DOC"
fi
