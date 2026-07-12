#!/usr/bin/env bash
# ci/check_wasm_graph.sh — bd-2z0.8.16 (program bd-2js6)
#
# Two guards protecting build-budget and wasm-cleanliness boundaries:
#
# GUARD A (wasm denylist + snapshot): the scriptbots-web wasm32 dependency
# graph must never contain native-only crates. NO franken numeric/graph/frame
# library supports wasm32 (fnx-algorithms hard-requires rayon; frankenpandas
# defaults pull bundled-C rusqlite; fsci/ft are thread/SIMD-native). One
# careless feature edge breaks the wasm build at best, silently bloats it at
# worst. Additionally, the FULL resolved graph is snapshotted: ANY new crate
# appearing in the wasm graph requires a conscious snapshot update in the same
# PR (golden-file pattern — drift is visible in review, never silent).
#
# GUARD B (reverse boundary): scriptbots-core's default-feature graph must not
# contain analytics/brain-side franken crates (fsci-*/fnx-*/fp-*/ft-* enter
# only through scriptbots-analytics / brain-ml opt-in features per the plan's
# build budgets).
#
# Uses `cargo metadata --locked` / `cargo tree --locked` ONLY — resolution, no
# compilation; --locked guarantees this guard can never mutate Cargo.lock
# (single-lane rule, bd-2z0.8). If the lock is out of sync with the manifest
# the guard fails LOUDLY — that is a feature (see bd-2z0.8.9.14).
#
# Snapshot update procedure (deliberate, reviewed):
#   ci/check_wasm_graph.sh --update-snapshot
#   git add ci/fixtures/wasm_graph_snapshot.txt   # justify in the PR body
#
# Usage:
#   ci/check_wasm_graph.sh                  # both guards
#   ci/check_wasm_graph.sh --update-snapshot
#   ci/check_wasm_graph.sh --self-test      # denylist logic fixture proof
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT="${REPO_ROOT}/ci/fixtures/wasm_graph_snapshot.txt"

# Native-only crates that must NEVER appear in the wasm32 graph of
# scriptbots-web. Extend in the same PR that admits a new franken family
# (docs/licenses.md §5 checklist item 4).
DENYLIST=(
  rayon
  wide
  tokio
  rusqlite
  fsci-
  fnx-
  fp-
  frankenpandas
  ft-
  fnp-
)

wasm_graph() {
  ( cd "$REPO_ROOT" && cargo tree --locked \
      --target wasm32-unknown-unknown \
      -p scriptbots-web \
      --edges normal,build \
      --prefix none --format '{p}' 2>&1 ) \
    | sed -E 's/ \(.*//' | sed -E 's/ v[0-9].*$//' | sort -u
}

core_default_graph() {
  ( cd "$REPO_ROOT" && cargo tree --locked \
      -p scriptbots-core \
      --edges normal \
      --prefix none --format '{p}' 2>&1 ) \
    | sed -E 's/ \(.*//' | sed -E 's/ v[0-9].*$//' | sort -u
}

check_denylist() {
  local graph="$1" label="$2" denied=()
  echo "== $label: denylist scan =="
  for pat in "${DENYLIST[@]}"; do
    local hits
    hits="$(printf '%s\n' "$graph" | grep -E "^${pat}" || true)"
    if [[ -n "$hits" ]]; then
      denied+=("$pat -> $hits")
      # Print the dependency chain that introduced the offender — the error
      # message is this guard's UX. Skipped under self-test (fixture graphs
      # have no real chains and cargo resolution would just slow the test).
      while IFS= read -r crate; do
        echo "::error::denied crate '$crate' in $label graph; introduction chain:"
        if [[ -z "${WASM_GUARD_SELF_TEST:-}" ]]; then
          ( cd "$REPO_ROOT" && cargo tree --locked \
              --target wasm32-unknown-unknown -p scriptbots-web -i "$crate" 2>/dev/null \
            || echo "  (run: cargo tree --locked -i $crate for the chain)" )
        else
          echo "  (chain lookup skipped under self-test)"
        fi
      done <<< "$hits"
    else
      echo "  clean: $pat"
    fi
  done
  (( ${#denied[@]} == 0 )) || return 1
}

main() {
  echo "== resolving wasm32 graph for scriptbots-web (cargo tree --locked) =="
  local graph
  if ! graph="$(wasm_graph)"; then
    echo "::error::cargo tree --locked failed — lock/manifest drift? (see bd-2z0.8.9.14)"
    printf '%s\n' "$graph"
    return 1
  fi
  printf '  %d crates in wasm graph\n' "$(printf '%s\n' "$graph" | grep -c .)"

  check_denylist "$graph" "scriptbots-web/wasm32" || return 1

  echo "== snapshot comparison (golden-file) =="
  if [[ ! -f "$SNAPSHOT" ]]; then
    echo "::error::snapshot missing: $SNAPSHOT — run --update-snapshot and commit it"
    return 1
  fi
  if ! diff -u "$SNAPSHOT" <(printf '%s\n' "$graph") ; then
    echo "::error::wasm graph drifted from snapshot. If intentional: ci/check_wasm_graph.sh --update-snapshot, commit, and justify in the PR."
    return 1
  fi
  echo "  snapshot OK"

  echo "== reverse boundary: scriptbots-core default graph =="
  local core_graph
  core_graph="$(core_default_graph)" || { echo "::error::core graph resolution failed"; return 1; }
  local bad=""
  for pat in fsci- fnx- fp- frankenpandas ft- fnp-; do
    bad+="$(printf '%s\n' "$core_graph" | grep -E "^${pat}" || true)"
  done
  if [[ -n "$bad" ]]; then
    echo "::error::analytics/brain franken crates leaked into scriptbots-core default graph:"
    printf '  %s\n' $bad
    return 1
  fi
  echo "  core default graph clean"
}

self_test() {
  export WASM_GUARD_SELF_TEST=1
  echo "== self-test: denylist logic against synthetic graphs =="
  local good_graph=$'scriptbots-core\nscriptbots-web\nserde\npostcard'
  local bad_graph=$'scriptbots-core\nrayon\nfnx-classes'
  if ! check_denylist "$good_graph" "fixture-good" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — clean graph rejected"; return 1
  fi
  echo "  PASS: clean graph accepted"
  if check_denylist "$bad_graph" "fixture-bad" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — rayon/fnx-classes not caught"; return 1
  fi
  echo "  PASS: denied crates caught"
  echo "== self-test PASSED =="
}

case "${1:-}" in
  --self-test) self_test ;;
  --update-snapshot)
    mkdir -p "$(dirname "$SNAPSHOT")"
    wasm_graph > "$SNAPSHOT"
    echo "snapshot written: $SNAPSHOT ($(grep -c . "$SNAPSHOT") crates)"
    ;;
  *) main ;;
esac
