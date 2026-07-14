#!/usr/bin/env bash
# ci/check_wasm_graph.sh — bd-2z0.8.16 (program bd-2js6)
#
# Three guards protecting build-budget and wasm-cleanliness boundaries:
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
# GUARD C (Frankentorch admission): every normal/build/dev edge in the default
# brain-ml and app graphs must remain ft-free. The explicit brain-ft graph must
# retain the reviewed eight-crate closure, immutable source revision, and
# numeric dependency universe.
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
#   ci/check_wasm_graph.sh                  # all three guards
#   ci/check_wasm_graph.sh --update-snapshot
#   ci/check_wasm_graph.sh --self-test      # denylist logic fixture proof
set -euo pipefail

# Golden ordering must not depend on the runner's locale (BSD/macOS and GNU
# `sort` otherwise disagree about '-' versus '_' in crate names).
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT="${REPO_ROOT}/ci/fixtures/wasm_graph_snapshot.txt"
FRANKENTORCH_REV="e4c6bdd5ec629ae70b40da9314da345ade012ca7"

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

# NB: stderr is DROPPED (not merged) — cargo emits "Blocking waiting for
# file lock…" chatter to stderr under contention, and one such line once
# leaked into the golden snapshot. Failures are still caught via exit code
# (pipefail) plus the plausibility checks in --update-snapshot.
wasm_graph() {
  ( cd "$REPO_ROOT" && cargo tree --locked \
      --target wasm32-unknown-unknown \
      -p scriptbots-web \
      --edges normal,build \
      --prefix none --format '{p}' 2>/dev/null ) \
    | sed -E 's/ \(.*//' | sed -E 's/ v[0-9].*$//' | sort -u
}

core_default_graph() {
  ( cd "$REPO_ROOT" && cargo tree --locked \
      -p scriptbots-core \
      --edges normal \
      --prefix none --format '{p}' 2>/dev/null ) \
    | sed -E 's/ \(.*//' | sed -E 's/ v[0-9].*$//' | sort -u
}

brain_ml_default_graph() {
  ( cd "$REPO_ROOT" && cargo tree --locked \
      -p scriptbots-brain-ml \
      --edges normal,build,dev \
      --prefix none --format '{p}' 2>/dev/null ) \
    | sed -E 's/ \(.*//' | sed -E 's/ v[0-9].*$//' | sort -u
}

app_default_graph() {
  ( cd "$REPO_ROOT" && cargo tree --locked \
      -p scriptbots-app \
      --edges normal,build,dev \
      --prefix none --format '{p}' 2>/dev/null ) \
    | sed -E 's/ \(.*//' | sed -E 's/ v[0-9].*$//' | sort -u
}

brain_ft_graph() {
  ( cd "$REPO_ROOT" && cargo tree --locked \
      -p scriptbots-brain-ml --no-default-features --features brain-ft \
      --edges normal \
      --prefix none --format '{p}' 2>/dev/null ) \
    | sed -E 's/ \(\*\)$//' | sort -u
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

check_no_ft() {
  local graph="$1" label="$2" hits
  hits="$(printf '%s\n' "$graph" | grep -E '^ft-' || true)"
  if [[ -n "$hits" ]]; then
    echo "::error::Frankentorch leaked into the feature-off $label graph:"
    while IFS= read -r hit; do
      printf '  %s\n' "$hit"
    done <<< "$hits"
    return 1
  fi
  echo "  $label: ft-* absent"
}

check_brain_ft_closure() {
  local graph="$1" normalized actual expected forbidden
  expected=$'ft-api\nft-autograd\nft-core\nft-dispatch\nft-kernel-cpu\nft-nn\nft-optim\nft-runtime'
  normalized="$(printf '%s\n' "$graph" \
    | sed -E 's/ \(.*//' | sed -E 's/ v[0-9].*$//' | sort -u)"
  actual="$(printf '%s\n' "$normalized" | grep -E '^ft-' || true)"
  if [[ "$actual" != "$expected" ]]; then
    echo "::error::brain-ft crate closure drifted from the reviewed eight-crate admission set"
    printf 'Expected:\n%s\nActual:\n%s\n' "$expected" "$actual"
    return 1
  fi
  forbidden="$(printf '%s\n' "$normalized" | grep -E '^(asupersync|ftui)$' || true)"
  if [[ -n "$forbidden" ]]; then
    echo "::error::brain-ft enabled an excluded runtime/serialization dependency:"
    while IFS= read -r dependency; do
      printf '  %s\n' "$dependency"
    done <<< "$forbidden"
    return 1
  fi
  echo "  brain-ft: exact eight-crate closure; ft-serialize/asupersync/ftui absent"
}

check_brain_ft_provenance() {
  local graph="$1" expected_suffix package unexpected=""
  expected_suffix=" v0.1.0 (https://github.com/Dicklesworthstone/frankentorch?rev=${FRANKENTORCH_REV}#${FRANKENTORCH_REV:0:8})"
  while IFS= read -r package; do
    [[ -z "$package" ]] && continue
    if [[ "$package" != ft-*"$expected_suffix" ]]; then
      unexpected+="${package}"$'\n'
    fi
  done < <(printf '%s\n' "$graph" | grep -E '^ft-' || true)
  if [[ -n "$unexpected" ]]; then
    echo "::error::brain-ft version/source drifted from 0.1.0 at ${FRANKENTORCH_REV}:"
    printf '%s' "$unexpected"
    return 1
  fi
  echo "  brain-ft: every ft-* crate is 0.1.0 at ${FRANKENTORCH_REV}"
}

check_brain_ft_numeric_universe() {
  local graph="$1" actual expected
  expected=$'half v2.7.1\nmatrixmultiply v0.3.10\nsafe_arch v0.7.4\nsafe_arch v1.0.0\nwide v0.7.33\nwide v1.5.0'
  actual="$(printf '%s\n' "$graph" \
    | awk '$1 ~ /^(half|matrixmultiply|safe_arch|wide)$/ { print $1 " " $2 }' \
    | sort -u)"
  if [[ "$actual" != "$expected" ]]; then
    echo "::error::brain-ft numeric dependency universe drifted"
    printf 'Expected:\n%s\nActual:\n%s\n' "$expected" "$actual"
    return 1
  fi
  echo "  brain-ft: half/matrixmultiply unified; reviewed wide/safe_arch split retained"
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
    while IFS= read -r dependency; do
      printf '  %s\n' "$dependency"
    done <<< "$bad"
    return 1
  fi
  echo "  core default graph clean"

  echo "== feature-off Frankentorch boundaries =="
  local brain_ml_graph app_graph
  brain_ml_graph="$(brain_ml_default_graph)" \
    || { echo "::error::scriptbots-brain-ml default graph resolution failed"; return 1; }
  app_graph="$(app_default_graph)" \
    || { echo "::error::scriptbots-app default graph resolution failed"; return 1; }
  check_no_ft "$brain_ml_graph" "scriptbots-brain-ml/default" || return 1
  check_no_ft "$app_graph" "scriptbots-app/default" || return 1

  echo "== brain-ft opt-in closure =="
  local ft_graph
  ft_graph="$(brain_ft_graph)" \
    || { echo "::error::scriptbots-brain-ml brain-ft graph resolution failed"; return 1; }
  check_brain_ft_closure "$ft_graph" || return 1
  check_brain_ft_provenance "$ft_graph" || return 1
  check_brain_ft_numeric_universe "$ft_graph" || return 1
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
  if ! check_no_ft "$good_graph" "fixture-feature-off" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — feature-off graph rejected"; return 1
  fi
  if check_no_ft $'scriptbots-brain-ml\nft-api' "fixture-feature-leak" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — ft-api leak not caught"; return 1
  fi
  echo "  PASS: feature-off ft-* boundary enforced"
  local good_ft_graph suffix
  suffix=" v0.1.0 (https://github.com/Dicklesworthstone/frankentorch?rev=${FRANKENTORCH_REV}#${FRANKENTORCH_REV:0:8})"
  good_ft_graph="$(printf '%s%s\n' \
    ft-api "$suffix" ft-autograd "$suffix" ft-core "$suffix" \
    ft-dispatch "$suffix" ft-kernel-cpu "$suffix" ft-nn "$suffix" \
    ft-optim "$suffix" ft-runtime "$suffix")"
  good_ft_graph+=$'\nhalf v2.7.1\nmatrixmultiply v0.3.10\nsafe_arch v0.7.4\nsafe_arch v1.0.0\nwide v0.7.33\nwide v1.5.0\nscriptbots-brain-ml v0.1.0'
  if ! check_brain_ft_closure "$good_ft_graph" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — reviewed brain-ft closure rejected"; return 1
  fi
  if check_brain_ft_closure "${good_ft_graph}"$'\nasupersync' >/dev/null 2>&1; then
    echo "::error::self-test FAILED — excluded asupersync not caught"; return 1
  fi
  local missing_ft_graph
  missing_ft_graph="$(printf '%s\n' "$good_ft_graph" | grep -v '^ft-runtime ')"
  if check_brain_ft_closure "$missing_ft_graph" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — missing ft-runtime not caught"; return 1
  fi
  if check_brain_ft_closure "${good_ft_graph}"$'\nft-serialize v0.1.0' >/dev/null 2>&1; then
    echo "::error::self-test FAILED — excluded ft-serialize not caught"; return 1
  fi
  if ! check_brain_ft_provenance "$good_ft_graph" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — reviewed brain-ft provenance rejected"; return 1
  fi
  if check_brain_ft_provenance "${good_ft_graph/e4c6bdd5/deadbeef}" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — wrong Frankentorch revision not caught"; return 1
  fi
  if ! check_brain_ft_numeric_universe "$good_ft_graph" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — reviewed numeric universe rejected"; return 1
  fi
  if check_brain_ft_numeric_universe "${good_ft_graph}"$'\nwide v9.9.9' >/dev/null 2>&1; then
    echo "::error::self-test FAILED — numeric version drift not caught"; return 1
  fi
  echo "  PASS: brain-ft closure, provenance, exclusions, and numeric universe enforced"
  echo "== self-test PASSED =="
}

case "${1:-}" in
  --self-test) self_test ;;
  --update-snapshot)
    mkdir -p "$(dirname "$SNAPSHOT")"
    # Harden against a failing/queued cargo: build to a temp file, validate,
    # and only then replace the snapshot. A failed resolution must never
    # leave error text in the golden file (this happened once: an offloaded
    # cargo run wrote its --locked refusal into the snapshot).
    TMP_SNAP="$(mktemp)"
    if ! wasm_graph > "$TMP_SNAP"; then
      echo "::error::cargo resolution failed; snapshot NOT updated. Output was:"
      cat "$TMP_SNAP"
      rm -f "$TMP_SNAP"
      exit 1
    fi
    if ! grep -q '^scriptbots-web$' "$TMP_SNAP" || grep -qiE '^[[:space:]]*(error|warning|blocking)' "$TMP_SNAP"; then
      echo "::error::resolution output failed plausibility checks (must contain scriptbots-web, no error/lock-wait lines); snapshot NOT updated. Output was:"
      cat "$TMP_SNAP"
      rm -f "$TMP_SNAP"
      exit 1
    fi
    mv "$TMP_SNAP" "$SNAPSHOT"
    echo "snapshot written: $SNAPSHOT ($(grep -c . "$SNAPSHOT") crates)"
    ;;
  *) main ;;
esac
