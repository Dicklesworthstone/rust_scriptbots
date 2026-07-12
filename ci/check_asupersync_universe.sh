#!/usr/bin/env bash
# ci/check_asupersync_universe.sh — bd-2z0.8.17 (program bd-2js6)
#
# Guards the single-version-universe invariant for crates whose TYPES cross
# consumer boundaries. Three consumers of asupersync are converging in this
# workspace: fsqlite (pinned git dep), fastmcp_rust (bd-2z0.8.7.1), and our own
# direct dependency (bd-2z0.4.12; bd-2z0.4.3 decided =0.3.6). Cargo unifies
# caret-compatible 0.3.x into ONE compiled crate — which is required, because
# a Cx built by our runtime must be THE SAME TYPE as the Cx fsqlite's
# AsyncConnection methods take. Two entries in the lock = two type universes =
# compile errors at best, subtle trait-object incompatibilities at worst.
#
# Failure directions caught:
#   1. Split universe: >1 lock entry for the crate (semver-incompatible reqs).
#   2. (Informational) consumer list printed on every run for drift forensics.
#
# Fix playbook (in order):
#   a. Prefer widening OUR requirement to a caret floor (e.g. ">=0.3.6, <0.4")
#      over exact "=x.y.z" pins — exact pins FORCE a split the moment another
#      consumer floats forward.
#   b. If two consumers declare incompatible ranges, escalate to the
#      bd-2z0.8 lane owner; do not "fix" by vendoring or renaming.
#
# Parameterized: check additional families with
#   ci/check_asupersync_universe.sh asupersync ftui
# (ftui coverage becomes mandatory when bd-2z0.8.8 lands the ftui family —
#  all ftui-* crates must resolve to one version set.)
#
# Usage:
#   ci/check_asupersync_universe.sh                 # default: asupersync
#   ci/check_asupersync_universe.sh --self-test     # fixture proof
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="${REPO_ROOT}/Cargo.lock"

check_crate() {
  local lock_file="$1" crate="$2"
  local versions
  versions="$(awk -v crate="$crate" '
    $0 == "[[package]]" { in_pkg=1; name=""; ver=""; next }
    in_pkg && /^name = / { gsub(/name = |"/,""); name=$0 }
    in_pkg && /^version = / { gsub(/version = |"/,""); ver=$0
      if (name == crate) print ver
      in_pkg=0 }
  ' "$lock_file")"
  local count
  count="$(printf '%s' "$versions" | grep -c . || true)"

  echo "== universe check: $crate =="
  if (( count == 0 )); then
    echo "  not present in lock (nothing to check)"
    return 0
  fi
  echo "  resolved version(s):"
  printf '    %s\n' $versions
  if (( count > 1 )); then
    echo "::error::$crate resolves to $count versions — split type universe!"
    cat <<REMEDY
Remediation (bd-2z0.8.17 playbook):
  1. Run: cargo tree --locked -i $crate   (once per resolved version:
     cargo tree --locked -i $crate@<ver>) to list the consumers forcing each.
  2. Prefer widening OUR requirement to a caret floor over exact pins.
  3. If consumer ranges are irreconcilable, escalate to the bd-2z0.8 lane —
     never vendor/rename around it.
REMEDY
    return 1
  fi
  echo "  OK: single universe ($versions)"
  return 0
}

check_family_prefix() {
  # For crate FAMILIES (e.g. ftui-*): every member must share one version.
  local lock_file="$1" prefix="$2"
  local pairs
  pairs="$(awk -v pre="$prefix" '
    $0 == "[[package]]" { in_pkg=1; name=""; next }
    in_pkg && /^name = / { gsub(/name = |"/,""); name=$0 }
    in_pkg && /^version = / { gsub(/version = |"/,""); ver=$0
      if (index(name, pre) == 1) print name "@" ver
      in_pkg=0 }
  ' "$lock_file")"
  [[ -z "$pairs" ]] && { echo "== family check: ${prefix}* not present =="; return 0; }
  local nvers
  nvers="$(printf '%s\n' "$pairs" | sed -E 's/^.*@//' | sort -u | wc -l | tr -d ' ')"
  echo "== family check: ${prefix}* =="
  printf '    %s\n' $pairs
  if (( nvers > 1 )); then
    echo "::error::${prefix}* family spans $nvers distinct versions — must be one release set"
    return 1
  fi
  echo "  OK: family uniform"
  return 0
}

self_test() {
  local tmp; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
  cat > "$tmp/split.lock" <<'FIXTURE'
[[package]]
name = "asupersync"
version = "0.3.4"

[[package]]
name = "asupersync"
version = "0.3.6"
FIXTURE
  cat > "$tmp/single.lock" <<'FIXTURE'
[[package]]
name = "asupersync"
version = "0.3.6"
FIXTURE
  echo "== self-test 1: split lock must FAIL =="
  if check_crate "$tmp/split.lock" asupersync >/dev/null 2>&1; then
    echo "::error::self-test FAILED — split universe not caught"; return 1
  fi
  echo "  PASS"
  echo "== self-test 2: single lock must PASS =="
  check_crate "$tmp/single.lock" asupersync >/dev/null 2>&1 || {
    echo "::error::self-test FAILED — single universe rejected"; return 1; }
  echo "  PASS"
  echo "== self-test PASSED =="
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit $?
fi

rc=0
for crate in "${@:-asupersync}"; do
  case "$crate" in
    ftui) check_family_prefix "$LOCK" "ftui" || rc=1 ;;
    *)    check_crate "$LOCK" "$crate" || rc=1 ;;
  esac
done
exit $rc
