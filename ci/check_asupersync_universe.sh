#!/usr/bin/env bash
# ci/check_asupersync_universe.sh — bd-2z0.8.17 / bd-2d25 (program bd-2js6)
#
# Guards the single-version-universe invariant for crates whose TYPES cross
# consumer boundaries. Three consumers of asupersync converge in this
# workspace: fsqlite (pinned git dep), fastmcp_rust, and our own direct
# dependencies (scriptbots-runtime, scriptbots-app, and scriptbots-storage;
# workspace pin is exact =0.3.9). Cargo unifies caret-compatible 0.3.x into
# ONE compiled crate — which is required, because a Cx built by our runtime
# must be THE SAME TYPE as the Cx fsqlite's AsyncConnection methods take.
# Two entries in the lock = two type universes = compile errors at best,
# subtle trait-object incompatibilities at worst.
#
# Failure directions caught:
#   1. Split universe: >1 lock entry for the crate (different versions or split sources).
#   2. Consumer requirement conflict: one or more consumers declare an incompatible semver range.
#   3. Family discrepancy: members of a crate family (e.g. ftui-*) do not share one uniform release version.
#   4. Enumerates every consumer and declared requirement on both pass and fail for auditable forensics.
#
# Fix playbook (in order):
#   a. Keep the first-party exact pin (=0.3.9). Coordinate any version advancement with
#      every boundary consumer through the serialized bd-2z0.8 dependency lane.
#   b. If two consumers declare incompatible ranges, escalate to that lane;
#      do not widen ad hoc, vendor, or rename around the type split.
#
# Default covers both asupersync and ftui:
#   ci/check_asupersync_universe.sh
# Check specific crate or family:
#   ci/check_asupersync_universe.sh asupersync ftui
# Run committed fixture self-tests:
#   ci/check_asupersync_universe.sh --self-test
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="${REPO_ROOT}/Cargo.lock"
FIXTURES_DIR="${REPO_ROOT}/ci/fixtures/asupersync_universe"

evaluate_consumer_requirements() {
  local crate="$1" resolved_ver="$2" metadata_source="${3:-cargo}"
  python3 - "$crate" "$resolved_ver" "$metadata_source" <<'PYEOF'
import json, sys, subprocess

crate = sys.argv[1]
resolved_ver = sys.argv[2]
source = sys.argv[3]

if source == "none":
    sys.exit(0)

def parse_version(v):
    v = v.split('+')[0].split('-')[0].strip()
    return tuple(int(x) for x in v.split('.'))

def satisfies_req(ver_str, req_str):
    ver = parse_version(ver_str)
    req_str = req_str.strip()
    if req_str == '*' or not req_str:
        return True
    if req_str.startswith('='):
        return ver == parse_version(req_str[1:])
    if req_str.startswith('^'):
        target = parse_version(req_str[1:])
        if target[0] == 0 and target[1] > 0:
            upper = (0, target[1] + 1, 0)
        elif target[0] == 0 and target[1] == 0:
            upper = (0, 0, target[2] + 1)
        else:
            upper = (target[0] + 1, 0, 0)
        return target <= ver < upper
    if req_str.startswith('~'):
        target = parse_version(req_str[1:])
        upper = (target[0], target[1] + 1, 0)
        return target <= ver < upper
    return satisfies_req(ver_str, '^' + req_str)

if source == "cargo":
    try:
        raw = subprocess.check_output(['cargo', 'metadata', '--format-version', '1', '--locked'], stderr=subprocess.DEVNULL)
        meta = json.loads(raw)
    except Exception as e:
        print(f"  note: cargo metadata unavailable: {e}")
        sys.exit(0)
else:
    with open(source, 'r') as f:
        meta = json.load(f)

consumers = []
for pkg in meta.get('packages', []):
    for dep in pkg.get('dependencies', []):
        if dep.get('name') == crate:
            req = dep.get('req', '*')
            sat = satisfies_req(resolved_ver, req)
            consumers.append((pkg.get('name'), pkg.get('version'), req, sat))

if not consumers:
    print(f"  no consumers declaring explicit dependency on {crate}")
    sys.exit(0)

print(f"  consumers and declared requirements ({len(consumers)} total):")
conflicts = []
satisfied = []
for c in sorted(consumers):
    if c[3]:
        satisfied.append(c)
        print(f"    {c[0]} v{c[1]}: requires {c[2]} (satisfied by {resolved_ver})")
    else:
        conflicts.append(c)
        print(f"    ::error::{c[0]} v{c[1]}: requires {c[2]} (CONFLICT: incompatible with {resolved_ver})")

if conflicts:
    print(f"\n  ACTIONABLE REQUIREMENT CONFLICTS ({len(conflicts)}):")
    for bad in conflicts:
        print(f"    - {bad[0]} v{bad[1]} requires {bad[2]} but resolved version is {resolved_ver}")
    if satisfied:
        print(f"    Satisfied by current resolution ({len(satisfied)}):")
        for good in satisfied:
            print(f"      + {good[0]} v{good[1]} requires {good[2]}")
    sys.exit(1)

sys.exit(0)
PYEOF
}

check_crate() {
  local lock_file="$1" crate="$2" metadata_source="${3:-cargo}"
  local entries
  entries="$(awk -v crate="$crate" '
    function emit() {
      if (in_pkg && name == crate) {
        if (src == "") src = "local"
        print ver " (" src ")"
      }
    }
    $0 == "[[package]]" { emit(); in_pkg=1; name=""; ver=""; src=""; next }
    in_pkg && /^name = / { gsub(/name = |"/,""); name=$0 }
    in_pkg && /^version = / { gsub(/version = |"/,""); ver=$0 }
    in_pkg && /^source = / { gsub(/source = |"/,""); src=$0 }
    END { emit() }
  ' "$lock_file")"

  local count
  count="$(printf '%s' "$entries" | grep -c . || true)"

  echo "== universe check: $crate =="
  if (( count == 0 )); then
    echo "  not present in lock (nothing to check)"
    return 0
  fi

  echo "  resolved entry/entries ($count):"
  while IFS= read -r line; do
    echo "    $line"
  done <<< "$entries"

  if (( count > 1 )); then
    echo "::error::$crate resolves to $count lock entries — split type universe!"
    cat <<REMEDY
Remediation (bd-2z0.8.17 / bd-2d25 playbook):
  1. Run: cargo tree --locked -i $crate   (or cargo tree --locked -i $crate@<ver>)
     to inspect the dependency chains forcing each distinct entry.
  2. Preserve the first-party exact pin (=0.3.9) and coordinate any advancement across
     all consumers through the serialized bd-2z0.8 dependency lane.
  3. If consumer ranges are irreconcilable, escalate to that lane — never
     widen ad hoc, vendor, or rename around the type split.
REMEDY
    return 1
  fi

  local resolved_ver
  resolved_ver="$(echo "$entries" | awk '{print $1}')"

  # Validate against declared consumer requirements
  if ! evaluate_consumer_requirements "$crate" "$resolved_ver" "$metadata_source"; then
    return 1
  fi

  echo "  OK: single universe ($entries)"
  return 0
}

check_family_prefix() {
  # For crate FAMILIES (e.g. ftui-*): every member must share one uniform version.
  local lock_file="$1" prefix="$2"
  local pairs
  pairs="$(awk -v pre="$prefix" '
    function emit() {
      if (in_pkg && index(name, pre) == 1) {
        print name "@" ver
      }
    }
    $0 == "[[package]]" { emit(); in_pkg=1; name=""; ver=""; next }
    in_pkg && /^name = / { gsub(/name = |"/,""); name=$0 }
    in_pkg && /^version = / { gsub(/version = |"/,""); ver=$0 }
    END { emit() }
  ' "$lock_file")"

  echo "== family check: ${prefix}* =="
  if [[ -z "$pairs" ]]; then
    echo "  not present in lock (prepared, not yet adopted in workspace: policy =0.5.0)"
    return 0
  fi

  local nvers
  nvers="$(printf '%s\n' "$pairs" | sed -E 's/^.*@//' | sort -u | wc -l | tr -d ' ')"
  printf '  members:\n'
  while IFS= read -r line; do
    echo "    $line"
  done <<< "$pairs"

  if (( nvers > 1 )); then
    echo "::error::${prefix}* family spans $nvers distinct versions — must be one release set"
    return 1
  fi
  local uniform_ver
  uniform_ver="$(printf '%s\n' "$pairs" | sed -E 's/^.*@//' | head -1)"
  echo "  OK: family uniform ($uniform_ver)"
  return 0
}

self_test() {
  echo "=== RUNNING ASUPERSYNC UNIVERSE GUARD SELF-TEST ==="
  [ -d "$FIXTURES_DIR" ] || {
    echo "::error::fixtures directory $FIXTURES_DIR does not exist"
    return 1
  }

  echo "== self-test 1: committed positive fixture (single universe) must PASS =="
  check_crate "$FIXTURES_DIR/positive_single_universe.lock" asupersync "none" >/dev/null 2>&1 || {
    echo "::error::self-test FAILED — valid single universe fixture rejected"
    return 1
  }
  echo "  PASS"

  echo "== self-test 2: committed negative fixture (split versions) must FAIL =="
  if check_crate "$FIXTURES_DIR/negative_split_universe.lock" asupersync "none" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — split version fixture was not rejected"
    return 1
  fi
  echo "  PASS (correctly caught split versions)"

  echo "== self-test 3: committed negative fixture (split sources) must FAIL =="
  if check_crate "$FIXTURES_DIR/negative_split_sources.lock" asupersync "none" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — split sources fixture was not rejected"
    return 1
  fi
  echo "  PASS (correctly caught split sources)"

  echo "== self-test 4: committed positive ftui uniform fixture must PASS =="
  check_family_prefix "$FIXTURES_DIR/positive_ftui_uniform.lock" "ftui" >/dev/null 2>&1 || {
    echo "::error::self-test FAILED — uniform ftui family fixture rejected"
    return 1
  }
  echo "  PASS"

  echo "== self-test 5: committed negative ftui split fixture must FAIL =="
  if check_family_prefix "$FIXTURES_DIR/negative_ftui_split.lock" "ftui" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — split ftui family fixture was not rejected"
    return 1
  fi
  echo "  PASS (correctly caught split family versions)"

  echo "== self-test 6: valid consumer metadata requirements must PASS =="
  evaluate_consumer_requirements "asupersync" "0.3.9" "$FIXTURES_DIR/fixture_valid_metadata.json" >/dev/null 2>&1 || {
    echo "::error::self-test FAILED — valid consumer requirements rejected"
    return 1
  }
  echo "  PASS"

  echo "== self-test 7: conflicting consumer metadata requirements must FAIL =="
  if evaluate_consumer_requirements "asupersync" "0.3.9" "$FIXTURES_DIR/fixture_conflicting_metadata.json" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — conflicting consumer requirements was not rejected"
    return 1
  fi
  echo "  PASS (correctly caught incompatible consumer requirements)"

  echo "=== ALL SELF-TESTS PASSED ==="
  return 0
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit $?
fi

rc=0
if [[ $# -eq 0 ]]; then
  crates_to_check=(asupersync ftui)
else
  crates_to_check=("$@")
fi

for crate in "${crates_to_check[@]}"; do
  case "$crate" in
    ftui) check_family_prefix "$LOCK" "ftui" || rc=1 ;;
    *)    check_crate "$LOCK" "$crate" || rc=1 ;;
  esac
done
exit $rc
