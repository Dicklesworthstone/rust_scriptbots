#!/usr/bin/env bash
# ci/check_audit_suppressions.sh — bd-2z0.8.14 (program bd-2z0.8)
#
# Prohibits undocumented cargo-audit suppressions and suppressions whose
# flagged package no longer exists at the reviewed version in Cargo.lock.
#
# The contract (see ci/configs/audit_advisories.toml for the full policy):
#   1. Every `--ignore RUSTSEC-*` flag in .github/workflows/*.yml has exactly
#      one [[advisory]] entry in the registry (no undocumented suppression).
#   2. Every registry entry has a live `--ignore` flag (no stale
#      justification after the suppression is removed).
#   3. Every entry carries complete id/package/locked_version/owning_bead/
#      justification/removal_trigger/reviewed_at/reviewed_by fields.
#   4. The entry's package+locked_version is still present in Cargo.lock —
#      an absent package means the suppression must be deleted, and a version
#      drift means the risk was never re-reviewed at the new version.
#   5. .cargo/audit.toml must not exist: it would be a second suppression
#      channel bypassing the registry.
#
# Usage:
#   ci/check_audit_suppressions.sh              # check the repo
#   ci/check_audit_suppressions.sh --self-test  # doctored-fixture proof
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REQUIRED_FIELDS=(id package locked_version owning_bead justification removal_trigger reviewed_at reviewed_by)

# Emit one tab-separated line per [[advisory]] entry, fields in
# REQUIRED_FIELDS order; missing fields are empty (validation catches them).
parse_registry() { # $1 = registry path
  awk '
    function emit() {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
        f["id"], f["package"], f["locked_version"], f["owning_bead"], \
        f["justification"], f["removal_trigger"], f["reviewed_at"], f["reviewed_by"]
    }
    /^\[\[advisory\]\][ \t]*$/ { if (seen) emit(); seen=1; delete f; next }
    /^[a-z_]+[ \t]*=[ \t]*"/ {
      key=$0; sub(/[ \t]*=.*/, "", key)
      val=$0; sub(/^[a-z_]+[ \t]*=[ \t]*"/, "", val); sub(/"[ \t]*$/, "", val)
      f[key]=val
      next
    }
    END { if (seen) emit() }
  ' "$1"
}

extract_workflow_ignores() { # $1 = workflows dir
  grep -hoE -- '--ignore[[:space:]=]+RUSTSEC-[0-9]{4}-[0-9]+' "$1"/*.yml 2>/dev/null \
    | grep -oE 'RUSTSEC-[0-9]{4}-[0-9]+' | sort -u
}

lock_pkg_versions() { # $1 = lock path, $2 = package name
  awk -v pkg="$2" '
    /^\[\[package\]\]/ { name=""; next }
    /^name = / { v=$0; gsub(/name = "|"/, "", v); name=v; next }
    /^version = / { if (name == pkg) { v=$0; gsub(/version = "|"/, "", v); print v } }
  ' "$1" | sort -u
}

check() { # $1 = workflows dir, $2 = registry, $3 = lock, $4 = cargo dir
  local workflows="$1" registry="$2" lock="$3" cargo_dir="$4"
  echo "== cargo-audit suppression-documentation guard =="

  if [[ -f "$cargo_dir/audit.toml" ]]; then
    echo "::error::$cargo_dir/audit.toml exists — cargo-audit ignores must live only in .github/workflows/ci.yml flags plus the ci/configs/audit_advisories.toml registry (bd-2z0.8.14)"
    return 1
  fi

  local ignores
  ignores="$(extract_workflow_ignores "$workflows" || true)"
  echo "  workflow --ignore flags: ${ignores:-<none>}"

  local entries
  entries="$(parse_registry "$registry")"
  local registry_ids
  registry_ids="$(printf '%s\n' "$entries" | cut -f1 | grep -E '^RUSTSEC-' | sort -u || true)"
  echo "  registry advisory ids: ${registry_ids:-<none>}"

  # Direction 1: every workflow ignore must be documented.
  local bad=0
  while IFS= read -r adv; do
    [[ -z "$adv" ]] && continue
    if ! printf '%s\n' "$registry_ids" | grep -qxF "$adv"; then
      echo "::error::undocumented advisory suppression: $adv is ignored in the workflows but has no [[advisory]] entry in $registry — add a complete entry (justification, owning bead, removal trigger) or remove the flag"
      bad=1
    fi
  done <<< "$ignores"

  # Direction 2: every registry entry must have a live flag.
  while IFS= read -r adv; do
    [[ -z "$adv" ]] && continue
    if ! printf '%s\n' "$ignores" | grep -qxF "$adv"; then
      echo "::error::stale advisory registry entry: $adv is documented in $registry but no workflow ignores it — delete the entry together with its suppression (bd-2z0.8.14)"
      bad=1
    fi
  done <<< "$registry_ids"

  # Per-entry field and lock-graph validation.
  local n=0
  while IFS=$'\t' read -r id package locked_version owning_bead justification removal_trigger reviewed_at reviewed_by; do
    [[ -z "$id$package" ]] && continue
    n=$((n + 1))
    local entry_label="${id:-<entry $n>}"
    if ! printf '%s' "$id" | grep -qE '^RUSTSEC-[0-9]{4}-[0-9]+$'; then
      echo "::error::$entry_label: id must match RUSTSEC-YYYY-NNNN, got '${id:-<empty>}'"
      bad=1
    fi
    local field value
    for field in "${REQUIRED_FIELDS[@]}"; do
      value=""
      case "$field" in
        id) value="$id" ;; package) value="$package" ;; locked_version) value="$locked_version" ;;
        owning_bead) value="$owning_bead" ;; justification) value="$justification" ;;
        removal_trigger) value="$removal_trigger" ;; reviewed_at) value="$reviewed_at" ;;
        reviewed_by) value="$reviewed_by" ;;
      esac
      if [[ -z "$value" ]]; then
        echo "::error::$entry_label: required field '$field' is missing or empty — undocumented suppressions are prohibited (bd-2z0.8.14)"
        bad=1
      fi
    done
    if [[ -n "$owning_bead" ]] && ! printf '%s' "$owning_bead" | grep -qE '^bd-[A-Za-z0-9.-]+$'; then
      echo "::error::$entry_label: owning_bead '$owning_bead' must name a beads issue (bd-*)"
      bad=1
    fi
    if [[ -n "$package" ]]; then
      local versions
      versions="$(lock_pkg_versions "$lock" "$package" || true)"
      if [[ -z "$versions" ]]; then
        echo "::error::$entry_label: package '$package' is absent from Cargo.lock — a suppression for an absent package must be removed, not kept (bd-2z0.8.14)"
        bad=1
      elif [[ -n "$locked_version" ]] && ! printf '%s\n' "$versions" | grep -qxF "$locked_version"; then
        echo "::error::$entry_label: registry records $package $locked_version but Cargo.lock has $(printf '%s ' $versions)— the advisory path was never re-reviewed at the new version; re-audit and update or remove the suppression"
        bad=1
      fi
    fi
  done <<< "$entries"
  (( bad == 0 )) || return 1

  local entry_count
  entry_count="$(printf '%s\n' "$registry_ids" | grep -c . || true)"
  echo "  OK: $entry_count documented suppression(s), registry == workflow flags, lock graph matches"
  return 0
}

self_test() {
  local tmp; tmp="$(mktemp -d "${TMPDIR:-/tmp}/check_audit_suppressions.XXXXXX")"
  trap 'rm -rf "$tmp"' RETURN
  mkdir -p "$tmp/workflows" "$tmp/.cargo"

  cat > "$tmp/workflows/ci.yml" <<'YAML'
      run: |
        cargo audit \
          --ignore RUSTSEC-2026-0194 \
          --ignore RUSTSEC-2026-0195
YAML
  cat > "$tmp/registry.toml" <<'TOML'
schema = 1

[[advisory]]
id = "RUSTSEC-2026-0194"
package = "quick-xml"
locked_version = "0.39.4"
owning_bead = "bd-2z0.8.14"
justification = "Trusted vendored XML only."
removal_trigger = "wayland-scanner release on quick-xml >= 0.41."
reviewed_at = "2026-07-17"
reviewed_by = "self-test"

[[advisory]]
id = "RUSTSEC-2026-0195"
package = "quick-xml"
locked_version = "0.39.4"
owning_bead = "bd-2z0.8.14"
justification = "NsReader API unused by the sole consumer."
removal_trigger = "wayland-scanner release on quick-xml >= 0.41."
reviewed_at = "2026-07-17"
reviewed_by = "self-test"
TOML
  cat > "$tmp/Cargo.lock" <<'LOCK'
[[package]]
name = "quick-xml"
version = "0.39.4"

[[package]]
name = "quick-xml"
version = "0.41.0"

[[package]]
name = "wayland-scanner"
version = "0.31.10"
LOCK

  echo "== self-test 1: consistent fixture must PASS =="
  check "$tmp/workflows" "$tmp/registry.toml" "$tmp/Cargo.lock" "$tmp/.cargo" >/dev/null 2>&1 || {
    echo "::error::self-test FAILED — consistent fixture rejected"; return 1; }
  echo "  PASS"

  echo "== self-test 2: undocumented ignore must FAIL =="
  sed -i '' 's/--ignore RUSTSEC-2026-0195/--ignore RUSTSEC-2026-0195 \\\n          --ignore RUSTSEC-2099-9999/' "$tmp/workflows/ci.yml" 2>/dev/null || \
    sed -i 's/--ignore RUSTSEC-2026-0195/--ignore RUSTSEC-2026-0195 \\\n          --ignore RUSTSEC-2099-9999/' "$tmp/workflows/ci.yml"
  if check "$tmp/workflows" "$tmp/registry.toml" "$tmp/Cargo.lock" "$tmp/.cargo" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — undocumented suppression not caught"; return 1
  fi
  echo "  PASS"

  echo "== self-test 3: stale registry entry must FAIL =="
  cat > "$tmp/workflows/ci.yml" <<'YAML'
      run: |
        cargo audit \
          --ignore RUSTSEC-2026-0194 \
          --ignore RUSTSEC-2026-0195
YAML
  cat >> "$tmp/registry.toml" <<'TOML'

[[advisory]]
id = "RUSTSEC-2020-0001"
package = "quick-xml"
locked_version = "0.39.4"
owning_bead = "bd-2z0.8.14"
justification = "Removed suppression whose entry was left behind."
removal_trigger = "Already removed."
reviewed_at = "2026-07-17"
reviewed_by = "self-test"
TOML
  if check "$tmp/workflows" "$tmp/registry.toml" "$tmp/Cargo.lock" "$tmp/.cargo" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — stale registry entry not caught"; return 1
  fi
  echo "  PASS"

  echo "== self-test 4: missing required field must FAIL =="
  sed -i '' '/^justification = "Removed suppression/d' "$tmp/registry.toml" 2>/dev/null || \
    sed -i '/^justification = "Removed suppression/d' "$tmp/registry.toml"
  if check "$tmp/workflows" "$tmp/registry.toml" "$tmp/Cargo.lock" "$tmp/.cargo" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — missing field not caught"; return 1
  fi
  echo "  PASS"

  echo "== self-test 5: absent package must FAIL =="
  cat > "$tmp/registry.toml" <<'TOML'
schema = 1

[[advisory]]
id = "RUSTSEC-2026-0194"
package = "quick-xml"
locked_version = "0.39.4"
owning_bead = "bd-2z0.8.14"
justification = "Trusted vendored XML only."
removal_trigger = "wayland-scanner release on quick-xml >= 0.41."
reviewed_at = "2026-07-17"
reviewed_by = "self-test"

[[advisory]]
id = "RUSTSEC-2026-0195"
package = "gone-crate"
locked_version = "1.0.0"
owning_bead = "bd-2z0.8.14"
justification = "Package removed from the graph but suppression kept."
removal_trigger = "None — should have been deleted."
reviewed_at = "2026-07-17"
reviewed_by = "self-test"
TOML
  if check "$tmp/workflows" "$tmp/registry.toml" "$tmp/Cargo.lock" "$tmp/.cargo" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — absent-package suppression not caught"; return 1
  fi
  echo "  PASS"

  echo "== self-test 6: unreviewed version drift must FAIL =="
  sed -i '' 's/locked_version = "0.39.4"/locked_version = "0.40.0"/' "$tmp/registry.toml" 2>/dev/null || \
    sed -i 's/locked_version = "0.39.4"/locked_version = "0.40.0"/' "$tmp/registry.toml"
  sed -i '' 's/package = "gone-crate"/package = "quick-xml"/' "$tmp/registry.toml" 2>/dev/null || \
    sed -i 's/package = "gone-crate"/package = "quick-xml"/' "$tmp/registry.toml"
  if check "$tmp/workflows" "$tmp/registry.toml" "$tmp/Cargo.lock" "$tmp/.cargo" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — unreviewed version drift not caught"; return 1
  fi
  echo "  PASS"

  echo "== self-test 7: .cargo/audit.toml side channel must FAIL =="
  printf '[advisories]\nignore = ["RUSTSEC-2026-0194"]\n' > "$tmp/.cargo/audit.toml"
  if check "$tmp/workflows" "$tmp/registry.toml" "$tmp/Cargo.lock" "$tmp/.cargo" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — audit.toml side channel not caught"; return 1
  fi
  rm -f "$tmp/.cargo/audit.toml"
  echo "  PASS"
  echo "== self-test PASSED =="
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
else
  check \
    "$REPO_ROOT/.github/workflows" \
    "$REPO_ROOT/ci/configs/audit_advisories.toml" \
    "$REPO_ROOT/Cargo.lock" \
    "$REPO_ROOT/.cargo"
fi
