#!/usr/bin/env bash
# ci/check_brain_coupling.sh — bd-16g.13.1 (program bd-16g.13)
#
# Mechanically enforces the anti-coupling boundary around brain internals:
#   1. scriptbots-render must keep scriptbots-brain dev-only: the crate's
#      [dependencies] table must not name scriptbots-brain, and its
#      [dev-dependencies] table must (tests may construct fixtures).
#   2. No frontend may downcast to a concrete brain family: the pattern
#      downcast_ref::<(Mlp|Dwraon|Assembly) must not appear in
#      scriptbots-render, scriptbots-app, scriptbots-web, or scriptbots-core.
#      Concrete brain access belongs to scriptbots-brain and the protocol
#      adapters only.
#   3. scriptbots-core must expose exactly one sanctioned genome read path:
#      a public WorldState::agent_genome method (protocol envelope types
#      only, no brain struct crossing the boundary).
#
# A CI grep can only prove the boundary exists; it cannot prove nobody will
# route around it next week. This script keeps the rule enforced instead of
# merely true today.
#
# Usage:
#   ci/check_brain_coupling.sh              # check the repo
#   ci/check_brain_coupling.sh --self-test  # doctored-fixture proof
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

failures=0
note() { printf '%s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; failures=$((failures + 1)); }

section_deps() { # $1 = manifest path, $2 = section name to extract
  awk -v want="$2" '
    /^\[/ { section=""; if ($0 == "[" want "]") section=want; next }
    section == want { print }
  ' "$1"
}

check_render_dev_only() { # $1 = render manifest
  local manifest="$1"
  if section_deps "$manifest" dependencies | grep -q '^scriptbots-brain[[:space:]]*='; then
    fail "scriptbots-brain appears in scriptbots-render [dependencies] (must stay dev-only)"
  else
    note "ok: scriptbots-brain absent from scriptbots-render [dependencies]"
  fi
  if section_deps "$manifest" dev-dependencies | grep -q '^scriptbots-brain[[:space:]]*='; then
    note "ok: scriptbots-brain present in scriptbots-render [dev-dependencies] (fixtures allowed)"
  else
    fail "scriptbots-brain missing from scriptbots-render [dev-dependencies] (test fixtures need it)"
  fi
}

check_no_downcast() { # $1 = repo root
  local hits
  hits="$(grep -rnE 'downcast_ref::<[[:space:]]*(Mlp|Dwraon|Assembly)' \
    "$1/crates/scriptbots-render" \
    "$1/crates/scriptbots-app" \
    "$1/crates/scriptbots-web" \
    "$1/crates/scriptbots-core" 2>/dev/null || true)"
  if [ -n "$hits" ]; then
    fail "concrete-brain downcast found outside scriptbots-brain:"$'\n'"$hits"
  else
    note "ok: no concrete-brain downcast outside scriptbots-brain"
  fi
}

check_read_path() { # $1 = core lib
  if grep -q 'pub fn agent_genome(' "$1"; then
    note "ok: WorldState::agent_genome read path exists"
  else
    fail "WorldState::agent_genome read path missing (the only sanctioned genome read)"
  fi
}

run_repo_checks() {
  check_render_dev_only "$REPO_ROOT/crates/scriptbots-render/Cargo.toml"
  check_no_downcast "$REPO_ROOT"
  check_read_path "$REPO_ROOT/crates/scriptbots-core/src/lib.rs"
}

# Run the checks with a fresh failure counter and report the result code only.
rc() {
  failures=0
  run_repo_checks
  [ "$failures" -eq 0 ]
}

run_self_test() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  # 1) Baseline fixture passes.
  mkdir -p "$tmp/crates/scriptbots-render" "$tmp/crates/scriptbots-app" \
    "$tmp/crates/scriptbots-web" "$tmp/crates/scriptbots-core/src"
  cat > "$tmp/crates/scriptbots-render/Cargo.toml" <<'EOF'
[dependencies]
scriptbots-core = { path = "../scriptbots-core" }

[dev-dependencies]
scriptbots-brain = { path = "../scriptbots-brain" }
EOF
  printf 'pub fn agent_genome() {}\n' > "$tmp/crates/scriptbots-core/src/lib.rs"
  if REPO_ROOT="$tmp" rc >/dev/null 2>&1; then
    note "self-test 1 ok: compliant fixture passes"
  else
    fail "self-test 1: compliant fixture must pass"
  fi

  # 2) A production brain dep in render fails.
  cat > "$tmp/crates/scriptbots-render/Cargo.toml" <<'EOF'
[dependencies]
scriptbots-brain = { path = "../scriptbots-brain" }

[dev-dependencies]
scriptbots-brain = { path = "../scriptbots-brain" }
EOF
  if REPO_ROOT="$tmp" rc >/dev/null 2>&1; then
    fail "self-test 2: production render dependency was not detected"
  else
    note "self-test 2 ok: production render dependency detected"
  fi

  # 3) A downcast in a frontend fails.
  mkdir -p "$tmp/crates/scriptbots-app/src"
  printf 'let b = x.downcast_ref::<MlpBrain>().unwrap();\n' > "$tmp/crates/scriptbots-app/src/view.rs"
  if REPO_ROOT="$tmp" rc >/dev/null 2>&1; then
    fail "self-test 3: frontend downcast was not detected"
  else
    note "self-test 3 ok: frontend downcast detected"
  fi

  # 4) A missing read path fails.
  printf 'pub fn something_else() {}\n' > "$tmp/crates/scriptbots-core/src/lib.rs"
  if REPO_ROOT="$tmp" rc >/dev/null 2>&1; then
    fail "self-test 4: missing agent_genome read path was not detected"
  else
    note "self-test 4 ok: missing read path detected"
  fi
}

case "${1:-}" in
  "") run_repo_checks ;;
  --self-test) run_self_test ;;
  *) printf 'usage: %s [--self-test]\n' "$0" >&2; exit 2 ;;
esac

if [ "$failures" -gt 0 ]; then
  exit 1
fi
note "brain-coupling boundary checks passed"
