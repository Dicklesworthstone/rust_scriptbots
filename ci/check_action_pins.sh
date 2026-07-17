#!/usr/bin/env bash
# ci/check_action_pins.sh — bd-2z0.8.14 (program bd-2z0.8)
#
# Requires every GitHub Actions `uses:` reference in .github/workflows to be
# pinned immutably:
#   - third-party actions (owner/repo[@path]) MUST pin a full 40-hex commit
#     SHA — never a tag (@v4), branch (@main/@master), or floating ref;
#   - docker:// actions MUST pin a @sha256: digest;
#   - local actions (./...) need no pin (they are this repository).
#
# A tag or branch ref lets upstream silently change what CI runs; a SHA pin is
# the only reference that cannot move. This guard keeps the pin requirement
# enforced instead of merely true today.
#
# Usage:
#   ci/check_action_pins.sh              # check the repo
#   ci/check_action_pins.sh --self-test  # doctored-fixture proof
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Emit one line per uses: reference — file:line<TAB>ref (comments stripped).
extract_uses() { # $1 = workflows dir
  local f line lineno rest
  shopt -s nullglob
  for f in "$1"/*.yml "$1"/*.yaml; do
    while IFS= read -r line; do
      lineno="${line%%:*}"
      rest="${line#*:}"
      rest="$(printf '%s' "$rest" | sed -E 's/^[[:space:]]*-?[[:space:]]*uses:[[:space:]]*//')"
      rest="${rest%%#*}"
      rest="$(printf '%s' "$rest" | sed -E 's/^[[:space:]]+//;s/[[:space:]]+$//')"
      [[ -z "$rest" ]] && continue
      printf '%s:%s\t%s\n' "$f" "$lineno" "$rest"
    done < <(grep -nE '^[[:space:]]*-?[[:space:]]*uses:[[:space:]]*' "$f" || true)
  done
}

check() { # $1 = workflows dir
  local workflows="$1"
  echo "== GitHub Actions immutable-pin guard =="

  local bad=0 count=0 location ref
  while IFS=$'\t' read -r location ref; do
    [[ -z "$ref" ]] && continue
    count=$((count + 1))
    if [[ "$ref" == ./* ]]; then
      continue # local action: this repository is the pin
    fi
    if [[ "$ref" == docker://* ]]; then
      if [[ "$ref" != *@sha256:[0-9a-f]* ]]; then
        echo "::error::$location: docker action '$ref' is not pinned to a @sha256: digest — image tags are mutable"
        bad=1
      fi
      continue
    fi
    if [[ "$ref" != *@* ]]; then
      echo "::error::$location: action '$ref' has no @<ref> at all — pin an immutable 40-hex commit SHA"
      bad=1
      continue
    fi
    local pin="${ref##*@}"
    if ! printf '%s' "$pin" | grep -qE '^[0-9a-f]{40}$'; then
      echo "::error::$location: action '$ref' is pinned to '$pin', which can move — pin the full 40-hex commit SHA instead (resolve it, e.g., with: git ls-remote https://github.com/${ref%@*}.git)"
      bad=1
    fi
  done < <(extract_uses "$workflows")
  (( bad == 0 )) || return 1

  echo "  OK: $count uses reference(s), all immutably pinned"
  return 0
}

self_test() {
  local tmp; tmp="$(mktemp -d "${TMPDIR:-/tmp}/check_action_pins.XXXXXX")"
  trap 'rm -rf "$tmp"' RETURN
  mkdir -p "$tmp/workflows"

  cat > "$tmp/workflows/ci.yml" <<'YAML'
steps:
  - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
  - uses: ./.github/actions/local-helper
YAML

  echo "== self-test 1: SHA-pinned fixture must PASS =="
  check "$tmp/workflows" >/dev/null 2>&1 || {
    echo "::error::self-test FAILED — SHA-pinned fixture rejected"; return 1; }
  echo "  PASS"

  echo "== self-test 2: tag-pinned action must FAIL =="
  printf 'steps:\n  - uses: actions/checkout@v4\n' > "$tmp/workflows/ci.yml"
  if check "$tmp/workflows" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — mutable tag pin not caught"; return 1
  fi
  echo "  PASS"

  echo "== self-test 3: branch-pinned action must FAIL =="
  printf 'steps:\n  - uses: actions/checkout@main\n' > "$tmp/workflows/ci.yml"
  if check "$tmp/workflows" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — mutable branch pin not caught"; return 1
  fi
  echo "  PASS"

  echo "== self-test 4: unpinned docker action must FAIL =="
  printf 'steps:\n  - uses: docker://alpine:3.20\n' > "$tmp/workflows/ci.yml"
  if check "$tmp/workflows" >/dev/null 2>&1; then
    echo "::error::self-test FAILED — mutable docker tag not caught"; return 1
  fi
  echo "  PASS"

  echo "== self-test 5: digest-pinned docker action must PASS =="
  printf 'steps:\n  - uses: docker://alpine@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n' > "$tmp/workflows/ci.yml"
  check "$tmp/workflows" >/dev/null 2>&1 || {
    echo "::error::self-test FAILED — digest-pinned docker action rejected"; return 1; }
  echo "  PASS"
  echo "== self-test PASSED =="
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
else
  check "$REPO_ROOT/.github/workflows"
fi
