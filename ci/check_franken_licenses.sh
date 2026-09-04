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
#   ci/check_franken_licenses.sh                # check the real repo (audit doc)
#   ci/check_franken_licenses.sh --third-party  # staleness guard for THIRD-PARTY-LICENSES.md (bd-2z0.13.6)
#   ci/check_franken_licenses.sh --community    # community license inventory & verbatim text guard (bd-2z0.13.7)
#   ci/check_franken_licenses.sh --readme-guard # README license link guard (bd-2z0.13.7)
#   ci/check_franken_licenses.sh --verify-archive [DIR|FILE] # mock-free archive extraction verification (bd-2z0.13.7)
#   ci/check_franken_licenses.sh --all          # run all guards
#   ci/check_franken_licenses.sh --self-test    # comprehensive positive & negative tamper test suite (bd-2z0.13.7)
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

CANON_RIDER_SHA="32a82e0a5754e72e51fae44b65a936c831c07376f21c90f5fb9e76897fcc3509"

calc_sha256() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$@" | cut -d' ' -f1
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$@" | cut -d' ' -f1
  else
    python3 -c "import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())"
  fi
}

# --community (bd-2z0.13.7): verifies Section 3 community dependencies inventory
# and required verbatim open-source license texts.
check_community() {
  local notice="${1:-$REPO_ROOT/THIRD-PARTY-LICENSES.md}"
  local rc=0
  echo "== community license notice guard =="
  echo "notice: $notice"
  if [[ ! -f "$notice" ]]; then
    echo "::error::$notice missing — community license obligations cannot be verified"
    return 1
  fi

  # Section 3 header
  if ! grep -q "^## 3. Community dependencies and licenses" "$notice"; then
    echo "::error::$notice missing Section 3 ('## 3. Community dependencies and licenses')"
    rc=1
  fi

  # Inventory and text sections
  if ! grep -q "^### 3.1 License category inventory" "$notice"; then
    echo "::error::$notice missing '### 3.1 License category inventory'"
    rc=1
  fi
  if ! grep -q "^### 3.2 Standard community license texts" "$notice"; then
    echo "::error::$notice missing '### 3.2 Standard community license texts'"
    rc=1
  fi

  # License categories in the inventory table
  local categories=("MIT OR Apache-2.0" "MIT" "Apache-2.0" "BSD-3-Clause" "Zlib" "ISC" "Unicode-3.0")
  for cat in "${categories[@]}"; do
    if ! grep -q "$cat" "$notice"; then
      echo "::error::$notice missing inventory category: '$cat'"
      rc=1
    fi
  done

  # Verbatim license titles and signature phrases
  local license_signatures=(
    "MIT License|Permission is hereby granted, free of charge"
    "Apache License, Version 2.0|TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION"
    "BSD 3-Clause License|Redistribution and use in source and binary forms"
    "BSD 2-Clause License|Redistribution and use in source and binary forms"
    "ISC License|Permission to use, copy, modify, and/or distribute this software"
    "Zlib License|This software is provided 'as-is'"
    "Unicode License Agreement|UNICODE, INC. LICENSE AGREEMENT"
  )
  for entry in "${license_signatures[@]}"; do
    local title="${entry%%|*}" sig="${entry##*|}"
    if ! grep -q "$title" "$notice"; then
      echo "::error::$notice missing license text heading for '$title'"
      rc=1
    elif ! grep -q "$sig" "$notice"; then
      echo "::error::$notice missing signature phrase in '$title' ('$sig')"
      rc=1
    fi
  done

  echo "-- community summary: $( [[ $rc == 0 ]] && echo OK || echo FAILED )"
  return $rc
}

# --readme-guard (bd-2z0.13.7): verifies README.md disclosure and checks
# that relative markdown links point to existing files.
check_readme_guard() {
  local readme="${1:-$REPO_ROOT/README.md}"
  local base_dir
  base_dir="$(cd "$(dirname "$readme")" && pwd)"
  local rc=0
  echo "== README license links guard =="
  echo "readme: $readme"
  if [[ ! -f "$readme" ]]; then
    echo "::error::$readme does not exist"
    return 1
  fi

  if ! grep -q "^## Licensing" "$readme"; then
    echo "::error::$readme missing required '## Licensing' section"
    rc=1
  fi

  local licensing_section
  licensing_section="$(sed -n '/^## Licensing$/,/^## /p' "$readme")"
  for req in "LICENSE" "THIRD-PARTY-LICENSES.md" "docs/licenses.md"; do
    if ! grep -q "$req" <<< "$licensing_section"; then
      echo "::error::$readme Licensing section missing link or reference to '$req'"
      rc=1
    fi
  done

  local links
  links=$(grep -oE '\[[^]]+\]\([^)]+\)' "$readme" | sed -E 's/^\[[^]]+\]\(([^)]+)\)$/\1/' | sort -u)
  local checked_count=0
  while IFS= read -r link; do
    [[ -z "$link" ]] && continue
    if [[ "$link" =~ ^(https?://|mailto:) ]]; then continue; fi
    local target="${link%%\?*}"
    target="${target%%#*}"
    [[ -z "$target" ]] && continue
    local resolved="$base_dir/$target"
    if [[ ! -e "$resolved" ]]; then
      echo "::error::broken relative link in $readme: '$link' -> '$resolved' does not exist"
      rc=1
    else
      checked_count=$((checked_count + 1))
    fi
  done <<< "$links"

  echo "-- readme-guard summary: $checked_count relative link(s) checked, $( [[ $rc == 0 ]] && echo OK || echo FAILED )"
  return $rc
}

# --verify-archive (bd-2z0.13.7): mock-free release archive extraction proof.
# Verifies that release archives (*.tar.gz, *.tar.xz, *.zip) carry complete
# notices and canonical rider block.
verify_archive() {
  local target="${1:-$REPO_ROOT/dist}"
  local lock_file="${2:-$LOCK}"
  local rc=0

  echo "== release archive third-party license verification =="
  echo "target: $target"

  local archives=()
  if [[ -d "$target" ]]; then
    while IFS= read -r f; do
      [[ -n "$f" && -f "$f" ]] && archives+=("$f")
    done < <(find "$target" -maxdepth 2 -type f \( -name "*.tar.gz" -o -name "*.tar.xz" -o -name "*.zip" \) 2>/dev/null | sort)

    if [[ ${#archives[@]} -eq 0 ]]; then
      if [[ "$target" == "$REPO_ROOT/dist" ]]; then
        echo "::notice::no release archives found in ./dist — skipping archive extraction verification (explicit unverified-platform state)"
        return 0
      else
        echo "::error::no release archives (*.tar.gz, *.tar.xz, *.zip) found in directory: $target"
        return 1
      fi
    fi
  elif [[ -f "$target" ]]; then
    archives+=("$target")
  else
    if [[ "$target" == "$REPO_ROOT/dist" ]]; then
      echo "::notice::no release archives directory (./dist) — skipping archive extraction verification (explicit unverified-platform state)"
      return 0
    else
      echo "::error::target does not exist: $target"
      return 1
    fi
  fi

  local verified_count=0
  for arch in "${archives[@]}"; do
    echo "Verifying release archive: $arch"
    local ext=""
    case "$arch" in
      *.tar.xz) ext="tar.xz" ;;
      *.tar.gz) ext="tar.gz" ;;
      *.zip)    ext="zip" ;;
      *)
        echo "::error::$arch has unsupported/unverified archive format (supported: .tar.gz, .tar.xz, .zip)"
        rc=1
        continue
        ;;
    esac

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local extract_err=0
    case "$ext" in
      tar.xz)
        tar -xJf "$arch" -C "$tmp_dir" >/dev/null 2>&1 || extract_err=1
        ;;
      tar.gz)
        tar -xzf "$arch" -C "$tmp_dir" >/dev/null 2>&1 || extract_err=1
        ;;
      zip)
        if command -v unzip >/dev/null 2>&1; then
          unzip -q "$arch" -d "$tmp_dir" >/dev/null 2>&1 || extract_err=1
        else
          python3 -m zipfile -e "$arch" "$tmp_dir" >/dev/null 2>&1 || extract_err=1
        fi
        ;;
    esac

    if (( extract_err != 0 )); then
      echo "::error::$arch extraction failed (corrupted or truncated archive)"
      find "$tmp_dir" -type f -exec rm -f {} + 2>/dev/null || true
      find "$tmp_dir" -depth -type d -exec rmdir {} + 2>/dev/null || true
      rc=1
      continue
    fi

    # Locate THIRD-PARTY-LICENSES.md
    local notice
    notice="$(find "$tmp_dir" -maxdepth 4 -name "THIRD-PARTY-LICENSES.md" 2>/dev/null | head -n 1 || true)"
    if [[ -z "$notice" || ! -f "$notice" ]]; then
      echo "::error::$arch does not contain THIRD-PARTY-LICENSES.md — distribution license obligations unmet (bd-2z0.13.6, bd-2z0.13.7)"
      find "$tmp_dir" -type f -exec rm -f {} + 2>/dev/null || true
      find "$tmp_dir" -depth -type d -exec rmdir {} + 2>/dev/null || true
      rc=1
      continue
    fi

    # Verify canonical rider SHA
    local block_sha
    block_sha="$(sed -n '/^BEGIN LICENSE TEXT$/,/^END LICENSE TEXT$/p' "$notice" | sed '1d;$d' | calc_sha256)"
    if [[ "$block_sha" != "$CANON_RIDER_SHA" ]]; then
      echo "::error::$arch notice embedded rider sha ($block_sha) != canonical ($CANON_RIDER_SHA) — rider text drifted or tampered"
      find "$tmp_dir" -type f -exec rm -f {} + 2>/dev/null || true
      find "$tmp_dir" -depth -type d -exec rmdir {} + 2>/dev/null || true
      rc=1
      continue
    fi

    # Verify franken family coverage
    local names
    names="$(grep -E '^name = "' "$lock_file" | sed -E 's/^name = "([^"]+)"$/\1/' | sort -u)"
    local missing_families=0
    while IFS= read -r name; do
      [[ -z "$name" ]] && continue
      local token=""
      for entry in "${FAMILIES[@]}"; do
        local pat="${entry%%|*}" tok="${entry##*|}"
        if [[ "$name" =~ $pat ]]; then token="$tok"; break; fi
      done
      if [[ -z "$token" && "$name" == franken* ]]; then token="$name"; fi
      [[ -z "$token" ]] && continue
      if ! grep -q -- "$token" "$notice"; then
        echo "::error::$arch notice missing franken crate '$name' (token '$token')"
        missing_families=$((missing_families + 1))
      fi
    done <<< "$names"

    if (( missing_families > 0 )); then
      echo "::error::$arch notice missing $missing_families franken families from lockfile"
      find "$tmp_dir" -type f -exec rm -f {} + 2>/dev/null || true
      find "$tmp_dir" -depth -type d -exec rmdir {} + 2>/dev/null || true
      rc=1
      continue
    fi

    # Verify community license sections & texts
    if ! check_community "$notice" >/dev/null 2>&1; then
      echo "::error::$arch notice failed community license verification"
      find "$tmp_dir" -type f -exec rm -f {} + 2>/dev/null || true
      find "$tmp_dir" -depth -type d -exec rmdir {} + 2>/dev/null || true
      rc=1
      continue
    fi

    local arch_size
    arch_size="$(wc -c < "$arch" | tr -d ' ')"
    echo "  [VERIFIED ARCHIVE] path=$arch size=${arch_size}B notice_rel=$(basename "$notice") rider_sha=$block_sha franken_coverage=complete community_notices=complete"
    verified_count=$((verified_count + 1))

    find "$tmp_dir" -type f -exec rm -f {} + 2>/dev/null || true
    find "$tmp_dir" -depth -type d -exec rmdir {} + 2>/dev/null || true
  done

  echo "-- archive extraction summary: $verified_count archive(s) verified, $( [[ $rc == 0 ]] && echo OK || echo FAILED )"
  return $rc
}

# --third-party (bd-2z0.13.6, bd-2z0.13.7): THIRD-PARTY-LICENSES.md ships in release
# archives to satisfy the rider's include-with-distribution obligation.
# Invariants:
# (1) Embedded rider license block matches canonical SHA exactly;
# (2) Every franken family present in the lock is named in the notice file;
# (3) Community dependencies inventory and verbatim texts are complete;
# (4) README license disclosure links resolve to existing files.
check_third_party() {
  local lock_file="$1" notice="$2" rc=0
  echo "== third-party notice staleness guard =="
  if [[ ! -f "$notice" ]]; then
    echo "::error::$notice missing — release artifacts cannot satisfy the rider obligation"
    return 1
  fi
  local block_sha
  block_sha="$(sed -n '/^BEGIN LICENSE TEXT$/,/^END LICENSE TEXT$/p' "$notice" | sed '1d;$d' | calc_sha256)"
  if [[ "$block_sha" != "$CANON_RIDER_SHA" ]]; then
    echo "::error::embedded license block sha ($block_sha) != canonical ($CANON_RIDER_SHA) — rider text drifted or was reformatted"
    rc=1
  else
    echo "  license block: canonical ($block_sha)"
  fi

  local names
  names="$(grep -E '^name = "' "$lock_file" | sed -E 's/^name = "([^"]+)"$/\1/' | sort -u)"
  local missing=0
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    local token=""
    for entry in "${FAMILIES[@]}"; do
      local pat="${entry%%|*}" tok="${entry##*|}"
      if [[ "$name" =~ $pat ]]; then token="$tok"; break; fi
    done
    if [[ -z "$token" && "$name" == franken* ]]; then token="$name"; fi
    [[ -z "$token" ]] && continue
    if ! grep -q -- "$token" "$notice"; then
      echo "::error::franken crate '$name' (token '$token') in Cargo.lock but absent from $notice — add it to §2 in this PR"
      missing=$((missing + 1))
    fi
  done <<< "$names"
  echo "-- third-party summary: rider-sha $( [[ $rc == 0 ]] && echo OK || echo DRIFTED ), $missing family name(s) missing from notice"

  # Run community license check
  if ! check_community "$notice"; then
    rc=1
  fi

  # Run README link guard
  if ! check_readme_guard "$REPO_ROOT/README.md"; then
    rc=1
  fi

  (( rc == 0 && missing == 0 ))
}

self_test() {
  local tmp
  tmp="$(mktemp -d)"
  cleanup() {
    find "$tmp" -type f -exec rm -f {} + 2>/dev/null || true
    find "$tmp" -depth -type d -exec rmdir {} + 2>/dev/null || true
  }
  trap cleanup RETURN

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

  echo "== self-test 3: third-party rider SHA drift must FAIL =="
  cat > "$tmp/tampered_notice.md" <<'FIXTURE'
BEGIN LICENSE TEXT
This text is altered and does not match canonical sha.
END LICENSE TEXT
asupersync
FIXTURE
  cat > "$tmp/valid_lock.lock" <<'FIXTURE'
[[package]]
name = "asupersync"
version = "0.3.6"
FIXTURE
  local tp_out
  tp_out="$(check_third_party "$tmp/valid_lock.lock" "$tmp/tampered_notice.md" 2>&1 || true)"
  if grep -q "rider text drifted or was reformatted" <<< "$tp_out"; then
    echo "  PASS: tampered rider block failed check_third_party"
  else
    echo "::error::self-test FAILED — tampered rider block was not detected"
    printf '%s\n' "$tp_out"
    return 1
  fi

  echo "== self-test 4: third-party missing franken family must FAIL =="
  cat > "$tmp/missing_notice.md" <<FIXTURE
BEGIN LICENSE TEXT
$(sed -n '/^BEGIN LICENSE TEXT$/,/^END LICENSE TEXT$/p' "$REPO_ROOT/THIRD-PARTY-LICENSES.md" | sed '1d;$d')
END LICENSE TEXT
FIXTURE
  cat > "$tmp/missing_lock.lock" <<'FIXTURE'
[[package]]
name = "asupersync"
version = "0.3.6"
FIXTURE
  local mf_out
  mf_out="$(check_third_party "$tmp/missing_lock.lock" "$tmp/missing_notice.md" 2>&1 || true)"
  if grep -q "absent from" <<< "$mf_out"; then
    echo "  PASS: missing franken family failed check_third_party"
  else
    echo "::error::self-test FAILED — missing franken family was not detected"
    printf '%s\n' "$mf_out"
    return 1
  fi

  echo "== self-test 5: community license inventory and verbatim texts guard =="
  if check_community "$REPO_ROOT/THIRD-PARTY-LICENSES.md" >/dev/null 2>&1; then
    echo "  PASS: canonical THIRD-PARTY-LICENSES.md passed community check"
  else
    echo "::error::self-test FAILED — canonical THIRD-PARTY-LICENSES.md failed community check"
    return 1
  fi
  cat > "$tmp/truncated_notice.md" <<'FIXTURE'
## 3. Community dependencies and licenses
### 3.1 License category inventory
MIT
FIXTURE
  if ! check_community "$tmp/truncated_notice.md" >/dev/null 2>&1; then
    echo "  PASS: truncated community notice rejected as required"
  else
    echo "::error::self-test FAILED — truncated community notice was accepted"
    return 1
  fi

  echo "== self-test 6: README license links guard =="
  if check_readme_guard "$REPO_ROOT/README.md" >/dev/null 2>&1; then
    echo "  PASS: canonical README.md passed readme guard"
  else
    echo "::error::self-test FAILED — canonical README.md failed readme guard"
    return 1
  fi
  cat > "$tmp/broken_readme.md" <<'FIXTURE'
## Licensing
[LICENSE](LICENSE)
[THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md)
[docs/licenses.md](docs/licenses.md)
[broken link](nonexistent_file_xyz123.md)
FIXTURE
  touch "$tmp/LICENSE" "$tmp/THIRD-PARTY-LICENSES.md"
  mkdir -p "$tmp/docs"
  touch "$tmp/docs/licenses.md"
  if ! check_readme_guard "$tmp/broken_readme.md" >/dev/null 2>&1; then
    echo "  PASS: broken relative link in README rejected as required"
  else
    echo "::error::self-test FAILED — broken relative link in README was accepted"
    return 1
  fi

  echo "== self-test 7: mock-free release archive verification & tamper controls =="
  local pkg_dir="$tmp/mock_dist_pkg"
  mkdir -p "$pkg_dir"
  cat > "$pkg_dir/scriptbots-app" <<'FIXTURE'
#!/bin/sh
echo "scriptbots v0.3.0 mock binary"
FIXTURE
  chmod +x "$pkg_dir/scriptbots-app"
  cp "$REPO_ROOT/THIRD-PARTY-LICENSES.md" "$pkg_dir/THIRD-PARTY-LICENSES.md"

  local tgz_good="$tmp/good.tar.gz"
  tar -czf "$tgz_good" -C "$pkg_dir" .

  local zip_good="$tmp/good.zip"
  if command -v zip >/dev/null 2>&1; then
    (cd "$pkg_dir" && zip -q "$zip_good" scriptbots-app THIRD-PARTY-LICENSES.md)
  else
    python3 -c "import zipfile, os; zf = zipfile.ZipFile('$zip_good', 'w'); zf.write('$pkg_dir/scriptbots-app', arcname='scriptbots-app'); zf.write('$pkg_dir/THIRD-PARTY-LICENSES.md', arcname='THIRD-PARTY-LICENSES.md'); zf.close()"
  fi

  # Positive control: valid .tar.gz and .zip archives pass
  if verify_archive "$tgz_good" "$LOCK" >/dev/null 2>&1 \
     && verify_archive "$zip_good" "$LOCK" >/dev/null 2>&1; then
    echo "  PASS: mock-free .tar.gz and .zip archives passed extraction verification"
  else
    echo "::error::self-test FAILED — valid archive extraction verification failed"
    return 1
  fi

  # Negative tamper 1: missing notice in archive
  local pkg_no_notice="$tmp/mock_no_notice"
  mkdir -p "$pkg_no_notice"
  cp "$pkg_dir/scriptbots-app" "$pkg_no_notice/scriptbots-app"
  local tgz_no_notice="$tmp/no_notice.tar.gz"
  tar -czf "$tgz_no_notice" -C "$pkg_no_notice" .
  if ! verify_archive "$tgz_no_notice" "$LOCK" >/dev/null 2>&1; then
    echo "  PASS: archive missing THIRD-PARTY-LICENSES.md rejected as required"
  else
    echo "::error::self-test FAILED — archive missing THIRD-PARTY-LICENSES.md was accepted"
    return 1
  fi

  # Negative tamper 2: tampered rider block in archive notice
  local pkg_tampered_rider="$tmp/mock_tampered_rider"
  mkdir -p "$pkg_tampered_rider"
  cp "$pkg_dir/scriptbots-app" "$pkg_tampered_rider/scriptbots-app"
  sed 's/OpenAI/TamperedAI/g' "$REPO_ROOT/THIRD-PARTY-LICENSES.md" > "$pkg_tampered_rider/THIRD-PARTY-LICENSES.md"
  local tgz_tampered_rider="$tmp/tampered_rider.tar.gz"
  tar -czf "$tgz_tampered_rider" -C "$pkg_tampered_rider" .
  if ! verify_archive "$tgz_tampered_rider" "$LOCK" >/dev/null 2>&1; then
    echo "  PASS: archive with tampered rider block rejected as required"
  else
    echo "::error::self-test FAILED — archive with tampered rider block was accepted"
    return 1
  fi

  # Negative tamper 3: tampered community notice in archive
  local pkg_tampered_comm="$tmp/mock_tampered_comm"
  mkdir -p "$pkg_tampered_comm"
  cp "$pkg_dir/scriptbots-app" "$pkg_tampered_comm/scriptbots-app"
  sed '/^## 3. Community dependencies and licenses/,$d' "$REPO_ROOT/THIRD-PARTY-LICENSES.md" > "$pkg_tampered_comm/THIRD-PARTY-LICENSES.md"
  local tgz_tampered_comm="$tmp/tampered_comm.tar.gz"
  tar -czf "$tgz_tampered_comm" -C "$pkg_tampered_comm" .
  if ! verify_archive "$tgz_tampered_comm" "$LOCK" >/dev/null 2>&1; then
    echo "  PASS: archive with stripped community notices rejected as required"
  else
    echo "::error::self-test FAILED — archive with stripped community notices was accepted"
    return 1
  fi

  # Negative tamper 4: corrupted archive bytes
  local tgz_corrupted="$tmp/corrupted.tar.gz"
  echo "GARBAGE NOT A TAR GZ DATA 1234567890" > "$tgz_corrupted"
  if ! verify_archive "$tgz_corrupted" "$LOCK" >/dev/null 2>&1; then
    echo "  PASS: corrupted archive rejected as required"
  else
    echo "::error::self-test FAILED — corrupted archive was accepted"
    return 1
  fi

  # Negative tamper 5: unsupported archive format
  local bad_format="$tmp/unsupported_format.rar"
  touch "$bad_format"
  if ! verify_archive "$bad_format" "$LOCK" >/dev/null 2>&1; then
    echo "  PASS: unsupported archive format rejected as required"
  else
    echo "::error::self-test FAILED — unsupported archive format was accepted"
    return 1
  fi

  echo "== all self-tests PASSED =="
  return 0
}

case "${1:-}" in
  --self-test)
    self_test
    ;;
  --third-party)
    check_third_party "$LOCK" "$REPO_ROOT/THIRD-PARTY-LICENSES.md"
    ;;
  --community)
    check_community "${2:-$REPO_ROOT/THIRD-PARTY-LICENSES.md}"
    ;;
  --readme-guard)
    check_readme_guard "${2:-$REPO_ROOT/README.md}"
    ;;
  --verify-archive)
    verify_archive "${2:-$REPO_ROOT/dist}" "$LOCK"
    ;;
  --all)
    check "$LOCK" "$DOC"
    check_third_party "$LOCK" "$REPO_ROOT/THIRD-PARTY-LICENSES.md"
    verify_archive "${2:-$REPO_ROOT/dist}" "$LOCK"
    ;;
  *)
    check "$LOCK" "$DOC"
    ;;
esac
