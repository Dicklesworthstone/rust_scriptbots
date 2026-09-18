#!/usr/bin/env bash
# MAP-Elites Archive Persistence Roundtrip E2E Proof (bd-16g.6.1).
#
# Verifies:
# 1. Archive space and cell persistence into FrankenSQLite database.
# 2. Additive schema migration V2 -> V3 preserving standard scientific tables.
# 3. Reloading archive into memory and byte-identical cell/space roundtrip.
# 4. StorageReader offline verification after database close.
# 5. Core export, reload, diff, and resurrect lifecycle verification.

set -euo pipefail

fail() {
  printf 'e2e_map_elites_archive_roundtrip: %s\n' "$1" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

if [[ "${CI:-}" == "true" ]]; then
  cargo_runner=(cargo)
else
  command -v rch >/dev/null 2>&1 ||
    fail "rch is required outside CI; local Cargo fallback is forbidden"
  cargo_runner=(rch exec -- cargo)
fi

log_file="$(mktemp)"
printf 'e2e_map_elites_archive_roundtrip: running storage persistence roundtrip test...\n'

"${cargo_runner[@]}" test \
  -p scriptbots-storage \
  --test persistence_integration \
  test_map_elites_archive_persistence_roundtrip \
  -- \
  --exact \
  --nocapture 2>&1 | tee "$log_file"

core_log_file="$(mktemp)"
printf 'e2e_map_elites_archive_roundtrip: running core export/reload/diff/resurrect test...\n'

"${cargo_runner[@]}" test \
  -p scriptbots-core \
  --features economy-faults \
  --test map_elites \
  test_map_elites_export_reload_diff_resurrect_replay_e2e \
  -- \
  --exact \
  --nocapture 2>&1 | tee "$core_log_file"

printf 'e2e_map_elites_archive_roundtrip: verifying test results and phase evidence...\n'

python3 -c '
import sys, re, json

storage_log = open(sys.argv[1]).read()
core_log = open(sys.argv[2]).read()

# 1. Assert storage persistence test passed
if not re.search(r"test result: ok\.\s+1 passed;\s+0 failed", storage_log):
    print("Storage archive persistence test failed", file=sys.stderr)
    sys.exit(1)

# 2. Assert core e2e test passed
if not re.search(r"test result: ok\.\s+1 passed;\s+0 failed", core_log):
    print("Core MAP-Elites E2E test failed", file=sys.stderr)
    sys.exit(1)

# 3. Assert all required E2E phases were completed
required_phases = {
    "start",
    "archive_populated",
    "archive_exported_and_reloaded",
    "archive_diff_verified",
    "resurrect_applied",
    "replay_digest_verified",
    "completed",
}
seen_phases = set()
for line in core_log.splitlines():
    m = re.search(r"\{\"schema\":\"scriptbots\.qd-archive\.e2e\.v1\".*\}", line)
    if m:
        data = json.loads(m.group(0))
        seen_phases.add(data.get("phase"))

missing = required_phases - seen_phases
if missing:
    print(f"Missing required phases in core E2E run: {missing}", file=sys.stderr)
    sys.exit(1)

print("e2e_map_elites_archive_roundtrip: all persistence and lifecycle checks passed.")
' "$log_file" "$core_log_file"

rm -f "$log_file" "$core_log_file"
printf 'e2e_map_elites_archive_roundtrip: PASS\n'
