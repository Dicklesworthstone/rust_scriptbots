#!/usr/bin/env bash
# GPU Sense Parity & Certification E2E Proof (bd-16g.15.3).
#
# Verifies:
# 1. Exact bit-level parity between CPU reference and GPU compute sensing pipeline.
# 2. Fault injection mode detection with exact divergence coordinates (tick, agent_id, eye_idx, channel).
# 3. Documentation drift check for docs/gpu-lane.md against authoritative generator.
# 4. CLI pre-storage refusal for unvalidated GPU target without --allow-approximate-sense.
# 5. RunManifestV3 propagation of approximate GPU sensing (reproducible=false, warning, limitation).

set -euo pipefail

fail() {
  printf 'e2e_gpu_sense_parity: %s\n' "$1" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

if [[ "${CI:-}" == "true" ]]; then
  cargo_runner=(env RUST_LOG="scriptbots::sense=info,info" cargo)
else
  command -v rch >/dev/null 2>&1 ||
    fail "rch is required outside CI; local Cargo fallback is forbidden"
  cargo_runner=(rch exec -- env RUST_LOG="scriptbots::sense=info,info" cargo)
fi

log_file="$(mktemp)"
printf 'e2e_gpu_sense_parity: running GPU sense parity and drift test suite...\n'

"${cargo_runner[@]}" test \
  -p scriptbots-world-gfx \
  --lib sense_parity \
  -- \
  --nocapture 2>&1 | tee "$log_file"

printf 'e2e_gpu_sense_parity: running CLI policy and manifest propagation test suite...\n'

app_log_file="$(mktemp)"
"${cargo_runner[@]}" test \
  -p scriptbots-app \
  --test run_manifest_emitted \
  -- test_gpu_sense \
  --nocapture 2>&1 | tee "$app_log_file"

printf 'e2e_gpu_sense_parity: verifying structured parity logs and artifacts...\n'

python3 -c '
import sys, os, json, re

gfx_log = open(sys.argv[1]).read()
app_log = open(sys.argv[2]).read()

# 1. Assert gfx parity tests passed
if not re.search(r"test result: ok\.\s+\d+ passed;\s+0 failed", gfx_log):
    print("GPU sense parity tests failed in scriptbots-world-gfx", file=sys.stderr)
    sys.exit(1)

# 2. Assert app tests passed
if not re.search(r"test result: ok\.\s+\d+ passed;\s+0 failed", app_log):
    print("GPU sense CLI policy tests failed in scriptbots-app", file=sys.stderr)
    sys.exit(1)

# 3. Assert structured GPU initialization and parity run logs
combined_logs = gfx_log + "\n" + app_log

if "Starting CPU-vs-GPU sense parity gate" not in combined_logs and "Initialized GPU compute sensing pipeline" not in combined_logs:
    print("Missing GPU pipeline startup/parity diagnostic logs", file=sys.stderr)
    sys.exit(1)

if "Completed CPU-vs-GPU sense parity run" not in combined_logs and "sense numeric contract" not in combined_logs:
    print("Missing parity run completion/contract diagnostic logs", file=sys.stderr)
    sys.exit(1)

# 4. Check fixture file ci/fixtures/sense_lane_parity.json
fixture_path = "ci/fixtures/sense_lane_parity.json"
if not os.path.isfile(fixture_path):
    print(f"Missing fixture file {fixture_path}", file=sys.stderr)
    sys.exit(1)

with open(fixture_path) as f:
    report = json.load(f)

assert report["tick_count"] == 1000, "Should evaluate 1000 ticks"
assert "target" in report and "adapter" in report, "Missing adapter metadata in parity report"
assert "verdict" in report and report["verdict"] in ("exact", "approximate"), "Invalid parity report verdict"
assert report["dispatches"] == 1000, "Should record 1000 dispatches"

# 5. Check docs/gpu-lane.md
doc_path = "docs/gpu-lane.md"
if not os.path.isfile(doc_path):
    print(f"Missing doc file {doc_path}", file=sys.stderr)
    sys.exit(1)

with open(doc_path) as f:
    doc = f.read()

assert "# GPU Compute Sense Lane & Parity Gate" in doc
assert "## 2. Hardware & Target Matrix" in doc
assert "## 3. CLI Policy & Flag Semantics" in doc
assert "## 4. Manifest & Downstream Propagation" in doc

print("e2e_gpu_sense_parity: all assertions passed successfully.")
' "$log_file" "$app_log_file"

rm -f "$log_file" "$app_log_file"
printf 'e2e_gpu_sense_parity: PASS\n'
