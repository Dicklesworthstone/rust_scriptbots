#!/usr/bin/env bash
# calibrate_species.sh: Calibration runner for speciation persistence and separation constants (bd-3l5d)

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

echo "==> [1/2] Running empirical speciation calibration suite via RCH..."
rch exec -- cargo test -p scriptbots-core --test species_calibration -- --nocapture

echo "==> [2/2] Calibration verification completed successfully."
cat <<EOF
{
  "pipeline": "calibrate_species",
  "status": "success",
  "bead": "bd-3l5d",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "constants": {
    "SPECIATION_PERSISTENCE_SAMPLES": 3,
    "REPRODUCTIVE_SEPARATION_MAX_RATE": 0.05
  },
  "verdict": "calibrated"
}
EOF
