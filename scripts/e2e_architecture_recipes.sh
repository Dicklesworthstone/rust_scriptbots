#!/usr/bin/env bash
# Execute literal architecture recipes through a materialized native DSR profile.
set -euo pipefail

if (( $# != 2 )) || [[ ! $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    echo "Usage: DSR_CONFIG_DIR=/absolute/config $0 PROFILE UNIQUE_VERSION" >&2
    echo "Materialize ci/dsr_verify.yaml with SCRIPTBOTS_VERIFY_LANE=recipes first." >&2
    exit 2
fi
tool=$1
version=$2
[[ ${DSR_CONFIG_DIR:-} = /* ]] || { echo "DSR_CONFIG_DIR must be absolute" >&2; exit 2; }
profile="$DSR_CONFIG_DIR/repos.d/$tool.yaml"
[[ -f "$profile" ]] || { echo "Missing DSR profile: $profile" >&2; exit 2; }
[[ $(yq -r '.tool_name' "$profile") == "$tool" ]] || { echo "Profile identity mismatch" >&2; exit 2; }
[[ $(yq -r '.env.SCRIPTBOTS_VERIFY_LANE' "$profile") == recipes ]] || {
    echo "Profile must select the recipes correctness lane" >&2
    exit 2
}
proof_root=$(yq -r '.env.SCRIPTBOTS_PROOF_ROOT' "$profile")
bash "$(dirname "${BASH_SOURCE[0]}")/dsr_verify.sh" --run "$tool" "$version"
proof="$proof_root/$version"
echo "Literal Rust recipes and scenario controls executed; evidence: $proof"
echo "This verifies the named library recipes, not production GUI/PTY/browser migration."
