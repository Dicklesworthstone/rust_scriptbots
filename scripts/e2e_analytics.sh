#!/usr/bin/env bash
set -euo pipefail

# The real storage/CLI graph suite is one component of bd-2z0.11.9.
# The full seeded-run/report journey remains open; do not manufacture its result.
if (( $# != 2 )) || [[ ! $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    echo "Usage: DSR_CONFIG_DIR=/absolute/config $0 PROFILE UNIQUE_VERSION" >&2
    echo "Materialize ci/dsr_verify.yaml with SCRIPTBOTS_VERIFY_LANE=graphs first." >&2
    exit 2
fi
tool=$1
version=$2
[[ ${DSR_CONFIG_DIR:-} = /* ]] || { echo "DSR_CONFIG_DIR must be absolute" >&2; exit 2; }
profile="$DSR_CONFIG_DIR/repos.d/$tool.yaml"
[[ -f "$profile" ]] || { echo "Missing DSR profile: $profile" >&2; exit 2; }
[[ $(yq -r '.tool_name' "$profile") == "$tool" ]] || { echo "Profile identity mismatch" >&2; exit 2; }
lane=$(yq -r '.env.SCRIPTBOTS_VERIFY_LANE' "$profile")
[[ "$lane" == graphs || "$lane" == graphs-and-recipes ]] || {
    echo "Profile must select the graphs or graphs-and-recipes correctness lane" >&2
    exit 2
}
proof_root=$(yq -r '.env.SCRIPTBOTS_PROOF_ROOT' "$profile")
bash "$(dirname "${BASH_SOURCE[0]}")/dsr_verify.sh" --run "$tool" "$version"
echo "Storage/CLI graph acceptance executed; evidence: $proof_root/$version"
echo "Full seeded simulation/report acceptance remains with bd-2z0.11.9."
