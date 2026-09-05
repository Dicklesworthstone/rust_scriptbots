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
expected=$(yq -r '.env.SCRIPTBOTS_EXPECTED_COMMIT' "$profile")
proof_root=$(yq -r '.env.SCRIPTBOTS_PROOF_ROOT' "$profile")
target=$(yq -r '.targets[0]' "$profile")
profile_hash=$(sha256sum "$profile" | cut -d ' ' -f 1)

dsr build --tool "$tool" --target "$target" --only-native --no-sync --version "$version"

[[ $(sha256sum "$profile" | cut -d ' ' -f 1) == "$profile_hash" ]] || {
    echo "Profile changed during execution" >&2
    exit 1
}
proof="$proof_root/$version"
jq -e --arg source "$expected" \
    '.status == "pass" and .source == $source and .lane == "recipes" and .exit_code == 0' \
    "$proof/verdict.json"
jq -e '.passed > 0' "$proof/architecture-doc-examples.tests.json"
jq -e '.passed > 0' "$proof/architecture-recipes.tests.json"
echo "Literal Rust recipes and scenario controls executed; evidence: $proof"
echo "This verifies the named library recipes, not production GUI/PTY/browser migration."
