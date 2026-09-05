#!/usr/bin/env bash
# Correctness acceptance invoked by the pinned DSR profile, never nested RCH.
set -euo pipefail

refuse() {
    jq -n --arg reason "$1" '{schema:"scriptbots.verification.v1",status:"refused",reason:$reason}'
    exit 2
}

proof_version=${1:-}
[[ "$proof_version" =~ ^[a-zA-Z0-9][a-zA-Z0-9._+-]*$ ]] || refuse "missing or unsafe proof version"
[[ ${SCRIPTBOTS_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || refuse "missing pinned source commit"
[[ ${SCRIPTBOTS_VERIFY_LANE:-} == workspace || ${SCRIPTBOTS_VERIFY_LANE:-} == graphs || ${SCRIPTBOTS_VERIFY_LANE:-} == recipes ]] || refuse "unknown correctness lane"
[[ ${RCH_DISABLED:-} == 1 && ${RCH_CARGO_WRAPPER_BYPASS:-} == 1 ]] || refuse "invoke through the native DSR profile"
[[ ${SCRIPTBOTS_VERIFY_PROFILE:-} = /* && -f "$SCRIPTBOTS_VERIFY_PROFILE" ]] || refuse "missing materialized DSR profile"
[[ ${SCRIPTBOTS_PROOF_ROOT:-} = /* && -d "$SCRIPTBOTS_PROOF_ROOT" ]] || refuse "missing external proof root"
source_root=$(pwd -P)
proof_root=$(cd "$SCRIPTBOTS_PROOF_ROOT" && pwd -P)
case "$proof_root/" in "$source_root/"*) refuse "proof root must be outside source" ;; esac
[[ $(git branch --show-current) == main ]] || refuse "source must be on main"
actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$SCRIPTBOTS_EXPECTED_COMMIT" ]] || refuse "source commit mismatch"
[[ -z $(git status --porcelain --untracked-files=all) ]] || refuse "dirty source checkout"
toolchain=$(sed -nE 's/^channel = "([^"]+)"$/\1/p' rust-toolchain.toml)
[[ -n "$toolchain" ]] || refuse "missing toolchain declaration"
if ! compiler_identity=$(rustup run "$toolchain" rustc -vV 2>&1); then
    printf '%s\n' "$compiler_identity" >&2
    refuse "pinned toolchain is unavailable"
fi
[[ $(rustc -vV) == "$compiler_identity" ]] || refuse "active compiler differs from pinned toolchain"
actual_target=$(printf '%s\n' "$compiler_identity" | sed -n 's/^host: //p')
[[ "$actual_target" == "${SCRIPTBOTS_VERIFY_TARGET:-}" ]] || refuse "host/target mismatch"
[[ ${CARGO_TARGET_DIR:-} = /* ]] || refuse "missing external Cargo target directory"
case "$CARGO_TARGET_DIR/" in "$source_root/"*) refuse "Cargo target must be outside source" ;; esac

proof_dir="$proof_root/$proof_version"
mkdir "$proof_dir" || refuse "proof directory already exists or cannot be created"
cp "$SCRIPTBOTS_VERIFY_PROFILE" "$proof_dir/profile.yaml"
sha256sum "$proof_dir/profile.yaml" > "$proof_dir/profile.sha256"
git show --no-patch --format=fuller HEAD > "$proof_dir/source.txt"
rustc -vV > "$proof_dir/rustc.txt"
cargo -V > "$proof_dir/cargo.txt"
uname -a > "$proof_dir/host.txt"
sha256sum Cargo.lock rust-toolchain.toml > "$proof_dir/inputs.sha256"
mkdir "$proof_dir/tmp"
export TMPDIR="$proof_dir/tmp"
steps=0
touch "$proof_dir/commands.jsonl"
trap 'rc=$?; if (( rc != 0 )); then jq -n --arg source "$actual_commit" --arg lane "$SCRIPTBOTS_VERIFY_LANE" --argjson exit_code "$rc" --argjson completed_steps "$steps" '\''{schema:"scriptbots.verification.v1",status:"failed",source:$source,lane:$lane,exit_code:$exit_code,completed_steps:$completed_steps}'\'' > "$proof_dir/verdict.json"; fi' EXIT

run_step() {
    local name=$1 kind=$2
    shift 2
    printf '%q ' "$@" > "$proof_dir/$name.command"
    printf '\n' >> "$proof_dir/$name.command"
    local command_exit=0 log_exit=0
    set +e
    "$@" 2>&1 | tee "$proof_dir/$name.log"
    local pipeline_status=("${PIPESTATUS[@]}")
    set -e
    command_exit=${pipeline_status[0]}
    log_exit=${pipeline_status[1]}
    jq -n -c --arg name "$name" --arg kind "$kind" --arg source "$actual_commit" \
        --arg target "$actual_target" --arg command "$(cat "$proof_dir/$name.command")" \
        --arg log "$name.log" --arg log_sha256 "$(sha256sum "$proof_dir/$name.log" | cut -d ' ' -f 1)" \
        --argjson command_exit "$command_exit" --argjson log_exit "$log_exit" \
        '{name:$name,kind:$kind,source:$source,target:$target,command:$command,log:$log,log_sha256:$log_sha256,command_exit:$command_exit,log_exit:$log_exit}' \
        >> "$proof_dir/commands.jsonl"
    (( command_exit == 0 && log_exit == 0 )) || return 1
    if [[ "$kind" == test ]]; then
        # Require real executed tests; compile-only and zero-filter runs cannot pass.
        local passed
        passed=$(sed -nE 's/^test result: ok\. ([0-9]+) passed;.*/\1/p' "$proof_dir/$name.log" | awk '{n += $1} END {print n+0}')
        (( passed > 0 )) || { echo "no executed tests in $name" >&2; return 1; }
        jq -n --arg name "$name" --argjson passed "$passed" '{name:$name,passed:$passed}' > "$proof_dir/$name.tests.json"
    fi
    steps=$((steps + 1))
}

run_step formatting check cargo fmt --all --check
run_step fsqlite-pin check bash ci/check_fsqlite_pin.sh
run_step franken-licenses check bash ci/check_franken_licenses.sh
run_step asupersync-universe check bash ci/check_asupersync_universe.sh
run_step wasm-graph check bash ci/check_wasm_graph.sh
case "$SCRIPTBOTS_VERIFY_LANE" in
    workspace)
        run_step workspace-check check cargo check --locked --workspace --all-targets
        run_step workspace-clippy check cargo clippy --locked --workspace --all-targets -- -D warnings
        run_step workspace-tests test cargo test --locked --workspace -- --nocapture
        run_step core-economy-faults test cargo test --locked -p scriptbots-core --features economy-faults -- --nocapture
        ;;
    graphs)
        run_step graph-check check cargo check --locked -p scriptbots-analytics --all-targets
        run_step graph-tests test cargo test --locked -p scriptbots-analytics --test graph_reports -- --nocapture
        run_step archive-unit test cargo test --locked -p scriptbots-storage --lib test_storage_map_elites_archive_persistence_and_reload -- --nocapture
        run_step archive-integration test cargo test --locked -p scriptbots-storage --test persistence_integration map_elites -- --nocapture
        ;;
    recipes)
        run_step architecture-doc-examples test cargo test --locked -p scriptbots-app --doc -- --nocapture
        run_step architecture-recipes test cargo test --locked -p scriptbots-app --test architecture_contract -- --nocapture
        run_step recipe-link-artifacts check cargo build --locked -p scriptbots-app --lib --message-format=json
        run_step recipe-dependencies check cargo metadata --locked --format-version 1 --filter-platform "$actual_target"
        export SCRIPTBOTS_RECIPE_ARTIFACT_LOG="$proof_dir/recipe-link-artifacts.log"
        export SCRIPTBOTS_RECIPE_METADATA_LOG="$proof_dir/recipe-dependencies.log"
        run_step architecture-mutations test cargo test --locked -p scriptbots-app --test architecture_contract literal_recipe_compiler_and_runtime_mutations -- --ignored --exact --nocapture
        ;;
esac
run_step analytics-binary check cargo build --locked -p scriptbots-analytics --bin sb-analyze
cmp "$SCRIPTBOTS_VERIFY_PROFILE" "$proof_dir/profile.yaml"
[[ $(git rev-parse HEAD) == "$actual_commit" && -z $(git status --porcelain --untracked-files=all) ]]
jq -n --arg source "$actual_commit" --arg lane "$SCRIPTBOTS_VERIFY_LANE" --arg target "$actual_target" --argjson completed_steps "$steps" \
    '{schema:"scriptbots.verification.v1",status:"pass",source:$source,lane:$lane,target:$target,exit_code:0,completed_steps:$completed_steps,scope:"named correctness lane only; no GUI, PTY, browser or performance claim"}' > "$proof_dir/verdict.json"
jq -e --arg source "$SCRIPTBOTS_EXPECTED_COMMIT" --arg lane "$SCRIPTBOTS_VERIFY_LANE" \
    '.status == "pass" and .exit_code == 0 and .source == $source and .lane == $lane and .completed_steps > 0' "$proof_dir/verdict.json"
