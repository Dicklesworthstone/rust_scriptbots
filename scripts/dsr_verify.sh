#!/usr/bin/env bash
# Correctness acceptance invoked by the pinned DSR profile, never nested RCH.
set -euo pipefail

refuse() {
    jq -n --arg reason "$1" '{schema:"scriptbots.verification.v1",status:"refused",reason:$reason}'
    exit 2
}

verify_evidence() {
    local directory=$1 expected=$2 lane=$3 record name kind log expected_hash passed recorded_passed target
    [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || refuse "invalid evidence source identity"
    local required=(formatting fsqlite-pin franken-licenses asupersync-universe wasm-graph)
    case "$lane" in
        workspace) required+=(workspace-check workspace-clippy workspace-tests core-economy-faults) ;;
        terminal) required+=(frankentui-metadata workspace-check workspace-clippy terminal-runtime-tests terminal-unit-tests terminal-end-to-end-tests terminal-smoke-tests) ;;
        connectivity) required+=(workspace-check workspace-clippy storage-checkpoint-tests checkpoint-unit-tests sensing-manifest-tests manifest-storage-tests bundle-tests control-unit-tests control-process-tests replay-process-tests) ;;
        tournament-smoke|tournament-full) required+=(workspace-check workspace-clippy assembly-tests tournament-tests tournament-cli) ;;
        tournament-smoke-and-analytics) required+=(workspace-check workspace-clippy assembly-tests tournament-tests tournament-cli analytics-pipeline) ;;
        analytics) required+=(workspace-check workspace-clippy analytics-pipeline) ;;
        graphs) required+=(graph-check graph-tests archive-unit archive-integration) ;;
        recipes) required+=(architecture-doc-examples architecture-recipes recipe-dependencies architecture-mutations) ;;
        graphs-and-recipes) required+=(graph-check graph-tests archive-unit archive-integration architecture-doc-examples architecture-recipes recipe-dependencies architecture-mutations) ;;
        archipelago) required+=(archipelago-check archipelago-tests capture-tests archipelago-multi-run-tests storage-tests) ;;
        server) required+=(server-check server-tests server-storage-tests) ;;
        *) refuse "unknown evidence lane" ;;
    esac
    required+=(analytics-binary)
    jq -e --arg source "$expected" --arg lane "$lane" --argjson count "${#required[@]}" \
        '.schema == "scriptbots.verification.v1" and .status == "pass" and .exit_code == 0 and .source == $source and .lane == $lane and .completed_steps == $count' \
        "$directory/verdict.json" >/dev/null || refuse "missing, stale or non-pass verdict"
    jq -s -e --arg source "$expected" --argjson count "${#required[@]}" \
        'length == $count and ([.[].name] | unique | length) == $count and all(.[]; .source == $source and .command_exit == 0 and .log_exit == 0)' \
        "$directory/commands.jsonl" >/dev/null || refuse "missing, duplicate or failed command evidence"
    for name in "${required[@]}"; do
        record=$(jq -c --arg name "$name" 'select(.name == $name)' "$directory/commands.jsonl")
        [[ -n "$record" ]] || refuse "missing required command: $name"
        log="$directory/$name.log"
        [[ -f "$log" && $(jq -r '.log' <<< "$record") == "$name.log" ]] || refuse "missing command log: $name"
        expected_hash=$(jq -r '.log_sha256' <<< "$record")
        [[ $(sha256sum "$log" | cut -d ' ' -f 1) == "$expected_hash" ]] || refuse "changed command log: $name"
        [[ -f "$directory/$name.command" && $(cat "$directory/$name.command") == "$(jq -r '.command' <<< "$record")" ]] || refuse "missing or changed command arguments: $name"
        kind=$(jq -r '.kind' <<< "$record")
        case "$name" in
            *-tests|core-economy-faults|archive-unit|archive-integration|architecture-doc-examples|architecture-recipes|architecture-mutations)
                [[ "$kind" == test ]] || refuse "test classified as compile-only: $name"
                passed=$(sed -nE 's/^test result: ok\. ([0-9]+) passed;.*/\1/p' "$log" | awk '{n += $1} END {print n+0}')
                recorded_passed=$(jq -er --arg name "$name" 'select(.name == $name) | .passed' "$directory/$name.tests.json") || refuse "missing executed-test evidence: $name"
                [[ "$passed" -gt 0 && "$recorded_passed" == "$passed" ]] || refuse "zero or inconsistent executed-test count: $name"
                ;;
            *) [[ "$kind" == check ]] || refuse "wrong command kind: $name" ;;
        esac
    done
    [[ -f "$directory/profile.yaml" && -f "$directory/profile.sha256" ]] || refuse "missing pinned profile evidence"
    [[ $(sha256sum "$directory/profile.yaml" | cut -d ' ' -f 1) == "$(cut -d ' ' -f 1 "$directory/profile.sha256")" ]] || refuse "changed profile evidence"
    [[ $(yq -r '.env.SCRIPTBOTS_EXPECTED_COMMIT' "$directory/profile.yaml") == "$expected" && $(yq -r '.env.SCRIPTBOTS_VERIFY_LANE' "$directory/profile.yaml") == "$lane" ]] || refuse "profile source or lane mismatch"
    target=$(yq -r '.env.SCRIPTBOTS_VERIFY_TARGET' "$directory/profile.yaml")
    jq -e --arg target "$target" '.target == $target' "$directory/verdict.json" >/dev/null || refuse "verdict target mismatch"
    jq -s -e --arg target "$target" 'all(.[]; .target == $target)' "$directory/commands.jsonl" >/dev/null || refuse "command target mismatch"
    for name in source.txt rustc.txt cargo.txt host.txt inputs.sha256; do
        [[ -s "$directory/$name" ]] || refuse "missing source/toolchain evidence: $name"
    done
    [[ $(head -n 1 "$directory/source.txt") == "commit $expected" ]] || refuse "source record mismatch"
    [[ $(sed -n 's/^host: //p' "$directory/rustc.txt") == "$target" ]] || refuse "compiler target mismatch"
    if [[ "$lane" == terminal ]]; then
        jq -R -s -e '
            split("\n") | map(fromjson? | select(type == "object" and has("packages") and has("resolve")))
            | length == 1 and (.[0] as $metadata
                | [$metadata.packages[] | select(.name == "ftui")] as $ftui
                | [$metadata.packages[] | select(.name == "scriptbots-app")] as $app
                | ($ftui | length) == 1 and ($app | length) == 1
                and $ftui[0].version == "0.5.0"
                and $ftui[0].source == "git+https://github.com/Dicklesworthstone/frankentui?rev=15cc6543f76b814394c590f9e7719dedd6684e4c#15cc6543f76b814394c590f9e7719dedd6684e4c"
                and any($metadata.resolve.nodes[]; .id == $app[0].id
                    and any(.deps[]; .name == "ftui" and .pkg == $ftui[0].id)))
        ' "$directory/frankentui-metadata.log" >/dev/null || refuse "missing exact locked ftui source and application consumer"
        jq -R -s -e --arg source "$expected" '
            split("\n") | map(fromjson? | select(.schema == "scriptbots.live-tui-fault-pty.v1"))
            | length == 3 and ([.[].fault] | sort) == ["panic", "renderer", "subscription"]
            and all(.[]; .source_commit == $source and .fault_observed == true
                and .raw_mode_observed == true and .terminal_attributes_restored == true
                and .alternate_screen_entered == true
                and .alternate_screen_restored == true
                and .cursor_hidden_observed == true and .cursor_visible_at_exit == true
                and .enabled_capture_modes_at_exit == [])
        ' "$directory/terminal-unit-tests.log" >/dev/null || refuse "missing source-bound actual PTY fault restoration"
        while IFS= read -r record; do
            local fault_directory fault_transcript_hash fault_attributes_hash
            fault_directory=$(jq -er '.artifact_directory' <<< "$record") || refuse "missing fault PTY artifact path"
            [[ "$fault_directory" = /* && -d "$fault_directory" ]] || refuse "missing fault PTY artifacts"
            fault_transcript_hash=$(jq -er '.transcript_sha256' <<< "$record") || refuse "missing fault PTY transcript identity"
            [[ $(sha256sum "$fault_directory/terminal.raw" | cut -d ' ' -f 1) == "$fault_transcript_hash" ]] || refuse "changed fault PTY transcript"
            fault_attributes_hash=$(jq -er '.attributes_sha256' <<< "$record") || refuse "missing fault PTY attributes identity"
            [[ $(sha256sum "$fault_directory/terminal-attributes.json" | cut -d ' ' -f 1) == "$fault_attributes_hash" ]] || refuse "changed fault PTY attributes"
            jq -e '.initial == .final' "$fault_directory/terminal-attributes.json" >/dev/null || refuse "fault PTY attributes differ"
        done < <(jq -R -c 'fromjson? | select(.schema == "scriptbots.live-tui-fault-pty.v1")' "$directory/terminal-unit-tests.log")
        jq -R -s -e --arg source "$expected" '
            split("\n") | map(fromjson? | select(.schema == "scriptbots.live-tui-pty.v1"))
            | length == 2 and ([.[].capability_profile] | sort) == ["no-color", "truecolor"]
            and all(.[]; .source_commit == $source and .exit_code == 0
                and .binary_source_observed == true
                and (if .capability_profile == "truecolor" then .observed_truecolor_sgr_count > 0
                     else .observed_color_sgr_count == 0 end)
                and (if .capability_profile == "truecolor" then
                    .data_provider_evidence as $e
                    | $e.checkpoint.new_catalog_entries == 1 and $e.checkpoint.byte_size > 0
                    and $e.checkpoint.visible_application_receipt == "[APPLIED] journal=unobserved CreateCheckpoint: applied id=checkpoint-request-1"
                    and ($e.checkpoint.checkpoint_id | type) == "string"
                    and $e.experiment.status == "completed" and $e.experiment.total_runs == 1
                    and $e.experiment.completed_runs == 1 and $e.experiment.failed_runs == 0
                    and $e.experiment.run.state == "completed" and $e.experiment.run.total_ticks == 3
                    and ($e.experiment.run.final_digest | test("^[0-9a-f]{64}$"))
                    and $e.refresh.pause_receipt.application_state == "applied"
                    and $e.refresh.step_receipt.application_state == "applied"
                    and $e.refresh.after_tick == ($e.refresh.before_tick + 1)
                    and $e.refresh.after_snapshot_revision > $e.refresh.before_snapshot_revision
                    and $e.refresh.paused_before == true and $e.refresh.paused_after == true
                    and $e.refresh.terminal_input_count_during_change == 0
                    and ($e.artifacts | length) > 0
                    and ([$e.artifacts[].file] | length) == ([$e.artifacts[].file] | unique | length)
                  else .data_provider_evidence == null end)
                and .raw_mode_observed == true and .terminal_attributes_restored == true
                and .alternate_screen_restored == true and .frankentui_shutdown_observed == true
                and .cursor_hidden_observed == true and .cursor_visible_at_exit == true
                and .enabled_capture_modes_at_exit == []
                and (.observed_science_routes | length) > 0
                and (.observed_science_routes | length) == (.observed_science_routes | unique | length)
                and (.observed_science_routes | sort) == ([.observed_screenshot_routes[].route] | sort))
        ' "$directory/terminal-end-to-end-tests.log" >/dev/null || refuse "missing source-bound live terminal and screenshot observations"
        while IFS= read -r record; do
            local artifact_directory transcript_hash runtime_hash screenshot_hash terminal_hash index screen_record
            artifact_directory=$(jq -er '.artifact_directory' <<< "$record") || refuse "missing PTY artifact path"
            [[ "$artifact_directory" = /* && -d "$artifact_directory" ]] || refuse "missing PTY artifacts"
            transcript_hash=$(jq -er '.transcript_sha256' <<< "$record") || refuse "missing PTY transcript identity"
            [[ $(sha256sum "$artifact_directory/terminal.raw" | cut -d ' ' -f 1) == "$transcript_hash" ]] || refuse "changed PTY transcript"
            runtime_hash=$(jq -er '.runtime_log_sha256' <<< "$record") || refuse "missing runtime log identity"
            [[ $(sha256sum "$artifact_directory/runtime.log" | cut -d ' ' -f 1) == "$runtime_hash" ]] || refuse "changed runtime lifecycle log"
            rg -q 'terminal lifecycle finalized' "$artifact_directory/runtime.log" || refuse "missing actual runtime shutdown callback"
            index=4
            while IFS= read -r screen_record; do
                screenshot_hash=$(jq -er '.sha256' <<< "$screen_record") || refuse "missing presented screenshot identity"
                [[ $(sha256sum "$artifact_directory/screen-$index.txt" | cut -d ' ' -f 1) == "$screenshot_hash" ]] || refuse "changed presented screenshot"
                terminal_hash=$(jq -er '.terminal_sha256' <<< "$screen_record") || refuse "missing observed terminal screen identity"
                [[ $(sha256sum "$artifact_directory/terminal-screen-$index.txt" | cut -d ' ' -f 1) == "$terminal_hash" ]] || refuse "changed observed terminal screen"
                index=$((index + 1))
            done < <(jq -c '.observed_screenshot_routes[]' <<< "$record")
            if [[ $(jq -r '.capability_profile' <<< "$record") == truecolor ]]; then
                local provider_file provider_hash provider_record
                while IFS= read -r provider_record; do
                    provider_file=$(jq -er '.file' <<< "$provider_record") || refuse "missing provider artifact filename"
                    [[ "$provider_file" =~ ^[A-Za-z0-9._-]+$ ]] || refuse "invalid provider artifact filename"
                    provider_hash=$(jq -er '.sha256' <<< "$provider_record") || refuse "missing provider artifact identity"
                    [[ $(sha256sum "$artifact_directory/$provider_file" | cut -d ' ' -f 1) == "$provider_hash" ]] || refuse "changed provider observation"
                done < <(jq -c '.data_provider_evidence.artifacts[]' <<< "$record")
                jq -e '.total == 0 and (.items | length) == 0' "$artifact_directory/checkpoint-catalog-before.json" >/dev/null || refuse "checkpoint catalog was not initially empty"
                jq -e --arg id "$(jq -r '.data_provider_evidence.checkpoint.checkpoint_id' <<< "$record")" \
                    '.total == 1 and (.items | length) == 1 and .items[0].checkpoint_id == $id' \
                    "$artifact_directory/checkpoint-catalog-after.json" >/dev/null || refuse "new checkpoint observation differs"
                for provider_file in checkpoint-positive-terminal.txt checkpoint-positive-screen.txt; do
                    rg -Fq -- "$(jq -r '.data_provider_evidence.checkpoint.visible_application_receipt' <<< "$record")" "$artifact_directory/$provider_file" || refuse "missing visible checkpoint application receipt"
                    rg -Fq -- "Checkpoint $(jq -r '.data_provider_evidence.checkpoint.checkpoint_id' <<< "$record") recorded at tick $(jq -r '.data_provider_evidence.checkpoint.tick' <<< "$record")" "$artifact_directory/$provider_file" || refuse "missing visible checkpoint result identity"
                done
                jq -e --arg id "$(jq -r '.data_provider_evidence.experiment.experiment_id' <<< "$record")" \
                    '.experiment_id == $id and .status == "completed" and .completed_runs == 1 and .failed_runs == 0 and (.runs | length) == 1' \
                    "$artifact_directory/experiment-completed.json" >/dev/null || refuse "completed experiment observation differs"
            fi
        done < <(jq -R -c 'fromjson? | select(.schema == "scriptbots.live-tui-pty.v1")' "$directory/terminal-end-to-end-tests.log")
    fi
    if [[ "$lane" == connectivity ]]; then
        jq -R -s -e --arg source "$expected" --rawfile compiler "$directory/rustc.txt" '
            split("\n") | map(fromjson? | select(.schema == "scriptbots.real-process-e2e.v3"
                or .schema == "scriptbots.e2e-experiment-checkpoint-artifact.v2"))
            | length == 2 and ([.[].schema] | unique | length) == 2
            and all(.[]; .source_commit == $source and .compiler_identity == ($compiler | rtrimstr("\n"))
                and .ordered_shutdown_exit_code == 0 and .tools_count > 0)
            and all(.[] | select(.schema == "scriptbots.real-process-e2e.v3");
                .application_state == "applied" and .proved_level == "applied"
                and .status_code == 200 and .pause_code == 200 and .screenshot_code == 409)
            and all(.[] | select(.schema == "scriptbots.e2e-experiment-checkpoint-artifact.v2");
                .status == "pass" and .experiments_count > 0 and .checkpoints_count > 0
                and .checkpoints_count == .artifacts_count and .checksums_verified == true
                and .watermarks.admitted > 0 and .watermarks.admitted == .watermarks.applied
                and .watermarks.applied == .watermarks.durable
                and .injected_faults_handled == (.fault_observations | length)
                and ([.fault_observations[][0]] | sort) == ["artifact_path_traversal", "empty_variants", "missing_artifact", "missing_experiment", "unknown_brain_family"])
        ' "$directory/control-process-tests.log" >/dev/null || refuse "missing source-bound process and durable bundle observations"
    fi
    if [[ "$lane" == tournament-smoke || "$lane" == tournament-full || "$lane" == tournament-smoke-and-analytics ]]; then
        local tournament_mode=smoke
        [[ "$lane" != tournament-full ]] || tournament_mode=full
        jq -e --arg source "$expected" --arg mode "$tournament_mode" \
            '.schema == "scriptbots.tournament-proof.v1" and .status == "pass" and .source == $source and .mode == $mode and .identical_invocations == 2' \
            "$directory/tournament/verdict.json" >/dev/null || refuse "missing or mismatched tournament observations"
        [[ -s "$directory/tournament/artifacts.sha256" ]] || refuse "missing tournament artifact hashes"
        sha256sum --check "$directory/tournament/artifacts.sha256" >/dev/null || refuse "changed tournament artifacts"
    fi
    if [[ "$lane" == analytics || "$lane" == tournament-smoke-and-analytics ]]; then
        bash scripts/e2e_analytics.sh --verify-evidence "$directory/analytics" "$expected" || refuse "missing, changed or mismatched analytics observations"
        jq -e --arg source "$expected" '
            .schema == "scriptbots.analytics-falsification.v1" and .status == "pass"
            and .artifact_source == $source and .verifier_source == $source
            and .positive_before_exit == 0 and .positive_after_exit == 0
            and (.required_report_names | length) == (.required_report_names | unique | length)
            and (.required_invariant_names | length) == (.required_invariant_names | unique | length)
            and (.rejected_cases | length) == ((.required_report_names | length) + (.required_invariant_names | length) + 5)
            and all(.rejected_cases[]; .exit_code != 0)
            and ([.rejected_cases[] | select(.case == "missing_report") | .omitted_identity] | sort) == (.required_report_names | sort)
            and ([.rejected_cases[] | select(.case == "missing_invariant") | .omitted_identity] | sort) == (.required_invariant_names | sort)
            and ([.rejected_cases[] | select(.omitted_identity == null) | .case] | sort) == ["aliased_database", "changed_report", "collapsed_root", "fixture_only", "missing_named_test"]
        ' "$directory/analytics/falsification/verdict.json" >/dev/null || refuse "missing analytics mutation refusals"
        sha256sum --check "$directory/analytics/falsification/artifacts.sha256" >/dev/null || refuse "changed analytics mutation evidence"
    fi
}

if [[ ${1:-} == --verify-evidence ]]; then
    [[ $# == 4 ]] || refuse "usage: --verify-evidence DIRECTORY EXPECTED_COMMIT LANE"
    verify_evidence "$2" "$3" "$4"
    exit 0
fi

if [[ ${1:-} == --run ]]; then
    [[ $# == 3 && $2 =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ && $3 =~ ^[a-zA-Z0-9][a-zA-Z0-9._+-]*$ ]] || refuse "usage: --run PROFILE UNIQUE_VERSION"
    [[ ${DSR_CONFIG_DIR:-} = /* && -f "$DSR_CONFIG_DIR/config.yaml" ]] || refuse "missing DSR configuration"
    profile="$DSR_CONFIG_DIR/repos.d/$2.yaml"
    [[ -f "$profile" ]] || refuse "missing DSR repository profile"
    command -v yq >/dev/null || refuse "missing yq prerequisite"
    [[ $(yq -r '.tool_name' "$profile") == "$2" ]] || refuse "DSR profile identity mismatch"
    [[ $(yq -r '.targets | length' "$profile") == 1 ]] || refuse "correctness profile must declare one explicit target"
    target=$(yq -r '.targets[0]' "$profile")
    host=$(SCRIPTBOTS_DSR_TARGET="$target" yq -r '.cross_compile[strenv(SCRIPTBOTS_DSR_TARGET)].host' "$profile")
    [[ -n "$host" && "$host" != null && -f "$DSR_CONFIG_DIR/hosts.yaml" ]] || refuse "missing DSR host configuration"
    SCRIPTBOTS_DSR_HOST="$host" yq -e '.hosts[strenv(SCRIPTBOTS_DSR_HOST)] != null' "$DSR_CONFIG_DIR/hosts.yaml" >/dev/null || refuse "undeclared DSR host: $host"
    dsr config validate || refuse "invalid DSR configuration"
    checkout=$(yq -r '.local_path' "$profile")
    expected=$(yq -r '.env.SCRIPTBOTS_EXPECTED_COMMIT' "$profile")
    lane=$(yq -r '.env.SCRIPTBOTS_VERIFY_LANE' "$profile")
    proof_root=$(yq -r '.env.SCRIPTBOTS_PROOF_ROOT' "$profile")
    if [[ $(yq -r '.env | has("DSR_BUILD_TIMEOUT")' "$profile") == true ]]; then
        build_timeout=$(yq -r '.env.DSR_BUILD_TIMEOUT' "$profile")
        [[ "$build_timeout" =~ ^[1-9][0-9]*$ ]] || refuse "profile DSR build timeout must be a nonzero integer in seconds"
        # DSR captures its orchestration deadline before exporting recipe env.
        export DSR_BUILD_TIMEOUT="$build_timeout"
    fi
    seed_cache=$(yq -r '.env.CARGO_HOME // ""' "$profile")
    if [[ -n "$seed_cache" ]]; then
        [[ "$seed_cache" = /* && -d "$seed_cache" && ! -L "$seed_cache" ]] || refuse "profile Cargo seed cache must be an existing absolute real directory"
        # DSR selects the seed from its launcher environment, then replaces the
        # build's CARGO_HOME with a private copy. Bind both to the declared input.
        export CARGO_HOME="$seed_cache"
    fi
    [[ "$checkout" = /* && -d "$checkout" && "$expected" =~ ^[0-9a-f]{40}$ ]] || refuse "missing pinned DSR source"
    [[ $(git -C "$checkout" branch --show-current) == main ]] || refuse "source must be on main"
    [[ $(git -C "$checkout" rev-parse HEAD) == "$expected" ]] || refuse "source commit mismatch"
    [[ -z $(git -C "$checkout" status --porcelain --untracked-files=all) ]] || refuse "dirty source checkout"
    [[ "$proof_root" = /* && -d "$proof_root" && ! -e "$proof_root/$3" ]] || refuse "missing proof root or reused proof version"
    profile_hash=$(sha256sum "$profile" | cut -d ' ' -f 1)
    DSR_KEEP_BUILD_STAGES=1 dsr build --tool "$2" --target "$target" --only-native --no-sync --version "$3"
    [[ $(sha256sum "$profile" | cut -d ' ' -f 1) == "$profile_hash" ]] || refuse "DSR profile changed during execution"
    verify_evidence "$proof_root/$3" "$expected" "$lane"
    exit 0
fi

proof_version=${1:-}
[[ "$proof_version" =~ ^[a-zA-Z0-9][a-zA-Z0-9._+-]*$ ]] || refuse "missing or unsafe proof version"
[[ ${SCRIPTBOTS_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || refuse "missing pinned source commit"
[[ ${SCRIPTBOTS_VERIFY_LANE:-} == workspace || ${SCRIPTBOTS_VERIFY_LANE:-} == terminal || ${SCRIPTBOTS_VERIFY_LANE:-} == connectivity || ${SCRIPTBOTS_VERIFY_LANE:-} == tournament-smoke || ${SCRIPTBOTS_VERIFY_LANE:-} == tournament-full || ${SCRIPTBOTS_VERIFY_LANE:-} == tournament-smoke-and-analytics || ${SCRIPTBOTS_VERIFY_LANE:-} == analytics || ${SCRIPTBOTS_VERIFY_LANE:-} == graphs || ${SCRIPTBOTS_VERIFY_LANE:-} == recipes || ${SCRIPTBOTS_VERIFY_LANE:-} == graphs-and-recipes || ${SCRIPTBOTS_VERIFY_LANE:-} == archipelago || ${SCRIPTBOTS_VERIFY_LANE:-} == server ]] || refuse "unknown correctness lane"
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

# The source checkout's Git protection does not cover DSR's sibling Cargo home.
# bd-build-farm-reliability-lb19.3: SBH deleted an active registry during DSR52.
# Protect only our external build inputs/outputs; leave fleet cleanup running.
if command -v sbh >/dev/null; then
    [[ ${CARGO_HOME:-} = /* && -d "$CARGO_HOME" ]] || refuse "missing DSR Cargo home for SBH protection"
    case "$CARGO_HOME/" in "$source_root/"*) refuse "Cargo home must be outside source" ;; esac
    mkdir -p "$CARGO_TARGET_DIR"
    for protected_path in "$CARGO_HOME" "$CARGO_TARGET_DIR" "$proof_root"; do
        sbh protect "$protected_path" || refuse "failed SBH protection: $protected_path"
        [[ -f "$protected_path/.sbh-protect" ]] || refuse "missing SBH marker: $protected_path"
    done
fi

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
# Embed the verified build inputs in the executable itself. Runtime manifests
# deliberately do not infer a shipped binary's origin from its caller's checkout.
export SCRIPTBOTS_SOURCE_REVISION="$actual_commit"
SCRIPTBOTS_SOURCE_BRANCH="$(git branch --show-current)"
export SCRIPTBOTS_SOURCE_TREE_CLEAN=true
SCRIPTBOTS_SOURCE_STATUS_DIGEST="$(git status --porcelain --untracked-files=all | sha256sum | cut -d ' ' -f 1)"
SCRIPTBOTS_SOURCE_DIFF_DIGEST="$(git diff --binary HEAD | sha256sum | cut -d ' ' -f 1)"
export SCRIPTBOTS_SOURCE_BRANCH SCRIPTBOTS_SOURCE_STATUS_DIGEST SCRIPTBOTS_SOURCE_DIFF_DIGEST
export SCRIPTBOTS_RUSTC_VV="$compiler_identity"
steps=0
touch "$proof_dir/commands.jsonl"
record_failed_verdict() {
    local rc=$?
    if (( rc != 0 )); then
        jq -n --arg source "$actual_commit" --arg lane "$SCRIPTBOTS_VERIFY_LANE" \
            --argjson exit_code "$rc" --argjson completed_steps "$steps" \
            '{schema:"scriptbots.verification.v1",status:"failed",source:$source,lane:$lane,exit_code:$exit_code,completed_steps:$completed_steps}' > "$proof_dir/verdict.json"
    fi
}
trap record_failed_verdict EXIT

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
    archipelago)
        run_step archipelago-check check cargo check --locked --workspace --all-targets
        run_step archipelago-tests test cargo test --locked -p scriptbots-app --test archipelago_report_cli recorded_archipelago_cli -- --nocapture
        run_step capture-tests test cargo test --locked -p scriptbots-app --lib archipelago_report::tests:: -- --nocapture
        run_step archipelago-multi-run-tests test cargo test --locked -p scriptbots-storage --test multi_run_schema -- --nocapture
        run_step storage-tests test cargo test --locked -p scriptbots-storage --no-fail-fast -- --nocapture
        ;;
    server)
        run_step server-check check cargo check --locked --workspace --all-targets
        run_step server-tests test cargo test --locked -p scriptbots-app --test real_process_control_e2e -- --nocapture
        # Both original storage scenarios and the exact seeded Assembly overflow regression.
        run_step server-storage-tests test cargo test --locked -p scriptbots-app --test real_process_control_e2e -- --ignored --nocapture
        ;;
    workspace)
        run_step workspace-check check cargo check --locked --workspace --all-targets
        run_step workspace-clippy check cargo clippy --locked --workspace --all-targets -- -D warnings
        run_step workspace-tests test cargo test --locked --workspace -- --nocapture
        run_step core-economy-faults test cargo test --locked -p scriptbots-core --features economy-faults -- --nocapture
        ;;
    terminal)
        run_step frankentui-metadata check cargo metadata --locked --format-version 1
        run_step workspace-check check cargo check --locked --workspace --all-targets
        run_step workspace-clippy check cargo clippy --locked --workspace --all-targets -- -D warnings
        run_step terminal-runtime-tests test cargo test --locked -p scriptbots-runtime --lib -- --nocapture
        run_step terminal-unit-tests test cargo test --locked -p scriptbots-app --lib terminal:: -- --nocapture
        run_step terminal-end-to-end-tests test cargo test --locked -p scriptbots-app --test terminal_end_to_end -- --nocapture
        run_step terminal-smoke-tests test cargo test --locked -p scriptbots-app --test terminal_smoke -- --nocapture
        ;;
    connectivity)
        # Fold into bd-2z0.19's composed acceptance when bounded retention is delivered.
        run_step workspace-check check cargo check --locked --workspace --all-targets
        run_step workspace-clippy check cargo clippy --locked --workspace --all-targets -- -D warnings
        run_step storage-checkpoint-tests test cargo test --locked -p scriptbots-storage --lib checkpoint_ -- --nocapture
        run_step checkpoint-unit-tests test cargo test --locked -p scriptbots-app --lib control::tests::checkpoint_ -- --nocapture
        run_step sensing-manifest-tests test cargo test --locked -p scriptbots-app --test run_manifest_emitted -- --nocapture
        run_step manifest-storage-tests test cargo test --locked -p scriptbots-storage --test multi_run_schema -- --nocapture
        run_step bundle-tests test cargo test --locked -p scriptbots-storage --lib bundle::tests:: -- --nocapture
        run_step control-unit-tests test cargo test --locked -p scriptbots-app --lib servers::tests:: -- --nocapture
        run_step control-process-tests test cargo test --locked -p scriptbots-app --test real_process_control_e2e -- --nocapture
        run_step replay-process-tests test cargo test --locked -p scriptbots-app --test replay_e2e -- --nocapture
        ;;
    tournament-smoke|tournament-full|tournament-smoke-and-analytics)
        run_step workspace-check check cargo check --locked --workspace --all-targets
        run_step workspace-clippy check cargo clippy --locked --workspace --all-targets -- -D warnings
        run_step assembly-tests test cargo test --locked -p scriptbots-brain --features experimental assembly::tests:: -- --nocapture
        run_step tournament-tests test cargo test --locked -p scriptbots-app --lib tournament:: -- --nocapture
        export SCRIPTBOTS_TOURNAMENT_PROOF_DIR="$proof_dir/tournament"
        tournament_mode=smoke
        [[ "$SCRIPTBOTS_VERIFY_LANE" != tournament-full ]] || tournament_mode=full
        run_step tournament-cli check bash scripts/e2e_tournament_leaderboard.sh "$tournament_mode"
        if [[ "$SCRIPTBOTS_VERIFY_LANE" == tournament-smoke-and-analytics ]]; then
            run_step analytics-pipeline check bash scripts/e2e_analytics.sh --inside-dsr "$proof_dir/analytics"
        fi
        ;;
    analytics)
        run_step workspace-check check cargo check --locked --workspace --all-targets
        run_step workspace-clippy check cargo clippy --locked --workspace --all-targets -- -D warnings
        run_step analytics-pipeline check bash scripts/e2e_analytics.sh --inside-dsr "$proof_dir/analytics"
        ;;
    graphs|graphs-and-recipes)
        run_step graph-check check cargo check --locked -p scriptbots-analytics --all-targets
        run_step graph-tests test cargo test --locked -p scriptbots-analytics --test graph_reports -- --nocapture
        run_step archive-unit test cargo test --locked -p scriptbots-storage --lib test_storage_map_elites_archive_persistence_and_reload -- --nocapture
        run_step archive-integration test cargo test --locked -p scriptbots-storage --test persistence_integration map_elites -- --nocapture
        ;;
esac
case "$SCRIPTBOTS_VERIFY_LANE" in
    recipes|graphs-and-recipes)
        run_step architecture-doc-examples test cargo test --locked -p scriptbots-app --doc -- --nocapture
        run_step architecture-recipes test cargo test --locked -p scriptbots-app --test architecture_contract --message-format=json -- --nocapture
        run_step recipe-dependencies check cargo metadata --locked --format-version 1 --filter-platform "$actual_target"
        export SCRIPTBOTS_RECIPE_ARTIFACT_LOG="$proof_dir/architecture-recipes.log"
        export SCRIPTBOTS_RECIPE_METADATA_LOG="$proof_dir/recipe-dependencies.log"
        run_step architecture-mutations test cargo test --locked -p scriptbots-app --test architecture_contract literal_recipe_compiler_and_runtime_mutations -- --ignored --exact --nocapture
        ;;
esac
run_step analytics-binary check cargo build --locked -p scriptbots-analytics --bin sb-analyze
cmp "$SCRIPTBOTS_VERIFY_PROFILE" "$proof_dir/profile.yaml"
[[ $(git rev-parse HEAD) == "$actual_commit" && -z $(git status --porcelain --untracked-files=all) ]]
jq -n --arg source "$actual_commit" --arg lane "$SCRIPTBOTS_VERIFY_LANE" --arg target "$actual_target" --argjson completed_steps "$steps" \
    '{schema:"scriptbots.verification.v1",status:"pass",source:$source,lane:$lane,target:$target,exit_code:0,completed_steps:$completed_steps,scope:"named correctness lane only; terminal lane includes local PTY; no hardware GUI, browser, full research journey or performance claim"}' > "$proof_dir/verdict.json"
jq -e --arg source "$SCRIPTBOTS_EXPECTED_COMMIT" --arg lane "$SCRIPTBOTS_VERIFY_LANE" \
    '.status == "pass" and .exit_code == 0 and .source == $source and .lane == $lane and .completed_steps > 0' "$proof_dir/verdict.json"
verify_evidence "$proof_dir" "$SCRIPTBOTS_EXPECTED_COMMIT" "$SCRIPTBOTS_VERIFY_LANE"
