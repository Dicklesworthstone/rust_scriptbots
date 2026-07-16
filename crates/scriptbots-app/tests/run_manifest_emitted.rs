//! A real run must leave a provenance record behind.
//!
//! The former V1 run manifest was well-built and thoroughly tested and was NEVER WRITTEN by
//! the binary a user actually runs. Every claim about provenanced, reproducible
//! runs was therefore true of the library and false of the product: a user could
//! not tell which build, which seed, or which config produced a run directory.
//!
//! The current contract is stronger: a base `RunManifestV3` is registered in the
//! run database before tick zero, and the adjacent V3.1 sidecar supplements that
//! durable record with post-bootstrap evidence. These tests drive the real binary
//! and inspect both records on disk.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use scriptbots_app::{
    CHARACTERIZATION_TRACE_V2_SCHEMA, CharacterizationTraceV2, RUN_MANIFEST_V3_BOOTSTRAP_SCHEMA,
    RUN_MANIFEST_V3_SCHEMA, RunManifestV3,
};
use scriptbots_runtime::RunId;
use scriptbots_storage::StorageReader;

fn run_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "scriptbots_manifest_{label}_{}_{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("run dir");
    dir
}

fn binary() -> PathBuf {
    // The integration test binary sits next to the app binary.
    let mut path = std::env::current_exe().expect("test exe");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.join("scriptbots-app")
}

fn launch(dir: &Path) -> std::process::Output {
    launch_with(dir, &[], &[])
}

/// Launch the real binary with extra env vars and CLI args.
fn launch_with(dir: &Path, envs: &[(&str, &str)], args: &[&str]) -> std::process::Output {
    let db = dir.join("run.sqlite");
    // `--profile-steps` exits BEFORE the world is bootstrapped, so it never reaches
    // the manifest. A headless terminal run does: it bootstraps the world, renders a
    // couple of frames into a test backend, and exits.
    let mut command = Command::new(binary());
    command
        // The test process can inherit this from its verifier (DSR deliberately
        // pins it). Each case below must control the resolver's environment
        // input explicitly or it is testing the verifier rather than the app.
        .env_remove("SCRIPTBOTS_MAX_THREADS")
        .env("SCRIPTBOTS_STORAGE_PATH", &db)
        .env("SCRIPTBOTS_RNG_SEED", "4242")
        // Preserve real control-server startup while allowing parallel test cases
        // to reserve distinct REST and MCP sockets atomically through the operating system.
        .env("SCRIPTBOTS_CONTROL_REST_ENABLED", "1")
        .env("SCRIPTBOTS_CONTROL_REST_ADDR", "127.0.0.1:0")
        .env("SCRIPTBOTS_CONTROL_MCP", "http")
        .env("SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR", "127.0.0.1:0")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", "2")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS", "1")
        .args(["--mode", "terminal", "--bootstrap-ticks", "2"]);
    for (key, value) in envs {
        command.env(key, value);
    }
    command.args(args);
    command.output().expect("the app binary runs")
}

fn manifest_of(output: &std::process::Output, dir: &Path) -> serde_json::Value {
    assert!(
        output.status.success(),
        "the run did not complete, so this test proves nothing.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(dir.join("run.manifest.json")).expect("manifest exists");
    serde_json::from_slice(&bytes).expect("valid JSON")
}

fn characterization_trace_of(
    output: &std::process::Output,
    path: &Path,
) -> CharacterizationTraceV2 {
    assert!(
        output.status.success(),
        "the characterization run did not complete, so this test proves nothing.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(path).expect("characterization trace exists");
    serde_json::from_slice(&bytes).expect("valid characterization trace JSON")
}

fn manifest_digest_for_test(manifest: &RunManifestV3) -> String {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let bytes = manifest
        .canonical_json_bytes()
        .expect("manifest has canonical JSON");
    let mut hash = OFFSET_BASIS;
    for byte in b"run-manifest-v3"
        .iter()
        .copied()
        .chain(std::iter::once(0))
        .chain(bytes.iter().copied())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("fnv1a64:{hash:016x}")
}

#[test]
fn the_manifest_records_the_thread_policy_the_run_actually_resolved() {
    // A manifest that records the ENVIRONMENT rather than the DECISION describes a run that did
    // not happen.
    //
    // `BuildProvenanceV0` already captures SCRIPTBOTS_MAX_THREADS. But precedence exists precisely
    // because a more specific layer can override the environment: export 16 and pass `--threads 4`
    // and the run uses FOUR. A reader who trusted the environment capture would conclude the run
    // used sixteen — and would be comparing against a run that never existed.
    //
    // This is the exact conflict the bead's precedence rules were written for, driven through the
    // real binary rather than a unit test of the resolver.
    let dir = run_dir("policy_cli_beats_env");
    let output = launch_with(
        &dir,
        &[("SCRIPTBOTS_MAX_THREADS", "16")],
        &["--threads", "4"],
    );
    let manifest = manifest_of(&output, &dir);

    let policy = &manifest["thread_policy"];
    assert!(
        !policy.is_null(),
        "the run recorded NO thread policy. Nobody can say how many threads produced this run, \
         nor which layer decided — and the environment capture beside it would actively mislead \
         them."
    );
    assert_eq!(
        policy["threads"], 4,
        "the manifest reports a thread count other than the one the CLI asked for. The run used \
         4; anything else means the manifest is describing a different run."
    );
    assert_eq!(
        policy["source"], "cli-flag",
        "the manifest does not name the CLI as the deciding layer. Two runs that both used 4 \
         threads — one because the operator asked and one because a probe guessed — have \
         different provenance, and `source` is how a reader tells them apart."
    );

    // THE OVERRIDE MUST BE ON THE RECORD. The user exported 16; the run used 4. A manifest that
    // showed only the winner would leave them unable to see that their environment variable was
    // considered and declined — which is exactly the confusion the original bug caused.
    assert_eq!(
        policy["overridden"], "environment",
        "the manifest does not record that the exported SCRIPTBOTS_MAX_THREADS was DECLINED. A \
         user who exported 16 and got 4 must be able to see from the run's own record that their \
         variable was seen and outranked, rather than ignored or lost."
    );

    // AND THE ENVIRONMENT CAPTURE NOW AGREES WITH REALITY. The user exported 16; the run
    // used 4; the capture must say 16. Startup still communicates the resolved count to
    // Rayon through `set_var`, but provenance reads the launch-pinned
    // `LaunchEnvironmentV0` snapshot taken before any mutation (bd-3p7i), so our own
    // write can no longer masquerade as the user's environment. `thread_policy` above
    // records the DECISION; this field records the USER'S STATEMENT; they differ exactly
    // when precedence did its job.
    assert_eq!(
        manifest["build"]["scriptbots_max_threads"], "16",
        "the environment capture must report what the USER exported at launch (16), not \
         the resolved value startup wrote back into the environment (4). If this reports \
         4 again, the launch-environment snapshot is being taken after startup mutation — \
         the exact bd-3p7i clobber regressing."
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_manifest_names_the_layer_that_was_overridden() {
    // `--low-power` alongside an explicit `--threads` is the case a user is most likely to get
    // wrong, and the one the original bug punished: their explicit 8 was silently replaced by 2.
    //
    // The precedence rules now let the explicit value win — but the DECLINED layer must be
    // visible, or the user learns that low-power did nothing from their power bill rather than
    // from the record of their own run.
    let dir = run_dir("policy_overridden");
    let output = launch_with(&dir, &[], &["--threads", "8", "--low-power"]);
    let manifest = manifest_of(&output, &dir);

    let policy = &manifest["thread_policy"];
    assert_eq!(
        policy["threads"], 8,
        "the explicit --threads value must win"
    );
    assert_eq!(policy["source"], "cli-flag");
    assert_eq!(
        policy["overridden"], "low-power-default",
        "the manifest does not record that --low-power's suggestion was DECLINED. That is the \
         normal, correct outcome of the precedence rules — but a user who passed --low-power and \
         got 8 threads deserves to find out from the manifest, not from the electricity meter."
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_real_run_writes_a_manifest_next_to_its_database() {
    let dir = run_dir("emit");
    let output = launch(&dir);

    // If the binary refuses these flags the test would silently prove nothing, so
    // fail loudly rather than "pass" on a run that never happened.
    assert!(
        output.status.success(),
        "the run did not complete, so this test proves nothing about provenance.\n\
         stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let manifest_path = dir.join("run.manifest.json");
    assert!(
        manifest_path.is_file(),
        "the run left NO manifest at {}. Its provenance — which build, which seed, \
         which config — is unrecoverable, and every claim this project makes about \
         reproducible runs is a claim about the library rather than the product.",
        manifest_path.display()
    );

    let encoded = std::fs::read(&manifest_path).expect("manifest is readable");
    let manifest: serde_json::Value =
        serde_json::from_slice(&encoded).expect("the manifest must be valid JSON");

    // It must describe THIS run, not a plausible-looking default. A manifest that
    // parses but describes the wrong run is worse than none: it looks like evidence.
    assert_eq!(
        manifest["root_seed"], 4242,
        "the manifest records a different seed than the run actually used"
    );
    assert_eq!(
        manifest["schema"], RUN_MANIFEST_V3_BOOTSTRAP_SCHEMA,
        "bootstrap evidence must move the V3 manifest onto its compatible V3.1 schema"
    );
    assert_eq!(manifest["schema_version"], 3);
    assert_eq!(manifest["random_streams"]["root_seed"], 4242);
    let bootstrap = &manifest["bootstrap_evidence"];
    assert_eq!(bootstrap["requested"], 2);
    assert_eq!(bootstrap["completed"], 2);
    assert_eq!(bootstrap["start"]["tick"], 0);
    assert_eq!(bootstrap["end"]["tick"], 2);
    assert_ne!(
        bootstrap["start"]["overall"], bootstrap["end"]["overall"],
        "two completed bootstrap transitions must move the full-state digest"
    );
    for field in [
        "schema",
        "schema_version",
        "identity",
        "config_digest",
        "normalized_config",
        "build",
        "reproducible",
    ] {
        assert!(
            !manifest[field].is_null(),
            "the manifest is missing `{field}`, so it cannot answer what produced this run"
        );
    }

    let sidecar: RunManifestV3 = serde_json::from_value(manifest.clone())
        .expect("the supplemental sidecar must satisfy the typed V3 manifest contract");
    let run_id_text = manifest["identity"]["run_id"]
        .as_str()
        .expect("the run identity must encode its durable RunId as text");
    let parsed_run_id: RunId = run_id_text
        .parse()
        .expect("the run identity must use the canonical RunId wire format");
    assert_eq!(parsed_run_id, sidecar.identity.run_id);
    assert_ne!(
        parsed_run_id.get(),
        0,
        "a real run must not use the zero RunId sentinel"
    );
    assert!(
        sidecar.identity.started_at_unix_ms > 0,
        "the run identity must record a real launch boundary"
    );
    assert_ne!(
        sidecar.identity.requested_tick_budget.is_some(),
        sidecar.identity.live_run_policy.is_some(),
        "the run identity must carry exactly one finite or live execution boundary"
    );
    assert_eq!(sidecar.identity.requested_tick_budget, None);
    assert_eq!(
        sidecar.identity.live_run_policy.as_deref(),
        Some("operator-controlled-until-stop-v1"),
        "the terminal application is a live run whose stop policy must be explicit"
    );

    let build = manifest["build"]
        .as_object()
        .expect("build provenance must be a structured record");
    for field in [
        "source_revision",
        "source_tree_clean",
        "source_status_digest",
        "source_diff_digest",
        "declared_toolchain",
        "compiler_toolchain",
        "rustc_vv",
        "toolchain_file_digest",
        "lockfile_digest",
        "provenance_complete",
    ] {
        assert!(
            build.contains_key(field),
            "build provenance must represent `{field}` explicitly, using null when it is unknown"
        );
    }
    assert!(
        build["declared_toolchain"]
            .as_str()
            .is_some_and(|value| !value.trim().is_empty()),
        "the tracked Rust toolchain declaration must not be blank"
    );
    assert!(
        build["toolchain_file_digest"]
            .as_str()
            .is_some_and(|value| !value.is_empty()),
        "the tracked toolchain file must be content-addressed"
    );
    assert!(
        build["lockfile_digest"]
            .as_str()
            .is_some_and(|value| !value.is_empty()),
        "the Cargo lockfile must be content-addressed"
    );
    let provenance_is_complete = build["source_revision"]
        .as_str()
        .is_some_and(|value| !value.trim().is_empty())
        && build["source_tree_clean"] == true
        && build["source_status_digest"]
            .as_str()
            .is_some_and(|value| !value.trim().is_empty())
        && build["source_diff_digest"]
            .as_str()
            .is_some_and(|value| !value.trim().is_empty())
        && build["rustc_vv"]
            .as_str()
            .is_some_and(|value| !value.trim().is_empty());
    assert_eq!(
        build["provenance_complete"], provenance_is_complete,
        "the completeness flag must report the captured source/toolchain evidence honestly"
    );
    assert_eq!(
        sidecar.reproducible, provenance_is_complete,
        "the run must not claim reproducibility when source or compiler identity is unknown"
    );

    // The sidecar is supplemental. The authoritative run row must already contain the same
    // identity and launch provenance, but remain at base V3 because it was registered before any
    // bootstrap transition executed.
    let database_path = dir.join("run.sqlite");
    let database_path = database_path
        .to_str()
        .expect("the temporary database path must be valid Unicode");
    let reader = StorageReader::open(database_path)
        .expect("the completed run database must be queryable read-only");
    assert_eq!(reader.run_id(), parsed_run_id);
    let durable = reader
        .run_manifest()
        .expect("the run database must contain its validated durable manifest");
    assert_eq!(durable.run_id, parsed_run_id);
    assert_eq!(durable.manifest_schema_version, 3);
    assert_eq!(durable.root_seed, sidecar.root_seed);
    assert_eq!(durable.rng_algorithm, sidecar.random_streams.algorithm);
    assert_eq!(durable.rng_version, sidecar.random_streams.version);
    assert_eq!(durable.config_digest, sidecar.config_digest);
    assert_eq!(
        durable.requested_tick_budget,
        sidecar.identity.requested_tick_budget
    );
    assert_eq!(durable.live_run_policy, sidecar.identity.live_run_policy);
    assert_eq!(durable.source_revision, sidecar.build.source_revision);
    assert_eq!(
        durable.source_tree_dirty,
        sidecar.build.source_tree_clean.map(|clean| !clean)
    );
    assert_eq!(
        durable.rust_toolchain,
        sidecar
            .build
            .compiler_toolchain
            .clone()
            .unwrap_or_else(|| sidecar.build.declared_toolchain.clone())
    );
    assert_eq!(durable.cargo_lock_digest, sidecar.build.lockfile_digest);

    let pre_tick: RunManifestV3 = serde_json::from_str(&durable.manifest_json)
        .expect("the durable launch manifest must retain its typed V3 representation");
    assert_eq!(pre_tick.schema, RUN_MANIFEST_V3_SCHEMA);
    assert_eq!(pre_tick.schema_version, 3);
    assert_eq!(pre_tick.identity, sidecar.identity);
    assert_eq!(pre_tick.root_seed, sidecar.root_seed);
    assert_eq!(pre_tick.random_streams, sidecar.random_streams);
    assert_eq!(pre_tick.normalized_config, sidecar.normalized_config);
    assert_eq!(pre_tick.build, sidecar.build);
    assert!(
        pre_tick.bootstrap_evidence.is_none(),
        "the database manifest must prove registration before bootstrap, not rewrite history afterward"
    );
    drop(reader);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_manifest_records_which_config_layers_built_the_run_and_who_displaced_whom() {
    // defaults -> scenario file -> environment -> CLI, driven through the real binary.
    // The file says 2000, the environment says 1000, the CLI says 500. The run must use
    // 500, and the manifest must carry BOTH displacements plus a kind-tagged digest for
    // every layer that spoke — that is what lets two runs that disagree be told apart by
    // their layer provenance rather than only by their final config digest.
    let dir = run_dir("config_layer_provenance");
    let scenario_path = dir.join("scenario.toml");
    std::fs::write(&scenario_path, b"world_width = 2000\n").expect("write scenario layer");
    let scenario_arg = scenario_path
        .to_str()
        .expect("temp path is valid Unicode")
        .to_owned();

    let output = launch_with(
        &dir,
        &[("SCRIPTBOTS_CONFIG_OVERRIDES", "world_width = 1000")],
        &["--config", &scenario_arg, "--set", "world_width=500"],
    );
    let manifest = manifest_of(&output, &dir);

    assert_eq!(
        manifest["normalized_config"]["world_width"], 500,
        "the CLI layer names the value for this exact invocation and must win"
    );

    let overrides = manifest["config_overrides"]
        .as_array()
        .expect("cross-layer displacements must be in the run record");
    let displaced: Vec<(&str, &str, &str)> = overrides
        .iter()
        .filter(|entry| entry["path"] == "world_width")
        .map(|entry| {
            (
                entry["losing_kind"].as_str().expect("losing kind"),
                entry["winning_kind"].as_str().expect("winning kind"),
                entry["winning_layer"].as_str().expect("winning layer"),
            )
        })
        .collect();
    assert_eq!(
        displaced
            .iter()
            .map(|(loser, winner, _)| (*loser, *winner))
            .collect::<Vec<_>>(),
        vec![("file", "environment"), ("environment", "cli")],
        "both displacements must be recorded in application order: {overrides:?}"
    );
    assert!(
        displaced[1].2.contains("--set world_width=500"),
        "the winning CLI layer must be attributable to the exact flag text, got {:?}",
        displaced[1].2
    );

    let digests: Vec<String> = manifest["scenario"]["ordered_config_layer_digests"]
        .as_array()
        .expect("ordered layer digests")
        .iter()
        .map(|value| value.as_str().expect("digest entry").to_owned())
        .collect();
    let kinds: Vec<&str> = digests
        .iter()
        .map(|entry| entry.split(':').next().unwrap_or(""))
        .collect();
    // The harness exports SCRIPTBOTS_RNG_SEED for every run, so the environment speaks
    // twice: once through SCRIPTBOTS_CONFIG_OVERRIDES and once through the typed
    // variables. Every statement appears, kind-tagged, in application order — starting
    // with the defaults every run is built from.
    assert_eq!(
        kinds,
        vec!["defaults", "file", "environment", "environment", "cli"],
        "every layer that spoke must appear as a kind-tagged digest, in order: {digests:?}"
    );

    // The displacement record must reach BOTH provenance artifacts: the durable
    // manifest registered in the run database before tick zero, and the supplemental
    // sidecar. A record that exists only beside the database can be lost with the
    // directory; one that exists only inside it cannot be read without tooling.
    let sidecar: RunManifestV3 = serde_json::from_value(manifest.clone())
        .expect("the sidecar must satisfy the typed V3 manifest contract");
    assert!(
        !sidecar.config_overrides.is_empty(),
        "the typed sidecar must carry the displacement record"
    );
    let database_path = dir.join("run.sqlite");
    let database_path = database_path
        .to_str()
        .expect("the temporary database path must be valid Unicode");
    let reader = StorageReader::open(database_path)
        .expect("the completed run database must be queryable read-only");
    let durable = reader
        .run_manifest()
        .expect("the run database must contain its validated durable manifest");
    let pre_tick: RunManifestV3 = serde_json::from_str(&durable.manifest_json)
        .expect("the durable launch manifest must retain its typed V3 representation");
    assert_eq!(
        pre_tick.config_overrides, sidecar.config_overrides,
        "the durable manifest and the sidecar must agree on the displacement record"
    );
    assert_eq!(
        pre_tick.scenario.ordered_config_layer_digests,
        sidecar.scenario.ordered_config_layer_digests,
        "the durable manifest and the sidecar must agree on the ordered layer digests"
    );
    drop(reader);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn characterization_records_the_same_config_layer_provenance_in_its_bound_manifest() {
    let dir = run_dir("characterization_config_layer_provenance");
    let scenario_path = dir.join("scenario.toml");
    let trace_path = dir.join("characterization.json");
    std::fs::write(&scenario_path, b"world_width = 2000\n").expect("write scenario layer");
    let scenario_arg = scenario_path
        .to_str()
        .expect("temp path is valid Unicode")
        .to_owned();
    let trace_arg = trace_path
        .to_str()
        .expect("trace path is valid Unicode")
        .to_owned();

    let output = launch_with(
        &dir,
        &[("SCRIPTBOTS_CONFIG_OVERRIDES", "world_width = 1000")],
        &[
            "--config",
            &scenario_arg,
            "--set",
            "world_width=500",
            "--characterize-v0",
            "0",
            "--characterization-out",
            &trace_arg,
        ],
    );
    let trace = characterization_trace_of(&output, &trace_path);

    assert_eq!(trace.schema, CHARACTERIZATION_TRACE_V2_SCHEMA);
    assert_eq!(
        trace.manifest.normalized_config["world_width"], 500,
        "the characterization world must use the CLI layer's winning value"
    );
    let overrides = &trace.manifest.config_overrides;
    let displaced: Vec<(&str, &str)> = overrides
        .iter()
        .filter(|entry| entry.path == "world_width")
        .map(|entry| {
            (
                entry.losing_kind.wire_tag(),
                entry.winning_kind.wire_tag(),
            )
        })
        .collect();
    assert_eq!(
        displaced,
        vec![("file", "environment"), ("environment", "cli")],
        "the characterization manifest must preserve both displacements: {overrides:?}"
    );

    let digest_kinds: Vec<&str> = trace
        .manifest
        .scenario
        .ordered_config_layer_digests
        .iter()
        .map(|entry| entry.split(':').next().unwrap_or(""))
        .collect();
    assert_eq!(
        digest_kinds,
        vec!["defaults", "file", "environment", "environment", "cli"],
        "the characterization manifest must retain every ordered config layer"
    );

    assert_eq!(
        trace.manifest_digest,
        manifest_digest_for_test(&trace.manifest),
        "the trace digest must bind the manifest after override provenance is attached"
    );
    let mut without_overrides = trace.manifest.clone();
    without_overrides.config_overrides.clear();
    assert_ne!(
        trace.manifest_digest,
        manifest_digest_for_test(&without_overrides),
        "removing override provenance must change the bound manifest digest"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn two_identical_runs_produce_the_same_provenance() {
    // This is what makes "provenanced" mean something. If two runs of the same
    // build, seed and config disagreed about their own configuration, the manifest
    // would be describing something other than the run.
    let first = run_dir("same_a");
    let second = run_dir("same_b");

    let out_a = launch(&first);
    let out_b = launch(&second);
    assert!(
        out_a.status.success() && out_b.status.success(),
        "both runs must complete"
    );

    let read = |dir: &Path| -> serde_json::Value {
        let bytes = std::fs::read(dir.join("run.manifest.json")).expect("manifest exists");
        serde_json::from_slice(&bytes).expect("valid JSON")
    };
    let a = read(&first);
    let b = read(&second);

    // The CONFIG DIGEST and the seed are the parts that must match. Timestamps,
    // host, and durations are legitimately non-reproducible and are not compared —
    // but they must not be allowed to hide a real difference, which is why the
    // digest is compared rather than the whole document.
    assert_eq!(
        a["config_digest"], b["config_digest"],
        "two runs of the same config disagree about their own config digest"
    );
    assert_eq!(a["root_seed"], b["root_seed"]);
    assert_eq!(a["normalized_config"], b["normalized_config"]);

    let _ = std::fs::remove_dir_all(&first);
    let _ = std::fs::remove_dir_all(&second);
}
