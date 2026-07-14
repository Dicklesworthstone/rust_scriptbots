//! A real run must leave a provenance record behind.
//!
//! `RunManifestV1` was well-built and thoroughly tested and was NEVER WRITTEN by
//! the binary a user actually runs. Every claim about provenanced, reproducible
//! runs was therefore true of the library and false of the product: a user could
//! not tell which build, which seed, or which config produced a run directory.
//!
//! This test is the difference. It runs the real binary and looks on disk.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

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
        .env("SCRIPTBOTS_STORAGE_PATH", &db)
        .env("SCRIPTBOTS_RNG_SEED", "4242")
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

    // AND THE ENVIRONMENT CAPTURE CANNOT BE TRUSTED TO SHOW THIS — which is why the field above
    // had to exist.
    //
    // `BuildProvenanceV0` captures SCRIPTBOTS_MAX_THREADS by reading the environment at manifest
    // time. But startup calls `set_var("SCRIPTBOTS_MAX_THREADS", resolved)` to communicate the
    // decision to Rayon, so by the time provenance is captured WE HAVE OVERWRITTEN THE USER'S OWN
    // VARIABLE. The field claims to record the environment and actually records what we clobbered
    // it with: it reports 4 here, not the 16 the user exported.
    //
    // This is asserted rather than glossed over, because it is a real defect (filed separately)
    // and because it pins the reason `thread_policy` is the only honest record of what happened.
    // If someone later fixes the clobber, this assertion fails and points them straight at the
    // comment explaining why.
    assert_eq!(
        manifest["build"]["scriptbots_max_threads"], "4",
        "the environment capture no longer reports the CLOBBERED value. If the set_var clobber \
         has been fixed so that provenance captures the user's real environment (16), that is an \
         improvement — update this assertion to expect \"16\" and note it on the bead."
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
    for field in [
        "schema",
        "schema_version",
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
