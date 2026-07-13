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
    let db = dir.join("run.sqlite");
    // `--profile-steps` exits BEFORE the world is bootstrapped, so it never reaches
    // the manifest. A headless terminal run does: it bootstraps the world, renders a
    // couple of frames into a test backend, and exits.
    Command::new(binary())
        .env("SCRIPTBOTS_STORAGE_PATH", &db)
        .env("SCRIPTBOTS_RNG_SEED", "4242")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", "2")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS", "1")
        .args(["--mode", "terminal", "--bootstrap-ticks", "2"])
        .output()
        .expect("the app binary runs")
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
