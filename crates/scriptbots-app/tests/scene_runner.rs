//! Integration test for the scene scenario runner (bd-2z0.14.3.5.1).
//!
//! Loads every manifest in tests/scenes/, executes the NullDriver for all
//! and the real TerminalHeadlessDriver for the two fast scenes, and asserts
//! every expectation passes and the JSON log carries the required schema.

use std::path::PathBuf;

use scriptbots_app::scene::{
    NullDriver, SceneLog, SceneManifest, TerminalHeadlessDriver, run_scene,
};

fn scenes_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/scenes")
}

fn load_all() -> Vec<SceneManifest> {
    let mut manifests: Vec<SceneManifest> = std::fs::read_dir(scenes_dir())
        .expect("tests/scenes directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            (path.extension().and_then(|ext| ext.to_str()) == Some("toml")).then_some(path)
        })
        .map(|path| {
            SceneManifest::load(&path)
                .unwrap_or_else(|error| panic!("scene {} failed to load: {error}", path.display()))
        })
        .collect();
    manifests.sort_by(|a, b| a.name.cmp(&b.name));
    // Five bd-2z0.14.3.5.1 reference scenes plus the bd-2z0.14.3.4 offscreen
    // smoke scene; the count is explicit so a dropped scene fails loudly.
    assert_eq!(manifests.len(), 6, "all six reference scenes load");
    manifests
}

#[test]
fn every_reference_scene_validates_and_runs_null_driver() {
    for manifest in load_all() {
        manifest
            .validate()
            .unwrap_or_else(|error| panic!("scene {} invalid: {error}", manifest.name));
        let log = run_scene(&manifest, &mut NullDriver)
            .unwrap_or_else(|error| panic!("scene {} null run failed: {error}", manifest.name));
        assert_eq!(log.seed, manifest.seed);
        assert_eq!(log.ticks_executed, manifest.ticks);
        let json = serde_json::to_value(&log).expect("scene log serializes");
        for field in [
            "name",
            "frontend",
            "seed",
            "ticks_executed",
            "captures",
            "expectations",
            "timings_ms",
            "world_digest",
        ] {
            assert!(
                json.get(field).is_some(),
                "scene {} log missing field {field}",
                manifest.name
            );
        }
        // Field presence alone was ceremonial: `timings_ms` existed but shipped
        // permanently empty, so this loop passed while the log carried no timing
        // evidence at all (bd-2z0.14.3.5.1). Require the phase key set, and require
        // the log to pass its own structural validation.
        for phase in SceneLog::TIMING_PHASES {
            assert!(
                log.timings_ms.contains_key(phase),
                "scene {} log is missing the `{phase}` timing phase: {:?}",
                manifest.name,
                log.timings_ms
            );
        }
        log.validate().unwrap_or_else(|error| {
            panic!(
                "scene {} produced a structurally invalid log: {error}",
                manifest.name
            )
        });
    }
}

#[test]
fn terminal_driver_passes_fast_scenes() {
    for name in ["empty_world", "mixed_population_1k"] {
        let path = scenes_dir().join(format!("{name}.toml"));
        let manifest = SceneManifest::load(&path).expect("scene loads");
        let log = run_scene(&manifest, &mut TerminalHeadlessDriver::default())
            .unwrap_or_else(|error| panic!("scene {name} terminal run failed: {error}"));
        assert!(
            log.world_digest.is_some(),
            "scene {name} must bind a final world digest"
        );
        for result in &log.expectations {
            assert!(
                result.pass,
                "scene {name} expectation failed: {} ({})",
                result.desc, result.detail
            );
        }
    }
}

#[test]
fn scene_runs_are_deterministic_across_repeats() {
    for name in ["empty_world", "mixed_population_1k"] {
        let path = scenes_dir().join(format!("{name}.toml"));
        let manifest = SceneManifest::load(&path).expect("scene loads");
        let a = run_scene(&manifest, &mut TerminalHeadlessDriver::default())
            .expect("first terminal run");
        let b = run_scene(&manifest, &mut TerminalHeadlessDriver::default())
            .expect("second terminal run");
        assert_eq!(
            a.world_digest, b.world_digest,
            "scene {name} world digest must be deterministic"
        );
        assert_eq!(
            a.captures.len(),
            b.captures.len(),
            "capture cardinality stable"
        );
        for (left, right) in a.captures.iter().zip(b.captures.iter()) {
            assert_eq!(
                left.hash, right.hash,
                "capture {} hash must be deterministic",
                left.name
            );
        }
    }
}
