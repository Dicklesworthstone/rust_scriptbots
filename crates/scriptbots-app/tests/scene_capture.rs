//! End-to-end tests for the offscreen live-render capture harness
//! (bd-2z0.14.3.4): real Bevy GPU pipeline rendered headless, provenance
//! manifests, golden compare/regen workflow, and the corrupted-pipeline
//! alarm proof (a dishonest frame MUST fail the harness).
//!
//! GPU tests skip loudly (never silently pass) when no adapter exists; on
//! the CI software lane (llvmpipe/lavapipe via WGPU_BACKEND) they run for
//! real.
#![cfg(feature = "bevy_render")]

use scriptbots_app::scene::{
    BevyOffscreenDriver, CameraKey, CapturePoint, Expectation, FrontendKind, GoldenOutcome,
    SceneDriver, SceneError, SceneManifest, process_golden,
};
use scriptbots_bevy::capture::{
    CapturedFrame, CompareThresholds, compare_frames, encode_png, rgba8_is_visually_blank,
};
use serial_test::serial;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

/// One GPU context at a time: parallel capture contexts would race the
/// adapter and flake the determinism assertions.
static GPU_GUARD: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

struct ScopedEnvOverride {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl ScopedEnvOverride {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: every GPU/environment-sensitive test in this module is
        // serialized by GPU_GUARD, and Drop restores the exact prior value.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for ScopedEnvOverride {
    fn drop(&mut self) {
        // SAFETY: GPU_GUARD serializes this process-wide mutation. Restoring
        // in Drop also covers panics inside the capture path.
        unsafe {
            if let Some(previous) = &self.previous {
                std::env::set_var(self.key, previous);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }
}

fn tiny_manifest(name: &str) -> SceneManifest {
    SceneManifest {
        name: name.to_string(),
        seed: 3,
        config_overrides: Some(
            toml::from_str::<toml::Value>(
                r#"
world_width = 600
world_height = 600
"#,
            )
            .expect("override toml"),
        ),
        ticks: 4,
        quality: Some("medium".to_string()),
        frontend: FrontendKind::BevyOffscreen,
        camera: vec![],
        captures: vec![CapturePoint {
            tick: 2,
            name: "mid".to_string(),
            camera_key: None,
        }],
        expect: vec![Expectation::AgentCount {
            min: 1,
            max: 10_000,
        }],
    }
}

const ADAPTER_UNAVAILABLE_PROBLEM: &str = "offscreen capture: no GPU adapter available for offscreen capture \
     (software lane requires llvmpipe/lavapipe via WGPU_BACKEND, or a real GPU)";

#[test]
#[serial]
fn same_tick_camera_bookmarks_render_distinct_views_of_identical_science() {
    let _guard = GPU_GUARD.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().expect("capture directory");
    let retained = std::env::var_os("SCRIPTBOTS_LOOK_CAPTURE_DIR")
        .map(std::path::PathBuf::from)
        .map(|root| root.join("bookmarks"));
    if let Some(path) = &retained {
        std::fs::create_dir(path).expect("fresh retained bookmark directory");
    }
    let artifacts = retained.as_deref().unwrap_or(dir.path());
    let mut manifest = tiny_manifest("camera-bookmarks");
    manifest.camera = [450.0, 900.0]
        .into_iter()
        .map(|distance| CameraKey {
            tick: 0,
            pos: [0.0, distance, distance],
            yaw: std::f32::consts::PI,
            pitch: -std::f32::consts::FRAC_PI_4,
            fov: 55.0,
            follow_uid: None,
        })
        .collect();
    manifest.captures = ["near", "overview"]
        .into_iter()
        .enumerate()
        .map(|(index, name)| CapturePoint {
            tick: 0,
            name: name.to_string(),
            camera_key: Some(index),
        })
        .collect();
    let mut driver = BevyOffscreenDriver {
        seed_agents: 8,
        viewport: (256, 256),
        artifacts_dir: Some(artifacts.to_path_buf()),
    };
    let facts = driver.run(&manifest).expect("real GPU bookmark captures");
    assert_eq!(facts.captures.len(), manifest.captures.len());
    assert_ne!(
        facts.captures[0].2, facts.captures[1].2,
        "camera consumer disconnected"
    );
    let provenance: Vec<scriptbots_bevy::capture::CaptureProvenance> = ["near", "overview"]
        .into_iter()
        .map(|name| {
            serde_json::from_slice(
                &std::fs::read(artifacts.join(format!("{name}.provenance.json"))).unwrap(),
            )
            .unwrap()
        })
        .collect();
    assert!(provenance[0].world_digest.is_some());
    assert_eq!(provenance[0].world_digest, provenance[1].world_digest);
    eprintln!("bookmark capture hashes: {:?}", facts.captures);
    for frame in &provenance {
        assert_eq!(frame.tick, 0);
        assert_eq!(frame.seed, manifest.seed);
        eprintln!(
            "bookmark evidence: {}",
            serde_json::to_string(frame).unwrap()
        );
    }
}

#[test]
#[serial]
fn look_development_exposure_changes_pixels_without_changing_science() {
    let _guard = GPU_GUARD.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().expect("exposure capture directory");
    let retained = std::env::var_os("SCRIPTBOTS_LOOK_CAPTURE_DIR").map(std::path::PathBuf::from);
    let root = retained.as_deref().unwrap_or(dir.path());
    let mut hashes = Vec::new();
    let mut digests = Vec::new();
    for (name, exposure) in [("baseline", 0.0), ("proposed", -0.75), ("reset", 0.0)] {
        let artifacts = root.join(name);
        std::fs::create_dir(&artifacts).expect("fresh exposure directory");
        let mut manifest = tiny_manifest(name);
        manifest.config_overrides = Some(
            toml::from_str(&format!(
                "world_width = 600\nworld_height = 600\n[render]\ntonemap_exposure_bias = {exposure:.2}\n"
            ))
            .unwrap(),
        );
        let mut driver = BevyOffscreenDriver {
            seed_agents: 8,
            viewport: (256, 256),
            artifacts_dir: Some(artifacts.clone()),
        };
        let facts = driver.run(&manifest).expect("real GPU exposure capture");
        let provenance: scriptbots_bevy::capture::CaptureProvenance =
            serde_json::from_slice(&std::fs::read(artifacts.join("mid.provenance.json")).unwrap())
                .unwrap();
        eprintln!(
            "exposure={exposure} captures={:?} provenance={}",
            facts.captures,
            serde_json::to_string(&provenance).unwrap()
        );
        assert!(provenance.world_digest.is_some());
        hashes.push(facts.captures[0].2.clone());
        digests.push(provenance.world_digest);
    }
    assert_ne!(hashes[0], hashes[1], "exposure consumer disconnected");
    assert_eq!(
        hashes[0], hashes[2],
        "exposure leaked into the next session"
    );
    assert!(digests.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
#[serial]
fn follow_camera_matches_explicit_view_without_changing_science() {
    let _guard = GPU_GUARD.lock().unwrap_or_else(|error| error.into_inner());
    let dir = tempfile::tempdir().expect("follow capture directory");
    let retained = std::env::var_os("SCRIPTBOTS_LOOK_CAPTURE_DIR")
        .map(std::path::PathBuf::from)
        .map(|root| root.join("follow"));
    if let Some(path) = &retained {
        std::fs::create_dir(path).expect("fresh retained follow directory");
    }
    let artifacts = retained.as_deref().unwrap_or(dir.path());
    let mut manifest = tiny_manifest("off-center-follow");
    manifest.config_overrides =
        Some(toml::from_str("world_width = 900\nworld_height = 600\n").unwrap());
    let mut world = scriptbots_core::WorldState::new(manifest.compose_config().unwrap()).unwrap();
    let first = world
        .try_spawn_agent(scriptbots_core::AgentData::default())
        .unwrap();
    let uid = world.agent_uid(first).unwrap().get();
    // The scene lattice starts at science (60, 60). In a 900 x 600 world,
    // its ground projection is (-390, 0, 240). Aim at it from above/south;
    // the mirrored negative instead looks toward Z=-240.
    let explicit = CameraKey {
        tick: 0,
        pos: [-390.0, 500.0, 740.0],
        yaw: std::f32::consts::PI,
        pitch: -std::f32::consts::FRAC_PI_4,
        fov: 55.0,
        follow_uid: None,
    };
    let mut follow = explicit.clone();
    follow.follow_uid = Some(uid);
    // A disconnected follow consumer must not pass by using the same ray.
    follow.yaw = 0.0;
    follow.pitch = 0.0;
    let mut mirrored = explicit.clone();
    mirrored.yaw = std::f32::consts::PI;
    mirrored.pitch = (-500.0_f32).atan2(980.0);
    manifest.camera = vec![follow, explicit, mirrored];
    manifest.captures = ["follow", "explicit", "mirrored"]
        .into_iter()
        .enumerate()
        .map(|(index, name)| CapturePoint {
            tick: 0,
            name: name.to_string(),
            camera_key: Some(index),
        })
        .collect();
    let mut driver = BevyOffscreenDriver {
        seed_agents: 8,
        viewport: (256, 256),
        artifacts_dir: Some(artifacts.to_path_buf()),
    };
    let facts = driver.run(&manifest).expect("real follow capture");
    assert_eq!(facts.captures.len(), manifest.captures.len());
    let mut pixels = Vec::new();
    let mut digests = Vec::new();
    for capture in &manifest.captures {
        let (width, height, rgba) = scriptbots_bevy::capture::decode_png(
            &std::fs::read(artifacts.join(format!("{}.png", capture.name))).unwrap(),
        )
        .unwrap();
        assert_eq!((width, height), driver.viewport);
        assert!(!rgba8_is_visually_blank(&rgba));
        pixels.push(rgba);
        let provenance: scriptbots_bevy::capture::CaptureProvenance = serde_json::from_slice(
            &std::fs::read(artifacts.join(format!("{}.provenance.json", capture.name))).unwrap(),
        )
        .unwrap();
        assert!(provenance.world_digest.is_some());
        eprintln!(
            "follow-view={} provenance={}",
            capture.name,
            serde_json::to_string(&provenance).unwrap()
        );
        digests.push(provenance.world_digest);
    }
    assert!(digests.windows(2).all(|pair| pair[0] == pair[1]));
    let correct = compare_frames(
        &pixels[0],
        &pixels[1],
        driver.viewport.0,
        driver.viewport.1,
        &CompareThresholds::default(),
    )
    .unwrap();
    let incorrect = compare_frames(
        &pixels[0],
        &pixels[2],
        driver.viewport.0,
        driver.viewport.1,
        &CompareThresholds::default(),
    )
    .unwrap();
    eprintln!(
        "follow hashes={:?} explicit={correct:?} mirrored={incorrect:?}",
        facts.captures
    );
    assert!(
        correct.pass,
        "follow must frame the actual agent ground position"
    );
    assert!(
        !incorrect.pass,
        "mirrored coordinates must fail image comparison"
    );
}

fn is_adapter_unavailable(error: &SceneError) -> bool {
    matches!(
        error.problems.as_slice(),
        [problem] if problem == ADAPTER_UNAVAILABLE_PROBLEM
    )
}

/// Render the tiny scene once, returning the capture hash + frame, or `None`
/// only when the capture backend explicitly reports that no adapter exists.
///
/// Device, map, timeout, encode, artifact, and metadata failures are harness
/// failures. Treating all of them as "no GPU here" made broken render paths
/// disappear behind a green skipped test.
fn render_tiny(
    name: &str,
    artifacts_dir: Option<&Path>,
) -> Option<(String, CapturedFrame, SceneManifest)> {
    let manifest = tiny_manifest(name);
    let mut driver = BevyOffscreenDriver {
        seed_agents: 8,
        viewport: (256, 256),
        artifacts_dir: artifacts_dir.map(Path::to_path_buf),
    };
    let facts = match driver.run(&manifest) {
        Ok(facts) => facts,
        Err(error) if is_adapter_unavailable(&error) => {
            eprintln!("SKIP: offscreen capture unavailable on this host: {error}");
            return None;
        }
        Err(error) => panic!("offscreen capture failed instead of producing evidence: {error}"),
    };
    let (_, _, hash, _) = facts
        .captures
        .first()
        .cloned()
        .expect("capture driver succeeded without recording the requested capture");
    // Re-read the frame from disk artifacts when available; otherwise
    // re-render for the frame (artifacts keep the honest path).
    let dir = artifacts_dir.expect("render_tiny requires an artifact directory");
    let png = std::fs::read(dir.join("mid.png"))
        .expect("successful capture must materialize its requested PNG");
    let (width, height, rgba8) =
        scriptbots_bevy::capture::decode_png(&png).expect("successful capture PNG must decode");
    let provenance = serde_json::from_str(
        &std::fs::read_to_string(dir.join("mid.provenance.json"))
            .expect("successful capture must materialize its provenance"),
    )
    .expect("successful capture provenance must decode");
    Some((
        hash,
        CapturedFrame {
            width,
            height,
            rgba8,
            provenance,
        },
        manifest,
    ))
}

#[test]
fn only_explicit_adapter_unavailability_may_skip_capture_evidence() {
    assert!(is_adapter_unavailable(&SceneError {
        problems: vec![ADAPTER_UNAVAILABLE_PROBLEM.to_string()],
    }));
    for failure in [
        "offscreen capture: render device unavailable",
        "offscreen capture: device poll during readback: lost",
        "offscreen capture: readback map timed out",
        "offscreen capture: readback buffer map failed",
        "offscreen capture: capture target metadata mismatch",
        "offscreen capture: offscreen render at tick 2: no GPU adapter available for offscreen capture",
        "offscreen capture: no GPU adapter available for offscreen capture (extra context)",
    ] {
        assert!(
            !is_adapter_unavailable(&SceneError {
                problems: vec![failure.to_string()],
            }),
            "real GPU failure was incorrectly made skippable: {failure}"
        );
    }
    assert!(
        !is_adapter_unavailable(&SceneError {
            problems: vec![
                ADAPTER_UNAVAILABLE_PROBLEM.to_string(),
                "a second failure must not disappear".to_string(),
            ],
        }),
        "a compound harness failure cannot be reduced to adapter unavailability"
    );
}

#[test]
#[serial]
fn offscreen_capture_is_deterministic_on_the_same_adapter() {
    let _guard = GPU_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let temp = tempfile::tempdir().expect("tempdir");
    let first = render_tiny("det_a", Some(temp.path()));
    let second = render_tiny("det_b", Some(temp.path()));
    let ((hash_a, frame_a, _), (hash_b, frame_b, _)) = match (first, second) {
        (Some(first), Some(second)) => (first, second),
        (None, None) => return, // SKIP already logged for an adapterless host.
        (first, second) => panic!(
            "adapter availability changed across identical captures: first={}, second={}",
            first.is_some(),
            second.is_some()
        ),
    };
    assert_eq!(
        hash_a, hash_b,
        "same scene, same adapter: capture hashes must be byte-identical"
    );
    assert_eq!(frame_a.rgba8, frame_b.rgba8, "pixel bytes identical");
    assert_eq!(
        frame_a.provenance.adapter_name,
        frame_b.provenance.adapter_name
    );
    assert_eq!(frame_a.provenance.backend, frame_b.provenance.backend);
    assert_eq!(
        frame_a.provenance.device_type,
        frame_b.provenance.device_type
    );
    assert_eq!(
        frame_a.provenance.quality_tier,
        frame_b.provenance.quality_tier
    );
    assert_eq!(frame_a.provenance.viewport, frame_b.provenance.viewport);
    assert_eq!(frame_a.provenance.seed, frame_b.provenance.seed);
    assert_eq!(frame_a.provenance.tick, frame_b.provenance.tick);
    // Provenance contract: real adapter identity, never blank.
    assert_eq!(
        frame_a.provenance.schema,
        "scriptbots.capture-provenance.v1"
    );
    assert_ne!(frame_a.provenance.adapter_name, "unknown");
    assert_eq!(frame_a.provenance.viewport, [256, 256]);
    assert_eq!(frame_a.provenance.colorspace, "rgba8-srgb");
    // A real frame of a lit world is not black and not uniform.
    assert!(
        !rgba8_is_visually_blank(&frame_a.rgba8),
        "a live render must contain varying, nonzero RGB pixels"
    );
}

#[test]
#[serial]
fn corrupted_pipeline_fails_the_harness() {
    let _guard = GPU_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let temp = tempfile::tempdir().expect("tempdir");
    let honest = match render_tiny("alarm_honest", Some(temp.path())) {
        Some((_, honest, _)) => honest,
        None => return,
    };
    // The alarm: corruption mode blacks out sun + ambient light. A golden
    // comparison between the honest and corrupted frames MUST fail — this is
    // the proof that a broken lighting/post path cannot ship green.
    let corrupt_env = ScopedEnvOverride::set("SCRIPTBOTS_CAPTURE_CORRUPT", "1");
    let corrupted = render_tiny("alarm_corrupt", Some(temp.path()));
    drop(corrupt_env);
    let (_, corrupted, _) = corrupted.expect(
        "the honest capture succeeded, so adapter unavailability on the corruption pass is a failure",
    );
    assert!(
        corrupted.provenance.corrupt,
        "corruption is labeled in provenance"
    );
    let stats = compare_frames(
        &honest.rgba8,
        &corrupted.rgba8,
        honest.width,
        honest.height,
        &CompareThresholds::default(),
    )
    .expect("same-shape compare");
    assert!(
        !stats.pass,
        "ALARM FAILED: a blacked-out pipeline passed the golden comparison: {stats:?}"
    );
    assert!(
        stats.differing_ratio > 0.05,
        "corruption should move a large share of pixels: {stats:?}"
    );
}

#[test]
#[serial]
fn golden_workflow_pass_regen_missing() {
    let _guard = GPU_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let temp = tempfile::tempdir().expect("tempdir");
    let Some((_, frame, _)) = render_tiny("golden_flow", Some(temp.path())) else {
        return;
    };
    let golden = temp.path().join("goldens/golden_flow/mid.png");

    // 1. Missing golden: explicit failure with regeneration instructions,
    //    candidate written for review — never an auto-bless.
    let outcome = process_golden(&frame, &golden, false).expect("missing golden workflow");
    let GoldenOutcome::MissingGolden {
        candidate,
        instructions,
    } = outcome
    else {
        panic!("missing golden must produce MissingGolden, got {outcome:?}");
    };
    assert!(candidate.exists(), "candidate written for review");
    assert!(
        instructions.contains("RUST_REGEN_GOLDEN"),
        "instructions name the regen path"
    );
    assert!(!golden.exists(), "no auto-bless");

    // 2. Regen: blesses byte-for-byte and writes provenance.
    let outcome = process_golden(&frame, &golden, true).expect("regen workflow");
    assert!(matches!(outcome, GoldenOutcome::Regenerated { .. }));
    assert!(golden.exists());
    assert!(
        temp.path()
            .join("goldens/golden_flow/mid.provenance.json")
            .exists()
    );

    // 3. Compare against the just-blessed golden: pass.
    let outcome = process_golden(&frame, &golden, false).expect("pass workflow");
    let GoldenOutcome::Pass {
        differing_ratio, ..
    } = outcome
    else {
        panic!("same-adapter compare must pass, got {outcome:?}");
    };
    assert_eq!(differing_ratio, 0.0);

    // 4. Tampered candidate: mismatch with a written heatmap.
    let mut tampered = frame.clone();
    for px in tampered.rgba8.as_chunks_mut::<4>().0.iter_mut().take(4096) {
        px[0] = px[0].wrapping_add(64);
    }
    let outcome = process_golden(&tampered, &golden, false).expect("mismatch workflow");
    let GoldenOutcome::Mismatch {
        heatmap,
        max_channel_diff,
        ..
    } = outcome
    else {
        panic!("tampered candidate must mismatch, got {outcome:?}");
    };
    assert!(heatmap.exists(), "diff heatmap written");
    assert_eq!(max_channel_diff, 64);
}

#[test]
fn checked_in_scene_manifests_all_validate() {
    let scenes_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/scenes");
    let mut count = 0;
    for entry in std::fs::read_dir(&scenes_dir).expect("scenes dir") {
        let entry = entry.expect("entry");
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
            continue;
        }
        SceneManifest::load(&path).unwrap_or_else(|error| {
            panic!("checked-in scene {} must validate: {error}", path.display())
        });
        count += 1;
    }
    assert!(
        count >= 5,
        "expected the five reference scenes, found {count}"
    );
}

#[test]
fn missing_scene_manifest_fails_closed() {
    let missing = Path::new("definitely/not/a/scene.toml");
    let error = SceneManifest::load(missing).expect_err("missing file must fail");
    assert!(
        error.to_string().contains("read"),
        "fail-closed read error: {error}"
    );
}

#[test]
fn bevy_offscreen_driver_honors_manifest_quality() {
    // No-GPU path: only explicit adapter unavailability may skip. A successful
    // run must render a real frame whose provenance proves the requested tier
    // reached the renderer.
    let mut manifest = tiny_manifest("quality_gate");
    manifest.quality = Some("ultra".to_string());
    let temp = tempfile::tempdir().expect("quality proof tempdir");
    let mut driver = BevyOffscreenDriver {
        seed_agents: 4,
        viewport: (64, 64),
        artifacts_dir: Some(temp.path().to_path_buf()),
    };
    match driver.run(&manifest) {
        Ok(facts) => {
            assert_eq!(facts.agent_counts.len(), 5, "ticks 0..=4");
            assert!(facts.world_digest.is_some());
            assert_eq!(facts.captures.len(), 1, "the quality proof must render");
            let provenance: scriptbots_bevy::capture::CaptureProvenance = serde_json::from_str(
                &std::fs::read_to_string(temp.path().join("mid.provenance.json"))
                    .expect("quality proof provenance"),
            )
            .expect("decode quality proof provenance");
            assert_eq!(
                provenance.quality_tier, "Ultra",
                "an explicit Ultra request must be consumed by the actual renderer"
            );
        }
        Err(error) => {
            assert!(
                is_adapter_unavailable(&error),
                "only explicit adapter unavailability may skip this GPU proof, got: {error}"
            );
        }
    }
}

#[test]
fn png_roundtrip_preserves_bytes() {
    let rgba: Vec<u8> = (0..(32 * 32 * 4)).map(|i| (i * 7 % 251) as u8).collect();
    let png = encode_png(32, 32, &rgba).expect("encode");
    let (w, h, back) = scriptbots_bevy::capture::decode_png(&png).expect("decode");
    assert_eq!((w, h), (32, 32));
    assert_eq!(rgba, back, "PNG roundtrip is lossless for RGBA8");
}
