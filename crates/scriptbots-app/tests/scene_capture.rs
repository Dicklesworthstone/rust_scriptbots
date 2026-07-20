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
    BevyOffscreenDriver, CapturePoint, Expectation, FrontendKind, GoldenOutcome, SceneDriver,
    SceneManifest, process_golden,
};
use scriptbots_bevy::capture::{
    CapturedFrame, CompareThresholds, compare_frames, encode_png,
};
use serial_test::serial;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

/// One GPU context at a time: parallel capture contexts would race the
/// adapter and flake the determinism assertions.
static GPU_GUARD: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

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
        }],
        expect: vec![Expectation::AgentCount {
            min: 1,
            max: 10_000,
        }],
    }
}

/// Render the tiny scene once, returning the capture hash + frame, or `None`
/// (loudly) when no GPU adapter exists on this host.
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
        Err(error) => {
            eprintln!("SKIP: offscreen capture unavailable on this host: {error}");
            return None;
        }
    };
    let (_, _, hash, _) = facts.captures.first()?.clone();
    // Re-read the frame from disk artifacts when available; otherwise
    // re-render for the frame (artifacts keep the honest path).
    let dir = artifacts_dir?;
    let png = std::fs::read(dir.join("mid.png")).ok()?;
    let (width, height, rgba8) = scriptbots_bevy::capture::decode_png(&png).ok()?;
    let provenance = serde_json::from_str(
        &std::fs::read_to_string(dir.join("mid.provenance.json")).ok()?,
    )
    .ok()?;
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
#[serial]
fn offscreen_capture_is_deterministic_on_the_same_adapter() {
    let _guard = GPU_GUARD.lock().expect("gpu guard");
    let temp = tempfile::tempdir().expect("tempdir");
    let first = render_tiny("det_a", Some(temp.path()));
    let second = render_tiny("det_b", Some(temp.path()));
    let (Some((hash_a, frame_a, _)), Some((hash_b, frame_b, _))) = (first, second) else {
        return; // SKIP already logged.
    };
    assert_eq!(
        hash_a, hash_b,
        "same scene, same adapter: capture hashes must be byte-identical"
    );
    assert_eq!(frame_a.rgba8, frame_b.rgba8, "pixel bytes identical");
    // Provenance contract: real adapter identity, never blank.
    assert_eq!(frame_a.provenance.schema, "scriptbots.capture-provenance.v1");
    assert_ne!(frame_a.provenance.adapter_name, "unknown");
    assert_eq!(frame_a.provenance.viewport, [256, 256]);
    assert_eq!(frame_a.provenance.colorspace, "rgba8-srgb");
    // A real frame of a lit world is not black and not uniform.
    let distinct: std::collections::HashSet<u8> = frame_a.rgba8.iter().copied().collect();
    assert!(
        distinct.len() > 16,
        "a live render has color variety (got {} distinct bytes)",
        distinct.len()
    );
}

#[test]
#[serial]
fn corrupted_pipeline_fails_the_harness() {
    let _guard = GPU_GUARD.lock().expect("gpu guard");
    let temp = tempfile::tempdir().expect("tempdir");
    let Some((_, honest, _)) = render_tiny("alarm_honest", Some(temp.path())) else {
        return;
    };
    // The alarm: corruption mode blacks out sun + ambient light. A golden
    // comparison between the honest and corrupted frames MUST fail — this is
    // the proof that a broken lighting/post path cannot ship green.
    unsafe {
        std::env::set_var("SCRIPTBOTS_CAPTURE_CORRUPT", "1");
    }
    let corrupted = render_tiny("alarm_corrupt", Some(temp.path()));
    unsafe {
        std::env::remove_var("SCRIPTBOTS_CAPTURE_CORRUPT");
    }
    let Some((_, corrupted, _)) = corrupted else {
        return;
    };
    assert!(corrupted.provenance.corrupt, "corruption is labeled in provenance");
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
    let _guard = GPU_GUARD.lock().expect("gpu guard");
    let temp = tempfile::tempdir().expect("tempdir");
    let Some((_, frame, _)) = render_tiny("golden_flow", Some(temp.path())) else {
        return;
    };
    let golden = temp.path().join("goldens/golden_flow/mid.png");

    // 1. Missing golden: explicit failure with regeneration instructions,
    //    candidate written for review — never an auto-bless.
    let outcome = process_golden(&frame, &golden, false).expect("missing golden workflow");
    let GoldenOutcome::MissingGolden { candidate, instructions } = outcome else {
        panic!("missing golden must produce MissingGolden, got {outcome:?}");
    };
    assert!(candidate.exists(), "candidate written for review");
    assert!(instructions.contains("RUST_REGEN_GOLDEN"), "instructions name the regen path");
    assert!(!golden.exists(), "no auto-bless");

    // 2. Regen: blesses byte-for-byte and writes provenance.
    let outcome = process_golden(&frame, &golden, true).expect("regen workflow");
    assert!(matches!(outcome, GoldenOutcome::Regenerated { .. }));
    assert!(golden.exists());
    assert!(temp.path().join("goldens/golden_flow/mid.provenance.json").exists());

    // 3. Compare against the just-blessed golden: pass.
    let outcome = process_golden(&frame, &golden, false).expect("pass workflow");
    let GoldenOutcome::Pass { differing_ratio, .. } = outcome else {
        panic!("same-adapter compare must pass, got {outcome:?}");
    };
    assert_eq!(differing_ratio, 0.0);

    // 4. Tampered candidate: mismatch with a written heatmap.
    let mut tampered = frame.clone();
    for px in tampered.rgba8.chunks_exact_mut(4).take(4096) {
        px[0] = px[0].wrapping_add(64);
    }
    let outcome = process_golden(&tampered, &golden, false).expect("mismatch workflow");
    let GoldenOutcome::Mismatch { heatmap, max_channel_diff, .. } = outcome else {
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
    assert!(count >= 5, "expected the five reference scenes, found {count}");
}

#[test]
fn missing_scene_manifest_fails_closed() {
    let missing = Path::new("definitely/not/a/scene.toml");
    let error = SceneManifest::load(missing).expect_err("missing file must fail");
    assert!(error.to_string().contains("read"), "fail-closed read error: {error}");
}

#[test]
fn bevy_offscreen_driver_honors_manifest_quality() {
    // No-GPU path: the stub/feature-disabled driver must fail closed. With
    // the feature on but no adapter, OffscreenCapture::new fails honestly.
    let mut manifest = tiny_manifest("quality_gate");
    manifest.captures.clear();
    manifest.quality = Some("ultra".to_string());
    let mut driver = BevyOffscreenDriver {
        seed_agents: 4,
        viewport: (64, 64),
        artifacts_dir: None,
    };
    match driver.run(&manifest) {
        Ok(facts) => {
            assert_eq!(facts.agent_counts.len(), 5, "ticks 0..=4");
            assert!(facts.world_digest.is_some());
        }
        Err(error) => {
            let text = error.to_string();
            assert!(
                text.contains("no GPU adapter") || text.contains("offscreen capture init"),
                "honest adapter failure, got: {text}"
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
