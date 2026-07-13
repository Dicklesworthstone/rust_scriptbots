use std::{fs, path::PathBuf};

use image::{ImageBuffer, Rgba};
use scriptbots_bevy::render_png_offscreen;
use scriptbots_core::{AgentData, ScriptBotsConfig, WorldState};

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/rendering_reference/golden/bevy_default.png")
}

fn seed_visible_agents(world: &mut WorldState) {
    let world_width = world.config().world_width as f32;
    let world_height = world.config().world_height as f32;
    let mut agent = AgentData::default();

    for row in 0..4 {
        for column in 0..4 {
            agent.position.x = (column + 1) as f32 * world_width / 5.0;
            agent.position.y = (row + 1) as f32 * world_height / 5.0;
            agent.heading = (row * 4 + column) as f32 * std::f32::consts::FRAC_PI_8;
            agent.spike_length = 10.0;
            world
                .try_spawn_agent(agent)
                .expect("snapshot agent is finite");
        }
    }
}

#[test]
fn bevy_scene_cpu_surrogate_raster_matches_semantic_golden() {
    let path = golden_path();
    let config = ScriptBotsConfig {
        rng_seed: Some(0xBEEF_F00D),
        bot_radius: 30.0,
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("world init");
    for _ in 0..120 {
        world
            .step()
            .expect("snapshot test world should accept each simulation step");
    }
    let terrain_only = render_png_offscreen(&world, 1600, 900).expect("render terrain fixture");
    seed_visible_agents(&mut world);
    let produced = render_png_offscreen(&world, 1600, 900).expect("render bevy png");
    let terrain_img = image::load_from_memory(&terrain_only)
        .expect("decode terrain fixture")
        .to_rgba8();
    let produced_img = image::load_from_memory(&produced)
        .expect("decode produced")
        .to_rgba8();
    let agent_signal_pixels = terrain_img
        .pixels()
        .zip(produced_img.pixels())
        .filter(|(terrain, rendered)| terrain.0[..3] != rendered.0[..3])
        .count();
    assert!(
        agent_signal_pixels >= 1_000,
        "snapshot fixture rendered only {agent_signal_pixels} agent-signal pixels"
    );

    let regenerate = std::env::var("BEVY_REGEN_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false);
    assert!(
        !(regenerate && std::env::var_os("CI").is_some()),
        "CI must never regenerate or bless Bevy golden assets"
    );
    if regenerate {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create golden snapshot directory");
        }
        fs::write(&path, &produced).expect("write updated golden");
        return;
    }

    let golden = fs::read(&path).expect(
        "Bevy-scene CPU-surrogate semantic golden missing; this test does not construct a Bevy App, camera, PBR pipeline, render graph, or GPU framebuffer. Generate the candidate locally with: BEVY_REGEN_GOLDEN=1 cargo test -p scriptbots-bevy --test snapshot bevy_scene_cpu_surrogate_raster_matches_semantic_golden -- --exact --nocapture; then review the image and Git diff before committing it",
    );
    let golden_img = image::load_from_memory(&golden)
        .expect("decode golden")
        .to_rgba8();

    assert_eq!(golden_img.dimensions(), produced_img.dimensions());

    let (width, height) = golden_img.dimensions();
    let mut total_diff = 0u64;
    let mut max_diff = 0u8;

    for (g, p) in golden_img.pixels().zip(produced_img.pixels()) {
        for c in 0..3 {
            let diff = g[c].abs_diff(p[c]);
            total_diff += diff as u64;
            max_diff = max_diff.max(diff);
        }
    }

    let channel_count = (width as u64) * (height as u64) * 3;
    let mean_diff = total_diff as f64 / channel_count as f64;
    assert!(mean_diff <= 40.0, "mean channel diff too high: {mean_diff}");
    assert!(max_diff <= 200, "max channel diff too high: {max_diff}");

    let golden_hist = luminance_histogram(&golden_img);
    let produced_hist = luminance_histogram(&produced_img);
    let hist_delta: u64 = golden_hist
        .iter()
        .zip(produced_hist.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .sum();
    assert!(
        hist_delta <= (width as u64 * height as u64) / 5,
        "luminance histogram drift too large: {hist_delta}"
    );
}

fn luminance_histogram(image: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> [u64; 16] {
    let mut bins = [0u64; 16];
    for pixel in image.pixels() {
        let luminance = 0.2126 * f32::from(pixel[0])
            + 0.7152 * f32::from(pixel[1])
            + 0.0722 * f32::from(pixel[2]);
        let idx = ((luminance / 255.0) * 15.0).clamp(0.0, 15.0) as usize;
        bins[idx] += 1;
    }
    bins
}
