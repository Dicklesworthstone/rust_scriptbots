use std::{fs, path::PathBuf};

use image::{ImageBuffer, Rgba};
use scriptbots_bevy::render_png_offscreen;
use scriptbots_core::{ScriptBotsConfig, WorldState};

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/rendering_reference/golden/bevy_default.png")
        .canonicalize()
        .unwrap_or_else(|_| {
            panic!(
                "unable to resolve golden path from {}",
                env!("CARGO_MANIFEST_DIR")
            )
        })
}

#[test]
fn bevy_renderer_matches_golden() {
    let path = golden_path();
    let golden = fs::read(&path).expect("load golden bevy snapshot");

    let mut config = ScriptBotsConfig::default();
    config.rng_seed = Some(0xBEEF_F00D);
    let mut world = WorldState::new(config).expect("world init");
    for _ in 0..120 {
        world.step();
    }
    let produced = render_png_offscreen(&world, 1600, 900).expect("render bevy png");

    if std::env::var("BEVY_REGEN_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        fs::write(&path, &produced).expect("write updated golden");
        return;
    }

    let golden_img = image::load_from_memory(&golden)
        .expect("decode golden")
        .to_rgba8();
    let produced_img = image::load_from_memory(&produced)
        .expect("decode produced")
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
