use std::{fs, path::PathBuf};

use scriptbots_bevy::render_png_offscreen;
use scriptbots_core::{ScriptBotsConfig, WorldState};

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("docs/rendering_reference/golden/bevy_default.png")
}

#[test]
fn bevy_renderer_matches_golden() {
    let path = golden_path();
    let golden = fs::read(&path).expect("load golden bevy snapshot");

    let config = ScriptBotsConfig::default();
    let mut world = WorldState::new(config).expect("world init");
    for _ in 0..120 {
        world.step();
    }
    let produced = render_png_offscreen(&world, 1600, 900).expect("render bevy png");

    assert_eq!(produced.len(), golden.len(), "PNG length mismatch");

    let diff_bytes = produced
        .iter()
        .zip(golden.iter())
        .filter(|(a, b)| a != b)
        .count();
    if diff_bytes != 0 {
        panic!(
            "Bevy snapshot differs from golden ({} differing bytes); regenerate golden if intentional",
            diff_bytes
        );
    }
}
