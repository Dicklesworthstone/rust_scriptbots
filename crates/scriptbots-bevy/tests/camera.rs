use scriptbots_bevy::render_png_offscreen;
use scriptbots_core::{ScriptBotsConfig, WorldState};

#[test]
fn bevy_scene_cpu_surrogate_raster_smoke() {
    let config = ScriptBotsConfig::default();
    let mut world = WorldState::new(config).expect("world init");
    for _ in 0..60 {
        world
            .step()
            .expect("CPU-surrogate fixture world should accept each simulation step");
    }
    let png = render_png_offscreen(&world, 800, 600).expect("render CPU-surrogate PNG");
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
}
