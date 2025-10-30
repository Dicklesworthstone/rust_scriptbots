use scriptbots_bevy::render_png_offscreen;
use scriptbots_core::{ScriptBotsConfig, WorldState};

#[test]
fn camera_controls_do_not_panic() {
    let config = ScriptBotsConfig::default();
    let mut world = WorldState::new(config).expect("world init");
    for _ in 0..60 {
        world.step();
    }
    let _ = render_png_offscreen(&world, 800, 600).expect("render png");
}
