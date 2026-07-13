use std::{fs, path::PathBuf};

use scriptbots_brain::MlpBrain;
use scriptbots_core::{AgentData, ScriptBotsConfig, WorldState};
use scriptbots_render::render_png_offscreen;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("scriptbots-render crate nested under crates/")
        .to_path_buf()
}

fn golden_dir() -> PathBuf {
    project_root()
        .join("docs")
        .join("rendering_reference")
        .join("golden")
}

fn register_brains(world: &mut WorldState) -> u64 {
    world
        .brain_registry_mut()
        .register(MlpBrain::KIND.as_str(), |seed_rng| {
            Ok(MlpBrain::runner(seed_rng))
        })
}

fn seed_agents(world: &mut WorldState, brain_key: u64) {
    let mut agent = AgentData::default();
    let spacing = 120.0;
    for row in 0..4 {
        for col in 0..4 {
            agent.position.x = col as f32 * spacing + spacing * 0.5;
            agent.position.y = row as f32 * spacing + spacing * 0.5;
            agent.heading = 0.0;
            agent.spike_length = 10.0;
            let id = world
                .try_spawn_agent(agent)
                .expect("snapshot agent is finite");
            assert!(
                world
                    .bind_agent_brain(id, brain_key)
                    .expect("snapshot MLP factory")
            );
        }
    }
}

#[test]
fn scriptbots_render_cpu_surrogate_raster_matches_semantic_golden() -> Result<(), String> {
    let config = ScriptBotsConfig {
        rng_seed: Some(424_242),
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("initialize world");

    let brain_key = register_brains(&mut world);
    seed_agents(&mut world, brain_key);
    for _ in 0..120 {
        world.step().expect("snapshot fixture simulation step");
    }

    let png = render_png_offscreen(&world, 1600, 900);
    let golden_path = golden_dir().join("rust_default.png");
    let regenerate = std::env::var("RUST_REGEN_GOLDEN")
        .map(|value| value == "1")
        .unwrap_or(false);
    assert!(
        !(regenerate && std::env::var_os("CI").is_some()),
        "CI must never regenerate or bless Rust golden assets"
    );
    if regenerate {
        fs::create_dir_all(golden_dir()).expect("create golden snapshot directory");
        fs::write(&golden_path, &png).expect("write updated golden");
        return Ok(());
    }
    let expected = fs::read(&golden_path).expect(
        "CPU-surrogate semantic golden missing; this test does not exercise a GPUI window, GPUI paint, or wgpu framebuffer. Generate the candidate locally with: RUST_REGEN_GOLDEN=1 cargo test -p scriptbots-render --test snapshot scriptbots_render_cpu_surrogate_raster_matches_semantic_golden -- --exact --nocapture; then review the image and Git diff before committing it",
    );

    if png != expected {
        let rgba = image::load_from_memory(&png)
            .expect("decode generated PNG for mismatch diagnostics")
            .to_rgba8();
        let expected_rgba = image::load_from_memory(&expected)
            .expect("decode golden PNG for mismatch diagnostics")
            .to_rgba8();
        let transparent_pixels = rgba.pixels().filter(|pixel| pixel.0[3] == 0).count();
        let opaque_black_pixels = rgba
            .pixels()
            .filter(|pixel| pixel.0 == [0, 0, 0, 255])
            .count();
        let mut color_counts = std::collections::BTreeMap::<[u8; 4], usize>::new();
        for pixel in rgba.pixels() {
            *color_counts.entry(pixel.0).or_default() += 1;
        }
        let mut common_colors = color_counts.into_iter().collect::<Vec<_>>();
        common_colors.sort_unstable_by(|left, right| right.1.cmp(&left.1));
        common_colors.truncate(8);
        let mut differing_pixels = 0usize;
        let mut difference_bounds = None::<(u32, u32, u32, u32)>;
        let mut maximum_channel_delta = 0u8;
        for (x, y, actual) in rgba.enumerate_pixels() {
            let expected = expected_rgba.get_pixel(x, y);
            if actual != expected {
                differing_pixels += 1;
                difference_bounds = Some(match difference_bounds {
                    Some((min_x, min_y, max_x, max_y)) => {
                        (min_x.min(x), min_y.min(y), max_x.max(x), max_y.max(y))
                    }
                    None => (x, y, x, y),
                });
                for (actual, expected) in actual.0.iter().zip(expected.0) {
                    maximum_channel_delta = maximum_channel_delta.max(actual.abs_diff(expected));
                }
            }
        }
        let failure_dir = project_root().join("target").join("snapshot-failures");
        fs::create_dir_all(&failure_dir).expect("create failure dir");
        let actual_path = failure_dir.join("rust_default.actual.png");
        fs::write(&actual_path, &png).expect("write actual snapshot");
        return Err(format!(
            "scriptbots-render CPU-surrogate raster diverged from its semantic golden; this is not evidence about the shipped GPUI or wgpu framebuffer.\nexpected: {}\nactual: {}\ntransparent pixels: {transparent_pixels}\nopaque black pixels: {opaque_black_pixels}\ndiffering pixels: {differing_pixels}\ndifference bounds: {difference_bounds:?}\nmaximum channel delta: {maximum_channel_delta}\nmost common RGBA colors: {common_colors:?}",
            golden_path.display(),
            actual_path.display()
        ));
    }
    Ok(())
}
