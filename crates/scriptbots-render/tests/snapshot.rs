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
fn rust_renderer_matches_golden_snapshot() {
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
        return;
    }
    let expected = fs::read(&golden_path).expect(
        "golden snapshot missing; generate locally with: RUST_REGEN_GOLDEN=1 cargo test -p scriptbots-render --test snapshot rust_renderer_matches_golden_snapshot -- --exact --nocapture",
    );

    if png != expected {
        let failure_dir = project_root().join("target").join("snapshot-failures");
        fs::create_dir_all(&failure_dir).expect("create failure dir");
        let actual_path = failure_dir.join("rust_default.actual.png");
        fs::write(&actual_path, &png).expect("write actual snapshot");
        panic!(
            "Rust snapshot diverged from golden.\nexpected: {}\nactual: {}",
            golden_path.display(),
            actual_path.display()
        );
    }
}
