//! Live GPU probe for OffscreenCapture (debugging readback wedge).
//! Optional first argument saves the observed frame as a new PNG file.
//! A software-adapter capture is smoke evidence, not hardware acceptance.
use scriptbots_bevy::capture::{OffscreenCapture, OffscreenCaptureConfig};
use scriptbots_core::{AgentData, RenderSettings, ScriptBotsConfig, WorldState};

fn main() {
    let output = std::env::args_os().nth(1);
    let mut world = WorldState::new(ScriptBotsConfig {
        rng_seed: Some(42),
        world_width: 600,
        world_height: 600,
        ..ScriptBotsConfig::default()
    })
    .expect("world");
    let mut boosted = Vec::new();
    for i in 0..4u32 {
        let mut a = AgentData::default();
        a.position.x = 200.0 + (i % 2) as f32 * 200.0;
        a.position.y = 200.0 + (i / 2) as f32 * 200.0;
        a.spike_length = 1.0;
        let id = world.try_spawn_agent(a).expect("spawn");
        if i % 2 == 1 {
            boosted.push(id);
        }
    }
    // Step once so the probe exercises a completed science boundary.
    world.step().expect("step once");
    // Explicit diagnostic state: display active and inactive boost cues in
    // the same frame, without relying on a random brain choosing to boost.
    for id in boosted {
        assert!(
            world
                .try_update_agent(id, |agent, _| agent.boost = true)
                .expect("set diagnostic boost")
        );
    }
    let config = OffscreenCaptureConfig {
        viewport: (800, 600),
        render_settings: RenderSettings::default(),
        corrupt: false,
    };
    let (w, h, len, adapter, backend, spread, first, center) =
        OffscreenCapture::run(&config, |session| {
            eprintln!("session tier = {:?}", session.tier());
            session.set_camera_pose([0.0, 450.0, 450.0], [0.0, 0.0, 0.0], 55.0);
            let frame = session.render(&world, "probe", 42, 1)?;
            if let Some(path) = &output {
                use std::io::Write;
                let png =
                    scriptbots_bevy::capture::encode_png(frame.width, frame.height, &frame.rgba8)?;
                // Never overwrite an earlier capture or a user-owned file.
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?;
                file.write_all(&png)?;
                eprintln!(
                    "capture PNG saved: {}",
                    std::path::Path::new(path).display()
                );
            }
            let mut min = [255u8; 3];
            let mut max = [0u8; 3];
            for px in frame.rgba8.as_chunks::<4>().0 {
                for c in 0..3 {
                    min[c] = min[c].min(px[c]);
                    max[c] = max[c].max(px[c]);
                }
            }
            let spread: u32 = (0..3).map(|c| u32::from(max[c] - min[c])).sum();
            let center_idx = ((frame.height / 2 * frame.width + frame.width / 2) * 4) as usize;
            let first: [u8; 4] = frame.rgba8[0..4].try_into().unwrap_or([0; 4]);
            let center: [u8; 4] = frame.rgba8[center_idx..center_idx + 4]
                .try_into()
                .unwrap_or([0; 4]);
            Ok((
                frame.width,
                frame.height,
                frame.rgba8.len(),
                frame.provenance.adapter_name,
                frame.provenance.backend,
                spread,
                first,
                center,
            ))
        })
        .expect("render");
    eprintln!("frame: {w}x{h} rgba8={len} adapter={adapter} backend={backend} spread={spread}");
    eprintln!("first px = {first:?} center px = {center:?}");
    assert!(
        spread > 24,
        "frame must contain visual variance, got {spread}"
    );
}
