//! Live GPU probe for OffscreenCapture (debugging readback wedge).
use scriptbots_bevy::capture::{OffscreenCapture, OffscreenCaptureConfig};
use scriptbots_core::{AgentData, RenderSettings, ScriptBotsConfig, WorldState};

fn main() {
    let mut world = WorldState::new(ScriptBotsConfig {
        rng_seed: Some(42),
        ..ScriptBotsConfig::default()
    })
    .expect("world");
    for i in 0..4u32 {
        let mut a = AgentData::default();
        a.position.x = 100.0 + i as f32 * 60.0;
        a.position.y = 100.0;
        a.spike_length = 10.0;
        world.try_spawn_agent(a).expect("spawn");
    }
    // Tick 0 == SnapshotState::default().last_applied_tick, so sync_world
    // skips a never-stepped world; step once so the scene actually syncs.
    world.step().expect("step once");
    let config = OffscreenCaptureConfig {
        viewport: (320, 240),
        render_settings: RenderSettings::default(),
        corrupt: false,
    };
    let (w, h, len, adapter, backend, spread, first, center) =
        OffscreenCapture::run(&config, |session| {
            eprintln!("session tier = {:?}", session.tier());
            let frame = session.render(&world, "probe", 42, 0)?;
            let mut min = [255u8; 3];
            let mut max = [0u8; 3];
            for px in frame.rgba8.chunks_exact(4) {
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
