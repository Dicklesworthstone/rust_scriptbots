//! Minimal isolation: does ReadbackComplete fire in a manually-pumped app?
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::window::{ExitCondition, WindowPlugin};
use std::sync::{Arc, Mutex};

fn main() {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .build()
            .disable::<bevy::winit::WinitPlugin>()
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                close_when_requested: false,
                ..Default::default()
            }),
    );
    app.finish();
    app.cleanup();
    eprintln!("plugins ready");

    let image = Image::new_fill(
        Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[9, 9, 9, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);

    let slot: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let slot2 = Arc::clone(&slot);
    app.world_mut()
        .spawn(Readback::texture(target))
        .observe(move |event: On<ReadbackComplete>| {
            eprintln!("ReadbackComplete fired! {} bytes", event.data.len());
            let mut guard = slot2.lock().unwrap_or_else(|e| e.into_inner());
            *guard = Some(event.data.clone());
        });

    for i in 0..10 {
        app.update();
        eprintln!("frame {i} done");
        let done = slot.lock().unwrap_or_else(|e| e.into_inner()).is_some();
        if done {
            eprintln!("READBACK COMPLETED at frame {i}");
            return;
        }
    }
    eprintln!("READBACK NEVER COMPLETED after 10 frames");
}
