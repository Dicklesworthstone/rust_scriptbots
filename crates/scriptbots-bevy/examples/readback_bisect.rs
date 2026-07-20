//! Bisect: replicate capture.rs's exact build sequence to find the wedge.
use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::window::{ExitCondition, WindowPlugin};
use scriptbots_bevy::capture::OffscreenCaptureConfig;
use std::sync::{Arc, Mutex};

fn main() {
    let config = OffscreenCaptureConfig {
        viewport: (64, 64),
        render_settings: Default::default(),
        corrupt: false,
    };

    // --- build_capture_app sequence ---
    let mut app = App::new();
    app.insert_resource(AmbientLight {
        color: Color::srgb(0.45, 0.52, 0.65),
        brightness: 800.0,
        affects_lightmapped_meshes: true,
    })
    .add_plugins(
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

    // --- configure_session: wipe entities, create target, spawn rig via
    // run_system_once, warm up ---
    let mut entity_query = app.world_mut().query::<bevy::ecs::world::EntityRef>();
    let entities: Vec<Entity> = entity_query
        .iter(app.world())
        .map(|entity_ref| entity_ref.id())
        .collect();
    let wiped = entities.len();
    for entity in entities {
        let _ = app.world_mut().despawn(entity);
    }
    eprintln!("wiped {wiped} entities");
    let mut image = Image::new_fill(
        Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);

    // Camera bound to the offscreen target, spawned via run_system_once like
    // setup_capture_scene (commands path).
    let cam_target = target.clone();
    app.world_mut()
        .run_system_once(move |mut commands: Commands| {
            commands.spawn((
                Camera3d::default(),
                Camera {
                    clear_color: ClearColorConfig::Custom(Color::srgb(0.03, 0.05, 0.09)),
                    target: RenderTarget::Image(cam_target.clone().into()),
                    is_active: false,
                    ..Default::default()
                },
                Transform::from_xyz(0.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
                Tonemapping::AcesFitted,
            ));
        })
        .expect("setup system");
    app.world_mut().spawn((
        DirectionalLight {
            illuminance: 9000.0,
            shadows_enabled: false,
            ..Default::default()
        },
        Transform::from_xyz(-5.0, 10.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Two warmup frames like configure_session.
    app.update();
    app.update();
    eprintln!("warmup done");

    // Flip the camera on like render() does.
    let mut cameras = app.world_mut().query::<&mut Camera>();
    for mut camera in cameras.iter_mut(app.world_mut()) {
        camera.is_active = true;
    }
    app.update();
    app.update();

    // Now the render() readback sequence.
    let slot: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let slot2 = Arc::clone(&slot);
    app.world_mut()
        .spawn(Readback::texture(target.clone()))
        .observe(move |event: On<ReadbackComplete>| {
            eprintln!("ReadbackComplete fired! {} bytes", event.data.len());
            let mut guard = slot2.lock().unwrap_or_else(|e| e.into_inner());
            *guard = Some(event.data.clone());
        });

    for i in 0..16 {
        app.update();
        if let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp)
            && let Some(device) = render_app
                .world_mut()
                .get_resource::<bevy::render::renderer::RenderDevice>()
        {
            let _ = device.poll(wgpu::PollType::Wait);
        }
        let done = slot.lock().unwrap_or_else(|e| e.into_inner()).is_some();
        if done {
            let data = slot
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
                .unwrap_or_default();
            let first: [u8; 4] = data[0..4].try_into().unwrap_or([0; 4]);
            eprintln!("READBACK COMPLETED at pump {i}, first px = {first:?}");
            return;
        }
    }
    eprintln!("READBACK NEVER COMPLETED after 16 pumps");
    let _ = config;
}
