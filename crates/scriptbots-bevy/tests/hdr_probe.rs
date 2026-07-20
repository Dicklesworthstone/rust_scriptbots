//! bd-x6v6 upstream-regression canary (own process by necessity):
//! a minimal bevy-only app — Hdr + tonemapping camera targeting an image,
//! one lit cube, no scriptbots systems — proves the HDR image-target path
//! renders content on this platform. If a future bevy/wgpu/driver upgrade
//! breaks HDR render-to-texture headless, THIS test fails first and
//! loudly, before the capture harness's own e2e does.
//!
//! It lives in an integration-test binary (its own process) because bevy
//! 0.17 keeps a process-global empty bind group layout: two render apps in
//! one process panic, and the crate's lib tests already build the capture
//! singleton app.
//!
//! Skip loudly (never silently pass) when no GPU adapter exists.

use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::camera::prelude::*;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::math::primitives::Cuboid;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::view::Hdr;
use bevy::window::{ExitCondition, WindowPlugin};
use scriptbots_bevy::capture::unpad_readback;

#[test]
fn hdr_image_target_renders_content_upstream_canary() {
    if scriptbots_bevy::probe_gpu_capability().is_none() {
        eprintln!("SKIP: hdr canary needs a GPU adapter");
        return;
    }
    let mut app = App::new();
    app.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
        affects_lightmapped_meshes: true,
    })
    .add_plugins(
        DefaultPlugins
            .build()
            .disable::<bevy::winit::WinitPlugin>()
            .disable::<bevy::render::pipelined_rendering::PipelinedRenderingPlugin>()
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                close_when_requested: false,
                ..Default::default()
            }),
    );
    let size = Extent3d {
        width: 128,
        height: 128,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    app.world_mut().spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(0.2, 0.0, 0.4)),
            target: RenderTarget::Image(target.clone().into()),
            ..Default::default()
        },
        Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        Tonemapping::AcesFitted,
        Hdr,
    ));
    let cube = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Mesh::from(Cuboid::new(1.0, 1.0, 1.0)));
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.4, 0.1),
            ..Default::default()
        });
    app.world_mut()
        .spawn((Mesh3d(cube), MeshMaterial3d(material), Transform::default()));
    app.world_mut().spawn((
        PointLight {
            intensity: 800_000.0,
            ..Default::default()
        },
        Transform::from_xyz(3.0, 4.0, 5.0),
    ));
    app.finish();
    app.cleanup();
    for _ in 0..6 {
        app.update();
    }

    // Manual readback of the target (same path the capture harness uses).
    let data = {
        let render_app = app.get_sub_app_mut(RenderApp).expect("render app");
        let world = render_app.world_mut();
        let device = world.resource::<RenderDevice>().clone();
        let queue = world.resource::<RenderQueue>().clone();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>().clone();
        let gpu_image = gpu_images.get(&target).expect("gpu image prepared");
        let bytes_per_row = RenderDevice::align_copy_bytes_per_row(128 * 4) as u32;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hdr_canary_readback"),
            size: u64::from(bytes_per_row) * 128,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            gpu_image.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            gpu_image.size,
        );
        queue.submit(std::iter::once(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device.poll(wgpu::PollType::Wait).expect("poll");
        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("map timeout")
            .expect("map");
        let data = {
            let mapped = buffer.slice(..).get_mapped_range();
            Vec::from(&*mapped)
        };
        buffer.unmap();
        data
    };
    let tight = unpad_readback(&data, 128, 128);
    let distinct: std::collections::HashSet<u8> = tight.iter().copied().collect();
    let max_channel = tight
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| px[0].max(px[1]).max(px[2]))
        .max()
        .unwrap_or(0);
    // The clear color (0.2, 0, 0.4) alone yields ~2-3 distinct bytes and a
    // max channel of 102; a rendered lit cube adds variety and brighter
    // pixels.
    assert!(
        distinct.len() > 8,
        "HDR pipeline should draw content: distinct={} max_channel={max_channel}",
        distinct.len()
    );
    assert!(
        max_channel > 102,
        "lit cube must exceed clear color: max_channel={max_channel}"
    );
}
