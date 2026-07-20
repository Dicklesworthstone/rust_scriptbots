//! Offscreen live-render capture + golden evidence machinery (bd-2z0.14.3.4).
//!
//! The plan's evidence taxonomy (9.4 / bd-2z0.1.3) distinguishes a
//! SemanticProjection (CPU reference image) from an OffscreenLiveRenderer:
//! the REAL Bevy render graph (terrain chunks, agent parts, lighting, HDR
//! tonemapping) rendered into an offscreen texture and read back. Until this
//! module existed the only "goldens" were CPU circle rasterizers that could
//! ship a dead GPU pipeline green. Everything here is adapter-honest: the
//! provenance block records exactly which adapter/backend produced a frame,
//! and golden comparison is a per-lane decision (software adapters in CI,
//! DSR-blessed goldens for real GPUs).
//!
//! Determinism contract: fixed exposure (no AutoExposure), fixed camera
//! poses, fixed viewport, one warmup frame so pipeline compilation never
//! leaks into captures. Two captures of the same world state on the same
//! adapter are byte-identical (asserted in tests). Cross-adapter byte
//! identity is NOT claimed — that is what the provenance labels are for.

use anyhow::{Context, Result, anyhow};
use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::camera::prelude::*;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::math::primitives::{Capsule3d, Cone, Rectangle, Sphere, Torus};
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::view::{ColorGrading, Hdr};
use bevy::window::{ExitCondition, WindowPlugin};
use scriptbots_core::{RenderQuality, RenderSettings, WorldState};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

use crate::{
    AccessibilityState, AgentMeshes, AgentRegistry, ReflectionProbeAssets, SnapshotState,
    TerrainChunkRegistry, WorldSnapshot, resolve_effective_render_settings, sync_world,
};

// ---------------------------------------------------------------------------
// Provenance + comparison (pure, GPU-free).
// ---------------------------------------------------------------------------

/// Schema tag for provenance manifests (`scriptbots.capture-provenance.v1`).
pub const PROVENANCE_SCHEMA: &str = "scriptbots.capture-provenance.v1";

/// Full provenance for one captured frame: everything needed to decide
/// whether a golden comparison is same-class and to reproduce the frame.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CaptureProvenance {
    /// Schema tag (`scriptbots.capture-provenance.v1`).
    pub schema: String,
    /// Scene name from the manifest.
    pub scene: String,
    /// World RNG seed.
    pub seed: u64,
    /// Tick captured.
    pub tick: u64,
    /// Frontend identifier (`bevy_offscreen`).
    pub frontend: String,
    /// Adapter name from the GPU probe (`unknown` when probing failed).
    pub adapter_name: String,
    /// Backend (Metal/Vulkan/D3D12/GL) or `unknown`.
    pub backend: String,
    /// Device class (DiscreteGpu/IntegratedGpu/Cpu/...) or `unknown`.
    pub device_type: String,
    /// Effective quality tier the capture rendered at.
    pub quality_tier: String,
    /// Viewport `[width, height]` in pixels.
    pub viewport: [u32; 2],
    /// Color encoding of the PNG bytes (`rgba8-srgb`).
    pub colorspace: String,
    /// `rustc -vV` first line (toolchain identity).
    pub rustc_version: String,
    /// Compile target triple.
    pub target_triple: String,
    /// Whether the alarm-test corruption mode was active.
    pub corrupt: bool,
}

/// One captured frame with its provenance.
#[derive(Debug, Clone)]
pub struct CapturedFrame {
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Tightly packed RGBA8 pixels (row-major, no row alignment padding).
    pub rgba8: Vec<u8>,
    /// Provenance block.
    pub provenance: CaptureProvenance,
}

/// Golden comparison thresholds. Defaults encode "same adapter, same
/// pipeline": sub-ULP raster jitter and timestamp-query nondeterminism must
/// not fail, but any real visual change must.
#[derive(Debug, Clone, Copy)]
pub struct CompareThresholds {
    /// A pixel counts as differing when any channel deviates by more.
    pub differing_channel: u8,
    /// Maximum allowed ratio of differing pixels.
    pub max_differing_ratio: f32,
    /// Maximum allowed mean absolute channel deviation over all pixels.
    pub max_mean_abs_diff: f32,
}

impl Default for CompareThresholds {
    fn default() -> Self {
        Self {
            differing_channel: 8,
            max_differing_ratio: 0.005,
            max_mean_abs_diff: 0.5,
        }
    }
}

/// Statistics from one frame comparison.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffStats {
    /// Pixels whose channel deviation exceeded the threshold.
    pub differing_pixels: u64,
    /// Total pixels.
    pub total_pixels: u64,
    /// differing / total.
    pub differing_ratio: f64,
    /// Largest single-channel absolute deviation.
    pub max_channel_diff: u8,
    /// Mean absolute channel deviation over all channels of all pixels.
    pub mean_abs_diff: f64,
    /// Whether the comparison passed the thresholds.
    pub pass: bool,
}

/// Compare two tightly packed RGBA8 buffers of the same dimensions.
///
/// # Errors
/// Returns an error when the buffers do not both contain exactly
/// `width * height * 4` bytes (a fail-closed shape mismatch, never a pass).
pub fn compare_frames(
    golden: &[u8],
    candidate: &[u8],
    width: u32,
    height: u32,
    thresholds: &CompareThresholds,
) -> Result<DiffStats> {
    let expected = width as usize * height as usize * 4;
    if golden.len() != expected || candidate.len() != expected {
        return Err(anyhow!(
            "frame shape mismatch: expected {expected} bytes ({width}x{height} RGBA8), \
             golden {} bytes, candidate {} bytes",
            golden.len(),
            candidate.len()
        ));
    }
    let mut differing_pixels = 0_u64;
    let mut max_channel_diff = 0_u8;
    let mut abs_sum = 0_u64;
    for (g, c) in golden.iter().zip(candidate.iter()) {
        let diff = g.abs_diff(*c);
        max_channel_diff = max_channel_diff.max(diff);
        abs_sum += u64::from(diff);
        if diff > thresholds.differing_channel {
            differing_pixels += 1;
        }
    }
    // Channel diffs over-count pixels by 4x; divide back out.
    let differing_pixels = differing_pixels / 4;
    let total_pixels = u64::from(width) * u64::from(height);
    let differing_ratio = differing_pixels as f64 / total_pixels as f64;
    let mean_abs_diff = abs_sum as f64 / (total_pixels * 4) as f64;
    let pass = differing_ratio <= f64::from(thresholds.max_differing_ratio)
        && mean_abs_diff <= f64::from(thresholds.max_mean_abs_diff);
    Ok(DiffStats {
        differing_pixels,
        total_pixels,
        differing_ratio,
        max_channel_diff,
        mean_abs_diff,
        pass,
    })
}

/// Render the per-pixel absolute difference of two same-shape frames as an
/// RGBA8 heatmap (green = identical, red = max deviation), for artifact
/// triage on golden mismatches.
///
/// # Errors
/// Same shape contract as [`compare_frames`].
pub fn diff_heatmap(golden: &[u8], candidate: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let expected = width as usize * height as usize * 4;
    if golden.len() != expected || candidate.len() != expected {
        return Err(anyhow!(
            "heatmap shape mismatch: expected {expected} bytes, got {} + {}",
            golden.len(),
            candidate.len()
        ));
    }
    let mut out = Vec::with_capacity(expected);
    for px in golden.chunks_exact(4).zip(candidate.chunks_exact(4)) {
        let (g, c) = (px.0, px.1);
        let diff = g[0].abs_diff(c[0]).max(g[1].abs_diff(c[1])).max(g[2].abs_diff(c[2]));
        out.push(diff);
        out.push(255 - diff);
        out.push(0);
        out.push(255);
    }
    Ok(out)
}

/// Undo the 256-byte row-pitch alignment wgpu applies to texture readbacks.
/// Returns the input unchanged when the pitch is already tight.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn unpad_readback(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let tight_pitch = width as usize * 4;
    let aligned_pitch = tight_pitch.div_ceil(256) * 256;
    let mut out = Vec::with_capacity(tight_pitch * height as usize);
    for row in 0..height as usize {
        let start = row * aligned_pitch;
        let end = start + tight_pitch;
        if end > data.len() {
            break;
        }
        out.extend_from_slice(&data[start..end]);
    }
    out
}

/// Encode tightly packed RGBA8 pixels as PNG bytes.
///
/// # Errors
/// Returns an error on shape mismatch or encoder failure.
pub fn encode_png(width: u32, height: u32, rgba8: &[u8]) -> Result<Vec<u8>> {
    let expected = width as usize * height as usize * 4;
    if rgba8.len() != expected {
        return Err(anyhow!(
            "png encode shape mismatch: expected {expected} bytes, got {}",
            rgba8.len()
        ));
    }
    let mut out = std::io::Cursor::new(Vec::new());
    let buffer = image::RgbaImage::from_raw(width, height, rgba8.to_vec())
        .ok_or_else(|| anyhow!("png encode: buffer rejected"))?;
    image::DynamicImage::ImageRgba8(buffer)
        .write_to(&mut out, image::ImageFormat::Png)
        .context("png encode failed")?;
    Ok(out.into_inner())
}

/// Decode a PNG into `(width, height, tightly packed RGBA8)`.
///
/// # Errors
/// Returns an error on malformed input.
pub fn decode_png(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)
        .context("png decode failed")?
        .to_rgba8();
    let (width, height) = image.dimensions();
    Ok((width, height, image.into_raw()))
}

// ---------------------------------------------------------------------------
// Offscreen live renderer.
// ---------------------------------------------------------------------------

/// Configuration for one offscreen capture context.
#[derive(Debug, Clone)]
pub struct OffscreenCaptureConfig {
    /// Viewport `(width, height)` in pixels.
    pub viewport: (u32, u32),
    /// Render settings (quality tier resolved against the probed adapter).
    pub render_settings: RenderSettings,
    /// Alarm-test corruption: black out the sun and ambient light so the
    /// frame provably differs from any honest capture.
    pub corrupt: bool,
}

/// The offscreen image target handle shared with the scene setup.
#[derive(Resource, Clone)]
struct CaptureTarget(Handle<Image>);

/// Marker for the capture camera.
#[derive(Component)]
struct CaptureCamera;

/// Corruption flag resource (alarm tests).
#[derive(Resource, Clone, Copy)]
struct CaptureCorrupt(bool);

/// A headless, offscreen instance of the real Bevy render pipeline.
///
/// The app runs with no window; every [`Self::render`] call syncs the scene
/// from the world state through the exact `sync_world` system the
/// interactive renderer uses, renders one frame into the offscreen target,
/// and reads the pixels back. Construction compiles pipelines eagerly (one
/// warmup update) so the first real capture is representative.
pub struct OffscreenCapture {
    app: App,
    target: Handle<Image>,
    viewport: (u32, u32),
    effective: crate::EffectiveRenderSettings,
    corrupt: bool,
}

impl OffscreenCapture {
    /// Build the headless render app and compile its pipelines.
    ///
    /// # Errors
    /// Returns an error when plugin/render-device initialization fails
    /// (no adapter, driver failure). A missing GPU must surface here, never
    /// as a silently black frame.
    pub fn new(config: &OffscreenCaptureConfig) -> Result<Self> {
        let (width, height) = config.viewport;
        if width == 0 || height == 0 || width > 8192 || height > 8192 {
            return Err(anyhow!(
                "capture viewport {width}x{height} outside 1..=8192"
            ));
        }
        let effective = resolve_effective_render_settings(&config.render_settings);
        if effective.gpu.is_none() {
            return Err(anyhow!(
                "no GPU adapter available for offscreen capture (software lane requires \
                 llvmpipe/lavapipe via WGPU_BACKEND, or a real GPU)"
            ));
        }

        let mut app = App::new();
        app.insert_resource(AmbientLight {
            color: Color::srgb(0.45, 0.52, 0.65),
            brightness: if config.corrupt { 0.0 } else { 800.0 },
            affects_lightmapped_meshes: true,
        })
        .insert_resource(SnapshotState::default())
        .insert_resource(AgentRegistry::default())
        .insert_resource(AccessibilityState::new())
        .insert_resource(CaptureCorrupt(config.corrupt))
        .insert_resource(effective.clone())
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: None,
            exit_condition: ExitCondition::DontExit,
            close_when_requested: false,
            ..Default::default()
        }))
        .add_systems(Startup, setup_capture_scene)
        .add_systems(Update, sync_world);

        // Offscreen render target: sRGB RGBA8, render attachment + copy-src
        // for the readback path.
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let mut image = Image::new_fill(
            size,
            TextureDimension::D2,
            &[0, 0, 0, 255],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        image.texture_descriptor.usage = TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_SRC
            | TextureUsages::TEXTURE_BINDING;
        let target = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(image);
        app.insert_resource(CaptureTarget(target.clone()));

        // finish/cleanup are required before manual update() pumping: the
        // render device initializes in RenderPlugin::finish.
        app.finish();
        app.cleanup();
        // Warmup: run startup + one frame so pipeline compilation is out of
        // the capture path.
        app.update();
        info!(
            viewport = ?config.viewport,
            tier = ?effective.tier,
            corrupt = config.corrupt,
            "offscreen capture context ready"
        );
        Ok(Self {
            app,
            target,
            viewport: (width, height),
            effective,
            corrupt: config.corrupt,
        })
    }

    /// Probed GPU info (always `Some` after a successful [`Self::new`]).
    #[must_use]
    pub fn gpu_info(&self) -> Option<&scriptbots_core::GpuInfo> {
        self.effective.gpu.as_ref()
    }

    /// Effective quality tier the capture renders at.
    #[must_use]
    pub const fn tier(&self) -> RenderQuality {
        self.effective.tier
    }

    /// Pose the capture camera (Bevy space: world X-Z is the ground plane,
    /// +Y up). `look_at` orients the camera; `fov_deg` is vertical FOV.
    pub fn set_camera_pose(&mut self, pos: [f32; 3], look_at: [f32; 3], fov_deg: f32) {
        let mut cameras = self
            .app
            .world_mut()
            .query_filtered::<(&mut Transform, &mut Projection), With<CaptureCamera>>();
        for (mut transform, mut projection) in cameras.iter_mut(self.app.world_mut()) {
            *transform = Transform::from_xyz(pos[0], pos[1], pos[2])
                .looking_at(Vec3::new(look_at[0], look_at[1], look_at[2]), Vec3::Y);
            if let Projection::Perspective(perspective) = projection.as_mut() {
                perspective.fov = fov_deg.to_radians();
            }
        }
    }

    /// Provenance block for one capture (pixel hash lives in the scene log).
    #[must_use]
    pub fn provenance(&self, scene: &str, seed: u64, tick: u64) -> CaptureProvenance {
        let gpu = self.effective.gpu.as_ref();
        CaptureProvenance {
            schema: PROVENANCE_SCHEMA.to_string(),
            scene: scene.to_string(),
            seed,
            tick,
            frontend: "bevy_offscreen".to_string(),
            adapter_name: gpu.map_or_else(|| "unknown".to_string(), |g| g.name.clone()),
            backend: gpu.map_or_else(|| "unknown".to_string(), |g| g.backend.clone()),
            device_type: gpu.map_or_else(|| "unknown".to_string(), |g| format!("{:?}", g.class)),
            quality_tier: format!("{:?}", self.effective.tier),
            viewport: [self.viewport.0, self.viewport.1],
            colorspace: "rgba8-srgb".to_string(),
            rustc_version: option_env!("SCRIPTBOTS_RUSTC_VERSION")
                .or(option_env!("CARGO_PKG_RUST_VERSION"))
                .unwrap_or("unknown")
                .to_string(),
            target_triple: option_env!("SCRIPTBOTS_TARGET_TRIPLE")
                .unwrap_or(if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
                    "aarch64-apple-darwin"
                } else if cfg!(target_os = "macos") {
                    "x86_64-apple-darwin"
                } else if cfg!(target_os = "linux") && cfg!(target_arch = "aarch64") {
                    "aarch64-unknown-linux-gnu"
                } else if cfg!(target_os = "linux") {
                    "x86_64-unknown-linux-gnu"
                } else {
                    "windows-msvc"
                })
            .to_string(),
            corrupt: self.corrupt,
        }
    }

    /// Render one frame from the current world state and read it back.
    ///
    /// The camera stays inactive between calls, so non-capture ticks cost no
    /// GPU time; the readback pumps the app until the GPU copy lands (a few
    /// frames), then detaches so no per-frame readback traffic persists.
    ///
    /// # Errors
    /// Returns an error when the world snapshot cannot be built or the
    /// readback does not complete within the pump budget (a wedged GPU).
    pub fn render(&mut self, world: &WorldState, scene: &str, seed: u64, tick: u64) -> Result<CapturedFrame> {
        let snapshot = WorldSnapshot::from_world(world)
            .ok_or_else(|| anyhow!("world snapshot unavailable at tick {tick}"))?;
        self.app
            .world_mut()
            .resource_mut::<SnapshotState>()
            .latest = Some(snapshot);
        self.set_camera_active(true);
        // Frame 1: sync_world applies the snapshot, render graph draws it.
        self.app.update();
        // Readback: spawn, pump until the GPU copy lands, then detach.
        let slot: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
        let slot_in_observer = Arc::clone(&slot);
        let readback_entity = self
            .app
            .world_mut()
            .spawn(Readback::texture(self.target.clone()))
            .observe(move |event: On<ReadbackComplete>, mut commands: Commands| {
                if let Ok(mut guard) = slot_in_observer.lock() {
                    *guard = Some(event.data.clone());
                }
                commands.entity(event.entity).despawn();
            })
            .id();
        let mut pumps = 0_u32;
        let data = loop {
            self.app.update();
            pumps += 1;
            if let Ok(mut guard) = slot.lock()
                && let Some(data) = guard.take()
            {
                break data;
            }
            if pumps >= 16 {
                warn!(pumps, "offscreen readback did not complete in budget");
                // The observer may already have despawned the entity.
                if let Ok(mut entity) = self.app.world_mut().get_entity_mut(readback_entity) {
                    entity.despawn();
                }
                self.set_camera_active(false);
                return Err(anyhow!(
                    "offscreen readback wedged after {pumps} frames (GPU driver stall?)"
                ));
            }
        };
        self.set_camera_active(false);
        let (width, height) = self.viewport;
        let rgba8 = unpad_readback(&data, width, height);
        let expected = width as usize * height as usize * 4;
        if rgba8.len() != expected {
            return Err(anyhow!(
                "readback shape drift: expected {expected} bytes after unpad, got {}",
                rgba8.len()
            ));
        }
        Ok(CapturedFrame {
            width,
            height,
            rgba8,
            provenance: self.provenance(scene, seed, tick),
        })
    }

    fn set_camera_active(&mut self, active: bool) {
        let mut cameras = self
            .app
            .world_mut()
            .query_filtered::<&mut Camera, With<CaptureCamera>>();
        for mut camera in cameras.iter_mut(self.app.world_mut()) {
            camera.is_active = active;
        }
    }
}

/// Capture-scene setup: the world-relevant subset of the interactive
/// `setup_scene` (same meshes, chunk registry, lighting rig) with the camera
/// bound to the offscreen target and no HUD. Deliberate divergence: fixed
/// exposure (no AutoExposure) so captures are deterministic.
fn setup_capture_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    target: Res<CaptureTarget>,
    corrupt: Res<CaptureCorrupt>,
    effective: Res<crate::EffectiveRenderSettings>,
) {
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(0.03, 0.05, 0.09)),
            target: RenderTarget::Image(target.0.clone().into()),
            is_active: false,
            ..default()
        },
        Transform::from_xyz(0.0, 1800.0, 1400.0).looking_at(Vec3::ZERO, Vec3::Y),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        Tonemapping::AcesFitted,
        ColorGrading::default(),
        Hdr,
        CaptureCamera,
    ));

    let light_transform =
        Transform::from_xyz(-1200.0, 1800.0, 900.0).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((
        DirectionalLight {
            illuminance: if corrupt.0 { 0.0 } else { 9000.0 },
            shadows_enabled: effective.features.shadows,
            ..default()
        },
        light_transform,
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
    ));

    commands.insert_resource(AgentMeshes {
        base_radius: 1.0,
        body: meshes.add(Mesh::from(Capsule3d::new(0.5, 1.6))),
        wheel: meshes.add(Mesh::from(Torus::new(0.3, 0.6))),
        spike: meshes.add(Mesh::from(Cone {
            radius: 0.45,
            height: 1.0,
        })),
        sphere: meshes.add(Mesh::from(Sphere::new(0.5))),
        quad: meshes.add(Mesh::from(Rectangle::new(1.0, 1.0))),
        ring: meshes.add(Mesh::from(Torus::new(0.7, 1.0))),
    });
    commands.insert_resource(TerrainChunkRegistry {
        chunk_size: crate::TERRAIN_CHUNK_SIZE,
        height_scale: crate::TERRAIN_HEIGHT_SCALE,
        ..default()
    });
    commands.insert_resource(ReflectionProbeAssets {
        diffuse: Handle::default(),
        specular: Handle::default(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height * 4) as usize]
    }

    #[test]
    fn compare_identical_frames_passes() {
        let golden = frame(64, 64, 128);
        let stats = compare_frames(&golden, &golden.clone(), 64, 64, &CompareThresholds::default())
            .expect("compare");
        assert!(stats.pass);
        assert_eq!(stats.differing_pixels, 0);
        assert_eq!(stats.max_channel_diff, 0);
    }

    #[test]
    fn compare_sub_threshold_jitter_passes_but_real_change_fails() {
        let golden = frame(64, 64, 100);
        // Jitter within the differing_channel bound on a few pixels: pass.
        let mut jittered = golden.clone();
        for px in jittered.chunks_exact_mut(4).take(4) {
            px[0] += 3;
        }
        let stats = compare_frames(&golden, &jittered, 64, 64, &CompareThresholds::default())
            .expect("compare jitter");
        assert!(stats.pass, "jitter should pass: {stats:?}");
        // A real change: every pixel shifts hard. Fail.
        let shifted = frame(64, 64, 180);
        let stats = compare_frames(&golden, &shifted, 64, 64, &CompareThresholds::default())
            .expect("compare shifted");
        assert!(!stats.pass, "a real visual change must fail: {stats:?}");
        assert_eq!(stats.max_channel_diff, 80);
        assert!(stats.differing_ratio > 0.99);
    }

    #[test]
    fn compare_shape_mismatch_fails_closed() {
        let golden = frame(64, 64, 0);
        let short = frame(32, 32, 0);
        assert!(compare_frames(&golden, &short, 64, 64, &CompareThresholds::default()).is_err());
        assert!(compare_frames(&golden, &golden, 32, 32, &CompareThresholds::default()).is_err());
    }

    #[test]
    fn diff_heatmap_marks_deviations() {
        let golden = frame(8, 8, 0);
        let mut candidate = golden.clone();
        candidate[0] = 255; // one hot pixel
        let heat = diff_heatmap(&golden, &candidate, 8, 8).expect("heatmap");
        assert_eq!(&heat[0..4], &[255, 0, 0, 255], "hot pixel is red");
        assert_eq!(&heat[4..8], &[0, 255, 0, 255], "clean pixel is green");
    }

    #[test]
    fn unpad_readback_recovers_tight_rows() {
        // Width 3 => tight pitch 12 => aligned pitch 256.
        let (width, height) = (3_u32, 2_u32);
        let mut padded = vec![0_u8; 256 * 2];
        for row in 0..height as usize {
            for col in 0..12 {
                padded[row * 256 + col] = (row * 12 + col) as u8;
            }
        }
        let tight = unpad_readback(&padded, width, height);
        assert_eq!(tight.len(), (width * height * 4) as usize);
        assert_eq!(tight[0], 0);
        assert_eq!(tight[11], 11);
        assert_eq!(tight[12], 12, "second row starts at its tight offset");
        // Already-tight width (64*4 = 256) passes through unchanged.
        let aligned = frame(64, 4, 7);
        assert_eq!(unpad_readback(&aligned, 64, 4), aligned);
    }

    #[test]
    fn provenance_serializes_with_required_fields() {
        let provenance = CaptureProvenance {
            schema: PROVENANCE_SCHEMA.to_string(),
            scene: "unit".to_string(),
            seed: 42,
            tick: 12,
            frontend: "bevy_offscreen".to_string(),
            adapter_name: "test-adapter".to_string(),
            backend: "Metal".to_string(),
            device_type: "Integrated".to_string(),
            quality_tier: "Medium".to_string(),
            viewport: [256, 256],
            colorspace: "rgba8-srgb".to_string(),
            rustc_version: "rustc test".to_string(),
            target_triple: "aarch64-apple-darwin".to_string(),
            corrupt: false,
        };
        let json = serde_json::to_value(&provenance).expect("serialize");
        for field in [
            "schema",
            "scene",
            "seed",
            "tick",
            "frontend",
            "adapter_name",
            "backend",
            "device_type",
            "quality_tier",
            "viewport",
            "colorspace",
            "rustc_version",
            "target_triple",
            "corrupt",
        ] {
            assert!(json.get(field).is_some(), "missing provenance field {field}");
        }
        assert_eq!(json["schema"], PROVENANCE_SCHEMA);
    }
}
