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
use bevy::ecs::system::RunSystemOnce;
use bevy::math::primitives::{Capsule3d, Cone, Rectangle, Sphere, Torus};
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::renderer::RenderDevice;
use bevy::render::view::{ColorGrading, Hdr};
use bevy::window::{ExitCondition, WindowPlugin};
use scriptbots_core::{RenderQuality, RenderSettings, WorldState};
use std::sync::{LazyLock, Mutex};
use tracing::{debug, info, warn};

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
        let diff = g[0]
            .abs_diff(c[0])
            .max(g[1].abs_diff(c[1]))
            .max(g[2].abs_diff(c[2]));
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

/// Process-wide capture app. `bevy_render` keeps a process-global empty
/// bind group layout, so constructing a second render App in one process
/// panics; the harness therefore builds exactly one App (lazily) and
/// reconfigures it per scene via [`OffscreenCapture::run`].
static CAPTURE_APP: LazyLock<Mutex<Option<SendCaptureApp>>> = LazyLock::new(|| Mutex::new(None));

/// Send wrapper for the capture App. Bevy's `App` is `!Send` (it stores a
/// non-Send runner trait object and can hold non-Send resources).
///
/// SAFETY: the App is only ever accessed while holding `CAPTURE_APP`'s
/// mutex, so all access is exclusive and serialized; the App is never
/// cloned, never moved out of the static, and no reference to it escapes
/// the critical section (the session borrows it from the guard by
/// construction). Moving the App across threads at lock time therefore
/// cannot create aliasing or data races.
struct SendCaptureApp(App);

#[allow(unsafe_code)]
unsafe impl Send for SendCaptureApp {}

/// A headless, offscreen capture session on the process-wide render app.
///
/// Every [`Self::render`] call syncs the scene from the world state through
/// the exact `sync_world` system the interactive renderer uses, renders
/// into the offscreen target, and reads the pixels back. Sessions are
/// created by [`Self::run`], which fully resets the previous scene (all
/// entities despawned, registries reset, fresh render target) so scenes
/// cannot contaminate each other.
pub struct OffscreenCapture<'a> {
    app: &'a mut App,
    target: Handle<Image>,
    viewport: (u32, u32),
    effective: crate::EffectiveRenderSettings,
    corrupt: bool,
}

impl<'a> OffscreenCapture<'a> {
    /// Run `f` with a freshly configured capture session.
    ///
    /// The first call in the process builds the render app; every call
    /// reconfigures it for `config`: previous scene entities are wiped,
    /// registries reset, a new offscreen target of the requested viewport
    /// is created, and the scene rig is respawned.
    ///
    /// # Errors
    /// Returns an error when the viewport is out of range, no GPU adapter is
    /// available, or render-device initialization failed on first build. A
    /// missing GPU must surface here, never as a silently black frame.
    pub fn run<R>(
        config: &OffscreenCaptureConfig,
        f: impl FnOnce(&mut OffscreenCapture) -> Result<R>,
    ) -> Result<R> {
        let (width, height) = config.viewport;
        if width == 0 || height == 0 || width > 8192 || height > 8192 {
            return Err(anyhow!(
                "capture viewport {width}x{height} outside 1..=8192"
            ));
        }
        let mut guard = CAPTURE_APP.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = Some(SendCaptureApp(build_capture_app(config)?));
        }
        let SendCaptureApp(app) = guard.as_mut().expect("capture app present");
        let mut session = configure_session(app, config)?;
        f(&mut session)
    }

    /// Probed GPU info (always `Some` inside a session).
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
                .unwrap_or(
                    if cfg!(target_arch = "aarch64") && cfg!(target_os = "macos") {
                        "aarch64-apple-darwin"
                    } else if cfg!(target_os = "macos") {
                        "x86_64-apple-darwin"
                    } else if cfg!(target_os = "linux") && cfg!(target_arch = "aarch64") {
                        "aarch64-unknown-linux-gnu"
                    } else if cfg!(target_os = "linux") {
                        "x86_64-unknown-linux-gnu"
                    } else {
                        "windows-msvc"
                    },
                )
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
    pub fn render(
        &mut self,
        world: &WorldState,
        scene: &str,
        seed: u64,
        tick: u64,
    ) -> Result<CapturedFrame> {
        let snapshot = WorldSnapshot::from_world(world)
            .ok_or_else(|| anyhow!("world snapshot unavailable at tick {tick}"))?;
        self.app.world_mut().resource_mut::<SnapshotState>().latest = Some(snapshot);
        self.set_camera_active(true);
        // Frame 1: sync_world applies the snapshot; pipelines compile.
        self.app.update();
        // Frame 2: full draw of the settled scene (async pipeline
        // compilation can never leak a half-drawn frame into evidence).
        self.app.update();
        // Frames 3-4: attachment/ViewTarget preparation has frame latency
        // (prepare_assets runs after ManageViews on the first frame).
        self.app.update();
        self.app.update();
        {
            let world = self.app.world_mut();
            {
                let mut probe = world.query_filtered::<Entity, With<CaptureCamera>>();
                let entities: Vec<Entity> = probe.iter(world).collect();
                for entity in entities {
                    let has_camera_graph = world
                        .get::<bevy::render::camera::CameraRenderGraph>(entity)
                        .is_some();
                    let has_frustum = world
                        .get::<bevy::camera::primitives::Frustum>(entity)
                        .is_some();
                    let has_visible_entities = world
                        .get::<bevy::render::sync_world::RenderEntity>(entity)
                        .is_some();
                    debug!(
                        ?entity,
                        has_camera_graph,
                        has_frustum,
                        render_entity = has_visible_entities,
                        "capture camera component probe"
                    );
                }
            }
            let mut cameras = world.query_filtered::<&Camera, With<CaptureCamera>>();
            let mut camera_states = 0_u32;
            let mut any_active = false;
            let mut viewport: Option<(f32, f32)> = None;
            for camera in cameras.iter(world) {
                camera_states += 1;
                any_active |= camera.is_active;
                viewport = camera.logical_viewport_size().map(|v| (v.x, v.y));
            }
            let mut mesh_query = world.query::<&Mesh3d>();
            let mesh_count = mesh_query.iter(world).count();
            let mut light_query = world.query::<&DirectionalLight>();
            let light_count = light_query.iter(world).count();
            let (extracted_cameras, view_targets) = self
                .app
                .get_sub_app_mut(RenderApp)
                .map(|render_app| {
                    let render_world = render_app.world_mut();
                    let mut c = render_world.query::<&bevy::render::camera::ExtractedCamera>();
                    let camera_debug: Vec<String> = c
                        .iter(render_world)
                        .map(|camera| {
                            let attachment_present = camera
                                .target
                                .as_ref()
                                .and_then(|target| {
                                    render_world
                                        .get_resource::<bevy::render::view::ViewTargetAttachments>()
                                        .and_then(|attachments| attachments.get(target))
                                        .map(|_| ())
                                })
                                .is_some();
                            format!(
                                "physical_target_size={:?} target={:?} hdr={} attachment_present={}",
                                camera.physical_target_size, camera.target, camera.hdr,
                                attachment_present
                            )
                        })
                        .collect();
                    let extracted_count = camera_debug.len();
                    for line in &camera_debug {
                        debug!(%line, "extracted camera probe");
                    }
                    // prepare_view_targets requires the full tuple
                    // (ExtractedCamera, ExtractedView, CameraMainTextureUsages, Msaa);
                    // ExtractedView is crate-private, probe the public parts.
                    let with_usages_count = {
                        let mut with_usages = render_world.query::<(
                            &bevy::render::camera::ExtractedCamera,
                            &bevy::camera::CameraMainTextureUsages,
                        )>();
                        with_usages.iter(render_world).count()
                    };
                    let with_msaa_count = {
                        let mut with_msaa = render_world.query::<(
                            &bevy::render::camera::ExtractedCamera,
                            &bevy::render::view::Msaa,
                        )>();
                        with_msaa.iter(render_world).count()
                    };
                    debug!(
                        with_usages_count,
                        with_msaa_count,
                        "prepare_view_targets public-tuple probes"
                    );
                    let mut q = render_world.query::<&bevy::render::view::ViewTarget>();
                    (extracted_count, q.iter(render_world).count())
                })
                .unwrap_or((0, 0));
            debug!(
                camera_states,
                any_active,
                ?viewport,
                mesh_count,
                light_count,
                extracted_cameras,
                view_targets,
                "capture scene state before readback"
            );
        }
        // Manual texture readback (the official headless pattern): the
        // `Readback` component's completion depends on extraction/plugin
        // wiring, while this path drives the copy, submission, and device
        // poll directly and fails loudly at each step.
        let data = self.readback_target()?;
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

    /// Manual texture readback via the render sub-app: copy the target to a
    /// MAP_READ buffer, submit, poll the device until the map completes.
    /// Every step has a typed failure — a missing GPU image, a wedged
    /// device, or a map error can never masquerade as a black frame.
    fn readback_target(&mut self) -> Result<Vec<u8>> {
        use bevy::render::render_asset::RenderAssets;
        use bevy::render::renderer::RenderQueue;
        use bevy::render::texture::GpuImage;

        let render_app = self
            .app
            .get_sub_app_mut(RenderApp)
            .ok_or_else(|| anyhow!("render sub-app missing"))?;
        let world = render_app.world_mut();
        let device = world
            .get_resource::<RenderDevice>()
            .ok_or_else(|| anyhow!("render device unavailable"))?
            .clone();
        let queue = world
            .get_resource::<RenderQueue>()
            .ok_or_else(|| anyhow!("render queue unavailable"))?
            .clone();
        let gpu_images = world
            .get_resource::<RenderAssets<GpuImage>>()
            .ok_or_else(|| anyhow!("gpu image registry unavailable"))?;
        let gpu_image = gpu_images.get(&self.target).ok_or_else(|| {
            anyhow!("capture target was never prepared on the GPU (image extraction failed)")
        })?;
        // The target is hardcoded Rgba8UnormSrgb = 4 bytes/pixel.
        let bytes_per_row =
            RenderDevice::align_copy_bytes_per_row(gpu_image.size.width as usize * 4) as u32;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scriptbots_capture_readback"),
            size: u64::from(bytes_per_row) * u64::from(gpu_image.size.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scriptbots_capture_copy"),
        });
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
        device
            .poll(wgpu::PollType::Wait)
            .map_err(|error| anyhow!("device poll during readback: {error}"))?;
        rx.recv_timeout(std::time::Duration::from_secs(10))
            .map_err(|error| anyhow!("readback map timed out (wedged GPU?): {error}"))?
            .map_err(|error| anyhow!("readback buffer map failed: {error}"))?;
        let data = {
            let mapped = buffer.slice(..).get_mapped_range();
            Vec::from(&*mapped)
        };
        buffer.unmap();
        debug!(bytes = data.len(), "offscreen readback complete");
        Ok(data)
    }
}

/// Build the process-wide render app (once). No window, no winit (macOS
/// demands EventLoop creation on the main thread; captures run on
/// test/worker threads), fixed exposure (no AutoExposure) for determinism.
fn build_capture_app(config: &OffscreenCaptureConfig) -> Result<App> {
    let mut app = App::new();
    app.insert_resource(AmbientLight {
        color: Color::srgb(0.45, 0.52, 0.65),
        brightness: 800.0,
        affects_lightmapped_meshes: true,
    })
    .insert_resource(SnapshotState::default())
    .insert_resource(AgentRegistry::default())
    .insert_resource(AccessibilityState::new())
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
    )
    .add_systems(Update, sync_world);
    // finish/cleanup are required before manual update() pumping: the
    // render device initializes in RenderPlugin::finish. Pipelined rendering
    // is disabled above for exactly this reason: with the render app on its
    // own thread, `get_sub_app_mut(RenderApp)` returns None and the explicit
    // wgpu device poll the readback path needs never happens — the capture
    // wedges at "readback did not complete in budget" on Metal.
    app.finish();
    app.cleanup();
    info!(
        viewport = ?config.viewport,
        "process-wide offscreen capture app built"
    );
    Ok(app)
}

/// Reset the process app for a new scene: wipe every entity, reset the
/// registries, re-resolve the effective tier, create a fresh render target,
/// respawn the scene rig, and warm the pipelines.
fn configure_session<'a>(
    app: &'a mut App,
    config: &OffscreenCaptureConfig,
) -> Result<OffscreenCapture<'a>> {
    let effective = resolve_effective_render_settings(&config.render_settings);
    if effective.gpu.is_none() {
        return Err(anyhow!(
            "no GPU adapter available for offscreen capture (software lane requires \
             llvmpipe/lavapipe via WGPU_BACKEND, or a real GPU)"
        ));
    }
    // Wipe the previous SCENE completely (camera, lights, meshes, agents,
    // readbacks-in-flight) — but never infrastructure entities. A blanket
    // wipe of every entity kills plugin-owned startup entities the render
    // path depends on (the wiped app then produces no ViewTargets and every
    // capture reads back as the initial fill).
    let scene_entities: Vec<Entity> = {
        let world = app.world_mut();
        let mut found = Vec::new();
        let mut cameras = world.query_filtered::<Entity, With<Camera>>();
        found.extend(cameras.iter(world));
        let mut lights =
            world.query_filtered::<Entity, Or<(With<DirectionalLight>, With<PointLight>)>>();
        found.extend(lights.iter(world));
        let mut meshes = world.query_filtered::<Entity, With<Mesh3d>>();
        found.extend(meshes.iter(world));
        let mut readbacks =
            world.query_filtered::<Entity, With<bevy::render::gpu_readback::Readback>>();
        found.extend(readbacks.iter(world));
        found
    };
    for entity in scene_entities {
        let _ = app.world_mut().despawn(entity);
    }
    *app.world_mut().resource_mut::<SnapshotState>() = SnapshotState::default();
    *app.world_mut().resource_mut::<AgentRegistry>() = AgentRegistry::default();
    app.world_mut().resource_mut::<AmbientLight>().brightness =
        if config.corrupt { 0.0 } else { 800.0 };
    app.insert_resource(CaptureCorrupt(config.corrupt));
    app.insert_resource(effective.clone());

    // Fresh offscreen render target: sRGB RGBA8, render attachment +
    // copy-src for the readback path.
    let size = Extent3d {
        width: config.viewport.0,
        height: config.viewport.1,
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
    app.insert_resource(CaptureTarget(target.clone()));

    // Respawn the scene rig immediately (the same system a Startup schedule
    // would run, driven imperatively so every session gets a fresh rig).
    app.world_mut()
        .run_system_once(setup_capture_scene)
        .map_err(|error| anyhow!("capture scene setup: {error}"))?;
    // Warmup: two frames with an empty scene so base pipelines compile
    // outside the evidence path.
    app.update();
    app.update();
    info!(
        viewport = ?config.viewport,
        tier = ?effective.tier,
        corrupt = config.corrupt,
        "offscreen capture session configured"
    );
    Ok(OffscreenCapture {
        app,
        target,
        viewport: config.viewport,
        effective,
        corrupt: config.corrupt,
    })
}

/// Capture-scene setup: the world-relevant subset of the interactive
/// `setup_scene` (same meshes, chunk registry, lighting rig) with the camera
/// bound to the offscreen target and no HUD. Deliberate divergence: fixed
/// exposure (no AutoExposure) so captures are deterministic.
fn setup_capture_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
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
    // Valid 1x1 black CUBE textures for the reflection probes: the PBR
    // environment-map bind group expects Cube-dimension views, and the
    // default (empty) handle resolves to a 2D texture — which panics wgpu
    // validation ('expects dimension = Cube, but given a view with
    // dimension = D2') the moment a lit 3D scene renders.
    let mut cube = Image::new_fill(
        bevy::render::render_resource::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        bevy::render::render_resource::TextureDimension::D2,
        &[0, 0, 0, 255],
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    cube.texture_view_descriptor = Some(bevy::render::render_resource::TextureViewDescriptor {
        dimension: Some(bevy::render::render_resource::TextureViewDimension::Cube),
        ..Default::default()
    });
    let cube_handle = images.add(cube);
    commands.insert_resource(ReflectionProbeAssets {
        diffuse: cube_handle.clone(),
        specular: cube_handle,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height * 4) as usize]
    }

    fn seeded_world(seed: u64) -> scriptbots_core::WorldState {
        let mut world = scriptbots_core::WorldState::new(scriptbots_core::ScriptBotsConfig {
            rng_seed: Some(seed),
            ..scriptbots_core::ScriptBotsConfig::default()
        })
        .expect("world");
        for i in 0..4u32 {
            let mut agent = scriptbots_core::AgentData::default();
            agent.position.x = 100.0 + i as f32 * 60.0;
            agent.position.y = 100.0;
            agent.spike_length = 10.0;
            world.try_spawn_agent(agent).expect("spawn");
        }
        world.step().expect("step");
        world
    }

    /// The C4 alarm contract: a deliberately corrupted render (blackout sun +
    /// crushed exposure) MUST fail the golden comparison against the honest
    /// capture — if it passes, the harness cannot see a broken pipeline.
    #[test]
    fn corrupted_capture_fails_comparison_against_honest_frame() {
        if crate::probe_gpu_capability().is_none() {
            eprintln!("no GPU adapter; skipping live corruption alarm test");
            return;
        }
        let world = seeded_world(0xA14A2);
        let honest_config = OffscreenCaptureConfig {
            viewport: (160, 120),
            render_settings: RenderSettings::default(),
            corrupt: false,
        };
        let corrupt_config = OffscreenCaptureConfig {
            corrupt: true,
            ..honest_config.clone()
        };
        let honest = OffscreenCapture::run(&honest_config, |session| {
            session.render(&world, "alarm-honest", 42, 1)
        })
        .expect("honest capture");
        let corrupted = OffscreenCapture::run(&corrupt_config, |session| {
            session.render(&world, "alarm-corrupt", 42, 1)
        })
        .expect("corrupted capture");
        let stats = compare_frames(
            &honest.rgba8,
            &corrupted.rgba8,
            honest.width,
            honest.height,
            &CompareThresholds::default(),
        )
        .expect("comparison runs");
        assert!(
            !stats.pass,
            "the corrupted frame MUST fail the golden comparison (alarm fires): {stats:?}"
        );
        // And the honest pipeline is not monochrome: a dead pipeline would
        // also 'pass' a black-golden comparison, so assert content variance.
        let mut min = [255u8; 3];
        let mut max = [0u8; 3];
        for px in honest.rgba8.chunks_exact(4) {
            for channel in 0..3 {
                min[channel] = min[channel].min(px[channel]);
                max[channel] = max[channel].max(px[channel]);
            }
        }
        let spread: u32 = (0..3).map(|c| u32::from(max[c] - min[c])).sum();
        assert!(
            spread > 24,
            "honest capture must contain visual variance (live pipeline), got {spread}"
        );
    }

    #[test]
    fn compare_identical_frames_passes() {
        let golden = frame(64, 64, 128);
        let stats = compare_frames(
            &golden,
            &golden.clone(),
            64,
            64,
            &CompareThresholds::default(),
        )
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
            assert!(
                json.get(field).is_some(),
                "missing provenance field {field}"
            );
        }
        assert_eq!(json["schema"], PROVENANCE_SCHEMA);
    }
}
