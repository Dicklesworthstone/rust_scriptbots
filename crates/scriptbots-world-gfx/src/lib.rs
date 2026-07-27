#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use scriptbots_core::{NUM_EYES, RenderTonemapMode, visual};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
#[cfg(feature = "perf_counters")]
use std::time::Instant;

/// Animation time step per simulation tick, in seconds.
///
/// Shader animation derives from the world tick through this constant — never from a
/// wall clock — so identical tick sequences produce identical frames and captures
/// (bd-2z0.7.11). The value preserves the legacy ~60 ticks-per-second visual rate.
pub const ANIM_SECONDS_PER_TICK: f32 = 1.0 / 60.0;

pub mod sense_wgsl;

/// Public snapshot format the renderer expects. Keep minimal; the app will adapt
/// its internal world snapshot to this view before passing to the renderer.
#[derive(Clone, Debug)]
pub struct WorldSnapshot<'a> {
    pub world_size: (f32, f32),
    pub terrain: TerrainView<'a>,
    pub agents: &'a [AgentInstance],
    /// Animation clock for this frame, in seconds; callers derive it from the world
    /// tick (`tick * ANIM_SECONDS_PER_TICK`) or a recorded sequence so captures are
    /// deterministic.
    pub anim_seconds: f32,
    /// The selected tonemap control from `RenderSettings`, when any. `None` defers to
    /// the renderer's environment-driven default; `Some` makes the chosen curve drive
    /// the shader every frame rather than sitting unconsumed (bd-2z0.7.11).
    pub tonemap_mode: Option<RenderTonemapMode>,
}

#[derive(Clone, Debug)]
pub struct TerrainView<'a> {
    pub dims: (u32, u32),
    pub cell_size: u32,
    pub tiles: &'a [u32], // index into a tileset palette/atlas (kept simple for MVP)
    /// Final natural/accessibility-mapped sRGB terrain colors from the core
    /// visual authority. The GPU backend decodes these to linear before
    /// writing its sRGB attachment; it must not invent a second biome palette
    /// or shading model.
    pub colors: &'a [[f32; 4]],
    pub elevation: Option<&'a [f32]>, // optional elevation field for slope accents
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct AgentInstance {
    pub position: [f32; 2],
    pub quad_extent: [f32; 2],
    pub heading: [f32; 2],
    pub body_radius: f32,
    pub body_half_length: f32,
    pub wheel_offset: f32,
    pub wheel_radius: f32,
    pub mouth_open: f32,
    pub herbivore_tendency: f32,
    pub temperature_preference: f32,
    pub food_delta: f32,
    pub sound_level: f32,
    pub sound_output: f32,
    pub wheel_left: f32,
    pub wheel_right: f32,
    pub spike_length: f32,
    pub trait_smell: f32,
    pub trait_sound: f32,
    pub trait_hearing: f32,
    pub trait_eye: f32,
    pub trait_blood: f32,
    pub selection: f32, // 0=None, 1=Hovered, 2=Selected/Focused
    /// Authoritative semantic sRGB body color. The GPU staging boundary decodes
    /// RGB to linear exactly once before the sRGB render target encodes output.
    pub color: [f32; 4],
    /// Authoritative semantic sRGB mouth color, resolved by
    /// `scriptbots_core::visual` and carried here rather than recomputed.
    ///
    /// Packed as a RESOLVED colour rather than as its inputs (bd-rl1h). Core
    /// derives `mouth_color` by mixing the death and combat event colours by
    /// `mouth_activity`, and that activity needs `sound_multiplier`, which this
    /// instance never carried. Uploading the multiplier would let the shader
    /// re-implement core's expression and drift from it; uploading the answer
    /// moves the authority boundary instead, which is what the body `color`
    /// above already does.
    pub mouth_color: [f32; 3],
    pub glow: f32,  // 0..1 extra glow (e.g., reproduction/spike)
    pub boost: f32, // 0..1 boost intensity
    pub spiked: f32,
    pub eye_dirs: [f32; NUM_EYES],
    pub eye_fov: [f32; NUM_EYES],
}

pub struct WorldRenderer {
    renderer_id: u64,
    frame_generation: u64,
    device: wgpu::Device,
    device_fault: DeviceFaultMonitor,
    queue: wgpu::Queue,
    size: (u32, u32),
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    readback: ReadbackRing,
    terrain: TerrainPipeline,
    agents: AgentPipeline,
    view: ViewUniforms,
    cam_scale: f32,
    cam_offset: (f32, f32),
    /// Last tick-derived animation time consumed by `render`; reused by `resize` so the
    /// shader clock stays monotonic across viewport changes without a wall clock.
    last_anim_seconds: f32,
    post: Option<PostFx>,
    /// Whether the post pass actually ran during the most recent `render()`.
    /// `post` staying `Some` only means the resources exist; env flags can
    /// disable the pass at runtime, and readback must not copy a stale target.
    post_ran: bool,
    #[cfg(feature = "perf_counters")]
    last_render_ms: f32,
    #[cfg(feature = "perf_counters")]
    last_readback_ms: f32,
}

pub struct RenderFrame {
    pub extent: (u32, u32),
    renderer_id: u64,
    generation: u64,
}

static NEXT_RENDERER_ID: AtomicU64 = AtomicU64::new(1);
const WORLD_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

impl WorldRenderer {
    pub async fn new(adapter: &wgpu::Adapter, size: (u32, u32)) -> Result<Self, ReadbackError> {
        // Reject zero-sized viewports instead of clamping them (bd-2z0.7.11): early window
        // init races must surface as typed errors, never as silently 1x1 renderers.
        if size.0 == 0 || size.1 == 0 {
            return Err(ReadbackError::ZeroDimensions {
                width: size.0,
                height: size.1,
            });
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|error| {
                ReadbackError::Device(format!("wgpu device request failed: {error}"))
            })?;
        let device_fault = DeviceFaultMonitor::install(&device);

        let format = WORLD_COLOR_FORMAT;
        let initialized = scoped_gpu_result(&device, "renderer initialization", || {
            let readback = ReadbackRing::new(&device, size, format)?;
            let (color, color_view) = create_color(&device, format, size)?;
            let view = ViewUniforms::new(&device, &queue, size);
            let mut terrain = TerrainPipeline::new(&device, format, &view);
            terrain.init_atlas(&device, &queue)?;
            let agents = AgentPipeline::new(&device, format, &view);
            Ok((readback, color, color_view, view, terrain, agents))
        });
        device_fault.check()?;
        let (readback, color, color_view, view, terrain, agents) = initialized?;
        let renderer_id = NEXT_RENDERER_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| ReadbackError::Device("renderer identity space exhausted".to_owned()))?;

        Ok(Self {
            renderer_id,
            frame_generation: 0,
            device,
            device_fault,
            queue,
            size,
            color,
            color_view,
            format,
            readback,
            terrain,
            agents,
            view,
            cam_scale: 1.0,
            cam_offset: (0.0, 0.0),
            last_anim_seconds: 0.0,
            post: None,
            post_ran: false,
            #[cfg(feature = "perf_counters")]
            last_render_ms: 0.0,
            #[cfg(feature = "perf_counters")]
            last_readback_ms: 0.0,
        })
    }

    pub fn resize(&mut self, new_size: (u32, u32)) -> Result<(), ReadbackError> {
        self.device_fault.check()?;
        if new_size == self.size {
            return Ok(());
        }
        if new_size.0 == 0 || new_size.1 == 0 {
            return Err(ReadbackError::ZeroDimensions {
                width: new_size.0,
                height: new_size.1,
            });
        }
        let next_generation = self
            .frame_generation
            .checked_add(1)
            .ok_or_else(|| ReadbackError::Device("render generation overflow".to_owned()))?;
        let device = self.device.clone();
        let format = self.format;
        let needs_post_target = self.post.is_some();
        let queue = self.queue.clone();
        let last_anim_seconds = self.last_anim_seconds;
        let cam_scale = self.cam_scale;
        let cam_offset = self.cam_offset;
        let view_uniforms = &self.view;
        // Prepare every allocation under typed wgpu error scopes before replacing live
        // renderer resources. Validation and OOM failures therefore preserve the prior
        // color/readback/post targets, view uniform, size, and generation.
        let prepared = scoped_gpu_result(&device, "renderer resize", || {
            let readback = ReadbackRing::new(&device, new_size, format)?;
            let (tex, view) = create_color(&device, format, new_size)?;
            let post_target = needs_post_target
                .then(|| create_color(&device, format, new_size))
                .transpose()?;
            let prepared_view = view_uniforms.prepare_resize(
                &device,
                &queue,
                new_size,
                last_anim_seconds,
                cam_scale,
                cam_offset,
            );
            Ok((readback, tex, view, post_target, prepared_view))
        });
        self.device_fault.check()?;
        let (readback, tex, view, post_target, prepared_view) = prepared?;

        self.color = tex;
        self.color_view = view;
        self.readback = readback;
        self.size = new_size;
        self.view.install_resize(prepared_view);
        if let (Some(post), Some((target, target_view))) = (self.post.as_mut(), post_target) {
            post.install_resize(format, target, target_view);
        }
        self.frame_generation = next_generation;
        self.device_fault.check()?;
        Ok(())
    }

    pub fn set_camera(&mut self, scale: f32, offset: (f32, f32)) {
        self.cam_scale = scale;
        self.cam_offset = offset;
    }

    pub fn render(&mut self, snapshot: &WorldSnapshot) -> Result<RenderFrame, ReadbackError> {
        self.device_fault.check()?;
        let device = self.device.clone();
        let result = scoped_gpu_result(&device, "frame render", || self.render_scoped(snapshot));
        if let Err(error) = &result
            && error.is_terminal_gpu_fault()
        {
            self.device_fault.record(error.clone());
        }
        self.device_fault.check()?;
        result
    }

    fn render_scoped(&mut self, snapshot: &WorldSnapshot) -> Result<RenderFrame, ReadbackError> {
        validate_snapshot(snapshot)?;
        if !self.cam_scale.is_finite()
            || self.cam_scale <= 0.0
            || !self.cam_offset.0.is_finite()
            || !self.cam_offset.1.is_finite()
        {
            return Err(ReadbackError::MetadataMismatch {
                expected: "finite positive camera scale and finite offset".to_owned(),
                actual: format!("scale={} offset={:?}", self.cam_scale, self.cam_offset),
            });
        }
        #[cfg(feature = "perf_counters")]
        let t0 = Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("world.render"),
            });
        // Ensure view uniforms match current viewport, tick-derived time, and camera
        self.last_anim_seconds = snapshot.anim_seconds;
        self.view.update(
            &self.queue,
            self.size,
            snapshot.anim_seconds,
            self.cam_scale,
            self.cam_offset,
        );
        // Background clear
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world.clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if env_flag("SB_WGPU_DEBUG_BRIGHT_BG") {
                            wgpu::Color {
                                r: 0.10,
                                g: 0.10,
                                b: 0.25,
                                a: 1.0,
                            }
                        } else {
                            wgpu::Color {
                                r: 0.03,
                                g: 0.06,
                                b: 0.12,
                                a: 1.0,
                            }
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }
        // Terrain + agents
        let vis_tiles = self.terrain.encode(
            &self.device,
            &self.queue,
            &mut encoder,
            &self.color_view,
            &self.view,
            snapshot,
            self.size,
            self.cam_scale,
            self.cam_offset,
        );
        let vis_agents = self.agents.encode(
            &self.device,
            &self.queue,
            &mut encoder,
            &self.color_view,
            &self.view,
            snapshot,
            self.size,
            self.cam_scale,
            self.cam_offset,
        );
        tracing::info!(
            tiles = vis_tiles,
            agents = vis_agents,
            "wgpu visible instances"
        );
        // Freeze presentation controls once for this frame. Reading the
        // environment independently in `ensure_post` and `PostFx::run` made
        // one frozen snapshot depend on wall-clock cache refreshes.
        let post_controls = PostControls::from_env()?;
        let selected_tonemap = snapshot.tonemap_mode.map(|mode| match mode {
            RenderTonemapMode::Aces | RenderTonemapMode::Tony => 1_u32,
            RenderTonemapMode::Agx => 3_u32,
        });

        // Post‑FX (ACES + vignette; FXAA stub): color_view → post.target
        self.post_ran = false;
        if self.ensure_post(&post_controls, selected_tonemap)?
            && let Some(p) = self.post.as_mut()
        {
            // The selected RenderSettings tonemap control wins over the environment
            // default so the chosen curve is actually consumed (bd-2z0.7.11).
            p.run(
                &self.device,
                &self.queue,
                &mut encoder,
                &self.color_view,
                self.size,
                selected_tonemap,
                &post_controls,
            );
            self.post_ran = true;
        }
        self.queue.submit(Some(encoder.finish()));
        #[cfg(feature = "perf_counters")]
        {
            self.last_render_ms = t0.elapsed().as_secs_f32() * 1000.0;
        }
        self.frame_generation = self
            .frame_generation
            .checked_add(1)
            .ok_or_else(|| ReadbackError::Device("render generation overflow".to_owned()))?;
        Ok(RenderFrame {
            extent: self.size,
            renderer_id: self.renderer_id,
            generation: self.frame_generation,
        })
    }

    pub fn copy_to_readback(&mut self, frame: &RenderFrame) -> Result<(), ReadbackError> {
        self.device_fault.check()?;
        let device = self.device.clone();
        let result = scoped_gpu_result(&device, "readback copy", || {
            self.copy_to_readback_scoped(frame)
        });
        if let Err(error) = &result
            && error.is_terminal_gpu_fault()
        {
            self.device_fault.record(error.clone());
        }
        self.device_fault.check()?;
        result
    }

    fn copy_to_readback_scoped(&mut self, frame: &RenderFrame) -> Result<(), ReadbackError> {
        validate_frame_token(frame, self.size, self.renderer_id, self.frame_generation)?;
        #[cfg(feature = "perf_counters")]
        let t0 = Instant::now();
        let src_tex: &wgpu::Texture = match self.post.as_ref() {
            Some(post) if self.post_ran => &post.target,
            _ => &self.color,
        };
        self.readback
            .copy(&self.device, &self.queue, src_tex)
            .map(|_| {
                #[cfg(feature = "perf_counters")]
                {
                    self.last_readback_ms = t0.elapsed().as_secs_f32() * 1000.0;
                }
            })
    }

    pub fn mapped_rgba(&mut self) -> Result<ReadbackView, ReadbackError> {
        self.device_fault.check()?;
        self.readback.mapped()
    }

    #[cfg(feature = "perf_counters")]
    pub fn last_timings_ms(&self) -> (f32, f32) {
        (self.last_render_ms, self.last_readback_ms)
    }

    fn ensure_post(
        &mut self,
        controls: &PostControls,
        selected_tonemap: Option<u32>,
    ) -> Result<bool, ReadbackError> {
        if !controls.wants_post(selected_tonemap) {
            return Ok(false);
        }
        if self.post.is_none() {
            self.post = Some(PostFx::new(
                &self.device,
                self.format,
                &self.color_view,
                self.size,
            )?);
        }
        Ok(true)
    }
}

fn validate_frame_token(
    frame: &RenderFrame,
    renderer_extent: (u32, u32),
    renderer_id: u64,
    generation: u64,
) -> Result<(), ReadbackError> {
    if frame.extent != renderer_extent {
        return Err(ReadbackError::MetadataMismatch {
            expected: format!("{}x{} render extent", renderer_extent.0, renderer_extent.1),
            actual: format!("{}x{} render extent", frame.extent.0, frame.extent.1),
        });
    }
    if frame.renderer_id != renderer_id || frame.generation != generation {
        return Err(ReadbackError::MetadataMismatch {
            expected: format!("renderer {renderer_id} generation {generation}"),
            actual: format!(
                "renderer {} generation {}",
                frame.renderer_id, frame.generation
            ),
        });
    }
    Ok(())
}

fn validate_snapshot(snapshot: &WorldSnapshot<'_>) -> Result<(), ReadbackError> {
    let (width, height) = snapshot.terrain.dims;
    let expected = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| ReadbackError::MetadataMismatch {
            expected: "terrain dimensions with a representable cell count".to_owned(),
            actual: format!("{width}x{height}"),
        })?;
    if snapshot.terrain.tiles.len() != expected {
        return Err(ReadbackError::MetadataMismatch {
            expected: format!("{expected} terrain kind values"),
            actual: format!("{} terrain kind values", snapshot.terrain.tiles.len()),
        });
    }
    if snapshot.terrain.colors.len() != expected {
        return Err(ReadbackError::MetadataMismatch {
            expected: format!("{expected} authoritative terrain colors"),
            actual: format!("{} terrain colors", snapshot.terrain.colors.len()),
        });
    }
    if let Some(elevation) = snapshot.terrain.elevation
        && elevation.len() != expected
    {
        return Err(ReadbackError::MetadataMismatch {
            expected: format!("{expected} terrain elevations"),
            actual: format!("{} terrain elevations", elevation.len()),
        });
    }
    if !snapshot.world_size.0.is_finite()
        || !snapshot.world_size.1.is_finite()
        || snapshot.world_size.0 <= 0.0
        || snapshot.world_size.1 <= 0.0
    {
        return Err(ReadbackError::MetadataMismatch {
            expected: "finite positive world dimensions".to_owned(),
            actual: format!("{:?}", snapshot.world_size),
        });
    }
    if !snapshot.anim_seconds.is_finite() {
        return Err(ReadbackError::MetadataMismatch {
            expected: "finite tick-derived animation time".to_owned(),
            actual: snapshot.anim_seconds.to_string(),
        });
    }
    if let Some((index, color)) = snapshot
        .terrain
        .colors
        .iter()
        .enumerate()
        .find(|(_, color)| color.iter().any(|channel| !channel.is_finite()))
    {
        return Err(ReadbackError::MetadataMismatch {
            expected: "finite authoritative terrain colors".to_owned(),
            actual: format!("non-finite color at terrain cell {index}: {color:?}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod capture_smoke_test {
    use super::*;

    /// Acquire an adapter, or report that this host cannot supply live GPU
    /// evidence and let the caller skip.
    ///
    /// These tests used to `.expect("adapter")`, so on a host with no usable
    /// backend they went RED rather than skipping — observed directly on rch
    /// worker hz1, which reports `active_backends: Backends(0x0)` while hz2
    /// offers llvmpipe. A test that fails because the machine has no GPU is
    /// indistinguishable, in CI output, from a test that fails because the
    /// renderer broke, and the first kind teaches everyone to ignore the
    /// second. `scriptbots-bevy` already skips this way via
    /// `live_gpu_evidence_available`; world-gfx had no equivalent.
    ///
    /// Skipping is honest here precisely BECAUSE these are live-GPU claims: a
    /// host without an adapter has not proven the claim false, it has failed to
    /// test it, and the difference belongs in the log rather than in the result.
    fn live_adapter(label: &str) -> Option<wgpu::Adapter> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })) {
            Ok(adapter) => Some(adapter),
            Err(error) => {
                eprintln!(
                    "skipping {label}: this host exposes no usable wgpu adapter ({error:?}); \
                     live GPU evidence is unavailable, not disproven"
                );
                None
            }
        }
    }

    // This executes the real offscreen wgpu pipeline and blocking GPU readback.
    // It is not a GPUI window, Bevy render graph, or on-screen presentation test.
    #[test]
    fn wgpu_offscreen_gpu_framebuffer_readback_is_populated() {
        let Some(adapter) = live_adapter("wgpu offscreen readback smoke") else {
            return;
        };
        let size = (640, 360);
        let mut renderer =
            pollster::block_on(WorldRenderer::new(&adapter, size)).expect("renderer");
        let dims = (120u32, 60u32);
        let tiles = vec![3u32; (dims.0 * dims.1) as usize];
        let colors = vec![[0.15, 0.45, 0.2, 1.0]; tiles.len()];
        let snapshot = WorldSnapshot {
            world_size: (6000.0, 3000.0),
            terrain: TerrainView {
                dims,
                cell_size: 50,
                tiles: &tiles,
                colors: &colors,
                elevation: None,
            },
            agents: &[],
            anim_seconds: 0.0,
            tonemap_mode: None,
        };
        let frame = renderer.render(&snapshot).expect("valid render snapshot");
        renderer
            .copy_to_readback(&frame)
            .expect("real wgpu offscreen framebuffer copy");
        let view = renderer
            .mapped_rgba()
            .expect("real wgpu framebuffer readback must map instead of passing ceremonially");
        assert_eq!((view.width, view.height), size);
        assert!(view.bytes_per_row >= view.width * 4);
        let bytes = view.bytes();
        assert_eq!(
            bytes.len(),
            view.bytes_per_row as usize * view.height as usize
        );
        assert!(
            bytes.iter().any(|byte| *byte != 0),
            "real wgpu framebuffer readback must contain rendered color data"
        );
        let first = bytes.to_vec();
        drop(view);

        let second_frame = renderer
            .render(&snapshot)
            .expect("same valid snapshot must render twice");
        renderer
            .copy_to_readback(&second_frame)
            .expect("second framebuffer copy");
        let second = renderer
            .mapped_rgba()
            .expect("second framebuffer readback")
            .bytes()
            .to_vec();
        assert_eq!(
            first, second,
            "same snapshot, controls, adapter, and tick-derived animation time must be byte-identical"
        );
    }

    /// Deterministic ACROSS RENDERER INSTANCES, not merely across two draws.
    ///
    /// The test above renders twice through the SAME `WorldRenderer`, so it
    /// shares one device, one pipeline set, one readback ring and one set of
    /// GPU allocations. That proves a repeated draw is stable; it cannot
    /// distinguish "the pipeline is deterministic" from "the second draw
    /// happened to observe the state the first draw left behind". Anything
    /// that depends on residual buffer contents, allocation addresses or
    /// first-use initialisation is invisible to it and would only appear on a
    /// fresh process — which is where a golden comparison actually lives.
    ///
    /// This builds two INDEPENDENT renderers from the same adapter, each with
    /// its own device, pipelines and readback ring, and requires their frames
    /// to be byte-identical. That is as close to run-to-run as an in-process
    /// test can get.
    ///
    /// HONEST LIMITS, so this is not read as more than it is: two devices from
    /// one adapter in one process is not a second process, and it says nothing
    /// about a DIFFERENT adapter, driver or backend — cross-backend agreement
    /// is a separate open remainder of bd-2z0.7.11, and cross-platform live
    /// evidence needs the DSR lanes this machine does not have.
    #[test]
    fn independent_renderers_produce_byte_identical_frames() {
        let Some(adapter) = live_adapter("independent-renderer determinism") else {
            return;
        };
        let size = (320, 180);
        let dims = (64u32, 32u32);
        let tiles = vec![3u32; (dims.0 * dims.1) as usize];
        let colors = vec![[0.15, 0.45, 0.2, 1.0]; tiles.len()];
        let snapshot = WorldSnapshot {
            world_size: (3200.0, 1600.0),
            terrain: TerrainView {
                dims,
                cell_size: 50,
                tiles: &tiles,
                colors: &colors,
                elevation: None,
            },
            agents: &[],
            anim_seconds: 0.0,
            tonemap_mode: None,
        };

        // Each closure owns its renderer for its whole lifetime, so the second
        // cannot observe anything the first allocated or left mapped.
        let render_once = |label: &str| -> Vec<u8> {
            let mut renderer = pollster::block_on(WorldRenderer::new(&adapter, size))
                .unwrap_or_else(|error| panic!("{label} renderer must initialise: {error:?}"));
            let frame = renderer
                .render(&snapshot)
                .unwrap_or_else(|error| panic!("{label} render: {error:?}"));
            renderer
                .copy_to_readback(&frame)
                .unwrap_or_else(|error| panic!("{label} copy: {error:?}"));
            renderer
                .mapped_rgba()
                .unwrap_or_else(|error| panic!("{label} readback: {error:?}"))
                .bytes()
                .to_vec()
        };

        let first = render_once("first");
        let second = render_once("second");

        assert!(
            first.iter().any(|byte| *byte != 0),
            "a determinism claim over two blank frames would be vacuous"
        );
        assert_eq!(
            first.len(),
            second.len(),
            "independent renderers must agree on readback shape"
        );
        assert_eq!(
            first, second,
            "the same snapshot rendered by two independently constructed renderers must be \
             byte-identical; a difference here means the GPU path carries state across \
             construction and no golden can be trusted run to run"
        );
    }
}

fn create_color(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    size: (u32, u32),
) -> Result<(wgpu::Texture, wgpu::TextureView), ReadbackError> {
    // Defensive clamp to ensure valid texture extent
    let size = (size.0.max(1), size.1.max(1));
    scoped_gpu_value(device, "color-target allocation", || {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("world.color"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    })
}

// ---------------- Readback ring (triple-buffered) ----------------

struct ReadbackRing {
    slots: [ReadbackSlot; 3],
    curr: usize,
    bytes_per_row: u32,
    extent: (u32, u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReadbackLayout {
    bytes_per_row: u32,
    size_bytes: u64,
}

/// Typed failures for the adapter/device/readback/capture surface (bd-2z0.7.11).
/// No failure on this surface may be reported as an empty vector or a silent success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadbackError {
    /// No adapter satisfied the request.
    AdapterUnavailable,
    /// Device-level internal or host-side poll failure.
    Device(String),
    /// A wgpu code/data contract was rejected without losing the device.
    Validation {
        /// Public operation whose GPU contract was rejected.
        operation: String,
        /// Backend-provided validation detail.
        detail: String,
    },
    /// The device-lost callback reported a terminal GPU device loss.
    DeviceLost {
        /// Backend classification for the loss.
        reason: wgpu::DeviceLostReason,
        /// Optional backend detail.
        detail: String,
    },
    /// A scoped GPU operation exhausted device memory.
    OutOfMemory {
        /// Public operation whose allocation failed.
        operation: String,
    },
    /// Unsupported resize extent or readback layout.
    Resize(String),
    /// Buffer map request failed.
    Map(String),
    /// Mapping did not complete within the bounded wait.
    Timeout,
    /// Capture produced no frame content at all.
    Empty,
    /// Capture produced a frame with no nonzero pixels.
    Blank,
    /// Observed metadata disagrees with the required contract.
    MetadataMismatch { expected: String, actual: String },
    /// Zero-sized render or capture extent was rejected instead of clamped.
    ZeroDimensions { width: u32, height: u32 },
    /// Invalid renderer control or environment configuration.
    Configuration(String),
    /// Explicit capture artifact could not be encoded or persisted.
    Artifact(String),
}

impl std::fmt::Display for ReadbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AdapterUnavailable => write!(f, "no GPU adapter satisfied the request"),
            Self::Device(detail) => write!(f, "GPU device failure: {detail}"),
            Self::Validation { operation, detail } => {
                write!(f, "GPU validation rejected {operation}: {detail}")
            }
            Self::DeviceLost { reason, detail } if detail.is_empty() => {
                write!(f, "GPU device lost: {reason:?}")
            }
            Self::DeviceLost { reason, detail } => {
                write!(f, "GPU device lost ({reason:?}): {detail}")
            }
            Self::OutOfMemory { operation } => {
                write!(f, "GPU out of memory during {operation}")
            }
            Self::Resize(detail) => write!(f, "GPU resize failure: {detail}"),
            Self::Map(detail) => write!(f, "GPU buffer map failure: {detail}"),
            Self::Timeout => write!(f, "GPU readback did not map within the bounded wait"),
            Self::Empty => write!(f, "capture produced no frame content"),
            Self::Blank => write!(f, "capture produced a frame with no nonzero pixels"),
            Self::MetadataMismatch { expected, actual } => {
                write!(
                    f,
                    "capture metadata mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ZeroDimensions { width, height } => {
                write!(
                    f,
                    "zero-sized render/capture extent {width}x{height} is rejected"
                )
            }
            Self::Configuration(detail) => write!(f, "GPU renderer configuration error: {detail}"),
            Self::Artifact(detail) => write!(f, "GPU capture artifact failure: {detail}"),
        }
    }
}

impl std::error::Error for ReadbackError {}

impl ReadbackError {
    fn is_terminal_gpu_fault(&self) -> bool {
        matches!(
            self,
            Self::Device(_) | Self::DeviceLost { .. } | Self::OutOfMemory { .. }
        )
    }
}

#[derive(Clone, Default)]
struct DeviceFaultMonitor {
    first_fault: Arc<Mutex<Option<ReadbackError>>>,
}

impl DeviceFaultMonitor {
    fn install(device: &wgpu::Device) -> Self {
        let monitor = Self::default();

        let lost_monitor = monitor.clone();
        device.set_device_lost_callback(move |reason, detail| {
            lost_monitor.record(ReadbackError::DeviceLost { reason, detail });
        });

        let uncaptured_monitor = monitor.clone();
        device.on_uncaptured_error(Arc::new(move |error| {
            uncaptured_monitor.record(readback_error_from_wgpu(
                "uncaptured renderer operation",
                error,
            ));
        }));

        monitor
    }

    fn record(&self, error: ReadbackError) {
        if let Ok(mut first_fault) = self.first_fault.lock() {
            let definitive_device_loss = matches!(error, ReadbackError::DeviceLost { .. });
            let prior_is_device_loss =
                matches!(first_fault.as_ref(), Some(ReadbackError::DeviceLost { .. }));
            if first_fault.is_none() || definitive_device_loss && !prior_is_device_loss {
                *first_fault = Some(error);
            }
        }
    }

    fn check(&self) -> Result<(), ReadbackError> {
        let mut first_fault = self
            .first_fault
            .lock()
            .map_err(|_| ReadbackError::Device("GPU fault monitor lock was poisoned".to_owned()))?;
        let Some(error) = first_fault.as_ref() else {
            return Ok(());
        };
        if error.is_terminal_gpu_fault() {
            return Err(error.clone());
        }
        match first_fault.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

fn readback_error_from_wgpu(operation: &str, error: wgpu::Error) -> ReadbackError {
    match error {
        wgpu::Error::OutOfMemory { .. } => ReadbackError::OutOfMemory {
            operation: operation.to_owned(),
        },
        wgpu::Error::Validation { description, .. } => ReadbackError::Validation {
            operation: operation.to_owned(),
            detail: description,
        },
        wgpu::Error::Internal { description, .. } => {
            ReadbackError::Device(format!("{operation} failed internally: {description}"))
        }
    }
}

fn scoped_gpu_result<T>(
    device: &wgpu::Device,
    operation: &str,
    perform: impl FnOnce() -> Result<T, ReadbackError>,
) -> Result<T, ReadbackError> {
    // One filter per scope is required by wgpu. Push all three so allocation and
    // implementation failures become values instead of reaching the default
    // uncaptured-error panic handler.
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    device.push_error_scope(wgpu::ErrorFilter::Internal);
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

    let result = perform();

    // Native wgpu-core resolves these futures immediately. Drain every scope,
    // even after finding one error, so no stale scope captures a later frame.
    let out_of_memory = pollster::block_on(device.pop_error_scope());
    let internal = pollster::block_on(device.pop_error_scope());
    let validation = pollster::block_on(device.pop_error_scope());
    if let Some(error) = out_of_memory.or(internal).or(validation) {
        return Err(readback_error_from_wgpu(operation, error));
    }

    result
}

fn scoped_gpu_value<T>(
    device: &wgpu::Device,
    operation: &str,
    perform: impl FnOnce() -> T,
) -> Result<T, ReadbackError> {
    scoped_gpu_result(device, operation, || Ok(perform()))
}

struct ReadbackSlot {
    buf: wgpu::Buffer,
    ready: bool,
    mapped: std::sync::Arc<AtomicBool>,
    /// Last asynchronous map failure reported for this slot; surfaced by `mapped`.
    map_error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

pub struct ReadbackView {
    pub guard: wgpu::BufferView,
    pub bytes_per_row: u32,
    pub width: u32,
    pub height: u32,
}

impl ReadbackView {
    pub fn bytes(&self) -> &[u8] {
        &self.guard
    }
}

fn validate_readback_layout(
    extent: (u32, u32),
    limits: &wgpu::Limits,
) -> Result<ReadbackLayout, ReadbackError> {
    if extent.0 == 0 || extent.1 == 0 {
        return Err(ReadbackError::ZeroDimensions {
            width: extent.0,
            height: extent.1,
        });
    }
    if extent.0 > limits.max_texture_dimension_2d || extent.1 > limits.max_texture_dimension_2d {
        return Err(ReadbackError::Resize(format!(
            "requested extent {}x{} exceeds device max_texture_dimension_2d {}",
            extent.0, extent.1, limits.max_texture_dimension_2d
        )));
    }

    let unpadded_bytes_per_row = u64::from(extent.0) * 4;
    let alignment = u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let bytes_per_row_u64 = unpadded_bytes_per_row
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| {
            ReadbackError::Resize(format!(
                "RGBA8 row-pitch alignment overflow for extent {}x{}",
                extent.0, extent.1
            ))
        })?;
    let bytes_per_row = u32::try_from(bytes_per_row_u64).map_err(|_| {
        ReadbackError::Resize(format!(
            "aligned RGBA8 row pitch {bytes_per_row_u64} for extent {}x{} exceeds the u32 copy-layout limit",
            extent.0, extent.1
        ))
    })?;
    let size_bytes = bytes_per_row_u64
        .checked_mul(u64::from(extent.1))
        .ok_or_else(|| {
            ReadbackError::Resize(format!(
                "readback buffer size overflow for extent {}x{}",
                extent.0, extent.1
            ))
        })?;
    if size_bytes > limits.max_buffer_size {
        return Err(ReadbackError::Resize(format!(
            "readback buffer for extent {}x{} requires {size_bytes} bytes, exceeding device max_buffer_size {}",
            extent.0, extent.1, limits.max_buffer_size
        )));
    }

    Ok(ReadbackLayout {
        bytes_per_row,
        size_bytes,
    })
}

/// Resolve a texture-to-buffer copy layout's non-zero dimensions, or say which
/// contract broke (bd-2z0.7.11, bd-aqy8).
///
/// `copy_texture_to_buffer` requires a non-zero row pitch and row count. Both
/// call sites previously wrote `NonZeroU32::new(..).unwrap()`, so a zero
/// PANICKED from inside GPU work — the readback path, which this bead family
/// requires to be fallible, and `init_atlas`, which runs during initialisation.
/// In both cases an upstream validator was supposed to make zero impossible,
/// and that is exactly why the failure was invisible: an invariant enforced in
/// one place and asserted by panic in another reports a bug as a crash with no
/// attribution.
///
/// Module-scoped on purpose. `ReadbackRing` and `TerrainPipeline` both need it,
/// and giving each its own copy would recreate the duplicated-invariant problem
/// this bead family keeps having to undo — one validator, two callers. It is
/// also pure, so it is exercised without a GPU adapter, which matters because
/// this crate has none on the remote lane.
fn copy_layout_dimensions(
    bytes_per_row: u32,
    extent: (u32, u32),
) -> Result<(std::num::NonZeroU32, std::num::NonZeroU32), ReadbackError> {
    let (width, height) = extent;
    let rows =
        std::num::NonZeroU32::new(height).ok_or(ReadbackError::ZeroDimensions { width, height })?;
    let pitch = std::num::NonZeroU32::new(bytes_per_row).ok_or_else(|| {
        ReadbackError::Resize(format!(
            "copy row pitch is zero for a {width}x{height} extent; the layout was \
             admitted without a copyable row size"
        ))
    })?;
    Ok((pitch, rows))
}

impl ReadbackRing {
    fn new(
        device: &wgpu::Device,
        extent: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> Result<Self, ReadbackError> {
        if format != wgpu::TextureFormat::Rgba8UnormSrgb {
            return Err(ReadbackError::MetadataMismatch {
                expected: "Rgba8UnormSrgb readback format".to_owned(),
                actual: format!("{format:?}"),
            });
        }
        let layout = validate_readback_layout(extent, &device.limits())?;
        let slots = scoped_gpu_value(device, "readback-ring allocation", || {
            let mk = || {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("world.readback"),
                    size: layout.size_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            };
            let mk_slot = || ReadbackSlot {
                buf: mk(),
                ready: false,
                mapped: Arc::new(AtomicBool::new(false)),
                map_error: Arc::new(Mutex::new(None)),
            };
            [mk_slot(), mk_slot(), mk_slot()]
        })?;
        Ok(Self {
            slots,
            curr: 0,
            bytes_per_row: layout.bytes_per_row,
            extent,
        })
    }

    fn copy(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color: &wgpu::Texture,
    ) -> Result<(), ReadbackError> {
        // Validated BEFORE any slot state is disturbed, so a rejected copy
        // leaves the ring exactly as it was rather than half-reset (bd-2z0.7.11).
        let (bytes_per_row, rows_per_image) =
            copy_layout_dimensions(self.bytes_per_row, self.extent)?;

        let slot = &mut self.slots[self.curr];
        slot.ready = false;
        if slot.mapped.load(Ordering::Relaxed) {
            slot.buf.unmap();
            slot.mapped.store(false, Ordering::Relaxed);
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("world.readback.copy"),
        });
        encoder.copy_texture_to_buffer(
            color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row.get()),
                    rows_per_image: Some(rows_per_image.get()),
                },
            },
            wgpu::Extent3d {
                width: self.extent.0,
                height: self.extent.1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        // Map asynchronously; record any failure so `mapped` can surface it instead of
        // the caller spinning forever on a slot that will never become ready.
        let slice = slot.buf.slice(..);
        let mapped_flag = std::sync::Arc::clone(&slot.mapped);
        let map_error = std::sync::Arc::clone(&slot.map_error);
        slice.map_async(wgpu::MapMode::Read, move |res| match res {
            Ok(()) => mapped_flag.store(true, Ordering::Relaxed),
            Err(error) => {
                if let Ok(mut slot_error) = map_error.lock() {
                    *slot_error = Some(error.to_string());
                }
            }
        });
        // Bounded wait for the map to land: an unbounded wait would let a wedged device
        // stall a capture — or the opt-in diagnostic presentation — forever (bd-2z0.7.11).
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(2_000)),
            })
            .map_err(|error| match error {
                wgpu::PollError::Timeout => ReadbackError::Timeout,
                other => ReadbackError::Device(format!("readback poll failed: {other:?}")),
            })?;
        // Advance ring pointer
        self.curr = (self.curr + 1) % self.slots.len();
        Ok(())
    }

    fn mapped(&mut self) -> Result<ReadbackView, ReadbackError> {
        // Surface any recorded asynchronous map failure before scanning for readiness.
        for slot in &self.slots {
            if let Ok(mut slot_error) = slot.map_error.lock()
                && let Some(error) = slot_error.take()
            {
                return Err(ReadbackError::Map(error));
            }
        }
        // Prefer the most recently mapped slot (scan last -> older)
        for i in 0..self.slots.len() {
            let idx = (self.curr + self.slots.len() - 1 - i) % self.slots.len();
            let slot = &mut self.slots[idx];
            if !slot.ready && !slot.mapped.load(Ordering::Relaxed) {
                continue;
            }
            let slice = slot.buf.slice(..);
            let guard = slice.get_mapped_range();
            slot.ready = true; // latch until consumer takes a view at least once
            return Ok(ReadbackView {
                guard,
                bytes_per_row: self.bytes_per_row,
                width: self.extent.0,
                height: self.extent.1,
            });
        }
        Err(ReadbackError::Empty)
    }
}

/// Parse the `SB_WGPU_FXAA` toggle truthfully: FXAA is not implemented in the post
/// pipeline, so any nonzero request logs a one-time warning and resolves to disabled
/// rather than silently claiming the feature (bd-2z0.7.11).
fn parse_fxaa_env(previous: u32) -> Result<u32, ReadbackError> {
    static WARNED: AtomicBool = AtomicBool::new(false);
    match std::env::var("SB_WGPU_FXAA") {
        Err(std::env::VarError::NotPresent) => Ok(previous),
        Err(error) => Err(ReadbackError::Configuration(format!(
            "SB_WGPU_FXAA is not valid Unicode: {error}"
        ))),
        Ok(value) if parse_toggle_value(&value) == Some(false) => Ok(0),
        Ok(value) if parse_toggle_value(&value) == Some(true) => {
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "SB_WGPU_FXAA requests FXAA, but FXAA is not implemented in the post \
                     pipeline; the request is ignored"
                );
            }
            Ok(0)
        }
        Ok(value) => Err(ReadbackError::Configuration(format!(
            "SB_WGPU_FXAA={value:?}; expected a boolean toggle"
        ))),
    }
}

// ---------------- View uniforms (viewport size) ----------------

#[cfg(test)]
mod tests {
    use super::{
        AgentInstance, DeviceFaultMonitor, PostControls, ReadbackError, ReadbackLayout,
        RenderFrame, TerrainView, WorldSnapshot, parse_control_f32, parse_toggle,
        readback_error_from_wgpu, resolve_bloom_intensity, scoped_gpu_value, terrain_atlas_palette,
        validate_frame_token, validate_readback_layout, validate_snapshot,
    };
    use bytemuck::Zeroable;
    use scriptbots_core::{
        AccessibilityPalette, TerrainKind,
        visual::{self, SplatInput, TerrainSurfaceInput},
    };

    /// The atlas upload layout goes through the SAME validator as readback.
    ///
    /// `atlas_w`/`atlas_h` are derived (`grid_cols * tile_w`,
    /// `grid_rows * tile_h`), so a misconfigured tile size yields a zero that
    /// used to panic during GPU initialisation. This pins the shape of the
    /// atlas call — pitch is `atlas_w * 4` for RGBA8 — against the one shared
    /// decision, so the two callers cannot drift apart into separate rules.
    #[test]
    fn a_degenerate_atlas_extent_is_typed_rather_than_panicking() {
        // A zero tile size collapses the whole atlas.
        // Written as a literal zero rather than `0 * 4`: clippy::erasing_op
        // rejects the multiplication, and the point is the resulting pitch.
        let error = super::copy_layout_dimensions(0, (0, 64))
            .expect_err("a zero-width atlas cannot be uploaded");
        assert!(
            matches!(error, ReadbackError::ZeroDimensions { .. })
                || matches!(error, ReadbackError::Resize(_)),
            "a degenerate atlas must be typed, got {error:?}"
        );

        // Zero rows: the grid had no rows to bake.
        let rows_error = super::copy_layout_dimensions(256 * 4, (256, 0))
            .expect_err("a zero-height atlas cannot be uploaded");
        match rows_error {
            ReadbackError::ZeroDimensions { width, height } => {
                assert_eq!((width, height), (256, 0));
            }
            other => panic!("expected ZeroDimensions, got {other:?}"),
        }

        // The ordinary 3x2 atlas shape still resolves.
        let (pitch, atlas_rows) = super::copy_layout_dimensions(384 * 4, (384, 256))
            .expect("a real atlas extent must upload");
        assert_eq!(pitch.get(), 384 * 4, "RGBA8 pitch is four bytes per texel");
        assert_eq!(atlas_rows.get(), 256);
    }

    /// A zero row count must be reported, not asserted by panic.
    ///
    /// This path previously read `NonZeroU32::new(self.extent.1).unwrap()`
    /// inside the readback copy. A ring that reached it with a zero height
    /// aborted the process from within the GPU capture path — the exact place
    /// this bead requires a typed, attributable error.
    #[test]
    fn a_zero_readback_height_is_typed_rather_than_panicking() {
        let error = super::copy_layout_dimensions(256, (640, 0))
            .expect_err("a zero height cannot produce a copyable layout");
        match error {
            ReadbackError::ZeroDimensions { width, height } => {
                assert_eq!(
                    (width, height),
                    (640, 0),
                    "the refusal must name the extent"
                );
            }
            other => panic!("expected ZeroDimensions, got {other:?}"),
        }
    }

    /// A zero row pitch is a layout contract failure, and the message has to say
    /// so — `Resize` is documented as covering unsupported readback layouts.
    #[test]
    fn a_zero_readback_row_pitch_is_typed_rather_than_panicking() {
        let error = super::copy_layout_dimensions(0, (640, 360))
            .expect_err("a zero row pitch cannot produce a copyable layout");
        match error {
            ReadbackError::Resize(message) => {
                assert!(
                    message.contains("640x360"),
                    "the refusal must name the extent it was asked to copy: {message}"
                );
                assert!(
                    message.contains("row pitch"),
                    "the refusal must name what was wrong: {message}"
                );
            }
            other => panic!("expected Resize, got {other:?}"),
        }
    }

    /// Height is checked before pitch, so a ring that is wrong in both ways
    /// reports the zero extent — the more fundamental fact — rather than a
    /// derived row-pitch symptom.
    #[test]
    fn a_doubly_invalid_layout_reports_the_zero_extent_first() {
        let error = super::copy_layout_dimensions(0, (0, 0))
            .expect_err("nothing about this layout is copyable");
        assert!(
            matches!(error, ReadbackError::ZeroDimensions { .. }),
            "the zero extent is the cause; the zero pitch is downstream of it: {error:?}"
        );
    }

    /// An ordinary layout still resolves, so the guard rejects only the
    /// genuinely impossible.
    #[test]
    fn a_valid_readback_layout_resolves_to_its_dimensions() {
        let (pitch, rows) = super::copy_layout_dimensions(2_560, (640, 360))
            .expect("an aligned 640x360 layout is copyable");
        assert_eq!(pitch.get(), 2_560);
        assert_eq!(rows.get(), 360);
    }

    /// The agent shader must CONSUME the core palette, not restate it.
    ///
    /// Asserting on the generated source is the whole point: a constant that
    /// merely happens to match today would drift the moment core's palette
    /// changed, and nothing would notice until someone compared two backends by
    /// eye. Generating from core means the drift cannot happen; this test proves
    /// the generation is actually wired.
    #[test]
    fn the_agent_shader_consumes_the_core_food_palette() {
        let source = super::agent_shader_source();
        let halo = visual::BIOLUMINESCENT_DARK_FIELD_V1.food.halo_srgb;
        let core = visual::BIOLUMINESCENT_DARK_FIELD_V1.food.core_srgb;

        for (label, value) in [("halo", halo), ("core", core)] {
            let rendered = format!("vec3<f32>({:?}, {:?}, {:?})", value[0], value[1], value[2]);
            assert!(
                source.contains(&rendered),
                "the generated shader must carry core's {label} colour verbatim; expected \
                 {rendered}"
            );
        }

        assert!(
            source.contains("mix(CORE_FOOD_HALO_SRGB, CORE_FOOD_CORE_SRGB"),
            "the nose must be derived from the core endpoints, not from a literal"
        );
        assert!(
            !source.contains("vec3<f32>(0.92, 0.6, 0.28)"),
            "the hand-authored nose literal must be gone, or the shader still \
             disagrees with core while looking like it does not"
        );
    }

    /// Where the backends still DISAGREE, say so here rather than leaving it to
    /// be discovered by comparing frames.
    ///
    /// bd-2z0.7.11 asks that the shipped backends agree on what they produce, or
    /// that the difference be declared. Terrain, agent body, the nose and — as of
    /// bd-rl1h — the diet stripe, wheels and selection rim are routed through
    /// `scriptbots_core::visual`. What remains is declared here, so it cannot
    /// silently grow, and so removing an entry is a deliberate act in a diff.
    #[test]
    fn backend_local_agent_chroma_is_declared_not_discovered() {
        // EMPTY as of bd-rl1h: every ornament this bead named is routed. The
        // constant stays rather than being deleted, because an empty declaration
        // is a claim worth keeping honest — a future ornament that authors its
        // own chroma must be added here deliberately, and that shows in a diff.
        const STILL_BACKEND_LOCAL: [&str; 0] = [];
        let source = super::agent_shader_source();

        assert!(
            STILL_BACKEND_LOCAL.is_empty(),
            "growing this list means a new divergence was introduced without \
             routing it through core"
        );

        // The routed ornaments must NOT drift back to local authorship. Asserting
        // their absence is what makes an empty list trustworthy: without it,
        // someone could reintroduce a literal and the empty list would still read
        // as complete.
        for (label, literal) in [
            (
                "stripe",
                "var stripe = mix(carn_color, herb_color, herbivore);",
            ),
            ("selection rim", "let rim_color = vec3<f32>(1.0, 1.0, 1.0)"),
            (
                "wheels",
                "let wheel_base_color = vec3<f32>(0.14, 0.16, 0.21);",
            ),
            ("mouth", "let mouth_color = vec3<f32>("),
            ("flame", "mix(vec3<f32>(1.0, 0.62, 0.22)"),
            ("spike", "mix(vec3<f32>(0.96, 0.44, 0.24)"),
            // Each pattern must include enough of the ORIGINAL EXPRESSION to be
            // unmistakable, not just its colour literal. Routing a value into a
            // generated constant puts that literal into the prelude, so a bare
            // `vec3<f32>(0.32, 0.62, 0.92)` matches the very declaration that
            // proves the ear WAS routed — this test failed exactly that way when
            // the ear landed, which is the trap every future ornament inherits.
            (
                "ears",
                "vec3<f32>(0.32, 0.62, 0.92) * (0.9 + trait_sound * 0.45)",
            ),
            ("sclera", "let sclera_color = vec3<f32>(0.97, 0.98, 1.0);"),
            ("pupil", "let pupil_color = vec3<f32>(0.08, 0.11, 0.18);"),
        ] {
            assert!(
                !source.contains(literal),
                "{label} was routed through core in bd-rl1h, but its hand-authored \
                 literal is back in the shader; the backends disagree again"
            );
        }

        // The mouth is routed by CARRYING the resolved colour, not by a generated
        // constant, so its proof is that the shader consumes the per-instance
        // value it is handed.
        assert!(
            source.contains("let mouth_color = v.mouth_color.rgb;"),
            "the mouth must read the core-resolved colour carried on the instance"
        );
    }

    /// The routed ornaments must carry CORE's values, not lookalikes.
    ///
    /// The stripe endpoints are the reason this matters most. The shader authored
    /// green/red while core's palette is cyan/magenta — not a drifted value but a
    /// different scheme, and `visual.rs` restrains genome colour specifically so
    /// the cyan-to-magenta semantic cannot be erased. Asserting on the generated
    /// source proves the generation is wired, rather than that two constants
    /// happen to agree today.
    #[test]
    fn the_agent_shader_consumes_the_core_ornament_palette() {
        let source = super::agent_shader_source();
        let agents = visual::BIOLUMINESCENT_DARK_FIELD_V1.agents;

        let boost = visual::BIOLUMINESCENT_DARK_FIELD_V1.events.boost;
        for (label, value) in [
            ("herbivore", agents.herbivore_srgb),
            ("carnivore", agents.carnivore_srgb),
            ("wheel", agents.wheel_srgb),
            ("selection rim", agents.selection_rim_srgb),
            ("spike", agents.spike_srgb),
            ("ear", agents.ear_srgb),
            ("eye sclera", agents.eye_sclera_srgb),
            ("eye pupil", agents.eye_pupil_srgb),
            ("boost core", boost.core_srgb),
            ("boost accent", boost.accent_srgb),
        ] {
            let rendered = format!("vec3<f32>({:?}, {:?}, {:?})", value[0], value[1], value[2]);
            assert!(
                source.contains(&rendered),
                "the generated shader must carry core's {label} colour verbatim; \
                 expected {rendered}"
            );
        }

        assert!(
            source.contains("mix(CORE_AGENT_CARNIVORE_SRGB, CORE_AGENT_HERBIVORE_SRGB, herbivore)"),
            "the stripe must be derived from the core endpoints, in core's argument \
             order — swapping them inverts the diet ramp while still looking routed"
        );
        assert!(
            source.contains("CORE_AGENT_WHEEL_SRGB * (0.65 + wheel_left * 0.55)"),
            "the wheels must use core's single-colour speed brightening, not a mix \
             between two authored endpoints"
        );
        assert!(
            source.contains("CORE_AGENT_SELECTION_RIM_SRGB * (selection_glow + glow)"),
            "the selection rim must take its chroma from core and keep glow as a \
             local intensity"
        );
    }

    /// Every shipped WGSL source must parse and pass naga semantic validation.
    ///
    /// These four shaders are compiled by wgpu at PIPELINE CREATION, on a real
    /// adapter — so until now a syntax or type error in any of them could only
    /// be discovered by running the GPU product on a machine with a working
    /// backend. That is the most expensive possible place to find it, and it is
    /// unreachable from CI, from the software lane, and from every agent
    /// working without a GPU.
    ///
    /// `scriptbots-bevy` already gates its particle shader this way
    /// (`wgsl_source_compiles_and_validates_with_naga`); world-gfx, which owns
    /// the actual world render path, had no equivalent.
    ///
    /// This matters directly for the remaining bd-2z0.7.11 work: the agent
    /// ornament colours still authored inside `AGENTS_WGSL` have to be rerouted
    /// through the core `AgentVisualParams` authority, and editing a shader that
    /// nothing can validate is how that lands broken.
    #[test]
    fn every_world_gfx_wgsl_source_parses_and_validates() {
        let agents_source = super::agent_shader_source();
        for (name, source) in [
            ("TERRAIN_WGSL", super::TERRAIN_WGSL),
            // The GENERATED source, because that is what wgpu compiles. Checking
            // the raw literal would validate something the product never uses.
            ("AGENTS_WGSL", agents_source.as_str()),
            ("POST_WGSL", super::POST_WGSL),
            ("BLOOM_WGSL", super::BLOOM_WGSL),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name} must parse as valid WGSL: {error}"));
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|error| panic!("{name} must pass naga validation: {error}"));
        }
    }

    /// The shaders must not be empty or accidentally truncated.
    ///
    /// A raw string that lost its body still parses as a valid empty module and
    /// would sail through validation above, so the parse gate alone cannot tell
    /// "compiles" from "compiles because there is nothing left".
    #[test]
    fn every_world_gfx_wgsl_source_declares_entry_points() {
        let agents_source = super::agent_shader_source();
        for (name, source) in [
            ("TERRAIN_WGSL", super::TERRAIN_WGSL),
            // The GENERATED source, because that is what wgpu compiles. Checking
            // the raw literal would validate something the product never uses.
            ("AGENTS_WGSL", agents_source.as_str()),
            ("POST_WGSL", super::POST_WGSL),
            ("BLOOM_WGSL", super::BLOOM_WGSL),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name} must parse: {error}"));
            assert!(
                !module.entry_points.is_empty(),
                "{name} declares no entry point, so nothing it contains can ever run"
            );
        }
    }

    #[test]
    fn stride_alignment_is_multiple_of_256() {
        let widths = [1u32, 2, 63, 64, 65, 257, 1023, 1920, 2560, 3840];
        let limits = wgpu::Limits::default();
        for w in widths {
            let raw = u64::from(w) * 4; // RGBA8 bytes per row without alignment
            let aligned = u64::from(
                validate_readback_layout((w, 1), &limits)
                    .unwrap()
                    .bytes_per_row,
            );
            assert_eq!(
                aligned % 256,
                0,
                "aligned stride must be a multiple of 256 for width {w}"
            );
            assert!(aligned >= raw, "aligned stride must be >= raw stride");
            assert!(
                aligned <= raw + 255,
                "aligned stride must not exceed raw+255"
            );
        }
    }

    #[test]
    fn resize_layout_failures_are_typed_before_gpu_allocation() {
        let limits = wgpu::Limits {
            max_texture_dimension_2d: 128,
            max_buffer_size: 1_024,
            ..wgpu::Limits::default()
        };

        assert_eq!(
            validate_readback_layout((65, 2), &limits),
            Ok(ReadbackLayout {
                bytes_per_row: 512,
                size_bytes: 1_024,
            })
        );
        assert_eq!(
            validate_readback_layout((0, 2), &limits),
            Err(ReadbackError::ZeroDimensions {
                width: 0,
                height: 2,
            })
        );

        let texture_error = validate_readback_layout((129, 1), &limits)
            .expect_err("oversized texture must be rejected before create_texture");
        assert!(
            matches!(texture_error, ReadbackError::Resize(ref detail) if detail.contains("max_texture_dimension_2d")),
            "unexpected texture-limit attribution: {texture_error}"
        );

        let buffer_error = validate_readback_layout((65, 3), &limits)
            .expect_err("oversized readback buffer must be rejected before create_buffer");
        assert!(
            matches!(buffer_error, ReadbackError::Resize(ref detail) if detail.contains("max_buffer_size")),
            "unexpected buffer-limit attribution: {buffer_error}"
        );

        let unrestricted_limits = wgpu::Limits {
            max_texture_dimension_2d: u32::MAX,
            max_buffer_size: u64::MAX,
            ..wgpu::Limits::default()
        };
        let pitch_error = validate_readback_layout((u32::MAX, 1), &unrestricted_limits)
            .expect_err("an unrepresentable copy row pitch must return a typed error");
        assert!(
            matches!(pitch_error, ReadbackError::Resize(ref detail) if detail.contains("u32 copy-layout limit")),
            "unexpected row-pitch attribution: {pitch_error}"
        );
    }

    #[test]
    fn device_fault_destroyed_noop_device_is_typed() {
        let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let monitor = DeviceFaultMonitor::install(&device);
        assert_eq!(monitor.check(), Ok(()));

        device.destroy();
        device
            .poll(wgpu::PollType::Poll)
            .expect("noop device destruction should remain pollable");

        assert_eq!(
            monitor.check(),
            Err(ReadbackError::DeviceLost {
                reason: wgpu::DeviceLostReason::Destroyed,
                detail: String::new(),
            })
        );
    }

    #[test]
    fn device_fault_scoped_noop_validation_is_typed_and_non_terminal() {
        let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let invalid_size = device
            .limits()
            .max_buffer_size
            .checked_add(1)
            .expect("noop max_buffer_size must leave room for an invalid request");

        let error = scoped_gpu_value(&device, "validation regression", || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("intentionally-oversized-test-buffer"),
                size: invalid_size,
                usage: wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .expect_err("an oversized buffer must be returned as a scoped error");

        assert!(
            matches!(
                &error,
                ReadbackError::Validation {
                    operation,
                    detail
                } if operation == "validation regression"
                    && detail.contains("Buffer size")
            ),
            "unexpected scoped validation attribution: {error}"
        );
        assert!(
            !error.is_terminal_gpu_fault(),
            "validation is a rejected operation contract, not a lost device"
        );

        let monitor = DeviceFaultMonitor::default();
        monitor.record(error.clone());
        assert_eq!(monitor.check(), Err(error));
        assert_eq!(
            monitor.check(),
            Ok(()),
            "an uncaptured validation is surfaced once rather than permanently poisoning the device"
        );
    }

    #[test]
    fn device_fault_synthetic_out_of_memory_classification_is_typed_and_sticky() {
        let monitor = DeviceFaultMonitor::default();
        let out_of_memory = readback_error_from_wgpu(
            "synthetic allocation boundary",
            wgpu::Error::OutOfMemory {
                source: Box::new(std::io::Error::other("synthetic OOM classification proof")),
            },
        );
        assert_eq!(
            out_of_memory,
            ReadbackError::OutOfMemory {
                operation: "synthetic allocation boundary".to_owned(),
            }
        );

        monitor.record(out_of_memory.clone());
        assert_eq!(monitor.check(), Err(out_of_memory.clone()));
        assert_eq!(
            monitor.check(),
            Err(out_of_memory),
            "a terminal OOM remains sticky until the renderer is replaced"
        );
    }

    #[test]
    fn device_fault_definitive_loss_overrides_an_earlier_oom_classification() {
        let monitor = DeviceFaultMonitor::default();
        monitor.record(ReadbackError::OutOfMemory {
            operation: "earlier allocation".to_owned(),
        });
        monitor.record(ReadbackError::DeviceLost {
            reason: wgpu::DeviceLostReason::Unknown,
            detail: "later callback".to_owned(),
        });
        assert_eq!(
            monitor.check(),
            Err(ReadbackError::DeviceLost {
                reason: wgpu::DeviceLostReason::Unknown,
                detail: "later callback".to_owned(),
            }),
            "a definitive loss callback must outrank a less-specific allocation symptom"
        );
    }

    #[test]
    fn agent_boost_tint_within_bounds() {
        // approximate the WGSL tint effect: base.rgb + (boost*0.35)*vec3(0.6,0.2,0.0)
        for &boost in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let tint = boost * 0.35;
            let add = [tint * 0.6, tint * 0.2, 0.0];
            let mut base = [0.4f32, 0.5, 0.6];
            for i in 0..3 {
                base[i] = (base[i] + add[i]).clamp(0.0, 1.0);
            }
            assert!((0.0..=1.0).contains(&base[0]));
            assert!((0.0..=1.0).contains(&base[1]));
            assert!((0.0..=1.0).contains(&base[2]));
        }
    }

    #[test]
    fn terrain_atlas_palette_matches_the_core_visual_authority() {
        let actual = terrain_atlas_palette();
        let expected = visual::TERRAIN_BASE_COLORS.map(|rgb| {
            [
                (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                u8::MAX,
            ]
        });
        assert_eq!(
            actual, expected,
            "world-gfx must not own a competing terrain palette"
        );
    }

    #[test]
    fn authoritative_srgb_terrain_color_is_decoded_once_for_the_gpu_target() {
        assert_eq!(
            super::WORLD_COLOR_FORMAT,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            "the transfer contract is paired with an sRGB attachment"
        );
        let linear = super::semantic_srgba_to_linear([0.5, 0.25, 0.75, 0.4]);
        let expected = [0.214_041_14, 0.050_876_09, 0.522_521_56, 0.4];
        for (channel, (actual, expected)) in linear.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-7,
                "linear channel {channel}: expected {expected}, got {actual}"
            );
        }
        assert_eq!(
            super::semantic_srgba_to_linear([0.0, 1.0, 0.04045, 0.25]),
            [0.0, 1.0, 0.04045 / 12.92, 0.25],
            "sRGB endpoints and alpha ownership must be exact"
        );
    }

    #[test]
    fn shared_terrain_oracle_reaches_world_gfx_with_only_the_srgb_transfer() {
        let weights = visual::splat_weights(&SplatInput {
            kind: TerrainKind::Grass,
            elevation: 0.91,
            slope: 0.82,
            water_depth: 1.5,
        });
        for accessibility in [
            AccessibilityPalette::Natural,
            AccessibilityPalette::Deuteranopia,
            AccessibilityPalette::Protanopia,
            AccessibilityPalette::Tritanopia,
            AccessibilityPalette::HighContrast,
        ] {
            let semantic = visual::terrain_surface_srgb(&TerrainSurfaceInput {
                splat_weights: weights,
                moisture: 0.37,
                elevation: 0.91,
                slope: 0.82,
                accent: 0.63,
                daylight: 0.35,
                accessibility,
            });
            let actual =
                super::semantic_srgba_to_linear([semantic[0], semantic[1], semantic[2], 1.0]);
            let expected = [
                super::srgb_component_to_linear(semantic[0]),
                super::srgb_component_to_linear(semantic[1]),
                super::srgb_component_to_linear(semantic[2]),
                1.0,
            ];
            assert_eq!(
                actual.map(f32::to_bits),
                expected.map(f32::to_bits),
                "{accessibility:?} terrain color must not be reinterpreted by world-gfx"
            );
        }
    }

    #[test]
    fn authoritative_srgb_agent_body_color_is_decoded_in_the_gpu_instance() {
        assert_eq!(
            super::WORLD_COLOR_FORMAT,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            "the agent transfer contract is paired with an sRGB attachment"
        );
        let agent = AgentInstance {
            color: [0.5, 0.25, 0.75, 0.4],
            ..AgentInstance::zeroed()
        };
        let actual = super::agent_instance_gpu(&agent).data6;
        let expected = [0.214_041_14, 0.050_876_09, 0.522_521_56, 0.4];
        for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-7,
                "linear agent channel {channel}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn disabled_bloom_has_exactly_zero_composite_intensity() {
        assert_eq!(
            resolve_bloom_intensity(false, 0.65).to_bits(),
            0.0_f32.to_bits(),
            "disabling bloom must remove its additive contribution, not bind the source texture \
             at the configured intensity"
        );
    }

    #[test]
    fn bloom_toggle_accepts_documented_boolean_spellings() {
        for value in ["0", "off", "false", "no", "OFF", "False"] {
            assert!(
                !parse_toggle(Some(value), true),
                "{value} must disable bloom"
            );
        }
        for value in ["1", "on", "true", "yes", "ON", "True"] {
            assert!(
                parse_toggle(Some(value), false),
                "{value} must enable bloom"
            );
        }
    }

    #[test]
    fn non_finite_or_out_of_range_post_controls_fail_closed() {
        for value in ["NaN", "inf", "-0.01", "1.01", "not-a-number"] {
            assert!(
                parse_control_f32(
                    "SB_WGPU_VIGNETTE",
                    value,
                    |parsed| (0.0..=1.0).contains(&parsed),
                    "a finite number in 0..=1",
                )
                .is_err(),
                "{value:?} must not reach WGSL as a vignette control"
            );
        }
    }

    #[test]
    fn post_decision_is_a_pure_function_of_the_frozen_controls() {
        let controls = PostControls {
            vignette: 0.0,
            tonemap: 0,
            bloom_enabled: false,
            fog_enabled: false,
            fxaa: 0,
            ..PostControls::default()
        };
        assert!(!controls.wants_post(None));
        assert!(!controls.wants_post(None));
        assert!(
            controls.wants_post(Some(1)),
            "a snapshot-selected tonemap must not be skipped because environment defaults are off"
        );
    }

    #[test]
    fn readback_rejects_a_frame_from_a_different_extent() {
        let error = validate_frame_token(
            &RenderFrame {
                extent: (64, 32),
                renderer_id: 7,
                generation: 11,
            },
            (128, 64),
            7,
            11,
        )
        .expect_err("cross-extent readback must be rejected");
        assert!(
            matches!(error, super::ReadbackError::MetadataMismatch { .. }),
            "unexpected readback error: {error}"
        );
    }

    #[test]
    fn readback_rejects_a_same_extent_stale_or_foreign_frame() {
        let stale = RenderFrame {
            extent: (128, 64),
            renderer_id: 7,
            generation: 10,
        };
        assert!(matches!(
            validate_frame_token(&stale, (128, 64), 7, 11),
            Err(super::ReadbackError::MetadataMismatch { .. })
        ));

        let foreign = RenderFrame {
            extent: (128, 64),
            renderer_id: 8,
            generation: 11,
        };
        assert!(matches!(
            validate_frame_token(&foreign, (128, 64), 7, 11),
            Err(super::ReadbackError::MetadataMismatch { .. })
        ));
    }

    #[test]
    fn snapshot_metadata_mismatch_is_typed_before_gpu_submission() {
        let tiles = [3_u32];
        let missing_colors: [[f32; 4]; 0] = [];
        let snapshot = WorldSnapshot {
            world_size: (10.0, 10.0),
            terrain: TerrainView {
                dims: (1, 1),
                cell_size: 10,
                tiles: &tiles,
                colors: &missing_colors,
                elevation: None,
            },
            agents: &[],
            anim_seconds: 0.0,
            tonemap_mode: None,
        };
        assert!(matches!(
            validate_snapshot(&snapshot),
            Err(super::ReadbackError::MetadataMismatch { .. })
        ));
    }
}

struct ViewUniforms {
    buf: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bg: wgpu::BindGroup,
}

struct PreparedViewUniforms {
    buf: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewData {
    v0: [f32; 4],
    v1: [f32; 4],
}

impl ViewUniforms {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, size: (u32, u32)) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view.bg_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<ViewData>() as u64
                    ),
                },
                count: None,
            }],
        });
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view.ubuf"),
            size: std::mem::size_of::<ViewData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view.bg"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        let this = Self { buf, layout, bg };
        this.update(queue, size, 0.0, 1.0, (0.0, 0.0));
        this
    }

    fn prepare_resize(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: (u32, u32),
        time: f32,
        scale: f32,
        offset: (f32, f32),
    ) -> PreparedViewUniforms {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view.ubuf.resize"),
            size: std::mem::size_of::<ViewData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view.bg.resize"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        let data = ViewData {
            v0: [size.0 as f32, size.1 as f32, time, scale],
            v1: [offset.0, offset.1, 0.0, 0.0],
        };
        queue.write_buffer(&buf, 0, bytemuck::bytes_of(&data));
        PreparedViewUniforms { buf, bg }
    }

    fn install_resize(&mut self, prepared: PreparedViewUniforms) {
        self.buf = prepared.buf;
        self.bg = prepared.bg;
    }

    fn update(
        &self,
        queue: &wgpu::Queue,
        size: (u32, u32),
        time: f32,
        scale: f32,
        offset: (f32, f32),
    ) {
        let v0 = [size.0 as f32, size.1 as f32, time, scale];
        let v1 = [offset.0, offset.1, 0.0, 0.0];
        let data = ViewData { v0, v1 };
        queue.write_buffer(&self.buf, 0, bytemuck::bytes_of(&data));
    }
}

// ---------------- Terrain pipeline (instanced tiles with atlas) ----------------

struct TerrainPipeline {
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    atlas: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    bg_layout: wgpu::BindGroupLayout,
    bg: wgpu::BindGroup,
    tile_vbuf: wgpu::Buffer,
    _tile_count: u32,
    grid_cols: u32,
    grid_rows: u32,
    tile_w: u32,
    tile_h: u32,
    atlas_w: u32,
    atlas_h: u32,
    vbuf_capacity_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileInstance {
    pos: [f32; 2],
    size: [f32; 2],
    color: [f32; 4],
    kind: u32,
    slope: f32,
}

/// Decode an authoritative semantic sRGB color before a fragment writes it to
/// an `Rgba8UnormSrgb` attachment. wgpu performs the inverse linear-to-sRGB
/// transfer for RGB attachment writes; alpha is always linear and unchanged.
fn semantic_srgba_to_linear(srgba: [f32; 4]) -> [f32; 4] {
    [
        srgb_component_to_linear(srgba[0]),
        srgb_component_to_linear(srgba[1]),
        srgb_component_to_linear(srgba[2]),
        srgba[3],
    ]
}

fn srgb_component_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn terrain_atlas_palette() -> [[u8; 4]; 6] {
    visual::TERRAIN_BASE_COLORS.map(|rgb| {
        [
            (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            u8::MAX,
        ]
    })
}

impl TerrainPipeline {
    fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat, view: &ViewUniforms) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // 1x1 white atlas placeholder; real atlas supplied later via update
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.atlas"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());

        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain.bg_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(TERRAIN_WGSL)),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain.layout"),
            bind_group_layouts: &[&bg_layout, &view.layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<TileInstance>() as u64, step_mode: wgpu::VertexStepMode::Instance, attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4, 3 => Uint32, 4 => Float32] }] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format: color_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, strip_index_format: None, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let vbuf_capacity_bytes = (1024 * std::mem::size_of::<TileInstance>()) as u64;
        let tile_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain.instances"),
            size: vbuf_capacity_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // default atlas grid config (3x2 tiles of 64px)
        let grid_cols = 3;
        let grid_rows = 2;
        let tile_w = 64;
        let tile_h = 64;
        let atlas_w = grid_cols * tile_w;
        let atlas_h = grid_rows * tile_h;
        Self {
            pipeline,
            sampler,
            atlas,
            atlas_view,
            bg_layout,
            bg,
            tile_vbuf,
            _tile_count: 0,
            grid_cols,
            grid_rows,
            tile_w,
            tile_h,
            atlas_w,
            atlas_h,
            vbuf_capacity_bytes,
        }
    }
    /// Bake and upload the terrain atlas.
    ///
    /// # Errors
    /// Returns [`ReadbackError::ZeroDimensions`] or [`ReadbackError::Resize`]
    /// when the derived atlas extent cannot describe a texture upload.
    ///
    /// `atlas_w`/`atlas_h` are DERIVED (`grid_cols * tile_w`,
    /// `grid_rows * tile_h`), not constants, so a misconfigured tile size
    /// produces a zero. This used to reach `NonZeroU32::new(..).unwrap()` and
    /// abort the process during GPU INITIALISATION, before the renderer even
    /// existed to report anything (bd-aqy8). Validated up front, so the failure
    /// is typed and attributable and no GPU resource is created for an extent
    /// that cannot be uploaded.
    fn init_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), ReadbackError> {
        // Checked BEFORE allocating the texture: the same fail-before-mutate
        // ordering the resize and readback paths already use.
        let (atlas_pitch, atlas_rows) =
            copy_layout_dimensions(self.atlas_w * 4, (self.atlas_w, self.atlas_h))?;

        // Generate a simple 3x2 atlas (DeepWater, ShallowWater, Sand, Grass, Bloom, Rock)
        self.atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain.atlas.real"),
            size: wgpu::Extent3d {
                width: self.atlas_w,
                height: self.atlas_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.atlas_view = self
            .atlas
            .create_view(&wgpu::TextureViewDescriptor::default());
        // fill tiles with curated colors
        let mut pixels = vec![0u8; (self.atlas_w * self.atlas_h * 4) as usize];
        let colors = terrain_atlas_palette();
        for row in 0..self.grid_rows {
            for col in 0..self.grid_cols {
                let idx = (row * self.grid_cols + col) as usize;
                let color = colors.get(idx).copied().unwrap_or([255, 255, 255, 255]);
                for y in 0..self.tile_h {
                    for x in 0..self.tile_w {
                        let px = col * self.tile_w + x;
                        let py = row * self.tile_h + y;
                        let offset = ((py * self.atlas_w + px) * 4) as usize;
                        let mut rgba = color;
                        // add gentle vignette/variation for non-water tiles
                        if idx >= 2 {
                            let fx = (x as f32 / self.tile_w as f32 - 0.5).abs();
                            let fy = (y as f32 / self.tile_h as f32 - 0.5).abs();
                            let vignette = fx.max(fy) * 0.12;
                            let dim = (1.0 - vignette).clamp(0.85, 1.0);
                            rgba[0] = ((rgba[0] as f32) * dim) as u8;
                            rgba[1] = ((rgba[1] as f32) * dim) as u8;
                            rgba[2] = ((rgba[2] as f32) * dim) as u8;
                        }
                        pixels[offset..offset + 4].copy_from_slice(&rgba);
                    }
                }
            }
        }
        queue.write_texture(
            self.atlas.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_pitch.get()),
                rows_per_image: Some(atlas_rows.get()),
            },
            wgpu::Extent3d {
                width: self.atlas_w,
                height: self.atlas_h,
                depth_or_array_layers: 1,
            },
        );
        // refresh bind group to point to the new view (returns Ok below)
        self.bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain.bg.rebind"),
            layout: &self.bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        Ok(())
    }

    fn ensure_vbuf_capacity(&mut self, device: &wgpu::Device, needed_bytes: u64) {
        if needed_bytes <= self.vbuf_capacity_bytes {
            return;
        }
        let mut cap = self.vbuf_capacity_bytes.max(1024);
        while cap < needed_bytes {
            cap *= 2;
        }
        self.tile_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain.instances.realloc"),
            size: cap,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.vbuf_capacity_bytes = cap;
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view_tex: &wgpu::TextureView,
        view_uniforms: &ViewUniforms,
        snapshot: &WorldSnapshot,
        viewport: (u32, u32),
        scale: f32,
        offset: (f32, f32),
    ) -> u32 {
        // Build tile instances for visible terrain with simple CPU frustum culling.
        let (tw, th) = snapshot.terrain.dims;
        let cell = snapshot.terrain.cell_size as f32;
        let mut staging: Vec<TileInstance> = Vec::with_capacity((tw as usize) * (th as usize));
        let (vp_w, vp_h) = (viewport.0 as f32, viewport.1 as f32);
        let disable_cull = matches!(std::env::var("SB_WGPU_DISABLE_CULL").ok().map(|s| s.to_ascii_lowercase()), Some(ref v) if v == "1" || v == "true" || v == "yes" || v == "on");
        let elev_opt = snapshot.terrain.elevation;
        let get_elev = |x: i32, y: i32| -> f32 {
            if let Some(elev) = elev_opt {
                let xi = x.clamp(0, (tw as i32) - 1) as usize;
                let yi = y.clamp(0, (th as i32) - 1) as usize;
                let idx = yi * (tw as usize) + xi;
                elev.get(idx).copied().unwrap_or(0.5)
            } else {
                0.0
            }
        };
        for y in 0..th as i32 {
            for x in 0..tw as i32 {
                let px = x as f32 * cell;
                let py = y as f32 * cell;
                // Convert to pixel-space for culling using camera
                let min_x_px = px * scale + offset.0;
                let min_y_px = py * scale + offset.1;
                let max_x_px = min_x_px + cell * scale;
                let max_y_px = min_y_px + cell * scale;
                if !disable_cull
                    && (max_x_px < 0.0 || max_y_px < 0.0 || min_x_px > vp_w || min_y_px > vp_h)
                {
                    continue;
                }
                let idx = (y as usize) * (tw as usize) + (x as usize);
                let tile_id = snapshot.terrain.tiles.get(idx).copied().unwrap_or(3);
                // slope via central differences if elevation present
                let slope = if elev_opt.is_some() {
                    let dx = (get_elev(x + 1, y) - get_elev(x - 1, y)) * 0.5;
                    let dy = (get_elev(x, y + 1) - get_elev(x, y - 1)) * 0.5;
                    (dx * dx + dy * dy).sqrt().clamp(0.0, 1.0)
                } else {
                    0.0
                };
                staging.push(TileInstance {
                    pos: [px, py],
                    size: [cell, cell],
                    color: semantic_srgba_to_linear(snapshot.terrain.colors[idx]),
                    kind: tile_id,
                    slope,
                });
            }
        }
        if !staging.is_empty() {
            let needed = (staging.len() * std::mem::size_of::<TileInstance>()) as u64;
            self.ensure_vbuf_capacity(device, needed);
            queue.write_buffer(&self.tile_vbuf, 0, bytemuck::cast_slice(&staging));
            self._tile_count = staging.len() as u32;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("terrain.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: view_tex,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.set_bind_group(1, &view_uniforms.bg, &[]);
        pass.set_vertex_buffer(0, self.tile_vbuf.slice(..));
        pass.draw(0..4, 0..staging.len() as u32);
        staging.len() as u32
    }
}

const TERRAIN_WGSL: &str = r#"
struct VsIn {
  @location(0) pos: vec2<f32>,
  @location(1) size: vec2<f32>,
  @location(2) color: vec4<f32>,
  @location(3) kind: u32,
  @location(4) slope: f32,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
};

struct View { v0: vec4<f32>, v1: vec4<f32> }; // v0=(viewport.x,viewport.y,time,scale) v1=(offset_x,offset_y,_,_)
@group(1) @binding(0) var<uniform> view: View;

@vertex
fn vs_main(inst: VsIn, @builtin(vertex_index) vid: u32) -> VsOut {
  var o: VsOut;
  var quad = array<vec2<f32>, 4>(vec2<f32>(0.0,0.0), vec2<f32>(1.0,0.0), vec2<f32>(0.0,1.0), vec2<f32>(1.0,1.0));
  let p = quad[vid];
  let xy = inst.pos + p * inst.size;
  let viewport = view.v0.xy;
  let scale = view.v0.w;
  let offset = view.v1.xy;
  let pos = (inst.pos + p * inst.size) * scale + offset;
  let ndc = vec2<f32>(pos.x / viewport.x * 2.0 - 1.0, 1.0 - (pos.y / viewport.y * 2.0));
  o.pos = vec4<f32>(ndc, 0.0, 1.0);
  o.color = inst.color;
  return o;
}

@group(0) @binding(0) var atlas_tex: texture_2d<f32>;
@group(0) @binding(1) var atlas_smp: sampler;

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
  // Core projects the final terrain color in sRGB; CPU staging decodes it to
  // this linear value, and Rgba8UnormSrgb performs the sole output encoding.
  // The shader owns rasterization only and must not invent a competing biome
  // palette, shimmer, slope curve, or hash-noise model.
  return v.color;
}
"#;

// ---------------- Agent pipeline (instanced sprites with effects) ----------------

struct AgentPipeline {
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    vbuf_capacity_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AgentInstanceGpu {
    data0: [f32; 4],
    data1: [f32; 4],
    data2: [f32; 4],
    data3: [f32; 4],
    data4: [f32; 4],
    data5: [f32; 4],
    data6: [f32; 4],
    data7: [f32; 4],
    data8: [f32; 4],
    data9: [f32; 4],
    data10: [f32; 4],
}

fn agent_instance_gpu(agent: &AgentInstance) -> AgentInstanceGpu {
    AgentInstanceGpu {
        data0: [
            agent.position[0],
            agent.position[1],
            agent.quad_extent[0],
            agent.quad_extent[1],
        ],
        data1: [
            agent.heading[0],
            agent.heading[1],
            agent.body_radius,
            agent.body_half_length,
        ],
        data2: [
            agent.wheel_offset,
            agent.wheel_radius,
            agent.mouth_open,
            agent.herbivore_tendency,
        ],
        data3: [
            agent.temperature_preference,
            agent.food_delta,
            agent.sound_level,
            agent.sound_output,
        ],
        data4: [
            agent.wheel_left,
            agent.wheel_right,
            agent.trait_smell,
            agent.trait_sound,
        ],
        data5: [
            agent.trait_hearing,
            agent.trait_eye,
            agent.trait_blood,
            agent.selection,
        ],
        data6: semantic_srgba_to_linear(agent.color),
        data7: [agent.glow, agent.boost, agent.spiked, agent.spike_length],
        data8: agent.eye_dirs,
        data9: agent.eye_fov,
        // Kept in semantic sRGB, NOT converted to linear like the body colour on
        // data6. Every ornament in this shader is authored in sRGB and composited
        // by `layer` in that space, so converting only the mouth would make it the
        // one ornament in a different space (bd-rl1h).
        data10: [
            agent.mouth_color[0],
            agent.mouth_color[1],
            agent.mouth_color[2],
            0.0,
        ],
    }
}

/// The agent shader, with the canonical palette values it must not re-author
/// prepended as WGSL constants (bd-2z0.7.11).
///
/// The backend-correctness half of this bead asks that the shipped backends
/// agree on what they produce, or that a difference be DECLARED rather than
/// discovered. The nose tint was discovered: the shader painted a bare
/// `vec3<f32>(0.92, 0.6, 0.28)` while `scriptbots_core::visual` derives
/// `nose_color` by mixing the canonical food halo and core colours by
/// `trait_smell` — a value the shader already receives per instance. It had the
/// input and invented the endpoints, so GPUI and world-gfx disagreed on the
/// same agent for no reason anyone had recorded.
///
/// Generating the constants from core rather than retyping them means the two
/// cannot drift: a palette edit in core reaches the GPU without anybody
/// remembering that this shader exists. Only the small prelude is formatted;
/// the shader body stays a raw literal, so no brace in it needs escaping.
/// Extends to the diet stripe, wheels and selection rim (bd-rl1h). Those three
/// were not merely drifted values — the shader was painting a DIFFERENT COLOUR
/// SCHEME. It authored `herb_color = (0.24, 0.78, 0.36)` green and
/// `carn_color = (0.88, 0.26, 0.21)` red, while core's palette is
/// `herbivore_srgb = [0.18, 0.86, 1.00]` cyan and
/// `carnivore_srgb = [1.00, 0.60, 0.92]` magenta. `visual.rs` calls that the
/// "cyan-to-magenta semantic" and deliberately restrains genome colour so it
/// cannot be erased; world-gfx was erasing it wholesale. The selection rim was
/// pure white against core's `[0.72, 0.94, 1.00]`.
fn agent_shader_source() -> String {
    let palette = &scriptbots_core::visual::BIOLUMINESCENT_DARK_FIELD_V1;
    let halo = palette.food.halo_srgb;
    let core = palette.food.core_srgb;
    let herbivore = palette.agents.herbivore_srgb;
    let carnivore = palette.agents.carnivore_srgb;
    let wheel = palette.agents.wheel_srgb;
    let rim = palette.agents.selection_rim_srgb;
    let spike = palette.agents.spike_srgb;
    let ear = palette.agents.ear_srgb;
    let sclera = palette.agents.eye_sclera_srgb;
    let pupil = palette.agents.eye_pupil_srgb;
    let boost_core = palette.events.boost.core_srgb;
    let boost_accent = palette.events.boost.accent_srgb;
    format!(
        "// GENERATED from scriptbots_core::visual::BIOLUMINESCENT_DARK_FIELD_V1 (bd-2z0.7.11).\n\
         // Do not hand-edit these values; change the palette in core instead.\n\
         const CORE_FOOD_HALO_SRGB: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
         const CORE_FOOD_CORE_SRGB: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
         const CORE_AGENT_HERBIVORE_SRGB: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
         const CORE_AGENT_CARNIVORE_SRGB: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
         const CORE_AGENT_WHEEL_SRGB: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
         const CORE_AGENT_SELECTION_RIM_SRGB: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
         const CORE_AGENT_SPIKE_SRGB: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
         const CORE_AGENT_EAR_SRGB: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
         const CORE_AGENT_EYE_SCLERA_SRGB: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
         const CORE_AGENT_EYE_PUPIL_SRGB: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
         const CORE_EVENT_BOOST_CORE_SRGB: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
         const CORE_EVENT_BOOST_ACCENT_SRGB: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
         {AGENTS_WGSL}",
        halo[0],
        halo[1],
        halo[2],
        core[0],
        core[1],
        core[2],
        herbivore[0],
        herbivore[1],
        herbivore[2],
        carnivore[0],
        carnivore[1],
        carnivore[2],
        wheel[0],
        wheel[1],
        wheel[2],
        rim[0],
        rim[1],
        rim[2],
        spike[0],
        spike[1],
        spike[2],
        ear[0],
        ear[1],
        ear[2],
        sclera[0],
        sclera[1],
        sclera[2],
        pupil[0],
        pupil[1],
        pupil[2],
        boost_core[0],
        boost_core[1],
        boost_core[2],
        boost_accent[0],
        boost_accent[1],
        boost_accent[2],
    )
}

impl AgentPipeline {
    fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat, view: &ViewUniforms) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("agents.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(agent_shader_source())),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("agents.layout"),
            bind_group_layouts: &[&view.layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("agents.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<AgentInstanceGpu>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x4,
                        1 => Float32x4,
                        2 => Float32x4,
                        3 => Float32x4,
                        4 => Float32x4,
                        5 => Float32x4,
                        6 => Float32x4,
                        7 => Float32x4,
                        8 => Float32x4,
                        9 => Float32x4,
                        10 => Float32x4
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let vbuf_capacity_bytes = (1024 * std::mem::size_of::<AgentInstanceGpu>()) as u64;
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agents.instances"),
            size: vbuf_capacity_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            vbuf,
            vbuf_capacity_bytes,
        }
    }

    fn ensure_vbuf_capacity(&mut self, device: &wgpu::Device, needed_bytes: u64) {
        if needed_bytes <= self.vbuf_capacity_bytes {
            return;
        }
        let mut cap = self.vbuf_capacity_bytes.max(1024);
        while cap < needed_bytes {
            cap *= 2;
        }
        self.vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agents.instances.realloc"),
            size: cap,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.vbuf_capacity_bytes = cap;
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view_tex: &wgpu::TextureView,
        view_uniforms: &ViewUniforms,
        snapshot: &WorldSnapshot,
        viewport: (u32, u32),
        scale: f32,
        offset: (f32, f32),
    ) -> u32 {
        let mut staging: Vec<AgentInstanceGpu> = Vec::with_capacity(snapshot.agents.len());
        let (vp_w, vp_h) = (viewport.0 as f32, viewport.1 as f32);
        let disable_cull = matches!(std::env::var("SB_WGPU_DISABLE_CULL").ok().map(|s| s.to_ascii_lowercase()), Some(ref v) if v == "1" || v == "true" || v == "yes" || v == "on");
        for a in snapshot.agents {
            // CPU frustum culling (pixel-space); assumes positions/sizes are pixels in this pass
            let cx = a.position[0] * scale + offset.0;
            let cy = a.position[1] * scale + offset.1;
            let radius_x = a.quad_extent[0] * scale;
            let radius_y = a.quad_extent[1] * scale;
            if !disable_cull
                && (cx + radius_x < 0.0
                    || cx - radius_x > vp_w
                    || cy + radius_y < 0.0
                    || cy - radius_y > vp_h)
            {
                continue;
            }
            staging.push(agent_instance_gpu(a));
        }
        if !staging.is_empty() {
            let needed = (staging.len() * std::mem::size_of::<AgentInstanceGpu>()) as u64;
            self.ensure_vbuf_capacity(device, needed);
            queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&staging));
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("agents.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: view_tex,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &view_uniforms.bg, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.draw(0..4, 0..staging.len() as u32);
        staging.len() as u32
    }
}

const AGENTS_WGSL: &str = r#"
struct InInst {
  @location(0) data0: vec4<f32>,
  @location(1) data1: vec4<f32>,
  @location(2) data2: vec4<f32>,
  @location(3) data3: vec4<f32>,
  @location(4) data4: vec4<f32>,
  @location(5) data5: vec4<f32>,
  @location(6) data6: vec4<f32>,
  @location(7) data7: vec4<f32>,
  @location(8) data8: vec4<f32>,
  @location(9) data9: vec4<f32>,
  @location(10) data10: vec4<f32>,
};

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) extent: vec2<f32>,
  @location(2) heading: vec2<f32>,
  @location(3) body_params: vec4<f32>,
  @location(4) behavior: vec4<f32>,
  @location(5) audio: vec4<f32>,
  @location(6) traits_a: vec4<f32>,
  @location(7) traits_b: vec4<f32>,
  @location(8) color: vec4<f32>,
  @location(9) eye_dirs: vec4<f32>,
  @location(10) eye_fov: vec4<f32>,
  @location(11) extras: vec2<f32>,
  // Core-resolved mouth chroma, carried through rather than recomputed here.
  @location(12) mouth_color: vec4<f32>,
};

struct View { v0: vec4<f32>, v1: vec4<f32> }; // v0=(viewport.x,viewport.y,time,scale) v1=(offset_x,offset_y,_,_)
@group(0) @binding(0) var<uniform> view: View;

fn capsule_distance(p: vec2<f32>, half_length: f32, radius: f32) -> f32 {
  let clamped = clamp(p.y, -half_length + radius, half_length - radius);
  return length(vec2<f32>(p.x, p.y - clamped)) - radius;
}

fn circle_distance(p: vec2<f32>, radius: f32) -> f32 {
  return length(p) - radius;
}

fn smooth_mask(dist: f32) -> f32 {
  let aa = max(fwidth(dist), 1e-3);
  // smoothstep(aa, -aa, x) has reversed edges — undefined behavior per the WGSL spec
  // (edge0 must be < edge1). The spec-safe form is identical on every defined driver.
  return 1.0 - smoothstep(-aa, aa, dist);
}

fn layer(base_rgb: ptr<function, vec3<f32>>, base_alpha: ptr<function, f32>, color: vec3<f32>, alpha: f32) {
  let a = clamp(alpha, 0.0, 1.0);
  if (a <= 0.0001) {
    return;
  }
  let current = *base_alpha;
  let new_alpha = current + a * (1.0 - current);
  if (new_alpha <= 0.0001) {
    *base_rgb = color;
    *base_alpha = a;
    return;
  }
  let weight = a * (1.0 - current) / new_alpha;
  *base_rgb = mix(*base_rgb, color, weight);
  *base_alpha = new_alpha;
}

@vertex
fn vs_main(inst: InInst, @builtin(vertex_index) vid: u32) -> VsOut {
  var o: VsOut;
  let quad = array<vec2<f32>, 4>(vec2<f32>(-0.5,-0.5), vec2<f32>(0.5,-0.5), vec2<f32>(-0.5,0.5), vec2<f32>(0.5,0.5));
  let l = quad[vid];
  let extent = inst.data0.zw;
  let local = vec2<f32>(l.x * extent.x * 2.0, l.y * extent.y * 2.0);
  let viewport = view.v0.xy;
  let scale = view.v0.w;
  let offset = view.v1.xy;
  let center = inst.data0.xy;
  let world = (center + local) * scale + offset;
  let ndc = vec2<f32>(world.x / viewport.x * 2.0 - 1.0, 1.0 - (world.y / viewport.y * 2.0));
  o.pos = vec4<f32>(ndc, 0.0, 1.0);
  o.local = local;
  o.extent = extent;
  o.heading = inst.data1.xy;
  o.body_params = vec4<f32>(inst.data1.z, inst.data1.w, inst.data2.x, inst.data2.y);
  o.behavior = vec4<f32>(inst.data2.z, inst.data2.w, inst.data3.x, inst.data3.y);
  o.audio = vec4<f32>(inst.data3.z, inst.data3.w, inst.data4.x, inst.data4.y);
  o.traits_a = vec4<f32>(inst.data4.z, inst.data4.w, inst.data5.x, inst.data5.y);
  o.traits_b = vec4<f32>(inst.data5.z, inst.data5.w, inst.data7.x, inst.data7.y);
  o.color = inst.data6;
  o.eye_dirs = inst.data8;
  o.eye_fov = inst.data9;
  o.extras = vec2<f32>(inst.data7.z, inst.data7.w);
  o.mouth_color = inst.data10;
  return o;
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
  let heading = normalize(v.heading);
  let right = vec2<f32>(-heading.y, heading.x);
  let local = vec2<f32>(dot(v.local, right), dot(v.local, heading));

  let body_radius = max(v.body_params.x, 0.5);
  let body_half_length = max(v.body_params.y, body_radius);
  let wheel_offset = v.body_params.z;
  let wheel_radius = v.body_params.w;
  let mouth_open = v.behavior.x;
  let herbivore = clamp(v.behavior.y, 0.0, 1.0);
  let temperature = clamp(v.behavior.z, 0.0, 1.0);
  let food_delta = v.behavior.w;
  let sound_level = clamp(abs(v.audio.x), 0.0, 1.0);
  let sound_output = clamp(abs(v.audio.y), 0.0, 1.0);
  let wheel_left = clamp(v.audio.z, 0.0, 1.0);
  let wheel_right = clamp(v.audio.w, 0.0, 1.0);
  let trait_smell = v.traits_a.x;
  let trait_sound = v.traits_a.y;
  let trait_hearing = v.traits_a.z;
  let trait_eye = v.traits_a.w;
  let trait_blood = v.traits_b.x;
  let selection = v.traits_b.y;
  let glow = v.traits_b.z;
  let boost = v.traits_b.w;
  let spiked = v.extras.x;
  let spike_length = v.extras.y;

  let body_dist = capsule_distance(local, body_half_length, body_radius);
  let body_mask = smooth_mask(body_dist);

  var accum_rgb = vec3<f32>(0.0);
  var accum_alpha = 0.0;

  // Wheels
  let wheel_half_length = body_half_length * 0.96;
  // Core authority: visual::agent_visual_params brightens the single canonical
  // wheel colour by speed, WHEEL_BASE_RGB * (0.65 + clamp01(speed) * 0.55),
  // rather than interpolating between two authored endpoints. `wheel_left` and
  // `wheel_right` are already clamped where they are unpacked, so this is core's
  // expression verbatim (bd-rl1h).
  let left_dist = capsule_distance(vec2<f32>(local.x + wheel_offset, local.y), wheel_half_length, wheel_radius);
  let right_dist = capsule_distance(vec2<f32>(local.x - wheel_offset, local.y), wheel_half_length, wheel_radius);
  let left_color = CORE_AGENT_WHEEL_SRGB * (0.65 + wheel_left * 0.55);
  let right_color = CORE_AGENT_WHEEL_SRGB * (0.65 + wheel_right * 0.55);
  layer(&accum_rgb, &accum_alpha, left_color, smooth_mask(left_dist));
  layer(&accum_rgb, &accum_alpha, right_color, smooth_mask(right_dist));

  // Body shell
  let body_color = clamp(v.color.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  layer(&accum_rgb, &accum_alpha, body_color, body_mask);

  // Diet stripe
  // Core authority: visual::diet_stripe_color is
  // mix_vec3(CARNIVORE_RGB, HERBIVORE_RGB, clamp01(herbivore_tendency)), and
  // `herbivore` is already clamped where it is unpacked. WGSL mix is
  // a + (b - a) * t, identical to core's mix_vec3, so this is the same value
  // rather than a lookalike. The hand-authored green/red endpoints were not a
  // drifted palette but a different scheme entirely (bd-rl1h).
  var stripe = mix(CORE_AGENT_CARNIVORE_SRGB, CORE_AGENT_HERBIVORE_SRGB, herbivore);
  let blood_tint = clamp(0.7 + trait_blood * 0.2, 0.8, 1.35);
  stripe = clamp(stripe * blood_tint, vec3<f32>(0.0), vec3<f32>(1.2));
  let stripe_dist = capsule_distance(local, body_half_length * 0.82, body_radius * 0.45);
  layer(&accum_rgb, &accum_alpha, stripe, smooth_mask(stripe_dist) * body_mask * 0.9);

  // Flame (boost)
  if (boost > 0.05) {
    let flame_half = body_radius * (0.45 + boost * 0.5);
    let flame_radius = body_radius * (0.18 + boost * 0.4);
    let flame_center = vec2<f32>(0.0, -body_half_length - flame_half * 0.6);
    let flame_dist = capsule_distance(local - flame_center, flame_half, flame_radius);
    // Core authority: the boost trail is an event cue, and visual::visual_cue
    // builds BoostTrail from events.boost core/accent. The local orange endpoints
    // were a different scheme from core's cyan pair entirely (bd-sqji).
    let flame_color = mix(CORE_EVENT_BOOST_CORE_SRGB, CORE_EVENT_BOOST_ACCENT_SRGB, sound_output);
    layer(&accum_rgb, &accum_alpha, flame_color, smooth_mask(flame_dist) * 0.8);
  }

  // Spike approximation
  let spike_half = body_radius * 0.7 + spike_length * 0.6;
  let spike_radius = body_radius * 0.32;
  let spike_center = vec2<f32>(0.0, body_half_length + spike_radius);
  let spike_dist = capsule_distance(local - spike_center, spike_half, spike_radius);
  // Core authority: visual::agent_visual_params sets spike_color from
  // agents.spike_srgb, a single colour rather than a two-endpoint ramp. The
  // strike cue is preserved as a BRIGHTNESS step on that one chroma — an
  // intensity, which core does not author, the same split used for the selection
  // rim's glow. The old local endpoints were orange against core's pale pink
  // (bd-sqji).
  let spike_color = mix(CORE_AGENT_SPIKE_SRGB * 0.94, CORE_AGENT_SPIKE_SRGB, clamp(spiked, 0.0, 1.0));
  layer(&accum_rgb, &accum_alpha, spike_color, smooth_mask(spike_dist));

  // Mouth
  let mouth_half_length = body_radius * 0.62;
  let mouth_radius = max(body_radius * 0.14, 1.2) * mouth_open;
  let mouth_center = vec2<f32>(0.0, body_half_length - body_radius * 0.35);
  let mouth_local = local - mouth_center;
  let mouth_swapped = vec2<f32>(mouth_local.y, mouth_local.x);
  let mouth_dist = capsule_distance(mouth_swapped, mouth_half_length, mouth_radius);
  // Core authority: visual::agent_visual_params derives mouth_color as
  // mix(events.death.core, events.combat.core, mouth_activity), where
  // mouth_activity folds in sound_multiplier — a value this instance does not
  // carry. So the RESOLVED colour is packed per instance rather than its inputs
  // (bd-rl1h): re-deriving it here from food_delta/sound_output alone would be a
  // different function wearing the same name, which is how the local literal
  // below diverged in the first place.
  let mouth_color = v.mouth_color.rgb;
  layer(&accum_rgb, &accum_alpha, clamp(mouth_color, vec3<f32>(0.0), vec3<f32>(1.0)), smooth_mask(mouth_dist));

  // Nose
  let nose_radius = max(body_radius * 0.12, 1.0) * (0.6 + trait_smell * 0.8);
  let nose_center = vec2<f32>(0.0, body_half_length - body_radius * 0.2);
  let nose_dist = circle_distance(local - nose_center, nose_radius);
  // Core authority: visual::agent_visual_params derives nose_color as
  // mix(food.halo, food.core, clamp01(trait_smell * 0.4)). WGSL mix is
  // a + (b - a) * t, identical to core's mix_vec3, so this is the same value
  // rather than a lookalike (bd-2z0.7.11).
  let nose_tint = mix(CORE_FOOD_HALO_SRGB, CORE_FOOD_CORE_SRGB, clamp(trait_smell * 0.4, 0.0, 1.0));
  layer(&accum_rgb, &accum_alpha, nose_tint, smooth_mask(nose_dist));

  // Ears (sound/hearing)
  let ear_scale = clamp(0.6 + trait_hearing * 0.45, 0.6, 1.6);
  let ear_radius = max(body_radius * 0.28, 1.5) * ear_scale;
  let ear_offset = body_half_length * 0.15;
  // Core authority: agents.ear_srgb, added by bd-sqji because no authority
  // existed. The trait_sound term stays local — it is an intensity response, not
  // a chroma, and core does not author it.
  let ear_color_base = CORE_AGENT_EAR_SRGB * (0.9 + trait_sound * 0.45);
  let ear_left_center = vec2<f32>(-(body_radius + ear_radius * 0.45), -ear_offset);
  let ear_right_center = vec2<f32>(body_radius + ear_radius * 0.45, -ear_offset);
  let ear_left_dist = circle_distance(local - ear_left_center, ear_radius);
  let ear_right_dist = circle_distance(local - ear_right_center, ear_radius);
  layer(&accum_rgb, &accum_alpha, clamp(ear_color_base, vec3<f32>(0.0), vec3<f32>(1.0)), smooth_mask(ear_left_dist) * 0.9);
  layer(&accum_rgb, &accum_alpha, clamp(ear_color_base, vec3<f32>(0.0), vec3<f32>(1.0)), smooth_mask(ear_right_dist) * 0.9);

  // Eyes
  let eye_dirs = vec4<f32>(v.eye_dirs.x, v.eye_dirs.y, v.eye_dirs.z, v.eye_dirs.w);
  let eye_fov = vec4<f32>(v.eye_fov.x, v.eye_fov.y, v.eye_fov.z, v.eye_fov.w);
  let base_eye_radius = max(body_radius * 0.14, 1.2);
  // Core authority: agents.eye_sclera_srgb / eye_pupil_srgb, both added by
  // bd-sqji because no authority existed for either.
  let sclera_color = CORE_AGENT_EYE_SCLERA_SRGB;
  let pupil_color = CORE_AGENT_EYE_PUPIL_SRGB;
  for (var i: i32 = 0; i < 4; i = i + 1) {
    let angle = eye_dirs[i];
    let dir = vec2<f32>(cos(angle), sin(angle));
    let distance = body_radius * (0.4 + 0.35 * f32(i) / 4.0 + 0.25);
    let eye_center = dir * distance;
    var eye_radius = base_eye_radius * (0.65 + trait_eye * 0.35);
    eye_radius = clamp(eye_radius, 1.6, body_radius * 0.38);
    let eye_dist = circle_distance(local - eye_center, eye_radius);
    let eye_mask = smooth_mask(eye_dist);
    layer(&accum_rgb, &accum_alpha, sclera_color, eye_mask);

    let pupil_radius = eye_radius * (0.35 + clamp(eye_fov[i], 0.3, 3.0) * 0.12);
    let pupil_dist = circle_distance(local - eye_center, pupil_radius);
    layer(&accum_rgb, &accum_alpha, pupil_color, smooth_mask(pupil_dist));
  }

  // Temperature marker
  let temp_color = mix(vec3<f32>(0.20, 0.52, 0.96), vec3<f32>(0.98, 0.42, 0.18), temperature);
  let temp_center = vec2<f32>(0.0, -body_half_length * 0.25);
  let temp_radius = body_radius * 0.22;
  let temp_ring_dist = circle_distance(local - temp_center, temp_radius);
  let temp_ring = smooth_mask(temp_ring_dist) * 0.6;
  layer(&accum_rgb, &accum_alpha, temp_color, temp_ring * 0.6);

  // Sound arcs
  let vocal = max(sound_output, sound_level);
  if (vocal > 0.12) {
    let arc_origin = vec2<f32>(0.0, body_half_length + body_radius * 0.4);
    let arc_r1 = body_radius * (0.55 + vocal * 0.6);
    let arc_r2 = arc_r1 + body_radius * 0.35;
    let arc_color = vec3<f32>(0.95, 0.68, 0.32) * (0.6 + vocal * 0.6);
    let arc1 = circle_distance(local - arc_origin, arc_r1);
    let arc2 = circle_distance(local - arc_origin, arc_r2);
    layer(&accum_rgb, &accum_alpha, arc_color, smooth_mask(arc1) * 0.35);
    layer(&accum_rgb, &accum_alpha, arc_color, smooth_mask(arc2) * 0.25);
  }

  // Selection + indicator rim
  let sel_hover = step(0.5, selection) * (1.0 - step(1.5, selection));
  let sel_selected = step(1.5, selection);
  let selection_glow = sel_hover * 0.25 + sel_selected * 0.45;
  let rim_width = max(fwidth(body_dist), 0.001) * 1.5;
  // Same reversed-edge fix as smooth_mask: 1 - smoothstep(-w, 0, x), never smoothstep(0, -w, x).
  let rim = 1.0 - smoothstep(-rim_width, 0.0, body_dist + body_radius * 0.15);
  // Core authority: visual::agent_visual_params sets selection_rim_color from
  // the palette's agents.selection_rim_srgb. The local literal was pure white,
  // which is not that colour — the rim read as untinted here and cyan-tinted in
  // every core-authoritative backend (bd-rl1h). The glow scaling stays local:
  // it is an intensity, not a chroma, and core does not author it.
  let rim_color = CORE_AGENT_SELECTION_RIM_SRGB * (selection_glow + glow);
  layer(&accum_rgb, &accum_alpha, rim_color, rim);

  // Boost tint overlay
  if (boost > 0.05) {
    let boost_tint = vec3<f32>(0.98, 0.62, 0.32);
    layer(&accum_rgb, &accum_alpha, boost_tint, body_mask * boost * 0.2);
  }

  let alpha = clamp(accum_alpha, 0.0, 1.0);
  let rgb = clamp(accum_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(rgb, alpha);
}
"#;

struct PostFx {
    // Final composite (tonemap + vignette + fog + bloom composite)
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    src_layout: wgpu::BindGroupLayout, // src color + sampler + bloom texture
    src_bg: wgpu::BindGroup,
    params_layout: wgpu::BindGroupLayout,
    params_bg: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    color_format: wgpu::TextureFormat,

    // Bloom: extract brights to half-res, separable blur (ping-pong)
    bloom_extract_pipeline: wgpu::RenderPipeline,
    bloom_blur_pipeline: wgpu::RenderPipeline,
    bloom_src_layout: wgpu::BindGroupLayout, // single texture + sampler
    bloom_src_bg: wgpu::BindGroup,
    blur_params_h_bg: wgpu::BindGroup,
    blur_params_h_buf: wgpu::Buffer,
    blur_params_v_bg: wgpu::BindGroup,
    blur_params_v_buf: wgpu::Buffer,
    bloom_a: Option<wgpu::Texture>,
    bloom_a_view: Option<wgpu::TextureView>,
    bloom_b: Option<wgpu::Texture>,
    bloom_b_view: Option<wgpu::TextureView>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostParams {
    exposure: f32,
    vignette: f32,
    tonemap: u32,
    fxaa: u32,
    bloom_thresh: f32,
    bloom_intensity: f32,
    fog_density: f32,
    fog_enabled: u32,
    fog_color: [f32; 3],
    _pad0: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PostControls {
    exposure: f32,
    vignette: f32,
    tonemap: u32,
    fxaa: u32,
    bloom_enabled: bool,
    bloom_thresh: f32,
    bloom_intensity: f32,
    fog_enabled: bool,
    fog_density: f32,
    fog_color: [f32; 3],
}

impl Default for PostControls {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            vignette: 0.08,
            tonemap: 1,
            fxaa: 0,
            bloom_enabled: true,
            bloom_thresh: 0.8,
            bloom_intensity: 0.65,
            fog_enabled: false,
            fog_density: 0.6,
            fog_color: [0.6, 0.7, 0.8],
        }
    }
}

impl PostControls {
    fn from_env() -> Result<Self, ReadbackError> {
        let mut controls = Self::default();
        controls.exposure = env_control_f32(
            "SB_WGPU_EXPOSURE",
            controls.exposure,
            |value| value >= 0.0,
            "a finite non-negative number",
        )?;
        controls.vignette = env_control_f32(
            "SB_WGPU_VIGNETTE",
            controls.vignette,
            |value| (0.0..=1.0).contains(&value),
            "a finite number in 0..=1",
        )?;
        controls.tonemap = match std::env::var("SB_WGPU_TONEMAP") {
            Err(std::env::VarError::NotPresent) => controls.tonemap,
            Err(error) => {
                return Err(ReadbackError::Configuration(format!(
                    "SB_WGPU_TONEMAP is not valid Unicode: {error}"
                )));
            }
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "off" | "none" | "0" | "" => 0,
                "aces" | "filmic" => 1,
                "reinhard" => 2,
                "agx" => 3,
                _ => {
                    return Err(ReadbackError::Configuration(format!(
                        "SB_WGPU_TONEMAP={value:?}; expected off, aces, reinhard, or agx"
                    )));
                }
            },
        };
        controls.fxaa = parse_fxaa_env(controls.fxaa)?;
        controls.bloom_enabled = match std::env::var("SB_WGPU_BLOOM") {
            Err(std::env::VarError::NotPresent) => controls.bloom_enabled,
            Err(error) => {
                return Err(ReadbackError::Configuration(format!(
                    "SB_WGPU_BLOOM is not valid Unicode: {error}"
                )));
            }
            Ok(value) => parse_toggle_value(&value).ok_or_else(|| {
                ReadbackError::Configuration(format!(
                    "SB_WGPU_BLOOM={value:?}; expected a boolean toggle"
                ))
            })?,
        };
        controls.bloom_thresh = env_control_f32(
            "SB_WGPU_BLOOM_THRESH",
            controls.bloom_thresh,
            |value| value >= 0.0,
            "a finite non-negative number",
        )?;
        controls.bloom_intensity = env_control_f32(
            "SB_WGPU_BLOOM_INTENSITY",
            controls.bloom_intensity,
            |value| value >= 0.0,
            "a finite non-negative number",
        )?;

        let fog_mode = match std::env::var("SB_WGPU_FOG") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "off".to_owned(),
            Err(error) => {
                return Err(ReadbackError::Configuration(format!(
                    "SB_WGPU_FOG is not valid Unicode: {error}"
                )));
            }
        };
        let fog_mode = fog_mode.trim().to_ascii_lowercase();
        controls.fog_enabled = matches!(fog_mode.as_str(), "low" | "med" | "high");
        controls.fog_density = match fog_mode.as_str() {
            "low" => 0.6,
            "med" => 1.0,
            "high" => 1.6,
            "off" | "none" | "0" | "" => controls.fog_density,
            _ => {
                return Err(ReadbackError::Configuration(format!(
                    "SB_WGPU_FOG={fog_mode:?}; expected off, low, med, or high"
                )));
            }
        };
        match std::env::var("SB_WGPU_FOG_COLOR") {
            Ok(value) => {
                let color = parse_rgb(&value).ok_or_else(|| {
                    ReadbackError::Configuration(format!(
                        "SB_WGPU_FOG_COLOR={value:?}; expected three finite 0..=1 channels"
                    ))
                })?;
                if color
                    .iter()
                    .any(|channel| !channel.is_finite() || !(0.0..=1.0).contains(channel))
                {
                    return Err(ReadbackError::Configuration(format!(
                        "SB_WGPU_FOG_COLOR={value:?}; expected three finite 0..=1 channels"
                    )));
                }
                controls.fog_color = color;
            }
            Err(std::env::VarError::NotPresent) => {}
            Err(error) => {
                return Err(ReadbackError::Configuration(format!(
                    "SB_WGPU_FOG_COLOR is not valid Unicode: {error}"
                )));
            }
        }
        Ok(controls)
    }

    fn wants_post(self, selected_tonemap: Option<u32>) -> bool {
        self.exposure != 1.0
            || self.vignette != 0.0
            || self.fxaa != 0
            || selected_tonemap.unwrap_or(self.tonemap) != 0
            || self.bloom_enabled
            || self.fog_enabled
    }
}

fn parse_control_f32(
    name: &str,
    raw: &str,
    valid: impl FnOnce(f32) -> bool,
    expected: &str,
) -> Result<f32, ReadbackError> {
    let value = raw.parse::<f32>().map_err(|_| {
        ReadbackError::Configuration(format!("{name}={raw:?}; expected {expected}"))
    })?;
    if !value.is_finite() || !valid(value) {
        return Err(ReadbackError::Configuration(format!(
            "{name}={raw:?}; expected {expected}"
        )));
    }
    Ok(value)
}

fn env_control_f32(
    name: &str,
    default: f32,
    valid: impl FnOnce(f32) -> bool,
    expected: &str,
) -> Result<f32, ReadbackError> {
    match std::env::var(name) {
        Ok(raw) => parse_control_f32(name, &raw, valid, expected),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(ReadbackError::Configuration(format!(
            "{name} is not valid Unicode: {error}"
        ))),
    }
}

#[cfg(test)]
fn parse_toggle(value: Option<&str>, default: bool) -> bool {
    value.and_then(parse_toggle_value).unwrap_or(default)
}

fn parse_toggle_value(value: &str) -> Option<bool> {
    let value = value.trim();
    if let Ok(numeric) = value.parse::<u32>() {
        return Some(numeric != 0);
    }
    if value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("no")
    {
        Some(false)
    } else if value.eq_ignore_ascii_case("on")
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("yes")
    {
        Some(true)
    } else {
        None
    }
}

fn parse_rgb(value: &str) -> Option<[f32; 3]> {
    let mut channels = value.split(',').map(str::trim);
    let color = [
        channels.next()?.parse().ok()?,
        channels.next()?.parse().ok()?,
        channels.next()?.parse().ok()?,
    ];
    channels.next().is_none().then_some(color)
}

fn resolve_bloom_intensity(enabled: bool, configured: f32) -> f32 {
    if enabled { configured } else { 0.0 }
}

impl PostFx {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        src_view: &wgpu::TextureView,
        size: (u32, u32),
    ) -> Result<Self, ReadbackError> {
        // Final composite shader
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("postfx.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(POST_WGSL)),
        });
        // Final pass samples: src color (binding 0), sampler (1), bloom blurred (2)
        let src_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post.src_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // Placeholder bloom view: use source for now; real bloom bound during run()
        let src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post.src_bg"),
            layout: &src_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
            ],
        });
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post.params_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<PostParams>() as u64
                    ),
                },
                count: None,
            }],
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post.params_buf"),
            size: std::mem::size_of::<PostParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post.params_bg"),
            layout: &params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post.layout"),
            bind_group_layouts: &[&src_layout, &params_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("post.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &post_shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &post_shader,
                entry_point: Some("fs_post"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let (target, target_view) = create_color(device, format, size)?;

        // Bloom shaders and layouts
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom.wgsl"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BLOOM_WGSL)),
        });
        let bloom_src_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom.src_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bloom_src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom.src_bg"),
            layout: &bloom_src_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let blur_params_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bloom.blur_params_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(16),
                    },
                    count: None,
                }],
            });
        // Separate uniform buffers for the horizontal and vertical blur
        // directions: queue.write_buffer ordering means a single shared
        // buffer would make both passes sample the last-written direction.
        let blur_params_h_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom.blur_params_h_buf"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blur_params_h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom.blur_params_h_bg"),
            layout: &blur_params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: blur_params_h_buf.as_entire_binding(),
            }],
        });
        let blur_params_v_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bloom.blur_params_v_buf"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blur_params_v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom.blur_params_v_bg"),
            layout: &blur_params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: blur_params_v_buf.as_entire_binding(),
            }],
        });
        // Extract pipeline: src -> bloom_a (half res)
        let bloom_extract_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom.extract.layout"),
            bind_group_layouts: &[&bloom_src_layout, &params_layout],
            push_constant_ranges: &[],
        });
        let bloom_extract_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("bloom.extract.pipeline"),
                layout: Some(&bloom_extract_layout),
                vertex: wgpu::VertexState {
                    module: &bloom_shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &bloom_shader,
                    entry_point: Some("fs_extract"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        // Blur pipeline: bloom_a <-> bloom_b
        let bloom_blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom.blur.layout"),
            bind_group_layouts: &[&bloom_src_layout, &blur_params_layout],
            push_constant_ranges: &[],
        });
        let bloom_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bloom.blur.pipeline"),
            layout: Some(&bloom_blur_layout),
            vertex: wgpu::VertexState {
                module: &bloom_shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &bloom_shader,
                entry_point: Some("fs_blur"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Ok(Self {
            pipeline,
            sampler,
            src_layout,
            src_bg,
            params_layout,
            params_bg,
            params_buf,
            target,
            target_view,
            color_format: format,
            bloom_extract_pipeline,
            bloom_blur_pipeline,
            bloom_src_layout,
            bloom_src_bg,
            blur_params_h_bg,
            blur_params_h_buf,
            blur_params_v_bg,
            blur_params_v_buf,
            bloom_a: None,
            bloom_a_view: None,
            bloom_b: None,
            bloom_b_view: None,
        })
    }

    fn install_resize(
        &mut self,
        format: wgpu::TextureFormat,
        target: wgpu::Texture,
        target_view: wgpu::TextureView,
    ) {
        self.target = target;
        self.target_view = target_view;
        self.color_format = format;
        // Drop bloom targets; will be recreated lazily on next run
        self.bloom_a = None;
        self.bloom_a_view = None;
        self.bloom_b = None;
        self.bloom_b_view = None;
    }

    fn rebind(
        &mut self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
        bloom_view: &wgpu::TextureView,
    ) {
        self.src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post.src_bg.rebind"),
            layout: &self.src_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
            ],
        });
    }

    fn ensure_bloom_targets(
        &mut self,
        device: &wgpu::Device,
        _format: wgpu::TextureFormat,
        full: (u32, u32),
    ) {
        if self.bloom_a.is_some() && self.bloom_b.is_some() {
            return;
        }
        let half = (full.0.max(1) / 2).max(1);
        let half_h = (full.1.max(1) / 2).max(1);
        let make = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: half,
                    height: half_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm, // linear for correct blur/composite
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let a = make("post.bloom.a");
        let b = make("post.bloom.b");
        let av = a.create_view(&wgpu::TextureViewDescriptor::default());
        let bv = b.create_view(&wgpu::TextureViewDescriptor::default());
        self.bloom_a = Some(a);
        self.bloom_a_view = Some(av);
        self.bloom_b = Some(b);
        self.bloom_b_view = Some(bv);
    }

    fn bind_bloom_src(&mut self, device: &wgpu::Device, src_view: &wgpu::TextureView) {
        self.bloom_src_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom.src_bg.rebind"),
            layout: &self.bloom_src_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        size: (u32, u32),
        selected_tonemap: Option<u32>,
        controls: &PostControls,
    ) {
        let params = PostParams {
            exposure: controls.exposure,
            vignette: controls.vignette,
            tonemap: selected_tonemap.unwrap_or(controls.tonemap),
            fxaa: controls.fxaa,
            bloom_thresh: controls.bloom_thresh,
            bloom_intensity: resolve_bloom_intensity(
                controls.bloom_enabled,
                controls.bloom_intensity,
            ),
            fog_density: controls.fog_density,
            fog_enabled: u32::from(controls.fog_enabled),
            fog_color: controls.fog_color,
            _pad0: 0.0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        // Recreate params bind group (demonstrates use of params_layout; allows for dynamic layout changes in future)
        self.params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post.params_bg.rebind"),
            layout: &self.params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.params_buf.as_entire_binding(),
            }],
        });

        // Bloom pass chain if enabled
        let mut bloom_view_opt: Option<wgpu::TextureView> = None;
        if controls.bloom_enabled {
            self.ensure_bloom_targets(device, self.color_format, size);
            // Create fresh local views to avoid borrowing self across mutable calls
            let a_view_local = self
                .bloom_a
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor::default());
            let b_view_local = self
                .bloom_b
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor::default());
            // Extract brights from src -> A
            self.bind_bloom_src(device, src);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom.extract"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &a_view_local,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.bloom_extract_pipeline);
                pass.set_bind_group(0, &self.bloom_src_bg, &[]);
                pass.set_bind_group(1, &self.params_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            // Blur A -> B (horizontal)
            let dir_h: [f32; 4] = [1.0 / (size.0.max(1) as f32 * 0.5), 0.0, 0.0, 0.0];
            queue.write_buffer(&self.blur_params_h_buf, 0, bytemuck::bytes_of(&dir_h));
            self.bind_bloom_src(device, &a_view_local);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom.blur.h"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &b_view_local,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.bloom_blur_pipeline);
                pass.set_bind_group(0, &self.bloom_src_bg, &[]);
                pass.set_bind_group(1, &self.blur_params_h_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            // Blur B -> A (vertical)
            let dir_v: [f32; 4] = [0.0, 1.0 / (size.1.max(1) as f32 * 0.5), 0.0, 0.0];
            queue.write_buffer(&self.blur_params_v_buf, 0, bytemuck::bytes_of(&dir_v));
            self.bind_bloom_src(device, &b_view_local);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom.blur.v"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &a_view_local,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.bloom_blur_pipeline);
                pass.set_bind_group(0, &self.bloom_src_bg, &[]);
                pass.set_bind_group(1, &self.blur_params_v_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            bloom_view_opt = Some(a_view_local);
        }

        // Final composite: bind src + bloom (or src placeholder) and draw
        let bloom_view_ref = bloom_view_opt.as_ref().unwrap_or(src);
        self.rebind(device, src, bloom_view_ref);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("post.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.src_bg, &[]);
        pass.set_bind_group(1, &self.params_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

const POST_WGSL: &str = r#"
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  // Fullscreen triangle
  let x = f32((vid << 1u) & 2u);
  let y = f32(vid & 2u);
  return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
struct Params {
  exposure: f32, vignette: f32, tonemap: u32, fxaa: u32,
  bloom_thresh: f32, bloom_intensity: f32, fog_density: f32, fog_enabled: u32,
  fog_color: vec3<f32>, _pad0: f32,
};
@group(1) @binding(0) var<uniform> params: Params;

fn aces_tonemap(col: vec3<f32>) -> vec3<f32> {
  // Fitted ACES curve. Coefficients MUST match the CPU aces_fitted in
  // scriptbots-render (Stephen Hill fit); an earlier shader divided by (a-1.0)=1.51
  // instead of the correct c=2.43 and crushed the shoulder relative to the CPU path.
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  let numerator = col * (a * col + b);
  let denom = col * (c * col + d) + e;
  return clamp(numerator / denom, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn reinhard_tonemap(c: vec3<f32>) -> vec3<f32> {
  return clamp(c / (vec3<f32>(1.0) + c), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_post(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let tex_size = vec2<f32>(textureDimensions(src_tex));
  // Pixel coords from builtin position -> normalize to [0,1]
  let uv01 = pos.xy / tex_size;
  var col = textureSampleLevel(src_tex, src_smp, uv01, 0.0);
  // FXAA stub: none (placeholder for later)
  // Tonemap (0 = passthrough, 1 = aces, 2 = reinhard) + mild vignette
  var rgb = col.rgb * params.exposure;
  if (params.tonemap == 1u) {
    rgb = aces_tonemap(rgb);
  } else if (params.tonemap == 2u) {
    rgb = reinhard_tonemap(rgb);
  } else if (params.tonemap == 3u) {
    // AgX parity with the CPU tonemap_rgb Agx path: pre-compress x/(x+0.3),
    // then the same fitted-ACES curve.
    rgb = aces_tonemap(rgb / (rgb + vec3<f32>(0.3)));
  } else {
    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  }
  let p = uv01 * 2.0 - 1.0;
  let vign = clamp(1.0 - dot(p, p) * params.vignette, 0.85, 1.0);
  rgb *= vign;
  // Height-fog (screen-space Y proxy)
  if (params.fog_enabled != 0u) {
    let h = 1.0 - uv01.y; // bottom-heavy fog (denser near bottom)
    let fog_f = clamp(1.0 - exp(-params.fog_density * h), 0.0, 1.0);
    rgb = mix(rgb, params.fog_color, fog_f);
  }
  // Bloom composite (additive)
  if (params.bloom_intensity > 0.0) {
    let b = textureSampleLevel(bloom_tex, src_smp, uv01, 0.0).rgb;
    rgb = clamp(rgb + b * params.bloom_intensity, vec3<f32>(0.0), vec3<f32>(1.0));
  }
  return vec4<f32>(rgb, 1.0);
}
"#;

// Bloom helpers (extract + separable blur)
const BLOOM_WGSL: &str = r#"
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  let x = f32((vid << 1u) & 2u);
  let y = f32(vid & 2u);
  return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;
struct Params { exposure: f32, vignette: f32, tonemap: u32, fxaa: u32, bloom_thresh: f32, bloom_intensity: f32, fog_density: f32, fog_enabled: u32, fog_color: vec3<f32>, _pad0: f32 };
@group(1) @binding(0) var<uniform> params: Params;

@fragment
fn fs_extract(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let tex_size = vec2<f32>(textureDimensions(src_tex));
  let uv = pos.xy / tex_size;
  let c = textureSampleLevel(src_tex, src_smp, uv, 0.0).rgb;
  let luma = max(c.r, max(c.g, c.b));
  let m = smoothstep(params.bloom_thresh, params.bloom_thresh + 0.1, luma);
  return vec4<f32>(c * m, 1.0);
}

struct BlurParams { dir: vec2<f32>, _pad: vec2<f32> };
@group(1) @binding(0) var<uniform> blur: BlurParams;

@fragment
fn fs_blur(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let tex_size = vec2<f32>(textureDimensions(src_tex));
  let uv = pos.xy / tex_size;
  // 5-tap gaussian (weights approximate)
  let w0 = 0.227027;
  let w1 = 0.316216;
  let w2 = 0.070270;
  let off1 = blur.dir * 1.384615;
  let off2 = blur.dir * 3.230769;
  var c = textureSampleLevel(src_tex, src_smp, uv, 0.0).rgb * w0;
  c += textureSampleLevel(src_tex, src_smp, uv + off1, 0.0).rgb * w1;
  c += textureSampleLevel(src_tex, src_smp, uv - off1, 0.0).rgb * w1;
  c += textureSampleLevel(src_tex, src_smp, uv + off2, 0.0).rgb * w2;
  c += textureSampleLevel(src_tex, src_smp, uv - off2, 0.0).rgb * w2;
  return vec4<f32>(c, 1.0);
}
"#;

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}
