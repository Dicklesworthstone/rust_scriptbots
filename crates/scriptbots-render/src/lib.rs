//! GPUI rendering layer for ScriptBots.

mod camera;
/// Headless GPUI HUD capture (bd-abu3). Test-only: it depends on GPUI's
/// `test-support` feature, which is a dev-dependency and must not reach production.
#[cfg(test)]
mod hud_capture;
mod vfx;

use camera::{Camera, CameraSnapshot, ViewLayout};
use gpui::{
    AlignItems, App, Background, Bounds, Context, Corners, Div, FocusHandle, KeyDownEvent,
    Keystroke, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PathBuilder, Pixels,
    Point, QuitMode, RenderImage, Rgba, ScrollDelta, ScrollWheelEvent, SharedString,
    StyleRefinement, Window, WindowBounds, WindowOptions, canvas, div, fill, linear_color_stop,
    linear_gradient, point, prelude::*, px, rgb, size,
};
use gpui_platform::application;
use scriptbots_core::PresetKind;
use scriptbots_core::attribution::{
    AttributionMethod, EffectiveOutput, OutputExplanation, explain_outputs,
};
use scriptbots_core::narrative::{
    EventKind as NarrativeEventKind, EventRecord as NarrativeEventRecord,
};
use scriptbots_core::visual::{
    self, AgentVisualInput, AgentVisualParams, SplatInput, TerrainSurfaceInput, VisualSelection,
    WorldVisualEvent,
};
use scriptbots_core::{
    AccessibilityPalette, ActivationEdge, ActivationLayer, AgentColumns, AgentId, AgentRuntime,
    AgentUid, BrainActivations, BrainInspectionClientId, BrainInspectionRequest,
    BrainInspectionRevision, BrainInspectionUnavailable, ControlCommand, ControlDisposition,
    FoodGrid, Generation, IndicatorState, MutationRates, NUM_EYES, OutputChannel, OutputsExt,
    Position, RenderFogMode, RenderQuality, RenderTonemapMode, SENSOR_LAYOUT, ScriptBotsConfig,
    SelectedBrainTelemetryOutcome, SelectionMode, SelectionState, SelectionUpdate,
    SensorAttribution, SensorKind, SimulationCommand, TerrainKind, TerrainLayer, TerrainTile,
    TickSummary, TraitModifiers, Velocity, WorldState, WorldStepDriver, apply_control_command,
    tier_features,
};
use scriptbots_storage::{AnalyticsSnapshotProvider, MetricReading};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, VecDeque},
    f32::consts::{FRAC_PI_2, FRAC_PI_4, PI},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering as AtomicOrdering},
    },
    time::{Duration, Instant},
};

#[cfg(feature = "audio")]
use kira::{
    DefaultBackend,
    frame::Frame,
    manager::{AudioManager, AudioManagerSettings},
    sound::static_sound::{StaticSoundData, StaticSoundSettings},
};

use image::{Frame as ImageFrame, ImageBuffer, Rgba as ImgRgba};
use tracing::{debug, error, info, warn};

#[cfg(feature = "world_wgpu")]
pub mod world_compositor {
    use super::*;
    use scriptbots_core::{TerrainKind, WorldState};
    use scriptbots_world_gfx::{
        ReadbackError, ReadbackView, WorldRenderer, WorldSnapshot as GfxSnapshot,
    };

    pub struct GpuiImage {
        size: (u32, u32),
        // Raw RGBA buffer we reuse to avoid realloc; GPUI will copy per frame
        rgba: Vec<u8>,
        bytes_per_row: u32,
        // Previous frame for diff-based present
        prev: Vec<u8>,
    }

    impl GpuiImage {
        fn new(size: (u32, u32), bytes_per_row: u32) -> Self {
            let cap = (bytes_per_row as usize) * (size.1 as usize);
            Self {
                size,
                rgba: vec![0u8; cap],
                bytes_per_row,
                prev: vec![0u8; cap],
            }
        }
        fn ensure(&mut self, size: (u32, u32), bytes_per_row: u32) {
            if self.size != size || self.bytes_per_row != bytes_per_row {
                self.size = size;
                self.bytes_per_row = bytes_per_row;
                let cap = (bytes_per_row as usize) * (size.1 as usize);
                self.rgba.resize(cap, 0u8);
                self.prev.resize(cap, 0u8);
            }
        }
        fn upload_from_readback(&mut self, view: &ReadbackView) {
            self.ensure((view.width, view.height), view.bytes_per_row);
            let new_len = (self.bytes_per_row as usize) * (self.size.1 as usize);
            if self.prev.len() != new_len {
                self.prev.resize(new_len, 0);
            }
            // swap buffers: previous frame lands in self.prev; current buffer becomes the new target
            std::mem::swap(&mut self.prev, &mut self.rgba);
            let src = view.bytes();
            self.rgba[..src.len()].copy_from_slice(src);
        }

        // Paint the full image using row-run coalescing (viewport-scaled)
        fn paint_full(&self, bounds: Bounds<Pixels>, window: &mut Window) {
            let vw = f32::from(bounds.size.width).max(1.0);
            let vh = f32::from(bounds.size.height).max(1.0);
            let img_w = self.size.0.max(1);
            let img_h = self.size.1.max(1);
            let row = self.bytes_per_row as usize;

            let sx = (vw / img_w as f32).max(0.0001);
            let sy = (vh / img_h as f32).max(0.0001);

            // Background to avoid tiny gaps from float math
            window.paint_quad(fill(bounds, Background::from(rgb(0x000000))));

            for y in 0..(img_h as usize) {
                let y0 = f32::from(bounds.origin.y) + (y as f32) * sy;
                let y1 = (y0 + sy).min(f32::from(bounds.origin.y) + vh);
                let mut x = 0usize;
                while x < (img_w as usize) {
                    let off = y * row + x * 4;
                    let r0 = self.rgba[off];
                    let g0 = self.rgba[off + 1];
                    let b0 = self.rgba[off + 2];
                    let a0 = self.rgba[off + 3];
                    let mut run = x + 1;
                    while run < (img_w as usize) {
                        let o = y * row + run * 4;
                        if self.rgba[o] == r0
                            && self.rgba[o + 1] == g0
                            && self.rgba[o + 2] == b0
                            && self.rgba[o + 3] == a0
                        {
                            run += 1;
                        } else {
                            break;
                        }
                    }
                    let x0 = f32::from(bounds.origin.x) + (x as f32) * sx;
                    let x1 = (x0 + (run - x) as f32 * sx).min(f32::from(bounds.origin.x) + vw);
                    let cell_bounds = Bounds::new(
                        point(px(x0), px(y0)),
                        size(px((x1 - x0).max(0.0)), px((y1 - y0).max(0.0))),
                    );
                    window.paint_quad(fill(
                        cell_bounds,
                        Background::from(Rgba {
                            r: (r0 as f32) / 255.0,
                            g: (g0 as f32) / 255.0,
                            b: (b0 as f32) / 255.0,
                            a: (a0 as f32) / 255.0,
                        }),
                    ));
                    x = run;
                }
            }
        }

        // Paint only changed runs vs. previous frame
        fn paint_diff(&self, bounds: Bounds<Pixels>, window: &mut Window) {
            let vw = f32::from(bounds.size.width).max(1.0);
            let vh = f32::from(bounds.size.height).max(1.0);
            let img_w = self.size.0.max(1);
            let img_h = self.size.1.max(1);
            let row = self.bytes_per_row as usize;

            let sx = (vw / img_w as f32).max(0.0001);
            let sy = (vh / img_h as f32).max(0.0001);

            // Do not clear background; unchanged regions remain
            for y in 0..(img_h as usize) {
                let mut x = 0usize;
                while x < (img_w as usize) {
                    let off = y * row + x * 4;
                    let changed = self.rgba[off..off + 4] != self.prev[off..off + 4];
                    if !changed {
                        x += 1;
                        continue;
                    }
                    let r0 = self.rgba[off];
                    let g0 = self.rgba[off + 1];
                    let b0 = self.rgba[off + 2];
                    let a0 = self.rgba[off + 3];
                    let mut run = x + 1;
                    while run < (img_w as usize) {
                        let o = y * row + run * 4;
                        if self.rgba[o..o + 4] == self.prev[o..o + 4] {
                            break;
                        }
                        // extend run only while color matches to keep draw calls minimal
                        if self.rgba[o] == r0
                            && self.rgba[o + 1] == g0
                            && self.rgba[o + 2] == b0
                            && self.rgba[o + 3] == a0
                        {
                            run += 1;
                        } else {
                            break;
                        }
                    }
                    let y0 = f32::from(bounds.origin.y) + (y as f32) * sy;
                    let y1 = (y0 + sy).min(f32::from(bounds.origin.y) + vh);
                    let x0 = f32::from(bounds.origin.x) + (x as f32) * sx;
                    let x1 = (x0 + (run - x) as f32 * sx).min(f32::from(bounds.origin.x) + vw);
                    let cell_bounds = Bounds::new(
                        point(px(x0), px(y0)),
                        size(px((x1 - x0).max(0.0)), px((y1 - y0).max(0.0))),
                    );
                    window.paint_quad(fill(
                        cell_bounds,
                        Background::from(Rgba {
                            r: (r0 as f32) / 255.0,
                            g: (g0 as f32) / 255.0,
                            b: (b0 as f32) / 255.0,
                            a: (a0 as f32) / 255.0,
                        }),
                    ));
                    x = run;
                }
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct ReadbackDigest {
        width: u32,
        height: u32,
        stride: u32,
        checksum: u64,
        non_zero: usize,
        populated_bytes: usize,
        min_byte: u8,
        max_byte: u8,
        varied_rgb: bool,
        sample_rgba: [u8; 4],
    }

    impl ReadbackDigest {
        fn is_visually_blank(&self) -> bool {
            if self.populated_bytes == 0 {
                return true;
            }
            if self.non_zero == 0 {
                return true;
            }
            !self.varied_rgb
        }
    }

    fn rgba8_is_visually_blank(bytes: &[u8]) -> bool {
        let mut pixels = bytes.as_chunks::<4>().0.iter();
        let Some(first) = pixels.next() else {
            return true;
        };
        let first_rgb = [first[0], first[1], first[2]];
        let mut has_nonzero_rgb = first_rgb.iter().any(|channel| *channel != 0);
        let mut varied_rgb = false;
        for pixel in pixels {
            let rgb = [pixel[0], pixel[1], pixel[2]];
            has_nonzero_rgb |= rgb.iter().any(|channel| *channel != 0);
            varied_rgb |= rgb != first_rgb;
        }
        !has_nonzero_rgb || !varied_rgb
    }

    /// Honest outcome of one compositor request.
    ///
    /// Rate limiting is the only non-rendering success: it deliberately reuses
    /// a previously completed frame. Every unavailable or failed GPU stage is a
    /// typed [`ReadbackError`] so callers can choose and report their fallback.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RenderSnapshotOutcome {
        Rendered,
        ReusedRateLimitedFrame,
    }

    fn readback_stats(view: &ReadbackView) -> ReadbackDigest {
        let stride = view.bytes_per_row as usize;
        let row_bytes = (view.width as usize) * 4;
        let src = view.bytes();
        let mut checksum = 0u64;
        let mut non_zero = 0usize;
        let mut min_byte = u8::MAX;
        let mut max_byte = u8::MIN;
        let mut populated = 0usize;
        let mut first_rgb: Option<[u8; 3]> = None;
        let mut varied_rgb = false;
        for y in 0..(view.height as usize) {
            let start = y * stride;
            let end = start.saturating_add(row_bytes).min(src.len());
            let row = &src[start..end];
            populated = populated.saturating_add(row.len());
            for pixel in row.as_chunks::<4>().0 {
                for &byte in pixel {
                    checksum = checksum.wrapping_add(u64::from(byte));
                }
                let rgb = [pixel[0], pixel[1], pixel[2]];
                for byte in rgb {
                    if byte != 0 {
                        non_zero += 1;
                    }
                    min_byte = min_byte.min(byte);
                    max_byte = max_byte.max(byte);
                }
                if let Some(first) = first_rgb {
                    varied_rgb |= rgb != first;
                } else {
                    first_rgb = Some(rgb);
                }
            }
        }
        if populated == 0 {
            min_byte = 0;
            max_byte = 0;
        }
        let mut sample = [0u8; 4];
        if view.width > 0 && view.height > 0 {
            let sample_x = (view.width / 2) as usize;
            let sample_y = (view.height / 2) as usize;
            let offset = sample_y
                .saturating_mul(stride)
                .saturating_add(sample_x.saturating_mul(4));
            if offset + 4 <= src.len() {
                sample.copy_from_slice(&src[offset..offset + 4]);
            }
        }
        ReadbackDigest {
            width: view.width,
            height: view.height,
            stride: view.bytes_per_row,
            checksum,
            non_zero,
            populated_bytes: populated,
            min_byte,
            max_byte,
            varied_rgb,
            sample_rgba: sample,
        }
    }

    pub struct Compositor {
        renderer: Option<WorldRenderer>,
        image: Option<GpuiImage>,
        adapter: Option<wgpu::Adapter>,
        cam_scale: f32,
        cam_offset: (f32, f32),
        last_submit: Option<std::time::Instant>,
        min_interval: f32, // seconds; 0 = uncapped
        render_scale: f32, // 0<scale<=1; offscreen resolution scale
        allow_software_adapter: bool,
        adapter_failure: Option<String>,
        adapter_failure_reported: bool,
        // Optional frame capture
        save_enabled: bool,
        save_dir: std::path::PathBuf,
        save_every: u32,
        save_counter: u64,
        save_prefix: String,
        last_digest: Option<ReadbackDigest>,
        /// Whether the adapter that actually rasterized is a CPU surrogate.
        ///
        /// The compositor has always known this; it simply never told anyone. A
        /// capture test that cannot distinguish a live GPU framebuffer from an
        /// llvmpipe raster is not testing the thing its name claims.
        adapter_is_software: bool,
    }

    impl Compositor {
        pub fn new() -> Self {
            let max_fps = std::env::var("SB_WGPU_MAX_FPS")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(60.0);
            let min_interval = if max_fps > 0.0 {
                (1.0 / max_fps).max(0.001)
            } else {
                0.0
            };
            let render_scale = std::env::var("SB_WGPU_RES_SCALE")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .map(|v| v.clamp(0.25, 1.0))
                .unwrap_or(1.0);
            // Frame capture controls (opt-in)
            let save_enabled = env_flag("SB_WGPU_SAVE_FRAMES");
            let save_dir = std::env::var("SB_WGPU_SAVE_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| std::path::PathBuf::from("frames"));
            let save_every = std::env::var("SB_WGPU_SAVE_EVERY")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(1);
            let save_prefix =
                std::env::var("SB_WGPU_SAVE_PREFIX").unwrap_or_else(|_| "frame".to_string());
            Self {
                renderer: None,
                image: None,
                adapter: None,
                cam_scale: 1.0,
                cam_offset: (0.0, 0.0),
                last_submit: None,
                min_interval,
                render_scale,
                allow_software_adapter: env_flag("SB_WGPU_ALLOW_SOFTWARE_ADAPTER")
                    || env_flag("SB_WGPU_ALLOW_CPU"),
                adapter_failure: None,
                adapter_failure_reported: false,
                save_enabled,
                save_dir,
                save_every,
                save_counter: 0,
                save_prefix,
                last_digest: None,
                adapter_is_software: false,
            }
        }

        #[cfg(test)]
        pub(super) fn new_for_capture_test(
            save_dir: std::path::PathBuf,
            save_prefix: String,
        ) -> Self {
            let mut compositor = Self::new();
            compositor.min_interval = 0.0;
            compositor.render_scale = 1.0;
            compositor.allow_software_adapter = true;
            compositor.save_enabled = true;
            compositor.save_dir = save_dir;
            compositor.save_every = 1;
            compositor.save_counter = 0;
            compositor.save_prefix = save_prefix;
            compositor
        }

        #[cfg(test)]
        pub(super) fn adapter_failure(&self) -> Option<&str> {
            self.adapter_failure.as_deref()
        }

        /// Did a CPU surrogate rasterize the last frame?
        ///
        /// A test that asserts "the GPU drew this" while an llvmpipe adapter did
        /// the drawing is asserting a falsehood, however green it looks.
        #[cfg(test)]
        pub(super) fn adapter_is_software(&self) -> bool {
            self.adapter_is_software
        }

        pub fn set_camera_params(&mut self, scale: f32, offset: (f32, f32)) {
            self.cam_scale = scale;
            self.cam_offset = offset;
            if let Some(r) = self.renderer.as_mut() {
                r.set_camera(scale, offset);
            }
        }

        fn ensure_renderer(&mut self, size: (u32, u32)) -> Result<(), ReadbackError> {
            if self.adapter_failure.is_some() {
                return Err(ReadbackError::AdapterUnavailable);
            }
            if self.adapter.is_none() {
                // Create a headless adapter suitable for offscreen rendering
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
                let future = async {
                    let res = instance
                        .request_adapter(&wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::HighPerformance,
                            compatible_surface: None,
                            force_fallback_adapter: false,
                        })
                        .await;
                    res.map_err(|_| ReadbackError::AdapterUnavailable)
                };
                let adapter = match pollster::block_on(future) {
                    Ok(adapter) => adapter,
                    Err(error) => {
                        self.record_adapter_failure(error.to_string());
                        return Err(error);
                    }
                };
                let info = adapter.get_info();
                info!(
                    adapter = %info.name,
                    device_type = ?info.device_type,
                    backend = ?info.backend,
                    vendor = format!("{:#x}", info.vendor),
                    "Selected wgpu adapter"
                );
                let name_lower = info.name.to_ascii_lowercase();
                let is_software = matches!(
                    info.device_type,
                    wgpu::DeviceType::Cpu | wgpu::DeviceType::Other
                ) || name_lower.contains("llvmpipe")
                    || name_lower.contains("lavapipe")
                    || name_lower.contains("swiftshader");
                if is_software && !self.allow_software_adapter {
                    let reason = format!(
                        "wgpu adapter \"{}\" (vendor {:#x}) is software-only; using CPU canvas",
                        info.name, info.vendor
                    );
                    self.record_adapter_failure(reason.clone());
                    return Err(ReadbackError::AdapterUnavailable);
                } else if is_software {
                    self.adapter_is_software = true;
                    warn!(
                        adapter = %info.name,
                        vendor = format!("{:#x}", info.vendor),
                        "wgpu adapter is software-only; continuing due to SB_WGPU_ALLOW_SOFTWARE_ADAPTER"
                    );
                }
                self.adapter = Some(adapter);
            }
            if self.renderer.is_none() {
                let adapter = self.adapter.as_ref().unwrap();
                let future = scriptbots_world_gfx::WorldRenderer::new(adapter, size);
                let mut renderer = match pollster::block_on(future) {
                    Ok(renderer) => renderer,
                    Err(error) => return Err(error),
                };
                renderer.set_camera(self.cam_scale, self.cam_offset);
                self.renderer = Some(renderer);
            }
            Ok(())
        }

        #[allow(dead_code)]
        pub fn resize(&mut self, size: (u32, u32)) -> Result<(), ReadbackError> {
            let Some(renderer) = self.renderer.as_mut() else {
                return Err(ReadbackError::AdapterUnavailable);
            };
            renderer.resize(size)
        }

        pub fn render_snapshot(
            &mut self,
            snapshot: &GfxSnapshot,
            target_size: (u32, u32),
        ) -> Result<RenderSnapshotOutcome, ReadbackError> {
            if target_size.0 == 0 || target_size.1 == 0 {
                self.invalidate_presentable_frame();
                return Err(ReadbackError::ZeroDimensions {
                    width: target_size.0,
                    height: target_size.1,
                });
            }
            // Rate limiting is explicit and may only reuse a successfully completed image.
            if self.min_interval > 0.0
                && let Some(last) = self.last_submit
                && last.elapsed().as_secs_f32() < self.min_interval
                && self.has_presentable_frame()
            {
                return Ok(RenderSnapshotOutcome::ReusedRateLimitedFrame);
            }
            let render_size = if self.render_scale < 0.9999 {
                (
                    (((target_size.0 as f32) * self.render_scale) as u32)
                        .max(1)
                        .min(target_size.0),
                    (((target_size.1 as f32) * self.render_scale) as u32)
                        .max(1)
                        .min(target_size.1),
                )
            } else {
                (target_size.0.max(1), target_size.1.max(1))
            };
            if let Err(error) = self.ensure_renderer(render_size) {
                self.invalidate_presentable_frame();
                if matches!(error, ReadbackError::AdapterUnavailable) {
                    self.maybe_report_adapter_failure();
                }
                return Err(error);
            }
            let Some(r) = self.renderer.as_mut() else {
                self.invalidate_presentable_frame();
                return Err(ReadbackError::Device(
                    "compositor initialized without a world renderer".to_owned(),
                ));
            };
            // When rendering at a reduced offscreen resolution, scale the camera mapping
            // so CPU culling and shader NDC math remain consistent with the smaller viewport.
            let rs = if self.render_scale < 0.9999 {
                self.render_scale
            } else {
                1.0
            };
            let effective_scale = self.cam_scale * rs;
            let effective_offset = (self.cam_offset.0 * rs, self.cam_offset.1 * rs);
            r.set_camera(effective_scale, effective_offset);
            if let Err(error) = r.resize(render_size) {
                self.invalidate_presentable_frame();
                return Err(error);
            }
            let frame = match r.render(snapshot) {
                Ok(frame) => frame,
                Err(error) => {
                    self.invalidate_presentable_frame();
                    return Err(error);
                }
            };
            if let Err(error) = r.copy_to_readback(&frame) {
                self.invalidate_presentable_frame();
                return Err(error);
            }
            let view = match r.mapped_rgba() {
                Ok(view) => view,
                Err(error) => {
                    self.invalidate_presentable_frame();
                    return Err(error);
                }
            };
            let digest = readback_stats(&view);
            if digest.is_visually_blank() {
                self.invalidate_presentable_frame();
                return Err(ReadbackError::Blank);
            }
            self.last_digest = Some(digest);
            // Lazy-initialize image with known dimensions; avoid stale size from previous runs.
            if self.image.is_none() {
                self.image = Some(GpuiImage::new(
                    (view.width, view.height),
                    view.bytes_per_row,
                ));
            }
            if let Some(img) = self.image.as_mut() {
                img.ensure((view.width, view.height), view.bytes_per_row);
                img.upload_from_readback(&view);
            }
            // `ReadbackView` borrows the renderer's mapped buffer. Explicit
            // artifact capture is rare, so copy only on that opt-in path and
            // release the mapping before mutably borrowing compositor policy.
            let save_payload = self.save_enabled.then(|| {
                (
                    view.bytes().to_vec(),
                    view.width,
                    view.height,
                    view.bytes_per_row as usize,
                )
            });
            drop(view);
            if let Some((bytes, width, height, stride)) = save_payload
                && let Err(error) =
                    self.save_rgba_if_requested(&bytes, width, height, stride, &digest)
            {
                self.invalidate_presentable_frame();
                return Err(error);
            }
            if env_flag("SB_WGPU_READBACK_CHECKSUM") {
                tracing::info!(
                    width = digest.width,
                    height = digest.height,
                    stride = digest.stride,
                    checksum = digest.checksum,
                    checksum_hex = format!("{:016x}", digest.checksum),
                    non_zero_bytes = digest.non_zero,
                    populated_bytes = digest.populated_bytes,
                    min_byte = digest.min_byte,
                    max_byte = digest.max_byte,
                    varied_rgb = digest.varied_rgb,
                    sample_r = digest.sample_rgba[0],
                    sample_g = digest.sample_rgba[1],
                    sample_b = digest.sample_rgba[2],
                    sample_a = digest.sample_rgba[3],
                    "wgpu readback checksum"
                );
            }
            self.last_submit = Some(std::time::Instant::now());
            if env_flag("SB_WGPU_LOG_VIS") {
                tracing::info!(
                    width = digest.width,
                    height = digest.height,
                    stride = digest.stride,
                    checksum = digest.checksum,
                    non_zero_bytes = digest.non_zero,
                    populated_bytes = digest.populated_bytes,
                    min_byte = digest.min_byte,
                    max_byte = digest.max_byte,
                    varied_rgb = digest.varied_rgb,
                    sample_r = digest.sample_rgba[0],
                    sample_g = digest.sample_rgba[1],
                    sample_b = digest.sample_rgba[2],
                    sample_a = digest.sample_rgba[3],
                    "wgpu readback mapped"
                );
            }
            Ok(RenderSnapshotOutcome::Rendered)
        }

        fn has_presentable_frame(&self) -> bool {
            self.image.is_some()
                && self
                    .last_digest
                    .as_ref()
                    .is_some_and(|digest| !digest.is_visually_blank())
        }

        fn invalidate_presentable_frame(&mut self) {
            self.last_digest = None;
            self.last_submit = None;
        }

        fn record_adapter_failure(&mut self, reason: String) {
            self.adapter_failure = Some(reason);
            self.adapter_failure_reported = false;
            self.adapter = None;
            self.renderer = None;
        }

        fn maybe_report_adapter_failure(&mut self) {
            if self.adapter_failure_reported {
                return;
            }
            if let Some(reason) = self.adapter_failure.as_ref() {
                warn!(
                    %reason,
                    "Disabling wgpu compositor; falling back to CPU canvas renderer"
                );
                self.adapter_failure_reported = true;
            }
        }

        pub fn paint_world(&mut self, bounds: Bounds<Pixels>, window: &mut Window) -> bool {
            let Some(digest) = self.last_digest.as_ref() else {
                return false;
            };
            if digest.is_visually_blank() {
                if env_flag("SB_WGPU_LOG_VIS") {
                    tracing::warn!(
                        checksum = digest.checksum,
                        non_zero = digest.non_zero,
                        min_byte = digest.min_byte,
                        max_byte = digest.max_byte,
                        varied_rgb = digest.varied_rgb,
                        "wgpu readback appears blank; falling back to CPU canvas"
                    );
                }
                return false;
            }
            if let Some(img) = &self.image {
                let mode = std::env::var("SB_WGPU_PRESENT_MODE")
                    .ok()
                    .or_else(|| Some("full".to_string()));
                match mode.as_deref() {
                    Some("full") => img.paint_full(bounds, window),
                    _ => img.paint_diff(bounds, window),
                }
                true
            } else {
                // Diagnostic fallback: draw a subtle placeholder so we know paint was invoked
                if env_flag("SB_WGPU_LOG_VIS") {
                    tracing::info!("wgpu image not available; painting placeholder");
                }
                self.last_digest = None;
                window.paint_quad(fill(bounds, Background::from(rgb(0x091220))));
                false
            }
        }

        pub(crate) fn render_scale_factor(&self) -> f32 {
            self.render_scale
        }

        fn save_rgba_if_requested(
            &mut self,
            src: &[u8],
            width: u32,
            height: u32,
            stride: usize,
            digest: &ReadbackDigest,
        ) -> Result<(), ReadbackError> {
            if !self.save_enabled {
                return Ok(());
            }
            self.save_counter = self.save_counter.saturating_add(1);
            if !(self.save_counter - 1).is_multiple_of(u64::from(self.save_every)) {
                return Ok(());
            }

            // Ensure directory exists
            let target_dir = self.save_dir.clone();
            std::fs::create_dir_all(&target_dir).map_err(|error| {
                ReadbackError::Artifact(format!(
                    "create capture directory {}: {error}",
                    target_dir.display()
                ))
            })?;
            // Repack from padded rows into tightly packed RGBA8 buffer
            let row_bytes = (width as usize) * 4;
            let mut tight = vec![0u8; row_bytes * (height as usize)];
            for y in 0..(height as usize) {
                let src_off = y * stride;
                let dst_off = y * row_bytes;
                let end = src_off + row_bytes;
                if end > src.len() {
                    return Err(ReadbackError::MetadataMismatch {
                        expected: format!(
                            "{} mapped bytes for {height} rows at stride {stride}",
                            stride * height as usize
                        ),
                        actual: format!("{} mapped bytes", src.len()),
                    });
                }
                tight[dst_off..dst_off + row_bytes].copy_from_slice(&src[src_off..end]);
            }
            let filename = format!("{}_{:06}.png", self.save_prefix, self.save_counter);
            let path = target_dir.join(filename);
            image::save_buffer_with_format(
                &path,
                &tight,
                width,
                height,
                image::ColorType::Rgba8,
                image::ImageFormat::Png,
            )
            .map_err(|error| {
                ReadbackError::Artifact(format!("write capture PNG {}: {error}", path.display()))
            })?;
            let meta_path = path.with_extension("txt");
            let checksum_hex = format!("{:016x}", digest.checksum);
            let meta = format!(
                "width={}\nheight={}\nstride={}\npopulated_bytes={}\nnonzero_rgb_bytes={}\nchecksum_decimal={}\nchecksum_hex={}\nmin_rgb_byte={}\nmax_rgb_byte={}\nvaried_rgb={}\nsample_rgba={:?}\n",
                digest.width,
                digest.height,
                digest.stride,
                digest.populated_bytes,
                digest.non_zero,
                digest.checksum,
                checksum_hex,
                digest.min_byte,
                digest.max_byte,
                digest.varied_rgb,
                digest.sample_rgba
            );
            std::fs::write(&meta_path, meta).map_err(|error| {
                ReadbackError::Artifact(format!(
                    "write capture metadata {}: {error}",
                    meta_path.display()
                ))
            })?;
            info!(
                target = "scriptbots::render::wgpu",
                path = %path.display(),
                meta = %meta_path.display(),
                width = digest.width,
                height = digest.height,
                stride = digest.stride,
                populated_bytes = digest.populated_bytes,
                non_zero_bytes = digest.non_zero,
                checksum = digest.checksum,
                checksum_hex,
                min_byte = digest.min_byte,
                max_byte = digest.max_byte,
                varied_rgb = digest.varied_rgb,
                sample_r = digest.sample_rgba[0],
                sample_g = digest.sample_rgba[1],
                sample_b = digest.sample_rgba[2],
                sample_a = digest.sample_rgba[3],
                "wgpu readback frame saved"
            );
            Ok(())
        }
    }

    #[cfg(test)]
    mod capture_policy_tests {
        use super::{Compositor, ReadbackDigest, RenderSnapshotOutcome, rgba8_is_visually_blank};
        use scriptbots_world_gfx::{
            AgentInstance, ReadbackError, TerrainView, WorldSnapshot as GfxSnapshot,
        };

        fn unique_capture_dir(label: &str) -> std::path::PathBuf {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time must follow the Unix epoch")
                .as_nanos();
            std::env::temp_dir().join(format!(
                "scriptbots_capture_policy_{label}_{}_{nonce}",
                std::process::id()
            ))
        }

        fn test_digest() -> ReadbackDigest {
            ReadbackDigest {
                width: 2,
                height: 1,
                stride: 8,
                checksum: 0x1234,
                non_zero: 8,
                populated_bytes: 8,
                min_byte: 10,
                max_byte: 255,
                varied_rgb: true,
                sample_rgba: [10, 20, 30, 255],
            }
        }

        #[test]
        fn blank_detection_ignores_opaque_alpha_and_requires_rgb_variation() {
            let opaque_black = ReadbackDigest {
                non_zero: 0,
                varied_rgb: false,
                ..test_digest()
            };
            assert!(
                opaque_black.is_visually_blank(),
                "opaque alpha must not make an otherwise black frame look rendered"
            );

            let uniform_color = ReadbackDigest {
                non_zero: 6,
                varied_rgb: false,
                ..test_digest()
            };
            assert!(
                uniform_color.is_visually_blank(),
                "a uniform clear color is not framebuffer evidence"
            );

            assert!(
                !test_digest().is_visually_blank(),
                "varying nonzero RGB is render evidence"
            );

            assert!(rgba8_is_visually_blank(&[0, 0, 0, 255, 0, 0, 0, 255]));
            assert!(rgba8_is_visually_blank(&[20, 30, 40, 255, 20, 30, 40, 255]));
            assert!(!rgba8_is_visually_blank(&[
                20, 30, 40, 255, 21, 30, 40, 255
            ]));
        }

        fn minimal_snapshot<'a>(tiles: &'a [u32], colors: &'a [[f32; 4]]) -> GfxSnapshot<'a> {
            GfxSnapshot {
                world_size: (1.0, 1.0),
                terrain: TerrainView {
                    dims: (1, 1),
                    cell_size: 1,
                    tiles,
                    colors,
                    elevation: None,
                },
                agents: &[] as &[AgentInstance],
                anim_seconds: 0.0,
                tonemap_mode: None,
            }
        }

        #[test]
        fn compositor_rejects_zero_dimensions_instead_of_silently_reusing_pixels() {
            let tiles = [3];
            let colors = [[0.2, 0.5, 0.2, 1.0]];
            let mut compositor = Compositor::new();
            let error = compositor
                .render_snapshot(&minimal_snapshot(&tiles, &colors), (0, 64))
                .expect_err("zero-width capture must fail");
            assert_eq!(
                error,
                ReadbackError::ZeroDimensions {
                    width: 0,
                    height: 64
                }
            );
        }

        #[test]
        fn cached_adapter_failure_remains_a_typed_unavailable_error() {
            let tiles = [3];
            let colors = [[0.2, 0.5, 0.2, 1.0]];
            let mut compositor = Compositor::new();
            compositor.adapter_failure = Some("no compatible native adapter".to_owned());
            let error = compositor
                .render_snapshot(&minimal_snapshot(&tiles, &colors), (64, 64))
                .expect_err("cached adapter failure must not masquerade as success");
            assert_eq!(error, ReadbackError::AdapterUnavailable);
        }

        #[test]
        fn rate_limit_reuse_is_an_explicit_success_outcome() {
            let tiles = [3];
            let colors = [[0.2, 0.5, 0.2, 1.0]];
            let mut compositor = Compositor::new();
            compositor.last_submit = Some(std::time::Instant::now());
            compositor.min_interval = 60.0;
            compositor.image = Some(super::GpuiImage::new((64, 64), 256));
            compositor.last_digest = Some(test_digest());
            let outcome = compositor
                .render_snapshot(&minimal_snapshot(&tiles, &colors), (64, 64))
                .expect("rate limiting deliberately reuses the last completed frame");
            assert_eq!(outcome, RenderSnapshotOutcome::ReusedRateLimitedFrame);
        }

        #[test]
        fn a_failed_frame_cannot_be_reused_by_the_rate_limiter() {
            let tiles = [3];
            let colors = [[0.2, 0.5, 0.2, 1.0]];
            let mut compositor = Compositor::new();
            compositor.last_submit = Some(std::time::Instant::now());
            compositor.min_interval = 60.0;
            compositor.image = Some(super::GpuiImage::new((64, 64), 256));
            compositor.last_digest = Some(test_digest());

            compositor.invalidate_presentable_frame();
            compositor.adapter_failure = Some("no compatible native adapter".to_owned());
            let error = compositor
                .render_snapshot(&minimal_snapshot(&tiles, &colors), (64, 64))
                .expect_err("an invalidated image must not become a rate-limit success");
            assert_eq!(error, ReadbackError::AdapterUnavailable);
            assert!(
                !compositor.has_presentable_frame(),
                "the old allocation may remain reusable, but its pixels are not presentable"
            );
        }

        #[test]
        fn default_capture_policy_is_write_free_and_explicit_capture_writes_once() {
            let rgba = [10, 20, 30, 255, 40, 50, 60, 255];
            let digest = test_digest();

            let disabled_dir = unique_capture_dir("disabled");
            let mut default = Compositor::new();
            default.save_enabled = false;
            default.save_dir = disabled_dir.clone();
            default.save_prefix = "default".to_owned();
            default
                .save_rgba_if_requested(&rgba, 2, 1, 8, &digest)
                .expect("disabled capture is an intentional no-op");

            assert_eq!(
                default.save_counter, 0,
                "a disabled capture policy must not consume a capture sequence number"
            );
            assert!(
                !disabled_dir.exists(),
                "default capture policy must not create {}",
                disabled_dir.display()
            );

            let explicit_dir = unique_capture_dir("explicit");
            let mut explicit = Compositor::new();
            explicit.save_enabled = true;
            explicit.save_dir = explicit_dir.clone();
            explicit.save_every = 1;
            explicit.save_counter = 0;
            explicit.save_prefix = "explicit".to_owned();
            explicit
                .save_rgba_if_requested(&rgba, 2, 1, 8, &digest)
                .expect("explicit capture artifacts");

            assert_eq!(
                explicit.save_counter, 1,
                "one explicit capture request must consume one sequence number"
            );
            assert!(
                explicit_dir.join("explicit_000001.png").is_file(),
                "the explicit capture path must write exactly one PNG"
            );
            assert!(
                explicit_dir.join("explicit_000001.txt").is_file(),
                "the explicit capture path must retain its matching metadata"
            );
            assert_eq!(
                std::fs::read_dir(&explicit_dir)
                    .expect("explicit capture directory must be readable")
                    .count(),
                2,
                "one explicit capture must create only its PNG and metadata artifacts"
            );
        }
    }

    impl Default for Compositor {
        fn default() -> Self {
            Self::new()
        }
    }

    // Headless, one-shot offscreen render to PNG (bytes) using the same snapshot path as the GUI.
    // This allows verifying the wgpu pipeline without a display server.
    pub fn render_wgpu_png_offscreen(
        world: &WorldState,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, scriptbots_world_gfx::ReadbackError> {
        if width == 0 || height == 0 {
            return Err(scriptbots_world_gfx::ReadbackError::ZeroDimensions { width, height });
        }
        // Build snapshot from world
        let frame = crate::RenderFrame::from_world(world, crate::ColorPaletteMode::Natural)
            .ok_or_else(|| scriptbots_world_gfx::ReadbackError::MetadataMismatch {
                expected: "a finite non-empty world render snapshot".to_owned(),
                actual: format!(
                    "world {}x{} at tick {} could not produce a render frame",
                    world.config().world_width,
                    world.config().world_height,
                    world.tick().0
                ),
            })?;
        let world_size = frame.world_size;
        let dims = frame.terrain.dimensions;
        let tiles_u32: Vec<u32> = frame
            .terrain
            .tiles
            .iter()
            .map(|t| match t.kind {
                TerrainKind::DeepWater => 0,
                TerrainKind::ShallowWater => 1,
                TerrainKind::Sand => 2,
                TerrainKind::Grass => 3,
                TerrainKind::Bloom => 4,
                TerrainKind::Rock => 5,
            })
            .collect();
        let elevation: Vec<f32> = frame.terrain.tiles.iter().map(|t| t.elevation).collect();
        let terrain_colors = canonical_gpu_terrain_colors(&frame);
        let palette_is_natural = matches!(frame.palette, ColorPaletteMode::Natural);
        let agents_gpu: Vec<scriptbots_world_gfx::AgentInstance> = frame
            .agents
            .iter()
            .map(|a| build_gpu_agent_instance(&frame, a, frame.palette, palette_is_natural))
            .collect();

        let snapshot = GfxSnapshot {
            world_size,
            terrain: scriptbots_world_gfx::TerrainView {
                dims,
                cell_size: frame.terrain.cell_size,
                tiles: &tiles_u32,
                colors: &terrain_colors,
                elevation: Some(&elevation),
            },
            agents: &agents_gpu,
            anim_seconds: frame.tick as f32 * scriptbots_world_gfx::ANIM_SECONDS_PER_TICK,
            tonemap_mode: frame.tonemap_mode,
        };

        // Fit camera into the requested viewport
        let mut comp = Compositor::new();
        let width_px = width as f32;
        let height_px = height as f32;
        let base_scale = (width_px / world_size.0)
            .min(height_px / world_size.1)
            .max(0.0001);
        let pad_x = (width_px - world_size.0 * base_scale) * 0.5;
        let pad_y = (height_px - world_size.1 * base_scale) * 0.5;
        comp.set_camera_params(base_scale, (pad_x, pad_y));

        comp.render_snapshot(&snapshot, (width, height))?;

        // Extract mapped frame. Geometry must come from the readback view:
        // render_snapshot may render at a reduced resolution (SB_WGPU_RES_SCALE),
        // so the actual view size can differ from the requested width/height.
        // An unmapped frame is a typed Empty failure, never an empty-vector success.
        let view = comp
            .renderer
            .as_mut()
            .ok_or_else(|| {
                scriptbots_world_gfx::ReadbackError::Device(
                    "compositor reported success without a world renderer".to_owned(),
                )
            })?
            .mapped_rgba()?;
        let view_width = view.width;
        let view_height = view.height;
        let stride = view.bytes_per_row as usize;
        let row_bytes = (view_width as usize) * 4;
        let src = view.bytes();
        let mut tight = vec![0u8; row_bytes * view_height as usize];
        for y in 0..(view_height as usize) {
            let s = y * stride;
            let d = y * row_bytes;
            let end = s + row_bytes;
            if end > src.len() {
                return Err(scriptbots_world_gfx::ReadbackError::MetadataMismatch {
                    expected: format!(
                        "{} bytes for {} rows at stride {stride}",
                        stride * view_height as usize,
                        view_height
                    ),
                    actual: format!("{} mapped bytes", src.len()),
                });
            }
            tight[d..d + row_bytes].copy_from_slice(&src[s..end]);
        }
        if rgba8_is_visually_blank(&tight) {
            return Err(scriptbots_world_gfx::ReadbackError::Blank);
        }
        let mut png: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut png);
        let encoder = image::codecs::png::PngEncoder::new(&mut cursor);
        image::ImageEncoder::write_image(
            encoder,
            &tight,
            view_width,
            view_height,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|error| {
            scriptbots_world_gfx::ReadbackError::Artifact(format!("PNG encode failed: {error}"))
        })?;
        Ok(png)
    }
}

#[cfg(all(test, feature = "world_wgpu"))]
mod wgpu_capture_test {
    use super::world_compositor::Compositor;
    use scriptbots_world_gfx::{AgentInstance, TerrainView, WorldSnapshot as GfxSnapshot};

    fn test_terrain_colors(tiles: &[u32]) -> Vec<[f32; 4]> {
        tiles
            .iter()
            .map(|kind| {
                let kind = match kind {
                    0 => scriptbots_core::TerrainKind::DeepWater,
                    1 => scriptbots_core::TerrainKind::ShallowWater,
                    2 => scriptbots_core::TerrainKind::Sand,
                    3 => scriptbots_core::TerrainKind::Grass,
                    4 => scriptbots_core::TerrainKind::Bloom,
                    _ => scriptbots_core::TerrainKind::Rock,
                };
                let rgb = scriptbots_core::visual::terrain_shaded_color(
                    &scriptbots_core::visual::TerrainShadeInput {
                        kind,
                        moisture: 0.5,
                        elevation: 0.5,
                        slope: 0.0,
                        accent: 0.0,
                        daylight: scriptbots_core::visual::DAYLIGHT_STATIC,
                    },
                );
                [rgb[0], rgb[1], rgb[2], 1.0]
            })
            .collect()
    }

    fn capture_target(label: &str) -> (std::path::PathBuf, String, std::path::PathBuf) {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time must follow the Unix epoch")
            .as_nanos();
        let prefix = format!("scriptbots_{label}_{}_{nonce}", std::process::id());
        let dir = std::env::temp_dir();
        let path = dir.join(format!("{prefix}_000001.png"));
        (dir, prefix, path)
    }

    /// The captured frame, decoded into real pixels.
    ///
    /// The old tests asserted `metadata.len() > 0` — the size of the FILE, not
    /// the content of the IMAGE. A PNG of a completely black frame is several
    /// hundred bytes and sails through that check, so the renderer could have
    /// drawn nothing at all and the test would still have gone green.
    fn decode_capture(path: &std::path::Path) -> Result<image::RgbaImage, String> {
        image::open(path)
            .map(|decoded| decoded.to_rgba8())
            .map_err(|err| format!("captured frame at {} must decode: {err}", path.display()))
    }

    /// How many DISTINCT colours the frame contains.
    ///
    /// This is the blank-frame detector. A frame that was never drawn into is
    /// uniform (one colour: the clear colour, or transparent black), so a count
    /// above one is the minimum evidence that rasterization actually happened.
    fn distinct_colors(image: &image::RgbaImage) -> usize {
        let mut seen = std::collections::BTreeSet::new();
        for pixel in image.pixels() {
            seen.insert(pixel.0);
        }
        seen.len()
    }

    /// Do two frames disagree anywhere within `radius` of a point?
    ///
    /// This is the honest question to ask of a renderer whose colours are blended:
    /// not "is this pixel exactly the colour I passed in" (that tests the shader's
    /// arithmetic) but "did drawing this thing change the picture here".
    fn frames_differ_near(
        a: &image::RgbaImage,
        b: &image::RgbaImage,
        center: (u32, u32),
        radius: u32,
    ) -> bool {
        let (width, height) = a.dimensions();
        let x0 = center.0.saturating_sub(radius);
        let y0 = center.1.saturating_sub(radius);
        let x1 = (center.0 + radius).min(width.saturating_sub(1));
        let y1 = (center.1 + radius).min(height.saturating_sub(1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                if a.get_pixel(x, y) != b.get_pixel(x, y) {
                    return true;
                }
            }
        }
        false
    }

    /// Name the surface that actually rasterized, so the test never overclaims.
    ///
    /// These captures run green on a headless CI box with no GPU in it — which
    /// means llvmpipe, a CPU rasterizer, is doing the drawing. That is a
    /// perfectly good thing to test; it is NOT a live GPU framebuffer, and a test
    /// named as though it were is selling false confidence in the GPU path.
    fn raster_source(comp: &Compositor) -> &'static str {
        if comp.adapter_is_software() {
            "CPU surrogate (software adapter)"
        } else {
            "live GPU framebuffer"
        }
    }

    fn test_agent(
        position: [f32; 2],
        size: f32,
        color: [f32; 4],
        selection: f32,
        glow: f32,
        boost: f32,
    ) -> AgentInstance {
        let body_radius = size * 0.25;
        AgentInstance {
            position,
            quad_extent: [size * 0.5, size * 0.5],
            heading: [1.0, 0.0],
            body_radius,
            body_half_length: size * 0.32,
            wheel_offset: body_radius * 1.35,
            wheel_radius: body_radius * 0.38,
            mouth_open: 0.35,
            herbivore_tendency: 0.5,
            temperature_preference: 0.5,
            food_delta: 0.0,
            sound_level: 0.0,
            sound_output: 0.0,
            wheel_left: 0.0,
            wheel_right: 0.0,
            spike_length: 0.0,
            trait_smell: 0.5,
            trait_sound: 0.5,
            trait_hearing: 0.5,
            trait_eye: 0.5,
            trait_blood: 0.5,
            selection,
            color,
            glow,
            boost,
            spiked: 0.0,
            eye_dirs: [0.0; scriptbots_core::NUM_EYES],
            eye_fov: [0.0; scriptbots_core::NUM_EYES],
        }
    }

    /// A captured terrain frame must contain a RASTERIZED IMAGE, not merely a file.
    ///
    /// This test was called `wgpu_capture_smoke` and asserted two things: that a
    /// PNG existed, and that its file length was greater than zero. Both are true
    /// of a PNG containing a completely black frame — so the renderer could have
    /// drawn nothing whatsoever and this test would still have passed. It also
    /// claimed "wgpu" in its name while running green on a headless CI box with
    /// no GPU, i.e. on an llvmpipe CPU rasterizer. The name has been corrected and
    /// the assertions now look at the pixels.
    #[test]
    fn a_captured_terrain_frame_is_actually_rasterized_and_not_merely_written() {
        let (save_dir, save_prefix, expected_png) = capture_target("render");
        let mut comp = Compositor::new_for_capture_test(save_dir, save_prefix);
        let viewport = (640u32, 360u32);
        // Simple 120x60 grass snapshot (50-unit cells matching default config)
        let dims = (120u32, 60u32);
        let tiles: Vec<u32> = vec![3u32; (dims.0 * dims.1) as usize];
        let colors = test_terrain_colors(&tiles);
        let snapshot = GfxSnapshot {
            world_size: (6000.0, 3000.0),
            terrain: TerrainView {
                dims,
                cell_size: 50,
                tiles: &tiles,
                colors: &colors,
                elevation: None,
            },
            agents: &[] as &[AgentInstance],
            anim_seconds: 0.0,
            tonemap_mode: None,
        };
        comp.set_camera_params(1.0, (0.0, 0.0));
        comp.render_snapshot(&snapshot, viewport)
            .expect("terrain capture must either render or return a typed GPU failure");

        assert!(
            expected_png.is_file(),
            "expected a render capture at {}; adapter failure: {:?}",
            expected_png.display(),
            comp.adapter_failure()
        );

        // The evidence, at last: decode the frame and look at it.
        let frame = decode_capture(&expected_png).expect("captured frame must decode");
        assert_eq!(
            frame.dimensions(),
            viewport,
            "the captured frame must be the size we asked to render"
        );

        let colors = distinct_colors(&frame);
        assert!(
            colors > 1,
            "the captured frame is a single flat colour, which means NOTHING WAS \
             DRAWN — the old test passed on exactly this, because it only checked \
             that the file had a non-zero length. Rasterized by: {}",
            raster_source(&comp)
        );

        // The snapshot is wall-to-wall grass (tile kind 3), so the frame must be
        // predominantly green: a frame that decoded, had several colours, and was
        // still the wrong scene would otherwise slip through.
        let greener_than_red = frame
            .pixels()
            .filter(|pixel| pixel.0[3] > 0 && pixel.0[1] > pixel.0[0])
            .count();
        assert!(
            greener_than_red > (frame.pixels().len() / 4),
            "a frame of solid grass must be predominantly green, but only {} of {} \
             pixels were. Rasterized by: {}",
            greener_than_red,
            frame.pixels().len(),
            raster_source(&comp)
        );
    }

    /// The blank-frame detector must actually fire.
    ///
    /// Same discipline as any other alarm: an evidence check nobody has ever seen
    /// fail is an evidence check nobody knows works. This constructs the exact
    /// image the old test would have accepted — a uniform, fully black frame —
    /// and proves the new check rejects it.
    #[test]
    fn the_blank_frame_check_rejects_the_frame_the_old_test_would_have_accepted() {
        let blank = image::RgbaImage::from_pixel(64, 64, image::Rgba([0, 0, 0, 255]));
        assert_eq!(
            distinct_colors(&blank),
            1,
            "a frame nothing was drawn into is uniform; if this ever reports more \
             than one colour, the detector is broken and every capture test above \
             it is worthless"
        );

        let drawn = {
            let mut image = blank.clone();
            image.put_pixel(10, 10, image::Rgba([12, 200, 40, 255]));
            image
        };
        assert!(
            distinct_colors(&drawn) > 1,
            "a frame with something drawn in it must be distinguishable from a \
             blank one, or the check would reject everything and prove nothing"
        );
    }

    /// The agents must actually appear in the frame, where they were placed.
    ///
    /// This test was called `wgpu_capture_agents` and never looked at a single
    /// pixel: it built two agents with deliberately distinct colours, rendered
    /// them, and then asserted only that a file existed and was non-empty. The
    /// agent-drawing path could have been deleted entirely and it would still have
    /// passed. It now proves both agents were rasterized, at the positions they
    /// were given.
    #[test]
    fn both_agents_are_rasterized_into_the_frame_at_the_positions_they_were_given() {
        let (save_dir, save_prefix, expected_png) = capture_target("agents");
        // The compositor numbers its captures, so the second render lands here.
        let second_png = save_dir.join(format!("{save_prefix}_000002.png"));
        let mut comp = Compositor::new_for_capture_test(save_dir, save_prefix);
        let viewport = (640u32, 360u32);

        // Patterned 120x60 terrain across all six kinds
        let dims = (120u32, 60u32);
        let mut tiles: Vec<u32> = Vec::with_capacity((dims.0 * dims.1) as usize);
        for y in 0..dims.1 {
            for x in 0..dims.0 {
                tiles.push((x + y) % 6);
            }
        }
        let colors = test_terrain_colors(&tiles);

        // Fit entire world into the viewport (match the GPUI mapping)
        let world_size = (6000.0f32, 3000.0f32);
        let base_scale = (viewport.0 as f32 / world_size.0)
            .min(viewport.1 as f32 / world_size.1)
            .max(0.0001);
        let pad_x = (viewport.0 as f32 - world_size.0 * base_scale) * 0.5;
        let pad_y = (viewport.1 as f32 - world_size.1 * base_scale) * 0.5;
        comp.set_camera_params(base_scale, (pad_x, pad_y));

        // Two visible agents near center with distinct colors
        let agents = vec![
            test_agent(
                [world_size.0 * 0.5, world_size.1 * 0.5],
                48.0,
                [1.0, 0.25, 0.2, 1.0],
                2.0,
                0.4,
                0.0,
            ),
            test_agent(
                [world_size.0 * 0.55, world_size.1 * 0.48],
                36.0,
                [0.2, 0.9, 0.3, 1.0],
                1.0,
                0.2,
                1.0,
            ),
        ];

        let snapshot = GfxSnapshot {
            world_size,
            terrain: TerrainView {
                dims,
                cell_size: 50,
                tiles: &tiles,
                colors: &colors,
                elevation: None,
            },
            agents: &agents,
            anim_seconds: 0.0,
            tonemap_mode: None,
        };

        comp.render_snapshot(&snapshot, viewport)
            .expect("agent capture must either render or return a typed GPU failure");

        assert!(
            expected_png.is_file(),
            "expected an agents capture at {}; adapter failure: {:?}",
            expected_png.display(),
            comp.adapter_failure()
        );

        let frame = decode_capture(&expected_png).expect("captured frame must decode");
        assert!(
            distinct_colors(&frame) > 1,
            "nothing was drawn at all. Rasterized by: {}",
            raster_source(&comp)
        );

        // Where the two agents should have landed, in pixels.
        let to_pixel = |world: [f32; 2]| -> (u32, u32) {
            (
                (world[0] * base_scale + pad_x).round().max(0.0) as u32,
                (world[1] * base_scale + pad_y).round().max(0.0) as u32,
            )
        };
        let red_at = to_pixel([world_size.0 * 0.5, world_size.1 * 0.5]);
        let green_at = to_pixel([world_size.0 * 0.55, world_size.1 * 0.48]);

        // The agent colours are deliberately distinct from the terrain, and from
        // each other, so their presence is a real signal rather than a coincidence
        // of the background. A generous tolerance and search radius: the shader
        // blends and antialiases, so demanding the exact input colour at the exact
        // centre pixel would be a test of the blend function, not of whether the
        // agent was drawn.
        // A DIFFERENTIAL proof, rather than a guess about the shader's arithmetic.
        //
        // Demanding an exact input colour at the agent's centre pixel would be
        // testing the blend function, not the thing we care about: the agents
        // carry glow and selection, so the red agent lands on screen as a washed
        // pink rather than its literal input colour. Instead, render the SAME
        // scene with the agents removed and compare. If the agent-drawing path
        // were deleted, the two frames would be identical — and that is exactly
        // what the old test could not tell.
        let empty = GfxSnapshot {
            world_size,
            terrain: TerrainView {
                dims,
                cell_size: 50,
                tiles: &tiles,
                colors: &colors,
                elevation: None,
            },
            agents: &[] as &[AgentInstance],
            anim_seconds: 0.0,
            tonemap_mode: None,
        };
        comp.render_snapshot(&empty, viewport)
            .expect("comparison capture must either render or return a typed GPU failure");
        let without_agents =
            decode_capture(&second_png).expect("captured comparison frame must decode");
        assert_eq!(
            without_agents.dimensions(),
            frame.dimensions(),
            "the two frames must be comparable"
        );

        assert!(
            frames_differ_near(&frame, &without_agents, red_at, 12),
            "removing the agents changed NOTHING near the red agent at {red_at:?}, \
             which means the agent was never rasterized there — the agent-drawing \
             path could have been deleted and the old test would not have noticed. \
             Rasterized by: {}",
            raster_source(&comp)
        );
        assert!(
            frames_differ_near(&frame, &without_agents, green_at, 12),
            "removing the agents changed nothing near the green agent at \
             {green_at:?}. Rasterized by: {}",
            raster_source(&comp)
        );

        // And the difference must be LOCAL. If the whole frame changed, the two
        // renders differ for some reason that has nothing to do with the agents
        // (a moved camera, a different clear colour), and the assertions above
        // would be passing for the wrong reason.
        let corner = (8u32, 8u32);
        assert!(
            !frames_differ_near(&frame, &without_agents, corner, 6),
            "the frames differ in a corner far from any agent, so the difference \
             cannot be attributed to the agents and these assertions prove nothing"
        );
    }
}

#[cfg(feature = "world_wgpu")]
fn use_wgpu_renderer() -> bool {
    // Presentation containment (bd-2z0.7.11): the DEFAULT presentation is the native
    // GPUI canvas, which performs zero GPU readback. `SB_RENDERER=wgpu` is an explicit
    // diagnostic product whose per-frame readback/upload is intentional; captures
    // (--dump-png, frame saves) use the same one-shot readback path. Never widen this
    // default without a surface/swapchain presentation that needs no readback.
    static CHOICE: OnceLock<bool> = OnceLock::new();
    *CHOICE.get_or_init(|| {
        let choice = match std::env::var("SB_RENDERER").ok().as_deref() {
            Some("canvas") => false,
            Some("wgpu") => true,
            _ => false,
        };
        tracing::info!(choice, env = %std::env::var("SB_RENDERER").unwrap_or_default(), "use_wgpu_renderer decision");
        choice
    })
}

#[cfg(feature = "world_wgpu")]
fn emit_wgpu_paint_entry_diagnostic() {
    tracing::trace!("entered paint_world_with_wgpu");
}

#[cfg(feature = "world_wgpu")]
fn emit_wgpu_camera_mapping_diagnostic(
    width_px: f32,
    height_px: f32,
    world_dims: (f32, f32),
    layout: ViewLayout,
    cam_offset: (f32, f32),
) {
    tracing::trace!(
        vw = width_px,
        vh = height_px,
        world_w = world_dims.0,
        world_h = world_dims.1,
        base_scale = layout.base_scale,
        scale = layout.scale,
        pad_x = layout.pad.0,
        pad_y = layout.pad.1,
        offset_x = layout.offset.0,
        offset_y = layout.offset.1,
        cam_off_x = cam_offset.0,
        cam_off_y = cam_offset.1,
        "wgpu camera mapping"
    );
}

#[cfg(feature = "world_wgpu")]
fn paint_world_with_wgpu(state: &CanvasState, bounds: Bounds<Pixels>, window: &mut Window) {
    use scriptbots_world_gfx::WorldSnapshot as GfxSnapshot;
    use world_compositor::Compositor;
    static COMPOSITOR: OnceLock<std::sync::Mutex<Compositor>> = OnceLock::new();
    let comp = COMPOSITOR.get_or_init(|| std::sync::Mutex::new(Compositor::new()));
    let mut comp = comp.lock().expect("compositor mutex");

    emit_wgpu_paint_entry_diagnostic();

    let world_size = state.frame.world_size;
    // Root-cause guard: GPUI may report 0x0 during initial layout or when minimized
    let vw_u32 = u32::from(bounds.size.width);
    let vh_u32 = u32::from(bounds.size.height);
    if vw_u32 == 0 || vh_u32 == 0 {
        return;
    }
    // GPUI bounds are in logical pixels; render the offscreen target at
    // physical resolution so HiDPI displays don't get an upscaled
    // quarter-resolution image. Camera params are scaled to match below.
    let scale_factor = window.scale_factor().max(0.1);
    let viewport = (
        (((vw_u32 as f32) * scale_factor).round() as u32).max(1),
        (((vh_u32 as f32) * scale_factor).round() as u32).max(1),
    );

    // Calculate camera scale/offset to map world pixels -> viewport pixels exactly as GPUI would
    // Reuse the same mapping used in paint_frame
    let width_px = f32::from(bounds.size.width).max(1.0);
    let raw_height_px = f32::from(bounds.size.height).max(1.0);
    let window_bounds = window.bounds();
    let window_height_px = f32::from(window_bounds.size.height).max(1.0);
    let height_px = if raw_height_px <= 2.0 && window_height_px > 16.0 {
        window_height_px
    } else {
        raw_height_px
    };
    let origin = (f32::from(bounds.origin.x), f32::from(bounds.origin.y));
    let world_dims = (world_size.0.max(1.0), world_size.1.max(1.0));
    let mut camera_guard = state.camera.lock().expect("camera mutex poisoned");
    let mut layout = layout_camera_for_frame(
        &mut camera_guard,
        &state.frame,
        origin,
        (width_px, height_px),
    );

    let coverage_too_small =
        layout.render_size.0 < width_px * 0.25 || layout.render_size.1 < height_px * 0.25;
    let coverage_too_large =
        layout.render_size.0 > width_px * 6.0 || layout.render_size.1 > height_px * 6.0;
    if coverage_too_small || coverage_too_large {
        camera_guard.fit_world();
        layout = camera_guard.layout(origin, (width_px, height_px), world_dims);
    }

    if state.controls.follow_mode != FollowMode::Off
        && let Some(target) = state.follow_target
    {
        camera_guard.center_on(target);
        layout = camera_guard.layout(origin, (width_px, height_px), world_dims);
        tracing::debug!(
            follow_mode = ?state.controls.follow_mode,
            target_x = target.x,
            target_y = target.y,
            render_width = layout.render_size.0,
            render_height = layout.render_size.1,
            offset_x = layout.offset.0,
            offset_y = layout.offset.1,
            "camera_layout_follow"
        );

        let follow_coverage_small =
            layout.render_size.0 < width_px * 0.25 || layout.render_size.1 < height_px * 0.25;
        let follow_coverage_large =
            layout.render_size.0 > width_px * 6.0 || layout.render_size.1 > height_px * 6.0;
        if follow_coverage_small || follow_coverage_large {
            camera_guard.fit_world();
            layout = camera_guard.layout(origin, (width_px, height_px), world_dims);
            tracing::warn!(
                follow_mode = ?state.controls.follow_mode,
                render_width = layout.render_size.0,
                render_height = layout.render_size.1,
                viewport_width = width_px,
                viewport_height = height_px,
                "follow_mode_auto_fit_world"
            );
        }
    }
    drop(camera_guard);

    let scale = layout.scale;
    let base_scale = layout.base_scale;
    let pad_x = layout.pad.0;
    let pad_y = layout.pad.1;
    let off_px_x = layout.offset.0;
    let off_px_y = layout.offset.1;
    // Offscreen world renderer uses (0,0) origin; include only pad and camera offset relative to origin
    let cam_offset = (off_px_x - origin.0, off_px_y - origin.1);

    if env_flag("SB_WGPU_LAYOUT_LOG") {
        let render_scale = comp.render_scale_factor();
        tracing::info!(
            viewport_width = width_px,
            viewport_height = height_px,
            world_width = world_dims.0,
            world_height = world_dims.1,
            pad_x,
            pad_y,
            offset_x = cam_offset.0,
            offset_y = cam_offset.1,
            zoom = scale / base_scale,
            render_scale,
            "wgpu_canvas_layout"
        );
    }

    if env_flag("SB_WGPU_LOG_CAM") {
        tracing::info!(
            vw = width_px,
            vh = height_px,
            world_w = world_dims.0,
            world_h = world_dims.1,
            base_scale = base_scale,
            scale = scale,
            pad_x = pad_x,
            pad_y = pad_y,
            off_x = off_px_x,
            off_y = off_px_y,
            cam_off_x = cam_offset.0,
            cam_off_y = cam_offset.1,
            "wgpu camera mapping"
        );
    }

    // Build a minimal snapshot from the current RenderFrame
    let terrain_dims = state.frame.terrain.dimensions;
    let tiles_u32: Vec<u32> = state
        .frame
        .terrain
        .tiles
        .iter()
        .map(|t| match t.kind {
            TerrainKind::DeepWater => 0,
            TerrainKind::ShallowWater => 1,
            TerrainKind::Sand => 2,
            TerrainKind::Grass => 3,
            TerrainKind::Bloom => 4,
            TerrainKind::Rock => 5,
        })
        .collect();
    let elevation: Vec<f32> = state
        .frame
        .terrain
        .tiles
        .iter()
        .map(|t| t.elevation)
        .collect();
    let terrain_colors = canonical_gpu_terrain_colors(&state.frame);
    let palette_is_natural = matches!(state.frame.palette, ColorPaletteMode::Natural);
    let agents_gpu: Vec<scriptbots_world_gfx::AgentInstance> = state
        .frame
        .agents
        .iter()
        .map(|a| build_gpu_agent_instance(&state.frame, a, state.frame.palette, palette_is_natural))
        .collect();

    let terrain_view = scriptbots_world_gfx::TerrainView {
        dims: terrain_dims,
        cell_size: state.frame.terrain.cell_size,
        tiles: &tiles_u32,
        colors: &terrain_colors,
        elevation: Some(&elevation),
    };
    let snapshot = GfxSnapshot {
        world_size,
        terrain: terrain_view,
        agents: &agents_gpu,
        anim_seconds: state.frame.tick as f32 * scriptbots_world_gfx::ANIM_SECONDS_PER_TICK,
        tonemap_mode: state.frame.tonemap_mode,
    };

    emit_wgpu_camera_mapping_diagnostic(width_px, height_px, world_dims, layout, cam_offset);

    // Render using current camera mapping. The camera layout was computed in
    // logical pixels; scale it to the physical-resolution render target.
    comp.set_camera_params(
        scale * scale_factor,
        (cam_offset.0 * scale_factor, cam_offset.1 * scale_factor),
    );
    match comp.render_snapshot(&snapshot, viewport) {
        Ok(_) if comp.paint_world(bounds, window) => {}
        Ok(_) => {
            // A completed/reused image that cannot be presented is not allowed
            // to produce a blank window.
            paint_frame(state, bounds, window);
        }
        Err(scriptbots_world_gfx::ReadbackError::AdapterUnavailable) => {
            // `Compositor` reports this policy fallback once with adapter
            // diagnostics; never repaint a stale GPU image afterward.
            paint_frame(state, bounds, window);
        }
        Err(error) => {
            tracing::warn!(
                %error,
                "wgpu compositor failed; drawing the deterministic CPU frame instead"
            );
            paint_frame(state, bounds, window);
        }
    }
}

#[cfg(all(test, feature = "world_wgpu"))]
mod wgpu_paint_diagnostic_tests {
    use super::{
        ViewLayout, emit_wgpu_camera_mapping_diagnostic, emit_wgpu_paint_entry_diagnostic,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tracing::{
        Event, Level, Metadata, Subscriber,
        span::{Attributes, Id, Record},
    };

    #[derive(Default)]
    struct EventCounts {
        info: AtomicUsize,
        trace: AtomicUsize,
    }

    struct CountingSubscriber {
        counts: Arc<EventCounts>,
    }

    impl Subscriber for CountingSubscriber {
        fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _span: &Attributes<'_>) -> Id {
            panic!("this event-only subscriber must not observe spans")
        }

        fn record(&self, _span: &Id, _values: &Record<'_>) {}

        fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

        fn event(&self, event: &Event<'_>) {
            match *event.metadata().level() {
                Level::INFO => {
                    self.counts.info.fetch_add(1, Ordering::Relaxed);
                }
                Level::TRACE => {
                    self.counts.trace.fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }
        }

        fn enter(&self, _span: &Id) {}

        fn exit(&self, _span: &Id) {}
    }

    #[test]
    fn steady_state_wgpu_paint_diagnostics_are_below_info() {
        let counts = Arc::new(EventCounts::default());
        let subscriber = CountingSubscriber {
            counts: Arc::clone(&counts),
        };
        let layout = ViewLayout {
            base_scale: 0.5,
            scale: 0.75,
            pad: (8.0, 12.0),
            offset: (3.0, 4.0),
            render_size: (640.0, 480.0),
        };

        tracing::subscriber::with_default(subscriber, || {
            for _ in 0..2 {
                emit_wgpu_paint_entry_diagnostic();
                emit_wgpu_camera_mapping_diagnostic(
                    640.0,
                    480.0,
                    (1_200.0, 900.0),
                    layout,
                    (3.0, 4.0),
                );
            }
        });

        assert_eq!(
            counts.info.load(Ordering::Relaxed),
            0,
            "steady-state paint diagnostics must not emit INFO events"
        );
        assert_eq!(
            counts.trace.load(Ordering::Relaxed),
            4,
            "both diagnostics must remain available at TRACE for every sampled frame"
        );
    }
}

fn toroidal_delta(origin: f32, target: f32, extent: f32) -> f32 {
    let mut delta = target - origin;
    let half = extent * 0.5;
    if delta > half {
        delta -= extent;
    } else if delta < -half {
        delta += extent;
    }
    delta
}

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => {
            let s = value.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

static RENDER_WATERMARK: OnceLock<bool> = OnceLock::new();
static RENDER_SAFE: OnceLock<bool> = OnceLock::new();

fn watermark_enabled() -> bool {
    *RENDER_WATERMARK.get_or_init(|| env_flag("SCRIPTBOTS_RENDER_WATERMARK"))
}

fn safe_mode_enabled() -> bool {
    *RENDER_SAFE.get_or_init(|| env_flag("SCRIPTBOTS_RENDER_SAFE"))
}

/// Failure of the native GPUI application lifetime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuiRunError {
    /// One of the windows required by the application shell could not be created.
    WindowLaunch(String),
    /// A supervised host dependency failed after the windows were published.
    ControlRuntime(String),
}

impl std::fmt::Display for GuiRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WindowLaunch(detail) => write!(f, "GPUI launch failed: {detail}"),
            Self::ControlRuntime(detail) => write!(f, "GPUI control runtime failed: {detail}"),
        }
    }
}

impl std::error::Error for GuiRunError {}

/// Dependency-neutral liveness probe supplied by the application shell.
pub type GuiHealthProbe = Arc<dyn Fn() -> std::result::Result<(), String> + Send + Sync + 'static>;

fn record_gui_run_error(slot: &Mutex<Option<GuiRunError>>, error: GuiRunError) {
    let mut slot = match slot.lock() {
        Ok(slot) => slot,
        Err(poisoned) => poisoned.into_inner(),
    };
    if slot.is_none() {
        *slot = Some(error);
    }
}

trait GuiQuitRequest {
    fn request_gui_quit(&mut self);
}

impl GuiQuitRequest for App {
    fn request_gui_quit(&mut self) {
        self.quit();
    }
}

fn abort_gui_launch(
    app: &mut impl GuiQuitRequest,
    slot: &Mutex<Option<GuiRunError>>,
    detail: String,
) {
    tracing::error!(error = %detail, "aborting GPUI shell launch due to window creation failure");
    record_gui_run_error(slot, GuiRunError::WindowLaunch(detail));
    app.request_gui_quit();
    #[cfg(target_os = "windows")]
    {
        // On Windows, GPUI platform run calls ExitProcess(0) during quit, which preempts
        // run_demo from returning its recorded WindowLaunch error to main().
        // Force an immediate exit(1) on Windows when window launch fails.
        std::process::exit(1);
    }
}

fn gui_health_failure(probe: &GuiHealthProbe) -> Option<String> {
    #[cfg(panic = "unwind")]
    {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| probe())) {
            Ok(Ok(())) => None,
            Ok(Err(detail)) => Some(detail),
            Err(panic) => {
                let detail = if let Some(message) = panic.downcast_ref::<&str>() {
                    (*message).to_owned()
                } else if let Some(message) = panic.downcast_ref::<String>() {
                    message.clone()
                } else {
                    "non-string panic payload".to_owned()
                };
                Some(format!("health probe panicked: {detail}"))
            }
        }
    }
    #[cfg(panic = "abort")]
    {
        // The shipped release profile aborts on panic by design; only ordinary
        // health errors can be converted into a graceful GPUI shutdown there.
        probe().err()
    }
}

fn start_gui_health_monitor(
    app: &App,
    probe: GuiHealthProbe,
    error_slot: Arc<Mutex<Option<GuiRunError>>>,
) {
    app.spawn(async move |cx| {
        loop {
            if let Some(detail) = gui_health_failure(&probe) {
                record_gui_run_error(&error_slot, GuiRunError::ControlRuntime(detail));
                cx.update(|app| app.quit());
                return;
            }
            cx.background_executor()
                .timer(Duration::from_millis(50))
                .await;
        }
    })
    .detach();
}

#[derive(Clone)]
struct SimulationDriveSnapshot {
    paused: bool,
    speed_multiplier: f32,
    simulation_fault: Option<String>,
}

struct GuiSimulationDriver {
    world: Arc<Mutex<WorldState>>,
    simulation_step: WorldStepDriver,
    command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync + 'static>,
    pending_playback: VecDeque<SimulationCommand>,
    paused: bool,
    speed_multiplier: f32,
    sim_accumulator: f32,
    last_sim_instant: Option<Instant>,
    simulation_fault: Option<String>,
}

impl GuiSimulationDriver {
    fn new(
        world: Arc<Mutex<WorldState>>,
        simulation_step: WorldStepDriver,
        command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync + 'static>,
    ) -> Self {
        Self {
            world,
            simulation_step,
            command_drain,
            pending_playback: VecDeque::new(),
            paused: false,
            speed_multiplier: 1.0,
            sim_accumulator: 0.0,
            last_sim_instant: None,
            simulation_fault: None,
        }
    }

    fn snapshot(&self) -> SimulationDriveSnapshot {
        SimulationDriveSnapshot {
            paused: self.paused,
            speed_multiplier: self.speed_multiplier,
            simulation_fault: self.simulation_fault.clone(),
        }
    }

    fn pause_for_simulation_failure(&mut self, detail: String) {
        self.paused = true;
        self.sim_accumulator = 0.0;
        self.simulation_fault = Some(detail.clone());
        warn!(error = %detail, "Simulation paused after a terminal step failure");
    }

    fn apply_playback_state(&mut self, command: &SimulationCommand) {
        if let Some(paused) = command.paused {
            self.paused = paused;
            if paused {
                self.sim_accumulator = 0.0;
            }
        }
        if let Some(speed) = command.speed_multiplier {
            self.speed_multiplier = speed;
        }
        if command.step_once {
            self.paused = true;
            self.sim_accumulator = 0.0;
        }
    }

    fn world_tick_for_step_accounting(&self) -> Result<u64, String> {
        self.world.lock().map(|world| world.tick().0).map_err(|_| {
            "simulation world mutex poisoned while accounting for an admitted Step".to_owned()
        })
    }

    fn service_pending_playback(&mut self) -> bool {
        let mut attempted_steps = 0;
        while attempted_steps < MAX_SIM_STEPS_PER_FRAME {
            let Some(command) = self.pending_playback.front().cloned() else {
                break;
            };
            self.apply_playback_state(&command);
            if !command.step_once {
                self.pending_playback.pop_front();
                continue;
            }
            attempted_steps += 1;

            let before_tick = match self.world_tick_for_step_accounting() {
                Ok(tick) => tick,
                Err(error) => {
                    self.pause_for_simulation_failure(error);
                    break;
                }
            };
            let step_result = (self.simulation_step)();
            let after_tick = match self.world_tick_for_step_accounting() {
                Ok(tick) => tick,
                Err(error) => {
                    self.pause_for_simulation_failure(error);
                    break;
                }
            };

            let expected_after_tick = before_tick.checked_add(1);
            let advanced_exactly_once = expected_after_tick == Some(after_tick);
            if advanced_exactly_once {
                self.pending_playback.pop_front();
            }

            match step_result {
                Ok(_) if advanced_exactly_once => {}
                Ok(_) => {
                    self.pause_for_simulation_failure(format!(
                        "simulation step driver violated its one-tick contract: tick changed from \
                         {before_tick} to {after_tick}"
                    ));
                    break;
                }
                Err(error) if advanced_exactly_once => {
                    self.pause_for_simulation_failure(error.to_string());
                    break;
                }
                Err(error) => {
                    self.pause_for_simulation_failure(format!(
                        "{error}; admitted Step did not complete exactly one tick \
                         (before={before_tick}, after={after_tick})"
                    ));
                    break;
                }
            }
        }
        attempted_steps > 0
    }

    #[allow(clippy::collapsible_if)]
    fn drive_at(&mut self, now: Instant) {
        let last = self.last_sim_instant.replace(now);
        if self.simulation_fault.is_some() {
            return;
        }

        let mut playback = Vec::new();
        let mut step_error = None;
        if let Ok(mut world) = self.world.lock() {
            if let Some(error) = world.latched_step_error() {
                step_error = Some(error.to_string());
            } else {
                for command in (self.command_drain.as_ref())() {
                    match apply_control_command(&mut world, command) {
                        Ok(ControlDisposition::WorldApplied) => {}
                        Ok(ControlDisposition::Playback(command)) => playback.push(command),
                        Err(error) => warn!(%error, "GPUI rejected a drained control command"),
                    }
                }
            }
        }
        if let Some(error) = step_error {
            self.pause_for_simulation_failure(error);
            return;
        }

        self.pending_playback.extend(playback);
        let attempted_manual_step = self.service_pending_playback();
        if self.simulation_fault.is_some() || attempted_manual_step {
            return;
        }

        if !self.paused {
            if let Ok(world) = self.world.lock() {
                let control = world.config().control.clone();
                let agent_count = world.agent_count();
                let max_age = world.last_max_age();
                let spike_hits = world.last_spike_hits();
                drop(world);

                let mut reason: Option<String> = None;
                if control.auto_pause_on_spike_hit && spike_hits > 0 {
                    reason = Some(format!("spike hits detected ({spike_hits})"));
                } else if let Some(age_limit) = control.auto_pause_age_above {
                    if max_age >= age_limit {
                        reason = Some(format!("max age {max_age} ≥ {age_limit}"));
                    }
                } else if let Some(limit) = control.auto_pause_population_below {
                    if agent_count as u32 <= limit {
                        reason = Some(format!("population {agent_count} ≤ {limit}"));
                    }
                }

                if let Some(reason) = reason {
                    self.paused = true;
                    info!(reason = %reason, "Auto-paused due to control settings");
                }
            }
        }

        if self.paused || self.speed_multiplier <= 0.0 {
            self.sim_accumulator = 0.0;
            return;
        }

        let Some(last) = last else {
            return;
        };
        let delta = now.saturating_duration_since(last).as_secs_f32();
        self.sim_accumulator += delta * self.speed_multiplier;
        if self.sim_accumulator < SIM_TICK_INTERVAL {
            return;
        }

        let max_accumulator = SIM_TICK_INTERVAL * MAX_SIM_STEPS_PER_FRAME as f32;
        self.sim_accumulator = self.sim_accumulator.min(max_accumulator).min(0.5);
        let steps = (self.sim_accumulator / SIM_TICK_INTERVAL).floor() as usize;
        if steps == 0 {
            return;
        }
        let steps = steps.min(MAX_SIM_STEPS_PER_FRAME);
        self.sim_accumulator -= SIM_TICK_INTERVAL * steps as f32;

        let mut step_error = self
            .world
            .lock()
            .ok()
            .and_then(|world| world.latched_step_error().map(|error| error.to_string()));
        for _ in 0..steps {
            if step_error.is_some() {
                break;
            }
            if let Err(error) = (self.simulation_step)() {
                step_error = Some(error.to_string());
            }
        }
        if let Some(error) = step_error {
            self.pause_for_simulation_failure(error);
        }
    }
}

fn start_gui_simulation_driver(app: &App, driver: Arc<Mutex<GuiSimulationDriver>>) {
    app.spawn(async move |cx| {
        loop {
            {
                let mut driver = match driver.lock() {
                    Ok(driver) => driver,
                    Err(poisoned) => poisoned.into_inner(),
                };
                driver.drive_at(Instant::now());
            }
            cx.background_executor()
                .timer(Duration::from_secs_f32(SIM_TICK_INTERVAL))
                .await;
        }
    })
    .detach();
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuiViewRole {
    Hud,
    WorldCanvas,
}

impl GuiViewRole {
    fn window_options(self, app: &App) -> WindowOptions {
        let (title, width, height) = match self {
            Self::Hud => ("ScriptBots HUD", 1280.0, 720.0),
            Self::WorldCanvas => ("ScriptBots World", 1600.0, 900.0),
        };
        let bounds = Bounds::centered(None, size(px(width), px(height)), app);
        let mut options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            ..Default::default()
        };
        if let Some(titlebar) = options.titlebar.as_mut() {
            titlebar.title = Some(title.into());
        }
        options
    }

    fn view_title(self) -> SharedString {
        match self {
            Self::Hud => "ScriptBots HUD".into(),
            Self::WorldCanvas => "World".into(),
        }
    }

    fn launch_label(self) -> &'static str {
        match self {
            Self::Hud => "HUD",
            Self::WorldCanvas => "simulation",
        }
    }
}

struct GuiSession {
    simulation_driver: Arc<Mutex<GuiSimulationDriver>>,
    analytics: AnalyticsSnapshotProvider,
    command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync + 'static>,
    selection_projection: Arc<Mutex<Option<Vec<AgentId>>>>,
    selection_submission: Arc<Mutex<()>>,
}

impl GuiSession {
    fn new(
        world: Arc<Mutex<WorldState>>,
        simulation_step: WorldStepDriver,
        analytics: AnalyticsSnapshotProvider,
        command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync + 'static>,
        command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync + 'static>,
    ) -> Self {
        Self {
            simulation_driver: Arc::new(Mutex::new(GuiSimulationDriver::new(
                world,
                simulation_step,
                command_drain,
            ))),
            analytics,
            command_submit,
            selection_projection: Arc::new(Mutex::new(None)),
            selection_submission: Arc::new(Mutex::new(())),
        }
    }

    fn new_view(&self, role: GuiViewRole, focus_handle: FocusHandle) -> SimulationView {
        let mut view = SimulationView::new(
            Arc::clone(&self.simulation_driver),
            self.analytics.clone(),
            role.view_title(),
            Arc::clone(&self.command_submit),
            Arc::clone(&self.selection_projection),
            Arc::clone(&self.selection_submission),
        );
        view.focus_handle = Some(focus_handle);
        if role == GuiViewRole::WorldCanvas {
            view.set_minimal_canvas_mode();
        }
        view
    }

    fn install(
        self: &Arc<Self>,
        app: &mut App,
    ) -> std::result::Result<GuiWindowHandles, GuiWindowLaunchFailure> {
        let windows = open_gui_session_windows(app, self)?;
        app.on_window_closed(|app, _window_id| app.quit()).detach();
        start_gui_simulation_driver(app, Arc::clone(&self.simulation_driver));
        Ok(windows)
    }
}

struct GuiWindowHandles {
    hud: gpui::WindowHandle<SimulationView>,
    canvas: gpui::WindowHandle<SimulationView>,
}

#[derive(Debug)]
struct GuiWindowLaunchFailure {
    role: GuiViewRole,
    detail: String,
}

fn open_gui_session_windows(
    app: &mut App,
    session: &Arc<GuiSession>,
) -> std::result::Result<GuiWindowHandles, GuiWindowLaunchFailure> {
    let hud_options = GuiViewRole::Hud.window_options(app);
    let session_for_hud = Arc::clone(session);
    let hud = app
        .open_window(hud_options, move |window, cx| {
            cx.new(|cx| {
                let focus_handle = cx.focus_handle();
                focus_handle.focus(window, cx);
                session_for_hud.new_view(GuiViewRole::Hud, focus_handle)
            })
        })
        .map_err(|error| GuiWindowLaunchFailure {
            role: GuiViewRole::Hud,
            detail: format!("{error:?}"),
        })?;

    let canvas_options = GuiViewRole::WorldCanvas.window_options(app);
    let session_for_canvas = Arc::clone(session);
    let canvas = app
        .open_window(canvas_options, move |window, cx| {
            cx.new(|cx| {
                let focus_handle = cx.focus_handle();
                focus_handle.focus(window, cx);
                session_for_canvas.new_view(GuiViewRole::WorldCanvas, focus_handle)
            })
        })
        .map_err(|error| GuiWindowLaunchFailure {
            role: GuiViewRole::WorldCanvas,
            detail: format!("{error:?}"),
        })?;

    Ok(GuiWindowHandles { hud, canvas })
}

/// Launch the ScriptBots GPUI shell with an interactive HUD.
pub fn run_demo(
    world: Arc<Mutex<WorldState>>,
    simulation_step: WorldStepDriver,
    analytics: AnalyticsSnapshotProvider,
    command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync + 'static>,
    command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync + 'static>,
    health_probe: GuiHealthProbe,
) -> Result<(), GuiRunError> {
    if let Ok(world) = world.lock()
        && let Some(summary) = world.history().last()
    {
        info!(
            tick = summary.tick.0,
            agents = summary.agent_count,
            births = summary.births,
            deaths = summary.deaths,
            avg_energy = summary.average_energy,
            "Launching GPUI shell with latest world snapshot",
        );
    }

    let session = Arc::new(GuiSession::new(
        Arc::clone(&world),
        simulation_step,
        analytics,
        command_drain,
        command_submit,
    ));
    let session_for_app = Arc::clone(&session);
    let run_error = Arc::new(Mutex::new(None));
    let run_error_for_app = Arc::clone(&run_error);

    application()
        .with_quit_mode(QuitMode::LastWindowClosed)
        .run(move |app: &mut App| {
            let GuiWindowHandles { hud, canvas } = match session_for_app.install(app) {
                Ok(windows) => windows,
                Err(failure) => {
                    let label = failure.role.launch_label();
                    error!(
                        window = label,
                        error = %failure.detail,
                        "failed to open GPUI window"
                    );
                    abort_gui_launch(
                        app,
                        &run_error_for_app,
                        format!("could not open {label} window: {}", failure.detail),
                    );
                    // GPUI uses explicit application-lifetime quitting on macOS. Merely
                    // returning from this launch callback can otherwise leave a
                    // zero-window event loop alive forever, hiding the launch error
                    // from the caller.
                    return;
                }
            };
            let _ = (hud, canvas);

            start_gui_health_monitor(app, health_probe, Arc::clone(&run_error_for_app));
            app.activate(true);
        });

    let error = {
        let mut slot = match run_error.lock() {
            Ok(slot) => slot,
            Err(poisoned) => poisoned.into_inner(),
        };
        slot.take()
    };
    error.map_or(Ok(()), Err)
}

const MAX_SELECTION_EVENTS: usize = 64;
const SIM_TICK_INTERVAL: f32 = 1.0 / 60.0;
const MAX_SIM_STEPS_PER_FRAME: usize = 240;
static NEXT_BRAIN_INSPECTION_CLIENT: AtomicU64 = AtomicU64::new(1);

struct SimulationView {
    world: Arc<Mutex<WorldState>>,
    simulation_driver: Arc<Mutex<GuiSimulationDriver>>,
    analytics_provider: AnalyticsSnapshotProvider,
    title: SharedString,
    command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync + 'static>,
    camera: Arc<Mutex<Camera>>,
    /// Window-local GPUI atlas images for the continuously sampled world field.
    ///
    /// This must not be shared between the HUD and world windows: GPUI atlas
    /// eviction is window-scoped, and one view dropping another view's image
    /// would intermittently reveal stale terrain.
    world_raster_cache: Arc<Mutex<WorldRasterCache>>,
    world_raster_cleanup_registered: bool,
    #[cfg(test)]
    force_legacy_world_painter: bool,
    inspector: Arc<Mutex<InspectorState>>,
    selection_projection: Arc<Mutex<Option<Vec<AgentId>>>>,
    selection_submission: Arc<Mutex<()>>,
    playback: PlaybackState,
    perf: PerfStats,
    last_perf: PerfSnapshot,
    /// When set, `render` must NOT overwrite `last_perf` (bd-c7pg).
    ///
    /// Capture pinned perf at view construction and `render` clobbered it on frame one,
    /// so the pin controlled nothing after that. Captures rendering different numbers of
    /// frames then disagreed on sample_count and the ms values, which rendered as ~14k
    /// pixels of changed glyph text and was misread as harness nondeterminism.
    forced_perf: Option<PerfSnapshot>,
    accessibility: AccessibilitySettings,
    debug: DebugOverlayState,
    selection_events: VecDeque<SelectionEvent>,
    controls: SimulationControls,
    shift_inspect: bool,
    bindings: InputBindings,
    focus_handle: Option<FocusHandle>,
    key_capture: Option<CommandAction>,
    settings_panel: SettingsPanelState,
    /// User intent for HUD chrome (bd-v9cz). Never written by the resize rule.
    hud: HudLayout,
    /// Transient hysteresis latch: true once the window crossed below
    /// [`HUD_RAIL_COLLAPSE_WIDTH`], cleared only above [`HUD_RAIL_RESTORE_WIDTH`].
    /// Deliberately not part of [`HudLayout`] — it is window geometry, not intent.
    hud_rail_forced_closed: bool,
    analytics_cache: Option<HudAnalytics>,
    analytics_revision: Option<u64>,
    analytics_status: StorageUiStatus,
    #[cfg(feature = "audio")]
    audio: Option<AudioState>,
    // When true, render a minimal canvas-focused layout (used in the dedicated world window).
    minimal_canvas_mode: bool,
    // Per-agent short history of brain outputs for inspector sparklines
    brain_history: std::collections::HashMap<AgentId, OutputHistory>,
    brain_client_id: BrainInspectionClientId,
    brain_request_revision: BrainInspectionRevision,
    brain_inspection_cache: Option<BrainInspectorCapture>,
    /// Narrative rail visibility (bd-16g.2.4); toggled from the rail header.
    rail_visible: bool,
    /// Selected event index into the retained narrative, plus the selected event's
    /// identity so a ring wrap that drops it is detected rather than silently
    /// re-pointing the selection at a different event.
    rail_selection: Option<(usize, u64, NarrativeEventKind)>,
    /// Set when the ring dropped the user's selected event; cleared on the next
    /// explicit selection.
    rail_selection_aged_out: bool,
    /// One-shot latches for the rail's logging contract.
    rail_logged_first_show: bool,
    rail_warned_aged_out: bool,
    /// One-shot latches for the attribution panel's warn-once-per-(agent, reason)
    /// logging contract (bd-16g.4.3).
    attribution_warned: std::collections::HashSet<(u64, &'static str)>,
    /// The last (agent, tick) the panel's probed-tick debug line was emitted for.
    attribution_last_debug: Option<(u64, u64)>,
}
impl SimulationView {
    fn new(
        simulation_driver: Arc<Mutex<GuiSimulationDriver>>,
        analytics_provider: AnalyticsSnapshotProvider,
        title: SharedString,
        command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync + 'static>,
        selection_projection: Arc<Mutex<Option<Vec<AgentId>>>>,
        selection_submission: Arc<Mutex<()>>,
    ) -> Self {
        let world = {
            let driver = match simulation_driver.lock() {
                Ok(driver) => driver,
                Err(poisoned) => poisoned.into_inner(),
            };
            Arc::clone(&driver.world)
        };
        let mut inspector_state = InspectorState::default();
        if let Ok(world_guard) = world.lock() {
            let interval = world_guard.config().persistence_interval;
            if interval > 0 {
                inspector_state.persistence_last_enabled = interval;
            }
        }

        Self {
            world,
            simulation_driver,
            analytics_provider,
            title,
            command_submit,
            camera: Arc::new(Mutex::new(Camera::default())),
            world_raster_cache: Arc::new(Mutex::new(WorldRasterCache::default())),
            world_raster_cleanup_registered: false,
            #[cfg(test)]
            force_legacy_world_painter: false,
            inspector: Arc::new(Mutex::new(inspector_state)),
            selection_projection,
            selection_submission,
            playback: PlaybackState::new(240),
            perf: PerfStats::new(240),
            last_perf: PerfSnapshot::default(),
            forced_perf: None,
            accessibility: AccessibilitySettings::default(),
            debug: DebugOverlayState::default(),
            selection_events: VecDeque::with_capacity(MAX_SELECTION_EVENTS),
            controls: SimulationControls::default(),
            shift_inspect: false,
            bindings: InputBindings::default(),
            focus_handle: None,
            settings_panel: SettingsPanelState::default(),
            hud: HudLayout::default(),
            hud_rail_forced_closed: false,
            key_capture: None,
            analytics_cache: None,
            analytics_revision: None,
            analytics_status: StorageUiStatus::default(),
            #[cfg(feature = "audio")]
            audio: AudioState::new()
                .map_err(|err| {
                    error!(?err, "failed to initialize audio manager");
                    err
                })
                .ok(),
            minimal_canvas_mode: false,
            brain_history: std::collections::HashMap::new(),
            brain_client_id: BrainInspectionClientId::new(
                NEXT_BRAIN_INSPECTION_CLIENT.fetch_add(1, AtomicOrdering::Relaxed),
            ),
            brain_request_revision: BrainInspectionRevision::new(0),
            brain_inspection_cache: None,
            rail_visible: true,
            rail_selection: None,
            rail_selection_aged_out: false,
            rail_logged_first_show: false,
            rail_warned_aged_out: false,
            attribution_warned: std::collections::HashSet::new(),
            attribution_last_debug: None,
        }
    }

    fn set_minimal_canvas_mode(&mut self) {
        self.minimal_canvas_mode = true;
    }

    fn submit_control_command(&self, command: ControlCommand) -> bool {
        let tick = self
            .world
            .lock()
            .map(|world| world.tick().0)
            .unwrap_or_default();
        let accepted = (self.command_submit.as_ref())(command.clone());
        if accepted {
            debug!(tick, source = "gui", payload = ?command, "GPUI control command enqueued");
        } else {
            warn!(tick, source = "gui", payload = ?command, "failed to enqueue GPUI control command");
        }
        accepted
    }

    fn submit_simulation_command(&self, command: SimulationCommand) -> bool {
        self.submit_control_command(ControlCommand::UpdateSimulation(command))
    }

    fn submit_selection_update(&self, update: SelectionUpdate) -> bool {
        self.submit_control_command(ControlCommand::UpdateSelection(update))
    }

    fn camera_snapshot(&self) -> CameraSnapshot {
        self.camera
            .lock()
            .map(|camera| camera.snapshot())
            .unwrap_or_default()
    }

    fn simulation_drive_snapshot(&self) -> SimulationDriveSnapshot {
        let driver = match self.simulation_driver.lock() {
            Ok(driver) => driver,
            Err(poisoned) => poisoned.into_inner(),
        };
        driver.snapshot()
    }

    fn submit_config_update<F>(&self, update: F)
    where
        F: FnOnce(&mut ScriptBotsConfig),
    {
        if let Ok(world) = self.world.lock() {
            let mut new_config = world.config().clone();
            drop(world);
            update(&mut new_config);
            let _ = self.submit_control_command(ControlCommand::UpdateConfig(Box::new(new_config)));
        } else {
            warn!("failed to acquire world lock for config update");
        }
    }

    fn apply_preset(&mut self, preset: PresetKind, _cx: &mut Context<Self>) {
        self.submit_config_update(|config| preset.apply_to_config(config));
    }

    fn canvas_to_world(&self, position: Point<Pixels>) -> Option<(f32, f32)> {
        self.camera
            .lock()
            .ok()
            .and_then(|camera| camera.screen_to_world(position))
    }

    fn world_to_screen_coords(&self, position: Position) -> Option<(f32, f32)> {
        self.camera
            .lock()
            .ok()
            .and_then(|camera| camera.world_to_screen((position.x, position.y)))
    }

    fn fit_world_view(&self, cx: &mut Context<Self>) {
        if let Ok(mut camera) = self.camera.lock() {
            camera.cancel_initial_population_view();
            camera.fit_world();
        }
        cx.notify();
    }

    fn fit_selection_view(&self, bounds: (Position, Position), cx: &mut Context<Self>) {
        if let Ok(mut camera) = self.camera.lock() {
            camera.cancel_initial_population_view();
            camera.fit_bounds(bounds.0, bounds.1, 120.0);
        }
        cx.notify();
    }

    fn selection_bounds(&self, inspector: &InspectorSnapshot) -> Option<(Position, Position)> {
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        let mut push_pos = |pos: Position| {
            min_x = min_x.min(pos.x);
            min_y = min_y.min(pos.y);
            max_x = max_x.max(pos.x);
            max_y = max_y.max(pos.y);
        };

        for entry in &inspector.selected {
            push_pos(entry.position);
        }
        if let Some(detail) = inspector.focused.as_ref() {
            push_pos(detail.position);
        } else if let Some(hover) = inspector.hovered.as_ref() {
            push_pos(hover.position);
        }

        if min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite() {
            Some((Position::new(min_x, min_y), Position::new(max_x, max_y)))
        } else {
            None
        }
    }

    fn selection_pick_radius(&self, world: &WorldState) -> f32 {
        (world.config().bot_radius * 3.0).max(24.0)
    }

    fn effective_selected_agents(
        &self,
        canonical: &[AgentId],
        live_agents: &[AgentId],
    ) -> Vec<AgentId> {
        let Ok(mut projection) = self.selection_projection.lock() else {
            return canonical.to_vec();
        };
        if let Some(projected) = projection.as_mut() {
            projected.retain(|id| live_agents.contains(id));
        }
        if projection
            .as_deref()
            .is_some_and(|projected| projected == canonical)
        {
            *projection = None;
        }
        projection.clone().unwrap_or_else(|| canonical.to_vec())
    }

    fn pick_agent_near(
        &self,
        world: &WorldState,
        point: (f32, f32),
        radius: f32,
    ) -> Option<AgentId> {
        let arena = world.agents();
        let columns = arena.columns();
        let positions = columns.positions();
        let radius_sq = radius * radius;
        let extent_x = world.config().world_width as f32;
        let extent_y = world.config().world_height as f32;
        let mut best: Option<(AgentId, f32)> = None;

        for (idx, agent_id) in arena.iter_handles().enumerate() {
            let pos = positions[idx];
            let dx = toroidal_delta(point.0, pos.x, extent_x);
            let dy = toroidal_delta(point.1, pos.y, extent_y);
            let dist_sq = dx.mul_add(dx, dy * dy);
            if dist_sq <= radius_sq && best.is_none_or(|(_, best_dist)| dist_sq < best_dist) {
                best = Some((agent_id, dist_sq));
            }
        }

        best.map(|(id, _)| id)
    }

    fn clear_all_selections(&mut self) -> bool {
        let submission = Arc::clone(&self.selection_submission);
        let Ok(_submission_guard) = submission.lock() else {
            return false;
        };
        let (tick, canonical_selection, live_agents) = match self.world.lock() {
            Ok(world) => (
                world.tick().0,
                world
                    .runtime()
                    .iter()
                    .filter_map(|(id, entry)| {
                        matches!(entry.selection, SelectionState::Selected).then_some(id)
                    })
                    .collect::<Vec<_>>(),
                world.agents().iter_handles().collect::<Vec<_>>(),
            ),
            Err(_) => return false,
        };
        let selected = self.effective_selected_agents(&canonical_selection, &live_agents);
        let presentation_changed = self
            .inspector
            .lock()
            .map(|inspector| inspector.focused_agent.is_some() || inspector.hovered_agent.is_some())
            .unwrap_or(false);
        let changed = !selected.is_empty() || presentation_changed;
        if !self.submit_selection_update(SelectionUpdate {
            mode: SelectionMode::Clear,
            agent_ids: Vec::new(),
            state: SelectionState::None,
        }) {
            return false;
        }
        if let Ok(mut projection) = self.selection_projection.lock() {
            *projection = Some(Vec::new());
        }
        if let Ok(mut inspector) = self.inspector.lock() {
            inspector.focused_agent = None;
            inspector.hovered_agent = None;
        }
        if changed {
            self.record_selection_event_with_ids(tick, SelectionEventKind::Clear, &[]);
        }
        changed
    }

    fn update_selection_from_point(&mut self, position: Point<Pixels>, extend: bool) -> bool {
        let Some(world_point) = self.canvas_to_world(position) else {
            if extend {
                return false;
            }
            let cleared = self.clear_all_selections();
            return cleared;
        };
        let submission = Arc::clone(&self.selection_submission);
        let Ok(_submission_guard) = submission.lock() else {
            return false;
        };

        let prior_focus = self
            .inspector
            .lock()
            .map(|state| state.focused_agent)
            .unwrap_or(None);

        let world = match self.world.lock() {
            Ok(world) => world,
            Err(_) => return false,
        };

        let pick_radius = self.selection_pick_radius(&world);
        let candidate = self.pick_agent_near(&world, world_point, pick_radius);
        let tick = world.tick().0;
        let live_agents = world.agents().iter_handles().collect::<Vec<_>>();
        let canonical_selection = world
            .runtime()
            .iter()
            .filter_map(|(id, entry)| {
                matches!(entry.selection, SelectionState::Selected).then_some(id)
            })
            .collect::<Vec<_>>();
        drop(world);
        let selected_before = self.effective_selected_agents(&canonical_selection, &live_agents);
        let was_selected = candidate.is_some_and(|id| selected_before.contains(&id));

        let (mode, state, ids, selected_after, candidate_id) =
            match (extend, candidate, was_selected) {
                (false, Some(id), _) => (
                    SelectionMode::Replace,
                    SelectionState::Selected,
                    vec![id.raw()],
                    vec![id],
                    Some(id),
                ),
                (false, None, _) => (
                    SelectionMode::Clear,
                    SelectionState::None,
                    Vec::new(),
                    Vec::new(),
                    None,
                ),
                (true, Some(id), true) => {
                    let selected_after = selected_before
                        .iter()
                        .copied()
                        .filter(|selected| *selected != id)
                        .collect();
                    (
                        SelectionMode::Clear,
                        SelectionState::None,
                        vec![id.raw()],
                        selected_after,
                        None,
                    )
                }
                (true, Some(id), false) => {
                    let mut selected_after = selected_before.clone();
                    selected_after.push(id);
                    (
                        SelectionMode::Add,
                        SelectionState::Selected,
                        vec![id.raw()],
                        selected_after,
                        Some(id),
                    )
                }
                (true, None, _) => (
                    SelectionMode::Add,
                    SelectionState::Selected,
                    Vec::new(),
                    selected_before.clone(),
                    None,
                ),
            };
        let accepted = self.submit_selection_update(SelectionUpdate {
            mode,
            agent_ids: ids,
            state,
        });
        if !accepted {
            return false;
        }
        if let Ok(mut projection) = self.selection_projection.lock() {
            *projection = Some(selected_after.clone());
        }

        let mut focus_after = candidate_id.filter(|id| selected_after.contains(id));

        if extend {
            focus_after = focus_after
                .or_else(|| prior_focus.filter(|prev| selected_after.contains(prev)))
                .or_else(|| selected_after.first().copied());
        }

        let focus_changed = focus_after != prior_focus;
        let selection_changed = selected_after != selected_before;

        if let Ok(mut inspector) = self.inspector.lock() {
            inspector.focused_agent = focus_after;
            inspector.hovered_agent = None;
        }

        if selection_changed {
            self.record_selection_event_with_ids(tick, SelectionEventKind::Click, &selected_after);
        } else if focus_changed {
            self.record_selection_event_with_ids(tick, SelectionEventKind::Focus, &selected_after);
        }

        selection_changed || focus_changed
    }

    fn handle_canvas_hover(&mut self, event: &MouseMoveEvent) -> bool {
        let mut changed = self.set_shift_inspect(event.modifiers.shift);
        if self.update_hover_from_point(event.position) {
            changed = true;
        }
        changed
    }

    fn set_shift_inspect(&mut self, active: bool) -> bool {
        if self.shift_inspect != active {
            self.shift_inspect = active;
            true
        } else {
            false
        }
    }

    fn update_hover_from_point(&mut self, position: Point<Pixels>) -> bool {
        let hovered = if let Some(world_point) = self.canvas_to_world(position) {
            if let Ok(world) = self.world.lock() {
                let radius = self.selection_pick_radius(&world);
                self.pick_agent_near(&world, world_point, radius)
            } else {
                None
            }
        } else {
            None
        };

        self.apply_hover_change(hovered)
    }
    fn apply_hover_change(&mut self, hovered: Option<AgentId>) -> bool {
        let prev_hover = self
            .inspector
            .lock()
            .map(|state| state.hovered_agent)
            .unwrap_or(None);

        if prev_hover == hovered {
            return false;
        }

        let desired = if let Some(curr) = hovered {
            let Ok(world) = self.world.lock() else {
                return false;
            };
            let live_agents = world.agents().iter_handles().collect::<Vec<_>>();
            let canonical_selection = world
                .runtime()
                .iter()
                .filter_map(|(id, entry)| {
                    matches!(entry.selection, SelectionState::Selected).then_some(id)
                })
                .collect::<Vec<_>>();
            drop(world);
            (!self
                .effective_selected_agents(&canonical_selection, &live_agents)
                .contains(&curr))
            .then_some(curr)
        } else {
            None
        };

        let mut inspector_changed = false;
        if let Ok(mut inspector) = self.inspector.lock() {
            inspector_changed = inspector.hovered_agent != desired;
            inspector.hovered_agent = desired;
        }

        inspector_changed
    }
    fn snapshot(&mut self) -> HudSnapshot {
        let mut snapshot = HudSnapshot::default();
        let (canonical_selection, live_agents) = self
            .world
            .lock()
            .map(|world| {
                (
                    world
                        .runtime()
                        .iter()
                        .filter_map(|(id, entry)| {
                            matches!(entry.selection, SelectionState::Selected).then_some(id)
                        })
                        .collect::<Vec<_>>(),
                    world.agents().iter_handles().collect::<Vec<_>>(),
                )
            })
            .unwrap_or_default();
        let _ = self.effective_selected_agents(&canonical_selection, &live_agents);
        let inspector_state = self
            .inspector
            .lock()
            .map(|state| state.clone())
            .unwrap_or_default();
        let selection_projection = self
            .selection_projection
            .lock()
            .map(|projection| projection.clone())
            .unwrap_or_default();
        let brain_request = (!self.minimal_canvas_mode
            && matches!(self.playback.mode(), PlaybackMode::Live))
        .then(|| {
            let revision = self
                .brain_request_revision
                .get()
                .checked_add(1)
                .expect("GPUI brain-inspection revision exhausted");
            (self.brain_client_id, BrainInspectionRevision::new(revision))
        });
        let cached_brain_inspection = self.brain_inspection_cache.clone();
        let mut next_brain_inspection_cache = cached_brain_inspection.clone();
        let mut brain_request_issued = false;

        let analytics_trigger = {
            let mut trigger: Option<(u64, usize)> = None;
            if let Ok(world) = self.world.lock() {
                snapshot.tick = world.tick().0;
                snapshot.epoch = world.epoch();
                snapshot.is_closed = world.is_closed();
                snapshot.agent_count = world.agent_count();

                let config = world.config();
                snapshot.world_size = (config.world_width, config.world_height);
                snapshot.history_capacity = config.history_capacity;
                snapshot.narrative = world.narrative_events().iter().cloned().collect();
                snapshot.narrative_dropped = world.narrative_dropped_events();
                snapshot.narrative_capacity = config.narrative_capacity;
                snapshot.render_frame = RenderFrame::from_world(&world, self.accessibility.palette);
                if let Some(frame) = snapshot.render_frame.as_mut() {
                    for agent in &mut frame.agents {
                        agent.selection = if selection_projection
                            .as_ref()
                            .is_some_and(|selected| selected.contains(&agent.agent_id))
                            || (selection_projection.is_none()
                                && matches!(agent.selection, SelectionState::Selected))
                        {
                            SelectionState::Selected
                        } else {
                            SelectionState::None
                        };
                    }
                    if let Some(hovered_agent) = inspector_state.hovered_agent
                        && let Some(agent) = frame
                            .agents
                            .iter_mut()
                            .find(|agent| agent.agent_id == hovered_agent)
                        && !matches!(agent.selection, SelectionState::Selected)
                    {
                        agent.selection = SelectionState::Hovered;
                    }
                }

                let mut ring: VecDeque<TickSummary> = VecDeque::with_capacity(12);
                for summary in world.history() {
                    if ring.len() == 12 {
                        ring.pop_front();
                    }
                    ring.push_back(summary.clone());
                }
                if let Some(latest) = ring.back() {
                    snapshot.summary = Some(HudMetrics::from(latest));
                }
                snapshot.recent_history = ring.into_iter().map(HudHistoryEntry::from).collect();
                let (inspector, cache, issued) = InspectorSnapshot::from_world(
                    &world,
                    &inspector_state,
                    selection_projection.as_deref(),
                    cached_brain_inspection.as_ref(),
                    brain_request,
                );
                snapshot.inspector = inspector;
                next_brain_inspection_cache = cache;
                brain_request_issued = issued;

                trigger = Some((snapshot.tick, snapshot.agent_count));
            }
            trigger
        };

        self.brain_inspection_cache = next_brain_inspection_cache;
        if brain_request_issued && let Some((_, revision)) = brain_request {
            self.brain_request_revision = revision;
        }
        self.maybe_log_brain_panel(&snapshot);

        if let Some((tick, count)) = analytics_trigger {
            self.maybe_refresh_analytics(tick, count);
        }

        snapshot.analytics = self.analytics_cache.clone();
        snapshot.storage = self.analytics_status.clone();
        let simulation = self.simulation_drive_snapshot();
        snapshot.simulation_fault = simulation.simulation_fault;

        snapshot.perf = self.last_perf;
        snapshot.controls = self
            .controls
            .snapshot(simulation.paused, simulation.speed_multiplier);

        self.playback.record(&snapshot);

        snapshot
    }

    fn maybe_refresh_analytics(&mut self, live_tick: u64, live_agent_count: usize) {
        let published = self.analytics_provider.snapshot();
        let revision_changed = self.analytics_revision != Some(published.revision);
        self.analytics_revision = Some(published.revision);

        let committed_tick = published.committed_tick.unwrap_or(live_tick);
        self.analytics_status = StorageUiStatus {
            revision: published.revision,
            committed_tick: published.committed_tick,
            lag: published
                .committed_tick
                .map(|tick| live_tick.saturating_sub(tick)),
            last_error: published.last_error.as_deref().map(str::to_owned),
            stopped: published.stopped,
        };
        if !revision_changed {
            return;
        }

        let committed_agent_count = published.committed_agent_count.unwrap_or(live_agent_count);
        if let Some(analytics) = parse_analytics(
            committed_tick,
            committed_agent_count,
            published.readings.as_ref(),
        ) {
            self.analytics_cache = Some(analytics);
        }
    }

    fn render_header(&self, snapshot: &HudSnapshot, cx: &mut Context<Self>) -> Div {
        let theme = hud_theme(self.accessibility.palette);
        let controls = snapshot.controls;
        let subline = format!(
            "Tick #{}, epoch {}, {} active agents",
            snapshot.tick, snapshot.epoch, snapshot.agent_count
        );

        let badge_canvas = {
            let state = HeaderBadgeState {
                phase: snapshot.tick as f32 * 0.02,
                palette: self.accessibility.palette,
            };
            canvas(
                move |_, _, _| state,
                move |bounds, state, window, _| paint_header_badge(bounds, state, window),
            )
            .w(px(56.0))
            .h(px(56.0))
            .flex_none()
        };

        let mut chips_row = div().flex().gap_2().items_center();
        let run_label = if controls.paused { "Paused" } else { "Running" };
        let run_icon = if controls.paused { "⏸" } else { "▶" };
        let run_bg = if controls.paused {
            theme.chip_paused
        } else {
            theme.chip_running
        };
        chips_row = chips_row.child(self.header_chip(theme, run_icon, run_label, run_bg));

        let mut env_chip = self.header_chip(
            theme,
            if snapshot.is_closed { "🔒" } else { "🌐" },
            if snapshot.is_closed { "Closed" } else { "Open" },
            if snapshot.is_closed {
                theme.chip_closed
            } else {
                theme.chip_open
            },
        );
        env_chip = env_chip
            .cursor_pointer()
            .hover(|s| s.opacity(0.92))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _event: &MouseDownEvent, _, cx| {
                    this.toggle_closed_environment(cx);
                }),
            );
        chips_row = chips_row.child(env_chip);

        if !controls.paused {
            let speed = if (controls.speed_multiplier - 1.0).abs() < f32::EPSILON {
                "Speed ×1".to_string()
            } else {
                format!("Speed ×{}", self.format_float(controls.speed_multiplier, 2))
            };
            chips_row = chips_row.child(self.header_chip(theme, "⚡", speed, theme.chip_follow));
        }

        if !matches!(controls.follow_mode, FollowMode::Off) {
            chips_row = chips_row.child(self.header_chip(
                theme,
                "🎯",
                controls.follow_mode.label(),
                theme.chip_follow,
            ));
        }

        let fit_world_listener = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.fit_world_view(cx);
        });
        let fit_world_chip = self
            .header_chip(theme, "🧭", "Fit World", theme.chip_open)
            .cursor_pointer()
            .hover(|s| s.opacity(0.92))
            .on_mouse_down(MouseButton::Left, fit_world_listener);
        chips_row = chips_row.child(fit_world_chip);

        if let Some(bounds) = self.selection_bounds(&snapshot.inspector) {
            let min = (bounds.0.x, bounds.0.y);
            let max = (bounds.1.x, bounds.1.y);
            let fit_sel_listener = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
                this.fit_selection_view(
                    (Position::new(min.0, min.1), Position::new(max.0, max.1)),
                    cx,
                );
            });
            let fit_sel_chip = self
                .header_chip(theme, "🔎", "Fit Selection", theme.chip_follow)
                .cursor_pointer()
                .hover(|s| s.opacity(0.92))
                .on_mouse_down(MouseButton::Left, fit_sel_listener);
            chips_row = chips_row.child(fit_sel_chip);
        }

        div()
            .flex()
            .justify_between()
            .items_center()
            .gap_4()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(badge_canvas)
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_3xl()
                                    .text_color(rgb(theme.text_primary))
                                    .child(self.title.clone()),
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(rgb(theme.text_subtle))
                                    .child(subline),
                            ),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .items_end()
                    .gap_2()
                    .child(chips_row)
                    .child(
                        div()
                            .text_sm()
                            .text_color(rgb(theme.text_subtle))
                            .child(format!(
                                "World {}×{} · History cap {}",
                                snapshot.world_size.0,
                                snapshot.world_size.1,
                                snapshot.history_capacity
                            )),
                    ),
            )
    }
    fn render_summary(&self, snapshot: &HudSnapshot) -> Div {
        let theme = hud_theme(self.accessibility.palette);
        let mut cards: Vec<Div> = Vec::new();

        if let Some(metrics) = snapshot.summary.as_ref() {
            let growth = metrics.net_growth();
            let growth_accent = if growth >= 0 { 0x22c55e } else { 0xef4444 };
            let growth_label = if growth >= 0 {
                format!("Net +{}", growth)
            } else {
                format!("Net {}", growth)
            };

            let history = &snapshot.recent_history;
            let agents_series = sparkline_from_history(history, |entry| entry.agent_count as f32);
            let growth_series = sparkline_from_history(history, |entry| entry.net_growth() as f32);
            let energy_series = sparkline_from_history(history, |entry| entry.average_energy);
            let health_series = sparkline_from_history(history, |entry| entry.average_health);

            cards.push(self.metric_card(
                &theme,
                "Tick",
                metrics.tick.to_string(),
                0x38bdf8,
                Some(format!("Epoch {}", snapshot.epoch)),
                None,
            ));
            cards.push(self.metric_card(
                &theme,
                "Agents",
                metrics.agent_count.to_string(),
                0x22c55e,
                Some(format!("{} active", snapshot.agent_count)),
                agents_series.clone(),
            ));
            cards.push(self.metric_card(
                &theme,
                "Births / Deaths",
                format!("{} / {}", metrics.births, metrics.deaths),
                growth_accent,
                Some(growth_label),
                growth_series.clone(),
            ));
            cards.push(self.metric_card(
                &theme,
                "Avg Energy",
                self.format_float(metrics.average_energy, 2),
                0xf59e0b,
                Some(format!(
                    "Total {}",
                    self.format_float(metrics.total_energy, 1)
                )),
                energy_series.clone(),
            ));
            cards.push(self.metric_card(
                &theme,
                "Avg Health",
                self.format_float(metrics.average_health, 2),
                0x8b5cf6,
                None,
                health_series,
            ));
        } else {
            cards.push(
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .rounded_lg()
                    .border_1()
                    .border_color(rgb(theme.card_border))
                    .bg(rgb(theme.card_bg))
                    .p_5()
                    .child(
                        div()
                            .text_lg()
                            .text_color(rgb(theme.text_primary))
                            .child("No metrics yet"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(rgb(theme.text_subtle))
                            .child("Run the simulation to generate tick summaries."),
                    ),
            );
        }

        let perf = snapshot.perf;
        let frame_value = if perf.sample_count == 0 {
            "—".to_string()
        } else {
            format!("{} ms", self.format_float(perf.latest_ms, 2))
        };
        let frame_detail = if perf.sample_count == 0 {
            "Collecting samples…".to_string()
        } else {
            format!(
                "avg {} · min {} · max {}",
                self.format_float(perf.average_ms, 2),
                self.format_float(perf.min_ms, 2),
                self.format_float(perf.max_ms, 2)
            )
        };
        cards.push(self.metric_card(
            &theme,
            "Frame Time",
            frame_value,
            0x14b8a6,
            Some(frame_detail),
            None,
        ));

        let fps_value = if perf.sample_count == 0 {
            "—".to_string()
        } else {
            self.format_float(perf.fps, 1)
        };
        let fps_detail = if perf.sample_count == 0 {
            "Awaiting samples".to_string()
        } else {
            format!("Samples {}", perf.sample_count)
        };
        cards.push(self.metric_card(&theme, "FPS", fps_value, 0xf97316, Some(fps_detail), None));

        let controls = snapshot.controls;
        let speed_value = if controls.paused {
            "Paused".to_string()
        } else {
            format!("{}×", self.format_float(controls.speed_multiplier, 2))
        };
        let bool_label = |value: bool| if value { "On" } else { "Off" };
        let speed_detail = format!(
            "Agents {} · Food {} · Outline {} · {}",
            bool_label(controls.draw_agents),
            bool_label(controls.draw_food),
            bool_label(controls.agent_outline),
            controls.follow_mode.label()
        );
        cards.push(self.metric_card(
            &theme,
            "Sim Controls",
            speed_value,
            0x60a5fa,
            Some(speed_detail),
            None,
        ));

        // Use a conservative 3-column grid to avoid clipping on smaller window sizes; rows wrap gracefully.
        div().grid().grid_cols(3).gap_4().children(cards)
    }
    fn render_analytics_panel(&self, snapshot: &HudSnapshot) -> Div {
        let theme = hud_theme(self.accessibility.palette);
        let committed = snapshot
            .storage
            .committed_tick
            .map_or_else(|| "pending".to_owned(), |tick| format!("t{tick}"));
        let (storage_state, storage_color) = if let Some(error) = &snapshot.storage.last_error {
            (format!("error: {error}"), 0xf87171)
        } else if snapshot.storage.stopped {
            ("stopped".to_owned(), 0xfbbf24)
        } else {
            ("active".to_owned(), 0x4ade80)
        };
        let lag = snapshot
            .storage
            .lag
            .map_or_else(|| "unknown".to_owned(), |ticks| ticks.to_string());
        let status_text = format!(
            "FrankenSQLite r{} · committed {} · lag {} · {}",
            snapshot.storage.revision, committed, lag, storage_state
        );
        let storage_bar = div()
            .text_xs()
            .text_color(rgb(storage_color))
            .child(status_text);
        let simulation_bar = snapshot.simulation_fault.as_ref().map(|error| {
            div()
                .text_xs()
                .text_color(rgb(0xf87171))
                .child(format!("Simulation fault · {error}"))
        });

        let Some(analytics) = snapshot.analytics.as_ref() else {
            return div()
                .flex()
                .flex_col()
                .gap_2()
                .children(simulation_bar)
                .child(storage_bar)
                .child(
                    div()
                        .text_sm()
                        .text_color(rgb(theme.text_subtle))
                        .child("Analytics warming up; waiting for the first durable commit."),
                );
        };

        let total_agents = snapshot
            .summary
            .as_ref()
            .map(|metrics| metrics.agent_count)
            .unwrap_or(snapshot.agent_count)
            .max(1);

        let share_detail = |count: usize, avg_energy: f64| -> String {
            let share = (count as f64 / total_agents as f64 * 100.0).clamp(0.0, 100.0);
            format!("{share:.1}% share · avg ⚡ {avg_energy:.2}")
        };

        let trophic_cards = vec![
            self.metric_card(
                &theme,
                "Carnivores",
                analytics.carnivores.to_string(),
                0xcb2a3b,
                Some(share_detail(
                    analytics.carnivores,
                    analytics.carnivore_avg_energy,
                )),
                None,
            ),
            self.metric_card(
                &theme,
                "Herbivores",
                analytics.herbivores.to_string(),
                0x22c55e,
                Some(share_detail(
                    analytics.herbivores,
                    analytics.herbivore_avg_energy,
                )),
                None,
            ),
            self.metric_card(
                &theme,
                "Hybrids",
                analytics.hybrids.to_string(),
                0x8b5cf6,
                Some(share_detail(analytics.hybrids, analytics.hybrid_avg_energy)),
                None,
            ),
        ];

        let trophic_row = div().grid().grid_cols(3).gap_4().children(trophic_cards);

        let meta_bar = div()
            .flex()
            .justify_between()
            .gap_4()
            .text_xs()
            .text_color(rgb(theme.text_subtle))
            .child(div().child(format!("Tick {}", analytics.tick)))
            .child(div().child(format!("Boosts {}", analytics.boost_count)))
            .child(div().child(format!("Births {}", analytics.births_total)));

        let resource_panel = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_lg()
            .bg(rgb(theme.card_bg))
            .border_1()
            .border_color(rgb(theme.card_border))
            .child(div().text_sm().text_color(rgb(0x7dd3fc)).child("Resources"))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_primary))
                    .child(format!(
                        "Total {:.1} · Mean {:.3} · σ {:.3}",
                        analytics.food_total, analytics.food_mean, analytics.food_stddev
                    )),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_subtle))
                    .child(format!(
                        "Δ mean {:.4} · |Δ| {:.4}",
                        analytics.food_delta_mean, analytics.food_delta_mean_abs
                    )),
            );

        let mutation_panel = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_lg()
            .bg(rgb(theme.card_bg))
            .border_1()
            .border_color(rgb(theme.card_border))
            .child(div().text_sm().text_color(rgb(0xfbbf24)).child("Mutation"))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_primary))
                    .child(format!(
                        "Primary {:.4} ± {:.4}",
                        analytics.mutation_primary_mean, analytics.mutation_primary_stddev
                    )),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_primary))
                    .child(format!(
                        "Secondary {:.4} ± {:.4}",
                        analytics.mutation_secondary_mean, analytics.mutation_secondary_stddev
                    )),
            );

        let behavior_panel = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_lg()
            .bg(rgb(theme.card_bg))
            .border_1()
            .border_color(rgb(theme.card_border))
            .child(div().text_sm().text_color(rgb(0x93c5fd)).child("Behavior"))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_primary))
                    .child(format!(
                        "Sensors μ {:.3} · H {:.3}",
                        analytics.behavior_sensor_mean, analytics.behavior_sensor_entropy
                    )),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_primary))
                    .child(format!(
                        "Outputs μ {:.3} · H {:.3}",
                        analytics.behavior_output_mean, analytics.behavior_output_entropy
                    )),
            );
        let age_panel = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_lg()
            .bg(rgb(theme.card_bg))
            .border_1()
            .border_color(rgb(theme.card_border))
            .child(div().text_sm().text_color(rgb(0xf59e0b)).child("Age"))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_primary))
                    .child(format!(
                        "Mean {:.2} · Max {:.0} · Gen μ {:.1}",
                        analytics.age_mean, analytics.age_max, analytics.generation_mean
                    )),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_primary))
                    .child(format!(
                        "Gen max {:.0} · Hybrid births {} ({:.1}%)",
                        analytics.generation_max,
                        analytics.births_hybrid,
                        analytics.births_hybrid_ratio * 100.0
                    )),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_subtle))
                    .child(format!(
                        "Repro μ {:.2} · Boost {:.1}%",
                        analytics.reproduction_counter_mean,
                        analytics.boost_ratio * 100.0
                    )),
            );

        let temperature_panel = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_3()
            .rounded_lg()
            .bg(rgb(theme.card_bg))
            .border_1()
            .border_color(rgb(theme.card_border))
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x38bdf8))
                    .child("Temperature"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_primary))
                    .child(format!(
                        "Preference μ {:.3} · σ {:.3}",
                        analytics.temperature_preference_mean,
                        analytics.temperature_preference_stddev
                    )),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_subtle))
                    .child(format!(
                        "Discomfort μ {:.3} · σ {:.3}",
                        analytics.temperature_discomfort_mean,
                        analytics.temperature_discomfort_stddev
                    )),
            );

        let mortality_panel = {
            let total = analytics.deaths_total.max(1);
            let make_row = |label: &str, count: usize| {
                let ratio = (count as f64 / total as f64) * 100.0;
                let label_text: SharedString = label.to_string().into();
                div()
                    .flex()
                    .justify_between()
                    .text_xs()
                    .text_color(rgb(theme.text_primary))
                    .child(div().child(label_text))
                    .child(div().child(format!("{count} ({ratio:.1}%)")))
            };

            div()
                .flex()
                .flex_col()
                .gap_1()
                .p_3()
                .rounded_lg()
                .bg(rgb(theme.card_bg))
                .border_1()
                .border_color(rgb(theme.card_border))
                .child(div().text_sm().text_color(rgb(0xf472b6)).child("Mortality"))
                .child(make_row("Carnivore", analytics.deaths_combat_carnivore))
                .child(make_row("Herbivore", analytics.deaths_combat_herbivore))
                .child(make_row("Starvation", analytics.deaths_starvation))
                .child(make_row("Aging", analytics.deaths_aging))
                .child(make_row("Other", analytics.deaths_unknown))
                .child(
                    div()
                        .flex()
                        .justify_between()
                        .text_xs()
                        .text_color(rgb(theme.text_subtle))
                        .child(div().child("Total"))
                        .child(div().child(analytics.deaths_total.to_string())),
                )
        };

        // Reduce to 2 columns to prevent overlap on narrow screens; content flows to multiple rows.
        let insights_row = div().grid().grid_cols(2).gap_4().children(vec![
            resource_panel,
            mutation_panel,
            behavior_panel,
            age_panel,
            temperature_panel,
            mortality_panel,
        ]);

        let mut brain_rows: Vec<Div> = Vec::new();
        brain_rows.push(
            div()
                .flex()
                .text_xs()
                .text_color(rgb(theme.text_subtle))
                .gap_4()
                .child(div().w(px(140.0)).child("BRAIN"))
                .child(div().w(px(80.0)).child("COUNT"))
                .child(div().w(px(80.0)).child("SHARE"))
                .child(div().w(px(100.0)).child("AVG ENERGY")),
        );

        if analytics.brain_shares.is_empty() {
            brain_rows.push(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_subtle))
                    .child("No brain metrics yet"),
            );
        } else {
            for entry in analytics.brain_shares.iter().take(6) {
                let share = (entry.count as f64 / total_agents as f64 * 100.0).clamp(0.0, 100.0);
                brain_rows.push(
                    div()
                        .flex()
                        .gap_4()
                        .items_center()
                        .text_xs()
                        .text_color(rgb(theme.text_primary))
                        .child(div().w(px(140.0)).child(entry.label.clone()))
                        .child(div().w(px(80.0)).child(entry.count.to_string()))
                        .child(div().w(px(80.0)).child(format!("{share:.1}%")))
                        .child(div().w(px(100.0)).child(format!("{:.3}", entry.avg_energy))),
                );
            }
        }

        let brain_panel = div()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .rounded_lg()
            .bg(rgb(theme.card_bg))
            .border_1()
            .border_color(rgb(theme.card_border))
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x38bdf8))
                    .child("Brain Share"),
            )
            .children(brain_rows);

        div()
            .flex()
            .flex_col()
            .gap_4()
            .children(simulation_bar)
            .child(storage_bar)
            .child(meta_bar)
            .child(trophic_row)
            .child(insights_row)
            .child(brain_panel)
    }
    fn render_history(&self, snapshot: &HudSnapshot) -> Div {
        let theme = hud_theme(self.accessibility.palette);
        let header = div()
            .flex()
            .justify_between()
            .items_center()
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(theme.text_primary))
                    .child("Recent Tick History"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(theme.text_subtle))
                    .child(format!(
                        "Showing {} of {} entries",
                        snapshot.recent_history.len(),
                        snapshot.history_capacity
                    )),
            );

        let rows: Vec<Div> = if snapshot.recent_history.is_empty() {
            vec![
                div()
                    .rounded_lg()
                    .bg(rgb(theme.card_bg))
                    .border_1()
                    .border_color(rgb(theme.card_border))
                    .p_4()
                    .child(
                        div()
                            .text_sm()
                            .text_color(rgb(theme.text_subtle))
                            .child("No persisted tick history yet."),
                    ),
            ]
        } else {
            snapshot
                .recent_history
                .iter()
                .enumerate()
                .map(|(idx, entry)| {
                    let row_bg = if idx % 2 == 0 {
                        rgb(theme.card_bg)
                    } else {
                        rgb(0x101a2e)
                    };
                    let growth = entry.net_growth();
                    let growth_color = if growth >= 0 {
                        rgb(0x22c55e)
                    } else {
                        rgb(0xef4444)
                    };
                    let growth_label = if growth >= 0 {
                        format!("+{}", growth)
                    } else {
                        growth.to_string()
                    };

                    div()
                        .flex()
                        .justify_between()
                        .items_center()
                        .rounded_lg()
                        .bg(row_bg)
                        .p_3()
                        .child(
                            div()
                                .text_sm()
                                .text_color(rgb(theme.text_subtle))
                                .child(format!("Tick {}", entry.tick)),
                        )
                        .child(
                            div()
                                .flex()
                                .gap_4()
                                .items_center()
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(rgb(theme.text_primary))
                                        .child(format!("Agents {}", entry.agent_count)),
                                )
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(rgb(0xf97316))
                                        .child(format!("Births {}", entry.births)),
                                )
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(rgb(0x38bdf8))
                                        .child(format!("Deaths {}", entry.deaths)),
                                )
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(growth_color)
                                        .child(format!("Δ {}", growth_label)),
                                )
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(rgb(theme.text_primary))
                                        .child(format!("⌀ energy {:.2}", entry.average_energy)),
                                ),
                        )
                })
                .collect()
        };

        div()
            .flex()
            .flex_col()
            .w(px(280.0))
            .flex_none()
            .bg(rgb(theme.card_bg))
            .border_1()
            .border_color(rgb(theme.card_border))
            .rounded_xl()
            .shadow_lg()
            .p_4()
            .gap_3()
            .child(header)
            .children(rows)
    }

    /// The attribution panel's logging contract (bd-16g.4.3): a standalone debug
    /// line per probed tick, warn-once-per-(agent, reason) on Unavailable, and a
    /// warn whenever non-finite values had to be excluded.
    fn maybe_log_brain_panel(&mut self, snapshot: &HudSnapshot) {
        let Some(capture) = &self.brain_inspection_cache else {
            return;
        };
        let uid = capture.agent_uid.get();
        let Some(detail) = snapshot.inspector.focused.as_ref() else {
            return;
        };
        if detail.outputs.len() < scriptbots_core::OUTPUT_SIZE {
            return;
        }
        let outputs: &[f32; scriptbots_core::OUTPUT_SIZE] = detail.outputs
            [..scriptbots_core::OUTPUT_SIZE]
            .try_into()
            .expect("length checked above");
        let explanations = explain_outputs(
            outputs,
            detail.brain_bound,
            detail.brain_activations.as_ref(),
            BRAIN_PANEL_TOP_K,
        );
        if self.attribution_last_debug != Some((uid, snapshot.tick)) {
            self.attribution_last_debug = Some((uid, snapshot.tick));
            let named_outputs: Vec<String> = explanations
                .iter()
                .map(|explanation| {
                    let effective = match &explanation.effective {
                        EffectiveOutput::Continuous(value) => format!("{value:.3}"),
                        EffectiveOutput::Thresholded { raw, active, .. } => {
                            format!("{raw:.3}/{}", if *active { "ON" } else { "OFF" })
                        }
                        EffectiveOutput::Clamped { raw, applied } => {
                            format!("{raw:.3}>{applied:.3}")
                        }
                    };
                    format!("{}={effective}", explanation.output_name)
                })
                .collect();
            let wheels: Vec<String> = explanations
                .iter()
                .take(2)
                .map(|explanation| {
                    let top: Vec<String> = explanation
                        .inputs
                        .iter()
                        .take(3)
                        .map(|input| format!("{} {:+.3}", input.sensor_name, input.contribution))
                        .collect();
                    format!(
                        "{}={:.3}({})",
                        explanation.output_name,
                        explanation.raw_value,
                        top.join(", ")
                    )
                })
                .collect();
            tracing::debug!(
                target = "scriptbots::brain_panel",
                agent_uid = uid,
                tick = snapshot.tick,
                outputs = %named_outputs.join(" "),
                wheels = %wheels.join(" | "),
                "brain panel probed tick"
            );
        }
        for explanation in &explanations {
            if let AttributionMethod::Unavailable(reason) = explanation.method
                && self.attribution_warned.insert((uid, reason.reason()))
            {
                tracing::warn!(
                    target = "scriptbots::brain_panel",
                    agent_uid = uid,
                    reason = reason.reason(),
                    "brain attribution unavailable"
                );
            }
            if explanation.non_finite_skipped > 0 {
                tracing::warn!(
                    target = "scriptbots::brain_panel",
                    agent_uid = uid,
                    output = explanation.output_name,
                    non_finite_skipped = explanation.non_finite_skipped,
                    "non-finite values excluded from brain attribution"
                );
            }
        }
    }

    /// The narrative rail (bd-16g.2.4): a read-only projection of `RunNarrative`
    /// along the bottom of the canvas — one glyph per retained event, coloured by
    /// the shared core rail model (the same table the TUI reads), click to select.
    /// STAGE 1 of the seek contract: selection never moves the simulation clock.
    fn render_narrative_rail(&self, snapshot: &HudSnapshot, cx: &mut Context<Self>) -> Div {
        let events = &snapshot.narrative;
        let mut rail = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_2()
            .rounded_md()
            .bg(rgb(0x111c33))
            .border_1()
            .border_color(rgb(0x24334f));

        let mut header_row = div().flex().justify_between().items_center().child(
            div()
                .text_xs()
                .text_color(rgb(0x94a3b8))
                .child("Timeline — run history (select-only; rewind needs replay bd-2z0.5.3)"),
        );
        let toggle_listener = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.rail_visible = !this.rail_visible;
            cx.notify();
        });
        header_row = header_row.child(
            div()
                .text_xs()
                .text_color(rgb(0x60a5fa))
                .cursor_pointer()
                .child("hide")
                .on_mouse_down(MouseButton::Left, toggle_listener),
        );
        rail = rail.child(header_row);

        if events.is_empty() {
            return rail.child(div().text_xs().text_color(rgb(0x64748b)).child(
                "no narrative events yet — the run's story will appear here as it happens",
            ));
        }

        // Glyph row: an explicit truncation marker first, then one clickable glyph
        // per retained event with the selection highlighted. Bounded work: at most
        // `narrative_capacity` cells, reusing the retained slice directly.
        let selection = self
            .rail_selection
            .map_or(events.len() - 1, |(index, _, _)| {
                index.min(events.len() - 1)
            });
        let mut glyph_row = div().flex().gap_1().items_center().flex_wrap();
        if snapshot.narrative_dropped > 0 {
            glyph_row = glyph_row.child(
                div()
                    .text_xs()
                    .text_color(rgb(0x64748b))
                    .child(format!("+{}…", snapshot.narrative_dropped)),
            );
        }
        for (index, event) in events.iter().enumerate() {
            let [r, g, b] = event.kind.rail_rgb();
            let color = (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
            let selected = index == selection;
            let listener = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
                this.select_rail_event(index);
                cx.notify();
            });
            let mut glyph = div()
                .cursor_pointer()
                .text_sm()
                .text_color(rgb(color))
                .child(event.kind.rail_glyph().to_string());
            if selected {
                glyph = glyph
                    .rounded_sm()
                    .bg(rgb(0x24334f))
                    .border_1()
                    .border_color(rgb(color));
            }
            glyph_row = glyph_row.child(glyph.on_mouse_down(MouseButton::Left, listener));
        }
        rail = rail.child(glyph_row);

        // Detail line: the selected event's full text, plus the honest markers.
        let selected_event = &events[selection];
        let aged_out = if self.rail_selection_aged_out {
            " | selected event aged out of the ring"
        } else {
            ""
        };
        let truncated = if snapshot.narrative_dropped > 0 {
            format!(
                " | {} earlier events dropped — this is a TAIL of the run's history",
                snapshot.narrative_dropped
            )
        } else {
            String::new()
        };
        rail = rail.child(div().text_xs().text_color(rgb(0xe2e8f0)).child(format!(
            "tick {} | {} | severity {:.2} — {}{aged_out}{truncated}",
            selected_event.tick.0,
            selected_event.kind.as_str(),
            selected_event.severity,
            selected_event.human_text
        )));
        rail
    }

    /// Select a rail event by index, clamped to the retained events. A fresh
    /// selection clears the aged-out marker (bd-16g.2.4).
    fn select_rail_event(&mut self, index: usize) {
        let events_len = {
            let world = self.world.lock().expect("world mutex poisoned");
            world.narrative_events().len()
        };
        if events_len == 0 {
            self.rail_selection = None;
            return;
        }
        let clamped = index.min(events_len - 1);
        let (tick, kind) = {
            let world = self.world.lock().expect("world mutex poisoned");
            let event = &world.narrative_events()[clamped];
            (event.tick.0, event.kind)
        };
        self.rail_selection = Some((clamped, tick, kind));
        self.rail_selection_aged_out = false;
        self.rail_warned_aged_out = false;
        tracing::debug!(
            target = "scriptbots::timeline",
            event_index = clamped,
            event_tick = tick,
            event_kind = kind.as_str(),
            "narrative rail selection moved"
        );
    }

    /// Keep the rail selection pointing at a live event. The ring can drop the
    /// very event the user selected; that must clamp loudly, never index into a
    /// dropped slot (bd-16g.2.4).
    fn validate_rail_selection(&mut self, snapshot: &HudSnapshot) {
        let Some((index, tick, kind)) = self.rail_selection else {
            return;
        };
        let events = &snapshot.narrative;
        let still_live = events
            .get(index)
            .is_some_and(|event| event.tick.0 == tick && event.kind == kind);
        if still_live {
            return;
        }
        if events.is_empty() {
            self.rail_selection = None;
        } else {
            let newest = events.len() - 1;
            self.rail_selection = Some((newest, events[newest].tick.0, events[newest].kind));
        }
        self.rail_selection_aged_out = true;
        if !self.rail_warned_aged_out {
            self.rail_warned_aged_out = true;
            tracing::warn!(
                target = "scriptbots::timeline",
                event_tick = tick,
                dropped_count = snapshot.narrative_dropped,
                "selected narrative event was dropped by the bounded ring"
            );
        }
    }

    /// The rail's first-show logging contract (bd-16g.2.4): one line from which a
    /// reader can tell whether they are looking at a complete history or a tail.
    fn maybe_log_rail_first_show(&mut self, snapshot: &HudSnapshot) {
        if !self.rail_visible || self.rail_logged_first_show || snapshot.narrative.is_empty() {
            return;
        }
        self.rail_logged_first_show = true;
        tracing::info!(
            target = "scriptbots::timeline",
            retained_events = snapshot.narrative.len(),
            dropped_events = snapshot.narrative_dropped,
            capacity = snapshot.narrative_capacity,
            oldest_tick = snapshot.narrative.first().map_or(0, |event| event.tick.0),
            newest_tick = snapshot.narrative.last().map_or(0, |event| event.tick.0),
            "narrative rail first shown"
        );
    }

    fn render_canvas(
        &self,
        snapshot: &HudSnapshot,
        resolved: ResolvedHudLayout,
        cx: &mut Context<Self>,
    ) -> Div {
        if let Some(frame) = snapshot.render_frame.clone() {
            self.render_canvas_world(snapshot, frame, resolved, cx)
        } else {
            self.render_canvas_placeholder(snapshot)
        }
    }
    fn render_canvas_world(
        &self,
        snapshot: &HudSnapshot,
        frame: RenderFrame,
        _resolved: ResolvedHudLayout,
        cx: &mut Context<Self>,
    ) -> Div {
        let follow_target = self.compute_follow_target(&frame, &snapshot.inspector);
        let canvas_state = CanvasState {
            frame: frame.clone(),
            camera: Arc::clone(&self.camera),
            world_raster_cache: Arc::clone(&self.world_raster_cache),
            #[cfg(test)]
            force_legacy_world_painter: self.force_legacy_world_painter,
            focus_agent: snapshot.inspector.focus_id,
            controls: snapshot.controls,
            debug: self.debug,
            follow_target,
            perf: snapshot.perf,
        };

        let canvas_element = canvas(
            move |_, _, _| canvas_state.clone(),
            move |bounds, state, window, _| {
                #[cfg(feature = "world_wgpu")]
                if use_wgpu_renderer() {
                    paint_world_with_wgpu(&state, bounds, window);
                    return;
                }
                paint_frame(&state, bounds, window)
            },
        )
        .flex_1();

        let canvas_stack = div()
            .relative()
            .flex_1()
            .h_full()
            .min_h(px(400.0))
            .flex_grow(1.0)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &MouseDownEvent, _, cx| {
                    let extend = event.modifiers.shift;
                    // Snapshot focus state BEFORE mutating selection
                    let had_focus_before = if !extend {
                        this.inspector
                            .lock()
                            .map(|s| s.focused_agent.is_some())
                            .unwrap_or(false)
                    } else {
                        false
                    };

                    let mut changed = this.update_selection_from_point(event.position, extend);

                    // Auto-follow behavior: when a single-click focuses an agent (no Shift),
                    // enable Follow selected; clicking empty space turns follow off.
                    if !extend {
                        let has_focus_after = this
                            .inspector
                            .lock()
                            .map(|s| s.focused_agent.is_some())
                            .unwrap_or(false);
                        if has_focus_after {
                            this.controls.follow_mode = FollowMode::Selected;
                            cx.notify();
                        } else if had_focus_before && !has_focus_after {
                            this.controls.follow_mode = FollowMode::Off;
                            cx.notify();
                        }
                    }
                    if this.set_shift_inspect(event.modifiers.shift) {
                        changed = true;
                    }
                    if changed {
                        cx.notify();
                    }
                }),
            )
            .on_mouse_down(
                MouseButton::Middle,
                cx.listener(|this, event: &MouseDownEvent, _, cx| {
                    if let Ok(mut camera) = this.camera.lock() {
                        camera.start_pan(event.position);
                        cx.notify();
                    }
                }),
            )
            .on_mouse_up(
                MouseButton::Middle,
                cx.listener(|this, _event: &MouseUpEvent, _, _| {
                    if let Ok(mut camera) = this.camera.lock() {
                        camera.end_pan();
                    }
                }),
            )
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                let mut changed = false;
                let mut panning = false;
                if let Ok(mut camera) = this.camera.lock() {
                    if camera.update_pan(event.position) {
                        changed = true;
                    }
                    panning = camera.is_panning();
                }

                if panning {
                    if this.apply_hover_change(None) {
                        changed = true;
                    }
                    if this.set_shift_inspect(false) {
                        changed = true;
                    }
                } else if this.handle_canvas_hover(event) {
                    changed = true;
                }

                if changed {
                    cx.notify();
                }
            }))
            .on_scroll_wheel(cx.listener(|this, event: &ScrollWheelEvent, _, cx| {
                if let Ok(mut camera) = this.camera.lock()
                    && camera.apply_scroll(event)
                {
                    cx.notify();
                }
            }))
            .child(canvas_element);

        // bd-v9cz layout policy. The world pane and the rail are FLEX SIBLINGS, not a
        // stack: previously `render_overlay` and `render_history_chart` were absolutely
        // positioned children of this same element, so they punched opaque rectangles
        // through the world and — because the mouse handlers live on the parent — a
        // click on a stats panel also selected whatever agent sat underneath it.
        // Docking makes overlap structurally impossible rather than a rule to remember.
        let world_row = div()
            .flex()
            .flex_row()
            .flex_1()
            .h_full()
            .min_h(px(400.0))
            .gap_3()
            .child(canvas_stack);

        // The rail is no longer nested here — it is a sibling of this canvas in the
        // HUD row (see render()). Nesting it made the rail invisible to any path that
        // did not go through render_canvas, and fed the resize rule a width this
        // container never had.
        let canvas_stack = world_row;

        let camera_snapshot = self.camera_snapshot();
        let footer = div()
            .text_xs()
            .text_color(rgb(0x475569))
            .flex()
            .justify_between()
            .child(format!(
                "World {:.0}×{:.0} units • Zoom {:.2}×",
                frame.world_size.0, frame.world_size.1, camera_snapshot.zoom,
            ))
            .child(format!(
                "Pan X {:.1}, Y {:.1}",
                camera_snapshot.offset_px.0, camera_snapshot.offset_px.1
            ));

        div()
            .flex()
            .flex_col()
            .flex_1()
            .rounded_xl()
            .border_1()
            .border_color(rgb(0x0ea5e9))
            .bg(rgb(0x0b1120))
            .shadow_lg()
            .p_4()
            .gap_3()
            .child(canvas_stack)
            .child(footer)
    }

    fn compute_follow_target(
        &self,
        frame: &RenderFrame,
        inspector: &InspectorSnapshot,
    ) -> Option<Position> {
        match self.controls.follow_mode {
            FollowMode::Off => None,
            FollowMode::Selected => inspector.focus_id.and_then(|id| {
                frame
                    .agents
                    .iter()
                    .find(|agent| agent.agent_id == id)
                    .map(|agent| agent.position)
            }),
            FollowMode::Oldest => frame
                .agents
                .iter()
                .max_by_key(|agent| agent.age)
                .map(|agent| agent.position),
        }
    }

    fn focus_agent(&mut self, agent_id: AgentId, cx: &mut Context<Self>) {
        if let Ok(mut inspector) = self.inspector.lock() {
            inspector.focused_agent = Some(agent_id);
        }
        cx.notify();
    }

    fn set_brush_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if let Ok(mut inspector) = self.inspector.lock() {
            inspector.brush_enabled = enabled;
        }
        cx.notify();
    }

    fn adjust_brush_radius(&mut self, delta: f32, cx: &mut Context<Self>) {
        if let Ok(mut inspector) = self.inspector.lock() {
            let mut radius = inspector.brush_radius + delta;
            radius = radius.clamp(8.0, 256.0);
            inspector.brush_radius = radius;
        }
        cx.notify();
    }

    fn set_probe_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if let Ok(mut inspector) = self.inspector.lock() {
            inspector.probe_enabled = enabled;
        }
        cx.notify();
    }

    fn set_debug_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.debug.enabled = enabled;
        cx.notify();
    }

    fn set_debug_show_velocity(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.debug.show_velocity = enabled;
        cx.notify();
    }

    fn set_debug_show_sense_radius(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.debug.show_sense_radius = enabled;
        cx.notify();
    }

    fn record_selection_event_with_ids(
        &mut self,
        tick: u64,
        kind: SelectionEventKind,
        selected: &[AgentId],
    ) {
        self.push_selection_event(
            tick,
            kind,
            selected.len(),
            selected.iter().copied().take(5).collect(),
        );
    }

    fn push_selection_event(
        &mut self,
        tick: u64,
        kind: SelectionEventKind,
        total_selected: usize,
        sample_ids: Vec<AgentId>,
    ) {
        let event = SelectionEvent {
            tick,
            kind,
            total_selected,
            sample_ids,
        };
        if self.selection_events.len() >= MAX_SELECTION_EVENTS {
            self.selection_events.pop_front();
        }
        self.selection_events.push_back(event);
    }
    fn clear_selection(&mut self, cx: &mut Context<Self>) {
        if self.clear_all_selections() {
            cx.notify();
        }
    }
    fn select_all_agents(&mut self, cx: &mut Context<Self>) {
        let submission = Arc::clone(&self.selection_submission);
        let Ok(_submission_guard) = submission.lock() else {
            return;
        };
        let (tick, ids, canonical_selection) = match self.world.lock() {
            Ok(world) => (
                world.tick().0,
                world.agents().iter_handles().collect::<Vec<_>>(),
                world
                    .runtime()
                    .iter()
                    .filter_map(|(id, entry)| {
                        matches!(entry.selection, SelectionState::Selected).then_some(id)
                    })
                    .collect::<Vec<_>>(),
            ),
            Err(_) => return,
        };
        let selected_before = self.effective_selected_agents(&canonical_selection, &ids);
        let prior_presentation = self
            .inspector
            .lock()
            .map(|inspector| (inspector.focused_agent, inspector.hovered_agent))
            .unwrap_or_default();
        let changed = selected_before != ids
            || prior_presentation.0 != ids.first().copied()
            || prior_presentation.1.is_some();
        if !self.submit_selection_update(SelectionUpdate {
            mode: SelectionMode::Replace,
            agent_ids: ids.iter().copied().map(AgentId::raw).collect(),
            state: SelectionState::Selected,
        }) {
            return;
        }
        if let Ok(mut projection) = self.selection_projection.lock() {
            *projection = Some(ids.clone());
        }
        if let Ok(mut inspector) = self.inspector.lock() {
            inspector.focused_agent = ids.first().copied();
            inspector.hovered_agent = None;
        }
        if changed {
            self.record_selection_event_with_ids(tick, SelectionEventKind::SelectAll, &ids);
            cx.notify();
        }
    }

    fn focus_first_selected(&mut self, cx: &mut Context<Self>) {
        let (tick, canonical_selection, live_agents) = match self.world.lock() {
            Ok(world) => (
                world.tick().0,
                world
                    .runtime()
                    .iter()
                    .filter_map(|(id, entry)| {
                        matches!(entry.selection, SelectionState::Selected).then_some(id)
                    })
                    .collect::<Vec<_>>(),
                world.agents().iter_handles().collect::<Vec<_>>(),
            ),
            Err(_) => return,
        };
        let selected = self.effective_selected_agents(&canonical_selection, &live_agents);
        let selected_id = selected.first().copied();

        if let Some(id) = selected_id {
            self.focus_agent(id, cx);
            self.record_selection_event_with_ids(tick, SelectionEventKind::Focus, &selected);
        }
    }

    fn set_persistence_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if enabled {
            let interval = self
                .inspector
                .lock()
                .map(|mut inspector| {
                    inspector.persistence_last_enabled = inspector.persistence_last_enabled.max(1);
                    inspector.persistence_last_enabled
                })
                .unwrap_or(60);

            self.submit_config_update(|config| {
                config.persistence_interval = interval;
            });
        } else {
            let current_interval = self
                .world
                .lock()
                .map(|world| world.config().persistence_interval)
                .unwrap_or(0);

            if current_interval > 0
                && let Ok(mut inspector) = self.inspector.lock()
            {
                inspector.persistence_last_enabled = current_interval;
            }

            self.submit_config_update(|config| {
                config.persistence_interval = 0;
            });
        }

        cx.notify();
    }

    fn adjust_persistence_interval(&mut self, delta: i32, cx: &mut Context<Self>) {
        let (current_interval, was_enabled) = {
            if let Ok(world) = self.world.lock() {
                let interval = world.config().persistence_interval;
                (interval, interval > 0)
            } else {
                (0, false)
            }
        };

        let cached_interval = self
            .inspector
            .lock()
            .map(|inspector| inspector.persistence_last_enabled)
            .unwrap_or(60);

        let base_interval = if was_enabled {
            current_interval
        } else {
            cached_interval
        };

        let new_interval = ((base_interval as i32) + delta).clamp(1, 10_000) as u32;

        if let Ok(mut inspector) = self.inspector.lock() {
            inspector.persistence_last_enabled = new_interval;
        }

        if was_enabled {
            self.submit_config_update(|config| {
                config.persistence_interval = new_interval;
            });
        }

        cx.notify();
    }
    fn adjust_agent_mutation_rates(
        &mut self,
        agent_id: AgentId,
        delta_primary: f32,
        delta_secondary: f32,
        cx: &mut Context<Self>,
    ) {
        let Some(agent_uid) = self
            .world
            .lock()
            .ok()
            .and_then(|world| world.agent_uid(agent_id))
        else {
            warn!(
                agent = agent_id.raw(),
                "Mutation-rate edit target is no longer live"
            );
            return;
        };
        if self.submit_control_command(ControlCommand::AdjustAgentMutationRates {
            agent_uid,
            delta_primary,
            delta_secondary,
        }) {
            cx.notify();
        }
    }

    fn handle_key_down(&mut self, event: &KeyDownEvent, cx: &mut Context<Self>) {
        if let Some(target) = self.key_capture {
            if event.keystroke.key.eq_ignore_ascii_case("escape") {
                self.key_capture = None;
                cx.notify();
                return;
            }
            self.bindings.assign(target, event.keystroke.clone());
            self.key_capture = None;
            info!(
                "Rebound {} to {}",
                target.label(),
                format_keystroke(&event.keystroke)
            );
            cx.notify();
            return;
        }

        // When settings panel is open, give it exclusive keyboard control
        // Only allow ToggleSettings (comma) to close the panel
        if self.settings_panel.open {
            if let Some(action) = self.bindings.action_for(&event.keystroke)
                && matches!(action, CommandAction::ToggleSettings)
            {
                self.invoke_action(action, cx);
            }
            // Ignore all other key bindings - settings panel handles them
            return;
        }

        if let Some(action) = self.bindings.action_for(&event.keystroke) {
            self.invoke_action(action, cx);
        }
    }

    fn invoke_action(&mut self, action: CommandAction, cx: &mut Context<Self>) {
        match action {
            CommandAction::TogglePlayback => self.playback_toggle(cx),
            CommandAction::GoLive => self.playback_go_live(cx),
            CommandAction::ToggleBrush => self.toggle_brush_state(cx),
            CommandAction::ToggleNarration => self.toggle_narration(cx),
            CommandAction::CyclePalette => self.cycle_palette(cx),
            CommandAction::ToggleSimulationPause => {
                let paused = !self.simulation_drive_snapshot().paused;
                self.set_simulation_paused(paused, cx);
            }
            CommandAction::StepSimulation => self.step_simulation_once(cx),
            CommandAction::ToggleAgentDraw => {
                self.controls.draw_agents = !self.controls.draw_agents;
                cx.notify();
            }
            CommandAction::ToggleFoodOverlay => {
                self.controls.draw_food = !self.controls.draw_food;
                cx.notify();
            }
            CommandAction::ToggleAgentOutline => {
                let enabled = !self.controls.agent_outline;
                self.set_agent_outline(enabled, cx);
            }
            CommandAction::IncreaseSimulationSpeed => self.adjust_simulation_speed(0.25, cx),
            CommandAction::DecreaseSimulationSpeed => self.adjust_simulation_speed(-0.25, cx),
            CommandAction::AddCrossoverAgents => self.spawn_crossover_agent(cx),
            CommandAction::SpawnCarnivore => self.spawn_agent_with_tendency(0.0, cx),
            CommandAction::SpawnHerbivore => self.spawn_agent_with_tendency(1.0, cx),
            CommandAction::ToggleClosedEnvironment => self.toggle_closed_environment(cx),
            CommandAction::FollowSelected => self.toggle_follow_selected(cx),
            CommandAction::FollowOldest => self.toggle_follow_oldest(cx),
            CommandAction::ToggleDebugOverlay => {
                let enabled = !self.debug.enabled;
                self.set_debug_enabled(enabled, cx);
            }
            CommandAction::ClearSelection => self.clear_selection(cx),
            CommandAction::SelectAll => self.select_all_agents(cx),
            CommandAction::FocusFirstSelected => self.focus_first_selected(cx),
            CommandAction::FitWorld => self.fit_world_view(cx),
            CommandAction::ToggleSettings => self.toggle_settings(cx),
            // bd-v9cz: panels are dismissible. These write user INTENT; the resize
            // rule never does, so a narrow window cannot silently undo this choice.
            CommandAction::ToggleStatsPanel => {
                self.hud.stats_open = !self.hud.stats_open;
                cx.notify();
            }
            CommandAction::ToggleHistoryPanel => {
                self.hud.history_open = !self.hud.history_open;
                cx.notify();
            }
            CommandAction::TogglePerfPanel => {
                self.hud.perf_open = !self.hud.perf_open;
                cx.notify();
            }
        }
    }

    fn toggle_settings(&mut self, cx: &mut Context<Self>) {
        self.settings_panel.open = !self.settings_panel.open;

        // Reset scroll position and recalculate content height when opening panel
        if self.settings_panel.open {
            self.settings_panel.scroll_offset = 0.0;
            let total_categories = ConfigCategory::all().len();
            self.settings_panel.content_height = self
                .settings_panel
                .estimate_content_height(total_categories);
            // Note: viewport_height uses conservative default (400px) from state
            // This ensures content is never blocked, at cost of allowing blank space on large windows
        }

        info!(open = self.settings_panel.open, "Settings panel toggled");
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn clear_search(&mut self, cx: &mut Context<Self>) {
        self.settings_panel.search_query.clear();
        info!("Search cleared");
        cx.notify();
    }

    fn update_search(&mut self, query: String, cx: &mut Context<Self>) {
        self.settings_panel.search_query = query;
        info!(query = %self.settings_panel.search_query, "Search query updated");
        cx.notify();
    }

    /// Check if text matches the current search query (case-insensitive substring match)
    fn matches_search(&self, text: &str) -> bool {
        if self.settings_panel.search_query.is_empty() {
            return true; // No search filter - show everything
        }
        // Case-insensitive substring search
        text.to_lowercase()
            .contains(&self.settings_panel.search_query.to_lowercase())
    }

    /// Check if any params in a list match the current search query
    fn has_matching_params(&self, params: &[(&str, String, &str)]) -> bool {
        if self.settings_panel.search_query.is_empty() {
            return true; // No search active - always has "matches"
        }
        params.iter().any(|(label, value, desc)| {
            self.matches_search(label) || self.matches_search(value) || self.matches_search(desc)
        })
    }
    /// Check if a category has any matching parameters (for conditional rendering during search)
    /// Note: This duplicates param construction logic to avoid API changes, prioritizing clarity
    fn category_has_matches(&self, category: ConfigCategory) -> bool {
        if self.settings_panel.search_query.is_empty() {
            return true; // No search - show all categories
        }

        let config = if let Ok(world) = self.world.lock() {
            world.config().clone()
        } else {
            scriptbots_core::ScriptBotsConfig::default()
        };

        // Special handling for Topography (has toggle + readonly params)
        if matches!(category, ConfigCategory::Topography) {
            let toggle_matches = self.matches_search("Enabled")
                || self.matches_search("Enable terrain elevation effects");
            let readonly_params = [
                (
                    "Speed Gain",
                    self.format_float(config.topography_speed_gain, 3),
                    "Downhill boost per unit slope",
                ),
                (
                    "Energy Penalty",
                    self.format_float(config.topography_energy_penalty, 4),
                    "Uphill cost per unit slope",
                ),
            ];
            return toggle_matches || self.has_matching_params(&readonly_params);
        }

        // Check all other categories' params
        match category {
            ConfigCategory::World => {
                let params = [
                    (
                        "World Width",
                        format!("{} units", config.world_width),
                        "Horizontal extent of the simulation world",
                    ),
                    (
                        "World Height",
                        format!("{} units", config.world_height),
                        "Vertical extent of the simulation world",
                    ),
                    (
                        "Food Cell Size",
                        format!("{} units", config.food_cell_size),
                        "Size of each food grid cell",
                    ),
                    (
                        "Initial Food",
                        self.format_float(config.initial_food, 3),
                        "Starting food in each cell",
                    ),
                    (
                        "RNG Seed",
                        config
                            .rng_seed
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "Random".to_string()),
                        "Random number generator seed",
                    ),
                    (
                        "Chart Flush Interval",
                        format!("{} ticks", config.chart_flush_interval),
                        "History chart update frequency",
                    ),
                ];
                self.has_matching_params(&params)
            }
            ConfigCategory::Food => {
                let params = [
                    (
                        "Respawn Interval",
                        format!("{} ticks", config.food_respawn_interval),
                        "Ticks between food respawn events",
                    ),
                    (
                        "Respawn Amount",
                        self.format_float(config.food_respawn_amount, 3),
                        "Food added per respawn",
                    ),
                    (
                        "Maximum Food",
                        self.format_float(config.food_max, 3),
                        "Maximum food per cell",
                    ),
                    (
                        "Growth Rate",
                        self.format_float(config.food_growth_rate, 4),
                        "Logistic regrowth rate",
                    ),
                    (
                        "Decay Rate",
                        self.format_float(config.food_decay_rate, 4),
                        "Proportional decay rate",
                    ),
                    (
                        "Diffusion Rate",
                        self.format_float(config.food_diffusion_rate, 3),
                        "Neighbor exchange rate",
                    ),
                    (
                        "Intake Rate",
                        self.format_float(config.food_intake_rate, 3),
                        "Agent food consumption rate",
                    ),
                    (
                        "Sharing Radius",
                        self.format_float(config.food_sharing_radius, 1),
                        "Friendly neighbor sharing distance",
                    ),
                    (
                        "Sharing Rate",
                        self.format_float(config.food_sharing_rate, 3),
                        "Energy fraction shared per neighbor",
                    ),
                    (
                        "Transfer Rate",
                        self.format_float(config.food_transfer_rate, 4),
                        "Altruistic sharing amount",
                    ),
                    (
                        "Sharing Distance",
                        self.format_float(config.food_sharing_distance, 1),
                        "Altruistic sharing threshold",
                    ),
                ];
                self.has_matching_params(&params)
            }
            ConfigCategory::Agent => {
                let params = [
                    (
                        "Bot Speed",
                        self.format_float(config.bot_speed, 2),
                        "Base wheel speed multiplier",
                    ),
                    (
                        "Bot Radius",
                        self.format_float(config.bot_radius, 1),
                        "Agent radius for collisions",
                    ),
                    (
                        "Boost Multiplier",
                        format!("{}×", self.format_float(config.boost_multiplier, 2)),
                        "Speed boost when activated",
                    ),
                    (
                        "Sense Radius",
                        self.format_float(config.sense_radius, 1),
                        "Perception range",
                    ),
                    (
                        "Carnivore Threshold",
                        self.format_float(config.carnivore_threshold, 2),
                        "Herbivore tendency cutoff for carnivores",
                    ),
                ];
                self.has_matching_params(&params)
            }
            ConfigCategory::Metabolism => {
                let params = [
                    (
                        "Base Drain",
                        self.format_float(config.metabolism_drain, 4),
                        "Baseline energy cost",
                    ),
                    (
                        "Movement Drain",
                        self.format_float(config.movement_drain, 4),
                        "Cost per velocity",
                    ),
                    (
                        "Ramp Floor",
                        self.format_float(config.metabolism_ramp_floor, 2),
                        "Energy level for ramping",
                    ),
                    (
                        "Ramp Rate",
                        self.format_float(config.metabolism_ramp_rate, 4),
                        "Additional drain above floor",
                    ),
                    (
                        "Boost Penalty",
                        self.format_float(config.metabolism_boost_penalty, 4),
                        "Fixed boost cost",
                    ),
                ];
                self.has_matching_params(&params)
            }
            ConfigCategory::Temperature => {
                let params = [
                    (
                        "Discomfort Rate",
                        self.format_float(config.temperature_discomfort_rate, 4),
                        "Health drain multiplier",
                    ),
                    (
                        "Comfort Band",
                        format!("±{}", self.format_float(config.temperature_comfort_band, 3)),
                        "Tolerance threshold",
                    ),
                    (
                        "Gradient Exponent",
                        self.format_float(config.temperature_gradient_exponent, 2),
                        "Pole-to-equator shaping",
                    ),
                    (
                        "Discomfort Exp",
                        self.format_float(config.temperature_discomfort_exponent, 2),
                        "Discomfort scaling power",
                    ),
                ];
                self.has_matching_params(&params)
            }
            ConfigCategory::Reproduction => {
                let params = [
                    (
                        "Energy Threshold",
                        self.format_float(config.reproduction_energy_threshold, 2),
                        "Required energy to reproduce",
                    ),
                    (
                        "Energy Cost",
                        self.format_float(config.reproduction_energy_cost, 2),
                        "Parent's energy deduction",
                    ),
                    (
                        "Cooldown",
                        format!("{} ticks", config.reproduction_cooldown),
                        "Ticks between reproductions",
                    ),
                    (
                        "Herbivore Rate",
                        format!(
                            "{}×",
                            self.format_float(config.reproduction_rate_herbivore, 3)
                        ),
                        "Herbivore multiplier",
                    ),
                    (
                        "Carnivore Rate",
                        format!(
                            "{}×",
                            self.format_float(config.reproduction_rate_carnivore, 3)
                        ),
                        "Carnivore multiplier",
                    ),
                    (
                        "Child Energy",
                        self.format_float(config.reproduction_child_energy, 2),
                        "Starting energy for child",
                    ),
                    (
                        "Spawn Jitter",
                        format!(
                            "±{}",
                            self.format_float(config.reproduction_spawn_jitter, 1)
                        ),
                        "Position randomization",
                    ),
                    (
                        "Spawn Back Distance",
                        self.format_float(config.reproduction_spawn_back_distance, 1),
                        "Child spawn distance behind parent",
                    ),
                    (
                        "Color Jitter",
                        format!(
                            "±{}",
                            self.format_float(config.reproduction_color_jitter, 3)
                        ),
                        "RGB mutation range",
                    ),
                    (
                        "Mutation Scale",
                        self.format_float(config.reproduction_mutation_scale, 4),
                        "Trait mutation magnitude",
                    ),
                    (
                        "Partner Chance",
                        format!(
                            "{}%",
                            self.format_float(config.reproduction_partner_chance * 100.0, 1)
                        ),
                        "Crossover probability",
                    ),
                    (
                        "Gene Log Capacity",
                        format!("{}", config.reproduction_gene_log_capacity),
                        "Max gene history entries",
                    ),
                    (
                        "Meta-Mutation Chance",
                        format!(
                            "{}%",
                            self.format_float(config.reproduction_meta_mutation_chance * 100.0, 1)
                        ),
                        "Mutation rate mutation chance",
                    ),
                    (
                        "Meta-Mutation Scale",
                        self.format_float(config.reproduction_meta_mutation_scale, 4),
                        "Mutation rate change magnitude",
                    ),
                ];
                self.has_matching_params(&params)
            }
            ConfigCategory::Aging => {
                let params = [
                    (
                        "Decay Start Age",
                        format!("{} ticks", config.aging_health_decay_start),
                        "Age when decay begins",
                    ),
                    (
                        "Decay Rate",
                        self.format_float(config.aging_health_decay_rate, 5),
                        "Health loss per tick",
                    ),
                    (
                        "Decay Max",
                        self.format_float(config.aging_health_decay_max, 4),
                        "Maximum decay per tick",
                    ),
                    (
                        "Energy Penalty",
                        format!(
                            "{}×",
                            self.format_float(config.aging_energy_penalty_rate, 3)
                        ),
                        "Health-to-energy conversion",
                    ),
                ];
                self.has_matching_params(&params)
            }
            ConfigCategory::Combat => {
                let params = [
                    (
                        "Spike Radius",
                        self.format_float(config.spike_radius, 1),
                        "Base spike collision radius",
                    ),
                    (
                        "Spike Damage",
                        self.format_float(config.spike_damage, 2),
                        "Damage at full power",
                    ),
                    (
                        "Spike Energy Cost",
                        self.format_float(config.spike_energy_cost, 4),
                        "Energy cost to deploy",
                    ),
                    (
                        "Min Length",
                        self.format_float(config.spike_min_length, 2),
                        "Minimum for damage",
                    ),
                    (
                        "Alignment Cosine",
                        self.format_float(config.spike_alignment_cosine, 2),
                        "Directional threshold",
                    ),
                    (
                        "Speed Bonus",
                        format!("{}×", self.format_float(config.spike_speed_damage_bonus, 3)),
                        "Velocity scaling",
                    ),
                    (
                        "Length Bonus",
                        format!(
                            "{}×",
                            self.format_float(config.spike_length_damage_bonus, 3)
                        ),
                        "Length scaling",
                    ),
                    (
                        "Growth Rate",
                        self.format_float(config.spike_growth_rate, 4),
                        "Spike extension rate",
                    ),
                ];
                self.has_matching_params(&params)
            }
            ConfigCategory::Carcass => {
                let params = [
                    (
                        "Distribution Radius",
                        self.format_float(config.carcass_distribution_radius, 1),
                        "Reward share distance",
                    ),
                    (
                        "Health Reward",
                        self.format_float(config.carcass_health_reward, 2),
                        "Base health given",
                    ),
                    (
                        "Reproduction Reward",
                        self.format_float(config.carcass_reproduction_reward, 1),
                        "Cooldown reduction",
                    ),
                    (
                        "Neighbor Exponent",
                        self.format_float(config.carcass_neighbor_exponent, 2),
                        "Sharing normalization",
                    ),
                    (
                        "Maturity Age",
                        format!("{} ticks", config.carcass_maturity_age),
                        "Full reward age",
                    ),
                    (
                        "Energy Share",
                        format!(
                            "{}%",
                            self.format_float(config.carcass_energy_share_rate * 100.0, 1)
                        ),
                        "Health-to-energy conversion",
                    ),
                    (
                        "Indicator Scale",
                        self.format_float(config.carcass_indicator_scale, 2),
                        "Visual pulse intensity",
                    ),
                ];
                self.has_matching_params(&params)
            }
            ConfigCategory::Topography => unreachable!(), // Handled above
            ConfigCategory::Population => {
                let params = [
                    (
                        "Minimum Population",
                        format!("{}", config.population_minimum),
                        "Auto-seed threshold",
                    ),
                    (
                        "Spawn Interval",
                        format!("{} ticks", config.population_spawn_interval),
                        "Ticks between spawns",
                    ),
                    (
                        "Spawn Count",
                        format!("{}", config.population_spawn_count),
                        "Agents per interval",
                    ),
                    (
                        "Crossover Chance",
                        format!(
                            "{}%",
                            self.format_float(config.population_crossover_chance * 100.0, 1)
                        ),
                        "Breed vs. random spawn",
                    ),
                ];
                self.has_matching_params(&params)
            }
            ConfigCategory::Persistence => {
                let params = [
                    (
                        "Interval",
                        format!("{} ticks", config.persistence_interval),
                        "Database flush frequency",
                    ),
                    (
                        "History Capacity",
                        format!("{}", config.history_capacity),
                        "In-memory tick summaries",
                    ),
                ];
                self.has_matching_params(&params)
            }
        }
    }

    /// Render a list of parameters with search filtering - ONE central filter point for ALL 60+ params!
    fn render_filtered_params(&self, params: Vec<(&str, String, &str)>) -> Div {
        let mut container = div()
            .flex()
            .flex_col()
            .gap_3()
            .px_4()
            .py_4()
            .rounded_lg()
            .bg(rgb(0x0f172a))
            .border_1()
            .border_color(rgb(0x1e293b));

        // ✨ SINGLE CENTRALIZED FILTERING LOOP - this replaces 60+ individual checks!
        for (label, value, desc) in params {
            if self.matches_search(label)
                || self.matches_search(&value)
                || self.matches_search(desc)
            {
                container = container.child(self.render_param_readonly(label, &value, desc));
            }
        }

        container
    }

    fn toggle_category_collapse(&mut self, category: ConfigCategory, cx: &mut Context<Self>) {
        if let Some(pos) = self
            .settings_panel
            .collapsed_categories
            .iter()
            .position(|c| *c == category)
        {
            // Category is collapsed, expand it
            self.settings_panel.collapsed_categories.remove(pos);
        } else {
            // Category is expanded, collapse it
            self.settings_panel.collapsed_categories.push(category);
        }

        // Update content height and clamp scroll
        let total_categories = ConfigCategory::all().len();
        self.settings_panel.content_height = self
            .settings_panel
            .estimate_content_height(total_categories);
        // Clamp scroll with updated content height (viewport_height from state)
        self.settings_panel.clamp_scroll();

        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }
    fn toggle_brush_state(&mut self, cx: &mut Context<Self>) {
        if let Ok(mut inspector) = self.inspector.lock() {
            let new_state = !inspector.brush_enabled;
            inspector.brush_enabled = new_state;
        }
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn toggle_narration(&mut self, cx: &mut Context<Self>) {
        self.accessibility.narration_enabled = !self.accessibility.narration_enabled;
        if self.accessibility.narration_enabled {
            info!("Narration enabled");
        } else {
            info!("Narration disabled");
        }
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn set_simulation_paused(&mut self, paused: bool, cx: &mut Context<Self>) {
        let simulation = self.simulation_drive_snapshot();
        if simulation.paused == paused {
            return;
        }
        if !self.submit_simulation_command(SimulationCommand {
            paused: Some(paused),
            speed_multiplier: Some(simulation.speed_multiplier),
            step_once: false,
        }) {
            return;
        }
        info!(paused, "Simulation pause command enqueued");
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn step_simulation_once(&mut self, cx: &mut Context<Self>) {
        if !self.submit_control_command(ControlCommand::Step) {
            return;
        }
        self.playback.go_live();
        info!("Simulation single-step command enqueued");
        cx.notify();
    }

    fn set_draw_agents(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if self.controls.draw_agents == enabled {
            return;
        }
        self.controls.draw_agents = enabled;
        info!(draw_agents = enabled, "Agent rendering toggled");
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn set_draw_food(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if self.controls.draw_food == enabled {
            return;
        }
        self.controls.draw_food = enabled;
        info!(draw_food = enabled, "Food overlay toggled");
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn set_agent_outline(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if self.controls.agent_outline == enabled {
            return;
        }
        self.controls.agent_outline = enabled;
        info!(agent_outline = enabled, "Agent outline toggled");
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }
    fn adjust_simulation_speed(&mut self, delta: f32, cx: &mut Context<Self>) {
        let simulation = self.simulation_drive_snapshot();
        let mut speed = simulation.speed_multiplier + delta;
        speed = speed.clamp(0.25, 4.0);
        speed = (speed * 100.0).round() / 100.0;
        if (speed - simulation.speed_multiplier).abs() > f32::EPSILON
            && self.submit_simulation_command(SimulationCommand {
                paused: Some(simulation.paused),
                speed_multiplier: Some(speed),
                step_once: false,
            })
        {
            info!(speed, "Simulation speed command enqueued");
            #[cfg(feature = "audio")]
            if let Some(audio) = self.audio.as_mut() {
                audio.play(&audio.toggle_sound);
            }
            cx.notify();
        }
    }

    fn spawn_agent_with_tendency(&mut self, herbivore_bias: f32, cx: &mut Context<Self>) {
        if self.submit_control_command(ControlCommand::SpawnAgent {
            herbivore_tendency: herbivore_bias,
        }) {
            cx.notify();
        }
    }

    fn spawn_crossover_agent(&mut self, cx: &mut Context<Self>) {
        let submission = Arc::clone(&self.selection_submission);
        let Ok(_submission_guard) = submission.lock() else {
            warn!("failed to acquire selection submission lock for crossover command");
            return;
        };
        let (canonical_selection, live_agents, live_uids) = match self.world.lock() {
            Ok(world) => {
                let canonical_selection = world
                    .runtime()
                    .iter()
                    .filter(|(_, runtime)| matches!(runtime.selection, SelectionState::Selected))
                    .map(|(agent_id, _)| agent_id)
                    .collect::<Vec<_>>();
                let live_agents = world.agents().iter_handles().collect::<Vec<_>>();
                let live_uids = live_agents
                    .iter()
                    .filter_map(|agent_id| world.agent_uid(*agent_id).map(|uid| (*agent_id, uid)))
                    .collect::<HashMap<_, _>>();
                (canonical_selection, live_agents, live_uids)
            }
            Err(_) => {
                warn!("failed to acquire world lock for crossover command");
                return;
            }
        };
        let mut selected = self
            .effective_selected_agents(&canonical_selection, &live_agents)
            .into_iter()
            .filter_map(|agent_id| live_uids.get(&agent_id).copied())
            .collect::<Vec<_>>();
        selected.sort_unstable();
        let command = if let [parent_a, parent_b, ..] = selected.as_slice() {
            ControlCommand::SpawnCrossover {
                parent_a: *parent_a,
                parent_b: *parent_b,
            }
        } else {
            ControlCommand::SpawnAgent {
                herbivore_tendency: 0.5,
            }
        };
        if self.submit_control_command(command) {
            cx.notify();
        }
    }

    fn toggle_closed_environment(&mut self, cx: &mut Context<Self>) {
        self.submit_config_update(|config| {
            config.closed = !config.closed;
        });
        cx.notify();
    }

    fn toggle_follow_selected(&mut self, cx: &mut Context<Self>) {
        let next = match self.controls.follow_mode {
            FollowMode::Selected => FollowMode::Off,
            _ => FollowMode::Selected,
        };
        self.controls.follow_mode = next;
        cx.notify();
    }

    fn toggle_follow_oldest(&mut self, cx: &mut Context<Self>) {
        let next = match self.controls.follow_mode {
            FollowMode::Oldest => FollowMode::Off,
            _ => FollowMode::Oldest,
        };
        self.controls.follow_mode = next;
        cx.notify();
    }

    fn cycle_palette(&mut self, cx: &mut Context<Self>) {
        let next = self.accessibility.palette.next();
        self.accessibility.palette = next;
        #[cfg(feature = "audio")]
        if let Some(audio) = self.audio.as_mut() {
            audio.play(&audio.toggle_sound);
        }
        cx.notify();
    }

    fn set_palette(&mut self, palette: ColorPaletteMode, cx: &mut Context<Self>) {
        if self.accessibility.palette != palette {
            self.accessibility.palette = palette;
            cx.notify();
        }
    }

    fn playback_restart(&mut self, cx: &mut Context<Self>) {
        self.playback.restart();
        cx.notify();
    }

    fn playback_step_back(&mut self, cx: &mut Context<Self>) {
        self.playback.step_back();
        cx.notify();
    }

    fn playback_toggle(&mut self, cx: &mut Context<Self>) {
        self.playback.toggle_play();
        cx.notify();
    }

    fn playback_step_forward(&mut self, cx: &mut Context<Self>) {
        self.playback.step_forward();
        cx.notify();
    }

    fn playback_go_live(&mut self, cx: &mut Context<Self>) {
        self.playback.go_live();
        cx.notify();
    }

    #[cfg(feature = "audio")]
    fn update_audio(&mut self, snapshot: &HudSnapshot) {
        let audio = match self.audio.as_mut() {
            Some(audio) => audio,
            None => return,
        };

        if self.playback.mode() != PlaybackMode::Live {
            return;
        }

        if let Some(summary) = snapshot.summary.as_ref() {
            if summary.tick != audio.last_tick {
                if summary.births > audio.last_births {
                    audio.play(&audio.birth_sound);
                }
                if summary.deaths > audio.last_deaths {
                    audio.play(&audio.death_sound);
                }
                audio.last_births = summary.births;
                audio.last_deaths = summary.deaths;
                audio.last_tick = summary.tick;
            }
        }

        if let Some(frame) = snapshot.render_frame.as_ref() {
            let spiked = frame
                .agents
                .iter()
                .filter(|agent| agent.spike_victim)
                .count();
            if spiked > audio.last_spike_count {
                audio.play(&audio.spike_sound);
            }
            audio.last_spike_count = spiked;
        }
    }

    fn render_inspector(&self, snapshot: &HudSnapshot, cx: &mut Context<Self>) -> Div {
        let inspector = &snapshot.inspector;

        let header = div()
            .flex()
            .justify_between()
            .items_center()
            .child(div().text_lg().child("Inspector"))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child(format!("{} agents live", inspector.total_agents)),
            );

        let hovered_block = if let Some(entry) = inspector.hovered.as_ref() {
            div()
                .flex()
                .flex_col()
                .gap_1()
                .rounded_md()
                .bg(rgb(0x111b2b))
                .border_1()
                .border_color(rgb(0x1e3a8a))
                .px_3()
                .py_2()
                .child(div().text_xs().text_color(rgb(0x38bdf8)).child("Hovered"))
                .child(
                    div()
                        .flex()
                        .gap_2()
                        .items_center()
                        .child(color_swatch(entry.color))
                        .child(div().text_sm().child(entry.label.clone())),
                )
                .child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
                    "E {:.2} · H {:.2} · Age {} · Gen {}",
                    entry.energy, entry.health, entry.age, entry.generation.0
                )))
        } else {
            div()
                .rounded_md()
                .bg(rgb(0x111b2b))
                .border_1()
                .border_color(rgb(0x1e293b))
                .px_3()
                .py_2()
                .text_xs()
                .text_color(rgb(0x475569))
                .child("Hover an agent to preview")
        };

        let mut list_children: Vec<Div> = inspector
            .selected
            .iter()
            .map(|entry| self.render_inspector_entry(entry, cx))
            .collect();

        if list_children.is_empty() {
            list_children.push(
                div()
                    .text_xs()
                    .text_color(rgb(0x475569))
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0x1e293b))
                    .bg(rgb(0x0f172a))
                    .px_3()
                    .py_2()
                    .child("No agents selected"),
            );
        }

        let selection_list = div().flex().flex_col().gap_2().children(list_children);

        let brush_tools = self.render_inspector_brush_tools(inspector, cx);
        let debug_tools = self.render_debug_controls(cx);
        let persistence_controls = self.render_persistence_controls(inspector, cx);
        let playback_controls = self.render_inspector_playback_controls(cx);

        let detail = inspector
            .focused
            .as_ref()
            .map(|detail| self.render_inspector_detail(detail, cx))
            .unwrap_or_else(|| {
                div()
                    .text_xs()
                    .text_color(rgb(0x475569))
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0x1e293b))
                    .bg(rgb(0x0f172a))
                    .px_3()
                    .py_3()
                    .child("Select an agent to inspect stats")
            });

        div()
            .flex()
            .flex_col()
            .gap_3()
            .w(px(320.0))
            .flex_none()
            .bg(rgb(0x0b1223))
            .border_1()
            .border_color(rgb(0x1d4ed8))
            .rounded_xl()
            .shadow_lg()
            .p_4()
            .child(header)
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
                        "Focused: {}",
                        inspector
                            .focus_id
                            .map(|id| format!("{id:?}"))
                            .unwrap_or_else(|| "—".to_string())
                    )))
            .child(hovered_block)
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x38bdf8))
                    .child(format!("Selected agents: {}", inspector.selected.len())),
            )
            .child(self.render_selection_controls(inspector, cx))
            .child(self.render_selection_log())
            .child(self.render_simulation_controls(snapshot, cx))
            .child(selection_list)
            .child(brush_tools)
            .child(debug_tools)
            .child(persistence_controls)
            .child(self.render_accessibility_panel(cx))
            .child(playback_controls)
            .child(detail)
    }

    fn render_inspector_entry(&self, entry: &AgentListEntry, cx: &mut Context<Self>) -> Div {
        let agent_id = entry.agent_id;
        let highlight_bg = if entry.is_focused {
            rgb(0x1d4ed8)
        } else {
            rgb(0x111b2b)
        };
        let border_color = if entry.is_focused {
            rgb(0x38bdf8)
        } else {
            rgb(0x1e293b)
        };

        let focus_listener = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
            this.focus_agent(agent_id, cx);
        });

        div()
            .flex()
            .flex_col()
            .gap_1()
            .rounded_md()
            .border_1()
            .border_color(border_color)
            .bg(highlight_bg)
            .px_3()
            .py_2()
            .on_mouse_down(MouseButton::Left, focus_listener)
            .child(
                div()
                    .flex()
                    .justify_between()
                    .items_center()
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .items_center()
                            .child(color_swatch(entry.color))
                            .child(div().text_sm().child(entry.label.clone())),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xf8fafc))
                            .child(format!("E {:.2}", entry.energy)),
                    ),
            )
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
                "H {:.2} · Age {} · Gen {}",
                entry.health, entry.age, entry.generation.0
            )))
            .child({
                let world_text =
                    format!("World ({:.1}, {:.1})", entry.position.x, entry.position.y);
                let line = if let Some((sx, sy)) = self.world_to_screen_coords(entry.position) {
                    format!("{} · Screen ({:.0}, {:.0})", world_text, sx, sy)
                } else {
                    world_text
                };
                div().text_xs().text_color(rgb(0x64748b)).child(line)
            })
    }

    fn render_inspector_brush_tools(
        &self,
        inspector: &InspectorSnapshot,
        cx: &mut Context<Self>,
    ) -> Div {
        let brush_on = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_brush_enabled(true, cx);
        });
        let brush_off = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_brush_enabled(false, cx);
        });
        let radius_inc = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_brush_radius(8.0, cx);
        });
        let radius_dec = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_brush_radius(-8.0, cx);
        });
        let probe_on = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_probe_enabled(true, cx);
        });
        let probe_off = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_probe_enabled(false, cx);
        });

        let brush_on_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("On")
                .on_mouse_down(MouseButton::Left, brush_on);
            if inspector.brush_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let brush_off_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Off")
                .on_mouse_down(MouseButton::Left, brush_off);
            if !inspector.brush_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let brush_minus_button = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .px_2()
            .py_1()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .child("-")
            .on_mouse_down(MouseButton::Left, radius_dec);

        let brush_plus_button = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .px_2()
            .py_1()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .child("+")
            .on_mouse_down(MouseButton::Left, radius_inc);

        let probe_on_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("On")
                .on_mouse_down(MouseButton::Left, probe_on);
            if inspector.probe_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let probe_off_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Off")
                .on_mouse_down(MouseButton::Left, probe_off);
            if !inspector.probe_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Brush Tools"),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .children(vec![brush_on_button, brush_off_button]),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child(format!("Radius {:.0}", inspector.brush_radius)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .children(vec![brush_minus_button, brush_plus_button]),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x94a3b8))
                            .child("Debug probe"),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .children(vec![probe_on_button, probe_off_button]),
                    ),
            )
    }
    fn render_persistence_controls(
        &self,
        inspector: &InspectorSnapshot,
        cx: &mut Context<Self>,
    ) -> Div {
        let enable = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_persistence_enabled(true, cx);
        });
        let disable = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_persistence_enabled(false, cx);
        });
        let inc_small = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_persistence_interval(5, cx);
        });
        let dec_small = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_persistence_interval(-5, cx);
        });
        let inc_large = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_persistence_interval(25, cx);
        });
        let dec_large = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_persistence_interval(-25, cx);
        });

        let display_interval = if inspector.persistence_enabled {
            inspector.persistence_interval.max(1)
        } else {
            inspector.persistence_cached_interval.max(1)
        };

        let on_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("On")
                .on_mouse_down(MouseButton::Left, enable);
            if inspector.persistence_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let off_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Off")
                .on_mouse_down(MouseButton::Left, disable);
            if !inspector.persistence_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        fn build_interval_button<L>(label: &str, listener: L) -> Div
        where
            L: Fn(&MouseDownEvent, &mut Window, &mut App) + 'static,
        {
            div()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x111b2b))
                .px_2()
                .py_1()
                .text_xs()
                .text_color(rgb(0xcbd5f5))
                .child(label.to_string())
                .on_mouse_down(MouseButton::Left, listener)
        }

        let minus_large = build_interval_button("-25", dec_large);
        let minus_small = build_interval_button("-5", dec_small);
        let plus_small = build_interval_button("+5", inc_small);
        let plus_large = build_interval_button("+25", inc_large);

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Storage / Persistence"),
            )
            .child(div().flex().gap_2().children(vec![on_button, off_button]))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0xcbd5f5))
                    .child(format!("Interval {} ticks", display_interval)),
            )
            .child(div().flex().gap_1().children(vec![
                minus_large,
                minus_small,
                plus_small,
                plus_large,
            ]))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x64748b))
                    .child("Disabling stores the last interval for quick re-enable."),
            )
    }
    fn render_selection_controls(
        &self,
        inspector: &InspectorSnapshot,
        cx: &mut Context<Self>,
    ) -> Div {
        let clear = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.clear_selection(cx);
        });
        let select_all = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.select_all_agents(cx);
        });
        let focus_first = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.focus_first_selected(cx);
        });

        let clear_binding = self
            .bindings
            .map
            .get(&CommandAction::ClearSelection)
            .cloned()
            .unwrap_or_default();
        let select_all_binding = self
            .bindings
            .map
            .get(&CommandAction::SelectAll)
            .cloned()
            .unwrap_or_default();
        let focus_binding = self
            .bindings
            .map
            .get(&CommandAction::FocusFirstSelected)
            .cloned()
            .unwrap_or_default();

        let style_action_button = |button: Div| {
            button
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x111b2b))
                .text_color(rgb(0xcbd5f5))
        };

        let clear_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Clear")
                .on_mouse_down(MouseButton::Left, clear);
            style_action_button(base)
        };
        let select_all_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Select all")
                .on_mouse_down(MouseButton::Left, select_all);
            style_action_button(base)
        };
        let focus_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Focus first")
                .on_mouse_down(MouseButton::Left, focus_first);
            style_action_button(base)
        };

        let hover_label = inspector
            .hovered
            .as_ref()
            .map(|entry| entry.label.clone())
            .unwrap_or_else(|| "—".to_string());
        let focus_label = inspector
            .focus_id
            .map(|id| format!("{id:?}"))
            .unwrap_or_else(|| "—".to_string());

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Selection tools"),
            )
            .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                "Selected {} · Hover {} · Focus {}",
                inspector.selected.len(),
                hover_label,
                focus_label
            )))
            .child(div().flex().gap_1().children(vec![
                clear_button,
                select_all_button,
                focus_button,
            ]))
            .child(div().text_xs().text_color(rgb(0x64748b)).child(format!(
                "Shortcuts: Clear {} · Select all {} · Focus {}",
                format_keystroke(&clear_binding),
                format_keystroke(&select_all_binding),
                format_keystroke(&focus_binding)
            )))
    }

    fn render_selection_log(&self) -> Div {
        let mut items: Vec<Div> = Vec::new();
        if self.selection_events.is_empty() {
            items.push(
                div()
                    .text_xs()
                    .text_color(rgb(0x475569))
                    .bg(rgb(0x0f172a))
                    .border_1()
                    .border_color(rgb(0x1e293b))
                    .rounded_md()
                    .px_2()
                    .py_2()
                    .child("No recent selection changes"),
            );
        } else {
            for event in self.selection_events.iter().rev().take(8) {
                let sample = if event.sample_ids.is_empty() {
                    "—".to_string()
                } else {
                    event
                        .sample_ids
                        .iter()
                        .map(|id| format!("{:?}", id))
                        .collect::<Vec<_>>()
                        .join(", ")
                };

                items.push(
                    div()
                        .bg(rgb(0x0f172a))
                        .border_1()
                        .border_color(rgb(0x1e293b))
                        .rounded_md()
                        .px_2()
                        .py_2()
                        .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                            "Tick {} · {} · selected {}",
                            event.tick,
                            event.kind.label(),
                            event.total_selected
                        )))
                        .child(
                            div()
                                .text_xs()
                                .text_color(rgb(0x64748b))
                                .child(format!("Sample [{}]", sample)),
                        ),
                );
            }
        }

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Selection history"),
            )
            .children(items)
    }

    fn render_debug_controls(&self, cx: &mut Context<Self>) -> Div {
        let debug_state = self.debug;
        let overlay_binding = self
            .bindings
            .map
            .get(&CommandAction::ToggleDebugOverlay)
            .cloned()
            .unwrap_or_default();

        let enable_debug = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_debug_enabled(true, cx);
        });
        let disable_debug = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_debug_enabled(false, cx);
        });
        let show_velocity = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_debug_show_velocity(true, cx);
        });
        let hide_velocity = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_debug_show_velocity(false, cx);
        });
        let show_sense = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_debug_show_sense_radius(true, cx);
        });
        let hide_sense = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_debug_show_sense_radius(false, cx);
        });

        let style_toggle = |button: Div, active: bool| {
            if active {
                button
                    .border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                button
                    .border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let overlay_on = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("On")
                .on_mouse_down(MouseButton::Left, enable_debug);
            style_toggle(base, debug_state.enabled)
        };
        let overlay_off = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Off")
                .on_mouse_down(MouseButton::Left, disable_debug);
            style_toggle(base, !debug_state.enabled)
        };
        let velocity_on = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("On")
                .on_mouse_down(MouseButton::Left, show_velocity);
            style_toggle(base, debug_state.show_velocity)
        };
        let velocity_off = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Off")
                .on_mouse_down(MouseButton::Left, hide_velocity);
            style_toggle(base, !debug_state.show_velocity)
        };
        let sense_on = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("On")
                .on_mouse_down(MouseButton::Left, show_sense);
            style_toggle(base, debug_state.show_sense_radius)
        };
        let sense_off = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Off")
                .on_mouse_down(MouseButton::Left, hide_sense);
            style_toggle(base, !debug_state.show_sense_radius)
        };

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Debug overlays"),
            )
            .child(div().flex().gap_2().children(vec![overlay_on, overlay_off]))
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child("Velocity arrows"),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .children(vec![velocity_on, velocity_off]),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child("Sense radius"),
                    )
                    .child(div().flex().gap_2().children(vec![sense_on, sense_off])),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x64748b))
                    .child(format!("Shortcut {}", format_keystroke(&overlay_binding))),
            )
    }
    fn render_simulation_controls(&self, snapshot: &HudSnapshot, cx: &mut Context<Self>) -> Div {
        let controls = snapshot.controls;

        let run_listener = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_simulation_paused(false, cx);
        });
        let pause_listener = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_simulation_paused(true, cx);
        });
        let slower_listener = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_simulation_speed(-0.25, cx);
        });
        let faster_listener = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.adjust_simulation_speed(0.25, cx);
        });
        let agents_on = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_draw_agents(true, cx);
        });
        let agents_off = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_draw_agents(false, cx);
        });
        let food_on = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_draw_food(true, cx);
        });
        let food_off = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_draw_food(false, cx);
        });
        let outline_on = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_agent_outline(true, cx);
        });
        let outline_off = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.set_agent_outline(false, cx);
        });
        let follow_off = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.controls.follow_mode = FollowMode::Off;
            this.fit_world_view(cx);
        });
        let follow_selected = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.controls.follow_mode = FollowMode::Selected;
            cx.notify();
        });
        let follow_oldest = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.controls.follow_mode = FollowMode::Oldest;
            cx.notify();
        });
        let spawn_crossover = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.spawn_crossover_agent(cx);
        });
        let spawn_carnivore = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.spawn_agent_with_tendency(0.0, cx);
        });
        let spawn_herbivore = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.spawn_agent_with_tendency(1.0, cx);
        });
        let open_world = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.submit_config_update(|config| {
                config.closed = false;
            });
            cx.notify();
        });
        let close_world = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.submit_config_update(|config| {
                config.closed = true;
            });
            cx.notify();
        });

        let style_toggle = |button: Div, active: bool| {
            if active {
                button
                    .border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a5f))
                    .text_color(rgb(0xe0f2fe))
            } else {
                button
                    .border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let pause_binding = self
            .bindings
            .map
            .get(&CommandAction::ToggleSimulationPause)
            .cloned()
            .unwrap_or_default();
        let slower_binding = self
            .bindings
            .map
            .get(&CommandAction::DecreaseSimulationSpeed)
            .cloned()
            .unwrap_or_default();
        let faster_binding = self
            .bindings
            .map
            .get(&CommandAction::IncreaseSimulationSpeed)
            .cloned()
            .unwrap_or_default();
        let draw_binding = self
            .bindings
            .map
            .get(&CommandAction::ToggleAgentDraw)
            .cloned()
            .unwrap_or_default();
        let food_binding = self
            .bindings
            .map
            .get(&CommandAction::ToggleFoodOverlay)
            .cloned()
            .unwrap_or_default();
        let outline_binding = self
            .bindings
            .map
            .get(&CommandAction::ToggleAgentOutline)
            .cloned()
            .unwrap_or_default();
        let crossover_binding = self
            .bindings
            .map
            .get(&CommandAction::AddCrossoverAgents)
            .cloned()
            .unwrap_or_default();
        let carnivore_binding = self
            .bindings
            .map
            .get(&CommandAction::SpawnCarnivore)
            .cloned()
            .unwrap_or_default();
        let herbivore_binding = self
            .bindings
            .map
            .get(&CommandAction::SpawnHerbivore)
            .cloned()
            .unwrap_or_default();
        let closed_binding = self
            .bindings
            .map
            .get(&CommandAction::ToggleClosedEnvironment)
            .cloned()
            .unwrap_or_default();

        let run_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Run")
                .on_mouse_down(MouseButton::Left, run_listener),
            !controls.paused,
        );
        let pause_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(format!("Pause ({})", format_keystroke(&pause_binding)))
                .on_mouse_down(MouseButton::Left, pause_listener),
            controls.paused,
        );

        let slower_button = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .text_color(rgb(0xcbd5f5))
            .px_2()
            .py_1()
            .text_xs()
            .child(format!("– ({})", format_keystroke(&slower_binding)))
            .on_mouse_down(MouseButton::Left, slower_listener);
        let faster_button = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .text_color(rgb(0xcbd5f5))
            .px_2()
            .py_1()
            .text_xs()
            .child(format!("+ ({})", format_keystroke(&faster_binding)))
            .on_mouse_down(MouseButton::Left, faster_listener);

        let agents_on_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Agents ON")
                .on_mouse_down(MouseButton::Left, agents_on),
            controls.draw_agents,
        );
        let agents_off_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(format!("Agents OFF ({})", format_keystroke(&draw_binding)))
                .on_mouse_down(MouseButton::Left, agents_off),
            !controls.draw_agents,
        );

        let food_on_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Food ON")
                .on_mouse_down(MouseButton::Left, food_on),
            controls.draw_food,
        );
        let food_off_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(format!("Food OFF ({})", format_keystroke(&food_binding)))
                .on_mouse_down(MouseButton::Left, food_off),
            !controls.draw_food,
        );

        let outline_on_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Outline ON")
                .on_mouse_down(MouseButton::Left, outline_on),
            controls.agent_outline,
        );
        let outline_off_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(format!(
                    "Outline OFF ({})",
                    format_keystroke(&outline_binding)
                ))
                .on_mouse_down(MouseButton::Left, outline_off),
            !controls.agent_outline,
        );

        let follow_off_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Follow OFF")
                .on_mouse_down(MouseButton::Left, follow_off),
            matches!(controls.follow_mode, FollowMode::Off),
        );
        let follow_selected_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Follow selected")
                .on_mouse_down(MouseButton::Left, follow_selected),
            matches!(controls.follow_mode, FollowMode::Selected),
        );
        let follow_oldest_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Follow oldest")
                .on_mouse_down(MouseButton::Left, follow_oldest),
            matches!(controls.follow_mode, FollowMode::Oldest),
        );

        let spawn_row = div().flex().gap_2().children(vec![
            div()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x111b2b))
                .text_color(rgb(0xcbd5f5))
                .px_2()
                .py_1()
                .text_xs()
                .child(format!(
                    "Crossover ({})",
                    format_keystroke(&crossover_binding)
                ))
                .on_mouse_down(MouseButton::Left, spawn_crossover),
            div()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x111b2b))
                .text_color(rgb(0xcbd5f5))
                .px_2()
                .py_1()
                .text_xs()
                .child(format!(
                    "Carnivore ({})",
                    format_keystroke(&carnivore_binding)
                ))
                .on_mouse_down(MouseButton::Left, spawn_carnivore),
            div()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x111b2b))
                .text_color(rgb(0xcbd5f5))
                .px_2()
                .py_1()
                .text_xs()
                .child(format!(
                    "Herbivore ({})",
                    format_keystroke(&herbivore_binding)
                ))
                .on_mouse_down(MouseButton::Left, spawn_herbivore),
        ]);

        let closed_off_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(format!(
                    "Closed OFF ({})",
                    format_keystroke(&closed_binding)
                ))
                .on_mouse_down(MouseButton::Left, open_world),
            !snapshot.is_closed,
        );
        let closed_on_button = style_toggle(
            div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Closed ON")
                .on_mouse_down(MouseButton::Left, close_world),
            snapshot.is_closed,
        );

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Simulation controls"),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .children(vec![run_button, pause_button]),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child(format!("Speed {:.2}×", controls.speed_multiplier)),
                    )
                    .child(slower_button)
                    .child(faster_button),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .children(vec![agents_on_button, agents_off_button]),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .children(vec![food_on_button, food_off_button]),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .children(vec![outline_on_button, outline_off_button]),
            )
            .child(div().flex().gap_2().children(vec![
                follow_off_button,
                follow_selected_button,
                follow_oldest_button,
            ]))
            .child(spawn_row)
            .child(
                div()
                    .flex()
                    .gap_2()
                    .children(vec![closed_off_button, closed_on_button]),
            )
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child("Presets"))
            .child({
                let apply = |label: &'static str, preset: PresetKind| {
                    let listener = cx.listener(move |this, _e: &MouseDownEvent, _, cx| {
                        this.apply_preset(preset, cx);
                    });
                    div()
                        .rounded_md()
                        .border_1()
                        .border_color(rgb(0x1e293b))
                        .bg(rgb(0x111b2b))
                        .text_color(rgb(0xcbd5f5))
                        .px_2()
                        .py_1()
                        .text_xs()
                        .child(label)
                        .on_mouse_down(MouseButton::Left, listener)
                };
                div().flex().gap_2().children(vec![
                    apply("Arctic", PresetKind::Arctic),
                    apply("Boom–Bust", PresetKind::BoomBust),
                    apply("Closed World", PresetKind::ClosedWorld),
                ])
            })
    }

    fn render_inspector_playback_controls(&self, cx: &mut Context<Self>) -> Div {
        let status = self.playback.status();

        let restart = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.playback_restart(cx);
        });
        let prev = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.playback_step_back(cx);
        });
        let toggle = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.playback_toggle(cx);
        });
        let next = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.playback_step_forward(cx);
        });
        let live = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
            this.playback_go_live(cx);
        });

        let play_label = if status.mode == PlaybackMode::Playing {
            "⏸"
        } else {
            "▶"
        };

        let style_button = |button: Div, active: bool| {
            if active {
                button
                    .border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                button
                    .border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let restart_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("⏮")
                .on_mouse_down(MouseButton::Left, restart);
            style_button(base, false)
        };

        let prev_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("⏪")
                .on_mouse_down(MouseButton::Left, prev);
            style_button(base, false)
        };

        let play_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(play_label)
                .on_mouse_down(MouseButton::Left, toggle);
            style_button(base, status.mode == PlaybackMode::Playing)
        };

        let next_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("⏩")
                .on_mouse_down(MouseButton::Left, next);
            style_button(base, false)
        };

        let live_button = {
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child("Live")
                .on_mouse_down(MouseButton::Left, live);
            style_button(base, status.mode == PlaybackMode::Live)
        };

        let frame_summary = if status.total == 0 {
            "No frames captured yet".to_string()
        } else {
            let frame_num = status.index.saturating_add(1);
            let tick = status.current_tick.unwrap_or(0);
            format!("Frame {frame_num}/{} · Tick {tick}", status.total)
        };

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Deterministic playback"),
            )
            .child(div().flex().gap_2().children(vec![
                restart_button,
                prev_button,
                play_button,
                next_button,
                live_button,
            ]))
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0xcbd5f5))
                    .child(frame_summary),
            )
    }
    fn render_accessibility_panel(&self, cx: &mut Context<Self>) -> Div {
        let palette_buttons: Vec<Div> = ColorPaletteMode::ALL
            .iter()
            .map(|mode| {
                let mode = *mode;
                let active = self.accessibility.palette == mode;
                let listener = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
                    this.set_palette(mode, cx);
                });
                let preview_color = apply_palette(rgba_from_hex(0x58ff94, 1.0), mode);
                let preview = div()
                    .w(px(16.0))
                    .h(px(8.0))
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(0x1e293b))
                    .bg(preview_color);
                let base = div()
                    .rounded_md()
                    .border_1()
                    .px_2()
                    .py_1()
                    .text_xs()
                    .flex()
                    .gap_1()
                    .items_center()
                    .child(preview)
                    .child(mode.label())
                    .on_mouse_down(MouseButton::Left, listener);
                if active {
                    base.border_color(rgb(0x38bdf8))
                        .bg(rgb(0x1e3a8a))
                        .text_color(rgb(0xe0f2fe))
                } else {
                    base.border_color(rgb(0x1e293b))
                        .bg(rgb(0x111b2b))
                        .text_color(rgb(0xcbd5f5))
                }
            })
            .collect();

        let narration_button = {
            let listener = cx.listener(|this, _event: &MouseDownEvent, _, cx| {
                this.toggle_narration(cx);
            });
            let base = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(if self.accessibility.narration_enabled {
                    "Narration: On"
                } else {
                    "Narration: Off"
                })
                .on_mouse_down(MouseButton::Left, listener);
            if self.accessibility.narration_enabled {
                base.border_color(rgb(0x38bdf8))
                    .bg(rgb(0x1e3a8a))
                    .text_color(rgb(0xe0f2fe))
            } else {
                base.border_color(rgb(0x1e293b))
                    .bg(rgb(0x111b2b))
                    .text_color(rgb(0xcbd5f5))
            }
        };

        let mut bindings_rows: Vec<Div> = Vec::new();
        for (action, stroke) in self.bindings.iter() {
            let capturing = self.key_capture == Some(action);
            let label = action.label();
            let binding_text = if capturing {
                "Press new key...".to_string()
            } else {
                format_keystroke(&stroke)
            };
            let listener = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
                if this.key_capture == Some(action) {
                    this.key_capture = None;
                } else {
                    this.key_capture = Some(action);
                }
                cx.notify();
            });
            let button = div()
                .rounded_md()
                .border_1()
                .px_2()
                .py_1()
                .text_xs()
                .child(if capturing { "Cancel" } else { "Rebind" })
                .on_mouse_down(MouseButton::Left, listener);
            bindings_rows.push(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(label))
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x94a3b8))
                            .child(binding_text),
                    )
                    .child(button),
            );
        }

        if self.key_capture.is_some() {
            bindings_rows.push(
                div()
                    .text_xs()
                    .text_color(rgb(0xf97316))
                    .child("Press a key to assign, or Esc to cancel."),
            );
        }

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Accessibility"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0xcbd5f5))
                    .child("Color palette"),
            )
            .child(div().flex().gap_2().children(palette_buttons))
            .child(div().text_xs().text_color(rgb(0xcbd5f5)).child("Narration"))
            .child(narration_button)
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0xcbd5f5))
                    .child("Key bindings"),
            )
            .children(bindings_rows)
    }
    fn render_inspector_detail(
        &self,
        detail: &AgentInspectorDetails,
        cx: &mut Context<Self>,
    ) -> Div {
        // Brain bars (sensors/outputs)
        let sensor_bars = render_brain_bars(&detail.sensors, true);
        let output_bars = render_brain_bars(&detail.outputs, false);
        let _sensors_preview: Vec<String> = detail
            .sensors
            .iter()
            .take(6)
            .enumerate()
            .map(|(idx, value)| format!("s{idx}:{value:.2}"))
            .collect();
        let _outputs_preview: Vec<String> = detail
            .outputs
            .iter()
            .take(4)
            .enumerate()
            .map(|(idx, value)| format!("o{idx}:{value:.2}"))
            .collect();

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e3a8a))
            .bg(rgb(0x111b2b))
            .px_3()
            .py_3()
            .child(
                div()
                    .flex()
                    .justify_between()
                    .items_center()
                    .child(div().text_sm().child(detail.label.clone()))
                    .child(color_swatch(detail.color)),
            )
            .child(div().text_xs().text_color(rgb(0x64748b)).child(format!(
                "Agent {:?} · Gen {} · Age {}",
                detail.agent_id, detail.generation.0, detail.age
            )))
            .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                "Energy {:.2} · Health {:.2} · Spike {:.1}",
                detail.energy, detail.health, detail.spike_length
            )))
            .child({
                let mut text = format!(
                    "Pos ({:.1}, {:.1}) · Brain {}",
                    detail.position.x, detail.position.y, detail.brain_descriptor
                );
                if let Some((sx, sy)) = self.world_to_screen_coords(detail.position) {
                    text.push_str(&format!(" · Screen ({:.0}, {:.0})", sx, sy));
                }
                div().text_xs().text_color(rgb(0x94a3b8)).child(text)
            })
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
                "Mutation rates p{:.3} s{:.3}",
                detail.mutation_rates.primary, detail.mutation_rates.secondary
            )))
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
                "Traits smell {:.2} · sound {:.2} · hearing {:.2} · eye {:.2} · blood {:.2}",
                detail.trait_modifiers.smell,
                detail.trait_modifiers.sound,
                detail.trait_modifiers.hearing,
                detail.trait_modifiers.eye,
                detail.trait_modifiers.blood
            )))
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child("Sensors"))
            .child(sensor_bars)
            .child(render_sense_attribution(detail))
            .child(div().text_xs().text_color(rgb(0x94a3b8)).child("Outputs"))
            .child(output_bars)
            .child({
                let provenance = match (
                    detail.brain_source_tick,
                    detail.brain_request_revision,
                    detail.brain_payload_bytes,
                    detail.brain_inspection_status.as_deref(),
                ) {
                    (
                        Some(source_tick),
                        Some(request_revision),
                        Some(payload_bytes),
                        Some("ready"),
                    ) => {
                        let payload_status = if detail
                            .brain_activations
                            .as_ref()
                            .is_some_and(|activations| activations.truncated)
                        {
                            "payload CLIPPED"
                        } else {
                            "payload complete"
                        };
                        format!(
                            "Brain detail · tick {source_tick} · request {request_revision} · {payload_bytes} B · {payload_status}"
                        )
                    }
                    (
                        Some(source_tick),
                        Some(request_revision),
                        Some(payload_bytes),
                        Some(status),
                    ) => format!(
                        "Brain detail · tick {source_tick} · request {request_revision} · {payload_bytes} B · {status}"
                    ),
                    _ => "Brain detail · not requested".to_owned(),
                };
                div()
                    .text_xs()
                    .text_color(rgb(0x64748b))
                    .child(provenance)
            })
            .child(render_activation_heatmaps(&detail.brain_activations))
            .child(render_output_attributions(detail))
            .child(self.render_output_sparklines_for(detail.agent_id))
            .child(self.render_diet_gauges())
            .child(render_brain_card(detail))
            .child(self.render_mutation_controls(detail, cx))
    }

    fn render_diet_gauges(&self) -> Div {
        // Use analytics snapshot if available to show diet average energies
        let mut rows: Vec<Div> = Vec::new();
        if let Some(analytics) = self.analytics_cache.as_ref() {
            let h = analytics.herbivore_avg_energy as f32;
            let o = analytics.hybrid_avg_energy as f32;
            let c = analytics.carnivore_avg_energy as f32;
            let maxv = h.max(o).max(c).max(1e-3);
            let mk = |label: &str, v: f32, color: Rgba| -> Div {
                let w = (v / maxv).clamp(0.0, 1.0);
                let track = div().w(px(120.0)).h(px(6.0)).bg(rgb(0x1e293b)).rounded_sm();
                let fillw = 120.0 * w;
                let fill = div().w(px(fillw)).h(px(6.0)).bg(color).rounded_sm();
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x94a3b8))
                            .child(label.to_string()),
                    )
                    .child(div().relative().child(track).child(fill))
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child(format!("{:.2}", v)),
                    )
            };
            rows.push(mk("H", h, rgba_from_hex(0x22c55e, 0.95)));
            rows.push(mk("O", o, rgba_from_hex(0xf59e0b, 0.95)));
            rows.push(mk("C", c, rgba_from_hex(0xef4444, 0.95)));
        } else {
            rows.push(
                div()
                    .text_xs()
                    .text_color(rgb(0x475569))
                    .child("Diet gauges: analytics pending"),
            );
        }
        div()
            .flex()
            .flex_col()
            .gap_1()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Diet avg energy"),
            )
            .children(rows)
    }
    fn render_mutation_controls(
        &self,
        detail: &AgentInspectorDetails,
        cx: &mut Context<Self>,
    ) -> Div {
        let agent_id = detail.agent_id;
        let primary_step = 0.0005_f32;
        let secondary_step = 0.01_f32;

        let inc_primary = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
            this.adjust_agent_mutation_rates(agent_id, primary_step, 0.0, cx);
        });
        let dec_primary = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
            this.adjust_agent_mutation_rates(agent_id, -primary_step, 0.0, cx);
        });

        let agent_id_secondary = detail.agent_id;
        let inc_secondary = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
            this.adjust_agent_mutation_rates(agent_id_secondary, 0.0, secondary_step, cx);
        });
        let dec_secondary = cx.listener(move |this, _event: &MouseDownEvent, _, cx| {
            this.adjust_agent_mutation_rates(agent_id_secondary, 0.0, -secondary_step, cx);
        });

        let primary_minus = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .px_2()
            .py_1()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .child("-")
            .on_mouse_down(MouseButton::Left, dec_primary);

        let primary_plus = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .px_2()
            .py_1()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .child("+")
            .on_mouse_down(MouseButton::Left, inc_primary);

        let secondary_minus = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .px_2()
            .py_1()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .child("-")
            .on_mouse_down(MouseButton::Left, dec_secondary);

        let secondary_plus = div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x111b2b))
            .px_2()
            .py_1()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .child("+")
            .on_mouse_down(MouseButton::Left, inc_secondary);

        div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x94a3b8))
                    .child("Mutation controls"),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child(format!("Primary {:.4}", detail.mutation_rates.primary)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .children(vec![primary_minus, primary_plus]),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xcbd5f5))
                            .child(format!("Secondary {:.3}", detail.mutation_rates.secondary)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .children(vec![secondary_minus, secondary_plus]),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x64748b))
                    .child("Adjusts focused agent mutation rates in ± steps."),
            )
    }

    fn render_canvas_placeholder(&self, snapshot: &HudSnapshot) -> Div {
        div()
            .flex()
            .flex_col()
            .flex_1()
            .rounded_xl()
            .border_1()
            .border_color(rgb(0x0ea5e9))
            .bg(rgb(0x0b1120))
            .shadow_lg()
            .p_4()
            .justify_center()
            .items_center()
            .gap_2()
            .child(
                div()
                    .text_lg()
                    .text_color(rgb(0x38bdf8))
                    .child("Canvas viewport"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x64748b))
                    .child("Rendering pipeline will paint agents and terrain here."),
            )
            .child(div().text_xs().text_color(rgb(0x38bdf8)).child(format!(
                "Latest tick #{}, {} agents",
                snapshot.tick, snapshot.agent_count
            )))
    }
    /// The docked HUD rail (bd-v9cz). Returns `None` when nothing is open or the
    /// window is too narrow, in which case the world gets the full width.
    ///
    /// Panels are children of this rail, so they occupy reserved space and cannot
    /// overlap the world. Colours are left exactly as they were: bd-f4x0 owns
    /// typography and palette, and bd-9pqz defines the ramp once for everyone.
    fn render_hud_rail(&self, snapshot: &HudSnapshot, resolved: ResolvedHudLayout) -> Option<Div> {
        if !resolved.show_rail {
            return None;
        }

        let mut rail = div()
            .flex()
            .flex_col()
            .flex_none()
            .w(px(HUD_RAIL_WIDTH))
            .h_full()
            .gap_3()
            .overflow_hidden();

        // Each panel may SHRINK so they share the rail's height. Without this the first
        // panel takes everything it wants and `overflow_hidden` silently swallows the
        // rest: the bd-abu3 capture proved the history chart was being clipped out of
        // existence while the policy reported it open. A panel the layout says is
        // visible must be visible, or the toggle state is a lie.
        if resolved.stats_open {
            rail = rail.child(self.render_overlay(snapshot).flex_shrink(1.0));
        }
        if resolved.history_open {
            rail = rail.child(self.render_history_chart(snapshot).flex_shrink(1.0));
        }
        if resolved.perf_open {
            rail = rail.child(self.render_perf_overlay(self.last_perf).flex_shrink(1.0));
        }
        Some(rail)
    }

    fn render_overlay(&self, snapshot: &HudSnapshot) -> Div {
        let mut lines: Vec<String> = if let Some(summary) = snapshot.summary.as_ref() {
            vec![
                format!("Tick {} (epoch {})", summary.tick, snapshot.epoch),
                format!(
                    "Agents {} • Births {} • Deaths {}",
                    summary.agent_count, summary.births, summary.deaths
                ),
                format!(
                    "Avg energy {:.2} • Avg health {:.2}",
                    summary.average_energy, summary.average_health
                ),
            ]
        } else {
            vec![format!("Tick {} • epoch {}", snapshot.tick, snapshot.epoch)]
        };
        let camera = self.camera_snapshot();
        lines.push(format!(
            "Zoom {:.2}× • Pan ({:.0}, {:.0})",
            camera.zoom, camera.offset_px.0, camera.offset_px.1
        ));
        lines.push(format!(
            "Simulation {} · speed {:.2}×",
            if snapshot.controls.paused {
                "Paused"
            } else {
                "Running"
            },
            snapshot.controls.speed_multiplier
        ));
        lines.push(format!(
            "Draw agents {} · food {}",
            if snapshot.controls.draw_agents {
                "ON"
            } else {
                "OFF"
            },
            if snapshot.controls.draw_food {
                "ON"
            } else {
                "OFF"
            }
        ));
        lines.push(format!(
            "Outline {}",
            if snapshot.controls.agent_outline {
                "ON"
            } else {
                "OFF"
            }
        ));
        lines.push(snapshot.controls.follow_mode.label().to_string());

        let inspector = &snapshot.inspector;
        if let Some(detail) = inspector.focused.as_ref() {
            if let Some((sx, sy)) = self.world_to_screen_coords(detail.position) {
                lines.push(format!(
                    "Focus {:?} · world ({:.1}, {:.1}) · screen ({:.0}, {:.0})",
                    detail.agent_id, detail.position.x, detail.position.y, sx, sy
                ));
            } else {
                lines.push(format!(
                    "Focus {:?} · world ({:.1}, {:.1})",
                    detail.agent_id, detail.position.x, detail.position.y
                ));
            }
        } else if let Some(focus_id) = inspector.focus_id {
            lines.push(format!("Focus {:?}", focus_id));
        }
        if let Some(hover) = inspector.hovered.as_ref() {
            if let Some((sx, sy)) = self.world_to_screen_coords(hover.position) {
                lines.push(format!(
                    "Hover {} · world ({:.1}, {:.1}) · screen ({:.0}, {:.0})",
                    hover.label, hover.position.x, hover.position.y, sx, sy
                ));
            } else {
                lines.push(format!(
                    "Hover {} · world ({:.1}, {:.1})",
                    hover.label, hover.position.x, hover.position.y
                ));
            }
        }
        lines.push(format!(
            "Brush {} · radius {:.0} · Probe {}",
            if inspector.brush_enabled { "ON" } else { "OFF" },
            inspector.brush_radius,
            if inspector.probe_enabled { "ON" } else { "OFF" }
        ));
        lines.push(format!(
            "Palette {} · Narration {}",
            self.accessibility.palette.label(),
            if self.accessibility.narration_enabled {
                "ON"
            } else {
                "OFF"
            }
        ));

        if self.debug.enabled {
            lines.push(format!(
                "Debug overlay ON · velocity {} · sense {}",
                if self.debug.show_velocity {
                    "ON"
                } else {
                    "OFF"
                },
                if self.debug.show_sense_radius {
                    "ON"
                } else {
                    "OFF"
                }
            ));
        } else {
            lines.push("Debug overlay OFF".to_string());
        }
        lines.push(format!(
            "Persistence {} · interval {}",
            if inspector.persistence_enabled {
                "ON"
            } else {
                "OFF"
            },
            if inspector.persistence_enabled {
                inspector.persistence_interval.max(1)
            } else {
                inspector.persistence_cached_interval.max(1)
            }
        ));
        if let Some(action) = self.key_capture {
            lines.push(format!("Rebinding {}...", action.label()));
        }

        if self.shift_inspect {
            if let Some(hover) = inspector.hovered.as_ref() {
                let mut entry = format!(
                    "Inspect {} · E {:.2} · H {:.2} · Age {} · Gen {}",
                    hover.label, hover.energy, hover.health, hover.age, hover.generation.0
                );
                if let Some((sx, sy)) = self.world_to_screen_coords(hover.position) {
                    entry.push_str(&format!(" · screen ({:.0}, {:.0})", sx, sy));
                }
                lines.push(entry);
            } else if let Some(detail) = inspector.focused.as_ref() {
                let mut entry = format!(
                    "Inspect {:?} · E {:.2} · H {:.2} · Age {} · Gen {}",
                    detail.agent_id, detail.energy, detail.health, detail.age, detail.generation.0
                );
                if let Some((sx, sy)) = self.world_to_screen_coords(detail.position) {
                    entry.push_str(&format!(" · screen ({:.0}, {:.0})", sx, sy));
                }
                lines.push(entry);
                if let Some((best_idx, best_value)) = detail
                    .outputs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                {
                    lines.push(format!(
                        "Outputs max [{}]={:.2} · Mutation {:.3}/{:.3}",
                        best_idx,
                        best_value,
                        detail.mutation_rates.primary,
                        detail.mutation_rates.secondary
                    ));
                }
            } else {
                lines.push("Inspect overlay active (no agent)".to_string());
            }
        }

        let playback = self.playback.status();
        if playback.total > 0 {
            let mode_label = match playback.mode {
                PlaybackMode::Live => "LIVE",
                PlaybackMode::Paused => "PAUSED",
                PlaybackMode::Playing => "PLAY",
            };
            let current_tick = playback.current_tick.unwrap_or(snapshot.tick);
            let total_frames = playback.total;
            lines.push(format!(
                "Playback {mode_label} · frame {}/{} · tick {}",
                playback.index.saturating_add(1),
                total_frames,
                current_tick
            ));
        } else {
            lines.push("Playback LIVE · no frames".to_string());
        }

        let mut container = div().flex().gap_3().items_start();

        if let Some(vector_state) = VectorHudState::from_snapshot(snapshot) {
            let canvas_state = vector_state.clone();
            let heading_deg = vector_state.heading_rad.to_degrees();
            let cohesion = if vector_state.max_speed > f32::EPSILON {
                (vector_state.vector_magnitude / vector_state.max_speed * 100.0).clamp(0.0, 100.0)
            } else {
                0.0
            };
            let vector_canvas = canvas(
                move |_, _, _| canvas_state.clone(),
                move |bounds, state, window, _| paint_vector_hud(bounds, &state, window),
            )
            .w(px(148.0))
            .h(px(108.0));

            let vector_card = div()
                .flex()
                .flex_col()
                .gap_2()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x07111f))
                .px_3()
                .py_3()
                .child(vector_canvas)
                .child(
                    div()
                        .text_xs()
                        .text_color(rgb(0x64748b))
                        .child("Vector HUD gauges"),
                )
                .child(div().text_xs().text_color(rgb(0xa5b4fc)).child(format!(
                    "Avg speed {:.2} · heading {:+.0}° · cohesion {:>3.0}%",
                    vector_state.mean_speed, heading_deg, cohesion
                )));

            container = container.child(vector_card);
        }

        let text_column = div().flex().flex_col().gap_1().children(
            lines
                .into_iter()
                .map(|line| div().text_sm().text_color(rgb(0xe2e8f0)).child(line)),
        );

        container = container.child(text_column);

        // Docked into the HUD rail (bd-v9cz): no .absolute(), no top/left. It is a
        // flex sibling of the world, so it occupies reserved space instead of
        // covering the simulation. Colours unchanged — bd-f4x0/bd-9pqz own those.
        // bd-f4x0: one shared chrome surface, hairline border, no drop shadow. The
        // shadow implied the panel floated ABOVE the world, which is exactly the
        // reading bd-v9cz removed structurally.
        div()
            .flex_none()
            .bg(chrome::surface())
            .rounded_md()
            .border_1()
            .border_color(chrome::border())
            .px_3()
            .py_3()
            .child(container)
    }

    fn render_perf_overlay(&self, stats: PerfSnapshot) -> Div {
        let mut lines = Vec::new();

        if stats.sample_count == 0 {
            lines.push("Performance stats: collecting...".to_string());
        } else {
            lines.push(format!(
                "Frame {:.2} ms ({:.1} fps)",
                stats.latest_ms, stats.fps
            ));
            lines.push(format!(
                "Avg {:.2} ms · Min {:.2} · Max {:.2}",
                stats.average_ms, stats.min_ms, stats.max_ms
            ));
            lines.push(format!("Samples {}", stats.sample_count));
        }

        // Docked into the HUD rail (bd-v9cz) — see render_overlay.
        div()
            .flex_none()
            .bg(chrome::surface())
            .border_1()
            .border_color(rgb(0x1e293b))
            .rounded_md()
            .shadow_md()
            .px_3()
            .py_2()
            .text_xs()
            .text_color(rgb(0xcbd5f5))
            .flex()
            .flex_col()
            .gap_1()
            .children(lines.into_iter().map(|line| div().child(line)))
    }

    fn render_history_chart(&self, snapshot: &HudSnapshot) -> Div {
        const WIDTH: f32 = 220.0;
        const HEIGHT: f32 = 120.0;

        match HistoryChartData::from_entries(&snapshot.recent_history, WIDTH, HEIGHT) {
            Some(data) => {
                let chart_canvas = canvas(
                    move |_, _, _| data.clone(),
                    move |bounds, data, window, _| paint_history_chart(bounds, &data, window),
                )
                .w(px(WIDTH))
                .h(px(HEIGHT - 28.0))
                .flex_none();

                let legend = div()
                    .flex()
                    .gap_2()
                    .text_xs()
                    .text_color(rgb(0xcbd5f5))
                    .child(legend_item(chrome::series_population().into(), "Agents"))
                    .child(legend_item(chrome::series_births().into(), "Births"))
                    .child(legend_item(chrome::series_deaths().into(), "Deaths"));

                // Docked into the HUD rail (bd-v9cz) — see render_overlay.
                div()
                    .flex_none()
                    .w(px(WIDTH))
                    .bg(chrome::surface())
                    .border_1()
                    .border_color(chrome::border())
                    .rounded_md()
                    .shadow_md()
                    .px_3()
                    .py_2()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(chart_canvas)
                    .child(legend)
            }
            None => div()
                .flex_none()
                .bg(chrome::surface())
                .border_1()
                .border_color(chrome::border())
                .rounded_md()
                .shadow_md()
                .px_3()
                .py_2()
                .text_xs()
                .text_color(rgb(0x94a3b8))
                .child("History chart pending data"),
        }
    }

    fn render_footer(&self, snapshot: &HudSnapshot) -> Div {
        div()
            .flex()
            .justify_between()
            .items_center()
            .text_xs()
            .text_color(rgb(0x475569))
            .child(format!(
                "World {}×{} · History capacity {}",
                snapshot.world_size.0, snapshot.world_size.1, snapshot.history_capacity
            ))
            .child(format!(
                "Showing {} recent ticks",
                snapshot.recent_history.len()
            ))
    }

    fn header_chip(
        &self,
        theme: HudTheme,
        icon: &str,
        label: impl Into<String>,
        bg_hex: u32,
    ) -> Div {
        let label = label.into();
        div()
            .flex()
            .items_center()
            .gap_2()
            .px_3()
            .py_1()
            .rounded_full()
            .bg(rgb(bg_hex))
            .text_sm()
            .text_color(rgb(theme.chip_text))
            .child(div().text_sm().child(SharedString::from(icon.to_string())))
            .child(div().text_sm().child(label))
    }

    fn metric_card(
        &self,
        theme: &HudTheme,
        label: &str,
        value: String,
        accent_hex: u32,
        detail: Option<String>,
        sparkline: Option<SparklineSeries>,
    ) -> Div {
        let accent = rgb(accent_hex);
        let accent_rgba = rgba_from_hex(accent_hex, 1.0);
        let badge_state = MetricBadgeState {
            accent: accent_rgba,
        };
        let badge = canvas(
            move |_, _, _| badge_state.clone(),
            move |bounds, state: MetricBadgeState, window, _| {
                paint_metric_badge(bounds, state, window);
            },
        )
        .w(px(28.0))
        .h(px(28.0));

        let mut card = div()
            .flex()
            .flex_col()
            .gap_2()
            .rounded_lg()
            .border_1()
            .border_color(accent)
            .bg(rgb(theme.card_bg))
            .shadow_md()
            .p_4()
            .child(
                div().flex().justify_between().items_center().child(
                    div().flex().items_center().gap_2().child(badge).child(
                        div()
                            .text_xs()
                            .text_color(accent)
                            .child(label.to_uppercase()),
                    ),
                ),
            )
            .child(
                div()
                    .text_3xl()
                    .text_color(rgb(theme.text_primary))
                    .child(value),
            );

        if let Some(detail_text) = detail {
            card = card.child(
                div()
                    .text_sm()
                    .text_color(rgb(theme.text_subtle))
                    .child(detail_text),
            );
        }

        if let Some(series) = sparkline {
            let spark_state = SparklineState {
                values: series.normalized.clone(),
                accent: accent_rgba,
                trend: series.trend,
            };
            let spark_canvas = canvas(
                move |_, _, _| spark_state.clone(),
                move |bounds, state: SparklineState, window, _| {
                    paint_sparkline(bounds, state, window);
                },
            )
            .h(px(28.0))
            .w_full();

            card = card.child(
                div()
                    .mt(px(6.0))
                    .rounded_md()
                    .bg(rgb(theme.spark_bg))
                    .border_1()
                    .border_color(rgb(theme.spark_border))
                    .px_3()
                    .py_2()
                    .child(spark_canvas),
            );
        }

        card
    }
    fn render_settings_panel(&self, cx: &mut Context<Self>) -> Div {
        // Modern, world-class settings panel with beautiful design
        let backdrop = div()
            .absolute()
            .inset_0()
            .bg(rgb(0x020617))
            .opacity(0.5)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    // Click backdrop to close panel (standard modal UX)
                    this.toggle_settings(cx);
                }),
            );

        let panel = div()
            .absolute()
            .top(px(0.0))
            .left(px(0.0))
            .bottom(px(0.0))
            .w(px(540.0))
            .bg(rgb(0x0f172a))
            .border_r_1()
            .border_color(rgb(0x334155))
            .shadow_xl()
            .flex()
            .flex_col()
            .overflow_hidden()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|_, _, _, cx| {
                    // Prevent clicks on panel from propagating to backdrop
                    cx.stop_propagation();
                }),
            )
            .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                // Handle keyboard input for search when settings panel is open
                let key = &event.keystroke.key;

                if event.keystroke.key == "backspace" {
                    // Remove last character from search
                    let mut query = this.settings_panel.search_query.clone();
                    query.pop();
                    this.update_search(query, cx);
                } else if event.keystroke.key == "escape" {
                    // Clear search or close panel
                    if !this.settings_panel.search_query.is_empty() {
                        this.clear_search(cx);
                    } else {
                        this.toggle_settings(cx);
                    }
                } else if key.len() == 1
                    && key
                        .chars()
                        .all(|c| c.is_alphanumeric() || c.is_whitespace() || "._-±×%".contains(c))
                {
                    // Add alphanumeric characters, spaces, punctuation, and special chars (±, ×, %) to search
                    let mut query = this.settings_panel.search_query.clone();
                    query.push_str(key);
                    this.update_search(query, cx);
                }
            }));

        let header = div()
            .flex()
            .items_center()
            .justify_between()
            .px_6()
            .py_4()
            .border_b_1()
            .border_color(rgb(0x334155))
            .bg(rgb(0x1e293b))
            .child(
                div().flex().items_center().gap_3().child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .text_xl()
                                .text_color(rgb(0xf1f5f9))
                                .child("Configuration"),
                        )
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap_2()
                                .child(
                                    div()
                                        .text_sm()
                                        .text_color(rgb(0x94a3b8))
                                        .child("Simulation parameters & settings"),
                                )
                                .child(
                                    div()
                                        .px_2()
                                        .py_1()
                                        .rounded_md()
                                        .bg(rgb(0x334155))
                                        .text_xs()
                                        .text_color(rgb(0x94a3b8))
                                        .child("Press , to toggle"),
                                ),
                        ),
                ),
            )
            .child(
                div()
                    .px_4()
                    .py_2()
                    .rounded_lg()
                    .bg(rgb(0x475569))
                    .text_base()
                    .text_color(rgb(0xf1f5f9))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(0x64748b)))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _, _, cx| {
                            this.toggle_settings(cx);
                        }),
                    )
                    .child("✕ Close"),
            );

        // REAL functional search bar - displays current search query and allows filtering
        let search_query = self.settings_panel.search_query.clone();
        let has_search = !search_query.is_empty();

        let search_bar = div()
            .px_6()
            .py_4()
            .border_b_1()
            .border_color(rgb(0x334155))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .px_4()
                    .py_3()
                    .rounded_lg()
                    .bg(rgb(0x1e293b))
                    .border_1()
                    .border_color(if has_search {
                        rgb(0x60a5fa)
                    } else {
                        rgb(0x475569)
                    })
                    .child(
                        div()
                            .flex_1()
                            .text_sm()
                            .text_color(if has_search {
                                rgb(0xf1f5f9)
                            } else {
                                rgb(0x94a3b8)
                            })
                            .child(if has_search {
                                search_query.clone()
                            } else {
                                "Search parameters...".to_string()
                            }),
                    )
                    .when(has_search, |container| {
                        container.child(
                            div()
                                .px_2()
                                .py_1()
                                .rounded_md()
                                .bg(rgb(0x334155))
                                .text_xs()
                                .text_color(rgb(0x94a3b8))
                                .cursor_pointer()
                                .hover(|s: StyleRefinement| {
                                    s.bg(rgb(0x475569)).text_color(rgb(0xf1f5f9))
                                })
                                .on_mouse_down(
                                    MouseButton::Left,
                                    cx.listener(|this, _, _, cx| {
                                        this.clear_search(cx);
                                    }),
                                )
                                .child("✕ Clear"),
                        )
                    }),
            );

        // Scrollable container for categories with mouse wheel handling
        // Use cached dimensions (updated when panel opens or categories collapse/expand)
        let content_height = self.settings_panel.content_height;
        let viewport_height = self.settings_panel.viewport_height;
        let scroll_offset = self.settings_panel.scroll_offset;

        // Calculate scroll bounds (must match clamp_scroll logic for consistency)
        let max_scroll = (content_height - viewport_height).max(0.0);
        let has_scrollable_content = max_scroll > 1.0;

        let categories_content = div()
            .flex_1()
            .overflow_hidden()
            .relative()
            .px_6()
            .py_4()
            .on_scroll_wheel(cx.listener(move |this, event: &ScrollWheelEvent, _, cx| {
                // Handle scroll wheel to update offset
                let scroll_delta = match event.delta {
                    ScrollDelta::Pixels(delta) => f32::from(delta.y),
                    ScrollDelta::Lines(lines) => lines.y * 20.0, // ~20px per line
                };

                // Update scroll offset with proper bounds
                // Positive delta = scroll down = increase offset to show lower content
                this.settings_panel.scroll_offset += scroll_delta;
                this.settings_panel.clamp_scroll();
                cx.notify();
            }))
            .child(
                div()
                    .absolute()
                    .top(px(-scroll_offset))
                    .left(px(0.0))
                    .right(px(0.0))
                    .child(self.render_all_config_categories(cx)),
            )
            .when(has_scrollable_content, |node| {
                node.child(
                    // Visual scroll indicator at bottom
                    div()
                        .absolute()
                        .bottom(px(8.0))
                        .right(px(16.0))
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .bg(rgb(0x1e293b))
                        .border_1()
                        .border_color(rgb(0x475569))
                        .text_xs()
                        .text_color(rgb(0x94a3b8))
                        .child(format!(
                            "{:.0}%",
                            if max_scroll > 0.0 {
                                (scroll_offset / max_scroll * 100.0).min(100.0)
                            } else {
                                0.0
                            }
                        )),
                )
            });

        let panel_content = panel
            .child(header)
            .child(search_bar)
            .child(categories_content);

        div()
            .absolute()
            .inset_0()
            .child(backdrop)
            .child(panel_content)
    }

    fn render_all_config_categories(&self, cx: &mut Context<Self>) -> Div {
        let mut container = div().flex().flex_col().gap_4();
        let mut rendered_count = 0;

        // Only render categories that have matching parameters during search
        for category in ConfigCategory::all() {
            if self.category_has_matches(category) {
                container = container.child(self.render_config_category(category, cx));
                rendered_count += 1;
            }
        }

        // Global empty state if ALL categories filtered out during search
        if rendered_count == 0 && !self.settings_panel.search_query.is_empty() {
            container = container.child(
                div()
                    .flex()
                    .flex_col()
                    .items_center()
                    .justify_center()
                    .gap_3()
                    .py(px(64.0))
                    .child(div().text_3xl().child("🔍"))
                    .child(div().text_base().text_color(rgb(0x94a3b8)).child(format!(
                        "No parameters match \"{}\"",
                        self.settings_panel.search_query
                    )))
                    .child(div().text_sm().text_color(rgb(0x64748b)).child(
                        "Try a different search term or clear the search to see all parameters",
                    )),
            );
        }

        container
    }

    fn render_config_category(&self, category: ConfigCategory, cx: &mut Context<Self>) -> Div {
        let is_collapsed = self.settings_panel.collapsed_categories.contains(&category);

        let header = div()
            .flex()
            .items_center()
            .justify_between()
            .cursor_pointer()
            .px_4()
            .py_3()
            .rounded_lg()
            .bg(rgb(0x1e293b))
            .border_1()
            .border_color(rgb(0x334155))
            .hover(|s| s.bg(rgb(0x334155)))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, _, cx| {
                    this.toggle_category_collapse(category, cx);
                }),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(div().text_xl().child(category.icon()))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_base()
                                    .text_color(rgb(0xf1f5f9))
                                    .child(category.label()),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(0x94a3b8))
                                    .child(category.description()),
                            ),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_center()
                    .w_8()
                    .h_8()
                    .rounded_md()
                    .text_base()
                    .text_color(rgb(0x94a3b8))
                    .hover(|s| s.bg(rgb(0x475569)).text_color(rgb(0x60a5fa)))
                    .child(if is_collapsed { "▶" } else { "▼" }),
            );

        let mut category_div = div().flex().flex_col().gap_2().child(header);

        if !is_collapsed {
            category_div = category_div.child(self.render_category_parameters(category, cx));
        }

        category_div
    }
    fn render_category_parameters(&self, category: ConfigCategory, cx: &mut Context<Self>) -> Div {
        // Read current config from world
        let config = if let Ok(world) = self.world.lock() {
            world.config().clone()
        } else {
            scriptbots_core::ScriptBotsConfig::default()
        };

        // Match on category and return filtered params directly - ULTRA CLEAN data-driven approach!
        // ONE central filter loop in render_filtered_params handles ALL 60+ parameters!
        match category {
            ConfigCategory::World => {
                let params = vec![
                    (
                        "World Width",
                        format!("{} units", config.world_width),
                        "Horizontal extent of the simulation world",
                    ),
                    (
                        "World Height",
                        format!("{} units", config.world_height),
                        "Vertical extent of the simulation world",
                    ),
                    (
                        "Food Cell Size",
                        format!("{} units", config.food_cell_size),
                        "Size of each food grid cell",
                    ),
                    (
                        "Initial Food",
                        self.format_float(config.initial_food, 3),
                        "Starting food in each cell",
                    ),
                    (
                        "RNG Seed",
                        config
                            .rng_seed
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "Random".to_string()),
                        "Random number generator seed",
                    ),
                    (
                        "Chart Flush Interval",
                        format!("{} ticks", config.chart_flush_interval),
                        "History chart update frequency",
                    ),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Food => {
                let params = vec![
                    (
                        "Respawn Interval",
                        format!("{} ticks", config.food_respawn_interval),
                        "Ticks between food respawn events",
                    ),
                    (
                        "Respawn Amount",
                        self.format_float(config.food_respawn_amount, 3),
                        "Food added per respawn",
                    ),
                    (
                        "Maximum Food",
                        self.format_float(config.food_max, 3),
                        "Maximum food per cell",
                    ),
                    (
                        "Growth Rate",
                        self.format_float(config.food_growth_rate, 4),
                        "Logistic regrowth rate",
                    ),
                    (
                        "Decay Rate",
                        self.format_float(config.food_decay_rate, 4),
                        "Proportional decay rate",
                    ),
                    (
                        "Diffusion Rate",
                        self.format_float(config.food_diffusion_rate, 3),
                        "Neighbor exchange rate",
                    ),
                    (
                        "Intake Rate",
                        self.format_float(config.food_intake_rate, 3),
                        "Agent food consumption rate",
                    ),
                    (
                        "Sharing Radius",
                        self.format_float(config.food_sharing_radius, 1),
                        "Friendly neighbor sharing distance",
                    ),
                    (
                        "Sharing Rate",
                        self.format_float(config.food_sharing_rate, 3),
                        "Energy fraction shared per neighbor",
                    ),
                    (
                        "Transfer Rate",
                        self.format_float(config.food_transfer_rate, 4),
                        "Altruistic sharing amount",
                    ),
                    (
                        "Sharing Distance",
                        self.format_float(config.food_sharing_distance, 1),
                        "Altruistic sharing threshold",
                    ),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Agent => {
                let params = vec![
                    (
                        "Bot Speed",
                        self.format_float(config.bot_speed, 2),
                        "Base wheel speed multiplier",
                    ),
                    (
                        "Bot Radius",
                        self.format_float(config.bot_radius, 1),
                        "Agent radius for collisions",
                    ),
                    (
                        "Boost Multiplier",
                        format!("{}×", self.format_float(config.boost_multiplier, 2)),
                        "Speed boost when activated",
                    ),
                    (
                        "Sense Radius",
                        self.format_float(config.sense_radius, 1),
                        "Perception range",
                    ),
                    (
                        "Carnivore Threshold",
                        self.format_float(config.carnivore_threshold, 2),
                        "Herbivore tendency cutoff for carnivores",
                    ),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Metabolism => {
                let params = vec![
                    (
                        "Base Drain",
                        self.format_float(config.metabolism_drain, 4),
                        "Baseline energy cost",
                    ),
                    (
                        "Movement Drain",
                        self.format_float(config.movement_drain, 4),
                        "Cost per velocity",
                    ),
                    (
                        "Ramp Floor",
                        self.format_float(config.metabolism_ramp_floor, 2),
                        "Energy level for ramping",
                    ),
                    (
                        "Ramp Rate",
                        self.format_float(config.metabolism_ramp_rate, 4),
                        "Additional drain above floor",
                    ),
                    (
                        "Boost Penalty",
                        self.format_float(config.metabolism_boost_penalty, 4),
                        "Fixed boost cost",
                    ),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Temperature => {
                let params = vec![
                    (
                        "Discomfort Rate",
                        self.format_float(config.temperature_discomfort_rate, 4),
                        "Health drain multiplier",
                    ),
                    (
                        "Comfort Band",
                        format!("±{}", self.format_float(config.temperature_comfort_band, 3)),
                        "Tolerance threshold",
                    ),
                    (
                        "Gradient Exponent",
                        self.format_float(config.temperature_gradient_exponent, 2),
                        "Pole-to-equator shaping",
                    ),
                    (
                        "Discomfort Exp",
                        self.format_float(config.temperature_discomfort_exponent, 2),
                        "Discomfort scaling power",
                    ),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Reproduction => {
                let params = vec![
                    (
                        "Energy Threshold",
                        self.format_float(config.reproduction_energy_threshold, 2),
                        "Required energy to reproduce",
                    ),
                    (
                        "Energy Cost",
                        self.format_float(config.reproduction_energy_cost, 2),
                        "Parent's energy deduction",
                    ),
                    (
                        "Cooldown",
                        format!("{} ticks", config.reproduction_cooldown),
                        "Ticks between reproductions",
                    ),
                    (
                        "Herbivore Rate",
                        format!(
                            "{}×",
                            self.format_float(config.reproduction_rate_herbivore, 3)
                        ),
                        "Herbivore multiplier",
                    ),
                    (
                        "Carnivore Rate",
                        format!(
                            "{}×",
                            self.format_float(config.reproduction_rate_carnivore, 3)
                        ),
                        "Carnivore multiplier",
                    ),
                    (
                        "Child Energy",
                        self.format_float(config.reproduction_child_energy, 2),
                        "Starting energy for child",
                    ),
                    (
                        "Spawn Jitter",
                        format!(
                            "±{}",
                            self.format_float(config.reproduction_spawn_jitter, 1)
                        ),
                        "Position randomization",
                    ),
                    (
                        "Spawn Back Distance",
                        self.format_float(config.reproduction_spawn_back_distance, 1),
                        "Child spawn distance behind parent",
                    ),
                    (
                        "Color Jitter",
                        format!(
                            "±{}",
                            self.format_float(config.reproduction_color_jitter, 3)
                        ),
                        "RGB mutation range",
                    ),
                    (
                        "Mutation Scale",
                        self.format_float(config.reproduction_mutation_scale, 4),
                        "Trait mutation magnitude",
                    ),
                    (
                        "Partner Chance",
                        format!(
                            "{}%",
                            self.format_float(config.reproduction_partner_chance * 100.0, 1)
                        ),
                        "Crossover probability",
                    ),
                    (
                        "Gene Log Capacity",
                        format!("{}", config.reproduction_gene_log_capacity),
                        "Max gene history entries",
                    ),
                    (
                        "Meta-Mutation Chance",
                        format!(
                            "{}%",
                            self.format_float(config.reproduction_meta_mutation_chance * 100.0, 1)
                        ),
                        "Mutation rate mutation chance",
                    ),
                    (
                        "Meta-Mutation Scale",
                        self.format_float(config.reproduction_meta_mutation_scale, 4),
                        "Mutation rate change magnitude",
                    ),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Aging => {
                let params = vec![
                    (
                        "Decay Start Age",
                        format!("{} ticks", config.aging_health_decay_start),
                        "Age when decay begins",
                    ),
                    (
                        "Decay Rate",
                        self.format_float(config.aging_health_decay_rate, 5),
                        "Health loss per tick",
                    ),
                    (
                        "Decay Max",
                        self.format_float(config.aging_health_decay_max, 4),
                        "Maximum decay per tick",
                    ),
                    (
                        "Energy Penalty",
                        format!(
                            "{}×",
                            self.format_float(config.aging_energy_penalty_rate, 3)
                        ),
                        "Health-to-energy conversion",
                    ),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Combat => {
                let params = vec![
                    (
                        "Spike Radius",
                        self.format_float(config.spike_radius, 1),
                        "Base spike collision radius",
                    ),
                    (
                        "Spike Damage",
                        self.format_float(config.spike_damage, 2),
                        "Damage at full power",
                    ),
                    (
                        "Spike Energy Cost",
                        self.format_float(config.spike_energy_cost, 4),
                        "Energy cost to deploy",
                    ),
                    (
                        "Min Length",
                        self.format_float(config.spike_min_length, 2),
                        "Minimum for damage",
                    ),
                    (
                        "Alignment Cosine",
                        self.format_float(config.spike_alignment_cosine, 2),
                        "Directional threshold",
                    ),
                    (
                        "Speed Bonus",
                        format!("{}×", self.format_float(config.spike_speed_damage_bonus, 3)),
                        "Velocity scaling",
                    ),
                    (
                        "Length Bonus",
                        format!(
                            "{}×",
                            self.format_float(config.spike_length_damage_bonus, 3)
                        ),
                        "Length scaling",
                    ),
                    (
                        "Growth Rate",
                        self.format_float(config.spike_growth_rate, 4),
                        "Spike extension rate",
                    ),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Carcass => {
                let params = vec![
                    (
                        "Distribution Radius",
                        self.format_float(config.carcass_distribution_radius, 1),
                        "Reward share distance",
                    ),
                    (
                        "Health Reward",
                        self.format_float(config.carcass_health_reward, 2),
                        "Base health given",
                    ),
                    (
                        "Reproduction Reward",
                        self.format_float(config.carcass_reproduction_reward, 1),
                        "Cooldown reduction",
                    ),
                    (
                        "Neighbor Exponent",
                        self.format_float(config.carcass_neighbor_exponent, 2),
                        "Sharing normalization",
                    ),
                    (
                        "Maturity Age",
                        format!("{} ticks", config.carcass_maturity_age),
                        "Full reward age",
                    ),
                    (
                        "Energy Share",
                        format!(
                            "{}%",
                            self.format_float(config.carcass_energy_share_rate * 100.0, 1)
                        ),
                        "Health-to-energy conversion",
                    ),
                    (
                        "Indicator Scale",
                        self.format_float(config.carcass_indicator_scale, 2),
                        "Visual pulse intensity",
                    ),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Topography => {
                // Topography has a toggle - hybrid approach with search-filtered toggle, then readonly params
                let mut container = div()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .px_4()
                    .py_4()
                    .rounded_lg()
                    .bg(rgb(0x0f172a))
                    .border_1()
                    .border_color(rgb(0x1e293b));

                // Add toggle if it matches search filter
                if self.matches_search("Enabled")
                    || self.matches_search("Enable terrain elevation effects")
                {
                    container = container.child(self.render_param_toggle(
                        "Enabled",
                        config.topography_enabled,
                        "Enable terrain elevation effects",
                        cx,
                    ));
                }

                // Add filterable readonly params
                let params = vec![
                    (
                        "Speed Gain",
                        self.format_float(config.topography_speed_gain, 3),
                        "Downhill boost per unit slope",
                    ),
                    (
                        "Energy Penalty",
                        self.format_float(config.topography_energy_penalty, 4),
                        "Uphill cost per unit slope",
                    ),
                ];

                for (label, value, desc) in params {
                    if self.matches_search(label)
                        || self.matches_search(&value)
                        || self.matches_search(desc)
                    {
                        container =
                            container.child(self.render_param_readonly(label, &value, desc));
                    }
                }

                container
            }

            ConfigCategory::Population => {
                let params = vec![
                    (
                        "Minimum Population",
                        format!("{}", config.population_minimum),
                        "Auto-seed threshold",
                    ),
                    (
                        "Spawn Interval",
                        format!("{} ticks", config.population_spawn_interval),
                        "Ticks between spawns",
                    ),
                    (
                        "Spawn Count",
                        format!("{}", config.population_spawn_count),
                        "Agents per interval",
                    ),
                    (
                        "Crossover Chance",
                        format!(
                            "{}%",
                            self.format_float(config.population_crossover_chance * 100.0, 1)
                        ),
                        "Breed vs. random spawn",
                    ),
                ];
                self.render_filtered_params(params)
            }

            ConfigCategory::Persistence => {
                let params = vec![
                    (
                        "Interval",
                        format!("{} ticks", config.persistence_interval),
                        "Database flush frequency",
                    ),
                    (
                        "History Capacity",
                        format!("{}", config.history_capacity),
                        "In-memory tick summaries",
                    ),
                ];
                self.render_filtered_params(params)
            }
        }
    }

    /// Helper to safely format floats with NaN/Inf guards
    fn format_float(&self, value: f32, precision: usize) -> String {
        if !value.is_finite() {
            if value.is_nan() {
                "NaN".to_string()
            } else if value.is_infinite() {
                if value.is_sign_positive() {
                    "∞".to_string()
                } else {
                    "-∞".to_string()
                }
            } else {
                "ERR".to_string()
            }
        } else {
            format!("{:.prec$}", value, prec = precision)
        }
    }

    fn render_param_readonly(&self, label: &str, value: &str, description: &str) -> Div {
        let label_owned = label.to_string();
        let value_owned = value.to_string();
        let description_owned = description.to_string();

        div()
            .flex()
            .flex_col()
            .gap_2()
            .py_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(div().text_sm().text_color(rgb(0xf1f5f9)).child(label_owned))
                    .child(div().text_sm().text_color(rgb(0x60a5fa)).child(value_owned)),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x94a3b8))
                    .child(description_owned),
            )
    }

    fn render_param_toggle(
        &self,
        label: &str,
        enabled: bool,
        description: &str,
        _cx: &mut Context<Self>,
    ) -> Div {
        let label_owned = label.to_string();
        let description_owned = description.to_string();

        div()
            .flex()
            .flex_col()
            .gap_2()
            .py_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(div().text_sm().text_color(rgb(0xf1f5f9)).child(label_owned))
                    .child(
                        div()
                            .text_xs()
                            .text_color(if enabled {
                                rgb(0x86efac)
                            } else {
                                rgb(0xfca5a5)
                            })
                            .child(if enabled {
                                "✓ ENABLED"
                            } else {
                                "○ DISABLED"
                            }),
                    ),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x94a3b8))
                    .child(description_owned),
            )
    }
}

/// Render an agent's sensor or actuator vector as labelled bars.
///
/// The labels come from the canonical channel layout in `scriptbots-core`, not
/// from a list retyped here: an inspector that says "s0: 0.42" tells you nothing,
/// and an inspector that invents its own names eventually disagrees with the
/// simulation about what a slot means. That disagreement is exactly how combat
/// spent months treating the green colour channel as "boost".
///
/// Every channel is shown. The old version truncated sensors to the first eight,
/// which silently hid blood, temperature, and the whole of eye 3 — the channels a
/// user is most likely to be hunting for when an agent behaves strangely.
/// Bounded contributor list requested per focused agent (bd-16g.4.2).
const SENSE_PROBE_MAX_CONTRIBUTORS: usize = 12;
/// Contributor rows shown before the panel defers to the truncation count.
const SENSE_PROBE_VISIBLE_CONTRIBUTORS: usize = 8;

/// Truthful per-channel provenance tag (bd-16g.4.2): attribution only exists
/// for neighbour-derived channels, and the panel must say where every other
/// channel comes from instead of implying "no neighbours detected".
const fn sensor_source_tag(kind: SensorKind) -> &'static str {
    match kind {
        SensorKind::EyeDensity
        | SensorKind::EyeRed
        | SensorKind::EyeGreen
        | SensorKind::EyeBlue
        | SensorKind::Sound
        | SensorKind::Smell
        | SensorKind::Hearing
        | SensorKind::Blood => "nbr",
        SensorKind::Food => "grid",
        SensorKind::Health | SensorKind::Clock => "self",
        SensorKind::Temperature => "pos",
    }
}

/// Egocentric sense attribution for the focused agent (bd-16g.4.2).
///
/// Renders `SensorAttribution` verbatim: clamped values, an explicit `⚠raw`
/// marker on saturated channels (contributions legitimately sum above 1.0 —
/// normalising them would destroy the information), eye rows labeled with
/// their true relative angle and FOV, `SENSOR_LAYOUT`-derived source tags,
/// and the strongest contributors with their perceived colours.
fn render_sense_attribution(detail: &AgentInspectorDetails) -> Div {
    let container = div()
        .flex()
        .flex_col()
        .gap_1()
        .rounded_md()
        .border_1()
        .border_color(rgb(0x1e3a5f))
        .bg(rgb(0x0d1826))
        .px_2()
        .py_2();

    let Some(attribution) = detail.sense_attribution.as_ref() else {
        return container.child(
            div()
                .text_xs()
                .text_color(rgb(0x64748b))
                .child("Sense attribution unavailable (agent vanished this frame)"),
        );
    };

    let truncation = if attribution.truncated > 0 {
        format!(" (+{} truncated)", attribution.truncated)
    } else {
        String::new()
    };
    let mut panel = container.child(div().text_xs().text_color(rgb(0x94a3b8)).child(format!(
        "Sense attribution · t{} · {} contributors{truncation}",
        attribution.tick.0,
        attribution.contributions.len(),
    )));

    // True-angle strip: the agent's full [-180°, +180°] egocentric field
    // unrolled at 1px/degree. Each eye covers its real angular span at its
    // real direction, tinted by what it perceives; contributor ticks sit at
    // their true bearings. Heading (0°) is the center line.
    {
        let strip_width = 360.0_f32;
        let clamp_x = |x: f32| x.clamp(0.0, strip_width - 2.0);
        let mut strip = div()
            .relative()
            .w(px(strip_width))
            .h(px(16.0))
            .rounded_sm()
            .bg(rgb(0x0a1420));
        for eye in 0..NUM_EYES {
            let mut seen = [0.0_f32; 3];
            let mut density = 0.0_f32;
            for channel in SENSOR_LAYOUT.iter().filter(|c| c.eye == Some(eye)) {
                let clamped = attribution.clamped[channel.index];
                match channel.kind {
                    SensorKind::EyeDensity => density = clamped,
                    SensorKind::EyeRed => seen[0] = clamped,
                    SensorKind::EyeGreen => seen[1] = clamped,
                    SensorKind::EyeBlue => seen[2] = clamped,
                    _ => {}
                }
            }
            let direction = detail.eye_directions[eye].to_degrees();
            let fov = detail.eye_fovs[eye].to_degrees().max(1.0);
            let left = clamp_x(strip_width / 2.0 + direction - fov / 2.0);
            let width = fov.min(strip_width - left).max(1.0);
            // Density scales brightness; an empty cone stays faintly visible
            // so its coverage is still readable.
            let brightness = density.mul_add(0.75, 0.25);
            let tint = [
                seen[0] * brightness,
                seen[1] * brightness,
                seen[2] * brightness,
            ];
            strip = strip.child(
                div()
                    .absolute()
                    .left(px(left))
                    .top(px(3.0))
                    .w(px(width))
                    .h(px(10.0))
                    .rounded_sm()
                    .bg(rgb_from_triplet(tint)),
            );
        }
        // Contributor bearings as full-height ticks in their perceived colour.
        for contribution in &attribution.contributions {
            let x = clamp_x(strip_width / 2.0 + contribution.bearing.to_degrees());
            strip = strip.child(
                div()
                    .absolute()
                    .left(px(x))
                    .top(px(0.0))
                    .w(px(2.0))
                    .h(px(16.0))
                    .bg(rgb_from_triplet(contribution.color)),
            );
        }
        // Heading line at 0°.
        strip = strip.child(
            div()
                .absolute()
                .left(px(strip_width / 2.0 - 1.0))
                .top(px(0.0))
                .w(px(1.0))
                .h(px(16.0))
                .bg(rgb(0x64748b)),
        );
        panel = panel
            .child(strip)
            .child(div().text_xs().text_color(rgb(0x475569)).child(
                "-180°            eye cones at true angles · ticks = contributors            +180°",
            ));
    }

    for eye in 0..NUM_EYES {
        let mut rgb_seen = [0.0_f32; 3];
        let mut density = 0.0_f32;
        let mut saturated = false;
        let mut cells = String::new();
        for channel in SENSOR_LAYOUT.iter().filter(|c| c.eye == Some(eye)) {
            let index = channel.index;
            let clamped = attribution.clamped[index];
            match channel.kind {
                SensorKind::EyeDensity => density = clamped,
                SensorKind::EyeRed => rgb_seen[0] = clamped,
                SensorKind::EyeGreen => rgb_seen[1] = clamped,
                SensorKind::EyeBlue => rgb_seen[2] = clamped,
                _ => {}
            }
            if attribution.saturated[index] {
                saturated = true;
                cells.push_str(&format!(" ⚠{}={:.1}", channel.name, attribution.raw[index]));
            }
        }
        let direction = detail.eye_directions[eye].to_degrees();
        let fov = detail.eye_fovs[eye].to_degrees();
        let marker = if saturated { cells.as_str() } else { "" };
        panel = panel.child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(color_swatch(rgb_seen))
                .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                    "eye{eye} ∠{direction:+.0}° fov {fov:.0}° · ρ{density:.2}{marker}"
                ))),
        );
    }

    let mut scalar_line = String::new();
    for channel in SENSOR_LAYOUT.iter().filter(|c| c.eye.is_none()) {
        let index = channel.index;
        scalar_line.push_str(&format!(
            "{} {:.2}[{}]",
            channel.name,
            attribution.clamped[index],
            sensor_source_tag(channel.kind)
        ));
        if attribution.saturated[index] {
            scalar_line.push_str(&format!("⚠{:.1}", attribution.raw[index]));
        }
        scalar_line.push_str("  ");
    }
    panel = panel.child(
        div()
            .text_xs()
            .text_color(rgb(0x94a3b8))
            .child(scalar_line.trim_end().to_owned()),
    );

    if attribution.contributions.is_empty() {
        panel = panel.child(
            div()
                .text_xs()
                .text_color(rgb(0x64748b))
                .child("no neighbours within sense radius (self/grid channels stay live)"),
        );
    } else {
        for contribution in attribution
            .contributions
            .iter()
            .take(SENSE_PROBE_VISIBLE_CONTRIBUTORS)
        {
            let dominant_eye = contribution
                .eye_density
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(eye, _)| eye);
            panel = panel.child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(color_swatch(contribution.color))
                    .child(div().text_xs().text_color(rgb(0xcbd5f5)).child(format!(
                        "#{} ∠{:+.0}° d{:.0} eye{} Σ{:.2}",
                        contribution.source_uid.0,
                        contribution.bearing.to_degrees(),
                        contribution.distance,
                        dominant_eye,
                        contribution.total,
                    ))),
            );
        }
        let hidden = attribution
            .contributions
            .len()
            .saturating_sub(SENSE_PROBE_VISIBLE_CONTRIBUTORS);
        if hidden > 0 {
            panel = panel.child(
                div()
                    .text_xs()
                    .text_color(rgb(0x64748b))
                    .child(format!("… +{hidden} weaker contributors")),
            );
        }
    }

    panel
}

fn render_brain_bars(values: &[f32], is_sensor: bool) -> Div {
    let mut rows: Vec<Div> = Vec::new();
    let max_val = values
        .iter()
        .copied()
        .fold(0.0_f32, |m, v| m.max(v.abs()))
        .max(1e-3);
    for (idx, v) in values.iter().copied().enumerate() {
        let name = if is_sensor {
            SENSOR_LAYOUT
                .get(idx)
                .map_or_else(|| format!("s{idx}"), |channel| channel.name.to_owned())
        } else {
            OutputChannel::ALL
                .get(idx)
                .map_or_else(|| format!("o{idx}"), |channel| channel.name().to_owned())
        };
        let width = (v.abs() / max_val).clamp(0.0, 1.0);
        let color = if is_sensor {
            rgb(0x60a5fa)
        } else {
            rgb(0xf59e0b)
        };
        let bar = div().w(px(120.0 * width)).h(px(8.0)).bg(color).rounded_sm();
        let label = div()
            .text_xs()
            .text_color(rgb(0x94a3b8))
            .w(px(84.0))
            .child(name);
        rows.push(
            div()
                .flex()
                .gap_2()
                .items_center()
                .child(label)
                .child(bar)
                .child(
                    div()
                        .text_xs()
                        .text_color(rgb(0xcbd5f5))
                        .child(format!("{v:.2}")),
                ),
        );
    }
    div().flex().flex_col().gap_1().children(rows)
}
/// Top-k attribution rows computed per output in the brain panel (bd-16g.4.3).
const BRAIN_PANEL_TOP_K: usize = 3;

/// The brain panel's named-output attribution list (bd-16g.4.3): the same
/// `OutputExplanation` the TUI consumes — every output with its canonical
/// wire-map name, raw value, actuator state, and top-k driving sensors, or the
/// honest reason the snapshot cannot say. Unbound agents get the passthrough
/// explanation; the panel never fabricates attribution over an identity copy.
fn render_output_attributions(detail: &AgentInspectorDetails) -> Div {
    let mut rows: Vec<Div> = Vec::new();
    rows.push(
        div()
            .text_xs()
            .text_color(rgb(0x94a3b8))
            .child("Output attribution"),
    );

    let explanations: Vec<OutputExplanation> =
        if detail.outputs.len() >= scriptbots_core::OUTPUT_SIZE {
            let outputs: &[f32; scriptbots_core::OUTPUT_SIZE] = detail.outputs
                [..scriptbots_core::OUTPUT_SIZE]
                .try_into()
                .expect("length checked above");
            explain_outputs(
                outputs,
                detail.brain_bound,
                detail.brain_activations.as_ref(),
                BRAIN_PANEL_TOP_K,
            )
        } else {
            rows.push(
                div()
                    .text_xs()
                    .text_color(rgb(0xf59e0b))
                    .child("output vector not captured yet"),
            );
            return div().flex().flex_col().gap_1().children(rows);
        };

    // A shared Unavailable reason gets one line instead of nine.
    let shared_reason = explanations.first().and_then(|first| {
        if let AttributionMethod::Unavailable(reason) = first.method
            && explanations
                .iter()
                .all(|explanation| explanation.method == AttributionMethod::Unavailable(reason))
        {
            Some(reason.reason())
        } else {
            None
        }
    });
    if let Some(reason) = shared_reason {
        rows.push(
            div()
                .text_xs()
                .text_color(rgb(0xf59e0b))
                .child(format!("({reason})")),
        );
    }

    for explanation in &explanations {
        let effective = match &explanation.effective {
            EffectiveOutput::Continuous(value) => format!("{value:.2}"),
            EffectiveOutput::Thresholded { raw, active, .. } => {
                format!("{raw:.2} {}", if *active { "ON" } else { "off" })
            }
            EffectiveOutput::Clamped { raw, applied } => {
                if (raw - applied).abs() > f32::EPSILON {
                    format!("{raw:.2}>{applied:.2}")
                } else {
                    format!("{raw:.2}")
                }
            }
        };
        let drivers = match &explanation.method {
            AttributionMethod::Unavailable(reason) if shared_reason.is_none() => {
                format!(" ({})", reason.reason())
            }
            AttributionMethod::Unavailable(_) => String::new(),
            _ if explanation.inputs.is_empty() => " (no drivers above k)".to_owned(),
            _ => explanation
                .inputs
                .iter()
                .take(2)
                .map(|input| format!(" {} {:+.2}", input.sensor_name, input.contribution))
                .collect::<Vec<_>>()
                .join(""),
        };
        rows.push(
            div()
                .flex()
                .gap_2()
                .child(
                    div()
                        .w(px(110.0))
                        .text_xs()
                        .text_color(rgb(0xe2e8f0))
                        .child(explanation.output_name),
                )
                .child(
                    div()
                        .w(px(90.0))
                        .text_xs()
                        .text_color(rgb(0x94a3b8))
                        .child(effective),
                )
                .child(div().text_xs().text_color(rgb(0x7dd3fc)).child(drivers)),
        );
    }
    div().flex().flex_col().gap_1().children(rows)
}

fn render_activation_heatmaps(activations: &Option<BrainActivations>) -> Div {
    let mut rows: Vec<Div> = Vec::new();
    if let Some(act) = activations {
        if act.truncated {
            rows.push(
                div()
                    .text_xs()
                    .text_color(rgb(0xf59e0b))
                    .child("Activation view clipped to the explicit inspection budget"),
            );
        }
        for layer in &act.layers {
            let layer_name = layer.name.clone();
            let state = Arc::new(layer.clone());
            let state_for_canvas = Arc::clone(&state);
            let canvas_el = canvas(
                move |_, _, _| Arc::clone(&state_for_canvas),
                move |bounds, st, window, _| {
                    paint_activation_grid(bounds, st.as_ref(), window);
                },
            )
            .w(px(200.0))
            .h(px(120.0));

            // Optional edges overlay limited to within this layer's index space
            let cell_count = state.width.checked_mul(state.height).unwrap_or_default();
            let edges_in_layer: Vec<ActivationEdge> = act
                .connections
                .iter()
                .copied()
                .filter(|e| e.from < cell_count && e.to < cell_count)
                .take(64)
                .collect();

            let edges_state = Arc::new((state.width, state.height, edges_in_layer));
            let edges_state_for_canvas = Arc::clone(&edges_state);
            let edges_canvas = canvas(
                move |_, _, _| Arc::clone(&edges_state_for_canvas),
                move |bounds, st, window, _| {
                    paint_activation_edges(bounds, st.as_ref(), window);
                },
            )
            .w(px(200.0))
            .h(px(120.0));

            rows.push(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(div().text_xs().text_color(rgb(0x94a3b8)).child(layer_name))
                    .child(div().relative().child(canvas_el).child(edges_canvas)),
            );
        }
        // Optional: show top-N connections if provided (textual summary)
        if !act.connections.is_empty() {
            let mut lines: Vec<Div> = Vec::new();
            if act.connections.len() > 64 {
                lines.push(div().text_xs().text_color(rgb(0xf59e0b)).child(format!(
                    "Display overlay is limited to 64 of {} payload edges per layer",
                    act.connections.len()
                )));
            }
            for edge in act.connections.iter().take(8) {
                lines.push(div().text_xs().text_color(rgb(0x64748b)).child(format!(
                    "edge {}→{} w={:.2}",
                    edge.from, edge.to, edge.weight
                )));
            }
            if act.connections.len() > 8 {
                lines.push(div().text_xs().text_color(rgb(0xf59e0b)).child(format!(
                    "Text summary shows 8 of {} payload edges",
                    act.connections.len()
                )));
            }
            rows.push(div().flex().flex_col().gap_0().children(lines));
        }
    } else {
        rows.push(
            div()
                .text_xs()
                .text_color(rgb(0x475569))
                .child("Activations: not available"),
        );
    }
    div()
        .flex()
        .flex_col()
        .gap_2()
        .rounded_md()
        .border_1()
        .border_color(rgb(0x1e293b))
        .bg(rgb(0x0f172a))
        .px_3()
        .py_2()
        .child(
            div()
                .text_xs()
                .text_color(rgb(0x94a3b8))
                .child("Brain activations"),
        )
        .children(rows)
}

fn paint_activation_grid(bounds: Bounds<Pixels>, layer: &ActivationLayer, window: &mut Window) {
    window.paint_quad(fill(
        bounds,
        Background::from(rgba_from_hex(0x0b1223, 0.92)),
    ));
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let width = f32::from(bounds_size.width).max(1.0);
    let height = f32::from(bounds_size.height).max(1.0);
    let cols = layer.width.max(1) as u16;
    let rows = layer.height.max(1) as u16;
    let cell_w = width / cols as f32;
    let cell_h = height / rows as f32;
    for y in 0..rows {
        for x in 0..cols {
            let idx = y as usize * layer.width + x as usize;
            let v = layer
                .values
                .get(idx)
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let color = lerp_rgba(
                rgba_from_hex(0x1e293b, 0.95),
                rgba_from_hex(0x22d3ee, 0.95),
                v,
            );
            let rect = Bounds::new(
                point(
                    px(f32::from(origin.x) + x as f32 * cell_w),
                    px(f32::from(origin.y) + y as f32 * cell_h),
                ),
                size(px(cell_w.max(1.0)), px(cell_h.max(1.0))),
            );
            window.paint_quad(fill(rect, Background::from(color)));
        }
    }
}

fn paint_activation_edges(
    bounds: Bounds<Pixels>,
    state: &(usize, usize, Vec<ActivationEdge>),
    window: &mut Window,
) {
    let (cols, rows, edges) = (state.0, state.1, state.2.as_slice());
    if cols == 0 || rows == 0 || edges.is_empty() {
        return;
    }
    let origin = bounds.origin;
    let size = bounds.size;
    let width = f32::from(size.width).max(1.0);
    let height = f32::from(size.height).max(1.0);
    let cell_w = width / cols as f32;
    let cell_h = height / rows as f32;

    for edge in edges {
        let from_x = (edge.from % cols) as f32 + 0.5;
        let from_y = (edge.from / cols) as f32 + 0.5;
        let to_x = (edge.to % cols) as f32 + 0.5;
        let to_y = (edge.to / cols) as f32 + 0.5;
        let x1 = f32::from(origin.x) + from_x * cell_w;
        let y1 = f32::from(origin.y) + from_y * cell_h;
        let x2 = f32::from(origin.x) + to_x * cell_w;
        let y2 = f32::from(origin.y) + to_y * cell_h;
        let w = edge.weight.abs().clamp(0.1, 1.0);
        let mut path = PathBuilder::stroke(px(0.5 + 1.5 * w));
        path.move_to(point(px(x1), px(y1)));
        path.line_to(point(px(x2), px(y2)));
        if let Ok(path) = path.build() {
            let color = if edge.weight >= 0.0 {
                rgba_from_hex(0x22d3ee, 0.85)
            } else {
                rgba_from_hex(0xf471b5, 0.85)
            };
            window.paint_path(path, color);
        }
    }
}
fn render_brain_card(detail: &AgentInspectorDetails) -> Div {
    // Compact card summarizing brain type and a radial radar chart
    let (best_idx, best_val) = detail
        .outputs
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0));

    let outputs = detail.outputs.clone();
    let sensors = detail.sensors.clone();
    let radar_canvas = canvas(
        move |_, _, _| (sensors.clone(), outputs.clone()),
        move |bounds, state, window, _| {
            paint_brain_radar(bounds, state.0.as_slice(), state.1.as_slice(), window)
        },
    )
    .w(px(180.0))
    .h(px(120.0));

    div()
        .flex()
        .flex_col()
        .gap_2()
        .rounded_md()
        .border_1()
        .border_color(rgb(0x1e293b))
        .bg(rgb(0x0f172a))
        .px_3()
        .py_2()
        .child(div().text_xs().text_color(rgb(0x94a3b8)).child("Brain"))
        .child(
            div()
                .text_sm()
                .text_color(rgb(0xcbd5f5))
                .child(detail.brain_descriptor.clone()),
        )
        .child(
            div()
                .text_xs()
                .text_color(rgb(0x94a3b8))
                .child(format!("dominant o{} {:.2}", best_idx, best_val)),
        )
        .child(radar_canvas)
}
fn paint_brain_radar(
    bounds: Bounds<Pixels>,
    sensors: &[f32],
    outputs: &[f32],
    window: &mut Window,
) {
    let origin = bounds.origin;
    let size = bounds.size;
    let width = f32::from(size.width).max(1.0);
    let height = f32::from(size.height).max(1.0);
    window.paint_quad(fill(
        bounds,
        Background::from(rgba_from_hex(0x0b1223, 0.92)),
    ));
    let cx = f32::from(origin.x) + width * 0.5;
    let cy = f32::from(origin.y) + height * 0.58;
    let radius = (width.min(height) * 0.42).max(24.0);

    // Web-like spokes
    let n = outputs.len().max(3);
    for ring in [0.3, 0.6, 1.0] {
        let mut ring_path = PathBuilder::stroke(px(1.0));
        for i in 0..n {
            let ang = (i as f32) / (n as f32) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
            let x = cx + radius * ring * ang.cos();
            let y = cy + radius * ring * ang.sin();
            if i == 0 {
                ring_path.move_to(point(px(x), px(y)));
            } else {
                ring_path.line_to(point(px(x), px(y)));
            }
        }
        // close polygon
        let ang0 = -std::f32::consts::FRAC_PI_2;
        let x0 = cx + radius * ring * ang0.cos();
        let y0 = cy + radius * ring * ang0.sin();
        ring_path.line_to(point(px(x0), px(y0)));
        if let Ok(path) = ring_path.build() {
            window.paint_path(path, rgba_from_hex(0x1e293b, 0.9));
        }
    }

    // Spokes
    for i in 0..n {
        let ang = (i as f32) / (n as f32) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
        let x = cx + radius * ang.cos();
        let y = cy + radius * ang.sin();
        let mut spoke = PathBuilder::stroke(px(1.0));
        spoke.move_to(point(px(cx), px(cy)));
        spoke.line_to(point(px(x), px(y)));
        if let Ok(path) = spoke.build() {
            window.paint_path(path, rgba_from_hex(0x1e293b, 0.9));
        }
    }

    // Sensor heat ring around the radar (real sensor activations)
    let n = sensors.len().max(3);
    let ring_inner = radius * 1.05;
    let ring_outer = radius * 1.18;
    for i in 0..n {
        let val = sensors.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let ang = (i as f32) / (n as f32) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
        let x1 = cx + ring_inner * ang.cos();
        let y1 = cy + ring_inner * ang.sin();
        let x2 = cx + (ring_inner + (ring_outer - ring_inner) * val) * ang.cos();
        let y2 = cy + (ring_inner + (ring_outer - ring_inner) * val) * ang.sin();
        let mut seg = PathBuilder::stroke(px(3.0));
        seg.move_to(point(px(x1), px(y1)));
        seg.line_to(point(px(x2), px(y2)));
        if let Ok(path) = seg.build() {
            window.paint_path(path, rgba_from_hex(0xf97316, 0.85));
        }
    }

    // Filled polygon of output magnitudes (clamped 0..1)
    let m = outputs.len().max(3);
    let mut poly = PathBuilder::fill();
    for i in 0..m {
        let val = outputs.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let ang = (i as f32) / (m as f32) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
        let x = cx + radius * val * ang.cos();
        let y = cy + radius * val * ang.sin();
        if i == 0 {
            poly.move_to(point(px(x), px(y)));
        } else {
            poly.line_to(point(px(x), px(y)));
        }
    }
    if let Ok(path) = poly.build() {
        window.paint_path(path, rgba_from_hex(0x22d3ee, 0.35));
    }
}
/// Render a simple PNG snapshot of the current world without a live window.
/// This is a coarse, deterministic rasterization intended for REST exports.
/// Render-relevant world state for the offscreen PNG snapshot (bd-134).
///
/// Captured from the world in one cheap pass so that rasterization — the
/// expensive part — can run with **no world lock held at all**. Callers that
/// serve a live, contended world must capture under a short lock and then
/// rasterize the scene outside it; callers with exclusive worlds may use the
/// [`render_png_offscreen`] convenience composition.
pub struct OffscreenScene {
    tonemap_mode: RenderTonemapMode,
    exposure_factor: f32,
    world_size: (f32, f32),
    cell_size: f32,
    bot_radius: f32,
    terrain: TerrainLayer,
    food: FoodGrid,
    agents: Vec<OffscreenAgent>,
}

/// One agent's render-relevant sample inside an [`OffscreenScene`].
struct OffscreenAgent {
    x: f32,
    y: f32,
    energy: f32,
    herbivore_tendency: f32,
}

impl OffscreenScene {
    /// Copy everything the offscreen renderer reads, in one pass.
    #[must_use]
    pub fn capture(world: &WorldState) -> Self {
        let config = world.config();
        let tonemap_settings = &config.render;
        let columns = world.agents().columns();
        let positions = columns.positions();
        let agents = world
            .agents()
            .iter_handles()
            .enumerate()
            .map(|(idx, handle)| {
                let position = positions[idx];
                let runtime = world.runtime().get(handle);
                OffscreenAgent {
                    x: position.x,
                    y: position.y,
                    energy: runtime.map(|rt| rt.energy).unwrap_or(0.5),
                    herbivore_tendency: runtime.map(|rt| rt.herbivore_tendency).unwrap_or(0.5),
                }
            })
            .collect();
        Self {
            tonemap_mode: tonemap_settings
                .tonemap_mode
                .unwrap_or(RenderTonemapMode::Aces),
            exposure_factor: tonemap_settings
                .tonemap_exposure_bias
                .map(|bias| 2f32.powf(bias))
                .unwrap_or(1.0),
            world_size: (config.world_width as f32, config.world_height as f32),
            cell_size: config.food_cell_size as f32,
            bot_radius: config.bot_radius,
            terrain: world.terrain().clone(),
            food: world.food().clone(),
            agents,
        }
    }
}

/// Capture-and-render composition for callers with an uncontended world.
pub fn render_png_offscreen(world: &WorldState, width: u32, height: u32) -> Vec<u8> {
    render_offscreen_scene(&OffscreenScene::capture(world), width, height)
}

/// Rasterize a captured scene. Holds no world reference and takes no lock.
pub fn render_offscreen_scene(scene: &OffscreenScene, width: u32, height: u32) -> Vec<u8> {
    let mut img: ImageBuffer<ImgRgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    let tonemap_mode = scene.tonemap_mode;
    let exposure_factor = scene.exposure_factor;
    let world_size = scene.world_size;
    let cell_size = scene.cell_size;

    let mut camera = Camera::default();
    let layout = camera.layout((0.0, 0.0), (width as f32, height as f32), world_size);

    let terrain = &scene.terrain;
    let food = &scene.food;

    let default_tile = TerrainTile {
        kind: TerrainKind::Grass,
        elevation: 0.5,
        moisture: 0.5,
        accent: 0.0,
        fertility_bias: 0.0,
        temperature_bias: 0.5,
        palette_index: 0,
    };

    for y in 0..height {
        for x in 0..width {
            let world_point = camera.screen_to_world(Point {
                x: px(x as f32 + 0.5),
                y: px(y as f32 + 0.5),
            });

            let rgba = if let Some((world_x, world_y)) = world_point {
                let tx = (world_x / cell_size).floor() as i32;
                let ty = (world_y / cell_size).floor() as i32;
                if tx >= 0
                    && ty >= 0
                    && (tx as u32) < terrain.width()
                    && (ty as u32) < terrain.height()
                {
                    let tile = terrain
                        .tile(tx as u32, ty as u32)
                        .copied()
                        .unwrap_or(default_tile);
                    let food_val = food.get(tx as u32, ty as u32).unwrap_or(0.0);
                    let base = match tile.kind {
                        TerrainKind::DeepWater => (30u8, 63u8, 102u8),
                        TerrainKind::ShallowWater => (47, 115, 179),
                        TerrainKind::Sand => (177, 78, 7),
                        TerrainKind::Grass => (80, 169, 19),
                        TerrainKind::Bloom => (121, 212, 109),
                        TerrainKind::Rock => (169, 177, 186),
                    };
                    let food_shade = (food_val.clamp(0.0, 1.0) * 90.0) as u8;
                    let mapped = tonemap_rgb(
                        ColorVec3 {
                            r: base.0.saturating_add(food_shade),
                            g: base.1.saturating_add(food_shade / 2),
                            b: base.2,
                        },
                        exposure_factor,
                        tonemap_mode,
                    );
                    ImgRgba([mapped.r, mapped.g, mapped.b, 255])
                } else {
                    let mapped = tonemap_rgb(
                        ColorVec3 {
                            r: 10,
                            g: 16,
                            b: 24,
                        },
                        exposure_factor,
                        tonemap_mode,
                    );
                    ImgRgba([mapped.r, mapped.g, mapped.b, 255])
                }
            } else {
                let mapped = tonemap_rgb(
                    ColorVec3 {
                        r: 10,
                        g: 16,
                        b: 24,
                    },
                    exposure_factor,
                    tonemap_mode,
                );
                ImgRgba([mapped.r, mapped.g, mapped.b, 255])
            };

            img.put_pixel(x, y, rgba);
        }
    }

    for agent in &scene.agents {
        let Some((screen_x, screen_y)) = camera.world_to_screen((agent.x, agent.y)) else {
            continue;
        };
        let fx = screen_x.round() as i32;
        let fy = screen_y.round() as i32;
        let energy = agent.energy;
        let base_radius = scene.bot_radius.max(1.0);
        let scale = layout.scale.max(f32::EPSILON);
        let energy_boost = base_radius * (0.5 + energy.clamp(0.0, 1.0));
        let radius = (scale * energy_boost).round().max(2.0) as i32;
        let tendency = agent.herbivore_tendency;
        let color = if tendency <= 0.33 {
            ColorVec3 {
                r: 120,
                g: 200,
                b: 120,
            }
        } else if tendency >= 0.66 {
            ColorVec3 {
                r: 220,
                g: 80,
                b: 80,
            }
        } else {
            ColorVec3 {
                r: 200,
                g: 180,
                b: 90,
            }
        };
        let color = tonemap_rgb(color, exposure_factor, tonemap_mode);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    let px = fx + dx;
                    let py = fy + dy;
                    if px >= 0 && py >= 0 && (px as u32) < width && (py as u32) < height {
                        img.put_pixel(
                            px as u32,
                            py as u32,
                            ImgRgba([color.r, color.g, color.b, 255]),
                        );
                    }
                }
            }
        }
    }

    let mut bytes = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut bytes);
        let _ = img.write_to(&mut cursor, image::ImageFormat::Png);
    }
    bytes
}

#[derive(Clone, Copy)]
struct ColorVec3 {
    r: u8,
    g: u8,
    b: u8,
}

/// sRGB-to-linear decode (the exact inverse of the encode in `linear_to_srgb_byte`).
/// Ownership rule (bd-2z0.7.11): display bytes are ALWAYS sRGB; every tonemap/exposure
/// operation happens in linear space, and encoding happens exactly once on the way out.
fn srgb_to_linear(x: f32) -> f32 {
    if x <= 0.040_45 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn tonemap_rgb(color: ColorVec3, exposure: f32, mode: RenderTonemapMode) -> ColorVec3 {
    // `color` arrives as sRGB-encoded display bytes. Decode to linear BEFORE exposure
    // and tonemap — the GPU path does this in hardware via sRGB texture views; the CPU
    // path must do it explicitly or the mids are crushed (bd-2z0.7.11).
    let mut linear = [
        srgb_to_linear(color.r as f32 / 255.0) * exposure,
        srgb_to_linear(color.g as f32 / 255.0) * exposure,
        srgb_to_linear(color.b as f32 / 255.0) * exposure,
    ];

    match mode {
        RenderTonemapMode::Aces | RenderTonemapMode::Tony => {
            for c in linear.iter_mut() {
                *c = aces_fitted((*c).max(0.0));
            }
        }
        RenderTonemapMode::Agx => {
            for c in linear.iter_mut() {
                let x = (*c).max(0.0);
                let compressed = x / (x + 0.3);
                *c = aces_fitted(compressed);
            }
        }
    }

    ColorVec3 {
        r: linear_to_srgb_byte(linear[0]),
        g: linear_to_srgb_byte(linear[1]),
        b: linear_to_srgb_byte(linear[2]),
    }
}

fn aces_fitted(x: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

fn linear_to_srgb_byte(x: f32) -> u8 {
    let srgb = if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (srgb.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod srgb_ownership_tests {
    use super::*;

    /// Hand-derived golden for the decode -> exposure -> fitted-ACES -> encode chain
    /// (bd-2z0.7.11). Derivation for (120, 200, 120) at exposure 1.0:
    ///   r: 120/255 = 0.470588 -> srgb_to_linear = 0.18782 -> aces_fitted = 0.27985
    ///      -> encode = round(0.56564 * 255) = 144
    ///   g: 200/255 = 0.784314 -> srgb_to_linear = 0.57748 -> aces_fitted = 0.66173
    ///      -> encode = round(0.83332 * 255) = 212
    ///   b equals r.
    #[test]
    fn cpu_tonemap_golden_matches_the_explicit_srgb_chain() {
        let mapped = tonemap_rgb(
            ColorVec3 {
                r: 120,
                g: 200,
                b: 120,
            },
            1.0,
            RenderTonemapMode::Aces,
        );
        assert!((mapped.r as i32 - 144).abs() <= 2, "r: got {}", mapped.r);
        assert!((mapped.g as i32 - 212).abs() <= 2, "g: got {}", mapped.g);
        assert_eq!(mapped.b, mapped.r);
    }

    #[test]
    fn cpu_tonemap_preserves_order_and_black_point() {
        let black = tonemap_rgb(ColorVec3 { r: 0, g: 0, b: 0 }, 1.0, RenderTonemapMode::Aces);
        assert_eq!((black.r, black.g, black.b), (0, 0, 0));

        let mut previous = 0u8;
        for value in [32u8, 64, 96, 128, 160, 192, 224, 255] {
            let mapped = tonemap_rgb(
                ColorVec3 {
                    r: value,
                    g: value,
                    b: value,
                },
                1.0,
                RenderTonemapMode::Aces,
            );
            assert!(
                mapped.r >= previous,
                "tonemap must be monotonic: {value} -> {} after {previous}",
                mapped.r
            );
            assert_eq!(mapped.r, mapped.g, "grayscale channels must stay equal");
            assert_eq!(mapped.g, mapped.b, "grayscale channels must stay equal");
            previous = mapped.r;
        }
    }

    #[test]
    fn srgb_decode_is_the_exact_inverse_of_the_encode() {
        for byte in 0u8..=255 {
            let linear = srgb_to_linear(f32::from(byte) / 255.0);
            let encoded = linear_to_srgb_byte(linear);
            assert_eq!(encoded, byte, "round trip failed at {byte}");
        }
    }
}

impl Render for SimulationView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if !self.world_raster_cleanup_registered {
            cx.on_release_in(window, |this, window, _| {
                let images = match this.world_raster_cache.lock() {
                    Ok(mut cache) => cache.drain(),
                    Err(poisoned) => poisoned.into_inner().drain(),
                };
                for image in images {
                    if let Err(error) = window.drop_image(image) {
                        debug!(?error, "could not evict released world raster image");
                    }
                }
            })
            .detach();
            self.world_raster_cleanup_registered = true;
        }

        self.perf.begin_frame();

        let live_snapshot = self.snapshot();
        let snapshot = self.playback.snapshot_for_render(live_snapshot);

        // Update per-agent brain output history from the current inspector focus
        self.update_brain_history(&snapshot);
        self.validate_rail_selection(&snapshot);
        self.maybe_log_rail_first_show(&snapshot);

        // bd-v9cz resize rule. The latch is transient window geometry; `self.hud` holds
        // user intent and is deliberately NOT written here, so a window dragged briefly
        // narrow does not destroy the layout the user chose. Separate collapse and
        // restore thresholds give hysteresis, so dragging an edge cannot strobe the rail.
        let viewport_width = f32::from(window.bounds().size.width);
        if self.hud_rail_forced_closed {
            if viewport_width >= HUD_RAIL_RESTORE_WIDTH {
                self.hud_rail_forced_closed = false;
            }
        } else if viewport_width < HUD_RAIL_COLLAPSE_WIDTH {
            self.hud_rail_forced_closed = true;
        }
        let resolved = self
            .hud
            .resolve(viewport_width, self.hud_rail_forced_closed);

        let mut content = if self.minimal_canvas_mode {
            // Dedicated window: render only canvas + overlay and skip heavy HUD sections
            div()
                .size_full()
                .relative()
                .flex()
                .flex_col()
                .bg(rgb(0x0f172a))
                .text_color(rgb(0xf8fafc))
                .p_2()
                .child(self.render_canvas(&snapshot, resolved, cx))
        } else {
            div()
                .size_full()
                .relative()
                .flex()
                .flex_col()
                .bg(rgb(0x0f172a))
                .text_color(rgb(0xf8fafc))
                .p_6()
                .gap_4()
                .child(self.render_header(&snapshot, cx))
                .child(self.render_summary(&snapshot))
                .child(self.render_analytics_panel(&snapshot))
                .child({
                    let mut canvas_row = div()
                        .flex()
                        .gap_4()
                        .flex_1()
                        .h_full()
                        .flex_grow(1.0)
                        .child(self.render_history(&snapshot))
                        .child(self.render_canvas(&snapshot, resolved, cx))
                        .child(self.render_inspector(&snapshot, cx));
                    // bd-v9cz: the HUD rail is a FIRST-CLASS SIBLING here, not a child of
                    // render_canvas. Nesting it inside the canvas meant the capture
                    // harness and the real window could disagree about whether chrome
                    // exists, and it fed the resize rule the wrong width — the rule read
                    // the 1280px window while the rail actually lived in the ~592px
                    // canvas container, so it never collapsed when it should have.
                    if let Some(rail) = self.render_hud_rail(&snapshot, resolved) {
                        canvas_row = canvas_row.child(rail);
                    }
                    canvas_row.style().align_items = Some(AlignItems::Stretch);
                    canvas_row
                })
                .when(self.rail_visible, |content| {
                    content.child(self.render_narrative_rail(&snapshot, cx))
                })
                .child(self.render_footer(&snapshot))
        };
        if let Some(focus_handle) = &self.focus_handle {
            content = content.track_focus(focus_handle);
        }
        content = content.on_key_down(cx.listener(|this, event: &KeyDownEvent, _, cx| {
            this.handle_key_down(event, cx);
        }));

        let perf_snapshot = self.perf.end_frame();
        // Respect a pinned snapshot: assigning unconditionally made the capture pin
        // last exactly one frame (bd-c7pg).
        if self.forced_perf.is_none() {
            self.last_perf = perf_snapshot;
        }

        #[cfg(feature = "audio")]
        self.update_audio(&snapshot);

        // The perf readout is docked in the HUD rail (bd-v9cz), so it is no longer
        // appended here as a floating overlay. It previously rendered only on frames
        // where sample_count % 4 == 0; because GPUI rebuilds the element tree every
        // frame that throttled the panel's EXISTENCE rather than its data, strobing it
        // at a quarter of frame rate (bd-rzy3). The rail reads self.last_perf, which is
        // assigned every frame just above, so the panel is stable and the value fresh.

        if self.settings_panel.open {
            content = content.child(self.render_settings_panel(cx));
        }

        // Keep the presentation current while the session-level driver advances
        // scientific time independently of either window's paint cadence.
        window.request_animation_frame();

        content
    }
}

#[derive(Clone)]
struct OutputSparklineState {
    // Up to N series; each is a time-ordered vector of samples in [f32]
    series: Vec<Vec<f32>>,
}

struct OutputHistory {
    capacity: usize,
    series: Vec<VecDeque<f32>>, // per-output history
}

impl OutputHistory {
    fn new(outputs_len: usize, capacity: usize) -> Self {
        let mut series = Vec::with_capacity(outputs_len);
        for _ in 0..outputs_len {
            series.push(VecDeque::with_capacity(capacity));
        }
        Self { capacity, series }
    }

    fn push(&mut self, outputs: &[f32]) {
        if self.series.len() != outputs.len() {
            // Reinitialize to match new output vector size
            *self = OutputHistory::new(outputs.len(), self.capacity);
        }
        for (i, &v) in outputs.iter().enumerate() {
            let q = &mut self.series[i];
            if q.len() == self.capacity {
                q.pop_front();
            }
            q.push_back(v);
        }
    }

    fn as_state(&self, take: usize) -> OutputSparklineState {
        let mut series: Vec<Vec<f32>> = Vec::new();
        for q in &self.series {
            let len = q.len();
            let start = len.saturating_sub(take);
            series.push(q.iter().skip(start).copied().collect());
        }
        OutputSparklineState { series }
    }
}

impl SimulationView {
    fn update_brain_history(&mut self, snapshot: &HudSnapshot) {
        if let Some(detail) = snapshot.inspector.focused.as_ref() {
            let id = detail.agent_id;
            let entry = self
                .brain_history
                .entry(id)
                .or_insert_with(|| OutputHistory::new(detail.outputs.len(), 64));
            entry.push(&detail.outputs);
            // Keep only the focused agent to control memory
            self.brain_history.retain(|k, _| *k == id);
        }
    }

    fn render_output_sparklines_for(&self, agent_id: AgentId) -> Div {
        if let Some(history) = self.brain_history.get(&agent_id) {
            // Pick top 3 series by latest absolute value
            let state_full = history.as_state(64);
            let mut idx_vals: Vec<(usize, f32)> = state_full
                .series
                .iter()
                .enumerate()
                .map(|(i, s)| (i, s.last().copied().unwrap_or(0.0).abs()))
                .collect();
            idx_vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            let top_indices: Vec<usize> = idx_vals.into_iter().take(3).map(|(i, _)| i).collect();
            let mut series: Vec<Vec<f32>> = Vec::new();
            for i in top_indices {
                series.push(state_full.series[i].clone());
            }
            let state = OutputSparklineState { series };

            let spark_canvas = canvas(
                move |_, _, _| state.clone(),
                move |bounds, state, window, _| paint_output_sparklines(bounds, &state, window),
            )
            .w(px(200.0))
            .h(px(56.0));

            return div()
                .flex()
                .flex_col()
                .gap_1()
                .rounded_md()
                .border_1()
                .border_color(rgb(0x1e293b))
                .bg(rgb(0x0f172a))
                .px_3()
                .py_2()
                .child(
                    div()
                        .text_xs()
                        .text_color(rgb(0x94a3b8))
                        .child("Output history"),
                )
                .child(spark_canvas);
        }
        div()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x1e293b))
            .bg(rgb(0x0f172a))
            .px_3()
            .py_2()
            .child(
                div()
                    .text_xs()
                    .text_color(rgb(0x475569))
                    .child("Collecting output history…"),
            )
    }
}

fn paint_output_sparklines(
    bounds: Bounds<Pixels>,
    state: &OutputSparklineState,
    window: &mut Window,
) {
    let origin = bounds.origin;
    let size = bounds.size;
    let width = f32::from(size.width).max(1.0);
    let height = f32::from(size.height).max(1.0);

    // Background
    window.paint_quad(fill(
        bounds,
        Background::from(rgba_from_hex(0x0b1223, 0.92)),
    ));

    // Compute global min/max
    let mut min_v = 0.0_f32;
    let mut max_v = 1.0_f32;
    if !state.series.is_empty() {
        min_v = f32::INFINITY;
        max_v = f32::NEG_INFINITY;
        for s in &state.series {
            for &v in s {
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
        }
        if (max_v - min_v).abs() < 1e-6 {
            max_v = min_v + 1.0;
        }
    }

    let colors = [
        rgba_from_hex(0x22d3ee, 0.95), // cyan
        rgba_from_hex(0xf59e0b, 0.95), // amber
        rgba_from_hex(0xa78bfa, 0.95), // violet
    ];

    // Draw zero baseline if in range
    if min_v < 0.0 && max_v > 0.0 {
        let y0 = (1.0 - (-min_v / (max_v - min_v))) * height;
        let mut baseline = PathBuilder::stroke(px(1.0));
        baseline.move_to(point(px(f32::from(origin.x)), px(f32::from(origin.y) + y0)));
        baseline.line_to(point(
            px(f32::from(origin.x) + width),
            px(f32::from(origin.y) + y0),
        ));
        if let Ok(path) = baseline.build() {
            window.paint_path(path, rgba_from_hex(0x1e293b, 0.9));
        }
    }

    for (si, series) in state.series.iter().enumerate() {
        if series.len() < 2 {
            continue;
        }
        let color = colors
            .get(si)
            .copied()
            .unwrap_or(rgba_from_hex(0x94a3b8, 0.95));
        let step_x = width / (series.len() as f32 - 1.0);
        let mut path = PathBuilder::stroke(px(1.8));
        for (i, &v) in series.iter().enumerate() {
            let x = f32::from(origin.x) + step_x * i as f32;
            let norm = (v - min_v) / (max_v - min_v);
            let y = f32::from(origin.y) + (1.0 - norm) * height;
            if i == 0 {
                path.move_to(point(px(x), px(y)));
            } else {
                path.line_to(point(px(x), px(y)));
            }
        }
        if let Ok(path) = path.build() {
            window.paint_path(path, color);
        }
    }
}

#[derive(Default, Clone)]
struct HudSnapshot {
    tick: u64,
    epoch: u64,
    is_closed: bool,
    world_size: (u32, u32),
    history_capacity: usize,
    agent_count: usize,
    summary: Option<HudMetrics>,
    analytics: Option<HudAnalytics>,
    storage: StorageUiStatus,
    simulation_fault: Option<String>,
    recent_history: Vec<HudHistoryEntry>,
    render_frame: Option<RenderFrame>,
    inspector: InspectorSnapshot,
    controls: ControlsSnapshot,
    perf: PerfSnapshot,
    /// The run's retained narrative events, oldest first (bd-16g.2.4 rail).
    narrative: Vec<NarrativeEventRecord>,
    /// Events the bounded narrative ring has discarded so far.
    narrative_dropped: u64,
    /// The narrative ring's configured capacity.
    narrative_capacity: usize,
}

#[derive(Default, Clone)]
struct StorageUiStatus {
    revision: u64,
    committed_tick: Option<u64>,
    lag: Option<u64>,
    last_error: Option<String>,
    stopped: bool,
}

#[derive(Clone)]
struct HudAnalytics {
    tick: u64,
    carnivores: usize,
    herbivores: usize,
    hybrids: usize,
    carnivore_avg_energy: f64,
    herbivore_avg_energy: f64,
    hybrid_avg_energy: f64,
    age_mean: f64,
    age_max: f64,
    boost_count: usize,
    boost_ratio: f64,
    reproduction_counter_mean: f64,
    temperature_preference_mean: f64,
    temperature_preference_stddev: f64,
    temperature_discomfort_mean: f64,
    temperature_discomfort_stddev: f64,
    food_total: f64,
    food_mean: f64,
    food_stddev: f64,
    food_delta_mean: f64,
    food_delta_mean_abs: f64,
    mutation_primary_mean: f64,
    mutation_primary_stddev: f64,
    mutation_secondary_mean: f64,
    mutation_secondary_stddev: f64,
    behavior_sensor_mean: f64,
    behavior_sensor_entropy: f64,
    behavior_output_mean: f64,
    behavior_output_entropy: f64,
    generation_mean: f64,
    generation_max: f64,
    deaths_combat_carnivore: usize,
    deaths_combat_herbivore: usize,
    deaths_starvation: usize,
    deaths_aging: usize,
    deaths_unknown: usize,
    deaths_total: usize,
    births_total: usize,
    births_hybrid: usize,
    births_hybrid_ratio: f64,
    brain_shares: Vec<BrainShareEntry>,
}

#[derive(Clone)]
struct BrainShareEntry {
    label: String,
    count: usize,
    avg_energy: f64,
}
fn parse_analytics(
    tick: u64,
    agent_count: usize,
    readings: &[MetricReading],
) -> Option<HudAnalytics> {
    if readings.is_empty() {
        return None;
    }

    let mut metrics = HashMap::with_capacity(readings.len());
    for reading in readings {
        metrics.insert(reading.name.clone(), reading.value);
    }

    let value = |key: &str| metrics.get(key).copied();
    let as_count = |key: &str| value(key).unwrap_or(0.0).max(0.0).round() as usize;
    let carnivores = as_count("population.carnivore.count");
    let herbivores = as_count("population.herbivore.count");
    let hybrids = as_count("population.hybrid.count");

    let carnivore_avg_energy = value("population.carnivore.avg_energy").unwrap_or(0.0);
    let herbivore_avg_energy = value("population.herbivore.avg_energy").unwrap_or(0.0);
    let hybrid_avg_energy = value("population.hybrid.avg_energy").unwrap_or(0.0);
    let age_mean = value("population.age.mean").unwrap_or(0.0);
    let age_max = value("population.age.max").unwrap_or(0.0);
    let boost_count = as_count("behavior.boost.count");
    let boost_ratio = value("behavior.boost.ratio").unwrap_or_else(|| {
        if agent_count > 0 {
            boost_count as f64 / agent_count as f64
        } else {
            0.0
        }
    });
    let reproduction_counter_mean = value("reproduction.counter.mean").unwrap_or(0.0);
    let temperature_preference_mean = value("temperature.preference.mean").unwrap_or(0.0);
    let temperature_preference_stddev = value("temperature.preference.stddev").unwrap_or(0.0);

    let food_total = value("food.total").unwrap_or(0.0);
    let food_mean = value("food.mean").unwrap_or(0.0);
    let food_stddev = value("food.stddev").unwrap_or(0.0);
    let food_delta_mean = value("food_delta.mean").unwrap_or(0.0);
    let food_delta_mean_abs = value("food_delta.mean_abs").unwrap_or(0.0);
    let temperature_discomfort_mean = value("temperature.discomfort.mean").unwrap_or(0.0);
    let temperature_discomfort_stddev = value("temperature.discomfort.stddev").unwrap_or(0.0);
    let generation_mean = value("population.generation.mean").unwrap_or(0.0);
    let generation_max = value("population.generation.max").unwrap_or(0.0);

    let mutation_primary_mean = value("mutation.primary.mean").unwrap_or(0.0);
    let mutation_primary_stddev = value("mutation.primary.stddev").unwrap_or(0.0);
    let mutation_secondary_mean = value("mutation.secondary.mean").unwrap_or(0.0);
    let mutation_secondary_stddev = value("mutation.secondary.stddev").unwrap_or(0.0);

    let behavior_sensor_mean = value("behavior.sensors.mean").unwrap_or(0.0);
    let behavior_sensor_entropy = value("behavior.sensors.entropy").unwrap_or(0.0);
    let behavior_output_mean = value("behavior.outputs.mean").unwrap_or(0.0);
    let behavior_output_entropy = value("behavior.outputs.entropy").unwrap_or(0.0);

    let mut brain_map: HashMap<String, BrainShareEntry> = HashMap::new();
    for (name, &metric_value) in &metrics {
        if let Some(rest) = name.strip_prefix("brain.population.") {
            if let Some(label) = rest.strip_suffix(".count") {
                let entry = brain_map
                    .entry(label.to_string())
                    .or_insert(BrainShareEntry {
                        label: label.to_string(),
                        count: 0,
                        avg_energy: 0.0,
                    });
                entry.count = metric_value.max(0.0).round() as usize;
                continue;
            }
            if let Some(label) = rest.strip_suffix(".avg_energy") {
                let entry = brain_map
                    .entry(label.to_string())
                    .or_insert(BrainShareEntry {
                        label: label.to_string(),
                        count: 0,
                        avg_energy: 0.0,
                    });
                entry.avg_energy = metric_value;
            }
        }
    }

    let mut brain_shares: Vec<BrainShareEntry> = brain_map.into_values().collect();
    brain_shares.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.label.cmp(&b.label)));

    let deaths_combat_carnivore = as_count("mortality.combat_carnivore.count");
    let deaths_combat_herbivore = as_count("mortality.combat_herbivore.count");
    let deaths_starvation = as_count("mortality.starvation.count");
    let deaths_aging = as_count("mortality.aging.count");
    let deaths_unknown = as_count("mortality.unknown.count");
    let deaths_total = value("mortality.total.count")
        .map(|v| v.max(0.0).round() as usize)
        .unwrap_or(
            deaths_combat_carnivore
                + deaths_combat_herbivore
                + deaths_starvation
                + deaths_aging
                + deaths_unknown,
        );
    let births_total = as_count("births.total.count");
    let births_hybrid = as_count("births.hybrid.count");
    let births_hybrid_ratio = value("births.hybrid.ratio").unwrap_or_else(|| {
        if births_total > 0 {
            births_hybrid as f64 / births_total as f64
        } else {
            0.0
        }
    });

    Some(HudAnalytics {
        tick,
        carnivores,
        herbivores,
        hybrids,
        carnivore_avg_energy,
        herbivore_avg_energy,
        hybrid_avg_energy,
        age_mean,
        age_max,
        boost_count,
        boost_ratio,
        reproduction_counter_mean,
        temperature_preference_mean,
        temperature_preference_stddev,
        temperature_discomfort_mean,
        temperature_discomfort_stddev,
        food_total,
        food_mean,
        food_stddev,
        food_delta_mean,
        food_delta_mean_abs,
        mutation_primary_mean,
        mutation_primary_stddev,
        mutation_secondary_mean,
        mutation_secondary_stddev,
        behavior_sensor_mean,
        behavior_sensor_entropy,
        behavior_output_mean,
        behavior_output_entropy,
        generation_mean,
        generation_max,
        deaths_combat_carnivore,
        deaths_combat_herbivore,
        deaths_starvation,
        deaths_aging,
        deaths_unknown,
        deaths_total,
        births_total,
        births_hybrid,
        births_hybrid_ratio,
        brain_shares,
    })
}

#[derive(Clone)]
struct VectorHudState {
    population_ratio: f32,
    energy_ratio: f32,
    births_ratio: f32,
    deaths_ratio: f32,
    tick_phase: f32,
    mean_speed: f32,
    vector_magnitude: f32,
    max_speed: f32,
    heading_rad: f32,
}

impl VectorHudState {
    fn from_snapshot(snapshot: &HudSnapshot) -> Option<Self> {
        let metrics = snapshot.summary.as_ref()?;

        let max_agents = snapshot
            .recent_history
            .iter()
            .map(|entry| entry.agent_count)
            .chain(std::iter::once(metrics.agent_count))
            .max()
            .unwrap_or(metrics.agent_count)
            .max(1);

        let max_births = snapshot
            .recent_history
            .iter()
            .map(|entry| entry.births)
            .chain(std::iter::once(metrics.births))
            .max()
            .unwrap_or(metrics.births)
            .max(1);

        let max_deaths = snapshot
            .recent_history
            .iter()
            .map(|entry| entry.deaths)
            .chain(std::iter::once(metrics.deaths))
            .max()
            .unwrap_or(metrics.deaths)
            .max(1);

        let mut energy_max = metrics.average_energy.max(0.0);
        for entry in &snapshot.recent_history {
            energy_max = energy_max.max(entry.average_energy);
        }
        if energy_max <= f32::EPSILON {
            energy_max = 1.0;
        }

        let (mean_speed, vector_magnitude, max_speed, heading_rad) = snapshot
            .render_frame
            .as_ref()
            .map(|frame| {
                if frame.agents.is_empty() {
                    return (0.0, 0.0, 1.0, 0.0);
                }

                let mut sum_vx: f32 = 0.0;
                let mut sum_vy: f32 = 0.0;
                let mut sum_speed: f32 = 0.0;
                let mut max_speed: f32 = 0.0;

                for agent in &frame.agents {
                    let vel = agent.velocity;
                    let speed = (vel.vx * vel.vx + vel.vy * vel.vy).sqrt();
                    sum_vx += vel.vx;
                    sum_vy += vel.vy;
                    sum_speed += speed;
                    max_speed = max_speed.max(speed);
                }

                let count = frame.agents.len() as f32;
                let safe_count = if count <= f32::EPSILON { 1.0 } else { count };
                let avg_vx = sum_vx / safe_count;
                let avg_vy = sum_vy / safe_count;
                let mean_speed = sum_speed / safe_count;
                let vector_magnitude = (avg_vx * avg_vx + avg_vy * avg_vy).sqrt();
                let heading_rad = if vector_magnitude > f32::EPSILON {
                    avg_vy.atan2(avg_vx)
                } else {
                    0.0
                };
                let max_speed_final = max_speed.max(mean_speed).max(1e-3);

                (mean_speed, vector_magnitude, max_speed_final, heading_rad)
            })
            .unwrap_or((0.0, 0.0, 1.0, 0.0));

        Some(Self {
            population_ratio: metrics.agent_count as f32 / max_agents as f32,
            energy_ratio: (metrics.average_energy / energy_max).clamp(0.0, 1.0),
            births_ratio: metrics.births as f32 / max_births as f32,
            deaths_ratio: metrics.deaths as f32 / max_deaths as f32,
            tick_phase: (snapshot.tick % 960) as f32 / 960.0,
            mean_speed,
            vector_magnitude,
            max_speed,
            heading_rad,
        })
    }
}

#[derive(Clone)]
struct HudMetrics {
    tick: u64,
    agent_count: usize,
    births: usize,
    deaths: usize,
    total_energy: f32,
    average_energy: f32,
    average_health: f32,
}

impl HudMetrics {
    fn net_growth(&self) -> isize {
        self.births as isize - self.deaths as isize
    }
}

impl From<&TickSummary> for HudMetrics {
    fn from(summary: &TickSummary) -> Self {
        Self {
            tick: summary.tick.0,
            agent_count: summary.agent_count,
            births: summary.births,
            deaths: summary.deaths,
            total_energy: summary.total_energy,
            average_energy: summary.average_energy,
            average_health: summary.average_health,
        }
    }
}

#[derive(Clone)]
struct SparklineSeries {
    normalized: Vec<f32>,
    trend: f32,
}

#[derive(Clone)]
struct SparklineState {
    values: Vec<f32>,
    accent: Rgba,
    trend: f32,
}

#[derive(Clone)]
struct MetricBadgeState {
    accent: Rgba,
}

#[derive(Clone, Copy)]
struct HeaderBadgeState {
    phase: f32,
    palette: ColorPaletteMode,
}

#[derive(Clone, Copy)]
struct DebugOverlayState {
    enabled: bool,
    show_velocity: bool,
    show_sense_radius: bool,
}

impl Default for DebugOverlayState {
    fn default() -> Self {
        Self {
            enabled: false,
            show_velocity: true,
            show_sense_radius: true,
        }
    }
}
// PresetKind comes from core now
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum FollowMode {
    #[default]
    Off,
    Selected,
    Oldest,
}

impl FollowMode {
    fn label(self) -> &'static str {
        match self {
            FollowMode::Off => "Follow off",
            FollowMode::Selected => "Follow selected",
            FollowMode::Oldest => "Follow oldest",
        }
    }
}

#[derive(Clone)]
struct SimulationControls {
    draw_agents: bool,
    draw_food: bool,
    follow_mode: FollowMode,
    agent_outline: bool,
}

impl Default for SimulationControls {
    fn default() -> Self {
        Self {
            draw_agents: true,
            draw_food: true,
            follow_mode: FollowMode::Off,
            agent_outline: false,
        }
    }
}

impl SimulationControls {
    fn snapshot(&self, paused: bool, speed_multiplier: f32) -> ControlsSnapshot {
        ControlsSnapshot {
            paused,
            draw_agents: self.draw_agents,
            draw_food: self.draw_food,
            speed_multiplier,
            follow_mode: self.follow_mode,
            agent_outline: self.agent_outline,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ControlsSnapshot {
    paused: bool,
    draw_agents: bool,
    draw_food: bool,
    speed_multiplier: f32,
    follow_mode: FollowMode,
    agent_outline: bool,
}

fn sparkline_from_history<F>(history: &[HudHistoryEntry], map: F) -> Option<SparklineSeries>
where
    F: Fn(&HudHistoryEntry) -> f32,
{
    if history.len() < 2 {
        return None;
    }
    let raw: Vec<f32> = history.iter().map(map).collect();
    if raw.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let first = raw.first().copied()?;
    let last = raw.last().copied()?;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in &raw {
        min = min.min(*value);
        max = max.max(*value);
    }
    let span = (max - min).abs().max(1e-5);
    let normalized: Vec<f32> = if span <= 1e-5 {
        vec![0.5; raw.len()]
    } else {
        raw.iter()
            .map(|v| ((v - min) / span).clamp(0.0, 1.0))
            .collect()
    };

    Some(SparklineSeries {
        normalized,
        trend: last - first,
    })
}

#[derive(Clone)]
struct HudHistoryEntry {
    tick: u64,
    agent_count: usize,
    births: usize,
    deaths: usize,
    average_energy: f32,
    average_health: f32,
}

impl HudHistoryEntry {
    fn net_growth(&self) -> isize {
        self.births as isize - self.deaths as isize
    }
}

impl From<TickSummary> for HudHistoryEntry {
    fn from(summary: TickSummary) -> Self {
        Self {
            tick: summary.tick.0,
            agent_count: summary.agent_count,
            births: summary.births,
            deaths: summary.deaths,
            average_energy: summary.average_energy,
            average_health: summary.average_health,
        }
    }
}

#[derive(Clone)]
struct InspectorState {
    focused_agent: Option<AgentId>,
    hovered_agent: Option<AgentId>,
    brush_enabled: bool,
    brush_radius: f32,
    probe_enabled: bool,
    persistence_last_enabled: u32,
}

impl Default for InspectorState {
    fn default() -> Self {
        Self {
            focused_agent: None,
            hovered_agent: None,
            brush_enabled: false,
            brush_radius: 48.0,
            probe_enabled: false,
            persistence_last_enabled: 60,
        }
    }
}

#[derive(Default, Clone)]
struct InspectorSnapshot {
    focused: Option<AgentInspectorDetails>,
    selected: Vec<AgentListEntry>,
    hovered: Option<AgentListEntry>,
    focus_id: Option<AgentId>,
    total_agents: usize,
    brush_enabled: bool,
    brush_radius: f32,
    probe_enabled: bool,
    persistence_enabled: bool,
    persistence_interval: u32,
    persistence_cached_interval: u32,
}

/// HUD chrome system (bd-f4x0), derived from the bd-9pqz art direction.
///
/// Three rules, and they are the whole brief:
///
/// 1. NO INVENTED COLOUR. Every surface here traces to `visual.rs`. The HUD used a
///    dozen unrelated hand-picked hexes (`0x0b1120`, `0x111b2b`, `0x0a1629`, three
///    chart colours, four text greys) that matched neither each other nor the world.
/// 2. SURFACES ARE THE SUBSTRATE, LIFTED. Chrome sits on the same near-black
///    blue-violet as the world, one step lighter, so panels read as the same material
///    rather than as pasted-on cards.
/// 3. RESTRAINT. The world is the subject. Chrome carries exactly one accent — the
///    herbivore cyan — and otherwise moves only in value.
mod chrome {
    use super::rgb_from_triplet;
    use gpui::Hsla;
    use scriptbots_core::visual::{
        BIOLUMINESCENT_DARK_FIELD_V1 as STYLE, CARNIVORE_RGB, FOOD_MID_RGB, HERBIVORE_RGB,
    };

    /// Lift a substrate value toward legibility without leaving the ramp: value only,
    /// hue preserved, so chrome can never drift off the art direction.
    fn lifted(base: [f32; 3], lift: f32) -> Hsla {
        rgb_from_triplet([
            (base[0] + lift).clamp(0.0, 1.0),
            (base[1] + lift).clamp(0.0, 1.0),
            (base[2] + lift).clamp(0.0, 1.0),
        ])
        .into()
    }

    /// Panel fill. One surface for every docked panel — the old code had three.
    pub fn surface() -> Hsla {
        lifted(STYLE.substrate.base_srgb, 0.010)
    }

    /// Raised surface for nested content, one value step above [`surface`].
    pub fn surface_raised() -> Hsla {
        lifted(STYLE.substrate.depth_violet_srgb, 0.014)
    }

    /// Hairline border. Deliberately close to the surface: a panel edge should be
    /// felt, not read.
    pub fn border() -> Hsla {
        lifted(STYLE.substrate.distant_haze_srgb, 0.020)
    }

    /// Type scale, expressed as colour weight rather than size — a terminal-like HUD
    /// has one usable size, so hierarchy has to come from value.
    pub fn text_primary() -> Hsla {
        lifted(STYLE.substrate.distant_haze_srgb, 0.760)
    }
    pub fn text_secondary() -> Hsla {
        lifted(STYLE.substrate.distant_haze_srgb, 0.520)
    }
    pub fn text_muted() -> Hsla {
        lifted(STYLE.substrate.distant_haze_srgb, 0.300)
    }

    /// The single accent. One, so it still means something where it appears.
    pub fn accent() -> Hsla {
        rgb_from_triplet(HERBIVORE_RGB).into()
    }

    /// Chart series, on the same deliberate ramp as the world's agents and food
    /// instead of stock blue/green/red.
    pub fn series_population() -> Hsla {
        rgb_from_triplet(HERBIVORE_RGB).into()
    }
    pub fn series_births() -> Hsla {
        rgb_from_triplet(FOOD_MID_RGB).into()
    }
    pub fn series_deaths() -> Hsla {
        rgb_from_triplet(CARNIVORE_RGB).into()
    }
}

/// Minimum world viewport, in logical pixels. The world has absolute layout
/// priority: chrome collapses to make room, never the other way round.
const WORLD_MIN_WIDTH: f32 = 640.0;

/// Docked rail width.
const HUD_RAIL_WIDTH: f32 = 320.0;

/// Below this the rail is force-collapsed; above [`HUD_RAIL_RESTORE_WIDTH`] it comes
/// back. The gap between them is hysteresis — with a single threshold, dragging a
/// window edge across it makes the whole rail strobe.
const HUD_RAIL_COLLAPSE_WIDTH: f32 = WORLD_MIN_WIDTH + HUD_RAIL_WIDTH;
const HUD_RAIL_RESTORE_WIDTH: f32 = HUD_RAIL_COLLAPSE_WIDTH + 80.0;

/// Where HUD chrome is permitted to exist (bd-v9cz layout policy).
///
/// These fields are USER INTENT. They are read to decide what to draw, but the
/// resize rule must never write to them: a window briefly dragged narrow would
/// otherwise silently destroy the layout the user chose. Forced collapse is
/// derived per-frame in [`HudLayout::resolve`] and thrown away.
#[derive(Clone, Copy)]
struct HudLayout {
    stats_open: bool,
    history_open: bool,
    perf_open: bool,
}

impl Default for HudLayout {
    fn default() -> Self {
        // First run, no saved config: stats and history docked and visible, perf
        // collapsed because it is a diagnostic rather than a first-run readout.
        Self {
            stats_open: true,
            history_open: true,
            perf_open: false,
        }
    }
}

/// What the current window size actually permits, given the user's intent.
#[derive(Clone, Copy)]
struct ResolvedHudLayout {
    show_rail: bool,
    stats_open: bool,
    history_open: bool,
    perf_open: bool,
}

impl HudLayout {
    /// Fold intent together with the available width. Panels drop in a fixed
    /// order — perf, then history, then the whole rail — so a given window size
    /// always yields the same layout.
    fn resolve(self, available_width: f32, rail_forced_closed: bool) -> ResolvedHudLayout {
        let room_for_rail = !rail_forced_closed && available_width >= HUD_RAIL_COLLAPSE_WIDTH;
        let any_panel_open = self.stats_open || self.history_open || self.perf_open;
        ResolvedHudLayout {
            show_rail: room_for_rail && any_panel_open,
            stats_open: self.stats_open,
            history_open: self.history_open,
            perf_open: self.perf_open,
        }
    }
}

#[cfg(test)]
mod hud_rail_layout_tests {
    use super::*;

    /// bd-rzy3: the perf readout must be MOUNTED on every frame it is open.
    ///
    /// GPUI rebuilds the element tree on every render, so gating the panel's
    /// `.child()` call on a frame counter throttles its EXISTENCE, not its data.
    /// The panel used to be appended only when `sample_count % 4 == 0`, i.e. it
    /// was absent three frames in four and strobed at a quarter of frame rate.
    /// Update cadence belongs to the VALUE — `self.last_perf` — never to the
    /// mount. This inspects the `render_hud_rail` body because the assembled
    /// `Div` exposes no child list to assert against.
    #[test]
    fn perf_panel_mount_is_not_gated_on_a_frame_counter() {
        let rail = render_hud_rail_body();
        assert!(
            rail.contains("self.render_perf_overlay(self.last_perf)"),
            "the rail must render the retained perf sample, so the panel is stable \
             frame to frame while its value refreshes independently"
        );
        for forbidden in ["sample_count", "is_multiple_of", "% 4", "frame_index"] {
            assert!(
                !rail.contains(forbidden),
                "render_hud_rail must not gate panel mounting on a frame counter \
                 ({forbidden}); GPUI rebuilds the tree every frame, so that hides \
                 the panel instead of throttling its data"
            );
        }
    }

    /// Every rail panel mounts purely from resolved layout state. If a panel's
    /// `.child()` call ever grows a second condition, the mount can once again
    /// depend on something other than what the user asked for.
    #[test]
    fn rail_panels_mount_only_on_resolved_layout_state() {
        let rail = render_hud_rail_body();
        for (panel, gate) in [
            ("render_overlay", "if resolved.stats_open {"),
            ("render_history_chart", "if resolved.history_open {"),
            ("render_perf_overlay", "if resolved.perf_open {"),
        ] {
            assert!(
                rail.contains(gate),
                "{panel} must mount under exactly `{gate}` so visibility follows \
                 resolved layout state alone"
            );
        }
    }

    /// The resolved layout is a pure fold of intent and width, so an open panel
    /// stays open across successive frames when neither input changes. The bead
    /// symptom was a panel blinking while the user changed nothing.
    #[test]
    fn resolved_panel_visibility_is_stable_across_frames() {
        let layout = HudLayout {
            stats_open: true,
            history_open: true,
            perf_open: true,
        };
        for frame in 0..8 {
            let resolved = layout.resolve(HUD_RAIL_COLLAPSE_WIDTH, false);
            assert!(
                resolved.show_rail,
                "rail must resolve visible on every frame; frame {frame} disagreed"
            );
            assert!(
                resolved.perf_open,
                "perf panel must resolve visible on every frame; frame {frame} disagreed"
            );
        }
    }

    /// Closed intent stays closed, and a rail with nothing open does not reserve
    /// width. Guards the other direction of the same fold.
    #[test]
    fn resolve_honours_closed_intent_and_narrow_windows() {
        let perf_closed = HudLayout {
            stats_open: true,
            history_open: false,
            perf_open: false,
        };
        let resolved = perf_closed.resolve(HUD_RAIL_COLLAPSE_WIDTH, false);
        assert!(
            resolved.show_rail,
            "an open stats panel still needs the rail"
        );
        assert!(!resolved.perf_open, "closed perf intent must stay closed");

        let all_closed = HudLayout {
            stats_open: false,
            history_open: false,
            perf_open: false,
        };
        assert!(
            !all_closed.resolve(HUD_RAIL_COLLAPSE_WIDTH, false).show_rail,
            "a rail with no open panel must not reserve width"
        );

        let open = HudLayout {
            stats_open: true,
            history_open: true,
            perf_open: true,
        };
        assert!(
            !open.resolve(HUD_RAIL_COLLAPSE_WIDTH - 1.0, false).show_rail,
            "below the collapse width the rail yields the space to the world"
        );
        assert!(
            !open.resolve(HUD_RAIL_COLLAPSE_WIDTH, true).show_rail,
            "a forced-closed rail stays closed regardless of available width"
        );
    }

    /// Source of `render_hud_rail`, from its signature to the next method in the
    /// impl block. Scoped to the body so unrelated code — including this module —
    /// can never satisfy or trip the guards above.
    fn render_hud_rail_body() -> &'static str {
        let after_signature = include_str!("lib.rs")
            .split_once("fn render_hud_rail(")
            .expect("render_hud_rail definition")
            .1;
        after_signature
            .split_once("\n    fn ")
            .expect("method following render_hud_rail")
            .0
    }
}

/// Settings panel state for configuration management
#[derive(Clone)]
struct SettingsPanelState {
    open: bool,
    /// Search query for filtering parameters (future feature)
    #[allow(dead_code)]
    search_query: String,
    /// Currently active/focused category (future feature for single-category view)
    #[allow(dead_code)]
    active_category: Option<ConfigCategory>,
    /// List of collapsed categories (hidden parameters)
    collapsed_categories: Vec<ConfigCategory>,
    /// List of parameter names that have been modified (future feature for change tracking)
    #[allow(dead_code)]
    modified_params: Vec<String>,
    /// Name of current preset configuration (future feature for save/load)
    #[allow(dead_code)]
    preset_name: String,
    /// Vertical scroll offset for categories content (in pixels)
    scroll_offset: f32,
    /// Cached total content height for scroll bounds calculation
    content_height: f32,
    /// Cached viewport height for scroll bounds calculation
    viewport_height: f32,
}

impl Default for SettingsPanelState {
    fn default() -> Self {
        Self {
            open: false,
            search_query: String::new(),
            active_category: None,
            collapsed_categories: Vec::new(),
            modified_params: Vec::new(),
            preset_name: "Default".to_string(),
            scroll_offset: 0.0,
            content_height: 0.0,
            // CRITICAL: Must use conservative (small) value to prevent blocking content access
            // Default window: 720px - chrome (132px) = 588px actual viewport
            // Use 400px to ensure scrolling works even on small windows (600px)
            // Trade-off: Shows blank space on large windows vs. blocking content (blank space is acceptable)
            viewport_height: 400.0,
        }
    }
}

impl SettingsPanelState {
    /// Calculate maximum scroll offset based on content and viewport heights
    fn max_scroll_offset(&self) -> f32 {
        (self.content_height - self.viewport_height).max(0.0)
    }

    /// Clamp scroll offset to valid bounds
    fn clamp_scroll(&mut self) {
        self.scroll_offset = self.scroll_offset.clamp(0.0, self.max_scroll_offset());
    }

    /// Calculate accurate content height based on actual parameter counts
    /// Uses precise measurements from rendered categories
    fn estimate_content_height(&self, _total_categories: usize) -> f32 {
        let mut total_height = 0.0;

        for category in ConfigCategory::all() {
            let is_collapsed = self.collapsed_categories.contains(&category);

            if is_collapsed {
                // Collapsed: header (70px) + gap (16px) = 86px
                total_height += 86.0;
            } else {
                // Expanded: header (70px) + params container + gap (16px)
                // Params container: padding (32px) + params + gaps between params
                // Each param: ~44px + 12px gap = 56px per param
                let param_count = category.parameter_count();
                let params_height = 32.0 + (param_count as f32 * 56.0);
                total_height += 70.0 + params_height + 16.0;
            }
        }

        // Add top/bottom padding for categories container (py_4 = 16px each side = 32px total)
        total_height + 32.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ConfigCategory {
    World,
    Food,
    Agent,
    Metabolism,
    Temperature,
    Reproduction,
    Aging,
    Combat,
    Carcass,
    Topography,
    Population,
    Persistence,
}
impl ConfigCategory {
    fn label(self) -> &'static str {
        match self {
            ConfigCategory::World => "World",
            ConfigCategory::Food => "Food Dynamics",
            ConfigCategory::Agent => "Agent Behavior",
            ConfigCategory::Metabolism => "Metabolism & Energy",
            ConfigCategory::Temperature => "Temperature",
            ConfigCategory::Reproduction => "Reproduction",
            ConfigCategory::Aging => "Aging",
            ConfigCategory::Combat => "Combat & Spikes",
            ConfigCategory::Carcass => "Carcass Sharing",
            ConfigCategory::Topography => "Topography",
            ConfigCategory::Population => "Population Control",
            ConfigCategory::Persistence => "Data Persistence",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            ConfigCategory::World => "🌍",
            ConfigCategory::Food => "🌾",
            ConfigCategory::Agent => "🤖",
            ConfigCategory::Metabolism => "⚡",
            ConfigCategory::Temperature => "🌡️",
            ConfigCategory::Reproduction => "🧬",
            ConfigCategory::Aging => "⏳",
            ConfigCategory::Combat => "⚔️",
            ConfigCategory::Carcass => "🦴",
            ConfigCategory::Topography => "⛰️",
            ConfigCategory::Population => "👥",
            ConfigCategory::Persistence => "💾",
        }
    }

    fn description(self) -> &'static str {
        match self {
            ConfigCategory::World => "World dimensions and grid configuration",
            ConfigCategory::Food => "Food spawning, growth, decay, and diffusion",
            ConfigCategory::Agent => "Agent movement, sensing, and base behavior",
            ConfigCategory::Metabolism => "Energy consumption and metabolism mechanics",
            ConfigCategory::Temperature => "Environmental temperature and agent comfort",
            ConfigCategory::Reproduction => "Reproduction mechanics, mutation, and crossover",
            ConfigCategory::Aging => "Age-related health decay and penalties",
            ConfigCategory::Combat => "Spike damage, energy costs, and combat mechanics",
            ConfigCategory::Carcass => "Death rewards and resource distribution",
            ConfigCategory::Topography => "Terrain elevation effects on movement",
            ConfigCategory::Population => "Population maintenance and seeding",
            ConfigCategory::Persistence => "Database storage and analytics",
        }
    }

    fn all() -> Vec<Self> {
        vec![
            ConfigCategory::World,
            ConfigCategory::Food,
            ConfigCategory::Agent,
            ConfigCategory::Metabolism,
            ConfigCategory::Temperature,
            ConfigCategory::Reproduction,
            ConfigCategory::Aging,
            ConfigCategory::Combat,
            ConfigCategory::Carcass,
            ConfigCategory::Topography,
            ConfigCategory::Population,
            ConfigCategory::Persistence,
        ]
    }

    /// Returns the exact number of parameters displayed in this category
    fn parameter_count(self) -> usize {
        match self {
            ConfigCategory::World => 5, // ACTUAL: width, height, food_cell_size, initial_food, rng_seed
            ConfigCategory::Food => 11, // ACTUAL: respawn_interval, respawn_amount, max, growth_rate, decay_rate, diffusion_rate, intake_rate, sharing_radius, sharing_rate, transfer_rate, sharing_distance
            ConfigCategory::Agent => 5, // ACTUAL: bot_speed, bot_radius, boost_multiplier, sense_radius, carnivore_threshold
            ConfigCategory::Metabolism => 5, // ACTUAL: drain, movement_drain, ramp_floor, ramp_rate, boost_penalty
            ConfigCategory::Temperature => 4, // ACTUAL: discomfort_rate, comfort_band, gradient_exponent, discomfort_exponent
            ConfigCategory::Reproduction => 14, // ACTUAL: energy_threshold, energy_cost, cooldown, herbivore_rate, carnivore_rate, child_energy, spawn_jitter, spawn_back_distance, color_jitter, mutation_scale, partner_chance, gene_log_capacity, meta_mutation_chance, meta_mutation_scale
            ConfigCategory::Aging => 4, // ACTUAL: decay_start, decay_rate, decay_max, energy_penalty
            ConfigCategory::Combat => 8, // ACTUAL: spike_radius, spike_damage, spike_energy_cost, min_length, alignment_cosine, speed_bonus, length_bonus, growth_rate
            ConfigCategory::Carcass => 7, // ACTUAL: distribution_radius, health_reward, reproduction_reward, neighbor_exponent, maturity_age, energy_share, indicator_scale
            ConfigCategory::Topography => 3, // ACTUAL: enabled, speed_gain, energy_penalty
            ConfigCategory::Population => 4, // ACTUAL: minimum, spawn_interval, spawn_count, crossover_chance
            ConfigCategory::Persistence => 2, // ACTUAL: interval, enabled (2 params visible in render)
        }
    }
}

#[derive(Clone)]
struct SelectionEvent {
    tick: u64,
    kind: SelectionEventKind,
    total_selected: usize,
    sample_ids: Vec<AgentId>,
}

#[derive(Clone, Copy)]
enum SelectionEventKind {
    Clear,
    SelectAll,
    Click,
    Focus,
}

impl SelectionEventKind {
    fn label(&self) -> &'static str {
        match self {
            SelectionEventKind::Clear => "Cleared",
            SelectionEventKind::SelectAll => "Selected all",
            SelectionEventKind::Click => "Selection changed",
            SelectionEventKind::Focus => "Focus updated",
        }
    }
}
impl InspectorSnapshot {
    fn from_world(
        world: &WorldState,
        inspector: &InspectorState,
        selection_projection: Option<&[AgentId]>,
        cached_brain: Option<&BrainInspectorCapture>,
        brain_request: Option<(BrainInspectionClientId, BrainInspectionRevision)>,
    ) -> (Self, Option<BrainInspectorCapture>, bool) {
        let mut snapshot = InspectorSnapshot {
            total_agents: world.agent_count(),
            brush_enabled: inspector.brush_enabled,
            brush_radius: inspector.brush_radius,
            probe_enabled: inspector.probe_enabled,
            persistence_cached_interval: inspector.persistence_last_enabled,
            ..InspectorSnapshot::default()
        };

        let config = world.config();
        snapshot.persistence_interval = config.persistence_interval;
        snapshot.persistence_enabled = config.persistence_interval > 0;
        if !snapshot.persistence_enabled && snapshot.persistence_interval > 0 {
            snapshot.persistence_cached_interval = snapshot.persistence_interval.max(1);
        }

        let arena = world.agents();
        let runtime = world.runtime();
        let columns = arena.columns();

        let mut selected = Vec::new();
        let mut hovered: Option<AgentListEntry> = None;

        for (row, agent_id) in arena.iter_handles().enumerate() {
            if let Some(agent_runtime) = runtime.get(agent_id) {
                let entry = AgentListEntry::from_world(row, agent_id, agent_runtime, columns);
                let is_selected = selection_projection.map_or_else(
                    || matches!(agent_runtime.selection, SelectionState::Selected),
                    |projected| projected.contains(&agent_id),
                );
                if is_selected {
                    selected.push(entry);
                } else if inspector.hovered_agent.is_some_and(|id| id == agent_id) {
                    hovered = Some(entry);
                }
            }
        }

        let mut focus_candidate = inspector.focused_agent.filter(|id| arena.contains(*id));

        if focus_candidate.is_none() {
            focus_candidate = selected.first().map(|entry| entry.agent_id);
        }
        if focus_candidate.is_none() {
            focus_candidate = hovered.as_ref().map(|entry| entry.agent_id);
        }

        let focus_id = focus_candidate;

        let mut request_issued = false;
        let brain_capture = focus_id.and_then(|agent_id| {
            let agent_uid = world.agent_uid(agent_id)?;
            let (client_id, revision) = brain_request?;
            if let Some(cached) = cached_brain
                && cached.agent_uid == agent_uid
                && cached.source_tick == world.tick().0
            {
                return Some(cached.clone());
            }

            request_issued = true;
            let response = match world.inspect_brains(&BrainInspectionRequest::single(
                client_id, revision, agent_uid,
            )) {
                Ok(response) => response,
                Err(error) => {
                    warn!(%error, agent_uid = agent_uid.get(), "brain inspection request refused");
                    return Some(BrainInspectorCapture {
                        agent_uid,
                        source_tick: world.tick().0,
                        request_revision: revision.get(),
                        retained_payload_bytes: 0,
                        status: BrainInspectorCaptureStatus::Refused(error.to_string()),
                    });
                }
            };
            let source_tick = response.source_tick.0;
            let request_revision = response.request_revision.get();
            let status = match response.telemetry.into_iter().next() {
                Some(SelectedBrainTelemetryOutcome::Ready { telemetry }) => {
                    BrainInspectorCaptureStatus::Ready(telemetry.inspection.activations)
                }
                Some(SelectedBrainTelemetryOutcome::Unavailable { reason, .. }) => {
                    BrainInspectorCaptureStatus::Unavailable(reason)
                }
                None => BrainInspectorCaptureStatus::Refused(
                    "inspection returned no outcome for the requested agent".to_owned(),
                ),
            };
            let retained_payload_bytes = match &status {
                BrainInspectorCaptureStatus::Ready(_) => response.build.retained_payload_bytes,
                BrainInspectorCaptureStatus::Unavailable(_)
                | BrainInspectorCaptureStatus::Refused(_) => 0,
            };
            Some(BrainInspectorCapture {
                agent_uid,
                source_tick,
                request_revision,
                retained_payload_bytes,
                status,
            })
        });
        let focused = focus_id.and_then(|agent_id| {
            AgentInspectorDetails::from_world(world, agent_id, brain_capture.clone())
        });

        for entry in &mut selected {
            entry.is_focused = Some(entry.agent_id) == focus_id;
        }
        if let Some(entry) = hovered.as_mut() {
            entry.is_focused = Some(entry.agent_id) == focus_id;
        }

        snapshot.focused = focused;
        snapshot.selected = selected;
        snapshot.hovered = hovered;
        snapshot.focus_id = focus_id;
        (snapshot, brain_capture, request_issued)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum ColorPaletteMode {
    #[default]
    Natural,
    Deuteranopia,
    Protanopia,
    Tritanopia,
    HighContrast,
}
impl ColorPaletteMode {
    const ALL: [ColorPaletteMode; 5] = [
        ColorPaletteMode::Natural,
        ColorPaletteMode::Deuteranopia,
        ColorPaletteMode::Protanopia,
        ColorPaletteMode::Tritanopia,
        ColorPaletteMode::HighContrast,
    ];

    fn next(self) -> Self {
        match self {
            ColorPaletteMode::Natural => ColorPaletteMode::Deuteranopia,
            ColorPaletteMode::Deuteranopia => ColorPaletteMode::Protanopia,
            ColorPaletteMode::Protanopia => ColorPaletteMode::Tritanopia,
            ColorPaletteMode::Tritanopia => ColorPaletteMode::HighContrast,
            ColorPaletteMode::HighContrast => ColorPaletteMode::Natural,
        }
    }

    fn label(self) -> &'static str {
        match self {
            ColorPaletteMode::Natural => "Natural",
            ColorPaletteMode::Deuteranopia => "Deuteranopia",
            ColorPaletteMode::Protanopia => "Protanopia",
            ColorPaletteMode::Tritanopia => "Tritanopia",
            ColorPaletteMode::HighContrast => "High Contrast",
        }
    }
}

#[derive(Default, Clone)]
struct AccessibilitySettings {
    palette: ColorPaletteMode,
    narration_enabled: bool,
}

#[derive(Clone, Copy)]
struct HudTheme {
    card_bg: u32,
    card_border: u32,
    spark_bg: u32,
    spark_border: u32,
    text_primary: u32,
    text_subtle: u32,
    chip_text: u32,
    chip_running: u32,
    chip_paused: u32,
    chip_open: u32,
    chip_closed: u32,
    chip_follow: u32,
}

fn hud_theme(mode: ColorPaletteMode) -> HudTheme {
    match mode {
        ColorPaletteMode::HighContrast => HudTheme {
            card_bg: 0x06090f,
            card_border: 0xf8fafc,
            spark_bg: 0x0b121f,
            spark_border: 0xf8fafc,
            text_primary: 0xf8fafc,
            text_subtle: 0xcbd5f5,
            chip_text: 0x020617,
            chip_running: 0xf97316,
            chip_paused: 0xfacc15,
            chip_open: 0x38bdf8,
            chip_closed: 0xf87171,
            chip_follow: 0x8b5cf6,
        },
        ColorPaletteMode::Deuteranopia
        | ColorPaletteMode::Protanopia
        | ColorPaletteMode::Tritanopia => HudTheme {
            card_bg: 0x0f172a,
            card_border: 0x1f2a3d,
            spark_bg: 0x0b1626,
            spark_border: 0x27364c,
            text_primary: 0xf8fafc,
            text_subtle: 0x9aa7c0,
            chip_text: 0x0f172a,
            chip_running: 0x2dd4bf,
            chip_paused: 0xfacc15,
            chip_open: 0x60a5fa,
            chip_closed: 0xf97316,
            chip_follow: 0xa855f7,
        },
        ColorPaletteMode::Natural => HudTheme {
            card_bg: 0x0f172a,
            card_border: 0x1f2a3d,
            spark_bg: 0x0b1626,
            spark_border: 0x1f2a3d,
            text_primary: 0xf8fafc,
            text_subtle: 0x94a3b8,
            chip_text: 0x0f172a,
            chip_running: 0x22c55e,
            chip_paused: 0xf59e0b,
            chip_open: 0x38bdf8,
            chip_closed: 0xf97316,
            chip_follow: 0x8b5cf6,
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum CommandAction {
    TogglePlayback,
    GoLive,
    ToggleBrush,
    ToggleNarration,
    CyclePalette,
    ToggleSimulationPause,
    StepSimulation,
    ToggleAgentDraw,
    ToggleFoodOverlay,
    ToggleAgentOutline,
    IncreaseSimulationSpeed,
    DecreaseSimulationSpeed,
    AddCrossoverAgents,
    SpawnCarnivore,
    SpawnHerbivore,
    ToggleClosedEnvironment,
    FollowSelected,
    FollowOldest,
    ToggleDebugOverlay,
    ClearSelection,
    SelectAll,
    FocusFirstSelected,
    FitWorld,
    ToggleSettings,
    ToggleStatsPanel,
    ToggleHistoryPanel,
    TogglePerfPanel,
}

impl CommandAction {
    fn label(self) -> &'static str {
        match self {
            CommandAction::TogglePlayback => "Toggle playback",
            CommandAction::GoLive => "Jump to live",
            CommandAction::ToggleBrush => "Toggle brush",
            CommandAction::ToggleNarration => "Toggle narration",
            CommandAction::CyclePalette => "Cycle palette",
            CommandAction::ToggleSimulationPause => "Toggle simulation pause",
            CommandAction::StepSimulation => "Step simulation once",
            CommandAction::ToggleAgentDraw => "Toggle agent drawing",
            CommandAction::ToggleFoodOverlay => "Toggle food overlay",
            CommandAction::ToggleAgentOutline => "Toggle agent outline",
            CommandAction::IncreaseSimulationSpeed => "Increase simulation speed",
            CommandAction::DecreaseSimulationSpeed => "Decrease simulation speed",
            CommandAction::AddCrossoverAgents => "Spawn crossover agent",
            CommandAction::SpawnCarnivore => "Spawn carnivore",
            CommandAction::SpawnHerbivore => "Spawn herbivore",
            CommandAction::ToggleClosedEnvironment => "Toggle closed environment",
            CommandAction::FollowSelected => "Follow selected agent",
            CommandAction::FollowOldest => "Follow oldest agent",
            CommandAction::ToggleDebugOverlay => "Toggle debug overlay",
            CommandAction::ClearSelection => "Clear selection",
            CommandAction::SelectAll => "Select all agents",
            CommandAction::FocusFirstSelected => "Focus first selected agent",
            CommandAction::FitWorld => "Fit world",
            CommandAction::ToggleSettings => "Toggle settings panel",
            CommandAction::ToggleStatsPanel => "Toggle stats panel",
            CommandAction::ToggleHistoryPanel => "Toggle history panel",
            CommandAction::TogglePerfPanel => "Toggle performance panel",
        }
    }
}

#[derive(Clone)]
struct InputBindings {
    map: BTreeMap<CommandAction, Keystroke>,
}

impl Default for InputBindings {
    fn default() -> Self {
        let mut map = BTreeMap::new();
        map.insert(
            CommandAction::TogglePlayback,
            Keystroke::parse("space").unwrap_or_default(),
        );
        map.insert(
            CommandAction::GoLive,
            Keystroke::parse("g").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleBrush,
            Keystroke::parse("b").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleNarration,
            Keystroke::parse("n").unwrap_or_default(),
        );
        map.insert(
            CommandAction::CyclePalette,
            Keystroke::parse("ctrl-p").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleSimulationPause,
            Keystroke::parse("p").unwrap_or_default(),
        );
        map.insert(
            CommandAction::StepSimulation,
            Keystroke::parse("s").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleAgentDraw,
            Keystroke::parse("d").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleFoodOverlay,
            Keystroke::parse("f").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleAgentOutline,
            Keystroke::parse("ctrl-shift-o").unwrap_or_default(),
        );
        map.insert(
            CommandAction::IncreaseSimulationSpeed,
            Keystroke::parse("shift-=").unwrap_or_default(),
        );
        map.insert(
            CommandAction::DecreaseSimulationSpeed,
            Keystroke::parse("-").unwrap_or_default(),
        );
        map.insert(
            CommandAction::AddCrossoverAgents,
            Keystroke::parse("a").unwrap_or_default(),
        );
        map.insert(
            CommandAction::SpawnCarnivore,
            Keystroke::parse("q").unwrap_or_default(),
        );
        map.insert(
            CommandAction::SpawnHerbivore,
            Keystroke::parse("h").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleClosedEnvironment,
            Keystroke::parse("c").unwrap_or_default(),
        );
        map.insert(
            CommandAction::FollowSelected,
            Keystroke::parse("shift-s").unwrap_or_default(),
        );
        map.insert(
            CommandAction::FollowOldest,
            Keystroke::parse("o").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleDebugOverlay,
            Keystroke::parse("shift-f").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ClearSelection,
            Keystroke::parse("escape").unwrap_or_default(),
        );
        map.insert(
            CommandAction::SelectAll,
            Keystroke::parse("ctrl-a").unwrap_or_default(),
        );
        map.insert(
            CommandAction::FocusFirstSelected,
            Keystroke::parse("ctrl-f").unwrap_or_default(),
        );
        map.insert(
            CommandAction::FitWorld,
            Keystroke::parse("0").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleSettings,
            Keystroke::parse(",").unwrap_or_default(),
        );
        // bd-v9cz panel dismissal. Digits 1/2/3 were unbound; grouping them keeps the
        // rail's three panels adjacent and memorable.
        map.insert(
            CommandAction::ToggleStatsPanel,
            Keystroke::parse("1").unwrap_or_default(),
        );
        map.insert(
            CommandAction::ToggleHistoryPanel,
            Keystroke::parse("2").unwrap_or_default(),
        );
        map.insert(
            CommandAction::TogglePerfPanel,
            Keystroke::parse("3").unwrap_or_default(),
        );
        Self { map }
    }
}

impl InputBindings {
    fn iter(&self) -> Vec<(CommandAction, Keystroke)> {
        self.map
            .iter()
            .map(|(action, ks)| (*action, ks.clone()))
            .collect()
    }

    fn assign(&mut self, action: CommandAction, stroke: Keystroke) {
        if stroke.key.is_empty() {
            self.map.insert(action, Keystroke::default());
            return;
        }

        let conflict = self.map.iter().find_map(|(other, ks)| {
            if *other != action && keystrokes_equal(ks, &stroke) {
                Some(*other)
            } else {
                None
            }
        });

        if let Some(conflict_action) = conflict {
            self.map.insert(conflict_action, Keystroke::default());
        }

        self.map.insert(action, stroke);
    }

    fn action_for(&self, stroke: &Keystroke) -> Option<CommandAction> {
        self.map
            .iter()
            .find(|(_, binding)| keystrokes_equal(binding, stroke))
            .map(|(action, _)| *action)
    }
}

#[cfg(feature = "audio")]
struct AudioState {
    manager: AudioManager<DefaultBackend>,
    birth_sound: StaticSoundData,
    death_sound: StaticSoundData,
    spike_sound: StaticSoundData,
    toggle_sound: StaticSoundData,
    last_births: usize,
    last_deaths: usize,
    last_spike_count: usize,
    last_tick: u64,
}

#[cfg(feature = "audio")]
impl AudioState {
    fn new() -> Result<Self, String> {
        let manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())
            .map_err(|err| format!("{err:?}"))?;
        let birth_sound = generate_tone(523.25, 0.18, 0.4);
        let death_sound = generate_tone(196.0, 0.22, 0.45);
        let spike_sound = generate_tone(880.0, 0.12, 0.35);
        let toggle_sound = generate_tone(660.0, 0.10, 0.3);
        Ok(Self {
            manager,
            birth_sound,
            death_sound,
            spike_sound,
            toggle_sound,
            last_births: 0,
            last_deaths: 0,
            last_spike_count: 0,
            last_tick: u64::MAX,
        })
    }

    fn play(&mut self, sound: &StaticSoundData) {
        if let Err(err) = self.manager.play(sound.clone()) {
            error!(?err, "failed to play audio cue");
        }
    }
}

fn keystrokes_equal(a: &Keystroke, b: &Keystroke) -> bool {
    if a.key.is_empty() || b.key.is_empty() {
        return false;
    }

    if a.modifiers != b.modifiers {
        return false;
    }

    let key_matches = a.key.eq_ignore_ascii_case(&b.key)
        || (a.key.eq_ignore_ascii_case("space") && b.key.trim().is_empty())
        || (b.key.eq_ignore_ascii_case("space") && a.key.trim().is_empty());

    if key_matches {
        return true;
    }

    if let Some(ref key_char) = a.key_char {
        if key_char.eq_ignore_ascii_case(&b.key) {
            return true;
        }
        if a.key.eq_ignore_ascii_case("space") && key_char.trim().is_empty() {
            return true;
        }
    }

    if let Some(ref key_char) = b.key_char {
        if key_char.eq_ignore_ascii_case(&a.key) {
            return true;
        }
        if b.key.eq_ignore_ascii_case("space") && key_char.trim().is_empty() {
            return true;
        }
    }

    false
}

fn format_keystroke(keystroke: &Keystroke) -> String {
    if keystroke.key.is_empty() {
        return "Unbound".to_string();
    }

    let mut parts = Vec::new();
    if keystroke.modifiers.control {
        parts.push("Ctrl".to_string());
    }
    if keystroke.modifiers.alt {
        parts.push("Alt".to_string());
    }
    if keystroke.modifiers.shift {
        parts.push("Shift".to_string());
    }
    if keystroke.modifiers.platform {
        parts.push(if cfg!(target_os = "macos") {
            "Cmd".to_string()
        } else {
            "Super".to_string()
        });
    }
    if keystroke.modifiers.function {
        parts.push("Fn".to_string());
    }

    let key = if keystroke.key.len() == 1 {
        keystroke.key.to_uppercase()
    } else {
        keystroke.key.clone()
    };
    parts.push(key);
    parts.join(" + ")
}
#[derive(Clone)]
struct AgentListEntry {
    agent_id: AgentId,
    label: String,
    color: [f32; 3],
    energy: f32,
    health: f32,
    generation: Generation,
    age: u32,
    is_focused: bool,
    position: Position,
}

impl AgentListEntry {
    fn from_world(
        row: usize,
        agent_id: AgentId,
        runtime: &AgentRuntime,
        columns: &AgentColumns,
    ) -> Self {
        let generation = columns.generations()[row];
        let age = columns.ages()[row];
        let health = columns.health()[row];
        let color = columns.colors()[row];
        let position = columns.positions()[row];

        let label = format!("#{row} · {:?} · Gen {}", agent_id, generation.0);

        Self {
            agent_id,
            label,
            color,
            energy: runtime.energy,
            health,
            generation,
            age,
            is_focused: false,
            position,
        }
    }
}

#[derive(Clone)]
struct AgentInspectorDetails {
    agent_id: AgentId,
    label: String,
    color: [f32; 3],
    position: Position,
    energy: f32,
    health: f32,
    age: u32,
    generation: Generation,
    brain_descriptor: String,
    mutation_rates: MutationRates,
    trait_modifiers: TraitModifiers,
    spike_length: f32,
    sensors: Vec<f32>,
    outputs: Vec<f32>,
    /// Whether the agent's brain binding has a runner; an unbound agent's
    /// outputs are an identity copy of sensors and must not be "explained"
    /// (bd-16g.4.3).
    brain_bound: bool,
    brain_activations: Option<BrainActivations>,
    brain_source_tick: Option<u64>,
    brain_request_revision: Option<u64>,
    brain_payload_bytes: Option<usize>,
    brain_inspection_status: Option<String>,
    /// Per-neighbour attribution of what this agent perceives (bd-16g.4.2),
    /// computed in core and rendered verbatim — never re-derived by the UI.
    sense_attribution: Option<SensorAttribution>,
    /// Eye directions relative to the heading, radians; pairs with
    /// `sense_attribution` so cones can be labeled at their true angles.
    eye_directions: [f32; NUM_EYES],
    /// Clamped per-eye fields of view, radians.
    eye_fovs: [f32; NUM_EYES],
}

#[derive(Clone)]
struct BrainInspectorCapture {
    agent_uid: AgentUid,
    source_tick: u64,
    request_revision: u64,
    retained_payload_bytes: usize,
    status: BrainInspectorCaptureStatus,
}

#[derive(Clone)]
enum BrainInspectorCaptureStatus {
    Ready(BrainActivations),
    Unavailable(BrainInspectionUnavailable),
    Refused(String),
}

impl AgentInspectorDetails {
    fn from_world(
        world: &WorldState,
        agent_id: AgentId,
        brain_capture: Option<BrainInspectorCapture>,
    ) -> Option<Self> {
        let arena = world.agents();
        let columns = arena.columns();
        let runtime = world.runtime();

        let index = arena.index_of(agent_id)?;
        let agent_runtime = runtime.get(agent_id)?;

        let position = columns.positions()[index];
        let color = columns.colors()[index];
        let health = columns.health()[index];
        let age = columns.ages()[index];
        let generation = columns.generations()[index];
        let spike_length = columns.spike_lengths()[index];

        let sensors = agent_runtime.sensors.to_vec();
        let outputs = agent_runtime.outputs.to_vec();
        let brain_bound = agent_runtime.brain.is_bound();
        let (
            brain_activations,
            brain_source_tick,
            brain_request_revision,
            brain_payload_bytes,
            brain_inspection_status,
        ) = brain_capture.map_or((None, None, None, None, None), |capture| {
            let (activations, status) = match capture.status {
                BrainInspectorCaptureStatus::Ready(activations) => {
                    (Some(activations), "ready".to_owned())
                }
                BrainInspectorCaptureStatus::Unavailable(reason) => {
                    (None, format!("unavailable: {reason:?}"))
                }
                BrainInspectorCaptureStatus::Refused(detail) => {
                    (None, format!("refused: {detail}"))
                }
            };
            (
                activations,
                Some(capture.source_tick),
                Some(capture.request_revision),
                Some(capture.retained_payload_bytes),
                Some(status),
            )
        });

        let label = format!("Agent {:?} · Gen {} · Age {}", agent_id, generation.0, age);

        let brain_descriptor = agent_runtime.brain.describe().to_string();

        // On-demand, bounded, single-agent query (bd-16g.4.2): same cost
        // class as the per-frame brain inspection above.
        let sense_attribution = world.explain_sensors(agent_id, SENSE_PROBE_MAX_CONTRIBUTORS);
        let eye_directions = agent_runtime.eye_direction;
        let eye_fovs = agent_runtime.eye_fov;

        Some(Self {
            agent_id,
            label,
            color,
            position,
            energy: agent_runtime.energy,
            health,
            age,
            generation,
            brain_descriptor,
            mutation_rates: agent_runtime.mutation_rates,
            trait_modifiers: agent_runtime.trait_modifiers,
            spike_length,
            sensors,
            outputs,
            brain_bound,
            brain_activations,
            brain_source_tick,
            brain_request_revision,
            brain_payload_bytes,
            brain_inspection_status,
            sense_attribution,
            eye_directions,
            eye_fovs,
        })
    }
}

fn rgb_from_triplet(color: [f32; 3]) -> Rgba {
    Rgba {
        r: color[0].clamp(0.0, 1.0),
        g: color[1].clamp(0.0, 1.0),
        b: color[2].clamp(0.0, 1.0),
        a: 1.0,
    }
}

fn rgba_from_triplet_with_alpha(color: [f32; 3], alpha: f32) -> Rgba {
    Rgba {
        r: color[0].clamp(0.0, 1.0),
        g: color[1].clamp(0.0, 1.0),
        b: color[2].clamp(0.0, 1.0),
        a: alpha.clamp(0.0, 1.0),
    }
}
fn rgba_from_hex(hex: u32, alpha: f32) -> Rgba {
    let r = ((hex >> 16) & 0xff) as f32 / 255.0;
    let g = ((hex >> 8) & 0xff) as f32 / 255.0;
    let b = (hex & 0xff) as f32 / 255.0;
    Rgba {
        r,
        g,
        b,
        a: alpha.clamp(0.0, 1.0),
    }
}

fn lerp_rgba(a: Rgba, b: Rgba, t: f32) -> Rgba {
    let clamped = t.clamp(0.0, 1.0);
    Rgba {
        r: a.r + (b.r - a.r) * clamped,
        g: a.g + (b.g - a.g) * clamped,
        b: a.b + (b.b - a.b) * clamped,
        a: a.a + (b.a - a.a) * clamped,
    }
}

fn scale_rgb(color: Rgba, factor: f32) -> Rgba {
    let clamp = factor.clamp(0.0, 2.0);
    Rgba {
        r: (color.r * clamp).clamp(0.0, 1.0),
        g: (color.g * clamp).clamp(0.0, 1.0),
        b: (color.b * clamp).clamp(0.0, 1.0),
        a: color.a,
    }
}

#[cfg(feature = "audio")]
const AUDIO_SAMPLE_RATE: u32 = 44_100;

#[cfg(feature = "audio")]
fn generate_tone(frequency: f32, duration: f32, amplitude: f32) -> StaticSoundData {
    let total_samples = (duration * AUDIO_SAMPLE_RATE as f32).max(1.0) as usize;
    let mut frames = Vec::with_capacity(total_samples);
    for i in 0..total_samples {
        let t = i as f32 / AUDIO_SAMPLE_RATE as f32;
        let envelope = (1.0 - t / duration).clamp(0.0, 1.0);
        let sample = (2.0 * PI * frequency * t).sin() * amplitude * envelope;
        frames.push(Frame::from_mono(sample));
    }
    StaticSoundData {
        sample_rate: AUDIO_SAMPLE_RATE,
        frames: frames.into(),
        settings: StaticSoundSettings::default(),
        slice: None,
    }
}

#[derive(Clone, Copy)]
struct PerfSnapshot {
    latest_ms: f32,
    average_ms: f32,
    min_ms: f32,
    max_ms: f32,
    sample_count: usize,
    fps: f32,
}

impl Default for PerfSnapshot {
    fn default() -> Self {
        Self {
            latest_ms: 0.0,
            average_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            sample_count: 0,
            fps: 0.0,
        }
    }
}

struct PerfStats {
    start: Option<Instant>,
    samples: VecDeque<f32>,
    capacity: usize,
}

impl PerfStats {
    fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            start: None,
            samples: VecDeque::with_capacity(cap),
            capacity: cap,
        }
    }

    fn begin_frame(&mut self) {
        self.start = Some(Instant::now());
    }

    fn end_frame(&mut self) -> PerfSnapshot {
        let elapsed_ms = self
            .start
            .take()
            .map(|start| start.elapsed().as_secs_f32() * 1000.0)
            .unwrap_or(0.0);
        self.samples.push_back(elapsed_ms);
        if self.samples.len() > self.capacity {
            self.samples.pop_front();
        }
        self.snapshot(elapsed_ms)
    }

    fn snapshot(&self, latest: f32) -> PerfSnapshot {
        if self.samples.is_empty() {
            return PerfSnapshot {
                latest_ms: latest,
                ..PerfSnapshot::default()
            };
        }

        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum = 0.0;
        for &sample in &self.samples {
            min = min.min(sample);
            max = max.max(sample);
            sum += sample;
        }
        let count = self.samples.len();
        let avg = if count > 0 { sum / count as f32 } else { 0.0 };
        let fps = if latest > f32::EPSILON {
            1000.0 / latest
        } else {
            0.0
        };

        PerfSnapshot {
            latest_ms: latest,
            average_ms: avg,
            min_ms: if min.is_finite() { min } else { 0.0 },
            max_ms: if max.is_finite() { max } else { 0.0 },
            sample_count: count,
            fps,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PlaybackMode {
    Live,
    Paused,
    Playing,
}

#[derive(Clone, Copy)]
struct PlaybackStatus {
    mode: PlaybackMode,
    index: usize,
    total: usize,
    current_tick: Option<u64>,
}

struct PlaybackState {
    timeline: VecDeque<HudSnapshot>,
    max_frames: usize,
    mode: PlaybackMode,
    pointer: usize,
}
impl PlaybackState {
    fn new(max_frames: usize) -> Self {
        Self {
            timeline: VecDeque::with_capacity(max_frames),
            max_frames: max_frames.max(1),
            mode: PlaybackMode::Live,
            pointer: 0,
        }
    }

    fn mode(&self) -> PlaybackMode {
        self.mode
    }

    fn record(&mut self, snapshot: &HudSnapshot) {
        if !matches!(self.mode, PlaybackMode::Live)
            || self
                .timeline
                .back()
                .is_some_and(|latest| latest.tick == snapshot.tick)
        {
            return;
        }
        if self.timeline.len() == self.max_frames {
            self.timeline.pop_front();
        }
        self.timeline.push_back(snapshot.clone());
        self.pointer = self.timeline.len().saturating_sub(1);
    }

    fn snapshot_for_render(&mut self, live: HudSnapshot) -> HudSnapshot {
        match self.mode {
            PlaybackMode::Live => live,
            PlaybackMode::Paused => self.timeline.get(self.pointer).cloned().unwrap_or(live),
            PlaybackMode::Playing => {
                if self.timeline.is_empty() {
                    self.mode = PlaybackMode::Live;
                    return live;
                }
                let snapshot = self
                    .timeline
                    .get(self.pointer)
                    .cloned()
                    .unwrap_or_else(|| live.clone());
                if self.pointer + 1 < self.timeline.len() {
                    self.pointer += 1;
                } else {
                    self.mode = PlaybackMode::Live;
                    self.pointer = self.timeline.len().saturating_sub(1);
                }
                snapshot
            }
        }
    }

    fn restart(&mut self) {
        if self.timeline.is_empty() {
            return;
        }
        self.mode = PlaybackMode::Paused;
        self.pointer = 0;
    }

    fn step_back(&mut self) {
        if self.timeline.is_empty() {
            return;
        }
        self.mode = PlaybackMode::Paused;
        if self.pointer > 0 {
            self.pointer -= 1;
        }
    }

    fn step_forward(&mut self) {
        if self.timeline.is_empty() {
            return;
        }
        if self.pointer + 1 < self.timeline.len() {
            self.pointer += 1;
            self.mode = PlaybackMode::Paused;
        } else {
            self.go_live();
        }
    }

    fn toggle_play(&mut self) {
        match self.mode {
            PlaybackMode::Live => {
                if !self.timeline.is_empty() {
                    self.mode = PlaybackMode::Playing;
                    self.pointer = 0;
                }
            }
            PlaybackMode::Paused => {
                if !self.timeline.is_empty() {
                    self.mode = PlaybackMode::Playing;
                }
            }
            PlaybackMode::Playing => {
                self.mode = PlaybackMode::Paused;
            }
        }
    }

    fn go_live(&mut self) {
        self.mode = PlaybackMode::Live;
        if !self.timeline.is_empty() {
            self.pointer = self.timeline.len().saturating_sub(1);
        }
    }

    fn status(&self) -> PlaybackStatus {
        let current_tick = self.timeline.get(self.pointer).map(|snap| snap.tick);
        PlaybackStatus {
            mode: self.mode,
            index: if self.timeline.is_empty() {
                0
            } else {
                self.pointer.min(self.timeline.len() - 1)
            },
            total: self.timeline.len(),
            current_tick,
        }
    }
}

#[cfg(test)]
mod playback_state_tests {
    use super::*;

    fn snapshot(tick: u64) -> HudSnapshot {
        HudSnapshot {
            tick,
            ..HudSnapshot::default()
        }
    }

    fn paint(playback: &mut PlaybackState, live_tick: u64) -> HudSnapshot {
        let live = snapshot(live_tick);
        playback.record(&live);
        playback.snapshot_for_render(live)
    }

    #[test]
    fn full_playback_ring_advances_to_live() {
        let mut playback = PlaybackState::new(3);
        for tick in 0..3 {
            playback.record(&snapshot(tick));
        }
        playback.toggle_play();

        assert_eq!(paint(&mut playback, 3).tick, 0);
        assert_eq!(paint(&mut playback, 4).tick, 1);
        assert_eq!(paint(&mut playback, 5).tick, 2);
        assert!(matches!(playback.mode(), PlaybackMode::Live));
        assert_eq!(
            playback
                .timeline
                .iter()
                .map(|entry| entry.tick)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(paint(&mut playback, 6).tick, 6);
    }

    #[test]
    fn paused_frame_is_stable_while_live_paints_arrive() {
        let mut playback = PlaybackState::new(3);
        for tick in 0..3 {
            playback.record(&snapshot(tick));
        }
        playback.restart();

        for live_tick in 3..12 {
            assert_eq!(paint(&mut playback, live_tick).tick, 0);
            assert!(matches!(playback.mode(), PlaybackMode::Paused));
            let status = playback.status();
            assert_eq!(status.index, 0);
            assert_eq!(status.total, 3);
        }
        assert_eq!(playback.status().current_tick, Some(0));
        assert_eq!(
            playback
                .timeline
                .iter()
                .map(|entry| entry.tick)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn duplicate_science_ticks_are_recorded_once() {
        let mut playback = PlaybackState::new(3);
        playback.record(&snapshot(7));
        playback.record(&snapshot(7));
        playback.record(&snapshot(7));

        let status = playback.status();
        assert_eq!(status.total, 1);
        assert_eq!(status.current_tick, Some(7));
    }

    #[test]
    fn live_recording_evicts_only_the_oldest_frame() {
        let mut playback = PlaybackState::new(3);
        for tick in 0..4 {
            playback.record(&snapshot(tick));
        }

        assert_eq!(playback.timeline.len(), 3);
        assert_eq!(playback.timeline.front().map(|entry| entry.tick), Some(1));
        assert_eq!(playback.timeline.back().map(|entry| entry.tick), Some(3));
        let status = playback.status();
        assert!(matches!(status.mode, PlaybackMode::Live));
        assert_eq!(status.index, 2);
        assert_eq!(status.total, 3);
        assert_eq!(status.current_tick, Some(3));
    }
}

fn color_swatch(color: [f32; 3]) -> Div {
    div()
        .w(px(10.0))
        .h(px(10.0))
        .rounded_full()
        .border_1()
        .border_color(rgb(0x1e293b))
        .bg(rgb_from_triplet(color))
}

#[derive(Clone)]
struct RenderFrame {
    tick: u64,
    tonemap_mode: Option<RenderTonemapMode>,
    day_night_cycle_ticks: u32,
    day_night_start_phase: f32,
    world_size: (f32, f32),
    terrain: TerrainFrame,
    food_dimensions: (u32, u32),
    food_cell_size: u32,
    food_cells: Vec<f32>,
    food_max: f32,
    agents: Vec<AgentRenderData>,
    agent_reference_age: u64,
    agent_base_radius: f32,
    sense_radius: f32,
    post_stack: PostProcessStack,
    palette: ColorPaletteMode,
}

#[derive(Clone)]
struct TerrainFrame {
    dimensions: (u32, u32),
    cell_size: u32,
    tiles: Vec<TerrainTileVisual>,
}

#[derive(Clone, Copy)]
struct TerrainTileVisual {
    kind: TerrainKind,
    elevation: f32,
    moisture: f32,
    fertility_bias: f32,
    accent: f32,
    slope: f32,
    water_depth: f32,
    elevation_gradient: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WorldRasterKey {
    field_fingerprint: u64,
    tick: u64,
    daylight_bits: u32,
    palette: ColorPaletteMode,
}

struct CachedWorldRaster {
    key: WorldRasterKey,
    image: Arc<RenderImage>,
}

/// Bounded GPUI atlas ownership for a single [`SimulationView`].
///
/// GPUI caches a render image by its generated ID and does not evict it when
/// the last application `Arc` is dropped. Keeping the current and immediately
/// previous images gives the compositor one complete frame of overlap; the
/// image older than that is explicitly retired after the replacement paints.
#[derive(Default)]
struct WorldRasterCache {
    current: Option<CachedWorldRaster>,
    previous: Option<Arc<RenderImage>>,
}

impl WorldRasterCache {
    fn image_for(&self, key: WorldRasterKey) -> Option<Arc<RenderImage>> {
        self.current
            .as_ref()
            .filter(|entry| entry.key == key)
            .map(|entry| Arc::clone(&entry.image))
    }

    fn commit(&mut self, key: WorldRasterKey, image: Arc<RenderImage>) -> Option<Arc<RenderImage>> {
        let retired = self.previous.take();
        self.previous = self
            .current
            .replace(CachedWorldRaster { key, image })
            .map(|entry| entry.image);
        retired
    }

    fn drain(&mut self) -> Vec<Arc<RenderImage>> {
        let mut images = Vec::with_capacity(2);
        if let Some(entry) = self.current.take() {
            images.push(entry.image);
        }
        if let Some(previous) = self.previous.take()
            && images.iter().all(|image| image.id != previous.id)
        {
            images.push(previous);
        }
        images
    }
}

struct WorldRasterPixels {
    width: u32,
    height: u32,
    bgra: Vec<u8>,
}

#[derive(Clone)]
struct PostProcessStack {
    passes: Vec<PostProcessPass>,
}
#[derive(Clone, Copy)]
enum PostProcessPass {
    Exposure {
        factor: f32,
    },
    Vignette {
        strength: f32,
        smoothness: f32,
    },
    Bloom {
        strength: f32,
    },
    Fog {
        strength: f32,
        color: [f32; 3],
    },
    Scanlines {
        intensity: f32,
        spacing: f32,
    },
    FilmGrain {
        strength: f32,
        seed: u64,
    },
    ColorGrade {
        lift: f32,
        gain: f32,
        temperature: f32,
    },
}

fn build_post_process_stack(world: &WorldState, palette: ColorPaletteMode) -> PostProcessStack {
    let tick = world.tick().0;
    let render = &world.config().render;
    let quality = render.requested_quality();
    let features = tier_features(quality);
    let (cycle_ticks, start_phase) = render.resolved_day_night();
    let daylight = visual::daylight_factor(tick, cycle_ticks, start_phase);
    let night = 1.0 - daylight;
    let closed_bonus = if world.is_closed() { 0.18 } else { 0.0 };
    let agent_count = world.agent_count().max(1) as f32;
    let latest = world.history().last();
    let (births_ratio, deaths_ratio) = latest
        .map(|summary| {
            (
                summary.births as f32 / agent_count,
                summary.deaths as f32 / agent_count,
            )
        })
        .unwrap_or((0.0, 0.0));
    let life_delta = (births_ratio - deaths_ratio).clamp(-1.0, 1.0);
    let tension = life_delta.abs();
    let post = render.post.as_ref();
    let atmosphere = &visual::visual_style().atmosphere;
    let potato = matches!(quality, RenderQuality::Potato);

    // Potato is the canonical no-post tier. Explicit per-effect blocks and exposure
    // still win: an operator who asks for one effect should not have that intent
    // silently discarded merely because the baseline tier is Potato.
    let explicit_effect = post.is_some() || render.tonemap_exposure_bias.is_some();
    if potato && !explicit_effect {
        return PostProcessStack { passes: Vec::new() };
    }

    let exposure_bias = render.tonemap_exposure_bias.unwrap_or(0.0).clamp(-4.0, 4.0);
    let exposure_factor =
        (atmosphere.exposure * 2.0_f32.powf(exposure_bias) * (0.90 + daylight * 0.24))
            .clamp(0.25, 2.5);

    let vignette = post.and_then(|settings| settings.vignette.as_ref());
    let vignette_enabled = vignette.map_or(!potato, |settings| settings.enabled);
    let vignette_strength = vignette
        .and_then(|settings| settings.intensity)
        .unwrap_or(atmosphere.vignette + night * 0.42 + closed_bonus)
        .clamp(0.0, 0.9);
    let vignette_smoothness = vignette
        .and_then(|settings| settings.smoothness)
        .unwrap_or(0.68)
        .clamp(0.0, 1.0);

    let bloom = post.and_then(|settings| settings.bloom.as_ref());
    let bloom_enabled = bloom.map_or(features.bloom, |settings| settings.enabled);
    let bloom_strength = bloom
        .and_then(|settings| settings.intensity)
        .unwrap_or(
            atmosphere.bloom_intensity * (0.80 + daylight * 0.28) + life_delta.max(0.0) * 0.20,
        )
        .clamp(0.0, 1.0);

    let fog = post.and_then(|settings| settings.fog.as_ref());
    let fog_mode = fog
        .and_then(|settings| settings.mode)
        .unwrap_or(if features.fog {
            RenderFogMode::Medium
        } else {
            RenderFogMode::Off
        });
    let fog_strength = match fog_mode {
        RenderFogMode::Off => 0.0,
        RenderFogMode::Low => 0.07,
        RenderFogMode::Medium => 0.14,
        RenderFogMode::High => 0.24,
    };
    let fog_color = fog
        .and_then(|settings| settings.color)
        .unwrap_or(atmosphere.fog_srgb);

    let temperature = match palette {
        ColorPaletteMode::Natural => 0.07 - night * 0.13 - life_delta * 0.05,
        ColorPaletteMode::Deuteranopia => 0.05,
        ColorPaletteMode::Protanopia => -0.04,
        ColorPaletteMode::Tritanopia => 0.12,
        ColorPaletteMode::HighContrast => 0.20,
    };

    let color_grade = PostProcessPass::ColorGrade {
        lift: (0.025 + daylight * 0.025 + closed_bonus * 0.08 - life_delta * 0.04).clamp(0.0, 0.12),
        gain: (0.88 + daylight * 0.25 + life_delta.max(0.0) * 0.12).clamp(0.8, 1.3),
        temperature,
    };

    let mut passes = Vec::with_capacity(7);
    if fog_strength > 0.0 {
        passes.push(PostProcessPass::Fog {
            strength: fog_strength * (0.75 + night * 0.65),
            color: fog_color,
        });
    }
    if !potato {
        passes.push(color_grade);
    }
    if !potato || render.tonemap_exposure_bias.is_some() {
        passes.push(PostProcessPass::Exposure {
            factor: exposure_factor,
        });
    }
    if bloom_enabled && bloom_strength > 0.0 {
        passes.push(PostProcessPass::Bloom {
            strength: bloom_strength,
        });
    }
    if vignette_enabled && vignette_strength > 0.0 {
        passes.push(PostProcessPass::Vignette {
            strength: vignette_strength,
            smoothness: vignette_smoothness,
        });
    }
    if matches!(quality, RenderQuality::High | RenderQuality::Ultra) {
        passes.push(PostProcessPass::Scanlines {
            intensity: (0.10 + night * 0.18 + closed_bonus * 0.25).clamp(0.06, 0.42),
            spacing: 5.5,
        });
        passes.push(PostProcessPass::FilmGrain {
            strength: (0.14 + tension * 0.06).clamp(0.10, 0.24),
            seed: tick,
        });
    }

    PostProcessStack { passes }
}

#[derive(Clone)]
struct AgentRenderData {
    agent_id: AgentId,
    position: Position,
    color: [f32; 3],
    spike_length: f32,
    velocity: Velocity,
    heading: f32,
    health: f32,
    age: u32,
    boost: f32,
    wheel_left: f32,
    wheel_right: f32,
    herbivore_tendency: f32,
    temperature_preference: f32,
    food_delta: f32,
    sound_level: f32,
    sound_output: f32,
    sound_multiplier: f32,
    trait_smell: f32,
    trait_sound: f32,
    trait_hearing: f32,
    trait_eye: f32,
    trait_blood: f32,
    eye_dirs: [f32; NUM_EYES],
    eye_fov: [f32; NUM_EYES],
    selection: SelectionState,
    indicator: IndicatorState,
    spike_extended: bool,
    spike_struck: bool,
    spike_victim: bool,
    reproduction_intent: f32,
}

#[derive(Clone)]
struct CanvasState {
    frame: RenderFrame,
    camera: Arc<Mutex<Camera>>,
    world_raster_cache: Arc<Mutex<WorldRasterCache>>,
    #[cfg(test)]
    force_legacy_world_painter: bool,
    focus_agent: Option<AgentId>,
    controls: ControlsSnapshot,
    debug: DebugOverlayState,
    follow_target: Option<Position>,
    perf: PerfSnapshot,
}

impl RenderFrame {
    fn from_world(world: &WorldState, palette: ColorPaletteMode) -> Option<Self> {
        let food = world.food();
        let width = food.width();
        let height = food.height();
        if width == 0 || height == 0 {
            return None;
        }

        let config = world.config();
        let arena = world.agents();
        let columns = arena.columns();
        let runtime = world.runtime();

        let positions = columns.positions();
        let colors = columns.colors();
        let spikes = columns.spike_lengths();
        let healths = columns.health();
        let velocities = columns.velocities();
        let headings = columns.headings();
        let ages = columns.ages();
        let agent_reference_age = u64::from(config.aging_health_decay_start.max(1));

        let estimated_agents = arena.len();
        let mut agents = Vec::with_capacity(estimated_agents);
        for (idx, agent_id) in arena.iter_handles().enumerate() {
            let runtime_entry = runtime.get(agent_id);
            let selection = runtime_entry.map(|rt| rt.selection).unwrap_or_default();
            let indicator = runtime_entry.map(|rt| rt.indicator).unwrap_or_default();
            let boost = runtime_entry.is_some_and(|rt| rt.outputs.boost_engaged());
            let spike_extended = runtime_entry
                .is_some_and(|rt| rt.outputs.channel(OutputChannel::SpikeTarget) > 0.5);
            let spike_struck = runtime_entry.is_some_and(|rt| rt.combat.spike_attacker);
            let spike_victim = runtime_entry.is_some_and(|rt| rt.spiked);
            let reproduction_intent = runtime_entry.map(|rt| rt.give_intent).unwrap_or(0.0);

            let (
                wheel_left,
                wheel_right,
                herbivore_tendency,
                temperature_preference,
                food_delta,
                sound_level,
                sound_output,
                sound_multiplier,
                trait_smell,
                trait_sound,
                trait_hearing,
                trait_eye,
                trait_blood,
                eye_dirs,
                eye_fov,
            ) = runtime_entry
                .map(|rt| {
                    let mut eye_dirs = [0.0_f32; NUM_EYES];
                    let mut eye_fov = [0.0_f32; NUM_EYES];
                    eye_dirs.copy_from_slice(&rt.eye_direction);
                    eye_fov.copy_from_slice(&rt.eye_fov);
                    (
                        rt.outputs.channel(OutputChannel::WheelLeft),
                        rt.outputs.channel(OutputChannel::WheelRight),
                        rt.herbivore_tendency.clamp(0.0, 1.0),
                        rt.temperature_preference.clamp(0.0, 1.0),
                        rt.food_delta,
                        rt.outputs.channel(OutputChannel::SoundLevel),
                        rt.sound_output,
                        rt.sound_multiplier,
                        rt.trait_modifiers.smell,
                        rt.trait_modifiers.sound,
                        rt.trait_modifiers.hearing,
                        rt.trait_modifiers.eye,
                        rt.trait_modifiers.blood,
                        eye_dirs,
                        eye_fov,
                    )
                })
                .unwrap_or_else(|| {
                    (
                        0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.3,
                        0.4,
                        1.0,
                        1.5,
                        1.5,
                        [0.0; NUM_EYES],
                        [1.0; NUM_EYES],
                    )
                });

            agents.push(AgentRenderData {
                agent_id,
                position: positions[idx],
                color: colors[idx],
                spike_length: spikes[idx],
                velocity: velocities[idx],
                heading: headings[idx],
                health: healths[idx],
                age: ages[idx],
                boost: if boost { 1.0 } else { 0.0 },
                wheel_left,
                wheel_right,
                herbivore_tendency,
                temperature_preference,
                food_delta,
                sound_level,
                sound_output,
                sound_multiplier,
                trait_smell,
                trait_sound,
                trait_hearing,
                trait_eye,
                trait_blood,
                eye_dirs,
                eye_fov,
                selection,
                indicator,
                spike_extended,
                spike_struck,
                spike_victim,
                reproduction_intent,
            });
        }

        let food_cells = food.cells().to_vec();
        let terrain = build_terrain_frame(
            world.terrain(),
            world
                .hydrology()
                .map(scriptbots_core::HydrologyState::water_depth),
        );
        let (day_night_cycle_ticks, day_night_start_phase) = config.render.resolved_day_night();

        Some(Self {
            tick: world.tick().0,
            tonemap_mode: config.render.tonemap_mode,
            day_night_cycle_ticks,
            day_night_start_phase,
            world_size: (config.world_width as f32, config.world_height as f32),
            terrain,
            food_dimensions: (width, height),
            food_cell_size: config.food_cell_size,
            food_cells,
            food_max: config.food_max,
            agents,
            agent_reference_age,
            agent_base_radius: config.bot_radius.max(1.0),
            sense_radius: config.sense_radius,
            post_stack: build_post_process_stack(world, palette),
            palette,
        })
    }
}

fn layout_camera_for_frame(
    camera: &mut Camera,
    frame: &RenderFrame,
    origin: (f32, f32),
    canvas_size: (f32, f32),
) -> ViewLayout {
    camera.layout_with_initial_population(
        origin,
        canvas_size,
        frame.world_size,
        frame.agent_base_radius,
        frame.agents.iter().map(|agent| agent.position),
    )
}

fn build_terrain_frame(layer: &TerrainLayer, water_depth: Option<&[f32]>) -> TerrainFrame {
    let width = layer.width();
    let height = layer.height();
    let mut tiles = Vec::with_capacity((width as usize) * (height as usize));
    let source = layer.tiles();
    if width == 0 || height == 0 {
        return TerrainFrame {
            dimensions: (width, height),
            cell_size: layer.cell_size(),
            tiles,
        };
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    for y in 0..height_usize {
        for x in 0..width_usize {
            let idx = y * width_usize + x;
            let tile = source[idx];

            let left = if x > 0 { source[idx - 1] } else { tile };
            let right = if x + 1 < width_usize {
                source[idx + 1]
            } else {
                tile
            };
            let up = if y > 0 {
                source[idx - width_usize]
            } else {
                tile
            };
            let down = if y + 1 < height_usize {
                source[idx + width_usize]
            } else {
                tile
            };

            let slope = ((tile.elevation - left.elevation).abs()
                + (tile.elevation - right.elevation).abs()
                + (tile.elevation - up.elevation).abs()
                + (tile.elevation - down.elevation).abs())
                * 0.25;
            let slope = slope.min(1.0);
            let cell_size = layer.cell_size() as f32;
            let world_x = (x as f32 + 0.5) * cell_size;
            let world_y = (y as f32 + 0.5) * cell_size;
            let (gradient_x, gradient_y) = layer.gradient_world(world_x, world_y, cell_size);

            tiles.push(TerrainTileVisual {
                kind: tile.kind,
                elevation: tile.elevation,
                moisture: tile.moisture,
                fertility_bias: tile.fertility_bias,
                accent: tile.accent,
                slope,
                water_depth: water_depth
                    .and_then(|depths| depths.get(idx))
                    .copied()
                    .unwrap_or(0.0),
                elevation_gradient: [gradient_x, gradient_y],
            });
        }
    }

    TerrainFrame {
        dimensions: (width, height),
        cell_size: layer.cell_size(),
        tiles,
    }
}

#[derive(Clone)]
struct HistoryChartData {
    width: f32,
    height: f32,
    agents: Vec<(f32, f32)>,
    births: Vec<(f32, f32)>,
    deaths: Vec<(f32, f32)>,
}

impl HistoryChartData {
    fn from_entries(entries: &[HudHistoryEntry], width: f32, height: f32) -> Option<Self> {
        if entries.len() < 2 {
            return None;
        }
        // Decimate samples to a fixed budget to avoid heavy polylines
        const MAX_SAMPLES: usize = 120;
        let stride = entries.len().div_ceil(MAX_SAMPLES).max(1);
        let decimated: Vec<&HudHistoryEntry> = entries.iter().step_by(stride).collect();

        let max_agents = decimated.iter().map(|e| e.agent_count).max().unwrap_or(1);
        let max_births = decimated.iter().map(|e| e.births).max().unwrap_or(0);
        let max_deaths = decimated.iter().map(|e| e.deaths).max().unwrap_or(0);
        let scale_agents = max_agents.max(1) as f32;
        let scale_births = max_births.max(1) as f32;
        let scale_deaths = max_deaths.max(1) as f32;

        let samples = decimated.len();
        let step = if samples > 1 {
            width / (samples as f32 - 1.0)
        } else {
            width
        };

        let y_clamp = height - 1.0;

        let to_points = |values: Vec<f32>| -> Vec<(f32, f32)> {
            values
                .into_iter()
                .enumerate()
                .map(|(idx, v)| {
                    let x = step * idx as f32;
                    let y = height - (v * y_clamp).min(y_clamp);
                    (x, y)
                })
                .collect()
        };

        let agents = to_points(
            decimated
                .iter()
                .map(|e| e.agent_count as f32 / scale_agents)
                .collect(),
        );
        let births = to_points(
            decimated
                .iter()
                .map(|e| e.births as f32 / scale_births)
                .collect(),
        );
        let deaths = to_points(
            decimated
                .iter()
                .map(|e| e.deaths as f32 / scale_deaths)
                .collect(),
        );

        Some(Self {
            width,
            height,
            agents,
            births,
            deaths,
        })
    }
}

fn legend_item(color: Rgba, label: &str) -> Div {
    div()
        .flex()
        .items_center()
        .gap_1()
        .child(div().w(px(8.0)).h(px(8.0)).rounded_full().bg(color))
        .child(label.to_string())
}
fn paint_history_chart(bounds: Bounds<Pixels>, data: &HistoryChartData, window: &mut Window) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let chart_width = f32::from(bounds_size.width).max(1.0);
    let chart_height = f32::from(bounds_size.height).max(1.0);
    let scale_x = chart_width / data.width.max(1.0);
    let scale_y = chart_height / data.height.max(1.0);

    let to_point = |(x, y): (f32, f32)| {
        point(
            px(f32::from(origin.x) + x * scale_x),
            px(f32::from(origin.y) + y * scale_y),
        )
    };

    let mut draw_polyline = |points: &[(f32, f32)], color: Rgba| {
        if points.len() < 2 {
            return;
        }
        let mut builder = PathBuilder::stroke(px(1.6));
        builder.move_to(to_point(points[0]));
        for &pt in &points[1..] {
            builder.line_to(to_point(pt));
        }
        if let Ok(path) = builder.build() {
            window.paint_path(path, color);
        }
    };

    // bd-f4x0: series ride the bd-9pqz ramp (agents cyan, food mint, carnivore
    // magenta) instead of stock blue/green/red that matched nothing on screen.
    draw_polyline(&data.agents, chrome::series_population().into());
    draw_polyline(&data.births, chrome::series_births().into());
    draw_polyline(&data.deaths, chrome::series_deaths().into());
}
fn append_arc_polyline(
    builder: &mut PathBuilder,
    cx: f32,
    cy: f32,
    radius: f32,
    start_deg: f32,
    sweep_deg: f32,
) {
    let segments = (sweep_deg.abs() / 6.0).ceil().max(1.0) as usize;
    let start = start_deg.to_radians();
    let sweep = sweep_deg.to_radians();
    let step = sweep / segments as f32;
    for i in 0..=segments {
        let angle = start + step * i as f32;
        let x = cx + radius * angle.cos();
        let y = cy + radius * angle.sin();
        if i == 0 {
            builder.move_to(point(px(x), px(y)));
        } else {
            builder.line_to(point(px(x), px(y)));
        }
    }
}

fn append_circle_polygon(builder: &mut PathBuilder, cx: f32, cy: f32, radius: f32) {
    append_arc_polyline(builder, cx, cy, radius, 0.0, 360.0);
}

fn append_capsule_polygon(
    builder: &mut PathBuilder,
    center: (f32, f32),
    forward: (f32, f32),
    right: (f32, f32),
    half_length: f32,
    radius: f32,
    segments: usize,
) {
    let segments = segments.max(4);
    let radius = radius.max(0.5);
    let half_length = half_length.max(radius + 0.5);
    let step = std::f32::consts::PI / segments as f32;
    let offset = (half_length - radius).max(0.0);
    let front_center = (center.0 + forward.0 * offset, center.1 + forward.1 * offset);
    let back_center = (center.0 - forward.0 * offset, center.1 - forward.1 * offset);

    let mut first = true;
    for i in 0..=segments {
        let theta = i as f32 * step;
        let (sin_t, cos_t) = theta.sin_cos();
        let point_x = front_center.0 + right.0 * (cos_t * radius) + forward.0 * (sin_t * radius);
        let point_y = front_center.1 + right.1 * (cos_t * radius) + forward.1 * (sin_t * radius);
        if first {
            builder.move_to(point(px(point_x), px(point_y)));
            first = false;
        } else {
            builder.line_to(point(px(point_x), px(point_y)));
        }
    }
    for i in 0..=segments {
        let theta = i as f32 * step;
        let (sin_t, cos_t) = theta.sin_cos();
        let point_x = back_center.0 - right.0 * (cos_t * radius) - forward.0 * (sin_t * radius);
        let point_y = back_center.1 - right.1 * (cos_t * radius) - forward.1 * (sin_t * radius);
        builder.line_to(point(px(point_x), px(point_y)));
    }
    builder.close();
}

fn palette_color(color: Rgba, palette: ColorPaletteMode, palette_is_natural: bool) -> Rgba {
    if palette_is_natural {
        color
    } else {
        apply_palette(color, palette)
    }
}

/// Smallest on-screen avatar size (in px) that paints the full detail kit.
/// Below this, ears/eyes/mouth resolve to sub-pixel mush — the silhouette
/// branch carries the same information for a fraction of the instance-buffer
/// cost (bd-2z0.7.12).
const DETAIL_MIN_PX: f32 = 18.0;

const AGENT_LOD_DIET_BINS: usize = 8;
const AGENT_LOD_LUMA_BINS: usize = 8;
const AGENT_LOD_BODY_BINS: usize = AGENT_LOD_DIET_BINS * AGENT_LOD_LUMA_BINS;
const AGENT_LOD_SPIKE_BINS: usize = 4;

#[derive(Clone, Copy, Debug)]
struct AgentSilhouette {
    center: (f32, f32),
    forward: (f32, f32),
    right: (f32, f32),
    body_half_length: f32,
    body_radius: f32,
    spike: [(f32, f32); 3],
    boost: Option<[(f32, f32); 3]>,
    extent: f32,
}

fn agent_silhouette(
    position: (f32, f32),
    size_px: f32,
    scale: f32,
    visuals: &AgentVisualParams,
    boost: f32,
    sound_multiplier: f32,
) -> AgentSilhouette {
    let half = size_px * 0.5;
    let forward = (visuals.facing[0], visuals.facing[1]);
    let right = (visuals.right[0], visuals.right[1]);
    let body_half_length = half * 1.35;
    let body_radius = (half * 0.72).max(3.0);

    let spike_base = body_radius * 0.65;
    let base_offset = body_half_length - body_radius * 0.2;
    // Core's physical tip offset remains authoritative below, while readiness
    // provides a screen-space minimum so the production 0..=1 growth range does
    // not collapse to a sub-pixel difference at overview zoom.
    let spike_front_offset =
        body_half_length + body_radius * (0.7 + visuals.spike_readiness * 2.0) + 2.0;
    let spike = [
        (
            position.0 + forward.0 * base_offset - right.0 * spike_base,
            position.1 + forward.1 * base_offset - right.1 * spike_base,
        ),
        (
            position.0 + forward.0 * base_offset + right.0 * spike_base,
            position.1 + forward.1 * base_offset + right.1 * spike_base,
        ),
        (
            position.0 + forward.0 * spike_front_offset + visuals.spike_tip_offset[0] * scale,
            position.1 + forward.1 * spike_front_offset + visuals.spike_tip_offset[1] * scale,
        ),
    ];

    let boost = (boost > 0.05).then(|| {
        let boost_level = boost.clamp(0.0, 1.0);
        let tail_offset = body_half_length - body_radius * 0.3;
        let flame_length =
            body_radius * (1.2 + boost_level * 1.6) + sound_multiplier.max(1.0) * 4.0;
        [
            (
                position.0 - forward.0 * tail_offset - right.0 * (body_radius * 0.55),
                position.1 - forward.1 * tail_offset - right.1 * (body_radius * 0.55),
            ),
            (
                position.0 - forward.0 * tail_offset + right.0 * (body_radius * 0.55),
                position.1 - forward.1 * tail_offset + right.1 * (body_radius * 0.55),
            ),
            (
                position.0 - forward.0 * (body_half_length + flame_length),
                position.1 - forward.1 * (body_half_length + flame_length),
            ),
        ]
    });

    let mut extent = body_half_length + body_radius;
    for point in spike.into_iter().chain(boost.into_iter().flatten()) {
        extent = extent.max((point.0 - position.0).hypot(point.1 - position.1));
    }

    AgentSilhouette {
        center: position,
        forward,
        right,
        body_half_length,
        body_radius,
        spike,
        boost,
        extent,
    }
}

fn append_triangle(builder: &mut PathBuilder, triangle: [(f32, f32); 3]) {
    builder.move_to(point(px(triangle[0].0), px(triangle[0].1)));
    builder.line_to(point(px(triangle[1].0), px(triangle[1].1)));
    builder.line_to(point(px(triangle[2].0), px(triangle[2].1)));
    builder.close();
}

fn agent_spike_color(
    visuals: &AgentVisualParams,
    palette: ColorPaletteMode,
    palette_is_natural: bool,
) -> Rgba {
    palette_color(
        scale_rgb(
            rgba_from_triplet_with_alpha(visuals.spike_color, 0.55 + visuals.spike_readiness * 0.4),
            1.0 + visuals.spike_readiness * 0.3,
        ),
        palette,
        palette_is_natural,
    )
}

fn agent_boost_color(
    boost: f32,
    visuals: &AgentVisualParams,
    palette: ColorPaletteMode,
    palette_is_natural: bool,
) -> Rgba {
    let gain = (visuals.body_emissive_gain / visual::visual_style().agents.boost_emissive_gain)
        .clamp(0.0, 1.0);
    palette_color(
        rgba_from_triplet_with_alpha(
            visuals.body_emissive,
            0.35 + boost.clamp(0.0, 1.0) * 0.3 + gain * 0.15,
        ),
        palette,
        palette_is_natural,
    )
}

#[allow(clippy::too_many_arguments)]
fn paint_agent_avatar(
    window: &mut Window,
    agent: &AgentRenderData,
    position: (f32, f32),
    size_px: f32,
    scale: f32,
    body_color: Rgba,
    visuals: &AgentVisualParams,
    palette: ColorPaletteMode,
    palette_is_natural: bool,
    very_low_fps: bool,
) {
    let (px_x, px_y) = position;
    let silhouette = agent_silhouette(
        position,
        size_px,
        scale,
        visuals,
        agent.boost,
        agent.sound_multiplier,
    );
    let forward = silhouette.forward;
    let right = silhouette.right;
    let body_half_length = silhouette.body_half_length;
    let body_radius = silhouette.body_radius;
    let wheel_half_length = body_half_length * 0.96;
    let wheel_radius = (body_radius * 0.38).max(2.0);
    let wheel_offset = body_radius + wheel_radius * 0.55;

    let left_center = (px_x - right.0 * wheel_offset, px_y - right.1 * wheel_offset);
    let right_center = (px_x + right.0 * wheel_offset, px_y + right.1 * wheel_offset);

    // Wheels behind body
    let mut left_wheel = PathBuilder::fill();
    append_capsule_polygon(
        &mut left_wheel,
        left_center,
        forward,
        right,
        wheel_half_length,
        wheel_radius,
        10,
    );
    if let Ok(path) = left_wheel.build() {
        let mut wheel_color = rgba_from_triplet_with_alpha(visuals.wheel_colors[0], 1.0);
        wheel_color = palette_color(wheel_color, palette, palette_is_natural);
        window.paint_path(path, wheel_color);
    }

    let mut right_wheel = PathBuilder::fill();
    append_capsule_polygon(
        &mut right_wheel,
        right_center,
        forward,
        right,
        wheel_half_length,
        wheel_radius,
        10,
    );
    if let Ok(path) = right_wheel.build() {
        let mut wheel_color = rgba_from_triplet_with_alpha(visuals.wheel_colors[1], 1.0);
        wheel_color = palette_color(wheel_color, palette, palette_is_natural);
        window.paint_path(path, wheel_color);
    }

    if very_low_fps || size_px < DETAIL_MIN_PX {
        // LOD: below DETAIL_MIN_PX the avatar's ears/eyes/mouth resolve to sub-pixel
        // mush; the silhouette retains facing, spike, boost, diet/health/age body
        // color, and selection remains a separate overlay. No agent is dropped.
        if let Some(boost) = silhouette.boost {
            let mut boost_path = PathBuilder::fill();
            append_triangle(&mut boost_path, boost);
            if let Ok(path) = boost_path.build() {
                window.paint_path(
                    path,
                    agent_boost_color(agent.boost, visuals, palette, palette_is_natural),
                );
            }
        }

        let mut body_shape = PathBuilder::fill();
        append_capsule_polygon(
            &mut body_shape,
            silhouette.center,
            forward,
            right,
            body_half_length,
            body_radius,
            12,
        );
        if let Ok(path) = body_shape.build() {
            window.paint_path(path, body_color);
        }

        let mut spike_path = PathBuilder::fill();
        append_triangle(&mut spike_path, silhouette.spike);
        if let Ok(path) = spike_path.build() {
            window.paint_path(
                path,
                agent_spike_color(visuals, palette, palette_is_natural),
            );
        }
        return;
    }

    // Body shell
    let mut body_shape = PathBuilder::fill();
    append_capsule_polygon(
        &mut body_shape,
        (px_x, px_y),
        forward,
        right,
        body_half_length,
        body_radius,
        16,
    );
    if let Ok(path) = body_shape.build() {
        window.paint_path(path, body_color);
    }

    // Diet stripe
    let mut stripe_color = rgba_from_triplet_with_alpha(visuals.stripe_color, 0.42);
    stripe_color = palette_color(stripe_color, palette, palette_is_natural);
    let mut stripe = PathBuilder::fill();
    append_capsule_polygon(
        &mut stripe,
        (px_x, px_y),
        forward,
        right,
        body_half_length * 0.82,
        body_radius * 0.45,
        14,
    );
    if let Ok(path) = stripe.build() {
        window.paint_path(path, stripe_color);
    }

    // Boost exhaust
    if let Some(boost) = silhouette.boost {
        let mut flame = PathBuilder::fill();
        append_triangle(&mut flame, boost);
        if let Ok(path) = flame.build() {
            window.paint_path(
                path,
                agent_boost_color(agent.boost, visuals, palette, palette_is_natural),
            );
        }
    }

    // Spike spear
    let mut spike_path = PathBuilder::fill();
    append_triangle(&mut spike_path, silhouette.spike);
    if let Ok(path) = spike_path.build() {
        window.paint_path(
            path,
            agent_spike_color(visuals, palette, palette_is_natural),
        );
    }

    // Mouth
    let eating_level = agent.food_delta.abs().min(1.5);
    let yelling_level = agent.sound_output.abs().min(1.5);
    let mouth_open = (0.35 + eating_level * 0.4 + yelling_level * 0.55).clamp(0.35, 1.6);
    let mouth_half_length = body_radius * 0.62;
    let mouth_radius = (body_radius * 0.14).max(1.2) * mouth_open;
    let mouth_offset = body_half_length - body_radius * 0.35;
    let mouth_center = (
        px_x + forward.0 * mouth_offset,
        px_y + forward.1 * mouth_offset,
    );
    let mut mouth_color =
        rgba_from_triplet_with_alpha(visuals.mouth_color, 0.72 + visuals.mouth_activity * 0.2);
    mouth_color = palette_color(mouth_color, palette, palette_is_natural);
    let mut mouth_path = PathBuilder::fill();
    append_capsule_polygon(
        &mut mouth_path,
        mouth_center,
        right,
        forward,
        mouth_half_length,
        mouth_radius,
        10,
    );
    if let Ok(path) = mouth_path.build() {
        window.paint_path(path, mouth_color);
    }

    // Nose (smell trait)
    let nose_radius = (body_radius * 0.12).max(1.0) * (0.6 + agent.trait_smell * 0.8);
    let nose_center = (
        px_x + forward.0 * (body_half_length - body_radius * 0.2),
        px_y + forward.1 * (body_half_length - body_radius * 0.2),
    );
    let mut nose = PathBuilder::fill();
    append_circle_polygon(&mut nose, nose_center.0, nose_center.1, nose_radius);
    if let Ok(path) = nose.build() {
        let mut nose_color = rgba_from_triplet_with_alpha(visuals.nose_color, 0.8);
        nose_color = palette_color(nose_color, palette, palette_is_natural);
        window.paint_path(path, nose_color);
    }

    // Ears / auditory fins
    let ear_scale = (0.6 + agent.trait_hearing * 0.45).clamp(0.6, 1.6);
    let ear_radius = (body_radius * 0.28).max(1.5) * ear_scale;
    let ear_offset = body_half_length * 0.15;
    let ear_center_left = (
        px_x + forward.0 * (-ear_offset) - right.0 * (body_radius + ear_radius * 0.45),
        px_y + forward.1 * (-ear_offset) - right.1 * (body_radius + ear_radius * 0.45),
    );
    let ear_center_right = (
        px_x + forward.0 * (-ear_offset) + right.0 * (body_radius + ear_radius * 0.45),
        px_y + forward.1 * (-ear_offset) + right.1 * (body_radius + ear_radius * 0.45),
    );
    let base_ear_color = rgba_from_triplet_with_alpha(visuals.stripe_color, 0.85);
    for center in [ear_center_left, ear_center_right] {
        let mut ear = PathBuilder::fill();
        append_circle_polygon(&mut ear, center.0, center.1, ear_radius);
        if let Ok(path) = ear.build() {
            let mut ear_color = scale_rgb(
                base_ear_color,
                0.9 + agent.trait_sound.clamp(0.1, 1.4) * 0.45,
            );
            ear_color = palette_color(ear_color, palette, palette_is_natural);
            window.paint_path(path, ear_color);
        }
    }

    // Eyes
    let base_eye_radius = (body_radius * 0.14).max(1.2);
    for i in 0..NUM_EYES {
        let dir_angle = agent.heading + agent.eye_dirs[i];
        let (sin_eye, cos_eye) = dir_angle.sin_cos();
        let distance = body_radius * (0.4 + 0.35 * (i as f32 / NUM_EYES as f32) + 0.25);
        let eye_center = (px_x + cos_eye * distance, px_y + sin_eye * distance);
        let mut eye_radius = base_eye_radius * (0.65 + agent.trait_eye.clamp(0.4, 2.5) * 0.35);
        let max_eye_radius = body_radius * 0.38;
        let min_eye_radius = 1.6_f32.min(max_eye_radius);
        eye_radius = eye_radius.min(max_eye_radius).max(min_eye_radius);

        let mut eye = PathBuilder::fill();
        append_circle_polygon(&mut eye, eye_center.0, eye_center.1, eye_radius);
        if let Ok(path) = eye.build() {
            let mut sclera_color =
                rgba_from_triplet_with_alpha(visual::visual_style().chrome.primary_text_srgb, 0.95);
            sclera_color = palette_color(sclera_color, palette, palette_is_natural);
            window.paint_path(path, sclera_color);
        }

        let pupil_radius = eye_radius * (0.35 + agent.eye_fov[i].clamp(0.3, 3.0) * 0.12);
        let mut pupil = PathBuilder::fill();
        append_circle_polygon(&mut pupil, eye_center.0, eye_center.1, pupil_radius);
        if let Ok(path) = pupil.build() {
            let mut pupil_color =
                rgba_from_triplet_with_alpha(visual::visual_style().substrate.abyss_srgb, 0.98);
            pupil_color = palette_color(pupil_color, palette, palette_is_natural);
            window.paint_path(path, pupil_color);
        }
    }

    // Vocalization arcs
    let vocal_level = agent.sound_output.max(agent.sound_level).clamp(0.0, 1.5);
    if vocal_level > 0.12 {
        let intensity = vocal_level.clamp(0.0, 1.0);
        let mouth_radius = (body_radius * 0.14).max(1.2) * (0.35 + intensity * 1.6);
        let arc_origin = (
            px_x + forward.0 * (body_half_length + mouth_radius * 0.6),
            px_y + forward.1 * (body_half_length + mouth_radius * 0.6),
        );
        for ring in 0..2 {
            let radius = mouth_radius * (1.1 + ring as f32 * 0.6);
            let mut arc = PathBuilder::stroke(px(1.0 + ring as f32));
            let start_deg = agent.heading.to_degrees() - 35.0 - ring as f32 * 8.0;
            let sweep = 70.0 + ring as f32 * 6.0;
            append_arc_polyline(
                &mut arc,
                arc_origin.0,
                arc_origin.1,
                radius,
                start_deg,
                sweep,
            );
            if let Ok(path) = arc.build() {
                let mut arc_color =
                    rgba_from_triplet_with_alpha(visuals.mouth_color, 0.35 + intensity * 0.35);
                arc_color = palette_color(arc_color, palette, palette_is_natural);
                window.paint_path(path, arc_color);
            }
        }
    }

    // Temperature preference marker
    let temp_pref = agent.temperature_preference.clamp(0.0, 1.0);
    let temperature_strength = ((temp_pref - 0.5).abs() * 2.0).clamp(0.0, 1.0);
    let mut temp_color = rgba_from_triplet_with_alpha(
        visuals.selection_rim_color,
        0.16 + temperature_strength * 0.22,
    );
    temp_color = palette_color(temp_color, palette, palette_is_natural);
    let temp_center = (
        px_x - forward.0 * (body_half_length * 0.25),
        px_y - forward.1 * (body_half_length * 0.25),
    );
    let temp_radius = (body_radius * 0.22).max(1.2);
    let mut temp_ring = PathBuilder::stroke(px(1.2));
    append_circle_polygon(&mut temp_ring, temp_center.0, temp_center.1, temp_radius);
    if let Ok(path) = temp_ring.build() {
        window.paint_path(path, temp_color);
    }
}

#[derive(Clone, Copy)]
struct AgentLodView {
    offset: (f32, f32),
    scale: f32,
    bounds: (f32, f32, f32, f32),
}

#[derive(Clone, Copy)]
struct AgentLodProjection {
    silhouette: AgentSilhouette,
    body_bin: usize,
    body_color: Rgba,
    spike_bin: usize,
    spike_color: Rgba,
    boost_bin: usize,
    boost_color: Option<Rgba>,
    marker_radius: f32,
    marker_color: Rgba,
    selection: SelectionState,
    focused: bool,
}

#[inline]
fn agent_screen_radius(frame: &RenderFrame, agent: &AgentRenderData, scale: f32) -> f32 {
    let radius_world = (frame.agent_base_radius + agent.spike_length * 0.25).max(8.0);
    (radius_world * scale).max(4.0)
}

fn project_agent_lod(
    frame: &RenderFrame,
    agent: &AgentRenderData,
    view: AgentLodView,
    focus_agent: Option<AgentId>,
    palette_is_natural: bool,
) -> Option<AgentLodProjection> {
    let center = (
        view.offset.0 + agent.position.x * view.scale,
        view.offset.1 + agent.position.y * view.scale,
    );
    let marker_radius = agent_screen_radius(frame, agent, view.scale);
    let visuals = resolve_agent_visual(agent, frame.agent_reference_age);
    let silhouette = agent_silhouette(
        center,
        marker_radius * 2.0,
        view.scale,
        &visuals,
        agent.boost,
        agent.sound_multiplier,
    );
    let cull_extent = if agent.spike_struck {
        silhouette.extent.max(marker_radius * 2.2)
    } else {
        silhouette.extent
    };
    let (view_left, view_top, view_right, view_bottom) = view.bounds;
    if center.0 + cull_extent < view_left
        || center.0 - cull_extent > view_right
        || center.1 + cull_extent < view_top
        || center.1 - cull_extent > view_bottom
    {
        return None;
    }

    let mut body_color = agent_color(&visuals);
    if !palette_is_natural {
        body_color = apply_palette(body_color, frame.palette);
    }
    let semantic_luma = visual::health_factor(agent.health)
        * visual::age_factor(u64::from(agent.age), frame.agent_reference_age);
    let diet_bin = ((agent.herbivore_tendency.clamp(0.0, 1.0) * AGENT_LOD_DIET_BINS as f32)
        as usize)
        .min(AGENT_LOD_DIET_BINS - 1);
    let luma_bin =
        ((semantic_luma * AGENT_LOD_LUMA_BINS as f32) as usize).min(AGENT_LOD_LUMA_BINS - 1);
    let body_bin = diet_bin * AGENT_LOD_LUMA_BINS + luma_bin;
    let spike_bin = ((visuals.spike_readiness * AGENT_LOD_SPIKE_BINS as f32) as usize)
        .min(AGENT_LOD_SPIKE_BINS - 1);

    let boost_color = silhouette
        .boost
        .map(|_| agent_boost_color(agent.boost, &visuals, frame.palette, palette_is_natural));
    let marker_alpha = match agent.selection {
        SelectionState::Selected => 0.86,
        SelectionState::Hovered => 0.62,
        SelectionState::None => 0.72,
    };
    let marker_color = palette_color(
        rgba_from_triplet_with_alpha(visuals.selection_rim_color, marker_alpha),
        frame.palette,
        palette_is_natural,
    );

    Some(AgentLodProjection {
        silhouette,
        body_bin,
        body_color,
        spike_bin,
        spike_color: agent_spike_color(&visuals, frame.palette, palette_is_natural),
        boost_bin: diet_bin,
        boost_color,
        marker_radius,
        marker_color,
        selection: agent.selection,
        focused: focus_agent == Some(agent.agent_id),
    })
}

fn accumulate_color(sum: &mut [f32; 4], count: &mut u32, color: Rgba) {
    sum[0] += color.r;
    sum[1] += color.g;
    sum[2] += color.b;
    sum[3] += color.a;
    *count += 1;
}

fn averaged_color(sum: [f32; 4], count: u32) -> Rgba {
    let denominator = count.max(1) as f32;
    Rgba {
        r: sum[0] / denominator,
        g: sum[1] / denominator,
        b: sum[2] / denominator,
        a: sum[3] / denominator,
    }
}

fn paint_color_buckets<const N: usize>(
    mut builders: [Option<PathBuilder>; N],
    sums: [[f32; 4]; N],
    counts: [u32; N],
    window: &mut Window,
) {
    for index in 0..N {
        if counts[index] == 0 {
            continue;
        }
        if let Some(builder) = builders[index].take()
            && let Ok(path) = builder.build()
        {
            window.paint_path(path, averaged_color(sums[index], counts[index]));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_agent_lod_batches(
    window: &mut Window,
    frame: &RenderFrame,
    focus_agent: Option<AgentId>,
    offset_x: f32,
    offset_y: f32,
    scale: f32,
    view_left: f32,
    view_top: f32,
    view_right: f32,
    view_bottom: f32,
    palette_is_natural: bool,
) {
    let view = AgentLodView {
        offset: (offset_x, offset_y),
        scale,
        bounds: (view_left, view_top, view_right, view_bottom),
    };
    let mut body_builders: [Option<PathBuilder>; AGENT_LOD_BODY_BINS] =
        std::array::from_fn(|_| Some(PathBuilder::fill()));
    let mut body_sums = [[0.0; 4]; AGENT_LOD_BODY_BINS];
    let mut body_counts = [0_u32; AGENT_LOD_BODY_BINS];
    let mut spike_builders: [Option<PathBuilder>; AGENT_LOD_SPIKE_BINS] =
        std::array::from_fn(|_| Some(PathBuilder::fill()));
    let mut spike_sums = [[0.0; 4]; AGENT_LOD_SPIKE_BINS];
    let mut spike_counts = [0_u32; AGENT_LOD_SPIKE_BINS];
    let mut boost_builders: [Option<PathBuilder>; AGENT_LOD_DIET_BINS] =
        std::array::from_fn(|_| Some(PathBuilder::fill()));
    let mut boost_sums = [[0.0; 4]; AGENT_LOD_DIET_BINS];
    let mut boost_counts = [0_u32; AGENT_LOD_DIET_BINS];
    let mut strike_builder = PathBuilder::fill();
    let mut has_strikes = false;
    let mut marker_builders: [Option<PathBuilder>; 3] = [
        Some(PathBuilder::stroke(px(2.4))),
        Some(PathBuilder::stroke(px(1.8))),
        Some(PathBuilder::stroke(px(2.8))),
    ];
    let mut marker_sums = [[0.0; 4]; 3];
    let mut marker_counts = [0_u32; 3];

    for agent in &frame.agents {
        let Some(projection) =
            project_agent_lod(frame, agent, view, focus_agent, palette_is_natural)
        else {
            continue;
        };
        let silhouette = projection.silhouette;
        if let Some(builder) = body_builders[projection.body_bin].as_mut() {
            append_capsule_polygon(
                builder,
                silhouette.center,
                silhouette.forward,
                silhouette.right,
                silhouette.body_half_length,
                silhouette.body_radius,
                6,
            );
        }
        accumulate_color(
            &mut body_sums[projection.body_bin],
            &mut body_counts[projection.body_bin],
            projection.body_color,
        );

        if let Some(builder) = spike_builders[projection.spike_bin].as_mut() {
            append_triangle(builder, silhouette.spike);
        }
        accumulate_color(
            &mut spike_sums[projection.spike_bin],
            &mut spike_counts[projection.spike_bin],
            projection.spike_color,
        );

        if let (Some(boost), Some(boost_color)) = (silhouette.boost, projection.boost_color) {
            if let Some(builder) = boost_builders[projection.boost_bin].as_mut() {
                append_triangle(builder, boost);
            }
            accumulate_color(
                &mut boost_sums[projection.boost_bin],
                &mut boost_counts[projection.boost_bin],
                boost_color,
            );
        }

        if agent.spike_struck {
            append_circle_polygon(
                &mut strike_builder,
                silhouette.center.0,
                silhouette.center.1,
                projection.marker_radius * 2.2,
            );
            has_strikes = true;
        }

        let marker = match projection.selection {
            SelectionState::Selected => Some((0, 1.85)),
            SelectionState::Hovered => Some((1, 1.45)),
            SelectionState::None => None,
        };
        if let Some((index, factor)) = marker {
            if let Some(builder) = marker_builders[index].as_mut() {
                append_circle_polygon(
                    builder,
                    silhouette.center.0,
                    silhouette.center.1,
                    projection.marker_radius * factor,
                );
            }
            accumulate_color(
                &mut marker_sums[index],
                &mut marker_counts[index],
                projection.marker_color,
            );
        }
        if projection.focused {
            if let Some(builder) = marker_builders[2].as_mut() {
                append_circle_polygon(
                    builder,
                    silhouette.center.0,
                    silhouette.center.1,
                    projection.marker_radius * 2.05,
                );
            }
            accumulate_color(
                &mut marker_sums[2],
                &mut marker_counts[2],
                projection.marker_color,
            );
        }
    }

    // Motion trails and strike flashes sit behind bodies; spikes and selection marks stay above.
    paint_color_buckets(boost_builders, boost_sums, boost_counts, window);
    if has_strikes && let Ok(path) = strike_builder.build() {
        let cue = visual::visual_cue_for_event(&WorldVisualEvent::SpikeExtend);
        let color = palette_color(
            rgba_from_triplet_with_alpha(cue.color, 0.28),
            frame.palette,
            palette_is_natural,
        );
        window.paint_path(path, color);
    }
    paint_color_buckets(body_builders, body_sums, body_counts, window);
    paint_color_buckets(spike_builders, spike_sums, spike_counts, window);
    paint_color_buckets(marker_builders, marker_sums, marker_counts, window);
}

fn paint_vector_hud(bounds: Bounds<Pixels>, state: &VectorHudState, window: &mut Window) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let width = f32::from(bounds_size.width).max(1.0);
    let height = f32::from(bounds_size.height).max(1.0);

    let backdrop = rgba_from_hex(0x091220, 0.88);
    window.paint_quad(fill(bounds, Background::from(backdrop)));

    let cx = f32::from(origin.x) + width * 0.5;
    let cy = f32::from(origin.y) + height * 0.52;
    let center = point(px(cx), px(cy));
    let radius = (width.min(height) * 0.36).max(18.0);

    let mut base_arc = PathBuilder::stroke(px(3.2));
    append_arc_polyline(&mut base_arc, cx, cy, radius, -140.0, 280.0);
    if let Ok(path) = base_arc.build() {
        window.paint_path(path, rgba_from_hex(0x142033, 0.95));
    }

    let progress_deg = 280.0 * state.population_ratio.clamp(0.0, 1.0);
    if progress_deg > 0.5 {
        let mut progress_arc = PathBuilder::stroke(px(4.2));
        append_arc_polyline(&mut progress_arc, cx, cy, radius, -140.0, progress_deg);
        if let Ok(path) = progress_arc.build() {
            let progress_color = lerp_rgba(
                rgba_from_hex(0x38bdf8, 0.95),
                rgba_from_hex(0x22c55e, 0.95),
                state.energy_ratio.clamp(0.0, 1.0),
            );
            window.paint_path(path, progress_color);
        }
    }

    let mut halo_arc = PathBuilder::stroke(px(1.6));
    append_arc_polyline(&mut halo_arc, cx, cy, radius * 1.08, -140.0, 280.0);
    if let Ok(path) = halo_arc.build() {
        window.paint_path(path, rgba_from_hex(0x1d3559, 0.25));
    }

    let energy_scale = state.energy_ratio.clamp(0.0, 1.0);
    let inner_radius = radius * (0.46 + energy_scale * 0.18);
    let mut inner_fill = PathBuilder::fill();
    append_circle_polygon(&mut inner_fill, cx, cy, inner_radius);
    if let Ok(path) = inner_fill.build() {
        let inner_color = lerp_rgba(
            rgba_from_hex(0x122033, 0.92),
            rgba_from_hex(0x3b82f6, 0.88),
            energy_scale,
        );
        window.paint_path(path, inner_color);
    }

    let pointer_deg = -140.0 + 280.0 * state.tick_phase;
    let pointer_rad = pointer_deg.to_radians();
    let pointer_radius = radius * 1.02;
    let px_pointer = cx + pointer_radius * pointer_rad.cos();
    let py_pointer = cy + pointer_radius * pointer_rad.sin();
    let mut pointer = PathBuilder::stroke(px(1.8));
    pointer.move_to(center);
    pointer.line_to(point(px(px_pointer), px(py_pointer)));
    if let Ok(path) = pointer.build() {
        window.paint_path(path, rgba_from_hex(0xfacc15, 0.75));
    }

    let velocity_ratio = if state.max_speed > f32::EPSILON {
        (state.vector_magnitude / state.max_speed).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if velocity_ratio > 0.01 {
        let heading = state.heading_rad;
        let arrow_length = radius * 0.85 * velocity_ratio;
        let tip_x = cx + arrow_length * heading.cos();
        let tip_y = cy + arrow_length * heading.sin();

        let mut arrow = PathBuilder::stroke(px(2.1));
        arrow.move_to(center);
        arrow.line_to(point(px(tip_x), px(tip_y)));
        if let Ok(path) = arrow.build() {
            window.paint_path(path, rgba_from_hex(0x38bdf8, 0.88));
        }

        let head_size = (8.0 + velocity_ratio * 18.0).clamp(6.0, 18.0);
        let left_angle = heading + PI - 0.4;
        let right_angle = heading + PI + 0.4;

        let left_point = point(
            px(tip_x + head_size * left_angle.cos()),
            px(tip_y + head_size * left_angle.sin()),
        );
        let right_point = point(
            px(tip_x + head_size * right_angle.cos()),
            px(tip_y + head_size * right_angle.sin()),
        );

        let mut left_head = PathBuilder::stroke(px(1.4));
        left_head.move_to(point(px(tip_x), px(tip_y)));
        left_head.line_to(left_point);
        if let Ok(path) = left_head.build() {
            window.paint_path(path, rgba_from_hex(0xe0f2fe, 0.82));
        }

        let mut right_head = PathBuilder::stroke(px(1.4));
        right_head.move_to(point(px(tip_x), px(tip_y)));
        right_head.line_to(right_point);
        if let Ok(path) = right_head.build() {
            window.paint_path(path, rgba_from_hex(0xe0f2fe, 0.82));
        }

        let coherence = (velocity_ratio * 100.0).clamp(0.0, 100.0);
        if coherence > 1.0 {
            let ring_inner = radius * 0.55;
            let mut ring = PathBuilder::stroke(px(1.2 + coherence * 0.02));
            append_arc_polyline(
                &mut ring,
                cx,
                cy,
                ring_inner,
                -140.0,
                280.0 * velocity_ratio,
            );
            if let Ok(path) = ring.build() {
                window.paint_path(path, rgba_from_hex(0x60a5fa, 0.65));
            }
        }
    }

    let bar_origin_x = f32::from(origin.x) + 12.0;
    let bar_origin_y = f32::from(origin.y) + height - 18.0;
    let bar_width = (width - 24.0).max(12.0);
    let bar_height = 6.0;
    let bar_bounds = Bounds::new(
        point(px(bar_origin_x), px(bar_origin_y)),
        size(px(bar_width), px(bar_height)),
    );
    window.paint_quad(fill(
        bar_bounds,
        Background::from(rgba_from_hex(0x132036, 0.95)),
    ));

    let births_width = bar_width * state.births_ratio.clamp(0.0, 1.0);
    if births_width > 0.5 {
        let births_bounds = Bounds::new(
            point(px(bar_origin_x), px(bar_origin_y)),
            size(px(births_width), px(bar_height * 0.55)),
        );
        window.paint_quad(fill(
            births_bounds,
            Background::from(rgba_from_hex(0x22c55e, 0.92)),
        ));
    }

    let deaths_width = bar_width * state.deaths_ratio.clamp(0.0, 1.0);
    if deaths_width > 0.5 {
        let deaths_bounds = Bounds::new(
            point(px(bar_origin_x), px(bar_origin_y + bar_height * 0.45)),
            size(px(deaths_width), px(bar_height * 0.55)),
        );
        window.paint_quad(fill(
            deaths_bounds,
            Background::from(rgba_from_hex(0xef4444, 0.92)),
        ));
    }

    let marker_x = bar_origin_x + bar_width * state.population_ratio.clamp(0.0, 1.0);
    let marker_bounds = Bounds::new(
        point(px(marker_x - 1.0), px(bar_origin_y - 4.0)),
        size(px(2.0), px(bar_height + 8.0)),
    );
    window.paint_quad(fill(
        marker_bounds,
        Background::from(rgba_from_hex(0x93c5fd, 0.65)),
    ));
}

#[allow(clippy::too_many_arguments)]
fn paint_terrain_layer(
    terrain: &TerrainFrame,
    offset_x: f32,
    offset_y: f32,
    scale: f32,
    daylight: f32,
    palette: ColorPaletteMode,
    view_left: f32,
    view_top: f32,
    view_right: f32,
    view_bottom: f32,
    window: &mut Window,
) {
    let width = terrain.dimensions.0 as usize;
    let height = terrain.dimensions.1 as usize;
    if width == 0 || height == 0 {
        return;
    }

    let cell_world = terrain.cell_size as f32;
    let cell_px = (cell_world * scale).max(1.0);
    let highlight_shift = (daylight * 0.45 + 0.35).clamp(0.2, 0.9);

    // Adaptive tiling: when zoomed out, render coarse blocks to reduce draw calls while
    // preserving the overall look. When zoomed in, fall back to fine detail per cell.
    const MAX_TERRAIN_QUADS_DEFAULT: usize = 140_000;
    const MAX_TERRAIN_QUADS_SAFE: usize = 40_000;
    let total_cells = width.saturating_mul(height).max(1);
    let max_quads = if safe_mode_enabled() {
        MAX_TERRAIN_QUADS_SAFE
    } else {
        MAX_TERRAIN_QUADS_DEFAULT
    };
    let stride_quads = ((total_cells as f32 / max_quads as f32).sqrt().ceil() as usize).max(1);
    let stride_pixels = if cell_px < 1.5 {
        (1.5 / cell_px).ceil() as usize
    } else {
        1
    };
    let stride = stride_quads.max(stride_pixels).clamp(1, 64);

    if stride > 1 {
        // Coarse block path: fill aggregated blocks using the top-left sample
        let block_px = (cell_px * stride as f32).max(1.0);
        for y in (0..height).step_by(stride) {
            for x in (0..width).step_by(stride) {
                let idx = y * width + x;
                let Some(tile) = terrain.tiles.get(idx).copied() else {
                    continue;
                };
                let px_x = offset_x + (x as f32 * cell_world * scale);
                let px_y = offset_y + (y as f32 * cell_world * scale);
                if px_x > view_right
                    || px_y > view_bottom
                    || px_x + block_px < view_left
                    || px_y + block_px < view_top
                {
                    continue;
                }
                let surface = terrain_surface_color(tile, daylight, palette);
                let bounds =
                    Bounds::new(point(px(px_x), px(px_y)), size(px(block_px), px(block_px)));
                window.paint_quad(fill(bounds, Background::from(surface)));
            }
        }
        return;
    }

    // Fine path: per-cell render with culling and reused px conversions
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let Some(tile) = terrain.tiles.get(idx).copied() else {
                continue;
            };

            let px_x = offset_x + (x as f32 * cell_world * scale);
            let px_y = offset_y + (y as f32 * cell_world * scale);
            if px_x > view_right
                || px_y > view_bottom
                || px_x + cell_px < view_left
                || px_y + cell_px < view_top
            {
                continue;
            }

            let surface = terrain_surface_color(tile, daylight, palette);
            let px_xu = px(px_x);
            let px_yu = px(px_y);
            let cell_bounds = Bounds::new(point(px_xu, px_yu), size(px(cell_px), px(cell_px)));
            window.paint_quad(fill(cell_bounds, Background::from(surface)));

            if tile.slope > 0.12 {
                let stroke_width = (0.55 + tile.slope * 1.1) * scale.clamp(0.6, 3.0);
                let accent = terrain_slope_accent_color(tile, highlight_shift, palette);
                let mut builder = PathBuilder::stroke(px(stroke_width.min(cell_px * 0.85)));
                let diag_bias = if tile.accent > 0.5 {
                    (0.35, 0.9)
                } else {
                    (0.1, 0.65)
                };
                builder.move_to(point(px(px_x + cell_px * diag_bias.0), px(px_y)));
                builder.line_to(point(px(px_x), px(px_y + cell_px * diag_bias.1)));
                if let Ok(path) = builder.build() {
                    window.paint_path(path, accent);
                }
            }

            if matches!(tile.kind, TerrainKind::Bloom) && tile.accent > 0.66 {
                let blossom = terrain_bloom_color(tile, palette);
                let bloom_size = (cell_px * (0.25 + tile.accent * 0.18)).min(cell_px * 0.6);
                let half = bloom_size * 0.5;
                let bloom_bounds = Bounds::new(
                    point(
                        px(px_x + cell_px * 0.5 - half),
                        px(px_y + cell_px * 0.5 - half),
                    ),
                    size(px(bloom_size), px(bloom_size)),
                );
                window.paint_quad(fill(bloom_bounds, Background::from(blossom)));
            }

            if matches!(
                tile.kind,
                TerrainKind::ShallowWater | TerrainKind::DeepWater
            ) {
                let caustic = terrain_water_caustic_color(tile, daylight, palette);
                let wave_height =
                    (cell_px * 0.12 + tile.accent * cell_px * 0.18).min(cell_px * 0.35);
                let mut builder =
                    PathBuilder::stroke(px((cell_px * 0.08 + tile.accent * 0.18).clamp(0.5, 2.6)));
                builder.move_to(point(px(px_x), px(px_y + wave_height)));
                builder.line_to(point(
                    px(px_x + cell_px),
                    px(px_y + wave_height * 0.45 + tile.accent * 0.22 * cell_px),
                ));
                if let Ok(path) = builder.build() {
                    window.paint_path(path, caustic);
                }
            }
        }
    }
}

// Kept as literal-free compatibility aliases while the old helper names are
// retired call-site by call-site. Core visual semantics own every value.
const LEGACY_TERRAIN_BASE: [[f32; 3]; 6] = visual::TERRAIN_BASE_COLORS;
const LEGACY_TERRAIN_ACCENT: [[f32; 3]; 6] = [
    visual::BIOLUMINESCENT_DARK_FIELD_V1.terrain[0].emissive_srgb,
    visual::BIOLUMINESCENT_DARK_FIELD_V1.terrain[1].emissive_srgb,
    visual::BIOLUMINESCENT_DARK_FIELD_V1.terrain[2].emissive_srgb,
    visual::BIOLUMINESCENT_DARK_FIELD_V1.terrain[3].emissive_srgb,
    visual::BIOLUMINESCENT_DARK_FIELD_V1.terrain[4].emissive_srgb,
    visual::BIOLUMINESCENT_DARK_FIELD_V1.terrain[5].emissive_srgb,
];

#[inline]
fn terrain_kind_index(kind: TerrainKind) -> usize {
    match kind {
        TerrainKind::DeepWater => 0,
        TerrainKind::ShallowWater => 1,
        TerrainKind::Sand => 2,
        TerrainKind::Grass => 3,
        TerrainKind::Bloom => 4,
        TerrainKind::Rock => 5,
    }
}

#[inline]
#[allow(dead_code)]
fn terrain_base_color(kind: TerrainKind) -> [f32; 3] {
    LEGACY_TERRAIN_BASE[terrain_kind_index(kind)]
}

#[inline]
fn terrain_accent_color(kind: TerrainKind) -> [f32; 3] {
    LEGACY_TERRAIN_ACCENT[terrain_kind_index(kind)]
}

fn terrain_surface_color(
    tile: TerrainTileVisual,
    daylight: f32,
    palette: ColorPaletteMode,
) -> Rgba {
    let weights = visual::splat_weights(&SplatInput {
        kind: tile.kind,
        elevation: tile.elevation,
        slope: tile.slope,
        water_depth: tile.water_depth,
    });
    let srgb = visual::terrain_surface_srgb(&TerrainSurfaceInput {
        splat_weights: weights,
        moisture: tile.moisture,
        elevation: tile.elevation,
        slope: tile.slope,
        accent: tile.accent,
        daylight,
        accessibility: accessibility_palette(palette),
    });
    rgba_from_triplet_with_alpha(srgb, 1.0)
}

fn canonical_gpu_terrain_colors(frame: &RenderFrame) -> Vec<[f32; 4]> {
    let daylight = visual::daylight_factor(
        frame.tick,
        frame.day_night_cycle_ticks,
        frame.day_night_start_phase,
    );
    frame
        .terrain
        .tiles
        .iter()
        .map(|tile| {
            let color = terrain_surface_color(*tile, daylight, frame.palette);
            [color.r, color.g, color.b, color.a]
        })
        .collect()
}

const WORLD_RASTER_SAMPLES_PER_TERRAIN_CELL: u32 = 4;
const MAX_WORLD_RASTER_TEXELS: u64 = 1_048_576;

fn world_raster_key(frame: &RenderFrame, daylight: f32) -> WorldRasterKey {
    WorldRasterKey {
        field_fingerprint: world_field_fingerprint(frame),
        tick: frame.tick,
        daylight_bits: daylight.to_bits(),
        palette: frame.palette,
    }
}

fn world_field_fingerprint(frame: &RenderFrame) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    fn push(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    let mut hash = FNV_OFFSET;
    for value in [
        frame.terrain.dimensions.0,
        frame.terrain.dimensions.1,
        frame.terrain.cell_size,
    ] {
        push(&mut hash, &value.to_le_bytes());
    }
    for tile in &frame.terrain.tiles {
        push(&mut hash, &[terrain_kind_index(tile.kind) as u8]);
        for value in [
            tile.elevation,
            tile.moisture,
            tile.fertility_bias,
            tile.accent,
            tile.slope,
            tile.water_depth,
            tile.elevation_gradient[0],
            tile.elevation_gradient[1],
        ] {
            push(&mut hash, &value.to_bits().to_le_bytes());
        }
    }
    hash
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn world_raster_dimensions(terrain: &TerrainFrame) -> Option<(u32, u32)> {
    let (grid_width, grid_height) = terrain.dimensions;
    if grid_width == 0 || grid_height == 0 || terrain.tiles.is_empty() {
        return None;
    }

    let mut width = grid_width
        .saturating_mul(WORLD_RASTER_SAMPLES_PER_TERRAIN_CELL)
        .max(1);
    let mut height = grid_height
        .saturating_mul(WORLD_RASTER_SAMPLES_PER_TERRAIN_CELL)
        .max(1);
    let texels = u64::from(width).saturating_mul(u64::from(height));
    if texels > MAX_WORLD_RASTER_TEXELS {
        let scale = (MAX_WORLD_RASTER_TEXELS as f64 / texels as f64).sqrt();
        width = ((f64::from(width) * scale).floor() as u32).max(1);
        height = ((f64::from(height) * scale).floor() as u32).max(1);
    }
    Some((width, height))
}

#[inline]
fn blend_scalar(values: &[f32], corners: &visual::TerrainSampleCorners) -> f32 {
    corners
        .indices
        .iter()
        .zip(corners.weights)
        .map(|(index, weight)| values.get(*index).copied().unwrap_or(0.0) * weight)
        .sum()
}

#[inline]
fn blend_channels<const N: usize>(
    values: &[[f32; N]],
    corners: &visual::TerrainSampleCorners,
) -> [f32; N] {
    let mut blended = [0.0; N];
    for (index, weight) in corners.indices.iter().zip(corners.weights) {
        if let Some(value) = values.get(*index) {
            for channel in 0..N {
                blended[channel] += value[channel] * weight;
            }
        }
    }
    blended
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn rasterize_world_fields(frame: &RenderFrame, daylight: f32) -> Option<WorldRasterPixels> {
    let (raster_width, raster_height) = world_raster_dimensions(&frame.terrain)?;
    let expected_tiles =
        (frame.terrain.dimensions.0 as usize).saturating_mul(frame.terrain.dimensions.1 as usize);
    if frame.terrain.tiles.len() < expected_tiles {
        return None;
    }

    let mut kinds = Vec::with_capacity(expected_tiles);
    let mut moisture = Vec::with_capacity(expected_tiles);
    let mut fertility = Vec::with_capacity(expected_tiles);
    let mut elevation = Vec::with_capacity(expected_tiles);
    let mut slope = Vec::with_capacity(expected_tiles);
    let mut water_depth = Vec::with_capacity(expected_tiles);
    let mut accent = Vec::with_capacity(expected_tiles);
    let mut gradients = Vec::with_capacity(expected_tiles);
    let mut splats = Vec::with_capacity(expected_tiles);
    for tile in frame.terrain.tiles.iter().take(expected_tiles) {
        kinds.push(tile.kind);
        moisture.push(tile.moisture);
        fertility.push(tile.fertility_bias);
        elevation.push(tile.elevation);
        slope.push(tile.slope);
        water_depth.push(tile.water_depth);
        accent.push(tile.accent);
        gradients.push(tile.elevation_gradient);
        splats.push(visual::splat_weights(&SplatInput {
            kind: tile.kind,
            elevation: tile.elevation,
            slope: tile.slope,
            water_depth: tile.water_depth,
        }));
    }

    let terrain_field = visual::TerrainFieldView {
        width: frame.terrain.dimensions.0,
        height: frame.terrain.dimensions.1,
        cell_size: frame.terrain.cell_size as f32,
        kinds: &kinds,
        moisture: &moisture,
        elevation: &elevation,
        slope: &slope,
        water_depth: &water_depth,
    };
    let mut bgra = Vec::with_capacity(
        (raster_width as usize)
            .saturating_mul(raster_height as usize)
            .saturating_mul(4),
    );

    for raster_y in 0..raster_height {
        let world_y = (raster_y as f32 + 0.5) / raster_height as f32 * frame.world_size.1;
        for raster_x in 0..raster_width {
            let world_x = (raster_x as f32 + 0.5) / raster_width as f32 * frame.world_size.0;
            let corners = terrain_field.sample_corners(world_x, world_y);
            let weights = blend_channels(&splats, &corners);
            let sampled_moisture = blend_scalar(&moisture, &corners);
            let _sampled_fertility = blend_scalar(&fertility, &corners);
            let sampled_elevation = blend_scalar(&elevation, &corners);
            let sampled_slope = blend_scalar(&slope, &corners);
            let sampled_accent = blend_scalar(&accent, &corners);
            let _sampled_gradient = blend_channels(&gradients, &corners);

            let display_rgb = visual::terrain_surface_srgb(&TerrainSurfaceInput {
                splat_weights: weights,
                moisture: sampled_moisture,
                elevation: sampled_elevation,
                slope: sampled_slope,
                accent: sampled_accent,
                daylight,
                accessibility: accessibility_palette(frame.palette),
            });

            let to_byte = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            bgra.extend_from_slice(&[
                to_byte(display_rgb[2]),
                to_byte(display_rgb[1]),
                to_byte(display_rgb[0]),
                u8::MAX,
            ]);
        }
    }

    Some(WorldRasterPixels {
        width: raster_width,
        height: raster_height,
        bgra,
    })
}

fn render_image_from_world_raster(pixels: WorldRasterPixels) -> Option<Arc<RenderImage>> {
    let buffer = ImageBuffer::<ImgRgba<u8>, _>::from_raw(pixels.width, pixels.height, pixels.bgra)?;
    Some(Arc::new(RenderImage::new(vec![ImageFrame::new(buffer)])))
}

#[allow(clippy::too_many_arguments)]
fn paint_continuous_world_fields(
    state: &CanvasState,
    offset_x: f32,
    offset_y: f32,
    scale: f32,
    daylight: f32,
    window: &mut Window,
) -> bool {
    let frame = &state.frame;
    let key = world_raster_key(frame, daylight);
    let cached = match state.world_raster_cache.lock() {
        Ok(cache) => cache.image_for(key),
        Err(poisoned) => poisoned.into_inner().image_for(key),
    };
    let (image, newly_built) = if let Some(image) = cached {
        (image, false)
    } else {
        let Some(pixels) = rasterize_world_fields(frame, daylight) else {
            return false;
        };
        let Some(image) = render_image_from_world_raster(pixels) else {
            return false;
        };
        (image, true)
    };

    let world_bounds = Bounds::new(
        point(px(offset_x), px(offset_y)),
        size(
            px((frame.world_size.0 * scale).max(1.0)),
            px((frame.world_size.1 * scale).max(1.0)),
        ),
    );
    let paint_result = window.paint_layer(world_bounds, |window| {
        window.paint_image(
            world_bounds,
            Corners::default(),
            Arc::clone(&image),
            0,
            false,
        )
    });

    if let Err(error) = paint_result {
        if newly_built {
            let _ = window.drop_image(image);
        }
        warn!(
            ?error,
            "continuous world raster upload failed; using cell fallback"
        );
        return false;
    }

    if newly_built {
        let retired = match state.world_raster_cache.lock() {
            Ok(mut cache) => cache.commit(key, image),
            Err(poisoned) => poisoned.into_inner().commit(key, image),
        };
        if let Some(retired) = retired
            && let Err(error) = window.drop_image(retired)
        {
            debug!(?error, "could not evict retired world raster image");
        }
    }
    true
}

#[cfg(test)]
mod continuous_world_raster_tests {
    use super::*;

    fn tile(kind: TerrainKind) -> TerrainTileVisual {
        TerrainTileVisual {
            kind,
            elevation: 0.5,
            moisture: 0.5,
            fertility_bias: 0.5,
            accent: 0.25,
            slope: 0.0,
            water_depth: 0.0,
            elevation_gradient: [0.0, 0.0],
        }
    }

    fn frame(
        dimensions: (u32, u32),
        cell_size: u32,
        tiles: Vec<TerrainTileVisual>,
        food_dimensions: (u32, u32),
        food_cell_size: u32,
        food_cells: Vec<f32>,
    ) -> RenderFrame {
        RenderFrame {
            tick: 17,
            tonemap_mode: None,
            day_night_cycle_ticks: 0,
            day_night_start_phase: 0.25,
            world_size: (
                dimensions.0 as f32 * cell_size as f32,
                dimensions.1 as f32 * cell_size as f32,
            ),
            terrain: TerrainFrame {
                dimensions,
                cell_size,
                tiles,
            },
            food_dimensions,
            food_cell_size,
            food_cells,
            food_max: 1.0,
            agents: Vec::new(),
            agent_reference_age: 1,
            agent_base_radius: 1.0,
            sense_radius: 1.0,
            post_stack: PostProcessStack { passes: Vec::new() },
            palette: ColorPaletteMode::Natural,
        }
    }

    fn rgb_at(pixels: &WorldRasterPixels, x: u32, y: u32) -> [f32; 3] {
        let offset = ((y * pixels.width + x) * 4) as usize;
        [
            f32::from(pixels.bgra[offset + 2]) / 255.0,
            f32::from(pixels.bgra[offset + 1]) / 255.0,
            f32::from(pixels.bgra[offset]) / 255.0,
        ]
    }

    fn rgb_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
        (left[0] - right[0]).abs() + (left[1] - right[1]).abs() + (left[2] - right[2]).abs()
    }

    #[test]
    fn canonical_gpu_terrain_colors_use_the_core_surface_oracle_for_every_palette() {
        let mut grass = tile(TerrainKind::Grass);
        grass.elevation = 0.91;
        grass.moisture = 0.37;
        grass.accent = 0.63;
        grass.slope = 0.82;
        grass.water_depth = 1.5;

        let mut sand = tile(TerrainKind::Sand);
        sand.elevation = 0.08;
        sand.moisture = 0.14;
        sand.accent = 0.77;
        sand.slope = 0.18;

        let mut frame = frame((2, 1), 10, vec![grass, sand], (1, 1), 20, vec![0.0]);
        frame.tick = 137;
        frame.day_night_cycle_ticks = 480;
        frame.day_night_start_phase = 0.17;
        let daylight = visual::daylight_factor(
            frame.tick,
            frame.day_night_cycle_ticks,
            frame.day_night_start_phase,
        );

        for palette in ColorPaletteMode::ALL {
            frame.palette = palette;
            let actual = canonical_gpu_terrain_colors(&frame);
            assert_eq!(actual.len(), frame.terrain.tiles.len());
            for (index, (tile, actual)) in
                frame.terrain.tiles.iter().copied().zip(actual).enumerate()
            {
                let expected = visual::terrain_surface_srgb(&TerrainSurfaceInput {
                    splat_weights: visual::splat_weights(&SplatInput {
                        kind: tile.kind,
                        elevation: tile.elevation,
                        slope: tile.slope,
                        water_depth: tile.water_depth,
                    }),
                    moisture: tile.moisture,
                    elevation: tile.elevation,
                    slope: tile.slope,
                    accent: tile.accent,
                    daylight,
                    accessibility: accessibility_palette(palette),
                });
                assert_eq!(
                    actual.map(f32::to_bits),
                    [expected[0], expected[1], expected[2], 1.0].map(f32::to_bits),
                    "tile {index} diverged from the core semantic-sRGB oracle for {palette:?}"
                );
            }
        }
    }

    #[test]
    fn canonical_gpu_terrain_projection_preserves_the_world_digest() {
        let config = ScriptBotsConfig {
            world_width: 96,
            world_height: 64,
            food_cell_size: 16,
            initial_food: 0.25,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            rng_seed: Some(0xBAAC_E11D),
            ..ScriptBotsConfig::default()
        };
        let world = WorldState::new(config).expect("backend-agreement world");
        let before = world
            .world_digest_v1()
            .expect("pre-projection canonical world digest");

        for palette in ColorPaletteMode::ALL {
            let frame =
                RenderFrame::from_world(&world, palette).expect("backend-agreement render frame");
            let colors = canonical_gpu_terrain_colors(&frame);
            assert_eq!(
                colors.len(),
                frame.terrain.tiles.len(),
                "every semantic terrain tile must reach the GPU boundary"
            );
        }

        assert_eq!(
            world
                .world_digest_v1()
                .expect("post-projection canonical world digest"),
            before,
            "backend comparison and terrain projection must remain science-neutral"
        );
    }

    #[test]
    fn continuous_raster_blends_biome_boundaries_and_toroidal_seam() {
        let frame = frame(
            (2, 1),
            10,
            vec![tile(TerrainKind::DeepWater), tile(TerrainKind::Bloom)],
            (1, 1),
            20,
            vec![0.0],
        );
        let pixels = rasterize_world_fields(&frame, visual::DAYLIGHT_STATIC).expect("world raster");

        let left = rgb_at(&pixels, 1, 1);
        let right = rgb_at(&pixels, 5, 1);
        let interior_step = rgb_distance(rgb_at(&pixels, 3, 1), rgb_at(&pixels, 4, 1));
        let seam_step = rgb_distance(rgb_at(&pixels, 7, 1), rgb_at(&pixels, 0, 1));
        let endpoint_distance = rgb_distance(left, right);

        assert!(
            endpoint_distance > 0.05,
            "fixture must contain visibly different biome endpoints"
        );
        assert!(
            interior_step < endpoint_distance * 0.6,
            "the cell boundary remained a hard color step: {interior_step} vs endpoints \
             {endpoint_distance}"
        );
        assert!(
            seam_step < endpoint_distance * 0.6,
            "the wrapped seam remained a hard color step: {seam_step} vs endpoints \
             {endpoint_distance}"
        );
    }

    #[test]
    fn terrain_raster_does_not_absorb_food_overlay() {
        let terrain = vec![tile(TerrainKind::Grass); 4];
        let with_food = frame((4, 1), 10, terrain.clone(), (2, 1), 20, vec![1.0, 0.0]);
        let without_food = frame((4, 1), 10, terrain, (2, 1), 20, vec![0.0, 0.0]);
        let food_scene =
            rasterize_world_fields(&with_food, visual::DAYLIGHT_STATIC).expect("terrain raster");
        let empty_scene =
            rasterize_world_fields(&without_food, visual::DAYLIGHT_STATIC).expect("terrain raster");

        assert_eq!(
            food_scene.bgra, empty_scene.bgra,
            "food belongs to the discrete emissive overlay, not the smoothed terrain image"
        );
    }

    fn image(byte: u8) -> Arc<RenderImage> {
        render_image_from_world_raster(WorldRasterPixels {
            width: 1,
            height: 1,
            bgra: vec![byte, byte, byte, u8::MAX],
        })
        .expect("one-pixel render image")
    }

    fn key(tick: u64) -> WorldRasterKey {
        WorldRasterKey {
            field_fingerprint: tick,
            tick,
            daylight_bits: visual::DAYLIGHT_STATIC.to_bits(),
            palette: ColorPaletteMode::Natural,
        }
    }

    #[test]
    fn raster_cache_reuses_keys_and_retires_only_two_generations_old() {
        let mut cache = WorldRasterCache::default();
        let a = image(1);
        let b = image(2);
        let c = image(3);

        assert!(cache.commit(key(1), Arc::clone(&a)).is_none());
        assert!(Arc::ptr_eq(&cache.image_for(key(1)).expect("cached A"), &a));
        assert!(cache.commit(key(2), Arc::clone(&b)).is_none());
        let retired = cache.commit(key(3), Arc::clone(&c)).expect("retired A");
        assert!(Arc::ptr_eq(&retired, &a));

        let drained = cache.drain();
        assert_eq!(drained.len(), 2);
        assert!(drained.iter().any(|image| image.id == b.id));
        assert!(drained.iter().any(|image| image.id == c.id));
    }
}

fn terrain_slope_accent_color(
    tile: TerrainTileVisual,
    highlight_shift: f32,
    palette: ColorPaletteMode,
) -> Rgba {
    let material = visual::terrain_material(tile.kind);
    let accent = terrain_accent_color(tile.kind);
    let alpha = (0.06 + tile.slope * highlight_shift * material.normal_strength).clamp(0.03, 0.36);
    let mut color = rgba_from_triplet_with_alpha(accent, alpha);
    color = scale_rgb(color, 1.0 + material.emissive_gain + tile.accent * 0.25);
    apply_palette(color, palette)
}

fn terrain_bloom_color(tile: TerrainTileVisual, palette: ColorPaletteMode) -> Rgba {
    let material = visual::terrain_material(TerrainKind::Bloom);
    let strength = ((tile.accent - 0.66) * 1.6).clamp(0.0, 1.0);
    let alpha = (0.12 + strength * 0.28).clamp(0.08, 0.35);
    let mut color = rgba_from_triplet_with_alpha(material.emissive_srgb, alpha);
    color = scale_rgb(color, 1.0 + material.emissive_gain + tile.moisture * 0.25);
    apply_palette(color, palette)
}

fn terrain_water_caustic_color(
    tile: TerrainTileVisual,
    daylight: f32,
    palette: ColorPaletteMode,
) -> Rgba {
    let material = visual::terrain_material(tile.kind);
    let alpha = (0.05 + daylight * 0.08 + tile.accent * 0.12 + material.reflectance * 0.10)
        .clamp(0.04, 0.26);
    let mut color = rgba_from_triplet_with_alpha(material.emissive_srgb, alpha);
    color = scale_rgb(color, 1.0 + material.emissive_gain + tile.moisture * 0.15);
    apply_palette(color, palette)
}
fn paint_sparkline(bounds: Bounds<Pixels>, state: SparklineState, window: &mut Window) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let width = f32::from(bounds_size.width).max(1.0);
    let height = f32::from(bounds_size.height).max(1.0);
    let samples = state.values.len();
    if samples < 2 {
        return;
    }
    let step = width / (samples.saturating_sub(1) as f32);

    let mut fill_builder = PathBuilder::fill();
    fill_builder.move_to(point(
        px(f32::from(origin.x)),
        px(f32::from(origin.y) + height),
    ));
    for (idx, value) in state.values.iter().enumerate() {
        let x = f32::from(origin.x) + step * idx as f32;
        let y = f32::from(origin.y) + height - value.clamp(0.0, 1.0) * height;
        fill_builder.line_to(point(px(x), px(y)));
    }
    fill_builder.line_to(point(
        px(f32::from(origin.x) + width),
        px(f32::from(origin.y) + height),
    ));
    fill_builder.close();
    if let Ok(path) = fill_builder.build() {
        let mut fill_color = state.accent;
        fill_color.a = if state.trend >= 0.0 { 0.18 } else { 0.12 };
        window.paint_path(path, fill_color);
    }

    let mut stroke_builder = PathBuilder::stroke(px(1.6));
    for (idx, value) in state.values.iter().enumerate() {
        let x = f32::from(origin.x) + step * idx as f32;
        let y = f32::from(origin.y) + height - value.clamp(0.0, 1.0) * height;
        if idx == 0 {
            stroke_builder.move_to(point(px(x), px(y)));
        } else {
            stroke_builder.line_to(point(px(x), px(y)));
        }
    }
    if let Ok(path) = stroke_builder.build() {
        let mut stroke_color = state.accent;
        stroke_color.a = 0.85;
        window.paint_path(path, stroke_color);
    }

    let marker_value = state.values.last().copied().unwrap_or(0.5).clamp(0.0, 1.0);
    let marker_size = 4.0;
    let marker_x = f32::from(origin.x) + width;
    let marker_y = f32::from(origin.y) + height - marker_value * height;
    let marker_bounds = Bounds::new(
        point(
            px(marker_x - marker_size * 0.5),
            px(marker_y - marker_size * 0.5),
        ),
        size(px(marker_size), px(marker_size)),
    );
    let mut marker_color = state.accent;
    marker_color.a = 1.0;
    window.paint_quad(fill(marker_bounds, Background::from(marker_color)));
}

fn paint_metric_badge(bounds: Bounds<Pixels>, state: MetricBadgeState, window: &mut Window) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let width = f32::from(bounds_size.width).max(1.0);
    let height = f32::from(bounds_size.height).max(1.0);
    let center_x = f32::from(origin.x) + width * 0.5;
    let center_y = f32::from(origin.y) + height * 0.5;
    let radius = width.min(height) * 0.5;

    let mut hex_builder = PathBuilder::fill();
    for step_idx in 0..6 {
        let angle = (PI / 3.0) * step_idx as f32 - FRAC_PI_2;
        let x = center_x + angle.cos() * radius;
        let y = center_y + angle.sin() * radius;
        if step_idx == 0 {
            hex_builder.move_to(point(px(x), px(y)));
        } else {
            hex_builder.line_to(point(px(x), px(y)));
        }
    }
    hex_builder.close();
    if let Ok(path) = hex_builder.build() {
        let mut outer = state.accent;
        outer.a = 0.45;
        window.paint_path(path, outer);
    }

    let inner_radius = radius * 0.58;
    let mut diamond = PathBuilder::fill();
    for idx in 0..4 {
        let angle = FRAC_PI_2 * idx as f32 + FRAC_PI_4;
        let x = center_x + angle.cos() * inner_radius;
        let y = center_y + angle.sin() * inner_radius;
        if idx == 0 {
            diamond.move_to(point(px(x), px(y)));
        } else {
            diamond.line_to(point(px(x), px(y)));
        }
    }
    diamond.close();
    if let Ok(path) = diamond.build() {
        let mut inner = state.accent;
        inner.a = 0.9;
        window.paint_path(path, inner);
    }

    let bar_width = inner_radius * 0.36;
    let bar_bounds = Bounds::new(
        point(px(center_x - bar_width * 0.5), px(center_y - inner_radius)),
        size(px(bar_width), px(inner_radius * 2.0)),
    );
    let mut bar_color = state.accent;
    bar_color.a = 0.65;
    window.paint_quad(fill(bar_bounds, Background::from(bar_color)));
}

fn paint_header_badge(bounds: Bounds<Pixels>, state: HeaderBadgeState, window: &mut Window) {
    let origin = bounds.origin;
    let size = bounds.size;
    let cx = f32::from(origin.x) + f32::from(size.width) * 0.5;
    let cy = f32::from(origin.y) + f32::from(size.height) * 0.5;
    let radius = f32::from(size.width.min(size.height)) * 0.5 - 1.5;

    let base = apply_palette(rgba_from_hex(0x0f172a, 1.0), state.palette);
    window.paint_quad(fill(bounds, Background::from(base)));

    let phase = state.phase;
    let glow_radius = radius * 0.9;
    for i in 0..5 {
        let t = i as f32 / 5.0;
        let angle = phase + t * std::f32::consts::TAU;
        let px = cx + angle.cos() * glow_radius;
        let py = cy + angle.sin() * glow_radius;
        let orb_radius = 6.0 + (angle.sin() * 2.0);
        let mut orb = PathBuilder::fill();
        append_circle_polygon(&mut orb, px, py, orb_radius);
        if let Ok(path) = orb.build() {
            let color = apply_palette(rgba_from_hex(0x60a5fa, 0.18 + t * 0.2), state.palette);
            window.paint_path(path, color);
        }
    }

    let mut ring = PathBuilder::stroke(px(3.0));
    append_arc_polyline(&mut ring, cx, cy, radius, 0.0, 360.0);
    if let Ok(path) = ring.build() {
        let ring_color = apply_palette(rgba_from_hex(0x38bdf8, 0.85), state.palette);
        window.paint_path(path, ring_color);
    }

    let mut inner = PathBuilder::fill();
    append_circle_polygon(&mut inner, cx, cy, radius * 0.55);
    if let Ok(path) = inner.build() {
        let inner_color = apply_palette(rgba_from_hex(0x1d4ed8, 0.5), state.palette);
        window.paint_path(path, inner_color);
    }

    let mut pulse = PathBuilder::stroke(px(1.6));
    let sweep = 140.0 + (phase.sin() + 1.0) * 100.0;
    append_arc_polyline(&mut pulse, cx, cy, radius * 0.68, -sweep * 0.5, sweep);
    if let Ok(path) = pulse.build() {
        let pulse_color = apply_palette(rgba_from_hex(0xfacc15, 0.75), state.palette);
        window.paint_path(path, pulse_color);
    }
}
fn apply_post_processing(
    stack: &PostProcessStack,
    palette: ColorPaletteMode,
    bounds: Bounds<Pixels>,
    window: &mut Window,
    daylight: f32,
    scale: f32,
    low_fps: bool,
) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let width_px = f32::from(bounds_size.width).max(1.0);
    let height_px = f32::from(bounds_size.height).max(1.0);
    let origin_x = f32::from(origin.x);
    let origin_y = f32::from(origin.y);

    for pass in &stack.passes {
        match *pass {
            PostProcessPass::Exposure { factor } => {
                if factor > 1.001 {
                    let alpha = ((factor - 1.0) * 0.24).clamp(0.0, 0.36);
                    let exposure_base = rgba_from_triplet_with_alpha(
                        visual::visual_style().chrome.primary_text_srgb,
                        alpha,
                    );
                    window.paint_quad(fill(
                        bounds,
                        Background::from(apply_palette(exposure_base, palette)),
                    ));
                } else if factor < 0.999 {
                    let alpha = ((1.0 - factor) * 0.42).clamp(0.0, 0.55);
                    let exposure_base = rgba_from_triplet_with_alpha(
                        visual::visual_style().substrate.abyss_srgb,
                        alpha,
                    );
                    window.paint_quad(fill(
                        bounds,
                        Background::from(apply_palette(exposure_base, palette)),
                    ));
                }
            }
            PostProcessPass::ColorGrade {
                lift,
                gain,
                temperature,
            } => {
                if lift > 0.001 {
                    let lift_color =
                        apply_palette(rgba_from_hex(0x02060f, lift.clamp(0.0, 0.2)), palette);
                    window.paint_quad(fill(bounds, Background::from(lift_color)));
                }
                if temperature.abs() > 0.001 {
                    let style = visual::visual_style();
                    let temp_rgb = if temperature >= 0.0 {
                        style.chrome.warning_srgb
                    } else {
                        style.chrome.accent_cyan_srgb
                    };
                    let temp_alpha = temperature.abs().clamp(0.0, 0.25) * 0.6;
                    if temp_alpha > 0.0 {
                        let temp_color = apply_palette(
                            rgba_from_triplet_with_alpha(temp_rgb, temp_alpha),
                            palette,
                        );
                        window.paint_quad(fill(bounds, Background::from(temp_color)));
                    }
                }
                if gain > 1.0 {
                    let gain_alpha = (gain - 1.0).clamp(0.0, 0.4);
                    if gain_alpha > 0.0 {
                        let gain_base = rgba_from_triplet_with_alpha(
                            visual::visual_style().chrome.primary_text_srgb,
                            gain_alpha * 0.35,
                        );
                        window.paint_quad(fill(
                            bounds,
                            Background::from(apply_palette(gain_base, palette)),
                        ));
                    }
                } else if gain < 1.0 {
                    let gain_alpha = ((1.0 - gain) * 0.55).clamp(0.0, 0.30);
                    let gain_base = rgba_from_triplet_with_alpha(
                        visual::visual_style().substrate.abyss_srgb,
                        gain_alpha,
                    );
                    window.paint_quad(fill(
                        bounds,
                        Background::from(apply_palette(gain_base, palette)),
                    ));
                }
            }
            PostProcessPass::Vignette {
                strength,
                smoothness,
            } => {
                if strength > 0.01 {
                    // GPUI's CPU canvas has no framebuffer-sampling post pass, so
                    // four edge gradients approximate a smooth vignette. The
                    // configured smoothness controls the falloff width instead of
                    // being accepted and silently ignored.
                    let edge_fraction = 0.08 + smoothness * 0.15;
                    let alpha = ((0.24 + (1.0 - daylight) * 0.18) * strength).clamp(0.03, 0.50);
                    let edge_color = apply_palette(
                        rgba_from_triplet_with_alpha(
                            visual::visual_style().substrate.abyss_srgb,
                            alpha,
                        ),
                        palette,
                    );
                    let transparent = Rgba {
                        a: 0.0,
                        ..edge_color
                    };
                    let band_height = height_px * edge_fraction;
                    let band_width = width_px * edge_fraction * 0.62;
                    for (edge_bounds, angle) in [
                        (
                            Bounds::new(
                                point(px(origin_x), px(origin_y)),
                                size(px(width_px), px(band_height)),
                            ),
                            0.0,
                        ),
                        (
                            Bounds::new(
                                point(px(origin_x), px(origin_y + height_px - band_height)),
                                size(px(width_px), px(band_height)),
                            ),
                            180.0,
                        ),
                        (
                            Bounds::new(
                                point(px(origin_x), px(origin_y)),
                                size(px(band_width), px(height_px)),
                            ),
                            270.0,
                        ),
                        (
                            Bounds::new(
                                point(px(origin_x + width_px - band_width), px(origin_y)),
                                size(px(band_width), px(height_px)),
                            ),
                            90.0,
                        ),
                    ] {
                        window.paint_quad(fill(
                            edge_bounds,
                            linear_gradient(
                                angle,
                                linear_color_stop(transparent, 0.0),
                                linear_color_stop(edge_color, 1.0),
                            ),
                        ));
                    }
                }
            }
            PostProcessPass::Bloom { strength } => {
                if strength > 0.01 {
                    // Four gradients converge on the scene centre, approximating a
                    // broad emissive glow without the hard rectangle used before.
                    let style = visual::visual_style();
                    let bloom_base = lerp_rgba(
                        rgba_from_triplet_with_alpha(
                            style.chrome.accent_magenta_srgb,
                            0.045 * strength,
                        ),
                        rgba_from_triplet_with_alpha(
                            style.chrome.accent_cyan_srgb,
                            0.040 * strength,
                        ),
                        daylight.clamp(0.0, 1.0),
                    );
                    let bloom_color = apply_palette(bloom_base, palette);
                    let transparent = Rgba {
                        a: 0.0,
                        ..bloom_color
                    };
                    for (bloom_bounds, angle) in [
                        (
                            Bounds::new(
                                point(px(origin_x), px(origin_y)),
                                size(px(width_px), px(height_px * 0.5)),
                            ),
                            180.0,
                        ),
                        (
                            Bounds::new(
                                point(px(origin_x), px(origin_y + height_px * 0.5)),
                                size(px(width_px), px(height_px * 0.5)),
                            ),
                            0.0,
                        ),
                        (
                            Bounds::new(
                                point(px(origin_x), px(origin_y)),
                                size(px(width_px * 0.5), px(height_px)),
                            ),
                            90.0,
                        ),
                        (
                            Bounds::new(
                                point(px(origin_x + width_px * 0.5), px(origin_y)),
                                size(px(width_px * 0.5), px(height_px)),
                            ),
                            270.0,
                        ),
                    ] {
                        window.paint_quad(fill(
                            bloom_bounds,
                            linear_gradient(
                                angle,
                                linear_color_stop(transparent, 0.0),
                                linear_color_stop(bloom_color, 1.0),
                            ),
                        ));
                    }
                }
            }
            PostProcessPass::Fog { strength, color } => {
                if strength > 0.001 {
                    // Distance and height fog becomes a horizon gradient in this
                    // 2D canvas backend, preserving terrain silhouettes without
                    // the visible bands of the old approximation.
                    let fog_bounds = Bounds::new(
                        point(px(origin_x), px(origin_y + height_px * 0.34)),
                        size(px(width_px), px(height_px * 0.66)),
                    );
                    let fog_color = apply_palette(
                        rgba_from_triplet_with_alpha(color, strength * 0.42),
                        palette,
                    );
                    let transparent = Rgba {
                        a: 0.0,
                        ..fog_color
                    };
                    window.paint_quad(fill(
                        fog_bounds,
                        linear_gradient(
                            180.0,
                            linear_color_stop(transparent, 0.0),
                            linear_color_stop(fog_color, 1.0),
                        ),
                    ));
                }
            }
            PostProcessPass::Scanlines { intensity, spacing } => {
                if intensity > 0.01 && !low_fps {
                    let spacing_px = (spacing / scale).clamp(2.0, 9.0);
                    let alpha = (0.07 * intensity).clamp(0.01, 0.2);
                    let mut y = origin_y;
                    while y < origin_y + height_px {
                        let scanline_bounds =
                            Bounds::new(point(px(origin_x), px(y)), size(px(width_px), px(1.0)));
                        let scan_base = rgba_from_hex(0x020816, alpha);
                        let scan_color = if matches!(palette, ColorPaletteMode::Natural) {
                            scan_base
                        } else {
                            apply_palette(scan_base, palette)
                        };
                        window.paint_quad(fill(scanline_bounds, Background::from(scan_color)));
                        y += spacing_px;
                    }
                }
            }
            PostProcessPass::FilmGrain { strength, seed } => {
                if strength > 0.01 && !low_fps {
                    paint_film_grain(bounds, seed, strength, palette, window);
                }
            }
        }
    }
}

fn paint_film_grain(
    bounds: Bounds<Pixels>,
    seed: u64,
    strength: f32,
    palette: ColorPaletteMode,
    window: &mut Window,
) {
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let width_px = f32::from(bounds_size.width).max(1.0);
    let height_px = f32::from(bounds_size.height).max(1.0);
    let cols = (width_px / 24.0).clamp(16.0, 64.0) as u32;
    let rows = (height_px / 24.0).clamp(12.0, 48.0) as u32;
    let cell_w = width_px / cols as f32;
    let cell_h = height_px / rows as f32;

    for row in 0..rows {
        for col in 0..cols {
            let noise = hashed_noise(seed, row as u64, col as u64);
            if noise < 0.2 {
                continue;
            }
            let alpha = (0.02 + noise * 0.06) * strength;
            if alpha < 0.01 {
                continue;
            }
            let base_hex = if noise > 0.6 { 0xffffff } else { 0x0f172a };
            let base = rgba_from_hex(base_hex, alpha.clamp(0.01, 0.12));
            let color = if matches!(palette, ColorPaletteMode::Natural) {
                base
            } else {
                apply_palette(base, palette)
            };
            let x = f32::from(origin.x) + col as f32 * cell_w;
            let y = f32::from(origin.y) + row as f32 * cell_h;
            let cell_bounds = Bounds::new(point(px(x), px(y)), size(px(cell_w), px(cell_h)));
            window.paint_quad(fill(cell_bounds, Background::from(color)));
        }
    }
}

fn hashed_noise(seed: u64, row: u64, col: u64) -> f32 {
    let mut value = seed.wrapping_add(row.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        ^ col.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 29;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    value ^= value >> 32;
    (value as f64 / u64::MAX as f64) as f32
}
#[allow(clippy::too_many_arguments)]
fn paint_debug_overlays(
    frame: &RenderFrame,
    focus_agent: Option<AgentId>,
    debug: DebugOverlayState,
    camera: &CameraSnapshot,
    layout: &ViewLayout,
    bounds: Bounds<Pixels>,
    window: &mut Window,
) {
    if !debug.enabled {
        return;
    }

    let scale = layout.scale;
    if scale <= f32::EPSILON {
        return;
    }

    let wants_sense = debug.show_sense_radius && frame.sense_radius > 0.0;
    let wants_velocity = debug.show_velocity;
    // Lazily create paths only if they are actually used (reduces per-frame allocations)
    let mut sense_path: Option<PathBuilder> = None;
    let mut vel_shaft_path: Option<PathBuilder> = None;
    let mut vel_head_path: Option<PathBuilder> = None;
    let mut crosshair_path = if focus_agent.is_some() {
        Some(PathBuilder::stroke(px(1.6)))
    } else {
        None
    };

    // If nothing is requested, return early
    if !wants_sense && !wants_velocity && crosshair_path.is_none() {
        return;
    }

    let view_left = f32::from(bounds.origin.x);
    let view_top = f32::from(bounds.origin.y);
    let view_right = view_left + f32::from(bounds.size.width);
    let view_bottom = view_top + f32::from(bounds.size.height);

    for agent in &frame.agents {
        let Some((px_x, px_y)) = camera.world_to_screen((agent.position.x, agent.position.y))
        else {
            continue;
        };

        // Cull debug overlays off-screen (with a small margin)
        if px_x < view_left - 64.0
            || px_x > view_right + 64.0
            || px_y < view_top - 64.0
            || px_y > view_bottom + 64.0
        {
            continue;
        }

        if wants_sense
            && matches!(
                agent.selection,
                SelectionState::Selected | SelectionState::Hovered
            )
        {
            if sense_path.is_none() {
                sense_path = Some(PathBuilder::stroke(px(1.4)));
            }
            let builder = sense_path.as_mut().expect("sense_path just created");
            let radius_px = (frame.sense_radius * scale).max(4.0);
            append_arc_polyline(builder, px_x, px_y, radius_px, 0.0, 360.0);
        }

        if wants_velocity {
            let vx = agent.velocity.vx;
            let vy = agent.velocity.vy;
            let speed = (vx * vx + vy * vy).sqrt();
            if speed > 1e-3 {
                let norm_x = vx / speed;
                let norm_y = vy / speed;
                let min_length = (frame.agent_base_radius * 1.5).max(8.0) * scale;
                let dynamic_length = (speed * frame.sense_radius)
                    .clamp(frame.agent_base_radius, frame.sense_radius * 1.5)
                    * scale;
                let arrow_length = dynamic_length.max(min_length);
                let tip_x = px_x + norm_x * arrow_length;
                let tip_y = px_y + norm_y * arrow_length;

                if vel_shaft_path.is_none() {
                    vel_shaft_path = Some(PathBuilder::stroke(px(1.6)));
                }
                let builder = vel_shaft_path
                    .as_mut()
                    .expect("vel_shaft_path just created");
                builder.move_to(point(px(px_x), px(px_y)));
                builder.line_to(point(px(tip_x), px(tip_y)));

                // Arrow head using fixed-angle rotation (avoid per-agent atan2/cos/sin)
                // angle = 0.5 rad; cos ~= 0.87758256, sin ~= 0.47942555
                let head_size = (arrow_length * 0.18).clamp(4.0, 14.0);
                let back = head_size * 0.877_582_56_f32;
                let side = head_size * 0.479_425_55_f32;
                // Perpendicular to direction
                let perp_x = -norm_y;
                let perp_y = norm_x;
                // Left and right points relative to tip
                let left_x = tip_x - back * norm_x + side * perp_x;
                let left_y = tip_y - back * norm_y + side * perp_y;
                let right_x = tip_x - back * norm_x - side * perp_x;
                let right_y = tip_y - back * norm_y - side * perp_y;

                if vel_head_path.is_none() {
                    vel_head_path = Some(PathBuilder::stroke(px(1.2)));
                }
                let builder = vel_head_path.as_mut().expect("vel_head_path just created");
                builder.move_to(point(px(tip_x), px(tip_y)));
                builder.line_to(point(px(left_x), px(left_y)));
                builder.move_to(point(px(tip_x), px(tip_y)));
                builder.line_to(point(px(right_x), px(right_y)));
            }
        }

        if Some(agent.agent_id) == focus_agent
            && let Some(builder) = crosshair_path.as_mut()
        {
            let cross = (frame.agent_base_radius * scale).max(6.0);
            builder.move_to(point(px(px_x - cross), px(px_y)));
            builder.line_to(point(px(px_x + cross), px(px_y)));
            builder.move_to(point(px(px_x), px(px_y - cross)));
            builder.line_to(point(px(px_x), px(px_y + cross)));
        }
    }

    if let Some(builder) = sense_path
        && let Ok(path) = builder.build()
    {
        window.paint_path(path, rgba_from_hex(0x38bdf8, 0.35));
    }
    if let Some(builder) = vel_shaft_path
        && let Ok(path) = builder.build()
    {
        window.paint_path(path, rgba_from_hex(0x22d3ee, 0.85));
    }
    if let Some(builder) = vel_head_path
        && let Ok(path) = builder.build()
    {
        window.paint_path(path, rgba_from_hex(0xe0f2fe, 0.78));
    }
    if let Some(builder) = crosshair_path
        && let Ok(path) = builder.build()
    {
        window.paint_path(path, rgba_from_hex(0xfacc15, 0.9));
    }
}

fn paint_frame(state: &CanvasState, bounds: Bounds<Pixels>, window: &mut Window) {
    let frame = &state.frame;
    let camera = &state.camera;
    let focus_agent = state.focus_agent;
    let controls = state.controls;
    let debug = state.debug;
    let follow_target = state.follow_target;
    let origin = bounds.origin;
    let bounds_size = bounds.size;
    let low_fps = state.perf.fps > 0.0 && state.perf.fps < 30.0;
    let very_low_fps = state.perf.fps > 0.0 && state.perf.fps < 24.0;
    let palette_is_natural = matches!(frame.palette, ColorPaletteMode::Natural);

    let mut camera_guard = camera.lock().expect("camera lock poisoned");

    let width_px = f32::from(bounds_size.width).max(1.0);
    let raw_height_px = f32::from(bounds_size.height).max(1.0);
    let window_bounds = window.bounds();
    let window_height_px = f32::from(window_bounds.size.height).max(1.0);
    let height_px = if raw_height_px <= 2.0 && window_height_px > 16.0 {
        window_height_px
    } else {
        raw_height_px
    };
    // GPUI can report a one-pixel flex-canvas height during the custom element's
    // paint callback. The world painter already compensates with the window
    // height above; every screen-space post pass must consume the same effective
    // bounds or it technically runs while touching only a one-pixel strip.
    let effective_bounds = Bounds::new(origin, size(px(width_px), px(height_px)));
    let origin_x = f32::from(origin.x);
    let origin_y = f32::from(origin.y);

    let mut layout = layout_camera_for_frame(
        &mut camera_guard,
        frame,
        (origin_x, origin_y),
        (width_px, height_px),
    );

    tracing::debug!(
        viewport_width = width_px,
        viewport_height_raw = raw_height_px,
        viewport_height = height_px,
        window_height = window_height_px,
        render_width = layout.render_size.0,
        render_height = layout.render_size.1,
        pad_x = layout.pad.0,
        pad_y = layout.pad.1,
        offset_x = layout.offset.0,
        offset_y = layout.offset.1,
        base_scale = layout.base_scale,
        zoom = camera_guard.zoom(),
        "camera_layout_pre_follow"
    );

    let coverage_too_small =
        layout.render_size.0 < width_px * 0.25 || layout.render_size.1 < height_px * 0.25;
    let coverage_too_large =
        layout.render_size.0 > width_px * 6.0 || layout.render_size.1 > height_px * 6.0;

    if coverage_too_small || coverage_too_large {
        camera_guard.fit_world();
        layout = camera_guard.layout(
            (origin_x, origin_y),
            (width_px, height_px),
            frame.world_size,
        );
        tracing::warn!(
            render_width = layout.render_size.0,
            render_height = layout.render_size.1,
            viewport_width = width_px,
            viewport_height = height_px,
            "auto_fit_world_due_to_extreme_zoom"
        );
    }

    if controls.follow_mode != FollowMode::Off
        && let Some(target) = follow_target
    {
        camera_guard.center_on(target);
        layout = camera_guard.layout(
            (origin_x, origin_y),
            (width_px, height_px),
            frame.world_size,
        );

        let follow_coverage_small =
            layout.render_size.0 < width_px * 0.25 || layout.render_size.1 < height_px * 0.25;
        let follow_coverage_large =
            layout.render_size.0 > width_px * 6.0 || layout.render_size.1 > height_px * 6.0;
        if follow_coverage_small || follow_coverage_large {
            camera_guard.fit_world();
            layout = camera_guard.layout(
                (origin_x, origin_y),
                (width_px, height_px),
                frame.world_size,
            );
        }
    }

    let mut camera_snapshot = camera_guard.snapshot();

    let scale = layout.scale;
    let base_scale = layout.base_scale;
    let pad_x = layout.pad.0;
    let pad_y = layout.pad.1;
    let offset_x = layout.offset.0;
    let offset_y = layout.offset.1;

    // Compute view bounds in window space used for culling and offscreen checks
    let view_left = origin_x;
    let view_top = origin_y;
    let view_right = view_left + width_px;
    let view_bottom = view_top + height_px;

    if env_flag("SB_CPU_LAYOUT_LOG") {
        tracing::info!(
            viewport_width = width_px,
            viewport_height = height_px,
            world_width = frame.world_size.0,
            world_height = frame.world_size.1,
            origin_x,
            origin_y,
            pad_x,
            pad_y,
            offset_x,
            offset_y,
            zoom = scale / base_scale,
            "cpu_canvas_layout"
        );
    }

    // Final guard: if the computed render rect is still fully outside the view
    // (observed on some Windows setups on first frame), draw this frame using a
    // deterministic centered offset so content is visible. This does not mutate
    // camera state; subsequent frames will use the camera's offsets.
    camera_snapshot.last_canvas_origin = (origin_x, origin_y);
    camera_snapshot.last_canvas_size = (width_px, height_px);
    camera_snapshot.last_world_size = frame.world_size;
    camera_snapshot.last_scale = scale;
    camera_snapshot.last_base_scale = base_scale;
    camera_snapshot.offset_px = (offset_x - origin_x - pad_x, offset_y - origin_y - pad_y);

    drop(camera_guard);

    let daylight = visual::daylight_factor(
        frame.tick,
        frame.day_night_cycle_ticks,
        frame.day_night_start_phase,
    );
    let style = visual::visual_style();

    if safe_mode_enabled() {
        // Conservative background fill (bypass gradient blending that could expose format issues)
        let background = apply_palette(
            rgba_from_triplet_with_alpha(style.substrate.abyss_srgb, 1.0),
            frame.palette,
        );
        window.paint_quad(fill(bounds, Background::from(background)));
    } else {
        let sky_base = lerp_rgba(
            rgba_from_triplet_with_alpha(style.substrate.abyss_srgb, 1.0),
            rgba_from_triplet_with_alpha(style.substrate.depth_violet_srgb, 1.0),
            daylight * 0.65,
        );
        window.paint_quad(fill(
            bounds,
            Background::from(apply_palette(sky_base, frame.palette)),
        ));
    }

    let horizon_height = height_px * 0.35;
    if !safe_mode_enabled() && horizon_height > 1.0 {
        let horizon_bounds = Bounds::new(
            point(px(origin_x), px(origin_y + height_px - horizon_height)),
            size(px(width_px), px(horizon_height)),
        );
        let horizon_base = rgba_from_triplet_with_alpha(
            style.substrate.distant_haze_srgb,
            (0.06 + 0.16 * daylight).clamp(0.0, 0.22),
        );
        let horizon_color = apply_palette(horizon_base, frame.palette);
        window.paint_quad(fill(horizon_bounds, Background::from(horizon_color)));
    }

    let aurora_strength = (1.0 - daylight).clamp(0.0, 1.0);
    if aurora_strength > 0.05 {
        let aurora_bounds = Bounds::new(
            point(px(origin_x), px(origin_y)),
            size(px(width_px), px(height_px * 0.25)),
        );
        let aurora_base =
            rgba_from_triplet_with_alpha(style.substrate.base_srgb, 0.18 * aurora_strength);
        let aurora_color = apply_palette(aurora_base, frame.palette);
        window.paint_quad(fill(aurora_bounds, Background::from(aurora_color)));
    }

    let use_continuous_world_fields = {
        #[cfg(test)]
        {
            !state.force_legacy_world_painter
        }
        #[cfg(not(test))]
        {
            true
        }
    };
    let continuous_world_painted = use_continuous_world_fields
        && paint_continuous_world_fields(state, offset_x, offset_y, scale, daylight, window);
    if !continuous_world_painted {
        paint_terrain_layer(
            &frame.terrain,
            offset_x,
            offset_y,
            scale,
            daylight,
            frame.palette,
            view_left,
            view_top,
            view_right,
            view_bottom,
            window,
        );
    }

    // Food remains a discrete emissive presentation layer above both terrain paths.
    // Sampling it into the continuous terrain image removes the grid by erasing the
    // very local contrast that makes resources legible against the substrate.
    let food_w = frame.food_dimensions.0 as usize;
    let food_h = frame.food_dimensions.1 as usize;
    let max_food = frame.food_max.max(f32::EPSILON);
    let inv_max_food: f32 = if max_food > 0.0 {
        1.0_f32 / max_food
    } else {
        0.0_f32
    };

    if controls.draw_food {
        // Compute visible cell range to cull off-screen food cells
        let cell_world = frame.food_cell_size as f32;
        let cell_px = (cell_world * scale).max(1.0);
        let inv_cell_px = if cell_px > f32::EPSILON {
            1.0 / cell_px
        } else {
            0.0
        };
        let mut x_min = ((view_left - offset_x) * inv_cell_px).floor() as isize;
        let mut x_max = ((view_right - offset_x) * inv_cell_px).ceil() as isize;
        let mut y_min = ((view_top - offset_y) * inv_cell_px).floor() as isize;
        let mut y_max = ((view_bottom - offset_y) * inv_cell_px).ceil() as isize;
        x_min = x_min.clamp(0, food_w as isize - 1);
        x_max = x_max.clamp(0, food_w as isize - 1);
        y_min = y_min.clamp(0, food_h as isize - 1);
        y_max = y_max.clamp(0, food_h as isize - 1);

        if very_low_fps {
            // Quantized batching: approximate per-cell shading by grouping into bins, reducing draw calls
            const FOOD_BINS: usize = 24;
            for y in y_min as usize..=y_max as usize {
                let mut builders: [Option<PathBuilder>; FOOD_BINS] = [
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                    Some(PathBuilder::fill()),
                ];
                let mut colors: [Option<Rgba>; FOOD_BINS] = [None; FOOD_BINS];
                for x in x_min as usize..=x_max as usize {
                    let idx = y * food_w + x;
                    let value = frame.food_cells.get(idx).copied().unwrap_or_default();
                    if value <= 0.001 {
                        continue;
                    }
                    let intensity: f32 = (value * inv_max_food).clamp(0.0_f32, 1.0_f32);
                    let food_visuals = visual::food_visual_params(intensity);
                    let mut color = food_color(intensity);
                    let shade_wave = visual::shimmer(frame.tick, x as u32, y as u32);
                    let gain = food_visuals.emissive_gain
                        / visual::visual_style().food.dense_emissive_gain;
                    let shade = (0.78 + 0.18 * shade_wave + 0.22 * gain).clamp(0.0, 1.3);
                    color = scale_rgb(color, shade);
                    if !palette_is_natural {
                        color = apply_palette(color, frame.palette);
                    }

                    // Quantize based on luminance to group similar colors
                    let luma = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
                    let mut bin = ((luma * (FOOD_BINS as f32)) as isize)
                        .clamp(0, (FOOD_BINS as isize) - 1)
                        as usize;
                    // Avoid bin 0 swallowing very dark but present cells when alpha is tiny
                    if bin == 0 && luma > 0.0 {
                        bin = 1;
                    }

                    let px_x = offset_x + (x as f32 * cell_world * scale);
                    let px_y = offset_y + (y as f32 * cell_world * scale);
                    if let Some(builder) = builders[bin].as_mut() {
                        // Stable-hue motes shrink sparse food away from the cell edges.
                        let mote_px = (cell_px * food_visuals.relative_radius.min(1.0)).max(1.0);
                        append_circle_polygon(
                            builder,
                            px_x + cell_px * 0.5,
                            px_y + cell_px * 0.5,
                            mote_px * 0.5,
                        );
                        builder.close();
                    }
                    if colors[bin].is_none() {
                        colors[bin] = Some(color);
                    }
                }
                for b in 0..FOOD_BINS {
                    if let (Some(builder), Some(col)) = (builders[b].take(), colors[b])
                        && let Ok(path) = builder.build()
                    {
                        window.paint_path(path, col);
                    }
                }
            }
        } else {
            for y in y_min as usize..=y_max as usize {
                for x in x_min as usize..=x_max as usize {
                    let idx = y * food_w + x;
                    let value = frame.food_cells.get(idx).copied().unwrap_or_default();
                    if value <= 0.001 {
                        continue;
                    }
                    let intensity: f32 = (value * inv_max_food).clamp(0.0_f32, 1.0_f32);
                    let food_visuals = visual::food_visual_params(intensity);
                    let mut color = food_color(intensity);
                    let shade_wave = visual::shimmer(frame.tick, x as u32, y as u32);
                    let gain = food_visuals.emissive_gain
                        / visual::visual_style().food.dense_emissive_gain;
                    let shade = (0.78 + 0.18 * shade_wave + 0.22 * gain).clamp(0.0, 1.3);
                    color = scale_rgb(color, shade);
                    if !palette_is_natural {
                        color = apply_palette(color, frame.palette);
                    }
                    let px_x = offset_x + (x as f32 * cell_world * scale);
                    let px_y = offset_y + (y as f32 * cell_world * scale);
                    let mote_px = (cell_px * food_visuals.relative_radius.min(1.0)).max(1.0);
                    let inset = (cell_px - mote_px) * 0.5;
                    let cell_bounds = Bounds::new(
                        point(px(px_x + inset), px(px_y + inset)),
                        size(px(mote_px), px(mote_px)),
                    );
                    window.paint_quad(
                        fill(cell_bounds, Background::from(color)).corner_radii(px(mote_px * 0.5)),
                    );
                }
            }
        }
    }

    if controls.draw_agents {
        if very_low_fps {
            paint_agent_lod_batches(
                window,
                frame,
                focus_agent,
                offset_x,
                offset_y,
                scale,
                view_left,
                view_top,
                view_right,
                view_bottom,
                palette_is_natural,
            );
        } else {
            for agent in &frame.agents {
                let px_x = offset_x + agent.position.x * scale;
                let px_y = offset_y + agent.position.y * scale;
                let half = agent_screen_radius(frame, agent, scale);
                let size_px = half * 2.0;

                // Cull off-screen agents
                if px_x + half < view_left
                    || px_x - half > view_right
                    || px_y + half < view_top
                    || px_y - half > view_bottom
                {
                    continue;
                }
                let visuals = resolve_agent_visual(agent, frame.agent_reference_age);

                if !low_fps {
                    let shadow_radius = half * 1.25;
                    let shadow_offset_x = scale.clamp(1.0, 6.0) * 0.6;
                    let shadow_offset_y = scale.clamp(1.0, 7.0) * 1.1;
                    let mut shadow = PathBuilder::fill();
                    append_circle_polygon(
                        &mut shadow,
                        px_x + shadow_offset_x,
                        px_y + shadow_offset_y,
                        shadow_radius,
                    );
                    if let Ok(path) = shadow.build() {
                        let shadow_color = apply_palette(
                            rgba_from_triplet_with_alpha(
                                visual::visual_style().substrate.abyss_srgb,
                                0.38,
                            ),
                            frame.palette,
                        );
                        window.paint_path(path, shadow_color);
                    }
                }
                // Inline highlights without allocating
                if !very_low_fps {
                    match agent.selection {
                        SelectionState::Selected => {
                            let factor = 1.85;
                            let highlight_radius = half * factor;
                            let highlight = apply_palette(
                                rgba_from_triplet_with_alpha(visuals.selection_rim_color, 0.36),
                                frame.palette,
                            );
                            let mut highlight_path = PathBuilder::fill();
                            append_circle_polygon(
                                &mut highlight_path,
                                px_x,
                                px_y,
                                highlight_radius,
                            );
                            if let Ok(path) = highlight_path.build() {
                                window.paint_path(path, highlight);
                            }
                        }
                        SelectionState::Hovered => {
                            let factor = 1.45;
                            let highlight_radius = half * factor;
                            let highlight = apply_palette(
                                rgba_from_triplet_with_alpha(visuals.selection_rim_color, 0.26),
                                frame.palette,
                            );
                            let mut highlight_path = PathBuilder::fill();
                            append_circle_polygon(
                                &mut highlight_path,
                                px_x,
                                px_y,
                                highlight_radius,
                            );
                            if let Ok(path) = highlight_path.build() {
                                window.paint_path(path, highlight);
                            }
                        }
                        SelectionState::None => {}
                    }

                    if focus_agent == Some(agent.agent_id) {
                        let factor = 2.05;
                        let highlight_radius = half * factor;
                        let highlight = apply_palette(
                            rgba_from_triplet_with_alpha(visuals.selection_rim_color, 0.32),
                            frame.palette,
                        );
                        let mut focus_path = PathBuilder::fill();
                        append_circle_polygon(&mut focus_path, px_x, px_y, highlight_radius);
                        if let Ok(path) = focus_path.build() {
                            window.paint_path(path, highlight);
                        }
                    }
                }

                // Defer agent outlines to a single batched pass below

                if agent.indicator.intensity > 0.05 && !low_fps {
                    let effect = agent.indicator.intensity.clamp(0.0, 1.0);
                    let indicator_radius = half * (1.2 + effect * 1.4);
                    let indicator_color = apply_palette(
                        rgba_from_triplet_with_alpha(agent.indicator.color, 0.15 + 0.35 * effect),
                        frame.palette,
                    );
                    let mut indicator = PathBuilder::fill();
                    append_circle_polygon(&mut indicator, px_x, px_y, indicator_radius);
                    if let Ok(path) = indicator.build() {
                        window.paint_path(path, indicator_color);
                    }
                }

                if agent.reproduction_intent > 0.2 && !low_fps {
                    let pulse = agent.reproduction_intent.clamp(0.0, 1.0);
                    let pulse_radius = half * (1.8 + pulse * 1.6);
                    let cue = visual::visual_cue_for_event(&WorldVisualEvent::Reproduce);
                    let pulse_color = apply_palette(
                        rgba_from_triplet_with_alpha(cue.color, 0.18 + 0.28 * pulse),
                        frame.palette,
                    );
                    let mut pulse_path = PathBuilder::fill();
                    append_circle_polygon(&mut pulse_path, px_x, px_y, pulse_radius);
                    if let Ok(path) = pulse_path.build() {
                        window.paint_path(path, pulse_color);
                    }
                }

                if agent.spike_struck {
                    let spike_radius = half * 2.2;
                    let cue = visual::visual_cue_for_event(&WorldVisualEvent::SpikeExtend);
                    let spike_color =
                        apply_palette(rgba_from_triplet_with_alpha(cue.color, 0.28), frame.palette);
                    let mut spike_path = PathBuilder::fill();
                    append_circle_polygon(&mut spike_path, px_x, px_y, spike_radius);
                    if let Ok(path) = spike_path.build() {
                        window.paint_path(path, spike_color);
                    }
                }

                let mut body_color = agent_color(&visuals);
                if !palette_is_natural {
                    body_color = apply_palette(body_color, frame.palette);
                }

                paint_agent_avatar(
                    window,
                    agent,
                    (px_x, px_y),
                    size_px,
                    scale,
                    body_color,
                    &visuals,
                    frame.palette,
                    palette_is_natural,
                    very_low_fps,
                );
            }
        }
    }

    if controls.draw_agents {
        paint_debug_overlays(
            frame,
            focus_agent,
            debug,
            &camera_snapshot,
            &layout,
            bounds,
            window,
        );

        // Batched agent outlines pass
        if controls.agent_outline && !very_low_fps {
            let mut outline_builder = PathBuilder::stroke(px(2.6));

            for agent in &frame.agents {
                let Some((px_x, px_y)) =
                    camera_snapshot.world_to_screen((agent.position.x, agent.position.y))
                else {
                    continue;
                };
                let half = agent_screen_radius(frame, agent, scale);
                if px_x + half < view_left
                    || px_x - half > view_right
                    || px_y + half < view_top
                    || px_y - half > view_bottom
                {
                    continue;
                }

                let visuals = resolve_agent_visual(agent, frame.agent_reference_age);
                let forward = (visuals.facing[0], visuals.facing[1]);
                let right = (visuals.right[0], visuals.right[1]);
                let outline_half_length = (half * 1.35).max(half + 2.0);
                let outline_radius = (half * 0.72).max(3.0);
                append_capsule_polygon(
                    &mut outline_builder,
                    (px_x, px_y),
                    forward,
                    right,
                    outline_half_length,
                    outline_radius,
                    14,
                );
            }
            if let Ok(path) = outline_builder.build() {
                window.paint_path(
                    path,
                    rgba_from_triplet_with_alpha(visual::visual_style().substrate.abyss_srgb, 0.78),
                );
            }
        }
    }

    if !safe_mode_enabled() {
        apply_post_processing(
            &frame.post_stack,
            frame.palette,
            effective_bounds,
            window,
            daylight,
            scale,
            low_fps,
        );
    }

    if watermark_enabled() {
        let mark_bounds = Bounds::new(
            point(px(origin_x + 6.0), px(origin_y + 6.0)),
            size(px(6.0), px(6.0)),
        );
        window.paint_quad(fill(
            mark_bounds,
            Background::from(rgba_from_hex(0xff00aa, 1.0)),
        ));
    }
}

fn food_color(intensity: f32) -> Rgba {
    let params = visual::food_visual_params(intensity);
    rgba_from_triplet_with_alpha(visual::food_density_color(intensity), params.alpha)
}

fn resolve_agent_visual(agent: &AgentRenderData, reference_age_ticks: u64) -> AgentVisualParams {
    visual::agent_visual_params(&AgentVisualInput {
        genome_color: agent.color,
        health: agent.health,
        age_ticks: u64::from(agent.age),
        reference_age_ticks,
        herbivore_tendency: agent.herbivore_tendency,
        temperature_preference: agent.temperature_preference,
        wheel_left: agent.wheel_left,
        wheel_right: agent.wheel_right,
        heading: agent.heading,
        spike_extended: agent.spike_extended,
        spike_length: agent.spike_length,
        boosting: agent.boost > 0.05,
        sound_output: agent.sound_output,
        sound_multiplier: agent.sound_multiplier,
        sound_level: agent.sound_level,
        food_delta: agent.food_delta,
        trait_smell: agent.trait_smell,
        trait_hearing: agent.trait_hearing,
        selection: match agent.selection {
            SelectionState::None => VisualSelection::None,
            SelectionState::Hovered => VisualSelection::Hovered,
            SelectionState::Selected => VisualSelection::Selected,
        },
    })
}

fn agent_color(visuals: &AgentVisualParams) -> Rgba {
    Rgba {
        r: visuals.body_color[0],
        g: visuals.body_color[1],
        b: visuals.body_color[2],
        a: 0.96,
    }
}

#[cfg(feature = "world_wgpu")]
fn build_gpu_agent_instance(
    frame: &RenderFrame,
    agent: &AgentRenderData,
    palette: ColorPaletteMode,
    palette_is_natural: bool,
) -> scriptbots_world_gfx::AgentInstance {
    let dynamic_radius = (frame.agent_base_radius + agent.spike_length * 0.25).max(8.0);
    let half_world = dynamic_radius;
    let mut body_radius = half_world * 0.72;
    if body_radius < 3.0 {
        body_radius = 3.0;
    }
    let mut body_half_length = half_world * 1.35;
    if body_half_length < body_radius + 2.0 {
        body_half_length = body_radius + 2.0;
    }
    let mut wheel_radius = body_radius * 0.38;
    if wheel_radius < 2.0 {
        wheel_radius = 2.0;
    }
    let wheel_offset = body_radius + wheel_radius * 0.55;
    let spike_extension = body_radius * 0.7 + agent.spike_length * 0.85 + 2.0;
    let flame_length = if agent.boost > 0.05 {
        body_radius * (1.2 + agent.boost * 1.6) + agent.sound_multiplier.max(1.0) * 4.0
    } else {
        0.0
    };
    let half_width = wheel_offset + wheel_radius + 3.0;
    let half_height = body_half_length + spike_extension.max(flame_length) + 3.0;
    let eating_level = agent.food_delta.abs().min(1.5);
    let yelling_level = agent.sound_output.abs().min(1.5);
    let mouth_open = (0.35 + eating_level * 0.4 + yelling_level * 0.55).clamp(0.35, 1.6);

    let visuals = resolve_agent_visual(agent, frame.agent_reference_age);
    let mut body_color = agent_color(&visuals);
    if !palette_is_natural {
        body_color = apply_palette(body_color, palette);
    }

    let selection = match agent.selection {
        SelectionState::Hovered => 1.0,
        SelectionState::Selected => 2.0,
        SelectionState::None => 0.0,
    };
    let glow_indicator = (agent.indicator.intensity * 0.35).clamp(0.0, 1.0);
    let glow_spike = if agent.spike_struck { 0.45 } else { 0.0 };
    let glow_repro = (agent.reproduction_intent * 0.25).clamp(0.0, 0.6);
    let glow = glow_indicator.max(glow_spike).max(glow_repro);
    let boost = agent.boost.clamp(0.0, 1.0);
    let spiked = if agent.spike_struck { 1.0 } else { 0.0 };

    scriptbots_world_gfx::AgentInstance {
        position: [agent.position.x, agent.position.y],
        quad_extent: [half_width, half_height],
        heading: visuals.facing,
        body_radius,
        body_half_length,
        wheel_offset,
        wheel_radius,
        mouth_open,
        herbivore_tendency: agent.herbivore_tendency.clamp(0.0, 1.0),
        temperature_preference: agent.temperature_preference.clamp(0.0, 1.0),
        food_delta: agent.food_delta,
        sound_level: agent.sound_level,
        sound_output: agent.sound_output,
        wheel_left: agent.wheel_left,
        wheel_right: agent.wheel_right,
        spike_length: agent.spike_length,
        trait_smell: agent.trait_smell,
        trait_sound: agent.trait_sound,
        trait_hearing: agent.trait_hearing,
        trait_eye: agent.trait_eye,
        trait_blood: agent.trait_blood,
        selection,
        color: [body_color.r, body_color.g, body_color.b, body_color.a],
        glow,
        boost,
        spiked,
        eye_dirs: agent.eye_dirs,
        eye_fov: agent.eye_fov,
    }
}

fn apply_palette(color: Rgba, palette: ColorPaletteMode) -> Rgba {
    let [r, g, b] = visual::apply_accessibility_palette(
        [color.r, color.g, color.b],
        accessibility_palette(palette),
    );
    Rgba {
        r,
        g,
        b,
        a: color.a,
    }
}

const fn accessibility_palette(palette: ColorPaletteMode) -> AccessibilityPalette {
    match palette {
        ColorPaletteMode::Natural => AccessibilityPalette::Natural,
        ColorPaletteMode::Deuteranopia => AccessibilityPalette::Deuteranopia,
        ColorPaletteMode::Protanopia => AccessibilityPalette::Protanopia,
        ColorPaletteMode::Tritanopia => AccessibilityPalette::Tritanopia,
        ColorPaletteMode::HighContrast => AccessibilityPalette::HighContrast,
    }
}

#[allow(dead_code)]
fn transform_color(color: Rgba, matrix: [[f32; 3]; 3]) -> Rgba {
    let r =
        (color.r * matrix[0][0] + color.g * matrix[0][1] + color.b * matrix[0][2]).clamp(0.0, 1.0);
    let g =
        (color.r * matrix[1][0] + color.g * matrix[1][1] + color.b * matrix[1][2]).clamp(0.0, 1.0);
    let b =
        (color.r * matrix[2][0] + color.g * matrix[2][1] + color.b * matrix[2][2]).clamp(0.0, 1.0);
    Rgba {
        r,
        g,
        b,
        a: color.a,
    }
}

#[cfg(test)]
mod command_characterization_tests {
    use super::*;
    use scriptbots_core::{AgentData, CharacterizationDigestV0, WorldDigestV1};
    use std::time::Duration;

    #[test]
    fn production_renderer_has_no_direct_world_selection_writes() {
        let (production, _) = include_str!("lib.rs")
            .split_once("#[cfg(test)]\nmod command_characterization_tests")
            .expect("command characterization test boundary");
        assert!(
            !production.contains(".apply_selection_update("),
            "GPUI selection must be admitted through ControlCommand, never written to WorldState"
        );
    }

    #[test]
    fn production_renderer_has_no_direct_closed_world_writes() {
        let (production, _) = include_str!("lib.rs")
            .split_once("#[cfg(test)]\nmod command_characterization_tests")
            .expect("command characterization test boundary");
        assert!(
            !production.contains(".set_closed("),
            "GPUI closed-world changes must be admitted through ControlCommand, never written to WorldState"
        );
    }

    /// bd-37m acceptance guard: the driver command drain is the sole renderer-owned path that may
    /// borrow scientific world state mutably.
    #[test]
    fn production_renderer_has_no_direct_agent_science_writes_or_rng() {
        let (production, _) = include_str!("lib.rs")
            .split_once("#[cfg(test)]\nmod command_characterization_tests")
            .expect("command characterization test boundary");
        for (forbidden, reason) in [
            (
                "&mut WorldState",
                "production rendering must not accept mutable scientific world access",
            ),
            (
                ".try_update_agent(",
                "agent edits must be admitted through ControlCommand",
            ),
            (
                ".try_update_agent_runtime(",
                "agent mutation-rate edits must be admitted through ControlCommand",
            ),
            (
                ".agent_runtime_mut(",
                "production rendering must not borrow scientific runtime state mutably",
            ),
            (
                ".try_spawn_agent(",
                "agent creation must be admitted through ControlCommand",
            ),
            (
                ".try_spawn_agent_with(",
                "agent creation must be admitted through ControlCommand",
            ),
            (
                ".try_inject_agent(",
                "agent injection must be admitted through ControlCommand",
            ),
            (
                ".try_inject_agent_with(",
                "agent injection must be admitted through ControlCommand",
            ),
            (
                ".try_inject_crossover_agent_with(",
                "crossover injection must be admitted through ControlCommand",
            ),
            (
                ".rng(RngDomain::",
                "only the world-owned command application path may consume scientific RNG",
            ),
        ] {
            assert!(
                !production.contains(forbidden),
                "{reason}; production renderer still contains `{forbidden}`"
            );
        }
        assert_eq!(
            production
                .matches("if let Ok(mut world) = self.world.lock()")
                .count(),
            1,
            "the simulation driver command drain must be the sole mutable WorldState lock in production rendering"
        );
    }

    #[test]
    fn readme_gui_shortcuts_match_production_bindings() {
        let bindings = InputBindings::default();
        let entries = bindings.iter();
        for (index, (action, stroke)) in entries.iter().enumerate() {
            assert!(
                !stroke.key.is_empty(),
                "default action `{}` must have a shortcut",
                action.label()
            );
            for (other_action, other_stroke) in entries.iter().skip(index + 1) {
                assert!(
                    !keystrokes_equal(stroke, other_stroke),
                    "default actions `{}` and `{}` must not share `{}`",
                    action.label(),
                    other_action.label(),
                    format_keystroke(stroke)
                );
            }
        }

        for (binding, action) in [
            ("s", CommandAction::StepSimulation),
            ("shift-s", CommandAction::FollowSelected),
            ("space", CommandAction::TogglePlayback),
            ("g", CommandAction::GoLive),
            ("p", CommandAction::ToggleSimulationPause),
            ("ctrl-p", CommandAction::CyclePalette),
            ("0", CommandAction::FitWorld),
        ] {
            let stroke = Keystroke::parse(binding).expect("valid production shortcut");
            assert_eq!(
                bindings.action_for(&stroke),
                Some(action),
                "{binding} must invoke `{}`",
                action.label()
            );
        }

        let mut expected_table = String::from("| Action | Default shortcut |\n| --- | --- |\n");
        for (action, stroke) in entries {
            expected_table.push_str(&format!(
                "| {} | `{}` |\n",
                action.label(),
                format_keystroke(&stroke)
            ));
        }

        let readme = include_str!("../../../README.md");
        let (_, after_start) = readme
            .split_once("<!-- BEGIN GENERATED GPUI DEFAULT SHORTCUTS -->")
            .expect("README must start the generated GPUI shortcut table");
        let (documented_table, _) = after_start
            .split_once("<!-- END GENERATED GPUI DEFAULT SHORTCUTS -->")
            .expect("README must end the generated GPUI shortcut table");
        let documented_table = documented_table.replace("\r\n", "\n");
        assert_eq!(
            documented_table.trim(),
            expected_table.trim(),
            "README GPUI shortcuts must be rendered from the production default binding registry"
        );
    }

    #[test]
    fn production_render_frame_layout_is_legible_at_both_supported_viewports() {
        let config = ScriptBotsConfig {
            world_width: 6_000,
            world_height: 3_000,
            food_cell_size: 50,
            initial_food: 0.0,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        };
        let agent_radius = config.bot_radius;
        let mut world = WorldState::new(config).expect("production-layout world");
        for x in [120.0, 240.0, 360.0] {
            for y in [120.0, 240.0, 360.0] {
                world
                    .try_spawn_agent(AgentData {
                        position: Position::new(x, y),
                        ..AgentData::default()
                    })
                    .expect("seed clustered production-layout agent");
            }
        }
        world
            .try_spawn_agent(AgentData {
                position: Position::new(5_900.0, 2_900.0),
                ..AgentData::default()
            })
            .expect("seed distant production-layout outlier");
        let frame = RenderFrame::from_world(&world, ColorPaletteMode::Natural)
            .expect("production frame must assemble");

        for viewport in [(1_280.0, 720.0), (1_600.0, 900.0)] {
            let mut camera = Camera::default();
            let layout = layout_camera_for_frame(&mut camera, &frame, (0.0, 0.0), viewport);
            let visible_agent_diameters = viewport.0 / (layout.scale * agent_radius * 2.0);
            assert!(
                (visible_agent_diameters - 120.0).abs() <= 1e-3,
                "{viewport:?} production layout must show 120 agent diameters across, got \
                 {visible_agent_diameters}"
            );
            assert!(
                layout.scale * agent_radius >= 5.0,
                "{viewport:?} production layout must render an agent radius at five or more \
                 pixels, got {}",
                layout.scale * agent_radius
            );

            let population_center = camera
                .world_to_screen((240.0, 240.0))
                .expect("population median must remain visible");
            assert!(
                (population_center.0 - viewport.0 * 0.5).abs() <= 1e-3
                    && (population_center.1 - viewport.1 * 0.5).abs() <= 1e-3,
                "{viewport:?} production layout must center population density, got \
                 {population_center:?}"
            );
        }
    }

    #[test]
    fn render_frame_reads_live_boost_and_attacker_spike_state() {
        let config = ScriptBotsConfig {
            world_width: 600,
            world_height: 300,
            food_cell_size: 50,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("agent visual truth world");
        let agent = world
            .try_spawn_agent_with(
                AgentData {
                    position: Position::new(120.0, 120.0),
                    spike_length: 2.0,
                    // Deliberately stale opposite of the live output below.
                    boost: false,
                    ..AgentData::default()
                },
                |runtime| {
                    runtime.outputs[OutputChannel::Boost.index()] = 1.0;
                    runtime.outputs[OutputChannel::SpikeTarget.index()] = 1.0;
                    // This marks the victim, not this agent's attacking spike.
                    runtime.spiked = true;
                    runtime.combat.spike_attacker = false;
                },
            )
            .expect("seed visual truth agent");

        let frame = RenderFrame::from_world(&world, ColorPaletteMode::Natural)
            .expect("assemble visual truth frame");
        let rendered = &frame.agents[0];
        assert_eq!(rendered.agent_id, agent);
        assert_eq!(rendered.boost, 1.0, "live output, not stale SoA boost");
        assert!(rendered.spike_extended, "SpikeTarget drives extension");
        assert!(
            !rendered.spike_struck,
            "victim flag must not masquerade as an attacker strike"
        );
        assert!(rendered.spike_victim, "audio may still observe victim hits");

        world
            .try_update_agent_runtime(agent, |runtime| {
                runtime.outputs[OutputChannel::Boost.index()] = 0.0;
                runtime.outputs[OutputChannel::SpikeTarget.index()] = 0.0;
                runtime.spiked = false;
                runtime.combat.spike_attacker = true;
            })
            .expect("update visual truth runtime");
        let frame = RenderFrame::from_world(&world, ColorPaletteMode::Natural)
            .expect("reassemble visual truth frame");
        let rendered = &frame.agents[0];
        assert_eq!(rendered.boost, 0.0);
        assert!(!rendered.spike_extended);
        assert!(rendered.spike_struck);
        assert!(!rendered.spike_victim);
    }

    #[test]
    fn compact_lod_projects_all_five_thousand_agents_without_semantic_drop() {
        let config = ScriptBotsConfig {
            world_width: 2_400,
            world_height: 1_200,
            food_cell_size: 50,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("compact LOD world");
        world
            .try_spawn_agent_with(
                AgentData {
                    position: Position::new(20.0, 20.0),
                    heading: 0.0,
                    health: 2.0,
                    spike_length: 4.0,
                    boost: false,
                    ..AgentData::default()
                },
                |runtime| {
                    runtime.herbivore_tendency = 0.0;
                    runtime.selection = SelectionState::Selected;
                    runtime.outputs[OutputChannel::SpikeTarget.index()] = 1.0;
                },
            )
            .expect("seed compact LOD template");
        let frame = RenderFrame::from_world(&world, ColorPaletteMode::Natural)
            .expect("assemble compact LOD frame");
        let template = frame.agents[0].clone();
        let view = AgentLodView {
            offset: (0.0, 0.0),
            scale: 1.0,
            bounds: (0.0, 0.0, 2_400.0, 1_200.0),
        };
        let semantic_body_bin = |health: f32, age: u32| {
            let mut agent = template.clone();
            agent.health = health;
            agent.age = age;
            project_agent_lod(&frame, &agent, view, None, true)
                .expect("semantic compact projection")
                .body_bin
        };
        let full_health_bin = semantic_body_bin(2.0, 0);
        let mid_health_bin = semantic_body_bin(1.2, 0);
        let floor_health_bin = semantic_body_bin(0.05, 0);
        let old_mid_health_bin = semantic_body_bin(1.2, frame.agent_reference_age as u32);
        assert!(
            full_health_bin > mid_health_bin && mid_health_bin > floor_health_bin,
            "compact LOD must preserve full > mid > floor health ordering: \
             full={full_health_bin}, mid={mid_health_bin}, floor={floor_health_bin}"
        );
        assert!(
            mid_health_bin > old_mid_health_bin,
            "compact LOD must preserve age weathering at fixed health and diet: \
             young={mid_health_bin}, old={old_mid_health_bin}"
        );

        let mut agents = Vec::with_capacity(5_000);
        for index in 0_usize..5_000 {
            let mut agent = template.clone();
            agent.position = Position::new(
                12.0 + (index % 100) as f32 * 23.5,
                12.0 + (index / 100) as f32 * 23.5,
            );
            agent.heading = if index.is_multiple_of(2) {
                0.0
            } else {
                FRAC_PI_2
            };
            agent.herbivore_tendency = if index.is_multiple_of(2) { 0.0 } else { 1.0 };
            agent.health = if index.is_multiple_of(3) { 0.2 } else { 2.0 };
            agent.age = if index.is_multiple_of(5) {
                frame.agent_reference_age as u32
            } else {
                0
            };
            agent.boost = if index.is_multiple_of(7) { 1.0 } else { 0.0 };
            agent.selection = match index % 11 {
                0 => SelectionState::Selected,
                1 => SelectionState::Hovered,
                _ => SelectionState::None,
            };
            agents.push(agent);
        }

        let projected: Vec<_> = agents
            .iter()
            .filter_map(|agent| project_agent_lod(&frame, agent, view, None, true))
            .collect();
        assert_eq!(
            projected.len(),
            agents.len(),
            "compact LOD may simplify geometry but must never sample or drop visible agents"
        );
        let body_bins: std::collections::BTreeSet<_> =
            projected.iter().map(|agent| agent.body_bin).collect();
        assert!(
            body_bins.len() >= 4,
            "diet and health/age must survive compact LOD quantization: {body_bins:?}"
        );
        assert!(projected.iter().any(|agent| agent.boost_color.is_some()));
        assert!(
            projected
                .iter()
                .any(|agent| agent.selection == SelectionState::Selected)
        );
        assert!(
            projected
                .iter()
                .any(|agent| agent.selection == SelectionState::Hovered)
        );

        let facing_x =
            project_agent_lod(&frame, &agents[0], view, None, true).expect("x-facing projection");
        let facing_y =
            project_agent_lod(&frame, &agents[1], view, None, true).expect("y-facing projection");
        let x_tip = facing_x.silhouette.spike[2];
        let y_tip = facing_y.silhouette.spike[2];
        assert!(x_tip.0 > facing_x.silhouette.center.0);
        assert!(y_tip.1 > facing_y.silhouette.center.1);
        assert!(
            (x_tip.1 - facing_x.silhouette.center.1).abs()
                < (x_tip.0 - facing_x.silhouette.center.0).abs(),
            "x-facing compact silhouette must remain oriented"
        );
        assert!(
            (y_tip.0 - facing_y.silhouette.center.0).abs()
                < (y_tip.1 - facing_y.silhouette.center.1).abs(),
            "y-facing compact silhouette must remain oriented"
        );
    }

    #[derive(Default)]
    struct FakeGuiLifecycle {
        quit_requested: bool,
    }

    impl GuiQuitRequest for FakeGuiLifecycle {
        fn request_gui_quit(&mut self) {
            self.quit_requested = true;
        }
    }

    #[test]
    fn gui_launch_error_preserves_the_first_failure_and_quits_the_event_loop() {
        let slot = Mutex::new(None);
        let mut lifecycle = FakeGuiLifecycle::default();
        abort_gui_launch(
            &mut lifecycle,
            &slot,
            "could not open HUD window: adapter unavailable".into(),
        );
        abort_gui_launch(
            &mut lifecycle,
            &slot,
            "could not open simulation window: ignored".into(),
        );

        let error = slot
            .lock()
            .expect("launch error slot")
            .clone()
            .expect("recorded launch error");
        assert_eq!(
            error.to_string(),
            "GPUI launch failed: could not open HUD window: adapter unavailable"
        );
        assert!(
            lifecycle.quit_requested,
            "a window-open failure must terminate the GPUI application lifetime"
        );
    }

    #[test]
    fn gui_health_probe_preserves_errors() {
        let healthy: GuiHealthProbe = Arc::new(|| Ok(()));
        assert_eq!(gui_health_failure(&healthy), None);

        let failed: GuiHealthProbe = Arc::new(|| Err("injected REST serve failure".to_owned()));
        assert_eq!(
            gui_health_failure(&failed).as_deref(),
            Some("injected REST serve failure")
        );
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn gui_health_probe_contains_panics_in_unwinding_profiles() {
        let panicked: GuiHealthProbe = Arc::new(|| std::panic::panic_any("injected health panic"));
        assert_eq!(
            gui_health_failure(&panicked).as_deref(),
            Some("health probe panicked: injected health panic")
        );
    }

    #[test]
    fn gui_runtime_error_preserves_a_control_failure_over_later_launch_noise() {
        let slot = Mutex::new(None);
        record_gui_run_error(
            &slot,
            GuiRunError::ControlRuntime("injected MCP failure".to_owned()),
        );
        let mut lifecycle = FakeGuiLifecycle::default();
        abort_gui_launch(
            &mut lifecycle,
            &slot,
            "late window-close artifact".to_owned(),
        );

        let error = slot
            .lock()
            .expect("GUI error slot")
            .clone()
            .expect("recorded control failure");
        assert_eq!(
            error,
            GuiRunError::ControlRuntime("injected MCP failure".to_owned())
        );
        assert_eq!(
            error.to_string(),
            "GPUI control runtime failed: injected MCP failure"
        );
        assert!(lifecycle.quit_requested);
    }

    fn command_characterization_world() -> Arc<Mutex<WorldState>> {
        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 50,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        };
        Arc::new(Mutex::new(
            WorldState::new(config).expect("characterization world"),
        ))
    }

    #[test]
    fn manual_agent_injection_refuses_a_sealed_persistence_boundary() {
        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 50,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 1,
            rng_seed: Some(0x005E_A1ED),
            ..ScriptBotsConfig::default()
        };
        let (mut world, mut persistence) =
            WorldState::with_persistence(config, Box::new(scriptbots_core::NullPersistence))
                .expect("sealed-boundary world");
        persistence
            .step(&mut world)
            .expect("seal the first persistence boundary");
        assert_eq!(world.tick().0, 1);
        let world = Arc::new(Mutex::new(world));
        let drain = one_shot_command_drain(vec![ControlCommand::SpawnAgent {
            herbivore_tendency: 0.5,
        }]);
        let driver = gui_simulation_driver_with_step(
            &world,
            disabled_persistence_step_driver(&world),
            drain,
        );

        let (agent_count_before, rng_before, identity_before) = {
            let world = world.lock().expect("world lock");
            (
                world.agent_count(),
                world.random_streams_checkpoint(),
                world.identity_sequence_state(),
            )
        };
        driver
            .lock()
            .expect("GUI simulation driver lock")
            .drive_at(Instant::now());
        assert_eq!(
            world.lock().expect("world lock").agent_count(),
            agent_count_before,
            "a rejected manual ingress must not add an agent"
        );
        let world = world.lock().expect("world lock");
        assert_eq!(
            world.random_streams_checkpoint(),
            rng_before,
            "sealed GUI ingress must be rejected before Population-domain position/color sampling"
        );
        assert_eq!(world.identity_sequence_state(), identity_before);
    }

    fn disabled_persistence_step_driver(world: &Arc<Mutex<WorldState>>) -> WorldStepDriver {
        let world = Arc::clone(world);
        Arc::new(move || {
            world
                .lock()
                .expect("world mutex poisoned while executing test simulation step")
                .step()
        })
    }

    fn simulation_view(
        world: Arc<Mutex<WorldState>>,
        command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync + 'static>,
    ) -> SimulationView {
        let simulation_driver = gui_simulation_driver(&world, command_drain);
        simulation_view_with_driver(simulation_driver)
    }

    fn gui_simulation_driver(
        world: &Arc<Mutex<WorldState>>,
        command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync + 'static>,
    ) -> Arc<Mutex<GuiSimulationDriver>> {
        let simulation_step = disabled_persistence_step_driver(world);
        gui_simulation_driver_with_step(world, simulation_step, command_drain)
    }

    /// bd-jw6f: the speed control must change the OBSERVED TICK RATE, not merely assign
    /// a multiplier field.
    ///
    /// The failure mode this codebase keeps producing is a control wired to something
    /// nothing consumes, and a test asserting `speed_multiplier == 2.0` would pass for
    /// exactly that defect. So this drives the real consumer — GuiSimulationDriver's
    /// accumulator path — and counts ticks the world actually executed.
    ///
    /// Asserts the RATIO, not absolute rates. `drive_at` takes `now` as a parameter, so
    /// the clock is fabricated rather than slept on: the result is identical on an idle
    /// machine and on one running six agents. An intermittent test gets ignored, and an
    /// ignored test is the same as no coverage.
    #[test]
    fn simulation_speed_multiplier_changes_the_observed_tick_rate() {
        fn ticks_after(speed: f32, advance: Duration) -> u64 {
            let world = command_characterization_world();
            let driver = gui_simulation_driver_with_step(
                &world,
                disabled_persistence_step_driver(&world),
                Arc::new(Vec::new),
            );
            let mut driver = driver.lock().expect("driver lock");
            driver.apply_playback_state(&SimulationCommand {
                paused: Some(false),
                speed_multiplier: Some(speed),
                step_once: false,
            });
            let start = Instant::now();
            // Prime `last` without advancing; the first call establishes the baseline.
            driver.drive_at(start);
            let before = world.lock().expect("world lock").tick().0;
            driver.drive_at(start + advance);
            world.lock().expect("world lock").tick().0 - before
        }

        // 12 tick-intervals of clock: well under MAX_SIM_STEPS_PER_FRAME (240) and under
        // the 0.5s accumulator clamp, so neither limiter distorts the ratio.
        let advance = Duration::from_secs_f32(SIM_TICK_INTERVAL * 12.0);
        let single = ticks_after(1.0, advance);
        let double = ticks_after(2.0, advance);

        assert!(
            single > 0,
            "1x produced no ticks over {advance:?}; the speed control has no consumer"
        );
        assert_eq!(
            double,
            single * 2,
            "2x must execute exactly twice the ticks of 1x over the same clock advance \
             (1x={single}, 2x={double}); the multiplier is not reaching the accumulator"
        );
    }

    /// bd-jw6f: the spawn shortcuts must CHANGE POPULATION, not merely enqueue.
    ///
    /// Population is the right assertion because population drives every downstream
    /// metric: a spawn control that silently did nothing would look identical to a
    /// simulation that is simply not reproducing, and the debugger would go hunting in
    /// evolution or energy and find nothing wrong, because nothing is.
    ///
    /// This asserts the END of the chain - keystroke, ControlCommand::SpawnAgent,
    /// apply_control_command, world agent count - rather than that a command was
    /// enqueued. Enqueueing is what the old direct-mutation path could fake.
    #[test]
    fn spawn_shortcut_increases_population_through_the_control_path() {
        for (bias, label) in [(1.0_f32, "herbivore"), (0.0_f32, "carnivore")] {
            let world = command_characterization_world();
            let before = world.lock().expect("world lock").agent_count();

            let disposition = apply_control_command(
                &mut world.lock().expect("world lock"),
                ControlCommand::SpawnAgent {
                    herbivore_tendency: bias,
                },
            )
            .unwrap_or_else(|error| panic!("{label} spawn rejected: {error}"));

            let after = world.lock().expect("world lock").agent_count();
            assert!(
                matches!(disposition, ControlDisposition::WorldApplied),
                "{label} spawn must apply to the world, got {disposition:?}"
            );
            assert_eq!(
                after,
                before + 1,
                "{label} spawn must add exactly one agent (before={before}, after={after}); \
                 a control that enqueues but does not change population is indistinguishable \
                 from a simulation that is not reproducing"
            );
        }
    }

    fn gui_simulation_driver_with_step(
        world: &Arc<Mutex<WorldState>>,
        simulation_step: WorldStepDriver,
        command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync + 'static>,
    ) -> Arc<Mutex<GuiSimulationDriver>> {
        Arc::new(Mutex::new(GuiSimulationDriver::new(
            Arc::clone(world),
            simulation_step,
            command_drain,
        )))
    }

    fn one_shot_command_drain(
        commands: Vec<ControlCommand>,
    ) -> Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync + 'static> {
        let commands = Mutex::new(Some(commands));
        Arc::new(move || {
            commands
                .lock()
                .expect("one-shot command drain lock")
                .take()
                .unwrap_or_default()
        })
    }

    fn injected_step_error(detail: &str) -> scriptbots_core::WorldStepError {
        scriptbots_core::PersistenceSessionError::Unavailable {
            detail: detail.to_owned(),
        }
        .into()
    }

    fn simulation_view_with_driver(
        simulation_driver: Arc<Mutex<GuiSimulationDriver>>,
    ) -> SimulationView {
        SimulationView::new(
            simulation_driver,
            AnalyticsSnapshotProvider::empty(),
            "command characterization".into(),
            Arc::new(|_command: ControlCommand| true),
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(())),
        )
    }

    fn prime_exactly_one_driver_step(driver: &Arc<Mutex<GuiSimulationDriver>>, now: Instant) {
        let mut driver = driver.lock().expect("GUI simulation driver lock");
        driver.paused = false;
        driver.speed_multiplier = 1.0;
        driver.sim_accumulator = 0.0;
        driver.last_sim_instant = Some(now - Duration::from_secs_f32(SIM_TICK_INTERVAL * 1.25));
    }

    /// Mirror of the TUI-side e2e fixture (bd-16g.2.4): forced boom/crash cycles
    /// until the narrative ring holds at least `target` events. Both surfaces read
    /// the same ring, so pinning each surface to the ring proves parity.
    fn narrative_e2e_world(target: usize) -> Arc<Mutex<WorldState>> {
        let config = ScriptBotsConfig {
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            bot_speed: 0.0,
            population_minimum: 0,
            population_spawn_interval: 0,
            reproduction_energy_threshold: 0.0,
            persistence_interval: 0,
            chart_flush_interval: 0,
            narrative_interval: 1,
            narrative_capacity: 64,
            rng_seed: Some(0xE2E2_4A11),
            ..ScriptBotsConfig::default()
        };
        let world = Arc::new(std::sync::Mutex::new(
            WorldState::new(config).expect("narrative e2e world"),
        ));
        for _ in 0..24 {
            {
                let mut guard = world.lock().expect("world lock");
                if guard.narrative_events().len() >= target {
                    break;
                }
                // Scale the injection with the accumulated population so every boom
                // stays above the narrative policy's materiality floor.
                let to_inject = guard.agent_count() / 2 + 10;
                for _ in 0..to_inject {
                    guard
                        .try_inject_agent(AgentData::default())
                        .expect("e2e injection is finite");
                }
            }
            for _ in 0..40 {
                let mut guard = world.lock().expect("world lock");
                guard.step().expect("e2e boom step");
            }
            {
                let mut guard = world.lock().expect("world lock");
                let handles: Vec<AgentId> = guard.agents().iter_handles().collect();
                for id in handles {
                    guard
                        .try_update_agent_runtime(id, |runtime| {
                            runtime.energy = -1.0;
                        })
                        .expect("starve e2e population");
                }
            }
            for _ in 0..170 {
                let mut guard = world.lock().expect("world lock");
                guard.step().expect("e2e crash step");
            }
        }
        world
    }

    #[test]
    fn narrative_rail_e2e_snapshot_matches_the_worlds_ring() {
        let world = narrative_e2e_world(20);
        let (world_ticks, world_dropped) = {
            let guard = world.lock().expect("world lock");
            assert!(
                guard.narrative_events().len() >= 20,
                "e2e fixture must produce at least 20 narrative events, found {}",
                guard.narrative_events().len()
            );
            (
                guard
                    .narrative_events()
                    .iter()
                    .map(|event| (event.tick.0, event.kind))
                    .collect::<Vec<_>>(),
                guard.narrative_dropped_events(),
            )
        };

        let drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> = Arc::new(Vec::new);
        let mut view = simulation_view(Arc::clone(&world), drain);
        let snapshot = view.snapshot();
        let snapshot_sequence: Vec<(u64, NarrativeEventKind)> = snapshot
            .narrative
            .iter()
            .map(|event| (event.tick.0, event.kind))
            .collect();
        assert_eq!(
            snapshot_sequence, world_ticks,
            "the GPU lab rail must show the world's events in the world's order"
        );
        assert_eq!(
            snapshot.narrative_dropped, world_dropped,
            "the GPU lab rail must report the world's exact dropped count"
        );
    }

    /// bd-16g.4.3: the GPU lab panel consumes the same core explanations as the
    /// TUI — an unbound agent gets the identity-passthrough table, and the
    /// detail carries the bound flag the panel branches on.
    #[test]
    fn inspector_detail_carries_bound_state_and_passthrough_explanations() {
        let world = command_characterization_world();
        let agent = {
            let mut guard = world.lock().expect("world lock");
            let agent = guard
                .try_spawn_agent(AgentData {
                    position: Position::new(50.0, 50.0),
                    ..AgentData::default()
                })
                .expect("spawn unbound agent");
            guard.step().expect("one tick");
            agent
        };
        let guard = world.lock().expect("world lock");
        let detail = AgentInspectorDetails::from_world(&guard, agent, None)
            .expect("detail for the live agent");
        assert!(!detail.brain_bound, "fixture agent must be unbound");
        let outputs: &[f32; scriptbots_core::OUTPUT_SIZE] = detail.outputs
            [..scriptbots_core::OUTPUT_SIZE]
            .try_into()
            .expect("full output vector");
        let explanations = explain_outputs(outputs, detail.brain_bound, None, 3);
        assert_eq!(explanations.len(), scriptbots_core::OUTPUT_SIZE);
        for (index, explanation) in explanations.iter().enumerate() {
            assert_eq!(
                explanation.method,
                scriptbots_core::attribution::AttributionMethod::Unavailable(
                    scriptbots_core::attribution::AttributionUnavailable::IdentityPassthrough
                ),
                "unbound output {index} must be a passthrough, not an attribution"
            );
            assert!(explanation.inputs.is_empty());
            assert_eq!(explanation.raw_value, detail.outputs[index]);
        }
        assert_eq!(explanations[6].output_name, "boost");
        assert_eq!(explanations[3].output_name, "color_green");
    }

    #[test]
    fn minimal_canvas_is_a_presentation_only_projection() {
        let world = command_characterization_world();
        let drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> = Arc::new(Vec::new);
        let driver = gui_simulation_driver(&world, drain);
        let mut canvas = simulation_view_with_driver(Arc::clone(&driver));
        canvas.set_minimal_canvas_mode();
        let now = Instant::now();
        prime_exactly_one_driver_step(&driver, now);

        let _snapshot = canvas.snapshot();

        assert_eq!(world.lock().expect("world lock").tick().0, 0);
        assert!(canvas.minimal_canvas_mode);
        driver
            .lock()
            .expect("GUI simulation driver lock")
            .drive_at(now);
        assert_eq!(world.lock().expect("world lock").tick().0, 1);
    }

    #[test]
    fn gpui_brain_pull_is_client_isolated_cached_and_suppressed_for_canvas() {
        #[derive(Debug)]
        struct GuiInspectionBrain {
            calls: Arc<std::sync::atomic::AtomicUsize>,
        }

        impl scriptbots_core::BrainRunner for GuiInspectionBrain {
            fn kind(&self) -> &'static str {
                "gpui.inspection"
            }

            fn tick(
                &mut self,
                _inputs: &[f32; scriptbots_core::INPUT_SIZE],
            ) -> [f32; scriptbots_core::OUTPUT_SIZE] {
                [0.0; scriptbots_core::OUTPUT_SIZE]
            }

            fn inspect(
                &self,
                request: scriptbots_core::BrainInspection,
            ) -> Result<
                Option<scriptbots_core::BrainInspectionSnapshot>,
                scriptbots_core::BrainInspectionError,
            > {
                self.calls
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let scriptbots_core::BrainInspection::Activations(limits) = request;
                scriptbots_core::bound_brain_inspection(
                    self.kind(),
                    BrainActivations {
                        layers: vec![ActivationLayer {
                            name: "gpui".to_owned(),
                            width: 1,
                            height: 1,
                            values: vec![0.5],
                        }],
                        connections: Vec::new(),
                        truncated: false,
                    },
                    1,
                    limits,
                )
                .map(Some)
            }

            fn state_digest(&self) -> Option<u64> {
                Some(0x4750_5549_4252_4149)
            }
        }

        let world = command_characterization_world();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let focused_agent = {
            let mut guard = world.lock().expect("GPUI inspection world lock");
            let family = guard
                .brain_registry_mut()
                .expect("GPUI inspection registry mutation")
                .register_with_state_digest("gpui.inspection", 0x4750_5549_4252_4149, {
                    let calls = Arc::clone(&calls);
                    move |_rng| {
                        Ok(Box::new(GuiInspectionBrain {
                            calls: Arc::clone(&calls),
                        }))
                    }
                });
            let agent_id = guard
                .try_spawn_agent(AgentData::default())
                .expect("spawn GPUI inspection agent");
            guard
                .bind_agent_brain(agent_id, family)
                .expect("bind GPUI inspection brain");
            agent_id
        };
        let digest_before = world
            .lock()
            .expect("pre-GPUI-inspection world lock")
            .world_digest_v1()
            .expect("pre-GPUI-inspection digest");
        let drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> = Arc::new(Vec::new);
        let mut hud = simulation_view(Arc::clone(&world), Arc::clone(&drain));
        hud.inspector
            .lock()
            .expect("HUD inspector lock")
            .focused_agent = Some(focused_agent);

        let first = hud.snapshot();
        let first_detail = first
            .inspector
            .focused
            .as_ref()
            .expect("immediate paused HUD inspection");
        assert_eq!(first_detail.brain_request_revision, Some(1));
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        let second = hud.snapshot();
        assert_eq!(
            second
                .inspector
                .focused
                .as_ref()
                .expect("cached paused HUD inspection")
                .brain_request_revision,
            Some(1)
        );
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);

        let mut peer = simulation_view(Arc::clone(&world), Arc::clone(&drain));
        peer.inspector
            .lock()
            .expect("peer inspector lock")
            .focused_agent = Some(focused_agent);
        assert_ne!(hud.brain_client_id, peer.brain_client_id);
        assert_eq!(
            peer.snapshot()
                .inspector
                .focused
                .as_ref()
                .expect("peer client inspection")
                .brain_request_revision,
            Some(1)
        );
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 2);

        let mut canvas = simulation_view(Arc::clone(&world), drain);
        canvas.set_minimal_canvas_mode();
        canvas
            .inspector
            .lock()
            .expect("canvas inspector lock")
            .focused_agent = Some(focused_agent);
        let _ = canvas.snapshot();
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 2);
        assert_eq!(
            world
                .lock()
                .expect("post-GPUI-inspection world lock")
                .world_digest_v1()
                .expect("post-GPUI-inspection digest"),
            digest_before
        );
    }

    #[test]
    fn two_gpui_views_share_one_simulation_clock() {
        let world = command_characterization_world();
        let drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> = Arc::new(Vec::new);
        let driver = gui_simulation_driver(&world, drain);
        let mut hud = simulation_view_with_driver(Arc::clone(&driver));
        let mut canvas = simulation_view_with_driver(Arc::clone(&driver));
        canvas.set_minimal_canvas_mode();
        let now = Instant::now();
        prime_exactly_one_driver_step(&driver, now);

        let _ = hud.snapshot();
        let _ = canvas.snapshot();
        assert_eq!(
            world.lock().expect("world lock").tick().0,
            0,
            "neither GPUI paint callback may advance scientific time"
        );

        driver
            .lock()
            .expect("GUI simulation driver lock")
            .drive_at(now);
        let _ = hud.snapshot();
        let _ = canvas.snapshot();

        let tick = world.lock().expect("world lock").tick().0;
        assert_eq!(
            tick, 1,
            "two GPUI views must not independently advance one shared world"
        );
    }

    fn seeded_digest_trace_after_snapshot_schedule(
        snapshot_count: usize,
    ) -> Vec<(CharacterizationDigestV0, WorldDigestV1)> {
        assert!(
            snapshot_count <= 2,
            "the production GPUI shell has two views"
        );
        let config = ScriptBotsConfig {
            world_width: 128,
            world_height: 128,
            food_cell_size: 16,
            initial_food: 0.25,
            food_respawn_interval: 3,
            population_minimum: 0,
            population_spawn_interval: 0,
            reproduction_energy_threshold: 0.0,
            persistence_interval: 0,
            chart_flush_interval: 0,
            rng_seed: Some(0xD16E_57A7),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("seeded repaint-schedule world");
        let mut focused_agent = None;
        for position in [
            Position::new(12.0, 18.0),
            Position::new(64.0, 72.0),
            Position::new(111.0, 103.0),
        ] {
            let agent_id = world
                .try_spawn_agent(AgentData {
                    position,
                    ..AgentData::default()
                })
                .expect("seed deterministic repaint-schedule agent");
            focused_agent.get_or_insert(agent_id);
        }
        let focused_agent = focused_agent.expect("seeded repaint-schedule agent");
        let world = Arc::new(Mutex::new(world));

        let pending_commands = Arc::new(Mutex::new(Vec::new()));
        let drain_commands = Arc::clone(&pending_commands);
        let command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> =
            Arc::new(move || {
                let mut commands = drain_commands
                    .lock()
                    .expect("repaint-schedule command queue");
                std::mem::take(&mut *commands)
            });
        let driver = gui_simulation_driver(&world, command_drain);

        let mut views = Vec::with_capacity(snapshot_count);
        if snapshot_count > 0 {
            let hud = simulation_view_with_driver(Arc::clone(&driver));
            hud.inspector
                .lock()
                .expect("HUD inspector lock")
                .focused_agent = Some(focused_agent);
            views.push(hud);
        }
        if snapshot_count > 1 {
            let mut canvas = simulation_view_with_driver(Arc::clone(&driver));
            canvas.set_minimal_canvas_mode();
            views.push(canvas);
        }

        let now = Instant::now();
        let mut digest_trace = Vec::with_capacity(240);
        for expected_tick in 1..=240 {
            pending_commands
                .lock()
                .expect("repaint-schedule command queue")
                .push(ControlCommand::UpdateSimulation(SimulationCommand {
                    paused: None,
                    speed_multiplier: None,
                    step_once: true,
                }));
            driver
                .lock()
                .expect("GUI simulation driver lock")
                .drive_at(now);
            for view in &mut views {
                let _ = view.snapshot();
            }
            let world = world.lock().expect("repaint-schedule world");
            assert_eq!(
                world.tick().0,
                expected_tick,
                "every driver command must advance exactly one science tick"
            );
            digest_trace.push((
                world
                    .characterization_digest_v0()
                    .expect("characterization digest"),
                world.world_digest_v1().expect("canonical world digest"),
            ));
        }

        digest_trace
    }

    #[test]
    fn seeded_digest_trace_is_independent_of_gpui_snapshot_count() {
        let zero_snapshots = seeded_digest_trace_after_snapshot_schedule(0);
        let one_snapshot = seeded_digest_trace_after_snapshot_schedule(1);
        let two_snapshots = seeded_digest_trace_after_snapshot_schedule(2);

        assert_eq!(
            one_snapshot, zero_snapshots,
            "one GPUI snapshot per tick must not alter any science digest in the 240-tick trace"
        );
        assert_eq!(
            two_snapshots, zero_snapshots,
            "HUD plus canvas snapshots must not alter any science digest in the 240-tick trace"
        );
    }

    /// A live production GPUI session, driven by real keystrokes (bd-jw6f).
    ///
    /// The coverage map this exists to close found that 18 of 27 HUD shortcuts
    /// were never driven by any test, and that most of the ones which WERE
    /// driven appeared only in keybinding-registry checks — which prove a
    /// binding is registered, not that pressing it changes anything. A control
    /// that silently does nothing passes that kind of test. So this fixture
    /// dispatches through the same window the user types into and reads state
    /// back off the production view afterwards.
    struct ShortcutFixture {
        app: gpui::TestApp,
        hud: gpui::WindowHandle<SimulationView>,
        world: Arc<Mutex<WorldState>>,
        submitted: Arc<Mutex<Vec<ControlCommand>>>,
    }

    impl ShortcutFixture {
        fn install() -> Self {
            Self::install_with_agents(0)
        }

        /// Install with `agents` seeded, for controls whose effect is world-side
        /// (selection) rather than view-local.
        fn install_with_agents(agents: usize) -> Self {
            let world = command_characterization_world();
            {
                let mut guard = world.lock().expect("seed shortcut world");
                for _ in 0..agents {
                    guard
                        .try_spawn_agent(AgentData::default())
                        .expect("default agent is finite");
                }
            }
            // Two separate logs: `pending` is what the driver would drain, while
            // `submitted` retains every intent for assertions. Draining must not
            // erase the evidence a test is about to read.
            let pending: Arc<Mutex<Vec<ControlCommand>>> = Arc::new(Mutex::new(Vec::new()));
            let submitted: Arc<Mutex<Vec<ControlCommand>>> = Arc::new(Mutex::new(Vec::new()));
            let drain = Arc::clone(&pending);
            let command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> =
                Arc::new(move || {
                    let mut commands = drain.lock().expect("shortcut command queue");
                    std::mem::take(&mut *commands)
                });
            let record = Arc::clone(&submitted);
            let submit = Arc::clone(&pending);
            let command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync> =
                Arc::new(move |command| {
                    record
                        .lock()
                        .expect("shortcut submitted log")
                        .push(command.clone());
                    submit.lock().expect("shortcut command queue").push(command);
                    true
                });
            let step_world = Arc::clone(&world);
            let simulation_step: WorldStepDriver =
                Arc::new(move || step_world.lock().expect("shortcut world lock").step());
            let session = Arc::new(GuiSession::new(
                Arc::clone(&world),
                simulation_step,
                AnalyticsSnapshotProvider::empty(),
                command_drain,
                command_submit,
            ));
            let mut app = gpui::TestApp::new();
            let windows = app
                .update(|app| session.install(app))
                .expect("install production GPUI shortcut session");
            // One production repaint, so the view is in the state a user would
            // actually be typing at rather than a freshly constructed one.
            force_production_repaint(&mut app, windows.hud);
            Self {
                app,
                hud: windows.hud,
                world,
                submitted,
            }
        }

        /// Step the world and repaint until the playback timeline holds
        /// `frames` distinct snapshots.
        ///
        /// The stepping is essential, not incidental: `PlaybackTimeline::record`
        /// drops a snapshot whose tick matches the newest one it already holds,
        /// so repainting a frozen world records exactly one frame no matter how
        /// many times it is called.
        fn record_frames(&mut self, frames: usize) {
            for _ in 0..(frames * 4) {
                if self.read(|view| view.playback.timeline.len()) >= frames {
                    return;
                }
                {
                    let mut world = self.world.lock().expect("shortcut world lock");
                    let _ = world.step();
                }
                force_production_repaint(&mut self.app, self.hud);
            }
            let recorded = self.read(|view| view.playback.timeline.len());
            assert!(
                recorded >= frames,
                "the production renderer must record {frames} playback frames, got {recorded}"
            );
        }

        /// Every control intent the view has submitted so far.
        fn submitted(&self) -> Vec<ControlCommand> {
            self.submitted
                .lock()
                .expect("shortcut submitted log")
                .clone()
        }

        /// How many agents the WORLD currently marks selected.
        ///
        /// Read from the world rather than from any view-local mirror: selection
        /// is scientific state, and a control that only updated a renderer's copy
        /// of it would look correct here while the simulation disagreed.
        fn selected_agents(&self) -> usize {
            self.world
                .lock()
                .expect("shortcut world lock")
                .runtime()
                .iter()
                .filter(|(_, entry)| matches!(entry.selection, SelectionState::Selected))
                .count()
        }

        /// Dispatch a keystroke into the production HUD window.
        ///
        /// Every caller passes a literal that also appears in its own assertion
        /// message, so a parse failure is self-identifying without formatting the
        /// input into a panic here.
        fn press(&mut self, keystroke: &str) {
            let parsed = Keystroke::parse(keystroke).expect("test shortcut literal must parse");
            self.app.update(|app| {
                app.update_window(self.hud.into(), |_, window, app| {
                    window.dispatch_keystroke(parsed, app);
                })
                .expect("dispatch production HUD shortcut");
            });
        }

        /// Read a field off the production view.
        fn read<T>(&mut self, probe: impl Fn(&SimulationView) -> T) -> T {
            let hud = self
                .app
                .update(|app| self.hud.root(app))
                .expect("production HUD root");
            self.app.read_entity(&hud, |view, _| probe(view))
        }
    }

    fn force_production_repaint(
        app: &mut gpui::TestApp,
        handle: gpui::WindowHandle<SimulationView>,
    ) {
        app.update(|app| {
            app.update_window(handle.into(), |_, window, app| {
                window.refresh();
                window.draw(app).clear();
            })
            .expect("draw production GPUI test window");
        });
    }

    /// bd-jw6f, tier 3 and the panel/accessibility group: every view-local HUD
    /// toggle must change the state it names, and must toggle BACK.
    ///
    /// Both halves matter. Asserting only that the first press flips something
    /// would pass for a control that latches on and can never be undone — from
    /// the user's side that is just as broken as a control that does nothing,
    /// and it is the failure a "did the handler run" test cannot see either.
    ///
    /// Table-driven on purpose: the coverage map's finding was that these
    /// controls share one defect class, so they should share one proof and a
    /// newly added toggle should be one line to cover.
    #[test]
    fn every_view_local_hud_toggle_changes_the_state_it_names_and_toggles_back() {
        let cases: [(&str, &str, fn(&SimulationView) -> bool); 9] = [
            ("d", "ToggleAgentDraw", |view| view.controls.draw_agents),
            ("f", "ToggleFoodOverlay", |view| view.controls.draw_food),
            ("ctrl-shift-o", "ToggleAgentOutline", |view| {
                view.controls.agent_outline
            }),
            ("shift-f", "ToggleDebugOverlay", |view| view.debug.enabled),
            ("1", "ToggleStatsPanel", |view| view.hud.stats_open),
            ("2", "ToggleHistoryPanel", |view| view.hud.history_open),
            ("3", "TogglePerfPanel", |view| view.hud.perf_open),
            (",", "ToggleSettings", |view| view.settings_panel.open),
            ("n", "ToggleNarration", |view| {
                view.accessibility.narration_enabled
            }),
        ];

        for (keystroke, action, probe) in cases {
            let mut fixture = ShortcutFixture::install();
            let before = fixture.read(probe);

            fixture.press(keystroke);
            let after = fixture.read(probe);
            assert_ne!(
                before, after,
                "{action}: pressing '{keystroke}' must change the state it names \
                 (was {before}, still {after})"
            );

            fixture.press(keystroke);
            assert_eq!(
                fixture.read(probe),
                before,
                "{action}: '{keystroke}' must toggle, not latch"
            );
        }
    }

    /// Selection is SCIENTIFIC state, so both shortcuts are proven against the
    /// world rather than any renderer-side mirror. A control that only updated
    /// the renderer's copy would read as working here while the simulation
    /// disagreed — and selection feeds the inspector, the follow camera, and the
    /// brain panel, so a stale copy misattributes every number they show.
    #[test]
    fn select_all_and_clear_selection_submit_selection_intents() {
        let mut fixture = ShortcutFixture::install_with_agents(3);
        assert!(
            fixture.submitted().is_empty(),
            "the fixture must start with no submitted intents"
        );

        fixture.press("ctrl-a");
        let after_select = fixture.submitted();
        assert!(
            after_select.iter().any(|command| matches!(
                command,
                ControlCommand::UpdateSelection(update)
                    if update.state == SelectionState::Selected && !update.agent_ids.is_empty()
            )),
            "SelectAll: 'ctrl-a' must submit a selection intent naming agents, got {after_select:?}"
        );

        fixture.press("escape");
        let after_clear = fixture.submitted();
        assert!(
            after_clear.len() > after_select.len(),
            "ClearSelection: 'escape' must submit an intent of its own, not silently do nothing"
        );
        assert!(
            after_clear
                .iter()
                .skip(after_select.len())
                .any(|command| matches!(
                    command,
                    ControlCommand::UpdateSelection(update) if update.state == SelectionState::None
                )),
            "ClearSelection: 'escape' must submit a CLEARING selection intent, got {after_clear:?}"
        );

        // The routing contract bd-37m established: the renderer submits intent
        // and never writes selection itself. A handler that reached into the
        // world directly would pass the assertions above and still be wrong.
        assert_eq!(
            fixture.selected_agents(),
            0,
            "the GPUI handler must not mutate world selection before the intent is drained"
        );
    }

    /// ToggleSimulationPause was the last of the six "driven but not
    /// state-proven". Like the speed controls it routes through the command bus,
    /// so the observable state is the exact canonical intent.
    ///
    /// The speed field matters as much as the pause field: `set_simulation_paused`
    /// carries the CURRENT speed through, so a pause that silently reset speed to
    /// a default would be a real regression and is pinned here.
    #[test]
    fn the_pause_shortcut_submits_the_inverted_pause_intent() {
        let mut fixture = ShortcutFixture::install();
        let drive = fixture.read(|view| view.simulation_drive_snapshot());

        fixture.press("p");
        let submitted = fixture.submitted();
        assert!(
            submitted.iter().any(|command| matches!(
                command,
                ControlCommand::UpdateSimulation(update)
                    if update.paused == Some(!drive.paused)
                        && update.speed_multiplier == Some(drive.speed_multiplier)
                        && !update.step_once
            )),
            "ToggleSimulationPause: 'p' must submit paused={} carrying speed {}, got {submitted:?}",
            !drive.paused,
            drive.speed_multiplier
        );
    }

    /// TogglePlayback and GoLive, the last two of the six "driven but not
    /// state-proven".
    ///
    /// THE TIMELINE DEPTH IS LOAD-BEARING, and getting it wrong is what made
    /// this look broken (bd-zlmc). With a single recorded frame, `toggle_play`
    /// does set `Playing` — but the very next render calls `snapshot_for_render`,
    /// which finds `pointer + 1 == timeline.len()`, concludes playback has run
    /// off the end, and returns the mode to `Live`. That is correct behaviour:
    /// a one-frame timeline finishes the instant it starts. A test recording one
    /// frame therefore observes `Live` after pressing space and reads it as a
    /// dead control. Recording several frames gives playback somewhere to go.
    #[test]
    fn playback_shortcuts_move_the_playback_mode() {
        let mode = |view: &SimulationView| view.playback.mode();
        let mut fixture = ShortcutFixture::install();
        fixture.record_frames(4);
        assert_eq!(
            fixture.read(mode),
            PlaybackMode::Live,
            "a recording session must still be following live"
        );

        fixture.press("space");
        assert_eq!(
            fixture.read(mode),
            PlaybackMode::Playing,
            "TogglePlayback: 'space' must leave live and start playing the timeline"
        );

        fixture.press("space");
        assert_eq!(
            fixture.read(mode),
            PlaybackMode::Paused,
            "TogglePlayback: a second 'space' must pause, not latch on Playing"
        );

        fixture.press("g");
        assert_eq!(
            fixture.read(mode),
            PlaybackMode::Live,
            "GoLive: 'g' must return playback to live from a paused scrub"
        );
    }

    /// FocusFirstSelected, driven through the realistic flow: select, then focus.
    ///
    /// It resolves its target through `effective_selected_agents`, which prefers
    /// the pre-drain selection PROJECTION over the world's canonical selection —
    /// so this also proves the projection bd-37m's intent routing publishes is
    /// actually consumable by a downstream control, not just recorded.
    #[test]
    fn focus_first_selected_focuses_an_agent_from_the_selection() {
        let mut fixture = ShortcutFixture::install_with_agents(3);
        let focused =
            |view: &SimulationView| view.inspector.lock().expect("inspector lock").focused_agent;
        assert_eq!(
            fixture.read(focused),
            None,
            "the fixture must start with nothing focused"
        );

        // Pressing ctrl-f with no selection must do nothing rather than focus an
        // arbitrary agent — a focus that invents a target is worse than none.
        fixture.press("ctrl-f");
        assert_eq!(
            fixture.read(focused),
            None,
            "FocusFirstSelected must not invent a target when nothing is selected"
        );

        fixture.press("ctrl-a");
        fixture.press("ctrl-f");
        let after = fixture.read(focused);
        assert!(
            after.is_some(),
            "FocusFirstSelected: 'ctrl-f' must focus an agent once a selection exists"
        );

        // And it must be an agent that actually exists in the world, not a stale
        // or fabricated handle.
        let live: Vec<AgentId> = fixture
            .world
            .lock()
            .expect("shortcut world lock")
            .agents()
            .iter_handles()
            .collect();
        assert!(
            after.is_some_and(|id| live.contains(&id)),
            "the focused handle must name a live agent, got {after:?}"
        );
    }

    /// bd-jw6f tier 1, the group the map called costliest to get wrong: "if these
    /// are inert the app feels broken on first contact".
    ///
    /// Speed already had a state proof — the GuiSimulationDriver ratio test —
    /// but that drives the DRIVER directly and never presses a key, so the
    /// binding half was still unproven. This closes it from the other end:
    /// keystroke in, canonical intent out.
    ///
    /// Asserting the EXACT submitted value rather than "a command was submitted"
    /// is the point. Speed is computed as `clamp(current + delta, 0.25, 4.0)`
    /// rounded to 2dp, so a wrong delta, a wrong sign, or a rounding slip would
    /// all still submit *something* and pass a weaker assertion.
    #[test]
    fn speed_shortcuts_submit_the_exact_canonical_speed_intent() {
        let mut fixture = ShortcutFixture::install();
        let base = fixture.read(|view| view.simulation_drive_snapshot().speed_multiplier);

        fixture.press("shift-=");
        let after_increase = fixture.submitted();
        let expected_up = ((base + 0.25).clamp(0.25, 4.0) * 100.0).round() / 100.0;
        assert!(
            after_increase.iter().any(|command| matches!(
                command,
                ControlCommand::UpdateSimulation(update)
                    if update.speed_multiplier == Some(expected_up)
            )),
            "IncreaseSimulationSpeed: 'shift-=' must submit speed {expected_up} from {base}, \
             got {after_increase:?}"
        );

        fixture.press("-");
        let after_decrease = fixture.submitted();
        assert!(
            after_decrease.len() > after_increase.len(),
            "DecreaseSimulationSpeed: '-' must submit an intent of its own"
        );
        // The view recomputes from the DRIVER's speed, which the fixture never
        // advances, so the decrease is one step below the same base.
        let expected_down = ((base - 0.25).clamp(0.25, 4.0) * 100.0).round() / 100.0;
        assert!(
            after_decrease
                .iter()
                .skip(after_increase.len())
                .any(|command| matches!(
                    command,
                    ControlCommand::UpdateSimulation(update)
                        if update.speed_multiplier == Some(expected_down)
                )),
            "DecreaseSimulationSpeed: '-' must submit speed {expected_down}, got {after_decrease:?}"
        );

        // Every speed intent must carry the CURRENT pause state rather than
        // silently resuming a paused world — changing speed is not a play command.
        let paused_now = fixture.read(|view| view.simulation_drive_snapshot().paused);
        assert!(
            after_decrease.iter().all(|command| matches!(
                command,
                ControlCommand::UpdateSimulation(update)
                    if update.paused == Some(paused_now) && !update.step_once
            )),
            "a speed change must preserve pause state and never request a step"
        );
    }

    /// CyclePalette was "driven but not state-proven". The accessibility palette
    /// is a five-way cycle, so proving it means more than "it changed": pressing
    /// it five times must return to where it started, which a control that
    /// advanced by two — or that saturated at the last entry — would fail.
    #[test]
    fn cycling_the_palette_advances_and_returns_to_where_it_started() {
        let mut fixture = ShortcutFixture::install();
        let palette = |view: &SimulationView| view.accessibility.palette;

        let start = fixture.read(palette);
        fixture.press("ctrl-p");
        assert_ne!(
            start,
            fixture.read(palette),
            "CyclePalette: 'ctrl-p' must advance the accessibility palette"
        );

        // Four more presses complete the five-entry cycle.
        for _ in 0..4 {
            fixture.press("ctrl-p");
        }
        assert_eq!(
            fixture.read(palette),
            start,
            "CyclePalette must cycle through all five palettes and wrap, not saturate"
        );
    }

    /// FitWorld was "driven but not state-proven". Proven by moving the camera
    /// away first: asserting the post-press state alone would pass trivially if
    /// the camera happened to already be fitted, which is exactly the vacuous
    /// coverage this bead exists to replace.
    #[test]
    fn fit_world_returns_a_panned_camera_to_the_fitted_view() {
        let mut fixture = ShortcutFixture::install();

        // fit_world is a no-op until a render has recorded a base scale, so an
        // unrecorded camera would make this test assert against the guard.
        let fitted = fixture.read(|view| {
            let camera = view.camera.lock().expect("camera lock");
            (camera.zoom(), camera.offset())
        });

        fixture.read(|view| {
            let mut camera = view.camera.lock().expect("camera lock");
            camera.start_pan(gpui::point(gpui::px(0.0), gpui::px(0.0)));
            camera.update_pan(gpui::point(gpui::px(64.0), gpui::px(48.0)));
            camera.end_pan();
        });
        let panned = fixture.read(|view| view.camera.lock().expect("camera lock").offset());
        assert_ne!(
            panned, fitted.1,
            "fixture precondition: the camera must actually be off-centre before '0'"
        );

        fixture.press("0");
        let after = fixture.read(|view| {
            let camera = view.camera.lock().expect("camera lock");
            (camera.zoom(), camera.offset())
        });
        assert_eq!(
            after.1,
            (0.0, 0.0),
            "FitWorld: '0' must recentre the camera, got offset {:?}",
            after.1
        );
    }

    /// The brush lives behind a mutex on the shared inspector rather than on the
    /// view, so it gets its own probe — but the same two-way proof.
    #[test]
    fn the_brush_shortcut_toggles_the_shared_inspector_state() {
        let mut fixture = ShortcutFixture::install();
        let brush =
            |view: &SimulationView| view.inspector.lock().expect("inspector lock").brush_enabled;
        let before = fixture.read(brush);
        fixture.press("b");
        assert_ne!(
            before,
            fixture.read(brush),
            "ToggleBrush: 'b' must change inspector.brush_enabled"
        );
        fixture.press("b");
        assert_eq!(
            fixture.read(brush),
            before,
            "ToggleBrush: 'b' must toggle, not latch"
        );
    }

    /// Follow mode is a tri-state rather than a boolean, so "it changed" is not
    /// enough — the two follow shortcuts must reach DIFFERENT modes, or one of
    /// them is silently the other.
    #[test]
    fn the_two_follow_shortcuts_select_distinct_follow_modes() {
        let follow = |view: &SimulationView| view.controls.follow_mode;

        let mut oldest = ShortcutFixture::install();
        let initial = oldest.read(follow);
        oldest.press("o");
        let after_oldest = oldest.read(follow);
        assert_ne!(
            initial, after_oldest,
            "FollowOldest: 'o' must change controls.follow_mode"
        );

        let mut selected = ShortcutFixture::install();
        selected.press("shift-s");
        let after_selected = selected.read(follow);
        assert_ne!(
            initial, after_selected,
            "FollowSelected: 'shift-s' must change controls.follow_mode"
        );
        assert_ne!(
            after_oldest, after_selected,
            "'o' and 'shift-s' must select different follow modes, not the same one"
        );
    }

    #[test]
    fn gpui_s_shortcut_submits_one_step_and_leaves_simulation_paused() {
        let world = command_characterization_world();
        let pending_commands = Arc::new(Mutex::new(vec![ControlCommand::Pause]));
        let drain_commands = Arc::clone(&pending_commands);
        let command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> =
            Arc::new(move || {
                let mut commands = drain_commands.lock().expect("single-step command queue");
                std::mem::take(&mut *commands)
            });
        let submit_commands = Arc::clone(&pending_commands);
        let command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync> =
            Arc::new(move |command| {
                submit_commands
                    .lock()
                    .expect("single-step command queue")
                    .push(command);
                true
            });
        let step_world = Arc::clone(&world);
        let simulation_step: WorldStepDriver =
            Arc::new(move || step_world.lock().expect("single-step world lock").step());
        let session = Arc::new(GuiSession::new(
            Arc::clone(&world),
            simulation_step,
            AnalyticsSnapshotProvider::empty(),
            command_drain,
            command_submit,
        ));

        let mut app = gpui::TestApp::new();
        let windows = app
            .update(|app| session.install(app))
            .expect("install production GPUI single-step session");
        assert!(
            session
                .simulation_driver
                .lock()
                .expect("single-step driver lock")
                .snapshot()
                .paused,
            "the fixture must start scientifically paused"
        );
        assert_eq!(
            world.lock().expect("single-step world lock").tick().0,
            0,
            "installing the windows must not advance science"
        );
        assert!(
            pending_commands
                .lock()
                .expect("single-step command queue")
                .is_empty(),
            "the startup pause command must be drained before keyboard input"
        );

        force_production_repaint(&mut app, windows.hud);
        let hud = app
            .update(|app| windows.hud.root(app))
            .expect("production HUD root");
        app.update_entity(&hud, |view, _| {
            assert!(
                !view.playback.timeline.is_empty(),
                "the production render must record a frame before rewinding"
            );
            view.playback.restart();
            assert!(matches!(view.playback.mode(), PlaybackMode::Paused));
        });

        app.update(|app| {
            app.update_window(windows.hud.into(), |_, window, app| {
                window.dispatch_keystroke(
                    Keystroke::parse("s").expect("valid single-step shortcut"),
                    app,
                );
            })
            .expect("dispatch production HUD single-step shortcut");
        });
        assert_eq!(
            pending_commands
                .lock()
                .expect("single-step command queue")
                .as_slice(),
            &[ControlCommand::Step],
            "one S keypress must enqueue exactly one canonical step command"
        );
        app.read_entity(&hud, |view, _| {
            assert!(
                matches!(view.playback.mode(), PlaybackMode::Live),
                "single-step must return the invoking view to the live frame"
            );
        });

        app.advance_clock(Duration::from_secs_f32(SIM_TICK_INTERVAL));
        app.run_until_parked();
        assert_eq!(
            world.lock().expect("single-step world lock").tick().0,
            1,
            "one S keypress must advance science by exactly one tick"
        );
        assert!(
            session
                .simulation_driver
                .lock()
                .expect("single-step driver lock")
                .snapshot()
                .paused,
            "single-step must leave the simulation paused"
        );

        app.advance_clock(Duration::from_secs_f32(SIM_TICK_INTERVAL * 2.0));
        app.run_until_parked();
        assert_eq!(
            world.lock().expect("single-step world lock").tick().0,
            1,
            "a single-step must not schedule a second scientific tick"
        );
        app.update(|app| app.shutdown());
    }

    #[test]
    fn gpui_closed_world_shortcut_submits_one_intent_without_direct_mutation() {
        let world = command_characterization_world();
        let submitted_commands = Arc::new(Mutex::new(Vec::new()));
        let captured_commands = Arc::clone(&submitted_commands);
        let command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync> =
            Arc::new(move |command| {
                captured_commands
                    .lock()
                    .expect("closed-world submitted command queue")
                    .push(command);
                true
            });
        let drain_commands = Arc::clone(&submitted_commands);
        let command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> =
            Arc::new(move || {
                let mut commands = drain_commands
                    .lock()
                    .expect("closed-world submitted command queue");
                std::mem::take(&mut *commands)
            });
        let session = Arc::new(GuiSession::new(
            Arc::clone(&world),
            disabled_persistence_step_driver(&world),
            AnalyticsSnapshotProvider::empty(),
            command_drain,
            command_submit,
        ));

        let mut app = gpui::TestApp::new();
        let windows = app
            .update(|app| session.install(app))
            .expect("install production GPUI closed-world session");
        assert!(!world.lock().expect("closed-world world lock").is_closed());

        app.update(|app| {
            app.update_window(windows.hud.into(), |_, window, app| {
                window.dispatch_keystroke(
                    Keystroke::parse("c").expect("valid closed-world shortcut"),
                    app,
                );
            })
            .expect("dispatch production HUD closed-world shortcut");
        });

        let mut expected_config = world
            .lock()
            .expect("closed-world world lock")
            .config()
            .clone();
        expected_config.closed = true;
        assert_eq!(
            submitted_commands
                .lock()
                .expect("closed-world submitted command queue")
                .as_slice(),
            &[ControlCommand::UpdateConfig(Box::new(expected_config))],
            "one C keypress must enqueue the exact canonical closed-world intent"
        );
        assert!(
            !world.lock().expect("closed-world world lock").is_closed(),
            "the GPUI handler must not mutate scientific world state before the intent is drained"
        );
        app.advance_clock(Duration::from_secs_f32(SIM_TICK_INTERVAL));
        app.run_until_parked();
        assert!(
            submitted_commands
                .lock()
                .expect("closed-world submitted command queue")
                .is_empty(),
            "the production driver must drain the admitted closed-world intent"
        );
        assert!(
            world.lock().expect("closed-world world lock").is_closed(),
            "the shared command application path must apply the GPUI closed-world intent"
        );
        app.update(|app| app.shutdown());
    }

    #[test]
    fn gpui_select_all_shortcut_submits_selection_intent_without_direct_mutation() {
        let world = command_characterization_world();
        let agent_id = world
            .lock()
            .expect("select-all world lock")
            .try_spawn_agent(AgentData::default())
            .expect("spawn select-all fixture agent");
        let submitted_commands = Arc::new(Mutex::new(Vec::new()));
        let captured_commands = Arc::clone(&submitted_commands);
        let command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync> =
            Arc::new(move |command| {
                captured_commands
                    .lock()
                    .expect("select-all submitted command queue")
                    .push(command);
                true
            });
        let drain_commands = Arc::clone(&submitted_commands);
        let command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> =
            Arc::new(move || {
                let mut commands = drain_commands
                    .lock()
                    .expect("select-all submitted command queue");
                std::mem::take(&mut *commands)
            });
        let session = Arc::new(GuiSession::new(
            Arc::clone(&world),
            disabled_persistence_step_driver(&world),
            AnalyticsSnapshotProvider::empty(),
            command_drain,
            command_submit,
        ));

        let mut app = gpui::TestApp::new();
        let windows = app
            .update(|app| session.install(app))
            .expect("install production GPUI select-all session");

        app.update(|app| {
            app.update_window(windows.hud.into(), |_, window, app| {
                window.dispatch_keystroke(
                    Keystroke::parse("ctrl-a").expect("valid select-all shortcut"),
                    app,
                );
            })
            .expect("dispatch production HUD select-all shortcut");
        });

        assert_eq!(
            submitted_commands
                .lock()
                .expect("select-all submitted command queue")
                .as_slice(),
            &[ControlCommand::UpdateSelection(SelectionUpdate {
                mode: SelectionMode::Replace,
                agent_ids: vec![agent_id.raw()],
                state: SelectionState::Selected,
            })],
            "one Ctrl-A keypress must enqueue the exact canonical selection intent"
        );
        assert!(
            matches!(
                world
                    .lock()
                    .expect("select-all world lock")
                    .agent_runtime(agent_id)
                    .expect("select-all fixture runtime")
                    .selection,
                SelectionState::None
            ),
            "the GPUI handler must not mutate selection before the intent is drained"
        );
        assert_eq!(
            session
                .selection_projection
                .lock()
                .expect("select-all selection projection")
                .as_deref(),
            Some(&[agent_id][..]),
            "the HUD and canvas must share the admitted pre-drain selection projection"
        );
        app.advance_clock(Duration::from_secs_f32(SIM_TICK_INTERVAL));
        app.run_until_parked();
        assert!(
            submitted_commands
                .lock()
                .expect("select-all submitted command queue")
                .is_empty(),
            "the production driver must drain the admitted selection intent"
        );
        assert!(
            matches!(
                world
                    .lock()
                    .expect("select-all world lock")
                    .agent_runtime(agent_id)
                    .expect("select-all fixture runtime")
                    .selection,
                SelectionState::Selected
            ),
            "the shared command application path must apply the GPUI selection intent"
        );
        app.update(|app| app.shutdown());
    }

    #[test]
    fn rejected_selection_clear_preserves_scientific_and_presentation_state() {
        let world = command_characterization_world();
        let agent_id = {
            let mut world = world.lock().expect("rejected-clear world lock");
            let agent_id = world
                .try_spawn_agent(AgentData::default())
                .expect("spawn rejected-clear fixture agent");
            let _ = world.apply_selection_update(SelectionUpdate {
                mode: SelectionMode::Replace,
                agent_ids: vec![agent_id.raw()],
                state: SelectionState::Selected,
            });
            agent_id
        };
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let captured_attempts = Arc::clone(&attempts);
        let driver = gui_simulation_driver(&world, Arc::new(Vec::new));
        let projection = Arc::new(Mutex::new(None));
        let mut view = SimulationView::new(
            driver,
            AnalyticsSnapshotProvider::empty(),
            "rejected selection".into(),
            Arc::new(move |_command| {
                captured_attempts.fetch_add(1, AtomicOrdering::Relaxed);
                false
            }),
            Arc::clone(&projection),
            Arc::new(Mutex::new(())),
        );
        view.inspector
            .lock()
            .expect("rejected-clear inspector")
            .focused_agent = Some(agent_id);

        assert!(!view.clear_all_selections());
        assert_eq!(attempts.load(AtomicOrdering::Relaxed), 1);
        assert!(
            matches!(
                world
                    .lock()
                    .expect("rejected-clear world lock")
                    .agent_runtime(agent_id)
                    .expect("rejected-clear fixture runtime")
                    .selection,
                SelectionState::Selected
            ),
            "rejected admission must leave canonical selection untouched"
        );
        assert_eq!(
            view.inspector
                .lock()
                .expect("rejected-clear inspector")
                .focused_agent,
            Some(agent_id),
            "rejected admission must leave presentation focus untouched"
        );
        assert!(
            projection
                .lock()
                .expect("rejected-clear projection")
                .is_none(),
            "rejected admission must not create an optimistic selection projection"
        );
        assert!(
            view.selection_events.is_empty(),
            "rejected admission must not append an applied-selection event"
        );
    }

    #[test]
    fn rapid_shift_clicks_project_pending_selection_in_fifo_order() {
        let world = command_characterization_world();
        let agent_id = world
            .lock()
            .expect("rapid-selection world lock")
            .try_spawn_agent(AgentData {
                position: Position::new(50.0, 50.0),
                ..AgentData::default()
            })
            .expect("spawn rapid-selection fixture agent");
        let pending_commands = Arc::new(Mutex::new(Vec::new()));
        let drain_commands = Arc::clone(&pending_commands);
        let command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> =
            Arc::new(move || {
                let mut commands = drain_commands
                    .lock()
                    .expect("rapid-selection command queue");
                std::mem::take(&mut *commands)
            });
        let driver = gui_simulation_driver(&world, command_drain);
        let submit_commands = Arc::clone(&pending_commands);
        let projection = Arc::new(Mutex::new(None));
        let mut view = SimulationView::new(
            Arc::clone(&driver),
            AnalyticsSnapshotProvider::empty(),
            "rapid selection".into(),
            Arc::new(move |command| {
                submit_commands
                    .lock()
                    .expect("rapid-selection command queue")
                    .push(command);
                true
            }),
            Arc::clone(&projection),
            Arc::new(Mutex::new(())),
        );
        {
            let mut camera = view.camera.lock().expect("rapid-selection camera");
            camera.ensure_default_zoom(1.0);
            camera.record_render_metrics((0.0, 0.0), (100.0, 100.0), (100.0, 100.0), 1.0);
        }
        let agent_screen_position = point(px(50.0), px(50.0));

        assert!(view.update_selection_from_point(agent_screen_position, true));
        assert!(view.update_selection_from_point(agent_screen_position, true));

        assert_eq!(
            pending_commands
                .lock()
                .expect("rapid-selection command queue")
                .as_slice(),
            &[
                ControlCommand::UpdateSelection(SelectionUpdate {
                    mode: SelectionMode::Add,
                    agent_ids: vec![agent_id.raw()],
                    state: SelectionState::Selected,
                }),
                ControlCommand::UpdateSelection(SelectionUpdate {
                    mode: SelectionMode::Clear,
                    agent_ids: vec![agent_id.raw()],
                    state: SelectionState::None,
                }),
            ],
            "the second pre-drain shift-click must toggle the projected state back off"
        );
        assert!(
            matches!(
                world
                    .lock()
                    .expect("rapid-selection world lock")
                    .agent_runtime(agent_id)
                    .expect("rapid-selection fixture runtime")
                    .selection,
                SelectionState::None
            ),
            "the GUI must not mutate canonical selection before command drain"
        );
        assert!(
            projection
                .lock()
                .expect("rapid-selection projection")
                .as_deref()
                .is_some_and(|selected| selected.is_empty())
        );
        assert_eq!(view.selection_events.len(), 2);
        assert_eq!(view.selection_events[0].tick, 0);
        assert_eq!(view.selection_events[0].total_selected, 1);
        assert_eq!(view.selection_events[0].sample_ids, vec![agent_id]);
        assert_eq!(view.selection_events[1].tick, 0);
        assert_eq!(view.selection_events[1].total_selected, 0);

        driver
            .lock()
            .expect("rapid-selection driver")
            .drive_at(Instant::now());
        assert!(
            pending_commands
                .lock()
                .expect("rapid-selection command queue")
                .is_empty()
        );
        assert!(
            matches!(
                world
                    .lock()
                    .expect("rapid-selection world lock")
                    .agent_runtime(agent_id)
                    .expect("rapid-selection fixture runtime")
                    .selection,
                SelectionState::None
            ),
            "FIFO application of Add then Clear must reproduce the projected final state"
        );
        let _ = view.snapshot();
        assert!(
            projection
                .lock()
                .expect("rapid-selection projection")
                .is_none(),
            "a canonical state matching the projection must clear pending presentation state"
        );
    }

    #[test]
    fn hover_is_view_local_command_free_and_digest_neutral() {
        let world = command_characterization_world();
        let agent_id = world
            .lock()
            .expect("hover-isolation world lock")
            .try_spawn_agent(AgentData::default())
            .expect("spawn hover-isolation fixture agent");
        let driver = gui_simulation_driver(&world, Arc::new(Vec::new));
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let projection = Arc::new(Mutex::new(None));
        let submission = Arc::new(Mutex::new(()));
        let make_view = |title: &'static str| {
            let captured_attempts = Arc::clone(&attempts);
            SimulationView::new(
                Arc::clone(&driver),
                AnalyticsSnapshotProvider::empty(),
                title.into(),
                Arc::new(move |_command| {
                    captured_attempts.fetch_add(1, AtomicOrdering::Relaxed);
                    true
                }),
                Arc::clone(&projection),
                Arc::clone(&submission),
            )
        };
        let mut hud_view = make_view("hover HUD");
        let mut canvas_view = make_view("hover canvas");
        let digest_before = world
            .lock()
            .expect("hover-isolation world lock")
            .world_digest_v1()
            .expect("hover-isolation digest");

        assert!(hud_view.apply_hover_change(Some(agent_id)));
        assert_eq!(
            attempts.load(AtomicOrdering::Relaxed),
            0,
            "hover must not enter the scientific command bus"
        );
        assert!(
            matches!(
                world
                    .lock()
                    .expect("hover-isolation world lock")
                    .agent_runtime(agent_id)
                    .expect("hover-isolation fixture runtime")
                    .selection,
                SelectionState::None
            ),
            "hover must not write canonical selection state"
        );
        assert_eq!(
            world
                .lock()
                .expect("hover-isolation world lock")
                .world_digest_v1()
                .expect("hover-isolation digest"),
            digest_before,
            "presentation-only hover must be science-digest neutral"
        );

        let hud_snapshot = hud_view.snapshot();
        assert_eq!(
            hud_snapshot
                .inspector
                .hovered
                .as_ref()
                .map(|entry| entry.agent_id),
            Some(agent_id)
        );
        assert!(
            hud_snapshot
                .render_frame
                .as_ref()
                .and_then(|frame| { frame.agents.iter().find(|agent| agent.agent_id == agent_id) })
                .is_some_and(|agent| matches!(agent.selection, SelectionState::Hovered))
        );

        let canvas_snapshot = canvas_view.snapshot();
        assert!(canvas_snapshot.inspector.hovered.is_none());
        assert!(
            canvas_snapshot
                .render_frame
                .as_ref()
                .and_then(|frame| { frame.agents.iter().find(|agent| agent.agent_id == agent_id) })
                .is_some_and(|agent| matches!(agent.selection, SelectionState::None)),
            "hover in one GPUI view must not leak into the other view"
        );

        let _ = world
            .lock()
            .expect("hover-isolation world lock")
            .apply_selection_update(SelectionUpdate {
                mode: SelectionMode::Replace,
                agent_ids: vec![agent_id.raw()],
                state: SelectionState::Selected,
            });
        let selected_snapshot = hud_view.snapshot();
        assert!(selected_snapshot.inspector.hovered.is_none());
        assert!(
            selected_snapshot
                .render_frame
                .as_ref()
                .and_then(|frame| { frame.agents.iter().find(|agent| agent.agent_id == agent_id) })
                .is_some_and(|agent| matches!(agent.selection, SelectionState::Selected)),
            "canonical Selected must dominate local hover presentation"
        );
        assert_eq!(attempts.load(AtomicOrdering::Relaxed), 0);
    }

    fn seeded_digest_trace_after_production_repaints(
        repaint_count: usize,
    ) -> Vec<(CharacterizationDigestV0, WorldDigestV1)> {
        assert!(
            repaint_count <= 2,
            "the production GPUI shell has two views"
        );
        let config = ScriptBotsConfig {
            world_width: 128,
            world_height: 128,
            food_cell_size: 16,
            initial_food: 0.25,
            food_respawn_interval: 3,
            population_minimum: 0,
            population_spawn_interval: 0,
            reproduction_energy_threshold: 0.0,
            persistence_interval: 0,
            chart_flush_interval: 0,
            rng_seed: Some(0xD16E_57A7),
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).expect("seeded production-repaint world");
        let mut focused_agent = None;
        for position in [
            Position::new(12.0, 18.0),
            Position::new(64.0, 72.0),
            Position::new(111.0, 103.0),
        ] {
            let agent_id = world
                .try_spawn_agent(AgentData {
                    position,
                    ..AgentData::default()
                })
                .expect("seed production-repaint agent");
            focused_agent.get_or_insert(agent_id);
        }
        let focused_agent = focused_agent.expect("seeded production-repaint agent");
        let world = Arc::new(Mutex::new(world));

        let drain_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pending_commands = Arc::new(Mutex::new(Vec::new()));
        let drain_commands = Arc::clone(&pending_commands);
        let counted_drains = Arc::clone(&drain_calls);
        let command_drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> =
            Arc::new(move || {
                counted_drains.fetch_add(1, AtomicOrdering::Relaxed);
                let mut commands = drain_commands
                    .lock()
                    .expect("production-repaint command queue");
                std::mem::take(&mut *commands)
            });
        let submit_commands = Arc::clone(&pending_commands);
        let command_submit: Arc<dyn Fn(ControlCommand) -> bool + Send + Sync> =
            Arc::new(move |command| {
                submit_commands
                    .lock()
                    .expect("production-repaint command queue")
                    .push(command);
                true
            });
        let step_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted_steps = Arc::clone(&step_calls);
        let step_world = Arc::clone(&world);
        let simulation_step: WorldStepDriver = Arc::new(move || {
            counted_steps.fetch_add(1, AtomicOrdering::Relaxed);
            step_world
                .lock()
                .expect("production-repaint world lock")
                .step()
        });
        let session = Arc::new(GuiSession::new(
            Arc::clone(&world),
            simulation_step,
            AnalyticsSnapshotProvider::empty(),
            command_drain,
            command_submit,
        ));

        let digest_before_install = {
            let world = world.lock().expect("pre-install production world");
            (
                world
                    .characterization_digest_v0()
                    .expect("pre-install characterization digest"),
                world
                    .world_digest_v1()
                    .expect("pre-install canonical world digest"),
            )
        };
        let mut app = gpui::TestApp::new();
        let windows = app
            .update(|app| session.install(app))
            .expect("install production GPUI test session");
        assert_eq!(
            app.windows().len(),
            2,
            "the production GPUI session must install exactly two windows"
        );
        assert_eq!(
            drain_calls.load(AtomicOrdering::Relaxed),
            1,
            "one production driver task must perform one startup drain"
        );
        assert_eq!(
            step_calls.load(AtomicOrdering::Relaxed),
            0,
            "session installation must not advance scientific time"
        );
        let digest_after_install = {
            let world = world.lock().expect("post-install production world");
            (
                world
                    .characterization_digest_v0()
                    .expect("post-install characterization digest"),
                world
                    .world_digest_v1()
                    .expect("post-install canonical world digest"),
            )
        };
        assert_eq!(
            digest_after_install, digest_before_install,
            "production window installation and initial renders must be science-neutral"
        );

        let hud = app
            .update(|app| windows.hud.root(app))
            .expect("production HUD root");
        let canvas = app
            .update(|app| windows.canvas.root(app))
            .expect("production canvas root");
        app.update_entity(&hud, |view, _| {
            assert!(
                Arc::ptr_eq(&view.simulation_driver, &session.simulation_driver),
                "the production HUD must use the session-level simulation driver"
            );
            assert!(
                !view.minimal_canvas_mode,
                "the production HUD must use its full inspector render branch"
            );
            view.inspector
                .lock()
                .expect("HUD inspector lock")
                .focused_agent = Some(focused_agent);
        });
        app.read_entity(&canvas, |view, _| {
            assert!(
                Arc::ptr_eq(&view.simulation_driver, &session.simulation_driver),
                "the production canvas must use the session-level simulation driver"
            );
            assert!(
                view.minimal_canvas_mode,
                "the production world window must use its minimal canvas render branch"
            );
        });

        let mut digest_trace = Vec::with_capacity(240);
        for expected_tick in 1..=240 {
            assert!(
                (session.command_submit.as_ref())(ControlCommand::UpdateSimulation(
                    SimulationCommand {
                        paused: None,
                        speed_multiplier: None,
                        step_once: true,
                    }
                )),
                "production GPUI step command must be admitted"
            );
            app.advance_clock(Duration::from_secs_f32(SIM_TICK_INTERVAL));
            app.run_until_parked();
            assert_eq!(
                drain_calls.load(AtomicOrdering::Relaxed),
                expected_tick as usize + 1,
                "exactly one production driver task must drain once per timer wake"
            );
            assert_eq!(
                step_calls.load(AtomicOrdering::Relaxed),
                expected_tick as usize,
                "each admitted step-once command must execute exactly one science step"
            );

            let digest_before_repaint = {
                let world = world.lock().expect("production-repaint world");
                assert_eq!(
                    world.tick().0,
                    expected_tick,
                    "one session driver wake must advance exactly one science tick"
                );
                (
                    world
                        .characterization_digest_v0()
                        .expect("characterization digest"),
                    world.world_digest_v1().expect("canonical world digest"),
                )
            };
            if repaint_count > 0 {
                force_production_repaint(&mut app, windows.hud);
                let rendered_tick = app.read_entity(&hud, |view, _| {
                    view.playback.timeline.back().map(|snapshot| snapshot.tick)
                });
                assert_eq!(
                    rendered_tick,
                    Some(expected_tick),
                    "HUD draw must execute the real production Render::render path"
                );
                let digest_after_hud = {
                    let world = world.lock().expect("post-HUD production world");
                    (
                        world
                            .characterization_digest_v0()
                            .expect("post-HUD characterization digest"),
                        world
                            .world_digest_v1()
                            .expect("post-HUD canonical world digest"),
                    )
                };
                assert_eq!(
                    digest_after_hud, digest_before_repaint,
                    "the full production HUD render path must be science-neutral"
                );
            }
            if repaint_count > 1 {
                force_production_repaint(&mut app, windows.canvas);
                let rendered_tick = app.read_entity(&canvas, |view, _| {
                    view.playback.timeline.back().map(|snapshot| snapshot.tick)
                });
                assert_eq!(
                    rendered_tick,
                    Some(expected_tick),
                    "canvas draw must execute the real production Render::render path"
                );
                let digest_after_canvas = {
                    let world = world.lock().expect("post-canvas production world");
                    (
                        world
                            .characterization_digest_v0()
                            .expect("post-canvas characterization digest"),
                        world
                            .world_digest_v1()
                            .expect("post-canvas canonical world digest"),
                    )
                };
                assert_eq!(
                    digest_after_canvas, digest_before_repaint,
                    "the full production canvas render path must be science-neutral"
                );
            }

            digest_trace.push(digest_before_repaint);
        }

        let expected_hud_tick = Some(if repaint_count > 0 { 240 } else { 0 });
        let expected_canvas_tick = Some(if repaint_count > 1 { 240 } else { 0 });
        assert_eq!(
            app.read_entity(&hud, |view, _| {
                view.playback.timeline.back().map(|snapshot| snapshot.tick)
            }),
            expected_hud_tick,
            "HUD playback history must expose the requested repaint schedule"
        );
        assert_eq!(
            app.read_entity(&canvas, |view, _| {
                view.playback.timeline.back().map(|snapshot| snapshot.tick)
            }),
            expected_canvas_tick,
            "canvas playback history must expose the requested repaint schedule"
        );
        assert_eq!(
            drain_calls.load(AtomicOrdering::Relaxed),
            241,
            "one startup drain plus 240 timer wakes proves one session-level driver task"
        );
        assert_eq!(
            step_calls.load(AtomicOrdering::Relaxed),
            240,
            "the production session must execute exactly 240 science ticks"
        );
        app.update(|app| app.shutdown());

        digest_trace
    }

    #[test]
    fn production_gpui_repaint_trace_is_independent_of_window_schedule() {
        let zero_repaints = seeded_digest_trace_after_production_repaints(0);
        let hud_repaints = seeded_digest_trace_after_production_repaints(1);
        let both_repaint = seeded_digest_trace_after_production_repaints(2);

        assert_eq!(
            hud_repaints, zero_repaints,
            "240 real HUD Render::render calls must not alter any science digest"
        );
        assert_eq!(
            both_repaint, zero_repaints,
            "240 real HUD plus canvas Render::render calls must not alter any science digest"
        );
    }

    #[test]
    fn gpui_pending_playback_executes_every_admitted_step_in_one_driver_wake() {
        let world = command_characterization_world();
        let drain = one_shot_command_drain(vec![ControlCommand::Step, ControlCommand::Step]);
        let driver = gui_simulation_driver(&world, drain);
        let now = Instant::now();
        {
            let mut driver = driver.lock().expect("GUI simulation driver lock");
            driver.paused = true;
            driver.last_sim_instant = Some(now);
            driver.drive_at(now);
        }

        assert_eq!(
            world.lock().expect("world lock").tick().0,
            2,
            "every admitted Step command must advance one distinct scientific tick"
        );
        let driver = driver.lock().expect("GUI simulation driver lock");
        assert!(
            driver.snapshot().paused,
            "a Step burst must leave the session paused"
        );
        assert!(
            driver.pending_playback.is_empty(),
            "every successfully completed Step obligation must be consumed"
        );
    }

    #[test]
    fn gpui_pending_playback_preserves_order_without_an_automatic_extra_tick() {
        for (label, commands, expected_paused) in [
            (
                "step then resume",
                vec![ControlCommand::Step, ControlCommand::Resume],
                false,
            ),
            (
                "resume then step",
                vec![ControlCommand::Resume, ControlCommand::Step],
                true,
            ),
        ] {
            let world = command_characterization_world();
            let driver = gui_simulation_driver(&world, one_shot_command_drain(commands));
            let now = Instant::now();
            prime_exactly_one_driver_step(&driver, now);
            driver
                .lock()
                .expect("GUI simulation driver lock")
                .drive_at(now);

            assert_eq!(
                world.lock().expect("world lock").tick().0,
                1,
                "{label} must execute the admitted manual Step without also falling through to \
                 an accumulated wall-clock tick"
            );
            let driver = driver.lock().expect("GUI simulation driver lock");
            assert_eq!(
                driver.snapshot().paused,
                expected_paused,
                "{label} must preserve playback command order"
            );
            assert!(
                driver.pending_playback.is_empty(),
                "{label} must consume every successful playback command"
            );
        }
    }

    #[test]
    fn gpui_pending_playback_retains_first_failed_step_and_blocks_implicit_retry() {
        let world = command_characterization_world();
        let step_calls = Arc::new(AtomicU64::new(0));
        let observed_step_calls = Arc::clone(&step_calls);
        let simulation_step: WorldStepDriver = Arc::new(move || {
            observed_step_calls.fetch_add(1, AtomicOrdering::Relaxed);
            Err(injected_step_error("injected failure before science"))
        });
        let driver = gui_simulation_driver_with_step(
            &world,
            simulation_step,
            one_shot_command_drain(vec![ControlCommand::Step, ControlCommand::Step]),
        );
        let now = Instant::now();
        {
            let mut driver = driver.lock().expect("GUI simulation driver lock");
            driver.paused = true;
            driver.last_sim_instant = Some(now);
            driver.drive_at(now);
            assert_eq!(
                driver.pending_playback.len(),
                2,
                "a pre-transition failure must retain its own Step and every later obligation"
            );
            assert!(driver.simulation_fault.is_some());

            driver.drive_at(now + Duration::from_secs_f32(SIM_TICK_INTERVAL));
            assert_eq!(
                driver.pending_playback.len(),
                2,
                "a terminal driver fault must leave retained playback dormant"
            );
        }

        assert_eq!(world.lock().expect("world lock").tick().0, 0);
        assert_eq!(
            step_calls.load(AtomicOrdering::Relaxed),
            1,
            "a retained failed Step must not retry on the next timer wake"
        );
    }

    #[test]
    fn gpui_pending_playback_retains_failed_second_step_and_later_commands() {
        let world = command_characterization_world();
        let step_world = Arc::clone(&world);
        let step_calls = Arc::new(AtomicU64::new(0));
        let observed_step_calls = Arc::clone(&step_calls);
        let simulation_step: WorldStepDriver = Arc::new(move || {
            if observed_step_calls.fetch_add(1, AtomicOrdering::Relaxed) == 0 {
                step_world
                    .lock()
                    .expect("world lock for successful first step")
                    .step()
            } else {
                Err(injected_step_error(
                    "injected failure before second science tick",
                ))
            }
        });
        let driver = gui_simulation_driver_with_step(
            &world,
            simulation_step,
            one_shot_command_drain(vec![
                ControlCommand::Step,
                ControlCommand::Step,
                ControlCommand::Resume,
            ]),
        );
        let now = Instant::now();
        {
            let mut driver = driver.lock().expect("GUI simulation driver lock");
            driver.paused = true;
            driver.last_sim_instant = Some(now);
            driver.drive_at(now);
            assert_eq!(
                driver.pending_playback.len(),
                2,
                "the failed second Step and trailing Resume must remain ordered and pending"
            );
            assert!(
                driver
                    .pending_playback
                    .front()
                    .is_some_and(|command| command.step_once)
            );
            assert_eq!(
                driver
                    .pending_playback
                    .back()
                    .and_then(|command| command.paused),
                Some(false),
                "the trailing Resume must not overtake the failed Step"
            );

            driver.drive_at(now + Duration::from_secs_f32(SIM_TICK_INTERVAL));
            assert_eq!(driver.pending_playback.len(), 2);
        }

        assert_eq!(world.lock().expect("world lock").tick().0, 1);
        assert_eq!(
            step_calls.load(AtomicOrdering::Relaxed),
            2,
            "the driver must stop at the second failure and must not retry implicitly"
        );
    }

    #[test]
    fn gpui_pending_playback_consumes_step_after_completed_boundary_fault() {
        let world = command_characterization_world();
        let step_world = Arc::clone(&world);
        let step_calls = Arc::new(AtomicU64::new(0));
        let observed_step_calls = Arc::clone(&step_calls);
        let simulation_step: WorldStepDriver = Arc::new(move || {
            observed_step_calls.fetch_add(1, AtomicOrdering::Relaxed);
            step_world
                .lock()
                .expect("world lock for completed-boundary step")
                .step()?;
            Err(injected_step_error(
                "injected admission failure after completed science",
            ))
        });
        let driver = gui_simulation_driver_with_step(
            &world,
            simulation_step,
            one_shot_command_drain(vec![ControlCommand::Step, ControlCommand::Resume]),
        );
        let now = Instant::now();
        {
            let mut driver = driver.lock().expect("GUI simulation driver lock");
            driver.paused = true;
            driver.last_sim_instant = Some(now);
            driver.drive_at(now);

            assert_eq!(
                driver.pending_playback.len(),
                1,
                "a Step that advanced science is consumed even when its callback returns an error"
            );
            let pending = driver
                .pending_playback
                .front()
                .expect("trailing Resume remains pending");
            assert!(!pending.step_once);
            assert_eq!(pending.paused, Some(false));

            driver.drive_at(now + Duration::from_secs_f32(SIM_TICK_INTERVAL));
            assert_eq!(
                driver.pending_playback.len(),
                1,
                "the later Resume must remain dormant behind the terminal fault"
            );
        }

        assert_eq!(world.lock().expect("world lock").tick().0, 1);
        assert_eq!(step_calls.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn gpui_applies_drained_playback_before_stepping() {
        let world = command_characterization_world();
        let drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> = Arc::new(|| {
            vec![ControlCommand::UpdateSimulation(SimulationCommand {
                paused: Some(true),
                speed_multiplier: Some(0.0),
                step_once: false,
            })]
        });
        let driver = gui_simulation_driver(&world, drain);
        let now = Instant::now();
        prime_exactly_one_driver_step(&driver, now);
        driver
            .lock()
            .expect("GUI simulation driver lock")
            .drive_at(now);

        assert_eq!(world.lock().expect("world lock").tick().0, 0);
        let state = driver
            .lock()
            .expect("GUI simulation driver lock")
            .snapshot();
        assert!(state.paused);
        assert_eq!(state.speed_multiplier, 0.0);
    }

    #[test]
    fn gpui_services_world_and_resume_commands_while_paused() {
        let world = command_characterization_world();
        let mut updated = world.lock().expect("world lock").config().clone();
        updated.food_max = 0.73;
        let drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> = Arc::new(move || {
            vec![
                ControlCommand::UpdateConfig(Box::new(updated.clone())),
                ControlCommand::UpdateSimulation(SimulationCommand {
                    paused: Some(false),
                    speed_multiplier: Some(1.0),
                    step_once: false,
                }),
            ]
        });
        let driver = gui_simulation_driver(&world, drain);
        let now = Instant::now();
        {
            let mut driver = driver.lock().expect("GUI simulation driver lock");
            driver.paused = true;
            driver.sim_accumulator = 0.0;
            driver.last_sim_instant = Some(now);
            driver.drive_at(now);
        }

        assert!((world.lock().expect("world lock").config().food_max - 0.73).abs() < f32::EPSILON);
        let state = driver
            .lock()
            .expect("GUI simulation driver lock")
            .snapshot();
        assert!(!state.paused);
        assert_eq!(state.speed_multiplier, 1.0);
    }

    #[test]
    fn simulation_fault_survives_storage_health_refresh() {
        let world = command_characterization_world();
        let drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> = Arc::new(Vec::new);
        let driver = gui_simulation_driver(&world, drain);
        let mut view = simulation_view_with_driver(Arc::clone(&driver));
        driver
            .lock()
            .expect("GUI simulation driver lock")
            .pause_for_simulation_failure("deliberate brain construction failure".to_owned());

        view.maybe_refresh_analytics(0, 0);
        let snapshot = view.snapshot();

        assert_eq!(
            snapshot.simulation_fault.as_deref(),
            Some("deliberate brain construction failure")
        );
        assert!(snapshot.storage.last_error.is_none());
        assert!(!snapshot.storage.stopped);
        assert!(snapshot.controls.paused);
    }

    #[test]
    fn single_lock_toggle_closed_environment_updates_world_state() {
        let world = command_characterization_world();
        let drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> = Arc::new(Vec::new);
        let view = simulation_view(Arc::clone(&world), drain);

        assert!(!world.lock().expect("world lock").is_closed());
        world
            .lock()
            .expect("world lock")
            .set_closed(true)
            .expect("set closed");
        assert!(world.lock().expect("world lock").is_closed());
        world
            .lock()
            .expect("world lock")
            .set_closed(false)
            .expect("set open");
        assert!(!world.lock().expect("world lock").is_closed());
        assert_eq!(view.world.lock().expect("world lock").is_closed(), false);
    }

    #[test]
    fn dual_window_snapshots_share_pause_and_speed_state() {
        let world = command_characterization_world();
        let drain: Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync> = Arc::new(|| {
            vec![ControlCommand::UpdateSimulation(SimulationCommand {
                paused: Some(true),
                speed_multiplier: Some(2.5),
                step_once: false,
            })]
        });
        let driver = gui_simulation_driver(&world, drain);
        let mut hud = simulation_view_with_driver(Arc::clone(&driver));
        let mut canvas = simulation_view_with_driver(Arc::clone(&driver));
        canvas.set_minimal_canvas_mode();

        driver
            .lock()
            .expect("GUI simulation driver lock")
            .drive_at(Instant::now());

        let hud_snapshot = hud.snapshot();
        let canvas_snapshot = canvas.snapshot();
        assert!(hud_snapshot.controls.paused);
        assert!(canvas_snapshot.controls.paused);
        assert_eq!(hud_snapshot.controls.speed_multiplier, 2.5);
        assert_eq!(canvas_snapshot.controls.speed_multiplier, 2.5);
        assert_eq!(world.lock().expect("world lock").tick().0, 0);
    }
}
