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
//!
//! Documented pipeline contract: captures render through the real HDR +
//! tonemapping chain (AcesFitted, fixed exposure — no AutoExposure).
//! The bd-x6v6 investigation proved the capture camera must be spawned
//! BEFORE `App::finish()` (Startup), because camera-dependent plugin
//! initialization for the HDR path only runs for cameras present at finish
//! time; a camera spawned later renders into its internal HDR texture but
//! the out attachment stays initial-fill black. `setup_capture_rig` is
//! therefore a Startup system and the camera/sun are app-lifetime entities
//! (configure_session never respawns them).

use anyhow::{Context, Result, anyhow};
use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::camera::prelude::*;
use bevy::ecs::system::RunSystemOnce;
use bevy::math::primitives::{Capsule3d, Cone, Rectangle, Sphere, Torus};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::renderer::{
    RenderAdapter, RenderAdapterInfo, RenderDevice, RenderInstance, RenderQueue, WgpuWrapper,
};
use bevy::render::settings::{RenderCreation, WgpuSettings, WgpuSettingsPriority};
use bevy::render::{RenderApp, RenderPlugin};
use bevy::window::{ExitCondition, WindowPlugin};
use scriptbots_core::{RenderQuality, RenderSettings, WorldState};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::{
    AccessibilityState, AgentMeshes, AgentRegistry, ReflectionProbeAssets, SnapshotState,
    TerrainChunkRegistry, WorldSnapshot, sync_world,
};

// ---------------------------------------------------------------------------
// Provenance + comparison (pure, GPU-free).
// ---------------------------------------------------------------------------

/// Schema tag for provenance manifests (`scriptbots.capture-provenance.v1`).
pub const PROVENANCE_SCHEMA: &str = "scriptbots.capture-provenance.v1";

const READBACK_MAP_TIMEOUT: Duration = Duration::from_secs(10);
const READBACK_POLL_INTERVAL: Duration = Duration::from_millis(1);

fn wait_for_readback_map<E>(
    receiver: &std::sync::mpsc::Receiver<std::result::Result<(), E>>,
    timeout: Duration,
    mut poll_once: impl FnMut() -> Result<()>,
) -> Result<()>
where
    E: std::fmt::Display,
{
    let started = Instant::now();
    let timeout_error = || anyhow!("readback map timed out after {timeout:?} (wedged GPU?)");
    let mut has_polled = false;
    loop {
        // Permit one immediate progress check even for a zero timeout, then
        // never begin another poll after the deadline.
        if has_polled && started.elapsed() >= timeout {
            return Err(timeout_error());
        }
        has_polled = true;

        // wgpu 26 has no bounded `PollType::Wait`. Polling once is explicitly
        // nonblocking, so the deadline below remains reachable even when the
        // submitted GPU work never completes.
        let poll_result = poll_once();
        match receiver.try_recv() {
            Ok(result) => {
                return result.map_err(|error| anyhow!("readback buffer map failed: {error}"));
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(anyhow!(
                    "readback map callback channel disconnected before completion"
                ));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => poll_result?,
        }

        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Err(timeout_error());
        }
        std::thread::sleep(READBACK_POLL_INTERVAL.min(remaining));
    }
}

const fn readback_poll_type() -> wgpu::PollType {
    wgpu::PollType::Poll
}

fn poll_readback_device(device: &RenderDevice) -> Result<()> {
    device
        .poll(readback_poll_type())
        .map(|_| ())
        .map_err(|error| anyhow!("device poll during readback: {error}"))
}

/// Full provenance for one captured frame: everything needed to decide
/// whether a golden comparison is same-class and to reproduce the frame.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CaptureProvenance {
    /// Schema tag (`scriptbots.capture-provenance.v1`).
    pub schema: String,
    /// Scientific identity of the world state this frame depicts.
    ///
    /// `None` only when the digest could not be computed; it never means two
    /// worlds agreed. A capture that cannot say which world it shows is not
    /// evidence of anything (bd-2z0.14.3.4).
    #[serde(default)]
    pub world_digest: Option<String>,
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
    /// Commit the source tree was on when this binary's build script ran,
    /// suffixed `-dirty` if that tree had uncommitted changes (bd-2z0.14.3.10).
    ///
    /// This is what lets a golden mismatch distinguish "the renderer changed"
    /// from "a different build produced this". World identity alone cannot:
    /// two trees can depict the same world and render it differently.
    ///
    /// Read it as an identifier, not a reproducibility proof. Cargo re-runs the
    /// build script when `.git/HEAD` moves, not on every source edit, so a
    /// value without `-dirty` means the tree was clean when the script last
    /// ran — not that it is clean now. `-dirty` appearing is therefore
    /// conclusive; its absence is merely the best the build can honestly claim.
    ///
    /// `None` when git is unavailable or this is not a repository (release
    /// tarballs, vendored builds).
    #[serde(default)]
    pub source_commit: Option<String>,
    /// FNV-1a-64 of the workspace `Cargo.lock` at build time (bd-2z0.14.3.10).
    ///
    /// Equal digests mean both artifacts linked the same resolved dependency
    /// graph, which `source_commit` does not establish on its own: the lock can
    /// move without a commit, and a commit can be rebuilt against a changed
    /// lock. Non-cryptographic — it answers "same file", not "untampered".
    ///
    /// `None` when the lock file could not be read.
    #[serde(default)]
    pub lock_digest: Option<String>,
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
    /// Minimum allowed mean SSIM over all luma windows.
    pub min_ssim_mean: f32,
    /// Minimum allowed SSIM for the single worst luma window.
    pub min_ssim_window: f32,
}

impl Default for CompareThresholds {
    fn default() -> Self {
        Self {
            differing_channel: 8,
            max_differing_ratio: 0.005,
            max_mean_abs_diff: 0.5,
            // Structural gates are ADDITIVE: they can only fail a comparison the
            // per-channel gate would have passed, never rescue one it failed. A
            // perceptual score that could override a channel breach would be a
            // gate that talks itself out of a real regression, which is the
            // failure mode this whole harness exists to avoid.
            //
            // These defaults are deliberately loose because they are uncalibrated:
            // no GPU lane was available to measure real adapter jitter (bd-e6ff,
            // bd-oite). 0.98 mean rejects global structural change; 0.5 on a single
            // window rejects severe localized damage — a dead mesh, a material that
            // stopped shading, a bloom pass gone missing in one region — while
            // sitting far above anything raster jitter produces. Tighten once a
            // real-GPU lane can measure the true noise floor.
            min_ssim_mean: 0.98,
            min_ssim_window: 0.5,
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
    /// Mean SSIM over all 8x8 luma windows, in `[-1, 1]` (1.0 = identical).
    ///
    /// Structural, not per-channel: it answers "does this look like the same
    /// image" where [`mean_abs_diff`](Self::mean_abs_diff) answers "how far
    /// apart are the numbers" (bd-2z0.14.3.10 item 3).
    pub ssim_mean: f64,
    /// SSIM of the WORST 8x8 luma window.
    ///
    /// The mean is dominated by the many untouched windows, so localized damage
    /// — one dead mesh, a broken shader on one material, a bloom pass that
    /// stopped contributing in one region — barely moves it. The worst window
    /// is what actually notices, which is why the gate uses both.
    pub ssim_min: f64,
    /// Whether the comparison passed the thresholds.
    pub pass: bool,
}

/// Whether a lane may certify PIXELS, as opposed to merely proving the capture
/// path compiles and runs.
///
/// This is the bd-2z0.14.3.10 item 4 decision made explicit and given one home,
/// instead of an `eprintln!` inside a test helper that produced no record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PixelEvidenceVerdict {
    /// The adapter may produce pixel evidence: goldens rendered here mean something.
    Admissible,
    /// The lane may run as a smoke check but must never certify pixels. The
    /// payload is actionable text naming what was found and what to do.
    SmokeOnly(String),
}

impl PixelEvidenceVerdict {
    /// Whether pixel evidence from this lane can be trusted.
    #[must_use]
    pub const fn admissible(&self) -> bool {
        matches!(self, Self::Admissible)
    }
}

/// Decide whether an adapter may certify pixels (bd-2z0.14.3.10 item 4).
///
/// THE DECISION, and the evidence behind it. bd-2z0.14.3.4 scope item 4 made the
/// llvmpipe/lavapipe software lane the PR gate and asserted the unmodified
/// pipeline would pass on it. It does not. Measured on the Linux rch workers,
/// both live capture tests failed with the typed error "offscreen capture
/// produced a blank or uniform RGB frame".
///
/// The failure is NOT an obvious capability wall, which is what makes the
/// verdict "smoke only" rather than "abandon the lane". llvmpipe probes with
/// full capability — Vulkan backend, `max_texture_2d` 16384, timestamp queries
/// available — and `OffscreenCapture` configures a session on it successfully at
/// tier Potato. Only the READBACK comes back uniform. That is the signature of
/// bd-x6v6, which also looked like a hardware limit and turned out to be a
/// camera-lifecycle bug, so the lane is worth one real investigation before it
/// is written off (the capture session takes exactly two warmup frames before
/// reading back, an assumption never validated against a rasterizer this slow).
///
/// Until that investigation succeeds, a software adapter may run the capture
/// path as a smoke check and may NOT certify pixels. Gating goldens on a lane
/// that returns uniform frames would either fail every PR or, if the blank guard
/// were relaxed to accommodate it, certify blank images as evidence — which is
/// precisely the bd-2z0.14.3.8 failure this harness exists to prevent.
///
/// Scope note: `Virtual` and `Unknown` adapters stay admissible, preserving the
/// behaviour this replaced rather than quietly widening the refusal. Neither has
/// been measured returning blank frames. Whether an unclassifiable adapter should
/// be trusted to certify pixels is a real question, but it is a different one
/// than this bead asked, and answering it by side effect would be the same
/// overreach the decision above declines to make about llvmpipe.
#[must_use]
pub fn pixel_evidence_verdict(gpu: Option<&scriptbots_core::GpuInfo>) -> PixelEvidenceVerdict {
    match gpu {
        None => PixelEvidenceVerdict::SmokeOnly(
            "no GPU adapter was found, so nothing rendered; a capture from this lane \
             proves the code path runs, never what it looks like (bd-2z0.14.3.10)"
                .to_string(),
        ),
        Some(info) if info.class == scriptbots_core::GpuClass::Software => {
            PixelEvidenceVerdict::SmokeOnly(format!(
                "software rasterizer ({}) returns blank or uniform frames from this \
                 capture path, so it may run as a smoke check but must not certify \
                 pixels. It probes fully capable and configures a session, so the \
                 blank readback is a candidate camera-lifecycle or warmup-frame bug \
                 (bd-x6v6 shape), not a proven hardware limit — investigate before \
                 demoting the lane permanently (bd-2z0.14.3.10 item 4)",
                info.name
            ))
        }
        Some(_) => PixelEvidenceVerdict::Admissible,
    }
}

/// Explain a blank capture in terms of what the lane could ever have proven
/// (bd-2z0.14.3.3 item 4).
///
/// The blank guard used to report only its symptom — "produced a blank or
/// uniform RGB frame" — which reads identically in two situations that call for
/// opposite responses:
///
/// * a software rasterizer, where a uniform frame is the KNOWN behaviour of a
///   lane that was never able to certify pixels. Nothing is broken; the run
///   asked the wrong lane for evidence.
/// * a discrete or integrated GPU, where a uniform frame means the pipeline
///   genuinely failed. Something IS broken, and reading it as a lane limitation
///   would file a real rendering bug as an environment quirk.
///
/// Collapsing those into one message is how a harness trains its readers to
/// ignore it. The verdict is sourced from [`pixel_evidence_verdict`] rather than
/// re-deciding here, so there is exactly one place that knows which adapters can
/// certify pixels.
fn blank_frame_error(gpu: Option<&scriptbots_core::GpuInfo>) -> anyhow::Error {
    match pixel_evidence_verdict(gpu) {
        PixelEvidenceVerdict::SmokeOnly(reason) => anyhow!(
            "offscreen capture produced a blank or uniform RGB frame, and this lane \
             could not have certified pixels in the first place: {reason}"
        ),
        PixelEvidenceVerdict::Admissible => anyhow!(
            "offscreen capture produced a blank or uniform RGB frame on {}, an adapter \
             that IS admissible for pixel evidence — so this is a real pipeline failure \
             (camera, lighting, extraction or readback), not a lane limitation, and it \
             must not be dismissed as one",
            gpu.map_or("an unknown adapter", |info| info.name.as_str())
        ),
    }
}

/// Side length of the SSIM window, in pixels.
const SSIM_WINDOW: usize = 8;

/// Rec. 709 luma of an RGBA8 pixel, matching the luminance basis the rest of
/// the harness uses.
fn luma8(pixel: &[u8; 4]) -> f64 {
    0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
}

/// Mean and worst-window SSIM between two same-shape RGBA8 frames.
///
/// Standard single-scale SSIM over non-overlapping `SSIM_WINDOW` blocks of the
/// luma channel, with the usual C1/C2 stabilizers so flat identical regions
/// score exactly 1.0 instead of dividing by zero. Non-overlapping blocks (not a
/// sliding Gaussian) keep this O(pixels) and fully deterministic, which matters
/// because it runs inside a golden gate.
///
/// Edge blocks are skipped when the frame is not a multiple of the window; a
/// partial block has too few samples for a meaningful variance and would add
/// noise to the very statistic being gated on.
fn ssim_luma(golden: &[u8], candidate: &[u8], width: u32, height: u32) -> (f64, f64) {
    // 8-bit dynamic range, the standard SSIM constants.
    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);

    let width = width as usize;
    let height = height as usize;
    let g = golden.as_chunks::<4>().0;
    let c = candidate.as_chunks::<4>().0;

    let mut sum = 0.0_f64;
    let mut windows = 0_u64;
    let mut worst = f64::INFINITY;

    let blocks_y = height / SSIM_WINDOW;
    let blocks_x = width / SSIM_WINDOW;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let (mut gs, mut cs) = (0.0_f64, 0.0_f64);
            let (mut gss, mut css, mut gcs) = (0.0_f64, 0.0_f64, 0.0_f64);
            for row in 0..SSIM_WINDOW {
                let base = (by * SSIM_WINDOW + row) * width + bx * SSIM_WINDOW;
                for col in 0..SSIM_WINDOW {
                    let gl = luma8(&g[base + col]);
                    let cl = luma8(&c[base + col]);
                    gs += gl;
                    cs += cl;
                    gss += gl * gl;
                    css += cl * cl;
                    gcs += gl * cl;
                }
            }
            let n = (SSIM_WINDOW * SSIM_WINDOW) as f64;
            let g_mean = gs / n;
            let c_mean = cs / n;
            // Population variance/covariance: the window IS the population.
            let g_var = (gss / n) - g_mean * g_mean;
            let c_var = (css / n) - c_mean * c_mean;
            let cov = (gcs / n) - g_mean * c_mean;

            let numerator = (2.0 * g_mean * c_mean + C1) * (2.0 * cov + C2);
            let denominator = (g_mean * g_mean + c_mean * c_mean + C1) * (g_var + c_var + C2);
            let score = if denominator == 0.0 {
                1.0
            } else {
                numerator / denominator
            };
            sum += score;
            worst = worst.min(score);
            windows += 1;
        }
    }

    if windows == 0 {
        // Frame smaller than one window: no structural claim is possible, so
        // report the neutral value and let the per-channel gate decide alone.
        return (1.0, 1.0);
    }
    (sum / windows as f64, worst)
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
    // Count PIXELS, not channels. The previous form counted differing channels
    // and divided by four, which is only equivalent when every differing pixel
    // differs in all four channels. A single-channel regression — a red shift,
    // a dead blue term, an alpha-only change — was therefore reported at a
    // QUARTER of its true extent, making it four times less likely to breach
    // max_differing_ratio. A golden comparison that under-reports difference is
    // an evidence check weaker than it claims to be (bd-2z0.14.3.4 round-1
    // audit).
    let mut differing_pixels = 0_u64;
    let mut max_channel_diff = 0_u8;
    let mut abs_sum = 0_u64;
    for (g_px, c_px) in golden
        .as_chunks::<4>()
        .0
        .iter()
        .zip(candidate.as_chunks::<4>().0.iter())
    {
        let mut pixel_differs = false;
        for channel in 0..4 {
            let diff = g_px[channel].abs_diff(c_px[channel]);
            max_channel_diff = max_channel_diff.max(diff);
            abs_sum += u64::from(diff);
            pixel_differs |= diff > thresholds.differing_channel;
        }
        if pixel_differs {
            differing_pixels += 1;
        }
    }
    let total_pixels = u64::from(width) * u64::from(height);
    let differing_ratio = differing_pixels as f64 / total_pixels as f64;
    let mean_abs_diff = abs_sum as f64 / (total_pixels * 4) as f64;
    // Structural agreement, which the per-channel statistics above cannot see.
    // A localized regression can hold differing_ratio under its threshold and
    // mean_abs_diff near zero purely by being small, while looking obviously
    // wrong (bd-2z0.14.3.10 item 3).
    let (ssim_mean, ssim_min) = ssim_luma(golden, candidate, width, height);
    let pass = differing_ratio <= f64::from(thresholds.max_differing_ratio)
        && mean_abs_diff <= f64::from(thresholds.max_mean_abs_diff)
        && ssim_mean >= f64::from(thresholds.min_ssim_mean)
        && ssim_min >= f64::from(thresholds.min_ssim_window);
    Ok(DiffStats {
        differing_pixels,
        total_pixels,
        differing_ratio,
        max_channel_diff,
        mean_abs_diff,
        ssim_mean,
        ssim_min,
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
    for px in golden
        .as_chunks::<4>()
        .0
        .iter()
        .zip(candidate.as_chunks::<4>().0.iter())
    {
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

/// Minimum summed per-channel RGB range for a frame to count as evidence.
///
/// Chosen to sit far below any real render and far above sampling noise
/// (bd-2z0.14.3.8). The live alarm test asserts a spread above 24 on an honest
/// frame, so a production floor of 8 rejects the degenerate cases without
/// second-guessing a legitimately dark scene that the alarm would still accept.
const MIN_EVIDENCE_RGB_SPREAD: u32 = 8;

/// Return true when an RGBA8 buffer contains no visible RGB evidence.
///
/// Alpha is deliberately ignored: the render target's initial fill is opaque
/// black, so counting its `255` alpha byte as content would turn a dead
/// pipeline into a successful capture.
///
/// A uniform clear colour is not evidence either — but neither is a frame that
/// differs from uniform by a single unit. The original form of this guard
/// asked only whether ANY pixel differed from the first in ANY channel by ANY
/// amount, and bd-2z0.14.3.8 caught it accepting a 160x120 frame whose total
/// per-channel range was 1. That is blank to a human and blank for every
/// evidentiary purpose, yet it was returned as a legitimate `CapturedFrame`
/// and would have been admitted to the golden-image lane. Only the live alarm
/// test, with a stricter threshold than production, noticed.
///
/// So the bar is now a measured range rather than mere inequality: the summed
/// per-channel spread must reach [`MIN_EVIDENCE_RGB_SPREAD`].
#[must_use]
pub fn rgba8_is_visually_blank(bytes: &[u8]) -> bool {
    let pixels = bytes.as_chunks::<4>().0;
    if pixels.is_empty() {
        return true;
    }
    let mut min = [u8::MAX; 3];
    let mut max = [u8::MIN; 3];
    let mut has_nonzero_rgb = false;
    for pixel in pixels {
        for channel in 0..3 {
            let value = pixel[channel];
            has_nonzero_rgb |= value != 0;
            min[channel] = min[channel].min(value);
            max[channel] = max[channel].max(value);
        }
    }
    let spread: u32 = (0..3).map(|c| u32::from(max[c] - min[c])).sum();
    !has_nonzero_rgb || spread < MIN_EVIDENCE_RGB_SPREAD
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

/// Failures that can occur before the Bevy render app owns a device.
///
/// Bevy's automatic renderer initialization uses `expect` for adapter
/// selection and `unwrap` for device creation. The release profile aborts on
/// panic, so the capture harness creates these resources itself and reports
/// every initialization failure before handing them to [`RenderPlugin`].
#[derive(Debug)]
pub enum CaptureRendererInitError {
    /// No adapter satisfies the selected backend and fallback policy.
    AdapterUnavailable,
    /// `WGPU_ADAPTER_NAME` selected an adapter that is not present.
    RequestedAdapterUnavailable {
        /// Requested adapter name.
        requested: String,
        /// Backends enabled for this process.
        backends: wgpu::Backends,
    },
    /// Bevy requested features the selected adapter does not expose.
    UnsupportedFeatures {
        /// Actual adapter name.
        adapter: String,
        /// Required features absent from the adapter.
        missing: wgpu::Features,
    },
    /// Bevy's requested limits exceed the selected adapter's limits.
    UnsupportedLimits {
        /// Actual adapter name.
        adapter: String,
        /// Human-readable list of violated limits.
        violations: String,
    },
    /// Bevy changed its defaults to use a constrained-limit policy that this
    /// manually mirrored initializer does not yet implement.
    UnsupportedConfiguration(&'static str),
    /// The adapter was selected, but creating its logical device failed.
    DeviceRequest {
        /// Actual adapter name.
        adapter: String,
        /// Actual adapter backend.
        backend: wgpu::Backend,
        /// Original wgpu failure.
        source: wgpu::RequestDeviceError,
    },
}

impl std::fmt::Display for CaptureRendererInitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AdapterUnavailable => formatter.write_str(
                "no GPU adapter available for offscreen capture \
                 (software lane requires llvmpipe/lavapipe via WGPU_BACKEND, or a real GPU)",
            ),
            Self::RequestedAdapterUnavailable {
                requested,
                backends,
            } => write!(
                formatter,
                "requested GPU adapter {requested:?} is unavailable for backends {backends:?}"
            ),
            Self::UnsupportedFeatures { adapter, missing } => write!(
                formatter,
                "GPU adapter {adapter:?} lacks required Bevy features {missing:?}"
            ),
            Self::UnsupportedLimits {
                adapter,
                violations,
            } => write!(
                formatter,
                "GPU adapter {adapter:?} cannot satisfy Bevy limits: {violations}"
            ),
            Self::UnsupportedConfiguration(detail) => {
                write!(
                    formatter,
                    "unsupported Bevy renderer configuration: {detail}"
                )
            }
            Self::DeviceRequest {
                adapter,
                backend,
                source,
            } => write!(
                formatter,
                "GPU device creation failed for adapter {adapter:?} on {backend:?}: {source}"
            ),
        }
    }
}

impl std::error::Error for CaptureRendererInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DeviceRequest { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// The offscreen image target handle shared with the scene setup.
#[derive(Resource, Clone)]
struct CaptureTarget(Handle<Image>);

/// Marker for the capture camera.
#[derive(Component)]
struct CaptureCamera;

/// Corruption flag resource (alarm tests).
#[derive(Resource, Clone, Copy)]
struct CaptureCorrupt(
    // bd-tqpj: inserted at app build for the alarm-test corruption path; reading system is in-flight capture-harness work (minimal touch on active file).
    #[allow(dead_code)] bool,
);

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
    adapter_name: String,
    backend: String,
    device_type: String,
    corrupt: bool,
    next_snapshot_revision: u64,
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
    ///
    /// `world_digest` is the scientific identity of the state that produced the
    /// frame. Without it an artifact records how it was rendered but not WHAT
    /// it depicts, so two captures of different worlds are indistinguishable
    /// once the process exits — which is exactly the question a golden
    /// mismatch asks (bd-2z0.14.3.4 round-1 audit: provenance omits world
    /// identity). `None` means the digest was unavailable, never that the
    /// worlds matched.
    #[must_use]
    pub fn provenance(
        &self,
        scene: &str,
        seed: u64,
        tick: u64,
        world_digest: Option<String>,
    ) -> CaptureProvenance {
        CaptureProvenance {
            schema: PROVENANCE_SCHEMA.to_string(),
            scene: scene.to_string(),
            seed,
            tick,
            world_digest,
            frontend: "bevy_offscreen".to_string(),
            adapter_name: self.adapter_name.clone(),
            backend: self.backend.clone(),
            device_type: self.device_type.clone(),
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
            source_commit: option_env!("SCRIPTBOTS_SOURCE_COMMIT").map(str::to_string),
            lock_digest: option_env!("SCRIPTBOTS_LOCK_DIGEST").map(str::to_string),
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
        let mut snapshot = WorldSnapshot::from_world(world)
            .ok_or_else(|| anyhow!("world snapshot unavailable at tick {tick}"))?;
        validate_capture_tick(snapshot.tick, tick)?;
        validate_capture_seed(world.agent_substream_protocol_v1().root_seed(), seed)?;
        // Scientific identity of the state being depicted, recorded alongside
        // how it was rendered (bd-2z0.14.3.4). Taken BEFORE rendering, so it
        // describes the world the frame actually shows. A digest failure is
        // reported as absent rather than silently omitted or faked — capture
        // itself is read-only over the world and must not fail on it.
        let world_digest = match world.world_digest_v1() {
            Ok(digest) => Some(digest.overall),
            Err(error) => {
                warn!(%error, "world digest unavailable for capture provenance");
                None
            }
        };
        snapshot.revision = self.next_snapshot_revision;
        self.next_snapshot_revision = self
            .next_snapshot_revision
            .checked_add(1)
            .ok_or_else(|| anyhow!("offscreen capture snapshot revision overflow"))?;
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
        let data = self.readback_target();
        self.set_camera_active(false);
        let data = data?;
        let (width, height) = self.viewport;
        let rgba8 = unpad_readback(&data, width, height);
        let expected = width as usize * height as usize * 4;
        if rgba8.len() != expected {
            return Err(anyhow!(
                "readback shape drift: expected {expected} bytes after unpad, got {}",
                rgba8.len()
            ));
        }
        if rgba8_is_visually_blank(&rgba8) {
            return Err(blank_frame_error(self.effective.gpu.as_ref()));
        }
        Ok(CapturedFrame {
            width,
            height,
            rgba8,
            provenance: self.provenance(scene, seed, tick, world_digest),
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
        wait_for_readback_map(&rx, READBACK_MAP_TIMEOUT, || poll_readback_device(&device))?;
        let data = {
            let mapped = buffer.slice(..).get_mapped_range();
            Vec::from(&*mapped)
        };
        buffer.unmap();
        debug!(bytes = data.len(), "offscreen readback complete");
        Ok(data)
    }
}

fn validate_capture_tick(snapshot_tick: u64, requested_tick: u64) -> Result<()> {
    if snapshot_tick != requested_tick {
        return Err(anyhow!(
            "capture tick attribution mismatch: snapshot is tick {snapshot_tick}, \
             caller requested tick {requested_tick}"
        ));
    }
    Ok(())
}

fn validate_capture_seed(world_seed: u64, requested_seed: u64) -> Result<()> {
    if world_seed != requested_seed {
        return Err(anyhow!(
            "capture seed attribution mismatch: world root is {world_seed}, \
             caller requested seed {requested_seed}"
        ));
    }
    Ok(())
}

/// Build the process-wide render app (once). No window, no winit (macOS
/// demands EventLoop creation on the main thread; captures run on
/// test/worker threads), fixed exposure (no AutoExposure) for determinism.
///
/// The capture camera is spawned in Startup — BEFORE finish/cleanup —
/// because camera-dependent plugin initialization (the HDR/tonemapping
/// path) only runs for cameras that exist at finish time; a camera spawned
/// later renders into its internal HDR texture but the out attachment
/// stays initial-fill black (the bd-x6v6 root cause, isolated by the
/// minimal upstream canary in `tests/hdr_probe.rs`).
fn build_capture_app(config: &OffscreenCaptureConfig) -> Result<App> {
    let render_creation = initialize_capture_renderer()?;
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
            .set(RenderPlugin {
                render_creation,
                ..default()
            })
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                close_when_requested: false,
                ..Default::default()
            }),
    )
    .add_systems(Update, sync_world)
    .add_systems(
        Startup,
        (setup_capture_rig, setup_capture_resources).chain(),
    );
    // First session's render target: sRGB RGBA8, render attachment +
    // copy-src for the readback path. Later sessions create fresh targets
    // in configure_session and repoint this same camera.
    let target = add_target_image(&mut app, config.viewport);
    app.insert_resource(CaptureTarget(target));
    // finish/cleanup are required before manual update() pumping: the
    // render device initializes in RenderPlugin::finish. Pipelined rendering
    // is disabled above for exactly this reason: with the render app on its
    // own thread, `get_sub_app_mut(RenderApp)` returns None and the explicit
    // wgpu device poll the readback path needs never happens — the capture
    // wedges at "readback did not complete in budget" on Metal.
    app.finish();
    app.cleanup();
    // Warmup: startup + two frames so base pipelines compile outside the
    // evidence path.
    app.update();
    app.update();
    info!(
        viewport = ?config.viewport,
        "process-wide offscreen capture app built"
    );
    Ok(app)
}

/// Create the exact wgpu resources that Bevy will render with.
///
/// This mirrors Bevy 0.17's automatic `initialize_renderer` policy for its
/// default settings, but replaces the adapter `expect` and device `unwrap`
/// with typed failures. It also makes `WGPU_ADAPTER_NAME` authoritative:
/// asking for an unavailable adapter fails instead of silently selecting a
/// different backend and mislabelling the resulting evidence.
fn initialize_capture_renderer() -> std::result::Result<RenderCreation, CaptureRendererInitError> {
    let settings = WgpuSettings::default();
    let backends = settings
        .backends
        .ok_or(CaptureRendererInitError::UnsupportedConfiguration(
            "WgpuSettings.backends must be explicit for offscreen capture",
        ))?;
    if settings.constrained_limits.is_some() {
        return Err(CaptureRendererInitError::UnsupportedConfiguration(
            "constrained_limits require updating the manual Bevy initializer",
        ));
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        flags: settings.instance_flags,
        memory_budget_thresholds: settings.instance_memory_budget_thresholds,
        backend_options: wgpu::BackendOptions {
            gl: wgpu::GlBackendOptions {
                gles_minor_version: settings.gles3_minor_version,
                fence_behavior: wgpu::GlFenceBehavior::Normal,
            },
            dx12: wgpu::Dx12BackendOptions {
                shader_compiler: settings.dx12_shader_compiler.clone(),
            },
            noop: wgpu::NoopBackendOptions { enable: false },
        },
    });

    let force_fallback_adapter = force_fallback_adapter(
        settings.force_fallback_adapter,
        std::env::var("WGPU_FORCE_FALLBACK_ADAPTER").ok().as_deref(),
    );
    let requested_name = std::env::var("WGPU_ADAPTER_NAME")
        .ok()
        .filter(|name| !name.trim().is_empty())
        .or(settings.adapter_name.clone());
    let request_options = wgpu::RequestAdapterOptions {
        power_preference: settings.power_preference,
        compatible_surface: None,
        force_fallback_adapter,
    };
    let adapter = if let Some(requested) = requested_name {
        instance
            .enumerate_adapters(backends)
            .into_iter()
            .find(|adapter| adapter.get_info().name.eq_ignore_ascii_case(&requested))
            .ok_or(CaptureRendererInitError::RequestedAdapterUnavailable {
                requested,
                backends,
            })?
    } else {
        bevy::tasks::block_on(instance.request_adapter(&request_options))
            .map_err(|_| CaptureRendererInitError::AdapterUnavailable)?
    };
    let adapter_info = adapter.get_info();
    let adapter_features = adapter.features();
    let adapter_limits = adapter.limits();

    let mut features = wgpu::Features::empty();
    let mut limits = settings.limits.clone();
    if matches!(settings.priority, WgpuSettingsPriority::Functionality) {
        features = adapter_features;
        if adapter_info.device_type == wgpu::DeviceType::DiscreteGpu {
            features.remove(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        }
        limits = adapter_limits.clone();
    }
    if let Some(disabled_features) = settings.disabled_features {
        features.remove(disabled_features);
    }
    features |= settings.features;

    if !adapter_features.contains(features) {
        return Err(CaptureRendererInitError::UnsupportedFeatures {
            adapter: adapter_info.name,
            missing: features.difference(adapter_features),
        });
    }
    if !limits.check_limits(&adapter_limits) {
        let mut violations = Vec::new();
        limits.check_limits_with_fail_fn(&adapter_limits, false, |name, requested, allowed| {
            violations.push(format!("{name} requested={requested} available={allowed}"));
        });
        return Err(CaptureRendererInitError::UnsupportedLimits {
            adapter: adapter_info.name,
            violations: violations.join(", "),
        });
    }

    let descriptor = wgpu::DeviceDescriptor {
        label: settings.device_label.as_deref(),
        required_features: features,
        required_limits: limits,
        memory_hints: settings.memory_hints,
        trace: wgpu::Trace::Off,
    };
    let (device, queue) =
        bevy::tasks::block_on(adapter.request_device(&descriptor)).map_err(|source| {
            CaptureRendererInitError::DeviceRequest {
                adapter: adapter_info.name.clone(),
                backend: adapter_info.backend,
                source,
            }
        })?;
    debug!(
        adapter = %adapter_info.name,
        backend = ?adapter_info.backend,
        device_type = ?adapter_info.device_type,
        force_fallback_adapter,
        "initialized fallible Bevy offscreen renderer"
    );

    Ok(RenderCreation::manual(
        RenderDevice::from(device),
        RenderQueue(Arc::new(WgpuWrapper::new(queue))),
        RenderAdapterInfo(WgpuWrapper::new(adapter_info)),
        RenderAdapter(Arc::new(WgpuWrapper::new(adapter))),
        RenderInstance(Arc::new(WgpuWrapper::new(instance))),
    ))
}

fn force_fallback_adapter(default: bool, override_value: Option<&str>) -> bool {
    override_value.map_or(default, |value| {
        !(value.is_empty() || value == "0" || value == "false")
    })
}

/// Create the offscreen render target image (sRGB RGBA8, render attachment
/// + copy-src) and add it to the image assets.
fn add_target_image(app: &mut App, viewport: (u32, u32)) -> Handle<Image> {
    let size = Extent3d {
        width: viewport.0,
        height: viewport.1,
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
    app.world_mut().resource_mut::<Assets<Image>>().add(image)
}

/// Reset the process app for a new scene: wipe scene CONTENT (meshes,
/// readbacks, probes — never the capture camera or sun, which were spawned
/// pre-finish and keep their plugin initialization), reset the registries,
/// re-resolve the effective tier, create a fresh render target, repoint
/// the camera, and retune the lighting for tier/corrupt mode.
fn configure_session<'a>(
    app: &'a mut App,
    config: &OffscreenCaptureConfig,
) -> Result<OffscreenCapture<'a>> {
    let (actual_gpu, actual_device_type) = {
        let render_app = app
            .get_sub_app_mut(RenderApp)
            .ok_or_else(|| anyhow!("capture render sub-app unavailable"))?;
        let world = render_app.world();
        let info = world
            .get_resource::<RenderAdapterInfo>()
            .ok_or_else(|| anyhow!("actual Bevy render adapter identity unavailable"))?;
        let device = world
            .get_resource::<RenderDevice>()
            .ok_or_else(|| anyhow!("actual Bevy render device unavailable"))?;
        (
            scriptbots_core::GpuInfo {
                name: info.name.clone(),
                backend: format!("{:?}", info.backend),
                class: crate::gpu_class_from_device_type(info.device_type),
                vram_bytes: None,
                max_texture_2d: Some(device.limits().max_texture_dimension_2d),
                timestamp_queries: device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            },
            format!("{:?}", info.device_type),
        )
    };
    let effective =
        crate::resolve_effective_render_settings_for_gpu(&config.render_settings, Some(actual_gpu));
    // Wipe scene content only. A blanket wipe of every entity kills
    // plugin-owned startup entities the render path depends on (the wiped
    // app then produces no ViewTargets and every capture reads back as the
    // initial fill); the camera and sun are app-lifetime entities.
    let scene_entities: Vec<Entity> = {
        let world = app.world_mut();
        let mut found = Vec::new();
        let mut meshes = world.query_filtered::<Entity, With<Mesh3d>>();
        found.extend(meshes.iter(world));
        let mut probes = world.query_filtered::<Entity, Or<(
            With<bevy::light::LightProbe>,
            With<bevy::light::EnvironmentMapLight>,
        )>>();
        found.extend(probes.iter(world));
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

    // Fresh render target; repoint the app-lifetime capture camera and
    // retune the sun for tier/corrupt without respawning either.
    let target = add_target_image(app, config.viewport);
    app.insert_resource(CaptureTarget(target.clone()));
    {
        let world = app.world_mut();
        let mut cameras = world.query_filtered::<&mut Camera, With<CaptureCamera>>();
        for mut camera in cameras.iter_mut(world) {
            camera.target = RenderTarget::Image(target.clone().into());
            camera.clear_color = ClearColorConfig::Custom(if config.corrupt {
                Color::srgb(0.9, 0.0, 0.9)
            } else {
                Color::srgb(0.03, 0.05, 0.09)
            });
        }
        let mut lights = world.query::<&mut DirectionalLight>();
        for mut light in lights.iter_mut(world) {
            light.illuminance = if config.corrupt { 0.0 } else { 9000.0 };
            light.shadows_enabled = effective.features.shadows;
        }
    }
    // Refresh the mesh/registry/probe resources for the new scene.
    app.world_mut()
        .run_system_once(setup_capture_resources)
        .map_err(|error| anyhow!("capture resource setup: {error}"))?;
    // Warmup: two frames with an empty scene so fresh-target state settles
    // outside the evidence path.
    app.update();
    app.update();
    let actual_gpu = effective
        .gpu
        .as_ref()
        .ok_or_else(|| anyhow!("resolved Bevy render adapter identity unavailable"))?;
    let adapter_name = actual_gpu.name.clone();
    let backend = actual_gpu.backend.clone();
    info!(
        viewport = ?config.viewport,
        tier = ?effective.tier,
        adapter = %adapter_name,
        backend = %backend,
        corrupt = config.corrupt,
        "offscreen capture session configured"
    );
    Ok(OffscreenCapture {
        app,
        target,
        viewport: config.viewport,
        effective,
        adapter_name,
        backend,
        device_type: actual_device_type,
        corrupt: config.corrupt,
        next_snapshot_revision: 1,
    })
}

/// The app-lifetime rig: HDR capture camera + sun. Spawned once in Startup
/// (see [`build_capture_app`] for the pre-finish requirement). The camera
/// starts inactive; sessions toggle it around readbacks.
fn setup_capture_rig(mut commands: Commands, target: Res<CaptureTarget>) {
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
        bevy::core_pipeline::tonemapping::Tonemapping::AcesFitted,
        bevy::render::view::ColorGrading::default(),
        bevy::render::view::Hdr,
        CaptureCamera,
    ));

    let light_transform =
        Transform::from_xyz(-1200.0, 1800.0, 900.0).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((
        DirectionalLight {
            illuminance: 9000.0,
            shadows_enabled: true,
            ..default()
        },
        light_transform,
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
    ));
}

/// Per-session resources: creature/agent meshes, terrain chunk registry,
/// and valid (non-empty) reflection probe textures — empty `Handle`s panic
/// wgpu's cube/D2 texture validation on some backends.
fn setup_capture_resources(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
) {
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
    // Valid 1x1 CUBE textures for the reflection probes: the PBR
    // environment-map bind group expects Cube-dimension views, and the
    // default (empty) handle resolves to a 2D texture — which panics wgpu
    // validation the moment a lit 3D scene renders.
    let mut cube_image = Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        &[32, 32, 40, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    cube_image.texture_view_descriptor =
        Some(bevy::render::render_resource::TextureViewDescriptor {
            dimension: Some(bevy::render::render_resource::TextureViewDimension::Cube),
            ..Default::default()
        });
    let cube = images.add(cube_image);
    commands.insert_resource(ReflectionProbeAssets {
        diffuse: cube.clone(),
        specular: cube,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height * 4) as usize]
    }

    #[test]
    fn readback_map_wait_keeps_the_timeout_reachable() {
        assert!(
            matches!(readback_poll_type(), wgpu::PollType::Poll),
            "the production poll seam must remain explicitly nonblocking"
        );
        let (_sender, receiver) =
            std::sync::mpsc::channel::<std::result::Result<(), &'static str>>();
        let mut polls = 0;
        let error = wait_for_readback_map(&receiver, Duration::ZERO, || {
            polls += 1;
            Ok(())
        })
        .expect_err("a map that never completes must time out");

        assert_eq!(
            polls, 1,
            "the nonblocking poll must run before the deadline"
        );
        assert!(
            error
                .to_string()
                .contains("readback map timed out after 0ns"),
            "timeout must remain explicit and attributable: {error}"
        );
    }

    #[test]
    fn readback_map_wait_observes_completion_from_the_poll_callback() {
        let (sender, receiver) =
            std::sync::mpsc::channel::<std::result::Result<(), &'static str>>();
        let mut sender = Some(sender);
        let mut polls = 0;

        wait_for_readback_map(&receiver, Duration::from_secs(1), || {
            polls += 1;
            sender
                .take()
                .expect("completion sent once")
                .send(Ok(()))
                .expect("receiver remains connected");
            Ok(())
        })
        .expect("poll-driven callback completion must be observed");

        assert_eq!(polls, 1);
    }

    #[test]
    fn readback_map_wait_preserves_map_and_poller_failures() {
        let (map_sender, map_receiver) =
            std::sync::mpsc::channel::<std::result::Result<(), &'static str>>();
        map_sender
            .send(Err("map rejected"))
            .expect("receiver remains connected");
        let map_error = wait_for_readback_map(&map_receiver, Duration::from_secs(1), || Ok(()))
            .expect_err("map failure must be surfaced");
        assert!(
            map_error.to_string().contains("map rejected"),
            "map failure must retain its cause: {map_error}"
        );

        let (_poll_sender, poll_receiver) =
            std::sync::mpsc::channel::<std::result::Result<(), &'static str>>();
        let poll_error = wait_for_readback_map(&poll_receiver, Duration::from_secs(1), || {
            Err(anyhow!("poll hook failed"))
        })
        .expect_err("poller failure must be surfaced");
        assert!(
            poll_error.to_string().contains("poll hook failed"),
            "poller failure must retain its cause: {poll_error}"
        );

        let (disconnected_sender, disconnected_receiver) =
            std::sync::mpsc::channel::<std::result::Result<(), &'static str>>();
        drop(disconnected_sender);
        let disconnected_error =
            wait_for_readback_map(&disconnected_receiver, Duration::from_secs(1), || Ok(()))
                .expect_err("a vanished callback sender must be surfaced");
        assert!(
            disconnected_error
                .to_string()
                .contains("channel disconnected"),
            "callback disconnection must be attributable: {disconnected_error}"
        );
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

    /// Whether this host can produce trustworthy live GPU evidence.
    ///
    /// A software rasterizer (llvmpipe/lavapipe/warp) is not GPU-accelerated in
    /// any meaningful sense, and on this project's CI workers it returns blank
    /// frames. Before bd-2z0.14.3.8 those blanks slipped past the production
    /// blank guard and the live tests only failed when the honest frame
    /// happened to be uniform enough for the ALARM test's stricter threshold to
    /// notice — which read as flakiness and was really a dead pipeline being
    /// admitted as evidence.
    ///
    /// Skipping here is the honest outcome: it says this host cannot verify
    /// this, rather than passing on blank frames or failing on a defect that
    /// is not in the code under test. Real GPU hosts still run these.
    fn live_gpu_evidence_available(label: &str) -> bool {
        // One authority for "may this lane certify pixels", shared with any
        // non-test consumer, rather than a second copy of the rule here.
        match pixel_evidence_verdict(crate::probe_gpu_capability().as_ref()) {
            PixelEvidenceVerdict::Admissible => true,
            PixelEvidenceVerdict::SmokeOnly(reason) => {
                eprintln!("skipping {label}: {reason}");
                false
            }
        }
    }

    #[test]
    fn repeated_live_capture_is_byte_identical_and_science_digest_neutral() {
        if !live_gpu_evidence_available("live repeatability test") {
            return;
        }
        let world = seeded_world(0xD37E_2A11);
        let digest_before = world.world_digest_v1().expect("pre-render science digest");
        let config = OffscreenCaptureConfig {
            viewport: (160, 120),
            render_settings: RenderSettings::default(),
            corrupt: false,
        };
        let first = OffscreenCapture::run(&config, |session| {
            session.render(&world, "repeatability", 0xD37E_2A11, 1)
        })
        .expect("first live capture");
        assert_eq!(
            world.world_digest_v1().expect("digest after first render"),
            digest_before,
            "the first render must not advance or mutate scientific state"
        );
        let second = OffscreenCapture::run(&config, |session| {
            session.render(&world, "repeatability", 0xD37E_2A11, 1)
        })
        .expect("second live capture after a fresh session reset");

        assert_eq!(
            first.rgba8, second.rgba8,
            "the same frozen world across fresh sessions must produce identical RGBA bytes"
        );
        assert_eq!(
            world.world_digest_v1().expect("post-render science digest"),
            digest_before,
            "rendering must not advance or mutate scientific state"
        );
    }

    #[test]
    fn capture_tick_attribution_rejects_a_caller_mismatch() {
        let error = validate_capture_tick(41, 42).expect_err("mismatched tick must fail");
        assert!(
            error.to_string().contains("snapshot is tick 41"),
            "error must name the rendered snapshot tick: {error}"
        );
    }

    #[test]
    fn capture_seed_attribution_rejects_a_caller_mismatch() {
        let error = validate_capture_seed(41, 42).expect_err("mismatched seed must fail");
        assert!(
            error.to_string().contains("world root is 41"),
            "error must name the world root seed: {error}"
        );
    }

    #[test]
    fn fallback_adapter_override_matches_bevy_environment_contract() {
        assert_eq!(
            CaptureRendererInitError::AdapterUnavailable.to_string(),
            "no GPU adapter available for offscreen capture \
             (software lane requires llvmpipe/lavapipe via WGPU_BACKEND, or a real GPU)",
            "adapterless hosts must retain the one exact, auditable skip boundary"
        );
        assert!(!force_fallback_adapter(false, None));
        assert!(force_fallback_adapter(true, None));
        for disabled in ["", "0", "false"] {
            assert!(!force_fallback_adapter(true, Some(disabled)));
        }
        for enabled in ["1", "true", "TRUE", "yes"] {
            assert!(force_fallback_adapter(false, Some(enabled)));
        }
    }

    /// The C4 alarm contract: a deliberately corrupted render (blackout sun +
    /// crushed exposure) MUST fail the golden comparison against the honest
    /// capture — if it passes, the harness cannot see a broken pipeline.
    #[test]
    fn corrupted_capture_fails_comparison_against_honest_frame() {
        if !live_gpu_evidence_available("live corruption alarm test") {
            return;
        }
        const ROOT_SEED: u64 = 0xA14A2;
        let world = seeded_world(ROOT_SEED);
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
            session.render(&world, "alarm-honest", ROOT_SEED, 1)
        })
        .expect("honest capture");
        let corrupted = OffscreenCapture::run(&corrupt_config, |session| {
            session.render(&world, "alarm-corrupt", ROOT_SEED, 1)
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
        for px in honest.rgba8.as_chunks::<4>().0 {
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
        for px in jittered.as_chunks_mut::<4>().0.iter_mut().take(4) {
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

    /// A single-channel regression must be counted at full extent.
    ///
    /// The previous implementation counted differing CHANNELS and divided by
    /// four, so a change confined to one channel — a red shift, a dead blue
    /// term — reported at a quarter of its true size and was four times less
    /// likely to breach the ratio threshold. Under the old arithmetic the four
    /// differing pixels below would have been reported as one.
    #[test]
    fn compare_counts_pixels_not_channels_for_single_channel_drift() {
        let thresholds = CompareThresholds {
            differing_channel: 0,
            max_differing_ratio: 0.0,
            max_mean_abs_diff: 255.0,
            // Disabled: this test isolates per-channel counting, and the frame
            // is smaller than one SSIM window anyway.
            min_ssim_mean: -1.0,
            min_ssim_window: -1.0,
        };
        // 4 pixels, each differing in RED only.
        let golden = [10u8, 20, 30, 255].repeat(4);
        let mut candidate = golden.clone();
        for pixel in 0..4 {
            candidate[pixel * 4] = 40;
        }
        let stats = compare_frames(&golden, &candidate, 4, 1, &thresholds).expect("shape matches");
        assert_eq!(
            stats.differing_pixels, 4,
            "every pixel differs in red, so all four must be counted"
        );
        assert_eq!(stats.total_pixels, 4);
        assert!((stats.differing_ratio - 1.0).abs() < f64::EPSILON);
        assert!(!stats.pass, "a full-frame red shift must not pass");
    }

    /// A pixel differing in all four channels is still ONE pixel.
    #[test]
    fn compare_does_not_multiply_a_fully_differing_pixel() {
        let thresholds = CompareThresholds {
            differing_channel: 0,
            max_differing_ratio: 1.0,
            max_mean_abs_diff: 255.0,
            // Disabled for the same reason as the sibling test above.
            min_ssim_mean: -1.0,
            min_ssim_window: -1.0,
        };
        let golden = [10u8, 20, 30, 40].repeat(2);
        let mut candidate = golden.clone();
        for channel in 0..4 {
            candidate[channel] = 200;
        }
        let stats = compare_frames(&golden, &candidate, 2, 1, &thresholds).expect("shape matches");
        assert_eq!(
            stats.differing_pixels, 1,
            "one pixel differing in four channels is one differing pixel"
        );
    }

    /// Alpha must never count as content, and neither must trivial variation.
    ///
    /// This test previously asserted that a ONE UNIT difference in a single
    /// channel made a frame valid evidence. That assertion encoded the
    /// bd-2z0.14.3.8 defect: a 160x120 capture whose total per-channel range
    /// was 1 passed the production guard and was returned as a legitimate
    /// frame, while the live alarm test — holding a stricter bar than
    /// production — was the only thing that noticed. The expectation was
    /// wrong, so it is inverted here rather than preserved.
    #[test]
    fn blank_detection_ignores_opaque_alpha_and_requires_real_rgb_range() {
        // All-black with opaque alpha: blank. Alpha is not content.
        assert!(rgba8_is_visually_blank(&[0, 0, 0, 255, 0, 0, 0, 255]));
        // Uniform non-black clear colour: blank.
        assert!(rgba8_is_visually_blank(&[20, 30, 40, 255, 20, 30, 40, 255]));
        // One unit of variation: STILL blank. This is the case that shipped.
        assert!(rgba8_is_visually_blank(&[20, 30, 40, 255, 21, 30, 40, 255]));
        // Just under the floor: blank.
        assert!(rgba8_is_visually_blank(&[
            20,
            30,
            40,
            255,
            20 + (MIN_EVIDENCE_RGB_SPREAD as u8 - 1),
            30,
            40,
            255
        ]));
        // At the floor: real evidence.
        assert!(!rgba8_is_visually_blank(&[
            20,
            30,
            40,
            255,
            20 + MIN_EVIDENCE_RGB_SPREAD as u8,
            30,
            40,
            255
        ]));
        // Spread summed ACROSS channels also clears the floor, so a scene
        // varying a little in each channel is not mistaken for a dead frame.
        assert!(!rgba8_is_visually_blank(&[
            20, 30, 40, 255, 23, 33, 43, 255
        ]));
        // Empty input is blank rather than a panic.
        assert!(rgba8_is_visually_blank(&[]));
    }

    #[test]
    fn provenance_serializes_with_required_fields() {
        let provenance = CaptureProvenance {
            schema: PROVENANCE_SCHEMA.to_string(),
            world_digest: Some("blake3:deadbeef".to_string()),
            scene: "unit".to_string(),
            seed: 42,
            tick: 12,
            frontend: "bevy_offscreen".to_string(),
            adapter_name: "test-adapter".to_string(),
            backend: "Metal".to_string(),
            device_type: "IntegratedGpu".to_string(),
            quality_tier: "Medium".to_string(),
            viewport: [256, 256],
            colorspace: "rgba8-srgb".to_string(),
            rustc_version: "rustc test".to_string(),
            target_triple: "aarch64-apple-darwin".to_string(),
            source_commit: Some("0123456789abcdef0123456789abcdef01234567".to_string()),
            lock_digest: Some("0f1e2d3c4b5a6978".to_string()),
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
            // A capture that cannot say which world it depicts records how it
            // was rendered but not what it shows (bd-2z0.14.3.4).
            "world_digest",
            // ...and one that cannot say which source tree rendered it cannot
            // separate a renderer change from a different build (bd-2z0.14.3.10).
            "source_commit",
            "lock_digest",
        ] {
            assert!(
                json.get(field).is_some(),
                "missing provenance field {field}"
            );
        }
        assert_eq!(json["schema"], PROVENANCE_SCHEMA);
        assert_eq!(json["world_digest"], "blake3:deadbeef");
    }

    fn gpu_info(name: &str, class: scriptbots_core::GpuClass) -> scriptbots_core::GpuInfo {
        scriptbots_core::GpuInfo {
            name: name.to_string(),
            backend: "Vulkan".to_string(),
            class,
            vram_bytes: None,
            max_texture_2d: Some(16_384),
            timestamp_queries: true,
        }
    }

    /// The item 4 decision, pinned: a software rasterizer may smoke-test the
    /// capture path but must never certify pixels.
    ///
    /// Full reported capability must NOT buy admissibility. llvmpipe advertises a
    /// Vulkan backend, 16384 max texture size and timestamp queries, and still
    /// returns uniform frames from this path — so a verdict keyed on capability
    /// flags rather than device class would wave it straight through.
    #[test]
    fn a_software_rasterizer_may_smoke_test_but_never_certify_pixels() {
        let verdict = pixel_evidence_verdict(Some(&gpu_info(
            "llvmpipe (LLVM 21.1.8, 256 bits)",
            scriptbots_core::GpuClass::Software,
        )));
        assert!(
            !verdict.admissible(),
            "a software rasterizer must not certify pixel evidence"
        );
        let PixelEvidenceVerdict::SmokeOnly(reason) = verdict else {
            panic!("expected a smoke-only verdict");
        };
        assert!(
            reason.contains("llvmpipe"),
            "the refusal must name what it found: {reason}"
        );
        assert!(
            reason.contains("investigate"),
            "the refusal must stay actionable rather than final: {reason}"
        );
    }

    /// A missing adapter is smoke-only for a different reason, and must say so
    /// rather than reusing the software-rasterizer text.
    #[test]
    fn an_absent_adapter_is_smoke_only_and_says_why() {
        let verdict = pixel_evidence_verdict(None);
        assert!(!verdict.admissible());
        let PixelEvidenceVerdict::SmokeOnly(reason) = verdict else {
            panic!("expected a smoke-only verdict");
        };
        assert!(
            reason.contains("no GPU adapter"),
            "an absent adapter must not be reported as a software rasterizer: {reason}"
        );
    }

    /// A blank frame from a software rasterizer must be explained as a lane
    /// limitation, naming the adapter, and must stay actionable.
    #[test]
    fn a_blank_frame_on_a_software_lane_is_explained_as_a_lane_limitation() {
        let error = blank_frame_error(Some(&gpu_info(
            "llvmpipe (LLVM 21.1.8, 256 bits)",
            scriptbots_core::GpuClass::Software,
        )))
        .to_string();
        assert!(
            error.contains("could not have certified pixels"),
            "the message must say the lane was never able to prove this: {error}"
        );
        assert!(
            error.contains("llvmpipe"),
            "the message must name the adapter it is talking about: {error}"
        );
    }

    /// The same blank frame on capable hardware is the OPPOSITE finding, and the
    /// message must not let a reader file it as an environment quirk.
    ///
    /// This is the distinction the old single message destroyed: identical text
    /// for "this lane never could" and "your pipeline just broke".
    #[test]
    fn a_blank_frame_on_capable_hardware_is_reported_as_a_real_pipeline_failure() {
        for class in [
            scriptbots_core::GpuClass::Discrete,
            scriptbots_core::GpuClass::Integrated,
        ] {
            let error = blank_frame_error(Some(&gpu_info("Apple M4", class))).to_string();
            assert!(
                error.contains("real pipeline failure"),
                "{class:?} must be reported as a genuine failure: {error}"
            );
            assert!(
                error.contains("Apple M4"),
                "the message must name the adapter: {error}"
            );
            assert!(
                !error.contains("could not have certified pixels"),
                "capable hardware must never be excused as a lane limitation: {error}"
            );
        }
    }

    /// Real hardware stays admissible, so this decision narrows nothing that
    /// currently works.
    #[test]
    fn real_hardware_still_certifies_pixels() {
        for class in [
            scriptbots_core::GpuClass::Discrete,
            scriptbots_core::GpuClass::Integrated,
        ] {
            assert!(
                pixel_evidence_verdict(Some(&gpu_info("Apple M4", class))).admissible(),
                "{class:?} must still be able to certify pixels"
            );
        }
    }

    /// Build a frame from a per-pixel closure, for structural comparison tests.
    fn frame_from(width: u32, height: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
        let mut out = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                out.extend_from_slice(&f(x, y));
            }
        }
        out
    }

    /// Identical frames must score a perfect 1.0 on both structural statistics,
    /// including across flat regions where variance is zero.
    #[test]
    fn ssim_of_identical_frames_is_one_even_on_flat_regions() {
        let flat = frame_from(64, 64, |_, _| [40, 40, 40, 255]);
        let (mean, min) = ssim_luma(&flat, &flat, 64, 64);
        assert!(
            (mean - 1.0).abs() < 1e-9 && (min - 1.0).abs() < 1e-9,
            "identical flat frames must be 1.0/1.0, got {mean}/{min}"
        );

        let textured = frame_from(64, 64, |x, y| {
            let v = ((x * 7 + y * 13) % 256) as u8;
            [v, v / 2, 255 - v, 255]
        });
        let (mean, min) = ssim_luma(&textured, &textured, 64, 64);
        assert!(
            (mean - 1.0).abs() < 1e-9 && (min - 1.0).abs() < 1e-9,
            "identical textured frames must be 1.0/1.0, got {mean}/{min}"
        );
    }

    /// The defect item 3 names: a small, severe, LOCALIZED regression passes
    /// every per-channel threshold purely by being small.
    ///
    /// One 8x8 block of a 256x256 frame is 64 of 65,536 pixels — a differing
    /// ratio of 0.00098, well under the 0.005 allowance — and inverting it
    /// moves the mean absolute deviation by a fraction of a unit. Per-channel
    /// statistics therefore report "pass" on a frame with a visibly destroyed
    /// region, which is exactly what a dead mesh or an unshaded material looks
    /// like. The worst-window structural score is what refuses it.
    #[test]
    fn a_localized_structural_regression_passes_per_channel_and_fails_perceptually() {
        let golden = frame_from(256, 256, |x, y| {
            let v = ((x * 3 + y * 5) % 200 + 28) as u8;
            [v, v, v, 255]
        });
        let candidate = frame_from(256, 256, |x, y| {
            let v = ((x * 3 + y * 5) % 200 + 28) as u8;
            if x < 8 && y < 8 {
                [255 - v, 255 - v, 255 - v, 255]
            } else {
                [v, v, v, 255]
            }
        });

        let thresholds = CompareThresholds::default();
        let stats = compare_frames(&golden, &candidate, 256, 256, &thresholds).expect("compare");

        // The per-channel half of the gate is genuinely satisfied.
        assert!(
            stats.differing_ratio <= f64::from(thresholds.max_differing_ratio),
            "premise: differing ratio {} must sit under the allowance",
            stats.differing_ratio
        );
        assert!(
            stats.mean_abs_diff <= f64::from(thresholds.max_mean_abs_diff),
            "premise: mean abs diff {} must sit under the allowance",
            stats.mean_abs_diff
        );

        // ...and the structural half still refuses it.
        assert!(
            stats.ssim_min < f64::from(thresholds.min_ssim_window),
            "worst window {} must breach the structural floor",
            stats.ssim_min
        );
        assert!(
            !stats.pass,
            "a destroyed 8x8 region must not pass merely because it is small"
        );
    }

    /// The mean alone is not enough, which is why the worst window is gated too.
    ///
    /// Averaged over 1,024 windows, a single destroyed one moves the mean by
    /// under a thousandth — it would sail past any mean threshold loose enough
    /// to tolerate real adapter jitter.
    #[test]
    fn mean_ssim_alone_would_miss_localized_damage() {
        let golden = frame_from(256, 256, |x, y| {
            let v = ((x * 3 + y * 5) % 200 + 28) as u8;
            [v, v, v, 255]
        });
        let candidate = frame_from(256, 256, |x, y| {
            let v = ((x * 3 + y * 5) % 200 + 28) as u8;
            if x < 8 && y < 8 {
                [255 - v, 255 - v, 255 - v, 255]
            } else {
                [v, v, v, 255]
            }
        });
        let (mean, min) = ssim_luma(&golden, &candidate, 256, 256);
        assert!(
            mean > 0.98,
            "premise: the mean stays above a realistic threshold ({mean})"
        );
        assert!(
            min < mean,
            "the worst window must be strictly more sensitive than the mean ({min} vs {mean})"
        );
    }

    /// The structural gate must never rescue a comparison the per-channel gate
    /// failed. It is additive by construction; this pins that it stays so.
    #[test]
    fn structural_agreement_cannot_rescue_a_per_channel_failure() {
        // A large, uniform brightness lift: structurally near-identical (every
        // window keeps its shape) but a gross per-channel deviation.
        let golden = frame_from(64, 64, |x, y| {
            let v = ((x * 3 + y * 5) % 200 + 28) as u8;
            [v, v, v, 255]
        });
        let candidate = frame_from(64, 64, |x, y| {
            let v = ((x * 3 + y * 5) % 200 + 28) as u8;
            [
                v.saturating_add(60),
                v.saturating_add(60),
                v.saturating_add(60),
                255,
            ]
        });
        let stats = compare_frames(&golden, &candidate, 64, 64, &CompareThresholds::default())
            .expect("compare");
        assert!(
            stats.mean_abs_diff > f64::from(CompareThresholds::default().max_mean_abs_diff),
            "premise: this must be a per-channel failure"
        );
        assert!(
            !stats.pass,
            "high structural similarity must not override a channel breach"
        );
    }

    /// Provenance JSON written before build identity existed must still decode.
    ///
    /// This is the exact shape of the committed goldens under
    /// `scriptbots-app/tests/scenes/goldens/`, which are parsed rather than
    /// byte-compared. If the new fields were required, every one of those
    /// artifacts would become unreadable and the harness would fail on its own
    /// history rather than on a visual regression.
    #[test]
    fn provenance_v1_json_without_build_identity_still_decodes() {
        let legacy = r#"{
            "schema": "scriptbots.capture-provenance.v1",
            "scene": "empty_world",
            "seed": 1001,
            "tick": 0,
            "frontend": "bevy_offscreen",
            "adapter_name": "Apple M4",
            "backend": "Metal",
            "device_type": "Integrated",
            "quality_tier": "Medium",
            "viewport": [1280, 720],
            "colorspace": "rgba8-srgb",
            "rustc_version": "",
            "target_triple": "aarch64-apple-darwin",
            "corrupt": false
        }"#;
        let decoded: CaptureProvenance =
            serde_json::from_str(legacy).expect("a pre-existing v1 golden must still decode");
        assert_eq!(decoded.scene, "empty_world");
        assert!(
            decoded.source_commit.is_none(),
            "an artifact from before the field existed must read as unknown, not as a value"
        );
        assert!(decoded.lock_digest.is_none());
        assert!(
            decoded.world_digest.is_none(),
            "the same additive rule that let world_digest land must still hold"
        );
    }

    /// The build script must actually inject build identity, not merely offer a
    /// field for it.
    ///
    /// `rustc_version` is the cautionary case this guards against: it was read
    /// via `option_env!("SCRIPTBOTS_RUSTC_VERSION").or(option_env!("CARGO_PKG_RUST_VERSION"))`
    /// with nothing in the workspace setting either, and because cargo sets
    /// `CARGO_PKG_RUST_VERSION` to an EMPTY string when `rust-version` is absent,
    /// the `.or` succeeded with `""` and never reached the `"unknown"` fallback.
    /// Every committed golden therefore carries `"rustc_version": ""`. A field
    /// that exists and is always blank looks like provenance and proves nothing.
    #[test]
    fn build_script_injects_real_build_identity() {
        let workspace_lock = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../Cargo.lock")
            .exists();
        if !workspace_lock {
            // Vendored or tarball build: the script correctly injects nothing.
            return;
        }

        let lock_digest =
            option_env!("SCRIPTBOTS_LOCK_DIGEST").expect("build.rs must inject the lock digest");
        assert_eq!(lock_digest.len(), 16, "FNV-1a-64 renders as 16 hex digits");
        assert!(
            lock_digest.chars().all(|c| c.is_ascii_hexdigit()),
            "lock digest must be hex, got {lock_digest}"
        );

        let rustc = option_env!("SCRIPTBOTS_RUSTC_VERSION")
            .expect("build.rs must inject the toolchain identity");
        assert!(
            !rustc.trim().is_empty(),
            "the blank-rustc_version hole must stay closed"
        );

        let target =
            option_env!("SCRIPTBOTS_TARGET_TRIPLE").expect("build.rs must inject the true target");
        assert!(
            !target.trim().is_empty() && target.contains('-'),
            "target triple must be cargo's resolved TARGET, got {target}"
        );

        // Only assert shape when git was available; a repo-less build legitimately
        // has no commit to report.
        if let Some(commit) = option_env!("SCRIPTBOTS_SOURCE_COMMIT") {
            let base = commit.strip_suffix("-dirty").unwrap_or(commit);
            assert_eq!(
                base.len(),
                40,
                "git object names are 40 hex digits: {commit}"
            );
            assert!(
                base.chars().all(|c| c.is_ascii_hexdigit()),
                "commit must be hex, got {commit}"
            );
        }
    }

    /// An absent world digest must serialize as an explicit null, never vanish.
    ///
    /// A missing key would let a reader assume the field was simply not part of
    /// this schema version, rather than that identity was unavailable for this
    /// specific capture. Absence of evidence has to look different from absence
    /// of the question.
    #[test]
    fn provenance_records_an_unavailable_world_digest_explicitly() {
        let provenance = CaptureProvenance {
            schema: PROVENANCE_SCHEMA.to_string(),
            world_digest: None,
            scene: "unit".to_string(),
            seed: 1,
            tick: 1,
            frontend: "bevy_offscreen".to_string(),
            adapter_name: "test-adapter".to_string(),
            backend: "Vulkan".to_string(),
            device_type: "Software".to_string(),
            quality_tier: "Potato".to_string(),
            viewport: [16, 16],
            colorspace: "rgba8-srgb".to_string(),
            rustc_version: "rustc test".to_string(),
            target_triple: "x86_64-unknown-linux-gnu".to_string(),
            source_commit: None,
            lock_digest: None,
            corrupt: false,
        };
        let json = serde_json::to_value(&provenance).expect("serialize");
        assert!(
            json.get("world_digest").is_some(),
            "the key must be present even when the digest is unavailable"
        );
        assert!(json["world_digest"].is_null());
    }
}
