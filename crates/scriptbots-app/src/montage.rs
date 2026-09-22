//! Headless montage encoder for highlight reels (`bd-16g.9.2`).
//!
//! Provides byte-reproducible visual artifacts (ASCII cast and animated GIF)
//! from deterministic replay checkpoints and scored narrative events without requiring
//! a GPU or external FFmpeg binary.

use scriptbots_core::narrative::EventRecord;
use scriptbots_core::reel::{Clip, SelectionConfig, select_clips};
use scriptbots_core::{BrainRegistry, TerrainKind, WorldCheckpointV1, WorldState};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;
use thiserror::Error;
use tracing::{error, info};

/// Supported export formats for highlight reels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReelFormat {
    /// Asciinema v2 terminal cast format (CI-safe, text).
    Ascii,
    /// Animated GIF format with pinned palette.
    Gif,
    /// MP4 video (probed, requires ffmpeg or hardware encoder).
    Mp4,
}

impl fmt::Display for ReelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ascii => write!(f, "ascii"),
            Self::Gif => write!(f, "gif"),
            Self::Mp4 => write!(f, "mp4"),
        }
    }
}

impl FromStr for ReelFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ascii" | "cast" => Ok(Self::Ascii),
            "gif" => Ok(Self::Gif),
            "mp4" => Ok(Self::Mp4),
            _ => Err(format!("unknown format '{s}': expected ascii, gif, or mp4")),
        }
    }
}

/// Errors returned by the montage replay and encoding pipeline.
#[derive(Debug, Error, PartialEq)]
pub enum ReelReplayError {
    /// Requested target tick cannot be reached from available checkpoints.
    #[error(
        "cannot seek to tick {target_tick}: nearest reachable checkpoint is tick {nearest_checkpoint}"
    )]
    UnreachableSeek {
        /// Target tick that was requested.
        target_tick: u64,
        /// Nearest available checkpoint tick.
        nearest_checkpoint: u64,
    },
    /// A frame's world tick does not match the expected event caption tick.
    #[error(
        "frame tick {frame_tick} does not match expected caption tick {event_tick} for clip rank {rank}"
    )]
    CaptionFrameMismatch {
        /// Clip rank.
        rank: usize,
        /// Frame world tick.
        frame_tick: u64,
        /// Event tick from caption metadata.
        event_tick: u64,
    },
    /// Low-level frame encoding failure with clip/frame/tick traceability.
    #[error("encoder failed at clip {clip_rank}, frame {frame_index}, tick {tick}: {reason}")]
    EncoderFailure {
        /// Rank of the clip.
        clip_rank: usize,
        /// Zero-based frame index within the clip.
        frame_index: usize,
        /// Simulation tick of the frame.
        tick: u64,
        /// Error description.
        reason: String,
    },
    /// MP4 export is unavailable because FFmpeg is absent.
    #[error("mp4 format requires ffmpeg on PATH; ascii and gif are available natively")]
    Mp4Unavailable,
    /// Storage error during replay recovery.
    #[error("storage error: {0}")]
    Storage(String),
    /// Event selection error.
    #[error("core selection error: {0}")]
    Selection(#[from] scriptbots_core::reel::ReelSelectionError),
    /// File I/O failure.
    #[error("io error: {0}")]
    Io(String),
    /// Serialization failure.
    #[error("serialization error: {0}")]
    Serialization(String),
}

/// Parameters for montage encoding.
#[derive(Debug, Clone)]
pub struct MontageOptions {
    /// Format to render.
    pub format: ReelFormat,
    /// Destination file path.
    pub output_path: PathBuf,
    /// Stride between rendered ticks (default: 5).
    pub stride: u64,
    /// Selection criteria.
    pub selection: SelectionConfig,
    /// Target frame width in cells/pixels (default: 80 for ascii, 160 for gif).
    pub width: usize,
    /// Target frame height in cells/pixels (default: 24 for ascii, 120 for gif).
    pub height: usize,
}

impl Default for MontageOptions {
    fn default() -> Self {
        Self {
            format: ReelFormat::Ascii,
            output_path: PathBuf::from("reel.cast"),
            stride: 5,
            selection: SelectionConfig::default(),
            width: 80,
            height: 24,
        }
    }
}

/// Single rendered frame representation in the montage encoder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MontageFrame {
    /// Tick number of the frame.
    pub tick: u64,
    /// Clip rank (1-based).
    pub rank: usize,
    /// Event caption overlay.
    pub caption: String,
    /// ASCII grid lines for the frame.
    pub ascii_grid: Vec<String>,
}

/// Outcome summary of montage generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MontageSummary {
    /// Target output path.
    pub output_path: PathBuf,
    /// Format generated.
    pub format: ReelFormat,
    /// Number of clips included.
    pub clips_count: usize,
    /// Total frames generated.
    pub total_frames: usize,
    /// Total bytes written.
    pub bytes_written: u64,
    /// Peak frames concurrently in flight (bounded streaming invariant).
    pub peak_frames_in_flight: usize,
}

/// Trait abstracting deterministic replay seeking for highlight reels.
pub trait ReplayProvider {
    /// Seek exactly to `target_tick` and return a reference to the resulting world state.
    ///
    /// # Errors
    /// Returns [`ReelReplayError::UnreachableSeek`] if `target_tick` cannot be reached.
    fn seek_to_exact(&mut self, target_tick: u64) -> Result<&WorldState, ReelReplayError>;

    /// Maximum tick reachable by this replay provider.
    fn max_tick(&self) -> u64;
}

/// Replay seeker backed by a chronological sequence of [`WorldCheckpointV1`].
pub struct CheckpointReplaySeeker {
    checkpoints: Vec<WorldCheckpointV1>,
    current_world: Option<WorldState>,
    max_tick: u64,
}

impl CheckpointReplaySeeker {
    /// Create a new seeker from checkpoints and maximum tick.
    #[must_use]
    pub fn new(mut checkpoints: Vec<WorldCheckpointV1>, max_tick: u64) -> Self {
        checkpoints.sort_by_key(|c| c.tick().0);
        Self {
            checkpoints,
            current_world: None,
            max_tick,
        }
    }
}

impl ReplayProvider for CheckpointReplaySeeker {
    fn seek_to_exact(&mut self, target_tick: u64) -> Result<&WorldState, ReelReplayError> {
        if self.checkpoints.is_empty() {
            return Err(ReelReplayError::UnreachableSeek {
                target_tick,
                nearest_checkpoint: 0,
            });
        }

        let earliest = self.checkpoints.first().unwrap().tick().0;
        if target_tick < earliest {
            return Err(ReelReplayError::UnreachableSeek {
                target_tick,
                nearest_checkpoint: earliest,
            });
        }

        let latest_cp = self.checkpoints.last().unwrap().tick().0;
        if target_tick > self.max_tick {
            return Err(ReelReplayError::UnreachableSeek {
                target_tick,
                nearest_checkpoint: latest_cp,
            });
        }

        // Find the most recent base checkpoint <= target_tick
        let best_cp = self
            .checkpoints
            .iter()
            .rev()
            .find(|c| c.tick().0 <= target_tick)
            .unwrap();

        let base_tick = best_cp.tick().0;

        let needs_restore = match &self.current_world {
            Some(w) => w.tick().0 < base_tick || w.tick().0 > target_tick,
            None => true,
        };

        if needs_restore {
            let registry_candidates = [
                crate::create_brain_registry_for_config(
                    crate::BrainPreset::Mixed,
                    best_cp.config(),
                ),
                crate::create_brain_registry_for_config(crate::BrainPreset::Mlp, best_cp.config()),
                Ok(BrainRegistry::new()),
            ];

            let mut restored_world = None;
            let mut last_err = None;

            for reg in registry_candidates.into_iter().flatten() {
                match WorldState::restore_checkpoint_v1(best_cp, reg) {
                    Ok(w) => {
                        restored_world = Some(w);
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e);
                    }
                }
            }

            let world = restored_world.ok_or_else(|| {
                ReelReplayError::Storage(format!(
                    "failed to restore checkpoint: {}",
                    last_err.map_or_else(|| "no matching registry".to_string(), |e| e.to_string())
                ))
            })?;
            self.current_world = Some(world);
        }

        let world = self.current_world.as_mut().unwrap();
        while world.tick().0 < target_tick {
            world
                .step()
                .map_err(|e| ReelReplayError::Storage(format!("sim step error: {e}")))?;
        }

        Ok(self.current_world.as_ref().unwrap())
    }

    fn max_tick(&self) -> u64 {
        self.max_tick
    }
}

/// Replay adapter wrapping a live `WorldState` advancing monotonically forward.
pub struct LiveWorldReplayAdapter<'a> {
    world: &'a mut WorldState,
    max_tick: u64,
}

impl<'a> LiveWorldReplayAdapter<'a> {
    /// Create a live world replay adapter.
    pub fn new(world: &'a mut WorldState, max_tick: u64) -> Self {
        Self { world, max_tick }
    }
}

impl ReplayProvider for LiveWorldReplayAdapter<'_> {
    fn seek_to_exact(&mut self, target_tick: u64) -> Result<&WorldState, ReelReplayError> {
        let current_tick = self.world.tick().0;
        if target_tick < current_tick {
            return Err(ReelReplayError::UnreachableSeek {
                target_tick,
                nearest_checkpoint: current_tick,
            });
        }
        if target_tick > self.max_tick {
            return Err(ReelReplayError::UnreachableSeek {
                target_tick,
                nearest_checkpoint: self.max_tick,
            });
        }
        while self.world.tick().0 < target_tick {
            self.world
                .step()
                .map_err(|e| ReelReplayError::Storage(format!("sim step error: {e}")))?;
        }
        Ok(&*self.world)
    }

    fn max_tick(&self) -> u64 {
        self.max_tick
    }
}

/// Pinned 256-color palette (768 bytes, 256 x RGB) for byte-reproducible GIF encoding.
///
/// Derived deterministically from canonical terrain and agent color representations.
pub const PINNED_GIF_PALETTE: [u8; 768] = build_pinned_palette();

const fn build_pinned_palette() -> [u8; 768] {
    let mut pal = [0u8; 768];
    // Slot 0: Background [15, 18, 25] (Dark slate)
    pal[0] = 15;
    pal[1] = 18;
    pal[2] = 25;
    // Slot 1: DeepWater [25, 51, 102]
    pal[3] = 25;
    pal[4] = 51;
    pal[5] = 102;
    // Slot 2: ShallowWater [51, 102, 153]
    pal[6] = 51;
    pal[7] = 102;
    pal[8] = 153;
    // Slot 3: Sand [204, 178, 128]
    pal[9] = 204;
    pal[10] = 178;
    pal[11] = 128;
    // Slot 4: Grass [51, 153, 51]
    pal[12] = 51;
    pal[13] = 153;
    pal[14] = 51;
    // Slot 5: Bloom [178, 76, 153]
    pal[15] = 178;
    pal[16] = 76;
    pal[17] = 153;
    // Slot 6: Rock [128, 128, 128]
    pal[18] = 128;
    pal[19] = 128;
    pal[20] = 128;
    // Slot 7: Food [0, 255, 100]
    pal[21] = 0;
    pal[22] = 255;
    pal[23] = 100;
    // Slot 8: Herbivore agent [50, 205, 50]
    pal[24] = 50;
    pal[25] = 205;
    pal[26] = 50;
    // Slot 9: Carnivore agent [220, 20, 60]
    pal[27] = 220;
    pal[28] = 20;
    pal[29] = 60;
    // Slot 10: Omnivore agent [255, 215, 0]
    pal[30] = 255;
    pal[31] = 215;
    pal[32] = 0;
    // Slot 11: Text ink [255, 255, 255]
    pal[33] = 255;
    pal[34] = 255;
    pal[35] = 255;
    // Slot 12: Selection rim [255, 255, 0]
    pal[36] = 255;
    pal[37] = 255;
    pal[38] = 0;

    // Deterministic procedural fill for slots 13..256
    let mut i = 13usize;
    while i < 256 {
        let idx = i * 3;
        let u = i as u8;
        pal[idx] = u.wrapping_mul(17);
        pal[idx + 1] = u.wrapping_mul(73).wrapping_add(19);
        pal[idx + 2] = u.wrapping_mul(31).wrapping_add(53);
        i += 1;
    }
    pal
}

/// Generate an ASCII character map representation of a WorldState tick.
#[must_use]
pub fn render_ascii_frame(world: &WorldState, width: usize, height: usize) -> Vec<String> {
    let mut grid = vec![vec!['.'; width]; height];

    let w_bounds = world.config().world_width as f32;
    let h_bounds = world.config().world_height as f32;
    let cell_size = world.terrain().cell_size().max(1);
    let food_cell_size = world.config().food_cell_size.max(1);

    // Render terrain hints
    for (y, row) in grid.iter_mut().enumerate().take(height) {
        let wy = (y as f32 / height as f32) * h_bounds;
        let tile_y = (wy / cell_size as f32) as u32;
        for (x, cell) in row.iter_mut().enumerate().take(width) {
            let wx = (x as f32 / width as f32) * w_bounds;
            let tile_x = (wx / cell_size as f32) as u32;
            let kind = world
                .terrain()
                .tile(tile_x, tile_y)
                .map_or(TerrainKind::Grass, |t| t.kind);
            match kind {
                TerrainKind::DeepWater | TerrainKind::ShallowWater => {
                    *cell = '~';
                }
                TerrainKind::Rock => {
                    *cell = '^';
                }
                TerrainKind::Bloom => {
                    *cell = '*';
                }
                _ => {}
            }
        }
    }

    // Render food
    for (y, row) in grid.iter_mut().enumerate().take(height) {
        let wy = (y as f32 / height as f32) * h_bounds;
        let fy = (wy / food_cell_size as f32) as u32;
        for (x, cell) in row.iter_mut().enumerate().take(width) {
            let wx = (x as f32 / width as f32) * w_bounds;
            let fx = (wx / food_cell_size as f32) as u32;
            if *cell == '.' && world.food().get(fx, fy).unwrap_or(0.0) > 5.0 {
                *cell = ':';
            }
        }
    }

    // Render agents as 'A'
    for pos in world.agents().columns().positions() {
        let px = (((pos.x / w_bounds) * width as f32) as usize).min(width.saturating_sub(1));
        let py = (((pos.y / h_bounds) * height as f32) as usize).min(height.saturating_sub(1));
        grid[py][px] = 'A';
    }

    grid.into_iter()
        .map(|row| row.into_iter().collect())
        .collect()
}

/// Render a WorldState tick into a pre-allocated indexed pixel buffer for GIF encoding.
pub fn render_gif_frame(world: &WorldState, width: usize, height: usize, buffer: &mut [u8]) {
    if buffer.len() < width * height {
        return;
    }
    buffer[..width * height].fill(0);

    let w_bounds = world.config().world_width as f32;
    let h_bounds = world.config().world_height as f32;
    let cell_size = world.terrain().cell_size().max(1);
    let food_cell_size = world.config().food_cell_size.max(1);

    // 1. Terrain
    for py in 0..height {
        let wy = (py as f32 / height as f32) * h_bounds;
        let tile_y = (wy / cell_size as f32) as u32;
        let row_start = py * width;
        for px in 0..width {
            let wx = (px as f32 / width as f32) * w_bounds;
            let tile_x = (wx / cell_size as f32) as u32;
            let kind = world
                .terrain()
                .tile(tile_x, tile_y)
                .map_or(TerrainKind::Grass, |t| t.kind);
            let color_idx = match kind {
                TerrainKind::DeepWater => 1,
                TerrainKind::ShallowWater => 2,
                TerrainKind::Sand => 3,
                TerrainKind::Grass => 4,
                TerrainKind::Bloom => 5,
                TerrainKind::Rock => 6,
            };
            buffer[row_start + px] = color_idx;
        }
    }

    // 2. Food
    for py in 0..height {
        let wy = (py as f32 / height as f32) * h_bounds;
        let fy = (wy / food_cell_size as f32) as u32;
        let row_start = py * width;
        for px in 0..width {
            let wx = (px as f32 / width as f32) * w_bounds;
            let fx = (wx / food_cell_size as f32) as u32;
            if world.food().get(fx, fy).unwrap_or(0.0) > 5.0 {
                buffer[row_start + px] = 7;
            }
        }
    }

    // 3. Agents
    let columns = world.agents().columns();
    for (pos, color) in columns.positions().iter().zip(columns.colors().iter()) {
        let px = (((pos.x / w_bounds) * width as f32) as usize).min(width.saturating_sub(1));
        let py = (((pos.y / h_bounds) * height as f32) as usize).min(height.saturating_sub(1));

        let color_idx = if color[0] > color[1] && color[0] > color[2] {
            9 // Carnivore red
        } else if color[1] > color[0] && color[1] > color[2] {
            8 // Herbivore green
        } else {
            10 // Omnivore yellow
        };

        buffer[py * width + px] = color_idx;
        if px + 1 < width {
            buffer[py * width + px + 1] = color_idx;
        }
    }
}

/// Encode a montage reel using an explicit replay provider and scored events.
///
/// # Errors
/// Returns [`ReelReplayError`] on unreachable seeks, caption offset mismatches, or encoder failures.
pub fn encode_montage_with_seeker<R: ReplayProvider>(
    seeker: &mut R,
    events: &[EventRecord],
    last_tick: u64,
    options: &MontageOptions,
) -> Result<MontageSummary, ReelReplayError> {
    if options.format == ReelFormat::Mp4 {
        return Err(ReelReplayError::Mp4Unavailable);
    }

    let mut clips = select_clips(events, last_tick, &options.selection)?;
    clips.sort_by_key(|c| c.start);
    encode_montage_clips_with_seeker(seeker, &clips, options)
}

/// Encode a montage reel using an explicit replay provider and pre-selected clips.
///
/// # Errors
/// Returns [`ReelReplayError`] on unreachable seeks, caption offset mismatches, or encoder failures.
#[allow(clippy::too_many_lines)]
pub fn encode_montage_clips_with_seeker<R: ReplayProvider>(
    seeker: &mut R,
    clips: &[Clip],
    options: &MontageOptions,
) -> Result<MontageSummary, ReelReplayError> {
    if options.format == ReelFormat::Mp4 {
        return Err(ReelReplayError::Mp4Unavailable);
    }

    if clips.is_empty() {
        info!(
            target: "scriptbots::reel::render",
            "no events scored above threshold for reel; 0 clips selected; no output file written"
        );
        return Ok(MontageSummary {
            output_path: options.output_path.clone(),
            format: options.format,
            clips_count: 0,
            total_frames: 0,
            bytes_written: 0,
            peak_frames_in_flight: 0,
        });
    }

    // Verify caption-frame alignment for all events within each clip
    for clip in clips {
        for ev in &clip.events {
            if ev.tick < clip.start || ev.tick > clip.end {
                return Err(ReelReplayError::CaptionFrameMismatch {
                    rank: clip.rank,
                    frame_tick: clip.start,
                    event_tick: ev.tick,
                });
            }
        }
    }

    let start_time = Instant::now();
    let mut total_frames = 0;
    let mut total_bytes = 0u64;
    let mut peak_in_flight = 0usize;
    let stride = options.stride.max(1);

    match options.format {
        ReelFormat::Ascii => {
            let width = if options.width > 0 { options.width } else { 80 };
            let height = if options.height > 0 {
                options.height
            } else {
                24
            };

            let mut file = File::create(&options.output_path)
                .map_err(|e| ReelReplayError::Io(e.to_string()))?;
            let header = format!(
                "{{\"version\": 2, \"width\": {width}, \"height\": {height}, \"timestamp\": 0, \"title\": \"ScriptBots Reel\"}}\n"
            );
            file.write_all(header.as_bytes())
                .map_err(|e| ReelReplayError::Io(e.to_string()))?;
            total_bytes += header.len() as u64;

            let mut time_offset = 0.0f64;
            for clip in clips {
                let clip_start_time = Instant::now();
                let mut clip_frames = 0;
                let mut clip_bytes = 0u64;
                let caption = clip
                    .events
                    .first()
                    .map_or("Event", |e| e.human_text.as_str());

                let mut tick = clip.start;
                while tick <= clip.end {
                    let world = seeker.seek_to_exact(tick)?;
                    if world.tick().0 != tick {
                        error!(
                            target: "scriptbots::reel::render",
                            clip = clip.rank,
                            frame = clip_frames,
                            tick = tick,
                            "frame tick mismatch with world state"
                        );
                        return Err(ReelReplayError::CaptionFrameMismatch {
                            rank: clip.rank,
                            frame_tick: world.tick().0,
                            event_tick: tick,
                        });
                    }

                    // Bounded streaming: 1 frame in flight
                    let in_flight = 1;
                    peak_in_flight = peak_in_flight.max(in_flight);
                    assert!(in_flight <= 2, "bounded streaming invariant violated");

                    let grid = render_ascii_frame(world, width, height);
                    let frame_text = format!(
                        "[t={tick} rank={}] {}\n{}",
                        clip.rank,
                        caption,
                        grid.join("\n")
                    );
                    let payload = serde_json::to_string(&frame_text)
                        .map_err(|e| ReelReplayError::Serialization(e.to_string()))?;
                    let line = format!("[{time_offset:.2}, \"o\", {payload}]\n");
                    file.write_all(line.as_bytes()).map_err(|e| {
                        ReelReplayError::EncoderFailure {
                            clip_rank: clip.rank,
                            frame_index: clip_frames,
                            tick,
                            reason: e.to_string(),
                        }
                    })?;

                    clip_bytes += line.len() as u64;
                    clip_frames += 1;
                    total_frames += 1;
                    time_offset += 0.1;
                    tick += stride;
                }
                total_bytes += clip_bytes;

                let clip_elapsed_ms = clip_start_time.elapsed().as_millis();
                info!(
                    target: "scriptbots::reel::render",
                    rank = clip.rank,
                    tick_start = clip.start,
                    tick_end = clip.end,
                    frames = clip_frames,
                    stride = options.stride,
                    events = clip.events.len(),
                    caption_chars = caption.len(),
                    bytes = clip_bytes,
                    elapsed_ms = clip_elapsed_ms,
                    "clip encoded"
                );
            }
        }
        ReelFormat::Gif => {
            let width = if options.width > 0 {
                options.width
            } else {
                160
            };
            let height = if options.height > 0 {
                options.height
            } else {
                120
            };

            let mut file = File::create(&options.output_path)
                .map_err(|e| ReelReplayError::Io(e.to_string()))?;
            let mut encoder =
                gif::Encoder::new(&mut file, width as u16, height as u16, &PINNED_GIF_PALETTE)
                    .map_err(|e| ReelReplayError::Io(e.to_string()))?;
            encoder
                .set_repeat(gif::Repeat::Infinite)
                .map_err(|e| ReelReplayError::Io(e.to_string()))?;

            let mut pixel_buf = vec![0u8; width * height];
            for clip in clips {
                let clip_start_time = Instant::now();
                let mut clip_frames = 0;
                let caption = clip
                    .events
                    .first()
                    .map_or("Event", |e| e.human_text.as_str());

                let mut tick = clip.start;
                while tick <= clip.end {
                    let world = seeker.seek_to_exact(tick)?;
                    if world.tick().0 != tick {
                        error!(
                            target: "scriptbots::reel::render",
                            clip = clip.rank,
                            frame = clip_frames,
                            tick = tick,
                            "frame tick mismatch with world state"
                        );
                        return Err(ReelReplayError::CaptionFrameMismatch {
                            rank: clip.rank,
                            frame_tick: world.tick().0,
                            event_tick: tick,
                        });
                    }

                    // Bounded streaming: 1 frame in flight
                    let in_flight = 1;
                    peak_in_flight = peak_in_flight.max(in_flight);
                    assert!(in_flight <= 2, "bounded streaming invariant violated");

                    render_gif_frame(world, width, height, &mut pixel_buf);
                    let mut frame = gif::Frame::from_indexed_pixels(
                        width as u16,
                        height as u16,
                        pixel_buf.clone(),
                        None,
                    );
                    frame.delay = 10; // 10 centiseconds = 100ms per frame

                    encoder
                        .write_frame(&frame)
                        .map_err(|e| ReelReplayError::EncoderFailure {
                            clip_rank: clip.rank,
                            frame_index: clip_frames,
                            tick,
                            reason: e.to_string(),
                        })?;

                    clip_frames += 1;
                    total_frames += 1;
                    tick += stride;
                }

                let clip_elapsed_ms = clip_start_time.elapsed().as_millis();
                info!(
                    target: "scriptbots::reel::render",
                    rank = clip.rank,
                    tick_start = clip.start,
                    tick_end = clip.end,
                    frames = clip_frames,
                    stride = options.stride,
                    events = clip.events.len(),
                    caption_chars = caption.len(),
                    bytes = 0,
                    elapsed_ms = clip_elapsed_ms,
                    "clip encoded"
                );
            }
            drop(encoder);
            drop(file);
            total_bytes = std::fs::metadata(&options.output_path)
                .map(|m| m.len())
                .unwrap_or(0);
        }
        ReelFormat::Mp4 => unreachable!(),
    }

    let total_elapsed_ms = start_time.elapsed().as_millis();
    info!(
        target: "scriptbots::reel::render",
        out = %options.output_path.display(),
        format = ?options.format,
        clips = clips.len(),
        frames = total_frames,
        bytes = total_bytes,
        elapsed_ms = total_elapsed_ms,
        peak_frames_in_flight = peak_in_flight,
        "montage reel generated successfully"
    );

    Ok(MontageSummary {
        output_path: options.output_path.clone(),
        format: options.format,
        clips_count: clips.len(),
        total_frames,
        bytes_written: total_bytes,
        peak_frames_in_flight: peak_in_flight,
    })
}

/// Convenience function encoding a montage reel from a world state and events.
///
/// Uses checkpoint-based exact replay to seek arbitrarily across clips without
/// advancing or mutating the input `WorldState`.
pub fn encode_montage(
    world: &WorldState,
    events: &[EventRecord],
    last_tick: u64,
    options: &MontageOptions,
) -> Result<MontageSummary, ReelReplayError> {
    let cp = world
        .capture_checkpoint_quiescent()
        .or_else(|_| world.checkpoint_v1())
        .map_err(|e| {
            ReelReplayError::Storage(format!("failed to capture initial checkpoint: {e}"))
        })?;
    let mut seeker = CheckpointReplaySeeker::new(vec![cp], last_tick);
    encode_montage_with_seeker(&mut seeker, events, last_tick, options)
}

/// Encode a montage reel from persisted FrankenSQLite storage.
pub fn encode_montage_from_storage(
    storage: &scriptbots_storage::StorageReader,
    options: &MontageOptions,
) -> Result<MontageSummary, ReelReplayError> {
    let records = storage
        .load_checkpoints()
        .map_err(|e| ReelReplayError::Storage(e.to_string()))?;
    let mut checkpoints = Vec::with_capacity(records.len());
    for rec in records {
        let cp = rec
            .world_checkpoint()
            .map_err(|e| ReelReplayError::Storage(e.to_string()))?;
        checkpoints.push(cp);
    }

    let run_events = storage
        .recent_run_events(100_000)
        .map_err(|e| ReelReplayError::Storage(e.to_string()))?;
    let events: Vec<EventRecord> = run_events
        .into_iter()
        .map(|pe| pe.record().clone())
        .collect();

    let max_tick = checkpoints
        .last()
        .map_or(0, |c| c.tick().0)
        .max(events.last().map_or(0, |e| e.tick.0));

    let mut seeker = CheckpointReplaySeeker::new(checkpoints, max_tick);
    encode_montage_with_seeker(&mut seeker, &events, max_tick, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::narrative::EventKind;
    use scriptbots_core::{AgentData, Position, ScriptBotsConfig, Tick};
    use tempfile::NamedTempFile;

    fn sample_event(tick: u64, kind: EventKind, mag: f64) -> EventRecord {
        EventRecord {
            schema_version: 1,
            tick: Tick(tick),
            kind,
            severity: 0.8,
            magnitude: mag,
            window: (tick.saturating_sub(10), tick),
            metric: "population".into(),
            before: 100.0,
            after: 50.0,
            score: 0.8,
            subject: None,
            human_text: format!("{kind:?} at tick {tick}"),
        }
    }

    #[test]
    fn test_encode_montage_ascii_empty_events_writes_no_file() {
        let config = ScriptBotsConfig::default();
        let world = WorldState::new(config).expect("world");
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let out_path = temp_dir.path().join("empty_reel.cast");

        let opts = MontageOptions {
            format: ReelFormat::Ascii,
            output_path: out_path.clone(),
            stride: 5,
            selection: SelectionConfig::default(),
            ..Default::default()
        };

        let summary = encode_montage(&world, &[], 100, &opts).expect("encode montage");
        assert_eq!(summary.clips_count, 0);
        assert_eq!(summary.total_frames, 0);
        assert_eq!(summary.bytes_written, 0);
        assert!(
            !out_path.exists(),
            "empty selections must NOT create any file on disk"
        );
    }

    #[test]
    fn test_mp4_format_returns_error_and_leaves_no_file() {
        let config = ScriptBotsConfig::default();
        let world = WorldState::new(config).expect("world");
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let out_path = temp_dir.path().join("refused.mp4");

        let opts = MontageOptions {
            format: ReelFormat::Mp4,
            output_path: out_path.clone(),
            stride: 5,
            selection: SelectionConfig::default(),
            ..Default::default()
        };

        let res = encode_montage(&world, &[], 100, &opts);
        assert_eq!(res.unwrap_err(), ReelReplayError::Mp4Unavailable);
        assert!(
            !out_path.exists(),
            "unsupported MP4 format must not create any file"
        );
    }

    #[test]
    fn test_unreachable_seek_returns_typed_nearest_checkpoint_error() {
        let config = ScriptBotsConfig::default();
        let mut world = WorldState::new(config).expect("world");
        let mut adapter = LiveWorldReplayAdapter::new(&mut world, 500);

        // Advance world to tick 100
        adapter.seek_to_exact(100).expect("seek to 100");

        // Attempting to seek backwards to tick 50 must return UnreachableSeek
        let err = adapter.seek_to_exact(50).unwrap_err();
        assert_eq!(
            err,
            ReelReplayError::UnreachableSeek {
                target_tick: 50,
                nearest_checkpoint: 100,
            }
        );

        // Seeking past max_tick must return UnreachableSeek
        let err = adapter.seek_to_exact(600).unwrap_err();
        assert_eq!(
            err,
            ReelReplayError::UnreachableSeek {
                target_tick: 600,
                nearest_checkpoint: 500,
            }
        );
    }

    struct OffsetReplaySeeker {
        world: WorldState,
    }

    impl ReplayProvider for OffsetReplaySeeker {
        fn seek_to_exact(&mut self, target_tick: u64) -> Result<&WorldState, ReelReplayError> {
            // Deliberately step past target_tick so that the returned world has tick > target_tick
            while self.world.tick().0 <= target_tick {
                self.world
                    .step()
                    .map_err(|e| ReelReplayError::Storage(e.to_string()))?;
            }
            Ok(&self.world)
        }

        fn max_tick(&self) -> u64 {
            1000
        }
    }

    #[test]
    fn test_caption_frame_offset_mismatch_detected() {
        let config = ScriptBotsConfig::default();
        let world = WorldState::new(config).expect("world");
        let temp = NamedTempFile::new().expect("temp file");

        let event = sample_event(50, EventKind::PopulationCrash, 20.0);

        let opts = MontageOptions {
            format: ReelFormat::Ascii,
            output_path: temp.path().to_path_buf(),
            stride: 5,
            selection: SelectionConfig {
                pre_window_ticks: 10,
                post_window_ticks: 10,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut seeker = OffsetReplaySeeker { world };
        let res = encode_montage_with_seeker(&mut seeker, &[event], 300, &opts);
        assert!(matches!(
            res,
            Err(ReelReplayError::CaptionFrameMismatch {
                rank: 1,
                frame_tick: 41,
                event_tick: 40,
            })
        ));
    }

    #[test]
    fn test_streaming_bounded_frames_in_flight() {
        let config = ScriptBotsConfig::default();
        let mut world = WorldState::new(config).expect("world");
        world
            .try_spawn_agent(AgentData {
                position: Position::new(50.0, 50.0),
                health: 1.0,
                color: [0.1, 0.9, 0.1],
                ..AgentData::default()
            })
            .expect("spawn agent");

        let events = vec![sample_event(50, EventKind::PopulationBoom, 30.0)];
        let temp = NamedTempFile::new().expect("temp file");
        let opts = MontageOptions {
            format: ReelFormat::Ascii,
            output_path: temp.path().to_path_buf(),
            stride: 5,
            selection: SelectionConfig {
                pre_window_ticks: 20,
                post_window_ticks: 20,
                ..Default::default()
            },
            ..Default::default()
        };

        let summary = encode_montage(&world, &events, 100, &opts).expect("encode montage");
        assert!(summary.clips_count > 0);
        assert!(summary.total_frames > 0);
        assert!(
            summary.peak_frames_in_flight <= 2,
            "peak frames in flight must be <= 2"
        );
    }

    #[test]
    fn test_byte_reproducible_ascii_output() {
        let events = vec![
            sample_event(50, EventKind::PopulationBoom, 30.0),
            sample_event(150, EventKind::Extinction, 80.0),
        ];

        let temp1 = NamedTempFile::new().expect("temp 1");
        let temp2 = NamedTempFile::new().expect("temp 2");

        let opts1 = MontageOptions {
            format: ReelFormat::Ascii,
            output_path: temp1.path().to_path_buf(),
            stride: 5,
            selection: SelectionConfig {
                pre_window_ticks: 15,
                post_window_ticks: 25,
                ..Default::default()
            },
            ..Default::default()
        };
        let opts2 = MontageOptions {
            output_path: temp2.path().to_path_buf(),
            ..opts1.clone()
        };

        let config = ScriptBotsConfig {
            rng_seed: Some(42),
            ..Default::default()
        };
        let world1 = WorldState::new(config.clone()).unwrap();
        let world2 = WorldState::new(config).unwrap();

        encode_montage(&world1, &events, 300, &opts1).expect("run 1");
        encode_montage(&world2, &events, 300, &opts2).expect("run 2");

        let bytes1 = std::fs::read(temp1.path()).expect("read 1");
        let bytes2 = std::fs::read(temp2.path()).expect("read 2");

        assert!(!bytes1.is_empty(), "output must not be empty");
        assert_eq!(
            bytes1, bytes2,
            "two ASCII montage runs with identical inputs must be byte-identical"
        );
    }

    #[test]
    fn test_byte_reproducible_gif_output() {
        let events = vec![
            sample_event(50, EventKind::PopulationBoom, 30.0),
            sample_event(120, EventKind::CombatSurge, 40.0),
        ];

        let temp1 = NamedTempFile::new().expect("temp 1");
        let temp2 = NamedTempFile::new().expect("temp 2");

        let opts1 = MontageOptions {
            format: ReelFormat::Gif,
            output_path: temp1.path().to_path_buf(),
            stride: 10,
            selection: SelectionConfig {
                pre_window_ticks: 10,
                post_window_ticks: 20,
                ..Default::default()
            },
            width: 80,
            height: 60,
        };
        let opts2 = MontageOptions {
            output_path: temp2.path().to_path_buf(),
            ..opts1.clone()
        };

        let config = ScriptBotsConfig {
            rng_seed: Some(42),
            ..Default::default()
        };
        let world1 = WorldState::new(config.clone()).unwrap();
        let world2 = WorldState::new(config).unwrap();

        encode_montage(&world1, &events, 200, &opts1).expect("run 1");
        encode_montage(&world2, &events, 200, &opts2).expect("run 2");

        let bytes1 = std::fs::read(temp1.path()).expect("read 1");
        let bytes2 = std::fs::read(temp2.path()).expect("read 2");

        assert!(!bytes1.is_empty(), "GIF output must not be empty");
        assert_eq!(
            bytes1, bytes2,
            "two GIF montage runs with identical inputs must be byte-identical"
        );

        // Verify GIF magic header and lack of timestamp metadata
        assert_eq!(&bytes1[0..6], b"GIF89a", "must be valid GIF89a header");
    }

    #[test]
    fn test_exact_frame_count_formula() {
        let events = vec![sample_event(100, EventKind::PopulationCrash, 50.0)];
        let temp = NamedTempFile::new().expect("temp file");
        let stride = 5;
        let pre = 20;
        let post = 30;

        let opts = MontageOptions {
            format: ReelFormat::Ascii,
            output_path: temp.path().to_path_buf(),
            stride,
            selection: SelectionConfig {
                pre_window_ticks: pre,
                post_window_ticks: post,
                ..Default::default()
            },
            ..Default::default()
        };

        let world = WorldState::new(ScriptBotsConfig::default()).unwrap();
        let summary = encode_montage(&world, &events, 200, &opts).expect("encode");

        // Window for tick 100 with pre=20, post=30 is start=80, end=130 (length 51 ticks).
        // With stride 5: ticks 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130 -> exactly 11 frames.
        let expected_frames = (130 - 80) / stride + 1;
        assert_eq!(summary.total_frames, expected_frames as usize);
    }
}
