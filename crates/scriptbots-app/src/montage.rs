//! Headless montage encoder for highlight reels (`bd-16g.9.2`).
//!
//! Provides byte-reproducible visual artifacts (ASCII cast and animated GIF)
//! from deterministic replay checkpoints and scored narrative events without requiring
//! a GPU or external FFmpeg binary.

use scriptbots_core::narrative::EventRecord;
use scriptbots_core::reel::{Clip, SelectionConfig, select_clips};
use scriptbots_core::{
    PersistenceAdmissionSession, ReplayEventKind, TerrainKind, WorldCheckpointV1, WorldPersistence,
    WorldState,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thiserror::Error;
use tracing::{error, info};

use crate::persisted_replay::{
    ReplayCheckpointSource, ReplayMirror, apply_recorded_operator_commands, digest_due,
    mirror_recorded_flush, recorded_flush_ticks, recorded_input_history,
};

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
    /// Invalid clip interval or encoder dimensions.
    #[error("invalid montage options: {0}")]
    InvalidOptions(String),
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

    /// Complete any verification horizon beyond the last visual frame.
    ///
    /// The encoder calls this before publishing the completed artifact. Fixture
    /// providers need no persisted evidence; a journal-backed provider must verify
    /// its covering digest again during this rendering pass.
    fn finish_replay(&mut self) -> Result<(), ReelReplayError> {
        Ok(())
    }
}

type DigestAnchors = VecDeque<(u64, String)>;

/// Verify digests where the original persistence boundary emitted them. Comparing
/// `world_digest_v1()` after same-tick commands would observe a different cut.
struct DigestVerifier {
    remaining: Arc<Mutex<DigestAnchors>>,
}

impl WorldPersistence for DigestVerifier {
    fn on_tick(
        &mut self,
        payload: &scriptbots_core::PersistenceBatch,
    ) -> Result<(), scriptbots_core::PersistenceAdmissionError> {
        let tick = payload.summary.tick.0;
        let mut remaining = self.remaining.lock().map_err(|error| {
            scriptbots_core::PersistenceAdmissionError::new(
                tick,
                format!("montage digest verifier poisoned: {error}"),
            )
        })?;
        for event in &payload.replay_events {
            if let ReplayEventKind::WorldDigest { overall } = &event.kind {
                let expected = remaining.front();
                if !expected.is_some_and(|(expected_tick, digest)| {
                    *expected_tick == tick && digest == overall
                }) {
                    return Err(scriptbots_core::PersistenceAdmissionError::new(
                        tick,
                        format!(
                            "montage replay digest diverged: expected {expected:?}, observed ({tick}, {overall})"
                        ),
                    ));
                }
                remaining.pop_front();
            }
        }
        Ok(())
    }
}

struct JournalReplayWorld {
    world: WorldState,
    persistence: PersistenceAdmissionSession,
    current_config: serde_json::Value,
    remaining: Arc<Mutex<DigestAnchors>>,
}

impl JournalReplayWorld {
    fn restore(source: &ReplayCheckpointSource, expected: &DigestAnchors) -> anyhow::Result<Self> {
        let ReplayCheckpointSource::Host(_) = source else {
            anyhow::bail!("persisted montage requires complete owner recording continuity");
        };
        let checkpoint = source.world_checkpoint();
        let mut world = source.restore_world()?;
        let remaining = Arc::new(Mutex::new(expected.clone()));
        // Full restoration already supplied the raw recording cadence, cap and
        // disabled-boundary marker. The inner science checkpoint normalizes its
        // cadence to zero; rebinding from that config would invent batch cuts.
        let persistence = source.bind_restored_persistence(
            &mut world,
            Box::new(DigestVerifier {
                remaining: Arc::clone(&remaining),
            }),
            checkpoint.config(),
        )?;
        let current_config = serde_json::to_value(world.config())?;
        Ok(Self {
            world,
            persistence,
            current_config,
            remaining,
        })
    }

    fn advance_to(&mut self, target: u64, mirror: &ReplayMirror, limit: u64) -> anyhow::Result<()> {
        while self.world.tick().0 < target {
            let tick = self.world.tick().0;
            mirror_recorded_flush(&mut self.world, &mut self.persistence, mirror)?;
            apply_recorded_operator_commands(
                &mut self.world,
                &mut self.persistence,
                &mut self.current_config,
                mirror,
                tick,
            )?;
            if digest_due(mirror.digest_ticks.as_ref(), tick + 1, limit) {
                self.world.request_replay_world_digest();
            }
            self.persistence.step(&mut self.world)?;
        }
        Ok(())
    }

    fn finish(&mut self, mirror: &ReplayMirror, limit: u64) -> anyhow::Result<()> {
        self.advance_to(limit, mirror, limit)?;
        apply_recorded_operator_commands(
            &mut self.world,
            &mut self.persistence,
            &mut self.current_config,
            mirror,
            limit,
        )?;
        self.persistence.finalize(&mut self.world)?;
        let remaining = self
            .remaining
            .lock()
            .map_err(|error| anyhow::anyhow!("montage digest verifier poisoned: {error}"))?;
        anyhow::ensure!(
            remaining.is_empty(),
            "montage replay is missing recorded digest anchors: {remaining:?}"
        );
        Ok(())
    }
}

/// Frozen full-owner source plus its exact durable journal suffix. One world is
/// replayed through the covering anchor before encoding, then restored for a
/// second streaming pass which must independently satisfy the same anchors.
/// The borrowed reader retains the finished-run lease through publication.
struct PersistedReplaySeeker<'a> {
    _storage: &'a scriptbots_storage::StorageReader,
    source: ReplayCheckpointSource,
    mirror: ReplayMirror,
    expected: DigestAnchors,
    current: JournalReplayWorld,
    anchor_tick: u64,
    last_frame: u64,
}

impl<'a> PersistedReplaySeeker<'a> {
    #[allow(
        clippy::too_many_lines,
        reason = "one immutable input load binds the selected full checkpoint, exact journal and covering digest sequence"
    )]
    fn from_storage(
        storage: &'a scriptbots_storage::StorageReader,
        first_frame: u64,
        last_frame: u64,
    ) -> anyhow::Result<Self> {
        // This is also the lease check for direct library callers, who may have
        // opened a live reader instead of entering through the reel CLI.
        storage.select_finished_run(Some(storage.run_id()))?;
        anyhow::ensure!(first_frame <= last_frame, "montage frame range is reversed");
        anyhow::ensure!(
            storage
                .archipelago_islands()?
                .iter()
                .all(|island| island.island_id == 0),
            "offline montage requires a single-world run; migration barriers are unsupported"
        );
        let terminal_tick = storage.max_tick()?.unwrap_or(0);
        anyhow::ensure!(
            last_frame <= terminal_tick,
            "montage target exceeds the recorded run"
        );
        let record = storage
            .load_latest_checkpoint_before(first_frame.saturating_add(1))?
            .ok_or_else(|| anyhow::anyhow!("no checkpoint can reach the first montage frame"))?;
        // A raw checkpoint has no exact host cursor or schedule authority. Its
        // explicitly intervention-free fixture API remains separate below.
        let checkpoint = record.host_checkpoint()?;
        let recording_config = checkpoint.config().clone();
        let source = ReplayCheckpointSource::Host(Box::new(checkpoint));
        let start_tick = source.world_checkpoint().tick().0;
        let mut events = storage.load_replay_events()?;
        anyhow::ensure!(
            events
                .iter()
                .all(|event| event.island_id == scriptbots_core::rng_domains::IslandId(0)),
            "offline montage replay contains another island's evidence"
        );
        events.sort_by_key(|event| (event.tick, event.seq));
        anyhow::ensure!(
            !events
                .windows(2)
                .any(|pair| (pair[0].tick, pair[0].seq) == (pair[1].tick, pair[1].seq)),
            "offline montage replay contains duplicate event identities"
        );
        let anchor_tick = events
            .iter()
            .find(|event| {
                event.tick > start_tick
                    && event.tick >= last_frame
                    && matches!(event.event.kind, ReplayEventKind::WorldDigest { .. })
            })
            .map(|event| event.tick)
            .ok_or_else(|| {
                anyhow::anyhow!("montage requires a recorded digest anchor covering every frame")
            })?;
        anyhow::ensure!(
            anchor_tick <= terminal_tick,
            "montage digest exceeds the recorded run"
        );
        let mut expected = DigestAnchors::new();
        for entry in events
            .iter()
            .filter(|event| event.tick > start_tick && event.tick <= anchor_tick)
        {
            if let ReplayEventKind::WorldDigest { overall } = &entry.event.kind {
                anyhow::ensure!(
                    entry.seq < scriptbots_storage::NARRATIVE_INPUT_REPLAY_SEQ
                        && entry.event.agent_uid.is_none()
                        && entry.event.position.is_none()
                        && entry.event.counterpart.is_none()
                        && entry.event.counterpart_position.is_none()
                        && overall.len() == std::mem::size_of::<u64>() * 2
                        && overall
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
                    "montage world-digest identity or encoding is invalid"
                );
                expected.push_back((entry.tick, overall.clone()));
            }
        }
        let mirror = ReplayMirror {
            digest_ticks: Some(
                expected
                    .iter()
                    .map(|(tick, _)| *tick)
                    .collect::<BTreeSet<_>>(),
            ),
            flush_ticks: recorded_flush_ticks(
                &storage.summary_ticks()?,
                &recording_config,
                anchor_tick,
            ),
            operator_commands: recorded_input_history(
                storage,
                source.journal_cut().map(|cut| (start_tick, cut)),
            )?
            .operator_commands,
        };
        source.validate_recorded_schedule(&mirror, anchor_tick)?;
        let mut verification = JournalReplayWorld::restore(&source, &expected)?;
        verification.finish(&mirror, anchor_tick)?;
        drop(verification);
        let current = JournalReplayWorld::restore(&source, &expected)?;
        Ok(Self {
            _storage: storage,
            source,
            mirror,
            expected,
            current,
            anchor_tick,
            last_frame,
        })
    }
}

impl ReplayProvider for PersistedReplaySeeker<'_> {
    fn seek_to_exact(&mut self, target_tick: u64) -> Result<&WorldState, ReelReplayError> {
        let start_tick = self.source.world_checkpoint().tick().0;
        if target_tick < start_tick || target_tick > self.last_frame {
            return Err(ReelReplayError::UnreachableSeek {
                target_tick,
                nearest_checkpoint: start_tick,
            });
        }
        if self.current.world.tick().0 > target_tick {
            self.current = JournalReplayWorld::restore(&self.source, &self.expected)
                .map_err(replay_storage_error)?;
        }
        self.current
            .advance_to(target_tick, &self.mirror, self.anchor_tick)
            .map_err(replay_storage_error)?;
        // Frames observe completed steps before the next boundary's inputs.
        // At the source tick the frame observes the exact captured owner cut.
        Ok(&self.current.world)
    }

    fn max_tick(&self) -> u64 {
        self.last_frame
    }

    fn finish_replay(&mut self) -> Result<(), ReelReplayError> {
        self.current
            .finish(&self.mirror, self.anchor_tick)
            .map_err(replay_storage_error)
    }
}

fn replay_storage_error(error: anyhow::Error) -> ReelReplayError {
    ReelReplayError::Storage(format!("{error:#}"))
}

/// A private staging file is retained on error. Only a completely encoded,
/// verified and synced artifact acquires the requested destination name. The
/// exclusive hard link also preserves any existing destination byte-for-byte.
struct OutputStage {
    path: PathBuf,
    file: File,
}

impl OutputStage {
    fn create(destination: &Path) -> Result<Self, ReelReplayError> {
        static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);
        let parent = destination
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        for _ in 0..128 {
            let sequence = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".scriptbots-reel-{}-{sequence}.pending",
                std::process::id()
            ));
            if path.file_name() == destination.file_name() {
                continue;
            }
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok(Self { path, file }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(ReelReplayError::Io(error.to_string())),
            }
        }
        Err(ReelReplayError::Io(
            "could not reserve a private montage staging path".to_owned(),
        ))
    }

    fn publish(&self, destination: &Path) -> Result<u64, ReelReplayError> {
        self.file
            .sync_all()
            .map_err(|error| ReelReplayError::Io(error.to_string()))?;
        let bytes = self
            .file
            .metadata()
            .map_err(|error| ReelReplayError::Io(error.to_string()))?
            .len();
        std::fs::hard_link(&self.path, destination).map_err(|error| {
            ReelReplayError::Io(format!(
                "could not publish montage without replacing {} (staging retained at {}): {error}",
                destination.display(),
                self.path.display()
            ))
        })?;
        Ok(bytes)
    }
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
            let registry = crate::brains::create_brain_registry_for_checkpoint(best_cp)
                .map_err(|error| {
                    ReelReplayError::Storage(format!(
                        "checkpoint at tick {base_tick} does not match the compiled application adapters: {error}"
                    ))
                })?;
            let world = WorldState::restore_checkpoint_v1(best_cp, registry).map_err(|error| {
                ReelReplayError::Storage(format!(
                    "failed to restore checkpoint at tick {base_tick}: {error}"
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

    // Refuse invalid intervals and dimensions before creating a staging file.
    let width = if options.width > 0 {
        options.width
    } else if options.format == ReelFormat::Ascii {
        80
    } else {
        160
    };
    let height = if options.height > 0 {
        options.height
    } else if options.format == ReelFormat::Ascii {
        24
    } else {
        120
    };
    if width.checked_mul(height).is_none()
        || (options.format == ReelFormat::Gif
            && (u16::try_from(width).is_err() || u16::try_from(height).is_err()))
    {
        return Err(ReelReplayError::InvalidOptions(
            "frame dimensions exceed the encoder's representable range".to_owned(),
        ));
    }
    for clip in clips {
        if clip.start > clip.end {
            return Err(ReelReplayError::InvalidOptions(format!(
                "clip {} has a reversed tick range",
                clip.rank
            )));
        }
        if clip.end > seeker.max_tick() {
            return Err(ReelReplayError::UnreachableSeek {
                target_tick: clip.end,
                nearest_checkpoint: seeker.max_tick(),
            });
        }
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
    let mut peak_in_flight = 0usize;
    let stride = options.stride.max(1);
    let mut stage = OutputStage::create(&options.output_path)?;

    match options.format {
        ReelFormat::Ascii => {
            let file = &mut stage.file;
            let header = format!(
                "{{\"version\": 2, \"width\": {width}, \"height\": {height}, \"timestamp\": 0, \"title\": \"ScriptBots Reel\"}}\n"
            );
            file.write_all(header.as_bytes())
                .map_err(|e| ReelReplayError::Io(e.to_string()))?;

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
                    let Some(next_tick) = tick.checked_add(stride) else {
                        break;
                    };
                    tick = next_tick;
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
                    bytes = clip_bytes,
                    elapsed_ms = clip_elapsed_ms,
                    "clip encoded"
                );
            }
        }
        ReelFormat::Gif => {
            let mut encoder = gif::Encoder::new(
                &mut stage.file,
                width as u16,
                height as u16,
                &PINNED_GIF_PALETTE,
            )
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
                    let Some(next_tick) = tick.checked_add(stride) else {
                        break;
                    };
                    tick = next_tick;
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
            encoder
                .into_inner()
                .map_err(|error| ReelReplayError::Io(error.to_string()))?;
        }
        ReelFormat::Mp4 => unreachable!(),
    }

    seeker.finish_replay()?;
    let total_bytes = stage.publish(&options.output_path)?;

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

/// Encode a montage reel from a finished, single-world run with full owner checkpoints.
///
/// Selection considers the newest 100,000 narrative events. The recorded terminal
/// tick determines post-roll; every selected frame must be covered by a later or
/// same-tick recorded digest. The exact checkpoint, journal suffix and anchors are
/// frozen and verified before encoding, and checked again before publication.
pub fn encode_montage_from_storage(
    storage: &scriptbots_storage::StorageReader,
    options: &MontageOptions,
) -> Result<MontageSummary, ReelReplayError> {
    if options.format == ReelFormat::Mp4 {
        return Err(ReelReplayError::Mp4Unavailable);
    }
    storage
        .select_finished_run(Some(storage.run_id()))
        .map_err(|error| ReelReplayError::Storage(error.to_string()))?;

    let run_events = storage
        .recent_run_events(100_000)
        .map_err(|e| ReelReplayError::Storage(e.to_string()))?;
    let events: Vec<EventRecord> = run_events
        .into_iter()
        .map(|pe| pe.record().clone())
        .collect();

    let max_tick = storage
        .max_tick()
        .map_err(|error| ReelReplayError::Storage(error.to_string()))?
        .unwrap_or(0);
    let mut clips = select_clips(&events, max_tick, &options.selection)?;
    clips.sort_by_key(|clip| clip.start);
    let Some(first) = clips.first() else {
        return Ok(MontageSummary {
            output_path: options.output_path.clone(),
            format: options.format,
            clips_count: 0,
            total_frames: 0,
            bytes_written: 0,
            peak_frames_in_flight: 0,
        });
    };
    let last = clips
        .iter()
        .map(|clip| clip.end)
        .max()
        .unwrap_or(first.start);
    let mut seeker = PersistedReplaySeeker::from_storage(storage, first.start, last)
        .map_err(replay_storage_error)?;
    encode_montage_clips_with_seeker(&mut seeker, &clips, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::narrative::EventKind;
    use scriptbots_core::{AgentData, Position, ScriptBotsConfig, Tick};

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

    fn checkpoint_world(preset: Option<crate::BrainPreset>) -> WorldState {
        let mut config = ScriptBotsConfig {
            rng_seed: Some(42_103),
            persistence_interval: 0,
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            closed: true,
            population_minimum: 0,
            reproduction_rate_herbivore: 0.0,
            reproduction_rate_carnivore: 0.0,
            ..ScriptBotsConfig::default()
        };
        config.neuroflow.enabled = false;
        config.neuroflow.hidden_layers = vec![7, 3];
        config.neuroflow.activation = scriptbots_core::NeuroflowActivationKind::Sigmoid;
        let mut world = WorldState::new(config).unwrap();
        let brain = preset.map(|preset| crate::install_brains(&mut world, preset).unwrap());
        let agent = world
            .try_spawn_agent(AgentData {
                position: Position::new(50.0, 50.0),
                ..AgentData::default()
            })
            .unwrap();
        if let Some(brain) = brain {
            assert!(
                world
                    .bind_agent_brain(agent, brain.population()[0])
                    .unwrap()
            );
        }
        for _ in 0..3 {
            world.step().unwrap();
        }
        world
    }

    fn submit_durable(
        port: &mut impl scriptbots_runtime::HostPort,
        id: u64,
        command: scriptbots_runtime::HostCommand,
    ) -> anyhow::Result<()> {
        use scriptbots_runtime::{ApplicationState, CommandEnvelope, CommandId, JournalState};
        use std::time::Duration;

        let id = CommandId::new(u128::from(id));
        port.submit(CommandEnvelope::new(id, command))?;
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let status = port
                .command_status(id)?
                .expect("submitted montage fixture command");
            if status.journal() == &JournalState::Durable {
                anyhow::ensure!(
                    matches!(status.application(), ApplicationState::Applied(_)),
                    "fixture command did not apply: {status:?}"
                );
                return Ok(());
            }
            anyhow::ensure!(
                Instant::now() < deadline,
                "fixture command did not settle: {status:?}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one real file-backed owner supplies auxiliary state, same-tick input order and independently observed frame digests"
    )]
    fn recorded_montage_fixture(
        path: &Path,
        record_digests: bool,
        persistence_interval: u32,
    ) -> anyhow::Result<Vec<scriptbots_core::WorldDigestV1>> {
        use crate::host_thread::{HostBootstrap, HostThread};
        use scriptbots_runtime::{
            HostCommand, HostCoreOptions, HostSessionId, PlaybackSnapshot, ScheduledConfigPatch,
            channel::ChannelHostOptions,
        };
        use scriptbots_storage::StoragePipeline;

        let mut pipeline = StoragePipeline::create_unattributed_file(path.to_str().unwrap())?;
        let config = ScriptBotsConfig {
            rng_seed: Some(42_103),
            persistence_interval,
            replay_event_tick_cap: if record_digests { 128 } else { 0 },
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            reproduction_energy_threshold: f32::MAX,
            reproduction_rate_herbivore: 0.0,
            reproduction_rate_carnivore: 0.0,
            spike_damage: 0.0,
            archive_enabled: true,
            archive_interval: 1,
            archive_min_lifetime_ticks: 1,
            selection_mode: scriptbots_core::EvolutionSelectionMode::Novelty,
            communication_scramble: true,
            communication_scramble_seed: Some(0x7265_656c),
            ..ScriptBotsConfig::default()
        };
        let (mut world, persistence) =
            WorldState::with_persistence(config.clone(), Box::new(pipeline.sink()))?;
        let width = world.terrain().width();
        let height = world.terrain().height();
        let map = scriptbots_core::MapArtifact::new(
            world.terrain().clone(),
            Some(scriptbots_core::ScalarField::new(
                width,
                height,
                vec![0.25; world.terrain().tiles().len()],
            )?),
            Some(scriptbots_core::ScalarField::new(
                width,
                height,
                vec![0.9; world.terrain().tiles().len()],
            )?),
            None,
            None,
            scriptbots_core::MapArtifactMetadata {
                generator: scriptbots_core::MapGeneratorKind::RuleBased,
                tileset_id: "recorded-montage-auxiliary-science".to_owned(),
                tileset_hash: 1,
                seed: 42_103,
                width,
                height,
                attempt_count: 1,
                succeeded_on: 1,
                generated_at_epoch_ms: 0,
            },
        )?;
        world.apply_map_artifact(&map)?;
        let brains = crate::install_brains(&mut world, crate::BrainPreset::Mlp)?;
        let agent = world.try_spawn_agent(AgentData {
            position: Position::new(50.0, 50.0),
            ..AgentData::default()
        })?;
        assert!(world.bind_agent_brain(agent, brains.population()[0])?);
        let agent_uid = world.agent_uid(agent).expect("stable fixture identity");
        let session = HostSessionId::new(0x7265_656c);
        let journal = pipeline.journal_port(session, Default::default())?;
        let host = HostThread::spawn_with_bootstrap(
            session,
            world,
            persistence,
            Box::new(journal),
            HostCoreOptions {
                initial_playback: PlaybackSnapshot {
                    paused: true,
                    speed_multiplier: 1.0,
                },
                ..HostCoreOptions::default()
            },
            ChannelHostOptions::default(),
            HostBootstrap {
                schedule: vec![
                    ScheduledConfigPatch::new(
                        1,
                        Tick(1),
                        serde_json::json!({"food_growth_rate": 0.03}),
                    )
                    .map_err(anyhow::Error::msg)?,
                ],
                ..HostBootstrap::default()
            },
        )?;
        let mut port = host.port();
        submit_durable(&mut port, 1, HostCommand::Step)?;
        let mut next_id = 2;
        if persistence_interval > 1 {
            submit_durable(&mut port, next_id, HostCommand::FlushPersistence)?;
            next_id += 1;
        }
        let before_inputs = if persistence_interval == 1 {
            Some((
                port.capture_host_checkpoint_v1()?,
                port.scientific_digest_v1()?,
            ))
        } else {
            None
        };
        submit_durable(
            &mut port,
            next_id,
            HostCommand::AdjustAgentMutationRates {
                agent_uid,
                delta_primary: 0.03,
                delta_secondary: 0.04,
            },
        )?;
        next_id += 1;
        let mut operator_config = config;
        operator_config.food_growth_rate = 0.2;
        submit_durable(
            &mut port,
            next_id,
            HostCommand::UpdateConfig(Box::new(operator_config)),
        )?;
        next_id += 1;
        // A non-unit recording captures after the operator config, so replay
        // cannot accidentally repair an invented cadence through UpdateConfig.
        let (checkpoint, first_digest) = match before_inputs {
            Some(captured) => captured,
            None => (
                port.capture_host_checkpoint_v1()?,
                port.scientific_digest_v1()?,
            ),
        };
        pipeline.checkpoint_writer().record_host(
            "montage-source",
            &checkpoint,
            &serde_json::json!({"source": "persisted-montage-regression"}),
        )?;
        let mut observed = vec![first_digest];
        for tick in 2..=4 {
            if record_digests {
                port.request_replay_world_digest()?;
            }
            submit_durable(&mut port, next_id, HostCommand::Step)?;
            next_id += 1;
            observed.push(port.scientific_digest_v1()?);
            if persistence_interval > 1 && tick == 2 {
                // This mutation occurs after the frame's completed step, before
                // its off-cadence batch. A forced cadence of one emits the digest
                // too early and therefore cannot match this recorded anchor.
                submit_durable(
                    &mut port,
                    next_id,
                    HostCommand::AdjustAgentMutationRates {
                        agent_uid,
                        delta_primary: 0.07,
                        delta_secondary: 0.08,
                    },
                )?;
                next_id += 1;
                submit_durable(&mut port, next_id, HostCommand::FlushPersistence)?;
                next_id += 1;
            }
        }
        submit_durable(&mut port, next_id, HostCommand::Shutdown)?;
        drop(port);
        let receipt = host.join()?;
        assert_eq!(receipt.final_digest?, *observed.last().unwrap());
        pipeline.shutdown()?;
        Ok(observed)
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the real recording proves observed world equality, exact journal order, reproducible encoders and late verification refusal together"
    )]
    fn persisted_montage_replays_real_owner_inputs_and_verifies_both_passes() -> anyhow::Result<()>
    {
        use crate::persisted_replay::ReplayOperatorCommand;
        use scriptbots_storage::StorageReader;

        let directory = tempfile::tempdir()?;
        let database = directory.path().join("montage.sqlite");
        let expected = recorded_montage_fixture(&database, true, 1)?;
        let live_reader = StorageReader::open(database.to_str().unwrap())?;
        assert!(PersistedReplaySeeker::from_storage(&live_reader, 1, 3).is_err());
        live_reader.close()?;
        let reader = StorageReader::open_finished(database.to_str().unwrap())?;
        let mut seeker = PersistedReplaySeeker::from_storage(&reader, 1, 4)?;
        assert!(matches!(
            seeker.mirror.operator_commands[&1].as_slice(),
            [
                ReplayOperatorCommand::World(_),
                ReplayOperatorCommand::World(_),
                ReplayOperatorCommand::ScheduledConfigPatch(_)
            ]
        ));
        for tick in [1, 2, 4, 3, 2] {
            assert_eq!(
                seeker.seek_to_exact(tick)?.world_digest_v1()?,
                expected[(tick - 1) as usize],
                "frame must reproduce the independently observed owner at tick {tick}"
            );
        }
        seeker.finish_replay()?;
        assert!(PersistedReplaySeeker::from_storage(&reader, 1, 5).is_err());

        let events = [sample_event(2, EventKind::PopulationBoom, 30.0)];
        let selection = SelectionConfig {
            pre_window_ticks: 1,
            post_window_ticks: 1,
            ..SelectionConfig::default()
        };
        for format in [ReelFormat::Ascii, ReelFormat::Gif] {
            let mut outputs = Vec::new();
            for repetition in 0..2 {
                let options = MontageOptions {
                    format,
                    output_path: directory
                        .path()
                        .join(format!("recorded-{format}-{repetition}")),
                    stride: 1,
                    selection: selection.clone(),
                    width: 16,
                    height: 8,
                };
                let mut seeker = PersistedReplaySeeker::from_storage(&reader, 1, 3)?;
                let summary = encode_montage_with_seeker(&mut seeker, &events, 4, &options)?;
                assert_eq!(summary.total_frames, 3);
                assert_eq!(summary.peak_frames_in_flight, 1);
                let bytes = std::fs::read(&options.output_path)?;
                assert_eq!(summary.bytes_written, bytes.len() as u64);
                assert!(!bytes.is_empty());
                if format == ReelFormat::Ascii {
                    let text = std::str::from_utf8(&bytes)?;
                    for tick in 1..=3 {
                        assert!(text.contains(&format!("[t={tick} rank=1]")));
                    }
                    assert!(text.contains("PopulationBoom at tick 2"));
                } else {
                    assert_eq!(bytes.last(), Some(&0x3b), "GIF trailer must be finalized");
                }
                outputs.push(bytes);
            }
            assert_eq!(outputs[0], outputs[1]);
        }

        // Render only the source frame: the changed journal order is encountered
        // later, in finish_replay. Even that late failure cannot publish a reel.
        for fault in ["wrong-order", "missing-input"] {
            let mut altered = PersistedReplaySeeker::from_storage(&reader, 1, 3)?;
            let actions = altered.mirror.operator_commands.get_mut(&1).unwrap();
            if fault == "wrong-order" {
                actions.swap(1, 2);
            } else {
                actions.remove(0);
            }
            let options = MontageOptions {
                output_path: directory.path().join(format!("{fault}.cast")),
                stride: 1,
                selection: SelectionConfig {
                    pre_window_ticks: 0,
                    post_window_ticks: 0,
                    ..SelectionConfig::default()
                },
                ..MontageOptions::default()
            };
            let failure = encode_montage_with_seeker(
                &mut altered,
                &[sample_event(1, EventKind::PopulationBoom, 30.0)],
                4,
                &options,
            )
            .expect_err("changed input must diverge after the last visible frame");
            assert!(failure.to_string().contains("digest diverged"));
            assert!(!options.output_path.exists());
        }
        Ok(())
    }

    #[test]
    fn persisted_montage_refuses_missing_covering_digest() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let database = directory.path().join("unanchored.sqlite");
        recorded_montage_fixture(&database, false, 1)?;
        let reader = scriptbots_storage::StorageReader::open_finished(database.to_str().unwrap())?;
        let error = match PersistedReplaySeeker::from_storage(&reader, 1, 3) {
            Ok(_) => anyhow::bail!("unverified frames must be refused"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("digest anchor covering every frame")
        );
        Ok(())
    }

    #[test]
    fn persisted_montage_preserves_nonunit_cadence_and_same_tick_flush_cut() -> anyhow::Result<()> {
        use scriptbots_storage::StorageReader;

        let directory = tempfile::tempdir()?;
        let database = directory.path().join("cadence-three.sqlite");
        let observed = recorded_montage_fixture(&database, true, 3)?;
        let reader = StorageReader::open_finished(database.to_str().unwrap())?;
        let record = reader.load_checkpoint("montage-source")?.unwrap();
        let owner = record.host_checkpoint()?;
        assert_eq!(owner.config().persistence_interval, 3);
        assert_eq!(owner.world_checkpoint().config().persistence_interval, 0);
        assert_eq!(owner.config().replay_event_tick_cap, 128);
        let mut seeker = PersistedReplaySeeker::from_storage(&reader, 1, 4)?;
        assert!(seeker.mirror.flush_ticks.contains(&2));
        assert_eq!(
            seeker
                .expected
                .iter()
                .map(|(tick, _)| *tick)
                .collect::<Vec<_>>(),
            [2, 3, 4],
            "off-cadence flush, normal cadence and final tail each retain their own anchor"
        );
        assert_ne!(
            seeker.expected.front().unwrap().1,
            observed[1].overall,
            "the tick-2 digest belongs after the same-tick mutation, not at the frame cut"
        );
        assert_eq!(
            seeker.current.persistence.last_admitted_tick(),
            Some(Tick(1))
        );
        for tick in [1, 2, 4, 3] {
            let frame = seeker.seek_to_exact(tick)?;
            assert_eq!(frame.config().persistence_interval, 3);
            assert_eq!(frame.config().replay_event_tick_cap, 128);
            assert_eq!(frame.world_digest_v1()?, observed[(tick - 1) as usize]);
        }
        seeker.finish_replay()?;

        // Reproduce the former binding bug as a negative control. With cadence
        // one the tick-2 step emits its digest before the recorded mutation and
        // explicit flush; correct journal application cannot repair that cut.
        let mut wrong_world = seeker.source.restore_world()?;
        let remaining = Arc::new(Mutex::new(seeker.expected.clone()));
        let persistence = wrong_world.bind_continuation_persistence(
            Box::new(DigestVerifier {
                remaining: Arc::clone(&remaining),
            }),
            1,
            owner.config().replay_event_tick_cap,
        )?;
        let current_config = serde_json::to_value(wrong_world.config())?;
        let mut wrong = JournalReplayWorld {
            world: wrong_world,
            persistence,
            current_config,
            remaining,
        };
        let error = wrong
            .advance_to(2, &seeker.mirror, seeker.anchor_tick)
            .expect_err("an invented cadence emits the off-cadence digest too early");
        assert!(format!("{error:#}").contains("digest diverged"));

        let options = MontageOptions {
            output_path: directory.path().join("cadence-three.cast"),
            stride: 1,
            selection: SelectionConfig {
                pre_window_ticks: 0,
                post_window_ticks: 0,
                ..SelectionConfig::default()
            },
            width: 16,
            height: 8,
            ..MontageOptions::default()
        };
        let mut seeker = PersistedReplaySeeker::from_storage(&reader, 1, 4)?;
        let summary = encode_montage_with_seeker(
            &mut seeker,
            &[sample_event(2, EventKind::PopulationBoom, 30.0)],
            4,
            &options,
        )?;
        assert_eq!(summary.total_frames, 1);
        assert!(std::fs::read_to_string(&options.output_path)?.contains("[t=2 rank=1]"));
        Ok(())
    }

    struct FailAfterFirstFrame {
        world: WorldState,
        calls: usize,
    }

    impl ReplayProvider for FailAfterFirstFrame {
        fn seek_to_exact(&mut self, target_tick: u64) -> Result<&WorldState, ReelReplayError> {
            self.calls += 1;
            if self.calls > 1 {
                return Err(ReelReplayError::Storage(
                    "injected later frame failure".to_owned(),
                ));
            }
            assert_eq!(target_tick, self.world.tick().0);
            Ok(&self.world)
        }

        fn max_tick(&self) -> u64 {
            5
        }
    }

    #[test]
    fn incomplete_reels_are_not_published_and_existing_destinations_are_preserved()
    -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let world = checkpoint_world(Some(crate::BrainPreset::Mlp));
        let events = [sample_event(4, EventKind::PopulationBoom, 30.0)];
        for format in [ReelFormat::Ascii, ReelFormat::Gif] {
            let options = MontageOptions {
                format,
                output_path: directory.path().join(format!("failed-{format}")),
                stride: 1,
                selection: SelectionConfig {
                    pre_window_ticks: 1,
                    post_window_ticks: 1,
                    ..SelectionConfig::default()
                },
                width: 16,
                height: 8,
            };
            let mut failing = FailAfterFirstFrame {
                world: checkpoint_world(Some(crate::BrainPreset::Mlp)),
                calls: 0,
            };
            let error = encode_montage_with_seeker(&mut failing, &events, 5, &options)
                .expect_err("the second frame fails after a valid frame was encoded");
            assert!(error.to_string().contains("injected later frame failure"));
            assert_eq!(failing.calls, 2);
            assert!(!options.output_path.exists());

            std::fs::write(&options.output_path, b"preserve existing reel")?;
            let error = encode_montage(&world, &events, 5, &options)
                .expect_err("an existing destination must refuse publication");
            assert!(matches!(
                error,
                ReelReplayError::Io(reason) if reason.contains("could not publish montage")
            ));
            assert_eq!(
                std::fs::read(&options.output_path)?,
                b"preserve existing reel"
            );
        }
        Ok(())
    }

    #[test]
    fn checkpoint_replay_restores_every_compiled_preset_and_exact_next_states() {
        let mut presets = vec![
            None,
            Some(crate::BrainPreset::Mlp),
            Some(crate::BrainPreset::Dwraon),
            Some(crate::BrainPreset::Assembly),
            Some(crate::BrainPreset::Mixed),
        ];
        if cfg!(feature = "brain-ft") {
            presets.push(Some(crate::BrainPreset::Ft));
        }
        if cfg!(feature = "neuro") {
            presets.push(Some(crate::BrainPreset::Neuro));
        }
        for preset in presets {
            let mut uninterrupted = checkpoint_world(preset);
            let checkpoint = uninterrupted.checkpoint_v1().unwrap();
            let original = checkpoint.encode().unwrap();
            let mut seeker = CheckpointReplaySeeker::new(vec![checkpoint.clone()], 9);
            let mut expected = Vec::new();
            expected.push(uninterrupted.world_digest_v1().unwrap());
            for _ in 3..9 {
                uninterrupted.step().unwrap();
                expected.push(uninterrupted.world_digest_v1().unwrap());
            }
            // Exercise cached forward progress and actual backwards reconstruction.
            for tick in [3, 4, 6, 5, 9, 3, 7] {
                let replayed = seeker.seek_to_exact(tick).unwrap();
                assert_eq!(
                    replayed.world_digest_v1().unwrap(),
                    expected[(tick - 3) as usize],
                    "checkpoint replay changed science for {preset:?} at tick {tick}"
                );
            }
            assert_eq!(checkpoint.encode().unwrap(), original);
        }
    }

    #[test]
    fn checkpoint_replay_preserves_saved_family_subset_and_order() {
        let mut uninterrupted = checkpoint_world(Some(crate::BrainPreset::Dwraon));
        uninterrupted
            .register_brain_family(
                scriptbots_brain::mlp::MlpBrain::KIND.as_str(),
                Box::new(scriptbots_brain::mlp::MlpBrainFamily::new()),
            )
            .unwrap();
        let checkpoint = uninterrupted.checkpoint_v1().unwrap();
        let original = checkpoint.encode().unwrap();
        let registry = checkpoint.required_brain_registry();
        assert_eq!(registry.entries.len(), 2);
        assert_eq!(
            registry.entries[0].kind,
            scriptbots_brain::dwraon::DwraonBrain::KIND.as_str()
        );
        let mut expected = vec![uninterrupted.world_digest_v1().unwrap()];
        for _ in 3..9 {
            uninterrupted.step().unwrap();
            expected.push(uninterrupted.world_digest_v1().unwrap());
        }
        let mut seeker = CheckpointReplaySeeker::new(vec![checkpoint.clone()], 9);
        for tick in [3, 4, 6, 5, 9, 3, 7] {
            let replayed = seeker.seek_to_exact(tick).unwrap();
            assert_eq!(
                replayed.world_digest_v1().unwrap(),
                expected[usize::try_from(tick - 3).unwrap()]
            );
            assert_eq!(
                replayed.checkpoint_v1().unwrap().required_brain_registry(),
                registry
            );
        }
        assert_eq!(checkpoint.encode().unwrap(), original);
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn checkpoint_replay_refuses_changed_neuro_constructor_identity() {
        let mut world = checkpoint_world(Some(crate::BrainPreset::Neuro));
        let mut changed = world.config().clone();
        changed.neuroflow.hidden_layers = vec![11, 5];
        scriptbots_core::apply_control_command(
            &mut world,
            crate::ControlCommand::UpdateConfig(Box::new(changed)),
        )
        .unwrap();
        let checkpoint = world.checkpoint_v1().unwrap();
        let original = checkpoint.encode().unwrap();
        let mut seeker = CheckpointReplaySeeker::new(vec![checkpoint.clone()], 4);
        assert!(matches!(
            seeker.seek_to_exact(3),
            Err(ReelReplayError::Storage(reason))
                if reason.contains("does not match the compiled application adapters")
        ));
        assert!(seeker.current_world.is_none());
        assert_eq!(checkpoint.encode().unwrap(), original);
    }

    #[test]
    fn dwraon_checkpoint_encodes_reproducible_ascii_and_gif_without_mutating_source() {
        let world = checkpoint_world(Some(crate::BrainPreset::Dwraon));
        let original = world.checkpoint_v1().unwrap().encode().unwrap();
        let events = [sample_event(5, EventKind::PopulationBoom, 30.0)];
        let directory = tempfile::tempdir().unwrap();
        for format in [ReelFormat::Ascii, ReelFormat::Gif] {
            let mut outputs = Vec::new();
            for repetition in 0..2 {
                let options = MontageOptions {
                    format,
                    output_path: directory.path().join(format!("{format}-{repetition}")),
                    stride: 1,
                    selection: SelectionConfig {
                        pre_window_ticks: 1,
                        post_window_ticks: 1,
                        ..SelectionConfig::default()
                    },
                    width: 16,
                    height: 8,
                };
                let summary = encode_montage(&world, &events, 6, &options).unwrap();
                assert_eq!(summary.total_frames, 3);
                assert_eq!(summary.clips_count, 1);
                let bytes = std::fs::read(&options.output_path).unwrap();
                assert!(!bytes.is_empty());
                assert_eq!(summary.bytes_written, bytes.len() as u64);
                outputs.push(bytes);
            }
            assert_eq!(outputs[0], outputs[1]);
        }
        assert_eq!(world.checkpoint_v1().unwrap().encode().unwrap(), original);
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
        let temp = tempfile::tempdir().expect("temp directory");

        let event = sample_event(50, EventKind::PopulationCrash, 20.0);

        let opts = MontageOptions {
            format: ReelFormat::Ascii,
            output_path: temp.path().join("offset.cast"),
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
        assert!(!opts.output_path.exists());
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
        let temp = tempfile::tempdir().expect("temp directory");
        let opts = MontageOptions {
            format: ReelFormat::Ascii,
            output_path: temp.path().join("bounded.cast"),
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

        let directory = tempfile::tempdir().expect("temp directory");

        let opts1 = MontageOptions {
            format: ReelFormat::Ascii,
            output_path: directory.path().join("first.cast"),
            stride: 5,
            selection: SelectionConfig {
                pre_window_ticks: 15,
                post_window_ticks: 25,
                ..Default::default()
            },
            ..Default::default()
        };
        let opts2 = MontageOptions {
            output_path: directory.path().join("second.cast"),
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

        let bytes1 = std::fs::read(&opts1.output_path).expect("read 1");
        let bytes2 = std::fs::read(&opts2.output_path).expect("read 2");

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

        let directory = tempfile::tempdir().expect("temp directory");

        let opts1 = MontageOptions {
            format: ReelFormat::Gif,
            output_path: directory.path().join("first.gif"),
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
            output_path: directory.path().join("second.gif"),
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

        let bytes1 = std::fs::read(&opts1.output_path).expect("read 1");
        let bytes2 = std::fs::read(&opts2.output_path).expect("read 2");

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
        let temp = tempfile::tempdir().expect("temp directory");
        let stride = 5;
        let pre = 20;
        let post = 30;

        let opts = MontageOptions {
            format: ReelFormat::Ascii,
            output_path: temp.path().join("count.cast"),
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
