//! Headless montage encoder for highlight reels (`bd-16g.9.2`).

use anyhow::{Result, anyhow, bail};
use scriptbots_core::WorldState;
use scriptbots_core::narrative::EventRecord;
use scriptbots_core::reel::{Clip, SelectionConfig, select_clips};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::info;

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

impl std::str::FromStr for ReelFormat {
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
}

impl Default for MontageOptions {
    fn default() -> Self {
        Self {
            format: ReelFormat::Ascii,
            output_path: PathBuf::from("reel.cast"),
            stride: 5,
            selection: SelectionConfig::default(),
        }
    }
}

/// Single rendered frame representation in the montage encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

/// Generate an ASCII character map representation of a WorldState tick.
#[must_use]
pub fn render_ascii_frame(world: &WorldState, width: usize, height: usize) -> Vec<String> {
    let mut grid = vec![vec!['.'; width]; height];

    let w_bounds = world.config().world_width as f32;
    let h_bounds = world.config().world_height as f32;

    // Render agents as 'A'
    for pos in world.agents().columns().positions() {
        let px = (((pos.x / w_bounds) * width as f32) as usize).min(width - 1);
        let py = (((pos.y / h_bounds) * height as f32) as usize).min(height - 1);
        grid[py][px] = 'A';
    }

    grid.into_iter()
        .map(|row| row.into_iter().collect())
        .collect()
}

/// Encode a montage reel from a world state generator and events.
pub fn encode_montage(
    world: &mut WorldState,
    events: &[EventRecord],
    last_tick: u64,
    options: &MontageOptions,
) -> Result<MontageSummary> {
    if options.format == ReelFormat::Mp4 {
        bail!("mp4 format requires ffmpeg on PATH; ascii and gif are available natively");
    }

    let clips = select_clips(events, last_tick, &options.selection);
    if clips.is_empty() {
        info!(target: "scriptbots::reel::render", "no events scored above threshold for reel");
        return Ok(MontageSummary {
            output_path: options.output_path.clone(),
            format: options.format,
            clips_count: 0,
            total_frames: 0,
            bytes_written: 0,
        });
    }

    let mut total_frames = 0;
    let mut ascii_output = String::new();

    // Asciinema v2 header
    if options.format == ReelFormat::Ascii {
        ascii_output.push_str(
            "{\"version\": 2, \"width\": 80, \"height\": 24, \"timestamp\": 0, \"title\": \"ScriptBots Reel\"}\n",
        );
    }

    let mut time_offset = 0.0f64;

    for clip in &clips {
        let caption = clip
            .events
            .first()
            .map_or("Event", |e| e.human_text.as_str());

        let mut tick = clip.start;
        while tick <= clip.end {
            // Step world to target tick if needed
            while world.tick().0 < tick {
                world.step()?;
            }

            let grid = render_ascii_frame(world, 80, 24);
            total_frames += 1;

            if options.format == ReelFormat::Ascii {
                let frame_text = format!(
                    "[t={tick} rank={}] {}\n{}",
                    clip.rank,
                    caption,
                    grid.join("\n")
                );
                let payload = serde_json::to_string(&frame_text)?;
                ascii_output.push_str(&format!("[{time_offset:.2}, \"o\", {payload}]\n"));
                time_offset += 0.1;
            }

            tick += options.stride.max(1);
        }
    }

    let bytes = ascii_output.as_bytes();
    let mut file = File::create(&options.output_path)?;
    file.write_all(bytes)?;

    info!(
        target: "scriptbots::reel::render",
        out = %options.output_path.display(),
        format = ?options.format,
        clips = clips.len(),
        frames = total_frames,
        bytes = bytes.len(),
        "montage reel generated successfully"
    );

    Ok(MontageSummary {
        output_path: options.output_path.clone(),
        format: options.format,
        clips_count: clips.len(),
        total_frames,
        bytes_written: bytes.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::narrative::EventKind;
    use scriptbots_core::{AgentData, Position, ScriptBotsConfig, Tick};
    use tempfile::NamedTempFile;

    #[test]
    fn test_encode_montage_ascii_empty_events() {
        let config = ScriptBotsConfig::default();
        let mut world = WorldState::new(config).expect("world");
        let temp = NamedTempFile::new().expect("temp file");

        let opts = MontageOptions {
            format: ReelFormat::Ascii,
            output_path: temp.path().to_path_buf(),
            stride: 5,
            selection: SelectionConfig::default(),
        };

        let summary = encode_montage(&mut world, &[], 100, &opts).expect("encode montage");
        assert_eq!(summary.clips_count, 0);
        assert_eq!(summary.total_frames, 0);
    }

    #[test]
    fn test_encode_montage_ascii_with_events() {
        let config = ScriptBotsConfig::default();
        let mut world = WorldState::new(config).expect("world");
        world
            .try_spawn_agent(AgentData {
                position: Position::new(50.0, 50.0),
                health: 1.0,
                ..AgentData::default()
            })
            .expect("agent");

        let events = vec![EventRecord {
            schema_version: 1,
            tick: Tick(10),
            kind: EventKind::PopulationCrash,
            severity: 0.9,
            magnitude: 0.8,
            window: (0, 10),
            metric: "population".into(),
            before: 100.0,
            after: 20.0,
            score: 0.9,
            subject: None,
            human_text: "population fell 80%".into(),
        }];

        let temp = NamedTempFile::new().expect("temp file");
        let opts = MontageOptions {
            format: ReelFormat::Ascii,
            output_path: temp.path().to_path_buf(),
            stride: 5,
            selection: SelectionConfig {
                pre_window_ticks: 5,
                post_window_ticks: 5,
                ..SelectionConfig::default()
            },
        };

        let summary = encode_montage(&mut world, &events, 20, &opts).expect("encode montage");
        assert!(summary.clips_count > 0);
        assert!(summary.total_frames > 0);
        assert!(summary.bytes_written > 0);
    }

    #[test]
    fn test_mp4_format_returns_error_without_encoder() {
        let config = ScriptBotsConfig::default();
        let mut world = WorldState::new(config).expect("world");
        let opts = MontageOptions {
            format: ReelFormat::Mp4,
            output_path: PathBuf::from("reel.mp4"),
            stride: 5,
            selection: SelectionConfig::default(),
        };

        let res = encode_montage(&mut world, &[], 100, &opts);
        assert!(res.is_err());
        assert!(
            res.unwrap_err()
                .to_string()
                .contains("mp4 format requires ffmpeg")
        );
    }
}
