//! Pure, lock-free audio engine, frame mapper, and offline PCM renderer (`bd-16g.14.1`, `bd-16g.14.2`).

use serde::{Deserialize, Serialize};

/// Maximum oneshot sounds triggered per tick.
pub const MAX_ONESHOTS_PER_TICK: usize = 8;
/// Maximum concurrent active voices.
pub const MAX_VOICES: usize = 32;

/// A lightweight, lock-free snapshot of world state metrics relevant to audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioFrame {
    /// Simulation tick.
    pub tick: u64,
    /// Active agent population count.
    pub population: u32,
    /// Birth count in current tick.
    pub births: u32,
    /// Death count in current tick.
    pub deaths: u32,
    /// Spike combat hits in current tick.
    pub spike_hits: u32,
    /// Herbivore population share in `[0, 1]`.
    pub herbivore_share: f32,
    /// Mean agent energy level in `[0, 2]`.
    pub mean_energy: f32,
    /// Mean agent health in `[0, 2]`.
    pub mean_health: f32,
}

impl Default for AudioFrame {
    fn default() -> Self {
        Self {
            tick: 0,
            population: 0,
            births: 0,
            deaths: 0,
            spike_hits: 0,
            herbivore_share: 0.5,
            mean_energy: 1.0,
            mean_health: 1.0,
        }
    }
}

/// Continuous audio synthesis parameters computed from frame history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioParams {
    /// Continuous drone density in `[0, 1]`.
    pub drone_density: f32,
    /// Continuous drone brightness in `[0, 1]`.
    pub drone_brightness: f32,
    /// Harmonic dissonance index in `[0, 1]`.
    pub dissonance: f32,
    /// Transient gain scaling.
    pub transient_gain: f32,
    /// Master volume gain in `[0, 1]`.
    pub master_gain: f32,
}

impl Default for AudioParams {
    fn default() -> Self {
        Self {
            drone_density: 0.2,
            drone_brightness: 0.5,
            dissonance: 0.0,
            transient_gain: 1.0,
            master_gain: 0.8,
        }
    }
}

/// A single one-shot sound event triggered by a voice plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OneShot {
    /// Sound classification: "Bell", "Thud", "Transient".
    pub kind: String,
    /// Panning value in `[-1.0, 1.0]`.
    pub pan: f32,
    /// Amplitude gain in `[0.0, 1.0]`.
    pub gain: f32,
    /// Pitch detune offset in cents.
    pub detune_cents: f32,
}

/// Planned set of voice events for a single tick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct VoicePlan {
    /// One-shot sound events to trigger.
    pub one_shots: Vec<OneShot>,
    /// Number of overflow events dropped by rate limiting.
    pub dropped: u32,
}

/// Audio engine configuration settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Target sample rate (Hz), default: 48000.
    pub sample_rate: u32,
    /// Master volume gain limit.
    pub master_gain: f32,
    /// Maximum oneshots per tick.
    pub max_oneshots_per_tick: usize,
    /// Token bucket refill rate per tick.
    pub token_refill_rate: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            master_gain: 0.8,
            max_oneshots_per_tick: MAX_ONESHOTS_PER_TICK,
            token_refill_rate: 2.0,
        }
    }
}

/// Map a single AudioFrame into updated AudioParams and a rate-limited VoicePlan.
#[must_use]
pub fn map_frame(
    prev: &AudioParams,
    frame: &AudioFrame,
    config: &AudioConfig,
) -> (AudioParams, VoicePlan) {
    let mut next_params = prev.clone();

    // Smooth drone density based on population
    let target_density = (frame.population as f32 / 500.0).clamp(0.0, 1.0);
    next_params.drone_density += (target_density - next_params.drone_density) * 0.1;

    // Smooth dissonance based on diet ratio: carnivore-heavy -> higher dissonance
    let target_dissonance = (1.0 - frame.herbivore_share).clamp(0.0, 1.0);
    next_params.dissonance += (target_dissonance - next_params.dissonance) * 0.05;

    // Master gain override
    next_params.master_gain = config.master_gain;

    let mut plan = VoicePlan::default();
    let total_triggers = frame.births + frame.deaths + frame.spike_hits;

    if total_triggers == 0 {
        return (next_params, plan);
    }

    let allowed = total_triggers.min(config.max_oneshots_per_tick as u32) as usize;
    plan.dropped = total_triggers.saturating_sub(allowed as u32);

    // Generate bell oneshots for births
    let mut added = 0;
    for _ in 0..frame.births.min(allowed as u32) {
        if added >= allowed {
            break;
        }
        plan.one_shots.push(OneShot {
            kind: "Bell".into(),
            pan: 0.0,
            gain: 0.5,
            detune_cents: added as f32 * 5.0,
        });
        added += 1;
    }

    // Generate thud oneshots for deaths
    for _ in 0..frame.deaths.min(allowed as u32) {
        if added >= allowed {
            break;
        }
        plan.one_shots.push(OneShot {
            kind: "Thud".into(),
            pan: -0.2,
            gain: 0.6,
            detune_cents: -(added as f32 * 10.0),
        });
        added += 1;
    }

    // Generate transient oneshots for spike hits
    for _ in 0..frame.spike_hits.min(allowed as u32) {
        if added >= allowed {
            break;
        }
        plan.one_shots.push(OneShot {
            kind: "Transient".into(),
            pan: 0.3,
            gain: 0.7,
            detune_cents: 0.0,
        });
        added += 1;
    }

    (next_params, plan)
}

/// Render a sequence of AudioFrames to deterministic 48kHz mono 32-bit float PCM samples.
#[must_use]
pub fn render_offline_pcm(
    frames: &[AudioFrame],
    config: &AudioConfig,
    ticks_per_second: u64,
) -> Vec<f32> {
    if frames.is_empty() || ticks_per_second == 0 {
        return Vec::new();
    }

    let samples_per_tick = config.sample_rate as u64 / ticks_per_second;
    let total_samples = (frames.len() as u64 * samples_per_tick) as usize;
    let mut pcm = vec![0.0f32; total_samples];

    let mut current_params = AudioParams::default();
    let dt = 1.0 / config.sample_rate as f32;

    for (tick_idx, frame) in frames.iter().enumerate() {
        let (next_params, plan) = map_frame(&current_params, frame, config);
        current_params = next_params;

        let start_sample = tick_idx * samples_per_tick as usize;
        let end_sample = (start_sample + samples_per_tick as usize).min(total_samples);

        for (sample_offset, s_idx) in (start_sample..end_sample).enumerate() {
            let t = (tick_idx * samples_per_tick as usize + sample_offset) as f32 * dt;
            // Synthesize drone oscillator
            let freq = 110.0 + current_params.dissonance * 20.0;
            let drone =
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.1 * current_params.drone_density;

            // Add oneshot transient contributions
            let mut oneshot_sum = 0.0f32;
            for shot in &plan.one_shots {
                let shot_freq = match shot.kind.as_str() {
                    "Bell" => 440.0,
                    "Thud" => 80.0,
                    _ => 220.0,
                };
                let env = (-20.0 * (sample_offset as f32 * dt)).exp();
                oneshot_sum += (2.0 * std::f32::consts::PI * shot_freq * t).sin() * shot.gain * env;
            }

            let raw_sample = (drone + oneshot_sum) * current_params.master_gain;
            // Soft-knee limiter [-1.0, 1.0]
            pcm[s_idx] = raw_sample.tanh();
        }
    }

    pcm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_frame_rate_limits_bursts() {
        let prev = AudioParams::default();
        let frame = AudioFrame {
            tick: 1,
            population: 100,
            births: 50,
            deaths: 20,
            spike_hits: 10,
            herbivore_share: 0.7,
            mean_energy: 1.0,
            mean_health: 1.0,
        };
        let cfg = AudioConfig::default();

        let (next_params, plan) = map_frame(&prev, &frame, &cfg);
        assert!(plan.one_shots.len() <= cfg.max_oneshots_per_tick);
        assert_eq!(plan.dropped, 80 - plan.one_shots.len() as u32);
        assert!(next_params.drone_density > 0.0);
    }

    #[test]
    fn test_render_offline_pcm_bounds() {
        let frames = vec![
            AudioFrame {
                tick: 0,
                population: 10,
                births: 1,
                ..AudioFrame::default()
            },
            AudioFrame {
                tick: 1,
                population: 12,
                births: 2,
                ..AudioFrame::default()
            },
        ];
        let cfg = AudioConfig::default();
        let pcm = render_offline_pcm(&frames, &cfg, 60);

        assert!(!pcm.is_empty());
        assert_eq!(pcm.len(), (48_000 / 60) * 2);
        for &sample in &pcm {
            assert!(
                sample.abs() <= 1.0,
                "all PCM samples must remain within [-1, 1]"
            );
        }
    }

    #[test]
    fn test_empty_frames_renders_empty_pcm() {
        let pcm = render_offline_pcm(&[], &AudioConfig::default(), 60);
        assert!(pcm.is_empty());
    }
}
