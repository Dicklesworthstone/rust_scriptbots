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
    pub dropped: u64,
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

/// Map a single `AudioFrame` into updated `AudioParams` and a rate-limited `VoicePlan`.
#[must_use]
#[expect(
    clippy::suboptimal_flops,
    reason = "deterministic audio smoothing rounds its multiplication before adding the prior parameter"
)]
pub fn map_frame(
    prev: &AudioParams,
    frame: &AudioFrame,
    config: &AudioConfig,
) -> (AudioParams, VoicePlan) {
    let mut next_params = prev.clone();

    // Smooth drone density based on population
    let target_density = u16::try_from(frame.population).map_or(1.0, |population| {
        (f32::from(population) / 500.0).clamp(0.0, 1.0)
    });
    next_params.drone_density += (target_density - next_params.drone_density) * 0.1;

    // Smooth dissonance based on diet ratio: carnivore-heavy -> higher dissonance
    let target_dissonance = (1.0 - frame.herbivore_share).clamp(0.0, 1.0);
    next_params.dissonance += (target_dissonance - next_params.dissonance) * 0.05;

    // Master gain override
    next_params.master_gain = config.master_gain;

    let mut plan = VoicePlan::default();
    let total_triggers =
        u64::from(frame.births) + u64::from(frame.deaths) + u64::from(frame.spike_hits);

    if total_triggers == 0 {
        return (next_params, plan);
    }

    let allowed = total_triggers.min(config.max_oneshots_per_tick as u64);
    plan.dropped = total_triggers.saturating_sub(allowed);

    // Generate bell oneshots for births
    let mut added = 0;
    for _ in 0..u64::from(frame.births).min(allowed) {
        if added >= allowed {
            break;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "the PCM voice plan uses an f32 pitch offset; retain the existing integer-to-f32 rounding"
        )]
        let detune_cents = added as f32 * 5.0;
        plan.one_shots.push(OneShot {
            kind: "Bell".into(),
            pan: 0.0,
            gain: 0.5,
            detune_cents,
        });
        added += 1;
    }

    // Generate thud oneshots for deaths
    for _ in 0..u64::from(frame.deaths).min(allowed) {
        if added >= allowed {
            break;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "the PCM voice plan uses an f32 pitch offset; retain the existing integer-to-f32 rounding"
        )]
        let detune_cents = -(added as f32 * 10.0);
        plan.one_shots.push(OneShot {
            kind: "Thud".into(),
            pan: -0.2,
            gain: 0.6,
            detune_cents,
        });
        added += 1;
    }

    // Generate transient oneshots for spike hits
    for _ in 0..u64::from(frame.spike_hits).min(allowed) {
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

/// A PCM buffer whose size cannot be represented or allocated on this target.
#[derive(Debug, thiserror::Error)]
pub enum AudioRenderError {
    /// The number of output samples does not fit an addressable buffer length.
    #[error("PCM sample count overflows: {frames} frames with {samples_per_tick} samples per tick")]
    SampleCountOverflow {
        /// Number of input frames.
        frames: usize,
        /// Requested output samples for each frame.
        samples_per_tick: u64,
    },
    /// The sample count fits an index, but its byte length exceeds a vector's capacity.
    #[error("PCM buffer of {samples} f32 samples exceeds the target allocation capacity")]
    CapacityExceeded {
        /// Requested number of samples.
        samples: usize,
    },
    /// The allocator refused a representable PCM buffer.
    #[error("could not reserve PCM buffer for {samples} samples: {source}")]
    AllocationFailed {
        /// Requested number of samples.
        samples: usize,
        /// The allocator's exact refusal.
        #[source]
        source: std::collections::TryReserveError,
    },
}

fn pcm_sample_count(frames: usize, samples_per_tick: u64) -> Result<usize, AudioRenderError> {
    let samples_per_tick_usize =
        usize::try_from(samples_per_tick).map_err(|_| AudioRenderError::SampleCountOverflow {
            frames,
            samples_per_tick,
        })?;
    let samples = frames.checked_mul(samples_per_tick_usize).ok_or(
        AudioRenderError::SampleCountOverflow {
            frames,
            samples_per_tick,
        },
    )?;
    std::alloc::Layout::array::<f32>(samples)
        .map_err(|_| AudioRenderError::CapacityExceeded { samples })?;
    Ok(samples)
}

/// Render a sequence of `AudioFrame` values to deterministic mono 32-bit float PCM samples.
///
/// # Errors
/// Returns a typed error if the sample count or allocation byte size overflows,
/// or the allocator refuses the buffer. Empty input and a zero tick rate produce
/// an empty buffer.
#[expect(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    reason = "the deterministic PCM synthesis contract uses f32 sample coordinates and separately rounded oscillator products and sums"
)]
pub fn render_offline_pcm(
    frames: &[AudioFrame],
    config: &AudioConfig,
    ticks_per_second: u64,
) -> Result<Vec<f32>, AudioRenderError> {
    if frames.is_empty() || ticks_per_second == 0 {
        return Ok(Vec::new());
    }

    // A tick rate wider than the u32 sample rate necessarily yields zero samples.
    let samples_per_tick = u32::try_from(ticks_per_second).map_or(0, |tick_rate| {
        config.sample_rate as usize / tick_rate as usize
    });
    let total_samples = pcm_sample_count(frames.len(), samples_per_tick as u64)?;
    let mut pcm = Vec::new();
    pcm.try_reserve_exact(total_samples)
        .map_err(|source| AudioRenderError::AllocationFailed {
            samples: total_samples,
            source,
        })?;
    pcm.resize(total_samples, 0.0_f32);

    let mut current_params = AudioParams::default();
    let dt = 1.0 / config.sample_rate as f32;

    for (tick_idx, frame) in frames.iter().enumerate() {
        let (next_params, plan) = map_frame(&current_params, frame, config);
        current_params = next_params;

        let start_sample = tick_idx * samples_per_tick;
        let end_sample = (start_sample + samples_per_tick).min(total_samples);

        for (sample_offset, s_idx) in (start_sample..end_sample).enumerate() {
            let t = (tick_idx * samples_per_tick + sample_offset) as f32 * dt;
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

    Ok(pcm)
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
        assert_eq!(plan.dropped, 80 - plan.one_shots.len() as u64);
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
        let pcm = render_offline_pcm(&frames, &cfg, 60).expect("bounded PCM buffer");

        assert_ne!(pcm.len(), 0);
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
        let pcm = render_offline_pcm(&[], &AudioConfig::default(), 60).expect("empty PCM buffer");
        assert_eq!(pcm.len(), 0);
    }

    #[test]
    fn event_count_overflow_preserves_full_dropped_total() {
        let frame = AudioFrame {
            births: u32::MAX,
            deaths: u32::MAX,
            spike_hits: u32::MAX,
            ..AudioFrame::default()
        };
        let config = AudioConfig {
            max_oneshots_per_tick: 1,
            ..AudioConfig::default()
        };
        let (_, plan) = map_frame(&AudioParams::default(), &frame, &config);
        assert_eq!(plan.one_shots.len(), 1);
        assert_eq!(plan.one_shots[0].kind, "Bell");
        assert_eq!(plan.dropped, 3 * u64::from(u32::MAX) - 1);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn voice_budget_above_u32_does_not_wrap_to_zero() {
        let frame = AudioFrame {
            births: 1,
            deaths: 1,
            spike_hits: 1,
            ..AudioFrame::default()
        };
        let config = AudioConfig {
            max_oneshots_per_tick: u32::MAX as usize + 1,
            ..AudioConfig::default()
        };
        let (_, plan) = map_frame(&AudioParams::default(), &frame, &config);
        assert_eq!(plan.one_shots.len(), 3);
        assert_eq!(plan.dropped, 0);
        assert_eq!(
            plan.one_shots
                .iter()
                .map(|shot| shot.kind.as_str())
                .collect::<Vec<_>>(),
            ["Bell", "Thud", "Transient"]
        );
    }

    #[test]
    fn pcm_layout_refuses_sample_and_byte_overflow_before_allocation() {
        assert_eq!(
            pcm_sample_count(2, 800).expect("ordinary sample count"),
            1_600
        );
        assert!(matches!(
            pcm_sample_count(usize::MAX, 2),
            Err(AudioRenderError::SampleCountOverflow {
                frames: usize::MAX,
                samples_per_tick: 2,
            })
        ));
        let samples =
            usize::try_from(isize::MAX).expect("positive isize fits usize") / size_of::<f32>() + 1;
        assert!(matches!(
            pcm_sample_count(samples, 1),
            Err(AudioRenderError::CapacityExceeded { samples: actual }) if actual == samples
        ));
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn pcm_layout_refuses_samples_per_tick_that_do_not_fit_usize() {
        assert!(matches!(
            pcm_sample_count(1, u64::from(u32::MAX) + 1),
            Err(AudioRenderError::SampleCountOverflow { frames: 1, .. })
        ));
    }

    #[test]
    fn zero_and_above_sample_tick_rates_produce_empty_pcm() {
        let frames = [AudioFrame::default()];
        let config = AudioConfig::default();
        for tick_rate in [0, u64::from(config.sample_rate) + 1, u64::MAX] {
            assert_eq!(
                render_offline_pcm(&frames, &config, tick_rate)
                    .expect("empty PCM for this tick rate")
                    .len(),
                0
            );
        }
    }
}
