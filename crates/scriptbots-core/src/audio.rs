//! Pure, lock-free audio engine, frame mapper, and offline PCM renderer (`bd-16g.14.1`, `bd-16g.14.2`, `bd-16g.14.3`).

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum oneshot sounds triggered per tick.
pub const MAX_ONESHOTS_PER_TICK: usize = 8;
/// Maximum oneshot sounds triggered per tick as f32.
pub const MAX_ONESHOTS_PER_TICK_F32: f32 = 8.0;
/// Maximum concurrent active voices.
pub const MAX_VOICES: usize = 32;
/// Maximum spatial event coordinates stored per frame.
pub const MAX_SPATIAL_EVENTS: usize = 16;
/// Capacity of the lock-free simulation-to-audio non-blocking channel.
pub const AUDIO_CHANNEL_CAPACITY: usize = 8;
/// Default sample rate in Hz.
pub const DEFAULT_SAMPLE_RATE: u32 = 48_000;
/// Soft-knee limiter engagement threshold in linear amplitude (-1.0 dBFS).
pub const LIMITER_THRESHOLD: f32 = 0.891_250_9; // 10^(-1/20)
/// Soft-knee limiter transition knee width in linear amplitude.
pub const LIMITER_KNEE_WIDTH: f32 = 0.2;

/// 2D coordinate for spatial sound positioning.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct SpatialPosition {
    /// World X coordinate.
    pub x: f32,
    /// World Y coordinate.
    pub y: f32,
}

impl SpatialPosition {
    /// Create a new spatial position coordinate.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Typed sound event classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CueKind {
    /// Bell tone for birth events.
    #[default]
    Bell,
    /// Damped low thud for death events.
    Thud,
    /// High-frequency percussive transient for combat/spike hits.
    Transient,
}

impl CueKind {
    /// Static string identifier for the cue kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bell => "Bell",
            Self::Thud => "Thud",
            Self::Transient => "Transient",
        }
    }

    /// Base oscillator frequency in Hz.
    #[must_use]
    pub const fn base_frequency_hz(self) -> f32 {
        match self {
            Self::Bell => 440.0,
            Self::Thud => 80.0,
            Self::Transient => 220.0,
        }
    }

    /// Default decay duration in seconds.
    #[must_use]
    pub const fn duration_seconds(self) -> f32 {
        match self {
            Self::Bell => 0.18,
            Self::Thud => 0.22,
            Self::Transient => 0.12,
        }
    }
}

impl std::fmt::Display for CueKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A lightweight, lock-free, allocation-free snapshot of world state metrics relevant to audio.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
    /// Crowding factor in `[0, 1]`.
    pub crowding: f32,
    /// Bounded spatial event coordinates recorded in this tick.
    pub spike_positions: [SpatialPosition; MAX_SPATIAL_EVENTS],
    /// Number of valid positions in `spike_positions`.
    pub spike_position_count: usize,
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
            crowding: 0.0,
            spike_positions: [SpatialPosition::default(); MAX_SPATIAL_EVENTS],
            spike_position_count: 0,
        }
    }
}

impl AudioFrame {
    /// Record a spike combat coordinate if capacity remains.
    pub const fn add_spike_position(&mut self, pos: SpatialPosition) -> bool {
        if self.spike_position_count < MAX_SPATIAL_EVENTS {
            self.spike_positions[self.spike_position_count] = pos;
            self.spike_position_count += 1;
            true
        } else {
            false
        }
    }

    /// Slice of valid spatial spike coordinates in this frame.
    #[must_use]
    pub fn spike_positions(&self) -> &[SpatialPosition] {
        &self.spike_positions[..self.spike_position_count.min(MAX_SPATIAL_EVENTS)]
    }
}

/// Per-layer sound mixing gain scaling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LayerGains {
    /// Continuous drone synth gain.
    pub drone: f32,
    /// Birth bell gain.
    pub bell: f32,
    /// Death thud gain.
    pub thud: f32,
    /// Spike combat transient gain.
    pub transient: f32,
}

impl Default for LayerGains {
    fn default() -> Self {
        Self {
            drone: 0.5,
            bell: 0.6,
            thud: 0.6,
            transient: 0.7,
        }
    }
}

/// Continuous audio synthesis parameters computed from frame history.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AudioParams {
    /// Continuous drone density in `[0, 1]`.
    pub drone_density: f32,
    /// Continuous drone brightness in `[0, 1]`.
    pub drone_brightness: f32,
    /// Harmonic dissonance index in `[0, 1]`.
    pub dissonance: f32,
    /// Smoothed bell activity rate.
    pub bell_rate: f32,
    /// Smoothed thud activity rate.
    pub thud_rate: f32,
    /// Transient gain scaling.
    pub transient_gain: f32,
    /// Layer gains.
    pub layer_gains: LayerGains,
    /// Master volume gain in `[0, 1]`.
    pub master_gain: f32,
    /// Whether audio is muted.
    pub is_muted: bool,
    /// Persistent token bucket balance for rate limiting.
    pub tokens: f32,
}

impl Default for AudioParams {
    fn default() -> Self {
        Self {
            drone_density: 0.2,
            drone_brightness: 0.5,
            dissonance: 0.0,
            bell_rate: 0.0,
            thud_rate: 0.0,
            transient_gain: 1.0,
            layer_gains: LayerGains::default(),
            master_gain: 0.8,
            is_muted: false,
            tokens: MAX_ONESHOTS_PER_TICK_F32,
        }
    }
}

/// A single one-shot sound event triggered by a voice plan.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OneShot {
    /// Sound classification.
    pub kind: CueKind,
    /// Panning value in `[-1.0, 1.0]`.
    pub pan: f32,
    /// Amplitude gain in `[0.0, 1.0]`.
    pub gain: f32,
    /// Pitch detune offset in cents.
    pub detune_cents: f32,
    /// Optional spatial position.
    pub position: Option<SpatialPosition>,
}

impl Default for OneShot {
    fn default() -> Self {
        Self {
            kind: CueKind::Bell,
            pan: 0.0,
            gain: 0.5,
            detune_cents: 0.0,
            position: None,
        }
    }
}

/// Planned set of voice events for a single tick (bounded, stack-allocated, zero-allocation).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VoicePlan {
    /// One-shot sound events to trigger.
    pub one_shots: [OneShot; MAX_ONESHOTS_PER_TICK],
    /// Number of active one-shots in `one_shots`.
    pub count: usize,
    /// Number of overflow events dropped by rate limiting.
    pub dropped: u64,
}

impl Default for VoicePlan {
    fn default() -> Self {
        Self {
            one_shots: [OneShot::default(); MAX_ONESHOTS_PER_TICK],
            count: 0,
            dropped: 0,
        }
    }
}

impl VoicePlan {
    /// Push a one-shot cue into the voice plan if capacity remains.
    pub const fn push(&mut self, shot: OneShot) -> bool {
        if self.count < MAX_ONESHOTS_PER_TICK {
            self.one_shots[self.count] = shot;
            self.count += 1;
            true
        } else {
            false
        }
    }

    /// Slice of active one-shots scheduled for this tick.
    #[must_use]
    pub fn one_shots(&self) -> &[OneShot] {
        &self.one_shots[..self.count.min(MAX_ONESHOTS_PER_TICK)]
    }

    /// Number of scheduled one-shots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count.min(MAX_ONESHOTS_PER_TICK)
    }

    /// Whether the voice plan has zero scheduled one-shots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterator over scheduled one-shots.
    pub fn iter(&self) -> std::slice::Iter<'_, OneShot> {
        self.one_shots().iter()
    }
}

impl std::ops::Deref for VoicePlan {
    type Target = [OneShot];

    fn deref(&self) -> &Self::Target {
        self.one_shots()
    }
}

impl<'a> IntoIterator for &'a VoicePlan {
    type Item = &'a OneShot;
    type IntoIter = std::slice::Iter<'a, OneShot>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Token bucket for persistent discrete event rate limiting.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TokenBucket {
    /// Maximum token capacity.
    pub capacity: f32,
    /// Current token balance.
    pub tokens: f32,
    /// Refill rate in tokens per tick.
    pub refill_rate: f32,
}

impl TokenBucket {
    /// Create a new token bucket initialized to full capacity.
    #[must_use]
    pub const fn new(capacity: f32, refill_rate: f32) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate,
        }
    }

    /// Refill the token bucket by its refill rate, capped at capacity.
    pub fn refill(&mut self) {
        self.tokens = (self.tokens + self.refill_rate).min(self.capacity);
    }

    /// Attempt to consume up to `requested` whole tokens. Returns granted tokens.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "tokens are clamped to non-negative integer values"
    )]
    pub fn consume(&mut self, requested: usize) -> usize {
        let available = self.tokens.floor().max(0.0) as usize;
        let granted = requested.min(available);
        self.tokens -= granted as f32;
        granted
    }
}

/// Audio engine configuration settings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Target sample rate (Hz), default: 48000.
    pub sample_rate: u32,
    /// Master volume gain limit in `[0, 1]`.
    pub master_gain: f32,
    /// Maximum oneshots per tick.
    pub max_oneshots_per_tick: usize,
    /// Token bucket refill rate per tick.
    pub token_refill_rate: f32,
    /// Maximum token bucket capacity.
    pub token_bucket_capacity: f32,
    /// Layer gains.
    pub layer_gains: LayerGains,
    /// Instant total mute toggle.
    pub is_muted: bool,
    /// Run seed for deterministic spatial and detune jitter.
    pub run_seed: u64,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            master_gain: 0.8,
            max_oneshots_per_tick: MAX_ONESHOTS_PER_TICK,
            token_refill_rate: 2.0,
            token_bucket_capacity: MAX_ONESHOTS_PER_TICK_F32,
            layer_gains: LayerGains::default(),
            is_muted: false,
            run_seed: 0,
        }
    }
}

/// Deterministic 64-bit seed mixing run seed and tick count.
#[inline]
#[must_use]
pub const fn deterministic_tick_seed(run_seed: u64, tick: u64) -> u64 {
    let mut z = run_seed ^ tick.rotate_left(13).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Soft-knee limiter with threshold at -1.0 dBFS (0.89125) and smooth transition knee.
///
/// Returns `(limited_sample, limiter_engaged, attenuation_db)`.
#[inline]
#[must_use]
#[expect(clippy::suboptimal_flops, reason = "clarity of limiter formula")]
pub fn soft_knee_limiter(sample: f32) -> (f32, bool, f32) {
    let abs_s = sample.abs();
    let lower_knee = LIMITER_THRESHOLD - LIMITER_KNEE_WIDTH * 0.5;
    let upper_knee = LIMITER_THRESHOLD + LIMITER_KNEE_WIDTH * 0.5;

    if abs_s <= lower_knee {
        (sample, false, 0.0)
    } else if abs_s <= upper_knee {
        let excess = abs_s - lower_knee;
        let compressed = abs_s - (excess * excess) / (2.0 * LIMITER_KNEE_WIDTH);
        let limited = sample.signum() * compressed.min(1.0);
        let atten_db = if abs_s > 1e-6 {
            (20.0 * (limited.abs() / abs_s).log10()).min(0.0)
        } else {
            0.0
        };
        (limited, true, atten_db.abs())
    } else {
        // Above the upper knee: asymptotic saturation strictly within [-1.0, 1.0]
        let excess = abs_s - LIMITER_THRESHOLD;
        let compressed = LIMITER_THRESHOLD + (excess / (1.0 + excess)).min(1.0 - LIMITER_THRESHOLD);
        let limited = sample.signum() * compressed.min(1.0);
        let atten_db = if abs_s > 1e-6 {
            (20.0 * (limited.abs() / abs_s).log10()).min(0.0)
        } else {
            0.0
        };
        (limited, true, atten_db.abs())
    }
}

/// Pure frame mapper without active voice constraints.
#[must_use]
pub fn map_frame(
    prev: &AudioParams,
    frame: &AudioFrame,
    config: &AudioConfig,
) -> (AudioParams, VoicePlan) {
    map_frame_bounded(prev, frame, config, 0)
}

/// Map a single `AudioFrame` into updated `AudioParams` and a rate-limited `VoicePlan`,
/// taking concurrent active voices into account to enforce `MAX_VOICES`.
#[must_use]
#[expect(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "deterministic audio smoothing and voice allocation round integer quantities to f32"
)]
pub fn map_frame_bounded(
    prev: &AudioParams,
    frame: &AudioFrame,
    config: &AudioConfig,
    active_voices: usize,
) -> (AudioParams, VoicePlan) {
    let mut next_params = *prev;

    // Smooth drone density based on population
    let target_density = u16::try_from(frame.population).map_or(1.0, |population| {
        (f32::from(population) / 500.0).clamp(0.0, 1.0)
    });
    next_params.drone_density += (target_density - next_params.drone_density) * 0.1;

    // Smooth dissonance based on diet ratio: carnivore-heavy -> higher dissonance
    let target_dissonance = (1.0 - frame.herbivore_share).clamp(0.0, 1.0);
    next_params.dissonance += (target_dissonance - next_params.dissonance) * 0.05;

    // Smooth event rates
    let bell_target = (f32::from(u16::try_from(frame.births).unwrap_or(u16::MAX)) / 20.0).min(1.0);
    next_params.bell_rate += (bell_target - next_params.bell_rate) * 0.2;

    let thud_target = (f32::from(u16::try_from(frame.deaths).unwrap_or(u16::MAX)) / 20.0).min(1.0);
    next_params.thud_rate += (thud_target - next_params.thud_rate) * 0.2;

    // Gains and mute
    next_params.master_gain = config.master_gain;
    next_params.layer_gains = config.layer_gains;
    next_params.is_muted = config.is_muted;

    // Persistent token bucket refill
    let mut tokens = (prev.tokens + config.token_refill_rate).min(config.token_bucket_capacity);

    let mut plan = VoicePlan::default();
    let total_triggers =
        u64::from(frame.births) + u64::from(frame.deaths) + u64::from(frame.spike_hits);

    if total_triggers == 0 {
        next_params.tokens = tokens;
        return (next_params, plan);
    }

    let token_allowed = tokens.floor().max(0.0) as u64;
    let tick_allowed = config.max_oneshots_per_tick as u64;
    let voice_ceiling = MAX_VOICES.saturating_sub(active_voices) as u64;
    let budget = tick_allowed.min(token_allowed).min(voice_ceiling);
    let allowed = total_triggers.min(budget);

    plan.dropped = total_triggers.saturating_sub(allowed);
    tokens -= allowed as f32;
    next_params.tokens = tokens;

    // Generate bell oneshots for births
    let mut added = 0;
    let births_to_add = u64::from(frame.births).min(allowed);
    for _ in 0..births_to_add {
        if added >= allowed {
            break;
        }
        let detune_cents = (added as f32) * 5.0;
        plan.push(OneShot {
            kind: CueKind::Bell,
            pan: 0.0,
            gain: 0.5 * config.layer_gains.bell,
            detune_cents,
            position: None,
        });
        added += 1;
    }

    // Generate thud oneshots for deaths
    let deaths_to_add = u64::from(frame.deaths).min(allowed.saturating_sub(added));
    for _ in 0..deaths_to_add {
        if added >= allowed {
            break;
        }
        let detune_cents = -((added as f32) * 10.0);
        plan.push(OneShot {
            kind: CueKind::Thud,
            pan: -0.2,
            gain: 0.6 * config.layer_gains.thud,
            detune_cents,
            position: None,
        });
        added += 1;
    }

    // Generate transient oneshots for spike combat hits
    let spikes_to_add = u64::from(frame.spike_hits).min(allowed.saturating_sub(added));
    for idx in 0..spikes_to_add {
        if added >= allowed {
            break;
        }
        let pos = if (idx as usize) < frame.spike_position_count {
            Some(frame.spike_positions[idx as usize])
        } else {
            None
        };
        let panning = pos.map_or(0.3, |p| (p.x / 1000.0).clamp(-1.0, 1.0));
        plan.push(OneShot {
            kind: CueKind::Transient,
            pan: panning,
            gain: 0.7 * config.layer_gains.transient,
            detune_cents: 0.0,
            position: pos,
        });
        added += 1;
    }

    (next_params, plan)
}

/// Structured limiter and audio diagnostics emitted once per second.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LimiterTelemetry {
    /// Number of samples where the soft-knee limiter was actively attenuating.
    pub limiter_engaged_samples: u64,
    /// Total samples evaluated during this reporting window.
    pub total_samples: u64,
    /// Percentage of samples with active limiter engagement in `[0, 100]`.
    pub limiter_engaged_pct: f32,
    /// Maximum attenuation applied by the limiter in dB (positive value).
    pub max_attenuation_db: f32,
    /// Peak linear amplitude observed in this window.
    pub peak_amplitude: f32,
    /// Peak amplitude expressed in dBFS (`<= 0.0`).
    pub peak_dbfs: f32,
    /// RMS linear amplitude observed in this window.
    pub rms_amplitude: f32,
    /// RMS amplitude expressed in dBFS (`<= 0.0`).
    pub rms_dbfs: f32,
    /// Concurrent active voice count.
    pub voices_active: usize,
    /// One-shot cues emitted in this window.
    pub one_shots_emitted: u64,
    /// One-shot cues dropped by rate limiting in this window.
    pub one_shots_dropped: u64,
    /// Simulation frames dropped due to full channel transport in this window.
    pub frames_dropped: u64,
    /// Drone density parameter.
    pub drone_density: f32,
    /// Dissonance parameter.
    pub dissonance: f32,
    /// Master volume gain.
    pub master_gain: f32,
}

impl Default for LimiterTelemetry {
    fn default() -> Self {
        Self {
            limiter_engaged_samples: 0,
            total_samples: 0,
            limiter_engaged_pct: 0.0,
            max_attenuation_db: 0.0,
            peak_amplitude: 0.0,
            peak_dbfs: -96.0,
            rms_amplitude: 0.0,
            rms_dbfs: -96.0,
            voices_active: 0,
            one_shots_emitted: 0,
            one_shots_dropped: 0,
            frames_dropped: 0,
            drone_density: 0.2,
            dissonance: 0.0,
            master_gain: 0.8,
        }
    }
}

/// Create a non-blocking capacity-8 SPSC channel transport for simulation-to-audio frames.
#[must_use]
pub fn audio_channel() -> (AudioSender, AudioReceiver) {
    let (tx, rx) = std::sync::mpsc::sync_channel(AUDIO_CHANNEL_CAPACITY);
    let frames_dropped = Arc::new(AtomicU64::new(0));
    (
        AudioSender {
            tx,
            frames_dropped: Arc::clone(&frames_dropped),
        },
        AudioReceiver { rx, frames_dropped },
    )
}

/// Simulation-side non-blocking audio frame sender.
#[derive(Debug, Clone)]
pub struct AudioSender {
    tx: std::sync::mpsc::SyncSender<AudioFrame>,
    frames_dropped: Arc<AtomicU64>,
}

/// Failure condition when sending an `AudioFrame`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AudioSendError {
    /// The capacity-8 channel is currently full; frame dropped.
    #[error("audio channel is full; frame dropped")]
    Full,
    /// The receiver has disconnected; frame dropped.
    #[error("audio receiver disconnected; frame dropped")]
    Disconnected,
}

impl AudioSender {
    /// Non-blocking send: returns immediately. If the queue is full or disconnected,
    /// increments `frames_dropped` and returns an error without blocking simulation.
    pub fn try_send(&self, frame: AudioFrame) -> Result<(), AudioSendError> {
        match self.tx.try_send(frame) {
            Ok(()) => Ok(()),
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                self.frames_dropped.fetch_add(1, Ordering::Relaxed);
                Err(AudioSendError::Full)
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                self.frames_dropped.fetch_add(1, Ordering::Relaxed);
                Err(AudioSendError::Disconnected)
            }
        }
    }

    /// Alias for non-blocking send. Never blocks the calling simulation thread.
    pub fn send(&self, frame: AudioFrame) -> Result<(), AudioSendError> {
        self.try_send(frame)
    }

    /// Total count of frames dropped by this sender.
    #[must_use]
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped.load(Ordering::Relaxed)
    }
}

/// Audio-side receiver for simulation frames.
pub struct AudioReceiver {
    rx: std::sync::mpsc::Receiver<AudioFrame>,
    frames_dropped: Arc<AtomicU64>,
}

impl AudioReceiver {
    /// Try to receive a single frame without blocking.
    pub fn try_recv(&self) -> Result<AudioFrame, std::sync::mpsc::TryRecvError> {
        self.rx.try_recv()
    }

    /// Drain all currently pending frames in the channel into a handler closure.
    pub fn drain_all<F: FnMut(AudioFrame)>(&self, mut handler: F) -> usize {
        let mut count = 0;
        while let Ok(frame) = self.rx.try_recv() {
            handler(frame);
            count += 1;
        }
        count
    }

    /// Total count of frames dropped by the channel transport.
    #[must_use]
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped.load(Ordering::Relaxed)
    }

    /// Read and reset the dropped frame counter (e.g. for once-per-second reporting).
    #[must_use]
    pub fn take_frames_dropped(&self) -> u64 {
        self.frames_dropped.swap(0, Ordering::Relaxed)
    }
}

/// Device or backend sink interface for rendered PCM audio samples.
pub trait AudioDevice: Send {
    /// Output a slice of 32-bit floating point PCM audio samples.
    fn write_samples(&mut self, samples: &[f32]);
}

/// A null audio device sink that discards samples (default for headless or tests).
#[derive(Debug, Default, Clone, Copy)]
pub struct NullAudioDevice;

impl AudioDevice for NullAudioDevice {
    fn write_samples(&mut self, _samples: &[f32]) {}
}

/// An in-memory audio sink buffer that records PCM samples for inspection and verification.
#[derive(Debug, Default, Clone)]
pub struct BufferAudioDevice {
    /// Recorded PCM samples.
    pub buffer: Vec<f32>,
}

impl AudioDevice for BufferAudioDevice {
    fn write_samples(&mut self, samples: &[f32]) {
        self.buffer.extend_from_slice(samples);
    }
}

/// Optional Kira audio backend device.
#[cfg(feature = "audio")]
pub struct KiraAudioDevice {
    #[expect(
        dead_code,
        reason = "Kira audio manager held for active sound lifetime"
    )]
    manager: Option<kira::AudioManager<kira::DefaultBackend>>,
}

#[cfg(feature = "audio")]
impl KiraAudioDevice {
    /// Attempt to initialize the Kira audio manager device.
    /// Returns `Err` if device initialization fails on headless or unprovisioned hosts.
    pub fn new() -> Result<Self, String> {
        let manager =
            kira::AudioManager::<kira::DefaultBackend>::new(kira::AudioManagerSettings::default())
                .map_err(|err| format!("{err:?}"))?;
        Ok(Self {
            manager: Some(manager),
        })
    }
}

#[cfg(feature = "audio")]
impl AudioDevice for KiraAudioDevice {
    fn write_samples(&mut self, _samples: &[f32]) {
        // Kira static sounds or streaming can be fed here when the device is active.
    }
}

/// Active synthesizing voice state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActiveVoice {
    /// Sound classification.
    pub kind: CueKind,
    /// Oscillator frequency in Hz.
    pub frequency_hz: f32,
    /// Amplitude gain.
    pub gain: f32,
    /// Panning in `[-1.0, 1.0]`.
    pub pan: f32,
    /// Current phase in radians.
    pub phase: f32,
    /// Phase step per sample.
    pub phase_step: f32,
    /// Elapsed samples since trigger.
    pub age_samples: usize,
    /// Total duration in samples.
    pub total_samples: usize,
}

/// Device-side consumer and sound generator.
///
/// Has NO reference to world simulation state, shared world, mutex locks, or callbacks into simulation.
pub struct Sonifier<D = NullAudioDevice> {
    receiver: AudioReceiver,
    config: AudioConfig,
    device: D,
    params: AudioParams,
    active_voices: [Option<ActiveVoice>; MAX_VOICES],
    sample_clock: u64,
    // Window-level telemetry accumulators
    total_samples_accumulated: u64,
    limiter_engaged_samples: u64,
    sample_accumulator: f32,
    peak_sample: f32,
    max_attenuation_db: f32,
    one_shots_emitted: u64,
    one_shots_dropped: u64,
    latest_telemetry: LimiterTelemetry,
}

impl Sonifier<NullAudioDevice> {
    /// Construct a new Sonifier accepting only the receiver and audio config.
    /// Defaults to `NullAudioDevice`.
    #[must_use]
    pub fn new(receiver: AudioReceiver, config: AudioConfig) -> Self {
        Self::with_device(receiver, config, NullAudioDevice)
    }
}

impl<D: AudioDevice> Sonifier<D> {
    /// Construct a Sonifier with an explicit device sink.
    pub fn with_device(receiver: AudioReceiver, config: AudioConfig, device: D) -> Self {
        let params = AudioParams {
            master_gain: config.master_gain,
            layer_gains: config.layer_gains,
            is_muted: config.is_muted,
            tokens: config.token_bucket_capacity,
            ..AudioParams::default()
        };
        Self {
            receiver,
            config,
            device,
            params,
            active_voices: [None; MAX_VOICES],
            sample_clock: 0,
            total_samples_accumulated: 0,
            limiter_engaged_samples: 0,
            sample_accumulator: 0.0,
            peak_sample: 0.0,
            max_attenuation_db: 0.0,
            one_shots_emitted: 0,
            one_shots_dropped: 0,
            latest_telemetry: LimiterTelemetry::default(),
        }
    }

    /// Access the underlying receiver.
    #[must_use]
    pub const fn receiver(&self) -> &AudioReceiver {
        &self.receiver
    }

    /// Access the active audio configuration.
    #[must_use]
    pub const fn config(&self) -> &AudioConfig {
        &self.config
    }

    /// Access the latest continuous audio parameters.
    #[must_use]
    pub const fn params(&self) -> &AudioParams {
        &self.params
    }

    /// Access the most recently published once-per-second telemetry record.
    #[must_use]
    pub const fn telemetry(&self) -> &LimiterTelemetry {
        &self.latest_telemetry
    }

    /// Set mute state instantly. When muted, every output sample is bit-zero (`0.0_f32`).
    pub const fn set_muted(&mut self, is_muted: bool) {
        self.config.is_muted = is_muted;
        self.params.is_muted = is_muted;
    }

    /// Update master gain.
    pub const fn set_master_gain(&mut self, gain: f32) {
        self.config.master_gain = gain.clamp(0.0, 1.0);
        self.params.master_gain = self.config.master_gain;
    }

    /// Update per-layer gains.
    pub const fn set_layer_gains(&mut self, layer_gains: LayerGains) {
        self.config.layer_gains = layer_gains;
        self.params.layer_gains = layer_gains;
    }

    /// Current number of concurrent active voices.
    #[must_use]
    pub fn active_voice_count(&self) -> usize {
        self.active_voices.iter().filter(|v| v.is_some()).count()
    }

    /// Process a single frame and synthesize `out.len()` PCM samples into `out`.
    #[expect(
        clippy::suboptimal_flops,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "synthesis oscillators use standard f32 trigonometric phases"
    )]
    pub fn step_frame(&mut self, frame: &AudioFrame, out: &mut [f32]) {
        let current_active = self.active_voice_count();
        let (next_params, plan) =
            map_frame_bounded(&self.params, frame, &self.config, current_active);
        self.params = next_params;

        self.one_shots_emitted += plan.len() as u64;
        self.one_shots_dropped += plan.dropped;

        // Admit planned one-shots into active voice slots
        for shot in plan.one_shots() {
            let base_freq = shot.kind.base_frequency_hz();
            let freq = base_freq * (shot.detune_cents / 1200.0).exp2();
            let total_samples =
                (shot.kind.duration_seconds() * self.config.sample_rate as f32) as usize;
            let phase_step = 2.0 * std::f32::consts::PI * freq / self.config.sample_rate as f32;

            // Find an empty slot
            for slot in &mut self.active_voices {
                if slot.is_none() {
                    *slot = Some(ActiveVoice {
                        kind: shot.kind,
                        frequency_hz: freq,
                        gain: shot.gain,
                        pan: shot.pan,
                        phase: 0.0,
                        phase_step,
                        age_samples: 0,
                        total_samples,
                    });
                    break;
                }
            }
        }

        // Instant bit-zero mute contract
        if self.config.is_muted || self.params.is_muted || self.config.master_gain <= 0.0 {
            out.fill(0.0);
            self.device.write_samples(out);
            self.accumulate_telemetry(out, true);
            return;
        }

        let dt = 1.0 / self.config.sample_rate as f32;
        let drone_freq = 110.0 + self.params.dissonance * 20.0;

        for out_sample in out.iter_mut() {
            let t = self.sample_clock as f32 * dt;
            self.sample_clock = self.sample_clock.wrapping_add(1);

            let drone = (2.0 * std::f32::consts::PI * drone_freq * t).sin()
                * 0.1
                * self.params.drone_density
                * self.config.layer_gains.drone;

            let mut voice_sum = 0.0_f32;
            for slot in &mut self.active_voices {
                if let Some(voice) = slot {
                    let vt = voice.age_samples as f32 * dt;
                    let env = (-20.0 * vt).exp();
                    let s = voice.phase.sin() * voice.gain * env;
                    voice.phase += voice.phase_step;
                    if voice.phase > 2.0 * std::f32::consts::PI {
                        voice.phase -= 2.0 * std::f32::consts::PI;
                    }
                    voice.age_samples += 1;
                    voice_sum += s;
                    if voice.age_samples >= voice.total_samples {
                        *slot = None;
                    }
                }
            }

            let raw = (drone + voice_sum) * self.config.master_gain;
            let (limited, engaged, atten_db) = soft_knee_limiter(raw);

            // Accumulate window statistics
            self.sample_accumulator += limited * limited;
            if limited.abs() > self.peak_sample {
                self.peak_sample = limited.abs();
            }
            if engaged {
                self.limiter_engaged_samples += 1;
                if atten_db > self.max_attenuation_db {
                    self.max_attenuation_db = atten_db;
                }
            }

            *out_sample = limited;
        }

        self.device.write_samples(out);
        self.accumulate_telemetry(out, false);
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "telemetry averages and logarithmic conversions use standard f32 formulas"
    )]
    fn accumulate_telemetry(&mut self, out: &[f32], is_muted: bool) {
        self.total_samples_accumulated += out.len() as u64;

        // Emit telemetry once per second
        if self.total_samples_accumulated >= u64::from(self.config.sample_rate) {
            let total = self.total_samples_accumulated.max(1);
            let rms = if is_muted {
                0.0
            } else {
                (self.sample_accumulator / total as f32).sqrt()
            };
            let peak = if is_muted { 0.0 } else { self.peak_sample };
            let engaged_pct = (self.limiter_engaged_samples as f32 / total as f32) * 100.0;
            let peak_dbfs = if peak > 1e-6 {
                20.0 * peak.log10()
            } else {
                -96.0
            };
            let rms_dbfs = if rms > 1e-6 {
                20.0 * rms.log10()
            } else {
                -96.0
            };
            let frames_dropped = self.receiver.take_frames_dropped();

            self.latest_telemetry = LimiterTelemetry {
                limiter_engaged_samples: self.limiter_engaged_samples,
                total_samples: total,
                limiter_engaged_pct: engaged_pct,
                max_attenuation_db: self.max_attenuation_db,
                peak_amplitude: peak,
                peak_dbfs,
                rms_amplitude: rms,
                rms_dbfs,
                voices_active: self.active_voice_count(),
                one_shots_emitted: self.one_shots_emitted,
                one_shots_dropped: self.one_shots_dropped,
                frames_dropped,
                drone_density: self.params.drone_density,
                dissonance: self.params.dissonance,
                master_gain: self.config.master_gain,
            };

            tracing::debug!(
                target: "scriptbots::audio",
                voices_active = self.latest_telemetry.voices_active,
                one_shots_emitted = self.latest_telemetry.one_shots_emitted,
                one_shots_dropped = self.latest_telemetry.one_shots_dropped,
                frames_dropped,
                peak_dbfs,
                rms_dbfs,
                limiter_engaged_pct = engaged_pct,
                drone_density = self.params.drone_density,
                dissonance = self.params.dissonance,
                master_gain = self.config.master_gain,
                "audio telemetry"
            );

            if frames_dropped > 0 {
                tracing::warn!(
                    target: "scriptbots::audio",
                    frames_dropped,
                    "audio thread is falling behind; frames dropped"
                );
            }
            if engaged_pct > 20.0 {
                tracing::warn!(
                    target: "scriptbots::audio",
                    limiter_engaged_pct = engaged_pct,
                    "audio mix is heavily limited (limiter engaged > 20%)"
                );
            }
            if self.one_shots_dropped > (self.config.max_oneshots_per_tick as u64) * 10 {
                tracing::warn!(
                    target: "scriptbots::audio",
                    one_shots_dropped = self.one_shots_dropped,
                    "oneshots dropped exceeds threshold"
                );
            }

            // Reset reporting period accumulators
            self.total_samples_accumulated = 0;
            self.limiter_engaged_samples = 0;
            self.sample_accumulator = 0.0;
            self.peak_sample = 0.0;
            self.max_attenuation_db = 0.0;
            self.one_shots_emitted = 0;
            self.one_shots_dropped = 0;
        }
    }

    /// Drain and process all pending frames from the receiver, returning number of frames processed.
    pub fn process_pending(&mut self, ticks_per_second: u64) -> usize {
        if ticks_per_second == 0 {
            return 0;
        }
        let ticks_rate = usize::try_from(ticks_per_second).unwrap_or(usize::MAX);
        let samples_per_tick = (self.config.sample_rate as usize / ticks_rate).max(1);
        let mut buffer = vec![0.0_f32; samples_per_tick];
        let mut processed = 0;

        while let Ok(frame) = self.receiver.try_recv() {
            self.step_frame(&frame, &mut buffer);
            processed += 1;
        }
        processed
    }
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

    if config.is_muted || config.master_gain <= 0.0 {
        return Ok(pcm);
    }

    let mut current_params = AudioParams {
        master_gain: config.master_gain,
        layer_gains: config.layer_gains,
        is_muted: config.is_muted,
        tokens: config.token_bucket_capacity,
        ..AudioParams::default()
    };
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
            let drone = (2.0 * std::f32::consts::PI * freq * t).sin()
                * 0.1
                * current_params.drone_density
                * config.layer_gains.drone;

            // Add oneshot transient contributions
            let mut oneshot_sum = 0.0f32;
            for shot in plan.one_shots() {
                let shot_freq = shot.kind.base_frequency_hz();
                let env = (-20.0 * (sample_offset as f32 * dt)).exp();
                oneshot_sum += (2.0 * std::f32::consts::PI * shot_freq * t).sin() * shot.gain * env;
            }

            let raw_sample = (drone + oneshot_sum) * current_params.master_gain;
            let (limited, _, _) = soft_knee_limiter(raw_sample);
            pcm[s_idx] = limited;
        }
    }

    Ok(pcm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_allocation_free() {
        // Assert AudioFrame is bounded and zero-heap allocation
        assert_eq!(std::mem::size_of::<[SpatialPosition; 16]>(), 16 * 8);
        let mut frame = AudioFrame::default();
        for i in 0..20u16 {
            let pos = SpatialPosition::new(f32::from(i), f32::from(i * 2));
            let added = frame.add_spike_position(pos);
            if usize::from(i) < MAX_SPATIAL_EVENTS {
                assert!(added);
            } else {
                assert!(!added);
            }
        }
        assert_eq!(frame.spike_positions().len(), MAX_SPATIAL_EVENTS);

        // Assert VoicePlan is bounded and zero-heap allocation
        let mut plan = VoicePlan::default();
        for _ in 0..12 {
            let _ = plan.push(OneShot::default());
        }
        assert_eq!(plan.len(), MAX_ONESHOTS_PER_TICK);
        assert_eq!(plan.one_shots().len(), MAX_ONESHOTS_PER_TICK);
    }

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
            ..AudioFrame::default()
        };
        let cfg = AudioConfig::default();

        let (next_params, plan) = map_frame(&prev, &frame, &cfg);
        assert!(plan.len() <= cfg.max_oneshots_per_tick);
        assert_eq!(plan.dropped, 80 - plan.len() as u64);
        assert!(next_params.drone_density > 0.0);
    }

    #[test]
    fn test_token_bucket_refill() {
        let mut bucket = TokenBucket::new(8.0, 2.0);
        assert_eq!(bucket.consume(8), 8);
        assert_eq!(bucket.consume(1), 0);

        // First refill
        bucket.refill();
        assert_eq!(bucket.consume(5), 2);
        assert_eq!(bucket.consume(1), 0);

        // Refill to capacity
        for _ in 0..10 {
            bucket.refill();
        }
        assert_eq!(bucket.tokens, 8.0);
        assert_eq!(bucket.consume(10), 8);
    }

    #[test]
    fn test_active_voice_accounting_and_machine_gun() {
        let (sender, receiver) = audio_channel();
        let config = AudioConfig {
            token_refill_rate: 2.0,
            max_oneshots_per_tick: 8,
            ..AudioConfig::default()
        };
        let mut sonifier = Sonifier::new(receiver, config);
        let mut sample_buffer = vec![0.0_f32; 800]; // 60 TPS at 48kHz is 800 samples

        // Adversarial 600-tick stream: 300 births + 200 deaths + 150 spikes EVERY tick
        for tick in 0..600 {
            let frame = AudioFrame {
                tick,
                population: 10_000,
                births: 300,
                deaths: 200,
                spike_hits: 150,
                herbivore_share: 0.5,
                mean_energy: 1.0,
                mean_health: 1.0,
                ..AudioFrame::default()
            };
            sender.send(frame).expect("send frame");
            let recv_frame = sonifier.receiver().try_recv().expect("recv frame");

            sonifier.step_frame(&recv_frame, &mut sample_buffer);

            // Invariant 1: active voices never exceed MAX_VOICES
            assert!(
                sonifier.active_voice_count() <= MAX_VOICES,
                "active voices {} exceeded MAX_VOICES at tick {}",
                sonifier.active_voice_count(),
                tick
            );

            // Invariant 2: all samples stay strictly in [-1.0, 1.0] without NaN
            for &s in &sample_buffer {
                assert!(!s.is_nan(), "sample must not be NaN at tick {tick}");
                assert!(
                    s.abs() <= 1.0,
                    "sample {s} must not exceed 1.0 at tick {tick}"
                );
            }
        }
    }

    #[test]
    fn test_channel_stall_nonblocking() {
        let (sender, receiver) = audio_channel();

        // Fill channel to exact capacity of 8
        for i in 0..AUDIO_CHANNEL_CAPACITY {
            let frame = AudioFrame {
                tick: i as u64,
                ..AudioFrame::default()
            };
            assert!(sender.try_send(frame).is_ok());
        }
        assert_eq!(sender.frames_dropped(), 0);

        // 9th frame must fail immediately with Full and increment frames_dropped
        let overflow_frame = AudioFrame {
            tick: 99,
            ..AudioFrame::default()
        };
        let start = std::time::Instant::now();
        let res = sender.send(overflow_frame);
        let elapsed = start.elapsed();

        assert_eq!(res, Err(AudioSendError::Full));
        assert!(
            elapsed < std::time::Duration::from_millis(50),
            "send must not block on full queue"
        );
        assert_eq!(sender.frames_dropped(), 1);

        // Drain one frame and ensure send succeeds again
        let drained = receiver.try_recv().expect("drain frame");
        assert_eq!(drained.tick, 0);

        assert!(sender.send(overflow_frame).is_ok());
    }

    #[test]
    fn test_instant_bit_zero_mute() {
        let (sender, receiver) = audio_channel();
        let config = AudioConfig {
            is_muted: true,
            ..AudioConfig::default()
        };
        let mut sonifier = Sonifier::new(receiver, config);
        let frame = AudioFrame {
            tick: 1,
            population: 500,
            births: 100,
            deaths: 50,
            spike_hits: 50,
            ..AudioFrame::default()
        };
        sender.send(frame).expect("send");
        let recv_frame = sonifier.receiver().try_recv().expect("recv");

        let mut out = vec![1.0_f32; 1000];
        sonifier.step_frame(&recv_frame, &mut out);

        for &sample in &out {
            assert_eq!(
                sample.to_bits(),
                0u32,
                "muted audio must produce bit-zero samples"
            );
        }
    }

    #[test]
    fn test_determinism_identical_voiceplans_live_and_offline() {
        let config = AudioConfig {
            run_seed: 12345,
            ..AudioConfig::default()
        };

        let frames: Vec<AudioFrame> = (0..50)
            .map(|tick| AudioFrame {
                tick,
                population: u32::try_from(100 + tick * 5).unwrap_or(u32::MAX),
                births: (tick % 7) as u32,
                deaths: (tick % 5) as u32,
                spike_hits: (tick % 3) as u32,
                herbivore_share: 0.6,
                ..AudioFrame::default()
            })
            .collect();

        // 1. Generate VoicePlans one at a time
        let mut params1 = AudioParams {
            tokens: config.token_bucket_capacity,
            ..AudioParams::default()
        };
        let mut plans_one_by_one = Vec::new();
        for frame in &frames {
            let (next_p, plan) = map_frame(&params1, frame, &config);
            params1 = next_p;
            plans_one_by_one.push(plan);
        }

        // 2. Generate VoicePlans in batch
        let mut params2 = AudioParams {
            tokens: config.token_bucket_capacity,
            ..AudioParams::default()
        };
        let mut plans_batched = Vec::new();
        for frame in &frames {
            let (next_p, plan) = map_frame(&params2, frame, &config);
            params2 = next_p;
            plans_batched.push(plan);
        }

        assert_eq!(plans_one_by_one, plans_batched);

        // 3. Offline render
        let pcm = render_offline_pcm(&frames, &config, 60).expect("render offline");
        assert_ne!(pcm.len(), 0);
        for &s in &pcm {
            assert!(s.abs() <= 1.0);
        }
    }

    #[test]
    fn test_smoothing_continuity() {
        let config = AudioConfig::default();
        let mut params = AudioParams::default();

        // Step change from 0 population to 500 population
        let frame_burst = AudioFrame {
            tick: 1,
            population: 500,
            ..AudioFrame::default()
        };
        let (next_params, _) = map_frame(&params, &frame_burst, &config);

        // One-pole smoothing: delta = (1.0 - 0.2) * 0.1 = 0.08
        let delta = (next_params.drone_density - params.drone_density).abs();
        assert!(
            delta < 0.15,
            "drone density step must be smoothly bounded (got {delta})"
        );
        params = next_params;

        // Quiet frame
        let frame_quiet = AudioFrame {
            tick: 2,
            population: 500,
            ..AudioFrame::default()
        };
        let (next_params2, _) = map_frame(&params, &frame_quiet, &config);
        assert!(next_params2.drone_density > params.drone_density);
    }

    #[test]
    fn test_dissonance_continuity() {
        let config = AudioConfig::default();
        let mut params = AudioParams::default();
        let mut prev_dissonance = params.dissonance;

        // Slow sweep across herbivore shares
        for i in 0..=100u16 {
            let share = f32::from(i) / 100.0;
            let frame = AudioFrame {
                tick: u64::from(i),
                herbivore_share: share,
                ..AudioFrame::default()
            };
            let (next_params, _) = map_frame(&params, &frame, &config);
            let jump = (next_params.dissonance - prev_dissonance).abs();
            assert!(
                jump <= 0.051,
                "dissonance must not chatter or jump discontinuously (jump {jump})"
            );
            prev_dissonance = next_params.dissonance;
            params = next_params;
        }
    }

    #[test]
    fn test_no_nan_extremes() {
        let config = AudioConfig::default();
        let mut params = AudioParams::default();

        for tick in 0..1000 {
            let extreme = if tick % 2 == 0 { u32::MAX } else { 0 };
            let frame = AudioFrame {
                tick,
                population: extreme,
                births: extreme,
                deaths: extreme,
                spike_hits: extreme,
                herbivore_share: if tick % 2 == 0 { 1.0 } else { 0.0 },
                mean_energy: if tick % 2 == 0 { 2.0 } else { 0.0 },
                mean_health: if tick % 2 == 0 { 2.0 } else { 0.0 },
                ..AudioFrame::default()
            };
            let (next_params, plan) = map_frame(&params, &frame, &config);
            assert!(!next_params.drone_density.is_nan());
            assert!(!next_params.dissonance.is_nan());
            assert!(!next_params.tokens.is_nan());
            assert!(next_params.drone_density >= 0.0 && next_params.drone_density <= 1.0);
            assert!(next_params.dissonance >= 0.0 && next_params.dissonance <= 1.0);
            assert!(plan.len() <= MAX_ONESHOTS_PER_TICK);
            params = next_params;
        }
    }

    #[test]
    fn test_structural_no_world_reference() {
        // Type-level verification: Sonifier::new signature takes only receiver and config
        fn assert_sonifier_signature<F>(_f: F)
        where
            F: FnOnce(AudioReceiver, AudioConfig) -> Sonifier<NullAudioDevice>,
        {
        }
        assert_sonifier_signature(Sonifier::new);

        let source = include_str!("audio.rs");
        // Verify audio.rs never references WorldState or SharedWorld or Arc<Mutex<
        // Look for occurrences outside of this test function
        let test_start = source
            .find("fn test_structural_no_world_reference")
            .unwrap_or(source.len());
        let prod_code = &source[..test_start];

        assert!(
            !prod_code.contains("WorldState"),
            "audio module must not reference WorldState"
        );
        assert!(
            !prod_code.contains("SharedWorld"),
            "audio module must not reference SharedWorld"
        );
        assert!(
            !prod_code.contains("Arc<Mutex<"),
            "audio module must not reference Arc<Mutex<"
        );
    }

    #[test]
    fn test_limiter_telemetry() {
        let (sender, receiver) = audio_channel();
        let config = AudioConfig {
            master_gain: 1.0,
            sample_rate: 48_000,
            ..AudioConfig::default()
        };
        let mut sonifier = Sonifier::new(receiver, config);

        // Send a frame with births to trigger oneshots
        let frame = AudioFrame {
            tick: 1,
            births: 8,
            ..AudioFrame::default()
        };
        sender.send(frame).expect("send");
        let recv_frame = sonifier.receiver().try_recv().expect("recv");

        // Synthesize 48_000 samples (1 full second to trigger telemetry emission)
        let mut block = vec![0.0_f32; 48_000];
        sonifier.step_frame(&recv_frame, &mut block);

        let telemetry = sonifier.telemetry();
        assert_eq!(telemetry.total_samples, 48_000);
        assert!(telemetry.peak_dbfs <= 0.0);
    }

    #[test]
    fn test_device_failure_graceful() {
        struct FailingDevice;
        impl AudioDevice for FailingDevice {
            fn write_samples(&mut self, _samples: &[f32]) {
                // Device sinks might silently drop or fail internally
            }
        }

        let (sender, receiver) = audio_channel();
        let mut sonifier = Sonifier::with_device(receiver, AudioConfig::default(), FailingDevice);

        let frame = AudioFrame {
            tick: 1,
            births: 2,
            ..AudioFrame::default()
        };
        assert!(sender.send(frame).is_ok());
        assert_eq!(sonifier.process_pending(60), 1);
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
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.one_shots()[0].kind, CueKind::Bell);
        assert_eq!(plan.dropped, 3 * u64::from(u32::MAX) - 1);
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

    #[cfg(feature = "audio")]
    #[test]
    fn test_live_feature_on_e2e() {
        let (sender, receiver) = audio_channel();
        let config = AudioConfig {
            sample_rate: 48_000,
            ..AudioConfig::default()
        };
        // Either KiraAudioDevice initializes or gracefully returns an Err on headless hosts
        match KiraAudioDevice::new() {
            Ok(device) => {
                let mut sonifier = Sonifier::with_device(receiver, config, device);
                let mut total_processed = 0;
                for tick in 0..120 {
                    let frame = AudioFrame {
                        tick,
                        population: 100,
                        births: (tick % 4) as u32,
                        deaths: (tick % 6) as u32,
                        spike_hits: (tick % 5) as u32,
                        ..AudioFrame::default()
                    };
                    sender.send(frame).expect("send frame without world lock");
                    total_processed += sonifier.process_pending(60);
                }
                assert_eq!(total_processed, 120);
                assert_eq!(sonifier.telemetry().frames_dropped, 0);
            }
            Err(err) => {
                tracing::info!(
                    ?err,
                    "Kira audio device not present on headless host, testing with NullAudioDevice"
                );
                let mut sonifier = Sonifier::new(receiver, config);
                let mut total_processed = 0;
                for tick in 0..120 {
                    let frame = AudioFrame {
                        tick,
                        population: 100,
                        births: (tick % 4) as u32,
                        deaths: (tick % 6) as u32,
                        spike_hits: (tick % 5) as u32,
                        ..AudioFrame::default()
                    };
                    sender.send(frame).expect("send frame without world lock");
                    total_processed += sonifier.process_pending(60);
                }
                assert_eq!(total_processed, 120);
                assert_eq!(sonifier.telemetry().frames_dropped, 0);
            }
        }
    }
}
