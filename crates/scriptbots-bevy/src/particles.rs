//! GPU particle engine core (bd-2z0.14.1.7.1).
//!
//! ScriptBots' world events — births, deaths, combat, eating — are invisible
//! today except as log lines. Bevy has no built-in particle system and new
//! dependencies are off the table (bd-2z0.8 lane), so this module hand-rolls
//! the lean engine the A7 integration bead wires into the render world:
//!
//! - fixed-capacity [`ParticlePool`]s with O(1) spawn/kill and a documented
//!   overflow policy (combat/death cues are never evicted by ambient traffic);
//! - a deterministic [`CueScheduler`] mapping [`visual::VisualCue`]s (the
//!   shared art table from bd-2z0.14.3.2) into spawn batches — identical
//!   inputs produce identical particles on every platform and replay;
//! - a deterministic procedural sprite atlas (the repo stays asset-free);
//! - the WGSL billboard shader source, validated here only structurally and
//!   compiled for real by the integration bead (bd-2z0.14.1.7).
//!
//! Particles are PURELY visual: they never feed back into simulation state,
//! never enter any digest, and their scheduling derives only from
//! (tick, ordinal, world seed) so replays produce identical effects.

use scriptbots_core::visual::{VisualCue, VisualCueKind, value_noise_2d};

/// One particle instance. Plain data, `Copy`, no heap; the GPU instance
/// buffer is filled from pools each frame by the integration bead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Particle {
    /// World position at spawn.
    pub position: [f32; 3],
    /// Initial velocity (world units per tick).
    pub velocity: [f32; 3],
    /// Gravity factor (positive pulls down, negative buoys up in water).
    pub gravity: f32,
    /// Velocity damping per tick in `[0, 1]` (0 = none).
    pub drag: f32,
    /// Tick the particle was born (scheduling clock; never wall-clock).
    pub born_tick: u64,
    /// Lifetime in ticks.
    pub duration_ticks: u32,
    /// Quad size at birth (world units).
    pub size_start: f32,
    /// Quad size at death.
    pub size_end: f32,
    /// Initial rotation (radians).
    pub rotation: f32,
    /// Angular velocity (radians per tick).
    pub spin: f32,
    /// Base color.
    pub color: [f32; 3],
    /// Accent color for two-tone sprites.
    pub accent: [f32; 3],
    /// Emissive intensity in `[0, 1]` (drives bloom).
    pub intensity: f32,
    /// Which sprite tile to sample.
    pub sprite: SpriteKind,
    /// Eviction priority class.
    pub priority: ParticlePriority,
    /// Optional agent anchor (boost trails follow their agent).
    pub follow: Option<AgentAnchor>,
}

/// An agent a particle follows (boost trails).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentAnchor {
    /// Stable agent UID.
    pub uid: u64,
    /// Offset from the agent's position.
    pub offset: [f32; 3],
}

/// Eviction priority: on pool pressure, ambient particles are evicted first;
/// critical ones (combat/death cues) are never evicted by policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ParticlePriority {
    /// Ambient shimmer/idle effects — first to go.
    Ambient,
    /// Ordinary effects (eat nibbles, boost trails).
    Standard,
    /// Combat hits, deaths, births — never evicted by policy.
    Critical,
}

/// Sprite tiles in the baked atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpriteKind {
    /// Four-point star (births, sparkles).
    Spark,
    /// Soft disc (motes, wilts).
    Mote,
    /// Jagged triangle (combat shards).
    Shard,
    /// Noise blob (smoke/foam).
    Puff,
    /// Annulus (reproduction rings).
    Ring,
}

impl SpriteKind {
    /// All sprite kinds in atlas order.
    pub const ALL: [Self; 5] = [Self::Spark, Self::Mote, Self::Shard, Self::Puff, Self::Ring];

    /// Atlas tile index.
    #[must_use]
    pub const fn tile_index(self) -> usize {
        match self {
            Self::Spark => 0,
            Self::Mote => 1,
            Self::Shard => 2,
            Self::Puff => 3,
            Self::Ring => 4,
        }
    }
}

/// Pool overflow behavior when every slot is occupied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverflowPolicy {
    /// Evict the oldest `Ambient` particle; if none exists, reject the spawn.
    #[default]
    DropOldestAmbient,
    /// Reject the incoming spawn no matter what.
    DropNewest,
}

/// Stable handle for one live particle (slot index + generation stamp).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParticleHandle {
    slot: u32,
    generation: u32,
}

/// Fixed-capacity particle pool with O(1) spawn/kill and deterministic
/// slot-order iteration. Allocation happens only in [`Self::with_capacity`].
#[derive(Debug, Clone)]
pub struct ParticlePool {
    slots: Vec<Option<Particle>>,
    generations: Vec<u32>,
    free: Vec<u32>,
    live_count: usize,
    born_counter: u64,
    /// Overflow behavior when the pool is full.
    pub overflow: OverflowPolicy,
}

impl ParticlePool {
    /// Allocate a pool for `capacity` particles (single allocation).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let slots_u32 = u32::try_from(capacity).unwrap_or(u32::MAX);
        Self {
            slots: vec![None; capacity],
            generations: vec![0; capacity],
            free: (0..slots_u32).rev().collect(),
            live_count: 0,
            born_counter: 0,
            overflow: OverflowPolicy::default(),
        }
    }

    /// Maximum number of live particles.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Number of live particles.
    #[must_use]
    pub const fn live_count(&self) -> usize {
        self.live_count
    }

    /// Spawn a particle, applying the overflow policy when full.
    ///
    /// Returns `None` when the spawn was rejected (full pool, policy could
    /// not free a slot).
    pub fn spawn(&mut self, particle: Particle) -> Option<ParticleHandle> {
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None => self.evict_for(particle.priority)?,
        };
        let generation = self.generations[slot as usize];
        self.slots[slot as usize] = Some(particle);
        self.live_count += 1;
        self.born_counter += 1;
        Some(ParticleHandle { slot, generation })
    }

    fn evict_for(&mut self, incoming: ParticlePriority) -> Option<u32> {
        match self.overflow {
            OverflowPolicy::DropNewest => None,
            OverflowPolicy::DropOldestAmbient => {
                if incoming == ParticlePriority::Ambient {
                    // Don't churn: ambient spawns into a full pool simply lose.
                    return None;
                }
                // Find the oldest ambient slot (deterministic slot scan).
                let slot_count = u32::try_from(self.slots.len()).unwrap_or(u32::MAX);
                let victim = (0..slot_count).find(|&slot| {
                    self.slots[slot as usize]
                        .as_ref()
                        .is_some_and(|p| p.priority == ParticlePriority::Ambient)
                })?;
                self.kill_slot(victim);
                // kill_slot returned the victim to the free list; reclaim it
                // for the incoming particle so it is not double-allocated.
                debug_assert_eq!(self.free.last(), Some(&victim));
                self.free.pop();
                Some(victim)
            }
        }
    }

    fn kill_slot(&mut self, slot: u32) {
        if self.slots[slot as usize].take().is_some() {
            self.generations[slot as usize] = self.generations[slot as usize].wrapping_add(1);
            self.free.push(slot);
            self.live_count -= 1;
        }
    }

    /// Kill a live particle by handle; stale handles are ignored.
    pub fn kill(&mut self, handle: ParticleHandle) {
        let slot = handle.slot as usize;
        if slot >= self.slots.len() || self.generations[slot] != handle.generation {
            return;
        }
        self.kill_slot(handle.slot);
    }

    /// Fetch a live particle by handle.
    #[must_use]
    pub fn get(&self, handle: ParticleHandle) -> Option<&Particle> {
        let slot = handle.slot as usize;
        if slot >= self.slots.len() || self.generations[slot] != handle.generation {
            return None;
        }
        self.slots[slot].as_ref()
    }

    /// Iterate live particles in deterministic slot order.
    pub fn for_each_live(&self, mut f: impl FnMut(&Particle)) {
        for particle in self.slots.iter().flatten() {
            f(particle);
        }
    }

    /// Mutable access for per-tick CPU updates (test/CPU path only).
    pub fn for_each_live_mut(&mut self, mut f: impl FnMut(u32, &mut Particle)) {
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if let Some(particle) = slot {
                f(u32::try_from(index).unwrap_or(u32::MAX), particle);
            }
        }
    }

    /// Kill particles whose lifetime ended at or before `tick`.
    ///
    /// Returns the number of expirations. Slot-order determinism: the same
    /// pool state and tick always produce the same post-step state.
    pub fn expire(&mut self, tick: u64) -> usize {
        let mut expired = Vec::new();
        for (index, slot) in self.slots.iter().enumerate() {
            if let Some(particle) = slot
                && tick >= particle.born_tick + u64::from(particle.duration_ticks)
            {
                expired.push(u32::try_from(index).unwrap_or(u32::MAX));
            }
        }
        let count = expired.len();
        for slot in expired {
            self.kill_slot(slot);
        }
        count
    }

    /// Update anchor positions for following particles (boost trails).
    ///
    /// `lookup` resolves an anchor UID to the agent's current world position;
    /// when it returns `None` the particle detaches (keeps its last offset
    /// velocity, a dead agent must not freeze its trail mid-air).
    pub fn update_anchors(&mut self, mut lookup: impl FnMut(u64) -> Option<[f32; 3]>) {
        self.for_each_live_mut(|_slot, particle| {
            if let Some(anchor) = particle.follow {
                if let Some(position) = lookup(anchor.uid) {
                    particle.position = [
                        position[0] + anchor.offset[0],
                        position[1] + anchor.offset[1],
                        position[2] + anchor.offset[2],
                    ];
                } else {
                    particle.follow = None;
                }
            }
        });
    }

    /// Integrate one tick of motion on the CPU path (software adapters and
    /// determinism tests; the GPU path runs the identical math in WGSL).
    pub fn integrate_tick(&mut self) {
        self.for_each_live_mut(|_slot, particle| {
            let damping = 1.0 - particle.drag.clamp(0.0, 1.0);
            particle.velocity[1] -= particle.gravity;
            particle.velocity = [
                particle.velocity[0] * damping,
                particle.velocity[1] * damping,
                particle.velocity[2] * damping,
            ];
            particle.position = [
                particle.velocity[0].mul_add(1.0, particle.position[0]),
                particle.velocity[1].mul_add(1.0, particle.position[1]),
                particle.velocity[2].mul_add(1.0, particle.position[2]),
            ];
            particle.rotation += particle.spin;
        });
    }
}

// ---------------------------------------------------------------------------
// GPU instance projection + transparent-pass ordering.
// ---------------------------------------------------------------------------

/// GPU projection of one [`Particle`]; field-for-field the WGSL
/// `ParticleInstance` layout (23 f32 = 92 bytes). `born_tick` and `duration`
/// are carried as f32: exact for tick values below 2^24 (~77 hours at 60Hz),
/// and the shader only needs the *difference*, which stays exact far longer
/// for particles whose whole life fits in that window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleInstance {
    /// Spawn position.
    pub pos0: [f32; 3],
    /// Initial velocity.
    pub vel0: [f32; 3],
    /// Gravity factor.
    pub gravity: f32,
    /// Velocity damping per tick.
    pub drag: f32,
    /// Born tick (see struct docs for the f32 precision bound).
    pub born_tick: f32,
    /// Lifetime in ticks.
    pub duration: f32,
    /// Quad size at birth.
    pub size_start: f32,
    /// Quad size at death.
    pub size_end: f32,
    /// Initial rotation.
    pub rotation: f32,
    /// Angular velocity.
    pub spin: f32,
    /// Base color.
    pub color: [f32; 3],
    /// Accent color.
    pub accent: [f32; 3],
    /// Emissive intensity.
    pub intensity: f32,
    /// Sprite atlas tile index.
    pub sprite: f32,
    /// Reserved (0); future bitfield (follow-anchor, blend class).
    pub flags: f32,
}

impl ParticleInstance {
    /// Project a live particle into the GPU layout. `tick` is unused today
    /// (motion integrates shader-side from spawn attributes) and reserved
    /// for future spawn-relative encodings.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_particle(particle: &Particle) -> Self {
        Self {
            pos0: particle.position,
            vel0: particle.velocity,
            gravity: particle.gravity,
            drag: particle.drag,
            born_tick: particle.born_tick as f32,
            duration: particle.duration_ticks as f32,
            size_start: particle.size_start,
            size_end: particle.size_end,
            rotation: particle.rotation,
            spin: particle.spin,
            color: particle.color,
            accent: particle.accent,
            intensity: particle.intensity,
            sprite: particle.sprite.tile_index() as f32,
            flags: 0.0,
        }
    }
}

impl ParticlePool {
    /// Pack live particles into `out` in deterministic slot order.
    ///
    /// Write-combining friendly: gap-free sequential writes, and `out` is
    /// reused across frames so steady-state frames perform zero allocation
    /// after the first fill reaches capacity.
    pub fn write_instance_buffer(&self, out: &mut Vec<ParticleInstance>) {
        out.clear();
        out.reserve(self.live_count);
        for particle in self.slots.iter().flatten() {
            out.push(ParticleInstance::from_particle(particle));
        }
    }
}

/// How the transparent particle pass orders billboards (C1 tier contract):
/// Low draws unsorted and relies on the soft depth fade; Medium+ sorts
/// back-to-front by view-space depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendOrderPolicy {
    /// Draw in slot order (free); correctness comes from the depth fade.
    Unsorted,
    /// Bucket-sort back-to-front by view-space depth (Medium+ tiers).
    #[default]
    BucketSort,
}

/// Fill `order` with the draw order as indices into `view_depths`.
///
/// `view_depths` pairs each live particle's slot with its view-space depth
/// (positive forward, larger = farther). Deterministic: a true bucket sort
/// with 256 depth buckets, stable within a bucket by input (slot) order,
/// back-to-front (farthest first). Non-finite depths land in the nearest
/// bucket (drawn last, where the depth fade hides them). `order` and
/// `scratch` are caller-reused so steady-state frames allocate nothing.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn blend_order_into(
    view_depths: &[(u32, f32)],
    policy: BlendOrderPolicy,
    order: &mut Vec<u32>,
    scratch: &mut Vec<u32>,
) {
    order.clear();
    order.extend(0..view_depths.len() as u32);
    if policy == BlendOrderPolicy::Unsorted || view_depths.len() < 2 {
        return;
    }
    let mut min_d = f32::INFINITY;
    let mut max_d = f32::NEG_INFINITY;
    for &(_, depth) in view_depths {
        if depth.is_finite() {
            min_d = min_d.min(depth);
            max_d = max_d.max(depth);
        }
    }
    if !min_d.is_finite() || (max_d - min_d) < 1e-9 {
        return; // empty or uniform depths: slot order is already correct.
    }
    const BUCKETS: usize = 256;
    let span = max_d - min_d;
    let bucket_of = |depth: f32| -> usize {
        if !depth.is_finite() {
            return 0; // non-finite => nearest bucket => drawn last.
        }
        (((depth - min_d) / span) * (BUCKETS - 1) as f32).clamp(0.0, (BUCKETS - 1) as f32) as usize
    };
    let mut counts = [0_usize; BUCKETS];
    for &(_, depth) in view_depths {
        counts[bucket_of(depth)] += 1;
    }
    // Prefix offsets with farthest bucket first (back-to-front).
    let mut starts = [0_usize; BUCKETS];
    let mut acc = 0_usize;
    for bucket in (0..BUCKETS).rev() {
        starts[bucket] = acc;
        acc += counts[bucket];
    }
    scratch.clear();
    scratch.resize(view_depths.len(), 0);
    let mut cursor = starts;
    for (index, &(_, depth)) in view_depths.iter().enumerate() {
        let bucket = bucket_of(depth);
        scratch[cursor[bucket]] = index as u32;
        cursor[bucket] += 1;
    }
    order.copy_from_slice(scratch);
}

// ---------------------------------------------------------------------------
// Deterministic cue scheduling.
// ---------------------------------------------------------------------------

/// A spawn batch produced for one cue (bounded inline storage; cue fan-outs
/// are small by design — see [`CueScheduler::MAX_PER_CUE`]).
pub struct SpawnBatch {
    /// Particles to spawn, in deterministic order.
    pub particles: Vec<Particle>,
}

/// Deterministic FNV-1a64 hash over the scheduling identity.
fn schedule_hash(seed: u64, tick: u64, ordinal: u32, index: u32) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in [seed, tick, u64::from(ordinal), u64::from(index)] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Map a hash to a uniform float in `[0, 1)` (top 24 bits for stability).
#[allow(clippy::cast_precision_loss)]
fn hash_unit(hash: u64) -> f32 {
    // Deliberate: only the top 24 bits participate, so precision loss is
    // exactly the design (a full-mantissa uniform sample).
    (hash >> 40) as f32 / 16_777_216.0
}

/// Deterministic cue -> particle scheduler.
///
/// Every batch derives from `(world seed, tick, cue ordinal within the tick,
/// particle index)` so identical cue streams produce identical particle
/// tables across runs and platforms — the replay-equivalence contract.
#[derive(Debug, Clone)]
pub struct CueScheduler {
    /// World seed for the run.
    pub seed: u64,
    /// Maximum particles one cue may spawn.
    pub max_per_cue: u32,
}

impl Default for CueScheduler {
    fn default() -> Self {
        Self {
            seed: 0,
            max_per_cue: Self::MAX_PER_CUE,
        }
    }
}

impl CueScheduler {
    /// Hard cap on particles per cue (combat storms cannot drown the frame).
    pub const MAX_PER_CUE: u32 = 48;

    /// Create a scheduler for a world seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            ..Self::default()
        }
    }

    fn jitter(&self, tick: u64, ordinal: u32, index: u32, channel: u32) -> f32 {
        hash_unit(schedule_hash(
            self.seed,
            tick,
            ordinal,
            index ^ (channel << 16),
        ))
    }

    /// Integer jitter in `0..max` (the float cast lives only here).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn jitter_u32(&self, tick: u64, ordinal: u32, index: u32, channel: u32, max: u32) -> u32 {
        // jitter returns [0, 1), so the product is in [0, max) and fits.
        (self.jitter(tick, ordinal, index, channel) * max as f32) as u32
    }

    const fn priority_for(kind: VisualCueKind) -> ParticlePriority {
        match kind {
            VisualCueKind::Sparkle | VisualCueKind::SparkCone | VisualCueKind::Shards => {
                ParticlePriority::Critical
            }
            VisualCueKind::Wilt | VisualCueKind::PulseRing | VisualCueKind::Flash => {
                ParticlePriority::Standard
            }
            VisualCueKind::Nibble => ParticlePriority::Ambient,
        }
    }

    const fn sprite_for(kind: VisualCueKind) -> SpriteKind {
        match kind {
            VisualCueKind::Sparkle | VisualCueKind::Flash => SpriteKind::Spark,
            VisualCueKind::Wilt => SpriteKind::Mote,
            VisualCueKind::Shards | VisualCueKind::SparkCone => SpriteKind::Shard,
            VisualCueKind::Nibble => SpriteKind::Puff,
            VisualCueKind::PulseRing => SpriteKind::Ring,
        }
    }

    /// Number of particles a cue spawns: `4 + intensity * 20`, capped.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    #[must_use]
    pub fn count_for_cue(&self, cue: &VisualCue) -> u32 {
        // intensity is clamped to [0, 1], so scaled lands in [4, 24] and the
        // cast always fits.
        let scaled = cue.intensity.clamp(0.0, 1.0).mul_add(20.0, 4.0);
        (scaled as u32).clamp(1, self.max_per_cue)
    }

    /// Schedule the spawn batch for one cue at `position`.
    #[must_use]
    pub fn schedule(
        &self,
        tick: u64,
        ordinal: u32,
        cue: &VisualCue,
        position: [f32; 3],
    ) -> SpawnBatch {
        let count = self.count_for_cue(cue);
        let sprite = Self::sprite_for(cue.kind);
        let priority = Self::priority_for(cue.kind);
        let mut particles = Vec::with_capacity(count as usize);
        for index in 0..count {
            let angle = self.jitter(tick, ordinal, index, 1) * 2.0 * core::f32::consts::PI;
            let speed = self.jitter(tick, ordinal, index, 2).mul_add(0.6, 0.15)
                * (0.5 + cue.intensity.clamp(0.0, 1.0));
            let upward = match cue.kind {
                VisualCueKind::Sparkle => self.jitter(tick, ordinal, index, 3).mul_add(0.8, 0.6),
                VisualCueKind::Wilt => -(self.jitter(tick, ordinal, index, 3).mul_add(0.25, 0.1)),
                VisualCueKind::Nibble => self.jitter(tick, ordinal, index, 3).mul_add(0.35, 0.25),
                VisualCueKind::SparkCone | VisualCueKind::Shards => {
                    self.jitter(tick, ordinal, index, 3).mul_add(0.65, 0.35)
                }
                VisualCueKind::PulseRing | VisualCueKind::Flash => 0.05,
            };
            let (gravity, drag) = match cue.kind {
                VisualCueKind::Wilt => (-0.004, 0.02),
                VisualCueKind::Shards | VisualCueKind::SparkCone => (0.02, 0.06),
                _ => (0.008, 0.04),
            };
            let duration = cue
                .duration_ticks
                .saturating_add(self.jitter_u32(tick, ordinal, index, 4, 8))
                .max(2);
            particles.push(Particle {
                position,
                velocity: [angle.cos() * speed, upward * speed, angle.sin() * speed],
                gravity,
                drag,
                born_tick: tick,
                duration_ticks: duration,
                size_start: cue.radius.mul_add(0.08, 0.4),
                size_end: 0.05,
                rotation: angle,
                spin: (self.jitter(tick, ordinal, index, 5) - 0.5) * 0.3,
                color: cue.color,
                accent: cue.accent_color,
                intensity: cue.intensity,
                sprite,
                priority,
                follow: None,
            });
        }
        SpawnBatch { particles }
    }
}

// ---------------------------------------------------------------------------
// Event -> emitter bindings (bd-2z0.14.1.7.2): tick-exact cue queue,
// per-family rate limiting, boost trail emitters.
// ---------------------------------------------------------------------------

/// One cue queued for a specific tick.
#[derive(Debug, Clone)]
struct QueuedCue {
    /// Arrival order within the tick (stable tiebreak).
    ordinal: u32,
    /// The cue recipe.
    cue: VisualCue,
    /// World-space spawn position.
    position: [f32; 3],
}

/// Per-tick spawn statistics (observability for storms).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmitterTickStats {
    /// Cues accepted for spawning.
    pub cues_spawned: u32,
    /// Cues dropped by rate limits.
    pub cues_dropped: u32,
    /// Particles actually spawned (post-cap).
    pub particles_spawned: u32,
    /// Trail particles emitted this tick.
    pub trail_particles: u32,
}

/// One boost trail emitter bound to an agent anchor.
#[derive(Debug, Clone)]
struct TrailState {
    anchor: AgentAnchor,
    color: [f32; 3],
    accent: [f32; 3],
    interval_ticks: u32,
    last_emit_tick: u64,
    /// Particle lifetime; trail length = duration / interval, bounded.
    duration_ticks: u32,
}

/// The event->emitter binding: maps the world's typed [`VisualCue`] stream
/// into deterministic particle spawns.
///
/// Contracts:
/// - Tick-exact: cues queue for their tick and are only applied by
///   [`Self::apply_tick`] for that exact tick — never wall-clock, never
///   early.
/// - Rate limiting: when a tick exceeds budget, cues are selected by
///   (priority desc, arrival ordinal asc) — combat/death (Critical) always
///   win over ambience; drops are counted in [`EmitterTickStats`].
/// - Determinism: identical (seed, cue stream, positions) produce identical
///   particle tables across runs and platforms; proof via
///   [`ParticlePool::state_hash`].
#[derive(Debug, Clone)]
pub struct CueEmitter {
    scheduler: CueScheduler,
    queue_tick: Option<u64>,
    queue: Vec<QueuedCue>,
    next_ordinal: u32,
    trails: Vec<TrailState>,
    /// Max cues whose batches spawn per tick per priority class.
    pub budget_per_class_per_tick: u32,
    /// Max cues whose batches spawn per tick overall.
    pub budget_global_per_tick: u32,
}

impl CueEmitter {
    /// Default per-class per-tick cue budget (combat storms shed ambience
    /// first, then standard, but critical cues are never class-capped below
    /// this budget).
    pub const DEFAULT_CLASS_BUDGET: u32 = 32;
    /// Default global per-tick cue budget.
    pub const DEFAULT_GLOBAL_BUDGET: u32 = 96;
    /// Default trail emission interval (ticks between trail puffs).
    pub const TRAIL_INTERVAL_TICKS: u32 = 3;
    /// Default trail particle lifetime; trail length = 24/3 = 8 puffs.
    pub const TRAIL_DURATION_TICKS: u32 = 24;

    /// Create an emitter for a world seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            scheduler: CueScheduler::new(seed),
            queue_tick: None,
            queue: Vec::new(),
            next_ordinal: 0,
            trails: Vec::new(),
            budget_per_class_per_tick: Self::DEFAULT_CLASS_BUDGET,
            budget_global_per_tick: Self::DEFAULT_GLOBAL_BUDGET,
        }
    }

    /// Queue a cue for `tick` at a world position. Cues for different ticks
    /// must not interleave: enqueue is append-only within one tick and a new
    /// tick implicitly seals the previous queue (fail-closed via stats if
    /// [`Self::apply_tick`] was skipped).
    pub fn enqueue(&mut self, tick: u64, cue: VisualCue, position: [f32; 3]) {
        if self.queue_tick != Some(tick) {
            self.queue.clear();
            self.queue_tick = Some(tick);
            self.next_ordinal = 0;
        }
        let ordinal = self.next_ordinal;
        self.next_ordinal = self.next_ordinal.wrapping_add(1);
        self.queue.push(QueuedCue {
            ordinal,
            cue,
            position,
        });
    }

    /// Drain the queue for `tick` into `pool`, applying rate limits, then
    /// emit due boost-trail particles. Returns per-tick statistics.
    #[allow(clippy::cast_possible_truncation)]
    pub fn apply_tick(&mut self, tick: u64, pool: &mut ParticlePool) -> EmitterTickStats {
        let mut stats = EmitterTickStats::default();
        if self.queue_tick == Some(tick) && !self.queue.is_empty() {
            // Stable priority selection: Critical first, arrival order inside.
            let mut queued = std::mem::take(&mut self.queue);
            queued.sort_by(|a, b| {
                CueScheduler::priority_for(b.cue.kind)
                    .cmp(&CueScheduler::priority_for(a.cue.kind))
                    .then(a.ordinal.cmp(&b.ordinal))
            });
            let mut class_counts = [0_u32; 3];
            for entry in queued {
                let class = CueScheduler::priority_for(entry.cue.kind) as usize;
                if stats.cues_spawned >= self.budget_global_per_tick
                    || class_counts[class] >= self.budget_per_class_per_tick
                {
                    stats.cues_dropped += 1;
                    continue;
                }
                class_counts[class] += 1;
                stats.cues_spawned += 1;
                let batch = self
                    .scheduler
                    .schedule(tick, entry.ordinal, &entry.cue, entry.position);
                for particle in batch.particles {
                    if pool.spawn(particle).is_some() {
                        stats.particles_spawned += 1;
                    }
                }
            }
        } else {
            // Stale or missing tick: cues queue for their tick and expire
            // unapplied rather than firing late (tick-exactness).
            stats.cues_dropped += self.queue.len() as u32;
            self.queue.clear();
            self.queue_tick = None;
        }
        stats
    }

    /// Bind a boost trail emitter to an agent anchor (one per agent; binding
    /// again refreshes colors). The emitter follows the agent via the
    /// `lookup` passed to [`Self::emit_trails`]; emitted puffs are static
    /// (they stay where the agent WAS — a trail, not an aura).
    pub fn bind_boost_trail(&mut self, anchor: AgentAnchor, color: [f32; 3], accent: [f32; 3]) {
        if let Some(existing) = self.trails.iter_mut().find(|t| t.anchor.uid == anchor.uid) {
            existing.color = color;
            existing.accent = accent;
            existing.anchor = anchor;
            return;
        }
        self.trails.push(TrailState {
            anchor,
            color,
            accent,
            interval_ticks: Self::TRAIL_INTERVAL_TICKS,
            last_emit_tick: 0,
            duration_ticks: Self::TRAIL_DURATION_TICKS,
        });
    }

    /// Detach every trail bound to `uid` (agent died or stopped boosting).
    pub fn detach_trails_for(&mut self, uid: u64) {
        self.trails.retain(|trail| trail.anchor.uid != uid);
    }

    /// Number of live trail bindings.
    #[must_use]
    pub fn trail_count(&self) -> usize {
        self.trails.len()
    }

    /// Emit due trail puffs for `tick`. `lookup` resolves an anchor UID to
    /// the agent's current world position; a dead agent detaches its trail
    /// (same semantics as [`ParticlePool::update_anchors`]).
    #[allow(clippy::cast_possible_truncation)]
    pub fn emit_trails(
        &mut self,
        tick: u64,
        pool: &mut ParticlePool,
        mut lookup: impl FnMut(u64) -> Option<[f32; 3]>,
    ) -> u32 {
        let mut emitted = 0_u32;
        let mut detached = Vec::new();
        for (index, trail) in self.trails.iter_mut().enumerate() {
            if tick < trail.last_emit_tick + u64::from(trail.interval_ticks) {
                continue;
            }
            let Some(position) = lookup(trail.anchor.uid) else {
                detached.push(index);
                continue;
            };
            trail.last_emit_tick = tick;
            // Deterministic per-trail jitter through the scheduler's hash.
            let cue = VisualCue {
                kind: VisualCueKind::Nibble,
                color: trail.color,
                accent_color: trail.accent,
                intensity: 0.5,
                radius: 2.0,
                duration_ticks: trail.duration_ticks,
            };
            let batch = self.scheduler.schedule(tick, index as u32, &cue, [
                position[0] + trail.anchor.offset[0],
                position[1] + trail.anchor.offset[1],
                position[2] + trail.anchor.offset[2],
            ]);
            // One puff per emission: the trail is a breadcrumb, not a burst.
            if let Some(particle) = batch.particles.into_iter().next()
                && pool.spawn(particle).is_some()
            {
                emitted += 1;
            }
        }
        for index in detached.into_iter().rev() {
            self.trails.remove(index);
        }
        emitted
    }
}

impl ParticlePool {
    /// Deterministic state fingerprint for replay-equivalence proofs: FNV-1a64
    /// over every slot's occupancy, generation, and live particle fields in
    /// slot order. Two pools with equal hashes hold equivalent state.
    #[must_use]
    pub fn state_hash(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut absorb_u64 = |value: u64| {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for (slot, generation) in self.slots.iter().zip(self.generations.iter()) {
            absorb_u64(u64::from(*generation));
            match slot {
                None => absorb_u64(0),
                Some(particle) => {
                    absorb_u64(1);
                    for value in [
                        particle.position[0].to_bits(),
                        particle.position[1].to_bits(),
                        particle.position[2].to_bits(),
                        particle.velocity[0].to_bits(),
                        particle.velocity[1].to_bits(),
                        particle.velocity[2].to_bits(),
                        particle.gravity.to_bits(),
                        particle.drag.to_bits(),
                    ] {
                        absorb_u64(u64::from(value));
                    }
                    absorb_u64(particle.born_tick);
                    absorb_u64(u64::from(particle.duration_ticks));
                }
            }
        }
        hash
    }
}

// ---------------------------------------------------------------------------
// Deterministic sprite atlas (SDF-authored, byte-identical across runs).
// ---------------------------------------------------------------------------

/// Signed distance of a four-point star (Spark).
fn sdf_star(x: f32, y: f32) -> f32 {
    const SQRT2_INV: f32 = core::f32::consts::FRAC_1_SQRT_2;
    let ax = x.abs();
    let ay = y.abs();
    let d_cross = (ax.min(ay) - 0.12).max(ax.max(ay) - 0.5);
    let d_diag = (ax + ay)
        .mul_add(SQRT2_INV, -0.35)
        .max((ax - ay).abs().mul_add(SQRT2_INV, -0.08));
    d_cross.min(d_diag)
}

/// Sprite alpha for a tile at normalized coordinates in `[-1, 1]`.
fn sprite_alpha(kind: SpriteKind, x: f32, y: f32, seed: u64) -> f32 {
    let dist = match kind {
        SpriteKind::Spark => sdf_star(x, y),
        SpriteKind::Mote => x.hypot(y) - 0.45,
        SpriteKind::Shard => {
            // Jagged triangle: half-plane intersection with noise-chipped edges.
            let tri = (y + 0.5).max(-x - 0.4).max(x - 0.4);
            let chip = value_noise_2d(seed ^ 0x51A7, x * 6.0, y * 6.0) * 0.15;
            tri + chip
        }
        SpriteKind::Puff => {
            let base = x.hypot(y) - 0.4;
            let wobble = value_noise_2d(seed ^ 0xB10B, x * 3.0, y * 3.0) * 0.25;
            base + wobble
        }
        SpriteKind::Ring => (x.hypot(y) - 0.38).abs() - 0.12,
    };
    // Smooth 2-pixel edge falloff in normalized space.
    (0.5 - dist * 2.5).clamp(0.0, 1.0)
}

/// Bake the deterministic sprite atlas: five `size x size` RGBA8 tiles laid
/// out side by side (`size * 5 x size` bytes total). Identical `(seed, size)`
/// produce byte-identical output on every platform.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
#[must_use]
pub fn bake_sprite_atlas(seed: u64, size: u32) -> Vec<u8> {
    // Pixel coordinates are bounded by `size` (atlas texels), so the u32->f32
    // and alpha->u8 casts are exact within the atlas domain.
    let size = size.max(1);
    let tile_px = usize::try_from(size).unwrap_or(1);
    let mut atlas = vec![0_u8; tile_px * 5 * tile_px * 4];
    for (tile, kind) in SpriteKind::ALL.iter().enumerate() {
        for y in 0..size {
            for x in 0..size {
                let nx = ((x as f32 + 0.5) / size as f32).mul_add(2.0, -1.0);
                let ny = ((y as f32 + 0.5) / size as f32).mul_add(2.0, -1.0);
                let alpha = sprite_alpha(*kind, nx, ny, seed);
                let offset = ((y as usize) * tile_px * 5 + tile * tile_px + x as usize) * 4;
                // White sprite body; tinting happens per-instance in the shader.
                atlas[offset] = 255;
                atlas[offset + 1] = 255;
                atlas[offset + 2] = 255;
                atlas[offset + 3] = alpha.mul_add(255.0, 0.5) as u8;
            }
        }
    }
    atlas
}

// ---------------------------------------------------------------------------
// WGSL billboard shader source. Structurally validated here (balanced
// braces, required entry points); compiled for real by the integration bead
// (bd-2z0.14.1.7) which owns the render-world plumbing.
// ---------------------------------------------------------------------------

/// The instanced billboard particle shader.
///
/// Per-instance data matches [`Particle`]'s GPU projection: spawn position,
/// velocity, gravity/drag, born tick + duration, size curve, rotation/spin,
/// colors, intensity, sprite tile. Motion integrates from spawn attributes
/// against the tick uniform — the same math as [`ParticlePool::integrate_tick`].
pub const PARTICLE_WGSL: &str = r"
struct ParticleInstance {
    pos0: vec3<f32>,
    vel0: vec3<f32>,
    gravity: f32,
    drag: f32,
    born_tick: f32,
    duration: f32,
    size_start: f32,
    size_end: f32,
    rotation: f32,
    spin: f32,
    color: vec3<f32>,
    accent: vec3<f32>,
    intensity: f32,
    sprite: f32,
    flags: f32,
};

struct ParticleUniforms {
    tick: f32,
    atlas_cols: f32,
    // Soft-particle fade: alpha scales with clamp((scene_depth -
    // particle_depth) * soft_fade_scale, 0, 1). Large values approximate a
    // hard depth cut; the Low tier uses this instead of sort order.
    soft_fade_scale: f32,
    _pad: f32,
    viewport: vec2<f32>,
    // Camera basis in world space: billboards orient to the camera, never
    // to a fixed world plane.
    camera_right: vec3<f32>,
    _pad2: f32,
    camera_up: vec3<f32>,
    _pad3: f32,
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: ParticleUniforms;
@group(0) @binding(1) var<storage, read> instances: array<ParticleInstance>;
@group(0) @binding(2) var atlas_tex: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;
@group(0) @binding(4) var scene_depth: texture_depth_2d;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) accent: vec3<f32>,
    @location(3) intensity: f32,
};

@vertex
fn vs_particle(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let p = instances[instance_index];
    let age = max(uniforms.tick - p.born_tick, 0.0);
    let life = clamp(age / max(p.duration, 1.0), 0.0, 1.0);
    // Same integration as the CPU path: vel(t) = vel0 * drag^t + g * t (approx,
    // closed form uses the geometric-series factor for exactness).
    let damping = 1.0 - clamp(p.drag, 0.0, 1.0);
    let drag_t = pow(damping, age);
    let drag_integral = select((1.0 - drag_t) / max(1.0 - damping, 1e-5), age, damping >= 1.0);
    let velocity_now = p.vel0 * drag_t + vec3<f32>(0.0, -p.gravity, 0.0) * age;
    let position = p.pos0 + p.vel0 * drag_integral + vec3<f32>(0.0, -0.5 * p.gravity, 0.0) * age * age;

    let corner = vec2<f32>(
        f32((vertex_index & 1u) * 2u) - 1.0,
        f32((vertex_index >> 1u) * 2u) - 1.0,
    );
    let size = mix(p.size_start, p.size_end, life);
    let rot = p.rotation + p.spin * age;
    let c = cos(rot);
    let s = sin(rot);
    let offset = vec2<f32>(
        corner.x * c - corner.y * s,
        corner.x * s + corner.y * c,
    ) * size;

    // Camera-facing billboard: rotate the corner offset into the camera's
    // world-space basis so the quad always fronts the viewer.
    let world = position + uniforms.camera_right * offset.x + uniforms.camera_up * offset.y;

    var out: VertexOutput;
    out.clip = uniforms.view_proj * vec4<f32>(world, 1.0);
    // Atlas UVs: tile column from sprite index, quad corner within the tile.
    let corner01 = corner * 0.5 + vec2<f32>(0.5, 0.5);
    out.uv = vec2<f32>((p.sprite + corner01.x) / uniforms.atlas_cols, corner01.y);
    out.color = mix(p.color, p.accent, life);
    out.accent = p.accent;
    out.intensity = p.intensity * (1.0 - life * life);
    return out;
}

@fragment
fn fs_particle(in: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(atlas_tex, atlas_sampler, in.uv);
    var alpha = sample.a * clamp(in.intensity, 0.0, 1.0);
    // Soft particles: fade the billboard where it approaches scene geometry.
    // @builtin(position) in the fragment stage carries framebuffer pixel
    // coordinates (xy) and the NDC depth (z) of this fragment.
    let pixel = vec2<i32>(floor(in.clip.xy));
    let scene = textureLoad(scene_depth, pixel, 0);
    let fade = clamp((scene - in.clip.z) * uniforms.soft_fade_scale, 0.0, 1.0);
    alpha *= fade;
    if (alpha < 0.01) {
        discard;
    }
    return vec4<f32>(in.color * sample.rgb, alpha);
}
";

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::visual::{VisualCue, VisualCueKind};

    fn cue(kind: VisualCueKind) -> VisualCue {
        VisualCue {
            kind,
            color: [1.0, 0.8, 0.4],
            accent_color: [0.6, 0.9, 1.0],
            intensity: 0.8,
            radius: 5.0,
            duration_ticks: 20,
        }
    }

    fn particle(priority: ParticlePriority) -> Particle {
        Particle {
            position: [0.0; 3],
            velocity: [0.0; 3],
            gravity: 0.0,
            drag: 0.0,
            born_tick: 0,
            duration_ticks: 10,
            size_start: 1.0,
            size_end: 0.1,
            rotation: 0.0,
            spin: 0.0,
            color: [1.0; 3],
            accent: [1.0; 3],
            intensity: 1.0,
            sprite: SpriteKind::Spark,
            priority,
            follow: None,
        }
    }

    #[test]
    fn pool_spawn_kill_reuse_and_live_count() {
        let mut pool = ParticlePool::with_capacity(3);
        let a = pool.spawn(particle(ParticlePriority::Standard)).expect("a");
        let b = pool.spawn(particle(ParticlePriority::Standard)).expect("b");
        let c = pool.spawn(particle(ParticlePriority::Standard)).expect("c");
        assert_eq!(pool.live_count(), 3);
        assert!(
            pool.spawn(particle(ParticlePriority::Ambient)).is_none(),
            "full pool rejects ambient"
        );
        pool.kill(b);
        assert_eq!(pool.live_count(), 2);
        assert!(pool.get(b).is_none(), "killed handle is stale");
        let d = pool.spawn(particle(ParticlePriority::Standard)).expect("d");
        assert_eq!(d.slot, b.slot, "slot is reused");
        assert_ne!(d.generation, b.generation, "generation bumps on reuse");
        assert!(pool.get(a).is_some());
        assert!(pool.get(c).is_some());
    }

    #[test]
    fn pool_overflow_evicts_oldest_ambient_but_never_critical() {
        let mut pool = ParticlePool::with_capacity(2);
        let ambient = pool
            .spawn(particle(ParticlePriority::Ambient))
            .expect("ambient");
        let _standard = pool
            .spawn(particle(ParticlePriority::Standard))
            .expect("standard");
        let critical = pool
            .spawn(particle(ParticlePriority::Critical))
            .expect("critical spawn");
        assert!(pool.get(ambient).is_none(), "ambient evicted for critical");
        assert!(pool.get(critical).is_some());
        // Pool full of Standard+Critical: a new Critical is rejected (nothing evictable).
        let rejected = pool.spawn(particle(ParticlePriority::Critical));
        assert!(rejected.is_none(), "no evictable ambient -> rejection");
        // DropNewest policy rejects everything at capacity.
        let mut pool2 = ParticlePool::with_capacity(1);
        pool2.overflow = OverflowPolicy::DropNewest;
        let _ = pool2
            .spawn(particle(ParticlePriority::Ambient))
            .expect("first");
        assert!(pool2.spawn(particle(ParticlePriority::Critical)).is_none());
    }

    #[test]
    fn pool_ambient_into_full_pool_of_ambients_loses_without_churn() {
        let mut pool = ParticlePool::with_capacity(2);
        let a = pool.spawn(particle(ParticlePriority::Ambient)).expect("a");
        let b = pool.spawn(particle(ParticlePriority::Ambient)).expect("b");
        assert!(pool.spawn(particle(ParticlePriority::Ambient)).is_none());
        assert!(
            pool.get(a).is_some() && pool.get(b).is_some(),
            "no churn for ambient"
        );
    }

    #[test]
    fn pool_expiry_hits_exact_tick_boundary() {
        let mut pool = ParticlePool::with_capacity(4);
        let born_at_five = Particle {
            born_tick: 5,
            duration_ticks: 10,
            ..particle(ParticlePriority::Standard)
        };
        let _ = pool.spawn(born_at_five);
        assert_eq!(pool.expire(14), 0, "alive at born+duration-1");
        assert_eq!(pool.expire(15), 1, "expires at born+duration");
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn pool_anchor_follow_and_detach() {
        let mut pool = ParticlePool::with_capacity(2);
        let mut p = particle(ParticlePriority::Standard);
        p.follow = Some(AgentAnchor {
            uid: 42,
            offset: [1.0, 0.0, 0.0],
        });
        let handle = pool.spawn(p).expect("spawn");
        pool.update_anchors(|uid| (uid == 42).then_some([10.0, 0.0, 0.0]));
        assert_eq!(pool.get(handle).expect("live").position, [11.0, 0.0, 0.0]);
        pool.update_anchors(|_uid| None);
        assert!(
            pool.get(handle).expect("live").follow.is_none(),
            "dead agent detaches trail"
        );
    }

    #[test]
    fn pool_integrate_matches_documented_motion() {
        let mut pool = ParticlePool::with_capacity(1);
        let mut p = particle(ParticlePriority::Standard);
        p.velocity = [1.0, 2.0, 0.0];
        p.gravity = 0.5;
        p.drag = 0.5;
        let handle = pool.spawn(p).expect("spawn");
        pool.integrate_tick();
        let after = pool.get(handle).expect("live");
        // vel.y -= 0.5 -> 1.5; then damping 0.5: [0.5, 0.75, 0]
        assert!((after.velocity[0] - 0.5).abs() < 1e-6);
        assert!((after.velocity[1] - 0.75).abs() < 1e-6);
        assert!((after.position[0] - 0.5).abs() < 1e-6);
        assert!((after.position[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn scheduler_is_deterministic_and_ordinal_decorrelated() {
        let scheduler = CueScheduler::new(99);
        let c = cue(VisualCueKind::Sparkle);
        let a = scheduler.schedule(1000, 0, &c, [1.0, 2.0, 3.0]);
        let b = scheduler.schedule(1000, 0, &c, [1.0, 2.0, 3.0]);
        assert_eq!(
            a.particles, b.particles,
            "identical inputs, identical batch"
        );
        let d = scheduler.schedule(1000, 1, &c, [1.0, 2.0, 3.0]);
        assert_ne!(a.particles, d.particles, "different ordinals decorrelate");
        // Velocity streams decorrelate across ordinals (sanity, not statistics).
        let differing = (0..16)
            .map(|ordinal| scheduler.schedule(1000, ordinal, &c, [0.0; 3]))
            .map(|batch| {
                let v = batch.particles[0].velocity;
                (v[0].to_bits(), v[1].to_bits(), v[2].to_bits())
            })
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert!(differing > 8, "velocity streams decorrelate: {differing}");
    }

    #[test]
    fn scheduler_count_scales_with_intensity_and_respects_cap() {
        let scheduler = CueScheduler::new(1);
        let low = VisualCue {
            intensity: 0.0,
            ..cue(VisualCueKind::Sparkle)
        };
        let high = VisualCue {
            intensity: 1.0,
            ..cue(VisualCueKind::Sparkle)
        };
        let low_count = scheduler.count_for_cue(&low);
        let high_count = scheduler.count_for_cue(&high);
        assert!(
            high_count > low_count,
            "intensity scales count: {low_count} vs {high_count}"
        );
        let crazy = VisualCue {
            intensity: 1.0,
            ..cue(VisualCueKind::Shards)
        };
        assert!(scheduler.count_for_cue(&crazy) <= CueScheduler::MAX_PER_CUE);
    }

    #[test]
    fn scheduler_particles_are_sane() {
        let scheduler = CueScheduler::new(7);
        for kind in [
            VisualCueKind::Sparkle,
            VisualCueKind::Shards,
            VisualCueKind::Wilt,
            VisualCueKind::Nibble,
            VisualCueKind::SparkCone,
            VisualCueKind::PulseRing,
            VisualCueKind::Flash,
        ] {
            let batch = scheduler.schedule(50, 0, &cue(kind), [0.0; 3]);
            assert!(!batch.particles.is_empty());
            for particle in &batch.particles {
                for v in particle.velocity {
                    assert!(v.is_finite());
                }
                assert!(particle.duration_ticks >= 2);
                assert!((0.0..=1.0).contains(&particle.intensity));
            }
        }
    }

    #[test]
    fn sprite_atlas_is_deterministic_and_shaped() {
        let a = bake_sprite_atlas(5, 32);
        let b = bake_sprite_atlas(5, 32);
        assert_eq!(a, b, "byte-identical across bakes");
        assert_eq!(a.len(), 32 * 5 * 32 * 4);
        // Every tile has some coverage.
        for tile in 0..5 {
            let mut covered = 0usize;
            for y in 0..32usize {
                for x in 0..32usize {
                    let offset = (y * 32 * 5 + tile * 32 + x) * 4;
                    if a[offset + 3] > 0 {
                        covered += 1;
                    }
                }
            }
            assert!(
                covered > 100,
                "tile {tile} has meaningful coverage: {covered}"
            );
        }
        // Ring tile has a transparent center.
        let center = (16 * 32 * 5 + 4 * 32 + 16) * 4;
        assert_eq!(a[center + 3], 0, "ring center is transparent");
    }

    #[test]
    fn golden_cue_particle_count_vectors_per_recipe() {
        // The recipe contract: 4 + intensity * 20, clamped to [1, max_per_cue].
        let scheduler = CueScheduler::new(7);
        let golden: [(f32, u32); 5] = [
            (0.0, 4),
            (0.25, 9),
            (0.5, 14),
            (0.75, 19),
            (1.0, 24),
        ];
        for kind in [
            VisualCueKind::Sparkle,
            VisualCueKind::Shards,
            VisualCueKind::Wilt,
            VisualCueKind::Nibble,
            VisualCueKind::SparkCone,
            VisualCueKind::PulseRing,
            VisualCueKind::Flash,
        ] {
            for (intensity, expected) in golden {
                let mut c = cue(kind);
                c.intensity = intensity;
                assert_eq!(
                    scheduler.count_for_cue(&c),
                    expected,
                    "{kind:?} at intensity {intensity}"
                );
            }
        }
        // Extreme intensity still respects the per-cue cap.
        let mut storm = cue(VisualCueKind::Shards);
        storm.intensity = 99.0;
        assert!(scheduler.count_for_cue(&storm) <= CueScheduler::MAX_PER_CUE);
    }

    #[test]
    fn emitter_is_tick_exact_and_seals_stale_queues() {
        let mut emitter = CueEmitter::new(11);
        let mut pool = ParticlePool::with_capacity(64);
        emitter.enqueue(5, cue(VisualCueKind::Sparkle), [0.0; 3]);
        // Wrong tick: nothing spawns; the stale queue is sealed and dropped.
        let stats = emitter.apply_tick(4, &mut pool);
        assert_eq!(stats.particles_spawned, 0, "cues never fire early");
        assert_eq!(stats.cues_dropped, 1, "stale cue counted");
        // Right tick on a fresh enqueue: fires exactly then.
        emitter.enqueue(5, cue(VisualCueKind::Sparkle), [0.0; 3]);
        let stats = emitter.apply_tick(5, &mut pool);
        assert_eq!(stats.cues_spawned, 1);
        assert!(stats.particles_spawned >= 4, "recipe count applied");
        // Late apply of the same tick after a new tick started: dropped.
        emitter.enqueue(6, cue(VisualCueKind::Wilt), [0.0; 3]);
        let stats = emitter.apply_tick(7, &mut pool);
        assert_eq!(stats.particles_spawned, 0);
        assert_eq!(stats.cues_dropped, 1);
    }

    #[test]
    fn emitter_rate_limits_by_priority_then_ordinal() {
        let mut emitter = CueEmitter::new(13);
        emitter.budget_per_class_per_tick = 1;
        emitter.budget_global_per_tick = 2;
        let mut pool = ParticlePool::with_capacity(512);
        // Arrival order: ambient, standard, critical — over budget after two.
        emitter.enqueue(9, cue(VisualCueKind::Nibble), [0.0; 3]);
        emitter.enqueue(9, cue(VisualCueKind::Wilt), [0.0; 3]);
        emitter.enqueue(9, cue(VisualCueKind::Sparkle), [0.0; 3]);
        emitter.enqueue(9, cue(VisualCueKind::Wilt), [0.0; 3]);
        let stats = emitter.apply_tick(9, &mut pool);
        assert_eq!(stats.cues_spawned, 2, "global budget binds");
        assert_eq!(stats.cues_dropped, 2);
        // Determinism of the selection: replay the same stream.
        let mut emitter_b = CueEmitter::new(13);
        emitter_b.budget_per_class_per_tick = 1;
        emitter_b.budget_global_per_tick = 2;
        let mut pool_b = ParticlePool::with_capacity(512);
        emitter_b.enqueue(9, cue(VisualCueKind::Nibble), [0.0; 3]);
        emitter_b.enqueue(9, cue(VisualCueKind::Wilt), [0.0; 3]);
        emitter_b.enqueue(9, cue(VisualCueKind::Sparkle), [0.0; 3]);
        emitter_b.enqueue(9, cue(VisualCueKind::Wilt), [0.0; 3]);
        let stats_b = emitter_b.apply_tick(9, &mut pool_b);
        assert_eq!(stats, stats_b);
        assert_eq!(pool.state_hash(), pool_b.state_hash(), "identical selection");
        // Critical won over ambient: the spawned batch was sparkle-tagged.
        let mut saw_spark = false;
        pool.for_each_live(|particle| {
            if particle.sprite == SpriteKind::Spark {
                saw_spark = true;
            }
        });
        assert!(saw_spark, "critical cue outranked ambience under budget");
    }

    #[test]
    fn emitter_is_deterministic_across_replays() {
        let stream = [
            (100_u64, VisualCueKind::Sparkle, [1.0, 2.0, 3.0]),
            (100, VisualCueKind::Shards, [4.0, 5.0, 6.0]),
            (101, VisualCueKind::Wilt, [7.0, 8.0, 9.0]),
            (101, VisualCueKind::Nibble, [10.0, 11.0, 12.0]),
            (102, VisualCueKind::PulseRing, [0.0, 0.0, 0.0]),
        ];
        let run = || {
            let mut emitter = CueEmitter::new(42);
            let mut pool = ParticlePool::with_capacity(1024);
            for tick in 100..=102 {
                for (at, kind, pos) in &stream {
                    if *at == tick {
                        emitter.enqueue(tick, cue(*kind), *pos);
                    }
                }
                emitter.apply_tick(tick, &mut pool);
                pool.integrate_tick();
                pool.expire(tick);
            }
            pool.state_hash()
        };
        assert_eq!(run(), run(), "identical cue stream -> identical pool state hash");
        // A different stream must diverge (the hash actually observes state).
        let divergent = || {
            let mut emitter = CueEmitter::new(42);
            let mut pool = ParticlePool::with_capacity(1024);
            emitter.enqueue(100, cue(VisualCueKind::Sparkle), [9.0, 9.0, 9.0]);
            emitter.apply_tick(100, &mut pool);
            pool.state_hash()
        };
        assert_ne!(run(), divergent(), "different streams produce different states");
    }

    #[test]
    fn boost_trail_follows_a_moving_agent_and_stays_bounded() {
        let mut emitter = CueEmitter::new(17);
        let mut pool = ParticlePool::with_capacity(256);
        emitter.bind_boost_trail(
            AgentAnchor {
                uid: 99,
                offset: [0.0, 0.5, 0.0],
            },
            [0.2, 0.6, 1.0],
            [0.8, 0.9, 1.0],
        );
        assert_eq!(emitter.trail_count(), 1);
        // Agent moves +1 x per tick; trail puffs appear at successive spots.
        let mut emitted_total = 0_u32;
        for tick in 0..=u64::from(CueEmitter::TRAIL_DURATION_TICKS) {
            let x = tick as f32;
            emitted_total += emitter.emit_trails(tick, &mut pool, |uid| {
                (uid == 99).then_some([x, 0.0, 0.0])
            });
        }
        let expected = CueEmitter::TRAIL_DURATION_TICKS / CueEmitter::TRAIL_INTERVAL_TICKS;
        assert_eq!(emitted_total, expected, "one puff per interval");
        // Agent dies: lookup yields None -> trail detaches and stops.
        let emitted_after_death = emitter.emit_trails(999, &mut pool, |_| None);
        assert_eq!(emitted_after_death, 0);
        assert_eq!(emitter.trail_count(), 0, "dead agent's trail detached");
        // Rebinding refreshes instead of duplicating.
        let anchor = AgentAnchor {
            uid: 7,
            offset: [0.0; 3],
        };
        emitter.bind_boost_trail(anchor, [1.0; 3], [1.0; 3]);
        emitter.bind_boost_trail(anchor, [0.5; 3], [0.5; 3]);
        assert_eq!(emitter.trail_count(), 1, "rebind refreshes");
        emitter.detach_trails_for(7);
        assert_eq!(emitter.trail_count(), 0);
    }

    #[test]
    fn state_hash_distinguishes_pool_states() {
        let mut a = ParticlePool::with_capacity(4);
        let b = ParticlePool::with_capacity(4);
        assert_eq!(a.state_hash(), b.state_hash(), "empty pools equal");
        a.spawn(particle(ParticlePriority::Standard)).expect("spawn");
        assert_ne!(a.state_hash(), b.state_hash(), "occupancy is observed");
        let handle = a.spawn(particle(ParticlePriority::Critical)).expect("spawn");
        let before = a.state_hash();
        a.kill(handle);
        assert_ne!(a.state_hash(), before, "kill changes generations");
    }

    #[test]
    fn wgsl_source_has_entry_points_and_balanced_braces() {
        assert!(PARTICLE_WGSL.contains("@vertex"));
        assert!(PARTICLE_WGSL.contains("@fragment"));
        // Camera-facing billboards + soft depth fade (bd-2z0.14.1.7.1 contract).
        assert!(PARTICLE_WGSL.contains("camera_right"));
        assert!(PARTICLE_WGSL.contains("camera_up"));
        assert!(PARTICLE_WGSL.contains("scene_depth"));
        assert!(PARTICLE_WGSL.contains("soft_fade_scale"));
        let mut depth = 0i32;
        for ch in PARTICLE_WGSL.chars() {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            assert!(depth >= 0, "unbalanced closing brace");
        }
        assert_eq!(depth, 0, "balanced braces");
    }

    #[test]
    fn wgsl_source_compiles_and_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(PARTICLE_WGSL)
            .expect("PARTICLE_WGSL must parse as valid WGSL");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .expect("PARTICLE_WGSL must pass naga semantic validation");
        // The instance layout consumed by the shader matches ParticleInstance.
        assert_eq!(
            core::mem::size_of::<ParticleInstance>(),
            23 * 4,
            "WGSL ParticleInstance layout drift from Rust projection"
        );
    }

    #[test]
    fn instance_buffer_fill_is_slot_ordered_and_grow_only() {
        let mut pool = ParticlePool::with_capacity(4);
        let a = pool.spawn(particle(ParticlePriority::Standard)).expect("a");
        let b = pool.spawn(particle(ParticlePriority::Critical)).expect("b");
        let _c = pool.spawn(particle(ParticlePriority::Ambient)).expect("c");
        pool.kill(b);
        let d = pool
            .spawn(particle(ParticlePriority::Standard))
            .expect("d reuses slot");
        let mut out = Vec::new();
        pool.write_instance_buffer(&mut out);
        assert_eq!(out.len(), 3);
        // Slot order: a (slot 0), d (slot 1, reused), c (slot 2).
        let slots = [a.slot, d.slot, _c.slot];
        assert!(slots.windows(2).all(|w| w[0] < w[1]), "slot-ordered fill");
        assert_eq!(out[0].duration, 10.0);
        assert_eq!(out[1].flags, 0.0);
        let capacity_after_first_fill = out.capacity();
        pool.write_instance_buffer(&mut out);
        assert_eq!(
            out.capacity(),
            capacity_after_first_fill,
            "grow-only buffer"
        );
    }

    #[test]
    fn blend_order_unsorted_is_identity_and_sorted_is_back_to_front() {
        let depths = [(0, 10.0), (1, 30.0), (2, 20.0), (3, f32::NAN)];
        let mut order = Vec::new();
        let mut scratch = Vec::new();
        blend_order_into(
            &depths,
            BlendOrderPolicy::Unsorted,
            &mut order,
            &mut scratch,
        );
        assert_eq!(order, vec![0, 1, 2, 3], "unsorted = slot order");
        blend_order_into(
            &depths,
            BlendOrderPolicy::BucketSort,
            &mut order,
            &mut scratch,
        );
        assert_eq!(
            order,
            vec![1, 2, 0, 3],
            "back-to-front; NaN lands in the nearest bucket (drawn last)"
        );
    }

    #[test]
    fn blend_order_is_stable_deterministic_and_handles_ties() {
        let depths = [(0, 5.0), (1, 5.0), (2, 5.0), (3, 9.0), (4, 1.0)];
        let mut first = Vec::new();
        let mut second = Vec::new();
        let mut scratch = Vec::new();
        blend_order_into(
            &depths,
            BlendOrderPolicy::BucketSort,
            &mut first,
            &mut scratch,
        );
        blend_order_into(
            &depths,
            BlendOrderPolicy::BucketSort,
            &mut second,
            &mut scratch,
        );
        assert_eq!(first, second, "deterministic across calls");
        assert_eq!(
            first,
            vec![3, 0, 1, 2, 4],
            "ties keep slot order within a bucket"
        );
        // Uniform depths: slot order (no reshuffle).
        let uniform = [(0, 7.0), (1, 7.0), (2, 7.0)];
        blend_order_into(
            &uniform,
            BlendOrderPolicy::BucketSort,
            &mut first,
            &mut scratch,
        );
        assert_eq!(first, vec![0, 1, 2]);
    }
}
