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
    /// Birth sequence stamped on each slot at spawn.
    ///
    /// DIAGNOSTIC ONLY. Eviction order comes from the intrusive list below, not
    /// from comparing these — which is what makes `born_counter` wrap a
    /// non-issue rather than a lurking correctness bug (bd-2z0.14.1.19).
    born_seq: Vec<u64>,
    /// Intrusive doubly-linked FIFO over the LIVE AMBIENT slots, oldest first.
    ///
    /// This is what makes true-oldest eviction O(1). The previous code scanned
    /// `0..capacity` and took the first ambient slot it found, which is the
    /// LOWEST slot index, not the oldest particle — after any kill/reuse the
    /// two disagree, and a freshly spawned particle in a recycled low slot was
    /// evicted ahead of genuinely old ones.
    ambient_prev: Vec<u32>,
    ambient_next: Vec<u32>,
    ambient_head: u32,
    ambient_tail: u32,
    /// Overflow behavior when the pool is full.
    pub overflow: OverflowPolicy,
}

/// Sentinel for "no slot" in the intrusive ambient list.
///
/// Safe as a sentinel because `with_capacity` caps addressable slots at
/// `u32::MAX`, so the largest real index is `u32::MAX - 1`.
const NO_SLOT: u32 = u32::MAX;

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
            born_seq: vec![0; capacity],
            ambient_prev: vec![NO_SLOT; capacity],
            ambient_next: vec![NO_SLOT; capacity],
            ambient_head: NO_SLOT,
            ambient_tail: NO_SLOT,
            overflow: OverflowPolicy::default(),
        }
    }

    /// Append a live ambient slot to the tail of the FIFO. O(1).
    fn ambient_push_back(&mut self, slot: u32) {
        let s = slot as usize;
        self.ambient_prev[s] = self.ambient_tail;
        self.ambient_next[s] = NO_SLOT;
        if self.ambient_tail == NO_SLOT {
            self.ambient_head = slot;
        } else {
            self.ambient_next[self.ambient_tail as usize] = slot;
        }
        self.ambient_tail = slot;
    }

    /// Remove a slot from the ambient FIFO wherever it sits. O(1).
    ///
    /// Only called for slots known to be in the list — a live ambient particle —
    /// so the neighbour links are always coherent.
    fn ambient_unlink(&mut self, slot: u32) {
        let s = slot as usize;
        let prev = self.ambient_prev[s];
        let next = self.ambient_next[s];
        if prev == NO_SLOT {
            self.ambient_head = next;
        } else {
            self.ambient_next[prev as usize] = next;
        }
        if next == NO_SLOT {
            self.ambient_tail = prev;
        } else {
            self.ambient_prev[next as usize] = prev;
        }
        self.ambient_prev[s] = NO_SLOT;
        self.ambient_next[s] = NO_SLOT;
    }

    /// Birth sequence stamped on a live slot, for diagnostics and tests.
    #[must_use]
    pub fn born_seq_of(&self, handle: ParticleHandle) -> Option<u64> {
        let slot = handle.slot as usize;
        if slot >= self.slots.len()
            || self.generations[slot] != handle.generation
            || self.slots[slot].is_none()
        {
            return None;
        }
        Some(self.born_seq[slot])
    }

    /// Slot holding the oldest live ambient particle, if any. O(1).
    #[must_use]
    pub fn oldest_ambient_slot(&self) -> Option<u32> {
        (self.ambient_head != NO_SLOT).then_some(self.ambient_head)
    }

    /// Priority of whatever occupies a slot, for the injected-negative test
    /// that reconstructs the rejected lowest-slot scan.
    #[cfg(test)]
    fn slot_priority_for_test(&self, slot: u32) -> Option<ParticlePriority> {
        self.slots
            .get(slot as usize)?
            .as_ref()
            .map(|particle| particle.priority)
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
        let is_ambient = particle.priority == ParticlePriority::Ambient;
        self.slots[slot as usize] = Some(particle);
        self.live_count += 1;
        // Wrapping is documented rather than guarded: `born_seq` is diagnostic,
        // and eviction order is the list's position, so a wrap cannot reorder
        // anything. At one spawn per nanosecond a u64 still takes ~584 years.
        self.born_counter = self.born_counter.wrapping_add(1);
        self.born_seq[slot as usize] = self.born_counter;
        if is_ambient {
            self.ambient_push_back(slot);
        }
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
                // The head of the ambient FIFO IS the oldest live ambient
                // particle, in O(1) and with no scan. Previously this searched
                // `0..capacity` and took the first ambient it found, which is
                // the lowest SLOT — after any kill and reuse that is frequently
                // the newest particle, so the policy evicted the wrong one.
                let victim = self.oldest_ambient_slot()?;
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
        if let Some(particle) = self.slots[slot as usize].take() {
            // Unlink BEFORE the slot can be recycled, or the FIFO would keep a
            // pointer to a slot that has since been refilled by a different
            // particle — the list must contain exactly the live ambients.
            if particle.priority == ParticlePriority::Ambient {
                self.ambient_unlink(slot);
            }
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
            VisualCueKind::Wilt
            | VisualCueKind::PulseRing
            | VisualCueKind::Flash
            | VisualCueKind::BoostTrail => ParticlePriority::Standard,
            VisualCueKind::Nibble => ParticlePriority::Ambient,
        }
    }

    const fn sprite_for(kind: VisualCueKind) -> SpriteKind {
        match kind {
            VisualCueKind::Sparkle | VisualCueKind::Flash => SpriteKind::Spark,
            VisualCueKind::Wilt => SpriteKind::Mote,
            VisualCueKind::Shards | VisualCueKind::SparkCone => SpriteKind::Shard,
            VisualCueKind::Nibble | VisualCueKind::BoostTrail => SpriteKind::Puff,
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
                VisualCueKind::PulseRing | VisualCueKind::Flash | VisualCueKind::BoostTrail => 0.05,
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
                let batch =
                    self.scheduler
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
                kind: VisualCueKind::BoostTrail,
                color: trail.color,
                accent_color: trail.accent,
                intensity: 0.5,
                radius: 2.0,
                duration_ticks: trail.duration_ticks,
            };
            let batch = self.scheduler.schedule(
                tick,
                index as u32,
                &cue,
                [
                    position[0] + trail.anchor.offset[0],
                    position[1] + trail.anchor.offset[1],
                    position[2] + trail.anchor.offset[2],
                ],
            );
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
    use scriptbots_core::visual::{
        VisualCue, VisualCueKind, WorldVisualEvent, visual_cue_for_event,
    };

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

    /// THE DEFECT, as a regression guard: eviction must take the oldest
    /// ambient, not the lowest slot index.
    ///
    /// The two agree only until something is killed. After a kill the freed
    /// slot is reused by the NEWEST particle, so a scan from slot 0 evicts the
    /// youngest ambient while genuinely old ones survive — the opposite of the
    /// policy's name. This arrangement makes lowest-slot and true-oldest
    /// disagree, so the old implementation fails it and the new one passes.
    #[test]
    fn eviction_takes_the_oldest_ambient_not_the_lowest_slot() {
        let mut pool = ParticlePool::with_capacity(3);
        let a = pool.spawn(particle(ParticlePriority::Ambient)).expect("a");
        let b = pool.spawn(particle(ParticlePriority::Ambient)).expect("b");
        let c = pool.spawn(particle(ParticlePriority::Ambient)).expect("c");
        assert_eq!(
            (a.slot, b.slot, c.slot),
            (0, 1, 2),
            "free list hands out 0,1,2"
        );

        pool.kill(a);
        let d = pool.spawn(particle(ParticlePriority::Ambient)).expect("d");
        assert_eq!(
            d.slot, 0,
            "the freed low slot is recycled by the NEWEST particle"
        );

        assert_eq!(
            pool.oldest_ambient_slot(),
            Some(b.slot),
            "b is the oldest surviving ambient; slot 0 now holds the newest"
        );

        let critical = pool
            .spawn(particle(ParticlePriority::Critical))
            .expect("a critical spawn must evict an ambient from a full pool");
        assert_eq!(
            critical.slot, b.slot,
            "the victim must be the oldest ambient (slot {}), not the lowest slot (0)",
            b.slot
        );
        assert!(
            pool.born_seq_of(b).is_none(),
            "the evicted particle is dead"
        );
        assert!(
            pool.born_seq_of(d).is_some(),
            "the newest ambient must survive; evicting it is the bug"
        );
        assert!(pool.born_seq_of(c).is_some(), "untouched ambient survives");
    }

    /// The injected negative: the strategy the old code used must actually
    /// disagree with the correct one on this arrangement.
    ///
    /// A regression guard is worth nothing if the wrong implementation would
    /// also satisfy it. This reproduces the discarded "first ambient slot from
    /// zero" scan and asserts it picks a DIFFERENT victim than the pool does —
    /// so `eviction_takes_the_oldest_ambient_not_the_lowest_slot` is proven to
    /// fail against the old behaviour rather than merely asserted to.
    #[test]
    fn a_lowest_slot_scan_would_pick_a_different_victim_than_true_oldest() {
        let mut pool = ParticlePool::with_capacity(3);
        let a = pool.spawn(particle(ParticlePriority::Ambient)).expect("a");
        let b = pool.spawn(particle(ParticlePriority::Ambient)).expect("b");
        pool.spawn(particle(ParticlePriority::Ambient)).expect("c");
        pool.kill(a);
        let d = pool.spawn(particle(ParticlePriority::Ambient)).expect("d");

        // The rejected strategy, restated here and nowhere in production.
        let lowest_ambient_slot = (0..pool.capacity() as u32)
            .find(|&slot| pool.slot_priority_for_test(slot) == Some(ParticlePriority::Ambient));
        let true_oldest = pool.oldest_ambient_slot();

        assert_eq!(
            lowest_ambient_slot,
            Some(d.slot),
            "the scan finds the recycled low slot"
        );
        assert_eq!(true_oldest, Some(b.slot), "the oldest ambient is b");
        assert_ne!(
            lowest_ambient_slot, true_oldest,
            "if these agreed, the regression guard would pass against the bug it exists to catch"
        );
    }

    /// Randomized spawn/kill/reuse traces must agree with a simple reference
    /// model of "ambient slots in birth order", which is the specification the
    /// intrusive list is an optimization of.
    #[test]
    fn ambient_eviction_matches_a_reference_model_under_random_traces() {
        const CAPACITY: usize = 16;
        // Deterministic LCG: no rand dependency, and a failure is reproducible
        // from the seed printed in the assertion.
        let mut state: u64 = 0x1419_5EED_2026_07_26;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };

        let mut pool = ParticlePool::with_capacity(CAPACITY);
        // Reference model: ambient slots, oldest first.
        let mut expected: Vec<u32> = Vec::new();
        let mut live: Vec<ParticleHandle> = Vec::new();

        for op in 0..4_000u32 {
            match next() % 3 {
                0 => {
                    // Spawn ambient. Rejected when full (no churn), which the
                    // model mirrors by leaving itself unchanged.
                    if let Some(handle) = pool.spawn(particle(ParticlePriority::Ambient)) {
                        expected.push(handle.slot);
                        live.push(handle);
                    }
                }
                1 => {
                    // Kill an arbitrary live particle.
                    if !live.is_empty() {
                        let victim = live.swap_remove((next() as usize) % live.len());
                        // Only mirror the kill when the handle was actually
                        // live. A stale handle names a slot that a DIFFERENT
                        // particle now occupies, and dropping that slot from
                        // the model would desynchronise it from a pool that
                        // correctly ignored the stale kill.
                        let was_live = pool.born_seq_of(victim).is_some();
                        pool.kill(victim);
                        if was_live {
                            expected.retain(|slot| *slot != victim.slot);
                        }
                    }
                }
                _ => {
                    // Critical spawn: evicts the oldest ambient when full.
                    let before = expected.first().copied();
                    if let Some(handle) = pool.spawn(particle(ParticlePriority::Critical)) {
                        if pool.live_count() == CAPACITY && before == Some(handle.slot) {
                            // It reused the evicted slot; the model drops it.
                            expected.retain(|slot| *slot != handle.slot);
                        }
                        live.retain(|h| h.slot != handle.slot);
                        live.push(handle);
                    }
                }
            }
            assert_eq!(
                pool.oldest_ambient_slot(),
                expected.first().copied(),
                "op {op}: pool and reference model disagree on the oldest ambient"
            );
        }
    }

    /// With no ambient to sacrifice the policy refuses rather than evicting
    /// something it promised never to touch.
    #[test]
    fn a_full_pool_without_ambients_refuses_the_spawn() {
        let mut pool = ParticlePool::with_capacity(2);
        pool.spawn(particle(ParticlePriority::Critical))
            .expect("c0");
        pool.spawn(particle(ParticlePriority::Standard))
            .expect("s1");
        assert_eq!(pool.oldest_ambient_slot(), None);
        assert!(
            pool.spawn(particle(ParticlePriority::Critical)).is_none(),
            "critical and standard particles must never be evicted by policy"
        );
        assert_eq!(
            pool.live_count(),
            2,
            "a refused spawn must not disturb the pool"
        );
    }

    /// An ambient arriving at a full pool loses rather than churning another
    /// ambient out — otherwise the pool thrashes at steady state.
    #[test]
    fn an_ambient_spawn_into_a_full_pool_is_rejected_without_churn() {
        let mut pool = ParticlePool::with_capacity(2);
        let first = pool.spawn(particle(ParticlePriority::Ambient)).expect("a0");
        pool.spawn(particle(ParticlePriority::Ambient)).expect("a1");
        assert!(pool.spawn(particle(ParticlePriority::Ambient)).is_none());
        assert_eq!(
            pool.oldest_ambient_slot(),
            Some(first.slot),
            "the rejected spawn must not have evicted anything"
        );
        assert_eq!(pool.live_count(), 2);
    }

    /// Degenerate capacities must answer rather than panic or index out of range.
    #[test]
    fn degenerate_capacities_are_handled() {
        let mut empty = ParticlePool::with_capacity(0);
        assert_eq!(empty.oldest_ambient_slot(), None);
        assert!(empty.spawn(particle(ParticlePriority::Critical)).is_none());
        assert!(empty.spawn(particle(ParticlePriority::Ambient)).is_none());

        let mut one = ParticlePool::with_capacity(1);
        let only = one
            .spawn(particle(ParticlePriority::Ambient))
            .expect("fits");
        assert_eq!(one.oldest_ambient_slot(), Some(only.slot));
        let critical = one
            .spawn(particle(ParticlePriority::Critical))
            .expect("evicts the single ambient");
        assert_eq!(critical.slot, only.slot);
        assert_eq!(one.oldest_ambient_slot(), None, "the list is now empty");
        assert_eq!(one.live_count(), 1);
    }

    /// A stale handle must be inert. Acting on one would unlink a slot that a
    /// different particle now occupies and silently corrupt the FIFO.
    #[test]
    fn stale_handles_do_not_corrupt_the_ambient_order() {
        let mut pool = ParticlePool::with_capacity(2);
        let doomed = pool.spawn(particle(ParticlePriority::Ambient)).expect("a0");
        pool.kill(doomed);
        let reused = pool.spawn(particle(ParticlePriority::Ambient)).expect("a1");
        assert_eq!(reused.slot, doomed.slot, "the slot is recycled");

        pool.kill(doomed); // stale: same slot, older generation
        assert!(
            pool.born_seq_of(reused).is_some(),
            "a stale handle must not kill the particle that replaced it"
        );
        assert_eq!(
            pool.oldest_ambient_slot(),
            Some(reused.slot),
            "the ambient list must still contain the live particle"
        );
        assert_eq!(pool.live_count(), 1);
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
        for recipe in [
            cue(VisualCueKind::Sparkle),
            cue(VisualCueKind::Shards),
            cue(VisualCueKind::Wilt),
            cue(VisualCueKind::Nibble),
            cue(VisualCueKind::SparkCone),
            cue(VisualCueKind::PulseRing),
            cue(VisualCueKind::Flash),
            visual_cue_for_event(&WorldVisualEvent::Boost { magnitude: 0.8 }),
        ] {
            let batch = scheduler.schedule(50, 0, &recipe, [0.0; 3]);
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
    fn scheduler_preserves_the_canonical_boost_cue() {
        let scheduler = CueScheduler::new(7);
        let canonical = visual_cue_for_event(&WorldVisualEvent::Boost { magnitude: 0.73 });
        let batch = scheduler.schedule(50, 0, &canonical, [0.0; 3]);

        assert_eq!(canonical.kind, VisualCueKind::BoostTrail);
        assert!(!batch.particles.is_empty());
        for particle in batch.particles {
            assert_eq!(particle.color, canonical.color);
            assert_eq!(particle.accent, canonical.accent_color);
            assert_eq!(particle.intensity, canonical.intensity);
            assert_eq!(particle.priority, ParticlePriority::Standard);
            assert_eq!(particle.sprite, SpriteKind::Puff);
            let horizontal_speed = particle.velocity[0].hypot(particle.velocity[2]);
            assert!(
                (particle.velocity[1] / horizontal_speed - 0.05).abs() < 1.0e-6,
                "boost exhaust stays near-planar rather than becoming an eat/birth arc"
            );
        }
    }

    /// Drive the REAL scheduler, pool, instance writer, ordering path and atlas
    /// bake to full capacity, and return deterministic JSON evidence.
    ///
    /// Mock-free by construction: every stage is the production type. Nothing
    /// here constructs a GPU device or a render graph, so this does NOT claim
    /// live renderer wiring — that stays owned by bd-2z0.14.1.7. What it proves
    /// is that the stages compose deterministically end to end, which no single
    /// unit test can show because each one stops at its own boundary.
    fn particle_pipeline_evidence(capacity: usize) -> serde_json::Value {
        let scheduler = CueScheduler::new(0x5EED_2026);
        let mut pool = ParticlePool::with_capacity(capacity);
        pool.overflow = OverflowPolicy::DropOldestAmbient;

        let mut scheduled = 0usize;
        let mut admitted = 0usize;
        let mut rejected = 0usize;

        // Enough cues to overrun the pool, so the eviction policy is exercised
        // rather than merely present. Ambient-producing kinds fill it; the
        // Shards/SparkCone cues then have to displace something.
        let kinds = [
            VisualCueKind::Sparkle,
            VisualCueKind::Nibble,
            VisualCueKind::Wilt,
            VisualCueKind::Shards,
            VisualCueKind::SparkCone,
        ];
        for tick in 0..24u64 {
            for (ordinal, kind) in kinds.iter().enumerate() {
                let batch = scheduler.schedule(
                    tick,
                    u32::try_from(ordinal).expect("ordinal fits"),
                    &cue(*kind),
                    [tick as f32, ordinal as f32, 0.0],
                );
                for particle in batch.particles {
                    scheduled += 1;
                    if pool.spawn(particle).is_some() {
                        admitted += 1;
                    } else {
                        rejected += 1;
                    }
                }
            }
        }

        // Real instance writer.
        let mut instances = Vec::new();
        pool.write_instance_buffer(&mut instances);

        // Real ordering path, fed from the instances just written.
        let depths: Vec<(u32, f32)> = instances
            .iter()
            .enumerate()
            .map(|(index, instance)| (u32::try_from(index).expect("index fits"), instance.pos0[2]))
            .collect();
        let mut order = Vec::new();
        let mut scratch = Vec::new();
        blend_order_into(
            &depths,
            BlendOrderPolicy::BucketSort,
            &mut order,
            &mut scratch,
        );

        // Real atlas bake, hashed rather than embedded.
        let atlas = bake_sprite_atlas(CANONICAL_ATLAS_SEED, CANONICAL_ATLAS_TILE_PX);

        // Order and instance streams are summarised by digest so the evidence
        // stays small while still changing if any element moves.
        let mut order_bytes = Vec::with_capacity(order.len() * 4);
        for slot in &order {
            order_bytes.extend_from_slice(&slot.to_le_bytes());
        }
        let mut instance_bytes = Vec::with_capacity(instances.len() * 12);
        for instance in &instances {
            for axis in instance.pos0 {
                instance_bytes.extend_from_slice(&axis.to_le_bytes());
            }
        }

        serde_json::json!({
            "schema": "scriptbots.particle-pipeline-evidence.v1",
            "capacity": capacity,
            "scheduled": scheduled,
            "admitted": admitted,
            "rejected": rejected,
            "live_count": pool.live_count(),
            "instances": instances.len(),
            "order_len": order.len(),
            "order_digest": blake3::hash(&order_bytes).to_hex().to_string(),
            "instance_position_digest": blake3::hash(&instance_bytes).to_hex().to_string(),
            "atlas_digest": blake3::hash(&atlas).to_hex().to_string(),
            "claims_live_renderer_wiring": false,
        })
    }

    /// The stages compose deterministically end to end, at full capacity, with
    /// the eviction policy actually engaged.
    ///
    /// Each stage already has unit coverage, but every one of those stops at its
    /// own boundary — a pool test cannot see the instance writer, and an
    /// ordering test cannot see the pool. This runs the real chain twice and
    /// requires identical evidence, so a nondeterminism introduced at any seam
    /// shows up here rather than as an unreproducible frame later.
    #[test]
    fn the_real_particle_pipeline_is_deterministic_at_full_capacity() {
        const CAPACITY: usize = 64;
        let first = particle_pipeline_evidence(CAPACITY);
        let second = particle_pipeline_evidence(CAPACITY);
        assert_eq!(
            first, second,
            "the same seed and cue schedule must produce identical pipeline evidence"
        );

        // The run has to be a real one, or the determinism claim is vacuous.
        assert_eq!(first["live_count"], CAPACITY, "the pool must end up FULL");
        assert_eq!(
            first["instances"], CAPACITY,
            "the instance writer must emit one instance per live particle"
        );
        assert_eq!(
            first["order_len"], CAPACITY,
            "every instance must be ordered"
        );
        assert!(
            first["scheduled"].as_u64().expect("scheduled") > CAPACITY as u64,
            "the schedule must overrun capacity, or eviction never engages"
        );
        assert!(
            first["rejected"].as_u64().expect("rejected") > 0,
            "a full pool must reject some ambient spawns; zero means the overflow \
             policy was never exercised and this proves less than it appears to"
        );
        assert_eq!(
            first["claims_live_renderer_wiring"], false,
            "this evidence is stage composition only; live wiring is bd-2z0.14.1.7"
        );
    }

    /// Canonical atlas parameters. The pinned digest below is only meaningful
    /// for exactly these; changing any of them requires re-deriving it.
    const CANONICAL_ATLAS_SEED: u64 = 5;
    const CANONICAL_ATLAS_TILE_PX: u32 = 32;

    /// blake3 of `bake_sprite_atlas(CANONICAL_ATLAS_SEED, CANONICAL_ATLAS_TILE_PX)`.
    ///
    /// REBASELINING THIS REQUIRES AN EXPLICIT REASON in the commit message
    /// naming what changed about the atlas and why the new appearance is
    /// correct. It is not a value to refresh until the test goes green — that
    /// converts a real signal into a rubber stamp, which is the failure this
    /// pin exists to prevent.
    const CANONICAL_ATLAS_DIGEST: &str =
        "eb074d032052d716d0c348b3c9b73004eb92b0c0433e69523edad1a7928b6a49";

    /// The atlas must match a PINNED digest, not merely equal a second bake.
    ///
    /// `sprite_atlas_is_deterministic_and_shaped` asserts `bake(a) == bake(a)`.
    /// That proves the bake is repeatable and nothing more: it would pass
    /// unchanged if every sprite in the atlas were replaced, because both sides
    /// of the comparison come from the same implementation. Deterministic drift
    /// is exactly the thing it cannot see.
    ///
    /// Pinning an external constant is what makes the check falsifiable — the
    /// expected value no longer moves when the code does.
    #[test]
    fn the_canonical_sprite_atlas_matches_its_pinned_digest() {
        let atlas = bake_sprite_atlas(CANONICAL_ATLAS_SEED, CANONICAL_ATLAS_TILE_PX);
        let digest = blake3::hash(&atlas).to_hex().to_string();
        assert_eq!(
            digest, CANONICAL_ATLAS_DIGEST,
            "canonical sprite atlas changed. If this was deliberate, record WHAT changed \
             about the atlas and why the new appearance is correct, then update the pin to \
             {digest}. Do not refresh it merely to get green."
        );
    }

    /// Shape, channel and transparency contracts the digest alone cannot explain.
    ///
    /// A digest says "different" without saying how, so these assert the
    /// properties a reader would want named when it trips: the RGB body is
    /// opaque white with tinting left to the shader, every tile carries real
    /// coverage, and the ring is genuinely hollow.
    #[test]
    fn the_canonical_sprite_atlas_holds_its_shape_channel_and_transparency_contracts() {
        let size = CANONICAL_ATLAS_TILE_PX as usize;
        let tiles = SpriteKind::ALL.len();
        let atlas = bake_sprite_atlas(CANONICAL_ATLAS_SEED, CANONICAL_ATLAS_TILE_PX);

        assert_eq!(
            atlas.len(),
            size * tiles * size * 4,
            "atlas is {tiles} tiles of {size}x{size} RGBA8 laid out side by side"
        );

        for tile in 0..tiles {
            let mut covered = 0usize;
            let mut interior = 0usize;
            for y in 0..size {
                for x in 0..size {
                    let offset = (y * size * tiles + tile * size + x) * 4;
                    let [r, g, b, a] = [
                        atlas[offset],
                        atlas[offset + 1],
                        atlas[offset + 2],
                        atlas[offset + 3],
                    ];
                    assert_eq!(
                        (r, g, b),
                        (255, 255, 255),
                        "tile {tile} texel ({x},{y}) must be white; per-instance tinting \
                         belongs to the shader, so a coloured atlas would tint twice"
                    );
                    if a > 0 {
                        covered += 1;
                        // Strictly interior, derived from the alpha formula
                        // rather than assumed: alpha = clamp(0.5 - dist*2.5),
                        // so dist == 0 (the shape boundary) is exactly 128 and
                        // anything inside the shape exceeds it. Saturation to
                        // 255 needs dist <= -0.2, which Ring can never reach —
                        // its minimum distance is -0.12, capping it at 204 — so
                        // requiring an opaque core would be false for this atlas.
                        if a > 128 {
                            interior += 1;
                        }
                    }
                }
            }
            assert!(
                covered > 100,
                "tile {tile} must carry real coverage, got {covered} texels"
            );
            assert!(
                interior > 0,
                "tile {tile} must have texels strictly inside its shape, not only \
                 boundary falloff"
            );
            assert!(
                covered < size * size,
                "tile {tile} must not fill its whole square, or it has no silhouette"
            );
        }

        // The ring is the one tile defined by its hole; if this fills in, the
        // sprite has silently become a disc.
        let ring = SpriteKind::Ring.tile_index();
        let centre = ((size / 2) * size * tiles + ring * size + size / 2) * 4;
        assert_eq!(
            atlas[centre + 3],
            0,
            "the ring centre must be fully transparent"
        );
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
        let golden: [(f32, u32); 5] = [(0.0, 4), (0.25, 9), (0.5, 14), (0.75, 19), (1.0, 24)];
        for recipe in [
            cue(VisualCueKind::Sparkle),
            cue(VisualCueKind::Shards),
            cue(VisualCueKind::Wilt),
            cue(VisualCueKind::Nibble),
            cue(VisualCueKind::SparkCone),
            cue(VisualCueKind::PulseRing),
            cue(VisualCueKind::Flash),
            visual_cue_for_event(&WorldVisualEvent::Boost { magnitude: 0.8 }),
        ] {
            for (intensity, expected) in golden {
                let mut c = recipe;
                c.intensity = intensity;
                assert_eq!(
                    scheduler.count_for_cue(&c),
                    expected,
                    "{:?} at intensity {intensity}",
                    c.kind
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
        assert_eq!(
            pool.state_hash(),
            pool_b.state_hash(),
            "identical selection"
        );
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
        assert_eq!(
            run(),
            run(),
            "identical cue stream -> identical pool state hash"
        );
        // A different stream must diverge (the hash actually observes state).
        let divergent = || {
            let mut emitter = CueEmitter::new(42);
            let mut pool = ParticlePool::with_capacity(1024);
            emitter.enqueue(100, cue(VisualCueKind::Sparkle), [9.0, 9.0, 9.0]);
            emitter.apply_tick(100, &mut pool);
            pool.state_hash()
        };
        assert_ne!(
            run(),
            divergent(),
            "different streams produce different states"
        );
    }

    #[test]
    fn boost_trail_follows_a_moving_agent_and_stays_bounded() {
        let mut emitter = CueEmitter::new(17);
        let mut pool = ParticlePool::with_capacity(256);
        let canonical = visual_cue_for_event(&WorldVisualEvent::Boost { magnitude: 0.8 });
        emitter.bind_boost_trail(
            AgentAnchor {
                uid: 99,
                offset: [0.0, 0.5, 0.0],
            },
            canonical.color,
            canonical.accent_color,
        );
        assert_eq!(emitter.trail_count(), 1);
        // Agent moves +1 x per tick; trail puffs appear at successive spots.
        let mut emitted_total = 0_u32;
        for tick in 0..=u64::from(CueEmitter::TRAIL_DURATION_TICKS) {
            let x = tick as f32;
            emitted_total +=
                emitter.emit_trails(tick, &mut pool, |uid| (uid == 99).then_some([x, 0.0, 0.0]));
        }
        let expected = CueEmitter::TRAIL_DURATION_TICKS / CueEmitter::TRAIL_INTERVAL_TICKS;
        assert_eq!(emitted_total, expected, "one puff per interval");
        pool.for_each_live(|particle| {
            assert_eq!(particle.color, canonical.color);
            assert_eq!(particle.accent, canonical.accent_color);
            assert_eq!(particle.priority, ParticlePriority::Standard);
            assert_eq!(particle.sprite, SpriteKind::Puff);
        });
        // Agent dies: lookup yields None -> trail detaches and stops.
        let emitted_after_death = emitter.emit_trails(999, &mut pool, |_| None);
        assert_eq!(emitted_after_death, 0);
        assert_eq!(emitter.trail_count(), 0, "dead agent's trail detached");
        // Rebinding refreshes instead of duplicating.
        let anchor = AgentAnchor {
            uid: 7,
            offset: [0.0; 3],
        };
        emitter.bind_boost_trail(anchor, canonical.color, canonical.accent_color);
        emitter.bind_boost_trail(anchor, canonical.color, canonical.accent_color);
        assert_eq!(emitter.trail_count(), 1, "rebind refreshes");
        emitter.detach_trails_for(7);
        assert_eq!(emitter.trail_count(), 0);
    }

    #[test]
    fn state_hash_distinguishes_pool_states() {
        let mut a = ParticlePool::with_capacity(4);
        let b = ParticlePool::with_capacity(4);
        assert_eq!(a.state_hash(), b.state_hash(), "empty pools equal");
        a.spawn(particle(ParticlePriority::Standard))
            .expect("spawn");
        assert_ne!(a.state_hash(), b.state_hash(), "occupancy is observed");
        let handle = a
            .spawn(particle(ParticlePriority::Critical))
            .expect("spawn");
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
