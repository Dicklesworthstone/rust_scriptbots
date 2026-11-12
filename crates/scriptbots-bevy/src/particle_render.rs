//! Positioned core events rendered by the same bounded material in live and capture apps.
//!
//! The host publishes a latest-value boundary, not a lossless event journal. We consume
//! accepted ordinals once, expose skipped boundaries and producer/budget drops, and never
//! manufacture missing events. Scientific state and its random streams remain untouched.

use std::{cmp::Reverse, sync::Arc};

use bevy::app::AppExit;
use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::camera::visibility::{NoFrustumCulling, VisibilitySystems};
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::mesh::{Indices, Mesh};
use bevy::pbr::{Material, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    AsBindGroup, BufferId, BufferUsages, Extent3d, PrimitiveTopology, TextureDimension,
    TextureFormat,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::storage::{GpuShaderStorageBuffer, ShaderStorageBuffer};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy::shader::ShaderRef;
use scriptbots_core::{
    LocatedWorldVisualEvent, RenderQuality, WorldState, tier_features,
    visual::{WorldVisualEvent, visual_cue_for_event},
};
use scriptbots_runtime::{HostSessionId, RenderSnapshot};
use tracing::{debug, error, warn};

use crate::particles::{CueScheduler, Particle, ParticlePool, ParticlePriority, bake_sprite_atlas};
use crate::{
    AccessibilityState, ColorPaletteMode, EffectiveRenderSettings, SnapshotState,
    TERRAIN_HEIGHT_SCALE, WorldSnapshot, apply_palette_rgb, sample_height_world,
    srgb_to_linear_rgb,
};

const ROW_BYTES: usize = 96;
const MAX_PARTICLES: usize = tier_features(RenderQuality::Ultra).particles_max as usize;
const MAX_EVENTS: usize = WorldState::MAX_TICK_VISUAL_EVENTS;
const MAX_LIFETIME: u32 = 4_096;
const CUES_PER_CLASS: usize = 32;
const ATLAS_TILE_SIZE: u32 = 32;

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct VisualFrame {
    seed: u64,
    epoch: u64,
    events: Arc<Vec<LocatedWorldVisualEvent>>,
    dropped: u64,
}

impl VisualFrame {
    pub(crate) fn from_snapshot(snapshot: &RenderSnapshot) -> Self {
        Self {
            seed: snapshot.visual_seed,
            epoch: snapshot.world.epoch,
            events: Arc::clone(&snapshot.visual_events),
            dropped: snapshot.visual_dropped_events,
        }
    }

    pub(crate) fn from_world(world: &WorldState) -> Self {
        Self {
            seed: world.agent_substream_protocol_v1().root_seed(),
            epoch: world.epoch(),
            events: Arc::new(world.visual_events().to_vec()),
            dropped: world.visual_dropped_events_this_tick(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EffectSource {
    session: Option<HostSessionId>,
    seed: u64,
    dimensions: [u32; 2],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ParticleRow([[f32; 4]; 6]);

impl ParticleRow {
    // All converted integers are bounded by MAX_LIFETIME or the five-tile atlas.
    // Absolute u64 ticks are subtracted before conversion, including beyond 2^53.
    #[allow(clippy::cast_precision_loss)]
    fn new(particle: &Particle, tick: u64, palette: ColorPaletteMode) -> Self {
        let age = u32::try_from(tick.saturating_sub(particle.born_tick)).unwrap_or(MAX_LIFETIME);
        debug_assert!(age < particle.duration_ticks && particle.duration_ticks <= MAX_LIFETIME);
        let color = srgb_to_linear_rgb(
            apply_palette_rgb(Vec3::from_array(particle.color), palette).to_array(),
        );
        let accent = srgb_to_linear_rgb(
            apply_palette_rgb(Vec3::from_array(particle.accent), palette).to_array(),
        );
        Self([
            [
                particle.position[0],
                particle.position[1],
                particle.position[2],
                particle.gravity,
            ],
            [
                particle.velocity[0],
                particle.velocity[1],
                particle.velocity[2],
                particle.drag,
            ],
            [
                age as f32,
                particle.duration_ticks as f32,
                particle.size_start,
                particle.size_end,
            ],
            [
                particle.rotation,
                particle.spin,
                particle.intensity,
                particle.sprite.tile_index() as f32,
            ],
            [color[0], color[1], color[2], 0.0],
            [accent[0], accent[1], accent[2], 0.0],
        ])
    }

    fn append_bytes(self, bytes: &mut Vec<u8>) {
        for vector in self.0 {
            for value in vector {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
}

#[derive(Default, Debug)]
struct ParticleStats {
    accepted_cues: u64,
    budget_dropped_cues: u64,
    refused_particles: u64,
    producer_dropped_events: u64,
    skipped_boundaries: u64,
    stale_boundaries: u64,
    source_resets: u64,
    tier_dropped_particles: u64,
}

#[derive(Resource)]
pub(crate) struct ParticleScene {
    pool: ParticlePool,
    source: Option<EffectSource>,
    last_tick: Option<u64>,
    last_publication: Option<u64>,
    last_ordinal: Option<u32>,
    observed_producer_drops: u64,
    class_counts: [usize; 3],
    pending: Vec<LocatedWorldVisualEvent>,
    rows: Vec<ParticleRow>,
    palette: Option<ColorPaletteMode>,
    revision: u64,
    max_rows: usize,
    gpu_capacity: usize,
    buffer: Option<Handle<ShaderStorageBuffer>>,
    mesh: Option<Handle<Mesh>>,
    material: Option<Handle<ParticleMaterial>>,
    entity: Option<Entity>,
    stats: ParticleStats,
    last_error: Option<&'static str>,
}

impl Default for ParticleScene {
    fn default() -> Self {
        Self {
            pool: ParticlePool::with_capacity(0),
            source: None,
            last_tick: None,
            last_publication: None,
            last_ordinal: None,
            observed_producer_drops: 0,
            class_counts: [0; 3],
            pending: Vec::with_capacity(MAX_EVENTS),
            rows: Vec::new(),
            palette: None,
            revision: 0,
            max_rows: MAX_PARTICLES,
            gpu_capacity: 0,
            buffer: None,
            mesh: None,
            material: None,
            entity: None,
            stats: ParticleStats::default(),
            last_error: None,
        }
    }
}

impl ParticleScene {
    fn reset_source(&mut self) {
        self.pool.clear();
        self.source = None;
        self.last_tick = None;
        self.last_publication = None;
        self.last_ordinal = None;
        self.observed_producer_drops = 0;
        self.class_counts = [0; 3];
        self.pending.clear();
        self.rows.clear();
        self.palette = None;
        self.last_error = None;
        self.revision = self.revision.wrapping_add(1);
    }

    /// Capture session teardown despawns mesh entities; keep reusable GPU assets.
    pub(crate) fn reset(&mut self) {
        self.reset_source();
        self.entity = None;
    }

    pub(crate) fn error(&self) -> Option<&'static str> {
        self.last_error
    }

    fn resize_pool(&mut self, capacity: usize) -> bool {
        if self.pool.capacity() == capacity {
            return false;
        }
        let mut resized = ParticlePool::with_capacity(capacity);
        for priority in [
            ParticlePriority::Critical,
            ParticlePriority::Standard,
            ParticlePriority::Ambient,
        ] {
            self.pool.for_each_live(|particle| {
                if particle.priority == priority {
                    let _ = resized.spawn(*particle);
                }
            });
        }
        let dropped = self.pool.live_count().saturating_sub(resized.live_count());
        self.stats.tier_dropped_particles = self
            .stats
            .tier_dropped_particles
            .saturating_add(u64::try_from(dropped).unwrap_or(u64::MAX));
        self.pool = resized;
        true
    }

    /// Return whether this boundary was admitted; stale publications never update GPU uniforms.
    fn observe(
        &mut self,
        snapshot: &WorldSnapshot,
        limit: usize,
        palette: ColorPaletteMode,
    ) -> Result<bool, &'static str> {
        let source = EffectSource {
            session: snapshot.owner_source.map(|(owner, _)| owner.session_id),
            seed: snapshot.visual.seed,
            dimensions: [
                snapshot.world_size.x.to_bits(),
                snapshot.world_size.y.to_bits(),
            ],
        };
        let publication = snapshot
            .owner_source
            .map(|(owner, _)| owner.published_snapshot.get());
        let same_source = self.source == Some(source);
        let stale_publication = self.source.is_some_and(|previous| {
            previous.session.is_some() && previous.session == source.session
        }) && publication
            .zip(self.last_publication)
            .is_some_and(|(new, old)| new < old);
        if stale_publication
            || (same_source && self.last_tick.is_some_and(|tick| snapshot.tick < tick))
        {
            self.stats.stale_boundaries = self.stats.stale_boundaries.saturating_add(1);
            return Ok(false);
        }
        validate_boundary(snapshot)?;
        if same_source && self.last_tick == Some(snapshot.tick) {
            if self.last_ordinal.is_some_and(|ordinal| {
                snapshot
                    .visual
                    .events
                    .last()
                    .is_none_or(|event| event.ordinal < ordinal)
            }) || snapshot.visual.dropped < self.observed_producer_drops
            {
                return Err("visual-event prefix regressed within one completed boundary");
            }
        }

        let mut changed = false;
        if !same_source {
            self.reset_source();
            self.source = Some(source);
            self.stats.source_resets = self.stats.source_resets.saturating_add(1);
            changed = true;
        }
        changed |= self.resize_pool(limit.min(self.max_rows).min(MAX_PARTICLES));
        if self.last_tick != Some(snapshot.tick) {
            if let Some(previous) = self.last_tick {
                self.stats.skipped_boundaries = self
                    .stats
                    .skipped_boundaries
                    .saturating_add(snapshot.tick.saturating_sub(previous).saturating_sub(1));
            }
            self.last_tick = Some(snapshot.tick);
            self.last_ordinal = None;
            self.observed_producer_drops = 0;
            self.class_counts = [0; 3];
            self.pool.expire(snapshot.tick);
            changed = true;
        }
        self.last_publication = publication;
        self.stats.producer_dropped_events = self.stats.producer_dropped_events.saturating_add(
            snapshot
                .visual
                .dropped
                .saturating_sub(self.observed_producer_drops),
        );
        self.observed_producer_drops = snapshot.visual.dropped;
        self.pending.clear();
        self.pending.extend(
            snapshot
                .visual
                .events
                .iter()
                .copied()
                .filter(|event| self.last_ordinal.is_none_or(|last| event.ordinal > last)),
        );
        self.last_ordinal = snapshot.visual.events.last().map(|event| event.ordinal);
        self.pending
            .sort_unstable_by_key(|event| (Reverse(event_priority(event.event)), event.ordinal));
        let scheduler = CueScheduler::new(source.seed);
        for event in &self.pending {
            let priority = event_priority(event.event);
            let class = match priority {
                ParticlePriority::Ambient => 0,
                ParticlePriority::Standard => 1,
                ParticlePriority::Critical => 2,
            };
            if self.class_counts[class] >= CUES_PER_CLASS {
                self.stats.budget_dropped_cues = self.stats.budget_dropped_cues.saturating_add(1);
                continue;
            }
            self.class_counts[class] += 1;
            self.stats.accepted_cues = self.stats.accepted_cues.saturating_add(1);
            let cue = visual_cue_for_event(&event.event);
            let location = Vec2::new(event.position.x, event.position.y);
            let position = [
                location.x - snapshot.world_size.x * 0.5,
                sample_height_world(&snapshot.terrain_height, location, TERRAIN_HEIGHT_SCALE)
                    + snapshot.agent_radius * 0.75,
                snapshot.world_size.y * 0.5 - location.y,
            ];
            let direction = Vec2::new(event.direction[0], -event.direction[1]).normalize_or_zero();
            // Interventions can stamp births before the current step advances.
            // Use the event's own integer tick for its recipe and age.
            scheduler.schedule_into(
                event.tick.0,
                event.ordinal,
                &cue,
                position,
                |mut particle| {
                    particle.priority = priority;
                    if matches!(
                        event.event,
                        WorldVisualEvent::CombatHit { .. } | WorldVisualEvent::SpikeExtend
                    ) && direction != Vec2::ZERO
                    {
                        let speed = Vec2::new(particle.velocity[0], particle.velocity[2]).length();
                        let side = particle.velocity[2] * 0.3;
                        particle.velocity[0] = direction.x * speed - direction.y * side;
                        particle.velocity[2] = direction.y * speed + direction.x * side;
                    }
                    if snapshot.tick.saturating_sub(particle.born_tick)
                        < u64::from(particle.duration_ticks)
                        && self.pool.spawn(particle).is_none()
                    {
                        self.stats.refused_particles =
                            self.stats.refused_particles.saturating_add(1);
                    }
                },
            );
            changed = true;
        }
        if changed || self.palette != Some(palette) {
            self.rows.clear();
            self.pool.for_each_live(|particle| {
                self.rows
                    .push(ParticleRow::new(particle, snapshot.tick, palette));
            });
            self.palette = Some(palette);
            self.revision = self.revision.wrapping_add(1);
        }
        if changed && snapshot.tick.is_multiple_of(120) {
            debug!(
                tick = snapshot.tick,
                live_particles = self.rows.len(),
                capacity = self.pool.capacity(),
                counters = ?self.stats,
                "native visual-event particle boundary"
            );
        }
        Ok(true)
    }
}

fn event_priority(event: WorldVisualEvent) -> ParticlePriority {
    match event {
        WorldVisualEvent::Birth { .. }
        | WorldVisualEvent::Death { .. }
        | WorldVisualEvent::CombatHit { .. } => ParticlePriority::Critical,
        WorldVisualEvent::Eat { .. } => ParticlePriority::Ambient,
        _ => ParticlePriority::Standard,
    }
}

fn validate_boundary(snapshot: &WorldSnapshot) -> Result<(), &'static str> {
    if snapshot.visual.events.len() > MAX_EVENTS {
        return Err("visual-event boundary exceeds the core event cap");
    }
    if !snapshot.world_size.is_finite() || snapshot.world_size.min_element() <= 0.0 {
        return Err("visual-event world dimensions are not finite and positive");
    }
    for (ordinal, event) in snapshot.visual.events.iter().enumerate() {
        if event.tick.0 > snapshot.tick || usize::try_from(event.ordinal).ok() != Some(ordinal) {
            return Err("visual-event tick or accepted ordinal is invalid");
        }
        let payload_is_finite = match event.event {
            WorldVisualEvent::CombatHit { damage } => damage.is_finite(),
            WorldVisualEvent::Eat { amount } => amount.is_finite(),
            WorldVisualEvent::Boost { magnitude } => magnitude.is_finite(),
            _ => true,
        };
        if !event.position.x.is_finite()
            || !event.position.y.is_finite()
            || !event.direction.iter().all(|value| value.is_finite())
            || !payload_is_finite
        {
            return Err("visual-event spatial data or magnitude is nonfinite");
        }
        let cue = visual_cue_for_event(&event.event);
        if !cue.intensity.is_finite()
            || !cue.radius.is_finite()
            || cue.duration_ticks.saturating_add(7) > MAX_LIFETIME
        {
            return Err("visual-event recipe exceeds the finite lifetime contract");
        }
    }
    Ok(())
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct ParticleMaterial {
    #[storage(0, read_only)]
    rows: Handle<ShaderStorageBuffer>,
    // Live count, nonpositive presentation offset, world width, world height.
    #[uniform(1)]
    parameters: Vec4,
    #[texture(2)]
    #[sampler(3)]
    atlas: Handle<Image>,
}

impl Material for ParticleMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://scriptbots_bevy/particle_render.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://scriptbots_bevy/particle_render.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }
}

#[derive(Component)]
struct ParticleBatch;

// Every slot/index is bounded by MAX_PARTICLES; f32 represents them exactly.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn billboard_mesh(capacity: usize) -> Mesh {
    let mut positions = Vec::with_capacity(capacity * 4);
    let mut indices = Vec::with_capacity(capacity * 6);
    for slot in 0..capacity {
        let base = (slot * 4) as u32;
        for corner in [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]] {
            positions.push([corner[0], corner[1], slot as f32]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_indices(Indices::U32(indices))
}

// Presentation can interpolate between observed cuts, never extrapolate science.
// Absolute ticks never enter a float uniform; the bounded difference is exact.
#[allow(clippy::cast_precision_loss)]
fn presentation_offset(snapshot: &WorldSnapshot) -> f32 {
    if snapshot.paused || snapshot.reduced_motion {
        return 0.0;
    }
    let Some(previous) = snapshot
        .motion
        .as_ref()
        .and_then(|motion| motion.previous.as_ref())
    else {
        return 0.0;
    };
    let span = snapshot.tick.saturating_sub(previous.tick);
    if span == 0 || span > u64::from(MAX_LIFETIME) || !snapshot.motion_alpha.is_finite() {
        return 0.0;
    }
    (snapshot.motion_alpha.clamp(0.0, 1.0) - 1.0) * span as f32
}

// GPU admission and handle installation form one transaction; splitting these
// parameters would obscure which assets belong to the same resident batch.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]
fn synchronize_particles(
    mut commands: Commands,
    state: Res<SnapshotState>,
    settings: Res<EffectiveRenderSettings>,
    accessibility: Res<AccessibilityState>,
    mut scene: ResMut<ParticleScene>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ParticleMaterial>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut images: ResMut<Assets<Image>>,
    batches: Query<(), With<ParticleBatch>>,
) {
    let Some(snapshot) = state.latest.as_ref() else {
        return;
    };
    if scene.max_rows == 0 {
        scene.last_error = Some("adapter cannot bind the native particle storage buffer");
        return;
    }
    let limit = usize::try_from(settings.features.particles_max).unwrap_or(MAX_PARTICLES);
    match scene.observe(snapshot, limit, accessibility.palette()) {
        Ok(true) => {}
        Ok(false) => return,
        Err(reason) => {
            if scene.last_error != Some(reason) {
                warn!(reason, "native visual-event boundary rejected");
            }
            scene.last_error = Some(reason);
            return;
        }
    }
    scene.last_error = None;
    let required = scene.pool.capacity().max(1);
    let rebuilt_buffer =
        scene.buffer.is_none() || scene.mesh.is_none() || required > scene.gpu_capacity;
    if rebuilt_buffer {
        let capacity = required
            .next_power_of_two()
            .min(scene.max_rows)
            .min(MAX_PARTICLES);
        let mut buffer =
            ShaderStorageBuffer::with_size(capacity * ROW_BYTES, RenderAssetUsages::default());
        buffer.buffer_description.label = Some("scriptbots visual-event particles");
        buffer.buffer_description.usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        scene.buffer = Some(buffers.add(buffer));
        scene.mesh = Some(meshes.add(billboard_mesh(capacity)));
        scene.gpu_capacity = capacity;
    }
    let buffer = scene
        .buffer
        .as_ref()
        .expect("particle buffer admitted")
        .clone();
    let parameters = Vec4::new(
        scene.rows.len() as f32,
        presentation_offset(snapshot),
        snapshot.world_size.x,
        snapshot.world_size.y,
    );
    let mut rebuilt_material = false;
    if let Some(handle) = scene.material.as_ref() {
        let Some(material) = materials.get(handle) else {
            scene.last_error = Some("native particle material disappeared");
            return;
        };
        if material.rows != buffer || material.parameters != parameters {
            let material = materials
                .get_mut(handle)
                .expect("particle material present");
            material.rows = buffer.clone();
            material.parameters = parameters;
        }
    } else {
        let atlas = images.add(Image::new(
            Extent3d {
                width: ATLAS_TILE_SIZE * 5,
                height: ATLAS_TILE_SIZE,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            bake_sprite_atlas(0, ATLAS_TILE_SIZE),
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        ));
        scene.material = Some(materials.add(ParticleMaterial {
            rows: buffer,
            parameters,
            atlas,
        }));
        rebuilt_material = true;
    }
    let mesh = scene.mesh.as_ref().expect("particle mesh admitted").clone();
    let material = scene
        .material
        .as_ref()
        .expect("particle material admitted")
        .clone();
    if let Some(entity) = scene.entity.filter(|entity| batches.contains(*entity)) {
        if rebuilt_buffer || rebuilt_material {
            commands
                .entity(entity)
                .insert((Mesh3d(mesh), MeshMaterial3d(material)));
        }
    } else {
        scene.entity = Some(
            commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(material),
                    Transform::IDENTITY,
                    NoFrustumCulling,
                    NotShadowCaster,
                    NotShadowReceiver,
                    ParticleBatch,
                ))
                .id(),
        );
    }
}

#[derive(Resource, Default)]
struct ExtractedParticles {
    rows: Vec<ParticleRow>,
    bytes: Vec<u8>,
    revision: u64,
    buffer: Option<Handle<ShaderStorageBuffer>>,
    uploaded: Option<(BufferId, u64)>,
}

fn extract_particles(
    scene: Extract<Res<ParticleScene>>,
    mut extracted: ResMut<ExtractedParticles>,
) {
    if extracted.revision != scene.revision || extracted.buffer != scene.buffer {
        extracted.rows.clone_from(&scene.rows);
        extracted.revision = scene.revision;
        if extracted.buffer != scene.buffer {
            extracted.buffer.clone_from(&scene.buffer);
            extracted.uploaded = None;
        }
    }
}

fn upload_particles(
    mut extracted: ResMut<ExtractedParticles>,
    buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    queue: Res<RenderQueue>,
) {
    let Some(buffer) = extracted
        .buffer
        .as_ref()
        .and_then(|handle| buffers.get(handle))
    else {
        return;
    };
    let key = (buffer.buffer.id(), extracted.revision);
    if extracted.uploaded == Some(key) {
        return;
    }
    let extracted = &mut *extracted;
    extracted.bytes.clear();
    for row in &extracted.rows {
        row.append_bytes(&mut extracted.bytes);
    }
    if !extracted.bytes.is_empty() {
        queue.write_buffer(&buffer.buffer, 0, &extracted.bytes);
    }
    extracted.uploaded = Some(key);
}

fn report_particle_failure(
    scene: Res<ParticleScene>,
    mut exit: MessageWriter<AppExit>,
    mut reported: Local<bool>,
) {
    if let Some(reason) = scene.error()
        && !*reported
    {
        error!(reason, "native visual-event particles failed");
        *reported = true;
        exit.write(AppExit::error());
    }
}

pub(crate) struct ParticleRenderPlugin;

impl Plugin for ParticleRenderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "particle_render.wgsl");
        app.init_resource::<ParticleScene>()
            .add_plugins(MaterialPlugin::<ParticleMaterial> {
                prepass_enabled: false,
                shadows_enabled: false,
                ..default()
            })
            .add_systems(
                PostUpdate,
                synchronize_particles.before(VisibilitySystems::CalculateBounds),
            )
            .add_systems(Last, report_particle_failure);
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .init_resource::<ExtractedParticles>()
                .add_systems(ExtractSchedule, extract_particles)
                .add_systems(
                    Render,
                    upload_particles.in_set(RenderSystems::PrepareResources),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(device) = app.world().get_resource::<RenderDevice>() {
            let limits = device.limits();
            let rows = if limits.max_storage_buffers_per_shader_stage == 0 {
                0
            } else {
                limits.max_storage_buffer_binding_size as usize / ROW_BYTES
            };
            app.world_mut().resource_mut::<ParticleScene>().max_rows = rows.min(MAX_PARTICLES);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particles::SpriteKind;
    use scriptbots_core::{BirthOrigin, DeathCause, Position, ScriptBotsConfig, Tick};
    use scriptbots_runtime::{HostCore, HostCoreOptions};

    fn located(tick: u64, ordinal: u32, event: WorldVisualEvent) -> LocatedWorldVisualEvent {
        LocatedWorldVisualEvent {
            tick: Tick(tick),
            ordinal,
            source: None,
            target: None,
            position: Position::new(30.0, 40.0),
            direction: [1.0, 0.0],
            event,
        }
    }

    fn birth() -> WorldVisualEvent {
        WorldVisualEvent::Birth {
            origin: BirthOrigin::Born,
        }
    }

    fn frame(tick: u64, events: &[WorldVisualEvent]) -> WorldSnapshot {
        let mut snapshot = crate::terrain_tests::sample_world_snapshot();
        snapshot.tick = tick;
        snapshot.visual = VisualFrame {
            seed: 731,
            epoch: tick / 10_000,
            events: Arc::new(
                events
                    .iter()
                    .enumerate()
                    .map(|(ordinal, event)| located(tick, u32::try_from(ordinal).unwrap(), *event))
                    .collect(),
            ),
            dropped: 0,
        };
        snapshot
    }

    fn observe(scene: &mut ParticleScene, snapshot: &WorldSnapshot, capacity: usize) {
        scene
            .observe(snapshot, capacity, ColorPaletteMode::Natural)
            .unwrap();
    }

    #[test]
    fn actual_host_events_and_entropy_root_reach_renderer_without_changing_science() {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 50,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            rng_seed: None,
            ..default()
        })
        .unwrap();
        world.push_visual_event(
            Tick(0),
            None,
            None,
            Position::new(30.0, 40.0),
            [1.0, 0.0],
            birth(),
        );
        let root = world.agent_substream_protocol_v1().root_seed();
        let digest = world.world_digest_v1().unwrap();
        let direct = VisualFrame::from_world(&world);
        let host = HostCore::new(
            HostSessionId::new(42),
            world,
            HostCoreOptions {
                capture_agent_visuals: true,
                ..default()
            },
        )
        .unwrap();
        let published = host.latest_snapshot();
        assert_eq!(published.config.rng_seed, None);
        assert_eq!(published.visual_seed, root);
        assert_eq!(VisualFrame::from_snapshot(&published), direct);
        let snapshot = WorldSnapshot::from_snapshot(&published).unwrap();
        let mut scene = ParticleScene::default();
        for _ in 0..5 {
            observe(&mut scene, &snapshot, 256);
        }
        assert!(!scene.rows.is_empty());
        assert_eq!(scene.stats.accepted_cues, 1);
        assert_eq!(host.world().tick(), Tick(0));
        assert_eq!(host.world().world_digest_v1().unwrap(), digest);

        let mut encoded = serde_json::to_value(published.as_ref()).unwrap();
        let decoded: RenderSnapshot = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded.visual_seed, root);
        encoded.as_object_mut().unwrap().remove("visual_seed");
        assert!(serde_json::from_value::<RenderSnapshot>(encoded).is_err());
    }

    #[test]
    fn repeated_boundaries_and_same_tick_controls_consume_each_ordinal_once() {
        let mut snapshot = frame(10, &[birth(), WorldVisualEvent::Eat { amount: 0.3 }]);
        snapshot.visual.dropped = 3;
        let mut scene = ParticleScene::default();
        observe(&mut scene, &snapshot, 256);
        let rows = scene.rows.clone();
        let revision = scene.revision;
        for index in 0..20 {
            snapshot.revision += 1;
            snapshot.paused = index % 2 == 0;
            observe(&mut scene, &snapshot, 256);
        }
        assert_eq!(scene.rows, rows);
        assert_eq!(scene.revision, revision);
        assert_eq!(scene.stats.accepted_cues, 2);
        assert_eq!(scene.stats.producer_dropped_events, 3);

        Arc::make_mut(&mut snapshot.visual.events).push(located(
            10,
            2,
            WorldVisualEvent::Reproduce,
        ));
        observe(&mut scene, &snapshot, 256);
        assert_eq!(scene.stats.accepted_cues, 3);
        let rows = scene.rows.clone();
        Arc::make_mut(&mut snapshot.visual.events).truncate(1);
        assert_eq!(
            scene.observe(&snapshot, 256, ColorPaletteMode::Natural),
            Err("visual-event prefix regressed within one completed boundary")
        );
        assert_eq!(scene.rows, rows);
    }

    #[test]
    fn missing_boundaries_are_counted_stale_data_ignored_and_sessions_reset() {
        let first = frame(7, &[birth()]);
        let mut scene = ParticleScene::default();
        observe(&mut scene, &first, 256);
        let later = frame(11, &[]);
        observe(&mut scene, &later, 256);
        assert_eq!(scene.stats.skipped_boundaries, 3);
        let rows = scene.rows.clone();
        observe(&mut scene, &first, 256);
        assert_eq!(scene.rows, rows);
        assert_eq!(scene.stats.stale_boundaries, 1);

        let mut next = frame(0, &[birth()]);
        next.visual.seed += 1;
        observe(&mut scene, &next, 256);
        let mut fresh = ParticleScene::default();
        observe(&mut fresh, &next, 256);
        assert_eq!(scene.rows, fresh.rows);
        assert_eq!(scene.stats.source_resets, 2);
        crate::terrain_tests::motion_test_source(&mut next);
        observe(&mut scene, &next, 256);
        assert_eq!(scene.stats.source_resets, 3);
        assert_eq!(scene.stats.accepted_cues, 3);
    }

    #[test]
    fn ordinary_epoch_rollover_retains_live_particles_and_stamped_interventions_age_correctly() {
        let mut scene = ParticleScene::default();
        observe(&mut scene, &frame(9_999, &[birth()]), 256);
        let count = scene.rows.len();
        let mut next = frame(10_000, &[birth()]);
        Arc::make_mut(&mut next.visual.events)[0].tick = Tick(9_999);
        observe(&mut scene, &next, 256);
        assert_eq!(scene.stats.source_resets, 1);
        assert_eq!(scene.stats.skipped_boundaries, 0);
        assert_eq!(scene.rows.len(), count * 2);
        assert!(scene.rows.iter().all(|row| row.0[2][0] == 1.0));
        let mut invalid = frame(10_001, &[birth()]);
        Arc::make_mut(&mut invalid.visual.events)[0].tick = Tick(10_002);
        assert!(
            scene
                .observe(&invalid, 256, ColorPaletteMode::Natural)
                .is_err()
        );
    }

    #[test]
    fn exact_accepted_ordinals_drive_all_event_recipes_and_deaths_are_critical() {
        let snapshot = frame(
            19,
            &[
                WorldVisualEvent::Eat { amount: 0.4 },
                birth(),
                WorldVisualEvent::Death {
                    cause: DeathCause::Starvation,
                },
                WorldVisualEvent::Death {
                    cause: DeathCause::CombatCarnivore,
                },
                WorldVisualEvent::CombatHit { damage: 0.2 },
                WorldVisualEvent::Reproduce,
                WorldVisualEvent::SpikeExtend,
                WorldVisualEvent::Boost { magnitude: 0.8 },
            ],
        );
        let mut scene = ParticleScene::default();
        observe(&mut scene, &snapshot, 512);
        let mut sprites = [false; 5];
        let mut actual_birth = None;
        scene.pool.for_each_live(|particle| {
            sprites[particle.sprite.tile_index()] = true;
            if particle.priority == ParticlePriority::Critical
                && particle.sprite == SpriteKind::Spark
                && actual_birth.is_none()
            {
                actual_birth = Some(*particle);
            }
            if particle.sprite == SpriteKind::Mote {
                assert_eq!(particle.priority, ParticlePriority::Critical);
            }
        });
        assert!(sprites.into_iter().all(|present| present));
        let expected = CueScheduler::new(snapshot.visual.seed).schedule(
            19,
            1,
            &visual_cue_for_event(&birth()),
            [0.0; 3],
        );
        let actual = actual_birth.unwrap();
        assert_eq!(actual.velocity, expected.particles[0].velocity);
        assert_eq!(actual.rotation, expected.particles[0].rotation);
        assert_eq!(scene.stats.accepted_cues, 8);
    }

    #[test]
    fn event_storms_and_quality_changes_respect_caps_and_preserve_critical_particles() {
        let mut events = vec![WorldVisualEvent::Eat { amount: 0.4 }; 100];
        events.extend(std::iter::repeat_n(
            WorldVisualEvent::Death {
                cause: DeathCause::Starvation,
            },
            40,
        ));
        let snapshot = frame(12, &events);
        let mut scene = ParticleScene {
            max_rows: 64,
            ..default()
        };
        observe(&mut scene, &snapshot, MAX_PARTICLES);
        assert_eq!(scene.pool.capacity(), 64);
        assert_eq!(scene.rows.len(), 64);
        assert_eq!(scene.stats.accepted_cues, 64);
        assert_eq!(scene.stats.budget_dropped_cues, 76);
        assert!(scene.stats.refused_particles > 0);
        scene.pool.for_each_live(|particle| {
            assert_eq!(particle.priority, ParticlePriority::Critical);
        });
        observe(&mut scene, &snapshot, 8);
        assert_eq!(scene.rows.len(), 8);
        assert_eq!(scene.stats.tier_dropped_particles, 56);
        assert_eq!(scene.stats.accepted_cues, 64);
        let before = scene.rows.clone();
        let oversized = frame(13, &vec![birth(); MAX_EVENTS + 1]);
        assert!(
            scene
                .observe(&oversized, 8, ColorPaletteMode::Natural)
                .is_err()
        );
        let mut corrupt = frame(13, &[birth()]);
        Arc::make_mut(&mut corrupt.visual.events)[0].ordinal = 2;
        assert!(
            scene
                .observe(&corrupt, 8, ColorPaletteMode::Natural)
                .is_err()
        );
        assert_eq!(scene.rows, before);
    }

    #[test]
    fn huge_ticks_keep_one_tick_precision_and_pause_stops_fractional_motion() {
        let tick = u64::MAX - 20;
        let mut scene = ParticleScene::default();
        observe(&mut scene, &frame(tick, &[birth()]), 256);
        let later = frame(tick + 1, &[]);
        observe(&mut scene, &later, 256);
        assert!(!scene.rows.is_empty());
        assert!(scene.rows.iter().all(|row| row.0[2][0] == 1.0));
        let rows = scene.rows.clone();
        for _ in 0..50 {
            observe(&mut scene, &later, 256);
        }
        assert_eq!(scene.rows, rows);
        let previous = frame(10, &[]);
        let mut current = frame(11, &[]);
        crate::terrain_tests::explicit_test_motion(&previous, &mut current, 0.25);
        assert_eq!(presentation_offset(&current), -0.75);
        current.paused = true;
        assert_eq!(presentation_offset(&current), 0.0);
        current.paused = false;
        current.reduced_motion = true;
        assert_eq!(presentation_offset(&current), 0.0);
    }

    #[test]
    fn synchronization_installs_one_resident_batch_and_capture_reset_reuses_assets() {
        let mut app = App::new();
        app.insert_resource(SnapshotState {
            latest: Some(frame(4, &[birth()])),
            ..default()
        })
        .insert_resource(EffectiveRenderSettings {
            tier: RenderQuality::Low,
            requested: RenderQuality::Low,
            features: tier_features(RenderQuality::Low),
            fog_settings: None,
            bloom_settings: None,
            gpu: None,
        })
        .insert_resource(AccessibilityState::new())
        .init_resource::<ParticleScene>()
        .init_resource::<Assets<Mesh>>()
        .init_resource::<Assets<ParticleMaterial>>()
        .init_resource::<Assets<ShaderStorageBuffer>>()
        .init_resource::<Assets<Image>>()
        .add_systems(PostUpdate, synchronize_particles);
        app.update();
        let scene = app.world().resource::<ParticleScene>();
        let original_entity = scene.entity.unwrap();
        let handles = (
            scene.buffer.clone(),
            scene.mesh.clone(),
            scene.material.clone(),
        );
        assert!(!scene.rows.is_empty());
        assert!(app.world().get::<Mesh3d>(original_entity).is_some());
        assert!(
            app.world()
                .get::<MeshMaterial3d<ParticleMaterial>>(original_entity)
                .is_some()
        );
        for _ in 0..8 {
            app.update();
        }
        let scene = app.world().resource::<ParticleScene>();
        assert_eq!(scene.stats.accepted_cues, 1);
        assert_eq!(
            (
                scene.buffer.clone(),
                scene.mesh.clone(),
                scene.material.clone()
            ),
            handles
        );
        assert_eq!(
            app.world().resource::<Assets<ShaderStorageBuffer>>().len(),
            1
        );

        app.world_mut().despawn(original_entity);
        app.world_mut().resource_mut::<ParticleScene>().reset();
        app.update();
        let scene = app.world().resource::<ParticleScene>();
        assert_ne!(scene.entity.unwrap(), original_entity);
        assert!(!scene.rows.is_empty());
        assert_eq!(
            (
                scene.buffer.clone(),
                scene.mesh.clone(),
                scene.material.clone()
            ),
            handles
        );
        assert_eq!(
            app.world().resource::<Assets<ShaderStorageBuffer>>().len(),
            1
        );
    }

    #[test]
    fn shader_storage_layout_and_byte_packing_match_the_actual_billboard_source() {
        let source = include_str!("particle_render.wgsl")
            .replace(
                "#import bevy_pbr::mesh_view_bindings::view",
                "struct TestView { clip_from_world: mat4x4<f32>, world_from_view: mat4x4<f32>, }; \
                 @group(0) @binding(0) var<uniform> view: TestView;",
            )
            .replace("#{MATERIAL_BIND_GROUP}", "3");
        let module = naga::front::wgsl::parse_str(&source).expect("actual particle WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("actual particle WGSL validates");
        let (_, particle_type) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("ParticleRow"))
            .unwrap();
        let naga::TypeInner::Struct { members, span } = &particle_type.inner else {
            panic!("particle row must be a storage struct");
        };
        assert_eq!(usize::try_from(*span).unwrap(), ROW_BYTES);
        assert_eq!(
            members
                .iter()
                .map(|member| member.offset)
                .collect::<Vec<_>>(),
            vec![0, 16, 32, 48, 64, 80]
        );
        let row = ParticleRow(std::array::from_fn(|vector| {
            std::array::from_fn(|component| {
                f32::from(u16::try_from(vector * 4 + component).unwrap())
            })
        }));
        let mut bytes = Vec::new();
        row.append_bytes(&mut bytes);
        assert_eq!(bytes.len(), ROW_BYTES);
        for (index, chunk) in bytes.chunks_exact(4).enumerate() {
            assert_eq!(
                f32::from_le_bytes(chunk.try_into().unwrap()),
                f32::from(u16::try_from(index).unwrap())
            );
        }
        let mesh = billboard_mesh(4);
        assert_eq!(mesh.count_vertices(), 16);
        assert_eq!(mesh.indices().unwrap().len(), 24);
    }
}
