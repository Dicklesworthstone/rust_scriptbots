//! Bounded, versioned presentation frames shared by native servers and browser clients.
//!
//! The six-byte header is `SBWS` followed by a little-endian protocol version.
//! The remainder is one complete Postcard value. This envelope leaves the existing
//! dynamic-world v2 codec unchanged and deliberately does not serialize an entire
//! host snapshot, checkpoint, or JSON-oriented narrative record positionally.

use crate::{PlaybackSnapshot, RenderSnapshot};
use scriptbots_core::{
    DynamicAgentSnapshot, DynamicSnapshotSummary, DynamicSnapshotWorld, DynamicWorldSnapshot,
    Region, TerrainKind, Tick, TickSummaryDto,
    narrative::{EVENT_RECORD_SCHEMA_VERSION, EventKind, EventRecord, SubjectRef},
};
use serde::{Deserialize, Deserializer, Serialize, de};
use std::{fmt, marker::PhantomData, sync::Arc};
use thiserror::Error;

/// Version of the positional payload following the fixed header.
pub const REMOTE_PROTOCOL_VERSION: u16 = 2;
/// Maximum complete binary frame, including its header.
pub const REMOTE_MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;
/// Maximum narrative records in one event page.
pub const REMOTE_MAX_EVENTS: usize = 64;
/// Maximum dynamic agents in one canvas frame.
pub const REMOTE_MAX_AGENTS: usize = 65_536;
/// Largest side of a sampled environment overview. Large worlds retain their full extent.
pub const REMOTE_ENVIRONMENT_MAX_SIDE: u32 = 128;
/// Maximum environment cells allocated by either decoder or projection.
pub const REMOTE_MAX_ENVIRONMENT_CELLS: usize =
    (REMOTE_ENVIRONMENT_MAX_SIDE * REMOTE_ENVIRONMENT_MAX_SIDE) as usize;
/// Maximum bytes in a narrative metric name or canonical subject reference.
pub const REMOTE_MAX_EVENT_LABEL_BYTES: usize = 256;
/// Maximum bytes in one canonical narrative description.
pub const REMOTE_MAX_EVENT_TEXT_BYTES: usize = 4096;
const MAGIC: &[u8; 4] = b"SBWS";
const HEADER_BYTES: usize = MAGIC.len() + 2;

/// Exact host publication used to build every payload in a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteFrameIdentity {
    /// Host identity; browsers expose it as a decimal string.
    pub session_id: u128,
    /// Publication revision within that host session.
    pub publication_revision: u64,
    /// Scientific-state revision, independent of tick and publication revision.
    pub scientific_revision: u64,
    /// Command-visible control revision.
    pub control_revision: u64,
    /// Active configuration revision.
    pub config_revision: u64,
    /// Completed scientific tick of the publication.
    pub tick: u64,
}

impl From<&RenderSnapshot> for RemoteFrameIdentity {
    fn from(snapshot: &RenderSnapshot) -> Self {
        Self {
            session_id: u128::from(snapshot.session_id.get()),
            publication_revision: snapshot.revision.get(),
            scientific_revision: snapshot.revisions.scientific.get(),
            control_revision: snapshot.revisions.control.get(),
            config_revision: snapshot.revisions.config.get(),
            tick: snapshot.world.tick,
        }
    }
}

/// One coherent presentation payload and the playback state observed with it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoteFrame {
    /// Source identity shared by metadata and payload.
    pub identity: RemoteFrameIdentity,
    /// Observed owner playback state; a command submission is not this observation.
    pub playback: PlaybackSnapshot,
    /// Earlier publication revisions skipped by this client's latest-value lane.
    pub coalesced_publications: u64,
    /// Coherent bounded environment overview for canvas frames; absent on other channels.
    #[serde(with = "crate::serde_optional_arc")]
    pub environment: Option<Arc<RemoteEnvironmentSnapshot>>,
    /// Exactly one negotiated channel's payload.
    pub payload: RemoteFramePayload,
}

/// Positional payload discriminants; changing their order changes the protocol.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RemoteFramePayload {
    /// The complete canonical aggregate for this scientific tick.
    Telemetry(TickSummaryDto),
    /// The unchanged dynamic-world v2 projection consumed by existing renderers.
    Canvas(#[serde(deserialize_with = "decode_canvas")] DynamicWorldSnapshot),
    /// Bounded canonical narrative records with explicit retention loss.
    Events(RemoteEventPage),
}

/// One actual source-grid sample, not an inferred scientific aggregate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteEnvironmentCell {
    /// Source terrain class.
    pub terrain: TerrainKind,
    /// Source elevation.
    pub elevation: f32,
    /// Source food quantity.
    pub food: f32,
    /// Source water depth, zero when hydrology is disabled.
    pub water_depth: f32,
}

/// Bounded environment samples covering the entire authoritative grid.
///
/// Small grids are exact. Larger grids use the source cell at each overview cell's
/// center; values are never described as averages or used for scientific metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoteEnvironmentSnapshot {
    /// Overview width, at most [`REMOTE_ENVIRONMENT_MAX_SIDE`].
    pub width: u32,
    /// Overview height, at most [`REMOTE_ENVIRONMENT_MAX_SIDE`].
    pub height: u32,
    /// Original source grid width.
    pub source_width: u32,
    /// Original source grid height.
    pub source_height: u32,
    /// Source cell edge length in world units.
    pub cell_size: u32,
    /// Exact source terrain content revision.
    pub terrain_revision: u64,
    /// Exact source food content revision.
    pub food_revision: u64,
    /// Exact source hydrology content revision.
    pub hydrology_revision: u64,
    /// Configured food scale, captured with the same publication.
    pub food_max: f32,
    /// Whether water samples come from an active hydrology layer.
    pub hydrology_enabled: bool,
    /// Row-major source samples, bounded before allocation during decode.
    #[serde(deserialize_with = "decode_environment_cells")]
    pub cells: Vec<RemoteEnvironmentCell>,
}

impl RemoteEnvironmentSnapshot {
    /// Sample one immutable publication without taking an owner lock or advancing science.
    ///
    /// # Errors
    /// Refuses inconsistent geometry, incomplete layers and invalid sampled values.
    pub fn capture(snapshot: &RenderSnapshot) -> Result<Self, RemoteProtocolError> {
        let terrain = &snapshot.layers.terrain;
        let food = &snapshot.layers.food;
        let source_len = usize::try_from(u64::from(terrain.width) * u64::from(terrain.height))
            .map_err(|_| RemoteProtocolError::InvalidPayload("environment source dimensions"))?;
        if terrain.width == 0
            || terrain.height == 0
            || terrain.cell_size == 0
            || terrain.tiles.len() != source_len
            || food.width != terrain.width
            || food.height != terrain.height
            || food.cells.len() != source_len
            || snapshot.layers.hydrology.as_ref().is_some_and(|water| {
                water.width != terrain.width
                    || water.height != terrain.height
                    || water.water_depth.len() != source_len
            })
        {
            return Err(RemoteProtocolError::InvalidPayload(
                "environment source layers",
            ));
        }
        let width = terrain.width.min(REMOTE_ENVIRONMENT_MAX_SIDE);
        let height = terrain.height.min(REMOTE_ENVIRONMENT_MAX_SIDE);
        let projection_len = usize::try_from(width * height)
            .map_err(|_| RemoteProtocolError::InvalidPayload("environment overview dimensions"))?;
        let source_width = usize::try_from(terrain.width)
            .map_err(|_| RemoteProtocolError::InvalidPayload("environment source width"))?;
        let mut cells = Vec::with_capacity(projection_len);
        for y in 0..height {
            let source_y = usize::try_from(
                (2 * u64::from(y) + 1) * u64::from(terrain.height) / (2 * u64::from(height)),
            )
            .map_err(|_| RemoteProtocolError::InvalidPayload("environment source row"))?;
            for x in 0..width {
                let source_x = usize::try_from(
                    (2 * u64::from(x) + 1) * u64::from(terrain.width) / (2 * u64::from(width)),
                )
                .map_err(|_| RemoteProtocolError::InvalidPayload("environment source column"))?;
                let index = source_y * source_width + source_x;
                let tile = terrain.tiles[index];
                cells.push(RemoteEnvironmentCell {
                    terrain: tile.kind,
                    elevation: tile.elevation,
                    food: food.cells[index],
                    water_depth: snapshot
                        .layers
                        .hydrology
                        .as_ref()
                        .map_or(0.0, |water| water.water_depth[index]),
                });
            }
        }
        let result = Self {
            width,
            height,
            source_width: terrain.width,
            source_height: terrain.height,
            cell_size: terrain.cell_size,
            terrain_revision: snapshot.layers.revisions.terrain.get(),
            food_revision: snapshot.layers.revisions.food.get(),
            hydrology_revision: snapshot.layers.revisions.hydrology.get(),
            food_max: snapshot.config.food_max,
            hydrology_enabled: snapshot.layers.hydrology.is_some(),
            cells,
        };
        result.validate(&snapshot.world.world)?;
        Ok(result)
    }

    fn validate(&self, world: &DynamicSnapshotWorld) -> Result<(), RemoteProtocolError> {
        require_limit(
            "environment cells",
            self.cells.len(),
            REMOTE_MAX_ENVIRONMENT_CELLS,
        )?;
        if self.width == 0
            || self.height == 0
            || self.width != self.source_width.min(REMOTE_ENVIRONMENT_MAX_SIDE)
            || self.height != self.source_height.min(REMOTE_ENVIRONMENT_MAX_SIDE)
            || self.cell_size == 0
            || u64::from(self.source_width) * u64::from(self.cell_size) != u64::from(world.width)
            || u64::from(self.source_height) * u64::from(self.cell_size) != u64::from(world.height)
            || usize::try_from(self.width * self.height).ok() != Some(self.cells.len())
            || !self.food_max.is_finite()
            || self.food_max <= 0.0
            || self.cells.iter().any(|cell| {
                !cell.elevation.is_finite()
                    || !cell.food.is_finite()
                    || cell.food < 0.0
                    || !cell.water_depth.is_finite()
                    || cell.water_depth < 0.0
                    || (!self.hydrology_enabled && cell.water_depth != 0.0)
            })
        {
            return Err(RemoteProtocolError::InvalidPayload(
                "environment geometry or values",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EnvironmentSource {
    session: u128,
    epoch: u64,
    world: (u32, u32),
    terrain: u64,
    food: u64,
    hydrology: u64,
    config: u64,
}

/// One bounded cached overview; no queue of past publications is retained.
#[derive(Debug, Clone, Default)]
pub struct RemoteEnvironmentCache {
    cached: Option<(EnvironmentSource, Arc<RemoteEnvironmentSnapshot>)>,
}

impl RemoteEnvironmentCache {
    /// Reuse exact revisioned layers, rebuilding only when their source changes.
    ///
    /// # Errors
    /// Propagates the same source refusals as [`RemoteEnvironmentSnapshot::capture`].
    pub fn snapshot(
        &mut self,
        source: &RenderSnapshot,
    ) -> Result<Arc<RemoteEnvironmentSnapshot>, RemoteProtocolError> {
        let key = EnvironmentSource {
            session: u128::from(source.session_id.get()),
            epoch: source.world.epoch,
            world: (source.world.world.width, source.world.world.height),
            terrain: source.layers.revisions.terrain.get(),
            food: source.layers.revisions.food.get(),
            hydrology: source.layers.revisions.hydrology.get(),
            config: source.revisions.config.get(),
        };
        if let Some((previous, value)) = &self.cached
            && *previous == key
        {
            return Ok(Arc::clone(value));
        }
        let value = Arc::new(RemoteEnvironmentSnapshot::capture(source)?);
        self.cached = Some((key, Arc::clone(&value)));
        Ok(value)
    }
}

/// Missing narrative ordinals explicitly reported before the page's retained suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteEventGap {
    /// First requested ordinal that is no longer retained.
    pub first_missing: u64,
    /// Last missing ordinal, inclusive.
    pub last_missing: u64,
}

/// A page after a session-bound narrative cursor, never an unlabelled replacement tail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoteEventPage {
    /// Cursor requested by this client before reading the page.
    pub after_event: u64,
    /// First ordinal still retained by the source; latest + 1 when empty.
    pub earliest_retained: u64,
    /// Last narrative ordinal published by the source at this boundary.
    pub latest_event: u64,
    /// Exact unavailable prefix, if retention overtook the requested cursor.
    pub gap: Option<RemoteEventGap>,
    /// Contiguous records following the cursor or declared gap, oldest first.
    #[serde(deserialize_with = "decode_events")]
    pub events: Vec<RemoteSequencedEvent>,
}

impl RemoteEventPage {
    /// Cursor reached by accepting this page, including any explicitly signalled gap.
    #[must_use]
    pub fn next_after_event(&self) -> u64 {
        self.events.last().map_or_else(
            || self.gap.map_or(self.after_event, |gap| gap.last_missing),
            |event| event.sequence,
        )
    }
}

/// One canonical narrative record at its retained-ring ordinal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoteSequencedEvent {
    /// One-based ordinal: source dropped count + retained index + 1.
    pub sequence: u64,
    /// Scientific narrative content, preserved without re-narration.
    pub record: RemoteEventRecord,
}

/// Postcard-compatible projection of every field in a canonical `EventRecord`.
// EventRecord omits absent subjects when serializing, and SubjectRef uses a
// JSON map serializer. Neither is a positional binary schema. An explicit
// always-present Option and the existing canonical subject representation avoid
// changing the core type, its characterization hashes, or its JSON consumers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoteEventRecord {
    /// Canonical narrative schema version.
    pub schema_version: u32,
    /// Tick at which the narrative detector identified the event.
    pub tick: u64,
    /// Canonical event category.
    pub kind: EventKind,
    /// Canonical importance in the inclusive range zero through one.
    pub severity: f32,
    /// Magnitude reported by the detector.
    pub magnitude: f64,
    /// Detection window, inclusive start and end ticks.
    pub window: (u64, u64),
    /// Canonical metric name.
    #[serde(deserialize_with = "decode_label")]
    pub metric: String,
    /// Representative value before the change.
    pub before: f64,
    /// Representative value after the change.
    pub after: f64,
    /// Detector score.
    pub score: f64,
    /// Optional exact `SubjectRef::to_db_string` representation.
    #[serde(deserialize_with = "decode_subject")]
    pub subject: Option<String>,
    /// Canonical deterministic prose; this transport never regenerates it.
    #[serde(deserialize_with = "decode_event_text")]
    pub human_text: String,
}

impl From<&EventRecord> for RemoteEventRecord {
    fn from(record: &EventRecord) -> Self {
        Self {
            schema_version: record.schema_version,
            tick: record.tick.0,
            kind: record.kind,
            severity: record.severity,
            magnitude: record.magnitude,
            window: record.window,
            metric: record.metric.clone(),
            before: record.before,
            after: record.after,
            score: record.score,
            subject: record.subject.map(SubjectRef::to_db_string),
            human_text: record.human_text.clone(),
        }
    }
}

impl RemoteEventRecord {
    /// Restore the exact canonical narrative fields after validating the wire record.
    ///
    /// # Errors
    /// Refuses unknown schemas, non-finite values, invalid bounds, or noncanonical subjects.
    pub fn to_event_record(&self) -> Result<EventRecord, RemoteProtocolError> {
        self.validate(u64::MAX)?;
        Ok(EventRecord {
            schema_version: self.schema_version,
            tick: Tick(self.tick),
            kind: self.kind,
            severity: self.severity,
            magnitude: self.magnitude,
            window: self.window,
            metric: self.metric.clone(),
            before: self.before,
            after: self.after,
            score: self.score,
            subject: self.subject.as_deref().map(parse_subject).transpose()?,
            human_text: self.human_text.clone(),
        })
    }

    fn validate(&self, tick: u64) -> Result<(), RemoteProtocolError> {
        if self.schema_version != EVENT_RECORD_SCHEMA_VERSION
            || self.tick > tick
            || self.window.0 > self.window.1
            || self.window.1 > tick
            || !(0.0..=1.0).contains(&self.severity)
            || ![self.magnitude, self.before, self.after, self.score]
                .iter()
                .all(|number| number.is_finite())
        {
            return Err(RemoteProtocolError::InvalidPayload(
                "invalid narrative record",
            ));
        }
        require_limit(
            "event metric",
            self.metric.len(),
            REMOTE_MAX_EVENT_LABEL_BYTES,
        )?;
        require_limit(
            "event text",
            self.human_text.len(),
            REMOTE_MAX_EVENT_TEXT_BYTES,
        )?;
        if let Some(subject) = &self.subject {
            require_limit("event subject", subject.len(), REMOTE_MAX_EVENT_LABEL_BYTES)?;
            parse_subject(subject)?;
        }
        Ok(())
    }
}

/// Fail-closed framing, allocation, or source-coherence refusal.
#[derive(Debug, Error)]
pub enum RemoteProtocolError {
    /// A binary frame did not begin with the complete framing header.
    #[error("missing or invalid remote frame magic")]
    InvalidMagic,
    /// The version is not implemented by this decoder.
    #[error("unsupported remote protocol version {0}")]
    UnsupportedVersion(u16),
    /// A complete frame exceeds the transport byte cap.
    #[error("remote frame has {0} bytes; limit is {REMOTE_MAX_FRAME_BYTES}")]
    FrameTooLarge(usize),
    /// A bounded collection or text field exceeds its declared protocol limit.
    #[error("{field} has {observed} entries/bytes; limit is {limit}")]
    LimitExceeded {
        /// Bounded field being decoded or encoded.
        field: &'static str,
        /// Actual observed cardinality or byte length.
        observed: usize,
        /// Maximum accepted value.
        limit: usize,
    },
    /// Postcard could not encode or decode the complete typed payload.
    #[error("malformed remote payload: {0}")]
    Codec(#[from] postcard::Error),
    /// A frame contains bytes after its single complete value.
    #[error("remote frame has trailing bytes")]
    TrailingBytes,
    /// Payload evidence disagrees with its source identity or protocol invariants.
    #[error("invalid remote payload: {0}")]
    InvalidPayload(&'static str),
}

impl RemoteFrame {
    /// Validate all payload and resource invariants without changing scientific state.
    ///
    /// # Errors
    /// Refuses mismatched ticks, invalid playback, non-finite data, broken event
    /// continuity, or a collection/text field over the protocol bounds.
    pub fn validate(&self) -> Result<(), RemoteProtocolError> {
        if !self.playback.speed_multiplier.is_finite() || self.playback.speed_multiplier < 0.0 {
            return Err(RemoteProtocolError::InvalidPayload(
                "invalid playback speed",
            ));
        }
        if let Some(environment) = &self.environment {
            let RemoteFramePayload::Canvas(snapshot) = &self.payload else {
                return Err(RemoteProtocolError::InvalidPayload(
                    "environment on non-canvas channel",
                ));
            };
            environment.validate(&snapshot.world)?;
        }
        match &self.payload {
            RemoteFramePayload::Telemetry(summary) => {
                if summary.tick != self.identity.tick
                    || ![
                        summary.total_energy,
                        summary.average_energy,
                        summary.average_health,
                    ]
                    .iter()
                    .all(|number| number.is_finite())
                {
                    return Err(RemoteProtocolError::InvalidPayload(
                        "telemetry identity or values",
                    ));
                }
            }
            RemoteFramePayload::Canvas(snapshot) => {
                require_limit("canvas agents", snapshot.agents.len(), REMOTE_MAX_AGENTS)?;
                if snapshot.tick != self.identity.tick
                    || snapshot.world.width == 0
                    || snapshot.world.height == 0
                    || snapshot.summary.agent_count != snapshot.agents.len()
                    || ![
                        snapshot.summary.total_energy,
                        snapshot.summary.average_energy,
                        snapshot.summary.average_health,
                    ]
                    .iter()
                    .all(|number| number.is_finite())
                    || !snapshot.agents.iter().all(|agent| {
                        agent
                            .position
                            .iter()
                            .chain(&agent.velocity)
                            .chain(&agent.color)
                            .all(|number| number.is_finite())
                            && [
                                agent.heading,
                                agent.health,
                                agent.energy,
                                agent.spike_length,
                                agent.herbivore_tendency,
                            ]
                            .iter()
                            .all(|number| number.is_finite())
                    })
                {
                    return Err(RemoteProtocolError::InvalidPayload(
                        "canvas identity or values",
                    ));
                }
            }
            RemoteFramePayload::Events(page) => {
                require_limit("event page", page.events.len(), REMOTE_MAX_EVENTS)?;
                if page.earliest_retained == 0
                    || page.after_event > page.latest_event
                    || page.earliest_retained > page.latest_event.saturating_add(1)
                {
                    return Err(RemoteProtocolError::InvalidPayload(
                        "event retention cursor",
                    ));
                }
                let first_requested = page.after_event.checked_add(1);
                let expected_gap = first_requested
                    .filter(|first| *first < page.earliest_retained)
                    .map(|first_missing| RemoteEventGap {
                        first_missing,
                        last_missing: page.earliest_retained - 1,
                    });
                if page.gap != expected_gap {
                    return Err(RemoteProtocolError::InvalidPayload(
                        "event gap differs from retention",
                    ));
                }
                let mut next = first_requested.map(|first| first.max(page.earliest_retained));
                for event in &page.events {
                    if Some(event.sequence) != next || event.sequence > page.latest_event {
                        return Err(RemoteProtocolError::InvalidPayload(
                            "non-contiguous event page",
                        ));
                    }
                    event.record.validate(self.identity.tick)?;
                    next = event.sequence.checked_add(1);
                }
            }
        }
        Ok(())
    }
}

/// Encode one validated frame with its explicit magic and version.
///
/// # Errors
/// Refuses invalid or oversized payloads. Cardinality and text limits are checked
/// before serialization, bounding its allocation independently of supplied lengths.
pub fn encode_remote_frame(frame: &RemoteFrame) -> Result<Vec<u8>, RemoteProtocolError> {
    frame.validate()?;
    let payload = postcard::to_allocvec(frame)?;
    let length = HEADER_BYTES.saturating_add(payload.len());
    if length > REMOTE_MAX_FRAME_BYTES {
        return Err(RemoteProtocolError::FrameTooLarge(length));
    }
    let mut bytes = Vec::with_capacity(length);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&REMOTE_PROTOCOL_VERSION.to_le_bytes());
    bytes.extend_from_slice(&payload);
    Ok(bytes)
}

/// Decode exactly one bounded frame, preserving every integer identity bit.
///
/// # Errors
/// Refuses oversized input before decoding, unsupported versions, malformed
/// payloads, trailing bytes, allocation-length attacks and incoherent identities.
pub fn decode_remote_frame(bytes: &[u8]) -> Result<RemoteFrame, RemoteProtocolError> {
    if bytes.len() > REMOTE_MAX_FRAME_BYTES {
        return Err(RemoteProtocolError::FrameTooLarge(bytes.len()));
    }
    if bytes.len() < HEADER_BYTES || &bytes[..MAGIC.len()] != MAGIC {
        return Err(RemoteProtocolError::InvalidMagic);
    }
    let version = u16::from_le_bytes([bytes[MAGIC.len()], bytes[MAGIC.len() + 1]]);
    if version != REMOTE_PROTOCOL_VERSION {
        return Err(RemoteProtocolError::UnsupportedVersion(version));
    }
    let (frame, trailing) = postcard::take_from_bytes::<RemoteFrame>(&bytes[HEADER_BYTES..])?;
    if !trailing.is_empty() {
        return Err(RemoteProtocolError::TrailingBytes);
    }
    frame.validate()?;
    Ok(frame)
}

const fn require_limit(
    field: &'static str,
    observed: usize,
    limit: usize,
) -> Result<(), RemoteProtocolError> {
    if observed > limit {
        return Err(RemoteProtocolError::LimitExceeded {
            field,
            observed,
            limit,
        });
    }
    Ok(())
}

fn parse_subject(encoded: &str) -> Result<SubjectRef, RemoteProtocolError> {
    let invalid = || RemoteProtocolError::InvalidPayload("noncanonical narrative subject");
    let subject = SubjectRef::from_db_string(encoded).map_err(|_| invalid())?;
    let valid_region = match subject {
        SubjectRef::Region(Region::Disc { x, y, radius }) => {
            x.is_finite() && y.is_finite() && radius.is_finite() && radius >= 0.0
        }
        SubjectRef::Region(Region::Rect { x, y, w, h }) => {
            [x, y, w, h].iter().all(|value| value.is_finite()) && w >= 0.0 && h >= 0.0
        }
        _ => true,
    };
    if !valid_region || subject.to_db_string() != encoded {
        return Err(invalid());
    }
    Ok(subject)
}

// Deserialize the unchanged v2 field order while checking its Vec length before
// reserving memory. These helpers intentionally do not change core serialization.
fn decode_canvas<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<DynamicWorldSnapshot, D::Error> {
    #[derive(Deserialize)]
    struct BoundedWorld {
        tick: u64,
        epoch: u64,
        world: DynamicSnapshotWorld,
        summary: DynamicSnapshotSummary,
        #[serde(deserialize_with = "decode_agents")]
        agents: Vec<DynamicAgentSnapshot>,
    }
    let world = BoundedWorld::deserialize(deserializer)?;
    Ok(DynamicWorldSnapshot {
        tick: world.tick,
        epoch: world.epoch,
        world: world.world,
        summary: world.summary,
        agents: world.agents,
    })
}

fn bounded_vec<'de, D, T, const MAX: usize>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Visitor<T, const MAX: usize>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const MAX: usize> de::Visitor<'de> for Visitor<T, MAX> {
        type Value = Vec<T>;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "at most {MAX} entries")
        }
        fn visit_seq<A: de::SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> Result<Self::Value, A::Error> {
            if sequence.size_hint().is_some_and(|length| length > MAX) {
                return Err(de::Error::custom("remote collection exceeds its entry cap"));
            }
            let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(MAX));
            while let Some(value) = sequence.next_element()? {
                if values.len() == MAX {
                    return Err(de::Error::custom("remote collection exceeds its entry cap"));
                }
                values.push(value);
            }
            Ok(values)
        }
    }
    deserializer.deserialize_seq(Visitor::<T, MAX>(PhantomData))
}

fn decode_agents<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<DynamicAgentSnapshot>, D::Error> {
    bounded_vec::<D, DynamicAgentSnapshot, REMOTE_MAX_AGENTS>(deserializer)
}

fn decode_environment_cells<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<RemoteEnvironmentCell>, D::Error> {
    bounded_vec::<D, RemoteEnvironmentCell, REMOTE_MAX_ENVIRONMENT_CELLS>(deserializer)
}

fn decode_events<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<RemoteSequencedEvent>, D::Error> {
    bounded_vec::<D, RemoteSequencedEvent, REMOTE_MAX_EVENTS>(deserializer)
}

struct BoundedText<const MAX: usize>(String);

impl<'de, const MAX: usize> Deserialize<'de> for BoundedText<MAX> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor<const MAX: usize>;
        impl<const MAX: usize> de::Visitor<'_> for Visitor<MAX> {
            type Value = BoundedText<MAX>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "a string of at most {MAX} bytes")
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                if value.len() > MAX {
                    return Err(E::custom("remote text exceeds its byte cap"));
                }
                Ok(BoundedText(value.to_owned()))
            }
        }
        deserializer.deserialize_str(Visitor::<MAX>)
    }
}

fn decode_label<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    BoundedText::<REMOTE_MAX_EVENT_LABEL_BYTES>::deserialize(deserializer).map(|text| text.0)
}

fn decode_event_text<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    BoundedText::<REMOTE_MAX_EVENT_TEXT_BYTES>::deserialize(deserializer).map(|text| text.0)
}

fn decode_subject<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
    Option::<BoundedText<REMOTE_MAX_EVENT_LABEL_BYTES>>::deserialize(deserializer)
        .map(|text| text.map(|text| text.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CommandEnvelope, CommandId, HostCommand, HostCore, HostCoreOptions, HostPort,
        HostSessionId, ManualHostDriver, ManualInstant,
    };
    use scriptbots_core::{AgentData, AgentUid, Intervention, ScriptBotsConfig, WorldState};

    fn host_with_narrative() -> HostCore {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            closed: true,
            population_spawn_interval: 0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            aging_health_decay_rate: 0.0,
            spike_damage: 0.0,
            spike_energy_cost: 0.0,
            reproduction_energy_threshold: f32::MAX,
            persistence_interval: 0,
            rng_seed: Some(0x16_020_901),
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        for _ in 0..12 {
            world
                .try_spawn_agent(AgentData {
                    health: 2.0,
                    ..AgentData::default()
                })
                .unwrap();
        }
        for tick in 1..=240 {
            if tick == 97 {
                world
                    .enqueue_intervention(Intervention::Meteor {
                        region: Region::All,
                        lethality: 10.0,
                        scorch: 0.0,
                    })
                    .unwrap();
            }
            world.step().unwrap();
        }
        let mut host = HostCore::new(
            HostSessionId::new(81),
            world,
            HostCoreOptions {
                initial_playback: PlaybackSnapshot {
                    paused: true,
                    speed_multiplier: 1.0,
                },
                ..HostCoreOptions::default()
            },
        )
        .unwrap();
        host.local_port()
            .submit(CommandEnvelope::new(CommandId::new(1), HostCommand::Step))
            .unwrap();
        host.drive(ManualInstant::from_nanos(1)).unwrap();
        host
    }

    fn canvas_frame() -> RemoteFrame {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 80,
            world_height: 80,
            food_cell_size: 40,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            rng_seed: Some(17),
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        world.try_spawn_agent(AgentData::default()).unwrap();
        let host =
            HostCore::new(HostSessionId::new(17), world, HostCoreOptions::default()).unwrap();
        let publication = host.latest_snapshot();
        RemoteFrame {
            identity: RemoteFrameIdentity::from(publication.as_ref()),
            playback: publication.playback,
            coalesced_publications: 0,
            environment: Some(Arc::new(
                RemoteEnvironmentSnapshot::capture(&publication).unwrap(),
            )),
            payload: RemoteFramePayload::Canvas(publication.world.clone()),
        }
    }

    fn unchecked_frame(frame: &RemoteFrame) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&REMOTE_PROTOCOL_VERSION.to_le_bytes());
        bytes.extend(postcard::to_allocvec(frame).unwrap());
        bytes
    }

    #[test]
    fn real_host_snapshot_and_narrative_round_trip_without_changing_science() {
        let host = host_with_narrative();
        let before = host.scientific_digest_v1().unwrap();
        let snapshot = host.latest_snapshot();
        assert_eq!(snapshot.world.tick, 241);
        assert!(
            !snapshot.narrative_events.is_empty(),
            "real extinction must emit narrative"
        );
        let page = RemoteEventPage {
            after_event: snapshot.narrative_dropped_events,
            earliest_retained: snapshot.narrative_dropped_events + 1,
            latest_event: snapshot.narrative_dropped_events
                + snapshot.narrative_events.len() as u64,
            gap: None,
            events: snapshot
                .narrative_events
                .iter()
                .enumerate()
                .map(|(index, record)| RemoteSequencedEvent {
                    sequence: snapshot.narrative_dropped_events + index as u64 + 1,
                    record: RemoteEventRecord::from(record),
                })
                .collect(),
        };
        for payload in [
            RemoteFramePayload::Canvas(snapshot.world.clone()),
            RemoteFramePayload::Telemetry(snapshot.completed_summary.clone().unwrap().into()),
            RemoteFramePayload::Events(page),
        ] {
            let frame = RemoteFrame {
                identity: RemoteFrameIdentity::from(snapshot.as_ref()),
                playback: snapshot.playback,
                coalesced_publications: 7,
                environment: matches!(&payload, RemoteFramePayload::Canvas(_))
                    .then(|| Arc::new(RemoteEnvironmentSnapshot::capture(&snapshot).unwrap())),
                payload,
            };
            let decoded = decode_remote_frame(&encode_remote_frame(&frame).unwrap()).unwrap();
            assert_eq!(decoded, frame);
            if let RemoteFramePayload::Events(page) = decoded.payload {
                let records = page
                    .events
                    .iter()
                    .map(|event| event.record.to_event_record().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(records, *snapshot.narrative_events);
            }
        }
        assert_eq!(host.scientific_digest_v1().unwrap(), before);
    }

    #[test]
    fn every_narrative_subject_and_absent_subject_survive_positional_encoding() {
        let mut frame = canvas_frame();
        frame.identity.tick = 100;
        for subject in [
            None,
            Some(SubjectRef::Agent(AgentUid(u64::MAX))),
            Some(SubjectRef::Species(7)),
            Some(SubjectRef::Island(4)),
            Some(SubjectRef::Region(Region::All)),
            Some(SubjectRef::Region(Region::Disc {
                x: 1.5,
                y: 2.0,
                radius: 3.25,
            })),
            Some(SubjectRef::Region(Region::Rect {
                x: -1.5,
                y: 2.0,
                w: 3.25,
                h: 4.0,
            })),
        ] {
            let record = EventRecord {
                schema_version: EVENT_RECORD_SCHEMA_VERSION,
                tick: Tick(100),
                kind: EventKind::PopulationCrash,
                severity: 0.8,
                magnitude: 50.0,
                window: (90, 100),
                metric: "population".to_owned(),
                before: 100.0,
                after: 50.0,
                score: 4.5,
                subject,
                human_text: "population fell 50%".to_owned(),
            };
            frame.environment = None;
            frame.payload = RemoteFramePayload::Events(RemoteEventPage {
                after_event: 0,
                earliest_retained: 1,
                latest_event: 1,
                gap: None,
                events: vec![RemoteSequencedEvent {
                    sequence: 1,
                    record: (&record).into(),
                }],
            });
            let decoded = decode_remote_frame(&encode_remote_frame(&frame).unwrap()).unwrap();
            let RemoteFramePayload::Events(page) = decoded.payload else {
                panic!("event frame")
            };
            assert_eq!(page.events[0].record.to_event_record().unwrap(), record);
        }
    }

    #[test]
    fn rejects_wrong_versions_trailing_bytes_and_incoherent_publications() {
        let frame = canvas_frame();
        let bytes = encode_remote_frame(&frame).unwrap();
        for length in 0..HEADER_BYTES {
            assert!(decode_remote_frame(&bytes[..length]).is_err());
        }
        let mut bad = bytes.clone();
        bad[0] ^= 1;
        assert!(matches!(
            decode_remote_frame(&bad),
            Err(RemoteProtocolError::InvalidMagic)
        ));
        bad = bytes.clone();
        bad[4..6].copy_from_slice(&1_u16.to_le_bytes());
        assert!(matches!(
            decode_remote_frame(&bad),
            Err(RemoteProtocolError::UnsupportedVersion(1))
        ));
        bad = bytes.clone();
        bad.push(0);
        assert!(matches!(
            decode_remote_frame(&bad),
            Err(RemoteProtocolError::TrailingBytes)
        ));
        assert!(decode_remote_frame(&bytes[..bytes.len() - 1]).is_err());
        let mut wrong = frame.clone();
        wrong.identity.tick += 1;
        assert!(encode_remote_frame(&wrong).is_err());
        assert!(decode_remote_frame(&unchecked_frame(&wrong)).is_err());
        wrong = frame;
        wrong.playback.speed_multiplier = f32::NAN;
        assert!(decode_remote_frame(&unchecked_frame(&wrong)).is_err());
        for length in [
            REMOTE_MAX_FRAME_BYTES - 1,
            REMOTE_MAX_FRAME_BYTES,
            REMOTE_MAX_FRAME_BYTES + 1,
        ] {
            let result = decode_remote_frame(&vec![0; length]);
            assert_eq!(
                matches!(result, Err(RemoteProtocolError::FrameTooLarge(_))),
                length > REMOTE_MAX_FRAME_BYTES
            );
        }
    }

    #[test]
    fn canvas_caps_apply_before_allocation_and_keep_the_dynamic_v2_payload_exact() {
        let mut frame = canvas_frame();
        let RemoteFramePayload::Canvas(world) = &mut frame.payload else {
            panic!("canvas frame")
        };
        let agent = world.agents[0].clone();
        for count in [REMOTE_MAX_AGENTS - 1, REMOTE_MAX_AGENTS] {
            let RemoteFramePayload::Canvas(world) = &mut frame.payload else {
                panic!("canvas frame")
            };
            world.agents = vec![agent.clone(); count];
            world.summary.agent_count = count;
            let expected = postcard::to_allocvec(world).unwrap();
            let decoded = decode_remote_frame(&encode_remote_frame(&frame).unwrap()).unwrap();
            let RemoteFramePayload::Canvas(decoded) = decoded.payload else {
                panic!("canvas frame")
            };
            assert_eq!(postcard::to_allocvec(&decoded).unwrap(), expected);
        }
        let RemoteFramePayload::Canvas(world) = &mut frame.payload else {
            panic!("canvas frame")
        };
        world.agents.push(agent);
        world.summary.agent_count += 1;
        assert!(encode_remote_frame(&frame).is_err());
        assert!(decode_remote_frame(&unchecked_frame(&frame)).is_err());
        let RemoteFramePayload::Canvas(world) = &mut frame.payload else {
            panic!("canvas frame")
        };
        world.agents.clear();
        world.summary.agent_count = 0;
        let mut poisoned_length = unchecked_frame(&frame);
        assert_eq!(poisoned_length.pop(), Some(0));
        poisoned_length.extend(postcard::to_allocvec(&usize::MAX).unwrap());
        assert!(
            decode_remote_frame(&poisoned_length).is_err(),
            "an attacker length must refuse before reserve"
        );
    }

    #[test]
    fn environment_exact_samples_revisions_and_cache_preserve_owner_science() {
        let host = host_with_narrative();
        let before = host.scientific_digest_v1().unwrap();
        let publication = host.latest_snapshot();
        let mut cache = RemoteEnvironmentCache::default();
        let first = cache.snapshot(&publication).unwrap();
        assert_eq!(first.cells.len(), publication.layers.food.cells.len());
        for (index, cell) in first.cells.iter().enumerate() {
            assert_eq!(cell.terrain, publication.layers.terrain.tiles[index].kind);
            assert_eq!(
                cell.elevation.to_bits(),
                publication.layers.terrain.tiles[index].elevation.to_bits()
            );
            assert_eq!(
                cell.food.to_bits(),
                publication.layers.food.cells[index].to_bits()
            );
            assert_eq!(
                cell.water_depth.to_bits(),
                publication
                    .layers
                    .hydrology
                    .as_ref()
                    .map_or(0.0, |water| water.water_depth[index])
                    .to_bits()
            );
        }
        let repeated = cache.snapshot(&publication).unwrap();
        assert!(Arc::ptr_eq(&first, &repeated));
        let mut changed = (*publication).clone();
        changed.revisions.control = crate::ControlRevision::new(100);
        changed.revision = crate::SnapshotRevision::new(100);
        assert!(Arc::ptr_eq(&first, &cache.snapshot(&changed).unwrap()));
        changed.layers.revisions.food = crate::LayerRevision::new(first.food_revision + 1);
        Arc::make_mut(&mut changed.layers.food).cells[0] = 0.125;
        let next = cache.snapshot(&changed).unwrap();
        assert!(!Arc::ptr_eq(&first, &next));
        assert_eq!(next.cells[0].food.to_bits(), 0.125_f32.to_bits());
        changed.session_id = HostSessionId::new(999);
        let new_session = cache.snapshot(&changed).unwrap();
        assert!(!Arc::ptr_eq(&next, &new_session));
        changed.world.epoch += 1;
        assert!(!Arc::ptr_eq(
            &new_session,
            &cache.snapshot(&changed).unwrap()
        ));
        assert_eq!(host.scientific_digest_v1().unwrap(), before);
    }

    #[test]
    fn environment_overview_covers_large_non_square_grid_with_bounded_center_samples() {
        let host = host_with_narrative();
        let mut source = (*host.latest_snapshot()).clone();
        source.world.world.width = 259;
        source.world.world.height = 131;
        let terrain = Arc::make_mut(&mut source.layers.terrain);
        let tile = terrain.tiles[0];
        terrain.width = 259;
        terrain.height = 131;
        terrain.cell_size = 1;
        terrain.tiles = vec![tile; 259 * 131];
        let food = Arc::make_mut(&mut source.layers.food);
        food.width = 259;
        food.height = 131;
        food.cells = (0..259 * 131_u16).map(f32::from).collect();
        source.layers.hydrology = None;
        let view = RemoteEnvironmentSnapshot::capture(&source).unwrap();
        assert_eq!((view.width, view.height), (128, 128));
        assert_eq!(view.cells.len(), REMOTE_MAX_ENVIRONMENT_CELLS);
        assert_eq!(
            (view.source_width, view.source_height, view.cell_size),
            (259, 131, 1)
        );
        assert_eq!(view.cells[0].food.to_bits(), 1.0_f32.to_bits());
        assert_eq!(view.cells[127].food.to_bits(), 257.0_f32.to_bits());
        assert_eq!(
            view.cells[127 * 128].food.to_bits(),
            f32::from(130 * 259 + 1_u16).to_bits()
        );
        assert_eq!(
            view.cells[128 * 128 - 1].food.to_bits(),
            f32::from(130 * 259 + 257_u16).to_bits()
        );
        assert_eq!(
            postcard::from_bytes::<RemoteEnvironmentSnapshot>(
                &postcard::to_allocvec(&view).unwrap()
            )
            .unwrap(),
            view
        );
    }

    #[test]
    fn environment_decode_refuses_geometry_values_channel_mix_and_allocation_attacks() {
        let valid = canvas_frame();
        let mut invalid = valid.clone();
        let env = Arc::make_mut(invalid.environment.as_mut().unwrap());
        env.width = 1;
        env.source_width = 1;
        env.cells.truncate(2);
        assert!(
            encode_remote_frame(&invalid).is_err(),
            "a coherent smaller grid must not hide half the world"
        );
        assert!(decode_remote_frame(&unchecked_frame(&invalid)).is_err());
        for bad in [f32::NAN, f32::INFINITY, -0.5] {
            let mut invalid = valid.clone();
            Arc::make_mut(invalid.environment.as_mut().unwrap()).cells[0].food = bad;
            assert!(encode_remote_frame(&invalid).is_err());
            assert!(decode_remote_frame(&unchecked_frame(&invalid)).is_err());
        }
        let mut invalid = valid.clone();
        Arc::make_mut(invalid.environment.as_mut().unwrap()).cells[0].water_depth = 1.0;
        assert!(
            encode_remote_frame(&invalid).is_err(),
            "disabled hydrology cannot fabricate water"
        );
        invalid.payload = RemoteFramePayload::Events(RemoteEventPage {
            after_event: 0,
            earliest_retained: 1,
            latest_event: 0,
            gap: None,
            events: Vec::new(),
        });
        assert!(
            decode_remote_frame(&unchecked_frame(&invalid)).is_err(),
            "non-canvas channel cannot carry environment"
        );
        let mut env = valid.environment.unwrap().as_ref().clone();
        env.cells.clear();
        let mut attack = postcard::to_allocvec(&env).unwrap();
        assert_eq!(attack.pop(), Some(0));
        attack.extend(postcard::to_allocvec(&usize::MAX).unwrap());
        assert!(postcard::from_bytes::<RemoteEnvironmentSnapshot>(&attack).is_err());
        env.cells = vec![
            RemoteEnvironmentCell {
                terrain: TerrainKind::Grass,
                elevation: 0.0,
                food: 0.0,
                water_depth: 0.0
            };
            REMOTE_MAX_ENVIRONMENT_CELLS + 1
        ];
        assert!(
            postcard::from_bytes::<RemoteEnvironmentSnapshot>(
                &postcard::to_allocvec(&env).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn event_caps_continuity_and_explicit_gaps_are_validated_on_both_sides() {
        let mut frame = canvas_frame();
        let record = RemoteEventRecord {
            schema_version: EVENT_RECORD_SCHEMA_VERSION,
            tick: 0,
            kind: EventKind::PopulationBoom,
            severity: 0.5,
            magnitude: 1.0,
            window: (0, 0),
            metric: "population".to_owned(),
            before: 1.0,
            after: 2.0,
            score: 2.0,
            subject: None,
            human_text: "population rose".to_owned(),
        };
        for count in [
            REMOTE_MAX_EVENTS - 1,
            REMOTE_MAX_EVENTS,
            REMOTE_MAX_EVENTS + 1,
        ] {
            frame.environment = None;
            frame.payload = RemoteFramePayload::Events(RemoteEventPage {
                after_event: 2,
                earliest_retained: 5,
                latest_event: count as u64 + 4,
                gap: Some(RemoteEventGap {
                    first_missing: 3,
                    last_missing: 4,
                }),
                events: (0..count)
                    .map(|index| RemoteSequencedEvent {
                        sequence: index as u64 + 5,
                        record: record.clone(),
                    })
                    .collect(),
            });
            assert_eq!(
                encode_remote_frame(&frame).is_ok(),
                count <= REMOTE_MAX_EVENTS
            );
            assert_eq!(
                decode_remote_frame(&unchecked_frame(&frame)).is_ok(),
                count <= REMOTE_MAX_EVENTS
            );
        }
        let RemoteFramePayload::Events(page) = &mut frame.payload else {
            panic!("event frame")
        };
        page.events.truncate(REMOTE_MAX_EVENTS);
        let valid = frame.clone();
        for fault in 0..6 {
            let mut broken = valid.clone();
            let RemoteFramePayload::Events(page) = &mut broken.payload else {
                panic!("event frame")
            };
            match fault {
                0 => page.gap = None,
                1 => page.events[1].sequence += 1,
                2 => page.events[0].record.tick += 1,
                3 => page.events[0].record.subject = Some("agent:0001".to_owned()),
                4 => page.events[0].record.human_text = "x".repeat(REMOTE_MAX_EVENT_TEXT_BYTES + 1),
                _ => page.events[0].record.metric = "x".repeat(REMOTE_MAX_EVENT_LABEL_BYTES + 1),
            }
            assert!(encode_remote_frame(&broken).is_err(), "fault {fault}");
            assert!(
                decode_remote_frame(&unchecked_frame(&broken)).is_err(),
                "fault {fault}"
            );
        }
    }
}
