use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use scriptbots_core::{
    AgentDebugInfo, AgentDebugQuery, ControlCommand, DietClass, DraftGeneratorConfig,
    ExistingAgentPolicy, HydrologyFlowDirection, MapArtifact, RuleBasedMapGenerator,
    SampleBasedMapGenerator, ScriptBotsConfig, SelectionMode, SelectionState, SelectionUpdate,
    TerrainKind, Tick, WorldState, default_tileset_spec,
};

use scriptbots_core::ConfigAuditEntry;
use scriptbots_core::check_knob_ranges;
#[cfg(feature = "gui")]
use scriptbots_render::{OffscreenScene, render_offscreen_scene};
use scriptbots_runtime::{
    ApplicationState, CommandEnvelope, CommandId, HostCommand, HostPort, JournalState,
    RenderSnapshot, ScientificRevision, channel::ChannelHostPort,
};
use smallvec::SmallVec;

/// Snapshot of configuration state returned to external clients.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ConfigSnapshot {
    pub tick: u64,
    pub config: Value,
}

impl ConfigSnapshot {
    fn from_world(config: &ScriptBotsConfig, tick: Tick) -> Result<Self, ControlError> {
        let config_value = serde_json::to_value(config).map_err(ControlError::serialization)?;
        Ok(Self {
            tick: tick.0,
            config: config_value,
        })
    }
}

/// Status summary of the running simulation for control clients.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SimulationStatusDto {
    pub tick: u64,
    pub agent_count: usize,
    pub is_closed: bool,
    pub config_revision: u64,
    /// Owner scientific revision for guarding operations against stale world observations.
    pub scientific_revision: u64,
    /// Automatic ticking state observed at this publication, not a queued request.
    pub paused: bool,
    /// Owner lifecycle; stopping does not establish completed shutdown.
    #[schema(value_type = String)]
    pub lifecycle: scriptbots_runtime::HostLifecycle,
    /// Exact owner blocker or fault, independent of storage durability.
    #[schema(value_type = Object)]
    pub health: scriptbots_runtime::HostHealth,
    /// Most recently applied command; application does not prove journal durability.
    pub last_applied_command: Option<String>,
    /// Admitted commands still waiting at the observed owner boundary.
    pub command_queue_depth: usize,
    /// Publication identity for detecting stale observations while science is paused.
    pub snapshot_revision: u64,
}

/// Exact, bounded browser observation of one immutable owner publication.
#[derive(Debug, Clone, PartialEq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct RemoteOwnerStatusDto {
    pub schema: String,
    pub identity: RemoteOwnerIdentityDto,
    #[schema(value_type = String)]
    pub lifecycle: scriptbots_runtime::HostLifecycle,
    pub playback: RemoteOwnerPlaybackDto,
    pub health: String,
    pub blocker: Option<String>,
    pub fault: Option<RemoteOwnerFaultDto>,
}

/// Decimal strings preserve full-width identities in ordinary browser JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct RemoteOwnerIdentityDto {
    pub session_id: String,
    pub publication_revision: String,
    pub scientific_revision: String,
    pub control_revision: String,
    pub config_revision: String,
    pub tick: String,
}

/// Browser playback uses the same field names as its remote canvas projection.
#[derive(Debug, Clone, PartialEq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct RemoteOwnerPlaybackDto {
    pub paused: bool,
    pub speed_multiplier: f32,
}

/// Typed fault identity; bounded diagnostic text never becomes HTML in the client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RemoteOwnerFaultDto {
    Scientific {
        tick: String,
        code: String,
        message: String,
        diagnostic_truncated: bool,
    },
    Journal {
        session_id: String,
        batch_sequence: String,
        code: String,
        message: String,
        diagnostic_truncated: bool,
    },
    Protocol {
        code: String,
        message: String,
        diagnostic_truncated: bool,
    },
}

fn bounded_owner_diagnostic(value: &str, byte_limit: usize) -> (String, bool) {
    let mut end = value.len().min(byte_limit);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    (value[..end].to_owned(), end != value.len())
}

impl RemoteOwnerStatusDto {
    fn from_snapshot(snapshot: &RenderSnapshot) -> Self {
        use scriptbots_runtime::{HostFault, HostHealth};

        // Even worst-case JSON escaping of these two text budgets fits the
        // browser's 16 KiB response ceiling, with room for all exact identities.
        let fault = snapshot.health.fault().map(|fault| {
            let (code, message) = match fault {
                HostFault::Scientific { code, message, .. }
                | HostFault::Protocol { code, message } => (code.as_str(), message.as_str()),
                HostFault::Journal { failure, .. } => {
                    (failure.code.as_str(), failure.message.as_str())
                }
            };
            let (code, code_truncated) = bounded_owner_diagnostic(code, 128);
            let (message, message_truncated) = bounded_owner_diagnostic(message, 2048);
            let diagnostic_truncated = code_truncated || message_truncated;
            match fault {
                HostFault::Scientific { tick, .. } => RemoteOwnerFaultDto::Scientific {
                    tick: tick.0.to_string(),
                    code,
                    message,
                    diagnostic_truncated,
                },
                HostFault::Journal { batch_id, .. } => RemoteOwnerFaultDto::Journal {
                    session_id: batch_id.session_id().get().to_string(),
                    batch_sequence: batch_id.sequence().to_string(),
                    code,
                    message,
                    diagnostic_truncated,
                },
                HostFault::Protocol { .. } => RemoteOwnerFaultDto::Protocol {
                    code,
                    message,
                    diagnostic_truncated,
                },
            }
        });
        Self {
            schema: "scriptbots.remote-owner-status.v1".to_owned(),
            identity: RemoteOwnerIdentityDto {
                session_id: snapshot.session_id.get().to_string(),
                publication_revision: snapshot.revision.get().to_string(),
                scientific_revision: snapshot.revisions.scientific.get().to_string(),
                control_revision: snapshot.revisions.control.get().to_string(),
                config_revision: snapshot.revisions.config.get().to_string(),
                tick: snapshot.world.tick.to_string(),
            },
            lifecycle: snapshot.lifecycle,
            playback: RemoteOwnerPlaybackDto {
                paused: snapshot.playback.paused,
                speed_multiplier: snapshot.playback.speed_multiplier,
            },
            health: match &snapshot.health {
                HostHealth::Healthy => "healthy",
                HostHealth::Blocked(_) => "blocked",
                HostHealth::Faulted(_) => "faulted",
            }
            .to_owned(),
            blocker: snapshot
                .health
                .blocker()
                .map(|blocker| format!("{blocker:?}")),
            fault,
        }
    }
}

/// Snapshot describing the current hydrology state.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct HydrologySnapshot {
    pub width: u32,
    pub height: u32,
    pub total_water_depth: f32,
    pub mean_water_depth: f32,
    pub flooded_shallow_count: u32,
    pub flooded_deep_count: u32,
    pub shallow_threshold: f32,
    pub deep_threshold: f32,
    #[schema(value_type = Vec<f32>)]
    pub water_depth: Vec<f32>,
    #[schema(value_type = Vec<String>)]
    pub flow_directions: Vec<String>,
    #[schema(value_type = Vec<u32>)]
    pub basin_ids: Vec<u32>,
    #[schema(value_type = Vec<f32>)]
    pub accumulation: Vec<f32>,
    #[schema(value_type = Vec<f32>)]
    pub spill_elevation: Vec<f32>,
}

impl HydrologySnapshot {
    const SHALLOW_THRESHOLD: f32 = 0.05;
    const DEEP_THRESHOLD: f32 = 0.2;

    fn from_snapshot(state: &scriptbots_runtime::HydrologyLayerSnapshot) -> Self {
        let total_water_depth: f32 = state.water_depth.iter().sum();
        let cell_count = state.water_depth.len().max(1) as f32;
        let shallow = state
            .water_depth
            .iter()
            .filter(|&&depth| depth >= Self::SHALLOW_THRESHOLD)
            .count();
        let deep = state
            .water_depth
            .iter()
            .filter(|&&depth| depth >= Self::DEEP_THRESHOLD)
            .count();

        let flow_directions = state
            .flow_directions
            .iter()
            .map(|direction| {
                match direction {
                    HydrologyFlowDirection::North => "N",
                    HydrologyFlowDirection::South => "S",
                    HydrologyFlowDirection::East => "E",
                    HydrologyFlowDirection::West => "W",
                    HydrologyFlowDirection::None => "-",
                }
                .to_owned()
            })
            .collect();

        Self {
            width: state.width,
            height: state.height,
            total_water_depth,
            mean_water_depth: total_water_depth / cell_count,
            flooded_shallow_count: saturating_u32(shallow),
            flooded_deep_count: saturating_u32(deep),
            shallow_threshold: Self::SHALLOW_THRESHOLD,
            deep_threshold: Self::DEEP_THRESHOLD,
            water_depth: state.water_depth.clone(),
            flow_directions,
            basin_ids: state.basin_ids.clone(),
            accumulation: state.accumulation.clone(),
            spill_elevation: state.spill_elevation.clone(),
        }
    }
}

fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Enumeration describing the primitive type of a knob.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum KnobKind {
    Number,
    Integer,
    Boolean,
    String,
    Array,
    Object,
    Null,
}

/// Public descriptor for a single configuration knob.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KnobEntry {
    pub path: String,
    pub kind: KnobKind,
    pub value: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Request payload for updating a configuration knob.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct KnobUpdate {
    pub path: String,
    #[schema(value_type = Object, nullable = false)]
    pub value: Value,
}

/// Errors produced by the control domain when mutating configuration.
#[derive(Debug, Error)]
pub enum ControlError {
    #[error("experiment execution capacity exhausted: {0}")]
    ExperimentCapacity(String),
    #[error("an experiment comparison is already running; retry after it completes")]
    ExperimentComparisonBusy,
    #[error("experiment service is shutting down; resume retained work in a new control runtime")]
    ExperimentShuttingDown,
    #[error("experiment worker failed: {0}")]
    ExperimentWorkerFailed(String),
    #[error("checkpoint retention exhausted: {0}")]
    CheckpointCapacity(String),
    #[error("checkpoint retention unavailable: {0}")]
    CheckpointRetentionUnavailable(String),
    #[error(transparent)]
    Host(#[from] scriptbots_runtime::HostAccessError),
    #[error("failed to lock world state")]
    Lock,
    #[error("{0}")]
    InvalidPatch(String),
    #[error("unknown knob path: {0}")]
    UnknownPath(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("command queue is full; retry later")]
    CommandQueueFull,
    #[error("command queue has been closed")]
    CommandQueueClosed,
    #[error("narrative search error: {0}")]
    NarrativeSearch(#[from] crate::narrative_search::NarrativeSearchError),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("payload too large: {0}")]
    PayloadTooLarge(String),
    #[error("conflict: {0}")]
    Conflict(String),
}

impl ControlError {
    fn serialization(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

impl From<scriptbots_runtime::remote::BrowserInterventionError> for ControlError {
    fn from(error: scriptbots_runtime::remote::BrowserInterventionError) -> Self {
        use scriptbots_runtime::remote::BrowserInterventionError;
        match error {
            BrowserInterventionError::InvalidRequest(message) => Self::BadRequest(message),
            error @ (BrowserInterventionError::SourceConflict(_)
            | BrowserInterventionError::PauseRequired) => Self::Conflict(error.to_string()),
        }
    }
}

impl From<PoisonError<MutexGuard<'_, WorldState>>> for ControlError {
    fn from(_: PoisonError<MutexGuard<'_, WorldState>>) -> Self {
        ControlError::Lock
    }
}

type KnobsCache = std::sync::Arc<Mutex<Option<(u64, Vec<KnobEntry>)>>>;

/// Lock a derived cache, adopting the contents even if a previous holder panicked.
///
/// Knobs are revalidated against the published configuration revision. Recovering
/// this derived cache cannot repair or alter scientific state; the host owns that
/// state separately. Unwrapping here would make later knob requests panic after
/// an unrelated cache holder panicked (bd-2t3k).
fn lock_cache<T>(cache: &Mutex<T>) -> MutexGuard<'_, T> {
    cache.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Wire tag of `scriptbots_runtime::ApplicationState::Admitted`.
///
/// Admission establishes identity and ordering; it does not prove application.
pub const APPLICATION_STATE_ADMITTED: &str = "admitted";

/// Wire tag of `scriptbots_runtime::ApplicationState::Applied`.
///
/// Published by the host after it observes the command's application.
pub const APPLICATION_STATE_APPLIED: &str = "applied";

/// Wire tag of `scriptbots_runtime::ApplicationState::Rejected`.
///
/// A command that reached the applier and was refused there — distinct from one
/// refused at admission, which never gets a receipt at all.
pub const APPLICATION_STATE_REJECTED: &str = "rejected";

/// Wire tag of `scriptbots_runtime::JournalState::NotRequired`.
///
/// Used for non-runtime and historical producers that require no journal record.
pub const JOURNAL_STATE_NOT_REQUIRED: &str = "not_required";

/// Two-axis status representation returned by REST, MCP, and CLI interfaces for commands.
///
/// The two axes are independent by design: application tracks
/// `admitted`/`applied`/`rejected`/`failed`, journal tracks
/// `not_required`/`pending`/`committed_volatile`/`durable`. Both axes come from the
/// authoritative host receipt; admission alone does not advance either axis.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CommandStatusDto {
    pub command_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub admission_sequence: Option<u64>,
    pub application_state: String,
    /// Exact owner rejection, including requested and actual revisions for a stale guard.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Object)]
    pub rejection_reason: Option<Value>,
    pub journal_state: String,
    pub control_revision: u64,
    pub scientific_revision: u64,
}

/// Retry authority for exactly the observed failed experiment generation.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ExperimentRetryFailedRequest {
    /// Generation from the last observed experiment status. A repeated request
    /// cannot authorize another attempt after the status has advanced.
    pub expected_generation: u64,
}

/// Request payload for setting simulation speed multiplier.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SpeedRequest {
    pub speed: f32,
}

/// Request payload for generating a procedural map sandbox artifact.
#[derive(Debug, Clone, Default, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct MapGenerateRequestBody {
    /// Width in cells. If omitted, uses the live terrain width (100 if unavailable).
    pub width: Option<u32>,
    /// Height in cells. If omitted, uses the live terrain height (100 if unavailable).
    pub height: Option<u32>,
    /// Cell size in world units. If omitted, uses the live terrain cell size (50 if unavailable).
    pub cell_size: Option<u32>,
    /// Random seed for deterministic generation. Defaults to a constant seed if omitted.
    pub seed: Option<u64>,
    /// Optional declarative tileset. If both tileset and sample are omitted, uses the default biome tileset.
    #[schema(value_type = Option<Object>)]
    pub tileset: Option<Value>,
    /// Exemplar for overlapping sample generation. Mutually exclusive with tileset.
    #[schema(value_type = Option<Object>)]
    pub sample: Option<Value>,
}

impl MapGenerateRequestBody {
    /// Resolve the requested backend without silently substituting invalid inputs.
    pub fn generator_config(&self) -> Result<Option<DraftGeneratorConfig>, ControlError> {
        match (&self.tileset, &self.sample) {
            (Some(_), Some(_)) => Err(ControlError::InvalidPatch(
                "tileset and sample are mutually exclusive".into(),
            )),
            (Some(value), None) => serde_json::from_value(value.clone())
                .map(|tileset| Some(DraftGeneratorConfig::RuleBased { tileset }))
                .map_err(|error| {
                    ControlError::InvalidPatch(format!("invalid tileset specification: {error}"))
                }),
            (None, Some(value)) => serde_json::from_value(value.clone())
                .map(|sample| Some(DraftGeneratorConfig::SampleBased { sample }))
                .map_err(|error| {
                    ControlError::InvalidPatch(format!("invalid sample specification: {error}"))
                }),
            (None, None) => Ok(None),
        }
    }
}

/// Existing-agent policy shared by CLI parsing and REST/MCP request schemas.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    utoipa::ToSchema,
    clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum MapApplyAgentPolicy {
    /// Keep existing agents at their current positions.
    #[default]
    Retain,
    /// Move deep-water agents to the nearest non-deep-water tile.
    RepositionToSafeTile,
    /// Record deaths for deep-water agents at an open recorded lifecycle boundary.
    CullIncompatible,
}

impl From<MapApplyAgentPolicy> for ExistingAgentPolicy {
    fn from(policy: MapApplyAgentPolicy) -> Self {
        match policy {
            MapApplyAgentPolicy::Retain => Self::Retain,
            MapApplyAgentPolicy::RepositionToSafeTile => Self::RepositionToSafeTile,
            MapApplyAgentPolicy::CullIncompatible => Self::CullIncompatible,
        }
    }
}

/// Request payload for applying a map artifact to the simulation.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct MapApplyRequestBody {
    /// Map artifact payload: either a JSON representation of MapArtifact or a base64/hex postcard string, or file path.
    #[schema(value_type = Object)]
    pub artifact: Value,
    /// Existing-agent policy; defaults to retaining positions.
    #[serde(default)]
    pub existing_agent_policy: MapApplyAgentPolicy,
    /// Reject at ordered application if the owner's scientific revision differs.
    /// Obtain this value from simulation status before preparing a preview; omission is unguarded.
    pub expected_scientific_revision: Option<u64>,
    /// Optional idempotency key to prevent double application on retry.
    pub idempotency_key: Option<String>,
}

/// Largest step batch a single control request may submit.
pub const MAX_STEP_BATCH: u64 = 100_000;
/// Largest procedural map (in cells) a single generate request may build.
pub const MAX_GENERATED_MAP_CELLS: u64 = 4096 * 4096;
/// Deepest dotted knob path accepted from a request.
const MAX_KNOB_PATH_SEGMENTS: usize = 32;
/// Longest dotted knob path accepted from a request.
const MAX_KNOB_PATH_BYTES: usize = 512;

/// Upper bound on a map artifact read from a server-side file path.
///
/// The path arrives from a REST or MCP request, so an unbounded `fs::read` would let a caller
/// point it at an endless or enormous file (`/dev/zero`, a multi-GB log) and exhaust memory.
/// A real map artifact is far below this.
const MAX_MAP_ARTIFACT_FILE_BYTES: u64 = 64 * 1024 * 1024;

fn hex_to_bytes(s: &str) -> Result<Vec<u8>, ()> {
    let s = s.trim().as_bytes();
    if !s.len().is_multiple_of(2) {
        return Err(());
    }
    // Decode on bytes, not `&str` slices: a non-ASCII request string such as "a\u{e9}b" has
    // an even byte length but a char boundary inside a pair, and slicing it would panic
    // (and, with `panic = "abort"`, take the whole process down).
    s.as_chunks::<2>()
        .0
        .iter()
        .map(|&[hi, lo]| {
            let hi = char::from(hi).to_digit(16).ok_or(())?;
            let lo = char::from(lo).to_digit(16).ok_or(())?;
            u8::try_from(hi * 16 + lo).map_err(|_| ())
        })
        .collect()
}

/// Read a map artifact file, refusing anything that is not a regular file or exceeds
/// [`MAX_MAP_ARTIFACT_FILE_BYTES`].
fn read_map_artifact_file(path: &Path) -> Result<Vec<u8>, ControlError> {
    use std::io::Read as _;
    // Stat before opening: opening a FIFO or device would block or stream forever.
    let meta = fs::metadata(path)
        .map_err(|e| ControlError::InvalidPatch(format!("failed to read map file: {e}")))?;
    if !meta.is_file() {
        return Err(ControlError::InvalidPatch(
            "map artifact path is not a regular file".into(),
        ));
    }
    let file = fs::File::open(path)
        .map_err(|e| ControlError::InvalidPatch(format!("failed to read map file: {e}")))?;
    let mut bytes = Vec::new();
    file.take(MAX_MAP_ARTIFACT_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| ControlError::InvalidPatch(format!("failed to read map file: {e}")))?;
    if bytes.len() as u64 > MAX_MAP_ARTIFACT_FILE_BYTES {
        return Err(ControlError::InvalidPatch(format!(
            "map artifact file exceeds {MAX_MAP_ARTIFACT_FILE_BYTES} bytes"
        )));
    }
    Ok(bytes)
}

/// Parse a `MapArtifact` from either a JSON object or string (file path, raw JSON, or hex-encoded postcard).
pub fn parse_map_artifact(value: &Value) -> Result<MapArtifact, ControlError> {
    match value {
        Value::Object(_) => serde_json::from_value(value.clone())
            .map_err(|e| ControlError::InvalidPatch(format!("invalid map artifact JSON: {e}"))),
        Value::String(s) => {
            let path = Path::new(s);
            if path.exists() {
                let bytes = read_map_artifact_file(path)?;
                if let Ok(artifact) = postcard::from_bytes::<MapArtifact>(&bytes) {
                    return Ok(artifact);
                }
                if let Ok(artifact) = serde_json::from_slice::<MapArtifact>(&bytes) {
                    return Ok(artifact);
                }
                return Err(ControlError::InvalidPatch(
                    "file is neither a valid postcard nor JSON MapArtifact".into(),
                ));
            }
            if let Ok(artifact) = serde_json::from_str::<MapArtifact>(s) {
                return Ok(artifact);
            }
            if let Ok(bytes) = hex_to_bytes(s)
                && let Ok(artifact) = postcard::from_bytes::<MapArtifact>(&bytes)
            {
                return Ok(artifact);
            }
            Err(ControlError::InvalidPatch(
                "invalid map artifact string: not a valid file path, JSON string, or hex postcard"
                    .into(),
            ))
        }
        _ => Err(ControlError::InvalidPatch(
            "map artifact must be a JSON object or string".into(),
        )),
    }
}

/// Request payload for applying an intervention to the simulation.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct InterveneRequestBody {
    /// Canonical intervention specification or command.
    #[schema(value_type = Object)]
    pub intervention: Value,
    /// Originating surface: gpu, tui, rest, mcp, cli, script (defaults to rest).
    #[serde(default)]
    pub surface: Option<String>,
    /// Actor identifier (defaults to "rest_client").
    #[serde(default)]
    pub actor: Option<String>,
    /// Optional idempotency key to prevent double application on retry.
    pub idempotency_key: Option<String>,
}

/// Parse an `InterventionCommand` from an `InterveneRequestBody`.
pub fn parse_intervention_command(
    body: &InterveneRequestBody,
) -> Result<scriptbots_core::interventions::InterventionCommand, ControlError> {
    let intervention: scriptbots_core::Intervention = if let Ok(cmd) =
        serde_json::from_value::<scriptbots_core::interventions::InterventionCommand>(
            body.intervention.clone(),
        ) {
        return Ok(cmd);
    } else if let Ok(interv) =
        serde_json::from_value::<scriptbots_core::Intervention>(body.intervention.clone())
    {
        interv
    } else if let Some(s) = body.intervention.as_str() {
        if let Ok(bytes) = hex_to_bytes(s) {
            scriptbots_core::interventions::intervention_from_param_bytes(&bytes).map_err(|e| {
                ControlError::InvalidPatch(format!("invalid postcard intervention: {e}"))
            })?
        } else {
            serde_json::from_str(s).map_err(|e| {
                ControlError::InvalidPatch(format!("invalid JSON intervention: {e}"))
            })?
        }
    } else {
        serde_json::from_value(body.intervention.clone())
            .map_err(|e| ControlError::InvalidPatch(format!("invalid intervention object: {e}")))?
    };

    let surface = body
        .surface
        .as_deref()
        .map(|s| match s.to_ascii_lowercase().as_str() {
            "gpu" => scriptbots_core::interventions::InterventionSurface::Gpu,
            "tui" => scriptbots_core::interventions::InterventionSurface::Tui,
            "rest" => scriptbots_core::interventions::InterventionSurface::Rest,
            "mcp" => scriptbots_core::interventions::InterventionSurface::Mcp,
            "cli" => scriptbots_core::interventions::InterventionSurface::Cli,
            "script" => scriptbots_core::interventions::InterventionSurface::Script,
            _ => scriptbots_core::interventions::InterventionSurface::Rest,
        })
        .unwrap_or(scriptbots_core::interventions::InterventionSurface::Rest);

    let actor = body
        .actor
        .clone()
        .unwrap_or_else(|| "rest_client".to_owned());

    Ok(scriptbots_core::interventions::InterventionCommand {
        intervention,
        surface,
        actor,
    })
}

/// Version and runtime discovery descriptor.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ApiVersionDto {
    pub service_name: String,
    pub version: String,
    pub git_commit: Option<String>,
    pub rustc_version: String,
    pub protocol_version: u32,
    pub features: Vec<String>,
    pub capabilities: Vec<String>,
}

/// OpenAPI and schema discovery descriptor.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ApiSchemaDto {
    pub openapi_version: String,
    pub title: String,
    pub version: String,
    pub routes: Vec<String>,
    pub mcp_tools: Vec<String>,
    pub schemas: Vec<String>,
}

/// Scenario variant arm within an experiment plan.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ExperimentVariantDto {
    pub variant_id: String,
    pub brain_family: String,
    #[schema(value_type = Option<Object>)]
    pub config_overrides: Option<Value>,
    /// Ordered live numeric configuration patches at completed-tick boundaries.
    /// Each tick must precede the run horizon; statements at the same tick keep order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interventions: Vec<crate::ScenarioInterventionV1>,
}

/// Request payload for creating a matched-seed experiment batch.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ExperimentCreateRequest {
    pub experiment_id: Option<String>,
    pub description: Option<String>,
    pub variants: Vec<ExperimentVariantDto>,
    pub seeds: Vec<u64>,
    /// Exact positive horizon within the runner's per-run limit; never clamped.
    /// Omitting this field selects 1,000 scientific ticks.
    pub ticks_per_run: Option<u64>,
    pub max_concurrency: Option<usize>,
    pub idempotency_key: Option<String>,
}

/// Record of an individual run within an experiment.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ExperimentRunRecordDto {
    pub run_id: String,
    pub variant_id: String,
    pub brain_family: String,
    pub seed: u64,
    pub state: String,
    /// Terminal recorded transitions; remains zero for pending and running runs.
    pub total_ticks: u64,
    /// Requested horizon from the validated experiment plan.
    pub target_ticks: u64,
    /// Last observed sealed cut, independently of terminal completion/status generation.
    #[schema(inline)]
    pub checkpoint_progress: crate::experiment_runner::ExperimentCheckpointProgress,
    pub final_digest: Option<String>,
    pub bundle_path: Option<String>,
    pub error_reason: Option<String>,
}

/// Complete, bounded file inventory for one source-verified completed run.
///
/// The status generation identifies the observation, not a mutable page cursor.
/// Companion runs may still be running or have failed; only this run's exact
/// plan, final owner and bundle are verified before exposing downloadable files.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ExperimentRunArtifactsDto {
    pub experiment_id: String,
    pub plan_digest: String,
    pub status_generation: u64,
    pub run: ExperimentRunRecordDto,
    /// Manifest and all indexed files, with at most 33 entries and no pagination.
    pub artifacts: Vec<ArtifactMetadataDto>,
    /// Exact sum of every listed file, including the bundle manifest.
    pub total_bytes: u64,
}

/// Maximum encoded inventory size shared by REST, MCP and remote clients.
pub const MAX_EXPERIMENT_RUN_ARTIFACTS_RESPONSE_BYTES: usize = 256 * 1024;
/// Complete inventory cardinality, including the bundle manifest.
pub const MAX_EXPERIMENT_RUN_ARTIFACTS: usize = crate::experiment_runner::MAX_BUNDLE_ARTIFACTS + 1;
/// Maximum bytes of any one downloadable experiment file.
pub const MAX_EXPERIMENT_RUN_ARTIFACT_BYTES: u64 =
    crate::experiment_runner::MAX_BUNDLE_ARTIFACT_BYTES;
/// Maximum sum of all inventory files, including the separately bounded manifest.
pub const MAX_EXPERIMENT_RUN_TOTAL_ARTIFACT_BYTES: u64 =
    crate::experiment_runner::MAX_BUNDLE_TOTAL_ARTIFACT_BYTES
        + crate::experiment_runner::MAX_STATUS_BYTES;

/// Complete batch status report for an experiment.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ExperimentBatchStatusDto {
    pub schema_version: u16,
    pub generation: u64,
    pub plan_digest: String,
    pub experiment_id: String,
    pub status: String,
    pub total_runs: usize,
    pub completed_runs: usize,
    pub failed_runs: usize,
    pub runs: Vec<ExperimentRunRecordDto>,
    /// Batch-level failure (plan, status-file, or bundle verification), if any.
    pub error_reason: Option<String>,
    pub created_at_utc: String,
    pub updated_at_utc: String,
}

/// One source-verified, matched-seed endpoint comparison within an experiment.
/// Differences are challenger minus baseline; a larger value is not a fitness verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ExperimentComparisonDto {
    pub experiment_id: String,
    pub baseline_variant: String,
    pub challenger_variant: String,
    pub metric: String,
    pub endpoint_tick: u64,
    pub matched_seeds: Vec<u64>,
    pub mean_difference: f64,
    pub ci_95: Option<(f64, f64)>,
    /// Paired Cohen's d_z, unavailable for a single pair or zero-spread differences.
    pub standardized_effect: Option<f64>,
    /// Inference is deliberately absent when only one independent seed pair exists.
    pub p_value: Option<f64>,
    pub adjusted_p_value: Option<f64>,
    pub significant: Option<bool>,
    pub descriptive_only: bool,
    pub underpowered_reason: Option<String>,
    pub undefined_effect_reason: Option<String>,
    pub plan_digest: String,
    pub status_generation: u64,
}

fn validate_experiment_comparison(
    experiment_id: &str,
    baseline_variant: &str,
    challenger_variant: &str,
    metric: &str,
) -> Result<(), ControlError> {
    use crate::experiment_runner::{
        EXPERIMENT_ENDPOINT_METRICS, MAX_RUN_COMPONENT_BYTES, validate_path_component,
    };

    for (label, value) in [
        ("experiment_id", experiment_id),
        ("baseline_variant", baseline_variant),
        ("challenger_variant", challenger_variant),
    ] {
        if value.len() > MAX_RUN_COMPONENT_BYTES {
            return Err(ControlError::BadRequest(format!(
                "{label} exceeds the {MAX_RUN_COMPONENT_BYTES}-byte experiment identity bound"
            )));
        }
        validate_path_component(label, value)
            .map_err(|error| ControlError::BadRequest(error.to_string()))?;
    }
    if baseline_variant == challenger_variant {
        return Err(ControlError::BadRequest(
            "comparison requires two distinct variants".into(),
        ));
    }
    if !EXPERIMENT_ENDPOINT_METRICS.contains(&metric) {
        return Err(ControlError::BadRequest(format!(
            "unsupported experiment endpoint metric; expected one of {}",
            EXPERIMENT_ENDPOINT_METRICS.join(", ")
        )));
    }
    Ok(())
}

/// Compare two arms of a standalone saved experiment using the same verified
/// bundles, restored endpoints and paired statistics as the managed service.
/// This path never starts workers, resumes runs, or writes experiment state.
pub fn compare_saved_experiment_variants(
    runner: &crate::experiment_runner::MatchedSeedExperimentRunner,
    state_file: &Path,
    baseline_variant: &str,
    challenger_variant: &str,
    metric: &str,
) -> Result<ExperimentComparisonDto, ControlError> {
    validate_experiment_comparison(
        &runner.experiment_id,
        baseline_variant,
        challenger_variant,
        metric,
    )?;
    compare_verified_variant_endpoints(
        runner,
        state_file,
        baseline_variant,
        challenger_variant,
        metric,
    )
}

fn compare_verified_variant_endpoints(
    runner: &crate::experiment_runner::MatchedSeedExperimentRunner,
    state_file: &Path,
    baseline_variant: &str,
    challenger_variant: &str,
    metric: &str,
) -> Result<ExperimentComparisonDto, ControlError> {
    use crate::experiment_runner::RunState;
    use crate::lab::stats::{
        AnalysisParams, RunSummary, RunSummaryParts, analyze_matched_seed_runs,
    };

    let experiment_id = &runner.experiment_id;
    let variants = [baseline_variant, challenger_variant]
        .map(|id| {
            runner
                .variants
                .iter()
                .find(|variant| variant.variant_id == id)
                .ok_or_else(|| {
                    ControlError::NotFound(format!(
                        "variant '{id}' is not in experiment '{experiment_id}'"
                    ))
                })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let status = runner
        .status_snapshot(state_file)
        .map_err(|error| ControlError::Serialization(error.to_string()))?;
    if let Some(run) = status.runs.iter().find(|run| {
        (run.variant_id == baseline_variant || run.variant_id == challenger_variant)
            && run.state != RunState::Completed
    }) {
        return Err(ControlError::Conflict(format!(
            "comparison run '{}' is {:?}; every declared seed must be completed",
            run.run_id, run.state
        )));
    }
    let endpoints = runner
        .verified_comparison_endpoints(state_file, &status, baseline_variant, challenger_variant)
        .map_err(|error| ControlError::Serialization(error.to_string()))?;
    let matched_seeds = endpoints[0]
        .iter()
        .map(|state| state.summary.seed)
        .collect::<Vec<_>>();
    let mut summaries = Vec::with_capacity(matched_seeds.len().saturating_mul(2));
    for (arm, states) in endpoints.into_iter().enumerate() {
        let variant = variants[arm];
        for state in states {
            let metrics = state.summary.available_metrics();
            if !metrics.get(metric).is_some_and(|value| value.is_finite()) {
                return Err(ControlError::Conflict(format!(
                    "endpoint '{metric}' is unavailable for run '{}' at tick {}; missing values are not replaced with zero",
                    state.run_id, state.tick
                )));
            }
            summaries.push(RunSummary::from_verified_parts(RunSummaryParts {
                summary_path: Some(
                    runner
                        .output_dir
                        .join(&state.run_id)
                        .join("exports/summary.csv")
                        .to_string_lossy()
                        .into_owned(),
                ),
                run_id: state.run_id,
                arm_id: u16::try_from(arm).expect("two comparison arms fit u16"),
                seed: state.summary.seed,
                config_digest: state.config_digest,
                digest: state.scientific_digest,
                ticks: state.tick,
                metrics,
                summary_artifact_digest: state.summary_artifact_digest,
                variant_id: variant.variant_id.clone(),
                config_overrides: variant.config_overrides.clone(),
                interventions: variant.interventions.clone(),
                brain_family: state.summary.brain_family,
            }));
        }
    }
    let mut analysis =
        analyze_matched_seed_runs(&summaries, &[metric.to_owned()], AnalysisParams::default())
            .map_err(|error| ControlError::Conflict(format!("comparison unavailable: {error}")))?;
    let effect = analysis.effects.pop().ok_or_else(|| {
        ControlError::Serialization("comparison produced no endpoint effect".into())
    })?;
    if !effect.mean_difference.is_finite()
        || effect
            .standardized_effect
            .is_some_and(|value| !value.is_finite())
        || effect
            .ci_95
            .is_some_and(|(low, high)| !low.is_finite() || !high.is_finite())
        || !effect.raw_p_value.is_finite()
        || !effect.adjusted.adjusted_p_value.is_finite()
    {
        return Err(ControlError::Serialization(
            "comparison produced a non-finite endpoint estimate".into(),
        ));
    }
    let descriptive_only = effect.n_pairs < 2;
    Ok(ExperimentComparisonDto {
        experiment_id: experiment_id.clone(),
        baseline_variant: baseline_variant.to_owned(),
        challenger_variant: challenger_variant.to_owned(),
        metric: metric.to_owned(),
        endpoint_tick: runner.max_ticks,
        matched_seeds,
        mean_difference: effect.mean_difference,
        ci_95: effect.ci_95,
        standardized_effect: effect.standardized_effect,
        p_value: (!descriptive_only).then_some(effect.raw_p_value),
        adjusted_p_value: (!descriptive_only).then_some(effect.adjusted.adjusted_p_value),
        significant: (!descriptive_only).then_some(effect.adjusted.rejected),
        descriptive_only,
        underpowered_reason: effect
            .underpowered_reason
            .map(|reason| reason.description()),
        undefined_effect_reason: effect.undefined_reason.map(|reason| reason.description()),
        plan_digest: status.plan_digest,
        status_generation: status.generation,
    })
}

/// Summary of an experiment for paginated listings.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ExperimentSummaryDto {
    pub experiment_id: String,
    pub status: String,
    pub total_runs: usize,
    pub completed_runs: usize,
    pub failed_runs: usize,
    pub created_at_utc: String,
    pub updated_at_utc: String,
}

/// Request payload for creating a simulation checkpoint.
#[derive(Debug, Clone, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CheckpointCreateRequest {
    pub description: Option<String>,
    pub idempotency_key: Option<String>,
}

/// Metadata descriptor for a simulation checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CheckpointMetadataDto {
    pub checkpoint_id: String,
    pub tick: u64,
    /// Live stable agents in the saved checkpoint, not the later live world.
    pub agent_count: usize,
    /// Captured WorldDigestV1 overall digest, distinct from either artifact checksum.
    pub world_digest_v1: String,
    pub byte_size: u64,
    pub checksum_blake3: String,
    pub checksum_sha256: String,
    pub schema: String,
    pub created_at_utc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Maximum future transitions admitted for one checkpoint branch.
pub const MAX_CHECKPOINT_BRANCH_TICKS: u64 = crate::branch_study::MAX_STUDY_HORIZON_TICKS;
/// Maximum encoded managed-branch request size, shared by all transports.
pub const MAX_CHECKPOINT_BRANCH_REQUEST_BYTES: usize = 16 * 1024;
/// Maximum safe human-readable branch name.
pub const MAX_CHECKPOINT_BRANCH_NAME_BYTES: usize = 64;
/// Maximum retry key length.
pub const MAX_CHECKPOINT_BRANCH_KEY_BYTES: usize = 1024;
/// Maximum checkpoint or branch identifier length accepted by the API.
pub const MAX_CHECKPOINT_BRANCH_ID_BYTES: usize = 128;
/// Completed and interrupted branches retain their identities instead of being evicted.
pub const MAX_RETAINED_CHECKPOINT_BRANCHES: usize = 32;

/// Run an independent scientific future from a retained checkpoint.
///
/// An intervention adds a treatment beside an unchanged control. This does not
/// replace the live owner or continue its session, schedule or retained history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckpointBranchRequest {
    pub checkpoint_id: String,
    pub branch_name: String,
    pub ticks: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Object)]
    pub intervention: Option<crate::branch_study::StudyIntervention>,
    /// Completed transitions after the saved cut before applying the intervention.
    /// Zero applies at the saved boundary. Must be less than `ticks`; a nonzero
    /// delay requires an intervention and leaves an observed transition afterward.
    #[serde(default, skip_serializing_if = "checkpoint_branch_delay_is_zero")]
    pub intervention_delay_ticks: u64,
    /// Required retry identity. A changed request cannot reuse an admitted key.
    pub idempotency_key: String,
}

const fn checkpoint_branch_delay_is_zero(delay: &u64) -> bool {
    *delay == 0
}

impl CheckpointBranchRequest {
    /// Validate transport-independent bounds before reading or reserving anything.
    pub fn validate_shape(&self) -> Result<(), ControlError> {
        if !valid_checkpoint_branch_identifier(&self.checkpoint_id) {
            return Err(ControlError::BadRequest(
                "checkpoint_id must be a bounded ASCII identifier".to_owned(),
            ));
        }
        if self.branch_name.is_empty()
            || self.branch_name.len() > MAX_CHECKPOINT_BRANCH_NAME_BYTES
            || !self
                .branch_name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(ControlError::BadRequest(format!(
                "branch_name must contain 1..={MAX_CHECKPOINT_BRANCH_NAME_BYTES} ASCII letters, digits, '_' or '-'"
            )));
        }
        if self.ticks == 0 || self.ticks > MAX_CHECKPOINT_BRANCH_TICKS {
            return Err(ControlError::BadRequest(format!(
                "ticks must be in 1..={MAX_CHECKPOINT_BRANCH_TICKS}"
            )));
        }
        if self.intervention_delay_ticks >= self.ticks {
            return Err(ControlError::BadRequest(
                "intervention_delay_ticks must be less than ticks so the intervention has an observed transition".to_owned(),
            ));
        }
        if self.intervention_delay_ticks != 0 && self.intervention.is_none() {
            return Err(ControlError::BadRequest(
                "a nonzero intervention_delay_ticks requires an intervention".to_owned(),
            ));
        }
        if self.idempotency_key.trim().is_empty()
            || self.idempotency_key.len() > MAX_CHECKPOINT_BRANCH_KEY_BYTES
        {
            return Err(ControlError::BadRequest(format!(
                "idempotency_key must contain 1..={MAX_CHECKPOINT_BRANCH_KEY_BYTES} bytes"
            )));
        }
        Ok(())
    }

    fn bounded_payload(&self) -> Result<Vec<u8>, ControlError> {
        struct Payload(Vec<u8>);
        impl std::io::Write for Payload {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if bytes.len() > MAX_CHECKPOINT_BRANCH_REQUEST_BYTES.saturating_sub(self.0.len()) {
                    return Err(std::io::Error::other(
                        "checkpoint branch request exceeds its byte bound",
                    ));
                }
                self.0.extend_from_slice(bytes);
                Ok(bytes.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut payload = Payload(Vec::new());
        serde_json::to_writer(&mut payload, self).map_err(|error| {
            if error.is_io() {
                ControlError::PayloadTooLarge(format!(
                    "checkpoint branch request exceeds {MAX_CHECKPOINT_BRANCH_REQUEST_BYTES} bytes"
                ))
            } else {
                ControlError::serialization(error)
            }
        })?;
        Ok(payload.0)
    }
}

fn valid_checkpoint_branch_identifier(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_CHECKPOINT_BRANCH_ID_BYTES
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

/// Latest observed managed-branch state. Source and target ticks are not live progress.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CheckpointBranchStatusDto {
    pub branch_id: String,
    pub checkpoint_id: String,
    pub branch_name: String,
    /// running, cancelling, completed, cancelled, failed or stopped after interruption.
    pub status: String,
    pub source_tick: u64,
    pub target_tick: u64,
    /// Digest of the exact saved source, not a forecast or final branch digest.
    pub world_digest_v1: String,
    /// Downloadable study report, published only after every branch bundle verifies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_reason: Option<String>,
}

/// Bounded discovery of admitted checkpoint studies, including recovered receipts.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct PaginatedCheckpointBranchesResponse {
    pub items: Vec<CheckpointBranchStatusDto>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    pub has_more: bool,
}

/// Metadata descriptor for a discoverable and downloadable artifact or bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ArtifactMetadataDto {
    pub artifact_id: String,
    pub filename: String,
    pub content_type: String,
    pub byte_size: u64,
    pub checksum_blake3: String,
    pub checksum_sha256: String,
    pub created_at_utc: String,
    pub relative_path: String,
}

/// Paginated response for experiments.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct PaginatedExperimentsResponse {
    pub items: Vec<ExperimentSummaryDto>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    pub has_more: bool,
    pub execution_capacity: ExperimentExecutionCapacityDto,
}

/// Service-wide reservations, including owners preparing or publishing a wave.
/// These counts describe reserved capacity, not simultaneous science activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ExperimentExecutionCapacityDto {
    pub max_run_slots: usize,
    pub reserved_run_slots: usize,
    /// Dense grid cardinality across every concurrently admitted experiment.
    /// This is not a bound on population or total resident memory.
    pub max_grid_cells: u64,
    pub reserved_grid_cells: u64,
    pub reserved_batches: usize,
}

/// A unique reservation spans preparation, execution and artifact publication.
/// Dropping an unstarted or unwinding worker returns its capacity as well.
struct ExperimentExecutionReservation {
    capacity: Arc<Mutex<ExperimentExecutionCapacityDto>>,
    run_slots: usize,
    grid_cells: u64,
    experiment_id: String,
    running: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl Drop for ExperimentExecutionReservation {
    fn drop(&mut self) {
        let observed = {
            // Reservation updates compute all fallible work before changing counts.
            // Poison must not prevent this unique owner's reservation being returned.
            let mut capacity = self.capacity.lock().unwrap_or_else(PoisonError::into_inner);
            capacity.reserved_run_slots -= self.run_slots;
            capacity.reserved_grid_cells -= self.grid_cells;
            capacity.reserved_batches -= 1;
            *capacity
        };
        if let Some(running) = &self.running {
            running.store(false, std::sync::atomic::Ordering::Release);
        }
        if !std::thread::panicking() {
            tracing::info!(
                experiment_id = %self.experiment_id,
                released_run_slots = self.run_slots,
                released_grid_cells = self.grid_cells,
                reserved_run_slots = observed.reserved_run_slots,
                reserved_grid_cells = observed.reserved_grid_cells,
                reserved_batches = observed.reserved_batches,
                max_run_slots = observed.max_run_slots,
                "experiment execution reservation released"
            );
        }
    }
}

/// Paginated response for checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct PaginatedCheckpointsResponse {
    pub items: Vec<CheckpointMetadataDto>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    pub has_more: bool,
    pub retention: CheckpointRetentionDto,
}

/// Actual file occupancy is separate from SQL's hex/metadata representation.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CheckpointRetentionDto {
    pub max_count: u64,
    pub max_bytes: u64,
    pub max_directory_leases: u64,
    pub directory_leases: usize,
    pub file_count: u64,
    pub file_bytes: u64,
    pub observed_file_count_high_water: u64,
    pub observed_file_bytes_high_water: u64,
    /// Immutable retry admissions, including incomplete requests from earlier owners.
    /// These do not represent requests adopted by the current owner.
    pub retained_request_records: u64,
    pub retained_request_record_bytes: u64,
    /// Union of binary artifact paths and checkpoint paths reserved by requests.
    pub reserved_checkpoint_count: u64,
    /// Derived from the configured count limit and the bounded record encoding.
    pub max_retained_request_record_bytes: u64,
    pub request_entries: usize,
    pub pending_entries: usize,
    pub payload_cache_bytes: u64,
    pub sql_count: Option<u64>,
    pub sql_bytes: Option<u64>,
    pub inventory_error: Option<String>,
    pub sql_usage_error: Option<String>,
}

/// Paginated response for artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct PaginatedArtifactsResponse {
    pub items: Vec<ArtifactMetadataDto>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    pub has_more: bool,
}

/// Maximum artifact download size enforced at boundary (64 MB).
pub const MAX_ARTIFACT_DOWNLOAD_BYTES: usize =
    scriptbots_runtime::lab_workflows::LAB_MAX_ARTIFACT_BYTES;

/// Standard FIPS 180-4 SHA-256 digest in pure Rust.
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::cast_possible_truncation
)]
pub fn compute_sha256(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let k: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, item) in w.iter_mut().take(16).enumerate() {
            *item = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut h_val = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h_val
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h_val = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(h_val);
    }

    format!(
        "{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
    )
}

/// Compute hex BLAKE3 hash of arbitrary bytes.
pub fn compute_blake3(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

fn read_artifact_file_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>, ControlError> {
    open_artifact_file_bounded(path, max_bytes, false).map(|(_, bytes)| bytes)
}

fn same_artifact_identity(expected: &fs::Metadata, actual: &fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        expected.dev() == actual.dev() && expected.ino() == actual.ino()
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        expected.volume_serial_number().is_some()
            && expected.volume_serial_number() == actual.volume_serial_number()
            && expected.file_index().is_some()
            && expected.file_index() == actual.file_index()
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (expected, actual);
        false
    }
}

fn artifact_is_singly_linked(metadata: &fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        metadata.nlink() == 1
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.number_of_links() == Some(1)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = metadata;
        false
    }
}

fn verify_artifact_identity(
    expected: &fs::Metadata,
    actual: &fs::Metadata,
) -> Result<(), ControlError> {
    if expected.file_type().is_symlink()
        || actual.file_type().is_symlink()
        || !expected.is_file()
        || !actual.is_file()
        || !artifact_is_singly_linked(expected)
        || !artifact_is_singly_linked(actual)
        || !same_artifact_identity(expected, actual)
    {
        return Err(ControlError::BadRequest(
            "artifact must remain the same singly linked regular file".to_owned(),
        ));
    }
    Ok(())
}

fn verify_open_artifact(path: &Path, file: &fs::File) -> Result<(), ControlError> {
    let current =
        fs::symlink_metadata(path).map_err(|error| ControlError::NotFound(error.to_string()))?;
    let opened = file
        .metadata()
        .map_err(|error| ControlError::Serialization(error.to_string()))?;
    verify_artifact_identity(&opened, &current)
}

/// Keep the verified descriptor available to callers that must sync the exact
/// bytes they read. A second pathname open could encounter a replacement FIFO.
fn open_artifact_file_bounded(
    path: &Path,
    max_bytes: usize,
    for_sync: bool,
) -> Result<(fs::File, Vec<u8>), ControlError> {
    let before =
        fs::symlink_metadata(path).map_err(|error| ControlError::NotFound(error.to_string()))?;
    verify_artifact_identity(&before, &before)?;
    let mut options = fs::OpenOptions::new();
    // Windows FlushFileBuffers needs write access. Downloads and acknowledged
    // retries remain read-only; Unix permits syncing a read-only descriptor.
    options.read(true).write(cfg!(windows) && for_sync);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // The preflight is not sufficient: a special file can replace the path
        // between metadata and open. Neither symlinks nor FIFOs may be followed.
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options
        .open(path)
        .map_err(|error| ControlError::NotFound(error.to_string()))?;
    let metadata = file
        .metadata()
        .map_err(|error| ControlError::Serialization(error.to_string()))?;
    verify_artifact_identity(&before, &metadata)?;
    let limit = u64::try_from(max_bytes).expect("artifact limit fits u64");
    if metadata.len() > limit {
        return Err(ControlError::PayloadTooLarge(
            "artifact exceeds the download byte bound".to_owned(),
        ));
    }
    let mut bytes = Vec::new();
    (&file)
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| ControlError::Serialization(error.to_string()))?;
    if bytes.len() > max_bytes {
        return Err(ControlError::PayloadTooLarge(
            "artifact grew beyond the download byte bound".to_owned(),
        ));
    }
    verify_open_artifact(path, &file)?;
    Ok((file, bytes))
}

#[cfg(unix)]
fn sync_checkpoint_directory(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;

    let before = fs::symlink_metadata(path)?;
    let directory = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let opened = directory.metadata()?;
    if !before.is_dir() || !opened.is_dir() || !same_artifact_identity(&before, &opened) {
        return Err(std::io::Error::other(
            "checkpoint parent directory changed before sync",
        ));
    }
    directory.sync_all()?;
    let after = fs::symlink_metadata(path)?;
    if !after.is_dir() || !same_artifact_identity(&opened, &after) {
        return Err(std::io::Error::other(
            "checkpoint parent directory changed during sync",
        ));
    }
    Ok(())
}

const NATIVE_ARTIFACT_EXPORT_SCHEMA: &str = "scriptbots.native-artifact-export.v1";
const MAX_NATIVE_ARTIFACT_EXPORT_MANIFEST_BYTES: usize = 32 * 1024;

/// Copy provenance, separate from checkpoint/journal durability receipts.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeArtifactExportManifestV1 {
    schema: String,
    request_id: String,
    // A decimal string preserves the complete identity in ordinary JSON tools.
    source_session_id: String,
    artifact: scriptbots_runtime::lab_workflows::LabArtifact,
    export_directory: String,
    output_path: String,
    manifest_path: String,
}

fn native_artifact_export_path_is_canonical(path: &Path, canonical: &Path) -> bool {
    if path == canonical {
        return true;
    }
    #[cfg(windows)]
    {
        use std::path::{Component, Prefix};
        let mut requested = path.components();
        let mut resolved = canonical.components();
        let (Some(Component::Prefix(left)), Some(Component::Prefix(right))) =
            (requested.next(), resolved.next())
        else {
            return false;
        };
        // Windows canonicalize adds this prefix without changing the location.
        let same_prefix = match (left.kind(), right.kind()) {
            (Prefix::Disk(left), Prefix::VerbatimDisk(right)) => left.eq_ignore_ascii_case(&right),
            (
                Prefix::UNC(left_server, left_share),
                Prefix::VerbatimUNC(right_server, right_share),
            ) => left_server == right_server && left_share == right_share,
            _ => false,
        };
        same_prefix && requested.eq(resolved)
    }
    #[cfg(not(windows))]
    {
        false
    }
}

fn verify_native_artifact_export_directory(
    path: &Path,
    expected: Option<&fs::Metadata>,
) -> Result<fs::Metadata, ControlError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| ControlError::Serialization(error.to_string()))?;
    let canonical = path
        .canonicalize()
        .map_err(|error| ControlError::Serialization(error.to_string()))?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || !native_artifact_export_path_is_canonical(path, &canonical)
        || expected.is_some_and(|expected| !same_artifact_identity(expected, &metadata))
    {
        return Err(ControlError::BadRequest(format!(
            "native artifact export directory {} is aliased, replaced or not a directory",
            path.display()
        )));
    }
    Ok(metadata)
}

fn native_artifact_export_existing_ancestor(
    path: &Path,
) -> Result<(PathBuf, fs::Metadata), ControlError> {
    for ancestor in path.ancestors() {
        match fs::symlink_metadata(ancestor) {
            Ok(_) => {
                return verify_native_artifact_export_directory(ancestor, None)
                    .map(|metadata| (ancestor.to_path_buf(), metadata));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(ControlError::Serialization(error.to_string())),
        }
    }
    Err(ControlError::BadRequest(
        "native artifact export has no existing absolute parent directory".to_owned(),
    ))
}

/// Keep a partial write intact when the filesystem cannot complete the copy.
/// The writer argument makes that real I/O failure boundary independently testable.
fn write_native_artifact_export_file(
    path: &Path,
    bytes: &[u8],
    write: impl FnOnce(&mut fs::File, &[u8]) -> std::io::Result<()>,
) -> Result<fs::File, ControlError> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let mut file = options
        .open(path)
        .map_err(|error| ControlError::Serialization(error.to_string()))?;
    verify_open_artifact(path, &file)?;
    write(&mut file, bytes).map_err(|error| ControlError::Serialization(error.to_string()))?;
    file.sync_all()
        .map_err(|error| ControlError::Serialization(error.to_string()))?;
    verify_open_artifact(path, &file)?;
    Ok(file)
}

fn verify_native_artifact_export_file(
    path: &Path,
    written: &fs::File,
    max_bytes: usize,
    byte_size: u64,
    checksum_blake3: &str,
    checksum_sha256: &str,
) -> Result<(), ControlError> {
    let (read, bytes) = open_artifact_file_bounded(path, max_bytes, false)?;
    verify_artifact_identity(
        &written
            .metadata()
            .map_err(|error| ControlError::Serialization(error.to_string()))?,
        &read
            .metadata()
            .map_err(|error| ControlError::Serialization(error.to_string()))?,
    )?;
    if bytes.len() as u64 != byte_size
        || compute_blake3(&bytes) != checksum_blake3
        || compute_sha256(&bytes) != checksum_sha256
    {
        return Err(ControlError::InvalidPatch(format!(
            "native artifact export {} failed copied-byte verification",
            path.display()
        )));
    }
    verify_open_artifact(path, written)
}

fn preserve_native_artifact_export_failure(path: &Path, error: ControlError) -> ControlError {
    ControlError::Serialization(format!(
        "native artifact export was not confirmed: {error}; any partial or existing files are preserved at {}",
        path.display()
    ))
}

const MAX_EXPERIMENT_DEFINITION_BYTES: u64 = 1024 * 1024;
const MAX_CATALOG_EXPERIMENTS: usize = 1024;

/// Immutable admission definition. Paths are derived from the configured root,
/// never deserialized. The checksum detects corruption, not hostile authorship.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperimentDefinitionV1 {
    schema_version: u16,
    experiment_id: String,
    cohort: crate::experiment_runner::MatchedSeedCohort,
    variants: Vec<crate::experiment_runner::ScenarioVariant>,
    max_ticks: u64,
    max_concurrency: usize,
    plan_digest: String,
    created_at_utc: String,
    request_identity: Option<(String, String)>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperimentCatalogRecord {
    definition: ExperimentDefinitionV1,
    checksum: String,
}

fn experiment_catalog_error(path: &Path, error: impl std::fmt::Display) -> ControlError {
    ControlError::Serialization(format!(
        "experiment catalog {}: {error}; preserve the files and reconcile the saved definition before retrying",
        path.display()
    ))
}

fn validate_experiment_directory(path: &Path) -> Result<(), ControlError> {
    let metadata = fs::symlink_metadata(path).map_err(|e| experiment_catalog_error(path, e))?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || path
            .canonicalize()
            .map_err(|e| experiment_catalog_error(path, e))?
            != path
    {
        return Err(experiment_catalog_error(
            path,
            "expected an unaliased directory",
        ));
    }
    Ok(())
}

impl ExperimentDefinitionV1 {
    fn persist(self, directory: &Path) -> Result<(), ControlError> {
        let payload = serde_json::to_vec(&self).map_err(ControlError::serialization)?;
        let record = ExperimentCatalogRecord {
            checksum: compute_blake3(&payload),
            definition: self,
        };
        let bytes = serde_json::to_vec(&record).map_err(ControlError::serialization)?;
        if bytes.len() as u64 > MAX_EXPERIMENT_DEFINITION_BYTES {
            return Err(ControlError::PayloadTooLarge(
                "experiment definition exceeds 1 MiB".into(),
            ));
        }
        // A new directory is the cross-process admission reservation. Never
        // overwrite a prior definition, status, cut, bundle, or partial admission.
        let parent = directory.parent().ok_or(ControlError::Lock)?;
        fs::create_dir_all(parent).map_err(|e| experiment_catalog_error(parent, e))?;
        validate_experiment_directory(parent)?;
        fs::create_dir(directory).map_err(|e| experiment_catalog_error(directory, e))?;
        let temporary = directory.join("definition.json.tmp");
        let committed = directory.join("definition.json");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| experiment_catalog_error(&temporary, e))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|e| experiment_catalog_error(&temporary, e))?;
        drop(file);
        fs::rename(&temporary, &committed).map_err(|e| experiment_catalog_error(&committed, e))?;
        #[cfg(unix)]
        for path in [Some(directory), Some(parent), parent.parent()]
            .into_iter()
            .flatten()
        {
            fs::File::open(path)
                .and_then(|f| f.sync_all())
                .map_err(|e| experiment_catalog_error(path, e))?;
        }
        Ok(())
    }
}

const MAX_CHECKPOINT_BRANCH_CATALOG_BYTES: usize = 1024 * 1024;
const MAX_CHECKPOINT_BRANCH_ARTIFACTS: usize =
    2 * (crate::branch_study::STUDY_BUNDLE_VERIFICATION_LIMITS.max_artifacts + 1) + 5;
const MAX_CHECKPOINT_BRANCH_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;

/// Immutable admission authority. Checksums detect corruption, not hostile authorship.
/// Paths are derived from the leased root and branch identity, never accepted from callers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointBranchDefinitionV1 {
    schema_version: u16,
    branch_id: String,
    request: CheckpointBranchRequest,
    request_hash: String,
    source_tick: u64,
    target_tick: u64,
    world_digest_v1: String,
    source_byte_size: u64,
    source_checksum_blake3: String,
    plan: crate::branch_study::BranchStudyPlan,
    created_at_utc: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointBranchDefinitionRecord {
    definition: CheckpointBranchDefinitionV1,
    checksum: String,
}

/// A terminal receipt records observed completion, not perpetual file integrity.
/// Artifact access independently revalidates the sealed files before publication.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointBranchReceiptV1 {
    schema_version: u16,
    definition_checksum: String,
    status: CheckpointBranchStatusDto,
    artifacts: Vec<ArtifactMetadataDto>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointBranchReceiptRecord {
    receipt: CheckpointBranchReceiptV1,
    checksum: String,
}

fn checkpoint_branch_catalog_error(path: &Path, error: impl std::fmt::Display) -> ControlError {
    ControlError::Serialization(format!(
        "checkpoint branch catalog {}: {error}; preserve the saved study and reconcile its admission or receipt before retrying",
        path.display()
    ))
}

fn valid_branch_digest(value: &str) -> bool {
    value.len() == blake3::OUT_LEN * 2
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn valid_world_digest_v1(value: &str) -> bool {
    value.len() == std::mem::size_of::<u64>() * 2
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn checkpoint_branch_specs(
    request: &CheckpointBranchRequest,
    source_tick: u64,
) -> Result<Vec<crate::branch_study::StudyBranchSpec>, ControlError> {
    use crate::branch_study::StudyBranchSpec;

    request.validate_shape()?;
    let mut branches = vec![StudyBranchSpec {
        branch_id: "control".to_owned(),
        description: format!("{}: unchanged scientific future", request.branch_name),
        intervention: None,
        scheduled_tick: None,
    }];
    if let Some(intervention) = request.intervention.clone() {
        let scheduled_tick = source_tick
            .checked_add(request.intervention_delay_ticks)
            .ok_or_else(|| {
                ControlError::BadRequest(
                    "saved checkpoint tick plus intervention delay overflows".to_owned(),
                )
            })?;
        branches.push(StudyBranchSpec {
            branch_id: "treatment".to_owned(),
            description: format!("{}: requested intervention", request.branch_name),
            intervention: Some(intervention),
            scheduled_tick: Some(scheduled_tick),
        });
    }
    Ok(branches)
}

fn write_checkpoint_branch_record(
    directory: &Path,
    filename: &str,
    bytes: &[u8],
) -> Result<(), ControlError> {
    if bytes.len() > MAX_CHECKPOINT_BRANCH_CATALOG_BYTES {
        return Err(ControlError::PayloadTooLarge(
            "checkpoint branch catalog record exceeds 1 MiB".to_owned(),
        ));
    }
    validate_experiment_directory(directory)?;
    let committed = directory.join(filename);
    match fs::symlink_metadata(&committed) {
        Ok(_) => {
            let (file, existing) =
                open_artifact_file_bounded(&committed, MAX_CHECKPOINT_BRANCH_CATALOG_BYTES, true)?;
            if existing == bytes {
                file.sync_all()
                    .map_err(|error| checkpoint_branch_catalog_error(&committed, error))?;
                verify_open_artifact(&committed, &file)?;
                #[cfg(unix)]
                for path in [
                    Some(directory),
                    directory.parent(),
                    directory.parent().and_then(Path::parent),
                ]
                .into_iter()
                .flatten()
                {
                    sync_checkpoint_directory(path)
                        .map_err(|error| checkpoint_branch_catalog_error(path, error))?;
                }
                return Ok(());
            }
            return Err(checkpoint_branch_catalog_error(
                &committed,
                "immutable catalog record already exists with different bytes",
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(checkpoint_branch_catalog_error(&committed, error)),
    }
    let temporary = directory.join(format!("{filename}.tmp"));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| checkpoint_branch_catalog_error(&temporary, error))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| checkpoint_branch_catalog_error(&temporary, error))?;
    verify_open_artifact(&temporary, &file)?;
    // The exclusive admission directory and branch registry own this single writer.
    // An interrupted temporary record is preserved and never silently retried.
    fs::rename(&temporary, &committed)
        .map_err(|error| checkpoint_branch_catalog_error(&committed, error))?;
    verify_open_artifact(&committed, &file)?;
    #[cfg(unix)]
    for path in [
        Some(directory),
        directory.parent(),
        directory.parent().and_then(Path::parent),
    ]
    .into_iter()
    .flatten()
    {
        sync_checkpoint_directory(path)
            .map_err(|error| checkpoint_branch_catalog_error(path, error))?;
    }
    Ok(())
}

impl CheckpointBranchDefinitionV1 {
    fn checksum(&self) -> Result<String, ControlError> {
        serde_json::to_vec(self)
            .map(|bytes| compute_blake3(&bytes))
            .map_err(ControlError::serialization)
    }

    fn status(&self, status: &str) -> CheckpointBranchStatusDto {
        CheckpointBranchStatusDto {
            branch_id: self.branch_id.clone(),
            checkpoint_id: self.request.checkpoint_id.clone(),
            branch_name: self.request.branch_name.clone(),
            status: status.to_owned(),
            source_tick: self.source_tick,
            target_tick: self.target_tick,
            world_digest_v1: self.world_digest_v1.clone(),
            artifact_id: None,
            error_reason: None,
        }
    }

    fn validate(&self, directory: &Path) -> Result<(), ControlError> {
        self.request.validate_shape()?;
        let expected_id = format!(
            "branch-{}",
            compute_blake3(self.request.idempotency_key.as_bytes())
        );
        if self.schema_version != 1
            || self.branch_id != expected_id
            || directory.file_name().and_then(|name| name.to_str()) != Some(expected_id.as_str())
            || self.request_hash != compute_blake3(&self.request.bounded_payload()?)
            || self.source_tick.checked_add(self.request.ticks) != Some(self.target_tick)
            || !valid_world_digest_v1(&self.world_digest_v1)
            || !valid_branch_digest(&self.source_checksum_blake3)
            || self.source_byte_size == 0
            || self.source_byte_size > MAX_ARTIFACT_DOWNLOAD_BYTES as u64
            || self.created_at_utc.parse::<u64>().is_err()
            || self.plan.study_id != self.branch_id
            || self.plan.checkpoint_tick != self.source_tick
            || self.plan.horizon_ticks != self.request.ticks
            || self.plan.source_checkpoint.as_deref()
                != Some(directory.join("study/source_checkpoint.bin").as_path())
            || serde_json::to_vec(&self.plan.branches).map_err(ControlError::serialization)?
                != serde_json::to_vec(&checkpoint_branch_specs(&self.request, self.source_tick)?)
                    .map_err(ControlError::serialization)?
        {
            return Err(checkpoint_branch_catalog_error(
                directory,
                "admission version, request, plan or source identity mismatch",
            ));
        }
        // Recovery never executes a future. Admission already validated the exact
        // envelope bound by source_checksum_blake3, including its continuity policy.
        // Apply universal limits here so complete-source receipts remain discoverable.
        crate::branch_study::validate_study_config(&self.plan.config)
            .map_err(|error| checkpoint_branch_catalog_error(directory, error))?;
        Ok(())
    }

    fn persist(&self, directory: &Path) -> Result<(), ControlError> {
        self.validate(directory)?;
        let record = CheckpointBranchDefinitionRecord {
            definition: self.clone(),
            checksum: self.checksum()?,
        };
        write_checkpoint_branch_record(
            directory,
            "definition.json",
            &serde_json::to_vec(&record).map_err(ControlError::serialization)?,
        )
    }
}

impl CheckpointBranchReceiptV1 {
    fn validate(&self, definition: &CheckpointBranchDefinitionV1) -> Result<(), ControlError> {
        let expected = definition.status(&self.status.status);
        let mut identity = self.status.clone();
        identity.artifact_id = None;
        identity.error_reason = None;
        if self.schema_version != 1
            || self.definition_checksum != definition.checksum()?
            || identity != expected
            || !matches!(
                self.status.status.as_str(),
                "completed" | "failed" | "cancelled"
            )
            || self
                .status
                .error_reason
                .as_ref()
                .is_some_and(|reason| reason.len() > 8192)
            || self.artifacts.len() > MAX_CHECKPOINT_BRANCH_ARTIFACTS
        {
            return Err(ControlError::InvalidPatch(
                "checkpoint branch receipt does not match its immutable admission".to_owned(),
            ));
        }
        if self.status.status != "completed" {
            if self.status.artifact_id.is_some()
                || self
                    .status
                    .error_reason
                    .as_ref()
                    .is_none_or(|reason| reason.is_empty())
                || !self.artifacts.is_empty()
            {
                return Err(ControlError::InvalidPatch(
                    "failed or cancelled branch receipt cannot publish artifacts".to_owned(),
                ));
            }
            return Ok(());
        }
        let prefix = format!("branches/{}/study/", definition.branch_id);
        let mut indexed = BTreeMap::new();
        let mut total_bytes = 0_u64;
        for artifact in &self.artifacts {
            let path = Path::new(&artifact.relative_path);
            total_bytes = total_bytes
                .checked_add(artifact.byte_size)
                .filter(|size| *size <= MAX_CHECKPOINT_BRANCH_ARTIFACT_BYTES)
                .ok_or_else(|| {
                    ControlError::PayloadTooLarge(
                        "checkpoint branch receipt exceeds its aggregate artifact byte cap"
                            .to_owned(),
                    )
                })?;
            if !artifact.relative_path.starts_with(&prefix)
                || artifact.relative_path.contains('\\')
                || !path
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_)))
                || path.file_name().and_then(|name| name.to_str())
                    != Some(artifact.filename.as_str())
                || artifact.artifact_id
                    != format!(
                        "checkpoint-branch-{}",
                        compute_blake3(artifact.relative_path.as_bytes())
                    )
                || artifact.byte_size > MAX_ARTIFACT_DOWNLOAD_BYTES as u64
                || !valid_branch_digest(&artifact.checksum_blake3)
                || !valid_branch_digest(&artifact.checksum_sha256)
                || artifact.created_at_utc != definition.created_at_utc
                || indexed
                    .insert(artifact.relative_path.as_str(), artifact)
                    .is_some()
            {
                return Err(ControlError::InvalidPatch(
                    "checkpoint branch receipt contains an invalid artifact identity".to_owned(),
                ));
            }
        }
        let report = indexed.get(format!("{prefix}study_report.json").as_str());
        let source = indexed.get(format!("{prefix}source_checkpoint.bin").as_str());
        let plan = indexed.get(format!("{prefix}study_plan.json").as_str());
        let expected_plan =
            serde_json::to_vec_pretty(&definition.plan).map_err(ControlError::serialization)?;
        if self.status.error_reason.is_some()
            || report.is_none_or(|artifact| {
                self.status.artifact_id.as_deref() != Some(artifact.artifact_id.as_str())
            })
            || source.is_none_or(|artifact| {
                artifact.byte_size != definition.source_byte_size
                    || artifact.checksum_blake3 != definition.source_checksum_blake3
            })
            || plan.is_none_or(|artifact| {
                artifact.byte_size != expected_plan.len() as u64
                    || artifact.checksum_blake3 != compute_blake3(&expected_plan)
            })
            || !indexed.contains_key(format!("{prefix}study_source.json").as_str())
            || !indexed.contains_key(format!("{prefix}study_summary.md").as_str())
            || definition.plan.branches.iter().any(|branch| {
                !indexed.contains_key(
                    format!("{prefix}bundle_{}/bundle_manifest.json", branch.branch_id).as_str(),
                )
            })
        {
            return Err(ControlError::InvalidPatch(
                "completed checkpoint branch receipt lacks its exact source, plan or report"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    fn persist(&self, directory: &Path) -> Result<(), ControlError> {
        let bytes = serde_json::to_vec(self).map_err(ControlError::serialization)?;
        let record = CheckpointBranchReceiptRecord {
            receipt: self.clone(),
            checksum: compute_blake3(&bytes),
        };
        write_checkpoint_branch_record(
            directory,
            "receipt.json",
            &serde_json::to_vec(&record).map_err(ControlError::serialization)?,
        )
    }
}

fn read_checkpoint_branch_receipt(
    directory: &Path,
    definition: &CheckpointBranchDefinitionV1,
) -> Result<Option<CheckpointBranchReceiptV1>, ControlError> {
    let path = directory.join("receipt.json");
    match fs::symlink_metadata(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(checkpoint_branch_catalog_error(&path, error)),
        Ok(_) => {}
    }
    let bytes = read_artifact_file_bounded(&path, MAX_CHECKPOINT_BRANCH_CATALOG_BYTES)
        .map_err(|error| checkpoint_branch_catalog_error(&path, error))?;
    let record: CheckpointBranchReceiptRecord = serde_json::from_slice(&bytes)
        .map_err(|error| checkpoint_branch_catalog_error(&path, error))?;
    if record.checksum
        != compute_blake3(
            &serde_json::to_vec(&record.receipt).map_err(ControlError::serialization)?,
        )
    {
        return Err(checkpoint_branch_catalog_error(
            &path,
            "terminal receipt checksum mismatch",
        ));
    }
    record.receipt.validate(definition)?;
    Ok(Some(record.receipt))
}

/// A REST/MCP-created matched-seed batch executing on a background runner thread.
///
/// The runner's atomically written status file is the source of terminal run
/// evidence. Separate source-bound observations expose its sealed checkpoint
/// progress; this job adds process-local execution state and the last status
/// generation whose completed artifacts were verified and published.
struct ExperimentJob {
    runner: std::sync::Arc<crate::experiment_runner::MatchedSeedExperimentRunner>,
    state_file: PathBuf,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    running: std::sync::Arc<std::sync::atomic::AtomicBool>,
    last_error: std::sync::Arc<Mutex<Option<String>>>,
    created_at_utc: String,
    artifact_root: PathBuf,
    artifact_generation: Mutex<Option<u64>>,
    /// Retain ownership until restart or control-runtime shutdown joins this worker.
    worker: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for ExperimentJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExperimentJob")
            .field("experiment_id", &self.runner.experiment_id)
            .field("state_file", &self.state_file)
            .finish_non_exhaustive()
    }
}

impl ExperimentJob {
    fn join_worker(&mut self) -> Result<(), String> {
        if let Some(worker) = self.worker.take()
            && let Err(panic) = worker.join()
        {
            let message = format!(
                "experiment '{}' worker panicked: {panic:?}",
                self.runner.experiment_id
            );
            *self
                .last_error
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Some(message.clone());
            return Err(message);
        }
        Ok(())
    }

    fn recover_artifacts(&self, services: &DataServices) -> Result<(), ControlError> {
        let status = self
            .runner
            .status_snapshot(&self.state_file)
            .map_err(|error| experiment_catalog_error(&self.state_file, error))?;
        let mut published = self
            .artifact_generation
            .lock()
            .map_err(|_| ControlError::Lock)?;
        if published.is_some_and(|generation| generation >= status.generation) {
            return Ok(());
        }
        self.runner
            .verify_completed_status(&self.state_file, &status)
            .map_err(|error| experiment_catalog_error(&self.state_file, error))?;
        DataServices::publish_experiment_artifacts(
            &services.artifacts,
            &status,
            &self.artifact_root,
        )?;
        *published = Some(status.generation);
        Ok(())
    }

    /// Start (or restart) the batch worker. `running` is raised before the
    /// thread exists so a status read can never observe a false idle gap.
    /// Retain only the artifact catalog, not the service that owns this worker:
    /// the final external owner must remain able to cancel and join the batch.
    fn start(
        &mut self,
        artifacts: Arc<Mutex<BTreeMap<String, RetainedArtifact>>>,
        mut reservation: ExperimentExecutionReservation,
    ) {
        use std::sync::atomic::Ordering;
        // New jobs have no worker; resume reaps the previous worker before admission.
        debug_assert!(self.worker.is_none());
        self.running.store(true, Ordering::Release);
        reservation.running = Some(std::sync::Arc::clone(&self.running));
        let runner = std::sync::Arc::clone(&self.runner);
        let state_file = self.state_file.clone();
        let cancel = std::sync::Arc::clone(&self.cancel);
        let last_error = std::sync::Arc::clone(&self.last_error);
        let artifact_root = self.artifact_root.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("experiment-{}", runner.experiment_id))
            .spawn(move || {
                let _reservation = reservation;
                let outcome = match runner.execute_batch_until_cancelled(&state_file, &cancel) {
                    Ok(status) => DataServices::publish_experiment_artifacts(
                        &artifacts,
                        &status,
                        &artifact_root,
                    )
                    .map_err(|error| format!("artifact publication failed: {error}")),
                    Err(error) => Err(error.to_string()),
                };
                if let Err(error) = outcome {
                    tracing::error!(
                        experiment_id = %runner.experiment_id,
                        %error,
                        "experiment batch stopped with an error"
                    );
                    if let Ok(mut slot) = last_error.lock() {
                        *slot = Some(error.to_string());
                    }
                }
            });
        match spawned {
            Ok(worker) => self.worker = Some(worker),
            Err(error) => {
                if let Ok(mut slot) = self.last_error.lock() {
                    *slot = Some(format!("cannot spawn experiment worker: {error}"));
                }
                self.running.store(false, Ordering::Release);
            }
        }
    }

    fn status_dto(
        &self,
        recover_checkpoint: bool,
    ) -> Result<ExperimentBatchStatusDto, ControlError> {
        use crate::experiment_runner::RunState;
        use std::sync::atomic::Ordering;
        let running = self.running.load(Ordering::Acquire);
        let cancel_requested = self.cancel.load(Ordering::Acquire);
        let error_reason = self.last_error.lock().ok().and_then(|slot| slot.clone());
        let status = self
            .runner
            .status_snapshot(&self.state_file)
            .map_err(|error| {
                ControlError::Serialization(format!(
                    "experiment status {} is unreadable: {error}",
                    self.state_file.display()
                ))
            })?;
        let updated_at_utc = fs::metadata(&self.state_file)
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or_else(|| self.created_at_utc.clone(), |d| d.as_secs().to_string());
        let finished = status.is_finished();
        let any_started = status.runs.iter().any(|run| run.state != RunState::Pending);
        let label = if running && cancel_requested {
            "cancelling"
        } else if running {
            if any_started { "running" } else { "pending" }
        } else if error_reason.is_some() {
            "failed"
        } else if finished {
            if status.failed_runs > 0 {
                "failed"
            } else {
                "completed"
            }
        } else if cancel_requested {
            "cancelled"
        } else {
            "stopped"
        };
        let mut checkpoint_progress = self
            .runner
            .checkpoint_progress_snapshot(&self.state_file, &status, running || !recover_checkpoint)
            .map_err(|error| experiment_catalog_error(&self.state_file, error))?;
        let runs = status
            .runs
            .into_iter()
            .map(|run| ExperimentRunRecordDto {
                target_ticks: self.runner.max_ticks,
                checkpoint_progress: checkpoint_progress
                    .remove(&run.run_id)
                    .expect("progress covers every validated run"),
                run_id: run.run_id,
                variant_id: run.variant_id,
                brain_family: run.brain_family,
                seed: run.seed,
                state: match run.state {
                    RunState::Pending => "pending",
                    RunState::Running => "running",
                    RunState::Completed => "completed",
                    RunState::Failed => "failed",
                }
                .to_string(),
                total_ticks: run.total_ticks,
                final_digest: run.final_digest,
                bundle_path: run.bundle_path,
                error_reason: run.error_reason,
            })
            .collect();
        Ok(ExperimentBatchStatusDto {
            schema_version: status.schema_version,
            generation: status.generation,
            plan_digest: status.plan_digest,
            experiment_id: status.experiment_id,
            status: label.to_string(),
            total_runs: status.total_runs,
            completed_runs: status.completed_runs,
            failed_runs: status.failed_runs,
            runs,
            error_reason,
            created_at_utc: self.created_at_utc.clone(),
            updated_at_utc,
        })
    }
}

/// Canonical in-process data services managing experiments, checkpoints, and artifacts.
/// Releasing the final owner cancels and joins its workers at their existing safe
/// boundaries. Use explicit shutdown to receive any worker errors directly.
#[derive(Debug)]
pub struct DataServices {
    experiments: Mutex<BTreeMap<String, ExperimentJob>>,
    /// One retry authority and worker owner for every admitted checkpoint branch.
    checkpoint_branches: Mutex<BTreeMap<String, CheckpointBranchJob>>,
    /// A nonblocking, shared admission slot for bundle restoration and statistics.
    experiment_comparison: Mutex<()>,
    /// Once closed, retained handles may inspect work but cannot start or resume it.
    experiment_shutdown_requested: std::sync::atomic::AtomicBool,
    execution_capacity: Arc<Mutex<ExperimentExecutionCapacityDto>>,
    checkpoints: Mutex<BTreeMap<String, CheckpointMetadataDto>>,
    artifacts: Arc<Mutex<BTreeMap<String, RetainedArtifact>>>,
    /// Admission authority, recovered from immutable experiment definitions:
    /// request key -> (payload hash, admitted ID).
    /// Status is read from the job rather than cached as a creation response.
    idempotency: Mutex<BTreeMap<String, (String, String)>>,
    artifacts_dir: RwLock<PathBuf>,
    checkpoint_retention: Mutex<CheckpointRetentionState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RetainedArtifact {
    metadata: ArtifactMetadataDto,
    directory: PathBuf,
}

/// A finite scientific fork. Workers never retain DataServices, so dropping the
/// final service handle can cancel and join them without an ownership cycle.
#[derive(Debug)]
struct CheckpointBranchJob {
    definition: CheckpointBranchDefinitionV1,
    directory: PathBuf,
    artifact_root: PathBuf,
    /// Recovered historical receipt; current files are checked on artifact access.
    receipt: Option<CheckpointBranchReceiptV1>,
    artifacts_published: bool,
    status: Arc<Mutex<CheckpointBranchStatusDto>>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl CheckpointBranchJob {
    fn start(
        &mut self,
        runner: crate::branch_study::BranchStudyOrchestrator,
        checkpoint: CatalogCheckpoint,
        artifact_root: PathBuf,
        artifacts: Arc<Mutex<BTreeMap<String, RetainedArtifact>>>,
        reservation: ExperimentExecutionReservation,
    ) {
        let status = Arc::clone(&self.status);
        let cancel = Arc::clone(&self.cancel);
        let definition = self.definition.clone();
        let directory = self.directory.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("checkpoint-branch-{}", runner.plan.study_id))
            // Checkpoint reconstruction uses the application's established reader stack.
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let _reservation = reservation;
                let result = match &checkpoint {
                    CatalogCheckpoint::Science(checkpoint) => {
                        runner.execute_from_checkpoint(checkpoint, Some(&cancel))
                    }
                    CatalogCheckpoint::Host(checkpoint) => {
                        runner.execute_from_host_checkpoint(checkpoint, Some(&cancel))
                    }
                };
                let result = match result {
                    Ok(report) => stage_checkpoint_branch_artifacts(
                        &artifact_root,
                        &runner.output_dir,
                        &report,
                        &definition,
                    )
                    .map_err(|error| (false, error.to_string())),
                    Err(crate::branch_study::BranchStudyError::Cancelled) => {
                        Err((true, "checkpoint branch cancellation observed".to_owned()))
                    }
                    Err(error) => Err((false, error.to_string())),
                };
                let mut terminal = definition.status("failed");
                let mut staged = BTreeMap::new();
                match result {
                    Ok((artifact_id, ready)) => {
                        terminal.status = "completed".to_owned();
                        terminal.artifact_id = Some(artifact_id);
                        staged = ready;
                    }
                    Err((cancelled, reason)) => {
                        terminal.status = if cancelled { "cancelled" } else { "failed" }.to_owned();
                        terminal.error_reason = Some(reason.chars().take(2048).collect());
                    }
                }
                let sealed = (|| {
                    // Producers sync their files; seal the complete directory tree too,
                    // before a durable terminal receipt names any downloadable output.
                    #[cfg(unix)]
                    {
                        let mut directories = std::collections::BTreeSet::new();
                        for artifact in staged.values() {
                            let path = artifact_root.join(&artifact.metadata.relative_path);
                            let mut parent = path.parent();
                            while let Some(path) = parent {
                                if !path.starts_with(&directory) {
                                    break;
                                }
                                directories.insert(path.to_path_buf());
                                parent = path.parent();
                            }
                        }
                        for path in directories.into_iter().rev() {
                            sync_checkpoint_directory(&path)
                                .map_err(|error| checkpoint_branch_catalog_error(&path, error))?;
                        }
                    }
                    let receipt = CheckpointBranchReceiptV1 {
                        schema_version: 1,
                        definition_checksum: definition.checksum()?,
                        status: terminal.clone(),
                        artifacts: staged
                            .values()
                            .map(|artifact| artifact.metadata.clone())
                            .collect(),
                    };
                    receipt.validate(&definition)?;
                    receipt.persist(&directory)?;
                    // Completion is the synced terminal observation. A volatile
                    // publication failure cannot undo that observation; cold lookup
                    // can revalidate and publish the same sealed artifact set.
                    if let Err(error) = publish_checkpoint_branch_artifact_set(&artifacts, staged) {
                        tracing::error!(branch_id = %definition.branch_id, %error,
                            "completed checkpoint branch artifact publication requires retry");
                    }
                    Ok::<(), ControlError>(())
                })();
                if let Err(error) = sealed {
                    terminal.status = "failed".to_owned();
                    terminal.artifact_id = None;
                    terminal.error_reason = Some(format!(
                        "checkpoint branch receipt/publication failed: {error}"
                    ));
                }
                let mut state = status.lock().unwrap_or_else(PoisonError::into_inner);
                *state = terminal;
                tracing::info!(
                    branch_id = %state.branch_id,
                    checkpoint_id = %state.checkpoint_id,
                    status = %state.status,
                    source_tick = state.source_tick,
                    target_tick = state.target_tick,
                    "checkpoint branch execution stopped"
                );
            });
        match spawned {
            Ok(worker) => self.worker = Some(worker),
            Err(error) => {
                let mut state = self.status.lock().unwrap_or_else(PoisonError::into_inner);
                state.status = "failed".to_owned();
                state.error_reason =
                    Some(format!("cannot spawn checkpoint branch worker: {error}"));
                let retained = self.definition.checksum().and_then(|checksum| {
                    CheckpointBranchReceiptV1 {
                        schema_version: 1,
                        definition_checksum: checksum,
                        status: state.clone(),
                        artifacts: Vec::new(),
                    }
                    .persist(&self.directory)
                });
                if let Err(error) = retained {
                    state.error_reason = Some(format!(
                        "{}; receipt persistence failed: {error}",
                        state.error_reason.as_deref().unwrap_or("worker failed")
                    ));
                }
            }
        }
    }

    fn recover_artifacts(
        &mut self,
        artifacts: &Mutex<BTreeMap<String, RetainedArtifact>>,
    ) -> Result<(), ControlError> {
        if self.artifacts_published {
            return Ok(());
        }
        let Some(receipt) = &self.receipt else {
            return Ok(());
        };
        if receipt.status.status != "completed" {
            return Ok(());
        }
        let output = self.directory.join("study");
        validate_experiment_directory(&self.directory)?;
        validate_experiment_directory(&output)?;
        // Check the receipt's complete fixed file set before consulting any
        // recovered manifest. Each read is bounded by its sealed byte length.
        for artifact in &receipt.artifacts {
            let path = self.artifact_root.join(&artifact.relative_path);
            if path
                .canonicalize()
                .map_err(|error| checkpoint_branch_catalog_error(&path, error))?
                != path
            {
                return Err(checkpoint_branch_catalog_error(
                    &path,
                    "artifact path is aliased",
                ));
            }
            let bytes = read_artifact_file_bounded(
                &path,
                usize::try_from(artifact.byte_size)
                    .map_err(|error| checkpoint_branch_catalog_error(&path, error))?,
            )?;
            if bytes.len() as u64 != artifact.byte_size
                || compute_blake3(&bytes) != artifact.checksum_blake3
                || compute_sha256(&bytes) != artifact.checksum_sha256
            {
                return Err(checkpoint_branch_catalog_error(
                    &path,
                    "artifact differs from the terminal receipt",
                ));
            }
        }
        let report_artifact = receipt
            .artifacts
            .iter()
            .find(|artifact| {
                receipt.status.artifact_id.as_deref() == Some(artifact.artifact_id.as_str())
            })
            .ok_or_else(|| {
                checkpoint_branch_catalog_error(&self.directory, "receipt lost its report")
            })?;
        let bytes = read_artifact_file_bounded(
            &output.join("study_report.json"),
            MAX_ARTIFACT_DOWNLOAD_BYTES,
        )?;
        if bytes.len() as u64 != report_artifact.byte_size
            || compute_blake3(&bytes) != report_artifact.checksum_blake3
            || compute_sha256(&bytes) != report_artifact.checksum_sha256
        {
            return Err(checkpoint_branch_catalog_error(
                &output,
                "report differs from the terminal receipt",
            ));
        }
        let report = serde_json::from_slice(&bytes).map_err(ControlError::serialization)?;
        let (report_id, staged) = stage_checkpoint_branch_artifacts(
            &self.artifact_root,
            &output,
            &report,
            &self.definition,
        )?;
        let expected: Vec<_> = staged
            .values()
            .map(|artifact| artifact.metadata.clone())
            .collect();
        if receipt.status.artifact_id.as_deref() != Some(report_id.as_str())
            || expected != receipt.artifacts
        {
            return Err(checkpoint_branch_catalog_error(
                &output,
                "artifact set differs from the terminal receipt",
            ));
        }
        publish_checkpoint_branch_artifact_set(artifacts, staged)?;
        self.artifacts_published = true;
        Ok(())
    }

    fn request_cancel(&self) {
        let mut state = self.status.lock().unwrap_or_else(PoisonError::into_inner);
        if matches!(state.status.as_str(), "running" | "cancelling") {
            self.cancel
                .store(true, std::sync::atomic::Ordering::Release);
            state.status = "cancelling".to_owned();
        }
    }

    fn join_worker(&mut self) -> Result<(), String> {
        if let Some(worker) = self.worker.take() {
            if let Err(panic) = worker.join() {
                let mut state = self.status.lock().unwrap_or_else(PoisonError::into_inner);
                let message = format!(
                    "checkpoint branch '{}' worker panicked: {panic:?}",
                    state.branch_id
                );
                state.status = "failed".to_owned();
                state.artifact_id = None;
                state.error_reason = Some(message.clone());
                return Err(message);
            }
            self.receipt = read_checkpoint_branch_receipt(&self.directory, &self.definition)
                .map_err(|error| error.to_string())?;
            if let Some(receipt) = &self.receipt {
                *self.status.lock().unwrap_or_else(PoisonError::into_inner) =
                    receipt.status.clone();
            }
        }
        Ok(())
    }

    fn status_dto(&mut self) -> Result<CheckpointBranchStatusDto, ControlError> {
        if self
            .worker
            .as_ref()
            .is_some_and(std::thread::JoinHandle::is_finished)
        {
            // The failed observation remains available as a typed terminal job status.
            self.join_worker().map_err(ControlError::Serialization)?;
        }
        Ok(self
            .status
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone())
    }
}

impl Drop for CheckpointBranchJob {
    fn drop(&mut self) {
        self.cancel
            .store(true, std::sync::atomic::Ordering::Release);
        if let Err(error) = self.join_worker() {
            tracing::error!(%error, "checkpoint branch worker failed during drop");
        }
    }
}

/// Validate report-controlled identities before using any of them as paths.
fn validate_checkpoint_branch_report(
    definition: &CheckpointBranchDefinitionV1,
    report: &crate::branch_study::BranchStudyReport,
) -> Result<(), ControlError> {
    let expected_source = crate::branch_study::BranchStudySource::SavedCheckpoint {
        artifact_path: "source_checkpoint.bin".to_owned(),
        byte_size: definition.source_byte_size,
        checksum_blake3: definition.source_checksum_blake3.clone(),
    };
    if report.schema_version != crate::branch_study::BRANCH_STUDY_REPORT_SCHEMA_VERSION
        || report.study_id != definition.branch_id
        || report.base_seed != definition.plan.base_seed
        || report.source != expected_source
        || report.checkpoint_tick != definition.source_tick
        || report.horizon_ticks != definition.request.ticks
        || report.pre_intervention_digest != definition.world_digest_v1
        || !report.pre_intervention_digest_verified_for_all_branches
        || report.control_branch_id != "control"
        || report.branch_reports.len() != definition.plan.branches.len()
        || report
            .branch_reports
            .iter()
            .zip(&definition.plan.branches)
            .any(|(branch, spec)| {
                branch.branch_id != spec.branch_id
                    || branch.description != spec.description
                    || branch.command_applied_tick != spec.scheduled_tick
                    || branch.terminal_sample.tick != definition.target_tick
                    || branch.final_digest != branch.terminal_sample.digest_overall
                    || !valid_world_digest_v1(&branch.final_digest)
                    || !branch.bundle_verified
            })
    {
        return Err(ControlError::InvalidPatch(
            "checkpoint branch report differs from its admitted source, plan or complete branch roster".to_owned(),
        ));
    }
    Ok(())
}

/// Stage bounded, verified files before sealing or publishing any metadata.
fn stage_checkpoint_branch_artifacts(
    root: &Path,
    output: &Path,
    report: &crate::branch_study::BranchStudyReport,
    definition: &CheckpointBranchDefinitionV1,
) -> Result<(String, BTreeMap<String, RetainedArtifact>), ControlError> {
    validate_checkpoint_branch_report(definition, report)?;
    validate_experiment_directory(root)?;
    validate_experiment_directory(output)?;
    let report_path = output.join("study_report.json");
    let report_bytes = read_artifact_file_bounded(&report_path, MAX_ARTIFACT_DOWNLOAD_BYTES)?;
    if report_bytes != serde_json::to_vec_pretty(report).map_err(ControlError::serialization)? {
        return Err(ControlError::InvalidPatch(
            "checkpoint branch report changed before publication".to_owned(),
        ));
    }
    let mut files = BTreeMap::new();
    for (name, bytes) in [
        (
            "study_plan.json",
            serde_json::to_vec_pretty(&definition.plan).map_err(ControlError::serialization)?,
        ),
        (
            "study_source.json",
            serde_json::to_vec_pretty(&report.source).map_err(ControlError::serialization)?,
        ),
        (
            "study_summary.md",
            report.summary_markdown.as_bytes().to_vec(),
        ),
    ] {
        files.insert(
            output.join(name),
            Some((bytes.len() as u64, compute_blake3(&bytes))),
        );
    }
    files.insert(
        report_path.clone(),
        Some((report_bytes.len() as u64, compute_blake3(&report_bytes))),
    );
    let crate::branch_study::BranchStudySource::SavedCheckpoint {
        artifact_path,
        byte_size,
        checksum_blake3,
    } = &report.source
    else {
        return Err(ControlError::InvalidPatch(
            "managed branch report lost its saved source".to_owned(),
        ));
    };
    files.insert(
        output.join(artifact_path),
        Some((*byte_size, checksum_blake3.clone())),
    );
    // The report retains only the overall source digest. Derive the full run-ID
    // input from the exact admitted envelope, rather than trusting a bundle's
    // self-consistent manifest or reconstructing missing digest lanes.
    let source_bytes = read_artifact_file_bounded(
        &output.join(artifact_path),
        usize::try_from(*byte_size)
            .map_err(|error| ControlError::PayloadTooLarge(error.to_string()))?,
    )?;
    if source_bytes.len() as u64 != *byte_size || compute_blake3(&source_bytes) != *checksum_blake3 {
        return Err(ControlError::InvalidPatch(
            "checkpoint branch source differs from its immutable admission".to_owned(),
        ));
    }
    let source =
        CatalogCheckpoint::decode(scriptbots_runtime::HOST_CHECKPOINT_V1_SCHEMA, &source_bytes)
            .or_else(|_| {
                CatalogCheckpoint::decode(
                    scriptbots_core::WORLD_CHECKPOINT_V1_SCHEMA,
                    &source_bytes,
                )
            })?;
    let science = source.science();
    if science.tick().0 != definition.source_tick
        || science.source_digest().overall != definition.world_digest_v1
        || serde_json::to_vec(science.config()).map_err(ControlError::serialization)?
            != serde_json::to_vec(&definition.plan.config).map_err(ControlError::serialization)?
    {
        return Err(ControlError::InvalidPatch(
            "checkpoint branch source does not describe its admitted plan".to_owned(),
        ));
    }
    let source_digest = science.source_digest().clone();
    drop(source);
    drop(source_bytes);
    for (branch, spec) in report.branch_reports.iter().zip(&definition.plan.branches) {
        let bundle = output.join(format!("bundle_{}", branch.branch_id));
        if !branch.bundle_verified
            || branch.bundle_path.as_deref().map(Path::new) != Some(bundle.as_path())
        {
            return Err(ControlError::InvalidPatch(
                "checkpoint branch bundle identity mismatch".to_owned(),
            ));
        }
        validate_experiment_directory(&bundle)?;
        let manifest = bundle.join("bundle_manifest.json");
        let manifest_limit = usize::try_from(
            crate::branch_study::STUDY_BUNDLE_VERIFICATION_LIMITS.max_manifest_bytes,
        )
        .expect("study manifest byte limit fits usize");
        let bytes = read_artifact_file_bounded(&manifest, manifest_limit)?;
        scriptbots_storage::bundle::verify_run_bundle_bounded(
            &bundle,
            crate::branch_study::STUDY_BUNDLE_VERIFICATION_LIMITS,
        )
        .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
        if read_artifact_file_bounded(&manifest, manifest_limit)? != bytes {
            return Err(ControlError::InvalidPatch(
                "checkpoint branch manifest changed during verification".to_owned(),
            ));
        }
        let parsed: scriptbots_storage::RunBundleV1 =
            serde_json::from_slice(&bytes).map_err(ControlError::serialization)?;
        if parsed.artifacts.len()
            > crate::branch_study::STUDY_BUNDLE_VERIFICATION_LIMITS.max_artifacts
        {
            return Err(ControlError::PayloadTooLarge(
                "checkpoint branch bundle contains too many artifacts".to_owned(),
            ));
        }
        let expected_manifest = definition
            .plan
            .bundle_manifest(spec, &source_digest, &report.source)
            .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
        if parsed.manifest != expected_manifest
            || parsed.digests.max_tick != definition.target_tick
            || parsed.artifacts.len() != crate::branch_study::STUDY_BRANCH_ARTIFACTS.len()
            || crate::branch_study::STUDY_BRANCH_ARTIFACTS
                .iter()
                .any(|(path, artifact_type)| {
                    !parsed.artifacts.iter().any(|artifact| {
                        artifact.relative_path == *path && artifact.artifact_type == *artifact_type
                    })
                })
        {
            return Err(ControlError::InvalidPatch(
                "checkpoint branch bundle identity or artifact roster differs from its admitted study"
                    .to_owned(),
            ));
        }
        // Bundle checksums alone cannot bind an intact bundle copied from another
        // study to this report. Compare the actual evidence with the admitted arm.
        validate_experiment_directory(&bundle.join("evidence"))?;
        let evidence = read_artifact_file_bounded(
            &bundle.join("evidence/branch_evidence.json"),
            MAX_ARTIFACT_DOWNLOAD_BYTES,
        )?;
        let expected_evidence = serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": report.schema_version,
            "branch_id": spec.branch_id,
            "description": spec.description,
            "base_seed": report.base_seed,
            "source": report.source,
            "checkpoint_tick": report.checkpoint_tick,
            "horizon_ticks": report.horizon_ticks,
            "intervention": spec.intervention,
            "scheduled_tick": spec.scheduled_tick,
            "command_applied_tick": branch.command_applied_tick,
            "first_divergence": branch.first_divergence,
            "extinction": branch.extinction,
            "recovery": branch.recovery,
            "hysteresis": branch.hysteresis,
            "effect_sizes": branch.effect_sizes,
            "terminal_sample": branch.terminal_sample,
        }))
        .map_err(ControlError::serialization)?;
        if evidence != expected_evidence {
            return Err(ControlError::InvalidPatch(
                "checkpoint branch bundle evidence differs from its admitted arm or report"
                    .to_owned(),
            ));
        }
        files.insert(manifest, Some((bytes.len() as u64, compute_blake3(&bytes))));
        for artifact in parsed.artifacts {
            files.insert(
                bundle.join(artifact.relative_path),
                Some((artifact.bytes_len, artifact.blake3_hex)),
            );
        }
    }
    if files.len() > MAX_CHECKPOINT_BRANCH_ARTIFACTS {
        return Err(ControlError::PayloadTooLarge(
            "checkpoint branch artifact count exceeds its bound".to_owned(),
        ));
    }
    let mut staged = BTreeMap::new();
    let mut report_id = None;
    let mut total_bytes = 0_u64;
    for (path, expected) in files {
        let relative = path.strip_prefix(root).map_err(|_| {
            ControlError::BadRequest(
                "checkpoint branch artifact escapes its source root".to_owned(),
            )
        })?;
        if !relative
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
            || path
                .canonicalize()
                .map_err(|error| ControlError::Serialization(error.to_string()))?
                != path
        {
            return Err(ControlError::BadRequest(
                "checkpoint branch artifact is aliased or contains traversal".to_owned(),
            ));
        }
        let bytes = read_artifact_file_bounded(&path, MAX_ARTIFACT_DOWNLOAD_BYTES)?;
        total_bytes = total_bytes
            .checked_add(bytes.len() as u64)
            .filter(|size| *size <= MAX_CHECKPOINT_BRANCH_ARTIFACT_BYTES)
            .ok_or_else(|| {
                ControlError::PayloadTooLarge(
                    "checkpoint branch artifacts exceed their aggregate byte cap".to_owned(),
                )
            })?;
        let checksum_blake3 = compute_blake3(&bytes);
        if expected
            .as_ref()
            .is_some_and(|(size, digest)| *size != bytes.len() as u64 || digest != &checksum_blake3)
        {
            return Err(ControlError::InvalidPatch(
                "checkpoint branch artifact changed before publication".to_owned(),
            ));
        }
        let relative_path = relative
            .components()
            .map(|component| {
                component.as_os_str().to_str().ok_or_else(|| {
                    ControlError::BadRequest("non-UTF-8 branch artifact path".to_owned())
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .join("/");
        let artifact_id = format!(
            "checkpoint-branch-{}",
            compute_blake3(relative_path.as_bytes())
        );
        if path == report_path {
            report_id = Some(artifact_id.clone());
        }
        let metadata = ArtifactMetadataDto {
            artifact_id: artifact_id.clone(),
            filename: path
                .file_name()
                .ok_or(ControlError::Lock)?
                .to_string_lossy()
                .into_owned(),
            content_type: match path.extension().and_then(|extension| extension.to_str()) {
                Some("json") => "application/json",
                Some("csv") => "text/csv",
                Some("md") => "text/markdown",
                _ => "application/octet-stream",
            }
            .to_owned(),
            byte_size: bytes.len() as u64,
            checksum_blake3,
            checksum_sha256: compute_sha256(&bytes),
            created_at_utc: definition.created_at_utc.clone(),
            relative_path,
        };
        staged.insert(
            artifact_id,
            RetainedArtifact {
                metadata,
                directory: root.to_path_buf(),
            },
        );
    }
    let report_id = report_id.ok_or_else(|| {
        ControlError::InvalidPatch("completed branch has no report artifact".to_owned())
    })?;
    Ok((report_id, staged))
}

fn publish_checkpoint_branch_artifact_set(
    artifacts: &Mutex<BTreeMap<String, RetainedArtifact>>,
    staged: BTreeMap<String, RetainedArtifact>,
) -> Result<(), ControlError> {
    let mut catalog = artifacts.lock().map_err(|_| ControlError::Lock)?;
    if staged
        .iter()
        .any(|(id, artifact)| catalog.get(id).is_some_and(|known| known != artifact))
    {
        return Err(ControlError::Conflict(
            "checkpoint branch artifact identity was published with different metadata or root"
                .to_owned(),
        ));
    }
    catalog.extend(staged);
    Ok(())
}

#[derive(Debug, Clone)]
struct RetainedCheckpoint {
    request_hash: String,
    metadata: CheckpointMetadataDto,
    artifact: ArtifactMetadataDto,
    directory: PathBuf,
    database_metadata: Value,
    acknowledged: bool,
    source_session: scriptbots_runtime::HostSessionId,
    source_scientific_revision: scriptbots_runtime::ScientificRevision,
    source_config_revision: scriptbots_runtime::ConfigRevision,
    source_persistence_interval: u32,
    source_scheduled_input_count: u64,
}

/// An immutable catalog source keeps its declared codec and complete science authority.
enum CatalogCheckpoint {
    Science(Box<scriptbots_core::WorldCheckpointV1>),
    Host(Box<scriptbots_runtime::HostCheckpointV1>),
}

impl CatalogCheckpoint {
    fn decode(schema: &str, bytes: &[u8]) -> Result<Self, ControlError> {
        match schema {
            scriptbots_core::WORLD_CHECKPOINT_V1_SCHEMA => {
                scriptbots_core::WorldCheckpointV1::decode(bytes)
                    .map(|checkpoint| Self::Science(Box::new(checkpoint)))
                    .map_err(|error| ControlError::Serialization(error.to_string()))
            }
            scriptbots_runtime::HOST_CHECKPOINT_V1_SCHEMA => {
                scriptbots_runtime::HostCheckpointV1::decode(bytes)
                    .map(|checkpoint| Self::Host(Box::new(checkpoint)))
                    .map_err(|error| ControlError::Serialization(error.to_string()))
            }
            _ => Err(ControlError::InvalidPatch(
                "unsupported checkpoint catalog schema".into(),
            )),
        }
    }

    fn science(&self) -> &scriptbots_core::WorldCheckpointV1 {
        match self {
            Self::Science(checkpoint) => checkpoint,
            Self::Host(checkpoint) => checkpoint.world_checkpoint(),
        }
    }

    fn schema(&self) -> &'static str {
        match self {
            Self::Science(_) => scriptbots_core::WORLD_CHECKPOINT_V1_SCHEMA,
            Self::Host(_) => scriptbots_runtime::HOST_CHECKPOINT_V1_SCHEMA,
        }
    }

    fn encode(&self) -> Result<Vec<u8>, ControlError> {
        match self {
            Self::Science(checkpoint) => checkpoint
                .encode()
                .map_err(|error| ControlError::Serialization(error.to_string())),
            Self::Host(checkpoint) => checkpoint
                .encode()
                .map_err(|error| ControlError::Serialization(error.to_string())),
        }
    }

    /// Explicitly extracting the raw component must not discard retained auxiliary science.
    #[cfg(test)]
    fn standalone_science(&self) -> Result<scriptbots_core::WorldCheckpointV1, ControlError> {
        match self {
            Self::Science(checkpoint) => checkpoint.validate_standalone_restore(),
            Self::Host(checkpoint) => checkpoint.validate_standalone_projection(),
        }
        .map_err(|error| ControlError::BadRequest(error.to_string()))?;
        Ok(self.science().clone())
    }
}

const CHECKPOINT_CATALOG_SUFFIX: &str = ".checkpoint.json";
const CHECKPOINT_CATALOG_READY_SUFFIX: &str = ".checkpoint.ready";
const MAX_CHECKPOINT_CATALOG_RECORD_BYTES: usize = 16 * 1024;
const CHECKPOINT_REQUEST_PREFIX: &str = "checkpoint-request-";
const CHECKPOINT_REQUEST_SUFFIX: &str = ".json";
const MAX_CHECKPOINT_REQUEST_RECORD_BYTES: usize = 16 * 1024;
const MAX_CATALOG_CHECKPOINTS: usize = 1_024;
const MAX_CHECKPOINT_DISCOVERY_ENTRIES: usize = 16_384;
const MAX_CHECKPOINT_DISCOVERY_BYTES: u64 = 256 * 1024 * 1024;

// Admission limits for a saved remote canvas, not a total resident-memory guarantee.
const MAX_CHECKPOINT_PREVIEW_BYTES: usize = 16 * 1024 * 1024;
const MAX_CHECKPOINT_PREVIEW_GRID_CELLS: u64 = 262_144;
const MAX_CHECKPOINT_PREVIEW_HISTORY: usize = 10_000;

/// A completed file catalog entry, published only after the original writer
/// acknowledges. Recovery exposes saved science for review/download; it does
/// not adopt the original owner's requests, journal or persistence session.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointCatalogRecord {
    schema_version: u16,
    metadata: CheckpointMetadataDto,
    checksum_blake3: String,
}

/// Reserve the exact request before creating its payload. An admission alone
/// never proves completion: recovery requires the ready catalog and original bytes.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointRequestAdmission {
    schema_version: u16,
    idempotency_key: String,
    request_hash: String,
    metadata: CheckpointMetadataDto,
    /// Original capture context only. Recovery never adopts this session or owner,
    /// including when the saved artifact carries the complete owner envelope.
    source_session: scriptbots_runtime::HostSessionId,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointRequestRecord {
    admission: CheckpointRequestAdmission,
    checksum_blake3: String,
}

fn checkpoint_request_path(directory: &Path, key: &str) -> PathBuf {
    directory.join(format!(
        "{CHECKPOINT_REQUEST_PREFIX}{}{CHECKPOINT_REQUEST_SUFFIX}",
        compute_blake3(key.as_bytes())
    ))
}

fn write_checkpoint_request_admission(
    directory: &Path,
    key: &str,
    retained: &RetainedCheckpoint,
) -> Result<(), ControlError> {
    let path = checkpoint_request_path(directory, key);
    let admission = CheckpointRequestAdmission {
        schema_version: 1,
        idempotency_key: key.to_owned(),
        request_hash: retained.request_hash.clone(),
        metadata: retained.metadata.clone(),
        source_session: retained.source_session,
    };
    let encoded = serde_json::to_vec(&admission).map_err(ControlError::serialization)?;
    let record = CheckpointRequestRecord {
        admission,
        checksum_blake3: compute_blake3(&encoded),
    };
    let bytes = serde_json::to_vec(&record).map_err(ControlError::serialization)?;
    if bytes.len() > MAX_CHECKPOINT_REQUEST_RECORD_BYTES {
        return Err(checkpoint_catalog_error(
            &path,
            "request admission exceeds its byte bound",
        ));
    }
    let file = match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut file) => {
            file.write_all(&bytes)
                .map_err(|error| checkpoint_catalog_error(&path, error))?;
            file
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let (file, existing) =
                open_artifact_file_bounded(&path, MAX_CHECKPOINT_REQUEST_RECORD_BYTES, true)
                    .map_err(|error| checkpoint_catalog_error(&path, error))?;
            if existing != bytes {
                return Err(checkpoint_catalog_error(
                    &path,
                    "existing request admission is incomplete or different; retry cannot overwrite it",
                ));
            }
            file
        }
        Err(error) => return Err(checkpoint_catalog_error(&path, error)),
    };
    file.sync_all()
        .map_err(|error| checkpoint_catalog_error(&path, error))?;
    verify_open_artifact(&path, &file).map_err(|error| checkpoint_catalog_error(&path, error))?;
    #[cfg(unix)]
    sync_checkpoint_directory(directory)
        .map_err(|error| checkpoint_catalog_error(directory, error))?;
    Ok(())
}

fn read_checkpoint_request_record(path: &Path) -> Result<CheckpointRequestAdmission, ControlError> {
    let directory = path
        .parent()
        .ok_or_else(|| checkpoint_catalog_error(path, "request admission has no directory"))?;
    let bytes = read_artifact_file_bounded(path, MAX_CHECKPOINT_REQUEST_RECORD_BYTES)
        .map_err(|error| checkpoint_catalog_error(path, error))?;
    let record: CheckpointRequestRecord =
        serde_json::from_slice(&bytes).map_err(|error| checkpoint_catalog_error(path, error))?;
    let encoded = serde_json::to_vec(&record.admission).map_err(ControlError::serialization)?;
    let admission = record.admission;
    if admission.schema_version != 1
        || record.checksum_blake3 != compute_blake3(&encoded)
        || admission.idempotency_key.is_empty()
        || admission.idempotency_key.len() > 1024
        || checkpoint_request_path(directory, &admission.idempotency_key).as_path() != path
        || !valid_checkpoint_catalog_id(&admission.metadata.checkpoint_id)
        || !valid_branch_digest(&admission.request_hash)
    {
        return Err(checkpoint_catalog_error(
            path,
            "unsupported, inconsistent or corrupt request admission",
        ));
    }
    Ok(admission)
}

fn read_checkpoint_request_admission(
    directory: &Path,
    key: &str,
    request_hash: &str,
) -> Result<CheckpointRequestAdmission, ControlError> {
    let admission = read_checkpoint_request_record(&checkpoint_request_path(directory, key))?;
    if admission.idempotency_key != key || admission.request_hash != request_hash {
        return Err(ControlError::Conflict(
            "checkpoint idempotency key reused with a changed request".to_owned(),
        ));
    }
    Ok(admission)
}

fn checkpoint_catalog_error(path: &Path, error: impl std::fmt::Display) -> ControlError {
    ControlError::CheckpointRetentionUnavailable(format!(
        "checkpoint catalog {}: {error}; retained files are preserved",
        path.display()
    ))
}

fn valid_checkpoint_catalog_id(id: &str) -> bool {
    let Some((nonce, tick)) = id.strip_prefix("ckpt-").and_then(|id| id.split_once("-t")) else {
        return false;
    };
    nonce.len() == 32
        && nonce
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && tick
            .parse::<u64>()
            .is_ok_and(|value| value.to_string() == tick)
}

fn checkpoint_artifact_metadata(metadata: &CheckpointMetadataDto) -> ArtifactMetadataDto {
    let filename = format!("{}.bin", metadata.checkpoint_id);
    ArtifactMetadataDto {
        artifact_id: metadata.checkpoint_id.clone(),
        filename: filename.clone(),
        content_type: "application/octet-stream".to_owned(),
        byte_size: metadata.byte_size,
        checksum_blake3: metadata.checksum_blake3.clone(),
        checksum_sha256: metadata.checksum_sha256.clone(),
        created_at_utc: metadata.created_at_utc.clone(),
        relative_path: filename,
    }
}

fn read_checkpoint_catalog(
    directory: &Path,
    id: &str,
) -> Result<CheckpointMetadataDto, ControlError> {
    let metadata = read_checkpoint_catalog_record(directory, id)?;
    if metadata.byte_size > MAX_ARTIFACT_DOWNLOAD_BYTES as u64 {
        return Err(checkpoint_catalog_error(
            &directory.join(format!("{id}{CHECKPOINT_CATALOG_SUFFIX}")),
            "unsupported, inconsistent or corrupt descriptor",
        ));
    }
    Ok(metadata)
}

/// Read the small committed descriptor before choosing a consumer's payload bound.
fn read_checkpoint_catalog_record(
    directory: &Path,
    id: &str,
) -> Result<CheckpointMetadataDto, ControlError> {
    let path = directory.join(format!("{id}{CHECKPOINT_CATALOG_SUFFIX}"));
    if !valid_checkpoint_catalog_id(id) {
        return Err(checkpoint_catalog_error(
            &path,
            "invalid checkpoint identity",
        ));
    }
    let ready_path = directory.join(format!("{id}{CHECKPOINT_CATALOG_READY_SUFFIX}"));
    read_artifact_file_bounded(&ready_path, 0)
        .map_err(|error| checkpoint_catalog_error(&ready_path, error))?;
    let bytes = read_artifact_file_bounded(&path, MAX_CHECKPOINT_CATALOG_RECORD_BYTES)
        .map_err(|error| checkpoint_catalog_error(&path, error))?;
    let record: CheckpointCatalogRecord =
        serde_json::from_slice(&bytes).map_err(|error| checkpoint_catalog_error(&path, error))?;
    let encoded = serde_json::to_vec(&record.metadata).map_err(ControlError::serialization)?;
    let metadata = record.metadata;
    if record.schema_version != 1
        || record.checksum_blake3 != compute_blake3(&encoded)
        || metadata.checkpoint_id != id
        || !id.ends_with(&format!("-t{}", metadata.tick))
        || !matches!(
            metadata.schema.as_str(),
            scriptbots_core::WORLD_CHECKPOINT_V1_SCHEMA
                | scriptbots_runtime::HOST_CHECKPOINT_V1_SCHEMA
        )
        || metadata.byte_size == 0
        || metadata
            .description
            .as_ref()
            .is_some_and(|description| description.len() > 4096)
    {
        return Err(checkpoint_catalog_error(
            &path,
            "unsupported, inconsistent or corrupt descriptor",
        ));
    }
    Ok(metadata)
}

fn verify_checkpoint_catalog_payload(
    directory: &Path,
    metadata: &CheckpointMetadataDto,
) -> Result<ArtifactMetadataDto, ControlError> {
    let artifact = checkpoint_artifact_metadata(metadata);
    let path = directory.join(&artifact.relative_path);
    let declared_bytes = usize::try_from(metadata.byte_size)
        .map_err(|error| checkpoint_catalog_error(&path, error))?;
    // Discovery charges this exact declaration against its per-call budget.
    // Refuse a larger file before reading it, including sparse or forged inputs.
    let bytes = read_artifact_file_bounded(&path, declared_bytes)
        .map_err(|error| checkpoint_catalog_error(&path, error))?;
    if bytes.len() as u64 != metadata.byte_size
        || compute_blake3(&bytes) != metadata.checksum_blake3
        || compute_sha256(&bytes) != metadata.checksum_sha256
    {
        return Err(checkpoint_catalog_error(&path, "artifact identity changed"));
    }
    let source = CatalogCheckpoint::decode(&metadata.schema, &bytes)
        .map_err(|error| checkpoint_catalog_error(&path, error))?;
    let checkpoint = source.science();
    if checkpoint.tick().0 != metadata.tick
        || checkpoint.agent_count() != metadata.agent_count
        || checkpoint.source_digest().overall != metadata.world_digest_v1
    {
        return Err(checkpoint_catalog_error(
            &path,
            "descriptor disagrees with the saved scientific cut",
        ));
    }
    Ok(artifact)
}

fn write_checkpoint_catalog(
    directory: &Path,
    metadata: &CheckpointMetadataDto,
) -> Result<(), ControlError> {
    let path = directory.join(format!(
        "{}{CHECKPOINT_CATALOG_SUFFIX}",
        metadata.checkpoint_id
    ));
    let encoded = serde_json::to_vec(metadata).map_err(ControlError::serialization)?;
    let record = CheckpointCatalogRecord {
        schema_version: 1,
        metadata: metadata.clone(),
        checksum_blake3: compute_blake3(&encoded),
    };
    let bytes = serde_json::to_vec(&record).map_err(ControlError::serialization)?;
    if bytes.len() > MAX_CHECKPOINT_CATALOG_RECORD_BYTES {
        return Err(checkpoint_catalog_error(
            &path,
            "descriptor exceeds its byte bound",
        ));
    }
    let file = match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut file) => {
            file.write_all(&bytes)
                .map_err(|error| checkpoint_catalog_error(&path, error))?;
            file
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let (file, existing) =
                open_artifact_file_bounded(&path, MAX_CHECKPOINT_CATALOG_RECORD_BYTES, true)
                    .map_err(|error| checkpoint_catalog_error(&path, error))?;
            if existing != bytes {
                return Err(checkpoint_catalog_error(
                    &path,
                    "existing descriptor is incomplete or different; retry cannot overwrite it",
                ));
            }
            file
        }
        Err(error) => return Err(checkpoint_catalog_error(&path, error)),
    };
    file.sync_all()
        .map_err(|error| checkpoint_catalog_error(&path, error))?;
    verify_open_artifact(&path, &file).map_err(|error| checkpoint_catalog_error(&path, error))?;
    #[cfg(unix)]
    sync_checkpoint_directory(directory)
        .map_err(|error| checkpoint_catalog_error(directory, error))?;
    // An empty marker is an atomic, no-overwrite publication. A crash during
    // descriptor writing leaves an undiscoverable partial file, not a poisoned
    // catalog. Keep both files immutable and never remove interrupted evidence.
    let ready_path = directory.join(format!(
        "{}{CHECKPOINT_CATALOG_READY_SUFFIX}",
        metadata.checkpoint_id
    ));
    let ready = match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&ready_path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            open_artifact_file_bounded(&ready_path, 0, true)
                .map_err(|error| checkpoint_catalog_error(&ready_path, error))?
                .0
        }
        Err(error) => return Err(checkpoint_catalog_error(&ready_path, error)),
    };
    ready
        .sync_all()
        .map_err(|error| checkpoint_catalog_error(&ready_path, error))?;
    verify_open_artifact(&ready_path, &ready)
        .map_err(|error| checkpoint_catalog_error(&ready_path, error))?;
    #[cfg(unix)]
    sync_checkpoint_directory(directory)
        .map_err(|error| checkpoint_catalog_error(directory, error))?;
    Ok(())
}

/// Hard bound on the complete checkpoint-to-current verification interval.
pub(crate) const MAX_EVENT_REVIEW_TICKS: u64 = 1_024;

pub(crate) struct EventCanvasReview {
    pub snapshot: Arc<RenderSnapshot>,
    pub checkpoint_id: String,
    pub base_tick: u64,
    pub verification: scriptbots_runtime::lab_workflows::LabReplayVerification,
}

/// Admission checks for a remote reconstruction. These bound retained inputs and
/// completed populations, not a tick's transient allocations or execution time.
#[derive(Clone, Copy)]
struct CanvasReplayLimits {
    max_agents: usize,
}

#[derive(Default)]
struct CanvasReplayPendingInputs {
    arrivals: usize,
    closed: Option<bool>,
}

impl CanvasReplayLimits {
    const REMOTE: Self = Self {
        max_agents: scriptbots_runtime::remote::REMOTE_MAX_AGENTS,
    };

    fn validate_population(self, population: usize) -> Result<(), ControlError> {
        if population > self.max_agents {
            return Err(ControlError::PayloadTooLarge(
                "checkpoint replay exceeds its agent admission bound".into(),
            ));
        }
        Ok(())
    }

    fn validate_config(
        self,
        config: &ScriptBotsConfig,
        population: usize,
    ) -> Result<(), ControlError> {
        ControlHandle::validate_checkpoint_preview_config(config, population)?;
        self.validate_population(population)?;
        if !config.closed {
            self.validate_population(config.population_minimum)?;
            if config.population_spawn_interval != 0 {
                self.validate_population(config.population_spawn_count as usize)?;
            }
        }
        Ok(())
    }

    fn validate_world(self, world: &WorldState) -> Result<(), ControlError> {
        self.validate_config(world.config(), world.agent_count())
    }

    fn validate_arrivals(
        self,
        world: &WorldState,
        pending: usize,
        additional: usize,
    ) -> Result<(), ControlError> {
        let population = world
            .agent_count()
            .checked_add(pending)
            .and_then(|population| population.checked_add(additional))
            .ok_or_else(|| {
                ControlError::PayloadTooLarge(
                    "checkpoint replay population admission overflowed".into(),
                )
            })?;
        self.validate_population(population)
    }

    fn admit_command(
        self,
        world: &WorldState,
        command: &ControlCommand,
        pending: &mut CanvasReplayPendingInputs,
    ) -> Result<(), ControlError> {
        let arrivals = match command {
            ControlCommand::SpawnAgent { .. } | ControlCommand::SpawnCrossover { .. } => 1,
            ControlCommand::Intervention(command) => {
                match &command.intervention {
                    scriptbots_core::Intervention::InjectCohort { count, .. } => {
                        let count = usize::from(*count);
                        self.validate_arrivals(world, pending.arrivals, count)?;
                        pending.arrivals += count;
                    }
                    scriptbots_core::Intervention::SetClosedWorld { closed } => {
                        pending.closed = Some(*closed);
                    }
                    _ => {}
                }
                0
            }
            _ => 0,
        };
        self.validate_arrivals(world, pending.arrivals, arrivals)
    }

    fn validate_step(
        self,
        world: &WorldState,
        pending: &CanvasReplayPendingInputs,
    ) -> Result<(), ControlError> {
        self.validate_world(world)?;
        self.validate_arrivals(world, pending.arrivals, 0)?;
        let config = world.config();
        if !pending.closed.unwrap_or(config.closed) {
            let population = world
                .agent_count()
                .checked_add(pending.arrivals)
                .map(|population| population.max(config.population_minimum))
                .ok_or_else(|| {
                    ControlError::PayloadTooLarge(
                        "checkpoint replay population admission overflowed".into(),
                    )
                })?;
            let scheduled = config.population_spawn_interval != 0
                && world.tick().0.checked_add(1).is_some_and(|tick| {
                    tick.is_multiple_of(u64::from(config.population_spawn_interval))
                });
            let population = population
                .checked_add(if scheduled {
                    config.population_spawn_count as usize
                } else {
                    0
                })
                .ok_or_else(|| {
                    ControlError::PayloadTooLarge(
                        "checkpoint replay population admission overflowed".into(),
                    )
                })?;
            self.validate_population(population)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct CheckpointRetentionState {
    limits: scriptbots_storage::CheckpointRetentionLimits,
    directories: BTreeMap<PathBuf, scriptbots_storage::ArtifactDirectoryLease>,
    requests: BTreeMap<String, RetainedCheckpoint>,
    file_count: u64,
    file_bytes: u64,
    request_record_count: u64,
    request_record_bytes: u64,
    reservation_count: u64,
    count_high_water: u64,
    bytes_high_water: u64,
    inventory_error: Option<String>,
}

impl CheckpointRetentionState {
    fn max_directory_leases(limits: scriptbots_storage::CheckpointRetentionLimits) -> u64 {
        // Keep room for the initial service root and an explicitly configured root even when
        // capture is disabled. Beyond those roots, the checkpoint count budget bounds leases.
        limits.max_count.max(2)
    }

    fn verify_directory(&self, path: &Path) -> Result<(), ControlError> {
        self.directories
            .get(path)
            .ok_or_else(|| {
                ControlError::CheckpointRetentionUnavailable(
                    "the retained artifact directory is not leased".to_owned(),
                )
            })?
            .verify()
            .map_err(|error| ControlError::CheckpointRetentionUnavailable(error.to_string()))
    }

    /// A cached receipt is usable only while its completed catalog still names
    /// the same saved cut in exactly one retained root. Payload bytes are
    /// verified separately by downloads and exact capture retries.
    fn verify_completed_catalog(
        &self,
        directory: &Path,
        expected: &CheckpointMetadataDto,
    ) -> Result<(), ControlError> {
        if self.directories.len() > MAX_CHECKPOINT_DISCOVERY_ENTRIES {
            return Err(ControlError::CheckpointCapacity(
                "checkpoint discovery directory budget exceeded".into(),
            ));
        }
        let id = &expected.checkpoint_id;
        let mut found = None;
        for root in self.directories.keys() {
            self.verify_directory(root)?;
            let path = root.join(format!("{id}{CHECKPOINT_CATALOG_READY_SUFFIX}"));
            match fs::symlink_metadata(&path) {
                Ok(_) if found.is_some() => {
                    return Err(ControlError::Conflict(format!(
                        "checkpoint {id} appears in multiple retained directories"
                    )));
                }
                Ok(_) => found = Some(root),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(checkpoint_catalog_error(&path, error)),
            }
        }
        let Some(found) = found else {
            return Err(checkpoint_catalog_error(
                &directory.join(format!("{id}{CHECKPOINT_CATALOG_READY_SUFFIX}")),
                "cached checkpoint completion marker is missing",
            ));
        };
        if found != directory {
            return Err(ControlError::Conflict(format!(
                "checkpoint {id} changed its retained artifact root"
            )));
        }
        if read_checkpoint_catalog(directory, id)? != *expected {
            return Err(ControlError::Conflict(format!(
                "checkpoint {id} has conflicting retained descriptors"
            )));
        }
        Ok(())
    }

    fn request_admission(
        &self,
        key: &str,
        request_hash: &str,
    ) -> Result<Option<(PathBuf, CheckpointRequestAdmission)>, ControlError> {
        if self.directories.len() > MAX_CHECKPOINT_DISCOVERY_ENTRIES {
            return Err(ControlError::CheckpointRetentionUnavailable(
                "checkpoint request discovery exceeds its directory bound".to_owned(),
            ));
        }
        let mut found = None;
        for directory in self.directories.keys() {
            self.verify_directory(directory)?;
            let path = checkpoint_request_path(directory, key);
            match fs::symlink_metadata(&path) {
                Ok(_) => {
                    if found.is_some() {
                        return Err(ControlError::Conflict(
                            "checkpoint idempotency key is retained in multiple artifact directories"
                                .to_owned(),
                        ));
                    }
                    found = Some(directory.clone());
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(checkpoint_catalog_error(&path, error)),
            }
        }
        found
            .map(|directory| {
                read_checkpoint_request_admission(&directory, key, request_hash)
                    .map(|admission| (directory, admission))
            })
            .transpose()
    }

    fn completed_request(
        &self,
        directory: &Path,
        admission: &CheckpointRequestAdmission,
    ) -> Result<CheckpointMetadataDto, ControlError> {
        self.verify_directory(directory)?;
        let metadata = read_checkpoint_catalog(directory, &admission.metadata.checkpoint_id)
            .map_err(|error| {
                ControlError::CheckpointRetentionUnavailable(format!(
                    "checkpoint {} completion is unconfirmed for original source session {}: {error}; retained request identity forbids a new capture",
                    admission.metadata.checkpoint_id,
                    admission.source_session.get(),
                ))
            })?;
        if metadata != admission.metadata {
            return Err(checkpoint_catalog_error(
                &checkpoint_request_path(directory, &admission.idempotency_key),
                "request admission disagrees with the completed checkpoint",
            ));
        }
        verify_checkpoint_catalog_payload(directory, &metadata)?;
        self.verify_directory(directory)?;
        Ok(metadata)
    }

    fn add_directory(&mut self, path: &Path) -> Result<PathBuf, ControlError> {
        fs::create_dir_all(path)
            .map_err(|error| ControlError::CheckpointRetentionUnavailable(error.to_string()))?;
        let path = path
            .canonicalize()
            .map_err(|error| ControlError::CheckpointRetentionUnavailable(error.to_string()))?;
        if !self.directories.contains_key(&path) {
            let limit = Self::max_directory_leases(self.limits);
            if u64::try_from(self.directories.len()).unwrap_or(u64::MAX) >= limit {
                return Err(ControlError::CheckpointCapacity(format!(
                    "artifact-directory leases={} reached limit={limit}; existing roots and artifacts are preserved; reuse a leased directory or increase the count budget",
                    self.directories.len(),
                )));
            }
            if self
                .directories
                .keys()
                .any(|known| path.starts_with(known) || known.starts_with(&path))
            {
                return Err(ControlError::Conflict(
                    "checkpoint artifact directories must not overlap".to_owned(),
                ));
            }
            let lease = scriptbots_storage::ArtifactDirectoryLease::acquire(&path)
                .map_err(|error| ControlError::CheckpointRetentionUnavailable(error.to_string()))?;
            self.directories.insert(path.clone(), lease);
        }
        self.refresh()?;
        Ok(path)
    }

    fn refresh(&mut self) -> Result<(), ControlError> {
        if self.directories.is_empty() {
            return Err(ControlError::CheckpointRetentionUnavailable(
                "no artifact directory lease was acquired".to_owned(),
            ));
        }
        fn reserve(reservations: &mut BTreeSet<PathBuf>, path: PathBuf) -> std::io::Result<()> {
            if reservations.len() >= MAX_CHECKPOINT_DISCOVERY_ENTRIES
                && !reservations.contains(&path)
            {
                return Err(std::io::Error::other(
                    "checkpoint reservations exceed their entry bound",
                ));
            }
            reservations.insert(path);
            Ok(())
        }
        fn scan(
            path: &Path,
            depth: usize,
            entries: &mut usize,
            reservations: &mut BTreeSet<PathBuf>,
        ) -> std::io::Result<(u64, u64, u64, u64)> {
            if depth > 32 {
                return Err(std::io::Error::other(
                    "checkpoint inventory exceeds its directory depth bound",
                ));
            }
            let mut count = 0_u64;
            let mut bytes = 0_u64;
            let mut request_count = 0_u64;
            let mut request_bytes = 0_u64;
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                *entries = (*entries)
                    .checked_add(1)
                    .filter(|count| *count <= MAX_CHECKPOINT_DISCOVERY_ENTRIES)
                    .ok_or_else(|| {
                        std::io::Error::other("checkpoint inventory exceeds its entry bound")
                    })?;
                let metadata = fs::symlink_metadata(entry.path())?;
                let (added_count, added_bytes, added_requests, added_request_bytes) = if metadata
                    .is_dir()
                {
                    scan(&entry.path(), depth + 1, entries, reservations)?
                } else if entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "bin")
                {
                    if !metadata.is_file() {
                        return Err(std::io::Error::other(
                            "checkpoint inventory contains a non-regular binary artifact",
                        ));
                    }
                    reserve(reservations, entry.path())?;
                    (1, metadata.len(), 0, 0)
                } else if entry.file_name().to_str().is_some_and(|name| {
                    name.starts_with(CHECKPOINT_REQUEST_PREFIX)
                        && name.ends_with(CHECKPOINT_REQUEST_SUFFIX)
                }) {
                    if !metadata.is_file()
                        || metadata.len() > MAX_CHECKPOINT_REQUEST_RECORD_BYTES as u64
                    {
                        return Err(std::io::Error::other(
                            "checkpoint inventory contains an invalid request admission",
                        ));
                    }
                    let request_path = entry.path();
                    let reservation = read_checkpoint_request_record(&request_path).map_or_else(
                        |_| request_path,
                        |admission| path.join(format!("{}.bin", admission.metadata.checkpoint_id)),
                    );
                    // Unreadable/partial records reserve their own evidence path.
                    // Only a verified identity can merge with a payload reservation.
                    reserve(reservations, reservation)?;
                    (0, 0, 1, metadata.len())
                } else {
                    (0, 0, 0, 0)
                };
                count = count
                    .checked_add(added_count)
                    .ok_or_else(|| std::io::Error::other("checkpoint count overflow"))?;
                bytes = bytes
                    .checked_add(added_bytes)
                    .ok_or_else(|| std::io::Error::other("checkpoint bytes overflow"))?;
                request_count = request_count
                    .checked_add(added_requests)
                    .ok_or_else(|| std::io::Error::other("checkpoint request count overflow"))?;
                request_bytes = request_bytes
                    .checked_add(added_request_bytes)
                    .ok_or_else(|| std::io::Error::other("checkpoint request bytes overflow"))?;
            }
            Ok((count, bytes, request_count, request_bytes))
        }
        let mut count = 0_u64;
        let mut bytes = 0_u64;
        let mut request_count = 0_u64;
        let mut request_bytes = 0_u64;
        let mut entries = 0_usize;
        let mut reservations = BTreeSet::new();
        for (path, lease) in &self.directories {
            lease
                .verify()
                .map_err(|error| ControlError::CheckpointRetentionUnavailable(error.to_string()))?;
            let (directory_count, directory_bytes, directory_requests, directory_request_bytes) =
                scan(path, 0, &mut entries, &mut reservations).map_err(|error| {
                    ControlError::CheckpointRetentionUnavailable(error.to_string())
                })?;
            count = count.checked_add(directory_count).ok_or_else(|| {
                ControlError::CheckpointRetentionUnavailable("checkpoint count overflow".to_owned())
            })?;
            bytes = bytes.checked_add(directory_bytes).ok_or_else(|| {
                ControlError::CheckpointRetentionUnavailable(
                    "checkpoint byte count overflow".to_owned(),
                )
            })?;
            request_count = request_count
                .checked_add(directory_requests)
                .ok_or_else(|| {
                    ControlError::CheckpointRetentionUnavailable(
                        "checkpoint request count overflow".to_owned(),
                    )
                })?;
            request_bytes = request_bytes
                .checked_add(directory_request_bytes)
                .ok_or_else(|| {
                    ControlError::CheckpointRetentionUnavailable(
                        "checkpoint request byte count overflow".to_owned(),
                    )
                })?;
        }
        for retained in self.requests.values() {
            reserve(
                &mut reservations,
                retained.directory.join(&retained.artifact.relative_path),
            )
            .map_err(|error| ControlError::CheckpointRetentionUnavailable(error.to_string()))?;
        }
        self.file_count = count;
        self.file_bytes = bytes;
        self.request_record_count = request_count;
        self.request_record_bytes = request_bytes;
        self.reservation_count = u64::try_from(reservations.len())
            .map_err(|error| ControlError::CheckpointRetentionUnavailable(error.to_string()))?;
        self.count_high_water = self.count_high_water.max(count);
        self.bytes_high_water = self.bytes_high_water.max(bytes);
        self.inventory_error = None;
        Ok(())
    }

    fn admit(&self, requested_bytes: u64) -> Result<(), ControlError> {
        if let Some(error) = &self.inventory_error {
            return Err(ControlError::CheckpointRetentionUnavailable(error.clone()));
        }
        if self.reservation_count >= self.limits.max_count
            || self.request_record_count >= self.limits.max_count
            || u64::try_from(self.requests.len()).unwrap_or(u64::MAX) >= self.limits.max_count
            || self
                .file_bytes
                .checked_add(requested_bytes)
                .is_none_or(|bytes| bytes > self.limits.max_bytes)
        {
            return Err(ControlError::CheckpointCapacity(format!(
                "files={} / {} bytes, reserved_checkpoints={}, request_entries={}, retained_request_records={}, requested={} bytes; limits={} files / {} bytes; preserved artifacts require additional configured capacity or a separate run/directory",
                self.file_count,
                self.file_bytes,
                self.reservation_count,
                self.requests.len(),
                self.request_record_count,
                requested_bytes,
                self.limits.max_count,
                self.limits.max_bytes,
            )));
        }
        Ok(())
    }
}

impl DataServices {
    pub fn new(artifacts_dir: PathBuf) -> Self {
        let mut retention = CheckpointRetentionState {
            limits: scriptbots_storage::CheckpointRetentionLimits::default(),
            directories: BTreeMap::new(),
            requests: BTreeMap::new(),
            file_count: 0,
            file_bytes: 0,
            request_record_count: 0,
            request_record_bytes: 0,
            reservation_count: 0,
            count_high_water: 0,
            bytes_high_water: 0,
            inventory_error: None,
        };
        let artifacts_dir = match retention.add_directory(&artifacts_dir) {
            Ok(path) => path,
            Err(error) => {
                retention.inventory_error = Some(error.to_string());
                artifacts_dir
            }
        };
        Self {
            experiments: Mutex::new(BTreeMap::new()),
            checkpoint_branches: Mutex::new(BTreeMap::new()),
            experiment_comparison: Mutex::new(()),
            experiment_shutdown_requested: std::sync::atomic::AtomicBool::new(false),
            execution_capacity: Arc::new(Mutex::new(ExperimentExecutionCapacityDto {
                max_run_slots: crate::experiment_runner::MAX_CONCURRENT_RUNS,
                reserved_run_slots: 0,
                max_grid_cells: crate::experiment_runner::MAX_CONCURRENT_GRID_CELLS,
                reserved_grid_cells: 0,
                reserved_batches: 0,
            })),
            checkpoints: Mutex::new(BTreeMap::new()),
            artifacts: Arc::new(Mutex::new(BTreeMap::new())),
            idempotency: Mutex::new(BTreeMap::new()),
            artifacts_dir: RwLock::new(artifacts_dir),
            checkpoint_retention: Mutex::new(retention),
        }
    }

    pub fn set_artifacts_dir(&self, dir: PathBuf) -> Result<(), ControlError> {
        let mut retention = self
            .checkpoint_retention
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let dir = retention.add_directory(&dir)?;
        *self.artifacts_dir.write().map_err(|_| ControlError::Lock)? = dir;
        Ok(())
    }

    pub fn artifacts_dir(&self) -> PathBuf {
        self.artifacts_dir
            .read()
            .map(|p| p.clone())
            .unwrap_or_else(|_| std::env::temp_dir().join("scriptbots_artifacts"))
    }

    /// Recover bounded admission/receipt metadata only. Completed status names a
    /// historical synced receipt; it does not assert that output is still intact.
    /// The caller holds the same registry lock used by create and shutdown.
    fn recover_checkpoint_branches_locked(
        &self,
        jobs: &mut BTreeMap<String, CheckpointBranchJob>,
        requested: Option<&str>,
    ) -> Result<usize, ControlError> {
        let roots = {
            let retention = self
                .checkpoint_retention
                .lock()
                .map_err(|_| ControlError::Lock)?;
            if retention.directories.len() > MAX_CHECKPOINT_DISCOVERY_ENTRIES {
                return Err(ControlError::ExperimentCapacity(
                    "branch catalog root budget exceeded".to_owned(),
                ));
            }
            let mut roots = Vec::with_capacity(retention.directories.len());
            for root in retention.directories.keys() {
                retention.verify_directory(root)?;
                roots.push(root.clone());
            }
            roots
        };
        let mut locations: BTreeMap<String, Vec<(PathBuf, PathBuf)>> = BTreeMap::new();
        let mut entries_seen = 0_usize;
        for root in roots {
            let parent = root.join("branches");
            let entries = match fs::read_dir(&parent) {
                Ok(entries) => entries,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(checkpoint_branch_catalog_error(&parent, error)),
            };
            validate_experiment_directory(&parent)?;
            for entry in entries.take(MAX_RETAINED_CHECKPOINT_BRANCHES + 1) {
                let entry =
                    entry.map_err(|error| checkpoint_branch_catalog_error(&parent, error))?;
                entries_seen += 1;
                // Unknown and interrupted entries also consume capacity. Reopening
                // another retained root cannot evade the service-wide admission cap.
                if entries_seen > MAX_RETAINED_CHECKPOINT_BRANCHES {
                    return Err(ControlError::ExperimentCapacity(format!(
                        "retained branch catalog exceeds {MAX_RETAINED_CHECKPOINT_BRANCHES} entries across its leased roots"
                    )));
                }
                let id = entry.file_name().into_string().map_err(|_| {
                    checkpoint_branch_catalog_error(&parent, "non-UTF-8 branch entry")
                })?;
                locations
                    .entry(id)
                    .or_default()
                    .push((root.clone(), entry.path()));
            }
        }
        let mut first_error = None;
        for (id, sources) in locations {
            if requested.is_some_and(|requested| requested != id) {
                continue;
            }
            let recovered = (|| {
                if sources.len() != 1 {
                    return Err(ControlError::Conflict(format!(
                        "checkpoint branch '{id}' occurs in multiple retained artifact roots"
                    )));
                }
                let (root, directory) = &sources[0];
                validate_experiment_directory(directory)?;
                if let Some(job) = jobs.get(&id) {
                    if &job.artifact_root != root || &job.directory != directory {
                        return Err(ControlError::Conflict(format!(
                            "checkpoint branch '{id}' changed its retained artifact root"
                        )));
                    }
                    return Ok(());
                }
                if !valid_checkpoint_branch_identifier(&id) {
                    return Err(checkpoint_branch_catalog_error(
                        directory,
                        "invalid branch identifier",
                    ));
                }
                let path = directory.join("definition.json");
                if requested.is_none()
                    && matches!(fs::symlink_metadata(&path), Err(error) if error.kind() == std::io::ErrorKind::NotFound)
                {
                    // Earlier managed studies and interrupted admission attempts
                    // have no reconstructible request identity. Preserve and count
                    // them, but never invent a catalog entry or rerun their science.
                    return Ok(());
                }
                let bytes = read_artifact_file_bounded(&path, MAX_CHECKPOINT_BRANCH_CATALOG_BYTES)
                    .map_err(|error| checkpoint_branch_catalog_error(&path, error))?;
                let record: CheckpointBranchDefinitionRecord = serde_json::from_slice(&bytes)
                    .map_err(|error| checkpoint_branch_catalog_error(&path, error))?;
                let definition = record.definition;
                definition.validate(directory)?;
                if record.checksum != definition.checksum()? {
                    return Err(checkpoint_branch_catalog_error(
                        &path,
                        "definition checksum mismatch",
                    ));
                }
                let receipt = read_checkpoint_branch_receipt(directory, &definition)?;
                let status = receipt.as_ref().map_or_else(|| {
                    let mut status = definition.status("stopped");
                    status.error_reason = Some("Execution was interrupted before a terminal receipt was sealed. Saved output is preserved; this study is not automatically rerun or resumable.".to_owned());
                    status
                }, |receipt| receipt.status.clone());
                if jobs.len() >= MAX_RETAINED_CHECKPOINT_BRANCHES {
                    return Err(ControlError::ExperimentCapacity(
                        "retained branch identity capacity reached".to_owned(),
                    ));
                }
                jobs.insert(
                    id.clone(),
                    CheckpointBranchJob {
                        definition,
                        directory: directory.clone(),
                        artifact_root: root.clone(),
                        receipt,
                        artifacts_published: false,
                        status: Arc::new(Mutex::new(status)),
                        cancel: Arc::default(),
                        worker: None,
                    },
                );
                Ok(())
            })();
            if let Err(error) = recovered {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(entries_seen), Err)
    }

    /// A cold artifact lookup restores at most the one study named by its sealed
    /// index. Whole-artifact discovery has a separate aggregate I/O admission cap.
    fn recover_checkpoint_branch_artifacts(
        &self,
        requested: Option<&str>,
    ) -> Result<(), ControlError> {
        let mut jobs = self
            .checkpoint_branches
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let mut first_error = match self.recover_checkpoint_branches_locked(&mut jobs, None) {
            Err(
                error @ (ControlError::Conflict(_)
                | ControlError::ExperimentCapacity(_)
                | ControlError::CheckpointRetentionUnavailable(_)
                | ControlError::Lock),
            ) => return Err(error),
            result => result.err(),
        };
        if let Some(artifact_id) = requested {
            let branch_id = jobs.iter().find_map(|(id, job)| {
                job.receipt
                    .as_ref()
                    .filter(|receipt| {
                        receipt
                            .artifacts
                            .iter()
                            .any(|artifact| artifact.artifact_id == artifact_id)
                    })
                    .map(|_| id.clone())
            });
            if let Some(branch_id) = branch_id {
                // A healthy companion can survive an unrelated catalog refusal,
                // but never suppress a conflict for the requested identity itself.
                self.recover_checkpoint_branches_locked(&mut jobs, Some(&branch_id))?;
            }
        }
        let mut work_bytes = 0_u64;
        for job in jobs.values_mut() {
            if let Err(error) = job.status_dto() {
                first_error.get_or_insert(error);
                continue;
            }
            let Some(receipt) = &job.receipt else {
                continue;
            };
            if requested.is_some_and(|id| {
                !receipt
                    .artifacts
                    .iter()
                    .any(|artifact| artifact.artifact_id == id)
            }) {
                continue;
            }
            if !job.artifacts_published {
                work_bytes = receipt.artifacts.iter().try_fold(work_bytes, |sum, artifact| {
                    sum.checked_add(artifact.byte_size).filter(|sum| *sum <= MAX_CHECKPOINT_BRANCH_ARTIFACT_BYTES)
                }).ok_or_else(|| ControlError::PayloadTooLarge(
                    "branch artifact discovery exceeds its aggregate verification byte budget; download a known artifact from one study instead".to_owned(),
                ))?;
            }
            match job.recover_artifacts(&self.artifacts) {
                Ok(()) if requested.is_some() => return Ok(()),
                Ok(()) => {}
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn verify_checkpoint_branch_artifact_identity(
        &self,
        artifact: &RetainedArtifact,
    ) -> Result<(), ControlError> {
        if !artifact
            .metadata
            .artifact_id
            .starts_with("checkpoint-branch-")
        {
            return Ok(());
        }
        let id = artifact
            .metadata
            .relative_path
            .split('/')
            .nth(1)
            .ok_or_else(|| {
                ControlError::InvalidPatch("branch artifact lost its owner identity".to_owned())
            })?;
        let mut jobs = self
            .checkpoint_branches
            .lock()
            .map_err(|_| ControlError::Lock)?;
        self.recover_checkpoint_branches_locked(&mut jobs, Some(id))?;
        let job = jobs
            .get(id)
            .ok_or_else(|| ControlError::NotFound(format!("checkpoint branch '{id}' not found")))?;
        if artifact.directory != job.artifact_root {
            return Err(ControlError::Conflict(
                "branch artifact changed its retained root".to_owned(),
            ));
        }
        Ok(())
    }

    fn verify_checkpoint_catalog_identity(&self, id: &str) -> Result<(), ControlError> {
        if !valid_checkpoint_catalog_id(id) {
            return Ok(());
        }
        let retention = self
            .checkpoint_retention
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let metadata = self
            .checkpoints
            .lock()
            .map_err(|_| ControlError::Lock)?
            .get(id)
            .cloned()
            .ok_or_else(|| ControlError::NotFound(format!("checkpoint '{id}' not found")))?;
        let artifact = self
            .artifacts
            .lock()
            .map_err(|_| ControlError::Lock)?
            .get(id)
            .cloned()
            .ok_or_else(|| {
                ControlError::Conflict(format!("checkpoint {id} lost its retained artifact"))
            })?;
        if artifact.metadata != checkpoint_artifact_metadata(&metadata) {
            return Err(ControlError::Conflict(format!(
                "checkpoint {id} has conflicting retained artifact metadata"
            )));
        }
        retention.verify_completed_catalog(&artifact.directory, &metadata)
    }

    /// Recover only completed descriptors in leased roots. Uncatalogued binary
    /// files still consume retention capacity but cannot imply a completed
    /// capture. Reads never record historical checkpoints in the live database.
    fn recover_checkpoint_catalogs(&self, requested: Option<&str>) -> Result<(), ControlError> {
        if requested.is_some_and(|id| !valid_checkpoint_catalog_id(id)) {
            return Ok(());
        }
        let retention = self
            .checkpoint_retention
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let mut paths = Vec::new();
        let mut entries_seen = 0_usize;
        for directory in retention.directories.keys() {
            entries_seen += 1;
            if entries_seen > MAX_CHECKPOINT_DISCOVERY_ENTRIES {
                return Err(ControlError::CheckpointCapacity(
                    "checkpoint discovery directory budget exceeded".into(),
                ));
            }
            retention.verify_directory(directory)?;
            if let Some(id) = requested {
                let path = directory.join(format!("{id}{CHECKPOINT_CATALOG_READY_SUFFIX}"));
                match fs::symlink_metadata(&path) {
                    Ok(_) => paths.push((directory.clone(), id.to_owned())),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(checkpoint_catalog_error(&path, error)),
                }
            } else {
                for entry in fs::read_dir(directory)
                    .map_err(|error| checkpoint_catalog_error(directory, error))?
                {
                    let entry =
                        entry.map_err(|error| checkpoint_catalog_error(directory, error))?;
                    entries_seen += 1;
                    if entries_seen > MAX_CHECKPOINT_DISCOVERY_ENTRIES {
                        return Err(ControlError::CheckpointCapacity(format!(
                            "checkpoint discovery exceeds {MAX_CHECKPOINT_DISCOVERY_ENTRIES} directory entries; request a checkpoint by ID"
                        )));
                    }
                    let name = entry.file_name();
                    let Some(id) = name
                        .to_str()
                        .filter(|name| name.starts_with("ckpt-"))
                        .and_then(|name| name.strip_suffix(CHECKPOINT_CATALOG_READY_SUFFIX))
                    else {
                        continue;
                    };
                    paths.push((directory.clone(), id.to_owned()));
                    if paths.len() > MAX_CATALOG_CHECKPOINTS {
                        return Err(ControlError::CheckpointCapacity(format!(
                            "checkpoint discovery exceeds {MAX_CATALOG_CHECKPOINTS} descriptors; request a saved checkpoint by ID; no files were removed"
                        )));
                    }
                }
            }
        }
        paths.sort();
        let mut verified_bytes = 0_u64;
        let mut first_error = None;
        let mut identity_counts = BTreeMap::new();
        for (_, id) in &paths {
            *identity_counts.entry(id.clone()).or_insert(0_usize) += 1;
        }
        for id in self
            .checkpoints
            .lock()
            .map_err(|_| ControlError::Lock)?
            .keys()
            .filter(|id| {
                valid_checkpoint_catalog_id(id)
                    && requested.is_none_or(|requested| requested == id.as_str())
                    && !identity_counts.contains_key(*id)
            })
        {
            first_error.get_or_insert_with(|| {
                ControlError::CheckpointRetentionUnavailable(format!(
                    "checkpoint {id} lost its completed catalog marker; cached metadata cannot establish recoverability"
                ))
            });
        }
        for (id, count) in &identity_counts {
            if *count > 1 {
                // Invalidate derived caches as well: an observed ambiguity must
                // not become an apparently successful fast-path read later.
                self.checkpoints
                    .lock()
                    .map_err(|_| ControlError::Lock)?
                    .remove(id);
                self.artifacts
                    .lock()
                    .map_err(|_| ControlError::Lock)?
                    .remove(id);
                first_error.get_or_insert_with(|| {
                    ControlError::Conflict(format!(
                        "checkpoint {id} appears in multiple retained directories"
                    ))
                });
            }
        }
        for (directory, id) in paths {
            if identity_counts[&id] > 1 {
                continue;
            }
            let recovered = (|| {
                if retention
                    .requests
                    .values()
                    .any(|request| !request.acknowledged && request.metadata.checkpoint_id == id)
                {
                    // Only the original exact retry may finish a pending capture.
                    return Ok(());
                }
                let metadata = read_checkpoint_catalog(&directory, &id)?;
                {
                    let checkpoints = self.checkpoints.lock().map_err(|_| ControlError::Lock)?;
                    let artifacts = self.artifacts.lock().map_err(|_| ControlError::Lock)?;
                    if let Some(known) = checkpoints.get(&id) {
                        if serde_json::to_vec(known).map_err(ControlError::serialization)?
                            != serde_json::to_vec(&metadata).map_err(ControlError::serialization)?
                            || artifacts
                                .get(&id)
                                .is_none_or(|artifact| artifact.directory != directory)
                        {
                            return Err(ControlError::Conflict(format!(
                                "checkpoint {id} has conflicting retained descriptors"
                            )));
                        }
                        return Ok(());
                    }
                    let uncatalogued = retention
                        .requests
                        .values()
                        .filter(|request| {
                            !checkpoints.contains_key(&request.metadata.checkpoint_id)
                        })
                        .count();
                    let retained_id = retention
                        .requests
                        .values()
                        .any(|request| request.metadata.checkpoint_id == id);
                    if !retained_id
                        && checkpoints.len().saturating_add(uncatalogued) >= MAX_CATALOG_CHECKPOINTS
                    {
                        return Err(ControlError::CheckpointCapacity(format!(
                            "checkpoint metadata reached its {MAX_CATALOG_CHECKPOINTS}-entry bound; retained files remain available in their source directories"
                        )));
                    }
                }
                verified_bytes = verified_bytes
                    .checked_add(metadata.byte_size)
                    .filter(|bytes| *bytes <= MAX_CHECKPOINT_DISCOVERY_BYTES)
                    .ok_or_else(|| ControlError::CheckpointCapacity(format!(
                        "checkpoint discovery reached its {MAX_CHECKPOINT_DISCOVERY_BYTES}-byte verification budget; retry discovery or request a checkpoint by ID"
                    )))?;
                let artifact = verify_checkpoint_catalog_payload(&directory, &metadata)?;
                retention.verify_directory(&directory)?;
                let mut checkpoints = self.checkpoints.lock().map_err(|_| ControlError::Lock)?;
                let mut artifacts = self.artifacts.lock().map_err(|_| ControlError::Lock)?;
                if artifacts.contains_key(&id) {
                    return Err(ControlError::Conflict(format!(
                        "checkpoint {id} conflicts with an existing artifact"
                    )));
                }
                checkpoints.insert(id.clone(), metadata);
                artifacts.insert(
                    id,
                    RetainedArtifact {
                        metadata: artifact,
                        directory,
                    },
                );
                Ok(())
            })();
            if let Err(error) = recovered
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Reconstitute admission only. Reads never launch a worker or rewrite status.
    fn recover_experiment(&self, experiment_id: &str) -> Result<(), ControlError> {
        if experiment_id.is_empty()
            || experiment_id.len() > 128
            || experiment_id.contains("..")
            || !experiment_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        {
            return Err(ControlError::BadRequest("invalid experiment_id".into()));
        }
        // Use the same lock order as create, including unkeyed admissions.
        let mut identities = self.idempotency.lock().map_err(|_| ControlError::Lock)?;
        let mut jobs = self.experiments.lock().map_err(|_| ControlError::Lock)?;
        if jobs.contains_key(experiment_id) {
            return Ok(());
        }
        let root = self.artifacts_dir();
        let parent = root.join("experiments");
        let directory = parent.join(experiment_id);
        match fs::symlink_metadata(&directory) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(experiment_catalog_error(&directory, e)),
            Ok(_) => {}
        }
        validate_experiment_directory(&parent)?;
        validate_experiment_directory(&directory)?;
        let path = directory.join("definition.json");
        let bytes = crate::experiment_runner::read_experiment_file_bounded(
            &path,
            MAX_EXPERIMENT_DEFINITION_BYTES,
        )
        .map_err(|e| experiment_catalog_error(&path, e))?;
        let record: ExperimentCatalogRecord =
            serde_json::from_slice(&bytes).map_err(|e| experiment_catalog_error(&path, e))?;
        let definition = record.definition;
        let payload = serde_json::to_vec(&definition).map_err(ControlError::serialization)?;
        if record.checksum != compute_blake3(&payload)
            || definition.schema_version != 1
            || definition.experiment_id != experiment_id
            || definition.cohort.cohort_id != experiment_id
            || definition.cohort.seeds.len() > 128
            || definition.variants.len() > 64
            || definition.max_ticks == 0
            || definition.created_at_utc.parse::<u64>().is_err()
        {
            return Err(experiment_catalog_error(
                &path,
                "definition checksum, version or identity mismatch",
            ));
        }
        let runner = crate::experiment_runner::MatchedSeedExperimentRunner::new(
            experiment_id,
            definition.cohort,
            definition.variants,
            definition.max_ticks,
            definition.max_concurrency,
            &directory,
        );
        if runner
            .plan_digest()
            .map_err(|e| experiment_catalog_error(&path, e))?
            != definition.plan_digest
        {
            return Err(experiment_catalog_error(
                &path,
                "saved plan/source differs from the current runner",
            ));
        }
        if let Some((key, hash)) = &definition.request_identity
            && (key.is_empty()
                || key.len() > 1024
                || hash.len() != blake3::OUT_LEN * 2
                || !hash
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                || identities
                    .get(key)
                    .is_some_and(|stored| stored != &(hash.clone(), experiment_id.to_owned())))
        {
            return Err(experiment_catalog_error(
                &path,
                "invalid or conflicting keyed admission identity",
            ));
        }
        if jobs.len() >= MAX_CATALOG_EXPERIMENTS {
            return Err(experiment_catalog_error(
                &path,
                "experiment catalog capacity reached",
            ));
        }
        if let Some((key, hash)) = definition.request_identity {
            identities.insert(key, (hash, experiment_id.to_owned()));
        }
        jobs.insert(
            experiment_id.to_owned(),
            ExperimentJob {
                runner: Arc::new(runner),
                state_file: directory.join("status.json"),
                cancel: Arc::default(),
                running: Arc::default(),
                last_error: Arc::default(),
                created_at_utc: definition.created_at_utc,
                artifact_root: root,
                artifact_generation: Mutex::new(None),
                worker: None,
            },
        );
        Ok(())
    }

    fn recover_experiment_catalog(&self) -> Result<(), ControlError> {
        let directory = self.artifacts_dir().join("experiments");
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(experiment_catalog_error(&directory, e)),
        };
        validate_experiment_directory(&directory)?;
        let mut names = Vec::new();
        for entry in entries.take(MAX_CATALOG_EXPERIMENTS + 1) {
            let entry = entry.map_err(|e| experiment_catalog_error(&directory, e))?;
            names.push(
                entry
                    .file_name()
                    .into_string()
                    .map_err(|_| experiment_catalog_error(&directory, "non-UTF-8 experiment ID"))?,
            );
        }
        if names.len() > MAX_CATALOG_EXPERIMENTS {
            return Err(experiment_catalog_error(
                &directory,
                "experiment catalog capacity exceeded",
            ));
        }
        names.sort();
        let mut first_error = None;
        for name in names {
            if let Err(error) = self.recover_experiment(&name) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Recover healthy batches even when another saved batch is unreadable.
    /// Whole-list discovery still reports the error; a direct lookup can use a
    /// successfully published companion without concealing that global refusal.
    fn recover_experiment_artifacts(&self) -> Result<(), ControlError> {
        let mut first_error = self.recover_experiment_catalog().err();
        let jobs = self.experiments.lock().map_err(|_| ControlError::Lock)?;
        for job in jobs.values() {
            if let Err(error) = job.recover_artifacts(self) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Publish only completed, bounded, verified bundles. Stage the entire slice
    /// before taking the catalog lock so a failure cannot expose a partial slice.
    fn publish_experiment_artifacts(
        artifacts: &Mutex<BTreeMap<String, RetainedArtifact>>,
        status: &crate::experiment_runner::ExperimentBatchStatus,
        root: &Path,
    ) -> Result<(), ControlError> {
        Self::publish_experiment_artifact_records(
            artifacts,
            &status.experiment_id,
            status
                .runs
                .iter()
                .filter(|run| run.state == crate::experiment_runner::RunState::Completed),
            root,
        )
        .map(|_| ())
    }

    /// Publish a selected, semantically verified run or the completed batch slice.
    /// Keep catalog identities immutable and expose no files from a failed slice.
    fn publish_experiment_artifact_records<'a>(
        artifacts: &Mutex<BTreeMap<String, RetainedArtifact>>,
        experiment_id: &str,
        runs: impl IntoIterator<Item = &'a crate::experiment_runner::RunRecord>,
        root: &Path,
    ) -> Result<Vec<ArtifactMetadataDto>, ControlError> {
        use crate::experiment_runner::{
            MAX_BUNDLE_ARTIFACT_BYTES, MAX_BUNDLE_ARTIFACTS, MAX_BUNDLE_TOTAL_ARTIFACT_BYTES,
            MAX_STATUS_BYTES, MatchedSeedExperimentRunner, RunState,
        };
        let manifest_limit = usize::try_from(MAX_STATUS_BYTES).expect("manifest limit fits usize");
        let mut staged = BTreeMap::new();
        for run in runs {
            if run.state != RunState::Completed {
                return Err(ControlError::Conflict(
                    "only completed experiment runs publish artifacts".to_owned(),
                ));
            }
            let relative_bundle = PathBuf::from("experiments")
                .join(experiment_id)
                .join(&run.run_id);
            let bundle_dir = root.join(&relative_bundle);
            scriptbots_storage::bundle::verify_run_bundle_bounded(
                &bundle_dir,
                scriptbots_storage::bundle::RunBundleVerificationLimits {
                    max_manifest_bytes: MAX_STATUS_BYTES,
                    max_artifacts: MAX_BUNDLE_ARTIFACTS,
                    max_artifact_bytes: MAX_BUNDLE_ARTIFACT_BYTES,
                    max_total_artifact_bytes: MAX_BUNDLE_TOTAL_ARTIFACT_BYTES,
                },
            )
            .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
            let manifest_path = bundle_dir.join("bundle_manifest.json");
            let manifest_bytes = read_artifact_file_bounded(&manifest_path, manifest_limit)?;
            let bundle: scriptbots_storage::RunBundleV1 =
                serde_json::from_slice(&manifest_bytes)
                    .map_err(|error| ControlError::Serialization(error.to_string()))?;
            if bundle.manifest.run_id != MatchedSeedExperimentRunner::bundle_run_id(&run.run_id)
                || bundle.manifest.experiment_id.as_deref() != Some(experiment_id)
                || bundle.manifest.variant_id.as_deref() != Some(run.variant_id.as_str())
                || bundle.manifest.root_seed != run.seed
                || bundle.digests.max_tick != run.total_ticks
                || bundle.artifacts.len() > MAX_BUNDLE_ARTIFACTS
                || bundle.bundle_version != scriptbots_storage::bundle::RUN_BUNDLE_SCHEMA_VERSION
            {
                return Err(ControlError::InvalidPatch(
                    "bundle identity changed before artifact publication".to_owned(),
                ));
            }
            let total_bytes = bundle.artifacts.iter().try_fold(0_u64, |total, entry| {
                if entry.bytes_len > MAX_BUNDLE_ARTIFACT_BYTES {
                    None
                } else {
                    total.checked_add(entry.bytes_len)
                }
            });
            if !total_bytes.is_some_and(|total| total <= MAX_BUNDLE_TOTAL_ARTIFACT_BYTES) {
                return Err(ControlError::PayloadTooLarge(
                    "bundle artifact bytes exceed the runner's publication bounds".to_owned(),
                ));
            }
            for (relative_path, expected) in std::iter::once(("bundle_manifest.json", None)).chain(
                bundle
                    .artifacts
                    .iter()
                    .map(|entry| (entry.relative_path.as_str(), Some(entry))),
            ) {
                let relative = relative_bundle.join(relative_path);
                let artifact_path = root.join(&relative);
                let canonical = artifact_path
                    .canonicalize()
                    .map_err(|error| ControlError::Serialization(error.to_string()))?;
                if !canonical.starts_with(root)
                    || canonical != artifact_path
                    || relative
                        .components()
                        .any(|component| !matches!(component, std::path::Component::Normal(_)))
                {
                    return Err(ControlError::BadRequest(
                        "bundle artifact is aliased or escapes its source root".to_owned(),
                    ));
                }
                let bytes = if let Some(entry) = expected {
                    read_artifact_file_bounded(
                        &canonical,
                        usize::try_from(entry.bytes_len)
                            .expect("bounded artifact length fits usize"),
                    )?
                } else {
                    manifest_bytes.clone()
                };
                let byte_size = bytes.len() as u64;
                let checksum_blake3 = compute_blake3(&bytes);
                if let Some(entry) = expected
                    && (entry.bytes_len != byte_size || entry.blake3_hex != checksum_blake3)
                {
                    return Err(ControlError::InvalidPatch(
                        "bundle artifact changed before publication".to_owned(),
                    ));
                }
                let relative_path = relative
                    .components()
                    .map(|component| component.as_os_str().to_str().expect("UTF-8 bundle path"))
                    .collect::<Vec<_>>()
                    .join("/");
                let artifact_id =
                    format!("experiment-{}", compute_blake3(relative_path.as_bytes()));
                let metadata = ArtifactMetadataDto {
                    artifact_id: artifact_id.clone(),
                    filename: relative
                        .file_name()
                        .expect("validated normal path")
                        .to_string_lossy()
                        .into_owned(),
                    content_type: match relative
                        .extension()
                        .and_then(|extension| extension.to_str())
                    {
                        Some("json") => "application/json",
                        Some("csv") => "text/csv",
                        _ => "application/octet-stream",
                    }
                    .to_owned(),
                    byte_size,
                    checksum_blake3,
                    checksum_sha256: compute_sha256(&bytes),
                    created_at_utc: bundle.created_at_utc.clone(),
                    relative_path,
                };
                staged.insert(
                    artifact_id,
                    RetainedArtifact {
                        metadata,
                        directory: root.to_path_buf(),
                    },
                );
            }
        }
        let mut catalog = artifacts.lock().map_err(|_| ControlError::Lock)?;
        for (id, observed) in &staged {
            if let Some(retained) = catalog.get(id)
                && (retained.directory != observed.directory
                    || retained.metadata.relative_path != observed.metadata.relative_path
                    || retained.metadata.byte_size != observed.metadata.byte_size
                    || retained.metadata.checksum_blake3 != observed.metadata.checksum_blake3
                    || retained.metadata.checksum_sha256 != observed.metadata.checksum_sha256)
            {
                return Err(ControlError::Conflict(
                    "published experiment artifact changed; original identity is retained"
                        .to_owned(),
                ));
            }
        }
        Ok(staged
            .into_iter()
            .map(|(id, observed)| catalog.entry(id).or_insert(observed).metadata.clone())
            .collect())
    }

    fn reserve_experiment_execution(
        &self,
        experiment_id: &str,
        run_slots: usize,
        grid_cells: u64,
    ) -> Result<ExperimentExecutionReservation, ControlError> {
        if self
            .experiment_shutdown_requested
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(ControlError::ExperimentShuttingDown);
        }
        if run_slots == 0 || grid_cells == 0 {
            return Err(ControlError::BadRequest(
                "experiment execution must reserve nonzero run slots and grid cells".to_owned(),
            ));
        }
        let mut capacity = self
            .execution_capacity
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let reserved_run_slots = capacity
            .reserved_run_slots
            .checked_add(run_slots)
            .filter(|count| *count <= capacity.max_run_slots)
            .ok_or_else(|| ControlError::ExperimentCapacity(format!(
                "requested_run_slots={run_slots}, reserved_run_slots={}, max_run_slots={}; no experiment worker admitted; retry after capacity is released or change SCRIPTBOTS_EXPERIMENT_RUN_SLOTS at startup",
                capacity.reserved_run_slots, capacity.max_run_slots,
            )))?;
        let reserved_grid_cells = capacity
            .reserved_grid_cells
            .checked_add(grid_cells)
            .filter(|count| *count <= capacity.max_grid_cells)
            .ok_or_else(|| ControlError::ExperimentCapacity(format!(
                "requested_grid_cells={grid_cells}, reserved_grid_cells={}, max_grid_cells={}; no experiment worker admitted; retry after capacity is released or reduce concurrent world grids",
                capacity.reserved_grid_cells, capacity.max_grid_cells,
            )))?;
        let reserved_batches = capacity.reserved_batches.checked_add(1).ok_or_else(|| {
            ControlError::ExperimentCapacity("batch reservation count overflow".to_owned())
        })?;
        let reservation = ExperimentExecutionReservation {
            capacity: Arc::clone(&self.execution_capacity),
            run_slots,
            grid_cells,
            experiment_id: experiment_id.to_owned(),
            running: None,
        };
        capacity.reserved_run_slots = reserved_run_slots;
        capacity.reserved_grid_cells = reserved_grid_cells;
        capacity.reserved_batches = reserved_batches;
        let observed = *capacity;
        drop(capacity);
        tracing::info!(
            experiment_id,
            requested_run_slots = run_slots,
            requested_grid_cells = grid_cells,
            reserved_run_slots = observed.reserved_run_slots,
            reserved_grid_cells = observed.reserved_grid_cells,
            reserved_batches = observed.reserved_batches,
            max_run_slots = observed.max_run_slots,
            "experiment execution capacity reserved"
        );
        Ok(reservation)
    }

    /// Close admission, cancel active batches and checkpoint branches, and join all workers.
    ///
    /// The existing runner finishes its current durable tick before stopping.
    /// This waits for that work and filesystem operations; it is not a hard
    /// shutdown deadline. Matched-seed continuation artifacts remain resumable by a
    /// fresh service; interrupted branch-study output is preserved without claiming
    /// resumability. The registry locks serialize admission and repeated shutdown
    /// calls; workers never acquire it while finishing or publishing artifacts.
    pub(crate) fn shutdown_experiments(&self) -> anyhow::Result<()> {
        use std::sync::atomic::Ordering;
        self.experiment_shutdown_requested
            .store(true, Ordering::Release);
        let (mut jobs, poisoned) = match self.experiments.lock() {
            Ok(jobs) => (jobs, false),
            Err(error) => (error.into_inner(), true),
        };
        let (mut branches, branches_poisoned) = match self.checkpoint_branches.lock() {
            Ok(branches) => (branches, false),
            Err(error) => (error.into_inner(), true),
        };
        for job in jobs.values() {
            job.cancel.store(true, Ordering::Release);
        }
        for branch in branches.values() {
            branch.request_cancel();
        }
        let mut failures = Vec::new();
        if poisoned {
            failures.push("experiment registry was poisoned before shutdown".to_owned());
        }
        if branches_poisoned {
            failures.push("checkpoint branch registry was poisoned before shutdown".to_owned());
        }
        for job in jobs.values_mut() {
            if let Err(error) = job.join_worker() {
                tracing::error!(%error, "experiment worker failed during shutdown");
                failures.push(error);
            }
        }
        for branch in branches.values_mut() {
            if let Err(error) = branch.join_worker() {
                tracing::error!(%error, "checkpoint branch worker failed during shutdown");
                failures.push(error);
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(anyhow::anyhow!(failures.join("; ")))
        }
    }
}

impl Default for DataServices {
    fn default() -> Self {
        let dir = std::env::var("SCRIPTBOTS_ARTIFACTS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::temp_dir().join(format!(
                    "scriptbots-artifacts-{}-{:032x}",
                    std::process::id(),
                    rand::random::<u128>()
                ))
            });
        Self::new(dir)
    }
}

impl Drop for DataServices {
    fn drop(&mut self) {
        // Workers retain their catalog and reservation state, never this owner.
        // Thus its final external release can signal every sibling before joining
        // any of them, without a worker retaining itself or joining its own thread.
        // Explicit runtime shutdown remains the fallible, caller-visible path.
        if let Err(error) = self.shutdown_experiments() {
            tracing::error!(%error, "experiment services failed during drop");
        }
    }
}

/// Shared handle used by REST, CLI, and MCP surfaces to access the running world.
#[derive(Clone)]
pub struct ControlHandle {
    host: ChannelHostPort,
    knobs_cache: KnobsCache,
    command_counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
    command_namespace: u64,
    database_path: Option<std::path::PathBuf>,
    checkpoint_writer: Option<scriptbots_storage::StorageCheckpointWriter>,
    data_services: std::sync::Arc<DataServices>,
}

fn allocate_control_command_namespace() -> u64 {
    // A continued host retains its command ledger across processes. Process-local
    // counters and wall clocks cannot identify a new control client after restart.
    // Keep the high bit clear so unkeyed commands cannot collide with keyed IDs
    // or the runtime's reserved lifecycle/bootstrap namespaces. This RNG is not
    // the world's seeded scientific stream.
    loop {
        let namespace = rand::random::<u64>() & (u64::MAX >> 1);
        if namespace != 0 {
            return namespace;
        }
    }
}

impl ControlHandle {
    pub fn new(host: ChannelHostPort) -> Self {
        Self {
            host,
            knobs_cache: std::sync::Arc::new(Mutex::new(None)),
            command_counter: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            command_namespace: allocate_control_command_namespace(),
            database_path: None,
            checkpoint_writer: None,
            data_services: std::sync::Arc::new(DataServices::default()),
        }
    }

    /// Immutable owner identity, available without requesting or advancing a snapshot.
    pub(crate) fn session_id(&self) -> scriptbots_runtime::HostSessionId {
        self.host.session_id()
    }

    /// Return a reference to the canonical data services.
    pub fn data_services(&self) -> &std::sync::Arc<DataServices> {
        &self.data_services
    }

    /// Set custom artifacts directory for data services.
    pub fn with_artifacts_dir(self, dir: std::path::PathBuf) -> Result<Self, ControlError> {
        self.data_services.set_artifacts_dir(dir)?;
        Ok(self)
    }

    pub fn with_checkpoint_retention_limits(
        self,
        limits: scriptbots_storage::CheckpointRetentionLimits,
    ) -> Result<Self, ControlError> {
        let mut retention = self
            .data_services
            .checkpoint_retention
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let directory_limit = CheckpointRetentionState::max_directory_leases(limits);
        if u64::try_from(retention.directories.len()).unwrap_or(u64::MAX) > directory_limit {
            return Err(ControlError::CheckpointCapacity(format!(
                "requested count policy allows {directory_limit} directory leases, but {} are retained; existing roots and artifacts are preserved",
                retention.directories.len(),
            )));
        }
        retention.limits = limits;
        drop(retention);
        Ok(self)
    }

    /// Bound create/resume execution shared by every clone and control transport.
    /// Zero disables new execution while retained jobs and artifacts remain readable.
    pub fn with_experiment_run_capacity(self, max_run_slots: usize) -> Result<Self, ControlError> {
        let mut capacity = self
            .data_services
            .execution_capacity
            .lock()
            .map_err(|_| ControlError::Lock)?;
        if max_run_slots < capacity.reserved_run_slots {
            return Err(ControlError::ExperimentCapacity(format!(
                "cannot reduce max_run_slots to {max_run_slots} while {} slots are reserved",
                capacity.reserved_run_slots,
            )));
        }
        capacity.max_run_slots = max_run_slots;
        drop(capacity);
        Ok(self)
    }

    /// Return the active artifacts directory.
    pub fn artifacts_dir(&self) -> std::path::PathBuf {
        self.data_services.artifacts_dir()
    }

    /// Attach a FrankenSQLite database path for offline storage queries.
    pub fn with_database(mut self, path: Option<std::path::PathBuf>) -> Self {
        self.database_path = path;
        self
    }

    /// Attach the run's storage-worker checkpoint writer.
    #[must_use]
    pub fn with_checkpoint_writer(
        mut self,
        writer: Option<scriptbots_storage::StorageCheckpointWriter>,
    ) -> Self {
        self.checkpoint_writer = writer;
        self
    }

    /// Return the attached database path, if configured.
    pub fn database_path(&self) -> Option<&std::path::Path> {
        self.database_path.as_deref()
    }

    /// Return version and runtime discovery information.
    pub fn version(&self) -> ApiVersionDto {
        ApiVersionDto {
            service_name: "scriptbots-control".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            git_commit: option_env!("SCRIPTBOTS_GIT_SHA").map(str::to_string),
            rustc_version: "rustc 2024 nightly".to_string(),
            protocol_version: 1,
            features: vec![
                #[cfg(feature = "gui")]
                "gui".into(),
                #[cfg(feature = "ml")]
                "ml".into(),
                #[cfg(feature = "brain-ft")]
                "brain-ft".into(),
                #[cfg(feature = "neuro")]
                "neuro".into(),
                "fastmcp".into(),
                "rest".into(),
            ],
            capabilities: vec![
                "simulation_control".into(),
                "config_patching".into(),
                "experiments_batch".into(),
                "experiment_comparisons".into(),
                "experiment_run_artifacts".into(),
                "experiment_failed_run_retry".into(),
                "checkpoints_v1".into(),
                "checkpoint_branch_catalog_v1".into(),
                "artifacts_storage".into(),
                "map_generation".into(),
                "narrative_search".into(),
            ],
        }
    }

    /// Return schema and route discovery information.
    pub fn schema(&self) -> ApiSchemaDto {
        ApiSchemaDto {
            openapi_version: "3.0.3".to_string(),
            title: "ScriptBots Control API".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            routes: vec![
                "/api/version".into(),
                "/api/v1/version".into(),
                "/api/schema".into(),
                "/api/v1/schema".into(),
                "/api/status".into(),
                "/api/config".into(),
                "/api/knobs".into(),
                "/api/pause".into(),
                "/api/resume".into(),
                "/api/step".into(),
                "/api/speed".into(),
                "/api/v1/experiments".into(),
                "/api/v1/experiments/{experiment_id}".into(),
                "/api/v1/experiments/{experiment_id}/compare".into(),
                "/api/v1/experiments/{experiment_id}/runs/{run_id}/artifacts".into(),
                "/api/v1/experiments/{experiment_id}/cancel".into(),
                "/api/v1/experiments/{experiment_id}/resume".into(),
                "/api/v1/experiments/{experiment_id}/retry-failed".into(),
                "/api/v1/checkpoints".into(),
                "/api/v1/checkpoints/{checkpoint_id}".into(),
                "/api/v1/checkpoints/branches".into(),
                "/api/v1/checkpoints/branches/{branch_id}".into(),
                "/api/v1/checkpoints/branches/{branch_id}/artifacts".into(),
                "/api/v1/checkpoints/branches/{branch_id}/cancel".into(),
                "/api/v1/artifacts".into(),
                "/api/v1/artifacts/{artifact_id}".into(),
                "/api/v1/artifacts/{artifact_id}/download".into(),
            ],
            mcp_tools: vec![
                "list_presets".into(),
                "apply_preset".into(),
                "list_knobs".into(),
                "get_config".into(),
                "apply_updates".into(),
                "apply_patch".into(),
                "pause".into(),
                "resume".into(),
                "step".into(),
                "set_speed".into(),
                "get_status".into(),
                "shutdown".into(),
                "get_command_status".into(),
                "map_generate".into(),
                "map_apply".into(),
                "intervene".into(),
                "narrative_search".into(),
                "narrative_around".into(),
                "get_version".into(),
                "get_schema".into(),
                "experiment_create".into(),
                "experiment_list".into(),
                "experiment_status".into(),
                "experiment_compare".into(),
                "experiment_run_artifacts".into(),
                "experiment_cancel".into(),
                "experiment_resume".into(),
                "experiment_retry_failed".into(),
                "checkpoint_create".into(),
                "checkpoint_list".into(),
                "checkpoint_status".into(),
                "checkpoint_branch_create".into(),
                "checkpoint_branch_list".into(),
                "checkpoint_branch_status".into(),
                "checkpoint_branch_cancel".into(),
                "artifact_list".into(),
                "artifact_get".into(),
            ],
            schemas: vec![
                "ApiVersionDto".into(),
                "ApiSchemaDto".into(),
                "ExperimentBatchStatusDto".into(),
                "ExperimentSummaryDto".into(),
                "ExperimentComparisonDto".into(),
                "ExperimentRunArtifactsDto".into(),
                "ExperimentComparisonRequest".into(),
                "ExperimentCreateRequest".into(),
                "ExperimentRetryFailedRequest".into(),
                "ExperimentVariantDto".into(),
                "ScenarioInterventionV1".into(),
                "CheckpointCreateRequest".into(),
                "CheckpointMetadataDto".into(),
                "CheckpointBranchRequest".into(),
                "CheckpointBranchStatusDto".into(),
                "PaginatedCheckpointBranchesResponse".into(),
                "CheckpointRetentionDto".into(),
                "ArtifactMetadataDto".into(),
                "PaginatedExperimentsResponse".into(),
                "ExperimentExecutionCapacityDto".into(),
                "PaginatedCheckpointsResponse".into(),
                "PaginatedArtifactsResponse".into(),
                "CommandStatusDto".into(),
                "SimulationStatusDto".into(),
            ],
        }
    }

    /// Create and enqueue a deterministic matched-seed experiment batch.
    pub fn create_experiment(
        &self,
        request: ExperimentCreateRequest,
    ) -> Result<ExperimentBatchStatusDto, ControlError> {
        let ticks_per_run = request.ticks_per_run.unwrap_or(1_000);
        let maximum_ticks = crate::experiment_runner::MAX_TICKS_PER_RUN;
        if !(1..=maximum_ticks).contains(&ticks_per_run) {
            return Err(ControlError::BadRequest(format!(
                "ticks_per_run must be in 1..={maximum_ticks}, found {ticks_per_run}; the requested scientific horizon is never clamped"
            )));
        }
        crate::experiment_runner::validate_experiment_workload(
            request.variants.len(),
            request.seeds.len(),
            ticks_per_run,
        )
        .map_err(|error| ControlError::BadRequest(error.to_string()))?;
        self.data_services.recover_experiment_catalog()?;
        if request.variants.is_empty() {
            return Err(ControlError::BadRequest("variants cannot be empty".into()));
        }
        if request.variants.len() > 64 {
            return Err(ControlError::BadRequest(
                "variants exceed maximum limit of 64".into(),
            ));
        }
        for v in &request.variants {
            let trimmed_id = v.variant_id.trim();
            if trimmed_id.is_empty() || trimmed_id.len() > 64 {
                return Err(ControlError::BadRequest(format!(
                    "invalid variant_id '{}': must be 1..=64 characters",
                    v.variant_id
                )));
            }
            if !trimmed_id
                .chars()
                .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
            {
                return Err(ControlError::BadRequest(format!(
                    "invalid variant_id '{}': must contain only alphanumeric, dash, or underscore",
                    v.variant_id
                )));
            }
            let fam = v.brain_family.trim().to_lowercase();
            crate::experiment_runner::validate_experiment_brain_family(trimmed_id, &fam)
                .map_err(|error| ControlError::BadRequest(error.to_string()))?;
        }
        if request.seeds.is_empty() {
            return Err(ControlError::BadRequest("seeds cannot be empty".into()));
        }
        if request.seeds.len() > 128 {
            return Err(ControlError::BadRequest(
                "seeds exceed maximum limit of 128".into(),
            ));
        }
        let request_identity = request
            .idempotency_key
            .as_ref()
            .map(|key| {
                if key.is_empty() || key.len() > 1024 {
                    return Err(ControlError::BadRequest(
                        "idempotency key must contain 1..=1024 bytes".into(),
                    ));
                }
                let payload = serde_json::to_vec(&request)
                    .map_err(|error| ControlError::Serialization(error.to_string()))?;
                Ok((key.clone(), compute_blake3(&payload)))
            })
            .transpose()?;
        // Serialize discovery and every create through publication. The worker
        // never needs this lock; its status writer has its own OS lease.
        let mut idempotency = self
            .data_services
            .idempotency
            .lock()
            .map_err(|_| ControlError::Lock)?;
        if let Some((key, payload_hash)) = request_identity.as_ref()
            && let Some((stored_hash, experiment_id)) = idempotency.get(key)
        {
            if stored_hash != payload_hash {
                return Err(ControlError::Conflict(
                    "idempotency key reused with different payload".into(),
                ));
            }
            let experiment_id = experiment_id.clone();
            drop(idempotency);
            return self.get_experiment(&experiment_id);
        }

        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);

        let experiment_id = if let Some(ref id) = request.experiment_id {
            let trimmed = id.trim();
            if trimmed.is_empty() || trimmed.len() > 128 {
                return Err(ControlError::BadRequest(
                    "experiment_id must be 1..=128 characters".into(),
                ));
            }
            if !trimmed
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
                || trimmed.contains("..")
            {
                return Err(ControlError::BadRequest(
                    "experiment_id contains invalid characters or path traversal".into(),
                ));
            }
            if let Ok(lock) = self.data_services.experiments.lock()
                && lock.contains_key(trimmed)
            {
                return Err(ControlError::Conflict(format!(
                    "experiment_id '{trimmed}' already exists"
                )));
            }
            trimmed.to_string()
        } else {
            format!("exp-{secs}{nanos:04}")
        };

        let mut variants = Vec::with_capacity(request.variants.len());
        for v in &request.variants {
            let config_overrides = match &v.config_overrides {
                None | Some(Value::Null) => BTreeMap::new(),
                Some(Value::Object(map)) => map.clone().into_iter().collect(),
                Some(_) => {
                    return Err(ControlError::BadRequest(format!(
                        "variant '{}': config_overrides must be an object of dotted config paths",
                        v.variant_id
                    )));
                }
            };
            variants.push(crate::experiment_runner::ScenarioVariant {
                variant_id: v.variant_id.trim().to_string(),
                brain_family: v.brain_family.trim().to_lowercase(),
                config_overrides,
                interventions: v.interventions.clone(),
            });
        }
        let default_concurrency = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .clamp(1, 4);
        let artifact_root = self.data_services.artifacts_dir();
        let output_dir = artifact_root.join("experiments").join(&experiment_id);
        let runner = crate::experiment_runner::MatchedSeedExperimentRunner::new(
            experiment_id.clone(),
            crate::experiment_runner::MatchedSeedCohort {
                cohort_id: experiment_id.clone(),
                seeds: request.seeds.clone(),
            },
            variants,
            ticks_per_run,
            request.max_concurrency.unwrap_or(default_concurrency),
            &output_dir,
        );
        // Refuse an invalid plan synchronously, before any thread or file exists.
        let run_slots = runner
            .plan_batch()
            .map_err(|error| ControlError::BadRequest(error.to_string()))?
            .len()
            .min(runner.max_concurrency);
        let grid_cells = runner
            .concurrent_grid_cells()
            .map_err(|error| ControlError::BadRequest(error.to_string()))?;
        let definition = ExperimentDefinitionV1 {
            schema_version: 1,
            experiment_id: experiment_id.clone(),
            cohort: runner.cohort.clone(),
            variants: runner.variants.clone(),
            max_ticks: runner.max_ticks,
            max_concurrency: runner.max_concurrency,
            plan_digest: runner
                .plan_digest()
                .map_err(|e| ControlError::BadRequest(e.to_string()))?,
            created_at_utc: secs.to_string(),
            request_identity: request_identity.clone(),
        };
        let mut job = ExperimentJob {
            runner: std::sync::Arc::new(runner),
            state_file: output_dir.join("status.json"),
            cancel: std::sync::Arc::default(),
            running: std::sync::Arc::default(),
            last_error: std::sync::Arc::default(),
            created_at_utc: format!("{secs}"),
            artifact_root,
            artifact_generation: Mutex::new(None),
            worker: None,
        };
        {
            let mut lock = self
                .data_services
                .experiments
                .lock()
                .map_err(|_| ControlError::Lock)?;
            if lock.contains_key(&experiment_id) {
                return Err(ControlError::Conflict(format!(
                    "experiment_id '{experiment_id}' already exists"
                )));
            }
            if lock.len() >= MAX_CATALOG_EXPERIMENTS {
                return Err(ControlError::PayloadTooLarge(
                    "experiment catalog capacity reached".into(),
                ));
            }
            let reservation = self.data_services.reserve_experiment_execution(
                &experiment_id,
                run_slots,
                grid_cells,
            )?;
            definition.persist(&output_dir)?;
            job.start(Arc::clone(&self.data_services.artifacts), reservation);
            lock.insert(experiment_id.clone(), job);
        }
        // Admission has happened even if the first status read fails. Publish
        // its identity before that fallible observation so an exact retry can
        // only look up this job, never start a second sweep.
        if let Some((key, payload_hash)) = request_identity {
            idempotency.insert(key, (payload_hash, experiment_id.clone()));
        }
        drop(idempotency);
        self.get_experiment(&experiment_id)
    }

    /// Look up status of an experiment by ID.
    pub fn get_experiment(
        &self,
        experiment_id: &str,
    ) -> Result<ExperimentBatchStatusDto, ControlError> {
        self.data_services.recover_experiment(experiment_id)?;
        let lock = self
            .data_services
            .experiments
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let job = lock.get(experiment_id).ok_or_else(|| {
            ControlError::NotFound(format!("experiment '{experiment_id}' not found"))
        })?;
        job.status_dto(true)
    }

    /// Inspect the exact downloadable files of one completed experiment run.
    ///
    /// A single plan-validated status binds the response. Verification reads only
    /// the selected run, so a running or corrupt companion cannot hide healthy
    /// results. No status, checkpoint or live world is changed by this operation.
    pub fn get_experiment_run_artifacts(
        &self,
        experiment_id: &str,
        run_id: &str,
    ) -> Result<ExperimentRunArtifactsDto, ControlError> {
        use crate::experiment_runner::{
            MAX_RUN_COMPONENT_BYTES, RunState, validate_path_component,
        };
        for (label, value) in [("experiment_id", experiment_id), ("run_id", run_id)] {
            if value.len() > MAX_RUN_COMPONENT_BYTES {
                return Err(ControlError::BadRequest(format!(
                    "{label} exceeds the {MAX_RUN_COMPONENT_BYTES}-byte experiment identity bound"
                )));
            }
            validate_path_component(label, value)
                .map_err(|error| ControlError::BadRequest(error.to_string()))?;
        }
        self.data_services.recover_experiment(experiment_id)?;
        let (runner, state_file, root) = {
            let jobs = self
                .data_services
                .experiments
                .lock()
                .map_err(|_| ControlError::Lock)?;
            let job = jobs.get(experiment_id).ok_or_else(|| {
                ControlError::NotFound(format!("experiment '{experiment_id}' not found"))
            })?;
            (
                Arc::clone(&job.runner),
                job.state_file.clone(),
                job.artifact_root.clone(),
            )
        };
        let status = runner
            .status_snapshot(&state_file)
            .map_err(|error| experiment_catalog_error(&state_file, error))?;
        let record = status
            .runs
            .iter()
            .find(|run| run.run_id == run_id)
            .ok_or_else(|| {
                ControlError::NotFound(format!(
                    "run '{run_id}' is not in experiment '{experiment_id}'"
                ))
            })?;
        if record.state != RunState::Completed {
            return Err(ControlError::Conflict(format!(
                "run '{run_id}' is {:?}; downloadable results require a completed run",
                record.state
            )));
        }
        runner
            .verify_completed_run(&state_file, &status, run_id)
            .map_err(|error| experiment_catalog_error(&state_file, error))?;
        // Read the observation cache only: recovering a companion checkpoint is
        // unrelated to this run's independently verified final bundle.
        let mut progress = runner
            .checkpoint_progress_snapshot(&state_file, &status, true)
            .map_err(|error| experiment_catalog_error(&state_file, error))?;
        let artifacts = DataServices::publish_experiment_artifact_records(
            &self.data_services.artifacts,
            experiment_id,
            std::iter::once(record),
            &root,
        )?;
        let total_bytes = artifacts
            .iter()
            .try_fold(0_u64, |total, artifact| {
                total.checked_add(artifact.byte_size)
            })
            .filter(|total| *total <= MAX_EXPERIMENT_RUN_TOTAL_ARTIFACT_BYTES)
            .ok_or_else(|| {
                ControlError::PayloadTooLarge(
                    "completed run inventory exceeds its aggregate byte bound".to_owned(),
                )
            })?;
        if artifacts.len() > MAX_EXPERIMENT_RUN_ARTIFACTS {
            return Err(ControlError::PayloadTooLarge(
                "completed run inventory exceeds its artifact count bound".to_owned(),
            ));
        }
        let result = ExperimentRunArtifactsDto {
            experiment_id: status.experiment_id.clone(),
            plan_digest: status.plan_digest.clone(),
            status_generation: status.generation,
            run: ExperimentRunRecordDto {
                run_id: record.run_id.clone(),
                variant_id: record.variant_id.clone(),
                brain_family: record.brain_family.clone(),
                seed: record.seed,
                state: "completed".to_owned(),
                total_ticks: record.total_ticks,
                target_ticks: runner.max_ticks,
                checkpoint_progress: progress
                    .remove(run_id)
                    .expect("progress covers every validated run"),
                final_digest: record.final_digest.clone(),
                bundle_path: record.bundle_path.clone(),
                error_reason: None,
            },
            artifacts,
            total_bytes,
        };
        if serde_json::to_vec_pretty(&result)
            .map_err(ControlError::serialization)?
            .len()
            > MAX_EXPERIMENT_RUN_ARTIFACTS_RESPONSE_BYTES
        {
            return Err(ControlError::PayloadTooLarge(
                "completed run inventory exceeds its encoded response bound".to_owned(),
            ));
        }
        Ok(result)
    }

    /// Compare one declared endpoint across two complete, distinct variant cohorts.
    ///
    /// Bundle restoration and the canonical Lab statistics run on the caller's
    /// background worker, without holding the experiment catalog lock or touching
    /// the live simulation. Each seed contributes exactly one final observation.
    /// Uses the existing Lab's default two-sided paired analysis and 95% interval;
    /// a one-pair cohort returns only its descriptive difference and provenance.
    pub fn compare_experiment_variants(
        &self,
        experiment_id: &str,
        baseline_variant: &str,
        challenger_variant: &str,
        metric: &str,
    ) -> Result<ExperimentComparisonDto, ControlError> {
        validate_experiment_comparison(
            experiment_id,
            baseline_variant,
            challenger_variant,
            metric,
        )?;
        // The slot protects only read/analysis capacity, not mutable science.
        // Recover its empty value after a panic; RAII releases it on every return.
        let _comparison = match self.data_services.experiment_comparison.try_lock() {
            Ok(permit) => permit,
            Err(std::sync::TryLockError::WouldBlock) => {
                return Err(ControlError::ExperimentComparisonBusy);
            }
            Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
        };
        self.data_services.recover_experiment(experiment_id)?;
        let (runner, state_file) = {
            let jobs = self
                .data_services
                .experiments
                .lock()
                .map_err(|_| ControlError::Lock)?;
            let job = jobs.get(experiment_id).ok_or_else(|| {
                ControlError::NotFound(format!("experiment '{experiment_id}' not found"))
            })?;
            (Arc::clone(&job.runner), job.state_file.clone())
        };
        compare_verified_variant_endpoints(
            &runner,
            &state_file,
            baseline_variant,
            challenger_variant,
            metric,
        )
    }

    /// Request cooperative cancellation at the next durable tick boundary.
    /// Unfinished runs retain their continuation cuts; unadmitted runs stay pending.
    pub fn cancel_experiment(
        &self,
        experiment_id: &str,
    ) -> Result<ExperimentBatchStatusDto, ControlError> {
        self.data_services.recover_experiment(experiment_id)?;
        let lock = self
            .data_services
            .experiments
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let job = lock.get(experiment_id).ok_or_else(|| {
            ControlError::NotFound(format!("experiment '{experiment_id}' not found"))
        })?;
        job.cancel.store(true, std::sync::atomic::Ordering::Release);
        job.status_dto(true)
    }

    /// Resume a stopped experiment with pending or checkpointed unfinished runs.
    pub fn resume_experiment(
        &self,
        experiment_id: &str,
    ) -> Result<ExperimentBatchStatusDto, ControlError> {
        self.start_retained_experiment(experiment_id, None)
    }

    /// Explicitly retry failed runs from verified continuations, retaining their
    /// original failure evidence. Completed runs are never executed again.
    pub fn retry_failed_experiment(
        &self,
        experiment_id: &str,
        expected_generation: u64,
    ) -> Result<ExperimentBatchStatusDto, ControlError> {
        self.start_retained_experiment(experiment_id, Some(expected_generation))
    }

    fn start_retained_experiment(
        &self,
        experiment_id: &str,
        retry_generation: Option<u64>,
    ) -> Result<ExperimentBatchStatusDto, ControlError> {
        self.data_services.recover_experiment(experiment_id)?;
        let mut lock = self
            .data_services
            .experiments
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let job = lock.get_mut(experiment_id).ok_or_else(|| {
            ControlError::NotFound(format!("experiment '{experiment_id}' not found"))
        })?;
        let current = job.status_dto(false)?;
        let retry_failed = retry_generation.is_some();
        if let Some(expected) = retry_generation
            && current.generation != expected
        {
            return Err(ControlError::Conflict(format!(
                "experiment '{experiment_id}' generation changed: expected {expected}, observed {}; refresh status before retrying failed runs",
                current.generation
            )));
        }
        match current.status.as_str() {
            "running" | "pending" => {
                return Err(ControlError::Conflict(format!(
                    "experiment '{experiment_id}' is already executing"
                )));
            }
            "cancelling" => {
                return Err(ControlError::Conflict(format!(
                    "experiment '{experiment_id}' is still stopping at a durable tick boundary; retry after it reports cancelled"
                )));
            }
            _ => {}
        }
        if retry_failed && current.failed_runs == 0 {
            return Err(ControlError::Conflict(format!(
                "experiment '{experiment_id}' has no failed runs to retry"
            )));
        }
        if !retry_failed && current.completed_runs + current.failed_runs == current.total_runs {
            return Err(ControlError::Conflict(format!(
                "experiment '{experiment_id}' has no pending runs to resume (status {})",
                current.status
            )));
        }
        job.runner
            .validate_execution_workload()
            .map_err(|error| ControlError::BadRequest(error.to_string()))?;
        let run_slots = current
            .runs
            .iter()
            .filter(|run| {
                matches!(run.state.as_str(), "pending" | "running")
                    || (retry_failed && run.state == "failed")
            })
            .count()
            .min(job.runner.max_concurrency);
        let grid_cells = job
            .runner
            .concurrent_grid_cells()
            .map_err(|error| ControlError::BadRequest(error.to_string()))?;
        // A cleared running flag can also mean unwind. Observe the old exit
        // before clearing diagnostics, cancellation or reserving a new wave.
        job.join_worker()
            .map_err(ControlError::ExperimentWorkerFailed)?;
        let reservation = self.data_services.reserve_experiment_execution(
            experiment_id,
            run_slots,
            grid_cells,
        )?;
        if let Some(expected_generation) = retry_generation {
            // Reserve the whole execution wave before changing retained status.
            // A refusal must leave both failure evidence and continuation cuts
            // intact; the reservation drops if validation or publication fails.
            job.runner
                .retry_failed_runs(&job.state_file, expected_generation)
                .map_err(|error| match error {
                    error @ crate::experiment_runner::ExperimentRunnerError::StaleStatusGeneration { .. } => {
                        ControlError::Conflict(error.to_string())
                    }
                    error => experiment_catalog_error(&job.state_file, error),
                })?;
        }
        job.cancel
            .store(false, std::sync::atomic::Ordering::Release);
        if let Ok(mut error) = job.last_error.lock() {
            *error = None;
        }
        job.start(Arc::clone(&self.data_services.artifacts), reservation);
        job.status_dto(true)
    }

    /// List experiments with bounded pagination.
    pub fn list_experiments(
        &self,
        limit: Option<usize>,
        cursor: Option<&str>,
    ) -> Result<PaginatedExperimentsResponse, ControlError> {
        self.data_services.recover_experiment_catalog()?;
        let limit = limit.unwrap_or(20).clamp(1, 100);
        let offset = if let Some(c) = cursor {
            c.trim()
                .parse::<usize>()
                .map_err(|_| ControlError::BadRequest(format!("invalid cursor: '{c}'")))?
        } else {
            0
        };

        let lock = self
            .data_services
            .experiments
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let total = lock.len();
        let items: Vec<ExperimentSummaryDto> = lock
            .values()
            .skip(offset)
            .take(limit)
            .map(|job| {
                // The listing omits per-run progress. Do not decode one large
                // cold checkpoint for every listed experiment only to discard it.
                job.status_dto(false).map(|e| ExperimentSummaryDto {
                    experiment_id: e.experiment_id,
                    status: e.status,
                    total_runs: e.total_runs,
                    completed_runs: e.completed_runs,
                    failed_runs: e.failed_runs,
                    created_at_utc: e.created_at_utc,
                    updated_at_utc: e.updated_at_utc,
                })
            })
            .collect::<Result<_, _>>()?;

        let has_more = offset + items.len() < total;
        let next_cursor = if has_more {
            Some((offset + items.len()).to_string())
        } else {
            None
        };

        Ok(PaginatedExperimentsResponse {
            items,
            total,
            limit,
            offset,
            next_cursor,
            has_more,
            execution_capacity: *self
                .data_services
                .execution_capacity
                .lock()
                .map_err(|_| ControlError::Lock)?,
        })
    }

    /// Capture a typed WorldCheckpointV1 directly from the scientific world owner.
    pub fn capture_checkpoint_v1(
        &self,
    ) -> Result<scriptbots_core::WorldCheckpointV1, ControlError> {
        Ok(self.host.capture_checkpoint_v1()?)
    }

    /// Create a simulation checkpoint and register it as an artifact.
    /// Retain the full owner envelope when standalone science would lose future state.
    /// Keyed retries can recover the original completed cut after catalog reopen;
    /// this does not resume or adopt the original owner or persistence session.
    pub fn create_checkpoint(
        &self,
        request: CheckpointCreateRequest,
    ) -> Result<CheckpointMetadataDto, ControlError> {
        if request
            .description
            .as_ref()
            .is_some_and(|description| description.len() > 4096)
        {
            return Err(ControlError::BadRequest(
                "checkpoint description exceeds 4096 bytes".to_owned(),
            ));
        }
        if let Some(ref key) = request.idempotency_key
            && (key.is_empty() || key.len() > 1024)
        {
            return Err(ControlError::BadRequest(
                "idempotency key must contain 1..=1024 bytes".into(),
            ));
        }
        let request_hash =
            compute_blake3(&serde_json::to_vec(&request).map_err(ControlError::serialization)?);
        let mut retention = self
            .data_services
            .checkpoint_retention
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let request_identity = request
            .idempotency_key
            .as_ref()
            .map(|key| format!("checkpoint:{key}"));
        let admission = request
            .idempotency_key
            .as_deref()
            .map(|key| retention.request_admission(key, &request_hash))
            .transpose()?
            .flatten();
        if let Some(identity) = &request_identity
            && let Some(retained) = retention.requests.get(identity).cloned()
        {
            if retained.request_hash != request_hash {
                return Err(ControlError::Conflict(
                    "checkpoint idempotency key reused with a changed request".to_owned(),
                ));
            }
            if let Some((directory, admission)) = &admission
                && (directory != &retained.directory
                    || admission.metadata != retained.metadata
                    || admission.source_session != retained.source_session)
            {
                return Err(ControlError::Conflict(
                    "checkpoint request admission disagrees with its retained source".to_owned(),
                ));
            }
            if let Some(key) = request.idempotency_key.as_deref() {
                write_checkpoint_request_admission(&retained.directory, key, &retained)?;
            }
            return self.finish_retained_checkpoint(&mut retention, identity, retained);
        }
        if let Some((directory, admission)) = admission {
            let metadata = retention.completed_request(&directory, &admission)?;
            drop(retention);
            self.data_services
                .recover_checkpoint_catalogs(Some(&metadata.checkpoint_id))?;
            return Ok(metadata);
        }
        {
            let checkpoints = self
                .data_services
                .checkpoints
                .lock()
                .map_err(|_| ControlError::Lock)?;
            let uncatalogued = retention
                .requests
                .values()
                .filter(|request| !checkpoints.contains_key(&request.metadata.checkpoint_id))
                .count();
            if checkpoints.len().saturating_add(uncatalogued) >= MAX_CATALOG_CHECKPOINTS {
                return Err(ControlError::CheckpointCapacity(format!(
                    "checkpoint metadata reached its {MAX_CATALOG_CHECKPOINTS}-entry bound; no new capture was admitted"
                )));
            }
        }
        let dir = self.data_services.artifacts_dir();
        if !retention.directories.contains_key(&dir) {
            return Err(ControlError::CheckpointRetentionUnavailable(
                "the active artifact directory is not leased".to_owned(),
            ));
        }
        retention.refresh()?;
        retention.admit(0)?;
        // The raw owner query checks both enabled features and retained hidden science.
        // Preserve its existing admission contract; only a refusal needs the stricter
        // full-owner query. Each successful cut supplies all of its own provenance.
        let (
            source,
            source_session,
            source_scientific_revision,
            source_config_revision,
            source_persistence_interval,
            source_scheduled_input_count,
        ) = match self.host.capture_checkpoint_cut() {
            Ok(captured) => (
                CatalogCheckpoint::Science(Box::new(captured.checkpoint)),
                captured.session_id,
                captured.scientific_revision,
                captured.config_revision,
                captured.persistence_interval,
                captured.scheduled_input_count,
            ),
            Err(_) => {
                let captured =
                    self.host
                        .capture_host_checkpoint_v1()
                        .map_err(|error| match error {
                            scriptbots_runtime::HostCheckpointError::Host(error) => {
                                ControlError::Host(error)
                            }
                            other => ControlError::Host(
                                scriptbots_runtime::HostAccessError::ProtocolViolation {
                                    message: other.to_string(),
                                },
                            ),
                        })?;
                let source_session = captured.session_id();
                let source_scientific_revision = captured.scientific_revision();
                let source_config_revision = captured.config_revision();
                let source_persistence_interval = captured.config().persistence_interval;
                let source_scheduled_input_count = captured
                    .remaining_schedule()
                    .and_then(|remaining| remaining.last())
                    .map_or(
                        captured.completed_schedule_sequence().unwrap_or(0),
                        |patch| patch.sequence,
                    );
                let source = match captured.validate_standalone_projection() {
                    Ok(()) => {
                        CatalogCheckpoint::Science(Box::new(captured.world_checkpoint().clone()))
                    }
                    Err(scriptbots_core::WorldCheckpointError::RequiresHostContinuity {
                        ..
                    }) => CatalogCheckpoint::Host(Box::new(captured)),
                    Err(error) => return Err(ControlError::Serialization(error.to_string())),
                };
                (
                    source,
                    source_session,
                    source_scientific_revision,
                    source_config_revision,
                    source_persistence_interval,
                    source_scheduled_input_count,
                )
            }
        };
        let checkpoint = source.science();
        let encoded_bytes = source.encode()?;
        if encoded_bytes.len() > MAX_ARTIFACT_DOWNLOAD_BYTES {
            return Err(ControlError::PayloadTooLarge(
                "checkpoint exceeds the artifact download byte bound".to_owned(),
            ));
        }
        let byte_size = u64::try_from(encoded_bytes.len())
            .map_err(|error| ControlError::Serialization(error.to_string()))?;
        retention.admit(byte_size)?;
        let tick = checkpoint.tick().0;
        let schema = source.schema().to_string();

        let blake3_hex = compute_blake3(&encoded_bytes);
        let sha256_hex = compute_sha256(&encoded_bytes);
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let checkpoint_id = format!("ckpt-{:032x}-t{tick}", rand::random::<u128>());

        let meta = CheckpointMetadataDto {
            checkpoint_id: checkpoint_id.clone(),
            tick,
            agent_count: checkpoint.agent_count(),
            world_digest_v1: checkpoint.source_digest().overall.clone(),
            byte_size,
            checksum_blake3: blake3_hex,
            checksum_sha256: sha256_hex,
            schema,
            created_at_utc: format!("{secs}"),
            description: request.description.clone(),
        };

        let art_meta = checkpoint_artifact_metadata(&meta);
        let file_path = dir.join(&art_meta.relative_path);
        let identity = request_identity.unwrap_or_else(|| checkpoint_id.clone());
        let retained = RetainedCheckpoint {
            request_hash,
            metadata: meta,
            artifact: art_meta,
            directory: dir,
            database_metadata: serde_json::json!({"source": "control_api", "description": request.description}),
            acknowledged: false,
            source_session,
            source_scientific_revision,
            source_config_revision,
            source_persistence_interval,
            source_scheduled_input_count,
        };
        if let Some(key) = request.idempotency_key.as_deref() {
            let admitted = write_checkpoint_request_admission(&retained.directory, key, &retained);
            retention.refresh()?;
            admitted?;
            retention
                .requests
                .insert(identity.clone(), retained.clone());
        }
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&file_path)
            .map_err(|error| {
                ControlError::CheckpointRetentionUnavailable(format!(
                    "checkpoint file was not reserved: {error}"
                ))
            })?;
        retention
            .requests
            .insert(identity.clone(), retained.clone());
        let written = file
            .write_all(&encoded_bytes)
            .and_then(|()| file.sync_all());
        drop(file);
        retention.refresh()?;
        written.map_err(|error| ControlError::CheckpointRetentionUnavailable(format!(
            "checkpoint {checkpoint_id} has retained partial/unconfirmed file data: {error}; the file is preserved and never overwritten"
        )))?;
        self.finish_retained_checkpoint(&mut retention, &identity, retained)
    }

    fn finish_retained_checkpoint(
        &self,
        retention: &mut CheckpointRetentionState,
        identity: &str,
        mut retained: RetainedCheckpoint,
    ) -> Result<CheckpointMetadataDto, ControlError> {
        retention.verify_directory(&retained.directory)?;
        if retained.acknowledged {
            retention.verify_completed_catalog(&retained.directory, &retained.metadata)?;
        }
        let path = retained.directory.join(&retained.artifact.relative_path);
        let (file, bytes) = open_artifact_file_bounded(
            &path,
            MAX_ARTIFACT_DOWNLOAD_BYTES,
            !retained.acknowledged,
        )
            .map_err(|error| ControlError::CheckpointRetentionUnavailable(format!(
                "checkpoint {} retained artifact could not be verified: {error}; retained data and request identity are preserved",
                retained.metadata.checkpoint_id,
            )))?;
        if bytes.len() as u64 != retained.metadata.byte_size
            || compute_blake3(&bytes) != retained.metadata.checksum_blake3
            || compute_sha256(&bytes) != retained.metadata.checksum_sha256
        {
            return Err(ControlError::CheckpointRetentionUnavailable(format!(
                "checkpoint {} has incomplete or changed retained file data; retry cannot overwrite that evidence",
                retained.metadata.checkpoint_id,
            )));
        }
        if !retained.acknowledged {
            file.sync_all()
                .map_err(|error| ControlError::CheckpointRetentionUnavailable(format!(
                    "checkpoint {} file durability is unconfirmed: {error}; retained data permits exact retry",
                    retained.metadata.checkpoint_id,
                )))?;
            // File fsync does not persist a newly created directory entry. Do
            // this before either the SQL receipt or public catalogs can succeed.
            #[cfg(unix)]
            sync_checkpoint_directory(&retained.directory)
                .map_err(|error| ControlError::CheckpointRetentionUnavailable(format!(
                    "checkpoint {} directory durability is unconfirmed: {error}; retained data permits exact retry",
                    retained.metadata.checkpoint_id,
                )))?;
            verify_open_artifact(&path, &file)
                .map_err(|error| ControlError::CheckpointRetentionUnavailable(format!(
                    "checkpoint {} retained artifact changed before publication: {error}; retained data permits exact retry",
                    retained.metadata.checkpoint_id,
                )))?;
            retention.verify_directory(&retained.directory)?;
            let checkpoint = CatalogCheckpoint::decode(&retained.metadata.schema, &bytes)?;
            if let Some(writer) = &self.checkpoint_writer {
                let recorded = match &checkpoint {
                    CatalogCheckpoint::Science(science) => writer
                        .record(
                            &retained.metadata.checkpoint_id,
                            science,
                            &retained.database_metadata,
                        )
                        .map(|_| ()),
                    CatalogCheckpoint::Host(owner) => writer
                        .record_host(
                            &retained.metadata.checkpoint_id,
                            owner,
                            &retained.database_metadata,
                        )
                        .map(|_| ()),
                };
                recorded.map_err(|error| match error {
                    scriptbots_storage::StorageError::CheckpointCapacityExceeded { .. } => {
                        ControlError::CheckpointCapacity(error.to_string())
                    }
                    other => ControlError::CheckpointRetentionUnavailable(format!(
                        "checkpoint {} persistence receipt is unconfirmed: {other}; retained file and request identity permit exact retry",
                        retained.metadata.checkpoint_id,
                    )),
                })?;
            }
            // A payload .bin can exist after an unconfirmed write. Publish its
            // immutable descriptor only after the original persistence receipt,
            // so restart discovery cannot turn an incomplete capture into one.
            write_checkpoint_catalog(&retained.directory, &retained.metadata)?;
            retention.verify_directory(&retained.directory)?;
            self.data_services
                .checkpoints
                .lock()
                .map_err(|_| ControlError::Lock)?
                .insert(
                    retained.metadata.checkpoint_id.clone(),
                    retained.metadata.clone(),
                );
            self.data_services
                .artifacts
                .lock()
                .map_err(|_| ControlError::Lock)?
                .insert(
                    retained.artifact.artifact_id.clone(),
                    RetainedArtifact {
                        metadata: retained.artifact.clone(),
                        directory: retained.directory.clone(),
                    },
                );
            retained.acknowledged = true;
            retention
                .requests
                .insert(identity.to_owned(), retained.clone());
        }
        Ok(retained.metadata)
    }

    pub fn checkpoint_retention_usage(&self) -> Result<CheckpointRetentionDto, ControlError> {
        let mut retention = self
            .data_services
            .checkpoint_retention
            .lock()
            .map_err(|_| ControlError::Lock)?;
        if let Err(error) = retention.refresh() {
            retention.inventory_error = Some(error.to_string());
        }
        let mut usage = CheckpointRetentionDto {
            max_count: retention.limits.max_count,
            max_bytes: retention.limits.max_bytes,
            max_directory_leases: CheckpointRetentionState::max_directory_leases(retention.limits),
            directory_leases: retention.directories.len(),
            file_count: retention.file_count,
            file_bytes: retention.file_bytes,
            observed_file_count_high_water: retention.count_high_water,
            observed_file_bytes_high_water: retention.bytes_high_water,
            retained_request_records: retention.request_record_count,
            retained_request_record_bytes: retention.request_record_bytes,
            reserved_checkpoint_count: retention.reservation_count,
            max_retained_request_record_bytes: retention
                .limits
                .max_count
                .saturating_mul(MAX_CHECKPOINT_REQUEST_RECORD_BYTES as u64),
            request_entries: retention.requests.len(),
            pending_entries: retention
                .requests
                .values()
                .filter(|entry| !entry.acknowledged)
                .count(),
            payload_cache_bytes: 0,
            sql_count: None,
            sql_bytes: None,
            inventory_error: retention.inventory_error.clone(),
            sql_usage_error: None,
        };
        drop(retention);
        if let Some(writer) = &self.checkpoint_writer {
            match writer.retention_usage() {
                Ok(sql) => {
                    usage.sql_count = Some(sql.count);
                    usage.sql_bytes = Some(sql.bytes);
                }
                Err(error) => usage.sql_usage_error = Some(error.to_string()),
            }
        }
        Ok(usage)
    }

    /// Look up checkpoint metadata by ID.
    pub fn get_checkpoint(
        &self,
        checkpoint_id: &str,
    ) -> Result<CheckpointMetadataDto, ControlError> {
        let cached = self
            .data_services
            .checkpoints
            .lock()
            .map_err(|_| ControlError::Lock)?
            .get(checkpoint_id)
            .cloned();
        if let Some(metadata) = cached {
            self.data_services
                .verify_checkpoint_catalog_identity(checkpoint_id)?;
            return Ok(metadata);
        }
        self.data_services
            .recover_checkpoint_catalogs(Some(checkpoint_id))?;
        let lock = self
            .data_services
            .checkpoints
            .lock()
            .map_err(|_| ControlError::Lock)?;
        lock.get(checkpoint_id).cloned().ok_or_else(|| {
            ControlError::NotFound(format!("checkpoint '{checkpoint_id}' not found"))
        })
    }

    /// List checkpoints with bounded pagination.
    pub fn list_checkpoints(
        &self,
        limit: Option<usize>,
        cursor: Option<&str>,
    ) -> Result<PaginatedCheckpointsResponse, ControlError> {
        let limit = limit.unwrap_or(20).clamp(1, 100);
        let offset = if let Some(c) = cursor {
            c.trim()
                .parse::<usize>()
                .map_err(|_| ControlError::BadRequest(format!("invalid cursor: '{c}'")))?
        } else {
            0
        };

        self.data_services.recover_checkpoint_catalogs(None)?;
        let lock = self
            .data_services
            .checkpoints
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let total = lock.len();
        let items: Vec<CheckpointMetadataDto> =
            lock.values().skip(offset).take(limit).cloned().collect();
        drop(lock);

        let has_more = offset + items.len() < total;
        let next_cursor = if has_more {
            Some((offset + items.len()).to_string())
        } else {
            None
        };

        Ok(PaginatedCheckpointsResponse {
            items,
            total,
            limit,
            offset,
            next_cursor,
            has_more,
            retention: self.checkpoint_retention_usage()?,
        })
    }

    /// Look up artifact metadata by ID.
    pub fn get_artifact(&self, artifact_id: &str) -> Result<ArtifactMetadataDto, ControlError> {
        self.retained_artifact(artifact_id)
            .map(|artifact| artifact.metadata)
    }

    fn retained_artifact(&self, artifact_id: &str) -> Result<RetainedArtifact, ControlError> {
        let cached = self
            .data_services
            .artifacts
            .lock()
            .map_err(|_| ControlError::Lock)?
            .get(artifact_id)
            .cloned();
        if let Some(artifact) = cached {
            self.data_services
                .verify_checkpoint_catalog_identity(artifact_id)?;
            self.data_services
                .verify_checkpoint_branch_artifact_identity(&artifact)?;
            return Ok(artifact);
        }
        if valid_checkpoint_catalog_id(artifact_id) {
            self.data_services
                .recover_checkpoint_catalogs(Some(artifact_id))?;
            if let Some(artifact) = self
                .data_services
                .artifacts
                .lock()
                .map_err(|_| ControlError::Lock)?
                .get(artifact_id)
                .cloned()
            {
                return Ok(artifact);
            }
        }
        // Do not hold the artifact lock while discovery takes the job locks or
        // publishes verified files. The miss path also supports cold downloads
        // when the client already knows an ID and never asks for a list.
        let branch_discovery = self
            .data_services
            .recover_checkpoint_branch_artifacts(Some(artifact_id));
        let discovery = self.data_services.recover_experiment_artifacts();
        let recovered = self
            .data_services
            .artifacts
            .lock()
            .map_err(|_| ControlError::Lock)?
            .get(artifact_id)
            .cloned();
        if let Some(artifact) = recovered {
            self.data_services
                .verify_checkpoint_branch_artifact_identity(&artifact)?;
            return Ok(artifact);
        }
        branch_discovery?;
        discovery?;
        Err(ControlError::NotFound(format!(
            "artifact '{artifact_id}' not found"
        )))
    }

    /// List discoverable artifacts with bounded pagination.
    pub fn list_artifacts(
        &self,
        limit: Option<usize>,
        cursor: Option<&str>,
    ) -> Result<PaginatedArtifactsResponse, ControlError> {
        let limit = limit.unwrap_or(20).clamp(1, 100);
        let offset = if let Some(c) = cursor {
            c.trim()
                .parse::<usize>()
                .map_err(|_| ControlError::BadRequest(format!("invalid cursor: '{c}'")))?
        } else {
            0
        };

        self.data_services.recover_checkpoint_catalogs(None)?;
        self.data_services.recover_experiment_artifacts()?;
        self.data_services
            .recover_checkpoint_branch_artifacts(None)?;
        let lock = self
            .data_services
            .artifacts
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let total = lock.len();
        let items: Vec<ArtifactMetadataDto> = lock
            .values()
            .skip(offset)
            .take(limit)
            .map(|artifact| artifact.metadata.clone())
            .collect();

        let has_more = offset + items.len() < total;
        let next_cursor = if has_more {
            Some((offset + items.len()).to_string())
        } else {
            None
        };

        Ok(PaginatedArtifactsResponse {
            items,
            total,
            limit,
            offset,
            next_cursor,
            has_more,
        })
    }

    /// Read artifact bytes with fail-closed traversal checks, size caps, and checksum verification.
    pub fn read_artifact_bytes(
        &self,
        artifact_id: &str,
    ) -> Result<(ArtifactMetadataDto, Vec<u8>), ControlError> {
        let retained = self.retained_artifact(artifact_id)?;
        let artifact = retained.metadata;
        let rel_path = &artifact.relative_path;

        if rel_path.contains("..")
            || rel_path.contains('\0')
            || rel_path.starts_with('/')
            || rel_path.starts_with('\\')
        {
            return Err(ControlError::BadRequest(
                "invalid artifact path: traversal rejected".into(),
            ));
        }

        let dir = retained.directory;
        let file_path = dir.join(rel_path);
        let canonical_file = file_path
            .canonicalize()
            .map_err(|error| ControlError::NotFound(format!("artifact file not found: {error}")))?;
        if !canonical_file.starts_with(&dir) || canonical_file != file_path {
            return Err(ControlError::BadRequest(
                "artifact path is aliased or outside its source directory".to_owned(),
            ));
        }
        let bytes = read_artifact_file_bounded(&file_path, MAX_ARTIFACT_DOWNLOAD_BYTES)?;

        if bytes.len() > MAX_ARTIFACT_DOWNLOAD_BYTES {
            return Err(ControlError::PayloadTooLarge(format!(
                "artifact byte size {} exceeds 64MB download limit",
                bytes.len()
            )));
        }

        let actual_blake3 = compute_blake3(&bytes);
        if actual_blake3 != artifact.checksum_blake3 {
            return Err(ControlError::InvalidPatch(format!(
                "artifact integrity verification failed: expected blake3 {}, got {}",
                artifact.checksum_blake3, actual_blake3
            )));
        }
        if bytes.len() as u64 != artifact.byte_size
            || compute_sha256(&bytes) != artifact.checksum_sha256
        {
            return Err(ControlError::InvalidPatch(
                "artifact byte count or SHA-256 changed".to_owned(),
            ));
        }

        Ok((artifact, bytes))
    }

    /// Copy the exact selected retained bytes into one exclusive export capsule.
    ///
    /// Every file is synced and read back before a receipt is returned. Unix also
    /// syncs the directory entries; this is not a host-journal durability receipt.
    /// Parent identity checks detect aliases/replacements but do not make an
    /// untrusted, concurrently modified directory tree an isolation boundary.
    /// Failures preserve every created file and report its containing directory.
    pub(crate) fn export_lab_artifact(
        &self,
        request_id: &str,
        session_id: scriptbots_runtime::HostSessionId,
        expected: &scriptbots_runtime::lab_workflows::LabArtifact,
        export_directory: &str,
    ) -> Result<scriptbots_runtime::lab_workflows::LabArtifactExportReceipt, ControlError> {
        use scriptbots_runtime::lab_workflows::{
            LabArtifactExportReceipt, artifact_export_request_identity_is_valid,
        };

        if session_id != self.session_id() {
            return Err(ControlError::Conflict(
                "native artifact export belongs to a different source session".to_owned(),
            ));
        }
        if !artifact_export_request_identity_is_valid(request_id) {
            return Err(ControlError::BadRequest(
                "native artifact export request identity is unsafe or exceeds its bound".to_owned(),
            ));
        }
        expected.validate().map_err(ControlError::BadRequest)?;
        if expected.byte_size > MAX_ARTIFACT_DOWNLOAD_BYTES as u64 {
            return Err(ControlError::PayloadTooLarge(
                "native artifact exceeds the export byte bound".to_owned(),
            ));
        }
        let root = Path::new(export_directory);
        if !root.is_absolute()
            || export_directory.len() > 4096
            || export_directory.chars().any(char::is_control)
            || root
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(ControlError::BadRequest(
                "native artifact export requires a bounded absolute directory without traversal"
                    .to_owned(),
            ));
        }

        // Verify the retained source and every frozen catalog field before mkdir
        // or exclusive file creation. Never reconstruct a scientific artifact.
        let (actual, bytes) = self.read_artifact_bytes(&expected.artifact_id)?;
        if actual.artifact_id != expected.artifact_id
            || actual.filename != expected.filename
            || actual.content_type != expected.content_type
            || actual.byte_size != expected.byte_size
            || actual.checksum_blake3 != expected.checksum_blake3
            || actual.checksum_sha256 != expected.checksum_sha256
            || actual.created_at_utc != expected.created_at_utc
            || actual.relative_path != expected.relative_path
        {
            return Err(ControlError::Conflict(
                "native artifact catalog changed after selection; refresh before exporting"
                    .to_owned(),
            ));
        }
        let capsule = root.join(request_id);
        let output_path = capsule.join(&expected.filename);
        let manifest_path = capsule.join("export-manifest.json");
        let path_string = |path: &Path| {
            path.to_str().map(str::to_owned).ok_or_else(|| {
                ControlError::BadRequest("native artifact export path is not UTF-8".to_owned())
            })
        };
        let receipt = LabArtifactExportReceipt {
            request_id: request_id.to_owned(),
            session_id,
            artifact: expected.clone(),
            export_directory: export_directory.to_owned(),
            output_path: path_string(&output_path)?,
            manifest_path: path_string(&manifest_path)?,
        };
        let manifest = NativeArtifactExportManifestV1 {
            schema: NATIVE_ARTIFACT_EXPORT_SCHEMA.to_owned(),
            request_id: request_id.to_owned(),
            source_session_id: session_id.get().to_string(),
            artifact: expected.clone(),
            export_directory: export_directory.to_owned(),
            output_path: receipt.output_path.clone(),
            manifest_path: receipt.manifest_path.clone(),
        };
        let manifest_bytes =
            serde_json::to_vec_pretty(&manifest).map_err(ControlError::serialization)?;
        if manifest_bytes.len() > MAX_NATIVE_ARTIFACT_EXPORT_MANIFEST_BYTES {
            return Err(ControlError::PayloadTooLarge(
                "native artifact export manifest exceeds its byte bound".to_owned(),
            ));
        }
        let (ancestor, ancestor_identity) = native_artifact_export_existing_ancestor(root)?;
        let directories = (|| -> Result<Vec<(PathBuf, fs::Metadata)>, ControlError> {
            fs::create_dir_all(root)
                .map_err(|error| ControlError::Serialization(error.to_string()))?;
            verify_native_artifact_export_directory(&ancestor, Some(&ancestor_identity))?;
            let mut directories = Vec::new();
            for directory in root
                .ancestors()
                .take_while(|path| *path != ancestor.as_path())
            {
                directories.push((
                    directory.to_path_buf(),
                    verify_native_artifact_export_directory(directory, None)?,
                ));
            }
            directories.push((ancestor, ancestor_identity));
            Ok(directories)
        })()
        .map_err(|error| preserve_native_artifact_export_failure(root, error))?;

        let result = (|| -> Result<(), ControlError> {
            // An existing capsule, even an empty one from an unconfirmed attempt,
            // is never reused. A later explicit export must have a new identity.
            fs::create_dir(&capsule).map_err(|error| ControlError::Conflict(error.to_string()))?;
            let capsule_identity = verify_native_artifact_export_directory(&capsule, None)?;
            for (directory, identity) in &directories {
                verify_native_artifact_export_directory(directory, Some(identity))?;
            }
            let payload =
                write_native_artifact_export_file(&output_path, &bytes, |file, bytes| {
                    file.write_all(bytes)
                })?;
            verify_native_artifact_export_file(
                &output_path,
                &payload,
                MAX_ARTIFACT_DOWNLOAD_BYTES,
                expected.byte_size,
                &expected.checksum_blake3,
                &expected.checksum_sha256,
            )?;
            let saved_manifest = write_native_artifact_export_file(
                &manifest_path,
                &manifest_bytes,
                |file, bytes| file.write_all(bytes),
            )?;
            verify_native_artifact_export_file(
                &manifest_path,
                &saved_manifest,
                MAX_NATIVE_ARTIFACT_EXPORT_MANIFEST_BYTES,
                manifest_bytes.len() as u64,
                &compute_blake3(&manifest_bytes),
                &compute_sha256(&manifest_bytes),
            )?;
            for (directory, identity) in std::iter::once((&capsule, &capsule_identity)).chain(
                directories
                    .iter()
                    .map(|(directory, identity)| (directory, identity)),
            ) {
                verify_native_artifact_export_directory(directory, Some(identity))?;
                #[cfg(unix)]
                sync_checkpoint_directory(directory)
                    .map_err(|error| ControlError::Serialization(error.to_string()))?;
                verify_native_artifact_export_directory(directory, Some(identity))?;
            }
            verify_open_artifact(&output_path, &payload)?;
            verify_open_artifact(&manifest_path, &saved_manifest)?;
            Ok(())
        })();
        result.map_err(|error| preserve_native_artifact_export_failure(&capsule, error))?;
        Ok(receipt)
    }

    /// Admit a finite independent future from an exact retained scientific cut.
    ///
    /// Retry identities and terminal receipts survive reopening the same artifact
    /// root. Interrupted output is preserved without rerunning or resuming science.
    /// The source owner is neither stepped nor replaced, and its session/schedule
    /// is not continued.
    pub fn create_checkpoint_branch(
        &self,
        request: CheckpointBranchRequest,
    ) -> Result<CheckpointBranchStatusDto, ControlError> {
        use crate::branch_study::{BranchStudyOrchestrator, BranchStudyPlan};

        request.validate_shape()?;
        let payload = request.bounded_payload()?;
        let request_hash = compute_blake3(&payload);
        let branch_id = format!(
            "branch-{}",
            compute_blake3(request.idempotency_key.as_bytes())
        );
        // One bounded authority covers lookup, source validation, directory claim,
        // worker admission and identity publication. Workers never acquire it.
        let mut jobs = self
            .data_services
            .checkpoint_branches
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let retained_entries = self
            .data_services
            .recover_checkpoint_branches_locked(&mut jobs, Some(&branch_id))?;
        if let Some(job) = jobs.get_mut(&branch_id) {
            if job.definition.request.idempotency_key != request.idempotency_key
                || job.definition.request_hash != request_hash
            {
                return Err(ControlError::Conflict(
                    "checkpoint branch idempotency key reused with a different request".to_owned(),
                ));
            }
            return job.status_dto();
        }
        if jobs.len() >= MAX_RETAINED_CHECKPOINT_BRANCHES
            || retained_entries >= MAX_RETAINED_CHECKPOINT_BRANCHES
        {
            return Err(ControlError::ExperimentCapacity(format!(
                "retained checkpoint branches reached {MAX_RETAINED_CHECKPOINT_BRANCHES}; existing identities and artifacts are preserved"
            )));
        }
        let checkpoint = self.verified_canvas_source(&request.checkpoint_id)?;
        let science = checkpoint.science();
        crate::branch_study::validate_study_config(science.config())
            .map_err(|error| ControlError::BadRequest(error.to_string()))?;
        if science.agent_count() as u64 > crate::branch_study::MAX_STUDY_POPULATION {
            return Err(ControlError::PayloadTooLarge(format!(
                "checkpoint population exceeds branch-study limit {}",
                crate::branch_study::MAX_STUDY_POPULATION,
            )));
        }
        let (width, height) = science
            .config()
            .food_dimensions()
            .map_err(|error| ControlError::BadRequest(error.to_string()))?;
        let reservation = self.data_services.reserve_experiment_execution(
            &branch_id,
            1,
            u64::from(width) * u64::from(height),
        )?;
        let world = Self::restore_catalog_world(&checkpoint)?;
        let preset = crate::brains::checkpoint_brain_preset(science).ok_or_else(|| {
            ControlError::BadRequest(
                "checkpoint branching requires a versioned compiled application brain preset"
                    .to_owned(),
            )
        })?;
        let base_seed = world.agent_substream_protocol_v1().root_seed();
        drop(world);
        let source_tick = science.tick().0;
        let target_tick = source_tick.checked_add(request.ticks).ok_or_else(|| {
            ControlError::BadRequest("checkpoint branch target tick overflows".to_owned())
        })?;
        let root = self.artifacts_dir();
        let directory = root.join("branches").join(&branch_id);
        let output = directory.join("study");
        let branches = checkpoint_branch_specs(&request, source_tick)?;
        let runner = BranchStudyOrchestrator::new(
            BranchStudyPlan {
                study_id: branch_id.clone(),
                description: format!(
                    "Independent checkpoint branch {} from {}",
                    request.branch_name, request.checkpoint_id
                ),
                base_seed,
                checkpoint_tick: source_tick,
                horizon_ticks: request.ticks,
                brain_preset: preset,
                config: science.config().clone(),
                source_checkpoint: None,
                branches,
            },
            output.clone(),
        );
        // Reuse the runner's source-specific restrictions, supported adapters and
        // treatment/work bounds. No reconstructed world survives this preflight.
        let source = match &checkpoint {
            CatalogCheckpoint::Science(checkpoint) => runner.validate_from_checkpoint(checkpoint),
            CatalogCheckpoint::Host(checkpoint) => runner.validate_from_host_checkpoint(checkpoint),
        }
        .map_err(|error| ControlError::BadRequest(error.to_string()))?;
        let mut retained_plan = runner.plan.clone();
        retained_plan.source_checkpoint = Some(output.join("source_checkpoint.bin"));
        let definition = CheckpointBranchDefinitionV1 {
            schema_version: 1,
            branch_id: branch_id.clone(),
            request,
            request_hash,
            source_tick,
            target_tick,
            world_digest_v1: science.source_digest().overall.clone(),
            source_byte_size: source.len() as u64,
            source_checksum_blake3: compute_blake3(&source),
            plan: retained_plan,
            created_at_utc: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or_else(
                    |_| "0".to_owned(),
                    |duration| duration.as_secs().to_string(),
                ),
        };
        drop(source);
        self.data_services
            .checkpoint_retention
            .lock()
            .map_err(|_| ControlError::Lock)?
            .verify_directory(&root)?;
        let parent = root.join("branches");
        fs::create_dir_all(&parent).map_err(|error| experiment_catalog_error(&parent, error))?;
        validate_experiment_directory(&parent)?;
        let retained = fs::read_dir(&parent)
            .map_err(|error| experiment_catalog_error(&parent, error))?
            .take(MAX_RETAINED_CHECKPOINT_BRANCHES + 1)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| experiment_catalog_error(&parent, error))?;
        if retained.len() >= MAX_RETAINED_CHECKPOINT_BRANCHES {
            return Err(ControlError::ExperimentCapacity(format!(
                "branch artifact directory reached {MAX_RETAINED_CHECKPOINT_BRANCHES} retained entries; existing output is preserved"
            )));
        }
        // Exclusive create is also the cross-process no-overwrite admission.
        fs::create_dir(&directory).map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                ControlError::Conflict(
                    "checkpoint branch output already exists; retained output is never replaced"
                        .to_owned(),
                )
            } else {
                experiment_catalog_error(&directory, error)
            }
        })?;
        validate_experiment_directory(&directory)?;
        definition.persist(&directory)?;
        let status = definition.status("running");
        let mut job = CheckpointBranchJob {
            definition,
            directory,
            artifact_root: root.clone(),
            receipt: None,
            artifacts_published: false,
            status: Arc::new(Mutex::new(status)),
            cancel: Arc::default(),
            worker: None,
        };
        job.start(
            runner,
            checkpoint,
            root,
            Arc::clone(&self.data_services.artifacts),
            reservation,
        );
        // Publish identity before the first fallible status observation.
        jobs.insert(branch_id.clone(), job);
        jobs.get_mut(&branch_id)
            .ok_or(ControlError::Lock)?
            .status_dto()
    }

    /// Observe a managed scientific fork without reading or advancing the live world.
    pub fn get_checkpoint_branch(
        &self,
        branch_id: &str,
    ) -> Result<CheckpointBranchStatusDto, ControlError> {
        if !valid_checkpoint_branch_identifier(branch_id) {
            return Err(ControlError::BadRequest(
                "invalid checkpoint branch identifier".to_owned(),
            ));
        }
        let mut jobs = self
            .data_services
            .checkpoint_branches
            .lock()
            .map_err(|_| ControlError::Lock)?;
        self.data_services
            .recover_checkpoint_branches_locked(&mut jobs, Some(branch_id))?;
        jobs.get_mut(branch_id)
            .ok_or_else(|| {
                ControlError::NotFound(format!("checkpoint branch '{branch_id}' not found"))
            })?
            .status_dto()
    }

    /// Read the requested treatment boundary from the immutable admission, including
    /// recovered studies. A receipt's display name is never a treatment authority.
    pub(crate) fn checkpoint_branch_intervention_tick(
        &self,
        branch_id: &str,
    ) -> Result<Option<u64>, ControlError> {
        if !valid_checkpoint_branch_identifier(branch_id) {
            return Err(ControlError::BadRequest(
                "invalid checkpoint branch identifier".to_owned(),
            ));
        }
        let mut jobs = self
            .data_services
            .checkpoint_branches
            .lock()
            .map_err(|_| ControlError::Lock)?;
        self.data_services
            .recover_checkpoint_branches_locked(&mut jobs, Some(branch_id))?;
        let definition = &jobs
            .get(branch_id)
            .ok_or_else(|| {
                ControlError::NotFound(format!("checkpoint branch '{branch_id}' not found"))
            })?
            .definition;
        definition
            .request
            .intervention
            .as_ref()
            .map(|_| {
                definition
                    .source_tick
                    .checked_add(definition.request.intervention_delay_ticks)
                    .filter(|tick| *tick < definition.target_tick)
                    .ok_or_else(|| {
                        ControlError::Conflict(
                            "checkpoint branch treatment boundary is outside its admitted horizon"
                                .into(),
                        )
                    })
            })
            .transpose()
    }

    /// Discover admitted studies from bounded metadata, without reading bundle
    /// payloads or advancing science. Completion is a historical terminal receipt;
    /// artifact download separately revalidates the exact retained bytes.
    pub fn list_checkpoint_branches(
        &self,
        limit: Option<usize>,
        cursor: Option<&str>,
    ) -> Result<PaginatedCheckpointBranchesResponse, ControlError> {
        let limit = limit.unwrap_or(20).clamp(1, 100);
        let offset = cursor.map_or(Ok(0), |value| {
            value
                .trim()
                .parse::<usize>()
                .map_err(|_| ControlError::BadRequest(format!("invalid cursor: '{value}'")))
        })?;
        let mut jobs = self
            .data_services
            .checkpoint_branches
            .lock()
            .map_err(|_| ControlError::Lock)?;
        self.data_services
            .recover_checkpoint_branches_locked(&mut jobs, None)?;
        let total = jobs.len();
        let items = jobs
            .values_mut()
            .skip(offset)
            .take(limit)
            .map(CheckpointBranchJob::status_dto)
            .collect::<Result<Vec<_>, _>>()?;
        let next_offset = offset.saturating_add(items.len());
        let has_more = next_offset < total;
        Ok(PaginatedCheckpointBranchesResponse {
            items,
            total,
            limit,
            offset,
            next_cursor: has_more.then(|| next_offset.to_string()),
            has_more,
        })
    }

    /// Page the actual artifacts published for one completed checkpoint study.
    ///
    /// Publication verifies every branch bundle before exposing its report. An
    /// unfinished job's files are never inferred from the filesystem. Downloads
    /// still revalidate the retained file's size, path and checksums independently.
    pub fn list_checkpoint_branch_artifacts(
        &self,
        branch_id: &str,
        limit: Option<usize>,
        cursor: Option<&str>,
    ) -> Result<PaginatedArtifactsResponse, ControlError> {
        let limit = limit.unwrap_or(20).clamp(1, 100);
        let offset = cursor.map_or(Ok(0), |value| {
            value
                .trim()
                .parse::<usize>()
                .map_err(|_| ControlError::BadRequest(format!("invalid cursor: '{value}'")))
        })?;
        let status = self.get_checkpoint_branch(branch_id)?;
        let report_id = status
            .artifact_id
            .as_deref()
            .filter(|_| status.status == "completed")
            .ok_or_else(|| {
                ControlError::Conflict(
                    "checkpoint branch artifacts are available only after verified completion"
                        .to_owned(),
                )
            })?;
        // A recovered terminal status is metadata only. Restore this study's
        // verified artifact set before taking the catalog lock for pagination.
        self.get_artifact(report_id)?;
        let prefix = format!("branches/{branch_id}/study/");
        let artifacts = self
            .data_services
            .artifacts
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let report = artifacts.get(report_id).ok_or_else(|| {
            ControlError::Conflict("completed checkpoint branch report is unavailable".to_owned())
        })?;
        if report.metadata.relative_path != format!("{prefix}study_report.json") {
            return Err(ControlError::Conflict(
                "completed checkpoint branch report has a different source identity".to_owned(),
            ));
        }
        let belongs_to_branch = |artifact: &&RetainedArtifact| {
            artifact.directory == report.directory
                && artifact.metadata.relative_path.starts_with(&prefix)
        };
        let total = artifacts.values().filter(belongs_to_branch).count();
        let items: Vec<_> = artifacts
            .values()
            .filter(belongs_to_branch)
            .skip(offset)
            .take(limit)
            .map(|artifact| artifact.metadata.clone())
            .collect();
        let next_offset = offset.saturating_add(items.len());
        let has_more = next_offset < total;
        Ok(PaginatedArtifactsResponse {
            items,
            total,
            limit,
            offset,
            next_cursor: has_more.then(|| next_offset.to_string()),
            has_more,
        })
    }

    /// Request cooperative cancellation; completion wins if it was already observed.
    /// Interrupted output is preserved, and is not advertised as a resumable owner.
    pub fn cancel_checkpoint_branch(
        &self,
        branch_id: &str,
    ) -> Result<CheckpointBranchStatusDto, ControlError> {
        if !valid_checkpoint_branch_identifier(branch_id) {
            return Err(ControlError::BadRequest(
                "invalid checkpoint branch identifier".to_owned(),
            ));
        }
        let mut jobs = self
            .data_services
            .checkpoint_branches
            .lock()
            .map_err(|_| ControlError::Lock)?;
        self.data_services
            .recover_checkpoint_branches_locked(&mut jobs, Some(branch_id))?;
        let job = jobs.get_mut(branch_id).ok_or_else(|| {
            ControlError::NotFound(format!("checkpoint branch '{branch_id}' not found"))
        })?;
        job.request_cancel();
        job.status_dto()
    }

    /// Reconstruct the selected saved cut for read-only canvas inspection.
    ///
    /// File reads and reconstruction run on the caller's background worker. This
    /// never drives or replaces the live owner, advances science, or opens SQL.
    /// Saved combinations of compiled adapters are restored in their original
    /// registry order; the core's full registry and restored-digest checks apply.
    /// This is a reconstructed science projection, not the original host publication:
    /// arena handles, per-tick activity counts and presentation history are not retained.
    pub(crate) fn checkpoint_canvas(
        &self,
        checkpoint_id: &str,
    ) -> Result<Arc<RenderSnapshot>, ControlError> {
        let checkpoint = self.verified_canvas_source(checkpoint_id)?;
        self.project_canvas_world(Self::restore_catalog_world(&checkpoint)?)
    }

    /// Project saved science for remote review with explicit resource admission bounds.
    ///
    /// Descriptor discovery is exact-ID and does not run the general catalog recovery:
    /// a cold request must apply the 16 MiB source bound before decoding any payload.
    /// The one verified payload read supplies both metadata validation and restoration.
    pub(crate) fn checkpoint_canvas_bounded(
        &self,
        checkpoint_id: &str,
    ) -> Result<(CheckpointMetadataDto, Arc<RenderSnapshot>), ControlError> {
        let (metadata, checkpoint, _) = self.read_bounded_canvas_checkpoint(checkpoint_id)?;
        let snapshot = self.project_canvas_world(Self::restore_catalog_world(&checkpoint)?)?;
        Ok((metadata, snapshot))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one retained-directory lease covers source discovery, verification and revalidation"
    )]
    fn read_bounded_canvas_checkpoint(
        &self,
        checkpoint_id: &str,
    ) -> Result<(CheckpointMetadataDto, CatalogCheckpoint, PathBuf), ControlError> {
        if checkpoint_id.len() > MAX_CHECKPOINT_BRANCH_ID_BYTES
            || !valid_checkpoint_catalog_id(checkpoint_id)
        {
            return Err(ControlError::NotFound("checkpoint not found".into()));
        }
        let retention = self
            .data_services
            .checkpoint_retention
            .lock()
            .map_err(|_| ControlError::Lock)?;
        if retention.directories.len() > MAX_CHECKPOINT_DISCOVERY_ENTRIES {
            return Err(ControlError::CheckpointCapacity(
                "checkpoint discovery directory budget exceeded".into(),
            ));
        }
        if retention
            .requests
            .values()
            .any(|request| !request.acknowledged && request.metadata.checkpoint_id == checkpoint_id)
        {
            return Err(ControlError::NotFound(
                "checkpoint completion is pending; retry its original capture request".into(),
            ));
        }
        let find_directory = || {
            let mut found = None;
            for directory in retention.directories.keys() {
                retention.verify_directory(directory)?;
                let ready =
                    directory.join(format!("{checkpoint_id}{CHECKPOINT_CATALOG_READY_SUFFIX}"));
                match fs::symlink_metadata(&ready) {
                    Ok(_) if found.is_some() => {
                        return Err(ControlError::Conflict(format!(
                            "checkpoint {checkpoint_id} appears in multiple retained directories"
                        )));
                    }
                    Ok(_) => found = Some(directory.clone()),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(checkpoint_catalog_error(&ready, error)),
                }
            }
            found.ok_or_else(|| {
                ControlError::NotFound(format!("checkpoint '{checkpoint_id}' not found"))
            })
        };
        let directory = find_directory()?;
        let metadata = read_checkpoint_catalog_record(&directory, checkpoint_id)?;
        if metadata.byte_size > MAX_CHECKPOINT_PREVIEW_BYTES as u64
            || metadata.agent_count > scriptbots_runtime::remote::REMOTE_MAX_AGENTS
        {
            return Err(ControlError::PayloadTooLarge(
                "checkpoint preview exceeds its source-byte or agent admission bound".into(),
            ));
        }
        let artifact = checkpoint_artifact_metadata(&metadata);
        let validate_cached_identity = || {
            let checkpoints = self
                .data_services
                .checkpoints
                .lock()
                .map_err(|_| ControlError::Lock)?;
            let artifacts = self
                .data_services
                .artifacts
                .lock()
                .map_err(|_| ControlError::Lock)?;
            match (checkpoints.get(checkpoint_id), artifacts.get(checkpoint_id)) {
                (None, None) => Ok(()),
                (Some(known), Some(retained))
                    if known == &metadata
                        && retained.directory == directory
                        && retained.metadata == artifact =>
                {
                    Ok(())
                }
                _ => Err(ControlError::Conflict(format!(
                    "checkpoint {checkpoint_id} has conflicting retained descriptors"
                ))),
            }
        };
        validate_cached_identity()?;
        let path = directory.join(&artifact.relative_path);
        let bytes = read_artifact_file_bounded(&path, MAX_CHECKPOINT_PREVIEW_BYTES)?;
        if bytes.len() as u64 != metadata.byte_size
            || compute_blake3(&bytes) != metadata.checksum_blake3
            || compute_sha256(&bytes) != metadata.checksum_sha256
        {
            return Err(ControlError::InvalidPatch(
                "artifact byte count or checksum changed".into(),
            ));
        }
        let checkpoint = Self::decode_canvas_checkpoint(&metadata, &artifact, &bytes)?;
        Self::validate_checkpoint_preview_config(
            checkpoint.science().config(),
            checkpoint.science().agent_count(),
        )?;
        if find_directory()? != directory
            || read_checkpoint_catalog_record(&directory, checkpoint_id)? != metadata
        {
            return Err(ControlError::Conflict(
                "checkpoint descriptor changed during preview verification".into(),
            ));
        }
        validate_cached_identity()?;
        Ok((metadata, checkpoint, directory))
    }

    fn validate_checkpoint_preview_config(
        config: &ScriptBotsConfig,
        agent_count: usize,
    ) -> Result<(), ControlError> {
        let (width, height) = config
            .food_dimensions()
            .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
        if agent_count > scriptbots_runtime::remote::REMOTE_MAX_AGENTS
            || u64::from(width) * u64::from(height) > MAX_CHECKPOINT_PREVIEW_GRID_CELLS
            || config.history_capacity > MAX_CHECKPOINT_PREVIEW_HISTORY
            || config.narrative_capacity > MAX_CHECKPOINT_PREVIEW_HISTORY
        {
            return Err(ControlError::PayloadTooLarge(
                "checkpoint preview exceeds its agent, grid or history admission bound".into(),
            ));
        }
        Ok(())
    }

    /// Read the raw science component only when it can stand alone without
    /// dropping auxiliary state retained by a complete source.
    #[cfg(test)]
    fn verified_canvas_checkpoint(
        &self,
        checkpoint_id: &str,
    ) -> Result<scriptbots_core::WorldCheckpointV1, ControlError> {
        self.verified_canvas_source(checkpoint_id)?
            .standalone_science()
    }

    fn verified_canvas_source(
        &self,
        checkpoint_id: &str,
    ) -> Result<CatalogCheckpoint, ControlError> {
        let metadata = self.get_checkpoint(checkpoint_id)?;
        let (artifact, bytes) = self.read_artifact_bytes(checkpoint_id)?;
        Self::decode_canvas_checkpoint(&metadata, &artifact, &bytes)
    }

    fn decode_canvas_checkpoint(
        metadata: &CheckpointMetadataDto,
        artifact: &ArtifactMetadataDto,
        bytes: &[u8],
    ) -> Result<CatalogCheckpoint, ControlError> {
        let checkpoint = CatalogCheckpoint::decode(&metadata.schema, bytes)?;
        let science = checkpoint.science();
        if metadata.tick != science.tick().0
            || metadata.agent_count != science.agent_count()
            || metadata.world_digest_v1 != science.source_digest().overall
            || metadata.byte_size != artifact.byte_size
            || metadata.checksum_blake3 != artifact.checksum_blake3
            || metadata.checksum_sha256 != artifact.checksum_sha256
        {
            return Err(ControlError::InvalidPatch(
                "checkpoint catalog and verified artifact disagree".into(),
            ));
        }
        Ok(checkpoint)
    }

    fn restore_catalog_world(checkpoint: &CatalogCheckpoint) -> Result<WorldState, ControlError> {
        match checkpoint {
            CatalogCheckpoint::Science(science) => Self::restore_canvas_world(science),
            CatalogCheckpoint::Host(owner) => {
                let science = owner.world_checkpoint();
                let registry = crate::brains::create_brain_registry_for_checkpoint(science)
                    .map_err(|error| {
                        ControlError::BadRequest(format!(
                            "checkpoint registry does not match the compiled application adapters: {error}"
                        ))
                    })?;
                let world = owner
                    .restore_science_projection(registry)
                    .map_err(|error| ControlError::Serialization(error.to_string()))?;
                let actual_digest = world
                    .world_digest_v1()
                    .map_err(|error| ControlError::Serialization(error.to_string()))?;
                if &actual_digest != science.source_digest() {
                    return Err(ControlError::InvalidPatch(
                        "reconstructed checkpoint scientific digest differs from its saved cut"
                            .into(),
                    ));
                }
                Ok(world)
            }
        }
    }

    fn restore_canvas_world(
        checkpoint: &scriptbots_core::WorldCheckpointV1,
    ) -> Result<WorldState, ControlError> {
        Self::restore_canvas_world_parts(checkpoint, None)
    }

    fn restore_canvas_world_parts(
        checkpoint: &scriptbots_core::WorldCheckpointV1,
        continuity: Option<&scriptbots_core::WorldHostCheckpointV1>,
    ) -> Result<WorldState, ControlError> {
        Self::restore_canvas_world_and_preset(checkpoint, continuity).map(|(world, _)| world)
    }

    fn restore_canvas_world_and_preset(
        checkpoint: &scriptbots_core::WorldCheckpointV1,
        continuity: Option<&scriptbots_core::WorldHostCheckpointV1>,
    ) -> Result<(WorldState, Option<crate::BrainPreset>), ControlError> {
        let registry =
            crate::brains::create_brain_registry_for_checkpoint(checkpoint).map_err(|error| {
                ControlError::BadRequest(format!(
                    "checkpoint registry does not match the compiled application adapters: {error}"
                ))
            })?;
        let world = match continuity {
            Some(continuity) => {
                WorldState::restore_host_checkpoint_v1(checkpoint, continuity, registry)
            }
            None => WorldState::restore_checkpoint_v1(checkpoint, registry),
        }
        .map_err(|error| match error {
            scriptbots_core::WorldCheckpointError::RegistryMismatch { .. } => {
                ControlError::BadRequest(format!(
                    "checkpoint registry does not match the compiled application adapters: {error}"
                ))
            }
            _ => ControlError::Serialization(error.to_string()),
        })?;
        let preset = crate::brains::checkpoint_brain_preset(checkpoint);
        let actual_digest = world
            .world_digest_v1()
            .map_err(|error| ControlError::Serialization(error.to_string()))?;
        if &actual_digest != checkpoint.source_digest() {
            return Err(ControlError::InvalidPatch(
                "reconstructed checkpoint scientific digest differs from its saved cut".into(),
            ));
        }
        Ok((world, preset))
    }

    fn project_canvas_world(&self, world: WorldState) -> Result<Arc<RenderSnapshot>, ControlError> {
        let projection = scriptbots_runtime::HostCore::new(
            self.host.session_id(),
            world,
            scriptbots_runtime::HostCoreOptions {
                initial_playback: scriptbots_runtime::PlaybackSnapshot {
                    paused: true,
                    speed_multiplier: 1.0,
                },
                capture_agent_visuals: true,
                ..scriptbots_runtime::HostCoreOptions::default()
            },
        )
        .map_err(|error| ControlError::Serialization(error.to_string()))?;
        Ok(projection.latest_snapshot())
    }

    /// Review a caller-selected checkpoint and target without choosing another source.
    ///
    /// Exact saved cuts need no current-owner replay history. Later targets require
    /// this owner's acknowledged cut and the same complete input/current-digest
    /// checks used for narrative review. Neither path changes the live owner.
    pub(crate) fn replay_canvas_from_checkpoint(
        &self,
        session_id: scriptbots_runtime::HostSessionId,
        expected: &scriptbots_runtime::lab_workflows::LabCheckpoint,
        target_tick: u64,
    ) -> Result<EventCanvasReview, ControlError> {
        if session_id != self.session_id() {
            return Err(ControlError::Conflict(
                "native replay review belongs to a different owner session".into(),
            ));
        }
        if expected.checkpoint_id.len() > MAX_CHECKPOINT_BRANCH_ID_BYTES
            || !valid_checkpoint_catalog_id(&expected.checkpoint_id)
        {
            return Err(ControlError::BadRequest(
                "native replay review checkpoint identity is invalid".into(),
            ));
        }
        if target_tick < expected.tick {
            return Err(ControlError::BadRequest(
                "review target precedes the selected checkpoint".into(),
            ));
        }
        let metadata = self.get_checkpoint(&expected.checkpoint_id)?;
        if metadata.checkpoint_id != expected.checkpoint_id
            || metadata.tick != expected.tick
            || metadata.agent_count != expected.agent_count
            || metadata.byte_size != expected.byte_size
            || metadata.world_digest_v1 != expected.world_digest
        {
            return Err(ControlError::Conflict(
                "selected checkpoint metadata changed; refresh before reviewing".into(),
            ));
        }

        let retained = self.retained_checkpoint_for_replay(session_id, &metadata, target_tick)?;
        if let Some(retained) = &retained {
            let artifact = self.retained_artifact(&metadata.checkpoint_id)?;
            if artifact.directory != retained.directory || artifact.metadata != retained.artifact {
                return Err(ControlError::Conflict(
                    "selected checkpoint changed its retained source directory or artifact".into(),
                ));
            }
        }
        let (artifact, bytes) = self.read_artifact_bytes(&metadata.checkpoint_id)?;
        let checkpoint = Self::decode_canvas_checkpoint(&metadata, &artifact, &bytes)?;
        match retained {
            Some(retained) => {
                self.replay_retained_canvas(&retained, &checkpoint, target_tick, None)
            }
            None => Ok(EventCanvasReview {
                snapshot: self.project_canvas_world(Self::restore_catalog_world(&checkpoint)?)?,
                checkpoint_id: metadata.checkpoint_id,
                base_tick: checkpoint.science().tick().0,
                verification: scriptbots_runtime::lab_workflows::LabReplayVerification::SavedCut,
            }),
        }
    }

    /// Review an explicitly selected saved source through the remote admission lane.
    ///
    /// Read and decode the bounded artifact once, without triggering general catalog
    /// recovery. Later targets reuse the native replay proof and check resource
    /// admission at every configuration/input/tick boundary of the complete interval.
    /// The caller owns concurrency and response deadlines; no live owner is changed.
    pub(crate) fn replay_canvas_from_checkpoint_bounded(
        &self,
        session_id: scriptbots_runtime::HostSessionId,
        checkpoint_id: &str,
        target_tick: u64,
    ) -> Result<(CheckpointMetadataDto, EventCanvasReview), ControlError> {
        if session_id != self.session_id() {
            return Err(ControlError::Conflict(
                "checkpoint replay belongs to a different owner session".into(),
            ));
        }
        let (metadata, checkpoint, directory) =
            self.read_bounded_canvas_checkpoint(checkpoint_id)?;
        if target_tick < metadata.tick {
            return Err(ControlError::BadRequest(
                "review target precedes the selected checkpoint".into(),
            ));
        }
        let retained = self.retained_checkpoint_for_replay(session_id, &metadata, target_tick)?;
        let review = match retained {
            Some(retained) => {
                if retained.directory != directory {
                    return Err(ControlError::Conflict(
                        "selected checkpoint changed its retained source directory".into(),
                    ));
                }
                self.replay_retained_canvas(
                    &retained,
                    &checkpoint,
                    target_tick,
                    Some(CanvasReplayLimits::REMOTE),
                )?
            }
            None => EventCanvasReview {
                snapshot: self.project_canvas_world(Self::restore_catalog_world(&checkpoint)?)?,
                checkpoint_id: metadata.checkpoint_id.clone(),
                base_tick: checkpoint.science().tick().0,
                verification: scriptbots_runtime::lab_workflows::LabReplayVerification::SavedCut,
            },
        };
        Ok((metadata, review))
    }

    fn retained_checkpoint_for_replay(
        &self,
        session_id: scriptbots_runtime::HostSessionId,
        metadata: &CheckpointMetadataDto,
        target_tick: u64,
    ) -> Result<Option<RetainedCheckpoint>, ControlError> {
        if target_tick == metadata.tick {
            return Ok(None);
        }
        let retention = self
            .data_services
            .checkpoint_retention
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let mut matches = retention
            .requests
            .values()
            .filter(|entry| entry.metadata.checkpoint_id == metadata.checkpoint_id);
        let retained = matches.next().ok_or_else(|| {
            ControlError::BadRequest(
                "selected checkpoint has no acknowledged current-owner replay history; only its saved cut can be reviewed".into(),
            )
        })?;
        if matches.next().is_some()
            || !retained.acknowledged
            || retained.source_session != session_id
            || retained.metadata != *metadata
            || retained.artifact != checkpoint_artifact_metadata(metadata)
        {
            return Err(ControlError::Conflict(
                "selected checkpoint replay history is ambiguous, unacknowledged or belongs to another source".into(),
            ));
        }
        retention.verify_completed_catalog(&retained.directory, metadata)?;
        Ok(Some(retained.clone()))
    }

    /// Reconstruct an older event from ticks and complete observed input histories.
    /// Configuration schedules require their complete committed outcome history;
    /// scientific commands retain the configuration revision observed at application.
    /// Other scientific revisions refuse before stepping. The atomic current digest
    /// must also match before the reconstructed target is published.
    /// Complete sources retain their companion science during reconstruction.
    /// Archive resurrection still requires its separate input-history admission
    /// contract and remains explicitly refused.
    pub(crate) fn event_canvas(&self, target_tick: u64) -> Result<EventCanvasReview, ControlError> {
        let retained = self
            .data_services
            .checkpoint_retention
            .lock()
            .map_err(|_| ControlError::Lock)?
            .requests
            .values()
            .filter(|entry| entry.acknowledged && entry.metadata.tick <= target_tick)
            .max_by_key(|entry| (entry.metadata.tick, entry.source_scientific_revision.get()))
            .cloned()
            .ok_or_else(|| {
                ControlError::NotFound(format!(
                    "no source-bound checkpoint at or before event tick {target_tick}"
                ))
            })?;
        let checkpoint_id = retained.metadata.checkpoint_id.clone();
        let checkpoint = self.verified_canvas_source(&checkpoint_id)?;
        let base_tick = checkpoint.science().tick().0;
        if target_tick == base_tick {
            return Ok(EventCanvasReview {
                snapshot: self.project_canvas_world(Self::restore_catalog_world(&checkpoint)?)?,
                checkpoint_id,
                base_tick,
                verification: scriptbots_runtime::lab_workflows::LabReplayVerification::SavedCut,
            });
        }
        self.replay_retained_canvas(&retained, &checkpoint, target_tick, None)
    }

    fn replay_retained_canvas(
        &self,
        retained: &RetainedCheckpoint,
        checkpoint: &CatalogCheckpoint,
        target_tick: u64,
        limits: Option<CanvasReplayLimits>,
    ) -> Result<EventCanvasReview, ControlError> {
        let current = self.host.scientific_digest_cut()?;
        Self::validate_event_review_boundary(retained, &current, target_tick)?;
        let published = self.read_snapshot()?;
        Self::validate_event_review_schedule(retained, &current, &published)?;
        let (config_audit, scientific_commands) =
            match Self::validate_event_review_span(retained, &current, target_tick) {
                Ok(()) => (Vec::new(), Vec::new()),
                Err(error) => {
                    if retained.source_config_revision > current.config_revision
                        || retained.source_config_revision.checked_next().is_none()
                        || current.config_revision.checked_next().is_none()
                    {
                        return Err(error);
                    }
                    let audit = if retained.source_config_revision == current.config_revision {
                        Vec::new()
                    } else {
                        Self::event_review_config_audit(retained, &current, &published)?
                    };
                    let commands = Self::event_review_scientific_commands(
                        retained,
                        &current,
                        target_tick,
                        &audit,
                    )?;
                    (audit, commands)
                }
            };
        if let Some(limits) = limits {
            limits.validate_config(
                checkpoint.science().config(),
                checkpoint.science().agent_count(),
            )?;
        }
        let world = Self::replay_event_canvas_world_restored(
            Self::restore_catalog_world(checkpoint)?,
            checkpoint.science(),
            &current.digest,
            &config_audit,
            &scientific_commands,
            target_tick,
            retained.source_persistence_interval,
            retained.source_config_revision,
            limits,
        )?;
        // The replay checked this exact captured endpoint. A target can retain
        // queued boundary inputs, so do not invent a settled target digest.
        Ok(EventCanvasReview {
            snapshot: self.project_canvas_world(world)?,
            checkpoint_id: retained.metadata.checkpoint_id.clone(),
            base_tick: checkpoint.science().tick().0,
            verification:
                scriptbots_runtime::lab_workflows::LabReplayVerification::CapturedInterval {
                    verified_through_tick: current.digest.tick.0,
                    captured_current_digest: current.digest.overall,
                },
        })
    }

    #[cfg(test)]
    #[allow(
        clippy::too_many_arguments,
        reason = "raw-science fixtures exercise the same admitted replay loop"
    )]
    fn replay_event_canvas_world(
        checkpoint: &scriptbots_core::WorldCheckpointV1,
        current: &scriptbots_core::WorldDigestV1,
        config_audit: &[ConfigAuditEntry],
        scientific_commands: &[scriptbots_runtime::AppliedScientificCommand],
        target_tick: u64,
        source_persistence_interval: u32,
        source_config_revision: scriptbots_runtime::ConfigRevision,
        limits: Option<CanvasReplayLimits>,
    ) -> Result<WorldState, ControlError> {
        if let Some(limits) = limits {
            limits.validate_config(checkpoint.config(), checkpoint.agent_count())?;
        }
        Self::replay_event_canvas_world_restored(
            Self::restore_canvas_world(checkpoint)?,
            checkpoint,
            current,
            config_audit,
            scientific_commands,
            target_tick,
            source_persistence_interval,
            source_config_revision,
            limits,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "replay inputs, original revision axes, full-interval proof and remote admission must share one ordered reconstruction"
    )]
    fn replay_event_canvas_world_restored(
        mut world: WorldState,
        checkpoint: &scriptbots_core::WorldCheckpointV1,
        current: &scriptbots_core::WorldDigestV1,
        config_audit: &[ConfigAuditEntry],
        scientific_commands: &[scriptbots_runtime::AppliedScientificCommand],
        target_tick: u64,
        source_persistence_interval: u32,
        source_config_revision: scriptbots_runtime::ConfigRevision,
        limits: Option<CanvasReplayLimits>,
    ) -> Result<WorldState, ControlError> {
        if let Some(limits) = limits {
            limits.validate_config(checkpoint.config(), checkpoint.agent_count())?;
        }
        if let Some(limits) = limits {
            limits.validate_world(&world)?;
        }
        let restored_config_revision = world.config_revision();
        let mut edits = config_audit.iter().peekable();
        let mut commands = scientific_commands.iter().peekable();
        let mut target = None;
        loop {
            if world.tick().0 == target_tick {
                // Capture before this boundary's inputs can enqueue interventions. A
                // previous tick must have consumed its pending inputs; the unchanged
                // quiescent capture refuses any deferred work that survived that tick.
                // Biased terrain changes derived food profiles which a science-only
                // checkpoint cannot restore. Preserve the existing typed companion for
                // this in-memory projection, without asserting live owner continuity.
                let (precursor, continuity) = world
                    .capture_host_checkpoint_parts_v1()
                    .map_err(|error| ControlError::Serialization(error.to_string()))?;
                target = Some((
                    precursor,
                    continuity,
                    Self::event_review_config_revision(
                        &world,
                        restored_config_revision,
                        source_config_revision,
                    )?,
                    edits.clone(),
                    commands.clone(),
                ));
            }
            let pending = Self::apply_event_review_boundary_inputs(
                &mut world,
                &mut edits,
                &mut commands,
                source_persistence_interval,
                source_config_revision,
                restored_config_revision,
                limits,
            )?;
            if world.tick() == current.tick {
                break;
            }
            if let Some(limits) = limits {
                limits.validate_step(&world, &pending)?;
            }
            world
                .step()
                .map_err(|error| ControlError::Serialization(error.to_string()))?;
            if let Some(limits) = limits {
                limits.validate_world(&world)?;
            }
        }
        if edits.peek().is_some() {
            return Err(ControlError::InvalidPatch(
                "event replay left recorded configuration edits unapplied".into(),
            ));
        }
        if commands.peek().is_some() {
            return Err(ControlError::InvalidPatch(
                "event replay left recorded scientific commands unapplied".into(),
            ));
        }
        if world
            .world_digest_v1()
            .map_err(|error| ControlError::Serialization(error.to_string()))?
            != *current
        {
            return Err(ControlError::InvalidPatch(
                "event replay differs from the captured current scientific digest; canvas unchanged".into(),
            ));
        }
        let (target, continuity, target_revision, mut target_edits, mut target_commands) =
            target.ok_or_else(|| {
                ControlError::InvalidPatch(
                    "event replay did not capture the requested target tick".into(),
                )
            })?;
        let mut target_world = Self::restore_canvas_world_parts(&target, Some(&continuity))?;
        if let Some(limits) = limits {
            limits.validate_world(&target_world)?;
        }
        let restored_target_revision = target_world.config_revision();
        // The current-digest oracle has now checked the entire interval. Reapply only
        // the target's same-tick inputs, including queued interventions, through the
        // identical boundary applier; no second science replay or live-owner mutation.
        Self::apply_event_review_boundary_inputs(
            &mut target_world,
            &mut target_edits,
            &mut target_commands,
            source_persistence_interval,
            target_revision,
            restored_target_revision,
            limits,
        )?;
        if let Some(limits) = limits {
            limits.validate_world(&target_world)?;
        }
        Ok(target_world)
    }

    fn event_review_config_revision(
        world: &WorldState,
        restored_config_revision: u64,
        source_config_revision: scriptbots_runtime::ConfigRevision,
    ) -> Result<scriptbots_runtime::ConfigRevision, ControlError> {
        world
            .config_revision()
            .checked_sub(restored_config_revision)
            .and_then(|delta| source_config_revision.get().checked_add(delta))
            .map(scriptbots_runtime::ConfigRevision::new)
            .ok_or_else(|| {
                ControlError::InvalidPatch(
                    "recorded scientific command configuration differs".into(),
                )
            })
    }

    fn apply_event_review_boundary_inputs(
        world: &mut WorldState,
        edits: &mut std::iter::Peekable<std::slice::Iter<'_, ConfigAuditEntry>>,
        commands: &mut std::iter::Peekable<
            std::slice::Iter<'_, scriptbots_runtime::AppliedScientificCommand>,
        >,
        source_persistence_interval: u32,
        source_config_revision: scriptbots_runtime::ConfigRevision,
        restored_config_revision: u64,
        limits: Option<CanvasReplayLimits>,
    ) -> Result<CanvasReplayPendingInputs, ControlError> {
        let mut pending = CanvasReplayPendingInputs::default();
        loop {
            if let Some(entry) = commands
                .peek()
                .filter(|entry| entry.boundary().tick == world.tick())
            {
                let mismatch = || {
                    ControlError::InvalidPatch(
                        "recorded scientific command configuration differs".into(),
                    )
                };
                let replayed_config_revision = Self::event_review_config_revision(
                    world,
                    restored_config_revision,
                    source_config_revision,
                )?;
                let command_config_revision = entry.boundary().revisions.config;
                if replayed_config_revision > command_config_revision {
                    return Err(mismatch());
                }
                if replayed_config_revision == command_config_revision {
                    let entry = commands.next().expect("peeked scientific command");
                    if let Some(limits) = limits {
                        limits.admit_command(world, entry.command(), &mut pending)?;
                    }
                    scriptbots_core::apply_control_command(world, entry.command().clone())
                        .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
                    if let Some(limits) = limits {
                        limits.validate_world(world)?;
                    }
                    continue;
                }
                if !edits
                    .peek()
                    .is_some_and(|entry| entry.tick == world.tick().0)
                {
                    return Err(mismatch());
                }
            }
            if !edits
                .peek()
                .is_some_and(|entry| entry.tick == world.tick().0)
            {
                return Ok(pending);
            }
            // Commands partition the config audit by the revision they observed,
            // preserving arbitrary input/edit ordering at a shared completed tick.
            let expected = world.config_revision().checked_add(1).ok_or_else(|| {
                ControlError::InvalidPatch("replayed configuration revision saturated".into())
            })?;
            Self::apply_event_review_config(
                world,
                edits.next().expect("peeked audit entry"),
                source_persistence_interval,
                limits,
            )?;
            if let Some(limits) = limits {
                limits.validate_world(world)?;
                limits.validate_arrivals(world, pending.arrivals, 0)?;
            }
            if world.config_revision() != expected {
                return Err(ControlError::InvalidPatch(
                    "recorded config edit did not advance its revision".into(),
                ));
            }
        }
    }

    fn validate_event_review_boundary(
        retained: &RetainedCheckpoint,
        current: &scriptbots_runtime::WorldDigestCut,
        target_tick: u64,
    ) -> Result<u64, ControlError> {
        let base_tick = retained.metadata.tick;
        let current_tick = current.digest.tick.0;
        let span = current_tick.checked_sub(base_tick).filter(|span| {
            *span <= MAX_EVENT_REVIEW_TICKS && target_tick >= base_tick && target_tick <= current_tick
        }).ok_or_else(|| ControlError::BadRequest(format!(
            "event tick {target_tick} requires a reachable checkpoint-to-current interval of at most {MAX_EVENT_REVIEW_TICKS} ticks"
        )))?;
        if retained.source_session != current.session_id {
            return Err(ControlError::InvalidPatch(
                "checkpoint owner session differs".into(),
            ));
        }
        if retained.source_scheduled_input_count != current.scheduled_input_count {
            return Err(ControlError::BadRequest(
                "event review needs matching source and current owner schedule declarations".into(),
            ));
        }
        Ok(span)
    }

    fn validate_event_review_schedule(
        retained: &RetainedCheckpoint,
        current: &scriptbots_runtime::WorldDigestCut,
        published: &RenderSnapshot,
    ) -> Result<(), ControlError> {
        if retained.source_scheduled_input_count == 0 && current.scheduled_input_count == 0 {
            return Ok(());
        }
        // HostCore installs configuration-only schedules once, before driving. Count
        // includes completed entries, so an exhausted queue cannot hide missing work.
        // Require the whole committed history, even entries before the saved cut.
        let mut previous_tick = 0;
        let mut previous_revision = 0;
        let complete = retained.source_scheduled_input_count == current.scheduled_input_count
            && published.session_id == current.session_id
            && published.scheduled_patches.len() as u64 == current.scheduled_input_count
            && published
                .scheduled_patches
                .iter()
                .enumerate()
                .all(|(index, status)| {
                    let evidence = &status.evidence;
                    let sequence = u64::try_from(index)
                        .ok()
                        .and_then(|index| index.checked_add(1));
                    let valid = Some(evidence.scheduled.sequence) == sequence
                        && status.batch_id.session_id() == current.session_id
                        && evidence.validate().is_ok()
                        && matches!(
                            evidence.outcome,
                            scriptbots_runtime::ScheduledPatchOutcome::Applied { .. }
                        )
                        && matches!(
                            status.journal,
                            scriptbots_runtime::JournalState::CommittedVolatile
                                | scriptbots_runtime::JournalState::Durable
                        )
                        && evidence.boundary.tick.0 >= previous_tick
                        && evidence.boundary.tick <= current.digest.tick
                        && evidence.boundary.revisions.config.get() > previous_revision
                        && evidence.boundary.revisions.config <= current.config_revision
                        && evidence.boundary.revisions.scientific <= current.scientific_revision;
                    previous_tick = evidence.boundary.tick.0;
                    previous_revision = evidence.boundary.revisions.config.get();
                    valid
                });
        if !complete {
            return Err(ControlError::BadRequest(
                "event review needs a complete committed successful owner schedule history".into(),
            ));
        }
        Ok(())
    }

    fn validate_event_review_span(
        retained: &RetainedCheckpoint,
        current: &scriptbots_runtime::WorldDigestCut,
        target_tick: u64,
    ) -> Result<(), ControlError> {
        let span = Self::validate_event_review_boundary(retained, current, target_tick)?;
        // Core config revision saturates; equality at its ceiling cannot prove no edits.
        if retained.source_config_revision.checked_next().is_none()
            || retained.source_config_revision != current.config_revision
        {
            return Err(ControlError::BadRequest(
                "event review needs recorded inputs for intervening configuration revisions".into(),
            ));
        }
        if current
            .scientific_revision
            .get()
            .checked_sub(retained.source_scientific_revision.get())
            != Some(span)
        {
            return Err(ControlError::BadRequest(
                "event review needs recorded inputs for intervening scientific revisions".into(),
            ));
        }
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "interval, input authority and revision-order checks form one completeness proof"
    )]
    fn event_review_scientific_commands(
        retained: &RetainedCheckpoint,
        current: &scriptbots_runtime::WorldDigestCut,
        target_tick: u64,
        config_audit: &[ConfigAuditEntry],
    ) -> Result<Vec<scriptbots_runtime::AppliedScientificCommand>, ControlError> {
        let span = Self::validate_event_review_boundary(retained, current, target_tick)?;
        let missing = || {
            ControlError::BadRequest(
                "event review needs complete recorded scientific inputs for intervening \
                 scientific revisions and their observed configuration revisions"
                    .into(),
            )
        };
        let config_count = u64::try_from(config_audit.len()).map_err(|_| missing())?;
        if retained.source_config_revision.checked_next().is_none()
            || current.config_revision.checked_next().is_none()
            || current
                .config_revision
                .get()
                .checked_sub(retained.source_config_revision.get())
                != Some(config_count)
            || config_audit
                .first()
                .is_some_and(|entry| entry.tick < retained.metadata.tick)
            || config_audit
                .last()
                .is_some_and(|entry| entry.tick > current.digest.tick.0)
            || config_audit
                .windows(2)
                .any(|entries| entries[0].tick > entries[1].tick)
        {
            return Err(missing());
        }
        let source_revision = retained.source_scientific_revision.get();
        let current_revision = current.scientific_revision.get();
        let commands: Vec<_> = current
            .scientific_commands
            .iter()
            .filter(|entry| {
                entry.boundary().revisions.scientific > retained.source_scientific_revision
                    && entry.boundary().revisions.scientific <= current.scientific_revision
            })
            .cloned()
            .collect();
        let mut previous_tick = retained.metadata.tick;
        let mut previous_config_revision = retained.source_config_revision;
        // A tick advances science once; each recorded scientific command advances it once.
        // Check every boundary, so an omitted or different scientific input leaves a gap.
        for (index, entry) in commands.iter().enumerate() {
            if let ControlCommand::Intervention(command) = entry.command()
                && matches!(
                    command.intervention,
                    scriptbots_core::Intervention::SpawnFromArchive { .. }
                )
            {
                return Err(ControlError::BadRequest(
                    "event review requires a source-bound behavioral archive \
                     for archive resurrection"
                        .into(),
                ));
            }
            let boundary = entry.boundary();
            let elapsed = boundary.tick.0.checked_sub(retained.metadata.tick);
            let ordinal = u64::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(1));
            let expected = elapsed
                .and_then(|ticks| source_revision.checked_add(ticks))
                .and_then(|revision| ordinal.and_then(|ordinal| revision.checked_add(ordinal)));
            // The observed revision partitions the complete audit at this command:
            // applied edits cannot be later, and unapplied edits cannot be earlier.
            // Equality deliberately allows either side of an edit at the same tick.
            let config_index = boundary
                .revisions
                .config
                .get()
                .checked_sub(retained.source_config_revision.get())
                .and_then(|index| usize::try_from(index).ok())
                .filter(|index| *index <= config_audit.len())
                .ok_or_else(missing)?;
            if expected != Some(boundary.revisions.scientific.get())
                || boundary.tick.0 < previous_tick
                || boundary.tick > current.digest.tick
                || boundary.revisions.config < previous_config_revision
                || config_audit[..config_index]
                    .last()
                    .is_some_and(|entry| entry.tick > boundary.tick.0)
                || config_audit
                    .get(config_index)
                    .is_some_and(|entry| entry.tick < boundary.tick.0)
                || !matches!(
                    entry.command(),
                    ControlCommand::AdjustAgentMutationRates { .. }
                        | ControlCommand::SpawnAgent { .. }
                        | ControlCommand::SpawnCrossover { .. }
                        | ControlCommand::Intervention(_)
                )
            {
                return Err(missing());
            }
            previous_tick = boundary.tick.0;
            previous_config_revision = boundary.revisions.config;
        }
        let expected_current = source_revision.checked_add(span).and_then(|revision| {
            u64::try_from(commands.len())
                .ok()
                .and_then(|count| revision.checked_add(count))
        });
        if expected_current != Some(current_revision) {
            return Err(missing());
        }
        Ok(commands)
    }

    fn event_review_config_audit(
        retained: &RetainedCheckpoint,
        current: &scriptbots_runtime::WorldDigestCut,
        published: &RenderSnapshot,
    ) -> Result<Vec<ConfigAuditEntry>, ControlError> {
        let missing = || {
            ControlError::BadRequest(
            "event review needs a complete ordered configuration audit for the captured interval".into(),
        )
        };
        let source_revision = retained.source_config_revision.get();
        let current_revision = current.config_revision.get();
        // Core records one entry per applied config revision, retaining a suffix. Never
        // infer ordinals at saturation, or assume this later publication equals the cut.
        let audit_base = published
            .revisions
            .config
            .get()
            .checked_sub(published.config_audit.len() as u64)
            .ok_or_else(missing)?;
        if published.session_id != current.session_id
            || current.config_revision.checked_next().is_none()
            || published.revisions.config.checked_next().is_none()
            || current_revision <= source_revision
            || current_revision > published.revisions.config.get()
        {
            return Err(missing());
        }
        let start = source_revision
            .checked_sub(audit_base)
            .ok_or_else(missing)?;
        let end = current_revision
            .checked_sub(audit_base)
            .ok_or_else(missing)?;
        let start = usize::try_from(start).map_err(|_| missing())?;
        let end = usize::try_from(end).map_err(|_| missing())?;
        let entries = published.config_audit.get(start..end).ok_or_else(missing)?;
        let mut preceding_tick = retained.metadata.tick;
        for entry in entries {
            if entry.tick < preceding_tick || entry.tick > current.digest.tick.0 {
                return Err(missing());
            }
            preceding_tick = entry.tick;
        }
        Ok(entries.to_vec())
    }

    fn apply_event_review_config(
        world: &mut WorldState,
        entry: &ConfigAuditEntry,
        source_persistence_interval: u32,
        limits: Option<CanvasReplayLimits>,
    ) -> Result<(), ControlError> {
        // set_closed records a sparse patch and deliberately does not run the full
        // config-update cache/food refresh. Preserve that operation's actual semantics.
        if let Some(patch) = entry.patch.as_object()
            && patch.len() == 1
            && let Some(closed) = patch.get("closed").and_then(Value::as_bool)
        {
            if let Some(limits) = limits {
                let mut proposed = world.config().clone();
                proposed.closed = closed;
                limits.validate_config(&proposed, world.agent_count())?;
            }
            world
                .set_closed(closed)
                .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
            return Ok(());
        }
        let mut config = world.config().clone();
        // Checkpoint restore disables persistence. Rehydrate only for patch validation;
        // replay never enables it or opens the live database.
        config.persistence_interval = source_persistence_interval;
        let mut value = serde_json::to_value(config).map_err(ControlError::serialization)?;
        merge_value(&mut value, &entry.patch, &mut SmallVec::new())?;
        let mut config: ScriptBotsConfig =
            serde_json::from_value(value).map_err(ControlError::serialization)?;
        if config.persistence_interval != source_persistence_interval {
            return Err(ControlError::BadRequest(
                "event review does not replay persistence interval changes".into(),
            ));
        }
        config
            .validate()
            .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
        if let Some(limits) = limits {
            limits.validate_config(&config, world.agent_count())?;
        }
        config.persistence_interval = 0;
        world
            .apply_config_update(config)
            .map_err(|error| ControlError::InvalidPatch(error.to_string()))
    }

    /// Produce a PNG snapshot of the world without a live window.
    pub fn snapshot_png(&self, width: u32, height: u32) -> Result<Vec<u8>, ControlError> {
        #[cfg(feature = "gui")]
        {
            const MAX_PIXELS: u64 = 64 * 1024 * 1024; // 64M px guardrail
            if (width as u64) * (height as u64) > MAX_PIXELS {
                return Err(ControlError::InvalidPatch(
                    "requested image too large".into(),
                ));
            }
            // Rasterization holds an immutable publication, never the owner.
            let snapshot = self.read_snapshot()?;
            let scene = OffscreenScene::capture(snapshot.as_ref());
            Ok(render_offscreen_scene(&scene, width, height))
        }
        #[cfg(not(feature = "gui"))]
        {
            // Reference params to avoid unused warnings in non-GUI builds
            let _ = (width, height);
            Err(ControlError::InvalidPatch(
                "PNG snapshot requires gui feature".into(),
            ))
        }
    }

    pub fn read_snapshot(&self) -> Result<Arc<RenderSnapshot>, ControlError> {
        self.host
            .clone()
            .snapshot_after(None)?
            .ok_or(ControlError::Lock)
    }

    /// Observe immutable owner publications without submitting work or advancing science.
    /// The hub carries latest-only data and independent publication wakeups.
    pub(crate) fn snapshot_hub(&self) -> scriptbots_runtime::SnapshotHub {
        self.host.snapshot_hub()
    }

    /// Retrieve the current configuration snapshot.
    pub fn snapshot(&self) -> Result<ConfigSnapshot, ControlError> {
        let snapshot = self.read_snapshot()?;
        ConfigSnapshot::from_world(&snapshot.config, Tick(snapshot.world.tick))
    }

    /// Retrieve the latest tick summary from the running world.
    pub fn latest_summary(&self) -> Result<scriptbots_core::TickSummary, ControlError> {
        let snapshot = self.read_snapshot()?;
        Ok(snapshot
            .completed_summary
            .clone()
            .or_else(|| snapshot.summary_history.last().cloned())
            .unwrap_or_else(|| scriptbots_core::TickSummary {
                tick: Tick(snapshot.world.tick),
                agent_count: snapshot.world.agents.len(),
                births: snapshot.world.summary.births,
                deaths: snapshot.world.summary.deaths,
                total_energy: snapshot.world.summary.total_energy,
                average_energy: snapshot.world.summary.average_energy,
                average_health: snapshot.world.summary.average_health,
                max_age: snapshot
                    .world
                    .agents
                    .iter()
                    .map(|agent| agent.age)
                    .max()
                    .unwrap_or(0),
                spike_hits: 0,
            }))
    }

    /// Retrieve a filtered debug listing of agents.
    pub fn debug_agents(
        &self,
        query: AgentDebugQuery,
    ) -> Result<Vec<AgentDebugInfo>, ControlError> {
        Ok(self.host.debug_agents(query)?)
    }

    /// Read one creature from the scientific owner, bound to the caller's session and epoch.
    /// Inspection is read-only and has a bounded owner-response deadline.
    pub fn inspect_agent(
        &self,
        uid: scriptbots_core::AgentUid,
        session_id: u64,
        epoch: u64,
    ) -> Result<Option<scriptbots_runtime::inspector::AgentInspection>, ControlError> {
        if self.host.session_id().get() != session_id {
            return Err(ControlError::Conflict(
                "inspector host session changed".to_owned(),
            ));
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
        let (observed_epoch, detail) = self
            .host
            .inspect_agent_basic_with_epoch_before(uid, deadline)?;
        if observed_epoch != epoch {
            return Err(ControlError::Conflict(
                "inspector source epoch changed".to_owned(),
            ));
        }
        if let Some(detail) = &detail
            && (detail.source.session_id.get() != session_id || detail.epoch != epoch)
        {
            return Err(ControlError::Conflict(
                "inspector source epoch or session changed".to_owned(),
            ));
        }
        Ok(detail.map(scriptbots_runtime::inspector::AgentInspection::from))
    }

    /// Read bounded brain activity from one owner boundary without installing a probe.
    pub fn inspect_brain_activity(
        &self,
        uid: scriptbots_core::AgentUid,
        session_id: u64,
        epoch: u64,
    ) -> Result<scriptbots_runtime::inspector::BrainActivityInspection, ControlError> {
        if self.host.session_id().get() != session_id {
            return Err(ControlError::Conflict(
                "brain inspector host session changed".to_owned(),
            ));
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
        let detail = self.host.inspect_brain_activity_before(uid, deadline)?;
        if detail.source.session_id != session_id.to_string()
            || detail.source.epoch != epoch.to_string()
        {
            return Err(ControlError::Conflict(
                "brain inspector source epoch or session changed".to_owned(),
            ));
        }
        Ok(detail)
    }

    /// Read one complete bounded genome page from the selected scientific owner.
    pub fn inspect_genome_page(
        &self,
        uid: scriptbots_core::AgentUid,
        session_id: u64,
        epoch: u64,
        locus_offset: usize,
        delta_offset: usize,
    ) -> Result<scriptbots_runtime::inspector::GenomeInspection, ControlError> {
        scriptbots_runtime::inspector::validate_genome_page_offsets(locus_offset, delta_offset)?;
        if self.host.session_id().get() != session_id {
            return Err(ControlError::Conflict(
                "genome inspector host session changed".to_owned(),
            ));
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
        let detail =
            self.host
                .inspect_genome_page_before(uid, locus_offset, delta_offset, deadline)?;
        if detail.source.session_id != session_id.to_string()
            || detail.source.epoch != epoch.to_string()
        {
            return Err(ControlError::Conflict(
                "genome inspector source epoch or session changed".to_owned(),
            ));
        }
        Ok(detail)
    }

    /// Submit a selection update and return its admission receipt.
    ///
    /// This used to return `Result<(), _>`, which made selection the only
    /// control surface a client could not follow. Every other command hands
    /// back a [`CommandStatusDto`] carrying a command id, the admission
    /// sequence and both revision axes; selection handed back nothing, so the
    /// REST layer had no identity to report and invented a bare `queued: true`
    /// instead. A client could not poll the outcome, could not tell one
    /// selection from another, and could not distinguish a command that was
    /// applied from one that was admitted and then dropped (bd-2z0.4.9).
    ///
    /// Poll the returned identity for the host's independent application and
    /// journal progress; admission itself does not establish application.
    pub fn update_selection(
        &self,
        update: SelectionUpdate,
        idempotency_key: Option<&str>,
    ) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command_with_key(
            ControlCommand::UpdateSelection(update),
            idempotency_key,
        )
    }

    /// Enqueue step commands for the simulation driver to advance `count` ticks.
    ///
    /// A single step shares the host's retry identity across transports. A batch
    /// still contains independent commands and returns only the last receipt;
    /// refuse a keyed batch rather than silently discard its retry guarantee.
    pub fn step_count(
        &self,
        count: u64,
        idempotency_key: Option<&str>,
    ) -> Result<CommandStatusDto, ControlError> {
        // Each step is a separately journalled command submitted in a loop, so an
        // unbounded request count (e.g. u64::MAX) would pin a worker thread forever and
        // grow the command journal without limit.
        if count > MAX_STEP_BATCH {
            return Err(ControlError::InvalidPatch(format!(
                "step count {count} exceeds the per-request limit of {MAX_STEP_BATCH}"
            )));
        }
        if count == 0 {
            return Err(ControlError::InvalidPatch(
                "step count must be at least 1".to_owned(),
            ));
        }
        if count == 1 {
            return self.submit_control_command_with_key(ControlCommand::Step, idempotency_key);
        }
        if idempotency_key.is_some() {
            return Err(ControlError::InvalidPatch(
                "idempotency keys require count=1; submit individually keyed steps for retryable batches"
                    .to_owned(),
            ));
        }
        let mut last_status = None;
        for _ in 0..count {
            last_status = Some(self.submit_control_command(ControlCommand::Step)?);
        }
        Ok(last_status.expect("at least one iteration"))
    }

    /// Read current status without waiting for a busy world owner.
    /// During contention, return the last published boundary with its observed tick.
    pub fn status(&self) -> Result<SimulationStatusDto, ControlError> {
        let snapshot = self.read_snapshot()?;
        Ok(SimulationStatusDto {
            tick: snapshot.world.tick,
            agent_count: snapshot.world.agents.len(),
            is_closed: snapshot.config.closed,
            config_revision: snapshot.revisions.config.get(),
            scientific_revision: snapshot.revisions.scientific.get(),
            paused: snapshot.playback.paused,
            lifecycle: snapshot.lifecycle,
            health: snapshot.health.clone(),
            last_applied_command: snapshot.last_applied_command.map(|id| id.to_string()),
            command_queue_depth: snapshot.command_queue_depth,
            snapshot_revision: snapshot.revision.get(),
        })
    }

    /// Read exact owner health for browsers without driving time or admitting work.
    pub fn remote_owner_status(&self) -> Result<RemoteOwnerStatusDto, ControlError> {
        let snapshot = self.read_snapshot()?;
        Ok(RemoteOwnerStatusDto::from_snapshot(&snapshot))
    }

    /// Retrieve a snapshot of the current hydrology state, if available.
    pub fn hydrology_snapshot(&self) -> Result<Option<HydrologySnapshot>, ControlError> {
        Ok(self
            .read_snapshot()?
            .layers
            .hydrology
            .as_deref()
            .map(HydrologySnapshot::from_snapshot))
    }

    /// Flatten the configuration into individual knob descriptors for discovery.
    pub fn list_knobs(&self) -> Result<Vec<KnobEntry>, ControlError> {
        let snapshot = self.read_snapshot()?;
        let rev = snapshot.revisions.config.get();
        if let Some((cached_rev, cached)) = lock_cache(&self.knobs_cache).as_ref()
            && *cached_rev == rev
        {
            return Ok(cached.clone());
        }
        let config_value =
            serde_json::to_value(snapshot.config.as_ref()).map_err(ControlError::serialization)?;
        let mut entries = Vec::with_capacity(256);
        let mut prefix = String::new();
        flatten_value(&mut prefix, &config_value, &mut entries);
        *lock_cache(&self.knobs_cache) = Some((rev, entries.clone()));
        Ok(entries)
    }

    /// Retrieve the configuration audit log accumulated since startup.
    pub fn audit(&self) -> Result<Vec<ConfigAuditEntry>, ControlError> {
        Ok(self.read_snapshot()?.config_audit.as_ref().clone())
    }

    /// Build a tail of recent narrative events from the world's tick history.
    /// Events include births, deaths, and combat spike hits.
    pub fn events_tail(&self, limit: usize) -> Result<Vec<EventEntry>, ControlError> {
        // Answered before the lock is taken. This used to sit AFTER it, so a
        // limit=0 request contended on the world mutex to return an empty vec.
        if limit == 0 {
            return Ok(Vec::new());
        }
        let snapshot = self.read_snapshot()?;
        // The limit arrives unclamped from the query string; cap it so a hostile
        // request cannot reserve unbounded memory (history yields ≤3 events/tick).
        let limit = limit.min(snapshot.summary_history.len().saturating_mul(3).max(1));
        let mut events = Vec::with_capacity(limit);
        for summary in snapshot.summary_history.iter().rev() {
            if summary.births > 0 {
                events.push(EventEntry::new(
                    summary.tick.0,
                    EventKind::Birth,
                    saturating_u32(summary.births),
                ));
                if events.len() >= limit {
                    break;
                }
            }
            if summary.deaths > 0 {
                events.push(EventEntry::new(
                    summary.tick.0,
                    EventKind::Death,
                    saturating_u32(summary.deaths),
                ));
                if events.len() >= limit {
                    break;
                }
            }
            if summary.spike_hits > 0 {
                events.push(EventEntry::new(
                    summary.tick.0,
                    EventKind::Combat,
                    summary.spike_hits,
                ));
                if events.len() >= limit {
                    break;
                }
            }
        }
        Ok(events)
    }

    /// Search narrative events using FrankenSQLite FTS5 full-text index with in-memory fallback.
    pub fn narrative_search(
        &self,
        query: crate::narrative_search::NarrativeSearchQuery,
    ) -> Result<Vec<crate::narrative_search::NarrativeSearchHitDto>, ControlError> {
        let snapshot = self.read_snapshot().ok();
        let events = snapshot.as_ref().map(|s| s.narrative_events.as_slice());
        Ok(crate::narrative_search::execute_narrative_search(
            self.database_path.as_deref(),
            events,
            query,
        )?)
    }

    /// Retrieve a chronological window of narrative events around a tick.
    pub fn narrative_around(
        &self,
        query: crate::narrative_search::NarrativeAroundQuery,
    ) -> Result<Vec<crate::narrative_search::NarrativeSearchHitDto>, ControlError> {
        let snapshot = self.read_snapshot().ok();
        let events = snapshot.as_ref().map(|s| s.narrative_events.as_slice());
        Ok(crate::narrative_search::execute_narrative_around(
            self.database_path.as_deref(),
            events,
            query,
        )?)
    }

    /// Retrieve current world selection state from the latest snapshot.
    pub fn current_selection(&self) -> Result<SelectionSnapshotDto, ControlError> {
        let snapshot = self.read_snapshot()?;
        let mut selected_agent_ids = Vec::new();
        for (i, state) in snapshot.agent_selection.iter().enumerate() {
            if *state == SelectionState::Selected
                && let Some(agent) = snapshot.world.agents.get(i)
            {
                selected_agent_ids.push(agent.id);
            }
        }
        Ok(SelectionSnapshotDto {
            revision: snapshot.revision.get(),
            selected_count: selected_agent_ids.len(),
            selected_agent_ids,
            last_applied_command: snapshot.last_applied_command.map(|c| c.to_string()),
        })
    }

    /// Retrieve applied interventions from the latest snapshot bounded ring.
    ///
    /// If `after_seq` is supplied, only records strictly greater than `after_seq`
    /// are returned. If `after_seq` is older than the oldest retained record in the ring,
    /// `gap_detected` is set to `true`, admitting that intermediate events were evicted.
    pub fn applied_interventions(
        &self,
        after_seq: Option<u64>,
    ) -> Result<InterventionsPollDto, ControlError> {
        let snapshot = self.read_snapshot()?;
        let min_seq = snapshot.applied_interventions.first().map(|r| r.seq);
        let max_seq = snapshot
            .applied_interventions
            .last()
            .map(|r| r.seq)
            .unwrap_or(0);
        let gap_detected = match (after_seq, min_seq) {
            (Some(requested), Some(oldest)) => requested + 1 < oldest,
            _ => false,
        };
        let records = snapshot
            .applied_interventions
            .iter()
            .filter(|r| after_seq.is_none_or(|seq| r.seq > seq))
            .map(|r| AppliedInterventionDto {
                seq: r.seq,
                tick: r.tick.0,
                kind: r.kind.to_string(),
                region: format!("{:?}", r.region),
                agents_affected: r.agents_affected,
                cells_affected: r.cells_affected,
                expires_at: r.expires_at.map(|t| t.0),
                expired: r.expired,
            })
            .collect();
        Ok(InterventionsPollDto {
            records,
            watermark_seq: max_seq,
            gap_detected,
        })
    }

    /// Render a coarse ASCII map of terrain, food, and agents — the server-side
    /// equivalent of the terminal renderer's saved snapshots.
    pub fn ascii_map(&self) -> Result<String, ControlError> {
        let snapshot = self.read_snapshot()?;
        let food = &snapshot.layers.food;
        let terrain = &snapshot.layers.terrain;
        let grid_w = food.width.max(1) as usize;
        let grid_h = food.height.max(1) as usize;
        let width = grid_w.clamp(16, 96);
        let height = grid_h.clamp(8, 48);
        let food_max = snapshot.config.food_max.max(f32::EPSILON);
        let world_w = (snapshot.config.world_width as f32).max(1.0);
        let world_h = (snapshot.config.world_height as f32).max(1.0);
        let tiles = &terrain.tiles;
        let cells = &food.cells;

        let mut rows = vec![vec![' '; width]; height];
        for (y, row) in rows.iter_mut().enumerate() {
            for (x, slot) in row.iter_mut().enumerate() {
                let cell_x = (x * grid_w) / width;
                let cell_y = (y * grid_h) / height;
                let idx = cell_y * grid_w + cell_x;
                let kind = tiles.get(idx).map(|tile| tile.kind);
                let food_level = cells.get(idx).copied().unwrap_or(0.0) / food_max;
                let base = match kind {
                    Some(TerrainKind::DeepWater) => '~',
                    Some(TerrainKind::ShallowWater) => '=',
                    Some(TerrainKind::Sand) => '.',
                    Some(TerrainKind::Grass) => ',',
                    Some(TerrainKind::Bloom) => '*',
                    Some(TerrainKind::Rock) => '^',
                    None => ' ',
                };
                *slot = if food_level > 0.66 {
                    '#'
                } else if food_level > 0.33 {
                    '+'
                } else {
                    base
                };
            }
        }
        for agent in &snapshot.world.agents {
            let x = (((agent.position[0] / world_w) * width as f32) as usize).min(width - 1);
            let y = (((agent.position[1] / world_h) * height as f32) as usize).min(height - 1);
            rows[y][x] = '@';
        }

        let mut out = format!("ScriptBots tick {}\n", snapshot.world.tick);
        for row in rows {
            out.extend(row);
            out.push('\n');
        }
        Ok(out)
    }

    /// Compute scoreboard snapshots: top predators (carnivores) by energy and oldest living agents.
    pub fn compute_scoreboard(&self, limit: usize) -> Result<Scoreboard, ControlError> {
        // Collection happens inside the seam and ranking happens outside it, so
        // the expensive sort provably cannot hold the world lock. That used to
        // rest on a hand-placed `drop(world)` with a comment; now the borrow
        // ends at the closure boundary and the compiler enforces it (bd-88yj).
        let snapshot = self.read_snapshot()?;
        let mut carnivores = Vec::with_capacity(snapshot.world.agents.len() / 2 + 1);
        let mut oldest = Vec::with_capacity(snapshot.world.agents.len());

        for agent in &snapshot.world.agents {
            let diet_core = DietClass::from_tendency(agent.herbivore_tendency);
            let diet = DietClassDto::from(diet_core);

            let entry = AgentScoreEntry {
                agent_id: agent.id,
                energy: agent.energy,
                health: agent.health,
                age: agent.age,
                generation: agent.generation.0,
                diet,
            };

            if matches!(diet_core, DietClass::Carnivore) {
                carnivores.push(entry.clone());
            }
            oldest.push(entry);
        }

        if limit == 0 {
            return Ok(Scoreboard {
                top_predators: Vec::new(),
                oldest: Vec::new(),
            });
        }

        partial_top_k(&mut carnivores, limit, cmp_score);
        if oldest.len() > limit {
            let nth = limit - 1;
            oldest.select_nth_unstable_by_key(nth, |e| Reverse(e.age));
            oldest.truncate(limit);
            oldest.sort_unstable_by_key(|e| Reverse(e.age));
        } else {
            oldest.sort_unstable_by_key(|e| Reverse(e.age));
        }

        Ok(Scoreboard {
            top_predators: carnivores,
            oldest,
        })
    }

    /// Apply a structured JSON patch object onto the observed configuration.
    ///
    /// The host rejects the command if another configuration change wins before
    /// application. The returned command status exposes that conflict so callers
    /// can read the new configuration before choosing to submit another patch.
    pub fn apply_patch(&self, patch: Value) -> Result<CommandStatusDto, ControlError> {
        let envelope = self.prepare_config_patch(patch)?;
        let status = self.host.clone().submit(envelope)?;
        let receipt = self.status_dto(status)?;
        *lock_cache(&self.knobs_cache) = None;
        Ok(receipt)
    }

    /// Bind the complete replacement to the same publication used for the merge.
    fn prepare_config_patch(&self, patch: Value) -> Result<CommandEnvelope, ControlError> {
        if !patch.is_object() {
            return Err(ControlError::InvalidPatch(
                "configuration patch must be a JSON object".into(),
            ));
        }

        // Every world read this function needs, captured ATOMICALLY in one
        // borrow. The original held the lock from here through validation,
        // merge and deserialization, reading config again near the end; holding
        // one guard is what made those two reads consistent. Capturing both up
        // front preserves that consistency exactly while moving the expensive
        // work - a JSON merge and a full config deserialization - off the lock
        // instead of running it under one (bd-88yj).
        let snapshot = self.read_snapshot()?;
        let (config_value, current_dims, current_bounds) = {
            let config = &snapshot.config;
            (
                serde_json::to_value(config.as_ref()),
                (snapshot.layers.food.width, snapshot.layers.food.height),
                (
                    config.world_width,
                    config.world_height,
                    config.food_cell_size,
                ),
            )
        };
        let mut config_value = config_value.map_err(ControlError::serialization)?;
        // Range-check the REQUESTED knobs before merging them. `validate()`
        // proves admissibility (finite, non-negative) but declares no upper
        // bounds, so `food_regrowth_rate = 1e9` used to sail through from REST,
        // from MCP, and therefore from any agent driving them. Every violation
        // is reported at once: a caller who has to fix one knob per round trip
        // gives up, and an autonomous one burns its entire budget doing it.
        let requested = flatten_numeric_assignments(&patch);
        let violations = check_knob_ranges(&requested);
        if !violations.is_empty() {
            let detail = violations
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ");
            return Err(ControlError::InvalidPatch(detail));
        }

        let mut path = SmallVec::<[&str; 8]>::new();
        merge_value(&mut config_value, &patch, &mut path)?;
        let json_str = serde_json::to_string(&config_value).map_err(ControlError::serialization)?;
        let mut de = serde_json::Deserializer::from_str(&json_str);
        let new_config: ScriptBotsConfig = serde_path_to_error::deserialize::<_, ScriptBotsConfig>(
            &mut de,
        )
        .map_err(|e: serde_path_to_error::Error<serde_json::Error>| {
            ControlError::InvalidPatch(format!("{} at {}", e, e.path()))
        })?;
        let (food_w, food_h) = new_config
            .food_dimensions()
            .map_err(|err| ControlError::InvalidPatch(err.to_string()))?;
        let (current_width, current_height, current_cell_size) = current_bounds;
        if current_dims != (food_w, food_h)
            || new_config.world_width != current_width
            || new_config.world_height != current_height
            || new_config.food_cell_size != current_cell_size
        {
            return Err(ControlError::InvalidPatch(
                "changing world dimensions at runtime is not supported; restart the simulation with the new configuration"
                    .into(),
            ));
        }
        // A patch becomes a full UpdateConfig command. Without this guard, two
        // clients merging disjoint fields from the same publication would both
        // apply and the later replacement would silently undo the first edit.
        // Keep the observed revision with the immutable config that produced it;
        // a fresh revision read here would authorize precisely that stale write.
        Ok(self
            .prepare_control_command(ControlCommand::UpdateConfig(Box::new(new_config)), None)?
            .expecting_config_revision(snapshot.revisions.config))
    }

    /// Apply a list of knob updates by path.
    pub fn apply_updates(&self, updates: &[KnobUpdate]) -> Result<CommandStatusDto, ControlError> {
        let mut patch_map = Map::new();
        for update in updates {
            insert_path(&mut patch_map, &update.path, update.value.clone())?;
        }
        self.apply_patch(Value::Object(patch_map))
    }

    /// Pause simulation ticks.
    pub fn pause(&self, idempotency_key: Option<&str>) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command_with_key(ControlCommand::Pause, idempotency_key)
    }

    /// Resume simulation ticks.
    pub fn resume(&self, idempotency_key: Option<&str>) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command_with_key(ControlCommand::Resume, idempotency_key)
    }

    /// Step simulation by one tick.
    pub fn step(&self) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command(ControlCommand::Step)
    }

    /// Set simulation playback speed multiplier.
    pub fn set_speed(
        &self,
        speed: f32,
        idempotency_key: Option<&str>,
    ) -> Result<CommandStatusDto, ControlError> {
        if !speed.is_finite() || speed < 0.0 {
            return Err(ControlError::InvalidPatch(
                "invalid speed multiplier".into(),
            ));
        }
        self.submit_control_command_with_key(ControlCommand::SetSpeed(speed), idempotency_key)
    }

    /// Issue graceful shutdown command.
    pub fn shutdown(&self) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command(ControlCommand::Shutdown)
    }

    /// Look up status of a command by ID.
    pub fn command_status(
        &self,
        command_id: &str,
    ) -> Result<Option<CommandStatusDto>, ControlError> {
        let id: CommandId = serde_json::from_value(Value::String(command_id.to_owned()))
            .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
        self.host
            .clone()
            .command_status(id)?
            .map(|status| self.status_dto(status))
            .transpose()
    }

    /// Submit a map command with an optional owner-side scientific revision guard.
    pub fn apply_map(
        &self,
        artifact: MapArtifact,
        existing_agent_policy: ExistingAgentPolicy,
        expected_scientific_revision: Option<u64>,
        idempotency_key: Option<&str>,
    ) -> Result<CommandStatusDto, ControlError> {
        let mut envelope = self.prepare_control_command(
            ControlCommand::ApplyMap {
                artifact: Box::new(artifact),
                existing_agent_policy,
            },
            idempotency_key,
        )?;
        envelope.expected_scientific_revision =
            expected_scientific_revision.map(ScientificRevision::new);
        let status = self.host.clone().submit(envelope)?;
        self.status_dto(status)
    }

    /// Submit an Intervention command.
    pub fn intervene(
        &self,
        cmd: scriptbots_core::interventions::InterventionCommand,
        idempotency_key: Option<&str>,
    ) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command_with_key(
            ControlCommand::Intervention(Box::new(cmd)),
            idempotency_key,
        )
    }

    /// Stage one bounded browser experiment at an observed paused boundary.
    /// Retrying a retained request reuses its complete envelope and actual owner status.
    pub fn browser_intervene(
        &self,
        request: &scriptbots_runtime::remote::BrowserInterventionRequest,
    ) -> Result<scriptbots_runtime::remote::BrowserInterventionReceipt, ControlError> {
        use scriptbots_runtime::remote::{
            BrowserInterventionReceipt, validate_browser_intervention_session,
        };
        let envelope = request.envelope()?;
        let snapshot = self.read_snapshot()?;
        validate_browser_intervention_session(&snapshot, &request.session_id)?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
        if self
            .host
            .command_status_before(envelope.command_id, deadline)?
            .is_none()
        {
            request.validate_new_source(&snapshot)?;
        }
        // Always resubmit the original envelope, even when a receipt already exists:
        // the owner must reject a changed body or guard reusing this identity.
        let status = self.host.submit_before(envelope, deadline)?;
        Ok(BrowserInterventionReceipt::from_status(
            &status,
            self.read_snapshot()?.as_ref(),
        ))
    }

    /// Read a retained browser command receipt without interpreting absence as non-admission.
    pub fn browser_intervention_status(
        &self,
        command_id: &str,
        session_id: &str,
    ) -> Result<Option<scriptbots_runtime::remote::BrowserInterventionReceipt>, ControlError> {
        use scriptbots_runtime::remote::{
            BrowserInterventionReceipt, browser_intervention_command_id,
            validate_browser_intervention_session,
        };
        let command_id = browser_intervention_command_id(command_id)?;
        validate_browser_intervention_session(self.read_snapshot()?.as_ref(), session_id)?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
        let status = self.host.command_status_before(command_id, deadline)?;
        let snapshot = self.read_snapshot()?;
        Ok(status.map(|status| BrowserInterventionReceipt::from_status(&status, &snapshot)))
    }

    /// Read bounded global effect observations; the core ring has no per-command linkage.
    pub fn browser_intervention_history(
        &self,
        session_id: &str,
        epoch: &str,
        after_seq: &str,
        limit: usize,
    ) -> Result<scriptbots_runtime::remote::BrowserInterventionHistory, ControlError> {
        Ok(
            scriptbots_runtime::remote::BrowserInterventionHistory::capture(
                self.read_snapshot()?.as_ref(),
                session_id,
                epoch,
                after_seq,
                limit,
            )?,
        )
    }

    /// Generate a procedural map artifact using the requested rule or sample backend.
    pub fn generate_map(
        &self,
        width: u32,
        height: u32,
        cell_size: Option<u32>,
        seed: u64,
        generator_config: Option<DraftGeneratorConfig>,
    ) -> Result<MapArtifact, ControlError> {
        let cells = u64::from(width) * u64::from(height);
        if cells > MAX_GENERATED_MAP_CELLS {
            return Err(ControlError::InvalidPatch(format!(
                "map of {width}x{height} cells exceeds the limit of {MAX_GENERATED_MAP_CELLS} cells"
            )));
        }
        let cell_size = cell_size.unwrap_or(50);
        let result = match generator_config.unwrap_or_else(|| DraftGeneratorConfig::RuleBased {
            tileset: default_tileset_spec(),
        }) {
            DraftGeneratorConfig::RuleBased { tileset } => RuleBasedMapGenerator::new(tileset)
                .and_then(|generator| generator.generate(width, height, cell_size, seed)),
            DraftGeneratorConfig::SampleBased { sample } => SampleBasedMapGenerator::new(sample)
                .and_then(|generator| generator.generate(width, height, cell_size, seed)),
        };
        result.map_err(|error| ControlError::InvalidPatch(format!("map generation error: {error}")))
    }

    /// Submit a command, honouring an idempotency key when one is supplied.
    ///
    /// [`scriptbots_runtime::HostPort::submit`] is documented as "submit or
    /// retry a logical command", with the command id as a "stable idempotency
    /// key" whose retry "returns its existing status". This path met none of
    /// that: ids were minted from a server-side counter, so a client that timed
    /// out and retried submitted a SECOND command. For `Pause` that is
    /// harmless, but a retried `Step` advances the simulation twice and a
    /// retried config patch applies twice — the client cannot tell, because
    /// both attempts return a cheerful receipt with different ids (bd-k7nq).
    ///
    /// Keyed retries always reach the authoritative host ledger: the same
    /// envelope reuses its identity, while a changed payload is a typed conflict.
    fn submit_control_command_with_key(
        &self,
        cmd: ControlCommand,
        idempotency_key: Option<&str>,
    ) -> Result<CommandStatusDto, ControlError> {
        let envelope = self.prepare_control_command(cmd, idempotency_key)?;
        let status = self.host.clone().submit(envelope)?;
        self.status_dto(status)
    }

    /// Validate and assign one identity before a caller starts a bounded retry.
    /// Retrying must reuse this complete envelope, not prepare the command again.
    pub(crate) fn prepare_control_command(
        &self,
        cmd: ControlCommand,
        idempotency_key: Option<&str>,
    ) -> Result<CommandEnvelope, ControlError> {
        cmd.validate()
            .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
        let command = HostCommand::try_from(cmd)
            .map_err(|error| ControlError::InvalidPatch(error.to_string()))?;
        let id = if let Some(key) = idempotency_key {
            if key.is_empty() || key.len() > 1024 {
                return Err(ControlError::InvalidPatch(
                    "idempotency key must contain 1..=1024 bytes".into(),
                ));
            }
            let mut hasher = blake3::Hasher::new_derive_key("scriptbots.control.idempotency.v1");
            hasher.update(&self.host.session_id().get().to_le_bytes());
            hasher.update(key.as_bytes());
            let mut bytes = [0_u8; 16];
            bytes.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
            u128::from_le_bytes(bytes) | (1_u128 << 127)
        } else {
            let sequence = self
                .command_counter
                .try_update(
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                    |value| value.checked_add(1),
                )
                .map_err(|_| {
                    ControlError::InvalidPatch("command identity sequence exhausted".into())
                })?;
            (u128::from(self.command_namespace) << 64) | u128::from(sequence)
        };
        Ok(CommandEnvelope::new(CommandId::new(id), command))
    }

    fn status_dto(
        &self,
        status: scriptbots_runtime::CommandStatus,
    ) -> Result<CommandStatusDto, ControlError> {
        let revisions = match status.application() {
            ApplicationState::Applied(applied) => applied.revisions,
            _ => self.read_snapshot()?.revisions,
        };
        Ok(CommandStatusDto {
            command_id: status.command_id().to_string(),
            admission_sequence: status.admission_sequence().map(|sequence| sequence.get()),
            application_state: match status.application() {
                ApplicationState::Admitted => "admitted",
                ApplicationState::Applied(_) => "applied",
                ApplicationState::Rejected(_) => "rejected",
                ApplicationState::Failed(_) => "failed",
            }
            .to_owned(),
            rejection_reason: match status.application() {
                ApplicationState::Rejected(reason) => {
                    Some(serde_json::to_value(reason).map_err(ControlError::serialization)?)
                }
                _ => None,
            },
            journal_state: match status.journal() {
                JournalState::NotRequired => "not_required",
                JournalState::Pending => "pending",
                JournalState::CommittedVolatile => "committed_volatile",
                JournalState::Durable => "durable",
                JournalState::Failed(_) => "failed",
            }
            .to_owned(),
            control_revision: revisions.control.get(),
            scientific_revision: revisions.scientific.get(),
        })
    }

    /// Submit a command without an idempotency key.
    ///
    /// Every existing caller keeps its previous behaviour: a retry here is a
    /// new command, because the caller supplied nothing to recognise it by.
    fn submit_control_command(
        &self,
        cmd: ControlCommand,
    ) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command_with_key(cmd, None)
    }

    /// Submit any control command, optionally keyed for safe retry.
    ///
    /// This is the [`scriptbots_runtime::HostPort`]-shaped entry point: one
    /// command, one optional stable key, one receipt. Surfaces that can carry a
    /// client-supplied key (an `Idempotency-Key` header, an MCP argument)
    /// should use it so a timeout-and-retry cannot double-apply. The
    /// command-specific helpers below remain for callers that have no key to
    /// offer.
    pub fn submit_command(
        &self,
        command: ControlCommand,
        idempotency_key: Option<&str>,
    ) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command_with_key(command, idempotency_key)
    }

    /// Submit a full configuration only at the caller's observed owner boundary.
    ///
    /// The owner checks all three revision guards when the command reaches its
    /// ordered application boundary. A concurrent resume, tick, or configuration
    /// edit therefore rejects this proposal instead of overwriting newer state.
    /// The typed receipt preserves both rejected and admitted command identities.
    pub fn submit_config_update(
        &self,
        config: Box<ScriptBotsConfig>,
        expected: scriptbots_runtime::HostRevisions,
    ) -> Result<scriptbots_runtime::CommandStatus, ControlError> {
        let envelope = self
            .prepare_control_command(ControlCommand::UpdateConfig(config), None)?
            .expecting_control_revision(expected.control)
            .expecting_scientific_revision(expected.scientific)
            .expecting_config_revision(expected.config);
        Ok(self.host.clone().submit(envelope)?)
    }
}

fn insert_path(map: &mut Map<String, Value>, path: &str, value: Value) -> Result<(), ControlError> {
    // Bound the nesting depth: the patch is later walked recursively, so a request with
    // tens of thousands of segments would overflow the stack and abort the process.
    if path.len() > MAX_KNOB_PATH_BYTES || path.split('.').count() > MAX_KNOB_PATH_SEGMENTS {
        return Err(ControlError::InvalidPatch(format!(
            "knob path exceeds {MAX_KNOB_PATH_SEGMENTS} segments or {MAX_KNOB_PATH_BYTES} bytes"
        )));
    }
    let mut segments = path.split('.').filter(|s| !s.is_empty());
    let Some(mut seg) = segments.next() else {
        return Err(ControlError::InvalidPatch("empty knob path".into()));
    };
    let mut cur = map;

    for next in segments {
        // Always use Entry API to avoid double-borrow; require objects for intermediate segments
        let entry = cur
            .entry(seg.to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        cur = entry.as_object_mut().ok_or_else(|| {
            ControlError::InvalidPatch(format!("intermediate segment '{seg}' is not an object"))
        })?;
        seg = next;
    }

    cur.insert(seg.to_owned(), value);
    Ok(())
}

fn path_display(path: &[&str]) -> String {
    path.join(".")
}

fn set_f64(target: &mut Value, v: f64, path: &[&str]) -> Result<(), ControlError> {
    if !v.is_finite() {
        return Err(ControlError::InvalidPatch(format!(
            "non-finite float at {}",
            path_display(path)
        )));
    }
    *target = Value::Number(serde_json::Number::from_f64(v).expect("checked finite above"));
    Ok(())
}

fn merge_value<'a>(
    target: &mut Value,
    patch: &'a Value,
    path: &mut SmallVec<[&'a str; 8]>,
) -> Result<(), ControlError> {
    match target {
        Value::Object(target_map) => {
            let Value::Object(patch_map) = patch else {
                return Err(ControlError::InvalidPatch(format!(
                    "type mismatch at {}",
                    path_display(path),
                )));
            };

            for (key, patch_value) in patch_map {
                path.push(key);
                let Some(target_value) = target_map.get_mut(key) else {
                    return Err(ControlError::UnknownPath(path_display(path)));
                };
                merge_value(target_value, patch_value, path)?;
                path.pop();
            }
            Ok(())
        }
        Value::Array(_) => {
            if matches!(patch, Value::Array(_)) {
                *target = patch.clone();
                Ok(())
            } else {
                Err(ControlError::InvalidPatch(format!(
                    "type mismatch at {}",
                    path_display(path),
                )))
            }
        }
        Value::Number(_) => match patch {
            Value::Number(n) => {
                *target = Value::Number(n.clone());
                Ok(())
            }
            Value::String(s) => {
                let s = s.trim();
                if target.as_i64().is_some() {
                    let v: i64 = s
                        .parse()
                        .map_err(|_| ControlError::InvalidPatch(path_display(path)))?;
                    *target = Value::from(v);
                } else if target.as_u64().is_some() {
                    let v: u64 = s
                        .parse()
                        .map_err(|_| ControlError::InvalidPatch(path_display(path)))?;
                    *target = Value::from(v);
                } else {
                    let v: f64 = s
                        .parse()
                        .map_err(|_| ControlError::InvalidPatch(path_display(path)))?;
                    set_f64(target, v, path)?;
                }
                Ok(())
            }
            Value::Null => {
                *target = Value::Null;
                Ok(())
            }
            _ => Err(ControlError::InvalidPatch(format!(
                "type mismatch at {}",
                path_display(path),
            ))),
        },
        Value::String(_) => match patch {
            Value::String(_) | Value::Null => {
                *target = patch.clone();
                Ok(())
            }
            _ => Err(ControlError::InvalidPatch(format!(
                "type mismatch at {}",
                path_display(path),
            ))),
        },
        Value::Bool(_) => match patch {
            Value::Bool(_) | Value::Null => {
                *target = patch.clone();
                Ok(())
            }
            Value::String(_) => {
                let parsed = match patch.as_str().map(|s| s.trim().to_ascii_lowercase()) {
                    Some(s) if matches!(s.as_str(), "true" | "1" | "yes" | "on" | "t" | "y") => {
                        true
                    }
                    Some(s) if matches!(s.as_str(), "false" | "0" | "no" | "off" | "f" | "n") => {
                        false
                    }
                    _ => {
                        return Err(ControlError::InvalidPatch(format!(
                            "cannot coerce '{:?}' to bool for {}",
                            patch,
                            path_display(path),
                        )));
                    }
                };
                *target = Value::from(parsed);
                Ok(())
            }
            _ => Err(ControlError::InvalidPatch(format!(
                "type mismatch at {}",
                path_display(path),
            ))),
        },
        Value::Null => {
            *target = patch.clone();
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Birth,
    Death,
    Combat,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct EventEntry {
    pub tick: u64,
    pub kind: EventKind,
    pub count: u32,
}

impl EventEntry {
    pub fn new(tick: u64, kind: EventKind, count: u32) -> Self {
        Self { tick, kind, count }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DietClassDto {
    Herbivore,
    Omnivore,
    Carnivore,
}

impl From<DietClass> for DietClassDto {
    fn from(value: DietClass) -> Self {
        match value {
            DietClass::Herbivore => Self::Herbivore,
            DietClass::Omnivore => Self::Omnivore,
            DietClass::Carnivore => Self::Carnivore,
        }
    }
}

impl From<DietClassDto> for DietClass {
    fn from(value: DietClassDto) -> Self {
        match value {
            DietClassDto::Herbivore => DietClass::Herbivore,
            DietClassDto::Omnivore => DietClass::Omnivore,
            DietClassDto::Carnivore => DietClass::Carnivore,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, utoipa::ToSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStateDto {
    None,
    Hovered,
    #[default]
    Selected,
}

impl From<SelectionState> for SelectionStateDto {
    fn from(value: SelectionState) -> Self {
        match value {
            SelectionState::None => Self::None,
            SelectionState::Hovered => Self::Hovered,
            SelectionState::Selected => Self::Selected,
        }
    }
}

impl From<SelectionStateDto> for SelectionState {
    fn from(value: SelectionStateDto) -> Self {
        match value {
            SelectionStateDto::None => SelectionState::None,
            SelectionStateDto::Hovered => SelectionState::Hovered,
            SelectionStateDto::Selected => SelectionState::Selected,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum SelectionModeDto {
    Replace,
    Add,
    Clear,
}

impl From<SelectionModeDto> for SelectionMode {
    fn from(value: SelectionModeDto) -> Self {
        match value {
            SelectionModeDto::Replace => SelectionMode::Replace,
            SelectionModeDto::Add => SelectionMode::Add,
            SelectionModeDto::Clear => SelectionMode::Clear,
        }
    }
}

impl From<SelectionMode> for SelectionModeDto {
    fn from(value: SelectionMode) -> Self {
        match value {
            SelectionMode::Replace => SelectionModeDto::Replace,
            SelectionMode::Add => SelectionModeDto::Add,
            SelectionMode::Clear => SelectionModeDto::Clear,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema, PartialEq, Eq)]
pub struct SelectionSnapshotDto {
    pub revision: u64,
    pub selected_count: usize,
    pub selected_agent_ids: Vec<u64>,
    pub last_applied_command: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema, PartialEq)]
pub struct AppliedInterventionDto {
    pub seq: u64,
    pub tick: u64,
    pub kind: String,
    pub region: String,
    pub agents_affected: usize,
    pub cells_affected: usize,
    pub expires_at: Option<u64>,
    pub expired: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema, PartialEq)]
pub struct InterventionsPollDto {
    pub records: Vec<AppliedInterventionDto>,
    pub watermark_seq: u64,
    pub gap_detected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct AgentScoreEntry {
    pub agent_id: u64,
    pub energy: f32,
    pub health: f32,
    pub age: u32,
    pub generation: u32,
    pub diet: DietClassDto,
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Scoreboard {
    pub top_predators: Vec<AgentScoreEntry>,
    pub oldest: Vec<AgentScoreEntry>,
}

fn cmp_score(a: &AgentScoreEntry, b: &AgentScoreEntry) -> std::cmp::Ordering {
    b.energy
        .total_cmp(&a.energy)
        .then_with(|| b.health.total_cmp(&a.health))
        .then_with(|| b.age.cmp(&a.age))
}

fn partial_top_k<T, F: Fn(&T, &T) -> std::cmp::Ordering>(v: &mut Vec<T>, k: usize, cmp: F) {
    if v.len() <= k {
        v.sort_by(cmp);
        return;
    }
    let nth = k.saturating_sub(1);
    v.select_nth_unstable_by(nth, &cmp);
    v.truncate(k);
    v.sort_by(cmp);
}

/// Flatten a JSON patch into dotted-path numeric assignments for range checking.
///
/// Only numbers are collected: strings, booleans and structural values are left
/// to serde, which already rejects type mismatches with a precise path.
fn flatten_numeric_assignments(patch: &Value) -> Vec<(String, f64)> {
    fn walk(prefix: &str, value: &Value, out: &mut Vec<(String, f64)>) {
        match value {
            Value::Object(map) => {
                for (key, child) in map {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    walk(&path, child, out);
                }
            }
            Value::Number(number) => {
                if let Some(as_f64) = number.as_f64() {
                    out.push((prefix.to_owned(), as_f64));
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk("", patch, &mut out);
    out
}

fn flatten_value(prefix: &mut String, value: &Value, entries: &mut Vec<KnobEntry>) {
    match value {
        Value::Object(map) => {
            let base = prefix.len();
            for (k, v) in map {
                if base != 0 {
                    prefix.push('.');
                }
                prefix.push_str(k);
                flatten_value(prefix, v, entries);
                prefix.truncate(base);
            }
        }
        _ => entries.push(KnobEntry {
            path: prefix.clone(),
            kind: knob_kind(value),
            value: value.clone(),
            description: None,
        }),
    }
}

fn knob_kind(value: &Value) -> KnobKind {
    match value {
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                KnobKind::Integer
            } else {
                KnobKind::Number
            }
        }
        Value::String(_) => KnobKind::String,
        Value::Bool(_) => KnobKind::Boolean,
        Value::Array(_) => KnobKind::Array,
        Value::Object(_) => KnobKind::Object,
        Value::Null => KnobKind::Null,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use slotmap::Key;
    use std::sync::{Arc, Mutex};

    pub(crate) struct TestHost {
        pub(crate) port: ChannelHostPort,
        worker: Option<std::thread::JoinHandle<Option<Vec<u8>>>>,
        clock_gate: Arc<Mutex<()>>,
        clock_blocked: std::sync::mpsc::Receiver<()>,
    }

    impl TestHost {
        pub(crate) fn spawn(world: WorldState) -> Self {
            Self::spawn_with_schedule(world, None)
        }

        fn spawn_with_schedule(
            world: WorldState,
            schedule: Option<Vec<scriptbots_runtime::ScheduledConfigPatch>>,
        ) -> Self {
            Self::spawn_with_identity(
                world,
                schedule,
                scriptbots_runtime::HostSessionId::new(0xc017),
            )
        }

        pub(crate) fn spawn_with_identity(
            world: WorldState,
            schedule: Option<Vec<scriptbots_runtime::ScheduledConfigPatch>>,
            session_id: scriptbots_runtime::HostSessionId,
        ) -> Self {
            Self::spawn_with_owner(move || {
                use scriptbots_runtime::{HostCore, HostCoreOptions, PlaybackSnapshot};
                let mut core = HostCore::new(
                    session_id,
                    world,
                    HostCoreOptions {
                        initial_playback: PlaybackSnapshot {
                            paused: true,
                            speed_multiplier: 1.0,
                        },
                        capture_agent_visuals: true,
                        ..HostCoreOptions::default()
                    },
                )
                .expect("test host");
                if let Some(schedule) = schedule {
                    core.install_schedule(schedule, crate::resolve_scheduled_config_patch)
                        .expect("actual application schedule resolver");
                }
                core
            })
        }

        fn spawn_with_owner(
            build_owner: impl FnOnce() -> scriptbots_runtime::HostCore + Send + 'static,
        ) -> Self {
            let (send, receive) = std::sync::mpsc::sync_channel(1);
            let clock_gate = Arc::new(Mutex::new(()));
            let owner_gate = Arc::clone(&clock_gate);
            let (blocked, clock_blocked) = std::sync::mpsc::channel();
            let worker = std::thread::spawn(move || {
                use scriptbots_runtime::{
                    FixedDeadlineHost, ManualInstant,
                    channel::{ChannelHostDriver, ChannelHostOptions, ChannelRunOutcome},
                };
                let (mut driver, port) = ChannelHostDriver::new(
                    FixedDeadlineHost::new(build_owner()),
                    ChannelHostOptions::default(),
                )
                .expect("channel owner");
                send.send(port).expect("publish test port");
                let epoch = std::time::Instant::now();
                let receipt = driver
                    .run(|| {
                        if matches!(
                            owner_gate.try_lock(),
                            Err(std::sync::TryLockError::WouldBlock)
                        ) {
                            blocked.send(()).expect("clock blockage receiver");
                            drop(owner_gate.lock().expect("test clock gate"));
                        }
                        ManualInstant::from_nanos(
                            u64::try_from(epoch.elapsed().as_nanos()).expect("test duration"),
                        )
                    })
                    .expect("test owner run");
                (receipt.outcome == ChannelRunOutcome::HandedOff).then(|| {
                    driver
                        .host()
                        .core()
                        .capture_volatile_host_checkpoint_v1()
                        .expect("retired owner and actual volatile journal")
                        .encode()
                        .expect("encode retired owner")
                })
            });
            Self {
                port: receive.recv().expect("test host rendezvous"),
                worker: Some(worker),
                clock_gate,
                clock_blocked,
            }
        }

        pub(crate) fn handle(&self) -> ControlHandle {
            ControlHandle::new(self.port.clone())
        }

        fn handoff(mut self) -> Vec<u8> {
            let acknowledged = self
                .port
                .handoff_host_checkpoint_v1()
                .expect("actual owner handoff");
            let bytes = self
                .worker
                .take()
                .expect("running test owner")
                .join()
                .expect("original owner retired")
                .expect("handoff retained its paired cut");
            let paired = scriptbots_runtime::VolatileHostCheckpointV1::decode(&bytes)
                .expect("reopen actual paired cut");
            assert_eq!(
                paired.host_checkpoint().encode().unwrap(),
                acknowledged.encode().unwrap(),
                "the joined owner is the acknowledged handoff, without another drive"
            );
            bytes
        }

        pub(crate) fn wait_applied(&self, status: &CommandStatusDto) -> CommandStatusDto {
            let observed = self.wait_finished(&status.command_id);
            assert_eq!(observed.application_state, "applied");
            observed
        }

        pub(crate) fn wait_finished(&self, command_id: &str) -> CommandStatusDto {
            let handle = self.handle();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                let observed = handle
                    .command_status(command_id)
                    .expect("host status")
                    .expect("retained command");
                if observed.application_state != "admitted" && observed.journal_state != "pending" {
                    return observed;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "host command did not finish"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    impl Drop for TestHost {
        fn drop(&mut self) {
            let _ = self.port.submit(CommandEnvelope::new(
                CommandId::new(u128::MAX - 1),
                HostCommand::Shutdown,
            ));
            if let Some(worker) = self.worker.take() {
                let _ = worker.join().expect("test host joined");
            }
        }
    }

    fn handle() -> (ControlHandle, TestHost) {
        let host = TestHost::spawn(
            WorldState::new(ScriptBotsConfig {
                rng_seed: Some(42),
                ..ScriptBotsConfig::default()
            })
            .expect("world"),
        );
        (host.handle(), host)
    }

    #[test]
    fn guarded_config_update_rejects_stale_axes_without_overwriting_the_owner() {
        use scriptbots_runtime::{HostRevisions, RejectionReason};

        let (handle, host) = handle();
        let original = handle.read_snapshot().expect("original paused owner");
        assert!(original.playback.paused);
        let mut proposal = original.config.as_ref().clone();
        proposal.food_growth_rate = 0.075;
        assert_ne!(
            proposal.food_growth_rate.to_bits(),
            original.config.food_growth_rate.to_bits(),
        );
        let mut competing = original.config.as_ref().clone();
        competing.food_max = 0.73;
        assert_ne!(
            competing.food_max.to_bits(),
            original.config.food_max.to_bits(),
        );
        let changed = handle
            .submit_command(ControlCommand::UpdateConfig(Box::new(competing)), None)
            .expect("competing configuration admission");
        host.wait_applied(&changed);
        host.wait_applied(&handle.step().expect("intervening scientific tick"));
        let current = handle.read_snapshot().expect("new owner boundary");
        assert!(current.playback.paused);
        assert!(current.revisions.control > original.revisions.control);
        assert!(current.revisions.scientific > original.revisions.scientific);
        assert!(current.revisions.config > original.revisions.config);
        let digest = host
            .port
            .clone()
            .scientific_digest_v1()
            .expect("current scientific digest");

        // Exercise each owner-side guard separately. A client precheck could
        // not protect an admitted command from these intervening boundaries.
        let stale_guards = [
            (
                HostRevisions {
                    control: original.revisions.control,
                    ..current.revisions
                },
                RejectionReason::ControlRevisionConflict {
                    expected: original.revisions.control,
                    actual: current.revisions.control,
                },
            ),
            (
                HostRevisions {
                    scientific: original.revisions.scientific,
                    ..current.revisions
                },
                RejectionReason::ScientificRevisionConflict {
                    expected: original.revisions.scientific,
                    actual: current.revisions.scientific,
                },
            ),
            (
                HostRevisions {
                    config: original.revisions.config,
                    ..current.revisions
                },
                RejectionReason::ConfigRevisionConflict {
                    expected: original.revisions.config,
                    actual: current.revisions.config,
                },
            ),
        ];
        let mut rejected_ids = BTreeSet::new();
        for (expected, reason) in stale_guards {
            let attempt = handle
                .submit_config_update(Box::new(proposal.clone()), expected)
                .expect("stale proposal has a typed command receipt");
            let id = attempt.command_id();
            assert!(
                rejected_ids.insert(id),
                "each explicit attempt has its own identity",
            );
            let settled = host.wait_finished(&id.to_string());
            assert_eq!(settled.application_state, "rejected");
            let receipt = host
                .port
                .clone()
                .command_status(id)
                .expect("authoritative command lookup")
                .expect("retained stale command");
            assert_eq!(receipt.application(), &ApplicationState::Rejected(reason));
            let unchanged = handle.read_snapshot().expect("unchanged owner");
            assert_eq!(unchanged.revisions, current.revisions);
            assert_eq!(unchanged.world.tick, current.world.tick);
            assert_eq!(unchanged.config, current.config);
            assert_eq!(
                host.port
                    .clone()
                    .scientific_digest_v1()
                    .expect("unchanged digest"),
                digest,
            );
        }

        let mut fresh = current.config.as_ref().clone();
        fresh.food_growth_rate = proposal.food_growth_rate;
        let accepted = handle
            .submit_config_update(Box::new(fresh.clone()), current.revisions)
            .expect("current guarded proposal");
        assert!(!rejected_ids.contains(&accepted.command_id()));
        let applied = host.wait_finished(&accepted.command_id().to_string());
        assert_eq!(applied.application_state, "applied");
        let after = handle.read_snapshot().expect("applied current proposal");
        assert_eq!(after.config.as_ref(), &fresh);
        assert_eq!(after.world.tick, current.world.tick);
        assert_eq!(after.revisions.scientific, current.revisions.scientific);
        assert_eq!(
            Some(after.revisions.config),
            current.revisions.config.checked_next(),
        );
        assert_eq!(
            Some(after.revisions.control),
            current.revisions.control.checked_next(),
        );
    }

    fn experiment_creation_request() -> ExperimentCreateRequest {
        ExperimentCreateRequest {
            experiment_id: None,
            description: Some("actual retry admission regression".to_owned()),
            variants: vec![ExperimentVariantDto {
                variant_id: "baseline".to_owned(),
                brain_family: "mlp".to_owned(),
                config_overrides: Some(serde_json::json!({
                    "world_width": 40,
                    "world_height": 40,
                    "food_cell_size": 10,
                    "population_minimum": 2,
                    "population_spawn_interval": 0
                })),
                interventions: Vec::new(),
            }],
            seeds: vec![7],
            ticks_per_run: Some(2),
            max_concurrency: Some(1),
            idempotency_key: Some("one-actual-sweep".to_owned()),
        }
    }

    fn pending_catalog_definition(root: &Path, id: &str) -> PathBuf {
        // Production DataServices canonicalizes its artifact root before it
        // creates catalog entries. Match that boundary here: on macOS the
        // system temporary directory commonly starts with /var, an alias of
        // /private/var. Keep the catalog's own symlink checks intact.
        let directory = root.canonicalize().unwrap().join("experiments").join(id);
        let runner = crate::experiment_runner::MatchedSeedExperimentRunner::new(
            id,
            crate::experiment_runner::MatchedSeedCohort {
                cohort_id: id.into(),
                seeds: vec![7],
            },
            vec![crate::experiment_runner::ScenarioVariant {
                variant_id: "mlp".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            }],
            2,
            1,
            &directory,
        );
        ExperimentDefinitionV1 {
            schema_version: 1,
            experiment_id: id.into(),
            cohort: runner.cohort.clone(),
            variants: runner.variants.clone(),
            max_ticks: runner.max_ticks,
            max_concurrency: runner.max_concurrency,
            plan_digest: runner.plan_digest().unwrap(),
            created_at_utc: "123456".into(),
            request_identity: Some((format!("key-{id}"), compute_blake3(id.as_bytes()))),
        }
        .persist(&directory)
        .unwrap();
        directory.join("definition.json")
    }

    #[test]
    fn experiment_horizon_refusal_precedes_catalog_and_execution_side_effects() {
        let root = tempfile::tempdir().unwrap().keep();
        let (handle, _host) = handle();
        let handle = handle.with_artifacts_dir(root.clone()).unwrap();
        let services = handle.data_services();
        // Leave an authentic prior admission undiscovered. An invalid new
        // horizon must not even populate its catalog or idempotency entries.
        let saved = pending_catalog_definition(&root, "saved-before-refusal");
        let saved_bytes = fs::read(&saved).unwrap();
        let catalog_entries = || {
            fs::read_dir(root.join("experiments"))
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect::<std::collections::BTreeSet<_>>()
        };
        let before_entries = catalog_entries();
        let before_capacity = *services.execution_capacity.lock().unwrap();
        for ticks in [0, crate::experiment_runner::MAX_TICKS_PER_RUN + 1, u64::MAX] {
            let mut request = experiment_creation_request();
            request.experiment_id = Some("refused-horizon".into());
            request.ticks_per_run = Some(ticks);
            let error = handle.create_experiment(request).unwrap_err();
            assert!(
                matches!(error, ControlError::BadRequest(ref message)
                    if message.contains("ticks_per_run")
                        && message.contains(&format!("found {ticks}"))),
                "{error}"
            );
            assert!(services.experiments.lock().unwrap().is_empty());
            assert!(services.idempotency.lock().unwrap().is_empty());
            assert!(services.artifacts.lock().unwrap().is_empty());
            assert_eq!(
                *services.execution_capacity.lock().unwrap(),
                before_capacity
            );
            assert_eq!(catalog_entries(), before_entries);
            assert_eq!(fs::read(&saved).unwrap(), saved_bytes);
            assert!(!saved.with_file_name("status.json").exists());
            assert!(!root.join("experiments/refused-horizon").exists());
        }
    }

    #[test]
    fn experiment_horizon_admission_preserves_exact_values_and_omitted_default() {
        for (requested, expected) in [
            (Some(1), 1),
            (Some(7), 7),
            (None, 1_000),
            (
                Some(crate::experiment_runner::MAX_TICKS_PER_RUN),
                crate::experiment_runner::MAX_TICKS_PER_RUN,
            ),
        ] {
            let root = tempfile::tempdir().unwrap().keep();
            let (handle, _host) = handle();
            let handle = handle.with_artifacts_dir(root.clone()).unwrap();
            let services = handle.data_services();
            let mut request = experiment_creation_request();
            request.experiment_id = Some("exact-horizon".into());
            request.ticks_per_run = requested;
            let admitted = handle.create_experiment(request).unwrap();
            // Exercise real admission at the upper bound without executing a
            // ten-million-tick test. The short positive controls finish normally.
            if expected > 7 {
                handle.cancel_experiment(&admitted.experiment_id).unwrap();
            }
            {
                let mut jobs = services.experiments.lock().unwrap();
                let job = jobs.get_mut(&admitted.experiment_id).unwrap();
                job.join_worker().unwrap();
                assert_eq!(job.runner.max_ticks, expected);
            }
            let observed = handle.get_experiment(&admitted.experiment_id).unwrap();
            assert_eq!(observed.total_runs, 1);
            assert_eq!(observed.runs[0].target_ticks, expected);
            assert_eq!(observed.failed_runs, 0, "{observed:?}");
            assert!(observed.error_reason.is_none(), "{observed:?}");
            assert!(
                matches!(observed.status.as_str(), "completed" | "cancelled"),
                "{observed:?}"
            );
            if expected <= 7 {
                assert_eq!(observed.status, "completed", "{observed:?}");
                assert_eq!(observed.completed_runs, 1);
                assert_eq!(observed.runs[0].total_ticks, expected);
            }
            let catalog: ExperimentCatalogRecord = serde_json::from_slice(
                &fs::read(root.join("experiments/exact-horizon/definition.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(catalog.definition.max_ticks, expected);
            assert_eq!(catalog.definition.plan_digest, observed.plan_digest);
            let capacity = *services.execution_capacity.lock().unwrap();
            assert_eq!(capacity.reserved_run_slots, 0);
            assert_eq!(capacity.reserved_grid_cells, 0);
            assert_eq!(capacity.reserved_batches, 0);
        }
    }

    #[test]
    fn experiment_total_tick_refusal_preserves_catalog_and_admission_state() {
        let root = tempfile::tempdir().unwrap().keep();
        let (handle, _host) = handle();
        let handle = handle.with_artifacts_dir(root.clone()).unwrap();
        let services = handle.data_services();
        let saved = pending_catalog_definition(&root, "saved-before-work-refusal");
        let saved_bytes = fs::read(&saved).unwrap();
        let before_capacity = *services.execution_capacity.lock().unwrap();
        for concurrency in [1, 4] {
            for two_variants in [false, true] {
                let mut request = experiment_creation_request();
                request.experiment_id = Some("refused-work".into());
                request.seeds = vec![7, 11];
                request.max_concurrency = Some(concurrency);
                request.ticks_per_run = Some(crate::experiment_runner::MAX_TICKS_PER_RUN);
                if two_variants {
                    let mut challenger = request.variants[0].clone();
                    challenger.variant_id = "challenger".into();
                    challenger.brain_family = "dwraon".into();
                    request.variants.push(challenger);
                    request.ticks_per_run =
                        Some(crate::experiment_runner::MAX_BATCH_TOTAL_TICKS / 4 + 1);
                }
                let outcome = handle.create_experiment(request);
                if let Ok(admitted) = &outcome {
                    // A restored admission bug must fail without leaving its
                    // long-running worker alive in the negative-control build.
                    let mut jobs = services.experiments.lock().unwrap();
                    let job = jobs.get_mut(&admitted.experiment_id).unwrap();
                    job.cancel.store(true, std::sync::atomic::Ordering::Release);
                    job.join_worker().unwrap();
                }
                let error = outcome.unwrap_err();
                assert!(
                    matches!(error, ControlError::BadRequest(ref message)
                        if message.contains("scientific tick transitions")
                            && message.contains("per-batch limit")),
                    "{error}"
                );
                assert!(services.experiments.lock().unwrap().is_empty());
                assert!(services.idempotency.lock().unwrap().is_empty());
                assert!(services.artifacts.lock().unwrap().is_empty());
                assert_eq!(
                    *services.execution_capacity.lock().unwrap(),
                    before_capacity
                );
                assert_eq!(fs::read(&saved).unwrap(), saved_bytes);
                assert!(!saved.with_file_name("status.json").exists());
                assert!(!root.join("experiments/refused-work").exists());
                assert_eq!(fs::read_dir(root.join("experiments")).unwrap().count(), 1);
            }
        }

        // A refusal did not consume the exact request key or execution slot.
        // The existing catalog can subsequently recover beside a real small run.
        let mut valid = experiment_creation_request();
        valid.experiment_id = Some("admitted-after-work-refusal".into());
        let admitted = handle.create_experiment(valid).unwrap();
        {
            let mut jobs = services.experiments.lock().unwrap();
            jobs.get_mut(&admitted.experiment_id)
                .unwrap()
                .join_worker()
                .unwrap();
        }
        let completed = handle.get_experiment(&admitted.experiment_id).unwrap();
        assert_eq!(completed.status, "completed", "{completed:?}");
        assert_eq!(completed.completed_runs, 1);
        assert_eq!(completed.failed_runs, 0);
        assert_eq!(completed.runs[0].total_ticks, 2);
        assert_eq!(
            *services.execution_capacity.lock().unwrap(),
            before_capacity
        );
    }

    #[test]
    fn experiment_past_workload_remains_discoverable_but_cannot_resume() {
        let root = tempfile::tempdir().unwrap().keep();
        let (handle, _host) = handle();
        let handle = handle.with_artifacts_dir(root.clone()).unwrap();
        let services = handle.data_services();
        let directory = root
            .canonicalize()
            .unwrap()
            .join("experiments")
            .join("past-workload");
        let runner = crate::experiment_runner::MatchedSeedExperimentRunner::new(
            "past-workload",
            crate::experiment_runner::MatchedSeedCohort {
                cohort_id: "past-workload".into(),
                seeds: vec![7, 11],
            },
            vec![crate::experiment_runner::ScenarioVariant {
                variant_id: "baseline".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
                interventions: Vec::new(),
            }],
            crate::experiment_runner::MAX_TICKS_PER_RUN,
            1,
            &directory,
        );
        // Write a genuine versioned definition for a previously admitted plan.
        // Pending records claim no completed science or missing bundle evidence.
        ExperimentDefinitionV1 {
            schema_version: 1,
            experiment_id: runner.experiment_id.clone(),
            cohort: runner.cohort.clone(),
            variants: runner.variants.clone(),
            max_ticks: runner.max_ticks,
            max_concurrency: runner.max_concurrency,
            plan_digest: runner.plan_digest().unwrap(),
            created_at_utc: "123456".into(),
            request_identity: None,
        }
        .persist(&directory)
        .unwrap();
        let definition = directory.join("definition.json");
        let saved_bytes = fs::read(&definition).unwrap();
        let before_capacity = *services.execution_capacity.lock().unwrap();
        let observed = handle.get_experiment("past-workload").unwrap();
        assert_eq!(observed.status, "stopped");
        assert_eq!(observed.total_runs, 2);
        assert!(observed.runs.iter().all(|run| {
            run.target_ticks == crate::experiment_runner::MAX_TICKS_PER_RUN
                && run.state == "pending"
                && run.total_ticks == 0
        }));
        let error = handle.resume_experiment("past-workload").unwrap_err();
        assert!(
            matches!(error, ControlError::BadRequest(ref message)
                if message.contains("scientific tick transitions")),
            "{error}"
        );
        let after = handle.get_experiment("past-workload").unwrap();
        assert_eq!(
            serde_json::to_value(&after).unwrap(),
            serde_json::to_value(&observed).unwrap()
        );
        assert_eq!(
            *services.execution_capacity.lock().unwrap(),
            before_capacity
        );
        assert_eq!(fs::read(&definition).unwrap(), saved_bytes);
        assert!(!directory.join("status.json").exists());
        assert!(services.artifacts.lock().unwrap().is_empty());
        let jobs = services.experiments.lock().unwrap();
        let job = jobs.get("past-workload").unwrap();
        assert!(job.worker.is_none());
        assert!(!job.cancel.load(std::sync::atomic::Ordering::Acquire));
        assert!(job.last_error.lock().unwrap().is_none());
    }

    pub(crate) fn comparison_catalog_runner(
        root: &Path,
        id: &str,
        seeds: Vec<u64>,
        metabolism_drain: f64,
    ) -> crate::experiment_runner::MatchedSeedExperimentRunner {
        use crate::experiment_runner::{
            MatchedSeedCohort, MatchedSeedExperimentRunner, ScenarioVariant,
        };
        const FIXTURE_TICKS: u64 = 2;
        let defaults = ScriptBotsConfig::default();
        // These values are not exposed as experiment knobs. The default
        // cadences cannot fire during this two-tick fixture, and both default
        // metabolism modifiers are zero, preserving the intended controls.
        assert!(
            defaults.food_respawn_interval == 0
                || u64::from(defaults.food_respawn_interval) > FIXTURE_TICKS
        );
        assert!(u64::from(defaults.reproduction_attempt_interval) > FIXTURE_TICKS);
        assert_eq!(defaults.metabolism_ramp_rate, 0.0);
        assert_eq!(defaults.metabolism_boost_penalty, 0.0);
        // The two conditions differ in actual retained food. No ingestion,
        // locomotion or reproduction can turn that fixture into another outcome.
        let common = BTreeMap::from([
            ("world_width".into(), serde_json::json!(480)),
            ("world_height".into(), serde_json::json!(480)),
            ("food_cell_size".into(), serde_json::json!(120)),
            ("population_minimum".into(), serde_json::json!(0)),
            ("population_spawn_interval".into(), serde_json::json!(0)),
            ("food_max".into(), serde_json::json!(10.0)),
            ("food_growth_rate".into(), serde_json::json!(0.0)),
            ("food_decay_rate".into(), serde_json::json!(0.0)),
            ("food_diffusion_rate".into(), serde_json::json!(0.0)),
            ("food_intake_rate".into(), serde_json::json!(0.0)),
            ("food_waste_rate".into(), serde_json::json!(0.0)),
            ("bot_speed".into(), serde_json::json!(0.0)),
            ("movement_drain".into(), serde_json::json!(0.0)),
            (
                "metabolism_drain".into(),
                serde_json::json!(metabolism_drain),
            ),
            ("temperature_discomfort_rate".into(), serde_json::json!(0.0)),
            ("spike_damage".into(), serde_json::json!(0.0)),
            ("spike_energy_cost".into(), serde_json::json!(0.0)),
        ]);
        let variants = [("baseline", 0.0), ("challenger", 10.0)]
            .into_iter()
            .map(|(variant_id, initial_food)| {
                let mut config_overrides = common.clone();
                config_overrides.insert("initial_food".into(), serde_json::json!(initial_food));
                ScenarioVariant {
                    variant_id: variant_id.into(),
                    brain_family: "mlp".into(),
                    config_overrides,
                    interventions: Vec::new(),
                }
            })
            .collect();
        let directory = root.canonicalize().unwrap().join("experiments").join(id);
        let runner = MatchedSeedExperimentRunner::new(
            id,
            MatchedSeedCohort {
                cohort_id: id.into(),
                seeds,
            },
            variants,
            FIXTURE_TICKS,
            1,
            &directory,
        );
        ExperimentDefinitionV1 {
            schema_version: 1,
            experiment_id: id.into(),
            cohort: runner.cohort.clone(),
            variants: runner.variants.clone(),
            max_ticks: runner.max_ticks,
            max_concurrency: runner.max_concurrency,
            plan_digest: runner.plan_digest().unwrap(),
            created_at_utc: "123456".into(),
            request_identity: None,
        }
        .persist(&directory)
        .unwrap();
        runner
    }

    #[test]
    fn experiment_run_artifacts_download_complete_portable_results_despite_broken_peer() {
        let root = tempfile::tempdir().unwrap();
        let (handle, host) = handle();
        let handle = handle
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap();
        let runner = comparison_catalog_runner(root.path(), "run-download", vec![7], 0.0);
        let status_path = runner.output_dir.join("status.json");
        let completed = runner.execute_batch(&status_path).unwrap();
        let run = &completed.runs[0];
        let peer = &completed.runs[1];
        let peer_path = Path::new(peer.bundle_path.as_ref().unwrap()).join("exports/summary.csv");
        let peer_before = fs::read(&peer_path).unwrap();
        fs::write(&peer_path, b"broken companion endpoint\n").unwrap();
        let status_before = fs::read(&status_path).unwrap();
        let live_before = host.port.scientific_digest_v1().unwrap();

        // Recover directly from the durable catalog; no global artifact scan or
        // successful prior status/download request is needed.
        let inventory = handle
            .get_experiment_run_artifacts("run-download", &run.run_id)
            .unwrap();
        assert_eq!(inventory.experiment_id, completed.experiment_id);
        assert_eq!(inventory.plan_digest, completed.plan_digest);
        assert_eq!(inventory.status_generation, completed.generation);
        assert_eq!(inventory.run.run_id, run.run_id);
        assert_eq!(inventory.run.variant_id, run.variant_id);
        assert_eq!(inventory.run.seed, run.seed);
        assert_eq!(inventory.run.final_digest, run.final_digest);
        assert_eq!(inventory.run.state, "completed");
        assert_eq!(inventory.run.total_ticks, runner.max_ticks);
        assert_eq!(inventory.artifacts.len(), 4);
        assert_eq!(
            inventory.total_bytes,
            inventory
                .artifacts
                .iter()
                .map(|file| file.byte_size)
                .sum::<u64>()
        );
        let portable = tempfile::tempdir().unwrap();
        let prefix = format!("experiments/run-download/{}/", run.run_id);
        for artifact in &inventory.artifacts {
            let relative = artifact.relative_path.strip_prefix(&prefix).unwrap();
            let (metadata, bytes) = handle.read_artifact_bytes(&artifact.artifact_id).unwrap();
            assert_eq!(&metadata, artifact);
            assert_eq!(compute_blake3(&bytes), artifact.checksum_blake3);
            assert_eq!(compute_sha256(&bytes), artifact.checksum_sha256);
            let output = portable.path().join(relative);
            fs::create_dir_all(output.parent().unwrap()).unwrap();
            fs::write(output, bytes).unwrap();
        }
        let verified = crate::experiment_runner::verify_portable_run_bundle(portable.path())
            .unwrap()
            .final_state
            .unwrap();
        assert_eq!(verified.run_id, run.run_id);
        assert_eq!(Some(verified.scientific_digest), run.final_digest);
        assert!(
            handle
                .get_experiment_run_artifacts("run-download", &peer.run_id)
                .is_err()
        );

        // A cold catalog and a retargeted active root both preserve the selected
        // bundle's source identity; an unrelated corrupt peer still cannot block it.
        handle.data_services.artifacts.lock().unwrap().clear();
        handle.data_services.experiments.lock().unwrap().clear();
        let recovered = handle
            .get_experiment_run_artifacts("run-download", &run.run_id)
            .unwrap();
        assert_eq!(recovered.artifacts, inventory.artifacts);
        let second_root = tempfile::tempdir().unwrap();
        let handle = handle
            .with_artifacts_dir(second_root.path().to_path_buf())
            .unwrap();
        let retained = handle
            .get_experiment_run_artifacts("run-download", &run.run_id)
            .unwrap();
        assert_eq!(retained.artifacts, inventory.artifacts);
        assert_eq!(fs::read(&status_path).unwrap(), status_before);
        assert_ne!(fs::read(&peer_path).unwrap(), peer_before);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), live_before);
    }

    #[test]
    fn experiment_run_artifacts_are_immutable_as_unfinished_companions_complete() {
        use crate::experiment_runner::RunState;
        let root = tempfile::tempdir().unwrap();
        let (handle, host) = handle();
        let handle = handle
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap();
        let runner = comparison_catalog_runner(root.path(), "partial-download", vec![7], 0.0);
        let path = runner.output_dir.join("status.json");
        let mut status = runner.load_or_create_status(&path).unwrap();
        let first_id = status.runs[0].run_id.clone();
        let second_id = status.runs[1].run_id.clone();
        assert!(matches!(
            handle.get_experiment_run_artifacts("partial-download", &first_id),
            Err(ControlError::Conflict(_))
        ));
        status.runs[0].state = RunState::Running;
        runner.save_status(&path, &mut status).unwrap();
        status.runs[0] = runner.execute_single_run(&runner.variants[0], 7).unwrap();
        status.completed_runs = 1;
        runner.save_status(&path, &mut status).unwrap();
        let live = host.port.scientific_digest_v1().unwrap();
        let first = handle
            .get_experiment_run_artifacts("partial-download", &first_id)
            .unwrap();
        assert_eq!(first.status_generation, status.generation);
        assert!(matches!(
            handle.get_experiment_run_artifacts("partial-download", &second_id),
            Err(ControlError::Conflict(_))
        ));

        // A running peer has no saved checkpoint at all. Inventory discovery
        // must not attempt to recover or verify that unrelated owner.
        status.runs[1].state = RunState::Running;
        runner.save_status(&path, &mut status).unwrap();
        let running = handle
            .get_experiment_run_artifacts("partial-download", &first_id)
            .unwrap();
        assert_eq!(running.artifacts, first.artifacts);
        assert!(matches!(
            handle.get_experiment_run_artifacts("partial-download", &second_id),
            Err(ControlError::Conflict(_))
        ));
        status.runs[1] = runner.execute_single_run(&runner.variants[1], 7).unwrap();
        status.completed_runs = 2;
        runner.save_status(&path, &mut status).unwrap();
        let saved_status = fs::read(&path).unwrap();
        let finished = handle
            .get_experiment_run_artifacts("partial-download", &first_id)
            .unwrap();
        assert!(finished.status_generation > first.status_generation);
        assert_eq!(finished.artifacts, first.artifacts);
        assert_eq!(finished.total_bytes, first.total_bytes);
        assert_eq!(finished.run.final_digest, first.run.final_digest);
        assert_eq!(fs::read(&path).unwrap(), saved_status);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
    }

    #[test]
    fn experiment_run_artifacts_refuse_foreign_ids_and_self_consistent_owner_substitution() {
        let root = tempfile::tempdir().unwrap();
        let (handle, _host) = handle();
        let handle = handle
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap();
        let runner = comparison_catalog_runner(root.path(), "identity-download", vec![7], 0.0);
        let path = runner.output_dir.join("status.json");
        let completed = runner.execute_batch(&path).unwrap();
        let run = &completed.runs[0];
        for invalid in ["", "../outside", &"x".repeat(129)] {
            assert!(matches!(
                handle.get_experiment_run_artifacts("identity-download", invalid),
                Err(ControlError::BadRequest(_))
            ));
            assert!(matches!(
                handle.get_experiment_run_artifacts(invalid, &run.run_id),
                Err(ControlError::BadRequest(_))
            ));
        }
        for (experiment_id, run_id) in [
            ("missing", run.run_id.as_str()),
            ("identity-download", "another-batch-baseline-seed7"),
        ] {
            assert!(matches!(
                handle.get_experiment_run_artifacts(experiment_id, run_id),
                Err(ControlError::NotFound(_))
            ));
        }
        let good = handle
            .get_experiment_run_artifacts("identity-download", &run.run_id)
            .unwrap();
        let directory = Path::new(run.bundle_path.as_ref().unwrap());
        let manifest_path = directory.join("bundle_manifest.json");
        let manifest_before = fs::read(&manifest_path).unwrap();
        let checkpoint_path = directory.join("checkpoints/final.checkpoint");
        let checkpoint_before = fs::read(&checkpoint_path).unwrap();
        let foreign = fs::read(
            Path::new(completed.runs[1].bundle_path.as_ref().unwrap())
                .join("checkpoints/final.checkpoint"),
        )
        .unwrap();
        let mut bundle: scriptbots_storage::RunBundleV1 =
            serde_json::from_slice(&manifest_before).unwrap();
        let entry = bundle
            .artifacts
            .iter_mut()
            .find(|entry| entry.relative_path == "checkpoints/final.checkpoint")
            .unwrap();
        entry.bytes_len = foreign.len() as u64;
        entry.blake3_hex = compute_blake3(&foreign);
        fs::write(&checkpoint_path, foreign).unwrap();
        fs::write(&manifest_path, serde_json::to_vec_pretty(&bundle).unwrap()).unwrap();
        // Byte integrity alone now succeeds, but the restored owner belongs to
        // another arm and cannot authorize a download inventory for this run.
        scriptbots_storage::bundle::verify_run_bundle_bounded(
            directory,
            scriptbots_storage::bundle::RunBundleVerificationLimits {
                max_manifest_bytes: crate::experiment_runner::MAX_STATUS_BYTES,
                max_artifacts: crate::experiment_runner::MAX_BUNDLE_ARTIFACTS,
                max_artifact_bytes: crate::experiment_runner::MAX_BUNDLE_ARTIFACT_BYTES,
                max_total_artifact_bytes: crate::experiment_runner::MAX_BUNDLE_TOTAL_ARTIFACT_BYTES,
            },
        )
        .unwrap();
        assert!(
            handle
                .get_experiment_run_artifacts("identity-download", &run.run_id)
                .is_err()
        );
        fs::write(&checkpoint_path, checkpoint_before).unwrap();
        fs::write(&manifest_path, manifest_before).unwrap();
        let restored = handle
            .get_experiment_run_artifacts("identity-download", &run.run_id)
            .unwrap();
        assert_eq!(restored.artifacts, good.artifacts);
    }

    #[test]
    fn experiment_comparison_uses_verified_matched_endpoints_without_advancing_live_science() {
        let root = tempfile::tempdir().unwrap();
        let (handle, host) = handle();
        let handle = handle
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap();
        let runner = comparison_catalog_runner(root.path(), "comparison", vec![29, 7, 11], 0.0);
        let path = runner.output_dir.join("status.json");
        assert!(matches!(
            handle.compare_experiment_variants(
                "comparison",
                "baseline",
                "challenger",
                "total_food"
            ),
            Err(ControlError::Conflict(_))
        ));
        let completed = runner.execute_batch(&path).unwrap();
        assert_eq!(
            completed.completed_runs, completed.total_runs,
            "{completed:?}"
        );
        let before = fs::read(&path).unwrap();
        let live = host.port.scientific_digest_v1().unwrap();
        let forward = handle
            .compare_experiment_variants("comparison", "baseline", "challenger", "total_food")
            .unwrap();
        assert_eq!(forward.matched_seeds, [7, 11, 29]);
        assert_eq!(forward.endpoint_tick, runner.max_ticks);
        assert_eq!(forward.plan_digest, completed.plan_digest);
        assert_eq!(forward.status_generation, completed.generation);
        assert!(
            forward.mean_difference > 0.0,
            "different food conditions collapsed: {forward:?}"
        );
        assert!(forward.ci_95.is_some());
        assert!(forward.p_value.is_some_and(|p| (0.0..=1.0).contains(&p)));
        assert!(!forward.descriptive_only);
        let reverse = handle
            .compare_experiment_variants("comparison", "challenger", "baseline", "total_food")
            .unwrap();
        assert_eq!(reverse.mean_difference, -forward.mean_difference);
        assert_eq!(reverse.matched_seeds, forward.matched_seeds);
        let population = handle
            .compare_experiment_variants("comparison", "baseline", "challenger", "alive_agents")
            .unwrap();
        assert_eq!(population.mean_difference, 0.0);
        assert_eq!(population.ci_95, Some((0.0, 0.0)));
        assert_eq!(population.standardized_effect, None);
        assert_eq!(
            population.undefined_effect_reason.as_deref(),
            Some("zero_variance")
        );
        assert_eq!(population.p_value, Some(1.0));
        assert_eq!(population.significant, Some(false));
        assert!(population.underpowered_reason.is_some());
        assert_eq!(
            serde_json::from_value::<ExperimentComparisonDto>(
                serde_json::to_value(&forward).unwrap()
            )
            .unwrap(),
            forward
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
    }

    #[test]
    fn experiment_comparison_preserves_descriptive_and_extinct_endpoint_boundaries() {
        let root = tempfile::tempdir().unwrap();
        let (handle, _host) = handle();
        let handle = handle
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap();
        let single = comparison_catalog_runner(root.path(), "single-pair", vec![7], 0.0);
        single
            .execute_batch(&single.output_dir.join("status.json"))
            .unwrap();
        let descriptive = handle
            .compare_experiment_variants("single-pair", "baseline", "challenger", "total_food")
            .unwrap();
        assert!(descriptive.mean_difference > 0.0);
        assert!(descriptive.descriptive_only);
        assert_eq!(descriptive.matched_seeds, [7]);
        assert_eq!(descriptive.ci_95, None);
        assert_eq!(descriptive.standardized_effect, None);
        assert_eq!(descriptive.p_value, None);
        assert_eq!(descriptive.adjusted_p_value, None);
        assert_eq!(descriptive.significant, None);
        assert_eq!(
            descriptive.undefined_effect_reason.as_deref(),
            Some("insufficient_pairs(have=1, need=2)")
        );

        let extinct = comparison_catalog_runner(root.path(), "extinct-pair", vec![7, 11], 1.0);
        let completed = extinct
            .execute_batch(&extinct.output_dir.join("status.json"))
            .unwrap();
        for run in &completed.runs {
            let observed = crate::experiment_runner::verify_portable_run_bundle(Path::new(
                run.bundle_path.as_ref().unwrap(),
            ))
            .unwrap()
            .final_state
            .unwrap();
            assert_eq!(
                observed.alive_agents, 0,
                "extinction fixture did not become extinct"
            );
        }
        let error = handle
            .compare_experiment_variants("extinct-pair", "baseline", "challenger", "average_energy")
            .unwrap_err();
        assert!(error.to_string().contains("unavailable"), "{error}");
        let zero = handle
            .compare_experiment_variants("extinct-pair", "baseline", "challenger", "total_energy")
            .unwrap();
        assert_eq!(zero.mean_difference, 0.0);
        assert_eq!(zero.standardized_effect, None);
    }

    #[test]
    fn experiment_comparison_bounds_admission_and_rejects_changed_plan_or_bundle_evidence() {
        use crate::experiment_runner::{ExperimentBatchStatus, MAX_RUN_COMPONENT_BYTES};
        let root = tempfile::tempdir().unwrap();
        let (handle, _host) = handle();
        let handle = handle
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap();
        let runner = comparison_catalog_runner(root.path(), "guarded", vec![11, 7], 0.0);
        let path = runner.output_dir.join("status.json");
        let completed = runner.execute_batch(&path).unwrap();
        assert_eq!(
            completed.completed_runs, completed.total_runs,
            "{completed:?}"
        );
        let compare = || {
            handle.compare_experiment_variants("guarded", "baseline", "challenger", "total_food")
        };
        let expected = compare().unwrap();
        let oversized = "x".repeat(MAX_RUN_COMPONENT_BYTES + 1);
        for invalid in ["", "../outside", oversized.as_str()] {
            for result in [
                handle.compare_experiment_variants(invalid, "baseline", "challenger", "total_food"),
                handle.compare_experiment_variants("guarded", invalid, "challenger", "total_food"),
                handle.compare_experiment_variants("guarded", "baseline", invalid, "total_food"),
            ] {
                assert!(matches!(result, Err(ControlError::BadRequest(_))));
            }
        }
        for (baseline, challenger, metric) in [
            ("baseline", "baseline", "total_food"),
            ("baseline", "challenger", "fitness"),
            ("baseline", "challenger", ""),
        ] {
            assert!(matches!(
                handle.compare_experiment_variants("guarded", baseline, challenger, metric),
                Err(ControlError::BadRequest(_))
            ));
        }
        assert!(matches!(
            handle.compare_experiment_variants("guarded", "unknown", "challenger", "total_food"),
            Err(ControlError::NotFound(_))
        ));
        let held = handle.data_services.experiment_comparison.lock().unwrap();
        assert!(matches!(
            handle.clone().compare_experiment_variants(
                "guarded",
                "baseline",
                "challenger",
                "total_food"
            ),
            Err(ControlError::ExperimentComparisonBusy)
        ));
        drop(held);
        assert_eq!(compare().unwrap(), expected);

        let original_status = fs::read(&path).unwrap();
        for defect in ["seed", "horizon", "missing_run"] {
            let mut changed: ExperimentBatchStatus =
                serde_json::from_slice(&original_status).unwrap();
            match defect {
                "seed" => changed.runs.last_mut().unwrap().seed = 999,
                "horizon" => changed.runs.last_mut().unwrap().total_ticks += 1,
                "missing_run" => {
                    changed.runs.pop();
                }
                _ => unreachable!(),
            }
            let bytes = serde_json::to_vec(&changed).unwrap();
            fs::write(&path, &bytes).unwrap();
            assert!(compare().is_err(), "accepted altered {defect} evidence");
            assert_eq!(
                fs::read(&path).unwrap(),
                bytes,
                "analysis rewrote refused evidence"
            );
            fs::write(&path, &original_status).unwrap();
        }
        assert_eq!(
            compare().unwrap(),
            expected,
            "refusal retained the admission permit"
        );

        // Rehash a false endpoint so the ordinary byte-integrity verifier accepts
        // it. Comparison must still reconstruct the actual final checkpoint.
        let bundle_dir = Path::new(completed.runs[0].bundle_path.as_ref().unwrap());
        let summary_path = bundle_dir.join("exports/summary.csv");
        let original_summary = fs::read(&summary_path).unwrap();
        let mut summary: crate::experiment_runner::ExperimentEndpointSummaryV2 =
            csv::Reader::from_reader(original_summary.as_slice())
                .deserialize()
                .next()
                .unwrap()
                .unwrap();
        summary.total_food += 1.0;
        let mut csv = csv::WriterBuilder::new()
            .terminator(csv::Terminator::Any(b'\n'))
            .from_writer(Vec::new());
        csv.serialize(summary).unwrap();
        let changed_summary = csv.into_inner().unwrap();
        fs::write(&summary_path, &changed_summary).unwrap();
        let manifest_path = bundle_dir.join("bundle_manifest.json");
        let original_manifest = fs::read(&manifest_path).unwrap();
        let mut manifest: scriptbots_storage::RunBundleV1 =
            serde_json::from_slice(&original_manifest).unwrap();
        let entry = manifest
            .artifacts
            .iter_mut()
            .find(|entry| entry.relative_path == "exports/summary.csv")
            .unwrap();
        entry.bytes_len = u64::try_from(changed_summary.len()).unwrap();
        entry.blake3_hex = compute_blake3(&changed_summary);
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        scriptbots_storage::verify_run_bundle(bundle_dir)
            .expect("false summary remains byte-valid");
        let error = compare().unwrap_err();
        assert!(
            error.to_string().contains("restored final checkpoint"),
            "{error}"
        );
        assert_eq!(fs::read(&summary_path).unwrap(), changed_summary);
        fs::write(&summary_path, original_summary).unwrap();
        fs::write(&manifest_path, original_manifest).unwrap();
        assert_eq!(compare().unwrap(), expected);
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one real owner teardown, cold recovery and uninterrupted comparison journey"
    )]
    fn final_control_handle_drop_joins_real_experiments_and_preserves_their_results() {
        use std::sync::atomic::Ordering;
        use std::time::{Duration, Instant};

        let root = tempfile::tempdir().unwrap();
        let oracle_root = tempfile::tempdir().unwrap();
        let (control, host) = handle();
        let control = control
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap();
        let mut request = experiment_creation_request();
        request.experiment_id = Some("drop-active".into());
        request.idempotency_key = Some("drop-active-key".into());
        request.seeds = vec![7, 9];
        request.max_concurrency = Some(2);
        request.ticks_per_run = Some(64);
        control.create_experiment(request.clone()).unwrap();

        let owner = Arc::downgrade(control.data_services());
        let capacity = Arc::clone(&control.data_services.execution_capacity);
        let (cancel, running) = {
            let jobs = control.data_services.experiments.lock().unwrap();
            let job = &jobs["drop-active"];
            (Arc::clone(&job.cancel), Arc::clone(&job.running))
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let status = control.get_experiment("drop-active").unwrap();
            assert!(
                matches!(status.status.as_str(), "pending" | "running" | "completed"),
                "{status:?}"
            );
            if status.runs.iter().any(|run| {
                run.checkpoint_progress
                    .observed_tick()
                    .is_some_and(|tick| tick > 0)
            }) {
                break;
            }
            assert!(Instant::now() < deadline, "a real owner must advance");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(!cancel.load(Ordering::Acquire));

        drop(control);
        // Clean up the old implementation before failing its regression: a worker
        // retaining DataServices otherwise keeps running after the assertion unwinds.
        let retained_by_worker = if let Some(leaked) = owner.upgrade() {
            leaked.shutdown_experiments().unwrap();
            true
        } else {
            false
        };
        assert!(
            !retained_by_worker,
            "dropping the final external owner left the real experiment retaining its service"
        );
        assert!(cancel.load(Ordering::Acquire));
        assert!(!running.load(Ordering::Acquire));
        let released = *capacity.lock().unwrap();
        assert_eq!(released.reserved_batches, 0);
        assert_eq!(released.reserved_run_slots, 0);
        assert_eq!(released.reserved_grid_cells, 0);

        // A new service must acquire the same source root and discover the exact
        // retained cuts. Reading their identities cannot restart scientific work.
        let reopened = host
            .handle()
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap();
        let stopped = reopened.get_experiment("drop-active").unwrap();
        assert!(matches!(stopped.status.as_str(), "stopped" | "completed"));
        assert!(
            stopped
                .runs
                .iter()
                .all(|run| matches!(run.state.as_str(), "running" | "completed"))
        );
        let directory = root.path().join("experiments/drop-active");
        let state_before_read = fs::read(directory.join("status.json")).unwrap();
        let cuts = stopped
            .runs
            .iter()
            .map(|run| {
                let path = directory.join(format!("{}.checkpoint", run.run_id));
                (path.clone(), fs::read(path).unwrap())
            })
            .collect::<Vec<_>>();
        for _ in &stopped.runs {
            reopened.get_experiment("drop-active").unwrap();
        }
        let observed = reopened.get_experiment("drop-active").unwrap();
        assert!(observed.runs.iter().any(|run| {
            run.checkpoint_progress
                .observed_tick()
                .is_some_and(|tick| tick > 0)
        }));
        for run in &observed.runs {
            assert!(
                run.checkpoint_progress
                    .observed_tick()
                    .is_some_and(|tick| tick <= run.target_ticks)
            );
        }
        assert_eq!(
            fs::read(directory.join("status.json")).unwrap(),
            state_before_read
        );
        for (path, bytes) in cuts {
            assert_eq!(fs::read(path).unwrap(), bytes);
        }

        let wait_completed = |control: &ControlHandle| {
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                let status = control.get_experiment("drop-active").unwrap();
                if status.status == "completed" {
                    break status;
                }
                assert!(
                    matches!(status.status.as_str(), "pending" | "running"),
                    "unexpected experiment result: {status:?}"
                );
                assert!(Instant::now() < deadline, "real experiment did not complete");
                std::thread::sleep(Duration::from_millis(1));
            }
        };
        if stopped.status == "stopped" {
            reopened.resume_experiment("drop-active").unwrap();
        }
        let resumed = wait_completed(&reopened);
        let oracle = host
            .handle()
            .with_artifacts_dir(oracle_root.path().to_path_buf())
            .unwrap();
        oracle.create_experiment(request).unwrap();
        let uninterrupted = wait_completed(&oracle);
        assert_eq!(resumed.completed_runs, 2);
        assert_eq!(resumed.failed_runs, 0);
        for (actual, expected) in resumed.runs.iter().zip(&uninterrupted.runs) {
            assert_eq!(actual.run_id, expected.run_id);
            assert_eq!(actual.seed, expected.seed);
            assert_eq!(actual.total_ticks, 64);
            assert_eq!(actual.final_digest, expected.final_digest);
            assert!(actual.final_digest.is_some());
            crate::experiment_runner::verify_portable_run_bundle(
                Path::new(actual.bundle_path.as_ref().unwrap()),
            )
            .unwrap();
        }
    }

    #[test]
    fn final_service_drop_cancels_all_workers_before_joining_despite_poison_or_panic() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::time::{Duration, Instant};

        let root = tempfile::tempdir().unwrap();
        let services = DataServices::new(root.path().to_path_buf());
        for id in ["a-panic", "z-sibling"] {
            pending_catalog_definition(root.path(), id);
            services.recover_experiment(id).unwrap();
        }
        let capacity = Arc::clone(&services.execution_capacity);
        let sibling_finished = Arc::new(AtomicBool::new(false));
        let observed_all_cancellations = Arc::new(AtomicUsize::new(0));
        let cancellations = {
            let mut jobs = services.experiments.lock().unwrap();
            let cancellations = jobs
                .values()
                .map(|job| Arc::clone(&job.cancel))
                .collect::<Vec<_>>();
            for (index, job) in jobs.values_mut().enumerate() {
                let flags = cancellations.clone();
                let finished = Arc::clone(&sibling_finished);
                let observed = Arc::clone(&observed_all_cancellations);
                let mut reservation = services
                    .reserve_experiment_execution(&job.runner.experiment_id, 1, 1)
                    .unwrap();
                job.running.store(true, Ordering::Release);
                reservation.running = Some(Arc::clone(&job.running));
                job.worker = Some(std::thread::spawn(move || {
                    let _reservation = reservation;
                    let deadline = Instant::now() + Duration::from_secs(30);
                    while !flags.iter().all(|flag| flag.load(Ordering::Acquire)) {
                        assert!(Instant::now() < deadline, "all siblings must be cancelled");
                        std::thread::yield_now();
                    }
                    observed.fetch_add(1, Ordering::AcqRel);
                    assert_ne!(index, 0, "injected experiment worker panic during owner drop");
                    finished.store(true, Ordering::Release);
                }));
            }
            cancellations
        };
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _jobs = services.experiments.lock().unwrap();
                panic!("injected registry poison before owner drop");
            }))
            .is_err()
        );
        drop(services);
        let joined_all = sibling_finished.load(Ordering::Acquire)
            && observed_all_cancellations.load(Ordering::Acquire) == 2
            && capacity.lock().unwrap().reserved_batches == 0;
        if !joined_all {
            // The old owner drops its JoinHandles. Stop those regression workers
            // before reporting the missing teardown, rather than leaking them.
            for flag in cancellations {
                flag.store(true, Ordering::Release);
            }
            let deadline = Instant::now() + Duration::from_secs(30);
            while capacity.lock().unwrap().reserved_batches != 0 {
                assert!(Instant::now() < deadline, "regression workers did not stop");
                std::thread::yield_now();
            }
        }
        assert!(joined_all, "owner drop did not cancel and join every sibling");
        let released = *capacity.lock().unwrap();
        assert_eq!(released.reserved_batches, 0);
        assert_eq!(released.reserved_run_slots, 0);
        assert_eq!(released.reserved_grid_cells, 0);
    }

    #[test]
    fn control_runtime_shutdown_and_drop_join_active_experiments() {
        use std::time::{Duration, Instant};

        for explicit_shutdown in [true, false] {
            let root = tempfile::tempdir().unwrap();
            let (_unused_handle, host) = handle();
            let (runtime, _submit) = crate::ControlRuntime::launch(
                host.port.clone(),
                crate::ControlServerConfig {
                    rest_enabled: false,
                    mcp_transport: crate::servers::McpTransportConfig::Disabled,
                    ..crate::ControlServerConfig::default()
                },
            )
            .unwrap();
            let retained = runtime
                .control_handle()
                .unwrap()
                .with_artifacts_dir(root.path().to_path_buf())
                .unwrap();
            let mut request = experiment_creation_request();
            request.experiment_id = Some("shutdown-active".into());
            request.idempotency_key = Some("shutdown-active-key".into());
            request.seeds = vec![7, 9];
            request.max_concurrency = Some(2);
            request.ticks_per_run = Some(10_000);
            let created = retained.create_experiment(request.clone()).unwrap();
            let directory = root.path().join("experiments/shutdown-active");
            let mut cuts = created
                .runs
                .iter()
                .map(|run| {
                    (
                        directory.join(format!("{}.checkpoint", run.run_id)),
                        std::collections::BTreeSet::new(),
                    )
                })
                .collect::<Vec<_>>();
            let deadline = Instant::now() + Duration::from_secs(30);
            while cuts.iter().any(|(_, seen)| seen.len() < 2) {
                for (path, seen) in &mut cuts {
                    if let Ok(bytes) = fs::read(path)
                        && let Some(end) = bytes.len().checked_sub(blake3::OUT_LEN)
                        && blake3::hash(&bytes[..end]).as_bytes() == &bytes[end..]
                    {
                        seen.insert(compute_blake3(&bytes));
                    }
                }
                assert!(
                    Instant::now() < deadline,
                    "both experiment owners must advance"
                );
                std::thread::sleep(Duration::from_millis(1));
            }

            if explicit_shutdown {
                runtime.shutdown().unwrap();
            } else {
                drop(runtime);
            }
            let stopped = retained.get_experiment("shutdown-active").unwrap();
            assert_eq!(stopped.status, "cancelled");
            // Cancelled records retain the resumable state; progress is in the
            // distinct, checksummed continuation cuts observed above.
            assert!(
                stopped
                    .runs
                    .iter()
                    .all(|run| run.state == "running" && run.total_ticks == 0)
            );
            for run in &stopped.runs {
                assert_eq!(run.target_ticks, 10_000);
                let crate::experiment_runner::ExperimentCheckpointProgress::Observed {
                    tick,
                    checkpoint_digest,
                    source,
                } = &run.checkpoint_progress
                else {
                    panic!("cancelled owner lost its published progress: {run:?}");
                };
                assert!(*tick > 0 && *tick < run.target_ticks);
                assert_eq!(
                    *source,
                    crate::experiment_runner::CheckpointProgressSource::Published
                );
                let checkpoint = directory.join(format!("{}.checkpoint", run.run_id));
                assert_eq!(
                    *checkpoint_digest,
                    compute_blake3(&fs::read(checkpoint).unwrap())
                );
                let encoded = serde_json::to_value(run).unwrap();
                assert_eq!(encoded["total_ticks"], 0);
                assert_eq!(encoded["checkpoint_progress"]["state"], "observed");
                assert_eq!(encoded["checkpoint_progress"]["tick"], *tick);
            }
            let capacity = retained
                .list_experiments(None, None)
                .unwrap()
                .execution_capacity;
            assert_eq!(capacity.reserved_run_slots, 0);
            assert_eq!(capacity.reserved_batches, 0);
            assert!(
                retained
                    .data_services
                    .experiments
                    .lock()
                    .unwrap()
                    .values()
                    .all(|job| job.worker.is_none())
            );

            let saved_status = fs::read(directory.join("status.json")).unwrap();
            let saved_cuts = cuts
                .iter()
                .map(|(path, _)| (path.clone(), fs::read(path).unwrap()))
                .collect::<Vec<_>>();
            assert!(matches!(
                retained.resume_experiment("shutdown-active"),
                Err(ControlError::ExperimentShuttingDown)
            ));
            // An exact creation retry remains a read of the admitted identity.
            assert_eq!(
                retained
                    .create_experiment(request.clone())
                    .unwrap()
                    .experiment_id,
                created.experiment_id
            );
            request.experiment_id = Some("after-shutdown".into());
            request.idempotency_key = Some("after-shutdown-key".into());
            assert!(matches!(
                retained.create_experiment(request),
                Err(ControlError::ExperimentShuttingDown)
            ));
            assert!(!root.path().join("experiments/after-shutdown").exists());
            retained.data_services.shutdown_experiments().unwrap();
            assert_eq!(
                fs::read(directory.join("status.json")).unwrap(),
                saved_status
            );
            for (path, bytes) in saved_cuts {
                assert_eq!(fs::read(path).unwrap(), bytes);
            }
        }
    }

    #[test]
    fn experiment_shutdown_joins_siblings_after_an_injected_worker_panic() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::time::{Duration, Instant};

        let root = tempfile::tempdir().unwrap();
        let services = DataServices::new(root.path().to_path_buf());
        for id in ["a-panic", "z-sibling"] {
            pending_catalog_definition(root.path(), id);
            services.recover_experiment(id).unwrap();
        }
        let sibling_finished = Arc::new(AtomicBool::new(false));
        let observed_all_cancellations = Arc::new(AtomicUsize::new(0));
        {
            let mut jobs = services.experiments.lock().unwrap();
            let cancellation = jobs
                .values()
                .map(|job| Arc::clone(&job.cancel))
                .collect::<Vec<_>>();
            for (index, job) in jobs.values_mut().enumerate() {
                let flags = cancellation.clone();
                let finished = Arc::clone(&sibling_finished);
                let observed = Arc::clone(&observed_all_cancellations);
                let mut reservation = services
                    .reserve_experiment_execution(&job.runner.experiment_id, 1, 1)
                    .unwrap();
                job.running.store(true, Ordering::Release);
                reservation.running = Some(Arc::clone(&job.running));
                // Inject thread lifecycle behavior only; the test above runs real science.
                job.worker = Some(std::thread::spawn(move || {
                    let _reservation = reservation;
                    let deadline = Instant::now() + Duration::from_secs(30);
                    while !flags.iter().all(|flag| flag.load(Ordering::Acquire)) {
                        assert!(
                            Instant::now() < deadline,
                            "all jobs must be cancelled before joining"
                        );
                        std::thread::yield_now();
                    }
                    observed.fetch_add(1, Ordering::AcqRel);
                    assert_ne!(index, 0, "injected experiment worker panic");
                    finished.store(true, Ordering::Release);
                }));
            }
        }
        let error = services.shutdown_experiments().unwrap_err().to_string();
        assert!(error.contains("a-panic"), "{error}");
        assert!(sibling_finished.load(Ordering::Acquire));
        assert_eq!(observed_all_cancellations.load(Ordering::Acquire), 2);
        let jobs = services.experiments.lock().unwrap();
        assert!(jobs.values().all(|job| job.worker.is_none()));
        assert!(
            jobs.values()
                .all(|job| !job.running.load(Ordering::Acquire))
        );
        assert!(jobs["a-panic"].last_error.lock().unwrap().is_some());
        drop(jobs);
        let capacity = *services.execution_capacity.lock().unwrap();
        assert_eq!(capacity.reserved_run_slots, 0);
        assert_eq!(capacity.reserved_batches, 0);
        assert!(matches!(
            services.reserve_experiment_execution("new", 1, 1),
            Err(ControlError::ExperimentShuttingDown)
        ));
    }

    #[test]
    fn experiment_shutdown_reports_registry_poison() {
        let root = tempfile::tempdir().unwrap();
        let services = DataServices::new(root.path().to_path_buf());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _jobs = services.experiments.lock().unwrap();
                panic!("injected registry-owner panic");
            }))
            .is_err()
        );
        let error = services.shutdown_experiments().unwrap_err().to_string();
        assert!(error.contains("registry was poisoned"), "{error}");
        assert!(matches!(
            services.reserve_experiment_execution("new", 1, 1),
            Err(ControlError::ExperimentShuttingDown)
        ));
    }

    pub(crate) fn failed_retry_catalog_fixture(
        handle: ControlHandle,
        root: &std::path::Path,
        experiment_id: &str,
    ) -> (
        ControlHandle,
        crate::experiment_runner::ExperimentBatchStatus,
        [crate::experiment_runner::RunRecord; 2],
    ) {
        use crate::experiment_runner::{
            MatchedSeedCohort, MatchedSeedExperimentRunner, RunState, ScenarioVariant,
        };
        use std::sync::atomic::AtomicBool;

        // Match production's canonical source root, including /var aliases on macOS.
        let root = root.canonicalize().unwrap();
        let handle = handle.with_artifacts_dir(root.clone()).unwrap();
        let directory = root.join("experiments").join(experiment_id);
        let runner = Arc::new(MatchedSeedExperimentRunner::new(
            experiment_id,
            MatchedSeedCohort {
                cohort_id: "cohort".into(),
                seeds: vec![7, 11],
            },
            vec![ScenarioVariant {
                variant_id: "mlp".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::from([
                    ("world_width".into(), json!(100)),
                    ("world_height".into(), json!(100)),
                    ("food_cell_size".into(), json!(50)),
                ]),
                interventions: Vec::new(),
            }],
            3,
            1,
            directory.clone(),
        ));
        // Real completed owners/bundles model a failure publishing one final
        // status; the successful peer's saved owner must remain unchanged.
        let completed = [7, 11].map(|seed| {
            runner
                .execute_single_run(&runner.variants[0], seed)
                .unwrap()
        });
        let state_file = directory.join("status.json");
        let mut status = runner.load_or_create_status(&state_file).unwrap();
        for run in &mut status.runs {
            run.state = RunState::Running;
        }
        runner.save_status(&state_file, &mut status).unwrap();
        status.runs[0].state = RunState::Failed;
        status.runs[0].error_reason = Some("completion status publication failed".into());
        status.runs[1] = completed[1].clone();
        status.completed_runs = 1;
        status.failed_runs = 1;
        runner.save_status(&state_file, &mut status).unwrap();
        handle.data_services.experiments.lock().unwrap().insert(
            experiment_id.into(),
            ExperimentJob {
                runner,
                state_file: state_file.clone(),
                cancel: Arc::new(AtomicBool::new(false)),
                running: Arc::new(AtomicBool::new(false)),
                last_error: Arc::new(Mutex::new(None)),
                created_at_utc: "1".into(),
                artifact_root: root.to_path_buf(),
                artifact_generation: Mutex::new(None),
                worker: None,
            },
        );
        (handle, status, completed)
    }

    #[test]
    fn experiment_run_artifacts_preserve_completed_results_in_a_failed_batch() {
        let root = tempfile::tempdir().unwrap();
        let (handle, host) = handle();
        let (handle, status, completed) =
            failed_retry_catalog_fixture(handle, root.path(), "failed-download");
        let state_file = root.path().join("experiments/failed-download/status.json");
        let before = fs::read(&state_file).unwrap();
        let live = host.port.scientific_digest_v1().unwrap();
        let inventory = handle
            .get_experiment_run_artifacts("failed-download", &completed[1].run_id)
            .unwrap();
        assert_eq!(inventory.run.final_digest, completed[1].final_digest);
        assert_eq!(inventory.status_generation, status.generation);
        // Even an intact bundle left by a torn publication does not promote a
        // failed run to completed evidence. Explicit retry/recovery owns that.
        assert!(matches!(
            handle.get_experiment_run_artifacts("failed-download", &completed[0].run_id),
            Err(ControlError::Conflict(_))
        ));
        assert_eq!(fs::read(state_file).unwrap(), before);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
    }

    #[test]
    fn experiment_failed_retry_reserves_capacity_and_keeps_generation_authority() {
        let root = tempfile::tempdir().unwrap();
        let (handle, _host) = handle();
        let (handle, status, completed) =
            failed_retry_catalog_fixture(handle, root.path(), "failed-retry");
        let directory = root.path().join("experiments/failed-retry");
        let state_file = directory.join("status.json");
        let generation = status.generation;
        let before = fs::read(&state_file).unwrap();
        let cuts = status
            .runs
            .iter()
            .map(|run| {
                let path = directory.join(format!("{}.checkpoint", run.run_id));
                let bytes = fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            handle.resume_experiment("failed-retry"),
            Err(ControlError::Conflict(_))
        ));
        assert!(matches!(
            handle.retry_failed_experiment("failed-retry", generation + 1),
            Err(ControlError::Conflict(message)) if message.contains("generation changed")
        ));
        handle.clone().with_experiment_run_capacity(0).unwrap();
        assert!(matches!(
            handle.retry_failed_experiment("failed-retry", generation),
            Err(ControlError::ExperimentCapacity(_))
        ));
        assert_eq!(fs::read(&state_file).unwrap(), before);
        for (path, bytes) in &cuts {
            assert_eq!(&fs::read(path).unwrap(), bytes);
        }
        assert!(!directory.join("status.json.retries").exists());
        handle.clone().with_experiment_run_capacity(1).unwrap();
        let admitted = handle
            .retry_failed_experiment("failed-retry", generation)
            .unwrap();
        assert!(admitted.generation > generation);
        assert_eq!(admitted.failed_runs, 0);
        handle
            .data_services
            .experiments
            .lock()
            .unwrap()
            .get_mut("failed-retry")
            .unwrap()
            .join_worker()
            .unwrap();
        let final_status = handle.get_experiment("failed-retry").unwrap();
        assert_eq!(final_status.status, "completed", "{final_status:?}");
        assert_eq!(final_status.completed_runs, 2);
        for (observed, expected) in final_status.runs.iter().zip(&completed) {
            assert_eq!(observed.total_ticks, expected.total_ticks);
            assert_eq!(observed.final_digest, expected.final_digest);
            assert_eq!(observed.bundle_path, expected.bundle_path);
        }
        assert_eq!(fs::read(&cuts[1].0).unwrap(), cuts[1].1);
        assert_eq!(
            fs::read(directory.join(format!(
                "status.json.retries/generation-{generation:020}.json"
            )))
            .unwrap(),
            before
        );
        let after = fs::read(&state_file).unwrap();
        assert!(matches!(
            handle.retry_failed_experiment("failed-retry", generation),
            Err(ControlError::Conflict(message)) if message.contains("generation changed")
        ));
        assert_eq!(fs::read(&state_file).unwrap(), after);
        let capacity = *handle.data_services.execution_capacity.lock().unwrap();
        assert_eq!(capacity.reserved_run_slots, 0);
        assert_eq!(capacity.reserved_batches, 0);
    }

    #[test]
    fn experiment_resume_reports_a_prior_worker_panic_before_admission() {
        let root = tempfile::tempdir().unwrap();
        let (handle, _host) = handle();
        let handle = handle
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap();
        pending_catalog_definition(root.path(), "failed-worker");
        handle
            .data_services
            .recover_experiment("failed-worker")
            .unwrap();
        {
            let mut jobs = handle.data_services.experiments.lock().unwrap();
            jobs.get_mut("failed-worker").unwrap().worker = Some(std::thread::spawn(|| {
                panic!("injected prior worker panic");
            }));
        }
        assert!(matches!(
            handle.resume_experiment("failed-worker"),
            Err(ControlError::ExperimentWorkerFailed(message)) if message.contains("failed-worker")
        ));
        let capacity = *handle.data_services.execution_capacity.lock().unwrap();
        assert_eq!(capacity.reserved_run_slots, 0);
        assert_eq!(capacity.reserved_batches, 0);
        assert!(
            !root
                .path()
                .join("experiments/failed-worker/status.json")
                .exists()
        );
        let jobs = handle.data_services.experiments.lock().unwrap();
        assert!(jobs["failed-worker"].worker.is_none());
        assert!(jobs["failed-worker"].last_error.lock().unwrap().is_some());
    }

    #[test]
    fn experiment_execution_capacity_is_shared_released_and_retryable() {
        use std::sync::atomic::Ordering;
        use std::time::{Duration, Instant};

        let root = tempfile::tempdir().unwrap().keep();
        println!("shared run-slot evidence: {}", root.display());
        let (handle, _host) = handle();
        let handle = handle
            .with_artifacts_dir(root.clone())
            .unwrap()
            .with_experiment_run_capacity(2)
            .unwrap();
        let wait_for = |id: &str, target: &str| {
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                let status = handle.get_experiment(id).unwrap();
                if status.status == target {
                    break status;
                }
                assert_ne!(status.status, "failed", "{status:?}");
                assert!(
                    Instant::now() < deadline,
                    "waiting for {target}: {status:?}"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        };
        let mut primary = experiment_creation_request();
        primary.experiment_id = Some("slots-primary".into());
        primary.idempotency_key = Some("slots-primary-key".into());
        primary.ticks_per_run = Some(40);
        primary.max_concurrency = Some(2);
        let mut dwraon = primary.variants[0].clone();
        dwraon.variant_id = "dwraon".into();
        dwraon.brain_family = "dwraon".into();
        primary.variants.push(dwraon);
        let created = handle.create_experiment(primary.clone()).unwrap();
        let full = handle
            .clone()
            .list_experiments(None, None)
            .unwrap()
            .execution_capacity;
        assert_eq!(full.reserved_run_slots, primary.variants.len());
        assert_eq!(full.reserved_batches, 1);
        assert_eq!(full.max_run_slots, primary.variants.len());
        assert_eq!(
            handle
                .create_experiment(primary.clone())
                .unwrap()
                .experiment_id,
            created.experiment_id
        );
        assert_eq!(
            handle
                .list_experiments(None, None)
                .unwrap()
                .execution_capacity,
            full
        );

        let mut secondary = experiment_creation_request();
        secondary.experiment_id = Some("slots-secondary".into());
        secondary.idempotency_key = Some("slots-secondary-key".into());
        secondary.seeds = vec![9];
        secondary.ticks_per_run = Some(40);
        secondary.max_concurrency = Some(crate::experiment_runner::MAX_CONCURRENT_RUNS);
        assert!(matches!(
            handle.create_experiment(secondary.clone()),
            Err(ControlError::ExperimentCapacity(_))
        ));
        assert!(!root.join("experiments/slots-secondary").exists());
        assert!(
            !handle
                .data_services
                .idempotency
                .lock()
                .unwrap()
                .contains_key("slots-secondary-key")
        );

        let mut cuts = BTreeMap::<PathBuf, std::collections::BTreeSet<String>>::new();
        for run in &created.runs {
            cuts.insert(
                root.join("experiments/slots-primary")
                    .join(format!("{}.checkpoint", run.run_id)),
                Default::default(),
            );
        }
        let deadline = Instant::now() + Duration::from_secs(30);
        while cuts.values().any(|observed| observed.len() < 2) {
            for (path, observed) in &mut cuts {
                if let Ok(bytes) = fs::read(path)
                    && let Some(end) = bytes.len().checked_sub(blake3::OUT_LEN)
                    && blake3::hash(&bytes[..end]).as_bytes() == &bytes[end..]
                {
                    observed.insert(compute_blake3(&bytes));
                }
            }
            assert!(Instant::now() < deadline, "both real owners must advance");
            std::thread::sleep(Duration::from_millis(1));
        }
        handle.cancel_experiment("slots-primary").unwrap();
        let stopped = wait_for("slots-primary", "cancelled");
        assert!(stopped.runs.iter().all(|run| run.state == "running"));
        assert_eq!(
            handle
                .list_experiments(None, None)
                .unwrap()
                .execution_capacity
                .reserved_run_slots,
            0
        );
        fs::write(
            root.join("cancelled.json"),
            serde_json::to_vec_pretty(&stopped).unwrap(),
        )
        .unwrap();
        let status_path = root.join("experiments/slots-primary/status.json");
        let saved_status = fs::read(&status_path).unwrap();
        let saved_cuts = cuts
            .keys()
            .map(|path| {
                let bytes = fs::read(path).unwrap();
                fs::write(
                    root.join(format!(
                        "stopped-{}",
                        path.file_name().unwrap().to_str().unwrap()
                    )),
                    &bytes,
                )
                .unwrap();
                (path.clone(), bytes)
            })
            .collect::<Vec<_>>();

        handle.create_experiment(secondary.clone()).unwrap();
        let occupied = handle
            .list_experiments(None, None)
            .unwrap()
            .execution_capacity;
        assert_eq!(occupied.reserved_run_slots, secondary.seeds.len());
        assert_eq!(occupied.reserved_batches, 1);
        assert!(matches!(
            handle.resume_experiment("slots-primary"),
            Err(ControlError::ExperimentCapacity(_))
        ));
        assert_eq!(fs::read(&status_path).unwrap(), saved_status);
        for (path, bytes) in saved_cuts {
            assert_eq!(fs::read(path).unwrap(), bytes);
        }
        assert!(
            handle.data_services.experiments.lock().unwrap()["slots-primary"]
                .cancel
                .load(Ordering::Acquire)
        );
        assert!(matches!(
            handle.clone().with_experiment_run_capacity(0),
            Err(ControlError::ExperimentCapacity(_))
        ));
        assert_eq!(
            handle
                .list_experiments(None, None)
                .unwrap()
                .execution_capacity,
            occupied
        );

        handle.clone().with_experiment_run_capacity(3).unwrap();
        handle.resume_experiment("slots-primary").unwrap();
        let combined = handle
            .list_experiments(None, None)
            .unwrap()
            .execution_capacity;
        assert_eq!(
            combined.reserved_run_slots,
            primary.variants.len() + secondary.seeds.len()
        );
        assert_eq!(combined.reserved_batches, 2);
        assert_eq!(combined.reserved_run_slots, combined.max_run_slots);
        let completed = [
            wait_for("slots-primary", "completed"),
            wait_for("slots-secondary", "completed"),
        ];
        let free = handle
            .list_experiments(None, None)
            .unwrap()
            .execution_capacity;
        assert_eq!(free.reserved_run_slots, 0);
        assert_eq!(free.reserved_batches, 0);
        for status in &completed {
            let original = Arc::clone(
                &handle.data_services.experiments.lock().unwrap()[&status.experiment_id].runner,
            );
            let oracle = crate::experiment_runner::MatchedSeedExperimentRunner::new(
                &original.experiment_id,
                original.cohort.clone(),
                original.variants.clone(),
                original.max_ticks,
                original.max_concurrency,
                root.join("oracle").join(&status.experiment_id),
            );
            let expected = oracle
                .execute_batch(&oracle.output_dir.join("status.json"))
                .unwrap();
            assert_eq!(status.runs.len(), expected.runs.len());
            for (actual, expected) in status.runs.iter().zip(expected.runs) {
                assert_eq!(actual.run_id, expected.run_id);
                assert_eq!(actual.brain_family, expected.brain_family);
                assert_eq!(actual.total_ticks, expected.total_ticks);
                assert_eq!(actual.final_digest, expected.final_digest);
                scriptbots_storage::verify_run_bundle(Path::new(
                    actual.bundle_path.as_ref().unwrap(),
                ))
                .unwrap();
            }
        }
        fs::write(
            root.join("completed.json"),
            serde_json::to_vec_pretty(&completed).unwrap(),
        )
        .unwrap();
        handle.clone().with_experiment_run_capacity(0).unwrap();
        let retry = handle.create_experiment(primary).unwrap();
        assert_eq!(retry.experiment_id, completed[0].experiment_id);
        let mut disabled = secondary;
        disabled.experiment_id = Some("disabled".into());
        disabled.idempotency_key = Some("disabled-key".into());
        assert!(matches!(
            handle.create_experiment(disabled),
            Err(ControlError::ExperimentCapacity(_))
        ));
        assert!(!root.join("experiments/disabled").exists());
        assert_eq!(
            handle
                .list_experiments(None, None)
                .unwrap()
                .execution_capacity
                .max_run_slots,
            0
        );
    }

    #[test]
    fn experiment_grid_capacity_is_shared_across_batches_and_released_atomically() {
        let directory = tempfile::tempdir().unwrap().keep();
        let (handle, _host) = handle();
        let handle = handle.with_artifacts_dir(directory.clone()).unwrap();
        let services = handle.data_services();
        let limit = crate::experiment_runner::MAX_CONCURRENT_GRID_CELLS;
        let first = services
            .reserve_experiment_execution("first", 1, limit - 1)
            .unwrap();
        let last = services.reserve_experiment_execution("last", 1, 1).unwrap();
        let full = *services.execution_capacity.lock().unwrap();
        assert_eq!(full.reserved_grid_cells, limit);
        assert!(full.reserved_run_slots < full.max_run_slots);
        for cells in [1, u64::MAX] {
            assert!(matches!(
                services.reserve_experiment_execution("overflow", 1, cells),
                Err(ControlError::ExperimentCapacity(_))
            ));
            assert_eq!(*services.execution_capacity.lock().unwrap(), full);
        }
        let mut request = experiment_creation_request();
        request.experiment_id = Some("grid-refused".into());
        assert!(matches!(
            handle.create_experiment(request),
            Err(ControlError::ExperimentCapacity(message)) if message.contains("grid_cells")
        ));
        assert!(!directory.join("experiments/grid-refused").exists());
        assert!(
            !services
                .experiments
                .lock()
                .unwrap()
                .contains_key("grid-refused")
        );
        assert_eq!(*services.execution_capacity.lock().unwrap(), full);
        drop(last);
        let retry = services
            .reserve_experiment_execution("retry", 1, 1)
            .unwrap();
        assert_eq!(*services.execution_capacity.lock().unwrap(), full);
        drop(first);
        drop(retry);
        let free = *services.execution_capacity.lock().unwrap();
        assert_eq!(free.reserved_grid_cells, 0);
        assert_eq!(free.reserved_run_slots, 0);
        assert_eq!(free.reserved_batches, 0);
        assert!(matches!(
            services.reserve_experiment_execution("empty-grid", 1, 0),
            Err(ControlError::BadRequest(_))
        ));
        assert_eq!(*services.execution_capacity.lock().unwrap(), free);
    }

    #[test]
    fn experiment_execution_reservation_returns_capacity_on_unstarted_drop_and_unwind() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let root = tempfile::tempdir().unwrap();
        let services = DataServices::new(root.path().to_path_buf());
        services.execution_capacity.lock().unwrap().max_run_slots = 1;
        let running = Arc::new(AtomicBool::new(false));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut reservation = services
                    .reserve_experiment_execution("unwind", 1, 1)
                    .unwrap();
                assert!(matches!(
                    services.reserve_experiment_execution("over-cap", 1, 1),
                    Err(ControlError::ExperimentCapacity(_))
                ));
                running.store(true, Ordering::Release);
                reservation.running = Some(Arc::clone(&running));
                panic!("intentional execution reservation unwind");
            }))
            .is_err()
        );
        assert!(!running.load(Ordering::Acquire));
        let reservation = services
            .reserve_experiment_execution("unstarted", 1, 1)
            .unwrap();
        drop(reservation);
        let free = *services.execution_capacity.lock().unwrap();
        assert_eq!(free.reserved_run_slots, 0);
        assert_eq!(free.reserved_grid_cells, 0);
        assert_eq!(free.reserved_batches, 0);
        assert!(matches!(
            services.reserve_experiment_execution("zero", 0, 1),
            Err(ControlError::BadRequest(_))
        ));
        assert_eq!(*services.execution_capacity.lock().unwrap(), free);
    }

    #[test]
    fn experiment_catalog_recovers_admission_without_executing_or_rewriting() {
        let root = tempfile::tempdir().unwrap().keep();
        let path = pending_catalog_definition(&root, "saved");
        let original = fs::read(&path).unwrap();
        let services = DataServices::new(root.clone());
        for _ in 0..2 {
            services.recover_experiment_catalog().unwrap();
            let jobs = services.experiments.lock().unwrap();
            assert_eq!(jobs.len(), 1);
            let job = jobs.get("saved").unwrap();
            let status = job.status_dto(true).unwrap();
            assert_eq!(status.status, "stopped");
            assert_eq!(status.created_at_utc, "123456");
            assert_eq!(status.total_runs, 1);
            assert_eq!(status.runs[0].state, "pending");
            assert!(
                !job.state_file.exists(),
                "read must not initialize status or science"
            );
            assert_eq!(fs::read(&path).unwrap(), original);
        }
        assert_eq!(
            services.idempotency.lock().unwrap().get("key-saved"),
            Some(&(compute_blake3(b"saved"), "saved".into()))
        );
        drop(services);
        let restarted = DataServices::new(root);
        restarted.recover_experiment("saved").unwrap();
        assert_eq!(restarted.experiments.lock().unwrap().len(), 1);
        assert_eq!(fs::read(path).unwrap(), original);
    }

    #[test]
    fn experiment_catalog_refuses_foreign_or_corrupt_definitions_and_retains_companions() {
        let root = tempfile::tempdir().unwrap().keep();
        let path = pending_catalog_definition(&root, "a-corrupt");
        let healthy = pending_catalog_definition(&root, "z-healthy");
        let original = fs::read(&path).unwrap();
        fs::copy(&path, root.join("preserved-definition.json")).unwrap();
        let valid: Value = serde_json::from_slice(&original).unwrap();
        let mutations = [
            ("/definition/schema_version", serde_json::json!(2)),
            (
                "/definition/plan_digest",
                serde_json::json!(compute_blake3(b"foreign source")),
            ),
            ("/definition/experiment_id", serde_json::json!("another")),
            ("/definition/cohort/cohort_id", serde_json::json!("another")),
            ("/definition/cohort/seeds/0", serde_json::json!(8)),
            (
                "/definition/variants/0/brain_family",
                serde_json::json!("dwraon"),
            ),
            ("/definition/max_concurrency", serde_json::json!(0)),
            ("/definition/request_identity/0", serde_json::json!("")),
            (
                "/definition/request_identity/1",
                serde_json::json!("unbound"),
            ),
            ("/checksum", serde_json::json!(compute_blake3(b"corrupt"))),
        ];
        let mut cases = std::collections::BTreeSet::new();
        for (pointer, value) in mutations {
            assert!(cases.insert(pointer));
            let mut changed = valid.clone();
            let field = changed.pointer_mut(pointer).unwrap();
            assert_ne!(*field, value, "distinct falsifying input: {pointer}");
            *field = value;
            // Semantic controls carry valid checksums, so they exercise the
            // identity/plan gates rather than all failing at the same byte guard.
            if pointer != "/checksum" {
                let definition: ExperimentDefinitionV1 =
                    serde_json::from_value(changed["definition"].clone()).unwrap();
                changed["checksum"] =
                    compute_blake3(&serde_json::to_vec(&definition).unwrap()).into();
            }
            let bytes = serde_json::to_vec(&changed).unwrap();
            fs::write(&path, &bytes).unwrap();
            let services = DataServices::new(root.clone());
            assert!(
                matches!(
                    services.recover_experiment_catalog(),
                    Err(ControlError::Serialization(_))
                ),
                "mutation admitted: {pointer}"
            );
            services.recover_experiment("z-healthy").unwrap();
            let jobs = services.experiments.lock().unwrap();
            assert!(!jobs.contains_key("a-corrupt"));
            assert_eq!(
                jobs["z-healthy"].status_dto(true).unwrap().status,
                "stopped"
            );
            assert_eq!(fs::read(&path).unwrap(), bytes);
            assert!(healthy.is_file());
            eprintln!(
                "catalog refusal {pointer}, sha256={}",
                compute_sha256(&bytes)
            );
        }
        fs::write(&path, &original).unwrap();
        let services = DataServices::new(root);
        services.recover_experiment_catalog().unwrap();
        assert_eq!(services.experiments.lock().unwrap().len(), 2);
    }

    #[test]
    fn experiment_catalog_refuses_oversized_aliased_and_interrupted_definitions() {
        let root = tempfile::tempdir().unwrap().keep();
        let path = pending_catalog_definition(&root, "saved");
        fs::copy(&path, root.join("original.json")).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(MAX_EXPERIMENT_DEFINITION_BYTES + 1)
            .unwrap();
        let services = DataServices::new(root.clone());
        assert!(services.recover_experiment("saved").is_err());
        assert!(services.experiments.lock().unwrap().is_empty());
        #[cfg(unix)]
        {
            let symlink_dir = root.join("experiments/symlink");
            std::os::unix::fs::symlink(path.parent().unwrap(), &symlink_dir).unwrap();
            assert!(services.recover_experiment("symlink").is_err());
            let alias = root.join("experiments/alias");
            fs::create_dir(&alias).unwrap();
            fs::hard_link(root.join("original.json"), alias.join("definition.json")).unwrap();
            assert!(services.recover_experiment("alias").is_err());
        }
        let interrupted = root.join("experiments/interrupted");
        fs::create_dir(&interrupted).unwrap();
        let temporary = interrupted.join("definition.json.tmp");
        fs::write(&temporary, b"partial").unwrap();
        assert!(services.recover_experiment("interrupted").is_err());
        assert_eq!(fs::read(&temporary).unwrap(), b"partial");
        assert!(services.recover_experiment("../outside").is_err());
        assert!(services.experiments.lock().unwrap().is_empty());
    }

    #[cfg(feature = "brain-ft")]
    #[test]
    fn experiment_creation_accepts_frankentorch_alias_and_exports_restorable_ft_bundle() {
        let root = tempfile::tempdir().unwrap();
        let (_unused_handle, host) = handle();
        // This guard joins experiment workers before the temporary root is dropped,
        // including when an assertion fails.
        let (runtime, _submit) = crate::ControlRuntime::launch(
            host.port.clone(),
            crate::ControlServerConfig {
                rest_enabled: false,
                mcp_transport: crate::servers::McpTransportConfig::Disabled,
                ..crate::ControlServerConfig::default()
            },
        )
        .unwrap();
        let handle = runtime
            .control_handle()
            .unwrap()
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap();
        let live_digest = host.port.scientific_digest_v1().unwrap();
        let live_snapshot = handle.read_snapshot().unwrap();
        assert!(handle.version().features.iter().any(|f| f == "brain-ft"));

        let mut request = experiment_creation_request();
        request.experiment_id = Some("ft-api-run".into());
        request.idempotency_key = Some("ft-api-run-key".into());
        request.variants[0].brain_family = " FrAnKeNtOrCh ".into();
        let expected_runs = request.variants.len() * request.seeds.len();
        let expected_seed = request.seeds[0];
        let expected_ticks = request.ticks_per_run.unwrap();
        let created = handle.create_experiment(request).expect("admit Ft alias");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let completed = loop {
            let status = handle.get_experiment(&created.experiment_id).unwrap();
            if status.status == "completed" {
                break status;
            }
            assert_ne!(status.status, "failed", "{status:?}");
            assert!(std::time::Instant::now() < deadline, "{status:?}");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert_eq!(completed.total_runs, expected_runs);
        assert_eq!(completed.completed_runs, expected_runs);
        assert_eq!(completed.failed_runs, 0);
        assert_eq!(completed.runs.len(), expected_runs);
        let run = &completed.runs[0];
        assert_eq!(run.brain_family, "ft");
        assert_eq!(run.state, "completed");
        assert_eq!(run.seed, expected_seed);
        assert_eq!(run.total_ticks, expected_ticks);
        let bundle_dir = Path::new(run.bundle_path.as_ref().expect("published Ft bundle"));
        let verified = crate::experiment_runner::verify_portable_run_bundle(bundle_dir)
            .expect("restore the actual Ft checkpoint from its portable bundle");
        let final_state = verified.final_state.expect("observed Ft final science");
        assert_eq!(final_state.run_id, run.run_id);
        assert_eq!(final_state.tick, expected_ticks);
        assert_eq!(
            Some(final_state.scientific_digest.as_str()),
            run.final_digest.as_deref()
        );
        let bundle: scriptbots_storage::RunBundleV1 =
            serde_json::from_slice(&fs::read(bundle_dir.join("bundle_manifest.json")).unwrap())
                .unwrap();
        let manifest: crate::RunManifestV3 =
            serde_json::from_str(&bundle.manifest.manifest_json).unwrap();
        assert_eq!(manifest.brain_roster.len(), 1);
        assert_eq!(manifest.brain_roster[0].kind, "ft");
        assert_eq!(manifest.root_seed, expected_seed);
        assert_eq!(
            manifest.identity.requested_tick_budget,
            Some(expected_ticks)
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), live_digest);
        let after = handle.read_snapshot().unwrap();
        assert_eq!(after.config, live_snapshot.config);
        assert_eq!(after.world.tick, live_snapshot.world.tick);
        runtime.shutdown().expect("join Ft experiment worker");
    }

    #[cfg(not(feature = "brain-ft"))]
    #[test]
    fn experiment_creation_refuses_ft_aliases_before_admission_without_feature() {
        let root = tempfile::tempdir().unwrap();
        let (_unused_handle, host) = handle();
        let (runtime, _submit) = crate::ControlRuntime::launch(
            host.port.clone(),
            crate::ControlServerConfig {
                rest_enabled: false,
                mcp_transport: crate::servers::McpTransportConfig::Disabled,
                ..crate::ControlServerConfig::default()
            },
        )
        .unwrap();
        let handle = runtime
            .control_handle()
            .unwrap()
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap()
            .with_experiment_run_capacity(1)
            .unwrap();
        let live_digest = host.port.scientific_digest_v1().unwrap();
        let capacity = handle
            .list_experiments(None, None)
            .unwrap()
            .execution_capacity;
        assert_eq!(capacity.reserved_run_slots, 0);
        assert_eq!(capacity.reserved_batches, 0);
        assert!(!handle.version().features.iter().any(|f| f == "brain-ft"));

        for alias in [" Ft ", " FrAnKeNtOrCh "] {
            let mut request = experiment_creation_request();
            request.experiment_id = Some("unavailable-ft".into());
            request.variants[0].brain_family = alias.into();
            let error = handle
                .create_experiment(request)
                .expect_err("disabled Ft must fail before any experiment side effects");
            assert!(matches!(
                error,
                ControlError::BadRequest(reason) if reason.contains("brain-ft")
            ));
            assert!(!root.path().join("experiments").exists());
            assert!(handle.data_services.experiments.lock().unwrap().is_empty());
            assert!(handle.data_services.idempotency.lock().unwrap().is_empty());
            let listed = handle.list_experiments(None, None).unwrap();
            assert_eq!(listed.total, 0);
            assert_eq!(listed.execution_capacity, capacity);
            assert_eq!(host.port.scientific_digest_v1().unwrap(), live_digest);
        }
        runtime
            .shutdown()
            .expect("join any unexpected experiment worker");
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn experiment_creation_compares_neuroflow_and_exports_restorable_configured_bundle() {
        let root = tempfile::tempdir().unwrap();
        let (_unused_handle, host) = handle();
        let (runtime, _submit) = crate::ControlRuntime::launch(
            host.port.clone(),
            crate::ControlServerConfig {
                rest_enabled: false,
                mcp_transport: crate::servers::McpTransportConfig::Disabled,
                ..crate::ControlServerConfig::default()
            },
        )
        .unwrap();
        let handle = runtime
            .control_handle()
            .unwrap()
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap();
        let live_digest = host.port.scientific_digest_v1().unwrap();
        let live_snapshot = handle.read_snapshot().unwrap();
        assert!(handle.version().features.iter().any(|f| f == "neuro"));

        let mut request = experiment_creation_request();
        request.experiment_id = Some("neuro-api-run".into());
        request.idempotency_key = Some("neuro-api-run-key".into());
        let mut neuro = request.variants[0].clone();
        neuro.variant_id = "neuro".into();
        neuro.brain_family = " NeUrOfLoW ".into();
        request.variants.push(neuro);
        let expected_runs = request.variants.len() * request.seeds.len();
        let expected_seed = request.seeds[0];
        let expected_ticks = request.ticks_per_run.unwrap();
        let created = handle
            .create_experiment(request)
            .expect("admit the real matched MLP/NeuroFlow cohort");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let completed = loop {
            let status = handle.get_experiment(&created.experiment_id).unwrap();
            if status.status == "completed" {
                break status;
            }
            assert_ne!(status.status, "failed", "{status:?}");
            assert!(std::time::Instant::now() < deadline, "{status:?}");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert_eq!(completed.total_runs, expected_runs);
        assert_eq!(completed.completed_runs, expected_runs);
        assert_eq!(completed.failed_runs, 0);
        assert_eq!(completed.runs.len(), expected_runs);
        let baseline = completed
            .runs
            .iter()
            .find(|run| run.variant_id == "baseline")
            .unwrap();
        assert_eq!(baseline.brain_family, "mlp.baseline");
        let run = completed
            .runs
            .iter()
            .find(|run| run.variant_id == "neuro")
            .unwrap();
        assert_eq!(run.brain_family, "ml.neuroflow");
        assert_eq!(run.state, "completed");
        assert_eq!(run.seed, expected_seed);
        assert_eq!(run.total_ticks, expected_ticks);
        assert_ne!(run.final_digest, baseline.final_digest);
        let bundle_dir = Path::new(
            run.bundle_path
                .as_ref()
                .expect("published NeuroFlow bundle"),
        );
        let verified = crate::experiment_runner::verify_portable_run_bundle(bundle_dir)
            .expect("restore configured NeuroFlow from its portable final checkpoint");
        let final_state = verified
            .final_state
            .expect("observed NeuroFlow final science");
        assert_eq!(final_state.run_id, run.run_id);
        assert_eq!(final_state.tick, expected_ticks);
        assert_eq!(
            Some(final_state.scientific_digest.as_str()),
            run.final_digest.as_deref()
        );
        let bundle: scriptbots_storage::RunBundleV1 =
            serde_json::from_slice(&fs::read(bundle_dir.join("bundle_manifest.json")).unwrap())
                .unwrap();
        let manifest: crate::RunManifestV3 =
            serde_json::from_str(&bundle.manifest.manifest_json).unwrap();
        assert_eq!(manifest.brain_roster.len(), 1);
        assert_eq!(manifest.brain_roster[0].kind, "ml.neuroflow");
        assert_eq!(manifest.root_seed, expected_seed);
        assert_eq!(
            manifest.identity.requested_tick_budget,
            Some(expected_ticks)
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), live_digest);
        let after = handle.read_snapshot().unwrap();
        assert_eq!(after.config, live_snapshot.config);
        assert_eq!(after.world.tick, live_snapshot.world.tick);
        runtime
            .shutdown()
            .expect("join NeuroFlow experiment worker");
    }

    #[cfg(not(feature = "neuro"))]
    #[test]
    fn experiment_creation_refuses_neuroflow_aliases_before_admission_without_feature() {
        let root = tempfile::tempdir().unwrap();
        let (_unused_handle, host) = handle();
        let (runtime, _submit) = crate::ControlRuntime::launch(
            host.port.clone(),
            crate::ControlServerConfig {
                rest_enabled: false,
                mcp_transport: crate::servers::McpTransportConfig::Disabled,
                ..crate::ControlServerConfig::default()
            },
        )
        .unwrap();
        let handle = runtime
            .control_handle()
            .unwrap()
            .with_artifacts_dir(root.path().to_path_buf())
            .unwrap()
            .with_experiment_run_capacity(1)
            .unwrap();
        let live_digest = host.port.scientific_digest_v1().unwrap();
        let capacity = handle
            .list_experiments(None, None)
            .unwrap()
            .execution_capacity;
        assert_eq!(capacity.reserved_run_slots, 0);
        assert_eq!(capacity.reserved_batches, 0);
        assert!(!handle.version().features.iter().any(|f| f == "neuro"));

        for alias in [" NeUrO ", " NeUrOfLoW ", " Ml.NeUrOfLoW "] {
            let mut request = experiment_creation_request();
            request.experiment_id = Some("unavailable-neuro".into());
            request.variants[0].brain_family = alias.into();
            let error = handle
                .create_experiment(request)
                .expect_err("disabled NeuroFlow must fail before experiment side effects");
            assert!(matches!(
                error,
                ControlError::BadRequest(reason) if reason.contains("neuro feature")
            ));
            assert!(!root.path().join("experiments").exists());
            assert!(handle.data_services.experiments.lock().unwrap().is_empty());
            assert!(handle.data_services.idempotency.lock().unwrap().is_empty());
            let listed = handle.list_experiments(None, None).unwrap();
            assert_eq!(listed.total, 0);
            assert_eq!(listed.execution_capacity, capacity);
            assert_eq!(host.port.scientific_digest_v1().unwrap(), live_digest);
        }
        runtime
            .shutdown()
            .expect("join any unexpected experiment worker");
    }

    #[test]
    fn experiment_creation_concurrent_retry_admits_one_real_batch() {
        let (handle, _host) = handle();
        let request = experiment_creation_request();
        let gate = std::sync::Barrier::new(4);
        let replies = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        gate.wait();
                        handle.create_experiment(request.clone())
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| {
                    worker
                        .join()
                        .expect("create caller joined")
                        .expect("create")
                })
                .collect::<Vec<_>>()
        });
        let identities: std::collections::BTreeSet<_> = replies
            .iter()
            .map(|reply| reply.experiment_id.as_str())
            .collect();
        assert_eq!(identities.len(), 1, "retry identities: {identities:?}");
        assert_eq!(handle.list_experiments(Some(100), None).unwrap().total, 1);
        let experiment_id = &replies[0].experiment_id;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let completed = loop {
            let status = handle.get_experiment(experiment_id).expect("actual status");
            if status.status == "completed" {
                break status;
            }
            assert_ne!(status.status, "failed", "{status:?}");
            assert!(std::time::Instant::now() < deadline, "{status:?}");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert_eq!(completed.completed_runs, 1);
        assert_eq!(completed.failed_runs, 0);
        assert_eq!(
            completed.runs[0].total_ticks,
            request.ticks_per_run.unwrap()
        );
        let bundle = scriptbots_storage::verify_run_bundle(Path::new(
            completed.runs[0].bundle_path.as_ref().expect("real bundle"),
        ))
        .expect("independently read the actual completed bundle");
        assert_eq!(bundle.max_tick, completed.runs[0].total_ticks);
        assert!(
            completed.runs[0]
                .final_digest
                .as_ref()
                .is_some_and(|digest| !digest.is_empty())
        );
        let retry = handle
            .create_experiment(request.clone())
            .expect("completed retry");
        assert_eq!(retry.experiment_id, *experiment_id);
        assert_eq!(retry.completed_runs, 1);
        let mut changed = request.clone();
        changed.seeds.push(8);
        assert!(matches!(
            handle.create_experiment(changed),
            Err(ControlError::Conflict(_))
        ));
        assert_eq!(handle.list_experiments(Some(100), None).unwrap().total, 1);
        let root = handle.artifacts_dir();
        let identity = experiment_id.clone();
        drop(handle);
        let (mut restarted, _restarted_host) = self::handle();
        restarted.data_services = Arc::new(DataServices::new(root));
        let recovered = restarted.create_experiment(request.clone()).unwrap();
        assert_eq!(recovered.experiment_id, identity);
        assert_eq!(recovered.status, "completed");
        assert_eq!(
            recovered.runs[0].final_digest,
            completed.runs[0].final_digest
        );
        let mut changed = request;
        changed.seeds.push(9);
        assert!(matches!(
            restarted.create_experiment(changed),
            Err(ControlError::Conflict(_))
        ));
        assert_eq!(
            restarted.list_experiments(Some(100), None).unwrap().total,
            1
        );
    }

    #[test]
    fn experiment_status_refuses_changed_plan_and_terminal_evidence() {
        let (handle, _host) = handle();
        let created = handle
            .create_experiment(experiment_creation_request())
            .expect("create actual experiment");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let completed = loop {
            let status = handle.get_experiment(&created.experiment_id).unwrap();
            if status.status == "completed" {
                break status;
            }
            assert_ne!(status.status, "failed", "{status:?}");
            assert!(std::time::Instant::now() < deadline, "{status:?}");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        let state_file = handle
            .artifacts_dir()
            .join("experiments")
            .join(&created.experiment_id)
            .join("status.json");
        let original = fs::read(&state_file).expect("actual completed status");
        fs::copy(&state_file, state_file.with_extension("preserved-original"))
            .expect("preserve completed status before fault inputs");
        let valid: Value = serde_json::from_slice(&original).unwrap();
        let mutations = [
            (
                "/schema_version",
                serde_json::json!(completed.schema_version + 1),
            ),
            (
                "/plan_digest",
                serde_json::json!(compute_blake3(b"foreign plan")),
            ),
            ("/experiment_id", serde_json::json!("foreign-experiment")),
            ("/total_runs", serde_json::json!(completed.total_runs + 1)),
            ("/completed_runs", serde_json::json!(0)),
            ("/failed_runs", serde_json::json!(1)),
            ("/runs/0/run_id", serde_json::json!("foreign-run")),
            ("/runs/0/variant_id", serde_json::json!("foreign-variant")),
            ("/runs/0/brain_family", serde_json::json!("dwraon")),
            (
                "/runs/0/seed",
                serde_json::json!(completed.runs[0].seed ^ 1),
            ),
            (
                "/runs/0/total_ticks",
                serde_json::json!(completed.runs[0].total_ticks + 1),
            ),
            ("/runs/0/final_digest", serde_json::json!("invalid-digest")),
            (
                "/runs/0/bundle_path",
                serde_json::json!("outside-the-experiment"),
            ),
            ("/runs/0/state", serde_json::json!("pending")),
        ];
        let mut observed_paths = std::collections::BTreeSet::new();
        let mut admitted = Vec::new();
        for (pointer, value) in mutations {
            assert!(observed_paths.insert(pointer), "duplicate mutation cell");
            let mut changed = valid.clone();
            let field = changed.pointer_mut(pointer).expect("actual declared field");
            assert_ne!(*field, value, "mutation must change its field: {pointer}");
            *field = value;
            let bytes = serde_json::to_vec(&changed).unwrap();
            fs::write(&state_file, &bytes).expect("actual changed status");
            let get_refused = matches!(
                handle.get_experiment(&created.experiment_id),
                Err(ControlError::Serialization(_))
            );
            let list_refused = matches!(
                handle.list_experiments(Some(100), None),
                Err(ControlError::Serialization(_))
            );
            eprintln!(
                "{}",
                serde_json::json!({
                    "case": pointer, "get_refused": get_refused, "list_refused": list_refused,
                    "input_sha256": compute_sha256(&bytes), "input_bytes": bytes.len()
                })
            );
            if !get_refused || !list_refused {
                admitted.push(pointer);
            }
            assert_eq!(
                fs::read(&state_file).unwrap(),
                bytes,
                "reads must not rewrite evidence"
            );
        }
        fs::write(&state_file, &original).expect("restore exact preserved completed bytes");
        let readback = handle.get_experiment(&created.experiment_id).unwrap();
        assert_eq!(
            serde_json::to_value(&readback.runs).unwrap(),
            serde_json::to_value(&completed.runs).unwrap()
        );
        assert_eq!(readback.plan_digest, completed.plan_digest);
        assert!(
            admitted.is_empty(),
            "unvalidated status cells admitted: {admitted:?}"
        );
    }

    #[test]
    fn experiment_status_enforces_the_runner_byte_bound_without_rewriting() {
        let (handle, _host) = handle();
        let created = handle
            .create_experiment(experiment_creation_request())
            .expect("create actual experiment");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let completed = loop {
            let status = handle.get_experiment(&created.experiment_id).unwrap();
            if status.status == "completed" {
                break status;
            }
            assert_ne!(status.status, "failed", "{status:?}");
            assert!(std::time::Instant::now() < deadline, "{status:?}");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        let state_file = handle
            .artifacts_dir()
            .join("experiments")
            .join(&created.experiment_id)
            .join("status.json");
        let original = fs::read(&state_file).unwrap();
        fs::copy(&state_file, state_file.with_extension("preserved-original"))
            .expect("preserve completed status");
        let cap = usize::try_from(crate::experiment_runner::MAX_STATUS_BYTES).unwrap();
        let mut observations = Vec::new();
        for size in [cap - 1, cap, cap + 1] {
            let mut bytes = original.clone();
            assert!(bytes.len() < size);
            bytes.resize(size, b' ');
            fs::write(&state_file, &bytes).unwrap();
            let get = handle.get_experiment(&created.experiment_id);
            let list = handle.list_experiments(Some(100), None);
            let get_matches = if size > cap {
                matches!(get, Err(ControlError::Serialization(_)))
            } else {
                get.is_ok_and(|status| {
                    serde_json::to_value(&status.runs).unwrap()
                        == serde_json::to_value(&completed.runs).unwrap()
                })
            };
            let list_matches = if size > cap {
                matches!(list, Err(ControlError::Serialization(_)))
            } else {
                list.is_ok_and(|page| {
                    page.total == 1
                        && page.items.len() == 1
                        && page.items[0].completed_runs == completed.completed_runs
                        && page.items[0].failed_runs == completed.failed_runs
                })
            };
            eprintln!(
                "{}",
                serde_json::json!({
                    "input_bytes": size, "cap": cap, "get_matches": get_matches,
                    "list_matches": list_matches, "input_sha256": compute_sha256(&bytes)
                })
            );
            observations.push(get_matches && list_matches);
            assert_eq!(fs::read(&state_file).unwrap(), bytes);
        }
        fs::copy(
            &state_file,
            state_file.with_extension("over-limit-preserved"),
        )
        .expect("preserve actual oversized input");
        fs::write(&state_file, original).expect("restore exact completed bytes");
        assert!(
            observations.iter().all(|matched| *matched),
            "bound observations: {observations:?}"
        );
    }

    #[test]
    fn experiment_creation_publishes_downloadable_bundle_files() {
        let (handle, _host) = handle();
        let created = handle
            .create_experiment(experiment_creation_request())
            .expect("create actual experiment");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let completed = loop {
            let status = handle
                .get_experiment(&created.experiment_id)
                .expect("actual experiment status");
            if status.status == "completed" {
                break status;
            }
            assert_ne!(status.status, "failed", "{status:?}");
            assert!(std::time::Instant::now() < deadline, "{status:?}");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        let original_root = handle.artifacts_dir();
        let bundle_dir = Path::new(completed.runs[0].bundle_path.as_ref().expect("bundle"));
        let bundle: scriptbots_storage::RunBundleV1 = serde_json::from_slice(
            &fs::read(bundle_dir.join("bundle_manifest.json")).expect("actual manifest"),
        )
        .expect("decode actual manifest");
        assert!(bundle_dir.starts_with(&original_root));
        let prefix = format!(
            "experiments/{}/{}",
            completed.experiment_id, completed.runs[0].run_id
        );
        let expected: std::collections::BTreeSet<_> = bundle
            .artifacts
            .iter()
            .map(|entry| format!("{prefix}/{}", entry.relative_path))
            .chain(std::iter::once(format!("{prefix}/bundle_manifest.json")))
            .collect();
        let catalog = handle.list_artifacts(Some(100), None).expect("catalog");
        let actual: std::collections::BTreeSet<_> = catalog
            .items
            .iter()
            .map(|item| item.relative_path.clone())
            .collect();
        assert_eq!(expected, actual, "completed files must be discoverable");
        assert!(!catalog.has_more);
        let (mut restarted, _restarted_host) = self::handle();
        restarted.data_services = Arc::new(DataServices::new(original_root.clone()));
        let recovered = restarted
            .list_artifacts(Some(100), None)
            .expect("recovered catalog");
        assert_eq!(
            recovered.total, catalog.total,
            "restart must recover the original files"
        );
        assert_eq!(
            serde_json::to_value(&recovered.items).unwrap(),
            serde_json::to_value(&catalog.items).unwrap(),
            "artifact identities and hashes survive restart"
        );
        let (mut direct, _direct_host) = self::handle();
        direct.data_services = Arc::new(DataServices::new(original_root.clone()));
        for artifact in &catalog.items {
            let (metadata, bytes) = direct
                .read_artifact_bytes(&artifact.artifact_id)
                .expect("cold download by the original ID without a list/create/resume");
            assert_eq!(metadata.artifact_id, artifact.artifact_id);
            assert_eq!(
                bytes,
                fs::read(original_root.join(&artifact.relative_path)).unwrap()
            );
        }
        let handle = handle
            .with_artifacts_dir(original_root.with_extension("rotated"))
            .expect("rotate the active artifact root");
        for artifact in &catalog.items {
            let (metadata, bytes) = handle
                .read_artifact_bytes(&artifact.artifact_id)
                .expect("download from the original root");
            assert_eq!(
                bytes,
                fs::read(original_root.join(&artifact.relative_path)).unwrap()
            );
            assert_eq!(metadata.byte_size, bytes.len() as u64);
            assert_eq!(metadata.checksum_blake3, compute_blake3(&bytes));
            assert_eq!(metadata.checksum_sha256, compute_sha256(&bytes));
        }
        let artifact = &catalog.items[0];
        let path = original_root.join(&artifact.relative_path);
        fs::copy(&path, path.with_extension("preserved-original"))
            .expect("retain the real pre-corruption bytes");
        fs::write(&path, b"deliberately changed after publication").expect("actual tamper");
        assert!(matches!(
            handle.read_artifact_bytes(&artifact.artifact_id),
            Err(ControlError::InvalidPatch(_))
        ));
    }

    #[test]
    fn experiment_artifact_recovery_rejects_semantic_tampering_and_preserves_healthy_jobs() {
        let (handle, _host) = handle();
        let mut finished = Vec::new();
        for (id, seeds) in [("a-corrupt", vec![7, 8]), ("z-healthy", vec![9])] {
            let mut request = experiment_creation_request();
            request.experiment_id = Some(id.into());
            request.idempotency_key = Some(format!("key-{id}"));
            request.seeds = seeds;
            handle.create_experiment(request).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                let status = handle.get_experiment(id).unwrap();
                if status.status == "completed" {
                    finished.push(status);
                    break;
                }
                assert_ne!(status.status, "failed", "{status:?}");
                assert!(std::time::Instant::now() < deadline, "{status:?}");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        let root = handle.artifacts_dir();
        let original_catalog = handle.list_artifacts(Some(100), None).unwrap();
        let statuses = finished
            .iter()
            .map(|batch| {
                let path = root
                    .join("experiments")
                    .join(&batch.experiment_id)
                    .join("status.json");
                let bytes = fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect::<Vec<_>>();
        let bad_batch = &finished[0];
        assert_ne!(
            bad_batch.runs[0].final_digest,
            bad_batch.runs[1].final_digest
        );
        let bundle_dir = Path::new(bad_batch.runs[1].bundle_path.as_ref().unwrap());
        let manifest_path = bundle_dir.join("bundle_manifest.json");
        let evidence_path = bundle_dir.join("evidence/run.json");
        let original_manifest = fs::read(&manifest_path).unwrap();
        let original_evidence = fs::read(&evidence_path).unwrap();
        fs::copy(&manifest_path, root.join("preserved-bundle-manifest.json")).unwrap();
        fs::copy(&evidence_path, root.join("preserved-run-evidence.json")).unwrap();
        let mut evidence: Value = serde_json::from_slice(&original_evidence).unwrap();
        evidence["final_digest"] = bad_batch.runs[0].final_digest.clone().unwrap().into();
        let changed = serde_json::to_vec(&evidence).unwrap();
        fs::write(&evidence_path, &changed).unwrap();
        let mut manifest: scriptbots_storage::RunBundleV1 =
            serde_json::from_slice(&original_manifest).unwrap();
        let entry = manifest
            .artifacts
            .iter_mut()
            .find(|entry| entry.relative_path == "evidence/run.json")
            .unwrap();
        entry.blake3_hex = compute_blake3(&changed);
        entry.bytes_len = changed.len() as u64;
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        scriptbots_storage::verify_run_bundle(bundle_dir)
            .expect("byte-valid control must pass the generic verifier");

        let (mut restarted, _restarted_host) = self::handle();
        restarted.data_services = Arc::new(DataServices::new(root.clone()));
        assert!(
            matches!(
                restarted.list_artifacts(Some(100), None),
                Err(ControlError::Serialization(_))
            ),
            "science mismatch must refuse publication"
        );
        let healthy = original_catalog
            .items
            .iter()
            .filter(|item| item.relative_path.starts_with("experiments/z-healthy/"))
            .collect::<Vec<_>>();
        assert!(!healthy.is_empty());
        {
            let retained = restarted.data_services.artifacts.lock().unwrap();
            assert_eq!(
                retained.len(),
                healthy.len(),
                "no partial files from the bad two-run batch"
            );
            assert!(retained.values().all(|item| {
                item.metadata
                    .relative_path
                    .starts_with("experiments/z-healthy/")
            }));
        }
        let (mut direct, _direct_host) = self::handle();
        direct.data_services = Arc::new(DataServices::new(root.clone()));
        for artifact in healthy {
            let (_, bytes) = direct
                .read_artifact_bytes(&artifact.artifact_id)
                .expect("cold healthy lookup succeeds despite the companion refusal");
            assert_eq!(bytes, fs::read(root.join(&artifact.relative_path)).unwrap());
        }
        let bad_artifact = original_catalog
            .items
            .iter()
            .find(|item| item.relative_path.starts_with("experiments/a-corrupt/"))
            .unwrap();
        assert!(matches!(
            restarted.get_artifact(&bad_artifact.artifact_id),
            Err(ControlError::Serialization(_))
        ));
        fs::write(&evidence_path, &original_evidence).unwrap();
        fs::write(&manifest_path, &original_manifest).unwrap();
        let restored = restarted.list_artifacts(Some(100), None).unwrap();
        assert_eq!(
            serde_json::to_value(restored.items).unwrap(),
            serde_json::to_value(original_catalog.items).unwrap(),
            "same-generation repair must retry verification without duplicate identities"
        );
        for (path, bytes) in statuses {
            assert_eq!(
                fs::read(path).unwrap(),
                bytes,
                "discovery never rewrites science status"
            );
        }
    }

    #[test]
    fn experiment_creation_catalog_failure_is_not_completion() {
        let (handle, _host) = handle();
        let services = std::sync::Arc::clone(&handle.data_services);
        let poisoned = std::thread::spawn(move || {
            let _guard = services
                .artifacts
                .lock()
                .expect("initially healthy catalog");
            panic!("intentional artifact catalog poison");
        });
        assert!(poisoned.join().is_err());
        let created = handle
            .create_experiment(experiment_creation_request())
            .expect("admit run");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let failed = loop {
            let status = handle
                .get_experiment(&created.experiment_id)
                .expect("actual status");
            assert_ne!(status.status, "completed", "publication failed: {status:?}");
            if status.status == "failed" {
                break status;
            }
            assert!(std::time::Instant::now() < deadline, "{status:?}");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        assert_eq!(
            failed.completed_runs, 1,
            "science completed before publication failed"
        );
        assert_eq!(failed.failed_runs, 0);
        assert!(
            failed
                .error_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("artifact publication failed"))
        );
        scriptbots_storage::verify_run_bundle(Path::new(
            failed.runs[0]
                .bundle_path
                .as_ref()
                .expect("retained bundle"),
        ))
        .expect("publication failure preserves the real completed bundle");
        let capacity = handle
            .list_experiments(None, None)
            .unwrap()
            .execution_capacity;
        assert_eq!(capacity.reserved_run_slots, 0);
        assert_eq!(capacity.reserved_batches, 0);
        assert!(
            handle
                .data_services
                .artifacts
                .lock()
                .unwrap_err()
                .into_inner()
                .is_empty()
        );
    }

    #[test]
    fn experiment_creation_status_failure_retains_the_admitted_identity() {
        let (handle, _host) = handle();
        let definition_path =
            pending_catalog_definition(&handle.artifacts_dir(), "status-unreadable");
        let mut record: ExperimentCatalogRecord =
            serde_json::from_slice(&fs::read(&definition_path).unwrap()).unwrap();
        let definition = &record.definition;
        let request = ExperimentCreateRequest {
            experiment_id: Some(definition.experiment_id.clone()),
            description: None,
            variants: definition
                .variants
                .iter()
                .map(|v| ExperimentVariantDto {
                    variant_id: v.variant_id.clone(),
                    brain_family: v.brain_family.clone(),
                    config_overrides: Some(serde_json::to_value(&v.config_overrides).unwrap()),
                    interventions: v.interventions.clone(),
                })
                .collect(),
            seeds: definition.cohort.seeds.clone(),
            ticks_per_run: Some(definition.max_ticks),
            max_concurrency: Some(definition.max_concurrency),
            idempotency_key: Some("status-unreadable-key".into()),
        };
        record.definition.request_identity = Some((
            request.idempotency_key.clone().unwrap(),
            compute_blake3(&serde_json::to_vec(&request).unwrap()),
        ));
        record.checksum = compute_blake3(&serde_json::to_vec(&record.definition).unwrap());
        fs::write(&definition_path, serde_json::to_vec(&record).unwrap()).unwrap();
        // Model a crash after durable definition admission, before its first
        // fallible status observation. An orphan directory without a definition
        // is now refused before admission and cannot model this boundary.
        // The real directory at status.json still makes every status read fail.
        let status_path = handle
            .artifacts_dir()
            .join("experiments/status-unreadable/status.json");
        fs::create_dir_all(&status_path).expect("actual unreadable status path");
        assert!(matches!(
            handle.create_experiment(request.clone()),
            Err(ControlError::Serialization(_))
        ));
        assert_eq!(handle.data_services.experiments.lock().unwrap().len(), 1);
        assert!(matches!(
            handle.create_experiment(request),
            Err(ControlError::Serialization(_))
        ));
        assert_eq!(handle.data_services.experiments.lock().unwrap().len(), 1);
    }

    #[test]
    fn experiment_creation_poisoned_retry_authority_admits_nothing() {
        let (handle, _host) = handle();
        let services = std::sync::Arc::clone(&handle.data_services);
        let poisoned = std::thread::spawn(move || {
            let _guard = services.idempotency.lock().expect("initially unpoisoned");
            panic!("intentional retry-authority poison");
        });
        assert!(poisoned.join().is_err());
        assert!(matches!(
            handle.create_experiment(experiment_creation_request()),
            Err(ControlError::Lock)
        ));
        assert!(handle.data_services.experiments.lock().unwrap().is_empty());
        assert!(!handle.artifacts_dir().join("experiments").exists());
    }

    fn checkpoint_test_handle(
        directory: &Path,
        count: u64,
        bytes: u64,
    ) -> (ControlHandle, TestHost) {
        let host = TestHost::spawn(
            WorldState::new(ScriptBotsConfig {
                world_width: 40,
                world_height: 40,
                food_cell_size: 10,
                rng_seed: Some(42),
                persistence_interval: 0,
                population_minimum: 0,
                population_spawn_interval: 0,
                ..ScriptBotsConfig::default()
            })
            .expect("checkpoint world"),
        );
        let mut handle = host.handle();
        handle.data_services = Arc::new(DataServices::new(directory.to_path_buf()));
        let handle = handle
            .with_checkpoint_retention_limits(scriptbots_storage::CheckpointRetentionLimits {
                max_count: count,
                max_bytes: bytes,
            })
            .expect("checkpoint policy");
        (handle, host)
    }

    fn lab_checkpoint_descriptor(
        metadata: &CheckpointMetadataDto,
    ) -> scriptbots_runtime::lab_workflows::LabCheckpoint {
        scriptbots_runtime::lab_workflows::LabCheckpoint {
            checkpoint_id: metadata.checkpoint_id.clone(),
            tick: metadata.tick,
            agent_count: metadata.agent_count,
            byte_size: metadata.byte_size,
            world_digest: metadata.world_digest_v1.clone(),
        }
    }

    struct AgentCanvasJourney {
        handle: ControlHandle,
        host: TestHost,
        directory: PathBuf,
        checkpoint_id: String,
        oracles: Vec<(Arc<RenderSnapshot>, scriptbots_core::WorldDigestV1)>,
    }

    fn agent_canvas_commands(parents: [scriptbots_core::AgentUid; 2]) -> [(u128, HostCommand); 3] {
        [
            (
                101,
                HostCommand::SpawnAgent {
                    herbivore_tendency: 0.25,
                },
            ),
            (
                102,
                HostCommand::AdjustAgentMutationRates {
                    agent_uid: parents[0],
                    delta_primary: 0.1,
                    delta_secondary: 0.05,
                },
            ),
            (
                103,
                HostCommand::SpawnCrossover {
                    parent_a: parents[0],
                    parent_b: parents[1],
                },
            ),
        ]
    }

    fn agent_canvas_journey() -> AgentCanvasJourney {
        let directory = tempfile::Builder::new()
            .prefix("event-agent-canvas-")
            .tempdir()
            .unwrap()
            .keep();
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(42),
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            persistence_interval: 0,
            closed: true,
            topography_enabled: false,
            reproduction_energy_threshold: f32::MAX,
            metabolism_drain: 0.0,
            metabolism_ramp_rate: 0.0,
            metabolism_boost_penalty: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            spike_damage: 0.0,
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        let brains = crate::install_brains(&mut world, crate::BrainPreset::Mlp)
            .unwrap()
            .population;
        crate::seed_founders(&mut world, &brains, crate::FounderRecipe::Spread).unwrap();
        let host = TestHost::spawn(world);
        let handle = host.handle().with_artifacts_dir(directory.clone()).unwrap();
        let parents = handle
            .read_snapshot()
            .unwrap()
            .world
            .agents
            .iter()
            .take(2)
            .map(|agent| agent.uid)
            .collect::<Vec<_>>();
        assert_eq!(parents.len(), 2);
        let checkpoint = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        let submit = |id: u128, command| {
            let mut port = host.port.clone();
            let id = CommandId::new(id);
            port.submit(CommandEnvelope::new(id, command)).unwrap();
            assert_eq!(
                host.wait_finished(&id.to_string()).application_state,
                "applied",
                "actual command {id}: {:?}",
                port.command_status(id).unwrap()
            );
        };
        let mut oracles = Vec::new();
        // Checkpoint capture sealed tick zero. A real new tick opens arrivals again.
        host.wait_applied(&handle.step().unwrap());
        for (id, command) in agent_canvas_commands([parents[0], parents[1]]) {
            let population = handle.read_snapshot().unwrap().world.agents.len();
            let mutation = host
                .port
                .inspect_agent(parents[0], 0)
                .unwrap()
                .unwrap()
                .mutation_rates;
            let edits_mutation = matches!(&command, HostCommand::AdjustAgentMutationRates { .. });
            submit(id, command);
            if edits_mutation {
                let changed = host
                    .port
                    .inspect_agent(parents[0], 0)
                    .unwrap()
                    .unwrap()
                    .mutation_rates;
                assert_ne!(changed.primary.to_bits(), mutation.primary.to_bits());
                assert_ne!(changed.secondary.to_bits(), mutation.secondary.to_bits());
            } else {
                assert_eq!(
                    handle.read_snapshot().unwrap().world.agents.len(),
                    population + 1
                );
            }
            host.wait_applied(&handle.step().unwrap());
            oracles.push((
                handle.read_snapshot().unwrap(),
                host.port.scientific_digest_v1().unwrap(),
            ));
            host.wait_applied(&handle.step().unwrap());
        }
        for _ in 0..3 {
            host.wait_applied(&handle.step().unwrap());
        }
        AgentCanvasJourney {
            handle,
            host,
            directory,
            checkpoint_id: checkpoint.checkpoint_id,
            oracles,
        }
    }

    fn mixed_canvas_journey() -> AgentCanvasJourney {
        let mut journey = agent_canvas_journey();
        let handle = &journey.handle;
        let host = &journey.host;
        // Retain a real input prefix before the saved cut. Replay must rebase config
        // revisions and exclude those older commands, rather than assuming tick zero.
        host.wait_applied(
            &handle
                .apply_patch(serde_json::json!({"food_growth_rate": 0.005}))
                .unwrap(),
        );
        let checkpoint = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        journey.checkpoint_id = checkpoint.checkpoint_id;
        journey.oracles.clear();
        host.wait_applied(&handle.step().unwrap());
        let parents = handle
            .read_snapshot()
            .unwrap()
            .world
            .agents
            .iter()
            .take(2)
            .map(|agent| agent.uid)
            .collect::<Vec<_>>();
        assert_eq!(parents.len(), 2);
        for offset in [100_u128, 200] {
            if offset == 200 {
                host.wait_applied(
                    &handle
                        .apply_patch(serde_json::json!({"food_growth_rate": 0.006}))
                        .unwrap(),
                );
            }
            for ((id, command), rates) in agent_canvas_commands([parents[0], parents[1]])
                .into_iter()
                .zip([[0.004, 0.003], [0.002, 0.001], [0.0004, 0.0002]])
            {
                let mut port = host.port.clone();
                let id = CommandId::new(id + offset);
                port.submit(CommandEnvelope::new(id, command)).unwrap();
                assert_eq!(
                    host.wait_finished(&id.to_string()).application_state,
                    "applied"
                );
                // Two edits between commands prevent an implementation from merely
                // alternating one edit and one command at this shared tick.
                for rate in rates {
                    host.wait_applied(
                        &handle
                            .apply_patch(serde_json::json!({"food_growth_rate": rate}))
                            .unwrap(),
                    );
                }
            }
            journey.oracles.push((
                handle.read_snapshot().unwrap(),
                host.port.scientific_digest_v1().unwrap(),
            ));
            host.wait_applied(&handle.step().unwrap());
        }
        host.wait_applied(&handle.step().unwrap());
        journey
    }

    #[test]
    fn event_canvas_replays_actual_agent_commands_with_full_target_oracles() {
        let AgentCanvasJourney {
            handle,
            host,
            directory,
            checkpoint_id,
            oracles,
        } = agent_canvas_journey();
        let source = handle
            .data_services
            .checkpoint_retention
            .lock()
            .unwrap()
            .requests
            .values()
            .next()
            .unwrap()
            .clone();
        let saved = handle.verified_canvas_checkpoint(&checkpoint_id).unwrap();
        let publication = handle.read_snapshot().unwrap();
        let current = host.port.scientific_digest_cut().unwrap();
        assert_eq!(current.scientific_commands.len(), oracles.len());
        for (oracle, digest) in &oracles {
            let commands = ControlHandle::event_review_scientific_commands(
                &source,
                &current,
                oracle.world.tick,
                &[],
            )
            .unwrap();
            let target = ControlHandle::replay_event_canvas_world(
                &saved,
                &current.digest,
                &[],
                &commands,
                oracle.world.tick,
                0,
                source.source_config_revision,
                None,
            )
            .unwrap();
            assert_eq!(
                &target.world_digest_v1().unwrap(),
                digest,
                "full actual target oracle"
            );
            assert_ne!(*digest, current.digest);
            let review = handle.event_canvas(oracle.world.tick).unwrap();
            assert_eq!(review.snapshot.world.tick, oracle.world.tick);
            assert_eq!(
                review.snapshot.world.agents.len(),
                oracle.world.agents.len()
            );
            let (metadata, remote) = handle
                .replay_canvas_from_checkpoint_bounded(
                    handle.session_id(),
                    &checkpoint_id,
                    oracle.world.tick,
                )
                .unwrap();
            assert_eq!(metadata, source.metadata);
            assert_eq!(remote.checkpoint_id, checkpoint_id);
            assert_eq!(remote.verification, review.verification);
            assert_eq!(
                serde_json::to_value(&remote.snapshot.world).unwrap(),
                serde_json::to_value(&review.snapshot.world).unwrap(),
            );
            assert_eq!(host.port.scientific_digest_v1().unwrap(), current.digest);
            assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &publication));
            fs::write(
                directory.join(format!("target-{}-digest.json", oracle.world.tick)),
                serde_json::to_vec_pretty(digest).unwrap(),
            )
            .unwrap();
        }
        fs::write(
            directory.join("current-digest.json"),
            serde_json::to_vec_pretty(&current.digest).unwrap(),
        )
        .unwrap();
        eprintln!(
            "agent replay: targets={:?} live={} full current={} commands={}; {}",
            oracles
                .iter()
                .map(|(snapshot, _)| snapshot.world.tick)
                .collect::<Vec<_>>(),
            current.digest.tick.0,
            current.digest.overall,
            current.scientific_commands.len(),
            directory.display()
        );
    }

    #[test]
    fn event_canvas_agent_commands_refuse_incomplete_history() {
        let AgentCanvasJourney {
            handle,
            host,
            checkpoint_id,
            ..
        } = agent_canvas_journey();
        let retention = &handle.data_services.checkpoint_retention;
        let source = retention.lock().unwrap().requests[&checkpoint_id].clone();
        let saved = handle.verified_canvas_checkpoint(&checkpoint_id).unwrap();
        let current = host.port.scientific_digest_cut().unwrap();
        for fault in [
            "missing",
            "order",
            "duplicate",
            "session",
            "config",
            "science",
        ] {
            let mut changed = current.clone();
            match fault {
                "missing" => {
                    changed.scientific_commands.remove(1);
                }
                "order" => changed.scientific_commands.swap(0, 1),
                "duplicate" => changed
                    .scientific_commands
                    .insert(1, changed.scientific_commands[0].clone()),
                "session" => {
                    changed.session_id =
                        scriptbots_runtime::HostSessionId::new(current.session_id.get() ^ 1)
                }
                "config" => {
                    changed.config_revision =
                        scriptbots_runtime::ConfigRevision::new(current.config_revision.get() + 1)
                }
                "science" => {
                    changed.scientific_revision = scriptbots_runtime::ScientificRevision::new(
                        current.scientific_revision.get() + 1,
                    )
                }
                _ => unreachable!(),
            }
            assert!(
                ControlHandle::event_review_scientific_commands(&source, &changed, 2, &[]).is_err(),
                "{fault}"
            );
        }
        let mut changed_digest = current.digest.clone();
        changed_digest.overall = "0000000000000000".into();
        assert!(
            ControlHandle::replay_event_canvas_world(
                &saved,
                &changed_digest,
                &[],
                &current.scientific_commands,
                2,
                0,
                source.source_config_revision,
                None,
            )
            .is_err()
        );
        assert!(
            matches!(
                ControlHandle::replay_event_canvas_world(
                    &saved,
                    &current.digest,
                    &[],
                    &current.scientific_commands,
                    2,
                    0,
                    source.source_config_revision.checked_next().unwrap(),
                    None,
                ),
                Err(ControlError::InvalidPatch(message)) if message.contains("command configuration")
            ),
            "a wrong absolute source revision must still fail at the command guard"
        );
    }

    #[test]
    // One real owner journey checks queued target inputs, applied effects and mixed config order.
    #[allow(clippy::too_many_lines)]
    fn event_canvas_replays_interventions_and_pending_target_boundaries() {
        use scriptbots_core::{
            CohortSource, Intervention, InterventionCommand, InterventionSurface, Placement,
            Region, TerrainKind,
        };
        let AgentCanvasJourney {
            handle,
            host,
            checkpoint_id,
            ..
        } = mixed_canvas_journey();
        let retention = &handle.data_services.checkpoint_retention;
        let source = retention.lock().unwrap().requests[&checkpoint_id].clone();
        let saved = handle.verified_canvas_checkpoint(&checkpoint_id).unwrap();
        let brain_key = saved.required_brain_registry().entries[0].key;
        let interventions = [
            Intervention::Drought {
                region: Region::All,
                ticks: 3,
                growth_scale: 0.0,
            },
            Intervention::Embargo {
                region: Region::All,
                ticks: 2,
            },
            Intervention::Bloom {
                region: Region::All,
                amount: 0.1,
            },
            Intervention::Meteor {
                region: Region::All,
                lethality: 0.01,
                scorch: 0.25,
            },
            Intervention::PaintTerrain {
                region: Region::All,
                terrain: TerrainKind::Bloom,
                fertility_bias: Some(0.1),
            },
            Intervention::InjectCohort {
                count: 3,
                genome: CohortSource::RegisteredBrain { key: brain_key },
                placement: Placement::Seeded {
                    region: Region::All,
                    seed: 91,
                },
                herbivore_tendency: 0.75,
            },
            Intervention::SetClosedWorld { closed: false },
        ];
        let expected_interventions = interventions.len();
        let mut oracles = Vec::new();
        for (index, intervention) in (0_u16..).zip(interventions) {
            let kind = intervention.kind_label();
            let rate = f32::from(index + 1) / 1_000.0;
            host.wait_applied(
                &handle
                    .apply_patch(serde_json::json!({"food_growth_rate": rate}))
                    .unwrap(),
            );
            let receipt = handle
                .submit_control_command(ControlCommand::Intervention(Box::new(
                    InterventionCommand::new(
                        intervention,
                        InterventionSurface::Script,
                        "canvas-oracle",
                    ),
                )))
                .unwrap();
            host.wait_applied(&receipt);
            // A config edit after enqueue must keep its real position relative to the input.
            host.wait_applied(
                &handle
                    .apply_patch(serde_json::json!({"food_growth_rate": rate / 2.0}))
                    .unwrap(),
            );
            let pending = handle.read_snapshot().unwrap();
            assert!(
                host.port.scientific_digest_v1().is_err(),
                "queued input is not a settled cut"
            );
            assert!(
                handle
                    .replay_canvas_from_checkpoint(
                        handle.session_id(),
                        &lab_checkpoint_descriptor(&source.metadata),
                        pending.world.tick,
                    )
                    .is_err(),
                "a queued current input cannot provide the required settled endpoint"
            );
            assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &pending));
            host.wait_applied(&handle.step().unwrap());
            let applied = handle.read_snapshot().unwrap();
            assert!(
                applied
                    .applied_interventions
                    .iter()
                    .any(|record| record.kind == kind && !record.expired)
            );
            let digest = host.port.scientific_digest_v1().unwrap();
            oracles.push((pending, applied, digest));
            host.wait_applied(&handle.step().unwrap());
        }
        let current = host.port.scientific_digest_cut().unwrap();
        let published = handle.read_snapshot().unwrap();
        let audit =
            ControlHandle::event_review_config_audit(&source, &current, &published).unwrap();
        let commands = ControlHandle::event_review_scientific_commands(
            &source,
            &current,
            oracles[0].0.world.tick,
            &audit,
        )
        .unwrap();
        assert_eq!(
            commands
                .iter()
                .filter(|entry| matches!(entry.command(), ControlCommand::Intervention(_)))
                .count(),
            expected_interventions
        );
        for (pending, applied, digest) in &oracles {
            let mut target = ControlHandle::replay_event_canvas_world(
                &saved,
                &current.digest,
                &audit,
                &commands,
                pending.world.tick,
                source.source_persistence_interval,
                source.source_config_revision,
                None,
            )
            .unwrap();
            assert!(matches!(
                target.capture_checkpoint_quiescent(),
                Err(scriptbots_core::WorldCheckpointError::DeferredHostOutput {
                    field: "pending_interventions"
                })
            ));
            let review = handle
                .replay_canvas_from_checkpoint(
                    handle.session_id(),
                    &lab_checkpoint_descriptor(&source.metadata),
                    pending.world.tick,
                )
                .unwrap();
            assert_eq!(review.checkpoint_id, checkpoint_id);
            assert_eq!(
                review.verification,
                scriptbots_runtime::lab_workflows::LabReplayVerification::CapturedInterval {
                    verified_through_tick: current.digest.tick.0,
                    captured_current_digest: current.digest.overall.clone(),
                }
            );
            let narrative = handle.event_canvas(pending.world.tick).unwrap();
            assert_eq!(narrative.verification, review.verification);
            assert_eq!(narrative.snapshot.world.tick, pending.world.tick);
            assert_eq!(review.snapshot.world.tick, pending.world.tick);
            assert_eq!(
                review.snapshot.world.agents.len(),
                pending.world.agents.len()
            );
            assert_eq!(
                review.snapshot.config.food_growth_rate.to_bits(),
                pending.config.food_growth_rate.to_bits()
            );
            target.step().unwrap();
            assert_eq!(
                &target.world_digest_v1().unwrap(),
                digest,
                "the actual next-tick oracle proves the queued target input"
            );
            let target = ControlHandle::replay_event_canvas_world(
                &saved,
                &current.digest,
                &audit,
                &commands,
                applied.world.tick,
                source.source_persistence_interval,
                source.source_config_revision,
                None,
            )
            .unwrap();
            assert_eq!(
                &target.world_digest_v1().unwrap(),
                digest,
                "full applied-effect target oracle"
            );
        }
        let intervention_index = current
            .scientific_commands
            .iter()
            .position(|entry| matches!(entry.command(), ControlCommand::Intervention(_)))
            .unwrap();
        for duplicate in [false, true] {
            let mut changed = current.clone();
            if duplicate {
                let entry = changed.scientific_commands[intervention_index].clone();
                changed
                    .scientific_commands
                    .insert(intervention_index, entry);
            } else {
                changed.scientific_commands.remove(intervention_index);
            }
            assert!(
                ControlHandle::event_review_scientific_commands(
                    &source,
                    &changed,
                    oracles[0].0.world.tick,
                    &audit,
                )
                .is_err()
            );
        }
        assert_eq!(host.port.scientific_digest_v1().unwrap(), current.digest);
        assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &published));
    }

    #[test]
    fn event_canvas_retains_archive_input_admission_refusal_for_full_sources() {
        let directory = tempfile::Builder::new()
            .prefix("event-archive-authority-")
            .tempdir()
            .unwrap()
            .keep();
        let host = TestHost::spawn(
            WorldState::new(ScriptBotsConfig {
                world_width: 40,
                world_height: 40,
                food_cell_size: 10,
                rng_seed: Some(42),
                persistence_interval: 0,
                population_minimum: 0,
                population_spawn_interval: 0,
                closed: true,
                archive_enabled: true,
                ..ScriptBotsConfig::default()
            })
            .unwrap(),
        );
        let handle = host.handle().with_artifacts_dir(directory).unwrap();
        let checkpoint = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        assert_eq!(
            checkpoint.schema,
            scriptbots_runtime::HOST_CHECKPOINT_V1_SCHEMA
        );
        host.wait_applied(&handle.step().unwrap());
        let target = handle.read_snapshot().unwrap().world.tick;
        let command = handle
            .submit_control_command(ControlCommand::Intervention(Box::new(
                scriptbots_core::InterventionCommand::new(
                    scriptbots_core::Intervention::SpawnFromArchive {
                        selector: scriptbots_core::CellSelector::All,
                        placement: scriptbots_core::Placement::Seeded {
                            region: scriptbots_core::Region::All,
                            seed: 1,
                        },
                        headroom: None,
                        herbivore_tendency: None,
                    },
                    scriptbots_core::InterventionSurface::Script,
                    "archive-authority-guard",
                ),
            )))
            .unwrap();
        host.wait_applied(&command);
        host.wait_applied(&handle.step().unwrap());
        let current = host.port.scientific_digest_cut().unwrap();
        assert_eq!(current.scientific_commands.len(), 1);
        assert!(matches!(
            handle.event_canvas(target),
            Err(ControlError::BadRequest(message)) if message.contains("source-bound behavioral archive")
        ));
        assert_eq!(
            handle
                .event_canvas(checkpoint.tick)
                .unwrap()
                .snapshot
                .world
                .tick,
            checkpoint.tick,
            "the independently verified source cut remains directly inspectable"
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), current.digest);
    }

    #[test]
    fn event_canvas_replays_mixed_inputs_in_observed_same_tick_order() {
        let AgentCanvasJourney {
            handle,
            host,
            directory,
            checkpoint_id,
            oracles,
        } = mixed_canvas_journey();
        let retention = &handle.data_services.checkpoint_retention;
        let source = retention.lock().unwrap().requests[&checkpoint_id].clone();
        let saved = handle.verified_canvas_checkpoint(&checkpoint_id).unwrap();
        let current = host.port.scientific_digest_cut().unwrap();
        let published = handle.read_snapshot().unwrap();
        let audit =
            ControlHandle::event_review_config_audit(&source, &current, &published).unwrap();
        let commands = ControlHandle::event_review_scientific_commands(
            &source,
            &current,
            oracles[0].0.world.tick,
            &audit,
        )
        .unwrap();
        assert_eq!(source.source_config_revision.get(), 1);
        assert_eq!(audit.len(), 13);
        assert_eq!(current.scientific_commands.len(), 9);
        assert_eq!(
            commands
                .iter()
                .map(|entry| entry.boundary().revisions.config.get())
                .collect::<Vec<_>>(),
            [1, 3, 5, 8, 10, 12],
            "actual commands must observe distinct positions within each tick's edits"
        );
        assert_eq!(
            commands
                .iter()
                .map(|entry| entry.boundary().tick.0)
                .collect::<Vec<_>>(),
            [11, 11, 11, 12, 12, 12]
        );
        for (oracle, digest) in &oracles {
            let target = ControlHandle::replay_event_canvas_world(
                &saved,
                &current.digest,
                &audit,
                &commands,
                oracle.world.tick,
                source.source_persistence_interval,
                source.source_config_revision,
                None,
            )
            .unwrap();
            assert_eq!(&target.world_digest_v1().unwrap(), digest);
            assert_ne!(*digest, current.digest);
            let review = handle.event_canvas(oracle.world.tick).unwrap();
            assert_eq!(review.checkpoint_id, checkpoint_id);
            assert_eq!(review.base_tick, saved.tick().0);
            assert_eq!(review.snapshot.world.tick, oracle.world.tick);
            assert_eq!(
                review.snapshot.world.agents.len(),
                oracle.world.agents.len()
            );
            assert_eq!(
                review.snapshot.config.food_growth_rate.to_bits(),
                oracle.config.food_growth_rate.to_bits()
            );
            assert_eq!(host.port.scientific_digest_v1().unwrap(), current.digest);
            assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &published));
            fs::write(
                directory.join(format!("mixed-target-{}-digest.json", oracle.world.tick)),
                serde_json::to_vec_pretty(digest).unwrap(),
            )
            .unwrap();
        }
        eprintln!(
            "mixed replay: source={} targets=11,12 live={} commands={} edits={} current={}; {}",
            saved.tick().0,
            current.digest.tick.0,
            commands.len(),
            audit.len(),
            current.digest.overall,
            directory.display()
        );
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one real owner journey checks historical review across handoff and a later omitted input"
    )]
    fn event_canvas_preserves_mixed_input_review_after_owner_continuation() {
        use scriptbots_runtime::{HostCore, ManualInstant, VolatileHostCheckpointV1};

        let AgentCanvasJourney {
            handle,
            host,
            directory,
            checkpoint_id,
            oracles,
        } = mixed_canvas_journey();
        let retention = &handle.data_services.checkpoint_retention;
        let source = retention.lock().unwrap().requests[&checkpoint_id].clone();
        let saved = handle.verified_canvas_checkpoint(&checkpoint_id).unwrap();
        let before = host.port.scientific_digest_cut().unwrap();
        let reviews = oracles
            .iter()
            .map(|(oracle, _)| handle.event_canvas(oracle.world.tick).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(before.scientific_commands.len(), 9);

        let bytes = host.handoff();
        fs::write(directory.join("mixed-review-owner.checkpoint"), &bytes).unwrap();
        let continued = TestHost::spawn_with_owner(move || {
            let paired = VolatileHostCheckpointV1::decode(&bytes).unwrap();
            let checkpoint = paired.host_checkpoint();
            let registry =
                crate::brains::create_brain_registry_for_checkpoint(checkpoint.world_checkpoint())
                    .unwrap();
            let world = checkpoint.restore_world(registry).unwrap();
            let persistence = world
                .bind_persistence(Box::new(scriptbots_core::NullPersistence))
                .unwrap();
            HostCore::restore_volatile_host_checkpoint_v1(
                &paired,
                world,
                persistence,
                None,
                ManualInstant::from_nanos(0),
            )
            .unwrap()
        });
        let mut restored_handle = continued.handle();
        // Source checkpoints remain the same retained application artifacts. Only
        // the actual owner and its journal were handed off and reconstructed.
        restored_handle.data_services = Arc::clone(&handle.data_services);
        let current = continued.port.scientific_digest_cut().unwrap();
        let published = restored_handle.read_snapshot().unwrap();
        assert_eq!(current.digest, before.digest);
        assert_eq!(current.session_id, before.session_id);
        assert_eq!(current.scientific_revision, before.scientific_revision);
        assert_eq!(current.config_revision, before.config_revision);
        assert_eq!(
            current.scientific_commands.len(),
            before.scientific_commands.len()
        );
        for (actual, expected) in current
            .scientific_commands
            .iter()
            .zip(&before.scientific_commands)
        {
            assert_eq!(actual.boundary(), expected.boundary());
            assert_eq!(actual.command(), expected.command());
        }
        let audit =
            ControlHandle::event_review_config_audit(&source, &current, &published).unwrap();
        for ((oracle, digest), before_review) in oracles.iter().zip(reviews) {
            let commands = ControlHandle::event_review_scientific_commands(
                &source,
                &current,
                oracle.world.tick,
                &audit,
            )
            .unwrap();
            let target = ControlHandle::replay_event_canvas_world(
                &saved,
                &current.digest,
                &audit,
                &commands,
                oracle.world.tick,
                source.source_persistence_interval,
                source.source_config_revision,
                None,
            )
            .unwrap();
            assert_eq!(&target.world_digest_v1().unwrap(), digest);
            let review = restored_handle.event_canvas(oracle.world.tick).unwrap();
            assert_eq!(review.checkpoint_id, checkpoint_id);
            assert_eq!(review.base_tick, saved.tick().0);
            assert_eq!(
                serde_json::to_value(&review.snapshot.world).unwrap(),
                serde_json::to_value(&before_review.snapshot.world).unwrap(),
                "the historical canvas is identical across real owner continuation"
            );
            assert_eq!(
                continued.port.scientific_digest_v1().unwrap(),
                current.digest,
                "historical review cannot mutate the continued owner"
            );
            assert!(Arc::ptr_eq(
                &restored_handle.read_snapshot().unwrap(),
                &published
            ));
        }

        // A real accepted input larger than the history budget leaves a revision
        // gap. Restoring earlier history must not turn that gap into a replay proof.
        let omitted = restored_handle
            .submit_command(
                ControlCommand::Intervention(Box::new(scriptbots_core::InterventionCommand::new(
                    scriptbots_core::Intervention::Embargo {
                        region: scriptbots_core::Region::All,
                        ticks: 2,
                    },
                    scriptbots_core::InterventionSurface::Script,
                    "x".repeat(scriptbots_runtime::MAX_SCIENTIFIC_COMMAND_HISTORY_BYTES + 1),
                ))),
                None,
            )
            .unwrap();
        continued.wait_applied(&omitted);
        continued.wait_applied(&restored_handle.step().unwrap());
        let incomplete = continued.port.scientific_digest_cut().unwrap();
        let publication = restored_handle.read_snapshot().unwrap();
        assert_eq!(
            incomplete.scientific_commands.len(),
            current.scientific_commands.len()
        );
        assert_eq!(
            incomplete.scientific_revision.get(),
            current.scientific_revision.get() + 2
        );
        assert!(matches!(
            restored_handle.event_canvas(oracles[0].0.world.tick),
            Err(ControlError::BadRequest(message))
                if message.contains("complete recorded scientific inputs")
        ));
        assert_eq!(
            continued.port.scientific_digest_v1().unwrap(),
            incomplete.digest
        );
        assert!(Arc::ptr_eq(
            &restored_handle.read_snapshot().unwrap(),
            &publication
        ));
    }

    #[test]
    // One real mixed-history cut supplies both cross-history and final-digest negatives.
    #[allow(clippy::too_many_lines)]
    fn event_canvas_mixed_inputs_refuse_inconsistent_histories() {
        let AgentCanvasJourney {
            handle,
            host,
            checkpoint_id,
            oracles,
            ..
        } = mixed_canvas_journey();
        let retention = &handle.data_services.checkpoint_retention;
        let source = retention.lock().unwrap().requests[&checkpoint_id].clone();
        let saved = handle.verified_canvas_checkpoint(&checkpoint_id).unwrap();
        let current = host.port.scientific_digest_cut().unwrap();
        let published = handle.read_snapshot().unwrap();
        let target_tick = oracles[0].0.world.tick;
        let audit =
            ControlHandle::event_review_config_audit(&source, &current, &published).unwrap();
        let commands =
            ControlHandle::event_review_scientific_commands(&source, &current, target_tick, &audit)
                .unwrap();
        for fault in ["missing", "early", "late", "future", "order"] {
            let mut changed = audit.clone();
            match fault {
                "missing" => {
                    changed.remove(0);
                }
                "early" => changed[0].tick = source.metadata.tick,
                "late" => {
                    for entry in &mut changed {
                        if entry.tick == target_tick {
                            entry.tick += 1;
                        }
                    }
                }
                "future" => changed.last_mut().unwrap().tick = current.digest.tick.0 + 1,
                "order" => changed.last_mut().unwrap().tick = source.metadata.tick,
                _ => unreachable!(),
            }
            if matches!(fault, "early" | "late") {
                let mut altered_publication = (*published).clone();
                let prefix = altered_publication.config_audit.len() - changed.len();
                Arc::make_mut(&mut altered_publication.config_audit)[prefix..]
                    .clone_from_slice(&changed);
                assert!(
                    ControlHandle::event_review_config_audit(
                        &source,
                        &current,
                        &altered_publication,
                    )
                    .is_ok(),
                    "the {fault} audit is internally complete but contradicts observed commands"
                );
            }
            assert!(
                ControlHandle::event_review_scientific_commands(
                    &source,
                    &current,
                    target_tick,
                    &changed,
                )
                .is_err(),
                "{fault}"
            );
        }
        for fault in [
            "missing",
            "duplicate",
            "order",
            "session",
            "science",
            "config",
            "saturated",
        ] {
            let mut changed = current.clone();
            let last = changed.scientific_commands.len() - 1;
            match fault {
                "missing" => {
                    changed.scientific_commands.remove(last);
                }
                "duplicate" => {
                    let duplicate = changed.scientific_commands[last].clone();
                    changed.scientific_commands.push(duplicate);
                }
                "order" => changed.scientific_commands.swap(last - 1, last),
                "session" => {
                    changed.session_id =
                        scriptbots_runtime::HostSessionId::new(current.session_id.get() ^ 1)
                }
                "science" => {
                    changed.scientific_revision =
                        current.scientific_revision.checked_next().unwrap()
                }
                "config" => {
                    changed.config_revision = current.config_revision.checked_next().unwrap()
                }
                "saturated" => {
                    changed.config_revision = scriptbots_runtime::ConfigRevision::new(u64::MAX)
                }
                _ => unreachable!(),
            }
            assert!(
                ControlHandle::event_review_scientific_commands(
                    &source,
                    &changed,
                    target_tick,
                    &audit,
                )
                .is_err(),
                "{fault}"
            );
        }
        for fault in ["audit-value", "current-digest"] {
            let mut changed = audit.clone();
            let mut digest = current.digest.clone();
            if fault == "audit-value" {
                changed.last_mut().unwrap().patch["food_growth_rate"] = serde_json::json!(0.009);
            } else {
                digest.overall = "0000000000000000".into();
            }
            assert!(
                matches!(
                    ControlHandle::replay_event_canvas_world(
                        &saved,
                        &digest,
                        &changed,
                        &commands,
                        target_tick,
                        source.source_persistence_interval,
                        source.source_config_revision,
                        None,
                    ),
                    Err(ControlError::InvalidPatch(message))
                        if message.contains("current scientific digest")
                ),
                "{fault}"
            );
        }
        assert_eq!(host.port.scientific_digest_v1().unwrap(), current.digest);
        assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &published));
    }

    #[test]
    // Actual scheduled applications plus corrupt outcome/audit negatives share one owner.
    #[allow(clippy::too_many_lines)]
    fn event_canvas_replays_completed_configuration_schedules_with_full_target_oracles() {
        let directory = tempfile::Builder::new()
            .prefix("event-scheduled-canvas-")
            .tempdir()
            .unwrap()
            .keep();
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(42),
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            persistence_interval: 0,
            closed: true,
            topography_enabled: false,
            food_growth_rate: 0.001,
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        let brains = crate::install_brains(&mut world, crate::BrainPreset::Mlp)
            .unwrap()
            .population;
        crate::seed_founders(&mut world, &brains, crate::FounderRecipe::Spread).unwrap();
        let schedule = [(1, 0.004), (1, 0.002), (4, 0.0002), (6, 0.001)]
            .into_iter()
            .enumerate()
            .map(|(index, (tick, rate))| {
                scriptbots_runtime::ScheduledConfigPatch::new(
                    index as u64 + 1,
                    Tick(tick),
                    serde_json::json!({"food_growth_rate":rate}),
                )
                .unwrap()
            })
            .collect();
        let host = TestHost::spawn_with_schedule(world, Some(schedule));
        let handle = host.handle().with_artifacts_dir(directory.clone()).unwrap();
        let checkpoint = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        let mut oracles = Vec::new();
        for _ in 0..8 {
            host.wait_applied(&handle.step().unwrap());
            oracles.push((
                handle.read_snapshot().unwrap(),
                host.port.scientific_digest_v1().unwrap(),
            ));
        }
        let current = host.port.scientific_digest_cut().unwrap();
        let published = handle.read_snapshot().unwrap();
        let retained = handle
            .data_services
            .checkpoint_retention
            .lock()
            .unwrap()
            .requests
            .values()
            .next()
            .unwrap()
            .clone();
        assert_eq!(retained.source_scheduled_input_count, 4);
        assert_eq!(published.scheduled_patches.len(), 4);
        assert_eq!(
            current.scheduled_input_count, 4,
            "exhaustion retains the declaration"
        );
        assert_eq!(
            published
                .scheduled_patches
                .iter()
                .map(|s| s.evidence.boundary.tick.0)
                .collect::<Vec<_>>(),
            [1, 1, 4, 6]
        );
        let audit =
            ControlHandle::event_review_config_audit(&retained, &current, &published).unwrap();
        let saved = handle
            .verified_canvas_checkpoint(&checkpoint.checkpoint_id)
            .unwrap();
        // Avoid same-boundary ambiguity: each target follows application of its due edits.
        for (oracle, digest) in [&oracles[1], &oracles[4], &oracles[6]] {
            let target = ControlHandle::replay_event_canvas_world(
                &saved,
                &current.digest,
                &audit,
                &[],
                oracle.world.tick,
                0,
                retained.source_config_revision,
                None,
            )
            .unwrap();
            assert_eq!(
                &target.world_digest_v1().unwrap(),
                digest,
                "full actual target digest"
            );
            let review = handle.event_canvas(oracle.world.tick).unwrap();
            assert_eq!(review.snapshot.world.tick, oracle.world.tick);
            assert_eq!(
                review.snapshot.config.food_growth_rate.to_bits(),
                oracle.config.food_growth_rate.to_bits()
            );
            assert_ne!(*digest, current.digest);
        }
        assert_ne!(
            oracles[1].0.config.food_growth_rate.to_bits(),
            oracles[4].0.config.food_growth_rate.to_bits()
        );
        for fault in [
            "missing",
            "order",
            "pending",
            "failed",
            "session",
            "future",
            "revision",
            "declaration",
        ] {
            let mut changed = (*published).clone();
            let mut source = retained.clone();
            let statuses = Arc::make_mut(&mut changed.scheduled_patches);
            match fault {
                "missing" => {
                    statuses.remove(0);
                }
                "order" => statuses.swap(0, 1),
                "pending" => statuses[0].journal = scriptbots_runtime::JournalState::Pending,
                "failed" => {
                    statuses[0].evidence.outcome = scriptbots_runtime::ScheduledPatchOutcome::Failed(
                        scriptbots_runtime::ApplicationFailure {
                            code: "test_failure".into(),
                            message: "not applied".into(),
                        },
                    )
                }
                "session" => {
                    statuses[0].batch_id = scriptbots_runtime::JournalBatchId::new(
                        scriptbots_runtime::HostSessionId::new(77),
                        1,
                    )
                }
                "future" => {
                    statuses[3].evidence.boundary.tick = Tick(9);
                    statuses[3].evidence.scheduled.due_tick = Tick(9);
                }
                "revision" => {
                    statuses[3].evidence.boundary.revisions.config =
                        scriptbots_runtime::ConfigRevision::new(99)
                }
                "declaration" => source.source_scheduled_input_count = 0,
                _ => unreachable!(),
            }
            assert!(
                matches!(
                    ControlHandle::validate_event_review_schedule(&source, &current, &changed),
                    Err(ControlError::BadRequest(_))
                ),
                "{fault}"
            );
        }
        let mut missing_audit = (*published).clone();
        Arc::make_mut(&mut missing_audit.config_audit).remove(0);
        assert!(
            ControlHandle::event_review_config_audit(&retained, &current, &missing_audit).is_err()
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), current.digest);
        assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &published));
        fs::write(
            directory.join("schedule.json"),
            serde_json::to_vec_pretty(published.scheduled_patches.as_ref()).unwrap(),
        )
        .unwrap();
        fs::write(
            directory.join("current-digest.json"),
            serde_json::to_vec_pretty(&current.digest).unwrap(),
        )
        .unwrap();
        eprintln!(
            "completed configuration schedule: source={} targets=2,5,7 live={} digest={}; {}",
            checkpoint.tick,
            current.digest.tick.0,
            current.digest.overall,
            directory.display()
        );
        // An exhausted schedule before the source cut also permits the tick-only
        // path; retaining old outcomes is not mistaken for new configuration work.
        let later = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        host.wait_applied(&handle.step().unwrap());
        let later_oracle = host.port.scientific_digest_v1().unwrap();
        host.wait_applied(&handle.step().unwrap());
        assert_eq!(
            handle.event_canvas(later_oracle.tick.0).unwrap().base_tick,
            later.tick
        );
        let science = handle
            .submit_control_command(ControlCommand::SpawnAgent {
                herbivore_tendency: 0.5,
            })
            .unwrap();
        host.wait_applied(&science);
        let after_spawn = host.port.scientific_digest_v1().unwrap();
        let review = handle.event_canvas(later_oracle.tick.0).unwrap();
        assert_eq!(review.base_tick, later.tick);
        assert_eq!(review.snapshot.world.tick, later_oracle.tick.0);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), after_spawn);
        // An actual queued intervention now has a replayable owner input.
        let intervention = handle
            .submit_control_command(ControlCommand::Intervention(Box::new(
                scriptbots_core::interventions::InterventionCommand::new(
                    scriptbots_core::Intervention::Embargo {
                        region: scriptbots_core::Region::All,
                        ticks: 2,
                    },
                    scriptbots_core::interventions::InterventionSurface::Script,
                    "event-review-intervention",
                ),
            )))
            .unwrap();
        host.wait_applied(&intervention);
        // Enqueue is not application: settle the intervention at a real tick before
        // requiring the same quiescent current-digest oracle used by event review.
        host.wait_applied(&handle.step().unwrap());
        assert!(
            handle
                .read_snapshot()
                .unwrap()
                .applied_interventions
                .iter()
                .any(|record| record.kind == "embargo" && !record.expired)
        );
        let after_intervention = host.port.scientific_digest_v1().unwrap();
        let review = handle.event_canvas(later_oracle.tick.0).unwrap();
        assert_eq!(review.base_tick, later.tick);
        assert_eq!(
            host.port.scientific_digest_v1().unwrap(),
            after_intervention
        );
        // Oversized provenance still applies through the real owner but cannot enter
        // bounded replay history. Its scientific revision must expose that missing input.
        let oversized = handle
            .submit_control_command(ControlCommand::Intervention(Box::new(
                scriptbots_core::InterventionCommand::new(
                    scriptbots_core::Intervention::Embargo {
                        region: scriptbots_core::Region::All,
                        ticks: 2,
                    },
                    scriptbots_core::InterventionSurface::Script,
                    "x".repeat(scriptbots_runtime::MAX_SCIENTIFIC_COMMAND_HISTORY_BYTES + 1),
                ),
            )))
            .unwrap();
        host.wait_applied(&oversized);
        host.wait_applied(&handle.step().unwrap());
        let refusal = handle
            .event_canvas(later_oracle.tick.0)
            .err()
            .expect("an applied input omitted from bounded history must refuse event review");
        assert!(
            matches!(&refusal, ControlError::BadRequest(message) if message.contains("scientific revisions")),
            "{refusal}"
        );
    }

    #[test]
    // One actual owner journey, including anchor corruption and pruning, per persistence mode.
    #[allow(clippy::too_many_lines)]
    fn event_canvas_replays_recorded_config_edits_at_their_actual_boundaries() {
        let mut distinct = std::collections::BTreeSet::new();
        for persistence_interval in [0, 3] {
            let directory = tempfile::Builder::new()
                .prefix("event-config-canvas-")
                .tempdir()
                .unwrap()
                .keep();
            let mut world = WorldState::new(ScriptBotsConfig {
                rng_seed: Some(42),
                world_width: 120,
                world_height: 120,
                food_cell_size: 20,
                persistence_interval,
                closed: true,
                topography_enabled: false,
                initial_food: 0.0,
                food_intake_rate: 0.0,
                food_decay_rate: 0.0,
                food_diffusion_rate: 0.0,
                food_respawn_interval: 0,
                metabolism_drain: 0.0,
                movement_drain: 0.0,
                reproduction_energy_threshold: 0.0,
                ..ScriptBotsConfig::default()
            })
            .unwrap();
            let brains = crate::install_brains(&mut world, crate::BrainPreset::Mlp)
                .unwrap()
                .population;
            crate::seed_founders(&mut world, &brains, crate::FounderRecipe::Spread).unwrap();
            let host = TestHost::spawn(world);
            let handle = host.handle().with_artifacts_dir(directory.clone()).unwrap();
            host.wait_applied(
                &handle
                    .apply_patch(serde_json::json!({"food_growth_rate":0.001}))
                    .unwrap(),
            );
            host.wait_applied(&handle.step().unwrap());
            let checkpoint = handle
                .create_checkpoint(CheckpointCreateRequest::default())
                .unwrap();
            // Two edits at the source boundary, plus edits at the target and later boundaries.
            for rate in [0.004, 0.002] {
                host.wait_applied(
                    &handle
                        .apply_patch(serde_json::json!({"food_growth_rate":rate}))
                        .unwrap(),
                );
            }
            for _ in 0..3 {
                host.wait_applied(&handle.step().unwrap());
            }
            host.wait_applied(
                &handle
                    .apply_patch(serde_json::json!({"food_growth_rate":0.0002}))
                    .unwrap(),
            );
            let oracle = handle.read_snapshot().unwrap();
            let oracle_digest = host.port.scientific_digest_v1().unwrap();
            for _ in 0..2 {
                host.wait_applied(&handle.step().unwrap());
            }
            host.wait_applied(
                &handle
                    .apply_patch(serde_json::json!({"food_growth_rate":0.001}))
                    .unwrap(),
            );
            for _ in 0..2 {
                host.wait_applied(&handle.step().unwrap());
            }
            let current = host.port.scientific_digest_cut().unwrap();
            let published = handle.read_snapshot().unwrap();
            let source = handle
                .data_services
                .checkpoint_retention
                .lock()
                .unwrap()
                .requests
                .values()
                .next()
                .unwrap()
                .clone();
            assert_eq!(
                source.source_config_revision.get(),
                1,
                "nonempty audit prefix"
            );
            assert_eq!(current.config_revision.get(), 5);
            assert_eq!(source.source_persistence_interval, persistence_interval);
            let audit =
                ControlHandle::event_review_config_audit(&source, &current, &published).unwrap();
            assert_eq!(
                audit.iter().map(|entry| entry.tick).collect::<Vec<_>>(),
                [1, 1, 4, 6]
            );
            let saved = handle
                .verified_canvas_checkpoint(&checkpoint.checkpoint_id)
                .unwrap();
            let target = ControlHandle::replay_event_canvas_world(
                &saved,
                &current.digest,
                &audit,
                &[],
                oracle.world.tick,
                source.source_persistence_interval,
                source.source_config_revision,
                None,
            )
            .unwrap();
            assert_eq!(
                target.world_digest_v1().unwrap(),
                oracle_digest,
                "full target science oracle"
            );
            assert_eq!(target.config().persistence_interval, 0);
            assert_ne!(
                oracle_digest, current.digest,
                "actual target/live science differs"
            );
            distinct.insert(current.digest.overall.clone());
            let review = handle.event_canvas(oracle.world.tick).unwrap();
            assert_eq!(review.base_tick, checkpoint.tick);
            assert_eq!(review.snapshot.world.tick, oracle.world.tick);
            assert_eq!(
                review.snapshot.config.food_growth_rate.to_bits(),
                oracle.config.food_growth_rate.to_bits()
            );
            assert_ne!(
                review.snapshot.config.food_growth_rate.to_bits(),
                published.config.food_growth_rate.to_bits()
            );
            for fault in ["value", "tick", "missing", "persistence"] {
                let mut changed = audit.clone();
                match fault {
                    "value" => changed[1].patch["food_growth_rate"] = serde_json::json!(0.000001),
                    "tick" => changed[2].tick += 1,
                    "missing" => {
                        changed.remove(2);
                    }
                    "persistence" => {
                        changed[0].patch["persistence_interval"] = serde_json::json!(7)
                    }
                    _ => unreachable!(),
                }
                assert!(
                    ControlHandle::replay_event_canvas_world(
                        &saved,
                        &current.digest,
                        &changed,
                        &[],
                        oracle.world.tick,
                        source.source_persistence_interval,
                        source.source_config_revision,
                        None,
                    )
                    .is_err(),
                    "{fault}"
                );
            }
            for fault in ["pruned", "order", "future", "session", "saturated", "older"] {
                let mut changed = (*published).clone();
                match fault {
                    "pruned" => {
                        Arc::make_mut(&mut changed.config_audit).drain(..2);
                    }
                    "order" => Arc::make_mut(&mut changed.config_audit)[3].tick = 0,
                    "future" => {
                        Arc::make_mut(&mut changed.config_audit)[3].tick = current.digest.tick.0 + 1
                    }
                    "session" => {
                        changed.session_id =
                            scriptbots_runtime::HostSessionId::new(current.session_id.get() ^ 1)
                    }
                    "saturated" => {
                        changed.revisions.config = scriptbots_runtime::ConfigRevision::new(u64::MAX)
                    }
                    "older" => changed.revisions.config = source.source_config_revision,
                    _ => unreachable!(),
                }
                assert!(
                    ControlHandle::event_review_config_audit(&source, &current, &changed).is_err(),
                    "{fault}"
                );
            }
            assert_eq!(host.port.scientific_digest_v1().unwrap(), current.digest);
            assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &published));
            fs::write(
                directory.join("target-digest.json"),
                serde_json::to_vec_pretty(&oracle_digest).unwrap(),
            )
            .unwrap();
            fs::write(
                directory.join("current-digest.json"),
                serde_json::to_vec_pretty(&current.digest).unwrap(),
            )
            .unwrap();
            eprintln!(
                "config replay: interval={persistence_interval} base={} target={} live={} revisions=1..5 full target={} current={}; {}",
                checkpoint.tick,
                oracle.world.tick,
                current.digest.tick.0,
                oracle_digest.overall,
                current.digest.overall,
                directory.display()
            );
            // The actual bounded ring must refuse rather than fabricate a missing prefix.
            let mut updates = 0;
            loop {
                let snapshot = handle.read_snapshot().unwrap();
                if snapshot.revisions.config.get() - source.source_config_revision.get()
                    > snapshot.config_audit.len() as u64
                {
                    break;
                }
                assert!(
                    updates < MAX_EVENT_REVIEW_TICKS,
                    "observe audit pruning within the work bound"
                );
                host.wait_applied(&handle.apply_patch(serde_json::json!({"food_growth_rate":if updates % 2 == 0 {0.02} else {0.01}})).unwrap());
                updates += 1;
            }
            assert!(
                matches!(handle.event_canvas(oracle.world.tick), Err(ControlError::BadRequest(message)) if message.contains("complete ordered configuration audit"))
            );
            assert!(matches!(
                handle.replay_canvas_from_checkpoint(
                    handle.session_id(),
                    &lab_checkpoint_descriptor(&checkpoint),
                    oracle.world.tick,
                ),
                Err(ControlError::BadRequest(message))
                    if message.contains("complete ordered configuration audit")
            ));
        }
        assert_eq!(
            distinct.len(),
            1,
            "persistence isolation preserves identical science"
        );
    }

    #[test]
    fn event_canvas_replays_real_tick_only_intervals_and_rejects_missing_inputs() {
        let mut digests = std::collections::BTreeSet::new();
        for preset in [
            None,
            Some(crate::BrainPreset::Mlp),
            Some(crate::BrainPreset::Dwraon),
            Some(crate::BrainPreset::Assembly),
            Some(crate::BrainPreset::Mixed),
            #[cfg(feature = "neuro")]
            Some(crate::BrainPreset::Neuro),
        ] {
            let directory = tempfile::Builder::new()
                .prefix("event-canvas-control-")
                .tempdir()
                .unwrap()
                .keep();
            eprintln!("event canvas preset {preset:?}: {}", directory.display());
            let mut config = ScriptBotsConfig {
                rng_seed: Some(42),
                world_width: 120,
                world_height: 120,
                food_cell_size: 20,
                persistence_interval: if preset.is_none() { 3 } else { 0 },
                closed: true,
                metabolism_drain: 0.0,
                movement_drain: 0.0,
                reproduction_energy_threshold: 0.0,
                ..ScriptBotsConfig::default()
            };
            if preset == Some(crate::BrainPreset::Neuro) {
                config.neuroflow.hidden_layers = vec![7, 3];
                config.neuroflow.activation = scriptbots_core::NeuroflowActivationKind::Sigmoid;
            }
            let mut world = WorldState::new(config).unwrap();
            if let Some(preset) = preset {
                let brains = crate::install_brains(&mut world, preset)
                    .unwrap()
                    .population;
                crate::seed_founders(&mut world, &brains, crate::FounderRecipe::Spread).unwrap();
            }
            let host = TestHost::spawn(world);
            let handle = host.handle().with_artifacts_dir(directory.clone()).unwrap();
            assert!(matches!(
                handle.event_canvas(0),
                Err(ControlError::NotFound(_))
            ));
            host.wait_applied(&handle.step().unwrap());
            let checkpoint = handle
                .create_checkpoint(CheckpointCreateRequest::default())
                .unwrap();
            for _ in 0..3 {
                host.wait_applied(&handle.step().unwrap());
            }
            let oracle = host.port.snapshot_hub().latest();
            let target_tick = oracle.world.tick;
            for _ in 0..3 {
                host.wait_applied(&handle.step().unwrap());
            }
            let current = host.port.snapshot_hub().latest();
            let current_digest = host.port.scientific_digest_v1().unwrap();
            let review = handle.event_canvas(target_tick).unwrap();
            let selected = handle
                .replay_canvas_from_checkpoint(
                    handle.session_id(),
                    &lab_checkpoint_descriptor(&checkpoint),
                    target_tick,
                )
                .unwrap();
            assert_eq!(selected.checkpoint_id, checkpoint.checkpoint_id);
            assert_eq!(selected.base_tick, checkpoint.tick);
            assert_eq!(selected.snapshot.world.tick, target_tick);
            assert_eq!(
                selected.verification,
                scriptbots_runtime::lab_workflows::LabReplayVerification::CapturedInterval {
                    verified_through_tick: current_digest.tick.0,
                    captured_current_digest: current_digest.overall.clone(),
                }
            );
            assert_eq!(review.verification, selected.verification);
            assert_eq!(review.checkpoint_id, checkpoint.checkpoint_id);
            assert_eq!(review.base_tick, checkpoint.tick);
            assert_eq!(review.snapshot.world.tick, target_tick);
            assert!(review.snapshot.agent_visuals_complete());
            let logical = |mut value: Value| {
                for agent in value["agents"].as_array_mut().unwrap() {
                    agent.as_object_mut().unwrap().remove("id").unwrap();
                }
                value["summary"]
                    .as_object_mut()
                    .unwrap()
                    .remove("deaths")
                    .unwrap();
                value
            };
            assert_eq!(
                logical(serde_json::to_value(&review.snapshot.world).unwrap()),
                logical(serde_json::to_value(&oracle.world).unwrap())
            );
            assert_eq!(
                logical(serde_json::to_value(&selected.snapshot.world).unwrap()),
                logical(serde_json::to_value(&oracle.world).unwrap())
            );
            assert!(
                digests.insert(current_digest.overall.clone()),
                "registry rows must differ"
            );
            assert_eq!(host.port.scientific_digest_v1().unwrap(), current_digest);
            assert!(Arc::ptr_eq(&host.port.snapshot_hub().latest(), &current));
            assert!(matches!(
                handle.event_canvas(current.world.tick + 1),
                Err(ControlError::BadRequest(_))
            ));
            let (key, original) = handle
                .data_services
                .checkpoint_retention
                .lock()
                .unwrap()
                .requests
                .iter()
                .find(|(_, entry)| entry.metadata.checkpoint_id == checkpoint.checkpoint_id)
                .map(|(key, entry)| (key.clone(), entry.clone()))
                .unwrap();
            let mut saturated_source = original.clone();
            let mut saturated_current = host.port.scientific_digest_cut().unwrap();
            saturated_source.source_config_revision =
                scriptbots_runtime::ConfigRevision::new(u64::MAX);
            saturated_current.config_revision = saturated_source.source_config_revision;
            assert!(
                matches!(ControlHandle::validate_event_review_span(&saturated_source, &saturated_current, target_tick),
                Err(ControlError::BadRequest(message)) if message.contains("configuration revisions")),
                "equal saturated config revisions cannot attest an input-free interval"
            );
            for fault in ["session", "science", "config", "schedule"] {
                let mut changed = original.clone();
                match fault {
                    "session" => {
                        changed.source_session = scriptbots_runtime::HostSessionId::new(
                            original.source_session.get() ^ 1,
                        )
                    }
                    "science" => {
                        changed.source_scientific_revision =
                            scriptbots_runtime::ScientificRevision::new(
                                original.source_scientific_revision.get() + 1,
                            )
                    }
                    "config" => {
                        changed.source_config_revision = scriptbots_runtime::ConfigRevision::new(
                            original.source_config_revision.get() + 1,
                        )
                    }
                    "schedule" => changed.source_scheduled_input_count = 1,
                    _ => unreachable!(),
                }
                handle
                    .data_services
                    .checkpoint_retention
                    .lock()
                    .unwrap()
                    .requests
                    .insert(key.clone(), changed);
                let error = match handle.event_canvas(target_tick) {
                    Ok(_) => panic!("source cut fault {fault} was admitted"),
                    Err(error) => error,
                };
                let expected = match fault {
                    "session" => "owner session",
                    "science" => "scientific revisions",
                    "config" => "configuration revisions",
                    "schedule" => "owner schedule",
                    _ => unreachable!(),
                };
                assert!(error.to_string().contains(expected), "{fault}: {error}");
                assert!(
                    handle
                        .replay_canvas_from_checkpoint(
                            handle.session_id(),
                            &lab_checkpoint_descriptor(&checkpoint),
                            target_tick,
                        )
                        .is_err(),
                    "the selected source must also refuse {fault}"
                );
                assert_eq!(host.port.scientific_digest_v1().unwrap(), current_digest);
                assert!(Arc::ptr_eq(&host.port.snapshot_hub().latest(), &current));
                handle
                    .data_services
                    .checkpoint_retention
                    .lock()
                    .unwrap()
                    .requests
                    .insert(key.clone(), original.clone());
            }
            if preset == Some(crate::BrainPreset::Mlp) {
                // An actual other world carries internally valid bytes and matching counters.
                // Bind its catalog to this owner to isolate the anchor-digest guard.
                let other = TestHost::spawn(
                    WorldState::new(ScriptBotsConfig {
                        rng_seed: Some(137),
                        closed: true,
                        persistence_interval: 0,
                        world_width: 120,
                        world_height: 120,
                        food_cell_size: 20,
                        ..ScriptBotsConfig::default()
                    })
                    .unwrap(),
                );
                let mut wrong = other
                    .handle()
                    .with_artifacts_dir(directory.join("wrong-world"))
                    .unwrap();
                other.wait_applied(&wrong.step().unwrap());
                let wrong_checkpoint = wrong
                    .create_checkpoint(CheckpointCreateRequest::default())
                    .unwrap();
                wrong.host = host.port.clone();
                assert!(
                    matches!(wrong.event_canvas(target_tick), Err(ControlError::InvalidPatch(message))
                    if message.contains("current scientific digest"))
                );
                assert!(matches!(
                    wrong.replay_canvas_from_checkpoint(
                        wrong.session_id(),
                        &lab_checkpoint_descriptor(&wrong_checkpoint),
                        target_tick,
                    ),
                    Err(ControlError::InvalidPatch(message))
                        if message.contains("current scientific digest")
                ));
                host.wait_applied(
                    &handle
                        .submit_control_command(ControlCommand::SpawnAgent {
                            herbivore_tendency: 1.0,
                        })
                        .unwrap(),
                );
                let after_spawn = host.port.scientific_digest_v1().unwrap();
                let after_spawn_publication = handle.read_snapshot().unwrap();
                let replayed = handle.event_canvas(target_tick).unwrap();
                assert_eq!(
                    logical(serde_json::to_value(&replayed.snapshot.world).unwrap()),
                    logical(serde_json::to_value(&oracle.world).unwrap())
                );
                assert_eq!(host.port.scientific_digest_v1().unwrap(), after_spawn);
                assert!(Arc::ptr_eq(
                    &handle.read_snapshot().unwrap(),
                    &after_spawn_publication
                ));
                // Complete recorded config edits remain replayable, including a round trip
                // after a recorded agent command at the same live boundary.
                let food_max = current.config.food_max;
                host.wait_applied(
                    &handle
                        .apply_patch(serde_json::json!({"food_max": food_max * 2.0}))
                        .unwrap(),
                );
                host.wait_applied(
                    &handle
                        .apply_patch(serde_json::json!({"food_max": food_max}))
                        .unwrap(),
                );
                let after_edits_digest = host.port.scientific_digest_v1().unwrap();
                let after_edits_publication = handle.read_snapshot().unwrap();
                let replayed = handle
                    .event_canvas(target_tick)
                    .expect("complete committed config history can be replayed");
                assert_eq!(
                    logical(serde_json::to_value(&replayed.snapshot.world).unwrap()),
                    logical(serde_json::to_value(&oracle.world).unwrap())
                );
                assert_eq!(
                    host.port.scientific_digest_v1().unwrap(),
                    after_edits_digest
                );
                assert!(Arc::ptr_eq(
                    &handle.read_snapshot().unwrap(),
                    &after_edits_publication
                ));
                let fresh = handle
                    .create_checkpoint(CheckpointCreateRequest::default())
                    .unwrap();
                assert!(fresh.tick > checkpoint.tick);
                host.wait_applied(&handle.step().unwrap());
                let fresh_target = host.port.snapshot_hub().latest().world.tick;
                host.wait_applied(&handle.step().unwrap());
                let review = handle.event_canvas(fresh_target).unwrap();
                assert_eq!(review.base_tick, fresh.tick);
                assert_eq!(review.checkpoint_id, fresh.checkpoint_id);
                assert_eq!(review.snapshot.world.tick, fresh_target);

                // Oversized inputs still apply but are omitted from bounded replay
                // history. The resulting revision gap must refuse historical review.
                let omitted = handle
                    .submit_control_command(ControlCommand::Intervention(Box::new(
                        scriptbots_core::interventions::InterventionCommand::new(
                            scriptbots_core::interventions::Intervention::Embargo {
                                region: scriptbots_core::Region::All,
                                ticks: 2,
                            },
                            scriptbots_core::interventions::InterventionSurface::Script,
                            "x".repeat(
                                scriptbots_runtime::MAX_SCIENTIFIC_COMMAND_HISTORY_BYTES + 1,
                            ),
                        ),
                    )))
                    .unwrap();
                host.wait_applied(&omitted);
                host.wait_applied(&handle.step().unwrap());
                let after_omitted = host.port.scientific_digest_v1().unwrap();
                let after_omitted_publication = handle.read_snapshot().unwrap();
                assert!(
                    after_omitted_publication
                        .applied_interventions
                        .iter()
                        .any(|record| record.kind == "embargo" && !record.expired)
                );
                let refusal = handle
                    .event_canvas(fresh_target)
                    .err()
                    .expect("an applied input omitted from history must refuse event review");
                assert!(
                    matches!(&refusal, ControlError::BadRequest(message)
                        if message.contains("scientific revisions")),
                    "{refusal}"
                );
                assert!(matches!(
                    handle.replay_canvas_from_checkpoint(
                        handle.session_id(),
                        &lab_checkpoint_descriptor(&fresh),
                        fresh_target,
                    ),
                    Err(ControlError::BadRequest(message))
                        if message.contains("scientific revisions")
                ));
                assert_eq!(host.port.scientific_digest_v1().unwrap(), after_omitted);
                assert!(Arc::ptr_eq(
                    &handle.read_snapshot().unwrap(),
                    &after_omitted_publication
                ));
            }
            fs::write(
                directory.join("target-oracle.json"),
                serde_json::to_vec_pretty(&oracle.world).unwrap(),
            )
            .unwrap();
            fs::write(
                directory.join("target-review.json"),
                serde_json::to_vec_pretty(&review.snapshot.world).unwrap(),
            )
            .unwrap();
            let (artifact, bytes) = handle
                .read_artifact_bytes(&checkpoint.checkpoint_id)
                .unwrap();
            fs::write(directory.join("before-corruption.bin"), bytes).unwrap();
            fs::OpenOptions::new()
                .append(true)
                .open(directory.join(artifact.relative_path))
                .unwrap()
                .write_all(&[0xff])
                .unwrap();
            assert!(matches!(
                handle.event_canvas(target_tick),
                Err(ControlError::InvalidPatch(_))
            ));
            assert!(matches!(
                handle.replay_canvas_from_checkpoint(
                    handle.session_id(),
                    &lab_checkpoint_descriptor(&checkpoint),
                    target_tick,
                ),
                Err(ControlError::InvalidPatch(_))
            ));
            eprintln!(
                "event canvas preset {preset:?}: base={} target={target_tick} captured_current={} captured_science={}; distinct state, matching projection, source guards and corrupt bytes checked",
                checkpoint.tick, current.world.tick, current_digest.overall
            );
        }
        assert_eq!(digests.len(), 5);
        // Both work and schedule refusals use real owners rather than invented ticks.
        for scheduled in [false, true] {
            let directory = tempfile::Builder::new()
                .prefix("event-canvas-bounds-")
                .tempdir()
                .unwrap()
                .keep();
            let world = WorldState::new(ScriptBotsConfig {
                rng_seed: Some(42),
                closed: true,
                persistence_interval: 0,
                world_width: 120,
                world_height: 120,
                food_cell_size: 20,
                ..ScriptBotsConfig::default()
            })
            .unwrap();
            let host = TestHost::spawn_with_schedule(
                world,
                scheduled.then(|| {
                    vec![
                        scriptbots_runtime::ScheduledConfigPatch::new(
                            1,
                            Tick(10),
                            serde_json::json!({"food_max": 0.61}),
                        )
                        .unwrap(),
                    ]
                }),
            );
            let handle = host.handle().with_artifacts_dir(directory.clone()).unwrap();
            let checkpoint = handle
                .create_checkpoint(CheckpointCreateRequest::default())
                .unwrap();
            let selected = lab_checkpoint_descriptor(&checkpoint);
            let ticks = if scheduled {
                2
            } else {
                MAX_EVENT_REVIEW_TICKS + 1
            };
            for completed in 1..=ticks {
                host.wait_applied(&handle.step().unwrap());
                if !scheduled && completed == MAX_EVENT_REVIEW_TICKS {
                    let endpoint = host.port.scientific_digest_v1().unwrap();
                    let boundary = handle
                        .replay_canvas_from_checkpoint(handle.session_id(), &selected, 1)
                        .expect("the full supported captured interval remains reviewable");
                    assert_eq!(boundary.snapshot.world.tick, 1);
                    assert_eq!(
                        boundary.verification,
                        scriptbots_runtime::lab_workflows::LabReplayVerification::CapturedInterval {
                            verified_through_tick: completed,
                            captured_current_digest: endpoint.overall.clone(),
                        }
                    );
                    assert_eq!(host.port.scientific_digest_v1().unwrap(), endpoint);
                    let (_, remote) = handle
                        .replay_canvas_from_checkpoint_bounded(
                            handle.session_id(),
                            &checkpoint.checkpoint_id,
                            1,
                        )
                        .expect("remote review accepts the unchanged complete interval bound");
                    assert_eq!(remote.verification, boundary.verification);
                    assert_eq!(remote.snapshot.world.tick, 1);
                }
            }
            let before = host.port.scientific_digest_v1().unwrap();
            assert_eq!(before.tick.0, ticks, "observe the completed owner interval");
            let error = match handle.event_canvas(1) {
                Ok(_) => panic!("bound admitted"),
                Err(error) => error,
            };
            assert!(error.to_string().contains(if scheduled {
                "owner schedule"
            } else {
                "at most"
            }));
            let selected_error = handle
                .replay_canvas_from_checkpoint(handle.session_id(), &selected, 1)
                .err()
                .expect("selected review must enforce the same interval and schedule bounds");
            assert!(selected_error.to_string().contains(if scheduled {
                "owner schedule"
            } else {
                "at most"
            }));
            let remote_error = handle
                .replay_canvas_from_checkpoint_bounded(
                    handle.session_id(),
                    &checkpoint.checkpoint_id,
                    1,
                )
                .err()
                .expect("remote target does not shorten the full captured verification interval");
            assert!(remote_error.to_string().contains(if scheduled {
                "owner schedule"
            } else {
                "at most"
            }));
            let saved = handle
                .replay_canvas_from_checkpoint(handle.session_id(), &selected, checkpoint.tick)
                .expect("a verified exact saved cut does not require interval replay");
            assert_eq!(saved.snapshot.world.tick, checkpoint.tick);
            assert_eq!(
                saved.verification,
                scriptbots_runtime::lab_workflows::LabReplayVerification::SavedCut
            );
            let (_, remote_saved) = handle
                .replay_canvas_from_checkpoint_bounded(
                    handle.session_id(),
                    &checkpoint.checkpoint_id,
                    checkpoint.tick,
                )
                .unwrap();
            assert_eq!(remote_saved.verification, saved.verification);
            assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
            if scheduled {
                let mut retention = handle.data_services.checkpoint_retention.lock().unwrap();
                let entry = retention.requests.values_mut().next().unwrap();
                assert_eq!(
                    entry.source_scheduled_input_count, 1,
                    "writer retains the declared input count from the real schedule"
                );
                entry.source_scheduled_input_count = 0;
                drop(retention);
                assert!(
                    matches!(handle.event_canvas(1), Err(ControlError::BadRequest(message)) if message.contains("owner schedule"))
                );
            }
            eprintln!(
                "event canvas bound: schedule={scheduled} actual_ticks={} refused; {}",
                before.tick.0,
                directory.display()
            );
        }
    }

    #[test]
    fn checkpoint_retry_recovers_the_original_cut_after_owner_restart_at_full_capacity() {
        let directory = tempfile::tempdir().unwrap().keep();
        let (handle, host) = checkpoint_test_handle(&directory, 4, 8 * 1024 * 1024);
        host.wait_applied(&handle.step().unwrap());
        let request = CheckpointCreateRequest {
            description: Some("original scientific cut".into()),
            idempotency_key: Some("restart-safe-create".into()),
        };
        let saved = handle.create_checkpoint(request.clone()).unwrap();
        let original = handle.read_artifact_bytes(&saved.checkpoint_id).unwrap().1;
        let admission_path =
            checkpoint_request_path(&directory, request.idempotency_key.as_deref().unwrap());
        let admission_bytes = fs::read(&admission_path).unwrap();
        let admission: CheckpointRequestRecord = serde_json::from_slice(&admission_bytes).unwrap();
        assert_eq!(
            admission.admission.source_session,
            scriptbots_runtime::HostSessionId::new(0xc017)
        );
        host.wait_applied(&handle.step().unwrap());
        assert_eq!(handle.create_checkpoint(request.clone()).unwrap(), saved);
        drop(handle);
        drop(host);

        let owner = TestHost::spawn_with_identity(
            WorldState::new(ScriptBotsConfig {
                world_width: 40,
                world_height: 40,
                food_cell_size: 10,
                rng_seed: Some(57),
                persistence_interval: 0,
                population_minimum: 0,
                population_spawn_interval: 0,
                ..ScriptBotsConfig::default()
            })
            .unwrap(),
            None,
            scriptbots_runtime::HostSessionId::new(0xc018),
        );
        let mut restarted = owner.handle();
        restarted.data_services = Arc::new(DataServices::new(directory.clone()));
        let restarted = restarted
            .with_checkpoint_retention_limits(scriptbots_storage::CheckpointRetentionLimits {
                max_count: 1,
                max_bytes: saved.byte_size,
            })
            .unwrap();
        let mut pipeline = scriptbots_storage::StoragePipeline::unattributed_memory().unwrap();
        let writer = pipeline.checkpoint_writer();
        writer
            .configure_retention(scriptbots_storage::CheckpointRetentionLimits {
                max_count: 0,
                max_bytes: 1024 * 1024,
            })
            .unwrap();
        let restarted = restarted.with_checkpoint_writer(Some(writer));
        let before = owner.port.scientific_digest_v1().unwrap();
        let publication = restarted.read_snapshot().unwrap();
        assert_eq!(before.tick.0, 0);
        let retries = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|_| {
                    let request = request.clone();
                    let restarted = &restarted;
                    scope.spawn(move || restarted.create_checkpoint(request).unwrap())
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        for retry in retries {
            assert_eq!(retry, saved);
        }
        assert_eq!(
            restarted.get_checkpoint(&saved.checkpoint_id).unwrap(),
            saved
        );
        assert_eq!(
            restarted
                .read_artifact_bytes(&saved.checkpoint_id)
                .unwrap()
                .1,
            original
        );
        let usage = restarted.checkpoint_retention_usage().unwrap();
        assert_eq!(usage.file_count, 1);
        assert_eq!(usage.file_bytes, saved.byte_size);
        assert_eq!(usage.request_entries, 0);
        assert_eq!(usage.pending_entries, 0);
        assert_eq!(usage.retained_request_records, 1);
        assert_eq!(usage.reserved_checkpoint_count, 1);
        assert_eq!(
            usage.retained_request_record_bytes,
            admission_bytes.len() as u64
        );
        assert!(usage.retained_request_record_bytes <= usage.max_retained_request_record_bytes);
        assert_eq!(usage.sql_count, Some(0));
        let changed = CheckpointCreateRequest {
            description: Some("different request contents".into()),
            ..request
        };
        assert!(matches!(
            restarted.create_checkpoint(changed),
            Err(ControlError::Conflict(_))
        ));
        assert!(matches!(
            restarted.create_checkpoint(CheckpointCreateRequest::default()),
            Err(ControlError::CheckpointCapacity(_))
        ));
        assert_eq!(owner.port.scientific_digest_v1().unwrap(), before);
        assert!(Arc::ptr_eq(
            &restarted.read_snapshot().unwrap(),
            &publication
        ));
        assert_eq!(fs::read(&admission_path).unwrap(), admission_bytes);
        pipeline.shutdown().unwrap();
    }

    #[test]
    fn checkpoint_retry_unfinished_admissions_never_recapture_after_reopen() {
        let source = tempfile::tempdir().unwrap().keep();
        let (handle, host) = checkpoint_test_handle(&source, 4, 8 * 1024 * 1024);
        host.wait_applied(&handle.step().unwrap());
        let request = CheckpointCreateRequest {
            description: Some("reserved original cut".into()),
            idempotency_key: Some("interrupted-create".into()),
        };
        let saved = handle.create_checkpoint(request.clone()).unwrap();
        let admission_name =
            checkpoint_request_path(&source, request.idempotency_key.as_deref().unwrap())
                .file_name()
                .unwrap()
                .to_owned();
        let admission_bytes = fs::read(source.join(&admission_name)).unwrap();
        let admission: CheckpointRequestRecord = serde_json::from_slice(&admission_bytes).unwrap();
        let original_session = format!(
            "original source session {}",
            admission.admission.source_session.get()
        );
        let payload_name = format!("{}.bin", saved.checkpoint_id);
        let descriptor_name = format!("{}{CHECKPOINT_CATALOG_SUFFIX}", saved.checkpoint_id);
        drop(handle);
        drop(host);

        for prefix in ["admission", "payload", "descriptor"] {
            let directory = tempfile::tempdir().unwrap().keep();
            fs::write(directory.join(&admission_name), &admission_bytes).unwrap();
            if prefix != "admission" {
                fs::copy(source.join(&payload_name), directory.join(&payload_name)).unwrap();
            }
            if prefix == "descriptor" {
                fs::copy(
                    source.join(&descriptor_name),
                    directory.join(&descriptor_name),
                )
                .unwrap();
            }
            let (restarted, owner) = checkpoint_test_handle(&directory, 1, 8 * 1024 * 1024);
            let entries = fs::read_dir(&directory).unwrap().count();
            let before = owner.port.scientific_digest_v1().unwrap();
            for _ in 0..2 {
                let error = restarted.create_checkpoint(request.clone()).unwrap_err();
                assert!(
                    matches!(error, ControlError::CheckpointRetentionUnavailable(ref detail)
                        if detail.contains("completion is unconfirmed")
                            && detail.contains(&original_session)),
                    "{prefix}: {error}"
                );
            }
            assert!(matches!(
                restarted.create_checkpoint(CheckpointCreateRequest::default()),
                Err(ControlError::CheckpointCapacity(_))
            ));
            let usage = restarted.checkpoint_retention_usage().unwrap();
            assert_eq!(usage.retained_request_records, 1);
            assert_eq!(usage.request_entries, 0);
            assert_eq!(restarted.list_checkpoints(None, None).unwrap().total, 0);
            assert_eq!(fs::read_dir(&directory).unwrap().count(), entries);
            assert_eq!(
                fs::read(directory.join(&admission_name)).unwrap(),
                admission_bytes
            );
            assert_eq!(owner.port.scientific_digest_v1().unwrap(), before);
        }
    }

    #[test]
    fn checkpoint_retry_counts_mixed_anonymous_and_unfinished_reservations_after_restart() {
        let source = tempfile::tempdir().unwrap().keep();
        let (handle, host) = checkpoint_test_handle(&source, 2, 8 * 1024 * 1024);
        let request = CheckpointCreateRequest {
            description: Some("reserved keyed cut".into()),
            idempotency_key: Some("mixed-pending-key".into()),
        };
        handle.create_checkpoint(request.clone()).unwrap();
        let first_usage = handle.checkpoint_retention_usage().unwrap();
        assert_eq!(first_usage.file_count, 1);
        assert_eq!(first_usage.retained_request_records, 1);
        assert_eq!(first_usage.request_entries, 1);
        assert_eq!(first_usage.reserved_checkpoint_count, 1);
        host.wait_applied(&handle.step().unwrap());
        let anonymous = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .expect("one completed keyed checkpoint consumes only one slot");
        assert_eq!(
            handle
                .checkpoint_retention_usage()
                .unwrap()
                .reserved_checkpoint_count,
            2
        );
        assert!(matches!(
            handle.create_checkpoint(CheckpointCreateRequest::default()),
            Err(ControlError::CheckpointCapacity(_))
        ));
        let admission_name =
            checkpoint_request_path(&source, request.idempotency_key.as_deref().unwrap())
                .file_name()
                .unwrap()
                .to_owned();
        let admission_bytes = fs::read(source.join(&admission_name)).unwrap();
        drop(handle);
        drop(host);

        for intact in [true, false] {
            let directory = tempfile::tempdir().unwrap().keep();
            let retained_admission = if intact {
                admission_bytes.as_slice()
            } else {
                &admission_bytes[..admission_bytes.len() / 2]
            };
            fs::write(directory.join(&admission_name), retained_admission).unwrap();
            for suffix in [
                ".bin",
                CHECKPOINT_CATALOG_SUFFIX,
                CHECKPOINT_CATALOG_READY_SUFFIX,
            ] {
                let name = format!("{}{suffix}", anonymous.checkpoint_id);
                fs::copy(source.join(&name), directory.join(&name)).unwrap();
            }
            let (restarted, owner) = checkpoint_test_handle(&directory, 2, 8 * 1024 * 1024);
            let before = owner.port.scientific_digest_v1().unwrap();
            let publication = restarted.read_snapshot().unwrap();
            let entries = fs::read_dir(&directory).unwrap().count();
            let usage = restarted.checkpoint_retention_usage().unwrap();
            assert_eq!(usage.file_count, 1);
            assert_eq!(usage.retained_request_records, 1);
            assert_eq!(usage.request_entries, 0);
            assert_eq!(usage.reserved_checkpoint_count, 2);
            assert_eq!(
                restarted.get_checkpoint(&anonymous.checkpoint_id).unwrap(),
                anonymous
            );
            for new_key in [None, Some("different-new-key".to_owned())] {
                assert!(matches!(
                    restarted.create_checkpoint(CheckpointCreateRequest {
                        description: None,
                        idempotency_key: new_key,
                    }),
                    Err(ControlError::CheckpointCapacity(_))
                ));
            }
            assert!(matches!(
                restarted.create_checkpoint(request.clone()),
                Err(ControlError::CheckpointRetentionUnavailable(_))
            ));
            assert_eq!(fs::read_dir(&directory).unwrap().count(), entries);
            assert_eq!(
                fs::read(directory.join(&admission_name)).unwrap(),
                retained_admission
            );
            assert_eq!(owner.port.scientific_digest_v1().unwrap(), before);
            assert!(Arc::ptr_eq(
                &restarted.read_snapshot().unwrap(),
                &publication
            ));
        }
    }

    #[test]
    fn checkpoint_retry_rejects_corrupt_and_rehashed_admissions_without_new_captures() {
        let source = tempfile::tempdir().unwrap().keep();
        let (handle, host) = checkpoint_test_handle(&source, 4, 8 * 1024 * 1024);
        let request = CheckpointCreateRequest {
            description: Some("original request".into()),
            idempotency_key: Some("corrupt-admission".into()),
        };
        let saved = handle.create_checkpoint(request.clone()).unwrap();
        let admission_name =
            checkpoint_request_path(&source, request.idempotency_key.as_deref().unwrap())
                .file_name()
                .unwrap()
                .to_owned();
        let admission_bytes = fs::read(source.join(&admission_name)).unwrap();
        drop(handle);
        drop(host);

        for corruption in [
            "truncated",
            "checksum",
            "schema",
            "key",
            "metadata",
            "payload",
        ] {
            let directory = tempfile::tempdir().unwrap().keep();
            for suffix in [
                ".bin",
                CHECKPOINT_CATALOG_SUFFIX,
                CHECKPOINT_CATALOG_READY_SUFFIX,
            ] {
                let name = format!("{}{suffix}", saved.checkpoint_id);
                fs::copy(source.join(&name), directory.join(&name)).unwrap();
            }
            let mut record: CheckpointRequestRecord =
                serde_json::from_slice(&admission_bytes).unwrap();
            match corruption {
                "checksum" | "metadata" => {
                    record.admission.metadata.description = Some("changed contents".into());
                }
                "schema" => record.admission.schema_version += 1,
                "key" => record.admission.idempotency_key = "different-original-key".into(),
                "truncated" | "payload" => {}
                _ => unreachable!(),
            }
            if corruption != "checksum" {
                record.checksum_blake3 =
                    compute_blake3(&serde_json::to_vec(&record.admission).unwrap());
            }
            let path = directory.join(&admission_name);
            if corruption == "truncated" {
                fs::write(&path, &admission_bytes[..admission_bytes.len() / 2]).unwrap();
            } else {
                fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
            }
            if corruption == "payload" {
                fs::OpenOptions::new()
                    .append(true)
                    .open(directory.join(format!("{}.bin", saved.checkpoint_id)))
                    .unwrap()
                    .write_all(&[0xff])
                    .unwrap();
            }
            let changed = fs::read(&path).unwrap();
            let (restarted, owner) = checkpoint_test_handle(&directory, 4, 8 * 1024 * 1024);
            let entries = fs::read_dir(&directory).unwrap().count();
            let before = owner.port.scientific_digest_v1().unwrap();
            let error = restarted.create_checkpoint(request.clone()).unwrap_err();
            assert!(
                matches!(error, ControlError::CheckpointRetentionUnavailable(_)),
                "{corruption}: {error}"
            );
            assert_eq!(
                restarted
                    .checkpoint_retention_usage()
                    .unwrap()
                    .request_entries,
                0
            );
            assert_eq!(fs::read_dir(&directory).unwrap().count(), entries);
            assert_eq!(fs::read(&path).unwrap(), changed);
            assert_eq!(owner.port.scientific_digest_v1().unwrap(), before);
        }
    }

    #[test]
    fn checkpoint_retry_refuses_duplicate_roots_even_with_a_cached_receipt() {
        let first = tempfile::tempdir().unwrap().keep();
        let second = tempfile::tempdir().unwrap().keep();
        let (handle, host) = checkpoint_test_handle(&first, 4, 8 * 1024 * 1024);
        let request = CheckpointCreateRequest {
            description: Some("one original request".into()),
            idempotency_key: Some("ambiguous-root".into()),
        };
        let saved = handle.create_checkpoint(request.clone()).unwrap();
        let first_path =
            checkpoint_request_path(&first, request.idempotency_key.as_deref().unwrap());
        let second_path =
            checkpoint_request_path(&second, request.idempotency_key.as_deref().unwrap());
        let original = fs::read(&first_path).unwrap();
        fs::write(&second_path, &original).unwrap();
        let handle = handle.with_artifacts_dir(second.clone()).unwrap();
        let before = host.port.scientific_digest_v1().unwrap();
        assert!(matches!(
            handle.create_checkpoint(request.clone()),
            Err(ControlError::Conflict(_))
        ));
        let usage = handle.checkpoint_retention_usage().unwrap();
        assert_eq!(usage.retained_request_records, 2);
        assert_eq!(usage.file_count, 1);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
        drop(handle);
        drop(host);

        let (restarted, owner) = checkpoint_test_handle(&first, 4, 8 * 1024 * 1024);
        let restarted = restarted.with_artifacts_dir(second).unwrap();
        let before = owner.port.scientific_digest_v1().unwrap();
        assert!(matches!(
            restarted.create_checkpoint(request),
            Err(ControlError::Conflict(_))
        ));
        assert_eq!(
            restarted
                .checkpoint_retention_usage()
                .unwrap()
                .request_entries,
            0
        );
        assert_eq!(owner.port.scientific_digest_v1().unwrap(), before);
        assert_eq!(fs::read(&first_path).unwrap(), original);
        assert_eq!(fs::read(&second_path).unwrap(), original);
        assert!(first.join(format!("{}.bin", saved.checkpoint_id)).is_file());
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_retry_refuses_aliased_admissions_without_using_the_cached_receipt() {
        for alias in ["symlink", "hardlink"] {
            let directory = tempfile::tempdir().unwrap().keep();
            let (handle, host) = checkpoint_test_handle(&directory, 4, 8 * 1024 * 1024);
            let request = CheckpointCreateRequest {
                description: None,
                idempotency_key: Some("aliased-admission".into()),
            };
            handle.create_checkpoint(request.clone()).unwrap();
            let path =
                checkpoint_request_path(&directory, request.idempotency_key.as_deref().unwrap());
            let original = fs::read(&path).unwrap();
            let preserved = directory.join("preserved-admission");
            fs::rename(&path, &preserved).unwrap();
            if alias == "symlink" {
                std::os::unix::fs::symlink(&preserved, &path).unwrap();
            } else {
                fs::hard_link(&preserved, &path).unwrap();
            }
            let before = host.port.scientific_digest_v1().unwrap();
            assert!(
                matches!(
                    handle.create_checkpoint(request),
                    Err(ControlError::CheckpointRetentionUnavailable(_))
                ),
                "{alias}"
            );
            assert_eq!(fs::read(&preserved).unwrap(), original);
            assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
        }
    }

    #[test]
    fn checkpoint_catalog_recovers_download_review_and_pagination_without_adopting_requests() {
        let directory = tempfile::tempdir().unwrap().keep();
        let (handle, host) = checkpoint_test_handle(&directory, 8, 8 * 1024 * 1024);
        host.wait_applied(&handle.step().unwrap());
        let first = handle
            .create_checkpoint(CheckpointCreateRequest {
                description: Some("saved before the second intervention".into()),
                idempotency_key: Some("original-request".into()),
            })
            .unwrap();
        let first_bytes = handle.read_artifact_bytes(&first.checkpoint_id).unwrap().1;
        host.wait_applied(&handle.step().unwrap());
        let second = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        let second_bytes = handle.read_artifact_bytes(&second.checkpoint_id).unwrap().1;
        let uncatalogued_id = format!("ckpt-{:032x}-t{}", 0, first.tick);
        fs::write(
            directory.join(format!("{uncatalogued_id}.bin")),
            &first_bytes,
        )
        .unwrap();
        fs::write(
            directory.join(format!("{uncatalogued_id}{CHECKPOINT_CATALOG_SUFFIX}")),
            b"interrupted descriptor without a completion marker",
        )
        .unwrap();
        let descriptors = [first.clone(), second.clone()].map(|metadata| {
            let path = directory.join(format!(
                "{}{CHECKPOINT_CATALOG_SUFFIX}",
                metadata.checkpoint_id
            ));
            let bytes = fs::read(&path).unwrap();
            (path, bytes)
        });
        drop(handle);
        drop(host);

        let (restarted, owner) = checkpoint_test_handle(&directory, 8, 8 * 1024 * 1024);
        let before = owner.port.scientific_digest_v1().unwrap();
        assert_eq!(before.tick.0, 0);
        // Cold download must work even when no catalog/list method was called.
        assert_eq!(
            restarted
                .read_artifact_bytes(&first.checkpoint_id)
                .unwrap()
                .1,
            first_bytes
        );
        let page = restarted.list_checkpoints(Some(1), None).unwrap();
        assert_eq!(page.total, 2);
        assert_eq!(page.items.len(), 1);
        assert!(page.has_more);
        let next = restarted
            .list_checkpoints(Some(1), page.next_cursor.as_deref())
            .unwrap();
        assert_eq!(next.total, 2);
        assert_eq!(next.items.len(), 1);
        assert!(!next.has_more);
        assert_ne!(page.items[0].checkpoint_id, next.items[0].checkpoint_id);
        for (expected, bytes) in [(&first, &first_bytes), (&second, &second_bytes)] {
            assert_eq!(
                serde_json::to_value(restarted.get_checkpoint(&expected.checkpoint_id).unwrap())
                    .unwrap(),
                serde_json::to_value(expected).unwrap(),
            );
            assert_eq!(
                &restarted
                    .read_artifact_bytes(&expected.checkpoint_id)
                    .unwrap()
                    .1,
                bytes
            );
            let snapshot = restarted
                .checkpoint_canvas(&expected.checkpoint_id)
                .unwrap();
            assert_eq!(snapshot.world.tick, expected.tick);
            assert_eq!(snapshot.world.agents.len(), expected.agent_count);
            let selected = lab_checkpoint_descriptor(expected);
            let review = restarted
                .replay_canvas_from_checkpoint(restarted.session_id(), &selected, expected.tick)
                .unwrap();
            assert_eq!(review.checkpoint_id, expected.checkpoint_id);
            assert_eq!(review.snapshot.world.tick, expected.tick);
            assert_eq!(review.snapshot.world.agents.len(), expected.agent_count);
            assert_eq!(
                review.verification,
                scriptbots_runtime::lab_workflows::LabReplayVerification::SavedCut
            );
            assert!(matches!(
                restarted.replay_canvas_from_checkpoint(
                    restarted.session_id(),
                    &selected,
                    expected.tick + 1,
                ),
                Err(ControlError::BadRequest(message))
                    if message.contains("no acknowledged current-owner replay history")
            ));
        }
        assert_eq!(restarted.list_artifacts(None, None).unwrap().total, 2);
        let usage = restarted.checkpoint_retention_usage().unwrap();
        assert_eq!(
            usage.file_count, 3,
            "uncatalogued files still consume quota"
        );
        assert_eq!(usage.request_entries, 0);
        assert_eq!(usage.pending_entries, 0);
        assert!(matches!(
            restarted.get_checkpoint(&uncatalogued_id),
            Err(ControlError::NotFound(_))
        ));
        assert!(matches!(
            restarted.event_canvas(first.tick),
            Err(ControlError::NotFound(_))
        ));
        assert_eq!(owner.port.scientific_digest_v1().unwrap(), before);
        for (path, bytes) in descriptors {
            assert_eq!(
                fs::read(path).unwrap(),
                bytes,
                "discovery must not rewrite evidence"
            );
        }
        assert_eq!(
            fs::read(directory.join(format!("{uncatalogued_id}.bin"))).unwrap(),
            first_bytes
        );
    }

    #[test]
    fn checkpoint_catalog_rejects_corrupt_and_rehashed_false_descriptors_without_mutating_owner() {
        for corruption in [
            "checksum",
            "population",
            "schema",
            "payload_bound",
            "record_bound",
        ] {
            let directory = tempfile::tempdir().unwrap().keep();
            let (handle, host) = checkpoint_test_handle(&directory, 4, 8 * 1024 * 1024);
            let metadata = handle
                .create_checkpoint(CheckpointCreateRequest::default())
                .unwrap();
            let path = directory.join(format!(
                "{}{CHECKPOINT_CATALOG_SUFFIX}",
                metadata.checkpoint_id
            ));
            let original = fs::read(&path).unwrap();
            fs::write(directory.join("descriptor.original"), &original).unwrap();
            let payload =
                fs::read(directory.join(format!("{}.bin", metadata.checkpoint_id))).unwrap();
            let mut record: CheckpointCatalogRecord = serde_json::from_slice(&original).unwrap();
            match corruption {
                "checksum" => {
                    record.metadata.description = Some("changed without rehashing".into())
                }
                "population" => record.metadata.agent_count += 1,
                "schema" => record.schema_version += 1,
                "payload_bound" => record.metadata.byte_size = 1,
                "record_bound" => {}
                _ => unreachable!(),
            }
            if corruption != "checksum" {
                record.checksum_blake3 =
                    compute_blake3(&serde_json::to_vec(&record.metadata).unwrap());
            }
            if corruption == "record_bound" {
                fs::write(&path, vec![b' '; MAX_CHECKPOINT_CATALOG_RECORD_BYTES + 1]).unwrap();
            } else {
                fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
            }
            let changed = fs::read(&path).unwrap();
            drop(handle);
            drop(host);
            let (restarted, owner) = checkpoint_test_handle(&directory, 4, 8 * 1024 * 1024);
            let before = owner.port.scientific_digest_v1().unwrap();
            let error = restarted
                .get_checkpoint(&metadata.checkpoint_id)
                .unwrap_err();
            assert!(
                matches!(error, ControlError::CheckpointRetentionUnavailable(_)),
                "{corruption}: {error}"
            );
            if corruption.ends_with("_bound") {
                assert!(
                    error.to_string().contains("byte bound"),
                    "{corruption}: {error}"
                );
            }
            assert!(restarted.list_checkpoints(None, None).is_err());
            assert!(
                restarted
                    .read_artifact_bytes(&metadata.checkpoint_id)
                    .is_err()
            );
            assert!(
                restarted
                    .data_services
                    .checkpoints
                    .lock()
                    .unwrap()
                    .is_empty()
            );
            assert!(restarted.data_services.artifacts.lock().unwrap().is_empty());
            assert_eq!(owner.port.scientific_digest_v1().unwrap(), before);
            assert_eq!(fs::read(&path).unwrap(), changed);
            assert_eq!(
                fs::read(directory.join(format!("{}.bin", metadata.checkpoint_id))).unwrap(),
                payload
            );
        }
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one retained save proves warm-cache refusal, exact repair and cold retry without recapture"
    )]
    fn checkpoint_cached_receipts_require_the_original_completed_catalog() {
        for corruption in [
            "missing_ready",
            "nonempty_ready",
            "missing_descriptor",
            "corrupt_descriptor",
            "changed_descriptor",
        ] {
            let directory = tempfile::tempdir().unwrap().keep();
            let (handle, host) = checkpoint_test_handle(&directory, 4, 8 * 1024 * 1024);
            let request = CheckpointCreateRequest {
                description: Some("original saved science".into()),
                idempotency_key: Some("lost-response-exact-retry".into()),
            };
            let saved = handle.create_checkpoint(request.clone()).unwrap();
            let (artifact, payload) = handle.read_artifact_bytes(&saved.checkpoint_id).unwrap();
            assert_eq!(handle.get_checkpoint(&saved.checkpoint_id).unwrap(), saved);
            assert_eq!(handle.get_artifact(&saved.checkpoint_id).unwrap(), artifact);
            assert_eq!(handle.create_checkpoint(request.clone()).unwrap(), saved);

            // A retry after a lost response must never substitute a fresh cut.
            host.wait_applied(&handle.step().unwrap());
            let live = host.port.scientific_digest_v1().unwrap();
            assert!(handle.status().unwrap().tick > saved.tick);
            let suffix = if corruption.ends_with("ready") {
                CHECKPOINT_CATALOG_READY_SUFFIX
            } else {
                CHECKPOINT_CATALOG_SUFFIX
            };
            let path = directory.join(format!("{}{suffix}", saved.checkpoint_id));
            let original = fs::read(&path).unwrap();
            fs::write(directory.join("original-completion-record"), &original).unwrap();
            match corruption {
                "missing_ready" | "missing_descriptor" => {
                    fs::rename(&path, directory.join("missing-completion-record")).unwrap();
                }
                "nonempty_ready" | "corrupt_descriptor" => {
                    fs::write(&path, b"incomplete completion record").unwrap();
                }
                "changed_descriptor" => {
                    let mut record: CheckpointCatalogRecord =
                        serde_json::from_slice(&original).unwrap();
                    record.metadata.description = Some("different retained request".into());
                    record.checksum_blake3 =
                        compute_blake3(&serde_json::to_vec(&record.metadata).unwrap());
                    fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
                }
                _ => unreachable!(),
            }
            let entries = fs::read_dir(&directory).unwrap().count();
            for _ in 0..2 {
                assert!(
                    handle.create_checkpoint(request.clone()).is_err(),
                    "{corruption}: a cached acknowledgement is not a recoverable saved cut"
                );
                assert!(
                    handle.get_checkpoint(&saved.checkpoint_id).is_err(),
                    "{corruption}"
                );
                assert!(
                    handle.get_artifact(&saved.checkpoint_id).is_err(),
                    "{corruption}"
                );
                assert!(
                    handle.read_artifact_bytes(&saved.checkpoint_id).is_err(),
                    "{corruption}"
                );
                assert!(handle.list_checkpoints(None, None).is_err(), "{corruption}");
                assert!(handle.list_artifacts(None, None).is_err(), "{corruption}");
            }
            assert_eq!(fs::read_dir(&directory).unwrap().count(), entries);
            assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
            assert_eq!(
                fs::read(directory.join(&artifact.relative_path)).unwrap(),
                payload
            );
            let usage = handle.checkpoint_retention_usage().unwrap();
            assert_eq!(usage.file_count, 1);
            assert_eq!(usage.request_entries, 1);
            assert_eq!(usage.pending_entries, 0);

            // Preserve the failed record and repair only the exact original
            // catalog bytes. The request and immutable payload remain unchanged.
            if path.exists() {
                fs::rename(&path, directory.join("refused-completion-record")).unwrap();
            }
            fs::write(&path, &original).unwrap();
            assert_eq!(handle.create_checkpoint(request.clone()).unwrap(), saved);
            assert_eq!(handle.get_checkpoint(&saved.checkpoint_id).unwrap(), saved);
            assert_eq!(handle.get_artifact(&saved.checkpoint_id).unwrap(), artifact);
            assert_eq!(handle.list_checkpoints(None, None).unwrap().total, 1);
            assert_eq!(handle.list_artifacts(None, None).unwrap().total, 1);
            assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
            drop(handle);
            drop(host);

            let (restarted, owner) = checkpoint_test_handle(&directory, 4, 8 * 1024 * 1024);
            let before = owner.port.scientific_digest_v1().unwrap();
            assert_eq!(restarted.create_checkpoint(request).unwrap(), saved);
            assert_eq!(
                restarted
                    .read_artifact_bytes(&saved.checkpoint_id)
                    .unwrap()
                    .1,
                payload
            );
            assert_eq!(owner.port.scientific_digest_v1().unwrap(), before);
        }
    }

    #[test]
    fn checkpoint_catalog_rejects_ambiguous_directories_even_after_a_cached_read() {
        let first = tempfile::tempdir().unwrap().keep();
        let second = tempfile::tempdir().unwrap().keep();
        let (handle, host) = checkpoint_test_handle(&first, 4, 8 * 1024 * 1024);
        let request = CheckpointCreateRequest {
            description: None,
            idempotency_key: Some("one-admission-two-completed-catalogs".into()),
        };
        let metadata = handle.create_checkpoint(request.clone()).unwrap();
        let expected = handle
            .read_artifact_bytes(&metadata.checkpoint_id)
            .unwrap()
            .1;
        for suffix in [
            ".bin",
            CHECKPOINT_CATALOG_SUFFIX,
            CHECKPOINT_CATALOG_READY_SUFFIX,
        ] {
            let name = format!("{}{suffix}", metadata.checkpoint_id);
            fs::copy(first.join(&name), second.join(&name)).unwrap();
        }
        let handle = handle.with_artifacts_dir(second.clone()).unwrap();
        let before = host.port.scientific_digest_v1().unwrap();
        assert!(matches!(
            handle.create_checkpoint(request.clone()),
            Err(ControlError::Conflict(_))
        ));
        assert!(matches!(
            handle.get_checkpoint(&metadata.checkpoint_id),
            Err(ControlError::Conflict(_))
        ));
        assert!(matches!(
            handle.read_artifact_bytes(&metadata.checkpoint_id),
            Err(ControlError::Conflict(_))
        ));
        assert!(matches!(
            handle.list_checkpoints(None, None),
            Err(ControlError::Conflict(_))
        ));
        for _ in 0..2 {
            assert!(matches!(
                handle.create_checkpoint(request.clone()),
                Err(ControlError::Conflict(_))
            ));
            assert!(matches!(
                handle.get_checkpoint(&metadata.checkpoint_id),
                Err(ControlError::Conflict(_))
            ));
            assert!(matches!(
                handle.read_artifact_bytes(&metadata.checkpoint_id),
                Err(ControlError::Conflict(_))
            ));
        }
        assert!(handle.data_services.checkpoints.lock().unwrap().is_empty());
        assert!(handle.data_services.artifacts.lock().unwrap().is_empty());
        assert_eq!(
            handle.checkpoint_retention_usage().unwrap().request_entries,
            1
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
        for directory in [first, second] {
            assert_eq!(
                fs::read(directory.join(format!("{}.bin", metadata.checkpoint_id))).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn checkpoint_catalog_failure_preserves_pending_identity_and_never_overwrites_descriptor() {
        let directory = tempfile::tempdir().unwrap().keep();
        let (handle, host) = checkpoint_test_handle(&directory, 4, 8 * 1024 * 1024);
        let request = CheckpointCreateRequest {
            description: None,
            idempotency_key: Some("catalog-retry".into()),
        };
        let metadata = handle.create_checkpoint(request.clone()).unwrap();
        let path = directory.join(format!(
            "{}{CHECKPOINT_CATALOG_SUFFIX}",
            metadata.checkpoint_id
        ));
        let original = fs::read(&path).unwrap();
        fs::write(directory.join("descriptor.original"), &original).unwrap();
        fs::write(&path, b"partial descriptor").unwrap();
        assert!(matches!(
            write_checkpoint_catalog(&directory, &metadata),
            Err(ControlError::CheckpointRetentionUnavailable(_))
        ));
        assert_eq!(fs::read(&path).unwrap(), b"partial descriptor");
        // Model the retained state immediately before descriptor publication.
        // Discovery must never promote that request; only its exact retry can.
        {
            let mut retention = handle.data_services.checkpoint_retention.lock().unwrap();
            retention
                .requests
                .get_mut("checkpoint:catalog-retry")
                .unwrap()
                .acknowledged = false;
            handle.data_services.checkpoints.lock().unwrap().clear();
            handle.data_services.artifacts.lock().unwrap().clear();
        }
        assert_eq!(handle.list_checkpoints(None, None).unwrap().total, 0);
        let before = host.port.scientific_digest_v1().unwrap();
        assert!(matches!(
            handle.create_checkpoint(request),
            Err(ControlError::CheckpointRetentionUnavailable(_))
        ));
        let usage = handle.checkpoint_retention_usage().unwrap();
        assert_eq!(usage.file_count, 1);
        assert_eq!(usage.request_entries, 1);
        assert_eq!(usage.pending_entries, 1);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
        assert_eq!(fs::read(path).unwrap(), b"partial descriptor");
    }

    fn checkpoint_branch_fixture(
        preset: Option<crate::BrainPreset>,
        initial_ticks: usize,
    ) -> (ControlHandle, TestHost, CheckpointMetadataDto, PathBuf) {
        let directory = tempfile::Builder::new()
            .prefix("managed-checkpoint-branch-")
            .tempdir()
            .unwrap()
            .keep();
        let mut config = ScriptBotsConfig {
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            persistence_interval: 0,
            rng_seed: Some(42),
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            topography_enabled: false,
            reproduction_energy_threshold: f32::MAX,
            ..ScriptBotsConfig::default()
        };
        if preset == Some(crate::BrainPreset::Neuro) {
            config.neuroflow.hidden_layers = vec![7, 3];
            config.neuroflow.activation = scriptbots_core::NeuroflowActivationKind::Sigmoid;
        }
        let mut world = WorldState::new(config).unwrap();
        if let Some(preset) = preset {
            let installed = crate::install_brains(&mut world, preset).unwrap();
            for _ in 0..2 {
                let id = world
                    .try_spawn_agent(scriptbots_core::AgentData::default())
                    .expect("seed checkpoint branch founder");
                assert!(
                    world
                        .bind_agent_brain(id, installed.population()[0])
                        .unwrap()
                );
            }
        }
        let host = TestHost::spawn(world);
        let handle = host.handle().with_artifacts_dir(directory.clone()).unwrap();
        for _ in 0..initial_ticks {
            host.wait_applied(&handle.step().unwrap());
        }
        let checkpoint = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        (handle, host, checkpoint, directory)
    }

    fn native_export_test_artifact(
        metadata: &ArtifactMetadataDto,
    ) -> scriptbots_runtime::lab_workflows::LabArtifact {
        serde_json::from_value(serde_json::to_value(metadata).unwrap()).unwrap()
    }

    #[test]
    fn native_artifact_export_preserves_real_nonempty_checkpoint_and_paused_owner() {
        let (handle, host, checkpoint, directory) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 2);
        assert!(checkpoint.agent_count > 0);
        let before = handle.status().unwrap();
        assert!(before.paused);
        let digest = host.port.scientific_digest_v1().unwrap();
        let (metadata, original) = handle
            .read_artifact_bytes(&checkpoint.checkpoint_id)
            .unwrap();
        let selected = native_export_test_artifact(&metadata);
        let root = directory
            .canonicalize()
            .unwrap()
            .join("exports")
            .join("nested");
        assert!(!root.exists());

        let receipt = handle
            .export_lab_artifact(
                "native-checkpoint-copy",
                handle.session_id(),
                &selected,
                root.to_str().unwrap(),
            )
            .unwrap();
        let capsule = root.join("native-checkpoint-copy");
        assert_eq!(receipt.request_id, "native-checkpoint-copy");
        assert_eq!(receipt.session_id, handle.session_id());
        assert_eq!(receipt.artifact, selected);
        assert_eq!(receipt.export_directory, root.to_str().unwrap());
        assert_eq!(
            Path::new(&receipt.output_path),
            capsule.join(&selected.filename)
        );
        assert_eq!(
            Path::new(&receipt.manifest_path),
            capsule.join("export-manifest.json")
        );
        let copied = fs::read(&receipt.output_path).unwrap();
        assert_eq!(copied, original);
        assert_eq!(compute_blake3(&copied), metadata.checksum_blake3);
        assert_eq!(compute_sha256(&copied), metadata.checksum_sha256);
        let manifest: NativeArtifactExportManifestV1 =
            serde_json::from_slice(&fs::read(&receipt.manifest_path).unwrap()).unwrap();
        assert_eq!(manifest.schema, "scriptbots.native-artifact-export.v1");
        assert_eq!(manifest.request_id, receipt.request_id);
        assert_eq!(
            manifest.source_session_id,
            handle.session_id().get().to_string()
        );
        assert_eq!(manifest.artifact, selected);
        assert_eq!(manifest.export_directory, receipt.export_directory);
        assert_eq!(manifest.output_path, receipt.output_path);
        assert_eq!(manifest.manifest_path, receipt.manifest_path);
        assert_eq!(fs::read_dir(&capsule).unwrap().count(), 2);
        assert_eq!(
            handle
                .read_artifact_bytes(&checkpoint.checkpoint_id)
                .unwrap(),
            (metadata, original)
        );
        assert_eq!(handle.status().unwrap(), before);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), digest);
    }

    #[test]
    fn native_artifact_export_rejects_every_stale_descriptor_before_creating_destination() {
        let (handle, host, checkpoint, directory) = checkpoint_branch_fixture(None, 0);
        let (metadata, original) = handle
            .read_artifact_bytes(&checkpoint.checkpoint_id)
            .unwrap();
        let selected = native_export_test_artifact(&metadata);
        let root = directory.canonicalize().unwrap().join("refused-exports");
        let digest = host.port.scientific_digest_v1().unwrap();
        for field in [
            "artifact_id",
            "filename",
            "content_type",
            "byte_size",
            "checksum_blake3",
            "checksum_sha256",
            "created_at_utc",
            "relative_path",
        ] {
            let mut stale = selected.clone();
            match field {
                "artifact_id" => stale.artifact_id = "missing-native-artifact".to_owned(),
                "filename" => stale.filename = "different.bin".to_owned(),
                "content_type" => stale.content_type = "application/x-different".to_owned(),
                "byte_size" => stale.byte_size += 1,
                "checksum_blake3" => stale.checksum_blake3 = compute_blake3(b"different"),
                "checksum_sha256" => stale.checksum_sha256 = compute_sha256(b"different"),
                "created_at_utc" => stale.created_at_utc = "2000-01-01T00:00:00Z".to_owned(),
                "relative_path" => stale.relative_path = "different-source.bin".to_owned(),
                _ => unreachable!(),
            }
            stale.validate().unwrap();
            assert!(
                handle
                    .export_lab_artifact(
                        field,
                        handle.session_id(),
                        &stale,
                        root.to_str().unwrap(),
                    )
                    .is_err(),
                "{field}"
            );
            assert!(!root.exists(), "{field} created an export directory");
        }
        assert_eq!(
            handle
                .read_artifact_bytes(&checkpoint.checkpoint_id)
                .unwrap()
                .1,
            original
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), digest);
    }

    #[test]
    fn native_artifact_export_rejects_wrong_session_unsafe_identity_and_size_before_output() {
        let (handle, _host, checkpoint, directory) = checkpoint_branch_fixture(None, 0);
        let selected = native_export_test_artifact(&checkpoint_artifact_metadata(&checkpoint));
        let root = directory.canonicalize().unwrap().join("refused-exports");
        let wrong_session = scriptbots_runtime::HostSessionId::new(handle.session_id().get() + 1);
        assert!(matches!(
            handle.export_lab_artifact(
                "wrong-session",
                wrong_session,
                &selected,
                root.to_str().unwrap(),
            ),
            Err(ControlError::Conflict(_))
        ));
        for request in ["", "../escape", "/absolute", "nested/request", "has space"] {
            assert!(matches!(
                handle.export_lab_artifact(
                    request,
                    handle.session_id(),
                    &selected,
                    root.to_str().unwrap(),
                ),
                Err(ControlError::BadRequest(_))
            ));
        }
        let mut oversized = selected.clone();
        oversized.byte_size = MAX_ARTIFACT_DOWNLOAD_BYTES as u64 + 1;
        assert!(matches!(
            handle.export_lab_artifact(
                "oversized",
                handle.session_id(),
                &oversized,
                root.to_str().unwrap(),
            ),
            Err(ControlError::PayloadTooLarge(_))
        ));
        let mut reserved = selected.clone();
        reserved.filename = "export-manifest.json".to_owned();
        assert!(matches!(
            handle.export_lab_artifact(
                "reserved-filename",
                handle.session_id(),
                &reserved,
                root.to_str().unwrap(),
            ),
            Err(ControlError::BadRequest(_))
        ));
        for destination in [
            "relative-export".to_owned(),
            format!("{}/../escaped-export", root.display()),
            format!("{}\ninvalid", root.display()),
        ] {
            assert!(matches!(
                handle.export_lab_artifact(
                    "invalid-directory",
                    handle.session_id(),
                    &selected,
                    &destination,
                ),
                Err(ControlError::BadRequest(_))
            ));
        }
        assert!(!root.exists());
    }

    #[test]
    fn native_artifact_export_refuses_corrupt_source_without_creating_destination() {
        let (handle, host, checkpoint, directory) = checkpoint_branch_fixture(None, 0);
        let metadata = checkpoint_artifact_metadata(&checkpoint);
        let selected = native_export_test_artifact(&metadata);
        let directory = directory.canonicalize().unwrap();
        let source = directory.join(&metadata.relative_path);
        let saved_original = directory.join("original-checkpoint-before-corruption.bin");
        fs::rename(&source, &saved_original).unwrap();
        fs::write(&source, b"corrupt source bytes").unwrap();
        let root = directory.join("refused-export");
        let digest = host.port.scientific_digest_v1().unwrap();
        assert!(
            handle
                .export_lab_artifact(
                    "corrupt-source",
                    handle.session_id(),
                    &selected,
                    root.to_str().unwrap(),
                )
                .is_err()
        );
        assert!(!root.exists());
        assert!(saved_original.is_file());
        assert_eq!(fs::read(&source).unwrap(), b"corrupt source bytes");
        assert_eq!(host.port.scientific_digest_v1().unwrap(), digest);
    }

    #[test]
    fn native_artifact_export_never_reuses_existing_request_directory() {
        let (handle, _host, checkpoint, directory) = checkpoint_branch_fixture(None, 0);
        let selected = native_export_test_artifact(&checkpoint_artifact_metadata(&checkpoint));
        let root = directory.canonicalize().unwrap().join("exports");
        let capsule = root.join("already-present");
        fs::create_dir_all(&capsule).unwrap();
        let sentinel = capsule.join("keep-this.txt");
        fs::write(&sentinel, b"original user content").unwrap();
        let error = handle
            .export_lab_artifact(
                "already-present",
                handle.session_id(),
                &selected,
                root.to_str().unwrap(),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("not confirmed"));
        assert!(error.contains("preserved"));
        assert!(error.contains(capsule.to_str().unwrap()));
        assert_eq!(fs::read(&sentinel).unwrap(), b"original user content");
        assert_eq!(fs::read_dir(&capsule).unwrap().count(), 1);
        assert!(!capsule.join(&selected.filename).exists());
        assert!(!capsule.join("export-manifest.json").exists());
    }

    #[cfg(unix)]
    #[test]
    fn native_artifact_export_refuses_source_root_and_request_symlinks() {
        use std::os::unix::fs::symlink;

        let (handle, _host, checkpoint, directory) = checkpoint_branch_fixture(None, 0);
        let metadata = checkpoint_artifact_metadata(&checkpoint);
        let selected = native_export_test_artifact(&metadata);
        let directory = directory.canonicalize().unwrap();
        let outside = directory.join("outside");
        fs::create_dir(&outside).unwrap();
        let sentinel = outside.join("sentinel");
        fs::write(&sentinel, b"untouched").unwrap();
        let alias = directory.join("aliased-export-root");
        symlink(&outside, &alias).unwrap();
        let aliased_root = alias.join("new-root");
        assert!(
            handle
                .export_lab_artifact(
                    "root-alias",
                    handle.session_id(),
                    &selected,
                    aliased_root.to_str().unwrap(),
                )
                .is_err()
        );
        assert!(!outside.join("new-root").exists());

        let root = directory.join("exports");
        fs::create_dir(&root).unwrap();
        symlink(&outside, root.join("request-alias")).unwrap();
        assert!(
            handle
                .export_lab_artifact(
                    "request-alias",
                    handle.session_id(),
                    &selected,
                    root.to_str().unwrap(),
                )
                .is_err()
        );
        assert_eq!(fs::read_dir(&outside).unwrap().count(), 1);
        assert_eq!(fs::read(&sentinel).unwrap(), b"untouched");

        let source = directory.join(&metadata.relative_path);
        let original = directory.join("original-before-source-symlink.bin");
        fs::rename(&source, &original).unwrap();
        symlink(&original, &source).unwrap();
        let refused = directory.join("refused-source-alias");
        assert!(
            handle
                .export_lab_artifact(
                    "source-alias",
                    handle.session_id(),
                    &selected,
                    refused.to_str().unwrap(),
                )
                .is_err()
        );
        assert!(!refused.exists());
        assert!(original.is_file());
    }

    #[test]
    fn native_artifact_export_accepts_an_exact_empty_retained_file() {
        let directory = tempfile::Builder::new()
            .prefix("native-empty-artifact-")
            .tempdir()
            .unwrap()
            .keep()
            .canonicalize()
            .unwrap();
        let (handle, host) = checkpoint_test_handle(&directory, 2, 8 * 1024 * 1024);
        let metadata = ArtifactMetadataDto {
            artifact_id: "ordinary-empty".to_owned(),
            filename: "empty.jsonl".to_owned(),
            content_type: "application/x-ndjson".to_owned(),
            byte_size: 0,
            checksum_blake3: compute_blake3(&[]),
            checksum_sha256: compute_sha256(&[]),
            created_at_utc: "2026-01-01T00:00:00Z".to_owned(),
            relative_path: "empty.jsonl".to_owned(),
        };
        fs::write(directory.join(&metadata.relative_path), b"").unwrap();
        handle.data_services.artifacts.lock().unwrap().insert(
            metadata.artifact_id.clone(),
            RetainedArtifact {
                metadata: metadata.clone(),
                directory: directory.clone(),
            },
        );
        let selected = native_export_test_artifact(&metadata);
        let digest = host.port.scientific_digest_v1().unwrap();
        let root = directory.join("exports");
        let receipt = handle
            .export_lab_artifact(
                "empty-copy",
                handle.session_id(),
                &selected,
                root.to_str().unwrap(),
            )
            .unwrap();
        assert!(fs::read(&receipt.output_path).unwrap().is_empty());
        let manifest: NativeArtifactExportManifestV1 =
            serde_json::from_slice(&fs::read(&receipt.manifest_path).unwrap()).unwrap();
        assert_eq!(manifest.artifact, selected);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), digest);
    }

    #[test]
    fn native_artifact_export_retains_partial_write_and_refuses_overwrite() {
        let directory = tempfile::Builder::new()
            .prefix("native-partial-export-")
            .tempdir()
            .unwrap()
            .keep();
        let path = directory.join("payload.bin");
        let bytes = b"the exact retained source bytes";
        let error = write_native_artifact_export_file(&path, bytes, |file, bytes| {
            file.write_all(&bytes[..4])?;
            Err(std::io::Error::other(
                "injected write failure after four bytes",
            ))
        })
        .map_err(|error| preserve_native_artifact_export_failure(&directory, error))
        .unwrap_err()
        .to_string();
        assert!(error.contains("injected write failure"));
        assert!(error.contains("preserved"));
        assert!(error.contains(directory.to_str().unwrap()));
        assert_eq!(fs::read(&path).unwrap(), &bytes[..4]);
        assert!(
            write_native_artifact_export_file(&path, bytes, |file, bytes| {
                file.write_all(bytes)
            })
            .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), &bytes[..4]);
    }

    #[test]
    fn native_artifact_export_readback_rejects_corruption_and_same_bytes_replacement() {
        let directory = tempfile::Builder::new()
            .prefix("native-export-readback-")
            .tempdir()
            .unwrap()
            .keep();
        let path = directory.join("payload.bin");
        let bytes = b"verified retained bytes";
        let original =
            write_native_artifact_export_file(&path, bytes, |file, bytes| file.write_all(bytes))
                .unwrap();
        fs::write(&path, b"corrupt copied bytes").unwrap();
        assert!(
            verify_native_artifact_export_file(
                &path,
                &original,
                MAX_ARTIFACT_DOWNLOAD_BYTES,
                bytes.len() as u64,
                &compute_blake3(bytes),
                &compute_sha256(bytes),
            )
            .is_err()
        );
        fs::rename(&path, directory.join("preserved-corrupt-copy.bin")).unwrap();
        fs::write(&path, bytes).unwrap();
        assert!(
            verify_native_artifact_export_file(
                &path,
                &original,
                MAX_ARTIFACT_DOWNLOAD_BYTES,
                bytes.len() as u64,
                &compute_blake3(bytes),
                &compute_sha256(bytes),
            )
            .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }

    #[cfg(windows)]
    #[test]
    fn native_artifact_export_accepts_only_equivalent_windows_canonical_prefixes() {
        assert!(native_artifact_export_path_is_canonical(
            Path::new(r"C:\exports"),
            Path::new(r"\\?\C:\exports"),
        ));
        assert!(native_artifact_export_path_is_canonical(
            Path::new(r"\\server\share\exports"),
            Path::new(r"\\?\UNC\server\share\exports"),
        ));
        assert!(!native_artifact_export_path_is_canonical(
            Path::new(r"C:\aliased"),
            Path::new(r"\\?\C:\different"),
        ));
    }

    fn checkpoint_branch_request(checkpoint_id: &str, key: &str) -> CheckpointBranchRequest {
        CheckpointBranchRequest {
            checkpoint_id: checkpoint_id.to_owned(),
            branch_name: "saved-future".to_owned(),
            ticks: 3,
            intervention: None,
            intervention_delay_ticks: 0,
            idempotency_key: key.to_owned(),
        }
    }

    fn wait_for_checkpoint_branch(
        handle: &ControlHandle,
        branch_id: &str,
    ) -> CheckpointBranchStatusDto {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let status = handle.get_checkpoint_branch(branch_id).unwrap();
            if !matches!(status.status.as_str(), "running" | "cancelling") {
                return status;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "branch did not finish: {status:?}"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[test]
    fn checkpoint_world_digest_and_artifact_checksums_have_distinct_formats() {
        assert!(valid_world_digest_v1("0123456789abcdef"));
        for invalid in [
            "0123456789abcde",
            "0123456789abcdef0",
            "0123456789abcdeF",
            "0123456789abcdeg",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        ] {
            assert!(!valid_world_digest_v1(invalid), "{invalid}");
        }
        assert!(!valid_branch_digest("0123456789abcdef"));
        assert!(valid_branch_digest(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        ));
        assert!(!valid_branch_digest(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeF"
        ));
    }

    #[test]
    fn managed_checkpoint_branches_restore_real_saved_cuts_and_preserve_live_owner() {
        let mut distinct = std::collections::BTreeSet::new();
        for (preset, ticks) in [
            (crate::BrainPreset::Mlp, 0),
            (crate::BrainPreset::Dwraon, 1),
            #[cfg(feature = "neuro")]
            (crate::BrainPreset::Neuro, 2),
        ] {
            let (handle, host, checkpoint, directory) =
                checkpoint_branch_fixture(Some(preset), ticks);
            assert_eq!(checkpoint.agent_count, 2);
            assert!(distinct.insert(checkpoint.world_digest_v1.clone()));
            let live = host.port.scientific_digest_v1().unwrap();
            let published = host.port.snapshot_hub().latest();
            let (_, original_bytes) = handle
                .read_artifact_bytes(&checkpoint.checkpoint_id)
                .unwrap();
            let request = checkpoint_branch_request(&checkpoint.checkpoint_id, "exact-source");
            let admitted = handle.create_checkpoint_branch(request.clone()).unwrap();
            assert_eq!(admitted.source_tick, ticks as u64);
            assert_eq!(admitted.target_tick, ticks as u64 + 3);
            assert_eq!(admitted.world_digest_v1, checkpoint.world_digest_v1);
            assert_eq!(
                handle
                    .create_checkpoint_branch(request.clone())
                    .unwrap()
                    .branch_id,
                admitted.branch_id
            );
            let mut conflict = request;
            conflict.ticks += 1;
            assert!(matches!(
                handle.create_checkpoint_branch(conflict),
                Err(ControlError::Conflict(_))
            ));
            let final_status = wait_for_checkpoint_branch(&handle, &admitted.branch_id);
            assert_eq!(final_status.status, "completed", "{final_status:?}");
            let (metadata, report_bytes) = handle
                .read_artifact_bytes(final_status.artifact_id.as_deref().unwrap())
                .unwrap();
            assert_eq!(compute_blake3(&report_bytes), metadata.checksum_blake3);
            let report: crate::branch_study::BranchStudyReport =
                serde_json::from_slice(&report_bytes).unwrap();
            assert_eq!(report.pre_intervention_digest, checkpoint.world_digest_v1);
            assert!(report.pre_intervention_digest_verified_for_all_branches);
            assert_eq!(report.branch_reports.len(), 1);
            let outcome = &report.branch_reports[0];
            assert_eq!(outcome.terminal_sample.tick, final_status.target_tick);
            assert!(outcome.bundle_verified);
            let decoded = scriptbots_core::WorldCheckpointV1::decode(&original_bytes).unwrap();
            let registry = crate::create_brain_registry_for_config(preset, decoded.config())
                .expect("known original preset and constructor configuration");
            let mut oracle = WorldState::restore_checkpoint_v1(&decoded, registry).unwrap();
            for _ in 0..3 {
                oracle.step().unwrap();
            }
            assert_eq!(
                outcome.final_digest,
                oracle.world_digest_v1().unwrap().overall
            );
            assert_eq!(
                fs::read(
                    directory
                        .join("branches")
                        .join(&admitted.branch_id)
                        .join("study")
                        .join("source_checkpoint.bin")
                )
                .unwrap(),
                original_bytes
            );
            let artifacts = handle.list_artifacts(Some(100), None).unwrap();
            assert!(
                artifacts
                    .items
                    .iter()
                    .any(|item| item.filename == "source_checkpoint.bin")
            );
            assert!(
                artifacts
                    .items
                    .iter()
                    .any(|item| item.filename == "bundle_manifest.json")
            );
            // The same published plan consumed by the offline CLI must restore
            // this saved cut, including tick zero, instead of seeding a new world.
            let plan_artifact = artifacts
                .items
                .iter()
                .find(|item| item.filename == "study_plan.json")
                .expect("published reproducible plan");
            let (_, plan_bytes) = handle
                .read_artifact_bytes(&plan_artifact.artifact_id)
                .unwrap();
            let retained_plan: crate::branch_study::BranchStudyPlan =
                serde_json::from_slice(&plan_bytes).unwrap();
            let retained_source = retained_plan
                .source_checkpoint
                .as_ref()
                .expect("saved-cut plan cannot imply seeded warmup");
            assert!(retained_source.is_absolute());
            assert_eq!(
                retained_source,
                &directory
                    .join("branches")
                    .join(&admitted.branch_id)
                    .join("study")
                    .join("source_checkpoint.bin")
            );
            assert_eq!(fs::read(retained_source).unwrap(), original_bytes);
            let rerun_output = directory.join("offline-reproduced");
            let rerunner = crate::branch_study::BranchStudyOrchestrator::new(
                retained_plan,
                rerun_output.clone(),
            );
            rerunner.validate().expect("published plan is executable");
            let reproduced = std::thread::Builder::new()
                .stack_size(32 * 1024 * 1024)
                .spawn(move || rerunner.execute_study(None))
                .unwrap()
                .join()
                .unwrap()
                .unwrap();
            assert_eq!(reproduced.source, report.source);
            assert_eq!(
                reproduced.pre_intervention_digest,
                report.pre_intervention_digest
            );
            assert_eq!(reproduced.branch_reports.len(), report.branch_reports.len());
            assert_eq!(
                reproduced.branch_reports[0].final_digest,
                outcome.final_digest
            );
            assert_eq!(
                reproduced.branch_reports[0].terminal_sample,
                outcome.terminal_sample
            );
            assert!(reproduced.branch_reports[0].bundle_verified);
            assert_eq!(
                fs::read(rerun_output.join("source_checkpoint.bin")).unwrap(),
                original_bytes
            );
            for artifact in artifacts.items {
                handle.read_artifact_bytes(&artifact.artifact_id).unwrap();
            }
            assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
            assert!(Arc::ptr_eq(&host.port.snapshot_hub().latest(), &published));
            assert_eq!(
                handle
                    .data_services
                    .checkpoint_branches
                    .lock()
                    .unwrap()
                    .len(),
                1
            );
            handle.data_services.shutdown_experiments().unwrap();
            assert_eq!(
                handle
                    .data_services
                    .execution_capacity
                    .lock()
                    .unwrap()
                    .reserved_run_slots,
                0
            );
        }
    }

    #[test]
    fn checkpoint_branch_delay_preserves_zero_delay_identity_and_rejects_invalid_timing() {
        let original = br#"{"checkpoint_id":"source","branch_name":"future","ticks":3,"idempotency_key":"legacy"}"#;
        let request: CheckpointBranchRequest = serde_json::from_slice(original).unwrap();
        assert_eq!(request.intervention_delay_ticks, 0);
        assert_eq!(request.bounded_payload().unwrap().as_slice(), original);
        let mut explicit_zero = serde_json::to_value(&request).unwrap();
        explicit_zero["intervention_delay_ticks"] = serde_json::json!(0);
        let explicit_zero: CheckpointBranchRequest =
            serde_json::from_value(explicit_zero).unwrap();
        assert_eq!(explicit_zero.bounded_payload().unwrap().as_slice(), original);

        let mut delayed = request.clone();
        delayed.intervention =
            Some(crate::branch_study::StudyIntervention::ClosedWorld { closed: true });
        for delay in 0..delayed.ticks {
            delayed.intervention_delay_ticks = delay;
            delayed.validate_shape().unwrap();
            let branches = checkpoint_branch_specs(&delayed, 17).unwrap();
            assert_eq!(branches[0].scheduled_tick, None);
            assert_eq!(branches[1].scheduled_tick, Some(17 + delay));
        }
        for delay in [delayed.ticks, delayed.ticks + 1, u64::MAX] {
            delayed.intervention_delay_ticks = delay;
            assert!(matches!(
                delayed.validate_shape(),
                Err(ControlError::BadRequest(_))
            ));
        }
        delayed.intervention_delay_ticks = 1;
        assert!(matches!(
            checkpoint_branch_specs(&delayed, u64::MAX),
            Err(ControlError::BadRequest(_))
        ));
        delayed.intervention = None;
        assert!(matches!(
            delayed.validate_shape(),
            Err(ControlError::BadRequest(_))
        ));
        for invalid in [
            serde_json::json!(-1),
            serde_json::json!(0.5),
            serde_json::json!("1"),
            serde_json::Value::Null,
        ] {
            let mut malformed = serde_json::to_value(&request).unwrap();
            malformed["intervention_delay_ticks"] = invalid;
            assert!(serde_json::from_value::<CheckpointBranchRequest>(malformed).is_err());
        }
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one real saved-source study proves delayed effects, exact retry, offline reproduction and cold recovery"
    )]
    fn managed_delayed_treatment_preserves_pre_command_science_and_recovered_timing() {
        let (handle, host, checkpoint, directory) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 2);
        let before = host.port.scientific_digest_v1().unwrap();
        let mut request = checkpoint_branch_request(&checkpoint.checkpoint_id, "delayed-treatment");
        request.intervention = Some(crate::branch_study::StudyIntervention::Drought {
            region: None,
            duration_ticks: 2,
            growth_scale: 0.0,
        });
        request.intervention_delay_ticks = request.ticks - 1;
        let due = checkpoint.tick + request.intervention_delay_ticks;
        let admitted = handle.create_checkpoint_branch(request.clone()).unwrap();
        assert_eq!(
            handle
                .checkpoint_branch_intervention_tick(&admitted.branch_id)
                .unwrap(),
            Some(due)
        );
        let completed = wait_for_checkpoint_branch(&handle, &admitted.branch_id);
        assert_eq!(completed.status, "completed", "{completed:?}");
        assert_eq!(
            handle.create_checkpoint_branch(request.clone()).unwrap(),
            completed
        );
        let (_, bytes) = handle
            .read_artifact_bytes(completed.artifact_id.as_deref().unwrap())
            .unwrap();
        let report: crate::branch_study::BranchStudyReport =
            serde_json::from_slice(&bytes).unwrap();
        let treatment = report
            .branch_reports
            .iter()
            .find(|branch| branch.branch_id == "treatment")
            .unwrap();
        assert_eq!(treatment.command_applied_tick, Some(due));
        assert_eq!(treatment.first_divergence.as_ref().unwrap().tick, due + 1);
        assert_eq!(treatment.terminal_sample.tick, completed.target_tick);
        let rows = |id: &str| {
            let branch = report
                .branch_reports
                .iter()
                .find(|branch| branch.branch_id == id)
                .unwrap();
            assert!(branch.bundle_verified);
            fs::read_to_string(
                Path::new(branch.bundle_path.as_deref().unwrap()).join("exports/summary.csv"),
            )
            .unwrap()
            .lines()
            .skip(1)
            .map(str::to_owned)
            .collect::<Vec<_>>()
        };
        let control_rows = rows("control");
        let treatment_rows = rows("treatment");
        let before_command = usize::try_from(request.intervention_delay_ticks).unwrap() + 1;
        assert_eq!(
            control_rows.len(),
            usize::try_from(request.ticks).unwrap() + 1
        );
        assert_eq!(
            control_rows[..before_command],
            treatment_rows[..before_command]
        );
        assert_ne!(control_rows.last(), treatment_rows.last());

        let plan: crate::branch_study::BranchStudyPlan = serde_json::from_slice(
            &fs::read(
                directory
                    .join("branches")
                    .join(&completed.branch_id)
                    .join("study/study_plan.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(plan.branches[1].scheduled_tick, Some(due));
        let reproduced = crate::branch_study::BranchStudyOrchestrator::new(
            plan,
            directory.join("delayed-offline-reproduction"),
        )
        .execute_study(None)
        .unwrap();
        assert_eq!(reproduced.source, report.source);
        for (original, replayed) in report.branch_reports.iter().zip(&reproduced.branch_reports) {
            assert_eq!(original.terminal_sample, replayed.terminal_sample);
            assert_eq!(original.command_applied_tick, replayed.command_applied_tick);
            assert_eq!(original.first_divergence, replayed.first_divergence);
        }
        let mut changed = request.clone();
        changed.intervention_delay_ticks -= 1;
        assert!(matches!(
            handle.create_checkpoint_branch(changed.clone()),
            Err(ControlError::Conflict(_))
        ));
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
        handle.data_services.shutdown_experiments().unwrap();
        drop(handle);

        let cold = host.handle().with_artifacts_dir(directory).unwrap();
        assert_eq!(cold.create_checkpoint_branch(request).unwrap(), completed);
        assert_eq!(
            cold.checkpoint_branch_intervention_tick(&completed.branch_id)
                .unwrap(),
            Some(due)
        );
        assert!(matches!(
            cold.create_checkpoint_branch(changed),
            Err(ControlError::Conflict(_))
        ));
        assert_eq!(cold.list_checkpoint_branches(None, None).unwrap().total, 1);
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
    }

    #[test]
    fn managed_checkpoint_branch_treatment_shares_source_and_records_actual_divergence() {
        let (handle, host, checkpoint, _) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 0);
        let before = host.port.scientific_digest_v1().unwrap();
        let mut request = checkpoint_branch_request(&checkpoint.checkpoint_id, "paired-treatment");
        request.intervention = Some(crate::branch_study::StudyIntervention::Drought {
            region: None,
            duration_ticks: 2,
            growth_scale: 0.0,
        });
        let admitted = handle.create_checkpoint_branch(request).unwrap();
        let final_status = wait_for_checkpoint_branch(&handle, &admitted.branch_id);
        assert_eq!(final_status.status, "completed", "{final_status:?}");
        let (_, bytes) = handle
            .read_artifact_bytes(final_status.artifact_id.as_deref().unwrap())
            .unwrap();
        let report: crate::branch_study::BranchStudyReport =
            serde_json::from_slice(&bytes).unwrap();
        assert_eq!(report.branch_reports.len(), 2);
        assert!(report.pre_intervention_digest_verified_for_all_branches);
        let treatment = report
            .branch_reports
            .iter()
            .find(|branch| branch.branch_id == "treatment")
            .unwrap();
        assert_eq!(treatment.command_applied_tick, Some(checkpoint.tick));
        assert!(treatment.first_divergence.is_some());
        assert!(
            report
                .branch_reports
                .iter()
                .all(|branch| branch.bundle_verified
                    && branch.terminal_sample.tick == final_status.target_tick)
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
        handle.data_services.shutdown_experiments().unwrap();
    }

    #[test]
    fn managed_checkpoint_branch_artifacts_page_verified_files_without_crossing_studies() {
        let (handle, host, checkpoint, directory) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 0);
        let before = host.port.scientific_digest_v1().unwrap();
        let first = handle
            .create_checkpoint_branch(checkpoint_branch_request(
                &checkpoint.checkpoint_id,
                "first",
            ))
            .unwrap();
        let first = wait_for_checkpoint_branch(&handle, &first.branch_id);
        assert_eq!(first.status, "completed");
        let second = handle
            .create_checkpoint_branch(checkpoint_branch_request(
                &checkpoint.checkpoint_id,
                "second",
            ))
            .unwrap();
        let second = wait_for_checkpoint_branch(&handle, &second.branch_id);
        assert_eq!(second.status, "completed");

        let prefix = format!("branches/{}/study/", first.branch_id);
        let mut cursor = None;
        let mut ids = std::collections::BTreeSet::new();
        let mut paths = std::collections::BTreeSet::new();
        let mut total = None;
        loop {
            let page = handle
                .list_checkpoint_branch_artifacts(&first.branch_id, Some(2), cursor.as_deref())
                .unwrap();
            assert_eq!(page.limit, 2);
            assert_eq!(page.offset, ids.len());
            assert_eq!(*total.get_or_insert(page.total), page.total);
            assert!(!page.items.is_empty());
            for item in page.items {
                assert!(item.relative_path.starts_with(&prefix));
                assert!(
                    ids.insert(item.artifact_id.clone()),
                    "pagination repeated a file"
                );
                paths.insert(item.relative_path.clone());
                let (download, bytes) = handle.read_artifact_bytes(&item.artifact_id).unwrap();
                assert_eq!(download.relative_path, item.relative_path);
                assert_eq!(bytes.len() as u64, item.byte_size);
                assert_eq!(compute_blake3(&bytes), item.checksum_blake3);
                assert_eq!(compute_sha256(&bytes), item.checksum_sha256);
                assert_eq!(
                    fs::read(directory.join(&item.relative_path)).unwrap(),
                    bytes
                );
            }
            if !page.has_more {
                assert!(page.next_cursor.is_none());
                break;
            }
            assert_eq!(page.next_cursor, Some(ids.len().to_string()));
            cursor = page.next_cursor;
        }
        assert_eq!(ids.len(), total.unwrap());
        assert!(ids.contains(first.artifact_id.as_ref().unwrap()));
        assert!(!ids.contains(second.artifact_id.as_ref().unwrap()));
        assert!(!ids.contains(&checkpoint.checkpoint_id));
        for filename in [
            "study_plan.json",
            "study_source.json",
            "study_summary.md",
            "study_report.json",
            "source_checkpoint.bin",
            "bundle_control/bundle_manifest.json",
        ] {
            assert!(paths.contains(&format!("{prefix}{filename}")), "{filename}");
        }
        let beyond = handle
            .list_checkpoint_branch_artifacts(&first.branch_id, Some(1000), Some("999999"))
            .unwrap();
        assert_eq!(beyond.limit, 100);
        assert!(beyond.items.is_empty());
        assert!(!beyond.has_more);
        assert!(beyond.next_cursor.is_none());
        assert!(matches!(
            handle.list_checkpoint_branch_artifacts(&first.branch_id, None, Some("invalid")),
            Err(ControlError::BadRequest(_))
        ));
        assert!(matches!(
            handle.list_checkpoint_branch_artifacts("unknown", None, None),
            Err(ControlError::NotFound(_))
        ));
        assert!(matches!(
            handle.list_checkpoint_branch_artifacts("../escape", None, None),
            Err(ControlError::BadRequest(_))
        ));
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
        handle.data_services.shutdown_experiments().unwrap();
        drop(handle);

        // The branch inventory itself must restore sealed artifacts after a
        // restart; it cannot depend on a preceding whole-catalog list or download.
        let cold = host.handle().with_artifacts_dir(directory).unwrap();
        assert!(cold.data_services.artifacts.lock().unwrap().is_empty());
        let inventory = cold
            .list_checkpoint_branch_artifacts(&first.branch_id, Some(100), None)
            .unwrap();
        assert_eq!(inventory.total, ids.len());
        assert_eq!(
            inventory
                .items
                .into_iter()
                .map(|item| item.artifact_id)
                .collect::<std::collections::BTreeSet<_>>(),
            ids
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
    }

    #[test]
    fn managed_checkpoint_branch_refuses_invalid_inputs_corrupt_sources_and_capacity_before_admission()
     {
        let (handle, host, checkpoint, directory) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 0);
        let before = host.port.scientific_digest_v1().unwrap();
        let request = checkpoint_branch_request(&checkpoint.checkpoint_id, "refusals");
        for ticks in [0, MAX_CHECKPOINT_BRANCH_TICKS + 1] {
            let mut invalid = request.clone();
            invalid.ticks = ticks;
            assert!(matches!(
                handle.create_checkpoint_branch(invalid),
                Err(ControlError::BadRequest(_))
            ));
        }
        let mut invalid = request.clone();
        invalid.branch_name = "../escape".to_owned();
        assert!(matches!(
            handle.create_checkpoint_branch(invalid),
            Err(ControlError::BadRequest(_))
        ));
        let mut oversized = request.clone();
        oversized.intervention = Some(crate::branch_study::StudyIntervention::ConfigPatch(
            serde_json::json!({"unknown": "x".repeat(MAX_CHECKPOINT_BRANCH_REQUEST_BYTES)}),
        ));
        assert!(matches!(
            handle.create_checkpoint_branch(oversized),
            Err(ControlError::PayloadTooLarge(_))
        ));
        assert!(matches!(
            handle.get_checkpoint_branch("../escape"),
            Err(ControlError::BadRequest(_))
        ));
        assert!(matches!(
            handle.cancel_checkpoint_branch("../escape"),
            Err(ControlError::BadRequest(_))
        ));
        let mut unknown = request.clone();
        unknown.checkpoint_id = "missing".to_owned();
        assert!(matches!(
            handle.create_checkpoint_branch(unknown),
            Err(ControlError::NotFound(_))
        ));
        let handle = handle.with_experiment_run_capacity(0).unwrap();
        assert!(matches!(
            handle.create_checkpoint_branch(request.clone()),
            Err(ControlError::ExperimentCapacity(_))
        ));
        let handle = handle.with_experiment_run_capacity(1).unwrap();
        let (artifact, original) = handle
            .read_artifact_bytes(&checkpoint.checkpoint_id)
            .unwrap();
        fs::write(directory.join("source-before-corruption.raw"), original).unwrap();
        fs::OpenOptions::new()
            .append(true)
            .open(directory.join(artifact.relative_path))
            .unwrap()
            .write_all(&[0xff])
            .unwrap();
        assert!(handle.create_checkpoint_branch(request).is_err());
        assert!(
            handle
                .data_services
                .checkpoint_branches
                .lock()
                .unwrap()
                .is_empty()
        );
        assert!(!directory.join("branches").exists());
        assert_eq!(
            handle
                .data_services
                .execution_capacity
                .lock()
                .unwrap()
                .reserved_run_slots,
            0
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);

        let (unsupported, unsupported_host, cut, unsupported_directory) =
            checkpoint_branch_fixture(None, 0);
        assert!(matches!(
            unsupported
                .create_checkpoint_branch(checkpoint_branch_request(&cut.checkpoint_id, "unbound")),
            Err(ControlError::BadRequest(_))
        ));
        assert!(
            unsupported
                .data_services
                .checkpoint_branches
                .lock()
                .unwrap()
                .is_empty()
        );
        assert!(!unsupported_directory.join("branches").exists());
        assert_eq!(
            unsupported_host
                .port
                .scientific_digest_v1()
                .unwrap()
                .overall,
            cut.world_digest_v1
        );
    }

    #[test]
    fn managed_checkpoint_branch_catalog_recovers_receipts_retries_pages_and_cold_downloads() {
        let (handle, host, checkpoint, directory) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 0);
        let before = host.port.scientific_digest_v1().unwrap();
        let request = checkpoint_branch_request(&checkpoint.checkpoint_id, "restart-control");
        let admitted = handle.create_checkpoint_branch(request.clone()).unwrap();
        let completed = wait_for_checkpoint_branch(&handle, &admitted.branch_id);
        assert_eq!(completed.status, "completed", "{completed:?}");
        let artifact_id = completed.artifact_id.clone().unwrap();
        let original = handle.read_artifact_bytes(&artifact_id).unwrap();
        let mut treatment =
            checkpoint_branch_request(&checkpoint.checkpoint_id, "restart-treatment");
        treatment.intervention =
            Some(crate::branch_study::StudyIntervention::ClosedWorld { closed: false });
        let treated = handle.create_checkpoint_branch(treatment).unwrap();
        assert_eq!(
            wait_for_checkpoint_branch(&handle, &treated.branch_id).status,
            "completed"
        );
        handle.data_services.shutdown_experiments().unwrap();
        drop(handle);

        let cold = host.handle().with_artifacts_dir(directory).unwrap();
        assert!(cold.data_services.artifacts.lock().unwrap().is_empty());
        // No list/status call precedes this known-ID download.
        assert_eq!(cold.read_artifact_bytes(&artifact_id).unwrap(), original);
        assert_eq!(
            cold.get_checkpoint_branch(&admitted.branch_id).unwrap(),
            completed
        );
        assert_eq!(
            cold.create_checkpoint_branch(request.clone()).unwrap(),
            completed
        );
        let mut conflicting = request;
        conflicting.ticks += 1;
        assert!(matches!(
            cold.create_checkpoint_branch(conflicting),
            Err(ControlError::Conflict(_))
        ));

        let first = cold.list_checkpoint_branches(Some(1), None).unwrap();
        assert_eq!(first.total, 2);
        assert_eq!(first.items.len(), 1);
        assert!(first.has_more);
        let second = cold
            .list_checkpoint_branches(Some(1), first.next_cursor.as_deref())
            .unwrap();
        assert_eq!(second.items.len(), 1);
        assert!(!second.has_more);
        assert_ne!(first.items[0].branch_id, second.items[0].branch_id);
        assert!(
            cold.list_checkpoint_branches(None, Some("not-a-cursor"))
                .is_err()
        );
        assert!(
            cold.list_checkpoint_branches(None, Some(&usize::MAX.to_string()))
                .unwrap()
                .items
                .is_empty()
        );
        assert!(
            cold.data_services
                .checkpoint_branches
                .lock()
                .unwrap()
                .values()
                .all(|job| job.worker.is_none())
        );
        assert_eq!(
            cold.data_services
                .execution_capacity
                .lock()
                .unwrap()
                .reserved_run_slots,
            0
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
    }

    #[test]
    fn managed_checkpoint_branch_catalog_preserves_legacy_output_without_blocking_new_studies() {
        let (handle, host, checkpoint, directory) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 0);
        let legacy_key = "pre-catalog-study";
        let legacy_id = format!("branch-{}", compute_blake3(legacy_key.as_bytes()));
        let legacy = directory.join("branches").join(&legacy_id);
        fs::create_dir_all(&legacy).unwrap();
        fs::write(
            legacy.join("study_report.json"),
            b"retained pre-catalog output",
        )
        .unwrap();
        assert!(handle.get_checkpoint_branch(&legacy_id).is_err());
        assert!(
            handle
                .create_checkpoint_branch(checkpoint_branch_request(
                    &checkpoint.checkpoint_id,
                    legacy_key
                ))
                .is_err()
        );

        let admitted = handle
            .create_checkpoint_branch(checkpoint_branch_request(
                &checkpoint.checkpoint_id,
                "new-catalog-study",
            ))
            .unwrap();
        let completed = wait_for_checkpoint_branch(&handle, &admitted.branch_id);
        assert_eq!(completed.status, "completed");
        assert_eq!(
            handle.list_checkpoint_branches(None, None).unwrap().total,
            1
        );
        handle.data_services.shutdown_experiments().unwrap();
        drop(handle);
        let cold = host.handle().with_artifacts_dir(directory).unwrap();
        assert_eq!(
            cold.list_checkpoint_branches(None, None).unwrap().items,
            vec![completed.clone()]
        );
        assert!(
            cold.read_artifact_bytes(completed.artifact_id.as_deref().unwrap())
                .is_ok()
        );
        assert_eq!(
            fs::read(legacy.join("study_report.json")).unwrap(),
            b"retained pre-catalog output"
        );
        assert!(!legacy.join("definition.json").exists());
    }

    #[test]
    fn managed_checkpoint_branch_without_a_seal_is_stopped_and_never_reexecuted() {
        let (handle, host, checkpoint, directory) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Dwraon), 1);
        let request = checkpoint_branch_request(&checkpoint.checkpoint_id, "interrupted-study");
        let admitted = handle.create_checkpoint_branch(request.clone()).unwrap();
        let completed = wait_for_checkpoint_branch(&handle, &admitted.branch_id);
        assert_eq!(completed.status, "completed");
        handle.data_services.shutdown_experiments().unwrap();
        let branch = directory.join("branches").join(&admitted.branch_id);
        let report_path = branch.join("study/study_report.json");
        let report = fs::read(&report_path).unwrap();
        // Model a crash after output creation but before the terminal rename.
        // Preserve the original seal as evidence rather than deleting output.
        fs::rename(
            branch.join("receipt.json"),
            branch.join("uncommitted-receipt.original"),
        )
        .unwrap();
        drop(handle);
        let before = host.port.scientific_digest_v1().unwrap();
        let cold = host.handle().with_artifacts_dir(directory).unwrap();
        let stopped = cold.get_checkpoint_branch(&admitted.branch_id).unwrap();
        assert_eq!(stopped.status, "stopped");
        assert!(stopped.artifact_id.is_none());
        assert!(
            stopped
                .error_reason
                .as_deref()
                .unwrap()
                .contains("interrupted")
        );
        assert_eq!(cold.create_checkpoint_branch(request).unwrap(), stopped);
        assert_eq!(
            cold.cancel_checkpoint_branch(&admitted.branch_id).unwrap(),
            stopped
        );
        assert!(
            cold.read_artifact_bytes(completed.artifact_id.as_deref().unwrap())
                .is_err()
        );
        assert_eq!(fs::read(report_path).unwrap(), report);
        assert!(!branch.join("receipt.json").exists());
        assert!(
            cold.data_services
                .checkpoint_branches
                .lock()
                .unwrap()
                .values()
                .all(|job| job.worker.is_none())
        );
        assert_eq!(
            cold.data_services
                .execution_capacity
                .lock()
                .unwrap()
                .reserved_run_slots,
            0
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
    }

    #[test]
    fn managed_checkpoint_branch_catalog_refuses_corrupt_or_oversized_identity_records() {
        let (handle, host, checkpoint, directory) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 0);
        let request = checkpoint_branch_request(&checkpoint.checkpoint_id, "catalog-corruption");
        let admitted = handle.create_checkpoint_branch(request.clone()).unwrap();
        assert_eq!(
            wait_for_checkpoint_branch(&handle, &admitted.branch_id).status,
            "completed"
        );
        handle.data_services.shutdown_experiments().unwrap();
        let branch = directory.join("branches").join(&admitted.branch_id);
        let definition_path = branch.join("definition.json");
        let receipt_path = branch.join("receipt.json");
        let definition_bytes = fs::read(&definition_path).unwrap();
        let receipt_bytes = fs::read(&receipt_path).unwrap();
        let before = host.port.scientific_digest_v1().unwrap();
        drop(handle);

        let mut changed: CheckpointBranchDefinitionRecord =
            serde_json::from_slice(&definition_bytes).unwrap();
        changed.definition.request.ticks += 1;
        changed.definition.request_hash =
            compute_blake3(&changed.definition.request.bounded_payload().unwrap());
        changed.checksum = changed.definition.checksum().unwrap();
        for bytes in [
            b"truncated admission".to_vec(),
            serde_json::to_vec(&changed).unwrap(),
            vec![b' '; MAX_CHECKPOINT_BRANCH_CATALOG_BYTES + 1],
        ] {
            fs::write(&definition_path, &bytes).unwrap();
            let cold = host.handle().with_artifacts_dir(directory.clone()).unwrap();
            assert!(cold.get_checkpoint_branch(&admitted.branch_id).is_err());
            assert!(cold.list_checkpoint_branches(None, None).is_err());
            assert!(cold.create_checkpoint_branch(request.clone()).is_err());
            assert_eq!(fs::read(&definition_path).unwrap(), bytes);
            assert_eq!(
                cold.data_services
                    .execution_capacity
                    .lock()
                    .unwrap()
                    .reserved_run_slots,
                0
            );
        }
        fs::write(&definition_path, definition_bytes).unwrap();
        let mut changed: CheckpointBranchReceiptRecord =
            serde_json::from_slice(&receipt_bytes).unwrap();
        changed.receipt.status.source_tick += 1;
        changed.checksum = compute_blake3(&serde_json::to_vec(&changed.receipt).unwrap());
        for bytes in [
            b"truncated receipt".to_vec(),
            serde_json::to_vec(&changed).unwrap(),
            vec![b' '; MAX_CHECKPOINT_BRANCH_CATALOG_BYTES + 1],
        ] {
            fs::write(&receipt_path, &bytes).unwrap();
            let cold = host.handle().with_artifacts_dir(directory.clone()).unwrap();
            assert!(cold.get_checkpoint_branch(&admitted.branch_id).is_err());
            assert!(cold.list_checkpoint_branches(None, None).is_err());
            assert!(cold.create_checkpoint_branch(request.clone()).is_err());
            assert_eq!(fs::read(&receipt_path).unwrap(), bytes);
        }
        fs::write(receipt_path, receipt_bytes).unwrap();
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
    }

    #[test]
    fn managed_checkpoint_branch_recovery_refuses_forged_reports_and_changed_bundle_bytes() {
        let (handle, host, checkpoint, directory) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 0);
        let admitted = handle
            .create_checkpoint_branch(checkpoint_branch_request(
                &checkpoint.checkpoint_id,
                "report-corruption",
            ))
            .unwrap();
        let completed = wait_for_checkpoint_branch(&handle, &admitted.branch_id);
        assert_eq!(completed.status, "completed");
        let artifact_id = completed.artifact_id.as_ref().unwrap().clone();
        handle.data_services.shutdown_experiments().unwrap();
        let branch = directory.join("branches").join(&admitted.branch_id);
        let report_path = branch.join("study/study_report.json");
        let receipt_path = branch.join("receipt.json");
        let report_bytes = fs::read(&report_path).unwrap();
        let receipt_bytes = fs::read(&receipt_path).unwrap();
        let report: crate::branch_study::BranchStudyReport =
            serde_json::from_slice(&report_bytes).unwrap();
        drop(handle);

        let mut forged = vec![report.clone(); 4];
        forged[0].branch_reports.clear();
        forged[1]
            .branch_reports
            .push(report.branch_reports[0].clone());
        forged[2].branch_reports[0].branch_id = "../outside".to_owned();
        if let crate::branch_study::BranchStudySource::SavedCheckpoint { artifact_path, .. } =
            &mut forged[3].source
        {
            *artifact_path = "../outside".to_owned();
        }
        for report in forged {
            let bytes = serde_json::to_vec_pretty(&report).unwrap();
            fs::write(&report_path, &bytes).unwrap();
            // Recompute the outer checksum to reach structural provenance guards,
            // rather than merely proving that a changed file has a different hash.
            let mut receipt: CheckpointBranchReceiptRecord =
                serde_json::from_slice(&receipt_bytes).unwrap();
            let metadata = receipt
                .receipt
                .artifacts
                .iter_mut()
                .find(|artifact| artifact.artifact_id == artifact_id)
                .unwrap();
            metadata.byte_size = bytes.len() as u64;
            metadata.checksum_blake3 = compute_blake3(&bytes);
            metadata.checksum_sha256 = compute_sha256(&bytes);
            receipt.checksum = compute_blake3(&serde_json::to_vec(&receipt.receipt).unwrap());
            fs::write(&receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
            let cold = host.handle().with_artifacts_dir(directory.clone()).unwrap();
            assert!(cold.read_artifact_bytes(&artifact_id).is_err());
            assert!(cold.data_services.artifacts.lock().unwrap().is_empty());
        }
        fs::write(&report_path, report_bytes).unwrap();
        fs::write(receipt_path, receipt_bytes).unwrap();
        let evidence = branch.join("study/bundle_control/evidence/branch_evidence.json");
        let original = fs::read(&evidence).unwrap();
        fs::write(&evidence, b"damaged evidence").unwrap();
        let cold = host.handle().with_artifacts_dir(directory).unwrap();
        // Status is a historical receipt. Download makes the new integrity observation.
        assert_eq!(
            cold.get_checkpoint_branch(&admitted.branch_id)
                .unwrap()
                .status,
            "completed"
        );
        assert!(cold.read_artifact_bytes(&artifact_id).is_err());
        assert!(cold.data_services.artifacts.lock().unwrap().is_empty());
        assert_eq!(fs::read(&evidence).unwrap(), b"damaged evidence");
        fs::write(evidence, original).unwrap();
    }

    #[test]
    fn managed_checkpoint_branch_rejects_self_consistent_bundle_relabelling_and_omissions() {
        let (handle, host, checkpoint, directory) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 0);
        let before = host.port.scientific_digest_v1().unwrap();
        let admitted = handle
            .create_checkpoint_branch(checkpoint_branch_request(
                &checkpoint.checkpoint_id,
                "bundle-semantic-identity",
            ))
            .unwrap();
        let completed = wait_for_checkpoint_branch(&handle, &admitted.branch_id);
        assert_eq!(completed.status, "completed", "{completed:?}");
        let artifact_id = completed.artifact_id.as_ref().unwrap().clone();
        let definition = handle.data_services.checkpoint_branches.lock().unwrap()
            [&admitted.branch_id]
            .definition
            .clone();
        handle.data_services.shutdown_experiments().unwrap();
        let branch = directory.join("branches").join(&admitted.branch_id);
        let output = branch.join("study");
        let bundle = output.join("bundle_control");
        let manifest_path = bundle.join("bundle_manifest.json");
        let receipt_path = branch.join("receipt.json");
        let manifest_bytes = fs::read(&manifest_path).unwrap();
        let receipt_bytes = fs::read(&receipt_path).unwrap();
        let report_bytes = fs::read(output.join("study_report.json")).unwrap();
        let report = serde_json::from_slice(&report_bytes).unwrap();
        let original: scriptbots_storage::RunBundleV1 =
            serde_json::from_slice(&manifest_bytes).unwrap();
        assert!(stage_checkpoint_branch_artifacts(&directory, &output, &report, &definition).is_ok());
        drop(handle);

        for case in [
            "run",
            "study",
            "arm",
            "scenario",
            "target",
            "source_revision",
            "root_seed",
            "missing_trajectory",
            "missing_evidence",
            "wrong_artifact_role",
        ] {
            let mut changed = original.clone();
            match case {
                "run" => {
                    let run_id =
                        scriptbots_runtime::RunId::new(if changed.manifest.run_id.get() == 1 {
                            2
                        } else {
                            1
                        });
                    changed.manifest.run_id = run_id;
                    changed.digests.run_id = run_id.to_string();
                    let mut manifest: Value =
                        serde_json::from_str(&changed.manifest.manifest_json).unwrap();
                    manifest["run_id"] = json!(run_id);
                    changed.manifest.manifest_json = serde_json::to_string(&manifest).unwrap();
                }
                "study" => changed.manifest.experiment_id = Some("another-study".to_owned()),
                "arm" => changed.manifest.variant_id = Some("another-arm".to_owned()),
                "scenario" => changed.manifest.scenario_id = "another-scenario".to_owned(),
                "target" => changed.digests.max_tick += 1,
                "source_revision" => {
                    changed.manifest.source_revision = Some("another-source".to_owned());
                    changed.digests.source_revision = changed.manifest.source_revision.clone();
                }
                "root_seed" => changed.manifest.root_seed = changed.manifest.root_seed.wrapping_add(1),
                "missing_trajectory" => changed
                    .artifacts
                    .retain(|artifact| artifact.relative_path != "exports/summary.csv"),
                "missing_evidence" => changed
                    .artifacts
                    .retain(|artifact| artifact.relative_path != "evidence/branch_evidence.json"),
                "wrong_artifact_role" => changed.artifacts[0].artifact_type = "evidence".to_owned(),
                _ => unreachable!(),
            }
            let bytes = serde_json::to_vec_pretty(&changed).unwrap();
            fs::write(&manifest_path, &bytes).unwrap();
            // Each mutation remains a valid, internally consistent artifact-only
            // bundle. The missing proof is its join to this admitted scientific arm.
            scriptbots_storage::bundle::verify_run_bundle_bounded(
                &bundle,
                crate::branch_study::STUDY_BUNDLE_VERIFICATION_LIMITS,
            )
            .unwrap_or_else(|error| panic!("{case} failed generic integrity: {error}"));
            assert!(
                matches!(
                    stage_checkpoint_branch_artifacts(&directory, &output, &report, &definition),
                    Err(ControlError::InvalidPatch(reason))
                        if reason.contains("bundle identity or artifact roster")
                ),
                "{case} escaped publication identity validation"
            );

            // Re-seal the exact changed manifest and roster to get beyond outer
            // receipt checksums on cold recovery. No artifact bytes are changed.
            let mut receipt: CheckpointBranchReceiptRecord =
                serde_json::from_slice(&receipt_bytes).unwrap();
            let prefix = format!("branches/{}/study/bundle_control/", admitted.branch_id);
            receipt.receipt.artifacts.retain(|artifact| {
                artifact
                    .relative_path
                    .strip_prefix(&prefix)
                    .is_none_or(|relative| {
                        relative == "bundle_manifest.json"
                            || changed
                                .artifacts
                                .iter()
                                .any(|entry| entry.relative_path == relative)
                    })
            });
            let metadata = receipt
                .receipt
                .artifacts
                .iter_mut()
                .find(|artifact| artifact.relative_path == format!("{prefix}bundle_manifest.json"))
                .unwrap();
            metadata.byte_size = bytes.len() as u64;
            metadata.checksum_blake3 = compute_blake3(&bytes);
            metadata.checksum_sha256 = compute_sha256(&bytes);
            receipt.receipt.validate(&definition).unwrap();
            receipt.checksum = compute_blake3(&serde_json::to_vec(&receipt.receipt).unwrap());
            fs::write(&receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
            let cold = host.handle().with_artifacts_dir(directory.clone()).unwrap();
            assert_eq!(
                cold.get_checkpoint_branch(&admitted.branch_id).unwrap().status,
                "completed",
                "{case}: history remains a terminal observation"
            );
            assert!(cold.read_artifact_bytes(&artifact_id).is_err(), "{case}");
            assert!(
                cold.data_services.artifacts.lock().unwrap().is_empty(),
                "{case}"
            );
            assert_eq!(fs::read(&manifest_path).unwrap(), bytes, "{case}");
        }
        fs::write(&manifest_path, manifest_bytes).unwrap();
        fs::write(&receipt_path, receipt_bytes).unwrap();
        let cold = host.handle().with_artifacts_dir(directory.clone()).unwrap();
        assert_eq!(
            cold.read_artifact_bytes(&artifact_id).unwrap().1,
            report_bytes
        );
        assert!(stage_checkpoint_branch_artifacts(&directory, &output, &report, &definition).is_ok());
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
    }

    #[test]
    fn managed_checkpoint_branch_catalog_refuses_ambiguous_roots_and_global_capacity_overflow() {
        let (handle, host, checkpoint, directory) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 0);
        let admitted = handle
            .create_checkpoint_branch(checkpoint_branch_request(
                &checkpoint.checkpoint_id,
                "root-conflict",
            ))
            .unwrap();
        let completed = wait_for_checkpoint_branch(&handle, &admitted.branch_id);
        assert_eq!(completed.status, "completed");
        let artifact_id = completed.artifact_id.unwrap();
        handle.read_artifact_bytes(&artifact_id).unwrap();
        handle.data_services.shutdown_experiments().unwrap();
        let other = tempfile::tempdir().unwrap().keep();
        fs::create_dir_all(other.join("branches").join(&admitted.branch_id)).unwrap();
        let handle = handle.with_artifacts_dir(other).unwrap();
        assert!(matches!(
            handle.get_checkpoint_branch(&admitted.branch_id),
            Err(ControlError::Conflict(_))
        ));
        assert!(matches!(
            handle.read_artifact_bytes(&artifact_id),
            Err(ControlError::Conflict(_))
        ));
        handle.data_services.artifacts.lock().unwrap().clear();
        assert!(matches!(
            handle.read_artifact_bytes(&artifact_id),
            Err(ControlError::Conflict(_))
        ));
        assert!(
            directory
                .join("branches")
                .join(&admitted.branch_id)
                .join("study/study_report.json")
                .is_file()
        );
        drop(handle);

        let first = tempfile::tempdir().unwrap().keep();
        let second = tempfile::tempdir().unwrap().keep();
        for (root, count) in [(&first, 17), (&second, 16)] {
            fs::create_dir(root.join("branches")).unwrap();
            for index in 0..count {
                fs::create_dir(root.join("branches").join(format!("interrupted-{index}"))).unwrap();
            }
        }
        let handle = host
            .handle()
            .with_artifacts_dir(first)
            .unwrap()
            .with_artifacts_dir(second)
            .unwrap();
        assert!(matches!(
            handle.list_checkpoint_branches(None, None),
            Err(ControlError::ExperimentCapacity(_))
        ));
        assert!(
            handle
                .data_services
                .checkpoint_branches
                .lock()
                .unwrap()
                .is_empty()
        );
    }

    #[cfg(unix)]
    #[test]
    fn managed_checkpoint_branch_catalog_rejects_symlinked_receipts() {
        let (handle, host, checkpoint, directory) =
            checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 0);
        let admitted = handle
            .create_checkpoint_branch(checkpoint_branch_request(
                &checkpoint.checkpoint_id,
                "receipt-alias",
            ))
            .unwrap();
        assert_eq!(
            wait_for_checkpoint_branch(&handle, &admitted.branch_id).status,
            "completed"
        );
        handle.data_services.shutdown_experiments().unwrap();
        let branch = directory.join("branches").join(&admitted.branch_id);
        fs::rename(branch.join("receipt.json"), branch.join("receipt.original")).unwrap();
        std::os::unix::fs::symlink("receipt.original", branch.join("receipt.json")).unwrap();
        drop(handle);
        let cold = host.handle().with_artifacts_dir(directory).unwrap();
        assert!(cold.get_checkpoint_branch(&admitted.branch_id).is_err());
        assert!(cold.list_checkpoint_branches(None, None).is_err());
        assert!(
            fs::symlink_metadata(branch.join("receipt.json"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[test]
    fn managed_checkpoint_branch_cancellation_shutdown_and_drop_join_workers() {
        for explicit_shutdown in [true, false] {
            let (handle, host, checkpoint, _) =
                checkpoint_branch_fixture(Some(crate::BrainPreset::Mlp), 0);
            let before = host.port.scientific_digest_v1().unwrap();
            let mut request =
                checkpoint_branch_request(&checkpoint.checkpoint_id, "cancel-and-join");
            request.ticks = MAX_CHECKPOINT_BRANCH_TICKS;
            let admitted = handle.create_checkpoint_branch(request.clone()).unwrap();
            let (state, capacity) = {
                let jobs = handle.data_services.checkpoint_branches.lock().unwrap();
                (
                    Arc::clone(&jobs[&admitted.branch_id].status),
                    Arc::clone(&handle.data_services.execution_capacity),
                )
            };
            let weak_services = Arc::downgrade(&handle.data_services);
            if explicit_shutdown {
                let cancelled = handle
                    .cancel_checkpoint_branch(&admitted.branch_id)
                    .unwrap();
                assert!(matches!(
                    cancelled.status.as_str(),
                    "cancelling" | "cancelled" | "completed"
                ));
                handle.data_services.shutdown_experiments().unwrap();
                assert!(
                    handle
                        .data_services
                        .checkpoint_branches
                        .lock()
                        .unwrap()
                        .values()
                        .all(|job| job.worker.is_none())
                );
                let terminal = handle.get_checkpoint_branch(&admitted.branch_id).unwrap();
                assert!(matches!(
                    terminal.status.as_str(),
                    "cancelled" | "completed"
                ));
                if terminal.status == "cancelled" {
                    assert!(terminal.artifact_id.is_none());
                    assert!(matches!(
                        handle.list_checkpoint_branch_artifacts(&admitted.branch_id, None, None),
                        Err(ControlError::Conflict(_))
                    ));
                }
                assert_eq!(
                    handle
                        .create_checkpoint_branch(request.clone())
                        .unwrap()
                        .branch_id,
                    admitted.branch_id
                );
                request.idempotency_key = "new-after-shutdown".to_owned();
                assert!(matches!(
                    handle.create_checkpoint_branch(request),
                    Err(ControlError::ExperimentShuttingDown)
                ));
            }
            drop(handle);
            assert!(
                weak_services.upgrade().is_none(),
                "workers must not keep their owning service alive"
            );
            assert!(matches!(
                state.lock().unwrap().status.as_str(),
                "cancelled" | "completed"
            ));
            assert_eq!(capacity.lock().unwrap().reserved_run_slots, 0);
            assert_eq!(capacity.lock().unwrap().reserved_grid_cells, 0);
            assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
        }
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one actual auxiliary-science owner proves full capture, exact replay, immutable retries and different-owner cold recovery"
    )]
    fn full_checkpoint_catalog_preserves_auxiliary_science_and_cold_recovery() {
        let directory = tempfile::tempdir().unwrap().keep();
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(42),
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            persistence_interval: 0,
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            reproduction_energy_threshold: f32::MAX,
            spike_damage: 0.0,
            archive_enabled: true,
            archive_interval: 1,
            archive_min_lifetime_ticks: 1,
            selection_mode: scriptbots_core::EvolutionSelectionMode::Novelty,
            communication_scramble: true,
            communication_scramble_seed: Some(0x6361_7461),
            communication_mi_interval: 1,
            communication_mi_window: 4,
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        let width = world.terrain().width();
        let height = world.terrain().height();
        let map = scriptbots_core::MapArtifact::new(
            world.terrain().clone(),
            Some(
                scriptbots_core::ScalarField::new(
                    width,
                    height,
                    vec![0.25; world.terrain().tiles().len()],
                )
                .unwrap(),
            ),
            Some(
                scriptbots_core::ScalarField::new(
                    width,
                    height,
                    vec![0.75; world.terrain().tiles().len()],
                )
                .unwrap(),
            ),
            None,
            None,
            scriptbots_core::MapArtifactMetadata {
                generator: scriptbots_core::MapGeneratorKind::RuleBased,
                tileset_id: "full-catalog-auxiliary-science".to_owned(),
                tileset_hash: 1,
                seed: 42,
                width,
                height,
                attempt_count: 1,
                succeeded_on: 1,
                generated_at_epoch_ms: 0,
            },
        )
        .unwrap();
        world.apply_map_artifact(&map).unwrap();
        let brains = crate::install_brains(&mut world, crate::BrainPreset::Mlp).unwrap();
        for _ in 0..2 {
            let agent = world
                .try_spawn_agent(scriptbots_core::AgentData {
                    position: scriptbots_core::Position::new(40.0, 40.0),
                    ..scriptbots_core::AgentData::default()
                })
                .unwrap();
            assert!(
                world
                    .bind_agent_brain(agent, brains.population()[0])
                    .unwrap()
            );
        }
        let host = TestHost::spawn_with_schedule(
            world,
            Some(vec![
                scriptbots_runtime::ScheduledConfigPatch::new(
                    1,
                    Tick(4),
                    serde_json::json!({"food_growth_rate": 0.017}),
                )
                .unwrap(),
            ]),
        );
        let handle = host.handle().with_artifacts_dir(directory.clone()).unwrap();
        for _ in 0..2 {
            host.wait_applied(&handle.step().unwrap());
        }
        let saved_snapshot = host.port.snapshot_hub().latest();
        let saved_digest = host.port.scientific_digest_v1().unwrap();
        let request = CheckpointCreateRequest {
            description: Some("full auxiliary-science cut".to_owned()),
            idempotency_key: Some("full-catalog-cut".to_owned()),
        };
        let saved = handle.create_checkpoint(request.clone()).unwrap();
        assert_eq!(saved.schema, scriptbots_runtime::HOST_CHECKPOINT_V1_SCHEMA);
        assert_eq!(saved.tick, 2);
        assert_eq!(saved.agent_count, 2);
        assert_eq!(saved.world_digest_v1, saved_digest.overall);
        let (artifact, bytes) = handle.read_artifact_bytes(&saved.checkpoint_id).unwrap();
        let owner = scriptbots_runtime::HostCheckpointV1::decode(&bytes).unwrap();
        assert_eq!(owner.session_id(), handle.session_id());
        assert_eq!(owner.remaining_schedule().unwrap().len(), 1);
        assert!(owner.validate_standalone_projection().is_err());
        let source = CatalogCheckpoint::decode(&saved.schema, &bytes).unwrap();
        let mut detached = ControlHandle::restore_catalog_world(&source).unwrap();
        assert_eq!(detached.world_digest_v1().unwrap(), saved_digest);
        assert!(detached.temperature().is_some());

        host.wait_applied(&handle.step().unwrap());
        let target_snapshot = host.port.snapshot_hub().latest();
        let target_digest = host.port.scientific_digest_v1().unwrap();
        detached.step().unwrap();
        assert_eq!(detached.world_digest_v1().unwrap(), target_digest);
        for _ in 0..2 {
            host.wait_applied(&handle.step().unwrap());
        }
        let live = host.port.scientific_digest_cut().unwrap();
        let publication = host.port.snapshot_hub().latest();
        assert_eq!(live.digest.tick.0, 5);
        assert_eq!(publication.scheduled_patches.len(), 1);
        let retained = handle
            .data_services
            .checkpoint_retention
            .lock()
            .unwrap()
            .requests
            .values()
            .next()
            .unwrap()
            .clone();
        let audit =
            ControlHandle::event_review_config_audit(&retained, &live, &publication).unwrap();
        let target = ControlHandle::replay_event_canvas_world_restored(
            ControlHandle::restore_catalog_world(&source).unwrap(),
            source.science(),
            &live.digest,
            &audit,
            &[],
            3,
            retained.source_persistence_interval,
            retained.source_config_revision,
            None,
        )
        .unwrap();
        assert_eq!(target.world_digest_v1().unwrap(), target_digest);

        // Slot handles and the prior-tick death counter are not scientific identity.
        // Compare every remaining projected field against the real owner observations.
        let logical = |mut value: Value| {
            for agent in value["agents"].as_array_mut().unwrap() {
                agent.as_object_mut().unwrap().remove("id").unwrap();
            }
            value["summary"]
                .as_object_mut()
                .unwrap()
                .remove("deaths")
                .unwrap();
            value
        };
        let saved_view = handle.checkpoint_canvas(&saved.checkpoint_id).unwrap();
        assert_eq!(
            logical(serde_json::to_value(&saved_view.world).unwrap()),
            logical(serde_json::to_value(&saved_snapshot.world).unwrap())
        );
        let selected = lab_checkpoint_descriptor(&saved);
        let native = handle
            .replay_canvas_from_checkpoint(handle.session_id(), &selected, 3)
            .unwrap();
        let (_, remote) = handle
            .replay_canvas_from_checkpoint_bounded(handle.session_id(), &saved.checkpoint_id, 3)
            .unwrap();
        let event = handle.event_canvas(3).unwrap();
        for review in [&native, &remote, &event] {
            assert_eq!(
                logical(serde_json::to_value(&review.snapshot.world).unwrap()),
                logical(serde_json::to_value(&target_snapshot.world).unwrap())
            );
            assert_eq!(
                review.verification,
                scriptbots_runtime::lab_workflows::LabReplayVerification::CapturedInterval {
                    verified_through_tick: 5,
                    captured_current_digest: live.digest.overall.clone(),
                }
            );
        }
        assert_eq!(handle.create_checkpoint(request.clone()).unwrap(), saved);
        let mut changed = request.clone();
        changed.description = Some("different request".to_owned());
        assert!(matches!(
            handle.create_checkpoint(changed),
            Err(ControlError::Conflict(_))
        ));
        let mut false_schema = saved.clone();
        false_schema.schema = scriptbots_core::WORLD_CHECKPOINT_V1_SCHEMA.to_owned();
        assert!(ControlHandle::decode_canvas_checkpoint(&false_schema, &artifact, &bytes).is_err());
        let branch_request =
            checkpoint_branch_request(&saved.checkpoint_id, "full-source-independent-branch");
        let admitted = handle
            .create_checkpoint_branch(branch_request.clone())
            .unwrap();
        let completed = wait_for_checkpoint_branch(&handle, &admitted.branch_id);
        assert_eq!(completed.status, "completed", "{completed:?}");
        assert_eq!(completed.source_tick, saved.tick);
        assert_eq!(completed.target_tick, saved.tick + branch_request.ticks);
        let (_, report_bytes) = handle
            .read_artifact_bytes(completed.artifact_id.as_deref().unwrap())
            .unwrap();
        let report: crate::branch_study::BranchStudyReport =
            serde_json::from_slice(&report_bytes).unwrap();
        assert_eq!(report.pre_intervention_digest, saved.world_digest_v1);
        assert!(report.pre_intervention_digest_verified_for_all_branches);
        assert_eq!(report.branch_reports.len(), 1);
        let mut future = ControlHandle::restore_catalog_world(&source).unwrap();
        for _ in 0..branch_request.ticks {
            future.step().unwrap();
        }
        assert_eq!(
            report.branch_reports[0].final_digest,
            future.world_digest_v1().unwrap().overall
        );
        assert_ne!(
            report.branch_reports[0].final_digest, live.digest.overall,
            "the original owner's pending schedule must not become a fork treatment"
        );
        let study_output = directory
            .join("branches")
            .join(&admitted.branch_id)
            .join("study");
        let retained_source = study_output.join("source_checkpoint.bin");
        assert_eq!(fs::read(&retained_source).unwrap(), bytes);
        assert_eq!(
            report.source,
            crate::branch_study::BranchStudySource::SavedCheckpoint {
                artifact_path: "source_checkpoint.bin".to_owned(),
                byte_size: bytes.len() as u64,
                checksum_blake3: compute_blake3(&bytes),
            }
        );
        let retained_plan: crate::branch_study::BranchStudyPlan =
            serde_json::from_slice(&fs::read(study_output.join("study_plan.json")).unwrap())
                .unwrap();
        assert_eq!(
            retained_plan.source_checkpoint.as_ref(),
            Some(&retained_source)
        );
        let offline_output = directory.join("full-source-offline-reproduced");
        let runner = crate::branch_study::BranchStudyOrchestrator::new(
            retained_plan,
            offline_output.clone(),
        );
        let reproduced = std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || runner.execute_study(None))
            .unwrap()
            .join()
            .unwrap()
            .unwrap();
        assert_eq!(reproduced.source, report.source);
        assert_eq!(
            reproduced.branch_reports[0].terminal_sample,
            report.branch_reports[0].terminal_sample
        );
        assert_eq!(
            fs::read(offline_output.join("source_checkpoint.bin")).unwrap(),
            bytes
        );
        assert_eq!(
            handle
                .create_checkpoint_branch(branch_request.clone())
                .unwrap(),
            completed
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), live.digest);
        assert!(Arc::ptr_eq(
            &host.port.snapshot_hub().latest(),
            &publication
        ));
        assert_eq!(
            handle.read_artifact_bytes(&saved.checkpoint_id).unwrap().1,
            bytes
        );
        drop(handle);
        drop(host);

        let cold_owner = TestHost::spawn_with_identity(
            WorldState::new(ScriptBotsConfig {
                rng_seed: Some(59),
                world_width: 120,
                world_height: 120,
                food_cell_size: 20,
                persistence_interval: 0,
                closed: true,
                population_minimum: 0,
                population_spawn_interval: 0,
                ..ScriptBotsConfig::default()
            })
            .unwrap(),
            None,
            scriptbots_runtime::HostSessionId::new(0xc018),
        );
        let cold = cold_owner.handle().with_artifacts_dir(directory).unwrap();
        let cold_digest = cold_owner.port.scientific_digest_v1().unwrap();
        let cold_publication = cold_owner.port.snapshot_hub().latest();
        assert_eq!(cold.create_checkpoint(request).unwrap(), saved);
        assert_eq!(
            cold.read_artifact_bytes(&saved.checkpoint_id).unwrap().1,
            bytes
        );
        assert_eq!(
            cold.get_checkpoint_branch(&admitted.branch_id).unwrap(),
            completed
        );
        assert_eq!(
            cold.create_checkpoint_branch(branch_request).unwrap(),
            completed
        );
        assert_eq!(
            cold.read_artifact_bytes(completed.artifact_id.as_deref().unwrap())
                .unwrap()
                .1,
            report_bytes
        );
        assert_eq!(
            cold.data_services
                .execution_capacity
                .lock()
                .unwrap()
                .reserved_run_slots,
            0,
            "cold receipt recovery must never launch scientific work"
        );
        assert!(
            cold.data_services
                .checkpoint_branches
                .lock()
                .unwrap()
                .get(&admitted.branch_id)
                .unwrap()
                .worker
                .is_none()
        );
        let (reopened, view) = cold
            .checkpoint_canvas_bounded(&saved.checkpoint_id)
            .unwrap();
        assert_eq!(reopened, saved);
        assert_eq!(
            logical(serde_json::to_value(&view.world).unwrap()),
            logical(serde_json::to_value(&saved_snapshot.world).unwrap())
        );
        assert!(matches!(
            cold.replay_canvas_from_checkpoint_bounded(cold.session_id(), &saved.checkpoint_id, 3),
            Err(ControlError::BadRequest(reason))
                if reason.contains("no acknowledged current-owner replay history")
        ));
        assert!(
            cold.data_services
                .checkpoint_retention
                .lock()
                .unwrap()
                .requests
                .is_empty()
        );
        assert_eq!(cold_owner.port.scientific_digest_v1().unwrap(), cold_digest);
        assert!(Arc::ptr_eq(
            &cold_owner.port.snapshot_hub().latest(),
            &cold_publication
        ));
        let branch_source_artifact = cold
            .list_artifacts(Some(100), None)
            .unwrap()
            .items
            .into_iter()
            .find(|artifact| artifact.filename == "source_checkpoint.bin")
            .unwrap();
        assert_eq!(
            cold.read_artifact_bytes(&branch_source_artifact.artifact_id)
                .unwrap()
                .1,
            bytes
        );
        let mut corrupted_source = bytes.clone();
        *corrupted_source.last_mut().unwrap() ^= 1;
        fs::write(&retained_source, &corrupted_source).unwrap();
        assert!(
            cold.read_artifact_bytes(&branch_source_artifact.artifact_id)
                .is_err()
        );
        assert_eq!(
            cold.read_artifact_bytes(&saved.checkpoint_id).unwrap().1,
            bytes
        );
        assert_eq!(cold_owner.port.scientific_digest_v1().unwrap(), cold_digest);
    }

    #[test]
    fn checkpoint_catalog_keeps_fitness_archive_and_disabled_feature_history() {
        let directory = tempfile::tempdir().unwrap().keep();
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(42),
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            persistence_interval: 0,
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            reproduction_energy_threshold: f32::MAX,
            archive_enabled: true,
            archive_interval: 1,
            archive_min_lifetime_ticks: 1,
            selection_mode: scriptbots_core::EvolutionSelectionMode::Fitness,
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        let brains = crate::install_brains(&mut world, crate::BrainPreset::Mlp).unwrap();
        let agent = world
            .try_spawn_agent(scriptbots_core::AgentData::default())
            .unwrap();
        assert!(
            world
                .bind_agent_brain(agent, brains.population()[0])
                .unwrap()
        );
        world.step().unwrap();
        world.step().unwrap();
        assert!(world.archive().unwrap().coverage_count() > 0);
        let statistics = world.agent_stats().clone();
        assert_eq!(statistics.len(), 1);
        assert_eq!(statistics.values().next().unwrap().ticks_observed, 2);
        let host = TestHost::spawn(world);
        let handle = host.handle().with_artifacts_dir(directory).unwrap();
        for enabled in [true, false] {
            if !enabled {
                host.wait_applied(
                    &handle
                        .apply_patch(serde_json::json!({"archive_enabled": false}))
                        .unwrap(),
                );
            }
            let saved = handle
                .create_checkpoint(CheckpointCreateRequest {
                    description: Some(format!("archive enabled: {enabled}")),
                    idempotency_key: Some(format!("archive-history-{enabled}")),
                })
                .unwrap();
            assert_eq!(saved.schema, scriptbots_runtime::HOST_CHECKPOINT_V1_SCHEMA);
            let (_, bytes) = handle.read_artifact_bytes(&saved.checkpoint_id).unwrap();
            let owner = scriptbots_runtime::HostCheckpointV1::decode(&bytes).unwrap();
            assert_eq!(owner.config().archive_enabled, enabled);
            assert!(matches!(
                owner.validate_standalone_projection(),
                Err(
                    scriptbots_core::WorldCheckpointError::RequiresHostContinuity {
                        feature: "quality-diversity history"
                    }
                )
            ));
            if !enabled {
                owner
                    .world_checkpoint()
                    .validate_standalone_restore()
                    .unwrap();
            }
            let source = CatalogCheckpoint::decode(&saved.schema, &bytes).unwrap();
            let restored = ControlHandle::restore_catalog_world(&source).unwrap();
            assert_eq!(restored.agent_stats(), &statistics);
            assert_eq!(restored.archive().is_some(), enabled);
            if enabled {
                assert!(restored.archive().unwrap().coverage_count() > 0);
            }
            let current = host.port.scientific_digest_v1().unwrap();
            assert_eq!(restored.world_digest_v1().unwrap(), current);
            assert_eq!(
                handle
                    .checkpoint_canvas_bounded(&saved.checkpoint_id)
                    .unwrap()
                    .1
                    .world
                    .tick,
                saved.tick
            );
            assert_eq!(host.port.scientific_digest_v1().unwrap(), current);
        }
    }

    #[test]
    fn full_checkpoint_capture_refusal_publishes_no_catalog_or_request() {
        struct ExternalSenseProvider;

        impl scriptbots_core::SenseProvider for ExternalSenseProvider {
            fn compute_accumulators(
                &mut self,
                _world: &WorldState,
                handles: &[scriptbots_core::AgentId],
            ) -> Result<
                Vec<scriptbots_core::sense_fixed::SenseAccum>,
                scriptbots_core::SenseProviderError,
            > {
                Ok(handles.iter().map(|_| Default::default()).collect())
            }
        }

        let directory = tempfile::tempdir().unwrap().keep();
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(42),
            world_width: 40,
            world_height: 40,
            food_cell_size: 10,
            persistence_interval: 0,
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            archive_enabled: true,
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        world.set_sense_provider(Box::new(ExternalSenseProvider));
        let host = TestHost::spawn(world);
        let handle = host.handle().with_artifacts_dir(directory.clone()).unwrap();
        let publication = host.port.snapshot_hub().latest();
        let request = CheckpointCreateRequest {
            description: None,
            idempotency_key: Some("full-capture-refusal".into()),
        };
        for _ in 0..2 {
            assert!(matches!(
                handle.create_checkpoint(request.clone()),
                Err(ControlError::Host(error)) if error.to_string().contains("host.sense_provider")
            ));
            let usage = handle.checkpoint_retention_usage().unwrap();
            assert_eq!(usage.file_count, 0);
            assert_eq!(usage.file_bytes, 0);
            assert_eq!(usage.request_entries, 0);
            assert_eq!(usage.pending_entries, 0);
            assert_eq!(handle.list_checkpoints(None, None).unwrap().total, 0);
            assert!(fs::read_dir(&directory).unwrap().next().is_none());
            assert!(Arc::ptr_eq(
                &host.port.snapshot_hub().latest(),
                &publication
            ));
        }
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one file-backed owner proves full catalog dispatch, durable storage identity, and exact retry after the owner advances"
    )]
    fn full_checkpoint_catalog_uses_the_matching_file_backed_owner() -> anyhow::Result<()> {
        use crate::host_thread::{HostBootstrap, HostThread};
        use scriptbots_runtime::{
            ApplicationState, CommandEnvelope, CommandId, HostCommand, HostCoreOptions, HostPort,
            HostSessionId, JournalState, PlaybackSnapshot, ScheduledConfigPatch,
            channel::ChannelHostOptions,
        };
        use scriptbots_storage::{StoragePipeline, StorageReader};
        use std::time::Duration;

        fn submit_durable(
            port: &mut impl HostPort,
            id: u128,
            command: HostCommand,
        ) -> anyhow::Result<()> {
            let id = CommandId::new(id);
            port.submit(CommandEnvelope::new(id, command))?;
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            loop {
                let status = port.command_status(id)?.expect("submitted fixture command");
                if status.journal() == &JournalState::Durable {
                    anyhow::ensure!(
                        matches!(status.application(), ApplicationState::Applied(_)),
                        "fixture command did not apply: {status:?}"
                    );
                    return Ok(());
                }
                anyhow::ensure!(
                    std::time::Instant::now() < deadline,
                    "fixture command did not settle: {status:?}"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
        }

        let root = tempfile::tempdir()?.keep();
        let database = root.join("full-catalog.sqlite");
        let path = database.to_str().expect("fixture path");
        let mut pipeline = StoragePipeline::create_unattributed_file(path)?;
        let run_id = pipeline.run_id();
        let config = ScriptBotsConfig {
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            rng_seed: Some(42_104),
            closed: true,
            persistence_interval: 3,
            population_minimum: 0,
            population_spawn_interval: 0,
            reproduction_energy_threshold: f32::MAX,
            spike_damage: 0.0,
            archive_enabled: true,
            archive_interval: 1,
            archive_min_lifetime_ticks: 1,
            selection_mode: scriptbots_core::EvolutionSelectionMode::Novelty,
            ..ScriptBotsConfig::default()
        };
        let (mut world, persistence) =
            WorldState::with_persistence(config, Box::new(pipeline.sink()))?;
        let brains = crate::install_brains(&mut world, crate::BrainPreset::Mlp)?;
        let agent = world.try_spawn_agent(scriptbots_core::AgentData {
            position: scriptbots_core::Position::new(40.0, 40.0),
            ..scriptbots_core::AgentData::default()
        })?;
        assert!(world.bind_agent_brain(agent, brains.population()[0])?);
        let session = HostSessionId::new(0xc018_f11e);
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
                        Tick(3),
                        serde_json::json!({"food_growth_rate": 0.017}),
                    )
                    .map_err(anyhow::Error::msg)?,
                ],
                ..HostBootstrap::default()
            },
        )?;
        let mut port = host.port();
        let handle = ControlHandle::new(port.clone())
            .with_artifacts_dir(root.join("artifacts"))?
            .with_checkpoint_writer(Some(pipeline.checkpoint_writer()));
        submit_durable(&mut port, 1, HostCommand::Step)?;
        let request = CheckpointCreateRequest {
            idempotency_key: Some("full-file-owner".to_owned()),
            description: Some("same durable owner and storage cut".to_owned()),
        };
        // Save uses the existing owner capture boundary to settle this cadence tail.
        let saved = handle.create_checkpoint(request.clone())?;
        let observed = port.scientific_digest_v1()?;
        assert_eq!(saved.schema, scriptbots_runtime::HOST_CHECKPOINT_V1_SCHEMA);
        assert_eq!(saved.tick, 1);
        assert_eq!(saved.agent_count, 1);
        assert_eq!(saved.world_digest_v1, observed.overall);
        let bytes = handle.read_artifact_bytes(&saved.checkpoint_id)?.1;
        let captured = scriptbots_runtime::HostCheckpointV1::decode(&bytes)?;
        assert_eq!(captured.session_id(), session);
        assert_eq!(captured.config().persistence_interval, 3);
        assert_eq!(captured.world_checkpoint().source_digest(), &observed);
        assert!(captured.validate_standalone_projection().is_err());
        assert_eq!(captured.remaining_schedule().unwrap().len(), 1);
        assert_eq!(captured.persistence_last_admitted_tick(), Some(Tick(1)));
        assert!(captured.journal_cut().last_journal_sequence > 0);

        submit_durable(&mut port, 2, HostCommand::Step)?;
        submit_durable(&mut port, 3, HostCommand::FlushPersistence)?;
        let live = port.scientific_digest_v1()?;
        assert_eq!(live.tick.0, 2);
        assert_eq!(handle.create_checkpoint(request)?, saved);
        assert_eq!(handle.read_artifact_bytes(&saved.checkpoint_id)?.1, bytes);
        assert_eq!(port.scientific_digest_v1()?, live);
        let usage = handle.checkpoint_retention_usage()?;
        assert_eq!(usage.file_count, 1);
        assert_eq!(usage.sql_count, Some(1));
        assert_eq!(usage.pending_entries, 0);
        drop(handle);
        port.submit(CommandEnvelope::new(
            CommandId::new(4),
            HostCommand::Shutdown,
        ))?;
        drop(port);
        let receipt = host.join()?;
        assert_eq!(receipt.final_digest?, live);
        pipeline.shutdown()?;

        let reader = StorageReader::open_finished(path)?;
        assert_eq!(reader.host_journal_sessions()?, vec![session]);
        assert_eq!(reader.summary_ticks()?, vec![1, 2]);
        let row = reader
            .load_checkpoint(&saved.checkpoint_id)?
            .expect("catalog SQL row");
        assert_eq!(row.checkpoint_ordinal, 0);
        assert_eq!(row.tick, saved.tick);
        assert_eq!(row.format, format!("{}+postcard_hex", saved.schema));
        assert_eq!(
            row.payload_digest,
            format!("blake3:{}", saved.checksum_blake3)
        );
        assert_eq!(row.host_checkpoint()?.encode()?, bytes);
        let metadata: serde_json::Value = serde_json::from_str(&row.metadata_json)?;
        assert_eq!(metadata["source"], "control_api");
        let cut = &metadata["host_storage_cut"];
        assert_eq!(cut["schema_version"], 1);
        assert_eq!(cut["run_id"], run_id.to_string());
        assert_eq!(cut["last_admitted_tick"], 1);
        assert!(!cut["admitted"].is_null());
        assert_eq!(cut["admitted"], cut["applied"]);
        assert_eq!(cut["applied"], cut["durable"]);
        reader.close()?;
        Ok(())
    }

    #[test]
    fn checkpoint_canvas_restores_real_application_presets_without_changing_the_owner() {
        let mut distinct_digests = std::collections::BTreeSet::new();
        for preset in [
            None,
            Some(crate::BrainPreset::Mlp),
            Some(crate::BrainPreset::Dwraon),
            Some(crate::BrainPreset::Assembly),
            Some(crate::BrainPreset::Mixed),
            #[cfg(feature = "neuro")]
            Some(crate::BrainPreset::Neuro),
        ] {
            let directory = tempfile::Builder::new()
                .prefix("checkpoint-canvas-")
                .tempdir()
                .unwrap()
                .keep();
            eprintln!(
                "checkpoint canvas preset {preset:?}: {}",
                directory.display()
            );
            let mut config = ScriptBotsConfig {
                rng_seed: Some(42),
                world_width: 120,
                world_height: 120,
                food_cell_size: 20,
                persistence_interval: 0,
                closed: true,
                ..ScriptBotsConfig::default()
            };
            if preset == Some(crate::BrainPreset::Neuro) {
                config.neuroflow.hidden_layers = vec![7, 3];
                config.neuroflow.activation = scriptbots_core::NeuroflowActivationKind::Sigmoid;
            }
            let mut world = WorldState::new(config).unwrap();
            if let Some(preset) = preset {
                let brains = crate::install_brains(&mut world, preset)
                    .unwrap()
                    .population;
                crate::seed_founders(&mut world, &brains, crate::FounderRecipe::Spread).unwrap();
            }
            let host = TestHost::spawn(world);
            let handle = host.handle().with_artifacts_dir(directory.clone()).unwrap();
            host.wait_applied(&handle.step().unwrap());
            let oracle = host.port.snapshot_hub().latest();
            let checkpoint = handle
                .create_checkpoint(CheckpointCreateRequest::default())
                .unwrap();
            assert!(
                distinct_digests.insert(checkpoint.world_digest_v1.clone()),
                "preset matrix must contain distinct science states"
            );
            let (artifact, bytes) = handle
                .read_artifact_bytes(&checkpoint.checkpoint_id)
                .unwrap();
            host.wait_applied(&handle.step().unwrap());
            let live = host.port.scientific_digest_v1().unwrap();
            let live_snapshot = host.port.snapshot_hub().latest();
            let frame = handle.checkpoint_canvas(&checkpoint.checkpoint_id).unwrap();
            assert_eq!(frame.world.tick, checkpoint.tick);
            assert_eq!(frame.world.agents.len(), checkpoint.agent_count);
            assert!(frame.agent_visuals_complete());
            let original = serde_json::to_value(&oracle.world).unwrap();
            let reconstructed = serde_json::to_value(&frame.world).unwrap();
            // Core checkpoints retain scientific UIDs/state, not arena slot handles
            // or the preceding tick's death counter. Keep their raw difference
            // visible, then compare every remaining projected field exactly.
            let logical = |mut value: Value| {
                for agent in value["agents"].as_array_mut().unwrap() {
                    agent.as_object_mut().unwrap().remove("id").unwrap();
                }
                value["summary"]
                    .as_object_mut()
                    .unwrap()
                    .remove("deaths")
                    .unwrap();
                value
            };
            assert_eq!(logical(reconstructed.clone()), logical(original.clone()));
            eprintln!(
                "checkpoint canvas preset {preset:?}: original deaths={} reconstructed deaths={}; arena handles are reallocated, scientific UIDs and remaining projection fields match",
                original["summary"]["deaths"], reconstructed["summary"]["deaths"]
            );
            assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
            assert!(Arc::ptr_eq(
                &host.port.snapshot_hub().latest(),
                &live_snapshot
            ));
            let artifact_path = directory.join(&artifact.relative_path);
            assert_eq!(fs::read(&artifact_path).unwrap(), bytes);
            fs::write(directory.join("before-corruption.bin"), &bytes).unwrap();
            fs::write(
                directory.join("checkpoint-metadata.json"),
                serde_json::to_vec_pretty(&checkpoint).unwrap(),
            )
            .unwrap();
            fs::write(
                directory.join("review-world.json"),
                serde_json::to_vec_pretty(&frame.world).unwrap(),
            )
            .unwrap();
            assert!(matches!(
                handle.checkpoint_canvas("unknown"),
                Err(ControlError::NotFound(_))
            ));
            for mismatch in ["tick", "population", "digest"] {
                let mut changed = checkpoint.clone();
                match mismatch {
                    "tick" => changed.tick += 1,
                    "population" => changed.agent_count += 1,
                    "digest" => changed.world_digest_v1.push('!'),
                    _ => unreachable!(),
                }
                handle
                    .data_services
                    .checkpoints
                    .lock()
                    .unwrap()
                    .insert(checkpoint.checkpoint_id.clone(), changed);
                assert!(
                    matches!(handle.checkpoint_canvas(&checkpoint.checkpoint_id),
                    Err(ControlError::InvalidPatch(message)) if message.contains("catalog and verified artifact disagree"))
                );
                handle
                    .data_services
                    .checkpoints
                    .lock()
                    .unwrap()
                    .insert(checkpoint.checkpoint_id.clone(), checkpoint.clone());
            }
            eprintln!(
                "checkpoint canvas preset {preset:?}: actual tick={} agents={} digest={}; owner digest={} unchanged; catalog tick/population/digest mismatches refused",
                checkpoint.tick, checkpoint.agent_count, checkpoint.world_digest_v1, live.overall
            );

            // A real retained artifact is made corrupt by appending, preserving its preimage.
            fs::OpenOptions::new()
                .append(true)
                .open(&artifact_path)
                .unwrap()
                .write_all(&[0xff])
                .unwrap();
            assert!(matches!(
                handle.checkpoint_canvas(&checkpoint.checkpoint_id),
                Err(ControlError::InvalidPatch(_))
            ));
            assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
        }
    }

    #[test]
    fn bounded_checkpoint_canvas_verifies_cold_and_cached_sources_without_changing_owner() {
        let directory = tempfile::tempdir().unwrap().keep();
        let (handle, host) = checkpoint_test_handle(&directory, 4, 64 * 1024 * 1024);
        let saved = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        host.wait_applied(&handle.step().unwrap());
        let live = host.port.scientific_digest_v1().unwrap();
        let publication = handle.read_snapshot().unwrap();
        for cold in [false, true] {
            if cold {
                handle.data_services.checkpoints.lock().unwrap().clear();
                handle.data_services.artifacts.lock().unwrap().clear();
            }
            let (metadata, frame) = handle
                .checkpoint_canvas_bounded(&saved.checkpoint_id)
                .unwrap();
            assert_eq!(metadata, saved);
            assert_eq!(frame.world.tick, saved.tick);
            assert_eq!(frame.world.agents.len(), saved.agent_count);
            assert!(frame.agent_visuals_complete());
            assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
            assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &publication));
        }
        // An acknowledged marker cannot bypass a pending original writer request.
        for request in handle
            .data_services
            .checkpoint_retention
            .lock()
            .unwrap()
            .requests
            .values_mut()
        {
            request.acknowledged = false;
        }
        assert!(matches!(
            handle.checkpoint_canvas_bounded(&saved.checkpoint_id),
            Err(ControlError::NotFound(reason)) if reason.contains("completion is pending")
        ));
        for request in handle
            .data_services
            .checkpoint_retention
            .lock()
            .unwrap()
            .requests
            .values_mut()
        {
            request.acknowledged = true;
        }
        let path = directory.join(format!("{}.bin", saved.checkpoint_id));
        let preimage = fs::read(&path).unwrap();
        fs::write(directory.join("preview-before-corruption.bin"), &preimage).unwrap();
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0xff])
            .unwrap();
        assert!(matches!(
            handle.checkpoint_canvas_bounded(&saved.checkpoint_id),
            Err(ControlError::InvalidPatch(_))
        ));
        assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
        assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &publication));
    }

    #[test]
    fn bounded_checkpoint_replay_keeps_selected_source_across_cold_catalogs() {
        let AgentCanvasJourney {
            handle,
            host,
            checkpoint_id,
            oracles,
            ..
        } = mixed_canvas_journey();
        let selected = handle.get_checkpoint(&checkpoint_id).unwrap();
        let target_tick = oracles[0].0.world.tick;
        let newer = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        assert!(newer.tick > target_tick);
        assert_ne!(newer.checkpoint_id, checkpoint_id);
        host.wait_applied(&handle.step().unwrap());
        let live = handle.read_snapshot().unwrap();
        let digest = host.port.scientific_digest_v1().unwrap();
        let native = handle
            .replay_canvas_from_checkpoint(
                handle.session_id(),
                &lab_checkpoint_descriptor(&selected),
                target_tick,
            )
            .unwrap();
        for cold in [false, true] {
            if cold {
                handle.data_services.checkpoints.lock().unwrap().clear();
                handle.data_services.artifacts.lock().unwrap().clear();
            }
            let (metadata, review) = handle
                .replay_canvas_from_checkpoint_bounded(
                    handle.session_id(),
                    &checkpoint_id,
                    target_tick,
                )
                .unwrap();
            assert_eq!(metadata, selected);
            assert_eq!(review.checkpoint_id, checkpoint_id);
            assert_eq!(review.base_tick, selected.tick);
            assert_eq!(review.snapshot.world.tick, target_tick);
            assert_eq!(review.verification, native.verification);
            assert_eq!(
                serde_json::to_value(&review.snapshot.world).unwrap(),
                serde_json::to_value(&native.snapshot.world).unwrap(),
            );
            assert_eq!(host.port.scientific_digest_v1().unwrap(), digest);
            assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &live));
        }
        assert!(matches!(
            handle.replay_canvas_from_checkpoint_bounded(
                scriptbots_runtime::HostSessionId::new(handle.session_id().get() + 1),
                "missing-checkpoint",
                target_tick,
            ),
            Err(ControlError::Conflict(_))
        ));
        assert!(matches!(
            handle.replay_canvas_from_checkpoint_bounded(
                handle.session_id(),
                &newer.checkpoint_id,
                target_tick,
            ),
            Err(ControlError::BadRequest(reason)) if reason.contains("precedes")
        ));
        assert_eq!(host.port.scientific_digest_v1().unwrap(), digest);
        assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &live));
    }

    #[test]
    fn bounded_checkpoint_replay_refuses_recorded_allocation_growth_without_narrowing_native() {
        for patch in [
            serde_json::json!({"history_capacity": MAX_CHECKPOINT_PREVIEW_HISTORY + 1}),
            serde_json::json!({"narrative_capacity": MAX_CHECKPOINT_PREVIEW_HISTORY + 1}),
            serde_json::json!({
                "closed": false,
                "population_minimum": scriptbots_runtime::remote::REMOTE_MAX_AGENTS + 1,
            }),
            serde_json::json!({
                "closed": false,
                "population_spawn_interval": 1,
                "population_spawn_count": scriptbots_runtime::remote::REMOTE_MAX_AGENTS + 1,
            }),
        ] {
            let directory = tempfile::tempdir().unwrap().keep();
            let (handle, host) = checkpoint_test_handle(&directory, 8, 64 * 1024 * 1024);
            let saved = handle
                .create_checkpoint(CheckpointCreateRequest::default())
                .unwrap();
            host.wait_applied(&handle.step().unwrap());
            host.wait_applied(&handle.apply_patch(patch.clone()).unwrap());
            let publication = handle.read_snapshot().unwrap();
            let current = host.port.scientific_digest_v1().unwrap();
            assert!(current.tick.0 > saved.tick);
            assert!(publication.world.agents.is_empty());
            let native = handle
                .replay_canvas_from_checkpoint(
                    handle.session_id(),
                    &lab_checkpoint_descriptor(&saved),
                    current.tick.0,
                )
                .expect("native replay retains its original resource policy");
            assert_eq!(native.snapshot.world.tick, current.tick.0);
            assert!(
                matches!(
                    handle.replay_canvas_from_checkpoint_bounded(
                        handle.session_id(),
                        &saved.checkpoint_id,
                        current.tick.0,
                    ),
                    Err(ControlError::PayloadTooLarge(_))
                ),
                "recorded patch {patch}"
            );
            assert_eq!(host.port.scientific_digest_v1().unwrap(), current);
            assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &publication));
        }
    }

    #[test]
    fn replay_population_admission_observes_real_agent_and_crossover_spawns() {
        let AgentCanvasJourney {
            handle,
            host,
            checkpoint_id,
            ..
        } = agent_canvas_journey();
        let source = handle
            .data_services
            .checkpoint_retention
            .lock()
            .unwrap()
            .requests[&checkpoint_id]
            .clone();
        let checkpoint = handle.verified_canvas_checkpoint(&checkpoint_id).unwrap();
        let current = host.port.scientific_digest_cut().unwrap();
        let publication = handle.read_snapshot().unwrap();
        assert_eq!(
            publication.world.agents.len(),
            checkpoint.agent_count() + 2,
            "the real owner applied both the injected agent and the crossover"
        );
        // A small explicit policy exercises the same admission checks without creating
        // 65,537 agents. The production ceiling is covered by the exact-boundary test.
        let limits = CanvasReplayLimits {
            max_agents: checkpoint.agent_count() + 1,
        };
        let commands = ControlHandle::event_review_scientific_commands(
            &source,
            &current,
            current.digest.tick.0,
            &[],
        )
        .unwrap();
        assert_eq!(
            commands.len(),
            3,
            "only the three actual post-source inputs"
        );
        assert!(matches!(
            ControlHandle::replay_event_canvas_world(
                &checkpoint,
                &current.digest,
                &[],
                &commands,
                current.digest.tick.0,
                source.source_persistence_interval,
                source.source_config_revision,
                Some(limits),
            ),
            Err(ControlError::PayloadTooLarge(_))
        ));
        let native = handle
            .replay_canvas_from_checkpoint(
                handle.session_id(),
                &lab_checkpoint_descriptor(&source.metadata),
                current.digest.tick.0,
            )
            .unwrap();
        assert_eq!(
            native.snapshot.world.agents.len(),
            publication.world.agents.len()
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), current.digest);
        assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &publication));
    }

    #[test]
    fn replay_population_admission_counts_cohorts_queued_at_the_same_boundary() {
        use scriptbots_core::{
            CohortSource, Intervention, InterventionCommand, InterventionSurface, Placement, Region,
        };
        let AgentCanvasJourney { handle, host, .. } = agent_canvas_journey();
        let saved = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        let checkpoint = handle
            .verified_canvas_checkpoint(&saved.checkpoint_id)
            .unwrap();
        let source = handle
            .data_services
            .checkpoint_retention
            .lock()
            .unwrap()
            .requests[&saved.checkpoint_id]
            .clone();
        let brain = checkpoint.required_brain_registry().entries[0].key;
        host.wait_applied(&handle.step().unwrap());
        for seed in [700, 701] {
            let command = InterventionCommand::new(
                Intervention::InjectCohort {
                    count: 2,
                    genome: CohortSource::RegisteredBrain { key: brain },
                    placement: Placement::Seeded {
                        region: Region::All,
                        seed,
                    },
                    herbivore_tendency: 0.75,
                },
                InterventionSurface::Script,
                "bounded replay cohort",
            );
            host.wait_applied(
                &handle
                    .submit_control_command(ControlCommand::Intervention(Box::new(command)))
                    .unwrap(),
            );
        }
        host.wait_applied(&handle.step().unwrap());
        let current = host.port.scientific_digest_cut().unwrap();
        let publication = handle.read_snapshot().unwrap();
        assert_eq!(publication.world.agents.len(), checkpoint.agent_count() + 4);
        let commands = ControlHandle::event_review_scientific_commands(
            &source,
            &current,
            current.digest.tick.0,
            &[],
        )
        .unwrap();
        let limits = CanvasReplayLimits {
            max_agents: checkpoint.agent_count() + 3,
        };
        assert!(matches!(
            ControlHandle::replay_event_canvas_world(
                &checkpoint,
                &current.digest,
                &[],
                &commands,
                current.digest.tick.0,
                source.source_persistence_interval,
                source.source_config_revision,
                Some(limits),
            ),
            Err(ControlError::PayloadTooLarge(_))
        ));
        let native = handle
            .replay_canvas_from_checkpoint(
                handle.session_id(),
                &lab_checkpoint_descriptor(&saved),
                current.digest.tick.0,
            )
            .unwrap();
        assert_eq!(
            native.snapshot.world.agents.len(),
            publication.world.agents.len()
        );
        assert_eq!(host.port.scientific_digest_v1().unwrap(), current.digest);
        assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &publication));
    }

    #[test]
    fn bounded_checkpoint_canvas_refuses_large_descriptors_before_payload_access() {
        let directory = tempfile::tempdir().unwrap().keep();
        let (handle, host) = checkpoint_test_handle(&directory, 8, 64 * 1024 * 1024);
        let saved = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        for case in 0..3 {
            let mut metadata = saved.clone();
            metadata.checkpoint_id = format!("ckpt-{:032x}-t{}", case + 1, saved.tick);
            match case {
                0 => metadata.byte_size = MAX_CHECKPOINT_PREVIEW_BYTES as u64 + 1,
                1 => metadata.byte_size = MAX_ARTIFACT_DOWNLOAD_BYTES as u64 + 1,
                2 => metadata.agent_count = scriptbots_runtime::remote::REMOTE_MAX_AGENTS + 1,
                _ => unreachable!(),
            }
            write_checkpoint_catalog(&directory, &metadata).unwrap();
            assert!(
                !directory
                    .join(format!("{}.bin", metadata.checkpoint_id))
                    .exists()
            );
            assert!(matches!(
                handle.checkpoint_canvas_bounded(&metadata.checkpoint_id),
                Err(ControlError::PayloadTooLarge(_))
            ));
            assert!(matches!(
                handle.replay_canvas_from_checkpoint_bounded(
                    handle.session_id(),
                    &metadata.checkpoint_id,
                    metadata.tick.saturating_add(1),
                ),
                Err(ControlError::PayloadTooLarge(_))
            ));
        }
        // A false small declaration cannot make an oversized physical file admissible.
        let mut metadata = saved.clone();
        metadata.checkpoint_id = format!("ckpt-{:032x}-t{}", 4, saved.tick);
        metadata.byte_size = 1;
        write_checkpoint_catalog(&directory, &metadata).unwrap();
        fs::File::create(directory.join(format!("{}.bin", metadata.checkpoint_id)))
            .unwrap()
            .set_len(MAX_CHECKPOINT_PREVIEW_BYTES as u64 + 1)
            .unwrap();
        assert!(matches!(
            handle.checkpoint_canvas_bounded(&metadata.checkpoint_id),
            Err(ControlError::PayloadTooLarge(_))
        ));
        assert!(matches!(
            handle.replay_canvas_from_checkpoint_bounded(
                handle.session_id(),
                &metadata.checkpoint_id,
                metadata.tick,
            ),
            Err(ControlError::PayloadTooLarge(_))
        ));
        assert_eq!(
            host.port.scientific_digest_v1().unwrap().overall,
            saved.world_digest_v1
        );
    }

    #[test]
    fn bounded_checkpoint_canvas_caps_decoded_histories_before_restoring() {
        let directory = tempfile::tempdir().unwrap().keep();
        let (handle, host) = checkpoint_test_handle(&directory, 8, 64 * 1024 * 1024);
        let saved = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        let before = host.port.scientific_digest_v1().unwrap();
        for narrative in [false, true] {
            let mut config = ScriptBotsConfig {
                rng_seed: Some(42),
                world_width: 40,
                world_height: 40,
                food_cell_size: 10,
                persistence_interval: 0,
                population_minimum: 0,
                population_spawn_interval: 0,
                ..ScriptBotsConfig::default()
            };
            if narrative {
                config.narrative_capacity = MAX_CHECKPOINT_PREVIEW_HISTORY + 1;
            } else {
                config.history_capacity = MAX_CHECKPOINT_PREVIEW_HISTORY + 1;
            }
            let checkpoint = WorldState::new(config).unwrap().checkpoint_v1().unwrap();
            let bytes = checkpoint.encode().unwrap();
            assert!(bytes.len() < MAX_CHECKPOINT_PREVIEW_BYTES);
            let mut metadata = saved.clone();
            metadata.checkpoint_id = format!("ckpt-{:032x}-t0", if narrative { 11 } else { 10 });
            metadata.world_digest_v1 = checkpoint.source_digest().overall.clone();
            metadata.byte_size = bytes.len() as u64;
            metadata.checksum_blake3 = compute_blake3(&bytes);
            metadata.checksum_sha256 = compute_sha256(&bytes);
            write_checkpoint_catalog(&directory, &metadata).unwrap();
            fs::write(
                directory.join(format!("{}.bin", metadata.checkpoint_id)),
                &bytes,
            )
            .unwrap();
            assert!(matches!(
                handle.checkpoint_canvas_bounded(&metadata.checkpoint_id),
                Err(ControlError::PayloadTooLarge(reason)) if reason.contains("history")
            ));
        }
        assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
    }

    #[test]
    fn checkpoint_preview_grid_and_agent_admission_include_exact_boundaries() {
        let config = ScriptBotsConfig {
            world_width: 512,
            world_height: 512,
            food_cell_size: 1,
            history_capacity: MAX_CHECKPOINT_PREVIEW_HISTORY,
            narrative_capacity: MAX_CHECKPOINT_PREVIEW_HISTORY,
            ..ScriptBotsConfig::default()
        };
        ControlHandle::validate_checkpoint_preview_config(
            &config,
            scriptbots_runtime::remote::REMOTE_MAX_AGENTS,
        )
        .unwrap();
        let mut excessive_grid = config.clone();
        excessive_grid.world_width = 513;
        for (config, agents) in [
            (&excessive_grid, 0),
            (&config, scriptbots_runtime::remote::REMOTE_MAX_AGENTS + 1),
        ] {
            assert!(matches!(
                ControlHandle::validate_checkpoint_preview_config(config, agents),
                Err(ControlError::PayloadTooLarge(_))
            ));
        }
    }

    #[test]
    fn checkpoint_canvas_restores_saved_subsets_without_claiming_a_founder_preset() {
        let directory = tempfile::tempdir().unwrap().keep();
        let config = ScriptBotsConfig {
            rng_seed: Some(42),
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            persistence_interval: 0,
            closed: true,
            population_minimum: 0,
            reproduction_attempt_chance: 0.0,
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config).unwrap();
        let first = crate::install_brains(&mut world, crate::BrainPreset::Mlp)
            .unwrap()
            .population[0];
        let second = world
            .register_brain_family(
                scriptbots_brain::dwraon::DwraonBrain::KIND.as_str(),
                Box::new(scriptbots_brain::dwraon::DwraonFamilyAdapter::default()),
            )
            .unwrap();
        for key in [first, second] {
            let agent = world
                .try_spawn_agent(scriptbots_core::AgentData::default())
                .unwrap();
            assert!(world.bind_agent_brain(agent, key).unwrap());
        }
        let host = TestHost::spawn(world);
        let handle = host.handle().with_artifacts_dir(directory.clone()).unwrap();
        host.wait_applied(&handle.step().unwrap());
        let saved = handle
            .create_checkpoint(CheckpointCreateRequest::default())
            .unwrap();
        let (_, bytes) = handle.read_artifact_bytes(&saved.checkpoint_id).unwrap();
        let checkpoint = scriptbots_core::WorldCheckpointV1::decode(&bytes).unwrap();
        let (restored, preset) =
            ControlHandle::restore_canvas_world_and_preset(&checkpoint, None).unwrap();
        assert_eq!(preset, None);
        assert_eq!(
            restored.world_digest_v1().unwrap(),
            *checkpoint.source_digest()
        );
        host.wait_applied(&handle.step().unwrap());
        let live = host.port.scientific_digest_v1().unwrap();
        let publication = handle.read_snapshot().unwrap();
        let review = handle.checkpoint_canvas(&saved.checkpoint_id).unwrap();
        assert_eq!(review.world.tick, saved.tick);
        assert_eq!(review.world.agents.len(), saved.agent_count);
        assert!(review.agent_visuals_complete());
        assert!(matches!(
            handle.create_checkpoint_branch(checkpoint_branch_request(
                &saved.checkpoint_id,
                "subset-preset-refusal",
            )),
            Err(ControlError::BadRequest(reason))
                if reason.contains("requires a versioned compiled application brain preset")
        ));
        assert!(!directory.join("branches").exists());
        assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
        assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &publication));
        assert_eq!(
            handle.read_artifact_bytes(&saved.checkpoint_id).unwrap().1,
            bytes
        );
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn neuro_checkpoint_consumers_refuse_constructor_drift_without_changing_the_owner() {
        for population in [0, 2] {
            let directory = tempfile::Builder::new()
                .prefix("neuro-checkpoint-constructor-")
                .tempdir()
                .unwrap()
                .keep();
            let mut original_config = ScriptBotsConfig {
                rng_seed: Some(42),
                world_width: 120,
                world_height: 120,
                food_cell_size: 20,
                persistence_interval: 0,
                closed: true,
                population_minimum: 0,
                population_spawn_interval: 0,
                reproduction_energy_threshold: f32::MAX,
                ..ScriptBotsConfig::default()
            };
            original_config.neuroflow.hidden_layers = vec![7, 3];
            original_config.neuroflow.activation =
                scriptbots_core::NeuroflowActivationKind::Sigmoid;
            let mut world = WorldState::new(original_config.clone()).unwrap();
            let installed = crate::install_brains(&mut world, crate::BrainPreset::Neuro).unwrap();
            for _ in 0..population {
                let id = world
                    .try_spawn_agent(scriptbots_core::AgentData::default())
                    .unwrap();
                assert!(
                    world
                        .bind_agent_brain(id, installed.population()[0])
                        .unwrap()
                );
            }
            // The raw cut retains the existing registry identity, not its former
            // constructor recipe. An edited config must never substitute a new factory.
            let mut changed = original_config.clone();
            changed.neuroflow.hidden_layers = vec![11, 5];
            scriptbots_core::apply_control_command(
                &mut world,
                ControlCommand::UpdateConfig(Box::new(changed)),
            )
            .unwrap();
            let host = TestHost::spawn(world);
            let handle = host.handle().with_artifacts_dir(directory.clone()).unwrap();
            let saved = handle
                .create_checkpoint(CheckpointCreateRequest::default())
                .unwrap();
            assert_eq!(saved.agent_count, population);
            let (_, bytes) = handle.read_artifact_bytes(&saved.checkpoint_id).unwrap();
            let checkpoint = scriptbots_core::WorldCheckpointV1::decode(&bytes).unwrap();
            let matching = crate::create_brain_registry_for_config(
                crate::BrainPreset::Neuro,
                &original_config,
            )
            .unwrap();
            let restored = WorldState::restore_checkpoint_v1(&checkpoint, matching)
                .expect("the known original registry still restores this valid source");
            assert_eq!(
                restored.world_digest_v1().unwrap(),
                *checkpoint.source_digest()
            );
            let substituted = crate::create_brain_registry_for_config(
                crate::BrainPreset::Neuro,
                checkpoint.config(),
            )
            .unwrap();
            assert!(matches!(
                WorldState::restore_checkpoint_v1(&checkpoint, substituted),
                Err(scriptbots_core::WorldCheckpointError::RegistryMismatch { .. })
            ));
            let before = host.port.scientific_digest_v1().unwrap();
            let published = handle.read_snapshot().unwrap();
            for error in [
                handle.checkpoint_canvas(&saved.checkpoint_id).err(),
                handle.event_canvas(saved.tick).err(),
                handle
                    .create_checkpoint_branch(checkpoint_branch_request(
                        &saved.checkpoint_id,
                        "constructor-drift",
                    ))
                    .err(),
            ] {
                assert!(
                    matches!(
                        &error,
                        Some(ControlError::BadRequest(message))
                            if message.contains(
                                "checkpoint registry does not match the compiled application adapters"
                            )
                    ),
                    "unexpected constructor-drift outcome: {error:?}"
                );
            }
            assert!(!directory.join("branches").exists());
            assert!(
                handle
                    .data_services
                    .checkpoint_branches
                    .lock()
                    .unwrap()
                    .is_empty()
            );
            let capacity = *handle.data_services.execution_capacity.lock().unwrap();
            assert_eq!(capacity.reserved_run_slots, 0);
            assert_eq!(capacity.reserved_grid_cells, 0);
            assert_eq!(host.port.scientific_digest_v1().unwrap(), before);
            assert!(Arc::ptr_eq(&handle.read_snapshot().unwrap(), &published));
            assert_eq!(
                handle.read_artifact_bytes(&saved.checkpoint_id).unwrap().1,
                bytes
            );
        }
    }

    #[test]
    fn checkpoint_retention_serializes_distinct_and_identical_concurrent_creates() {
        for identical in [false, true] {
            let directory = tempfile::tempdir().expect("checkpoint directory").keep();
            let (handle, _host) = checkpoint_test_handle(&directory, 1, 1024 * 1024);
            let barrier = Arc::new(std::sync::Barrier::new(12));
            let workers: Vec<_> = (0..12)
                .map(|index| {
                    let handle = handle.clone();
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        handle.create_checkpoint(CheckpointCreateRequest {
                            idempotency_key: Some(if identical {
                                "same".to_owned()
                            } else {
                                format!("request-{index}")
                            }),
                            description: Some("concurrent capture".to_owned()),
                        })
                    })
                })
                .collect();
            let outcomes: Vec<_> = workers
                .into_iter()
                .map(|worker| worker.join().expect("creator joined"))
                .collect();
            let accepted: Vec<_> = outcomes
                .iter()
                .filter_map(|outcome| outcome.as_ref().ok())
                .collect();
            assert_eq!(accepted.len(), if identical { outcomes.len() } else { 1 });
            for outcome in &outcomes {
                if let Err(error) = outcome {
                    assert!(
                        matches!(error, ControlError::CheckpointCapacity(_)),
                        "{error}"
                    );
                }
            }
            for checkpoint in &accepted {
                assert_eq!(checkpoint.checkpoint_id, accepted[0].checkpoint_id);
            }
            let (artifact, bytes) = handle
                .read_artifact_bytes(&accepted[0].checkpoint_id)
                .expect("accepted bytes");
            assert_eq!(compute_blake3(&bytes), artifact.checksum_blake3);
            let usage = handle
                .checkpoint_retention_usage()
                .expect("retention observation");
            assert_eq!(usage.file_count, 1);
            assert_eq!(usage.file_bytes, bytes.len() as u64);
            assert_eq!(usage.request_entries, 1);
            assert_eq!(usage.pending_entries, 0);
            assert_eq!(usage.payload_cache_bytes, 0);
            assert!(usage.observed_file_count_high_water <= usage.max_count);
            assert!(usage.observed_file_bytes_high_water <= usage.max_bytes);
            let original_key = handle
                .data_services
                .checkpoint_retention
                .lock()
                .expect("retained request")
                .requests
                .keys()
                .next()
                .expect("accepted identity")
                .strip_prefix("checkpoint:")
                .expect("keyed checkpoint")
                .to_owned();
            assert!(matches!(
                handle.create_checkpoint(CheckpointCreateRequest {
                    idempotency_key: Some(original_key),
                    description: Some("changed".to_owned()),
                }),
                Err(ControlError::Conflict(_))
            ));
        }
    }

    #[test]
    fn checkpoint_directory_leases_are_bounded_and_reusable_without_deletion() {
        let root = tempfile::tempdir().expect("directory lease proof").keep();
        let first = root.join("first");
        let second = root.join("second");
        let third = root.join("third");
        let (handle, host) = checkpoint_test_handle(&first, 0, 0);
        let handle = handle
            .with_artifacts_dir(second.clone())
            .expect("second root");
        let before = handle
            .checkpoint_retention_usage()
            .expect("actual directory usage");
        assert_eq!(before.directory_leases as u64, before.max_directory_leases);
        assert_eq!(before.file_count, 0);
        assert!(matches!(
            handle.data_services.set_artifacts_dir(third.clone()),
            Err(ControlError::CheckpointCapacity(_))
        ));
        let after = handle
            .checkpoint_retention_usage()
            .expect("usage after refusal");
        assert_eq!(after.directory_leases, before.directory_leases);
        assert_eq!(
            handle.artifacts_dir(),
            second.canonicalize().expect("second root exists")
        );
        handle
            .data_services
            .set_artifacts_dir(first.clone())
            .expect("reuse leased first root");
        assert_eq!(
            handle.artifacts_dir(),
            first.canonicalize().expect("first root exists")
        );
        assert!(
            first.is_dir() && second.is_dir() && third.is_dir(),
            "no root was removed"
        );
        drop(host);
    }

    #[test]
    fn checkpoint_retention_exact_bytes_directory_change_and_restart_preserve_files() {
        let first = tempfile::tempdir()
            .expect("first checkpoint directory")
            .keep();
        let second = tempfile::tempdir()
            .expect("second checkpoint directory")
            .keep();
        let (handle, host) = checkpoint_test_handle(&first, 1, 0);
        let checkpoint_bytes = handle
            .capture_checkpoint_v1()
            .expect("real checkpoint")
            .encode()
            .expect("encoded checkpoint");
        let required = checkpoint_bytes.len() as u64;
        let request = CheckpointCreateRequest {
            idempotency_key: Some("exact".to_owned()),
            description: None,
        };
        let handle = handle
            .with_checkpoint_retention_limits(scriptbots_storage::CheckpointRetentionLimits {
                max_count: 1,
                max_bytes: required - 1,
            })
            .expect("one-byte short policy");
        assert!(matches!(
            handle.create_checkpoint(request.clone()),
            Err(ControlError::CheckpointCapacity(_))
        ));
        assert_eq!(std::fs::read_dir(&first).expect("inventory").count(), 0);
        let handle = handle
            .with_checkpoint_retention_limits(scriptbots_storage::CheckpointRetentionLimits {
                max_count: 1,
                max_bytes: required,
            })
            .expect("exact policy");
        let retained = handle
            .create_checkpoint(request.clone())
            .expect("exact capacity admits");
        assert_eq!(retained.byte_size, required);
        assert_eq!(
            handle
                .create_checkpoint(request)
                .expect("idempotent retry")
                .checkpoint_id,
            retained.checkpoint_id
        );
        let before = handle
            .read_artifact_bytes(&retained.checkpoint_id)
            .expect("initial read")
            .1;
        let handle = handle
            .with_artifacts_dir(second)
            .expect("directory change retains earlier lease/accounting");
        assert!(matches!(
            handle.create_checkpoint(CheckpointCreateRequest::default()),
            Err(ControlError::CheckpointCapacity(_))
        ));
        assert_eq!(
            handle
                .read_artifact_bytes(&retained.checkpoint_id)
                .expect("original directory read")
                .1,
            before
        );
        assert_eq!(
            handle
                .checkpoint_retention_usage()
                .expect("all directories")
                .file_count,
            1
        );
        drop(handle);
        let mut restarted = host.handle();
        restarted.data_services = Arc::new(DataServices::new(first.clone()));
        let restarted = restarted
            .with_checkpoint_retention_limits(scriptbots_storage::CheckpointRetentionLimits {
                max_count: 1,
                max_bytes: required,
            })
            .expect("restart budget");
        let observed = restarted
            .checkpoint_retention_usage()
            .expect("restart occupancy");
        assert_eq!(observed.file_count, 1);
        assert_eq!(observed.file_bytes, required);
        assert_eq!(
            observed.request_entries, 0,
            "historical files must not be claimed as current-run requests"
        );
        assert!(matches!(
            restarted.create_checkpoint(CheckpointCreateRequest::default()),
            Err(ControlError::CheckpointCapacity(_))
        ));
        assert_eq!(
            fs::read(first.join(format!("{}.bin", retained.checkpoint_id)))
                .expect("preserved file"),
            before
        );
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_aliases_are_refused_without_blocking_exact_retry_or_control() {
        use std::os::unix::fs::symlink;

        for kind in ["symlink", "hardlink"] {
            let directory = tempfile::tempdir().expect("checkpoint directory").keep();
            let (handle, host) = checkpoint_test_handle(&directory, 1, 1024 * 1024);
            let request = CheckpointCreateRequest {
                idempotency_key: Some(kind.to_owned()),
                description: Some("immutable checkpoint source cut".to_owned()),
            };
            let checkpoint = handle.create_checkpoint(request.clone()).expect("capture");
            let (artifact, original) = handle
                .read_artifact_bytes(&checkpoint.checkpoint_id)
                .expect("original download");
            let path = directory.join(&artifact.relative_path);
            let preserved = directory.join("original.checkpoint");
            fs::rename(&path, &preserved).expect("preserve original inode and bytes");
            match kind {
                "symlink" => symlink(&preserved, &path).expect("same-directory symlink"),
                "hardlink" => fs::hard_link(&preserved, &path).expect("hardlink"),
                _ => unreachable!(),
            }
            assert!(matches!(
                handle.create_checkpoint(request.clone()),
                Err(ControlError::CheckpointRetentionUnavailable(_))
            ));
            assert!(matches!(
                handle.read_artifact_bytes(&checkpoint.checkpoint_id),
                Err(ControlError::BadRequest(_))
            ));
            let usage = handle
                .checkpoint_retention_usage()
                .expect("responsive quota read");
            assert_eq!(usage.request_entries, 1);
            assert_eq!(usage.pending_entries, 0);
            assert_eq!(handle.list_checkpoints(None, None).unwrap().total, 1);
            host.wait_applied(&handle.step().expect("responsive owner command"));
            let live = host.port.scientific_digest_v1().expect("advanced world");
            assert!(handle.status().unwrap().tick > checkpoint.tick);

            // Retain both the refused alias and its target. Repair only the
            // now-vacant pathname with the exact original checkpoint bytes.
            fs::rename(&path, directory.join("refused.alias")).expect("preserve alias");
            let mut repaired = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("reserve repaired pathname");
            repaired.write_all(&original).expect("restore exact bytes");
            repaired.sync_all().expect("sync restored bytes");
            drop(repaired);
            let retried = handle
                .create_checkpoint(request)
                .expect("exact repaired retry");
            assert_eq!(
                serde_json::to_value(&retried).unwrap(),
                serde_json::to_value(&checkpoint).unwrap(),
                "repair must not capture the advanced world or allocate another identity"
            );
            assert_eq!(
                handle
                    .read_artifact_bytes(&checkpoint.checkpoint_id)
                    .unwrap()
                    .1,
                original
            );
            assert_eq!(fs::read(&preserved).unwrap(), original);
            let usage = handle
                .checkpoint_retention_usage()
                .expect("repaired quota read");
            assert_eq!(usage.file_count, 1);
            assert_eq!(usage.file_bytes, original.len() as u64);
            assert_eq!(usage.request_entries, 1);
            assert!(usage.inventory_error.is_none());
            assert!(matches!(
                handle.create_checkpoint(CheckpointCreateRequest::default()),
                Err(ControlError::CheckpointCapacity(_))
            ));
            assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
        }
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_fifo_refusal_keeps_control_responsive() {
        use std::os::unix::fs::FileTypeExt;
        use std::process::Command;
        use std::time::Duration;

        const CHILD_DIRECTORY: &str = "SCRIPTBOTS_CHECKPOINT_FIFO_TEST_DIRECTORY";
        if let Some(root) = std::env::var_os(CHILD_DIRECTORY) {
            let root = PathBuf::from(root);
            let directory = root.join("artifacts");
            let (handle, host) = checkpoint_test_handle(&directory, 1, 1024 * 1024);
            let request = CheckpointCreateRequest {
                idempotency_key: Some("fifo-exact-retry".to_owned()),
                description: None,
            };
            let checkpoint = handle
                .create_checkpoint(request.clone())
                .expect("real capture");
            let (artifact, original) = handle
                .read_artifact_bytes(&checkpoint.checkpoint_id)
                .expect("real checkpoint bytes");
            let path = directory.join(&artifact.relative_path);
            fs::rename(&path, root.join("original.checkpoint")).expect("preserve original");
            assert!(
                Command::new("mkfifo")
                    .arg(&path)
                    .status()
                    .unwrap()
                    .success()
            );
            assert!(matches!(
                handle.create_checkpoint(request.clone()),
                Err(ControlError::CheckpointRetentionUnavailable(_))
            ));
            assert!(matches!(
                handle.read_artifact_bytes(&checkpoint.checkpoint_id),
                Err(ControlError::BadRequest(_))
            ));
            let usage = handle
                .checkpoint_retention_usage()
                .expect("retention lock released");
            assert!(usage.inventory_error.is_some());
            assert_eq!(usage.request_entries, 1);
            assert_eq!(usage.pending_entries, 0);
            assert!(matches!(
                handle.create_checkpoint(CheckpointCreateRequest::default()),
                Err(ControlError::CheckpointRetentionUnavailable(_))
            ));
            host.wait_applied(&handle.step().expect("owner remains responsive"));
            assert!(handle.status().unwrap().tick > checkpoint.tick);
            assert_eq!(handle.list_checkpoints(None, None).unwrap().total, 1);
            assert!(fs::symlink_metadata(&path).unwrap().file_type().is_fifo());
            fs::rename(&path, root.join("refused.fifo")).expect("preserve refused FIFO");
            let mut repaired = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("reserve repaired checkpoint");
            repaired.write_all(&original).unwrap();
            repaired.sync_all().unwrap();
            drop(repaired);
            assert_eq!(
                serde_json::to_value(handle.create_checkpoint(request).expect("exact retry"))
                    .unwrap(),
                serde_json::to_value(&checkpoint).unwrap()
            );
            assert_eq!(
                handle
                    .read_artifact_bytes(&checkpoint.checkpoint_id)
                    .unwrap()
                    .1,
                original
            );
            let usage = handle.checkpoint_retention_usage().unwrap();
            assert_eq!(usage.file_count, 1);
            assert_eq!(usage.file_bytes, original.len() as u64);
            assert!(usage.inventory_error.is_none());
            fs::write(
                root.join("refused.txt"),
                b"FIFO refused; control and exact retry responsive",
            )
            .unwrap();
            return;
        }

        let root = tempfile::tempdir().expect("bounded FIFO probe").keep();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "control::tests::checkpoint_fifo_refusal_keeps_control_responsive",
                "--nocapture",
            ])
            .env(CHILD_DIRECTORY, &root);
        // A regression to blocking open must fail in a killed/reaped child,
        // never strand the test harness behind the checkpoint retention mutex.
        crate::experiment_runner::MatchedSeedExperimentRunner::run_bounded_child(
            &mut command,
            Duration::from_secs(10),
        )
        .expect("checkpoint readers must refuse FIFOs without waiting for a writer");
        assert_eq!(
            fs::read(root.join("refused.txt")).unwrap(),
            b"FIFO refused; control and exact retry responsive",
            "the child must execute the complete real-checkpoint journey"
        );
        assert!(
            fs::symlink_metadata(root.join("refused.fifo"))
                .unwrap()
                .file_type()
                .is_fifo()
        );
    }

    #[test]
    fn checkpoint_pending_storage_refusal_retries_the_same_artifact_after_capacity_changes() {
        let root = tempfile::tempdir().expect("checkpoint fixture").keep();
        let directory = root.join("artifacts");
        let (handle, host) = checkpoint_test_handle(&directory, 1, 1024 * 1024);
        let mut pipeline =
            scriptbots_storage::StoragePipeline::unattributed_memory().expect("real storage owner");
        let writer = pipeline.checkpoint_writer();
        writer
            .configure_retention(scriptbots_storage::CheckpointRetentionLimits {
                max_count: 0,
                max_bytes: 1024 * 1024,
            })
            .expect("zero SQL capacity");
        let handle = handle.with_checkpoint_writer(Some(writer.clone()));
        let request = CheckpointCreateRequest {
            idempotency_key: Some("pending".to_owned()),
            description: Some("retained exact retry".to_owned()),
        };
        assert!(matches!(
            handle.create_checkpoint(request.clone()),
            Err(ControlError::CheckpointCapacity(_))
        ));
        let before = handle
            .checkpoint_retention_usage()
            .expect("pending observation");
        assert_eq!(before.file_count, 1);
        assert_eq!(before.pending_entries, 1);
        assert_eq!(before.sql_count, Some(0));
        assert_eq!(
            handle
                .list_checkpoints(None, None)
                .expect("published checkpoints")
                .total,
            0
        );
        let pending_id = handle
            .data_services
            .checkpoint_retention
            .lock()
            .expect("retained identity")
            .requests
            .values()
            .next()
            .expect("pending entry")
            .metadata
            .checkpoint_id
            .clone();
        let original =
            fs::read(directory.join(format!("{pending_id}.bin"))).expect("retained pending bytes");
        writer
            .configure_retention(scriptbots_storage::CheckpointRetentionLimits {
                max_count: 1,
                max_bytes: 1024 * 1024,
            })
            .expect("additional configured capacity");
        let source = scriptbots_core::WorldCheckpointV1::decode(&original).expect("source cut");
        host.wait_applied(&handle.step().expect("advance beyond the retained cut"));
        let live = host.port.scientific_digest_v1().expect("advanced science");
        assert!(handle.status().unwrap().tick > source.tick().0);

        // Matching bytes at the same pathname do not transfer a directory's
        // writer lease. A pending retry must not publish from a substituted root.
        let leased_directory = root.join("preserved-artifacts");
        fs::rename(&directory, &leased_directory).expect("preserve leased directory");
        fs::create_dir(&directory).expect("unleased replacement directory");
        fs::write(directory.join(format!("{pending_id}.bin")), &original)
            .expect("same bytes in replacement root");
        assert!(matches!(
            handle.create_checkpoint(request.clone()),
            Err(ControlError::CheckpointRetentionUnavailable(_))
        ));
        let unavailable = handle
            .checkpoint_retention_usage()
            .expect("responsive usage");
        assert!(unavailable.inventory_error.is_some());
        assert_eq!(unavailable.pending_entries, 1);
        assert_eq!(unavailable.sql_count, Some(0));
        assert!(matches!(
            handle.list_checkpoints(None, None),
            Err(ControlError::CheckpointRetentionUnavailable(message))
                if message.contains("artifact directory identity changed")
        ));
        assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
        fs::rename(&directory, root.join("refused-artifacts")).expect("preserve replacement");
        fs::rename(&leased_directory, &directory).expect("restore original leased root");
        let acknowledged = handle
            .create_checkpoint(request.clone())
            .expect("exact pending retry");
        assert_eq!(acknowledged.checkpoint_id, pending_id);
        assert_eq!(acknowledged.tick, source.tick().0);
        assert_eq!(acknowledged.world_digest_v1, source.source_digest().overall);
        assert_eq!(
            handle
                .create_checkpoint(request)
                .expect("completed retry")
                .checkpoint_id,
            pending_id
        );
        let after = handle
            .checkpoint_retention_usage()
            .expect("acknowledged observation");
        assert_eq!(after.file_count, 1);
        assert_eq!(after.file_bytes, before.file_bytes);
        assert_eq!(after.sql_count, Some(1));
        assert_eq!(after.pending_entries, 0);
        assert!(after.inventory_error.is_none());
        assert_eq!(host.port.scientific_digest_v1().unwrap(), live);
        assert_eq!(
            handle
                .read_artifact_bytes(&pending_id)
                .expect("download after retry")
                .1,
            original
        );
        pipeline
            .shutdown()
            .expect("ordinary storage shutdown succeeds");
    }

    #[test]
    fn checkpoint_indeterminate_owner_ack_retains_identity_and_retries_the_committed_row() {
        let directory = tempfile::tempdir().expect("checkpoint directory").keep();
        let (handle, _host) = checkpoint_test_handle(&directory, 1, 1024 * 1024);
        let mut pipeline =
            scriptbots_storage::StoragePipeline::unattributed_memory().expect("real owner");
        let normal_writer = pipeline.checkpoint_writer();
        normal_writer
            .configure_retention(scriptbots_storage::CheckpointRetentionLimits {
                max_count: 1,
                max_bytes: 1024 * 1024,
            })
            .expect("SQL capacity");
        let paused = pipeline
            .pause_worker_for_test()
            .expect("pause actual owner");
        pipeline
            .set_deadlines(scriptbots_storage::StorageDeadlines {
                flush_ack: std::time::Duration::from_millis(20),
                ..scriptbots_storage::StorageDeadlines::default()
            })
            .expect("short negative acknowledgement deadline");
        let timed_writer = pipeline.checkpoint_writer();
        pipeline
            .set_deadlines(scriptbots_storage::StorageDeadlines::default())
            .expect("restore normal controller deadlines");
        let handle = handle.with_checkpoint_writer(Some(timed_writer));
        let request = CheckpointCreateRequest {
            idempotency_key: Some("actual-owner-ack-timeout".to_owned()),
            description: Some("exact retry after the real queued write commits".to_owned()),
        };
        let error = handle
            .create_checkpoint(request.clone())
            .expect_err("paused actual owner must miss its acknowledgement deadline");
        assert!(matches!(
            error,
            ControlError::CheckpointRetentionUnavailable(ref detail)
                if detail.contains("outcome is unknown")
        ));
        let pending = handle
            .data_services
            .checkpoint_retention
            .lock()
            .expect("retained exact identity")
            .requests
            .values()
            .next()
            .expect("pending request")
            .clone();
        assert!(!pending.acknowledged);
        let original = fs::read(directory.join(&pending.artifact.relative_path))
            .expect("unconfirmed file retained");
        assert_eq!(original.len() as u64, pending.metadata.byte_size);
        assert_eq!(
            handle
                .list_checkpoints(None, None)
                .expect("published list")
                .total,
            0,
            "a timed-out receipt must not publish persisted metadata"
        );
        drop(paused);
        let committed = normal_writer
            .retention_usage()
            .expect("queued write has completed");
        assert_eq!(committed.count, 1);
        let handle = handle.with_checkpoint_writer(Some(normal_writer));
        let result = handle
            .create_checkpoint(request.clone())
            .expect("exact retry");
        assert_eq!(result.checkpoint_id, pending.metadata.checkpoint_id);
        let decoded = scriptbots_core::WorldCheckpointV1::decode(&original)
            .expect("actual retained empty checkpoint");
        assert_eq!(result.agent_count, 0);
        assert_eq!(result.agent_count, decoded.agent_count());
        assert_eq!(result.world_digest_v1, decoded.source_digest().overall);
        assert_ne!(result.world_digest_v1, result.checksum_blake3);
        assert_eq!(
            serde_json::to_value(handle.create_checkpoint(request).expect("repeat retry"))
                .expect("retry metadata"),
            serde_json::to_value(&result).expect("accepted metadata")
        );
        let usage = handle.checkpoint_retention_usage().expect("bounded usage");
        assert_eq!(usage.file_count, 1);
        assert_eq!(usage.sql_count, Some(1));
        assert_eq!(usage.pending_entries, 0);
        assert_eq!(usage.request_entries, 1);
        assert_eq!(
            handle
                .read_artifact_bytes(&result.checkpoint_id)
                .expect("download")
                .1,
            original
        );
        pipeline.shutdown().expect("normal ordered owner shutdown");
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn checkpoint_capture_refusal_for_a_real_legacy_brain_consumes_no_retention() {
        let directory = tempfile::tempdir().expect("checkpoint directory").keep();
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 40,
            world_height: 40,
            food_cell_size: 10,
            rng_seed: Some(43),
            persistence_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .expect("real world");
        let key = scriptbots_brain_neuro::NeuroflowBrain::register(
            &mut world,
            scriptbots_brain_neuro::NeuroflowBrainConfig::default(),
        )
        .expect("register real non-restorable brain");
        let agent = world
            .try_spawn_agent(scriptbots_core::AgentData::default())
            .expect("founder");
        assert!(
            world
                .bind_agent_brain(agent, key)
                .expect("bind actual brain")
        );
        let host = TestHost::spawn(world);
        let mut handle = host.handle();
        handle.data_services = Arc::new(DataServices::new(directory.clone()));
        let request = CheckpointCreateRequest {
            idempotency_key: Some("legacy-refusal".to_owned()),
            description: None,
        };
        for _ in 0..2 {
            let error = handle
                .create_checkpoint(request.clone())
                .expect_err("unsupported checkpoint");
            assert!(matches!(error, ControlError::Host(_)));
            assert!(error.to_string().contains("legacy brain"));
            let usage = handle
                .checkpoint_retention_usage()
                .expect("observed retention");
            assert_eq!(usage.file_count, 0);
            assert_eq!(usage.file_bytes, 0);
            assert_eq!(usage.request_entries, 0);
            assert_eq!(usage.pending_entries, 0);
            assert_eq!(
                handle.list_checkpoints(None, None).expect("metadata").total,
                0
            );
        }
        assert!(
            fs::read_dir(&directory)
                .expect("preserved directory")
                .next()
                .is_none()
        );
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_file_reservation_failure_preserves_capacity_and_allows_retry() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("checkpoint directory").keep();
        let (handle, _host) = checkpoint_test_handle(&directory, 1, 1024 * 1024);
        let original = fs::metadata(&directory)
            .expect("directory metadata")
            .permissions();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o500))
            .expect("make actual file creation unavailable");
        let request = CheckpointCreateRequest {
            idempotency_key: Some("reservation-retry".to_owned()),
            description: None,
        };
        let refused = handle.create_checkpoint(request.clone());
        fs::set_permissions(&directory, original).expect("restore fixture directory access");
        assert!(matches!(
            refused,
            Err(ControlError::CheckpointRetentionUnavailable(_))
        ));
        let usage = handle
            .checkpoint_retention_usage()
            .expect("failed-create observation");
        assert_eq!(usage.file_count, 0);
        assert_eq!(usage.file_bytes, 0);
        assert_eq!(usage.request_entries, 0);
        assert_eq!(
            handle
                .list_checkpoints(None, None)
                .expect("no published success")
                .total,
            0
        );
        let accepted = handle
            .create_checkpoint(request)
            .expect("retry after file access restored");
        let (_, bytes) = handle
            .read_artifact_bytes(&accepted.checkpoint_id)
            .expect("real file readback");
        assert_eq!(compute_blake3(&bytes), accepted.checksum_blake3);
        assert_eq!(
            handle
                .checkpoint_retention_usage()
                .expect("actual capacity")
                .file_count,
            1
        );
    }

    #[test]
    fn checkpoint_oversized_retained_file_is_refused_before_download_and_kept_in_inventory() {
        let directory = tempfile::tempdir().expect("checkpoint directory").keep();
        let path = directory.join("oversized.bin");
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("reserve real oversized artifact");
        let bytes = u64::try_from(MAX_ARTIFACT_DOWNLOAD_BYTES).expect("declared byte limit") + 1;
        file.set_len(bytes)
            .expect("sparse artifact above the actual read bound");
        file.sync_all().expect("retained fixture sync");
        drop(file);
        assert!(matches!(
            read_artifact_file_bounded(&path, MAX_ARTIFACT_DOWNLOAD_BYTES),
            Err(ControlError::PayloadTooLarge(_))
        ));
        let (handle, _host) = checkpoint_test_handle(&directory, 2, bytes - 1);
        let usage = handle
            .checkpoint_retention_usage()
            .expect("historical occupancy");
        assert_eq!(usage.file_count, 1);
        assert_eq!(usage.file_bytes, bytes);
        assert_eq!(usage.request_entries, 0);
        assert!(matches!(
            handle.create_checkpoint(CheckpointCreateRequest::default()),
            Err(ControlError::CheckpointCapacity(_))
        ));
        assert_eq!(
            fs::metadata(&path).expect("refused artifact remains").len(),
            bytes
        );
    }

    fn read_status_before_releasing_owner(
        handle: ControlHandle,
        owner: MutexGuard<'_, ()>,
    ) -> Result<SimulationStatusDto, ControlError> {
        let (reply, receipt) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            reply.send(handle.status()).expect("reply retained");
        });
        let result = receipt.recv_timeout(std::time::Duration::from_secs(2));
        drop(owner);
        reader.join().expect("status reader joins");
        result.expect("status must return while the owner still holds its lock")
    }

    #[test]
    fn status_contention_serves_observed_fields_and_refreshes_after_release() {
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0x513A),
            persistence_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 0,
            closed: true,
            ..ScriptBotsConfig::default()
        })
        .expect("status world");
        world.step().expect("first observed tick");
        world
            .try_inject_agent(scriptbots_core::AgentData::default())
            .expect("one observed agent");
        let mut config = world.config().clone();
        config.closed = false;
        world.apply_config_update(config).expect("first revision");
        let host = TestHost::spawn(world);
        let handle = host.handle();
        let expected = handle.status().expect("initial observed status");
        assert_eq!(expected.tick, 1);
        assert_eq!(expected.agent_count, 1);
        assert!(!expected.is_closed);
        assert_eq!(expected.config_revision, 1);
        assert!(expected.paused);
        let owner = host.clock_gate.lock().expect("hold owner clock");
        let submitting_handle = handle.clone();
        let update = std::thread::spawn(move || {
            submitting_handle.apply_patch(serde_json::json!({"closed": true}))
        });
        host.clock_blocked
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("owner must actually be waiting on the gate");
        assert_eq!(
            read_status_before_releasing_owner(handle.clone(), owner)
                .expect("nonblocking observed status"),
            expected
        );
        let receipt = update
            .join()
            .expect("submitter joined")
            .expect("config admitted");
        host.wait_applied(&receipt);
        let current = handle.status().expect("owner-applied status");
        assert_eq!(current.config_revision, 2);
        assert!(current.is_closed);
        assert_ne!(
            current, expected,
            "the projection must refresh after the queued change applies"
        );
    }

    #[test]
    fn remote_owner_status_is_exact_read_only_and_reports_real_lifecycle() {
        let world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(77),
            population_minimum: 0,
            population_spawn_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        let mut host = TestHost::spawn_with_identity(
            world,
            None,
            scriptbots_runtime::HostSessionId::new(u64::MAX),
        );
        let handle = host.handle();
        let publication = handle.read_snapshot().unwrap();
        let observed = serde_json::to_value(handle.remote_owner_status().unwrap()).unwrap();
        assert_eq!(observed["schema"], "scriptbots.remote-owner-status.v1");
        assert_eq!(observed["identity"]["sessionId"], u64::MAX.to_string());
        assert_eq!(
            observed["identity"]["publicationRevision"],
            publication.revision.get().to_string()
        );
        assert_eq!(observed["identity"]["tick"], "0");
        assert_eq!(observed["health"], "healthy");
        assert!(observed["fault"].is_null());
        assert_eq!(
            observed["playback"],
            serde_json::json!({"paused": true, "speedMultiplier": 1.0})
        );
        assert!(Arc::ptr_eq(&publication, &handle.read_snapshot().unwrap()));
        host.port
            .submit(CommandEnvelope::new(
                CommandId::new(u128::MAX - 9),
                HostCommand::Shutdown,
            ))
            .unwrap();
        assert!(host.worker.take().unwrap().join().unwrap().is_none());
        let stopped = serde_json::to_value(handle.remote_owner_status().unwrap()).unwrap();
        assert_eq!(stopped["identity"]["sessionId"], u64::MAX.to_string());
        assert_eq!(stopped["lifecycle"], "stopped");
        assert_eq!(stopped["identity"]["tick"], "0");
    }

    #[test]
    fn remote_owner_status_preserves_completed_fault_and_bounds_unicode_diagnostics() {
        use scriptbots_runtime::{
            HostCore, HostCoreOptions, HostFault, HostHealth, ManualHostDriver, ManualInstant,
        };

        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(77),
            population_minimum: 0,
            population_spawn_interval: 2,
            population_spawn_count: 1,
            closed: false,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        world
            .brain_registry_mut()
            .unwrap()
            .register("test.remote-owner-fault", |_rng| {
                Err(scriptbots_core::BrainSpawnError::new(
                    "test.remote-owner-fault",
                    std::io::Error::other("deliberate remote owner fault"),
                ))
            });
        let session = scriptbots_runtime::HostSessionId::new(u64::MAX);
        let mut owner = HostCore::new(session, world, HostCoreOptions::default()).unwrap();
        owner.drive(ManualInstant::from_nanos(0)).unwrap();
        assert_eq!(
            owner
                .drive(ManualInstant::from_nanos(100_000_000))
                .unwrap()
                .scientific_steps,
            2
        );
        let publication = owner.latest_snapshot();
        let actual =
            serde_json::to_value(RemoteOwnerStatusDto::from_snapshot(&publication)).unwrap();
        assert_eq!(actual["health"], "faulted");
        assert_eq!(
            actual["lifecycle"], "running",
            "a fault does not prove completed shutdown"
        );
        assert_eq!(actual["identity"]["tick"], "2");
        assert_eq!(actual["fault"]["tick"], "2");
        assert_eq!(actual["fault"]["code"], "brain_spawn");
        assert_eq!(actual["fault"]["diagnostic_truncated"], false);
        assert!(Arc::ptr_eq(&publication, &owner.latest_snapshot()));

        // Declared wire fixtures test impossible-to-run full-width counters and
        // hostile diagnostic strings, independently of the real fault above.
        let mut wide = (*publication).clone();
        wide.revision = scriptbots_runtime::SnapshotRevision::new(u64::MAX);
        wide.revisions.scientific = ScientificRevision::new((1_u64 << 53) + 1);
        wide.revisions.control = scriptbots_runtime::ControlRevision::new(1_u64 << 53);
        wide.health = HostHealth::Faulted(HostFault::Journal {
            batch_id: scriptbots_runtime::JournalBatchId::new(session, u64::MAX),
            failure: scriptbots_runtime::JournalFailure {
                code: "\0".repeat(200),
                message: "\0".repeat(3000),
            },
        });
        let bytes = serde_json::to_vec(&RemoteOwnerStatusDto::from_snapshot(&wide)).unwrap();
        assert!(bytes.len() < 16 * 1024);
        let journal: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            journal["identity"]["publicationRevision"],
            u64::MAX.to_string()
        );
        assert_eq!(
            journal["identity"]["scientificRevision"],
            "9007199254740993"
        );
        assert_eq!(journal["identity"]["controlRevision"], "9007199254740992");
        assert_eq!(journal["fault"]["session_id"], u64::MAX.to_string());
        assert_eq!(journal["fault"]["batch_sequence"], u64::MAX.to_string());
        assert_eq!(journal["fault"]["diagnostic_truncated"], true);
        wide.health = HostHealth::Faulted(HostFault::Protocol {
            code: "unicode".to_owned(),
            message: format!("x{}", "🦋".repeat(1000)),
        });
        let protocol = serde_json::to_value(RemoteOwnerStatusDto::from_snapshot(&wide)).unwrap();
        assert_eq!(
            protocol["fault"]["message"],
            format!("x{}", "🦋".repeat(511))
        );
        assert_eq!(protocol["fault"]["diagnostic_truncated"], true);
    }

    #[test]
    fn remote_owner_status_retains_a_scientific_fault_after_its_displayed_tick() {
        use scriptbots_runtime::{HostCore, HostCoreOptions, ManualHostDriver, ManualInstant};

        struct FailSecondSense {
            calls: Arc<std::sync::atomic::AtomicUsize>,
        }

        impl scriptbots_core::SenseProvider for FailSecondSense {
            fn compute_accumulators(
                &mut self,
                _world: &WorldState,
                handles: &[scriptbots_core::AgentId],
            ) -> Result<
                Vec<scriptbots_core::sense_fixed::SenseAccum>,
                scriptbots_core::SenseProviderError,
            > {
                if self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    Ok(vec![
                        scriptbots_core::sense_fixed::SenseAccum::default();
                        handles.len()
                    ])
                } else {
                    Err(scriptbots_core::SenseProviderError::Execution(
                        "remote sparse-publication sensing failure".to_owned(),
                    ))
                }
            }
        }

        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 40,
            world_height: 40,
            food_cell_size: 10,
            rng_seed: Some(77),
            persistence_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 0,
            reproduction_energy_threshold: f32::MAX,
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        world
            .try_spawn_agent(scriptbots_core::AgentData::default())
            .unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        world.set_sense_provider(Box::new(FailSecondSense {
            calls: Arc::clone(&calls),
        }));
        let mut owner = HostCore::new(
            scriptbots_runtime::HostSessionId::new(77),
            world,
            HostCoreOptions {
                tick_period_nanos: 10,
                snapshot_interval_ticks: 4,
                ..HostCoreOptions::default()
            },
        )
        .unwrap();
        owner.drive(ManualInstant::from_nanos(0)).unwrap();
        assert_eq!(
            owner
                .drive(ManualInstant::from_nanos(10))
                .unwrap()
                .scientific_steps,
            1
        );
        let displayed = owner.latest_snapshot();
        assert_eq!(
            displayed.world.tick, 0,
            "exercise a genuinely older publication"
        );
        assert_eq!(
            owner
                .drive(ManualInstant::from_nanos(20))
                .unwrap()
                .scientific_steps,
            0
        );
        let faulted = owner.latest_snapshot();
        let observed = serde_json::to_value(RemoteOwnerStatusDto::from_snapshot(&faulted)).unwrap();
        assert_eq!(observed["identity"]["tick"], "0");
        assert_eq!(observed["fault"]["tick"], "1");
        assert_eq!(observed["fault"]["code"], "world_step");
        assert_eq!(observed["health"], "faulted");
        assert_eq!(observed["playback"]["paused"], true);
        assert!(
            observed["fault"]["message"]
                .as_str()
                .unwrap()
                .contains("remote sparse-publication sensing failure")
        );
        assert!(faulted.revision > displayed.revision);
        assert_eq!(faulted.world, displayed.world);
        assert_eq!(faulted.revisions.scientific, displayed.revisions.scientific);
        assert_eq!(faulted.revisions.config, displayed.revisions.config);
        assert!(Arc::ptr_eq(&faulted, &owner.latest_snapshot()));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn status_reports_observed_playback_and_shutdown() {
        let (handle, mut host) = handle();
        let status = serde_json::to_value(handle.status().expect("paused status")).unwrap();
        assert_eq!(status["paused"], true);
        assert_eq!(status["lifecycle"], "running");
        let resumed = handle
            .submit_control_command(ControlCommand::Resume)
            .unwrap();
        host.wait_applied(&resumed);
        let status = serde_json::to_value(handle.status().expect("running status")).unwrap();
        assert_eq!(status["paused"], false);
        assert_eq!(status["health"]["state"], "healthy");
        let paused = handle
            .submit_control_command(ControlCommand::Pause)
            .unwrap();
        host.wait_applied(&paused);
        assert_eq!(
            serde_json::to_value(handle.status().unwrap()).unwrap()["paused"],
            true
        );
        host.port
            .submit(CommandEnvelope::new(
                CommandId::new(u128::MAX - 2),
                HostCommand::Shutdown,
            ))
            .expect("shutdown admitted");
        let _ = host
            .worker
            .take()
            .unwrap()
            .join()
            .expect("shutdown completed");
        let status = serde_json::to_value(handle.status().expect("terminal status")).unwrap();
        assert_eq!(status["lifecycle"], "stopped");
    }

    #[test]
    fn successful_host_construction_always_publishes_an_initial_observation() {
        let (handle, host) = handle();
        let snapshot = host.port.snapshot_hub().latest();
        let status = handle.status().expect("initial owner observation");
        assert_eq!(status.tick, snapshot.world.tick);
        assert_eq!(status.agent_count, snapshot.world.agents.len());
        assert_eq!(status.config_revision, snapshot.revisions.config.get());
    }

    /// bd-134: latest-summary reads remain available while the real owner is
    /// parked at an injected clock gate. This is not a live database-stall proof.
    #[test]
    fn latest_summary_reads_publication_while_owner_is_parked() {
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0xB134_5EED),
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .expect("world");
        world.step().expect("persistence-disabled step");
        let published = world
            .history()
            .next_back()
            .expect("completed tick summary")
            .clone();

        let host = TestHost::spawn(world);
        let handle = host.handle();
        let owner = host.clock_gate.lock().expect("owner clock gate");
        let submitter = handle.clone();
        let command = std::thread::spawn(move || submitter.step());
        host.clock_blocked
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("actual owner blockage");
        let reader_handle = handle.clone();
        let (send, receive) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || send.send(reader_handle.latest_summary()));
        let served = receive.recv_timeout(std::time::Duration::from_secs(2));
        drop(owner);
        reader
            .join()
            .expect("summary reader joined")
            .expect("summary receiver retained");
        let served = served
            .expect("summary read must finish while owner is parked")
            .expect("published summary");
        assert_eq!(served, published);
        let receipt = command
            .join()
            .expect("step submitter joined")
            .expect("step admission");
        host.wait_applied(&receipt);
        assert_eq!(
            handle
                .latest_summary()
                .expect("next completed summary")
                .tick
                .0,
            published.tick.0 + 1
        );
    }

    /// A poisoned derived knobs cache must not disable config reads or the
    /// independent host command-status path (bd-2t3k).
    #[test]
    fn poisoned_derived_caches_keep_serving_knobs_and_command_status() {
        let (handle, _receiver) = handle();

        let before = handle.list_knobs().expect("knobs list before poisoning");
        assert!(!before.is_empty(), "config must flatten to some knobs");
        let issued = handle.pause(None).expect("pause command accepted");

        let knobs_poisoner = Arc::clone(&handle.knobs_cache);
        let _ = std::thread::spawn(move || {
            let _guard = knobs_poisoner.lock().expect("pre-poison knobs cache");
            panic!("deliberate poison for the bd-2t3k knobs-cache test");
        })
        .join();
        assert!(
            handle.knobs_cache.lock().is_err(),
            "the derived cache must actually be poisoned for this test to prove anything"
        );

        let after = handle
            .list_knobs()
            .expect("knobs still served from a poisoned cache");
        assert_eq!(
            serde_json::to_value(&after).expect("knobs serialize"),
            serde_json::to_value(&before).expect("knobs serialize")
        );
        let looked_up = handle
            .command_status(&issued.command_id)
            .expect("status still served from a poisoned cache")
            .expect("the issued command is cached");
        assert_eq!(looked_up.command_id, issued.command_id);
        // Writes recover too: a later command must still land in the status cache.
        let next = handle.resume(None).expect("resume command accepted");
        assert!(
            handle
                .command_status(&next.command_id)
                .expect("lookup after a poisoned write")
                .is_some(),
            "a command issued after poisoning must still be recorded"
        );
    }

    #[test]
    fn patch_updates_single_field() {
        let (handle, receiver) = handle();
        let before = handle.snapshot().expect("config before submission");
        let updates = vec![KnobUpdate {
            path: "food_max".to_string(),
            value: Value::from(0.6),
        }];
        let receipt = handle.apply_updates(&updates).expect("patch");

        // A receipt, not a projection. This test used to read food_max ≈ 0.6
        // out of the RETURNED value and call that success, which encoded the
        // defect as the contract: the response was built from the requested
        // config and stamped with the tick at which it was not yet in effect
        // (bd-k7nq).
        assert!(
            receipt.admission_sequence.is_some(),
            "a config update must report the order it took on the bus"
        );

        receiver.wait_applied(&receipt);
        let after = handle.snapshot().expect("owner-applied config");
        assert_eq!(before.config["food_max"], serde_json::json!(0.5));
        assert!(
            (after.config["food_max"]
                .as_f64()
                .expect("numeric food maximum")
                - 0.6)
                .abs()
                < f64::from(f32::EPSILON),
            "the owner-applied receipt must agree with a later config read"
        );
    }

    #[test]
    fn concurrent_disjoint_patches_reject_stale_replacement_and_preserve_both_on_retry() {
        let world = WorldState::new(ScriptBotsConfig {
            world_width: 40,
            world_height: 40,
            food_cell_size: 10,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            closed: false,
            rng_seed: Some(42),
            ..ScriptBotsConfig::default()
        })
        .expect("paused patch world");
        let host = TestHost::spawn(world);
        let clients = [host.handle(), host.handle()];
        let before = clients[0].read_snapshot().expect("shared publication");
        let patches = [
            serde_json::json!({"closed": true}),
            serde_json::json!({"food_max": 0.75}),
        ];

        // Both requests finish the production read/validate/merge before either
        // is submitted. This deterministically reproduces concurrent clients
        // observing one publication, without depending on thread timing.
        let first = clients[0]
            .prepare_config_patch(patches[0].clone())
            .expect("first prepared patch");
        let second = clients[1]
            .prepare_config_patch(patches[1].clone())
            .expect("second prepared patch");
        let start = std::sync::Barrier::new(3);
        let submitted = std::thread::scope(|scope| {
            let pending = [
                (clients[0].host.clone(), first),
                (clients[1].host.clone(), second),
            ]
            .map(|(mut port, envelope)| {
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    port.submit(envelope).expect("real host admission")
                })
            });
            start.wait();
            pending.map(|worker| worker.join().expect("patch submitter joined"))
        });
        let outcomes = submitted.map(|status| host.wait_finished(&status.command_id().to_string()));
        let (winner, stale) = match (
            outcomes[0].application_state.as_str(),
            outcomes[1].application_state.as_str(),
        ) {
            ("applied", "rejected") => (0, 1),
            ("rejected", "applied") => (1, 0),
            other => panic!("one competing patch must win and the other conflict: {other:?}"),
        };
        assert!(
            outcomes[winner].admission_sequence.expect("winning order")
                < outcomes[stale]
                    .admission_sequence
                    .expect("conflicting order")
        );
        assert_eq!(
            outcomes[stale].rejection_reason,
            Some(serde_json::json!({
                "kind": "config_revision_conflict",
                "expected": before.revisions.config.get(),
                "actual": before.revisions.config.get() + 1,
            }))
        );
        let after = clients[0].read_snapshot().expect("resolved owner config");
        let mut expected = before.config.as_ref().clone();
        if winner == 0 {
            expected.closed = true;
        } else {
            expected.food_max = 0.75;
        }
        assert_eq!(after.config.as_ref(), &expected);
        assert_eq!(
            after.revisions.config.get(),
            before.revisions.config.get() + 1
        );
        assert_eq!(after.world.tick, before.world.tick);
        assert_eq!(
            after.last_applied_command.map(|id| id.to_string()),
            Some(outcomes[winner].command_id.clone())
        );

        // A new public request merges against the winning publication. It must
        // retain the peer's edit and obtain its own observable application.
        let retry = clients[stale]
            .apply_patch(patches[stale].clone())
            .expect("fresh patch after observed conflict");
        assert_ne!(retry.command_id, outcomes[stale].command_id);
        host.wait_applied(&retry);
        expected.closed = true;
        expected.food_max = 0.75;
        let final_snapshot = clients[0].read_snapshot().expect("both edits applied");
        assert_eq!(final_snapshot.config.as_ref(), &expected);
        assert_eq!(
            final_snapshot.revisions.config.get(),
            before.revisions.config.get() + 2
        );
        assert_eq!(final_snapshot.world.tick, before.world.tick);
    }

    #[test]
    fn prepared_config_patch_survives_unrelated_playback_revision() {
        let (handle, host) = handle();
        let before = handle.read_snapshot().expect("original publication");
        let envelope = handle
            .prepare_config_patch(serde_json::json!({"food_max": 0.75}))
            .expect("prepared configuration");
        let playback = handle.set_speed(2.0, None).expect("speed command");
        host.wait_applied(&playback);
        let after_playback = handle.read_snapshot().expect("applied playback");
        assert!(after_playback.revisions.control > before.revisions.control);
        assert_eq!(after_playback.revisions.config, before.revisions.config);

        let submitted = handle
            .host
            .clone()
            .submit(envelope)
            .expect("patch admission");
        let receipt = handle.status_dto(submitted).expect("patch receipt");
        host.wait_applied(&receipt);
        let after = handle.read_snapshot().expect("applied config");
        let mut expected = before.config.as_ref().clone();
        expected.food_max = 0.75;
        assert_eq!(after.config.as_ref(), &expected);
        assert_eq!(after.playback, after_playback.playback);
        assert_eq!(after.world.tick, before.world.tick);
        assert_eq!(
            after.revisions.config.get(),
            before.revisions.config.get() + 1
        );
    }

    #[test]
    fn patch_render_quality_and_post_stack_round_trip() {
        let (handle, receiver) = handle();
        let snapshot = handle
            .apply_patch(serde_json::json!({
                "render": {
                    "quality": "high",
                    "theme": "nordic_frost",
                    "palette": "tritanopia",
                    "post": {
                        "bloom": { "enabled": false, "threshold": 1.2 },
                        "fog": { "mode": "low" }
                    },
                    "day_night": { "cycle_ticks": 24000, "stars": true }
                }
            }))
            .expect("render patch applies");
        assert!(
            snapshot.admission_sequence.is_some(),
            "the render patch must report its admission order"
        );

        // Read the round trip only after authoritative application is observed.
        receiver.wait_applied(&snapshot);
        let applied = handle.snapshot().expect("config after drain");
        let render = &applied.config["render"];
        assert_eq!(render["quality"], serde_json::json!("high"));
        assert_eq!(render["theme"], serde_json::json!("nordic_frost"));
        assert_eq!(render["palette"], serde_json::json!("tritanopia"));
        assert_eq!(render["post"]["bloom"]["enabled"], serde_json::json!(false));
        assert!(
            (render["post"]["bloom"]["threshold"]
                .as_f64()
                .expect("threshold")
                - 1.2)
                .abs()
                < 1e-6
        );
        assert_eq!(render["post"]["fog"]["mode"], serde_json::json!("low"));
        assert_eq!(render["day_night"]["cycle_ticks"], serde_json::json!(24000));

        // Decode the authoritative projection back to its typed configuration.
        let config: ScriptBotsConfig =
            serde_json::from_value(applied.config).expect("typed applied config");
        assert_eq!(
            config.render.quality,
            Some(scriptbots_core::RenderQuality::High)
        );
        assert_eq!(
            config
                .render
                .post
                .as_ref()
                .and_then(|post| post.bloom.as_ref())
                .map(|bloom| bloom.enabled),
            Some(false)
        );
    }

    #[test]
    fn patch_render_rejects_invalid_enum_and_range() {
        let (handle, _receiver) = handle();
        let bad_enum = handle.apply_patch(serde_json::json!({
            "render": { "quality": "ludicrous" }
        }));
        assert!(
            matches!(bad_enum, Err(ControlError::InvalidPatch(_))),
            "unknown quality tier must fail closed: {bad_enum:?}"
        );

        let out_of_range = handle.apply_patch(serde_json::json!({
            "render": { "post": { "bloom": { "enabled": true, "intensity": 2.0 } } }
        }));
        assert!(
            matches!(out_of_range, Err(ControlError::InvalidPatch(_))),
            "bloom intensity 2.0 must be rejected by the knob range table: {out_of_range:?}"
        );

        let bad_nested = handle.apply_patch(serde_json::json!({
            "render": { "day_night": { "start_phase": 1.5 } }
        }));
        assert!(
            matches!(bad_nested, Err(ControlError::InvalidPatch(_))),
            "start_phase 1.5 must be rejected: {bad_nested:?}"
        );
    }

    /// The config response reports only applied state.
    ///
    /// This was a `#[should_panic]` target test carrying "KNOWN DEFECT
    /// bd-2z0.4.1: config response projects unapplied future state". The defect
    /// is fixed under bd-k7nq, so the target is now asserted directly rather
    /// than pinned as known-failing — a should_panic marker that no longer
    /// panics is worse than no test, because it fails for the right reason and
    /// reads like a regression.
    ///
    /// The response can no longer project, structurally: it is a receipt and
    /// carries no config field at all. What is left to prove is that the
    /// receipt contains no projected config, and later reads agree with actual
    /// owner application. Admission may race application on the owner thread.
    #[test]
    fn config_response_reports_only_applied_state() {
        let (handle, receiver) = handle();
        let before = handle.snapshot().expect("config before");
        let baseline = before.config["food_max"].as_f64().expect("food_max");

        let receipt = handle
            .apply_updates(&[KnobUpdate {
                path: "food_max".to_owned(),
                value: Value::from(0.6),
            }])
            .expect("accepted config patch");
        assert!(
            receipt.admission_sequence.is_some(),
            "an accepted config patch must report its admission order"
        );

        assert!(
            serde_json::to_value(&receipt)
                .expect("receipt wire")
                .get("config")
                .is_none(),
            "an admission response must not impersonate an applied configuration snapshot"
        );
        receiver.wait_applied(&receipt);
        assert_eq!(
            before.config["food_max"]
                .as_f64()
                .expect("retained baseline"),
            baseline
        );
        let applied = handle.snapshot().expect("config after drain");
        let applied_food_max = applied.config["food_max"].as_f64().expect("food_max");
        assert!(
            (applied_food_max - 0.6).abs() < 1.0e-6,
            "the admitted command never applied, got {applied_food_max}"
        );
    }

    #[test]
    fn absurd_knob_values_are_rejected_at_the_control_boundary() {
        // The end-to-end proof that the hole is closed: this exact request used
        // to be ACCEPTED. ScriptBotsConfig::validate() checks that values are
        // finite and non-negative but declares no upper bounds, so a growth rate
        // of one billion was admissible from REST, from MCP, and therefore from
        // any agent driving them.
        let (handle, _receiver) = handle();
        let err = handle
            .apply_updates(&[KnobUpdate {
                path: "food_growth_rate".into(),
                value: Value::from(1e9),
            }])
            .expect_err("an absurd growth rate must be refused");
        let message = err.to_string();
        assert!(
            message.contains("food_growth_rate") && message.contains("range"),
            "the rejection must name the knob and its range, got: {message}"
        );
    }

    #[test]
    fn every_violation_in_one_patch_is_reported_at_once() {
        // A caller who must fix one knob per round trip gives up; an autonomous
        // one burns its whole budget doing it.
        let (handle, _receiver) = handle();
        let err = handle
            .apply_updates(&[
                KnobUpdate {
                    path: "food_growth_rate".into(),
                    value: Value::from(1e9),
                },
                KnobUpdate {
                    path: "metabolism_drain".into(),
                    value: Value::from(50.0),
                },
            ])
            .expect_err("both knobs are out of range");
        let message = err.to_string();
        assert!(message.contains("food_growth_rate"), "{message}");
        assert!(message.contains("metabolism_drain"), "{message}");
    }

    #[test]
    fn a_harsh_but_sane_world_is_still_expressible() {
        // The bounds exist to reject the absurd, not to enforce taste: a
        // researcher must still be able to build a brutal world.
        let (handle, receiver) = handle();
        let receipt = handle
            .apply_updates(&[
                KnobUpdate {
                    path: "metabolism_drain".into(),
                    value: Value::from(0.9),
                },
                KnobUpdate {
                    path: "spike_damage".into(),
                    value: Value::from(9.0),
                },
            ])
            .expect("a hostile world is a legitimate experiment");
        receiver.wait_applied(&receipt);
        let config = handle.snapshot().expect("applied hostile world");
        assert!(
            (config.config["metabolism_drain"]
                .as_f64()
                .expect("metabolism")
                - 0.9)
                .abs()
                < 1.0e-6
        );
        assert_eq!(config.config["spike_damage"], serde_json::json!(9.0));
    }

    #[test]
    fn unknown_path_errors() {
        let (handle, _receiver) = handle();
        let err = handle
            .apply_updates(&[KnobUpdate {
                path: "does.not.exist".into(),
                value: Value::from(1),
            }])
            .expect_err("unknown path");
        assert!(matches!(err, ControlError::UnknownPath(_)));
    }

    #[test]
    fn bd_yw1j_retired_neighbor_normalizer_is_not_discoverable_or_mutable() {
        let (handle, _receiver) = handle();
        let knobs = handle.list_knobs().expect("list public knobs");
        assert!(
            knobs.iter().all(|knob| knob.path != "sense_max_neighbors"),
            "the retired no-op normalizer must not remain visible as a scientific control"
        );

        let err = handle
            .apply_updates(&[KnobUpdate {
                path: "sense_max_neighbors".into(),
                value: Value::from(12.0),
            }])
            .expect_err("the retired normalizer must fail closed");
        assert!(matches!(
            err,
            ControlError::UnknownPath(ref path) if path == "sense_max_neighbors"
        ));
    }

    #[test]
    fn dimension_updates_are_rejected() {
        let (handle, _receiver) = handle();
        let err = handle
            .apply_updates(&[KnobUpdate {
                path: "world_width".into(),
                value: Value::from(8_000),
            }])
            .expect_err("dimension update should fail");
        match err {
            ControlError::InvalidPatch(message) => {
                assert!(
                    message.contains("changing world dimensions")
                        || message.contains("world dimensions must be divisible"),
                    "unexpected error message: {message}"
                );
            }
            other => panic!("expected InvalidPatch, got {other:?}"),
        }
    }

    #[test]
    fn non_finite_knob_update_is_field_specific_and_not_admitted() {
        let (handle, receiver) = handle();
        let before = receiver.port.snapshot_hub().latest();
        let err = handle
            .apply_updates(&[KnobUpdate {
                path: "food_growth_rate".into(),
                value: Value::String("NaN".into()),
            }])
            .expect_err("non-finite string coercion must fail");
        assert!(
            matches!(&err, ControlError::InvalidPatch(_)),
            "expected InvalidPatch, got {err:?}"
        );
        let ControlError::InvalidPatch(message) = err else {
            return;
        };
        assert!(
            message.contains("food_growth_rate"),
            "error did not identify field: {message}"
        );
        assert_eq!(
            handle
                .command_counter
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            receiver.port.snapshot_hub().latest().revisions,
            before.revisions
        );
        let value = handle
            .snapshot()
            .expect("snapshot")
            .config
            .get("food_growth_rate")
            .and_then(Value::as_f64)
            .expect("food_growth_rate");
        assert!(
            (value as f32 - ScriptBotsConfig::default().food_growth_rate).abs() < f32::EPSILON,
            "rejected update changed food_growth_rate to {value}"
        );
    }

    #[test]
    fn unrepresentable_nested_float_reports_exact_path_without_partial_admission() {
        let (handle, receiver) = handle();
        let before = receiver.port.snapshot_hub().latest();
        let err = handle
            .apply_updates(&[
                KnobUpdate {
                    path: "food_max".into(),
                    value: Value::from(0.6),
                },
                KnobUpdate {
                    path: "render.auto_exposure.enabled".into(),
                    value: Value::from(true),
                },
                KnobUpdate {
                    path: "render.auto_exposure.speed_brighten".into(),
                    value: Value::from(1.0e40_f64),
                },
            ])
            .expect_err("f64 value outside the f32 domain must fail");
        assert!(
            matches!(&err, ControlError::InvalidPatch(_)),
            "expected InvalidPatch, got {err:?}"
        );
        let ControlError::InvalidPatch(message) = err else {
            return;
        };
        assert!(
            message.contains("render.auto_exposure.speed_brighten"),
            "error did not identify nested field: {message}"
        );
        assert_eq!(
            handle
                .command_counter
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            receiver.port.snapshot_hub().latest().revisions,
            before.revisions
        );
        assert_eq!(
            handle.snapshot().expect("snapshot").config.get("food_max"),
            Some(&Value::from(0.5))
        );
    }

    #[test]
    fn bounded_owner_queue_refuses_config_without_projecting_it_and_accepts_with_room() {
        use scriptbots_runtime::{
            FixedDeadlineHost, HostCore, HostCoreOptions, HostSessionId, ManualInstant,
            PlaybackSnapshot,
            channel::{ChannelHostDriver, ChannelHostOptions},
        };

        for capacity in [1, 2] {
            let (ready, client) = std::sync::mpsc::sync_channel(1);
            let (finish, finished) = std::sync::mpsc::channel::<CommandId>();
            let worker = std::thread::spawn(move || {
                let core = HostCore::new(
                    HostSessionId::new(0xca9),
                    WorldState::new(ScriptBotsConfig {
                        rng_seed: Some(42),
                        ..ScriptBotsConfig::default()
                    })
                    .expect("world"),
                    HostCoreOptions {
                        command_capacity: capacity,
                        initial_playback: PlaybackSnapshot {
                            paused: true,
                            speed_multiplier: 1.0,
                        },
                        ..HostCoreOptions::default()
                    },
                )
                .expect("bounded owner");
                let (mut driver, port) = ChannelHostDriver::new(
                    FixedDeadlineHost::new(core),
                    ChannelHostOptions::default(),
                )
                .expect("channel driver");
                ready.send(port).expect("publish port");
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                for sequence in 1_u64.. {
                    match finished.try_recv() {
                        Ok(id) => {
                            return (
                                driver.host().core().latest_snapshot(),
                                driver
                                    .host()
                                    .core()
                                    .local_port()
                                    .command_status(id)
                                    .expect("owner status")
                                    .expect("known command"),
                            );
                        }
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            panic!("client ended before reporting its receipt")
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => {}
                    }
                    assert!(
                        std::time::Instant::now() < deadline,
                        "client did not finish"
                    );
                    // Occupy one actual owner queue slot immediately before every
                    // ingress drain. Capacity two admits the config alongside it;
                    // capacity one must refuse it. No application is simulated.
                    let filler = driver
                        .host_mut()
                        .submit(CommandEnvelope::new(
                            CommandId::new(u128::from(sequence)),
                            HostCommand::Pause,
                        ))
                        .expect("filler submission");
                    assert!(matches!(filler.application(), ApplicationState::Admitted));
                    driver
                        .step(ManualInstant::from_nanos(sequence))
                        .expect("real owner boundary");
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                unreachable!("unbounded sequence")
            });
            let handle = ControlHandle::new(client.recv().expect("owner port"));
            let receipt = handle
                .apply_updates(&[KnobUpdate {
                    path: "food_max".to_owned(),
                    value: Value::from(0.6),
                }])
                .expect("host returns its admission decision");
            let id = serde_json::from_value(Value::String(receipt.command_id.clone()))
                .expect("canonical identity");
            finish.send(id).expect("client finished");
            let (snapshot, terminal) = worker.join().expect("owner joined");
            if capacity == 1 {
                assert_eq!(receipt.application_state, APPLICATION_STATE_REJECTED);
                assert!(receipt.admission_sequence.is_none());
                assert!(matches!(
                    terminal.application(),
                    ApplicationState::Rejected(scriptbots_runtime::RejectionReason::Overloaded {
                        capacity: 1
                    })
                ));
                assert_eq!(snapshot.config.food_max, 0.5);
                assert_eq!(snapshot.revisions.config.get(), 0);
            } else {
                assert_eq!(receipt.application_state, APPLICATION_STATE_ADMITTED);
                assert!(receipt.admission_sequence.is_some());
                assert!(matches!(
                    terminal.application(),
                    ApplicationState::Applied(_)
                ));
                assert!((snapshot.config.food_max - 0.6).abs() < f32::EPSILON);
                assert_eq!(snapshot.revisions.config.get(), 1);
            }
            assert_eq!(snapshot.world.tick, 0, "capacity test must not run science");
        }
    }

    #[test]
    fn disconnected_owner_returns_no_optimistic_config_snapshot() {
        let (handle, mut host) = handle();
        host.port
            .submit(CommandEnvelope::new(
                CommandId::new(u128::MAX - 2),
                HostCommand::Shutdown,
            ))
            .expect("ordered shutdown");
        let _ = host
            .worker
            .take()
            .expect("owner handle")
            .join()
            .expect("owner exited");

        let error = handle
            .apply_updates(&[KnobUpdate {
                path: "food_max".into(),
                value: Value::from(0.6),
            }])
            .expect_err("closed owner must reject config update");
        assert!(matches!(
            error,
            ControlError::Host(scriptbots_runtime::HostAccessError::Disconnected)
        ));
        assert_eq!(
            handle.snapshot().expect("snapshot").config.get("food_max"),
            Some(&Value::from(0.5))
        );
    }

    /// A real control submission advances science on the owner thread, and its
    /// polled receipt reports application and volatile journal commitment.
    #[test]
    fn an_applied_command_is_reported_by_the_applier_and_visible_to_the_submitter() {
        let (handle, receiver) = handle();
        let before = handle.status().expect("before step").tick;
        let admitted = handle.step().expect("step admitted");
        assert_eq!(admitted.application_state, APPLICATION_STATE_ADMITTED);
        receiver.wait_applied(&admitted);
        let polled = handle
            .command_status(&admitted.command_id)
            .expect("lookup")
            .expect("the command is known");
        assert_eq!(
            polled.application_state, APPLICATION_STATE_APPLIED,
            "the receipt did not advance, so `admitted` is still the end of the road"
        );
        assert_eq!(polled.journal_state, "committed_volatile");
        assert_eq!(handle.status().expect("completed step").tick, before + 1);
    }

    /// A stale revision is rejected by the owner without advancing science.
    /// This fails an implementation that reports every admitted step as applied.
    #[test]
    fn a_rejected_command_is_reported_as_rejected_not_applied() {
        let (handle, mut host) = handle();
        let command_id = CommandId::new(0xabc);
        host.port
            .submit(
                CommandEnvelope::new(command_id, HostCommand::Step)
                    .expecting_control_revision(scriptbots_runtime::ControlRevision::new(u64::MAX)),
            )
            .expect("admit revision-guarded step");
        let rejected = host.wait_finished(&command_id.to_string());
        assert_eq!(rejected.application_state, APPLICATION_STATE_REJECTED);
        assert_eq!(rejected.journal_state, "committed_volatile");
        assert_eq!(handle.status().expect("rejected step status").tick, 0);
    }

    /// A receipt can advance from admitted to applied.
    ///
    /// This is the guarantee that did not exist before: every prior fix taught
    /// a surface to say `admitted` honestly, but nothing could move a command
    /// past it, so a caller had no way to learn what happened next.
    #[test]
    fn a_receipt_advances_from_admitted_to_applied() {
        let (handle, host) = handle();
        let admitted = handle.pause(None).expect("pause admitted");
        assert_eq!(admitted.application_state, APPLICATION_STATE_ADMITTED);

        let applied = host.wait_applied(&admitted);
        assert_eq!(applied.application_state, APPLICATION_STATE_APPLIED);
        assert_eq!(
            applied.command_id, admitted.command_id,
            "advancing must not mint a new identity"
        );

        // And the advance is visible to anyone polling the id, which is the
        // only reason a caller was given an identity in the first place.
        let polled = handle
            .command_status(&admitted.command_id)
            .expect("lookup")
            .expect("the command is known");
        assert_eq!(polled.application_state, APPLICATION_STATE_APPLIED);
    }

    /// Unknown identities have no receipt; an exact retry preserves the outcome,
    /// while a conflicting command cannot overwrite it.
    #[test]
    fn the_ledger_refuses_invented_and_revised_outcomes() {
        let (handle, mut host) = handle();
        let unknown = handle
            .command_status(&CommandId::new(99999).to_string())
            .expect("unknown identity lookup");
        assert!(
            unknown.is_none(),
            "a receipt was advanced for a command that was never submitted"
        );

        let admitted = handle.pause(None).expect("pause admitted");
        host.wait_applied(&admitted);
        let command_id: CommandId =
            serde_json::from_value(Value::String(admitted.command_id.clone()))
                .expect("canonical identity");
        let replay = host
            .port
            .submit(CommandEnvelope::new(command_id, HostCommand::Pause))
            .expect("exact envelope retry");
        assert!(matches!(replay.application(), ApplicationState::Applied(_)));
        let contradiction = host
            .port
            .submit(CommandEnvelope::new(command_id, HostCommand::Resume));
        assert!(
            matches!(
                contradiction,
                Err(scriptbots_runtime::HostAccessError::CommandIdCollision { .. })
            ),
            "an applied command was quietly re-reported as rejected"
        );
        assert_eq!(
            handle
                .command_status(&admitted.command_id)
                .expect("lookup")
                .expect("known")
                .application_state,
            APPLICATION_STATE_APPLIED,
            "the refused write must not have changed the stored outcome"
        );
    }

    /// A keyed retry returns the original receipt and enqueues nothing.
    ///
    /// The receipt equality alone would be satisfied by a cache that answered
    /// correctly while still submitting a duplicate, so the queue depth is what
    /// actually carries this test: a retried `Step` that reaches the bus twice
    /// advances the simulation twice, and the client cannot tell (bd-k7nq).
    #[test]
    fn a_keyed_retry_does_not_submit_a_second_command() {
        let (handle, receiver) = handle();

        let first = handle
            .submit_command(ControlCommand::Step, Some("client-abc"))
            .expect("first submit");
        let retry = handle
            .submit_command(ControlCommand::Step, Some("client-abc"))
            .expect("retry of the same logical command");

        assert_eq!(first.command_id.len(), CommandId::new(1).to_string().len());
        assert_eq!(
            first.command_id, retry.command_id,
            "a retry must return the original receipt"
        );
        assert_eq!(
            first.admission_sequence, retry.admission_sequence,
            "a retry must not take a second admission order"
        );

        receiver.wait_applied(&retry);
        assert_eq!(
            handle.status().expect("status").tick,
            1,
            "a retried Step must advance once"
        );
    }

    #[test]
    fn prepared_command_retry_applies_once_and_new_preparation_advances_again() {
        let (handle, owner) = handle();
        let envelope = handle
            .prepare_control_command(ControlCommand::Step, None)
            .expect("prepare step");
        assert_eq!(handle.status().expect("preparation status").tick, 0);
        let mut port = handle.host.clone();
        let first = port.submit(envelope.clone()).expect("first submission");
        let retry = port.submit(envelope.clone()).expect("same envelope retry");
        assert_eq!(first.command_id(), retry.command_id());
        assert_eq!(first.admission_sequence(), retry.admission_sequence());
        owner.wait_applied(&handle.status_dto(retry).expect("retry DTO"));
        assert_eq!(handle.status().expect("after retry").tick, 1);

        let next = handle
            .prepare_control_command(ControlCommand::Step, None)
            .expect("next step");
        assert_ne!(next.command_id, envelope.command_id);
        let applied = port.submit(next).expect("new submission");
        owner.wait_applied(&handle.status_dto(applied).expect("new DTO"));
        assert_eq!(handle.status().expect("after distinct step").tick, 2);
    }

    /// Positive control: distinct keys are distinct commands.
    ///
    /// Without this, the test above would pass against an implementation that
    /// deduplicated everything and only ever submitted once.
    #[test]
    fn distinct_keys_submit_distinct_commands() {
        let (handle, receiver) = handle();

        let first = handle
            .submit_command(ControlCommand::Step, Some("step-1"))
            .expect("first");
        let second = handle
            .submit_command(ControlCommand::Step, Some("step-2"))
            .expect("second");

        assert_ne!(first.command_id, second.command_id);
        assert!(
            second.admission_sequence > first.admission_sequence,
            "distinct commands must take distinct admission orders"
        );

        receiver.wait_applied(&second);
        assert_eq!(
            handle.status().expect("status").tick,
            2,
            "distinct Steps must both apply"
        );
    }

    /// An unkeyed submit stays non-idempotent, as every existing caller expects.
    #[test]
    fn an_unkeyed_submit_is_still_a_new_command_each_time() {
        let (handle, receiver) = handle();

        let first = handle
            .submit_command(ControlCommand::Step, None)
            .expect("a");
        let second = handle
            .submit_command(ControlCommand::Step, None)
            .expect("b");

        assert_ne!(
            first.command_id, second.command_id,
            "without a key there is nothing to recognise a retry by, so these are two commands"
        );

        receiver.wait_applied(&second);
        assert_eq!(handle.status().expect("status").tick, 2);
    }

    /// A process restart must preserve explicit retries without recycling new
    /// commands into the continued owner's existing idempotency ledger.
    #[test]
    fn unkeyed_commands_advance_after_checkpoint_continuation_in_a_new_process() {
        const DIRECTORY: &str = "SCRIPTBOTS_TEST_CONTROL_RESTART_DIRECTORY";
        const PHASE: &str = "SCRIPTBOTS_TEST_CONTROL_RESTART_PHASE";
        if let Some(directory) = std::env::var_os(DIRECTORY) {
            let restore = match std::env::var(PHASE).as_deref() {
                Ok("create") => false,
                Ok("restore") => true,
                phase => panic!("invalid continuation fixture phase: {phase:?}"),
            };
            run_control_restart_phase(&PathBuf::from(directory), restore);
            return;
        }

        let directory = tempfile::tempdir().expect("cross-process continuation directory");
        for phase in ["create", "restore"] {
            let stdout_path = directory.path().join(format!("{phase}.stdout.log"));
            let stderr_path = directory.path().join(format!("{phase}.stderr.log"));
            let mut command = std::process::Command::new(
                std::env::current_exe().expect("current control test executable"),
            );
            command
                .args([
                    "--exact",
                    "control::tests::unkeyed_commands_advance_after_checkpoint_continuation_in_a_new_process",
                    "--nocapture",
                ])
                .env(DIRECTORY, directory.path())
                .env(PHASE, phase)
                .stdout(fs::File::create(&stdout_path).expect("phase stdout log"))
                .stderr(fs::File::create(&stderr_path).expect("phase stderr log"));
            if let Err(error) =
                crate::experiment_runner::MatchedSeedExperimentRunner::run_bounded_child(
                    &mut command,
                    std::time::Duration::from_secs(30),
                )
            {
                let read_log = |path: &Path| {
                    let mut text = String::new();
                    fs::File::open(path)
                        .expect("open phase log")
                        .take(64 * 1024)
                        .read_to_string(&mut text)
                        .expect("read bounded phase log");
                    text
                };
                panic!(
                    "continuation phase {phase} failed: {error}; stdout={} stderr={}",
                    read_log(&stdout_path),
                    read_log(&stderr_path),
                );
            }
            let (artifact, tick) = if phase == "create" {
                ("owner.checkpoint", 2)
            } else {
                ("continued.checkpoint", 3)
            };
            let bytes = fs::read(directory.path().join(artifact))
                .expect("child actually executed and retained its owner cut");
            let checkpoint = scriptbots_runtime::VolatileHostCheckpointV1::decode(&bytes)
                .expect("independently reopen the child artifact");
            assert_eq!(checkpoint.host_checkpoint().tick().0, tick);
        }
    }

    fn run_control_restart_phase(directory: &Path, restore: bool) {
        use scriptbots_runtime::{
            FixedDeadlineHost, HostCore, HostCoreOptions, HostSessionId, ManualInstant,
            PlaybackSnapshot, VolatileHostCheckpointV1,
            channel::{ChannelHostDriver, ChannelHostOptions, ChannelRunOutcome},
        };
        use std::time::{Duration, Instant};

        fn wait_settled(control: &ControlHandle, admitted: &CommandStatusDto) -> CommandStatusDto {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                let observed = control
                    .command_status(&admitted.command_id)
                    .expect("actual owner status")
                    .expect("retained command identity");
                if observed.application_state != "admitted" && observed.journal_state != "pending" {
                    assert_eq!(observed.application_state, "applied");
                    assert_eq!(observed.journal_state, "committed_volatile");
                    return observed;
                }
                assert!(
                    Instant::now() < deadline,
                    "continued command did not settle"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        let checkpoint_path = directory.join("owner.checkpoint");
        let receipts_path = directory.join("receipts.json");
        let saved = restore.then(|| fs::read(&checkpoint_path).expect("saved owner artifact"));
        let (ready, receive) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let core = if let Some(bytes) = saved {
                let checkpoint =
                    VolatileHostCheckpointV1::decode(&bytes).expect("decode actual owner cut");
                let world = checkpoint
                    .host_checkpoint()
                    .restore_world(scriptbots_core::BrainRegistry::new())
                    .expect("restore actual scientific world");
                let persistence = world
                    .bind_persistence(Box::new(scriptbots_core::NullPersistence))
                    .expect("bind continued persistence");
                HostCore::restore_volatile_host_checkpoint_v1(
                    &checkpoint,
                    world,
                    persistence,
                    None,
                    ManualInstant::from_nanos(0),
                )
                .expect("restore the complete owner and command authority")
            } else {
                let mut world = WorldState::new(ScriptBotsConfig {
                    world_width: 200,
                    world_height: 200,
                    food_cell_size: 20,
                    rng_seed: Some(42),
                    persistence_interval: 1,
                    closed: true,
                    population_spawn_interval: 0,
                    reproduction_energy_threshold: f32::MAX,
                    history_capacity: 16,
                    ..ScriptBotsConfig::default()
                })
                .expect("nonempty continuation world");
                for _ in 0..3 {
                    world
                        .try_spawn_agent(scriptbots_core::AgentData::default())
                        .expect("real founder");
                }
                HostCore::new(
                    HostSessionId::new(0xc017),
                    world,
                    HostCoreOptions {
                        initial_playback: PlaybackSnapshot {
                            paused: true,
                            speed_multiplier: 1.0,
                        },
                        ..HostCoreOptions::default()
                    },
                )
                .expect("real original owner")
            };
            let (mut driver, port) =
                ChannelHostDriver::new(FixedDeadlineHost::new(core), ChannelHostOptions::default())
                    .expect("actual owner channel");
            ready.send(port).expect("publish owner port");
            let epoch = Instant::now();
            let receipt = driver
                .run(|| {
                    ManualInstant::from_nanos(
                        u64::try_from(epoch.elapsed().as_nanos()).expect("test duration"),
                    )
                })
                .expect("owner handoff");
            assert_eq!(receipt.outcome, ChannelRunOutcome::HandedOff);
            driver
                .host()
                .core()
                .capture_volatile_host_checkpoint_v1()
                .expect("joint owner and journal cut")
                .encode()
                .expect("encode actual continuation artifact")
        });
        let port = receive.recv().expect("owner ready");
        let control = ControlHandle::new(port.clone());

        if restore {
            let [old_step, old_pause, old_keyed]: [CommandStatusDto; 3] = serde_json::from_slice(
                &fs::read(&receipts_path).expect("original process receipts"),
            )
            .expect("decode original receipts");
            assert_eq!(control.status().expect("restored science").tick, 2);

            // Do not obtain fresh command identities by dropping the old ledger.
            let old_id: CommandId =
                serde_json::from_value(Value::String(old_step.command_id.clone()))
                    .expect("old unkeyed identity");
            let old_retry = control
                .host
                .clone()
                .submit(CommandEnvelope::new(old_id, HostCommand::Step))
                .expect("an explicit old envelope still retries");
            assert_eq!(
                old_retry
                    .admission_sequence()
                    .map(|sequence| sequence.get()),
                old_step.admission_sequence,
            );
            assert!(matches!(
                control
                    .host
                    .clone()
                    .submit(CommandEnvelope::new(old_id, HostCommand::Pause)),
                Err(scriptbots_runtime::HostAccessError::CommandIdCollision { .. })
            ));
            let keyed = control
                .submit_command(ControlCommand::Step, Some("control-restart-key"))
                .expect("keyed retry across processes");
            assert_eq!(keyed.command_id, old_keyed.command_id);
            assert_eq!(keyed.admission_sequence, old_keyed.admission_sequence);
            wait_settled(&control, &keyed);
            assert_eq!(control.status().expect("retry did not advance").tick, 2);

            let step = control.step().expect("new unkeyed step after restart");
            assert_ne!(step.command_id, old_step.command_id);
            wait_settled(&control, &step);
            assert_eq!(control.status().expect("fresh step advanced").tick, 3);

            // The old process's second command was Pause. Reusing its namespace
            // here would collide with that payload instead of changing speed.
            let speed = control
                .clone()
                .set_speed(2.0, None)
                .expect("new command does not collide with the original Pause");
            assert_ne!(speed.command_id, old_pause.command_id);
            wait_settled(&control, &speed);
            let snapshot = control.read_snapshot().expect("continued playback");
            assert_eq!(snapshot.world.tick, 3);
            assert!(snapshot.playback.paused);
            assert_eq!(snapshot.playback.speed_multiplier, 2.0);
            let fresh: CommandId = serde_json::from_value(Value::String(step.command_id))
                .expect("fresh unkeyed identity");
            let cloned: CommandId = serde_json::from_value(Value::String(speed.command_id))
                .expect("cloned control identity");
            assert_ne!(fresh.client_namespace(), 0);
            assert_eq!(fresh.client_namespace(), cloned.client_namespace());
            assert_eq!(fresh.client_sequence(), 0);
            assert_eq!(cloned.client_sequence(), 1);
            assert_eq!(fresh.get() >> 127, 0);
        } else {
            let step = wait_settled(&control, &control.step().expect("original unkeyed step"));
            let pause = wait_settled(&control, &control.pause(None).expect("original pause"));
            let keyed = wait_settled(
                &control,
                &control
                    .submit_command(ControlCommand::Step, Some("control-restart-key"))
                    .expect("original keyed step"),
            );
            assert_eq!(control.status().expect("original science").tick, 2);
            fs::write(
                &receipts_path,
                serde_json::to_vec(&[step, pause, keyed]).expect("encode original receipts"),
            )
            .expect("retain original process receipts");
        }
        let handoff = port
            .handoff_host_checkpoint_v1()
            .expect("settled actual host handoff");
        let bytes = worker.join().expect("original owner actually retired");
        let paired = VolatileHostCheckpointV1::decode(&bytes).expect("final joint artifact");
        assert_eq!(
            paired.host_checkpoint().encode().expect("paired owner"),
            handoff.encode().expect("acknowledged handoff"),
        );
        assert_eq!(handoff.tick().0, if restore { 3 } else { 2 });
        let output = if restore {
            directory.join("continued.checkpoint")
        } else {
            checkpoint_path
        };
        fs::write(output, bytes).expect("retain actual continuation artifact");
    }

    /// A selection submission must hand back a receipt a client can follow.
    ///
    /// Selection used to return `()`, so the REST layer had no identity to
    /// report and answered with a hardcoded `queued: true`. This asserts the
    /// three properties that made that answer useless: there is a command id,
    /// it is distinct per submission, and the receipt is retrievable by that id
    /// afterwards (bd-2z0.4.9).
    #[test]
    fn selection_submission_returns_a_followable_receipt() {
        let (handle, receiver) = handle();
        let update = || SelectionUpdate {
            mode: SelectionMode::Clear,
            agent_ids: Vec::new(),
            state: SelectionState::None,
        };

        let first = handle
            .update_selection(update(), None)
            .expect("first selection");
        let second = handle
            .update_selection(update(), None)
            .expect("second selection");

        assert_ne!(
            first.command_id, second.command_id,
            "two selections must be distinguishable; a client correlating receipts \
             cannot work with a shared id"
        );
        assert!(
            first.admission_sequence.is_some(),
            "an admitted command must report the order it took on the bus"
        );
        assert!(
            second.admission_sequence > first.admission_sequence,
            "admission order must advance, got {:?} then {:?}",
            first.admission_sequence,
            second.admission_sequence
        );

        let looked_up = handle
            .command_status(&first.command_id)
            .expect("status lookup")
            .expect("the receipt must be retrievable by its own id");
        assert_eq!(looked_up.command_id, first.command_id);

        let terminal = receiver.wait_applied(&looked_up);
        assert_eq!(terminal.application_state, APPLICATION_STATE_APPLIED);
        assert_eq!(terminal.journal_state, "committed_volatile");
    }

    #[test]
    fn debug_agents_lists_selection() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let raw_id = {
            let id = world
                .try_spawn_agent(scriptbots_core::AgentData::default())
                .expect("default agent is finite");
            let _ = world.apply_selection_update(SelectionUpdate {
                mode: SelectionMode::Replace,
                agent_ids: vec![id.data().as_ffi()],
                state: SelectionState::Selected,
            });
            id.data().as_ffi()
        };
        let host = TestHost::spawn(world);
        let handle = host.handle();

        let entries = handle
            .debug_agents(AgentDebugQuery {
                selection: Some(SelectionState::Selected),
                ids: Some(vec![raw_id]),
                ..AgentDebugQuery::default()
            })
            .expect("debug agents");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].agent_id, raw_id);
    }

    #[test]
    fn update_selection_enqueues_and_applies() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let raw_id = {
            let id = world
                .try_spawn_agent(scriptbots_core::AgentData::default())
                .expect("default agent is finite");
            id.data().as_ffi()
        };
        let host = TestHost::spawn(world);
        let handle = host.handle();
        let receipt = handle
            .update_selection(
                SelectionUpdate {
                    mode: SelectionMode::Replace,
                    agent_ids: vec![raw_id],
                    state: SelectionState::Selected,
                },
                None,
            )
            .expect("enqueue selection command");

        host.wait_applied(&receipt);
        let entries = handle
            .debug_agents(AgentDebugQuery {
                ids: Some(vec![raw_id]),
                selection: Some(SelectionState::Selected),
                ..AgentDebugQuery::default()
            })
            .expect("selected agent query");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].agent_id, raw_id);
    }

    #[test]
    fn control_commands_generate_status_dtos_and_lookup() {
        let (handle, receiver) = handle();

        let status_pause = handle.pause(None).expect("pause command");
        // Submission returns the admission receipt. Application and journal
        // commitment are observed separately through command-status lookup.
        assert_eq!(status_pause.application_state, APPLICATION_STATE_ADMITTED);
        assert_eq!(status_pause.journal_state, "pending");
        assert_eq!(
            status_pause.command_id.len(),
            CommandId::new(1).to_string().len()
        );

        let status_resume = handle.resume(None).expect("resume command");
        assert_ne!(status_pause.command_id, status_resume.command_id);

        let status_step = handle.step().expect("step command");
        assert_eq!(status_step.application_state, APPLICATION_STATE_ADMITTED);

        let status_speed = handle.set_speed(2.5, None).expect("speed command");
        assert_eq!(status_speed.application_state, APPLICATION_STATE_ADMITTED);

        receiver.wait_applied(&status_speed);

        let looked_up = handle
            .command_status(&status_pause.command_id)
            .expect("lookup")
            .expect("found status");
        assert_eq!(looked_up.command_id, status_pause.command_id);

        let non_existent = handle
            .command_status(&CommandId::new(9999).to_string())
            .expect("lookup");
        assert!(non_existent.is_none());

        let err = handle
            .set_speed(-1.0, None)
            .expect_err("negative speed must fail");
        assert!(matches!(err, ControlError::InvalidPatch(_)));

        let status_shutdown = handle.shutdown().expect("shutdown command");
        assert_eq!(
            status_shutdown.application_state,
            APPLICATION_STATE_ADMITTED
        );
    }

    #[test]
    fn test_map_generate_deterministic_content_hash() {
        let (handle, _receiver) = handle();

        let map1 = handle
            .generate_map(20, 20, Some(50), 12345, None)
            .expect("generate map 1");
        let map2 = handle
            .generate_map(20, 20, Some(50), 12345, None)
            .expect("generate map 2");

        assert_eq!(map1.terrain().width(), 20);
        assert_eq!(map1.terrain().height(), 20);
        assert_eq!(map1.terrain().cell_size(), 50);
        assert_eq!(
            map1.scientific_content_hash(),
            map2.scientific_content_hash()
        );

        let map_diff_seed = handle
            .generate_map(20, 20, Some(50), 54321, None)
            .expect("generate map different seed");
        assert_ne!(
            map1.scientific_content_hash(),
            map_diff_seed.scientific_content_hash()
        );
    }

    #[test]
    fn test_map_generate_sample_preserves_backend_and_rejects_conflicts() {
        let (handle, _receiver) = handle();
        let sample = scriptbots_core::SamplePatternSpec {
            sample_id: "sand_control_v1".into(),
            sample_width: 2,
            sample_height: 2,
            pattern_size: 2,
            sample_tiles: vec![TerrainKind::Sand; 4],
            all_orientations: true,
            periodic: true,
        };
        let request: MapGenerateRequestBody = serde_json::from_value(serde_json::json!({
            "sample": sample,
        }))
        .expect("sample request");
        let map = handle
            .generate_map(
                8,
                8,
                Some(16),
                42,
                request.generator_config().expect("backend"),
            )
            .expect("sample generation");
        assert_eq!(
            map.metadata().generator,
            scriptbots_core::MapGeneratorKind::SampleBased
        );
        assert_eq!(map.metadata().tileset_id, sample.sample_id);
        assert_eq!(
            map.metadata().tileset_hash,
            scriptbots_core::compute_sample_hash(&sample)
        );
        assert!(
            map.terrain()
                .tiles()
                .iter()
                .all(|tile| tile.kind == TerrainKind::Sand)
        );
        let default = handle
            .generate_map(8, 8, Some(16), 42, None)
            .expect("default rule map");
        assert_eq!(
            default.metadata().generator,
            scriptbots_core::MapGeneratorKind::RuleBased
        );
        assert_ne!(
            map.scientific_content_hash(),
            default.scientific_content_hash()
        );

        let conflict = MapGenerateRequestBody {
            tileset: Some(serde_json::json!({})),
            sample: request.sample,
            ..MapGenerateRequestBody::default()
        };
        assert!(
            conflict
                .generator_config()
                .expect_err("conflicting backend")
                .to_string()
                .contains("mutually exclusive")
        );
        let malformed = MapGenerateRequestBody {
            tileset: Some(serde_json::json!({"unknown": true})),
            ..MapGenerateRequestBody::default()
        };
        assert!(
            malformed
                .generator_config()
                .expect_err("invalid tileset")
                .to_string()
                .contains("invalid tileset specification")
        );
    }

    #[test]
    fn test_map_apply_success_and_dimension_mismatch_rejection() {
        let world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let (width, height) = world.config().food_dimensions().expect("food dimensions");
        let cell_size = world.config().food_cell_size;

        let host = TestHost::spawn(world);
        let handle = host.handle();

        // 1. Valid map application matching world dimensions
        let artifact = handle
            .generate_map(width, height, Some(cell_size), 42, None)
            .expect("generate valid map");

        let status = handle
            .apply_map(artifact, ExistingAgentPolicy::Retain, None, None)
            .expect("apply valid map command");
        assert_eq!(status.application_state, APPLICATION_STATE_ADMITTED);

        let observed = host.wait_applied(&status);
        assert_eq!(observed.application_state, "applied");

        // Verify revisions in the host's world state
        let snapshot = handle.read_snapshot().expect("snapshot");
        assert!(snapshot.revisions.control.get() > 0);
        assert!(snapshot.revisions.scientific.get() > 0);

        // 2. Mismatched map dimensions rejected by host
        let bad_artifact = handle
            .generate_map(width + 10, height + 10, Some(cell_size), 42, None)
            .expect("generate mismatched map");
        let status_bad = handle
            .apply_map(bad_artifact, ExistingAgentPolicy::Retain, None, None)
            .expect("submit mismatched map");
        let observed_bad = host.wait_finished(&status_bad.command_id);
        assert_eq!(observed_bad.application_state, "failed");
    }

    #[test]
    fn test_map_apply_revision_guard_preserves_owner_and_keyed_identity() {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 48,
            world_height: 48,
            food_cell_size: 16,
            rng_seed: Some(123),
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        })
        .expect("world");
        world
            .try_spawn_agent(scriptbots_core::AgentData {
                position: scriptbots_core::Position::new(8.0, 24.0),
                ..scriptbots_core::AgentData::default()
            })
            .expect("agent");
        let host = TestHost::spawn(world);
        let handle = host.handle();
        let artifact = handle.generate_map(3, 3, Some(16), 42, None).expect("map");
        let observed = handle.status().expect("observed revision");
        let first = handle
            .apply_map(
                artifact.clone(),
                ExistingAgentPolicy::Retain,
                Some(observed.scientific_revision),
                None,
            )
            .expect("matching revision admitted");
        host.wait_applied(&first);
        let before = handle.read_snapshot().expect("applied map");
        assert_eq!(before.world.agents.len(), 1);
        assert!(before.revisions.scientific.get() > observed.scientific_revision);
        assert_eq!(
            handle
                .status()
                .expect("status revision")
                .scientific_revision,
            before.revisions.scientific.get()
        );
        let stale = handle
            .apply_map(
                artifact.clone(),
                ExistingAgentPolicy::Retain,
                Some(observed.scientific_revision),
                Some("stale-map"),
            )
            .expect("stale request reaches owner");
        let rejected = host.wait_finished(&stale.command_id);
        assert_eq!(rejected.application_state, "rejected");
        assert_eq!(
            rejected.rejection_reason,
            Some(serde_json::json!({
                "kind": "scientific_revision_conflict",
                "expected": observed.scientific_revision,
                "actual": before.revisions.scientific.get(),
            }))
        );
        let command_id: CommandId = serde_json::from_value(Value::String(stale.command_id.clone()))
            .expect("command identity");
        let status = host
            .port
            .clone()
            .command_status(command_id)
            .expect("owner status")
            .expect("known");
        assert!(matches!(
            status.application(),
            ApplicationState::Rejected(scriptbots_runtime::RejectionReason::ScientificRevisionConflict {
                expected, actual
            }) if expected.get() == observed.scientific_revision && *actual == before.revisions.scientific
        ));
        let retry = handle
            .apply_map(
                artifact.clone(),
                ExistingAgentPolicy::Retain,
                Some(observed.scientific_revision),
                Some("stale-map"),
            )
            .expect("identical guarded retry");
        assert_eq!(retry.command_id, rejected.command_id);
        assert_eq!(retry.admission_sequence, rejected.admission_sequence);
        assert_eq!(retry.application_state, "rejected");
        assert_eq!(retry.rejection_reason, rejected.rejection_reason);
        let changed = handle.apply_map(
            artifact,
            ExistingAgentPolicy::Retain,
            Some(before.revisions.scientific.get()),
            Some("stale-map"),
        );
        assert!(matches!(
            changed,
            Err(ControlError::Host(
                scriptbots_runtime::HostAccessError::CommandIdCollision { .. }
            ))
        ));
        let after = handle.read_snapshot().expect("rejected map snapshot");
        assert_eq!(after.layers.terrain, before.layers.terrain);
        assert_eq!(after.world.agents, before.world.agents);
        assert_eq!(after.world.tick, before.world.tick);
        assert_eq!(after.revisions.scientific, before.revisions.scientific);
        assert_eq!(after.last_applied_command, before.last_applied_command);
    }

    #[test]
    fn hex_decoder_rejects_non_ascii_without_panicking() {
        // Even byte length with a char boundary inside the first pair: slicing `&str` here
        // used to panic, which aborts the release binary (`panic = "abort"`).
        assert_eq!(hex_to_bytes("a\u{e9}b"), Err(()));
        assert_eq!(hex_to_bytes("\u{e9}\u{e9}"), Err(()));
        assert_eq!(hex_to_bytes("+f"), Err(()));
        assert_eq!(hex_to_bytes("0aFf"), Ok(vec![0x0a, 0xff]));
        assert_eq!(hex_to_bytes(" 10 "), Ok(vec![0x10]));
        assert_eq!(hex_to_bytes("abc"), Err(()));
        assert!(
            parse_map_artifact(&Value::String("a\u{e9}b".into())).is_err(),
            "a non-ASCII map artifact string must be refused, not panic"
        );
    }

    #[test]
    fn map_artifact_file_reads_are_bounded_and_regular_files_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = read_map_artifact_file(dir.path()).expect_err("a directory is refused");
        assert!(
            err.to_string().contains("not a regular file")
                || err.to_string().contains("failed to read")
        );

        let big = dir.path().join("big.bin");
        let file = fs::File::create(&big).expect("create");
        file.set_len(MAX_MAP_ARTIFACT_FILE_BYTES + 1)
            .expect("set_len");
        let err = read_map_artifact_file(&big).expect_err("oversized file is refused");
        assert!(err.to_string().contains("exceeds"), "{err}");

        let small = dir.path().join("small.bin");
        fs::write(&small, b"not a map").expect("write");
        assert_eq!(
            read_map_artifact_file(&small).expect("small read"),
            b"not a map"
        );
        assert!(parse_map_artifact(&Value::String(small.display().to_string())).is_err());
    }

    #[test]
    fn oversized_knob_paths_are_refused_without_recursing() {
        let mut map = Map::new();
        let deep = vec!["a"; 50_000].join(".");
        assert!(insert_path(&mut map, &deep, Value::from(1)).is_err());
        assert!(
            map.is_empty(),
            "a refused path must not be partially inserted"
        );
        assert!(insert_path(&mut map, "world_width", Value::from(100)).is_ok());
    }

    #[test]
    fn oversized_step_and_map_requests_are_refused_before_work() {
        let (handle, _receiver) = handle();
        let err = handle
            .step_count(u64::MAX, None)
            .expect_err("an unbounded step batch must be refused");
        assert!(err.to_string().contains("per-request limit"), "{err}");
        let err = handle
            .generate_map(65_535, 65_535, Some(50), 1, None)
            .expect_err("a map far beyond the cell limit must be refused");
        assert!(err.to_string().contains("exceeds the limit"), "{err}");
    }
}
