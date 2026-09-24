use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use scriptbots_core::{
    AgentDebugInfo, AgentDebugQuery, ControlCommand, DietClass, HydrologyFlowDirection,
    MapArtifact, RuleBasedMapGenerator, ScriptBotsConfig, SelectionMode, SelectionState,
    SelectionUpdate, TerrainKind, Tick, TilesetSpec, WorldState, default_tileset_spec,
};

use scriptbots_core::ConfigAuditEntry;
use scriptbots_core::check_knob_ranges;
#[cfg(feature = "gui")]
use scriptbots_render::{OffscreenScene, render_offscreen_scene};
use scriptbots_runtime::{
    ApplicationState, CommandEnvelope, CommandId, HostCommand, HostPort, JournalState,
    RenderSnapshot, channel::ChannelHostPort,
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
    pub journal_state: String,
    pub control_revision: u64,
    pub scientific_revision: u64,
}

/// Request payload for setting simulation speed multiplier.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SpeedRequest {
    pub speed: f32,
}

/// Request payload for generating a procedural map sandbox artifact.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MapGenerateRequestBody {
    /// Width of the map grid in cells. Defaults to 100.
    #[schema(default = 100)]
    pub width: Option<u32>,
    /// Height of the map grid in cells. Defaults to 100.
    #[schema(default = 100)]
    pub height: Option<u32>,
    /// Cell size in world units. Defaults to 50.
    #[schema(default = 50)]
    pub cell_size: Option<u32>,
    /// Random seed for deterministic generation. Defaults to a constant seed if omitted.
    pub seed: Option<u64>,
    /// Optional declarative tileset specification (as JSON object). If omitted, default biome tileset is used.
    #[schema(value_type = Option<Object>)]
    pub tileset: Option<Value>,
}

/// Request payload for applying a map artifact to the simulation.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MapApplyRequestBody {
    /// Map artifact payload: either a JSON representation of MapArtifact or a base64/hex postcard string, or file path.
    #[schema(value_type = Object)]
    pub artifact: Value,
    /// Optional idempotency key to prevent double application on retry.
    pub idempotency_key: Option<String>,
}

fn hex_to_bytes(s: &str) -> Result<Vec<u8>, ()> {
    let s = s.trim();
    if !s.len().is_multiple_of(2) {
        return Err(());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ()))
        .collect()
}

/// Parse a `MapArtifact` from either a JSON object or string (file path, raw JSON, or hex-encoded postcard).
pub fn parse_map_artifact(value: &Value) -> Result<MapArtifact, ControlError> {
    match value {
        Value::Object(_) => serde_json::from_value(value.clone())
            .map_err(|e| ControlError::InvalidPatch(format!("invalid map artifact JSON: {e}"))),
        Value::String(s) => {
            let path = Path::new(s);
            if path.exists() {
                let bytes = fs::read(path).map_err(|e| {
                    ControlError::InvalidPatch(format!("failed to read map file: {e}"))
                })?;
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
pub struct ExperimentVariantDto {
    pub variant_id: String,
    pub brain_family: String,
    #[schema(value_type = Option<Object>)]
    pub config_overrides: Option<Value>,
}

/// Request payload for creating a matched-seed experiment batch.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ExperimentCreateRequest {
    pub experiment_id: Option<String>,
    pub description: Option<String>,
    pub variants: Vec<ExperimentVariantDto>,
    pub seeds: Vec<u64>,
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
    pub total_ticks: u64,
    pub final_digest: Option<String>,
    pub bundle_path: Option<String>,
    pub error_reason: Option<String>,
}

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
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct CheckpointMetadataDto {
    pub checkpoint_id: String,
    pub tick: u64,
    pub byte_size: u64,
    pub checksum_blake3: String,
    pub checksum_sha256: String,
    pub schema: String,
    pub created_at_utc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Metadata descriptor for a discoverable and downloadable artifact or bundle.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
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
pub const MAX_ARTIFACT_DOWNLOAD_BYTES: usize = 64 * 1024 * 1024;

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

/// A REST/MCP-created matched-seed batch executing on a background runner thread.
///
/// The runner's atomically written status file is the only source of run
/// progress; this job adds only the process-local execution state (whether a
/// worker is live, whether cancellation was requested, and a batch-level error).
struct ExperimentJob {
    runner: std::sync::Arc<crate::experiment_runner::MatchedSeedExperimentRunner>,
    state_file: PathBuf,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    running: std::sync::Arc<std::sync::atomic::AtomicBool>,
    last_error: std::sync::Arc<Mutex<Option<String>>>,
    created_at_utc: String,
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
    /// Start (or restart) the batch worker. `running` is raised before the
    /// thread exists so a status read can never observe a false idle gap.
    fn start(&self) {
        use std::sync::atomic::Ordering;
        self.running.store(true, Ordering::Release);
        let runner = std::sync::Arc::clone(&self.runner);
        let state_file = self.state_file.clone();
        let cancel = std::sync::Arc::clone(&self.cancel);
        let running = std::sync::Arc::clone(&self.running);
        let last_error = std::sync::Arc::clone(&self.last_error);
        let spawned = std::thread::Builder::new()
            .name(format!("experiment-{}", runner.experiment_id))
            .spawn(move || {
                let outcome = runner.execute_batch_until_cancelled(&state_file, &cancel);
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
                running.store(false, Ordering::Release);
            });
        if let Err(error) = spawned {
            if let Ok(mut slot) = self.last_error.lock() {
                *slot = Some(format!("cannot spawn experiment worker: {error}"));
            }
            self.running.store(false, Ordering::Release);
        }
    }

    fn status_dto(&self, experiment_id: &str) -> Result<ExperimentBatchStatusDto, ControlError> {
        use crate::experiment_runner::{ExperimentBatchStatus, RunState};
        use std::sync::atomic::Ordering;
        let running = self.running.load(Ordering::Acquire);
        let cancel_requested = self.cancel.load(Ordering::Acquire);
        let error_reason = self.last_error.lock().ok().and_then(|slot| slot.clone());
        let (status, updated_at_utc) = match fs::read(&self.state_file) {
            Ok(bytes) => {
                let status: ExperimentBatchStatus =
                    serde_json::from_slice(&bytes).map_err(|error| {
                        ControlError::Serialization(format!(
                            "experiment status {} is unreadable: {error}",
                            self.state_file.display()
                        ))
                    })?;
                let modified = fs::metadata(&self.state_file)
                    .and_then(|meta| meta.modified())
                    .ok()
                    .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or_else(|| self.created_at_utc.clone(), |d| d.as_secs().to_string());
                (status, modified)
            }
            // The worker has not written its first status yet: the plan is the status.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let runs = self
                    .runner
                    .plan_batch()
                    .map_err(|error| ControlError::BadRequest(error.to_string()))?;
                (
                    ExperimentBatchStatus {
                        schema_version: 2,
                        generation: 0,
                        plan_digest: String::new(),
                        experiment_id: experiment_id.to_string(),
                        total_runs: runs.len(),
                        completed_runs: 0,
                        failed_runs: 0,
                        runs,
                    },
                    self.created_at_utc.clone(),
                )
            }
            Err(error) => {
                return Err(ControlError::Serialization(format!(
                    "experiment status {} is unreadable: {error}",
                    self.state_file.display()
                )));
            }
        };
        let finished = status.is_finished();
        let any_started = status.runs.iter().any(|run| run.state != RunState::Pending);
        let label = if running && cancel_requested {
            "cancelling"
        } else if running {
            if any_started { "running" } else { "pending" }
        } else if finished {
            if status.failed_runs > 0 {
                "failed"
            } else {
                "completed"
            }
        } else if error_reason.is_some() {
            "failed"
        } else if cancel_requested {
            "cancelled"
        } else {
            "stopped"
        };
        let runs = status
            .runs
            .into_iter()
            .map(|run| ExperimentRunRecordDto {
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
#[derive(Debug)]
pub struct DataServices {
    experiments: Mutex<BTreeMap<String, ExperimentJob>>,
    checkpoints: Mutex<BTreeMap<String, CheckpointMetadataDto>>,
    checkpoint_data: Mutex<BTreeMap<String, Vec<u8>>>,
    artifacts: Mutex<BTreeMap<String, ArtifactMetadataDto>>,
    idempotency: Mutex<BTreeMap<String, (String, Value)>>,
    artifacts_dir: RwLock<PathBuf>,
}

impl DataServices {
    pub fn new(artifacts_dir: PathBuf) -> Self {
        let _ = fs::create_dir_all(&artifacts_dir);
        Self {
            experiments: Mutex::new(BTreeMap::new()),
            checkpoints: Mutex::new(BTreeMap::new()),
            checkpoint_data: Mutex::new(BTreeMap::new()),
            artifacts: Mutex::new(BTreeMap::new()),
            idempotency: Mutex::new(BTreeMap::new()),
            artifacts_dir: RwLock::new(artifacts_dir),
        }
    }

    pub fn set_artifacts_dir(&self, dir: PathBuf) {
        let _ = fs::create_dir_all(&dir);
        if let Ok(mut lock) = self.artifacts_dir.write() {
            *lock = dir;
        }
    }

    pub fn artifacts_dir(&self) -> PathBuf {
        self.artifacts_dir
            .read()
            .map(|p| p.clone())
            .unwrap_or_else(|_| std::env::temp_dir().join("scriptbots_artifacts"))
    }

    /// Register an external artifact with computed checksums and metadata.
    pub fn register_artifact(&self, meta: ArtifactMetadataDto, data: Option<Vec<u8>>) {
        let id = meta.artifact_id.clone();
        if let Some(bytes) = data
            && let Ok(mut lock) = self.checkpoint_data.lock()
        {
            lock.insert(id.clone(), bytes);
        }
        if let Ok(mut lock) = self.artifacts.lock() {
            lock.insert(id, meta);
        }
    }
}

impl Default for DataServices {
    fn default() -> Self {
        let dir = std::env::var("SCRIPTBOTS_ARTIFACTS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("scriptbots_artifacts"));
        Self::new(dir)
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

impl ControlHandle {
    pub fn new(host: ChannelHostPort) -> Self {
        static NEXT_NAMESPACE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            host,
            knobs_cache: std::sync::Arc::new(Mutex::new(None)),
            command_counter: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            command_namespace: NEXT_NAMESPACE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            database_path: None,
            checkpoint_writer: None,
            data_services: std::sync::Arc::new(DataServices::default()),
        }
    }

    /// Return a reference to the canonical data services.
    pub fn data_services(&self) -> &std::sync::Arc<DataServices> {
        &self.data_services
    }

    /// Set custom artifacts directory for data services.
    pub fn with_artifacts_dir(self, dir: std::path::PathBuf) -> Self {
        self.data_services.set_artifacts_dir(dir);
        self
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
                #[cfg(feature = "neuro")]
                "neuro".into(),
                "fastmcp".into(),
                "rest".into(),
            ],
            capabilities: vec![
                "simulation_control".into(),
                "config_patching".into(),
                "experiments_batch".into(),
                "checkpoints_v1".into(),
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
                "/api/v1/experiments/{experiment_id}/cancel".into(),
                "/api/v1/experiments/{experiment_id}/resume".into(),
                "/api/v1/checkpoints".into(),
                "/api/v1/checkpoints/{checkpoint_id}".into(),
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
                "experiment_cancel".into(),
                "experiment_resume".into(),
                "checkpoint_create".into(),
                "checkpoint_list".into(),
                "checkpoint_status".into(),
                "artifact_list".into(),
                "artifact_get".into(),
            ],
            schemas: vec![
                "ApiVersionDto".into(),
                "ApiSchemaDto".into(),
                "ExperimentBatchStatusDto".into(),
                "ExperimentSummaryDto".into(),
                "ExperimentCreateRequest".into(),
                "CheckpointCreateRequest".into(),
                "CheckpointMetadataDto".into(),
                "ArtifactMetadataDto".into(),
                "PaginatedExperimentsResponse".into(),
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
            if !matches!(
                fam.as_str(),
                "mlp"
                    | "mlp.baseline"
                    | "dwraon"
                    | "dwraon.baseline"
                    | "assembly"
                    | "assembly.experimental"
            ) {
                return Err(ControlError::BadRequest(format!(
                    "unknown brain family '{}': must be mlp, dwraon, or assembly",
                    v.brain_family
                )));
            }
        }
        if request.seeds.is_empty() {
            return Err(ControlError::BadRequest("seeds cannot be empty".into()));
        }
        if request.seeds.len() > 128 {
            return Err(ControlError::BadRequest(
                "seeds exceed maximum limit of 128".into(),
            ));
        }
        let total_runs = request.variants.len().saturating_mul(request.seeds.len());
        if total_runs > 4096 {
            return Err(ControlError::BadRequest(format!(
                "total batch runs ({total_runs}) exceed maximum limit of 4096"
            )));
        }

        let ticks_per_run = request.ticks_per_run.unwrap_or(1000).clamp(1, 10_000_000);

        if let Some(ref key) = request.idempotency_key {
            if key.is_empty() || key.len() > 1024 {
                return Err(ControlError::BadRequest(
                    "idempotency key must contain 1..=1024 bytes".into(),
                ));
            }
            let payload_hash = compute_blake3(
                serde_json::to_string(&request)
                    .unwrap_or_default()
                    .as_bytes(),
            );
            if let Ok(lock) = self.data_services.idempotency.lock()
                && let Some((stored_hash, cached_val)) = lock.get(key)
            {
                if stored_hash == &payload_hash {
                    if let Ok(dto) =
                        serde_json::from_value::<ExperimentBatchStatusDto>(cached_val.clone())
                    {
                        return self.get_experiment(&dto.experiment_id);
                    }
                } else {
                    return Err(ControlError::Conflict(
                        "idempotency key reused with different payload".into(),
                    ));
                }
            }
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
                .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.')
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
            });
        }
        let default_concurrency = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .clamp(1, 4);
        let output_dir = self
            .data_services
            .artifacts_dir()
            .join("experiments")
            .join(&experiment_id);
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
        runner
            .plan_batch()
            .map_err(|error| ControlError::BadRequest(error.to_string()))?;
        fs::create_dir_all(&output_dir).map_err(|error| {
            ControlError::Serialization(format!(
                "cannot create experiment directory {}: {error}",
                output_dir.display()
            ))
        })?;

        let job = ExperimentJob {
            runner: std::sync::Arc::new(runner),
            state_file: output_dir.join("status.json"),
            cancel: std::sync::Arc::default(),
            running: std::sync::Arc::default(),
            last_error: std::sync::Arc::default(),
            created_at_utc: format!("{secs}"),
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
            job.start();
            lock.insert(experiment_id.clone(), job);
        }
        let status_dto = self.get_experiment(&experiment_id)?;

        if let Some(ref key) = request.idempotency_key {
            let payload_hash = compute_blake3(
                serde_json::to_string(&request)
                    .unwrap_or_default()
                    .as_bytes(),
            );
            if let Ok(mut lock) = self.data_services.idempotency.lock()
                && let Ok(val) = serde_json::to_value(&status_dto)
            {
                lock.insert(key.clone(), (payload_hash, val));
            }
        }

        Ok(status_dto)
    }

    /// Look up status of an experiment by ID.
    pub fn get_experiment(
        &self,
        experiment_id: &str,
    ) -> Result<ExperimentBatchStatusDto, ControlError> {
        let lock = self
            .data_services
            .experiments
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let job = lock.get(experiment_id).ok_or_else(|| {
            ControlError::NotFound(format!("experiment '{experiment_id}' not found"))
        })?;
        job.status_dto(experiment_id)
    }

    /// Request cooperative cancellation. The wave already executing finishes;
    /// unadmitted runs stay pending and can be resumed.
    pub fn cancel_experiment(
        &self,
        experiment_id: &str,
    ) -> Result<ExperimentBatchStatusDto, ControlError> {
        let lock = self
            .data_services
            .experiments
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let job = lock.get(experiment_id).ok_or_else(|| {
            ControlError::NotFound(format!("experiment '{experiment_id}' not found"))
        })?;
        job.cancel.store(true, std::sync::atomic::Ordering::Release);
        job.status_dto(experiment_id)
    }

    /// Resume a stopped experiment whose batch still has pending runs.
    pub fn resume_experiment(
        &self,
        experiment_id: &str,
    ) -> Result<ExperimentBatchStatusDto, ControlError> {
        let lock = self
            .data_services
            .experiments
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let job = lock.get(experiment_id).ok_or_else(|| {
            ControlError::NotFound(format!("experiment '{experiment_id}' not found"))
        })?;
        let current = job.status_dto(experiment_id)?;
        match current.status.as_str() {
            "running" | "pending" => {
                return Err(ControlError::Conflict(format!(
                    "experiment '{experiment_id}' is already executing"
                )));
            }
            "cancelling" => {
                return Err(ControlError::Conflict(format!(
                    "experiment '{experiment_id}' is still finishing its current wave; retry after it reports cancelled"
                )));
            }
            _ => {}
        }
        if current.completed_runs + current.failed_runs == current.total_runs {
            return Err(ControlError::Conflict(format!(
                "experiment '{experiment_id}' has no pending runs to resume (status {})",
                current.status
            )));
        }
        job.cancel
            .store(false, std::sync::atomic::Ordering::Release);
        if let Ok(mut error) = job.last_error.lock() {
            *error = None;
        }
        job.start();
        job.status_dto(experiment_id)
    }

    /// List experiments with bounded pagination.
    pub fn list_experiments(
        &self,
        limit: Option<usize>,
        cursor: Option<&str>,
    ) -> Result<PaginatedExperimentsResponse, ControlError> {
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
            .iter()
            .skip(offset)
            .take(limit)
            .map(|(id, job)| {
                job.status_dto(id).map(|e| ExperimentSummaryDto {
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
        })
    }

    /// Capture a typed WorldCheckpointV1 directly from the scientific world owner.
    pub fn capture_checkpoint_v1(
        &self,
    ) -> Result<scriptbots_core::WorldCheckpointV1, ControlError> {
        Ok(self.host.capture_checkpoint_v1()?)
    }

    /// Create a simulation checkpoint and register it as an artifact.
    pub fn create_checkpoint(
        &self,
        request: CheckpointCreateRequest,
    ) -> Result<CheckpointMetadataDto, ControlError> {
        if let Some(ref key) = request.idempotency_key {
            if key.is_empty() || key.len() > 1024 {
                return Err(ControlError::BadRequest(
                    "idempotency key must contain 1..=1024 bytes".into(),
                ));
            }
            let payload_hash =
                compute_blake3(format!("{}:{:?}", key, request.description).as_bytes());
            if let Ok(lock) = self.data_services.idempotency.lock()
                && let Some((stored_hash, cached_val)) = lock.get(key)
            {
                if stored_hash == &payload_hash {
                    if let Ok(dto) =
                        serde_json::from_value::<CheckpointMetadataDto>(cached_val.clone())
                    {
                        return Ok(dto);
                    }
                } else {
                    return Err(ControlError::Conflict(
                        "idempotency key reused with different payload".into(),
                    ));
                }
            }
        }

        // Only a real core checkpoint is ever labelled as one; a capture refusal is the answer.
        let checkpoint = self.capture_checkpoint_v1()?;
        let encoded_bytes = checkpoint
            .encode()
            .map_err(|e| ControlError::Serialization(e.to_string()))?;
        let tick = checkpoint.tick().0;
        let schema = scriptbots_core::WORLD_CHECKPOINT_V1_SCHEMA.to_string();

        let blake3_hex = compute_blake3(&encoded_bytes);
        let sha256_hex = compute_sha256(&encoded_bytes);
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let checkpoint_id = format!("ckpt-{secs}{nanos:04}-t{tick}");

        let meta = CheckpointMetadataDto {
            checkpoint_id: checkpoint_id.clone(),
            tick,
            byte_size: encoded_bytes.len() as u64,
            checksum_blake3: blake3_hex.clone(),
            checksum_sha256: sha256_hex.clone(),
            schema,
            created_at_utc: format!("{secs}"),
            description: request.description.clone(),
        };

        let filename = format!("{checkpoint_id}.bin");
        let art_meta = ArtifactMetadataDto {
            artifact_id: checkpoint_id.clone(),
            filename: filename.clone(),
            content_type: "application/octet-stream".into(),
            byte_size: encoded_bytes.len() as u64,
            checksum_blake3: blake3_hex.clone(),
            checksum_sha256: sha256_hex,
            created_at_utc: format!("{secs}"),
            relative_path: filename.clone(),
        };

        let dir = self.data_services.artifacts_dir();
        let file_path = dir.join(&filename);
        fs::write(&file_path, &encoded_bytes).map_err(|error| {
            ControlError::Serialization(format!(
                "cannot write checkpoint artifact {}: {error}",
                file_path.display()
            ))
        })?;
        // The run database row goes through the connection-owning storage worker.
        if let Some(writer) = &self.checkpoint_writer {
            let metadata = serde_json::json!({
                "source": "control_api",
                "description": request.description,
            });
            writer
                .record(&checkpoint_id, &checkpoint, &metadata)
                .map_err(|error| {
                    ControlError::Serialization(format!(
                        "checkpoint {checkpoint_id} was not recorded in the run database: {error}"
                    ))
                })?;
        }

        if let Ok(mut lock) = self.data_services.checkpoints.lock() {
            lock.insert(checkpoint_id.clone(), meta.clone());
        }
        if let Ok(mut lock) = self.data_services.checkpoint_data.lock() {
            lock.insert(checkpoint_id.clone(), encoded_bytes.clone());
        }
        if let Ok(mut lock) = self.data_services.artifacts.lock() {
            lock.insert(checkpoint_id.clone(), art_meta);
        }

        if let Some(ref key) = request.idempotency_key {
            let payload_hash = compute_blake3(format!("{}:{:?}", key, meta.description).as_bytes());
            if let Ok(mut lock) = self.data_services.idempotency.lock()
                && let Ok(val) = serde_json::to_value(&meta)
            {
                lock.insert(key.clone(), (payload_hash, val));
            }
        }

        Ok(meta)
    }

    /// Look up checkpoint metadata by ID.
    pub fn get_checkpoint(
        &self,
        checkpoint_id: &str,
    ) -> Result<CheckpointMetadataDto, ControlError> {
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

        let lock = self
            .data_services
            .checkpoints
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let total = lock.len();
        let items: Vec<CheckpointMetadataDto> =
            lock.values().skip(offset).take(limit).cloned().collect();

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
        })
    }

    /// Look up artifact metadata by ID.
    pub fn get_artifact(&self, artifact_id: &str) -> Result<ArtifactMetadataDto, ControlError> {
        let lock = self
            .data_services
            .artifacts
            .lock()
            .map_err(|_| ControlError::Lock)?;
        lock.get(artifact_id)
            .cloned()
            .ok_or_else(|| ControlError::NotFound(format!("artifact '{artifact_id}' not found")))
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

        let lock = self
            .data_services
            .artifacts
            .lock()
            .map_err(|_| ControlError::Lock)?;
        let total = lock.len();
        let items: Vec<ArtifactMetadataDto> =
            lock.values().skip(offset).take(limit).cloned().collect();

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
        let artifact = self.get_artifact(artifact_id)?;
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

        let bytes = if let Ok(lock) = self.data_services.checkpoint_data.lock() {
            if let Some(b) = lock.get(artifact_id) {
                b.clone()
            } else {
                let dir = self.data_services.artifacts_dir();
                let file_path = dir.join(rel_path);
                let canonical_dir = dir
                    .canonicalize()
                    .map_err(|e| ControlError::BadRequest(format!("invalid artifacts dir: {e}")))?;
                let canonical_file = file_path
                    .canonicalize()
                    .map_err(|e| ControlError::NotFound(format!("artifact file not found: {e}")))?;
                if !canonical_file.starts_with(&canonical_dir) {
                    return Err(ControlError::BadRequest(
                        "path traversal detected: outside artifacts dir".into(),
                    ));
                }
                let metadata = fs::metadata(&canonical_file)
                    .map_err(|e| ControlError::NotFound(format!("artifact metadata error: {e}")))?;
                if metadata.len() > MAX_ARTIFACT_DOWNLOAD_BYTES as u64 {
                    return Err(ControlError::PayloadTooLarge(format!(
                        "artifact byte size {} exceeds 64MB download limit",
                        metadata.len()
                    )));
                }
                fs::read(&canonical_file).map_err(|e| {
                    ControlError::BadRequest(format!("failed to read artifact: {e}"))
                })?
            }
        } else {
            return Err(ControlError::Lock);
        };

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

        Ok((artifact, bytes))
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
    pub fn step_count(&self, count: u64) -> Result<CommandStatusDto, ControlError> {
        let iterations = count.max(1);
        let mut last_status = None;
        for _ in 0..iterations {
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
            paused: snapshot.playback.paused,
            lifecycle: snapshot.lifecycle,
            health: snapshot.health.clone(),
            last_applied_command: snapshot.last_applied_command.map(|id| id.to_string()),
            command_queue_depth: snapshot.command_queue_depth,
            snapshot_revision: snapshot.revision.get(),
        })
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

    /// Apply a structured JSON patch object onto the configuration.
    pub fn apply_patch(&self, patch: Value) -> Result<CommandStatusDto, ControlError> {
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
        // No explicit drop is needed any more: the borrow ended at the seam
        // above, so the submit below - which reads tick and revision under the
        // same non-reentrant mutex - cannot deadlock against a guard this
        // function is still holding.
        // Return the receipt, not a projection. This used to build a
        // ConfigSnapshot from the REQUESTED config and hand it back as though
        // it were current, so a client was told the new configuration was in
        // effect when the command had only been admitted to a bounded queue. It
        // was worse than a plain projection: the requested config was stamped
        // with `current_tick`, the tick at which those values were NOT in
        // effect, making the response a chimera of a config that had not been
        // applied and a tick at which it had not been applied (bd-k7nq).
        //
        // Acceptance criterion 4 of the migration is explicit that reads use
        // immutable snapshots and never project future config, so the caller
        // now gets a command id it can poll and reads the configuration back
        // through /api/config when it wants the applied truth.
        let status =
            self.submit_control_command(ControlCommand::UpdateConfig(Box::new(new_config)))?;
        *lock_cache(&self.knobs_cache) = None;
        Ok(status)
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

    /// Submit an ApplyMap command.
    pub fn apply_map(
        &self,
        artifact: MapArtifact,
        idempotency_key: Option<&str>,
    ) -> Result<CommandStatusDto, ControlError> {
        self.submit_control_command_with_key(
            ControlCommand::ApplyMap(Box::new(artifact)),
            idempotency_key,
        )
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

    /// Generate a procedural map artifact deterministically using a rule-based generator.
    pub fn generate_map(
        &self,
        width: u32,
        height: u32,
        cell_size: Option<u32>,
        seed: u64,
        tileset: Option<TilesetSpec>,
    ) -> Result<MapArtifact, ControlError> {
        let spec = tileset.unwrap_or_else(default_tileset_spec);
        let generator = RuleBasedMapGenerator::new(spec)
            .map_err(|e| ControlError::InvalidPatch(format!("tileset compile error: {e}")))?;
        let cell_size = cell_size.unwrap_or(50);
        generator
            .generate(width, height, cell_size, seed)
            .map_err(|e| ControlError::InvalidPatch(format!("map generation error: {e}")))
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
}

fn insert_path(map: &mut Map<String, Value>, path: &str, value: Value) -> Result<(), ControlError> {
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
        worker: Option<std::thread::JoinHandle<()>>,
        clock_gate: Arc<Mutex<()>>,
        clock_blocked: std::sync::mpsc::Receiver<()>,
    }

    impl TestHost {
        pub(crate) fn spawn(world: WorldState) -> Self {
            let (send, receive) = std::sync::mpsc::sync_channel(1);
            let clock_gate = Arc::new(Mutex::new(()));
            let owner_gate = Arc::clone(&clock_gate);
            let (blocked, clock_blocked) = std::sync::mpsc::channel();
            let worker = std::thread::spawn(move || {
                use scriptbots_runtime::{
                    FixedDeadlineHost, HostCore, HostCoreOptions, HostSessionId, ManualInstant,
                    PlaybackSnapshot,
                    channel::{ChannelHostDriver, ChannelHostOptions},
                };
                let core = HostCore::new(
                    HostSessionId::new(0xc017),
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
                let (mut driver, port) = ChannelHostDriver::new(
                    FixedDeadlineHost::new(core),
                    ChannelHostOptions::default(),
                )
                .expect("channel owner");
                send.send(port).expect("publish test port");
                let epoch = std::time::Instant::now();
                driver
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

        pub(crate) fn wait_applied(&self, status: &CommandStatusDto) -> CommandStatusDto {
            let observed = self.wait_finished(&status.command_id);
            assert_eq!(observed.application_state, "applied");
            observed
        }

        fn wait_finished(&self, command_id: &str) -> CommandStatusDto {
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
                worker.join().expect("test host joined");
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
        host.worker
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
        host.worker
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
            .apply_map(artifact, None)
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
            .apply_map(bad_artifact, None)
            .expect("submit mismatched map");
        let observed_bad = host.wait_finished(&status_bad.command_id);
        assert_eq!(observed_bad.application_state, "failed");
    }
}
