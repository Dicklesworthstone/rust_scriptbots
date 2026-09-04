//! Offline science layer for `ScriptBots` (bd-2z0.11.5, program bd-2js6).
//!
//! This crate is the ONE blessed offline reader of finished run databases:
//! a report framework plus the `sb-analyze` CLI. Boundary rules it exists to
//! uphold (`docs/franken_integration.md` §4):
//!
//! - **Read-only**: all access goes through [`scriptbots_storage::StorageReader`],
//!   which exposes no mutating API. This crate never opens a writable
//!   connection and never competes with a live run's storage worker.
//! - **Native-only**: never a dependency of the app binaries and never part
//!   of any wasm graph (`ci/check_wasm_graph.sh` guard B enforces the
//!   reverse boundary).
//! - **Franken analytics adapters land here** (fsci-stats: bd-2z0.11.6,
//!   fnx graphs: bd-2z0.11.7, frankenpandas exports: bd-2z0.11.8) behind
//!   this crate's report registry — never in the tick path, never in core.
//! - **Export successor**: report coverage lands here before the app's direct-DB
//!   `control_cli Export` path is retired under bd-2z0.8.9.5.
//!
//! Every report execution is wrapped in a tracing span carrying the report
//! name, parameter set, row counts, and wall time, so detailed logging is a
//! property of the framework rather than a per-report afterthought.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::time::Instant;

use scriptbots_core::{AgentUid, Tick};
use scriptbots_storage::{
    PersistedAgentObservation, PersistedInteraction, PersistedInteractionCapture, PersistedMetric,
    PersistenceBatchId, StorageError, StorageReader,
};
use serde::Serialize;

pub use stats::StatsError;

/// Native, dependency-free statistics for offline detector certification (bd-2z0.11.6).
///
/// Bootstrap confidence intervals, permutation tests, and effect sizes as pure functions over
/// `&[f64]`. Implemented natively rather than via `fsci-stats` because the latter is git-only and
/// nightly-only, and the four estimators the certification actually needs require neither — the
/// module's calibration tests demonstrate the native path is sufficient, which is the evidence
/// bd-2z0.11.3's adapter decision consumes. Never enters core or any tick path.
pub mod stats;

/// Statistical certification of narrative events (bd-2z0.11.6, item 1).
///
/// Answers "is this detected event real, or the tail of noise?" for the events the detector
/// fires, with Benjamini-Hochberg false-discovery-rate control across a whole run — the
/// principled replacement for eyeballed per-event thresholds that bd-16g.2.3's false-positive
/// budget needs. Pure functions over a metric series; the report that reads real `EventRecord`s
/// from a database is a thin adapter on top.
pub mod certify;

/// Matched-seed treatment-effect analysis (bd-2z0.11.6 item 3; serves bd-16g.1.4).
///
/// Given the same seeds run under a control and a treatment, measures whether the treatment
/// changed each metric — with a paired design (sign-flip permutation, paired bootstrap CI,
/// Cohen's `d_z`) that exploits the matched seeds, and Benjamini-Hochberg across metrics so a
/// study measuring many outcomes cannot report a chance "effect". Pure functions; the DB glue
/// that pulls per-seed outcomes from two run databases is a thin adapter on top.
pub mod compare;

/// Single change-point detection over a metric series (bd-2z0.11.6).
///
/// Finds the split that maximizes the absolute mean shift in a metric — the "if this run had one
/// regime shift, where was it?" question — which the `metric-changepoints` report then certifies
/// via [`certify`]. Pure; the certification that consumes it is where the resampling lives.
pub mod changepoint;

/// Native distribution characterization (bd-2z0.11.6 item 2).
///
/// Moment-based shape summary — skewness, kurtosis, and the Jarque-Bera normality test with an
/// exact chi-square(2) p-value — so "is this metric normal, and how is it shaped?" is answered
/// natively, with no `erf` and no `fsci` dependency. Full distribution fitting (lognormal/gamma +
/// KS) is left for the adapter decision (bd-2z0.11.3). Pure functions over a slice.
pub mod distribution;

/// Run-scoped lineage fitness, uncertainty, and evolutionary explanations (bd-2z0.11.10).
pub mod lineage;

/// FrankenPandas export and summary layer (bd-2z0.11.8).
pub mod dataframe;
pub use dataframe::*;

/// NetworkX-parity graph analysis and report layer (bd-2z0.11.7).
pub mod graphs;
pub use graphs::*;

pub use lineage::{
    EvolutionaryChangeExplanation, FounderLineageRecord, GenerationMetricRow,
    LINEAGE_FITNESS_SCHEMA_ID_V1, LineageFitness, LineageFitnessMachine, LineageLifespanSummary,
    LineageRunReconciliation,
};

/// Schema version stamped into every machine-readable report payload.
///
/// Bump ONLY with a migration note in the owning Bead/release evidence. Full
/// envelope goldens assert the value so an accidental schema change is loud.
pub const REPORT_SCHEMA_VERSION: u32 = 1;

/// Maximum metric rows sampled by `metric-summary` in one bounded SQL page.
///
/// Keeping report reads capped prevents a finished multi-run database from being
/// materialized wholesale merely to compute an interactive summary.
const METRIC_SUMMARY_ROW_LIMIT: usize = 4_096;

/// Maximum recent tick summaries sampled by `run-summary` in one bounded SQL page.
const RUN_SUMMARY_TICK_LIMIT: usize = 4_096;

/// Default number of recent replay events rendered by `narrative-timeline`.
const NARRATIVE_TIMELINE_DEFAULT_LIMIT: usize = 1_024;

/// Hard ceiling for a caller-selected `narrative-timeline` SQL page.
const NARRATIVE_TIMELINE_MAX_LIMIT: usize = 4_096;

/// Stable schema identifier shared by reports and downstream behavior consumers.
pub const PHENOTYPE_FEATURE_SCHEMA_ID_V1: &str = "scriptbots.phenotype-features.v1";
/// Stable schema identifier for the directed interaction graph.
pub const INTERACTION_GRAPH_SCHEMA_ID_V1: &str = "scriptbots.interaction-graph.v1";
/// Number and order of canonical phenotype axes.
pub const PHENOTYPE_AXIS_COUNT_V1: usize = 6;

/// Provenance class of one phenotype feature.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FeatureEvidenceV1 {
    /// Directly measured from persisted state snapshots.
    PersistedObservation,
    /// A persisted heritable trait used explicitly as a proxy, not realized behavior.
    PersistedTraitProxy,
    /// A completed directed interaction persisted by core.
    PersistedInteraction,
    /// A parent edge from the persisted arrival ledger.
    PersistedArrival,
}

/// Immutable definition of one ordered phenotype axis.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct PhenotypeAxisV1 {
    /// Stable machine identifier.
    pub id: &'static str,
    /// Human-readable unit.
    pub unit: &'static str,
    /// Declared value domain.
    pub domain: &'static str,
    /// Exact aggregation rule.
    pub aggregation: &'static str,
    /// What happens when evidence is absent.
    pub missingness: &'static str,
    /// Whether this is direct behavior, a trait proxy, an interaction, or lineage evidence.
    pub evidence: FeatureEvidenceV1,
}

/// Canonical six-axis phenotype schema.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct PhenotypeFeatureSchemaV1 {
    /// Stable schema identifier.
    pub schema_id: &'static str,
    /// Version within the identifier family.
    pub version: u16,
    /// Ordered axes; array position is part of the schema.
    pub axes: [PhenotypeAxisV1; PHENOTYPE_AXIS_COUNT_V1],
}

/// The canonical schema value used by every v1 extractor.
pub const PHENOTYPE_FEATURE_SCHEMA_V1: PhenotypeFeatureSchemaV1 = PhenotypeFeatureSchemaV1 {
    schema_id: PHENOTYPE_FEATURE_SCHEMA_ID_V1,
    version: 1,
    axes: [
        PhenotypeAxisV1 {
            id: "movement.speed.mean",
            unit: "world_unit_per_tick",
            domain: "finite_nonnegative",
            aggregation: "mean_hypot_velocity_over_persisted_observations",
            missingness: "reject_agent_without_observation",
            evidence: FeatureEvidenceV1::PersistedObservation,
        },
        PhenotypeAxisV1 {
            id: "diet.herbivore_trait.mean",
            unit: "ratio",
            domain: "finite",
            aggregation: "mean_persisted_herbivore_tendency",
            missingness: "reject_agent_without_observation",
            evidence: FeatureEvidenceV1::PersistedTraitProxy,
        },
        PhenotypeAxisV1 {
            id: "sensing.trait_modifier.mean",
            unit: "trait_multiplier",
            domain: "finite",
            aggregation: "mean_of_smell_sound_hearing_eye_blood_traits_over_observations",
            missingness: "reject_agent_without_observation",
            evidence: FeatureEvidenceV1::PersistedTraitProxy,
        },
        PhenotypeAxisV1 {
            id: "interaction.combat.actor_rate",
            unit: "event_per_tick",
            domain: "finite_nonnegative",
            aggregation: "completed_combat_events_as_actor_divided_by_window_ticks",
            missingness: "zero_only_when_run_wide_capture_is_complete",
            evidence: FeatureEvidenceV1::PersistedInteraction,
        },
        PhenotypeAxisV1 {
            id: "interaction.food_share.actor_rate",
            unit: "event_per_tick",
            domain: "finite_nonnegative",
            aggregation: "completed_food_share_events_as_actor_divided_by_window_ticks",
            missingness: "zero_only_when_run_wide_capture_is_complete",
            evidence: FeatureEvidenceV1::PersistedInteraction,
        },
        PhenotypeAxisV1 {
            id: "reproduction.parent_rate",
            unit: "offspring_per_tick",
            domain: "finite_nonnegative",
            aggregation: "distinct_parent_edges_divided_by_window_ticks",
            missingness: "zero_when_no_persisted_child_edge",
            evidence: FeatureEvidenceV1::PersistedArrival,
        },
    ],
};

impl PhenotypeFeatureSchemaV1 {
    /// BLAKE3 digest of the canonical JSON schema bytes.
    pub fn digest(self) -> Result<String, PhenotypeExtractionError> {
        canonical_json_digest(&self)
    }
}

/// Half-open simulation tick window.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct PhenotypeTickWindowV1 {
    /// Inclusive start tick of the window.
    pub start_tick: u64,
    /// Exclusive end tick of the window.
    pub end_tick: u64,
}

impl PhenotypeTickWindowV1 {
    const fn validate(self) -> Result<(), PhenotypeExtractionError> {
        if self.start_tick >= self.end_tick {
            return Err(PhenotypeExtractionError::InvalidWindow {
                start_tick: self.start_tick,
                end_tick: self.end_tick,
            });
        }
        Ok(())
    }

    const fn duration(self) -> u64 {
        self.end_tick - self.start_tick
    }

    const fn contains(self, tick: u64) -> bool {
        self.start_tick <= tick && tick < self.end_tick
    }
}

/// Typed refusal from the canonical phenotype/interaction extractor.
#[allow(missing_docs)]
#[derive(Debug, thiserror::Error)]
pub enum PhenotypeExtractionError {
    #[error("invalid empty or reversed tick window [{start_tick}, {end_tick})")]
    InvalidWindow { start_tick: u64, end_tick: u64 },
    #[error("schema id mismatch: expected {expected}, found {found}")]
    SchemaIdMismatch { expected: String, found: String },
    #[error("schema digest mismatch: expected {expected}, found {found}")]
    SchemaDigestMismatch { expected: String, found: String },
    #[error("{source_name} row {index} belongs to run {found}, expected {expected}")]
    CrossRunSource {
        source_name: &'static str,
        index: usize,
        expected: String,
        found: String,
    },
    #[error("duplicate observation for uid {agent_uid:?} at tick {tick}")]
    DuplicateObservation { agent_uid: AgentUid, tick: u64 },
    #[error("duplicate interaction source key at tick {tick}, seq {seq}")]
    DuplicateInteraction { tick: u64, seq: u64 },
    #[error("duplicate arrival identity {agent_uid:?}")]
    DuplicateArrival { agent_uid: AgentUid },
    #[error("{source_name} references unknown agent identity {agent_uid:?}")]
    MissingAgentIdentity {
        source_name: &'static str,
        agent_uid: AgentUid,
    },
    #[error("non-finite {field} for uid {agent_uid:?} at tick {tick}")]
    NonFinite {
        field: &'static str,
        agent_uid: AgentUid,
        tick: u64,
    },
    #[error("interaction at tick {tick}, seq {seq} has no finite positive magnitude")]
    InvalidInteractionMagnitude { tick: u64, seq: u64 },
    #[error("interaction at tick {tick}, seq {seq} targets its actor {agent_uid:?}")]
    SelfInteraction {
        tick: u64,
        seq: u64,
        agent_uid: AgentUid,
    },
    #[error("unsupported interaction kind {kind:?} at tick {tick}, seq {seq}")]
    UnsupportedInteractionKind { tick: u64, seq: u64, kind: String },
    #[error("{source_name} tick {tick} is outside [{start_tick}, {end_tick})")]
    TickOutsideWindow {
        source_name: &'static str,
        tick: u64,
        start_tick: u64,
        end_tick: u64,
    },
    #[error(
        "interaction evidence is incomplete: observed={observed}, persisted={persisted}, sampled_out={sampled_out}, truncated={truncated}"
    )]
    IncompleteInteractionEvidence {
        observed: u64,
        persisted: u64,
        sampled_out: u64,
        truncated: u64,
    },
    #[error("interaction row count {rows} does not match certified persisted count {persisted}")]
    InteractionCountMismatch { rows: usize, persisted: u64 },
    #[error("no agent observations exist in [{start_tick}, {end_tick})")]
    InsufficientWindow { start_tick: u64, end_tick: u64 },
    #[error("canonical JSON serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("counter overflow while accumulating {0}")]
    CounterOverflow(&'static str),
}

/// One run-tagged persisted state observation.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RunScopedAgentObservationV1 {
    /// Source run identifier.
    pub run_id: String,
    /// Simulation tick.
    pub tick: u64,
    /// Stable agent identity.
    pub agent_uid: AgentUid,
    /// X velocity component.
    pub velocity_x: f64,
    /// Y velocity component.
    pub velocity_y: f64,
    /// Trait proxy, not realized diet.
    pub herbivore_tendency: f64,
    /// Trait proxy: smell.
    pub trait_smell: f64,
    /// Trait proxy: sound.
    pub trait_sound: f64,
    /// Trait proxy: hearing.
    pub trait_hearing: f64,
    /// Trait proxy: eye.
    pub trait_eye: f64,
    /// Trait proxy: blood.
    pub trait_blood: f64,
}

/// One run-tagged canonical directed interaction.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RunScopedInteractionV1 {
    /// Source run identifier.
    pub run_id: String,
    /// Simulation tick.
    pub tick: u64,
    /// Monotonic sequence within the tick.
    pub seq: u64,
    /// String interaction category.
    pub kind: String,
    /// Acting agent identity.
    pub actor: AgentUid,
    /// Target agent identity.
    pub target: AgentUid,
    /// Interaction magnitude if defined.
    pub magnitude: Option<f64>,
}

/// One run-tagged stable-identity arrival/parent record.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RunScopedArrivalV1 {
    /// Source run identifier.
    pub run_id: String,
    /// Arrival simulation tick.
    pub tick: u64,
    /// Stable agent identity.
    pub agent_uid: AgentUid,
    /// First parent identity if born.
    pub parent_a: Option<AgentUid>,
    /// Second parent identity if sexually reproduced.
    pub parent_b: Option<AgentUid>,
}

/// Run-tagged interaction-capture accounting.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RunScopedInteractionCaptureV1 {
    /// Source run identifier.
    pub run_id: String,
    /// Total interaction events observed.
    pub observed: u64,
    /// Total interaction events persisted.
    pub persisted: u64,
    /// Total interaction events dropped by sampling.
    pub sampled_out: u64,
    /// Total interaction events truncated by capacity.
    pub truncated: u64,
}

/// Immutable source ledger consumed by the pure extractor.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PhenotypeLedgerV1 {
    /// Canonical schema identifier.
    pub schema_id: String,
    /// Blake3 digest of the feature schema.
    pub schema_digest: String,
    /// Blake3 digest of the simulation configuration.
    pub config_digest: String,
    /// Source run identifier.
    pub run_id: String,
    /// Bounded analysis tick window.
    pub window: PhenotypeTickWindowV1,
    /// Ordered run-scoped observations.
    pub observations: Vec<RunScopedAgentObservationV1>,
    /// Ordered run-scoped interactions.
    pub interactions: Vec<RunScopedInteractionV1>,
    /// Ordered ancestry arrivals.
    pub arrivals: Vec<RunScopedArrivalV1>,
    /// Certified interaction completeness counters.
    pub interaction_capture: RunScopedInteractionCaptureV1,
}

/// Directed interaction category retained by the v1 graph.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum InteractionEdgeKindV1 {
    /// Directed combat event.
    Combat,
    /// Directed food sharing event.
    FoodShare,
}

/// Canonical aggregate of directed source events.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DirectedInteractionEdgeV1 {
    /// Acting agent identity.
    pub actor: AgentUid,
    /// Target agent identity.
    pub target: AgentUid,
    /// Interaction classification.
    pub kind: InteractionEdgeKindV1,
    /// Total events along this directed edge.
    pub event_count: u64,
    /// Sum of positive event magnitudes.
    pub magnitude_sum: f64,
    /// Earliest tick observed.
    pub first_tick: u64,
    /// Latest tick observed.
    pub last_tick: u64,
}

/// Permutation-invariant directed graph for one run and window.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DirectedInteractionGraphV1 {
    /// Schema identifier.
    pub schema_id: &'static str,
    /// Source run identifier.
    pub run_id: String,
    /// Tick window.
    pub window: PhenotypeTickWindowV1,
    /// Certified interaction capture counters.
    pub capture: RunScopedInteractionCaptureV1,
    /// Sorted unique agent identities.
    pub nodes: Vec<AgentUid>,
    /// Canonical directed interaction edges.
    pub edges: Vec<DirectedInteractionEdgeV1>,
    /// Blake3 digest of the graph payload.
    pub canonical_digest: String,
}

/// One canonical ordered phenotype vector.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PhenotypeFeatureRowV1 {
    /// Source run identifier.
    pub run_id: String,
    /// Stable agent identity.
    pub agent_uid: AgentUid,
    /// Number of ticks this agent was observed.
    pub observed_tick_count: u64,
    /// Values follow [`PHENOTYPE_FEATURE_SCHEMA_V1`] exactly.
    pub values: [f64; PHENOTYPE_AXIS_COUNT_V1],
}

impl From<&PhenotypeFeatureRowV1> for scriptbots_core::species::AgentPhenotypeVector {
    fn from(row: &PhenotypeFeatureRowV1) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self {
            agent_uid: row.agent_uid,
            movement_speed_mean: row.values[0] as f32,
            diet_herbivore_ratio: row.values[1] as f32,
            sensing_range_mean: row.values[2] as f32,
            aggression_index: row.values[3] as f32,
            giving_altruism_index: row.values[4] as f32,
            reproduction_rate: row.values[5] as f32,
        }
    }
}

impl PhenotypeFeatureRowV1 {
    /// Converts this row to a core [`scriptbots_core::species::AgentPhenotypeVector`].
    ///
    /// This establishes exact schema parity with the species segmentation pipeline (bd-2z0.11.2, bd-16g.3.6).
    #[must_use]
    pub fn to_agent_phenotype_vector(&self) -> scriptbots_core::species::AgentPhenotypeVector {
        self.into()
    }

    /// Extracts the canonical behavioral descriptor for MAP-Elites archive binning (bd-2z0.11.2, bd-16g.6.1).
    #[must_use]
    pub const fn behavior_descriptor(&self) -> [f64; PHENOTYPE_AXIS_COUNT_V1] {
        self.values
    }
}

/// Production analysis read model derived from persisted run facts.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PhenotypeInteractionAnalysisV1 {
    /// Canonical schema definition.
    pub schema: PhenotypeFeatureSchemaV1,
    /// Digest of the schema definition.
    pub schema_digest: String,
    /// Digest of the simulation configuration.
    pub config_digest: String,
    /// Source run identifier.
    pub run_id: String,
    /// Analyzed tick window.
    pub window: PhenotypeTickWindowV1,
    /// Extracted phenotype feature rows.
    pub features: Vec<PhenotypeFeatureRowV1>,
    /// Directed interaction network.
    pub interaction_graph: DirectedInteractionGraphV1,
    /// Blake3 digest of the full analysis model.
    pub canonical_digest: String,
}

/// Errors surfaced by the analytics layer.
#[derive(Debug, thiserror::Error)]
pub enum AnalyticsError {
    /// The underlying read-only storage access failed.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
    /// The requested report is not registered.
    #[error("unknown report '{0}' (run `sb-analyze <db> list` for the registry)")]
    UnknownReport(String),
    /// A parameter failed to parse or validate.
    #[error("bad parameter '{name}': {reason}")]
    BadParam {
        /// Parameter key as supplied by the caller.
        name: String,
        /// Human-readable validation failure.
        reason: String,
    },
    /// Serialization of the machine payload failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Writing a requested report artifact failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Canonical phenotype extraction refused invalid or censored evidence.
    #[error("phenotype extraction error: {0}")]
    Phenotype(#[from] PhenotypeExtractionError),
    /// Rebuilding or querying the ancestry graph failed.
    #[error("ancestry error: {0}")]
    Ancestry(#[from] scriptbots_core::ancestry::AncestryError),
    /// Dataframe export or summary error.
    #[error("dataframe error: {0}")]
    DataFrame(#[from] crate::dataframe::DataFrameError),
    /// Graph construction, algorithm, or export error (bd-2z0.11.7).
    #[error("graph error: {0}")]
    Graph(String),
}

/// Read-only context handed to every report.
pub struct ReaderCtx {
    /// Open read-only handle over the finished run database.
    pub reader: StorageReader,
    /// Path of the database, for provenance stamping in outputs.
    pub db_path: String,
}

impl ReaderCtx {
    /// Opens a finished run database read-only.
    ///
    /// Fails (rather than creating anything) when the path does not exist —
    /// asserted by the scaffold tests as the read-only contract. The finished-run
    /// lease rejects a live writer and remains held for every report query.
    pub fn open(db_path: &str) -> Result<Self, AnalyticsError> {
        let reader = StorageReader::open_finished(db_path)?;
        Ok(Self {
            reader,
            db_path: db_path.to_owned(),
        })
    }
}

/// Load the complete immutable run ledger needed by the v1 extractor.
///
/// The production adapter intentionally selects the whole persisted run. Core
/// records interaction-completeness counters at enclosing persistence
/// boundaries while edges retain their source ticks, so a sub-window cannot be
/// certified from the current schema without fabricating precision.
pub fn load_persisted_phenotype_ledger(
    reader: &StorageReader,
) -> Result<PhenotypeLedgerV1, AnalyticsError> {
    let max_tick = reader
        .max_tick()?
        .ok_or(PhenotypeExtractionError::InsufficientWindow {
            start_tick: 0,
            end_tick: 0,
        })?;
    let end_tick = max_tick
        .checked_add(1)
        .ok_or(PhenotypeExtractionError::CounterOverflow(
            "analysis end tick",
        ))?;
    let window = PhenotypeTickWindowV1 {
        start_tick: 0,
        end_tick,
    };
    let run_id = reader.run_id().to_string();
    let manifest = reader.run_manifest()?;
    let schema_digest = PHENOTYPE_FEATURE_SCHEMA_V1.digest()?;

    let observations = reader
        .load_agent_observations(window.start_tick, window.end_tick)?
        .into_iter()
        .map(|observation| run_scoped_observation(&run_id, &observation))
        .collect();
    let interactions = reader
        .load_interactions_window(window.start_tick, window.end_tick)?
        .into_iter()
        .map(|interaction| run_scoped_interaction(&run_id, &interaction))
        .collect();
    let arrivals = reader
        .load_ancestry_births()?
        .into_iter()
        .filter(|arrival| arrival.tick.0 < window.end_tick)
        .map(|arrival| RunScopedArrivalV1 {
            run_id: run_id.clone(),
            tick: arrival.tick.0,
            agent_uid: arrival.agent_uid,
            parent_a: arrival.parent_a,
            parent_b: arrival.parent_b,
        })
        .collect();
    let PersistedInteractionCapture {
        observed,
        persisted,
        sampled_out,
        truncated,
    } = reader.load_interaction_capture()?;

    Ok(PhenotypeLedgerV1 {
        schema_id: PHENOTYPE_FEATURE_SCHEMA_ID_V1.to_owned(),
        schema_digest,
        config_digest: manifest.config_digest,
        run_id: run_id.clone(),
        window,
        observations,
        interactions,
        arrivals,
        interaction_capture: RunScopedInteractionCaptureV1 {
            run_id,
            observed,
            persisted,
            sampled_out,
            truncated,
        },
    })
}

fn run_scoped_observation(
    run_id: &str,
    observation: &PersistedAgentObservation,
) -> RunScopedAgentObservationV1 {
    RunScopedAgentObservationV1 {
        run_id: run_id.to_owned(),
        tick: observation.tick,
        agent_uid: observation.agent_uid,
        velocity_x: observation.velocity_x,
        velocity_y: observation.velocity_y,
        herbivore_tendency: observation.herbivore_tendency,
        trait_smell: observation.trait_smell,
        trait_sound: observation.trait_sound,
        trait_hearing: observation.trait_hearing,
        trait_eye: observation.trait_eye,
        trait_blood: observation.trait_blood,
    }
}

fn run_scoped_interaction(
    run_id: &str,
    interaction: &PersistedInteraction,
) -> RunScopedInteractionV1 {
    RunScopedInteractionV1 {
        run_id: run_id.to_owned(),
        tick: interaction.tick,
        seq: interaction.seq,
        kind: interaction.kind.clone(),
        actor: interaction.actor,
        target: interaction.target,
        magnitude: interaction.value,
    }
}

#[derive(Debug, Default)]
struct FeatureAccumulator {
    observations: u64,
    movement_sum: f64,
    herbivore_trait_sum: f64,
    sensing_trait_sum: f64,
    combat_events: u64,
    food_share_events: u64,
    offspring_edges: u64,
}

#[derive(Debug)]
struct EdgeAccumulator {
    event_count: u64,
    magnitude_sum: f64,
    first_tick: u64,
    last_tick: u64,
}

fn validate_row_run(
    expected: &str,
    found: &str,
    source_name: &'static str,
    index: usize,
) -> Result<(), PhenotypeExtractionError> {
    if found == expected {
        Ok(())
    } else {
        Err(PhenotypeExtractionError::CrossRunSource {
            source_name,
            index,
            expected: expected.to_owned(),
            found: found.to_owned(),
        })
    }
}

const fn require_finite(
    value: f64,
    field: &'static str,
    observation: &RunScopedAgentObservationV1,
) -> Result<f64, PhenotypeExtractionError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PhenotypeExtractionError::NonFinite {
            field,
            agent_uid: observation.agent_uid,
            tick: observation.tick,
        })
    }
}

fn interaction_kind(
    interaction: &RunScopedInteractionV1,
) -> Result<InteractionEdgeKindV1, PhenotypeExtractionError> {
    match interaction.kind.as_str() {
        "combat" => Ok(InteractionEdgeKindV1::Combat),
        "food_share" => Ok(InteractionEdgeKindV1::FoodShare),
        _ => Err(PhenotypeExtractionError::UnsupportedInteractionKind {
            tick: interaction.tick,
            seq: interaction.seq,
            kind: interaction.kind.clone(),
        }),
    }
}

fn checked_increment(
    value: &mut u64,
    context: &'static str,
) -> Result<(), PhenotypeExtractionError> {
    *value = value
        .checked_add(1)
        .ok_or(PhenotypeExtractionError::CounterOverflow(context))?;
    Ok(())
}

fn canonical_json_digest<T: Serialize>(value: &T) -> Result<String, PhenotypeExtractionError> {
    let bytes = serde_json::to_vec(value)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Derive canonical finite feature rows and a directed interaction graph.
///
/// This function is pure: callers may use hand-audited fixtures, while
/// [`load_persisted_phenotype_ledger`] is the production storage adapter.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn extract_phenotype_interactions(
    ledger: &PhenotypeLedgerV1,
) -> Result<PhenotypeInteractionAnalysisV1, PhenotypeExtractionError> {
    ledger.window.validate()?;
    if ledger.schema_id != PHENOTYPE_FEATURE_SCHEMA_ID_V1 {
        return Err(PhenotypeExtractionError::SchemaIdMismatch {
            expected: PHENOTYPE_FEATURE_SCHEMA_ID_V1.to_owned(),
            found: ledger.schema_id.clone(),
        });
    }
    let expected_schema_digest = PHENOTYPE_FEATURE_SCHEMA_V1.digest()?;
    if ledger.schema_digest != expected_schema_digest {
        return Err(PhenotypeExtractionError::SchemaDigestMismatch {
            expected: expected_schema_digest,
            found: ledger.schema_digest.clone(),
        });
    }

    validate_row_run(
        &ledger.run_id,
        &ledger.interaction_capture.run_id,
        "interaction_capture",
        0,
    )?;
    let capture = &ledger.interaction_capture;
    let accounted = capture
        .persisted
        .checked_add(capture.sampled_out)
        .and_then(|value| value.checked_add(capture.truncated))
        .ok_or(PhenotypeExtractionError::CounterOverflow(
            "interaction capture",
        ))?;
    if capture.observed != accounted || capture.sampled_out != 0 || capture.truncated != 0 {
        return Err(PhenotypeExtractionError::IncompleteInteractionEvidence {
            observed: capture.observed,
            persisted: capture.persisted,
            sampled_out: capture.sampled_out,
            truncated: capture.truncated,
        });
    }
    let interaction_rows = u64::try_from(ledger.interactions.len())
        .map_err(|_| PhenotypeExtractionError::CounterOverflow("interaction row count"))?;
    if interaction_rows != capture.persisted {
        return Err(PhenotypeExtractionError::InteractionCountMismatch {
            rows: ledger.interactions.len(),
            persisted: capture.persisted,
        });
    }

    let mut arrivals = ledger.arrivals.clone();
    arrivals.sort_by_key(|arrival| (arrival.tick, arrival.agent_uid));
    let mut known_agents = BTreeSet::new();
    for (index, arrival) in arrivals.iter().enumerate() {
        validate_row_run(&ledger.run_id, &arrival.run_id, "arrival", index)?;
        if arrival.tick >= ledger.window.end_tick {
            return Err(PhenotypeExtractionError::TickOutsideWindow {
                source_name: "arrival",
                tick: arrival.tick,
                start_tick: 0,
                end_tick: ledger.window.end_tick,
            });
        }
        if !known_agents.insert(arrival.agent_uid) {
            return Err(PhenotypeExtractionError::DuplicateArrival {
                agent_uid: arrival.agent_uid,
            });
        }
    }

    let mut observations = ledger.observations.clone();
    observations.sort_by_key(|observation| (observation.agent_uid, observation.tick));
    let mut previous_observation = None;
    let mut feature_accumulators = BTreeMap::<AgentUid, FeatureAccumulator>::new();
    for (index, observation) in observations.iter().enumerate() {
        validate_row_run(&ledger.run_id, &observation.run_id, "observation", index)?;
        if !ledger.window.contains(observation.tick) {
            return Err(PhenotypeExtractionError::TickOutsideWindow {
                source_name: "observation",
                tick: observation.tick,
                start_tick: ledger.window.start_tick,
                end_tick: ledger.window.end_tick,
            });
        }
        if previous_observation == Some((observation.agent_uid, observation.tick)) {
            return Err(PhenotypeExtractionError::DuplicateObservation {
                agent_uid: observation.agent_uid,
                tick: observation.tick,
            });
        }
        previous_observation = Some((observation.agent_uid, observation.tick));
        if !known_agents.contains(&observation.agent_uid) {
            return Err(PhenotypeExtractionError::MissingAgentIdentity {
                source_name: "observation",
                agent_uid: observation.agent_uid,
            });
        }

        let velocity_x = require_finite(observation.velocity_x, "velocity_x", observation)?;
        let velocity_y = require_finite(observation.velocity_y, "velocity_y", observation)?;
        let movement = velocity_x.hypot(velocity_y);
        let herbivore = require_finite(
            observation.herbivore_tendency,
            "herbivore_tendency",
            observation,
        )?;
        let sensing = [
            ("trait_smell", observation.trait_smell),
            ("trait_sound", observation.trait_sound),
            ("trait_hearing", observation.trait_hearing),
            ("trait_eye", observation.trait_eye),
            ("trait_blood", observation.trait_blood),
        ]
        .into_iter()
        .try_fold(0.0, |sum, (field, value)| {
            require_finite(value, field, observation).map(|value| sum + value)
        })? / 5.0;
        if !movement.is_finite() || !sensing.is_finite() {
            return Err(PhenotypeExtractionError::NonFinite {
                field: "derived_observation",
                agent_uid: observation.agent_uid,
                tick: observation.tick,
            });
        }
        let accumulator = feature_accumulators
            .entry(observation.agent_uid)
            .or_default();
        checked_increment(&mut accumulator.observations, "observation count")?;
        accumulator.movement_sum += movement;
        accumulator.herbivore_trait_sum += herbivore;
        accumulator.sensing_trait_sum += sensing;
    }
    if feature_accumulators.is_empty() {
        return Err(PhenotypeExtractionError::InsufficientWindow {
            start_tick: ledger.window.start_tick,
            end_tick: ledger.window.end_tick,
        });
    }

    let mut interactions = ledger.interactions.clone();
    interactions.sort_by_key(|interaction| (interaction.tick, interaction.seq));
    let mut previous_interaction = None;
    let mut graph_edges =
        BTreeMap::<(AgentUid, AgentUid, InteractionEdgeKindV1), EdgeAccumulator>::new();
    for (index, interaction) in interactions.iter().enumerate() {
        validate_row_run(&ledger.run_id, &interaction.run_id, "interaction", index)?;
        if !ledger.window.contains(interaction.tick) {
            return Err(PhenotypeExtractionError::TickOutsideWindow {
                source_name: "interaction",
                tick: interaction.tick,
                start_tick: ledger.window.start_tick,
                end_tick: ledger.window.end_tick,
            });
        }
        if previous_interaction == Some((interaction.tick, interaction.seq)) {
            return Err(PhenotypeExtractionError::DuplicateInteraction {
                tick: interaction.tick,
                seq: interaction.seq,
            });
        }
        previous_interaction = Some((interaction.tick, interaction.seq));
        if interaction.actor == interaction.target {
            return Err(PhenotypeExtractionError::SelfInteraction {
                tick: interaction.tick,
                seq: interaction.seq,
                agent_uid: interaction.actor,
            });
        }
        for (source_name, agent_uid) in [
            ("interaction.actor", interaction.actor),
            ("interaction.target", interaction.target),
        ] {
            if !known_agents.contains(&agent_uid) {
                return Err(PhenotypeExtractionError::MissingAgentIdentity {
                    source_name,
                    agent_uid,
                });
            }
        }
        let magnitude = interaction
            .magnitude
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or(PhenotypeExtractionError::InvalidInteractionMagnitude {
                tick: interaction.tick,
                seq: interaction.seq,
            })?;
        let kind = interaction_kind(interaction)?;
        if let Some(accumulator) = feature_accumulators.get_mut(&interaction.actor) {
            match kind {
                InteractionEdgeKindV1::Combat => {
                    checked_increment(&mut accumulator.combat_events, "combat event count")?;
                }
                InteractionEdgeKindV1::FoodShare => {
                    checked_increment(
                        &mut accumulator.food_share_events,
                        "food-share event count",
                    )?;
                }
            }
        }
        let edge = graph_edges
            .entry((interaction.actor, interaction.target, kind))
            .or_insert(EdgeAccumulator {
                event_count: 0,
                magnitude_sum: 0.0,
                first_tick: interaction.tick,
                last_tick: interaction.tick,
            });
        checked_increment(&mut edge.event_count, "graph edge event count")?;
        edge.magnitude_sum += magnitude;
        if !edge.magnitude_sum.is_finite() {
            return Err(PhenotypeExtractionError::InvalidInteractionMagnitude {
                tick: interaction.tick,
                seq: interaction.seq,
            });
        }
        edge.first_tick = edge.first_tick.min(interaction.tick);
        edge.last_tick = edge.last_tick.max(interaction.tick);
    }

    for arrival in arrivals
        .iter()
        .filter(|arrival| ledger.window.contains(arrival.tick))
    {
        let mut parents = BTreeSet::new();
        parents.extend(arrival.parent_a);
        parents.extend(arrival.parent_b);
        for parent in parents {
            if !known_agents.contains(&parent) {
                return Err(PhenotypeExtractionError::MissingAgentIdentity {
                    source_name: "arrival.parent",
                    agent_uid: parent,
                });
            }
            if let Some(accumulator) = feature_accumulators.get_mut(&parent) {
                checked_increment(&mut accumulator.offspring_edges, "offspring edge count")?;
            }
        }
    }

    let window_ticks = ledger.window.duration() as f64;
    let mut features = Vec::with_capacity(feature_accumulators.len());
    for (agent_uid, accumulator) in feature_accumulators {
        let observations = accumulator.observations as f64;
        let values = [
            accumulator.movement_sum / observations,
            accumulator.herbivore_trait_sum / observations,
            accumulator.sensing_trait_sum / observations,
            accumulator.combat_events as f64 / window_ticks,
            accumulator.food_share_events as f64 / window_ticks,
            accumulator.offspring_edges as f64 / window_ticks,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(PhenotypeExtractionError::NonFinite {
                field: "feature_vector",
                agent_uid,
                tick: ledger.window.end_tick - 1,
            });
        }
        features.push(PhenotypeFeatureRowV1 {
            run_id: ledger.run_id.clone(),
            agent_uid,
            observed_tick_count: accumulator.observations,
            values,
        });
    }

    let edges = graph_edges
        .into_iter()
        .map(
            |((actor, target, kind), accumulator)| DirectedInteractionEdgeV1 {
                actor,
                target,
                kind,
                event_count: accumulator.event_count,
                magnitude_sum: accumulator.magnitude_sum,
                first_tick: accumulator.first_tick,
                last_tick: accumulator.last_tick,
            },
        )
        .collect::<Vec<_>>();
    let nodes = known_agents.into_iter().collect::<Vec<_>>();
    let graph_capture = ledger.interaction_capture.clone();
    let graph_digest = canonical_json_digest(&(
        INTERACTION_GRAPH_SCHEMA_ID_V1,
        &ledger.run_id,
        ledger.window,
        &graph_capture,
        &nodes,
        &edges,
    ))?;
    let interaction_graph = DirectedInteractionGraphV1 {
        schema_id: INTERACTION_GRAPH_SCHEMA_ID_V1,
        run_id: ledger.run_id.clone(),
        window: ledger.window,
        capture: graph_capture,
        nodes,
        edges,
        canonical_digest: graph_digest,
    };
    let canonical_digest = canonical_json_digest(&(
        PHENOTYPE_FEATURE_SCHEMA_ID_V1,
        &ledger.schema_digest,
        &ledger.config_digest,
        &ledger.run_id,
        ledger.window,
        &features,
        &interaction_graph,
    ))?;

    Ok(PhenotypeInteractionAnalysisV1 {
        schema: PHENOTYPE_FEATURE_SCHEMA_V1,
        schema_digest: ledger.schema_digest.clone(),
        config_digest: ledger.config_digest.clone(),
        run_id: ledger.run_id.clone(),
        window: ledger.window,
        features,
        interaction_graph,
        canonical_digest,
    })
}

/// Load and extract canonical phenotype/interaction analysis from a finished run.
pub fn analyze_persisted_phenotypes(
    reader: &StorageReader,
) -> Result<PhenotypeInteractionAnalysisV1, AnalyticsError> {
    let ledger = load_persisted_phenotype_ledger(reader)?;
    Ok(extract_phenotype_interactions(&ledger)?)
}

/// Single-feature comparison between two cohorts with effect size and bootstrap uncertainty.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FeatureCohortComparison {
    /// Stable axis name.
    pub feature_name: &'static str,
    /// Physical unit.
    pub unit: &'static str,
    /// Mean value for cohort A.
    pub mean_a: f64,
    /// Mean value for cohort B.
    pub mean_b: f64,
    /// Standard deviation for cohort A.
    pub std_dev_a: f64,
    /// Standard deviation for cohort B.
    pub std_dev_b: f64,
    /// Cohen's d standardized mean difference (B minus A).
    pub cohens_d: f64,
    /// Hedges' g small-sample-corrected effect size.
    pub hedges_g: f64,
    /// Cliff's delta non-parametric effect size.
    pub cliffs_delta: f64,
    /// 95% bootstrap confidence interval of the mean difference (B minus A).
    pub mean_diff_ci_95: Option<[f64; 2]>,
    /// Two-sided permutation test p-value for the mean difference.
    pub permutation_p_value: Option<f64>,
}

/// Comprehensive statistical comparison between two cohorts of phenotypes (bd-2z0.11.2).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PhenotypeCohortComparisonReport {
    /// Label for cohort A.
    pub cohort_a_name: String,
    /// Label for cohort B.
    pub cohort_b_name: String,
    /// Number of agents in cohort A.
    pub sample_size_a: usize,
    /// Number of agents in cohort B.
    pub sample_size_b: usize,
    /// Per-axis statistical comparisons with effect sizes and uncertainty.
    pub feature_comparisons: Vec<FeatureCohortComparison>,
    /// Multivariate Mahalanobis distance and covariance status.
    pub multivariate_cluster_comparison: scriptbots_core::species::PhenotypeClusterComparison,
}

/// Parent-to-offspring phenotype shift along canonical axes (bd-2z0.11.2).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LineagePhenotypeShift {
    /// Parent agent identity.
    pub parent_uid: AgentUid,
    /// Offspring agent identity.
    pub child_uid: AgentUid,
    /// Offspring feature value minus parent feature value along each canonical axis.
    pub deltas: [f64; PHENOTYPE_AXIS_COUNT_V1],
}

/// Lineage phenotype shift summary across a population.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LineagePhenotypeShiftSummary {
    /// Number of parent-offspring pairs observed.
    pub pair_count: usize,
    /// Mean feature shifts (child minus parent).
    pub mean_deltas: [f64; PHENOTYPE_AXIS_COUNT_V1],
    /// Standard deviation of feature shifts.
    pub std_dev_deltas: [f64; PHENOTYPE_AXIS_COUNT_V1],
}

/// Compares two phenotype cohorts across canonical axes with effect sizes and bootstrap uncertainty (bd-2z0.11.2).
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
pub fn compare_phenotype_cohorts(
    cohort_a_name: &str,
    cohort_a: &[PhenotypeFeatureRowV1],
    cohort_b_name: &str,
    cohort_b: &[PhenotypeFeatureRowV1],
    seed: u64,
) -> Result<PhenotypeCohortComparisonReport, stats::StatsError> {
    let mut feature_comparisons = Vec::with_capacity(PHENOTYPE_AXIS_COUNT_V1);

    let vectors_a: Vec<scriptbots_core::species::AgentPhenotypeVector> = cohort_a
        .iter()
        .map(PhenotypeFeatureRowV1::to_agent_phenotype_vector)
        .collect();
    let vectors_b: Vec<scriptbots_core::species::AgentPhenotypeVector> = cohort_b
        .iter()
        .map(PhenotypeFeatureRowV1::to_agent_phenotype_vector)
        .collect();

    let multivariate_cluster_comparison = scriptbots_core::species::compare_phenotype_clusters(
        cohort_a_name,
        &vectors_a,
        cohort_b_name,
        &vectors_b,
    );

    for axis_idx in 0..PHENOTYPE_AXIS_COUNT_V1 {
        let axis = &PHENOTYPE_FEATURE_SCHEMA_V1.axes[axis_idx];
        let values_a: Vec<f64> = cohort_a.iter().map(|r| r.values[axis_idx]).collect();
        let values_b: Vec<f64> = cohort_b.iter().map(|r| r.values[axis_idx]).collect();

        if values_a.is_empty() || values_b.is_empty() {
            feature_comparisons.push(FeatureCohortComparison {
                feature_name: axis.id,
                unit: axis.unit,
                mean_a: if values_a.is_empty() {
                    0.0
                } else {
                    stats::mean(&values_a)?
                },
                mean_b: if values_b.is_empty() {
                    0.0
                } else {
                    stats::mean(&values_b)?
                },
                std_dev_a: if values_a.is_empty() {
                    0.0
                } else {
                    stats::std_dev(&values_a)?
                },
                std_dev_b: if values_b.is_empty() {
                    0.0
                } else {
                    stats::std_dev(&values_b)?
                },
                cohens_d: 0.0,
                hedges_g: 0.0,
                cliffs_delta: 0.0,
                mean_diff_ci_95: None,
                permutation_p_value: None,
            });
            continue;
        }

        let mean_a = stats::mean(&values_a)?;
        let mean_b = stats::mean(&values_b)?;
        let std_dev_a = stats::std_dev(&values_a)?;
        let std_dev_b = stats::std_dev(&values_b)?;
        let cohens_d = stats::cohens_d(&values_a, &values_b)?;
        let hedges_g = stats::hedges_g(&values_a, &values_b)?;
        let cliffs_delta = stats::cliffs_delta(&values_a, &values_b)?;

        // Resampling with deterministic seed offset by axis index
        let axis_seed = seed.wrapping_add((axis_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let ci =
            stats::bootstrap_mean_difference_ci(&values_a, &values_b, 1_000, 0.95, axis_seed).ok();
        let mean_diff_ci_95 = ci.map(|c| [c.lower, c.upper]);

        let perm = stats::permutation_test_mean_difference(
            &values_a,
            &values_b,
            1_000,
            axis_seed.wrapping_add(1),
        )
        .ok();
        let permutation_p_value = perm.map(|p| p.p_value);

        feature_comparisons.push(FeatureCohortComparison {
            feature_name: axis.id,
            unit: axis.unit,
            mean_a,
            mean_b,
            std_dev_a,
            std_dev_b,
            cohens_d,
            hedges_g,
            cliffs_delta,
            mean_diff_ci_95,
            permutation_p_value,
        });
    }

    Ok(PhenotypeCohortComparisonReport {
        cohort_a_name: cohort_a_name.to_owned(),
        cohort_b_name: cohort_b_name.to_owned(),
        sample_size_a: cohort_a.len(),
        sample_size_b: cohort_b.len(),
        feature_comparisons,
        multivariate_cluster_comparison,
    })
}

/// Clusters a cohort of phenotype feature rows into species using the canonical segmentation engine (bd-2z0.11.2, bd-16g.3.6).
#[must_use]
pub fn cluster_cohort_species(
    cohort: &[PhenotypeFeatureRowV1],
    params: &scriptbots_core::species::SpeciesParams,
    previous: &scriptbots_core::species::SpeciesTable,
    tick: Tick,
) -> (
    scriptbots_core::species::SpeciesTable,
    scriptbots_core::species::SegmentReport,
) {
    let samples: Vec<(AgentUid, Vec<f32>)> = cohort
        .iter()
        .map(|r| {
            (
                r.agent_uid,
                r.to_agent_phenotype_vector().features().to_vec(),
            )
        })
        .collect();
    scriptbots_core::species::segment_species(tick, &samples, previous, params)
}

/// Derives the core [`scriptbots_core::species::PhenotypeAnalysisReport`] across cohorts (bd-2z0.11.2).
#[must_use]
pub fn compute_core_phenotype_analysis(
    run_id: &str,
    tick: Tick,
    cohorts: &[(&str, &[PhenotypeFeatureRowV1])],
) -> scriptbots_core::species::PhenotypeAnalysisReport {
    let mut all_vectors = Vec::new();
    let converted_cohorts: Vec<(&str, Vec<scriptbots_core::species::AgentPhenotypeVector>)> =
        cohorts
            .iter()
            .map(|(name, rows)| {
                let vecs: Vec<scriptbots_core::species::AgentPhenotypeVector> = rows
                    .iter()
                    .map(PhenotypeFeatureRowV1::to_agent_phenotype_vector)
                    .collect();
                all_vectors.extend(vecs.clone());
                (*name, vecs)
            })
            .collect();

    let cohort_refs: Vec<(&str, &[scriptbots_core::species::AgentPhenotypeVector])> =
        converted_cohorts
            .iter()
            .map(|(name, vecs)| (*name, vecs.as_slice()))
            .collect();

    scriptbots_core::species::compute_phenotype_analysis(run_id, tick, &all_vectors, &cohort_refs)
}

/// Measures parent-to-offspring phenotype shifts without causal overclaim (bd-2z0.11.2).
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn compute_lineage_phenotype_shifts(
    features: &[PhenotypeFeatureRowV1],
    arrivals: &[RunScopedArrivalV1],
) -> (Vec<LineagePhenotypeShift>, LineagePhenotypeShiftSummary) {
    let feature_map: BTreeMap<AgentUid, &[f64; PHENOTYPE_AXIS_COUNT_V1]> =
        features.iter().map(|f| (f.agent_uid, &f.values)).collect();

    let mut shifts = Vec::new();

    for arrival in arrivals {
        let Some(&child_vals) = feature_map.get(&arrival.agent_uid) else {
            continue;
        };

        for parent_uid in [arrival.parent_a, arrival.parent_b].into_iter().flatten() {
            if let Some(&parent_vals) = feature_map.get(&parent_uid) {
                let mut deltas = [0.0; PHENOTYPE_AXIS_COUNT_V1];
                for i in 0..PHENOTYPE_AXIS_COUNT_V1 {
                    deltas[i] = child_vals[i] - parent_vals[i];
                }
                shifts.push(LineagePhenotypeShift {
                    parent_uid,
                    child_uid: arrival.agent_uid,
                    deltas,
                });
            }
        }
    }

    let pair_count = shifts.len();
    let mut mean_deltas = [0.0; PHENOTYPE_AXIS_COUNT_V1];
    let mut std_dev_deltas = [0.0; PHENOTYPE_AXIS_COUNT_V1];

    if pair_count > 0 {
        for i in 0..PHENOTYPE_AXIS_COUNT_V1 {
            let sum: f64 = shifts.iter().map(|s| s.deltas[i]).sum();
            mean_deltas[i] = sum / pair_count as f64;
            let sq_diff: f64 = shifts
                .iter()
                .map(|s| (s.deltas[i] - mean_deltas[i]).powi(2))
                .sum();
            std_dev_deltas[i] = (sq_diff / pair_count as f64).sqrt();
        }
    }

    let summary = LineagePhenotypeShiftSummary {
        pair_count,
        mean_deltas,
        std_dev_deltas,
    };

    (shifts, summary)
}

/// String-keyed report parameters with typed accessors.
#[derive(Debug, Default, Clone)]
pub struct ReportParams(BTreeMap<String, String>);

impl ReportParams {
    /// Builds parameters from `key=value` pairs, rejecting malformed input.
    pub fn from_pairs<I: IntoIterator<Item = String>>(pairs: I) -> Result<Self, AnalyticsError> {
        let mut map = BTreeMap::new();
        for pair in pairs {
            let Some((k, v)) = pair.split_once('=') else {
                return Err(AnalyticsError::BadParam {
                    name: pair,
                    reason: "expected key=value".into(),
                });
            };
            let key = k.trim();
            if key.is_empty() {
                return Err(AnalyticsError::BadParam {
                    name: pair,
                    reason: "parameter name must not be empty".into(),
                });
            }
            if map.insert(key.to_owned(), v.trim().to_owned()).is_some() {
                return Err(AnalyticsError::BadParam {
                    name: key.to_owned(),
                    reason: "parameter was supplied more than once".into(),
                });
            }
        }
        Ok(Self(map))
    }

    /// Raw string lookup.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    /// Parses an optional `usize` parameter.
    pub fn get_usize(&self, key: &str) -> Result<Option<usize>, AnalyticsError> {
        self.get(key)
            .map(|raw| {
                raw.parse::<usize>().map_err(|e| AnalyticsError::BadParam {
                    name: key.to_owned(),
                    reason: e.to_string(),
                })
            })
            .transpose()
    }

    /// Parses an optional `u64` parameter.
    pub fn get_u64(&self, key: &str) -> Result<Option<u64>, AnalyticsError> {
        self.get(key)
            .map(|raw| {
                raw.parse::<u64>().map_err(|e| AnalyticsError::BadParam {
                    name: key.to_owned(),
                    reason: e.to_string(),
                })
            })
            .transpose()
    }

    /// Parses an optional `f64` parameter.
    pub fn get_f64(&self, key: &str) -> Result<Option<f64>, AnalyticsError> {
        self.get(key)
            .map(|raw| {
                raw.parse::<f64>().map_err(|e| AnalyticsError::BadParam {
                    name: key.to_owned(),
                    reason: e.to_string(),
                })
            })
            .transpose()
    }

    /// Iterates the raw pairs (stable order) for logging.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

/// A finished report: stable machine payload plus human-readable markdown.
#[derive(Debug, Serialize)]
pub struct ReportOutput {
    /// Machine payload schema version ([`REPORT_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Registered report name.
    pub report: String,
    /// Database path the report ran against (provenance).
    pub db_path: String,
    /// Latest tick present in the database when the report ran, if any.
    pub latest_tick: Option<u64>,
    /// Number of primary rows rendered by this report.
    pub row_count: usize,
    /// Machine-readable payload (stable per `schema_version`).
    pub machine: serde_json::Value,
    /// Human-readable markdown rendering of the same content.
    #[serde(skip)]
    pub human_md: String,
}

/// A single offline report over a finished run database.
pub trait Report {
    /// Stable registry name (kebab-case).
    fn name(&self) -> &'static str;
    /// One-line description shown by `sb-analyze list`.
    fn description(&self) -> &'static str;
    /// Executes the report read-only.
    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError>;
}

/// Registry of available reports.
pub struct Registry {
    reports: Vec<Box<dyn Report>>,
}

impl Registry {
    /// Builds the built-in registry.
    ///
    /// Franken-adapter reports (fsci/fnx/frankenpandas) register here as
    /// their beads land (bd-2z0.11.6/.7/.8).
    #[must_use]
    pub fn builtin() -> Self {
        Self {
            reports: vec![
                Box::new(RunSummary),
                Box::new(NarrativeTimeline),
                Box::new(MetricSummary),
                Box::new(MetricChangepoints),
                Box::new(RunComparison),
                Box::new(MetricDistribution),
                Box::new(PhenotypeInteractions),
                Box::new(lineage::LineageFitness),
                Box::new(graphs::LineageStructure),
                Box::new(graphs::DynastyCommunities),
                Box::new(graphs::InteractionCentrality),
            ],
        }
    }

    /// Lists `(name, description)` pairs in registration order.
    #[must_use]
    pub fn list(&self) -> Vec<(&'static str, &'static str)> {
        self.reports
            .iter()
            .map(|r| (r.name(), r.description()))
            .collect()
    }

    /// Runs a report by name with framework-level tracing.
    pub fn run(
        &self,
        name: &str,
        cx: &ReaderCtx,
        params: &ReportParams,
    ) -> Result<ReportOutput, AnalyticsError> {
        let report = self
            .reports
            .iter()
            .find(|r| r.name() == name)
            .ok_or_else(|| AnalyticsError::UnknownReport(name.to_owned()))?;
        let span = tracing::info_span!("report", name = %name, db = %cx.db_path);
        let _guard = span.enter();
        for (k, v) in params.iter() {
            tracing::debug!(param = %k, value = %v, "report parameter");
        }
        let started = Instant::now();
        tracing::info!("report started");
        let result = report.run(cx, params);
        match &result {
            Ok(out) => tracing::info!(
                elapsed_ms = elapsed_millis(&started),
                latest_tick = ?out.latest_tick,
                rows = out.row_count,
                "report completed"
            ),
            Err(err) => tracing::error!(
                elapsed_ms = elapsed_millis(&started),
                error = %err,
                "report failed"
            ),
        }
        result
    }
}

fn elapsed_millis(started: &Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn log_report_stage(stage: &'static str, started: &Instant, rows: usize) {
    tracing::debug!(
        stage,
        elapsed_ms = elapsed_millis(started),
        rows,
        "report stage completed"
    );
}

/// `metric-summary`: a per-metric distribution summary over a finished run.
///
/// This is the first report to put the native [`stats`] module (bd-2z0.11.6) to work on real
/// persisted data: for every metric present in the newest bounded SQL page, it reports n, mean,
/// standard deviation, the 5/50/95 quantiles, min/max, and the coefficient of variation — the
/// foundation of the `distribution-report` (bd-2z0.11.6 item 2). Distribution FITTING (the
/// candidate normal/lognormal/gamma fits + KS test) is the piece where `fsci`'s distribution zoo
/// would earn its keep and is left for the adapter decision (bd-2z0.11.3); the summary itself
/// needs nothing beyond the native estimators.
struct MetricSummary;

#[derive(Debug, Serialize)]
struct MetricSummaryMachine {
    metrics: Vec<MetricSummaryRow>,
}

#[derive(Debug, Serialize)]
struct MetricSummaryRow {
    name: String,
    n: usize,
    mean: f64,
    std_dev: f64,
    min: f64,
    q05: f64,
    median: f64,
    q95: f64,
    max: f64,
    /// `std_dev / |mean|` — a scale-free measure of spread. `None` when the mean is within
    /// `f64::EPSILON` of zero, where the ratio is meaningless rather than merely large.
    coefficient_of_variation: Option<f64>,
}

impl Report for MetricSummary {
    fn name(&self) -> &'static str {
        "metric-summary"
    }

    fn description(&self) -> &'static str {
        "Per-metric distribution summary over the newest bounded row page of a finished run"
    }

    fn run(&self, cx: &ReaderCtx, _params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let read_started = Instant::now();
        let readings = cx.reader.recent_metrics(METRIC_SUMMARY_ROW_LIMIT)?;
        log_report_stage("read", &read_started, readings.len());

        let render_started = Instant::now();
        // Group values by metric name. BTreeMap keeps the output in a stable, name-sorted order so
        // two runs of the report over the same data render identically — a report whose row order
        // wobbled could not be diffed across runs.
        let mut by_metric: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for PersistedMetric { name, value, .. } in readings {
            by_metric.entry(name).or_default().push(value);
        }

        let mut rows = Vec::with_capacity(by_metric.len());
        for (name, values) in by_metric {
            // A metric with non-finite values is a real problem worth surfacing, but the stats
            // functions already reject it; map that to a report-level error rather than a panic.
            let mean = stats::mean(&values).map_err(|error| metric_stats_error(&error))?;
            let std_dev = stats::std_dev(&values).map_err(|error| metric_stats_error(&error))?;
            let q05 = stats::quantile(&values, 0.05).map_err(|error| metric_stats_error(&error))?;
            let median =
                stats::quantile(&values, 0.50).map_err(|error| metric_stats_error(&error))?;
            let q95 = stats::quantile(&values, 0.95).map_err(|error| metric_stats_error(&error))?;
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let coefficient_of_variation =
                (mean.abs() > f64::EPSILON).then(|| std_dev / mean.abs());
            rows.push(MetricSummaryRow {
                name,
                n: values.len(),
                mean,
                std_dev,
                min,
                q05,
                median,
                q95,
                max,
                coefficient_of_variation,
            });
        }

        let machine = MetricSummaryMachine { metrics: rows };

        let mut md = String::new();
        let _ = writeln!(md, "# Metric summary\n");
        if machine.metrics.is_empty() {
            let _ = writeln!(md, "_No metrics persisted in this run._");
        } else {
            let _ = writeln!(
                md,
                "| metric | n | mean | sd | min | p05 | median | p95 | max | CV |"
            );
            let _ = writeln!(md, "|---|---|---|---|---|---|---|---|---|---|");
            for row in &machine.metrics {
                let _ = writeln!(
                    md,
                    "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {} |",
                    row.name,
                    row.n,
                    row.mean,
                    row.std_dev,
                    row.min,
                    row.q05,
                    row.median,
                    row.q95,
                    row.max,
                    row.coefficient_of_variation
                        .map_or_else(|| "-".to_owned(), |cv| format!("{cv:.4}")),
                );
            }
        }

        let output = base_output(
            self.name(),
            cx,
            machine.metrics.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

/// Map a statistics error to a report error. The stats module only errors on genuinely bad data
/// (empty or non-finite), which for persisted metrics means the run wrote something impossible —
/// a report-level failure, not a panic.
fn metric_stats_error(error: &crate::stats::StatsError) -> AnalyticsError {
    AnalyticsError::Storage(StorageError::InvalidData {
        context: "analytics.metric_summary",
        reason: error.to_string(),
    })
}

/// `metric-changepoints`: find and CERTIFY the most prominent regime shift in each metric.
///
/// This is the certification pipeline (bd-2z0.11.6) run over real persisted data. For every metric
/// the run recorded, it locates the single largest mean shift ([`changepoint::largest_shift`]),
/// certifies it — permutation test, bootstrap CI on the shift, effect sizes — and applies
/// Benjamini-Hochberg across all metrics, so a run with a dozen metrics cannot report a chance
/// "regime change" it never had. It answers "which metrics genuinely shifted in this run, and
/// when?" with statistics rather than an eyeballed threshold.
///
/// Distinct from `scriptbots-core::detect`, which is the ONLINE detector: this is the offline
/// certification of shifts, over the finished series.
struct MetricChangepoints;

#[derive(Debug, Serialize)]
struct ChangepointsMachine {
    /// Certification window (samples on each side of the shift).
    window: usize,
    /// Target false-discovery rate for the across-metrics correction.
    fdr: f64,
    /// Metrics whose series was long enough to admit a certified change-point.
    metrics_examined: usize,
    /// How many of those hold up under FDR control — the honest count of real regime shifts.
    significant: usize,
    /// True when the bounded metric read hit its cap, so early history was not analysed and a
    /// change-point in it would be invisible. A truncated analysis must not read as a complete one.
    truncated: bool,
    changepoints: Vec<ChangepointRow>,
}

#[derive(Debug, Serialize)]
struct ChangepointRow {
    metric: String,
    /// The tick at which the new regime begins.
    change_tick: u64,
    shift: f64,
    before_mean: f64,
    after_mean: f64,
    p_value: f64,
    ci_lower: f64,
    ci_upper: f64,
    cohens_d: f64,
    cliffs_delta: f64,
    /// Survives Benjamini-Hochberg across the run's metrics. The field to act on.
    significant_fdr: bool,
}

struct ChangepointCandidate {
    metric: String,
    change_tick: u64,
    shift: f64,
    before_mean: f64,
    after_mean: f64,
    certification: certify::EventCertification,
}

fn certify_metric_changepoints(
    by_metric: BTreeMap<String, Vec<(u64, f64)>>,
    window: usize,
    cert_params: &certify::CertificationParams,
) -> Result<Vec<ChangepointCandidate>, AnalyticsError> {
    let mut candidates = Vec::new();
    for (metric, mut points) in by_metric {
        points.sort_by_key(|(tick, _)| *tick);
        let series: Vec<f64> = points.iter().map(|(_, value)| *value).collect();
        let Some(cp) = changepoint::largest_shift(&series, window) else {
            continue;
        };
        let certification = certify::certify_event(&series, cp.index, cert_params)
            .map_err(|error| metric_stats_error(&error))?;
        candidates.push(ChangepointCandidate {
            metric,
            change_tick: points[cp.index].0,
            shift: cp.shift,
            before_mean: cp.before_mean,
            after_mean: cp.after_mean,
            certification,
        });
    }
    Ok(candidates)
}

fn render_changepoints_markdown(machine: &ChangepointsMachine) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Metric change-points\n");
    if machine.truncated {
        let _ = writeln!(
            md,
            "> **Note:** the metric read hit its {METRIC_SUMMARY_ROW_LIMIT}-row cap; early \
             history was not analysed and a shift in it is not reported here.\n"
        );
    }
    let _ = writeln!(
        md,
        "_window={}, FDR={}, {} of {} metrics show a certified regime shift._\n",
        machine.window, machine.fdr, machine.significant, machine.metrics_examined
    );
    if machine.changepoints.is_empty() {
        let _ = writeln!(
            md,
            "_No metric series was long enough to certify a change-point._"
        );
        return md;
    }

    let _ = writeln!(md, "| metric | tick | shift | p | 95% CI | d | δ | real? |");
    let _ = writeln!(md, "|---|---|---|---|---|---|---|---|");
    for row in &machine.changepoints {
        let _ = writeln!(
            md,
            "| {} | {} | {:+.4} | {:.4} | [{:.3}, {:.3}] | {:.3} | {:.3} | {} |",
            row.metric,
            row.change_tick,
            row.shift,
            row.p_value,
            row.ci_lower,
            row.ci_upper,
            row.cohens_d,
            row.cliffs_delta,
            if row.significant_fdr { "yes" } else { "no" },
        );
    }
    md
}

impl Report for MetricChangepoints {
    fn name(&self) -> &'static str {
        "metric-changepoints"
    }

    fn description(&self) -> &'static str {
        "Find and statistically certify the largest regime shift in each metric (FDR-controlled)"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let window = params.get_usize("window")?.unwrap_or(30);
        if window == 0 {
            return Err(AnalyticsError::BadParam {
                name: "window".to_owned(),
                reason: "must be at least 1".to_owned(),
            });
        }

        let read_started = Instant::now();
        let readings = cx.reader.recent_metrics(METRIC_SUMMARY_ROW_LIMIT)?;
        // `recent_metrics` is a bounded most-recent-N read. If it came back full, earlier history
        // was dropped and a change-point in it is invisible — say so rather than let a truncated
        // analysis read as a complete one.
        let truncated = readings.len() >= METRIC_SUMMARY_ROW_LIMIT;
        log_report_stage("read", &read_started, readings.len());

        let render_started = Instant::now();
        // Group into ordered (tick, value) series per metric. recent_metrics order is not
        // contractually chronological here, so we sort by tick explicitly — a change-point over a
        // mis-ordered series would be meaningless.
        let mut by_metric: BTreeMap<String, Vec<(u64, f64)>> = BTreeMap::new();
        for PersistedMetric { tick, name, value } in readings {
            by_metric.entry(name).or_default().push((tick, value));
        }

        let cert_params = certify::CertificationParams {
            window,
            fdr: params
                .get("fdr")
                .map(str::parse::<f64>)
                .transpose()
                .map_err(|e| AnalyticsError::BadParam {
                    name: "fdr".to_owned(),
                    reason: e.to_string(),
                })?
                .unwrap_or(0.05),
            ..certify::CertificationParams::default()
        };

        // The window doubles as `min_segment`, so every located shift leaves a full certification
        // window on both sides and `certify_event` cannot see an out-of-range window.
        let candidates = certify_metric_changepoints(by_metric, window, &cert_params)?;

        // Second pass: Benjamini-Hochberg across every metric's p-value at once.
        let p_values: Vec<f64> = candidates.iter().map(|c| c.certification.p_value).collect();
        let rejected = certify::benjamini_hochberg(&p_values, cert_params.fdr);

        let mut rows = Vec::with_capacity(candidates.len());
        let mut significant = 0usize;
        for (candidate, &is_rejected) in candidates.into_iter().zip(&rejected) {
            if is_rejected {
                significant += 1;
            }
            rows.push(ChangepointRow {
                metric: candidate.metric,
                change_tick: candidate.change_tick,
                shift: candidate.shift,
                before_mean: candidate.before_mean,
                after_mean: candidate.after_mean,
                p_value: candidate.certification.p_value,
                ci_lower: candidate.certification.shift_ci.lower,
                ci_upper: candidate.certification.shift_ci.upper,
                cohens_d: candidate.certification.cohens_d,
                cliffs_delta: candidate.certification.cliffs_delta,
                significant_fdr: is_rejected,
            });
        }

        let machine = ChangepointsMachine {
            window,
            fdr: cert_params.fdr,
            metrics_examined: rows.len(),
            significant,
            truncated,
            changepoints: rows,
        };

        let md = render_changepoints_markdown(&machine);

        let output = base_output(
            self.name(),
            cx,
            machine.changepoints.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

/// `compare-runs`: paired treatment-effect comparison of two run databases (serves bd-16g.1.4).
///
/// Given a control run (the database this report runs against) and a `treatment_db=<path>`, it
/// measures whether the treatment shifted each metric the two runs share. The runs are assumed to
/// share seeds, so each metric is compared TICK-ALIGNED — the control and treatment values at the
/// same tick form a matched pair — and the pairing is fed to [`compare`], which applies a
/// sign-flip permutation test, a paired-bootstrap CI, Cohen's `d_z`, and Benjamini-Hochberg across
/// the metrics. It is the DB-facing glue for the matched-seed statistics; the pure analysis was
/// proven in isolation, this wires it to two real databases.
struct RunComparison;

#[derive(Debug, Serialize)]
struct RunComparisonMachine {
    /// The treatment database this control run was compared against (provenance).
    treatment_db: String,
    /// Target false-discovery rate for the across-metrics correction.
    fdr: f64,
    /// Metrics present in BOTH runs with enough tick-aligned pairs to compare.
    metrics_compared: usize,
    /// How many hold up under FDR control — the honest count of real treatment effects.
    significant: usize,
    /// True when either bounded metric read hit its cap, so the comparison is over recent history
    /// rather than the whole run.
    truncated: bool,
    metrics: Vec<RunComparisonRow>,
}

#[derive(Debug, Serialize)]
struct RunComparisonRow {
    metric: String,
    /// Number of tick-aligned matched pairs.
    n_pairs: usize,
    /// Mean of `treatment - control` over the matched pairs. The treatment-effect estimate.
    mean_difference: f64,
    ci_lower: f64,
    ci_upper: f64,
    p_value: f64,
    /// Paired standardized effect size (`d_z`).
    cohens_dz: f64,
    /// Fraction of pairs where treatment exceeded control.
    fraction_positive: f64,
    /// Survives Benjamini-Hochberg across the run's shared metrics. The field to act on.
    significant_fdr: bool,
}

struct PairedMetricSeries {
    name: String,
    control: Vec<f64>,
    treatment: Vec<f64>,
}

type MetricsByTick = BTreeMap<String, BTreeMap<u64, f64>>;

fn index_metrics_by_tick(readings: Vec<PersistedMetric>) -> MetricsByTick {
    let mut indexed: MetricsByTick = BTreeMap::new();
    for PersistedMetric { tick, name, value } in readings {
        indexed.entry(name).or_default().insert(tick, value);
    }
    indexed
}

fn pair_shared_metrics(
    control: &MetricsByTick,
    treatment: &MetricsByTick,
) -> Vec<PairedMetricSeries> {
    let mut paired = Vec::new();
    for (name, control_ticks) in control {
        let Some(treatment_ticks) = treatment.get(name) else {
            continue;
        };
        let mut control_values = Vec::new();
        let mut treatment_values = Vec::new();
        for (tick, control_value) in control_ticks {
            if let Some(treatment_value) = treatment_ticks.get(tick) {
                control_values.push(*control_value);
                treatment_values.push(*treatment_value);
            }
        }
        if control_values.len() >= 3 {
            paired.push(PairedMetricSeries {
                name: name.clone(),
                control: control_values,
                treatment: treatment_values,
            });
        }
    }
    paired
}

fn compare_paired_metrics(
    paired: &[PairedMetricSeries],
    fdr: f64,
) -> Result<Vec<RunComparisonRow>, AnalyticsError> {
    let series: Vec<compare::MetricSeries<'_>> = paired
        .iter()
        .map(|pair| compare::MetricSeries {
            name: pair.name.as_str(),
            control: pair.control.as_slice(),
            treatment: pair.treatment.as_slice(),
        })
        .collect();
    let study = compare::compare_metrics(
        &series,
        &compare::CompareParams {
            fdr,
            ..compare::CompareParams::default()
        },
    )
    .map_err(|error| metric_stats_error(&error))?;

    Ok(study
        .metrics
        .iter()
        .map(|named| {
            let comparison = &named.comparison;
            RunComparisonRow {
                metric: named.metric.clone(),
                n_pairs: comparison.n_pairs,
                mean_difference: comparison.mean_difference,
                ci_lower: comparison.difference_ci.lower,
                ci_upper: comparison.difference_ci.upper,
                p_value: comparison.p_value,
                cohens_dz: comparison.cohens_dz,
                fraction_positive: comparison.fraction_positive,
                significant_fdr: comparison.significant_fdr,
            }
        })
        .collect())
}

fn render_run_comparison_markdown(machine: &RunComparisonMachine) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Run comparison\n");
    let _ = writeln!(
        md,
        "_treatment=`{}`, FDR={}, {} of {} shared metrics show a certified treatment effect._\n",
        machine.treatment_db, machine.fdr, machine.significant, machine.metrics_compared
    );
    if machine.truncated {
        let _ = writeln!(
            md,
            "> **Note:** a metric read hit its {METRIC_SUMMARY_ROW_LIMIT}-row cap; the \
             comparison is over recent history, not the whole run.\n"
        );
    }
    if machine.metrics.is_empty() {
        let _ = writeln!(
            md,
            "_No metric was present in both runs with enough matched ticks._"
        );
        return md;
    }

    let _ = writeln!(
        md,
        "| metric | pairs | Δ (treat−ctrl) | 95% CI | p | d_z | +frac | real? |"
    );
    let _ = writeln!(md, "|---|---|---|---|---|---|---|---|");
    for row in &machine.metrics {
        let _ = writeln!(
            md,
            "| {} | {} | {:+.4} | [{:.3}, {:.3}] | {:.4} | {:.3} | {:.2} | {} |",
            row.metric,
            row.n_pairs,
            row.mean_difference,
            row.ci_lower,
            row.ci_upper,
            row.p_value,
            row.cohens_dz,
            row.fraction_positive,
            if row.significant_fdr { "yes" } else { "no" },
        );
    }
    md
}

impl Report for RunComparison {
    fn name(&self) -> &'static str {
        "compare-runs"
    }

    fn description(&self) -> &'static str {
        "Paired treatment-effect comparison of two run databases (tick-aligned, FDR-controlled)"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let treatment_path =
            params
                .get("treatment_db")
                .ok_or_else(|| AnalyticsError::BadParam {
                    name: "treatment_db".to_owned(),
                    reason:
                        "compare-runs requires treatment_db=<path> (the treatment run database)"
                            .to_owned(),
                })?;
        let fdr = params
            .get("fdr")
            .map(str::parse::<f64>)
            .transpose()
            .map_err(|e| AnalyticsError::BadParam {
                name: "fdr".to_owned(),
                reason: e.to_string(),
            })?
            .unwrap_or(0.05);

        let read_started = Instant::now();
        let control_readings = cx.reader.recent_metrics(METRIC_SUMMARY_ROW_LIMIT)?;
        let treatment_reader = StorageReader::open_finished(treatment_path)?;
        let treatment_readings = treatment_reader.recent_metrics(METRIC_SUMMARY_ROW_LIMIT)?;
        let truncated = control_readings.len() >= METRIC_SUMMARY_ROW_LIMIT
            || treatment_readings.len() >= METRIC_SUMMARY_ROW_LIMIT;
        log_report_stage(
            "read",
            &read_started,
            control_readings.len() + treatment_readings.len(),
        );

        let render_started = Instant::now();
        // BTreeMap keeps both metric and tick order stable. Pair only ticks shared by both runs;
        // three pairs is the minimum accepted by the paired statistical test.
        let control = index_metrics_by_tick(control_readings);
        let treatment = index_metrics_by_tick(treatment_readings);
        let paired = pair_shared_metrics(&control, &treatment);
        let rows = compare_paired_metrics(&paired, fdr)?;
        let significant = rows.iter().filter(|row| row.significant_fdr).count();

        let machine = RunComparisonMachine {
            treatment_db: treatment_path.to_owned(),
            fdr,
            metrics_compared: rows.len(),
            significant,
            truncated,
            metrics: rows,
        };

        let md = render_run_comparison_markdown(&machine);

        let output = base_output(
            self.name(),
            cx,
            machine.metrics.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

/// `metric-distribution`: per-metric shape and normality (bd-2z0.11.6 item 2).
///
/// For every metric the run recorded, reports its skewness and excess kurtosis and runs a
/// Jarque-Bera normality test ([`distribution`]) — a native, `erf`-free assessment of "is this
/// metric normal, and how is it shaped?". A skewed or heavy-tailed metric is exactly the case
/// where a mean-and-SD summary (the `metric-summary` report) understates the story, so this is its
/// companion. Full distribution FITTING (candidate lognormal/gamma) stays with the `fsci` adapter
/// decision (bd-2z0.11.3).
struct MetricDistribution;

#[derive(Debug, Serialize)]
struct MetricDistributionMachine {
    /// Significance level for the normality verdict.
    alpha: f64,
    /// True when the bounded metric read hit its cap.
    truncated: bool,
    /// Metrics with at least four values (the minimum for a shape test).
    metrics_examined: usize,
    /// How many were flagged non-normal at `alpha`.
    non_normal: usize,
    metrics: Vec<MetricDistributionRow>,
}

#[derive(Debug, Serialize)]
struct MetricDistributionRow {
    name: String,
    n: usize,
    mean: f64,
    std_dev: f64,
    skewness: f64,
    excess_kurtosis: f64,
    jarque_bera: f64,
    jb_p_value: f64,
    /// A constant metric: no shape, reported as such rather than as "looks normal".
    degenerate: bool,
    /// Jarque-Bera rejects normality at `alpha`. Never true for a degenerate metric.
    non_normal: bool,
}

fn summarize_metric_distributions(
    readings: Vec<PersistedMetric>,
    alpha: f64,
) -> Result<Vec<MetricDistributionRow>, AnalyticsError> {
    let mut by_metric: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for PersistedMetric { name, value, .. } in readings {
        by_metric.entry(name).or_default().push(value);
    }

    let mut rows = Vec::with_capacity(by_metric.len());
    for (name, values) in by_metric {
        if values.len() < 4 {
            continue;
        }
        let summary =
            distribution::summarize(&values).map_err(|error| metric_stats_error(&error))?;
        rows.push(MetricDistributionRow {
            name,
            n: summary.n,
            mean: summary.mean,
            std_dev: summary.std_dev,
            skewness: summary.skewness,
            excess_kurtosis: summary.excess_kurtosis,
            jarque_bera: summary.jarque_bera,
            jb_p_value: summary.jb_p_value,
            degenerate: summary.degenerate,
            non_normal: summary.rejects_normality(alpha),
        });
    }
    Ok(rows)
}

fn render_metric_distribution_markdown(machine: &MetricDistributionMachine) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Metric distributions\n");
    let _ = writeln!(
        md,
        "_alpha={}, {} of {} metrics flagged non-normal (Jarque-Bera)._\n",
        machine.alpha, machine.non_normal, machine.metrics_examined
    );
    if machine.truncated {
        let _ = writeln!(
            md,
            "> **Note:** the metric read hit its {METRIC_SUMMARY_ROW_LIMIT}-row cap; the shape \
             is over recent history, not the whole run.\n"
        );
    }
    if machine.metrics.is_empty() {
        let _ = writeln!(md, "_No metric had at least four values to characterize._");
        return md;
    }

    let _ = writeln!(
        md,
        "| metric | n | mean | sd | skew | ex.kurt | JB | p | normal? |"
    );
    let _ = writeln!(md, "|---|---|---|---|---|---|---|---|---|");
    for row in &machine.metrics {
        let verdict = if row.degenerate {
            "constant"
        } else if row.non_normal {
            "no"
        } else {
            "yes"
        };
        let _ = writeln!(
            md,
            "| {} | {} | {:.4} | {:.4} | {:+.3} | {:+.3} | {:.2} | {:.4} | {} |",
            row.name,
            row.n,
            row.mean,
            row.std_dev,
            row.skewness,
            row.excess_kurtosis,
            row.jarque_bera,
            row.jb_p_value,
            verdict,
        );
    }
    md
}

impl Report for MetricDistribution {
    fn name(&self) -> &'static str {
        "metric-distribution"
    }

    fn description(&self) -> &'static str {
        "Per-metric shape (skewness, kurtosis) and a Jarque-Bera normality test over a finished run"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let alpha = params
            .get("alpha")
            .map(str::parse::<f64>)
            .transpose()
            .map_err(|e| AnalyticsError::BadParam {
                name: "alpha".to_owned(),
                reason: e.to_string(),
            })?
            .unwrap_or(0.05);

        let read_started = Instant::now();
        let readings = cx.reader.recent_metrics(METRIC_SUMMARY_ROW_LIMIT)?;
        let truncated = readings.len() >= METRIC_SUMMARY_ROW_LIMIT;
        log_report_stage("read", &read_started, readings.len());

        let render_started = Instant::now();
        let rows = summarize_metric_distributions(readings, alpha)?;
        let non_normal = rows.iter().filter(|row| row.non_normal).count();

        let machine = MetricDistributionMachine {
            alpha,
            truncated,
            metrics_examined: rows.len(),
            non_normal,
            metrics: rows,
        };

        let md = render_metric_distribution_markdown(&machine);

        let output = base_output(
            self.name(),
            cx,
            machine.metrics.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

fn base_output(
    name: &str,
    cx: &ReaderCtx,
    row_count: usize,
    machine: serde_json::Value,
    human_md: String,
) -> Result<ReportOutput, AnalyticsError> {
    Ok(ReportOutput {
        schema_version: REPORT_SCHEMA_VERSION,
        report: name.to_owned(),
        db_path: cx.db_path.clone(),
        latest_tick: cx.reader.max_tick()?,
        row_count,
        machine,
        human_md,
    })
}

/// `phenotype-interactions`: canonical run-wide phenotype and interaction analysis.
struct PhenotypeInteractions;

impl Report for PhenotypeInteractions {
    fn name(&self) -> &'static str {
        "phenotype-interactions"
    }

    fn description(&self) -> &'static str {
        "Versioned phenotype features and directed interaction graph from persisted AgentUid histories"
    }

    fn run(&self, cx: &ReaderCtx, _params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let read_started = Instant::now();
        let analysis = analyze_persisted_phenotypes(&cx.reader)?;
        log_report_stage("read_and_extract", &read_started, analysis.features.len());
        tracing::info!(
            run_id = %analysis.run_id,
            start_tick = analysis.window.start_tick,
            end_tick = analysis.window.end_tick,
            schema_id = analysis.schema.schema_id,
            schema_digest = %analysis.schema_digest,
            config_digest = %analysis.config_digest,
            accepted_agents = analysis.features.len(),
            graph_nodes = analysis.interaction_graph.nodes.len(),
            graph_edges = analysis.interaction_graph.edges.len(),
            canonical_digest = %analysis.canonical_digest,
            "canonical phenotype and interaction analysis completed"
        );

        let render_started = Instant::now();
        let mut md = String::new();
        let _ = writeln!(md, "# Phenotype and interaction analysis\n");
        let _ = writeln!(md, "- run: `{}`", analysis.run_id);
        let _ = writeln!(
            md,
            "- window: `[{}, {})`",
            analysis.window.start_tick, analysis.window.end_tick
        );
        let _ = writeln!(
            md,
            "- feature schema: `{}` (`{}`)",
            analysis.schema.schema_id, analysis.schema_digest
        );
        let _ = writeln!(md, "- config digest: `{}`", analysis.config_digest);
        let _ = writeln!(md, "- canonical digest: `{}`", analysis.canonical_digest);
        let _ = writeln!(
            md,
            "- agents / graph nodes / graph edges: {} / {} / {}\n",
            analysis.features.len(),
            analysis.interaction_graph.nodes.len(),
            analysis.interaction_graph.edges.len()
        );
        let _ = writeln!(
            md,
            "| uid | observations | speed | herbivore trait proxy | sensing trait proxy | combat/tick | share/tick | offspring/tick |"
        );
        let _ = writeln!(md, "|---:|---:|---:|---:|---:|---:|---:|---:|");
        for row in &analysis.features {
            let _ = writeln!(
                md,
                "| {} | {} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} |",
                row.agent_uid.0,
                row.observed_tick_count,
                row.values[0],
                row.values[1],
                row.values[2],
                row.values[3],
                row.values[4],
                row.values[5],
            );
        }

        let output = base_output(
            self.name(),
            cx,
            analysis.features.len(),
            serde_json::to_value(&analysis)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

/// `run-summary`: lifecycle totals and bounded recent population trajectory statistics.
struct RunSummary;

#[derive(Debug, Serialize)]
struct RunSummaryMachine {
    tick_count: u64,
    birth_records: u64,
    death_records: u64,
    population_first: Option<usize>,
    population_last: Option<usize>,
    population_min: Option<usize>,
    population_max: Option<usize>,
    population_mean: Option<f64>,
    total_energy_first: Option<f64>,
    total_energy_last: Option<f64>,
    watermarks: WatermarksMachine,
}

#[derive(Debug, Serialize)]
struct WatermarksMachine {
    admitted: Option<u64>,
    applied: Option<u64>,
    durable: Option<u64>,
}

impl Report for RunSummary {
    fn name(&self) -> &'static str {
        "run-summary"
    }

    fn description(&self) -> &'static str {
        "Lifecycle totals, recent bounded population trajectory stats, and persistence watermarks"
    }

    fn run(&self, cx: &ReaderCtx, _params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let read_started = Instant::now();
        let ledger = cx.reader.run_ledger_summary()?;
        // StorageReader returns the newest bounded page in chronological order.
        let ticks = cx.reader.recent_ticks(RUN_SUMMARY_TICK_LIMIT)?;
        let watermarks = cx.reader.persistence_watermarks()?;
        log_report_stage("read", &read_started, ticks.len());

        let render_started = Instant::now();
        let mut population_first = None;
        let mut population_last = None;
        let mut population_min = None;
        let mut population_max = None;
        let mut population_mean = 0.0_f64;
        let mut population_count = 0_u64;
        for tick in &ticks {
            population_first.get_or_insert(tick.agent_count);
            population_last = Some(tick.agent_count);
            population_min = Some(
                population_min.map_or(tick.agent_count, |value: usize| value.min(tick.agent_count)),
            );
            population_max = Some(
                population_max.map_or(tick.agent_count, |value: usize| value.max(tick.agent_count)),
            );
            population_count += 1;
            #[allow(clippy::cast_precision_loss)]
            let observation = tick.agent_count as f64;
            #[allow(clippy::cast_precision_loss)]
            let count = population_count as f64;
            population_mean += (observation - population_mean) / count;
        }
        let machine = RunSummaryMachine {
            tick_count: ledger.tick_count,
            birth_records: ledger.birth_records,
            death_records: ledger.death_records,
            population_first,
            population_last,
            population_min,
            population_max,
            population_mean: (population_count > 0).then_some(population_mean),
            total_energy_first: ticks.first().map(|t| t.total_energy),
            total_energy_last: ticks.last().map(|t| t.total_energy),
            watermarks: WatermarksMachine {
                admitted: watermarks.admitted.map(PersistenceBatchId::get),
                applied: watermarks.applied.map(PersistenceBatchId::get),
                durable: watermarks.durable.map(PersistenceBatchId::get),
            },
        };

        let mut md = String::new();
        let _ = writeln!(md, "# Run summary\n");
        let _ = writeln!(md, "| field | value |");
        let _ = writeln!(md, "|---|---|");
        let _ = writeln!(md, "| ticks persisted | {} |", machine.tick_count);
        let _ = writeln!(
            md,
            "| births / deaths | {} / {} |",
            machine.birth_records, machine.death_records
        );
        let _ = writeln!(
            md,
            "| recent-window population first→last (min/mean/max) | {:?}→{:?} ({:?}/{}/{:?}) |",
            machine.population_first,
            machine.population_last,
            machine.population_min,
            machine
                .population_mean
                .map_or_else(|| "-".into(), |m| format!("{m:.1}")),
            machine.population_max,
        );
        let _ = writeln!(
            md,
            "| total energy first→last | {:?}→{:?} |",
            machine.total_energy_first, machine.total_energy_last
        );

        let output = base_output(
            self.name(),
            cx,
            ticks.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

/// `narrative-timeline`: bounded page of the latest events for a finished run.
///
/// v1 renders aggregate kind counts plus a newest-first SQL page returned in
/// chronological order. When the
/// typed narrative tables land (bd-16g.2.2) and FTS search (bd-16g.2.6),
/// this report upgrades to the `run_events` stream + BM25 search parameters
/// WITHOUT changing its registry name; the machine schema will bump
/// [`REPORT_SCHEMA_VERSION`] per the documented migration policy.
struct NarrativeTimeline;

#[derive(Debug, Serialize)]
struct TimelineMachine {
    event_counts: Vec<(String, u64)>,
    events: Vec<TimelineRow>,
    truncated_to: Option<usize>,
}

#[derive(Debug, Serialize)]
struct TimelineRow {
    tick: u64,
    seq: u64,
    event: serde_json::Value,
}

impl Report for NarrativeTimeline {
    fn name(&self) -> &'static str {
        "narrative-timeline"
    }

    fn description(&self) -> &'static str {
        "Chronological replay-event timeline (upgrades to typed narrative events when bd-16g.2.2 lands)"
    }

    fn run(&self, cx: &ReaderCtx, params: &ReportParams) -> Result<ReportOutput, AnalyticsError> {
        let limit = narrative_timeline_limit(params)?;
        let read_started = Instant::now();
        let counts = cx.reader.replay_event_counts()?;
        let events = cx.reader.recent_replay_events(limit)?;
        let total = counts
            .iter()
            .fold(0_u64, |sum, count| sum.saturating_add(count.count));
        log_report_stage("read", &read_started, events.len());

        let render_started = Instant::now();
        let machine = TimelineMachine {
            event_counts: counts
                .iter()
                .map(|c| (c.event_type.clone(), c.count))
                .collect(),
            events: events
                .iter()
                .map(|e| {
                    Ok(TimelineRow {
                        tick: e.tick,
                        seq: e.seq,
                        event: serde_json::to_value(&e.event)?,
                    })
                })
                .collect::<Result<Vec<_>, AnalyticsError>>()?,
            truncated_to: (total > u64::try_from(limit).unwrap_or(u64::MAX)).then_some(limit),
        };

        let mut md = String::new();
        let _ = writeln!(
            md,
            "# Narrative timeline (latest bounded replay-event page, v1)\n"
        );
        if machine.events.is_empty() {
            let _ = writeln!(md, "_No replay events persisted in this run._");
        } else {
            let _ = writeln!(md, "| tick | seq | event |");
            let _ = writeln!(md, "|---|---|---|");
            for row in &machine.events {
                let _ = writeln!(md, "| {} | {} | `{}` |", row.tick, row.seq, row.event);
            }
            if let Some(t) = machine.truncated_to {
                let _ = writeln!(md, "\n_…showing {t} of {total} events (bounded SQL page)._");
            }
        }

        let output = base_output(
            self.name(),
            cx,
            machine.events.len(),
            serde_json::to_value(&machine)?,
            md,
        )?;
        log_report_stage("render", &render_started, output.row_count);
        Ok(output)
    }
}

fn narrative_timeline_limit(params: &ReportParams) -> Result<usize, AnalyticsError> {
    let limit = params
        .get_usize("limit")?
        .unwrap_or(NARRATIVE_TIMELINE_DEFAULT_LIMIT);
    if limit > NARRATIVE_TIMELINE_MAX_LIMIT {
        return Err(AnalyticsError::BadParam {
            name: "limit".to_owned(),
            reason: format!("must be at most {NARRATIVE_TIMELINE_MAX_LIMIT}"),
        });
    }
    Ok(limit)
}

#[cfg(test)]
#[allow(clippy::all, clippy::pedantic, clippy::nursery)]
mod phenotype_tests {
    use super::*;

    const RUN_A: &str = "0000000000000000000000000000000a";
    const RUN_B: &str = "0000000000000000000000000000000b";

    fn observation(
        uid: u64,
        tick: u64,
        velocity: (f64, f64),
        herbivore: f64,
        sensing: f64,
    ) -> RunScopedAgentObservationV1 {
        RunScopedAgentObservationV1 {
            run_id: RUN_A.to_owned(),
            tick,
            agent_uid: AgentUid(uid),
            velocity_x: velocity.0,
            velocity_y: velocity.1,
            herbivore_tendency: herbivore,
            trait_smell: sensing,
            trait_sound: sensing,
            trait_hearing: sensing,
            trait_eye: sensing,
            trait_blood: sensing,
        }
    }

    fn arrival(
        uid: u64,
        tick: u64,
        parent_a: Option<u64>,
        parent_b: Option<u64>,
    ) -> RunScopedArrivalV1 {
        RunScopedArrivalV1 {
            run_id: RUN_A.to_owned(),
            tick,
            agent_uid: AgentUid(uid),
            parent_a: parent_a.map(AgentUid),
            parent_b: parent_b.map(AgentUid),
        }
    }

    fn interaction(
        tick: u64,
        seq: u64,
        kind: &str,
        actor: u64,
        target: u64,
        magnitude: f64,
    ) -> RunScopedInteractionV1 {
        RunScopedInteractionV1 {
            run_id: RUN_A.to_owned(),
            tick,
            seq,
            kind: kind.to_owned(),
            actor: AgentUid(actor),
            target: AgentUid(target),
            magnitude: Some(magnitude),
        }
    }

    fn fixture() -> PhenotypeLedgerV1 {
        PhenotypeLedgerV1 {
            schema_id: PHENOTYPE_FEATURE_SCHEMA_ID_V1.to_owned(),
            schema_digest: PHENOTYPE_FEATURE_SCHEMA_V1.digest().expect("schema digest"),
            config_digest: "config-a".to_owned(),
            run_id: RUN_A.to_owned(),
            window: PhenotypeTickWindowV1 {
                start_tick: 0,
                end_tick: 4,
            },
            observations: vec![
                observation(1, 0, (3.0, 4.0), 0.25, 1.0),
                observation(1, 1, (0.0, 0.0), 0.5, 3.0),
                observation(2, 1, (0.0, 2.0), 0.8, 2.0),
                observation(3, 2, (1.0, 0.0), 0.5, 1.5),
            ],
            interactions: vec![
                interaction(1, 100, "combat", 1, 2, 2.0),
                interaction(2, 101, "food_share", 1, 2, 3.0),
                interaction(3, 102, "combat", 2, 1, 1.0),
            ],
            arrivals: vec![
                arrival(1, 0, None, None),
                arrival(2, 0, None, None),
                arrival(3, 2, Some(1), Some(2)),
            ],
            interaction_capture: RunScopedInteractionCaptureV1 {
                run_id: RUN_A.to_owned(),
                observed: 3,
                persisted: 3,
                sampled_out: 0,
                truncated: 0,
            },
        }
    }

    fn row(analysis: &PhenotypeInteractionAnalysisV1, uid: u64) -> &PhenotypeFeatureRowV1 {
        analysis
            .features
            .iter()
            .find(|row| row.agent_uid == AgentUid(uid))
            .expect("feature row")
    }

    #[test]
    fn schema_axes_pin_ids_units_evidence_and_order() {
        let axes = PHENOTYPE_FEATURE_SCHEMA_V1.axes;
        assert_eq!(
            axes.map(|axis| axis.id),
            [
                "movement.speed.mean",
                "diet.herbivore_trait.mean",
                "sensing.trait_modifier.mean",
                "interaction.combat.actor_rate",
                "interaction.food_share.actor_rate",
                "reproduction.parent_rate",
            ]
        );
        assert_eq!(axes[0].unit, "world_unit_per_tick");
        assert_eq!(axes[1].unit, "ratio");
        assert_eq!(axes[2].unit, "trait_multiplier");
        assert_eq!(axes[3].unit, "event_per_tick");
        assert_eq!(axes[4].unit, "event_per_tick");
        assert_eq!(axes[5].unit, "offspring_per_tick");
        assert_eq!(axes[1].evidence, FeatureEvidenceV1::PersistedTraitProxy);
        assert_eq!(axes[2].evidence, FeatureEvidenceV1::PersistedTraitProxy);
    }

    #[test]
    fn derives_every_axis_and_directed_edge_from_audited_ledger() {
        let analysis = extract_phenotype_interactions(&fixture()).expect("extract");

        assert_eq!(analysis.features.len(), 3);
        assert_eq!(row(&analysis, 1).observed_tick_count, 2);
        assert_eq!(
            row(&analysis, 1).values,
            [2.5, 0.375, 2.0, 0.25, 0.25, 0.25]
        );
        assert_eq!(row(&analysis, 2).values, [2.0, 0.8, 2.0, 0.25, 0.0, 0.25]);
        assert_eq!(row(&analysis, 3).values, [1.0, 0.5, 1.5, 0.0, 0.0, 0.0]);
        assert!(analysis.features.iter().all(|row| {
            row.values
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        }));

        let graph = &analysis.interaction_graph;
        assert_eq!(graph.nodes, vec![AgentUid(1), AgentUid(2), AgentUid(3)]);
        assert_eq!(graph.edges.len(), 3);
        assert_eq!(
            (
                graph.edges[0].actor,
                graph.edges[0].target,
                graph.edges[0].kind,
                graph.edges[0].magnitude_sum,
            ),
            (AgentUid(1), AgentUid(2), InteractionEdgeKindV1::Combat, 2.0,)
        );
        assert_eq!(
            (
                graph.edges[2].actor,
                graph.edges[2].target,
                graph.edges[2].kind,
            ),
            (AgentUid(2), AgentUid(1), InteractionEdgeKindV1::Combat,)
        );
        assert_ne!(graph.canonical_digest, analysis.canonical_digest);
    }

    #[test]
    fn shuffled_sources_produce_byte_identical_analysis() {
        let baseline = fixture();
        let expected =
            serde_json::to_vec(&extract_phenotype_interactions(&baseline).expect("baseline"))
                .expect("serialize baseline");
        let mut shuffled = baseline;
        shuffled.observations.reverse();
        shuffled.interactions.reverse();
        shuffled.arrivals.reverse();
        let actual =
            serde_json::to_vec(&extract_phenotype_interactions(&shuffled).expect("shuffled"))
                .expect("serialize shuffled");
        assert_eq!(actual, expected);
    }

    #[test]
    fn complete_absence_is_zero_but_censored_absence_is_rejected() {
        let mut no_events = fixture();
        no_events.interactions.clear();
        no_events.interaction_capture.observed = 0;
        no_events.interaction_capture.persisted = 0;
        let analysis = extract_phenotype_interactions(&no_events).expect("complete empty graph");
        assert!(analysis.interaction_graph.edges.is_empty());
        assert_eq!(row(&analysis, 1).values[3], 0.0);
        assert_eq!(row(&analysis, 1).values[4], 0.0);

        no_events.interaction_capture.observed = 1;
        no_events.interaction_capture.sampled_out = 1;
        assert!(matches!(
            extract_phenotype_interactions(&no_events),
            Err(PhenotypeExtractionError::IncompleteInteractionEvidence {
                observed: 1,
                persisted: 0,
                sampled_out: 1,
                truncated: 0,
            })
        ));
    }

    #[test]
    fn schema_run_identity_and_window_drift_fail_typed() {
        let mut bad_schema = fixture();
        bad_schema.schema_digest = "drift".to_owned();
        assert!(matches!(
            extract_phenotype_interactions(&bad_schema),
            Err(PhenotypeExtractionError::SchemaDigestMismatch { .. })
        ));

        let mut cross_run = fixture();
        cross_run.observations[0].run_id = RUN_B.to_owned();
        assert!(matches!(
            extract_phenotype_interactions(&cross_run),
            Err(PhenotypeExtractionError::CrossRunSource {
                source_name: "observation",
                ..
            })
        ));

        let mut outside = fixture();
        outside.observations[0].tick = outside.window.end_tick;
        assert!(matches!(
            extract_phenotype_interactions(&outside),
            Err(PhenotypeExtractionError::TickOutsideWindow {
                source_name: "observation",
                ..
            })
        ));

        let mut empty = fixture();
        empty.window.end_tick = empty.window.start_tick;
        assert!(matches!(
            extract_phenotype_interactions(&empty),
            Err(PhenotypeExtractionError::InvalidWindow { .. })
        ));
    }

    #[test]
    fn duplicate_and_missing_identities_fail_typed() {
        let mut duplicate_observation = fixture();
        duplicate_observation
            .observations
            .push(duplicate_observation.observations[0].clone());
        assert!(matches!(
            extract_phenotype_interactions(&duplicate_observation),
            Err(PhenotypeExtractionError::DuplicateObservation { .. })
        ));

        let mut duplicate_interaction = fixture();
        duplicate_interaction
            .interactions
            .push(duplicate_interaction.interactions[0].clone());
        duplicate_interaction.interaction_capture.observed += 1;
        duplicate_interaction.interaction_capture.persisted += 1;
        assert!(matches!(
            extract_phenotype_interactions(&duplicate_interaction),
            Err(PhenotypeExtractionError::DuplicateInteraction { .. })
        ));

        let mut duplicate_arrival = fixture();
        duplicate_arrival
            .arrivals
            .push(duplicate_arrival.arrivals[0].clone());
        assert!(matches!(
            extract_phenotype_interactions(&duplicate_arrival),
            Err(PhenotypeExtractionError::DuplicateArrival { .. })
        ));

        let mut missing = fixture();
        missing
            .arrivals
            .retain(|arrival| arrival.agent_uid != AgentUid(2));
        assert!(matches!(
            extract_phenotype_interactions(&missing),
            Err(PhenotypeExtractionError::MissingAgentIdentity {
                source_name: "observation",
                agent_uid: AgentUid(2),
            })
        ));
    }

    #[test]
    fn nonfinite_and_invalid_interactions_fail_typed() {
        for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut ledger = fixture();
            ledger.observations[0].velocity_x = nonfinite;
            assert!(matches!(
                extract_phenotype_interactions(&ledger),
                Err(PhenotypeExtractionError::NonFinite {
                    field: "velocity_x",
                    ..
                })
            ));
        }

        for magnitude in [None, Some(0.0), Some(-1.0), Some(f64::NAN)] {
            let mut ledger = fixture();
            ledger.interactions[0].magnitude = magnitude;
            assert!(matches!(
                extract_phenotype_interactions(&ledger),
                Err(PhenotypeExtractionError::InvalidInteractionMagnitude { .. })
            ));
        }

        let mut self_edge = fixture();
        self_edge.interactions[0].target = self_edge.interactions[0].actor;
        assert!(matches!(
            extract_phenotype_interactions(&self_edge),
            Err(PhenotypeExtractionError::SelfInteraction { .. })
        ));

        let mut unknown_kind = fixture();
        unknown_kind.interactions[0].kind = "courtship".to_owned();
        assert!(matches!(
            extract_phenotype_interactions(&unknown_kind),
            Err(PhenotypeExtractionError::UnsupportedInteractionKind { .. })
        ));
    }

    #[test]
    fn empty_observation_window_is_insufficient_not_a_nan_report() {
        let mut ledger = fixture();
        ledger.observations.clear();
        assert!(matches!(
            extract_phenotype_interactions(&ledger),
            Err(PhenotypeExtractionError::InsufficientWindow {
                start_tick: 0,
                end_tick: 4,
            })
        ));
    }

    #[test]
    fn test_phenotype_row_conversion_and_map_elites_descriptor() {
        let row = PhenotypeFeatureRowV1 {
            run_id: RUN_A.to_owned(),
            agent_uid: AgentUid(42),
            observed_tick_count: 10,
            values: [1.5, 0.8, 0.4, 0.2, 0.1, 0.05],
        };

        let vec = row.to_agent_phenotype_vector();
        assert_eq!(vec.agent_uid, AgentUid(42));
        assert!((vec.movement_speed_mean - 1.5).abs() < 1e-6);
        assert!((vec.diet_herbivore_ratio - 0.8).abs() < 1e-6);
        assert!((vec.sensing_range_mean - 0.4).abs() < 1e-6);
        assert!((vec.aggression_index - 0.2).abs() < 1e-6);
        assert!((vec.giving_altruism_index - 0.1).abs() < 1e-6);
        assert!((vec.reproduction_rate - 0.05).abs() < 1e-6);

        let desc = row.behavior_descriptor();
        assert_eq!(desc, [1.5, 0.8, 0.4, 0.2, 0.1, 0.05]);
    }

    #[test]
    fn test_compare_phenotype_cohorts_statistical_metrics() {
        let cohort_a: Vec<PhenotypeFeatureRowV1> = (0..20)
            .map(|i| PhenotypeFeatureRowV1 {
                run_id: RUN_A.to_owned(),
                agent_uid: AgentUid(i),
                observed_tick_count: 5,
                values: [1.0 + (i as f64) * 0.01, 0.2, 0.3, 0.1, 0.0, 0.01],
            })
            .collect();

        let cohort_b: Vec<PhenotypeFeatureRowV1> = (20..40)
            .map(|i| PhenotypeFeatureRowV1 {
                run_id: RUN_B.to_owned(),
                agent_uid: AgentUid(i),
                observed_tick_count: 5,
                values: [5.0 + (i as f64) * 0.01, 0.8, 0.7, 0.4, 0.2, 0.05],
            })
            .collect();

        let report = compare_phenotype_cohorts("cohort_a", &cohort_a, "cohort_b", &cohort_b, 42)
            .expect("cohort comparison succeeds");

        assert_eq!(report.sample_size_a, 20);
        assert_eq!(report.sample_size_b, 20);
        assert_eq!(report.feature_comparisons.len(), PHENOTYPE_AXIS_COUNT_V1);

        // Speed comparison (axis 0): cohort_b is significantly faster (~5.3 vs ~1.1)
        let speed_comp = &report.feature_comparisons[0];
        assert_eq!(speed_comp.feature_name, "movement.speed.mean");
        assert!(speed_comp.mean_b > speed_comp.mean_a);
        assert!(speed_comp.cohens_d > 10.0, "large effect size on speed");
        assert_eq!(
            speed_comp.cliffs_delta, 1.0,
            "cohort B completely dominates cohort A on speed"
        );
        assert!(speed_comp.mean_diff_ci_95.is_some());
        let ci = speed_comp.mean_diff_ci_95.unwrap();
        assert!(
            ci[0] > 3.0 && ci[1] < 5.0,
            "95% CI covers true difference ~4.0"
        );
        assert!(
            speed_comp.permutation_p_value.unwrap() < 0.01,
            "significant difference"
        );
    }

    #[test]
    fn test_cluster_cohort_species() {
        let cohort: Vec<PhenotypeFeatureRowV1> = (0..10)
            .map(|i| PhenotypeFeatureRowV1 {
                run_id: RUN_A.to_owned(),
                agent_uid: AgentUid(i),
                observed_tick_count: 5,
                values: [1.0, 0.5, 0.5, 0.1, 0.1, 0.01],
            })
            .collect();

        let params = scriptbots_core::species::SpeciesParams::default();
        let prev = scriptbots_core::species::SpeciesTable {
            tick: Tick(0),
            species: Vec::new(),
            next_id: scriptbots_core::species::SpeciesId(1),
        };

        let (table, report) = cluster_cohort_species(&cohort, &params, &prev, Tick(10));
        assert_eq!(report.total_agents_segmented, 10);
        assert!(!table.species.is_empty(), "minted at least one species");
    }

    #[test]
    fn test_compute_lineage_phenotype_shifts() {
        let parent = PhenotypeFeatureRowV1 {
            run_id: RUN_A.to_owned(),
            agent_uid: AgentUid(1),
            observed_tick_count: 10,
            values: [2.0, 0.5, 0.5, 0.1, 0.1, 0.02],
        };
        let child = PhenotypeFeatureRowV1 {
            run_id: RUN_A.to_owned(),
            agent_uid: AgentUid(2),
            observed_tick_count: 5,
            values: [2.5, 0.6, 0.4, 0.1, 0.1, 0.03],
        };

        let arrivals = vec![RunScopedArrivalV1 {
            run_id: RUN_A.to_owned(),
            tick: 5,
            agent_uid: AgentUid(2),
            parent_a: Some(AgentUid(1)),
            parent_b: None,
        }];

        let (shifts, summary) = compute_lineage_phenotype_shifts(&[parent, child], &arrivals);
        assert_eq!(shifts.len(), 1);
        assert_eq!(shifts[0].parent_uid, AgentUid(1));
        assert_eq!(shifts[0].child_uid, AgentUid(2));
        assert!((shifts[0].deltas[0] - 0.5).abs() < 1e-6); // speed child - parent = 0.5
        assert!((shifts[0].deltas[1] - 0.1).abs() < 1e-6); // diet child - parent = 0.1
        assert!((shifts[0].deltas[2] - (-0.1)).abs() < 1e-6); // sensing child - parent = -0.1

        assert_eq!(summary.pair_count, 1);
        assert!((summary.mean_deltas[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_compute_core_phenotype_analysis() {
        let cohort_a: Vec<PhenotypeFeatureRowV1> = (0..5)
            .map(|i| PhenotypeFeatureRowV1 {
                run_id: RUN_A.to_owned(),
                agent_uid: AgentUid(i),
                observed_tick_count: 5,
                values: [1.0, 0.2, 0.3, 0.1, 0.0, 0.01],
            })
            .collect();

        let cohort_b: Vec<PhenotypeFeatureRowV1> = (5..10)
            .map(|i| PhenotypeFeatureRowV1 {
                run_id: RUN_A.to_owned(),
                agent_uid: AgentUid(i),
                observed_tick_count: 5,
                values: [3.0, 0.8, 0.6, 0.4, 0.2, 0.05],
            })
            .collect();

        let report = compute_core_phenotype_analysis(
            RUN_A,
            Tick(50),
            &[("cohort_a", &cohort_a), ("cohort_b", &cohort_b)],
        );

        assert_eq!(report.run_id, RUN_A);
        assert_eq!(report.tick, Tick(50));
        assert_eq!(report.total_agents_analyzed, 10);
        assert_eq!(report.comparisons.len(), 1);
        assert_eq!(report.comparisons[0].cluster_a_name, "cohort_a");
        assert_eq!(report.comparisons[0].cluster_b_name, "cohort_b");
    }
}
