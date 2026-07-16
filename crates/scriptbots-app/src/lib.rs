//! Shared application plumbing for ScriptBots control surfaces.

use std::sync::{Arc, Mutex, OnceLock};

use scriptbots_core::{
    CharacterizationDigestV0, CharacterizationError, CoreBuildIdentityV0,
    PersistenceAdmissionSession, ScriptBotsConfig, TickEvents, WorldDigestV1,
    WorldDigestV1ContractError, WorldState, rng_domains::DomainStreamsCheckpoint,
};
use scriptbots_runtime::RunId;
pub use scriptbots_storage::STORAGE_SIDECAR_SUFFIXES;
use scriptbots_storage::{AnalyticsSnapshotProvider, RunManifestRecord};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type SharedWorld = Arc<Mutex<WorldState>>;
pub type SharedAnalytics = AnalyticsSnapshotProvider;

/// Schema identifier for the run-scoped stable-identity/domain-stream manifest.
pub const RUN_MANIFEST_V3_SCHEMA: &str = "scriptbots.run-manifest.v3";
/// Compatible V3 minor schema used when a manifest carries bootstrap execution evidence.
pub const RUN_MANIFEST_V3_BOOTSTRAP_SCHEMA: &str = "scriptbots.run-manifest.v3.1";
/// Schema identifier for a sequence of V2 world characterization points.
pub const CHARACTERIZATION_TRACE_V2_SCHEMA: &str = "scriptbots.characterization-trace.v2";
/// Safety bound for the temporary characterization runner.
pub const MAX_CHARACTERIZATION_TICKS_V2: u64 = 256;
/// Maximum UTF-8 byte length for an experiment or variant identifier.
pub const MAX_RUN_IDENTITY_ID_BYTES: usize = 128;
/// Maximum UTF-8 byte length for the explicit live-run policy identifier.
pub const MAX_LIVE_RUN_POLICY_BYTES: usize = 256;
/// Maximum UTF-8 byte length for the stable scenario identifier stored by FrankenSQLite.
pub const MAX_SCENARIO_ID_BYTES: usize = 512;
const CARGO_LOCK_BYTES: &[u8] = include_bytes!("../../../Cargo.lock");
const RUST_TOOLCHAIN_BYTES: &[u8] = include_bytes!("../../../rust-toolchain.toml");
const RUST_TOOLCHAIN_TEXT: &str = include_str!("../../../rust-toolchain.toml");

/// Build identity V0 embedded in the V3 run manifest and characterization trace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BuildProvenanceV0 {
    pub package_name: String,
    pub package_version: String,
    pub source_revision: Option<String>,
    pub source_branch: Option<String>,
    pub source_tree_clean: Option<bool>,
    pub source_status_digest: Option<String>,
    pub source_diff_digest: Option<String>,
    pub declared_toolchain: String,
    pub compiler_toolchain: Option<String>,
    pub rustc_vv: Option<String>,
    pub toolchain_file_digest: String,
    pub lockfile_digest: String,
    pub compiled_features: Vec<String>,
    pub core: CoreBuildIdentityV0,
    pub rustflags: Option<String>,
    pub rayon_num_threads: Option<String>,
    pub scriptbots_max_threads: Option<String>,
    pub provenance_complete: bool,
    pub warnings: Vec<String>,
}

/// The mutable runtime environment values provenance records, pinned at launch.
///
/// Startup legitimately mutates the process environment after resolving the thread
/// policy — it is still the only channel to the Rayon pool builder (the bd-3p7i
/// "env-as-IPC" disease note; the value-passing cure belongs to the `HostCore`
/// config lane). Capturing at manifest time therefore recorded OUR OWN write as the
/// user's environment: export 16, pass `--threads 4`, and the capture said "4" — wrong
/// about the one fact it purports to record. The snapshot is pinned by the first
/// caller, which is the binary's first statement, so the record is what the USER
/// launched with rather than what startup smeared over it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchEnvironmentV0 {
    /// `RAYON_NUM_THREADS` exactly as the process was launched.
    pub rayon_num_threads: Option<String>,
    /// `SCRIPTBOTS_MAX_THREADS` exactly as the process was launched.
    pub scriptbots_max_threads: Option<String>,
}

static LAUNCH_ENVIRONMENT: OnceLock<LaunchEnvironmentV0> = OnceLock::new();

impl LaunchEnvironmentV0 {
    /// Pin — or return the already pinned — launch environment snapshot.
    ///
    /// Call before any startup code mutates the process environment; every later call
    /// gets the pinned snapshot regardless of what has been written since.
    #[must_use]
    pub fn pin() -> &'static Self {
        LAUNCH_ENVIRONMENT.get_or_init(|| Self {
            rayon_num_threads: std::env::var("RAYON_NUM_THREADS").ok(),
            scriptbots_max_threads: std::env::var("SCRIPTBOTS_MAX_THREADS").ok(),
        })
    }
}

impl BuildProvenanceV0 {
    /// Capture provenance embedded in the current `scriptbots-app` build.
    ///
    /// Reproducible builds must embed `SCRIPTBOTS_SOURCE_REVISION`,
    /// `SCRIPTBOTS_SOURCE_TREE_CLEAN=true`, `SCRIPTBOTS_SOURCE_STATUS_DIGEST`,
    /// `SCRIPTBOTS_SOURCE_DIFF_DIGEST`, and `SCRIPTBOTS_RUSTC_VV` while invoking Cargo. These are
    /// deliberately compile-time inputs: a shipped binary must never describe the caller's
    /// current directory, current Git checkout, or currently installed compiler as the source of
    /// the already-built executable. Missing values remain visibly unknown; provenance is never
    /// fabricated. The two runtime environment fields come from the pinned
    /// [`LaunchEnvironmentV0`], never from a live read that startup may have overwritten.
    #[must_use]
    pub fn current() -> Self {
        let launch_environment = LaunchEnvironmentV0::pin();
        let source_revision = compile_time_text(option_env!("SCRIPTBOTS_SOURCE_REVISION"));
        let source_branch = compile_time_text(option_env!("SCRIPTBOTS_SOURCE_BRANCH"));
        let source_tree_clean =
            option_env!("SCRIPTBOTS_SOURCE_TREE_CLEAN").and_then(parse_compile_time_bool);
        let source_status_digest =
            compile_time_text(option_env!("SCRIPTBOTS_SOURCE_STATUS_DIGEST"));
        let source_diff_digest = compile_time_text(option_env!("SCRIPTBOTS_SOURCE_DIFF_DIGEST"));
        let compiler_toolchain = compile_time_text(option_env!("RUSTUP_TOOLCHAIN"));
        let rustc_vv = compile_time_text(option_env!("SCRIPTBOTS_RUSTC_VV"));

        let mut compiled_features = Vec::new();
        for (enabled, name) in [
            (cfg!(feature = "bevy_render"), "bevy_render"),
            (cfg!(feature = "brain-ft"), "brain-ft"),
            (cfg!(feature = "fast-alloc"), "fast-alloc"),
            (cfg!(feature = "gui"), "gui"),
            (cfg!(feature = "llm-anthropic"), "llm-anthropic"),
            (cfg!(feature = "ml"), "ml"),
            (cfg!(feature = "neuro"), "neuro"),
        ] {
            if enabled {
                compiled_features.push(name.to_owned());
            }
        }
        compiled_features.sort_unstable();

        let mut warnings = Vec::new();
        if source_revision.is_none() {
            warnings.push("source revision is unavailable".to_owned());
        }
        match source_tree_clean {
            Some(true) => {}
            Some(false) => warnings.push("source tree is dirty".to_owned()),
            None => warnings.push("source tree cleanliness is unavailable".to_owned()),
        }
        if source_status_digest.is_none() {
            warnings.push("source status digest is unavailable".to_owned());
        }
        if source_diff_digest.is_none() {
            warnings.push("source diff digest is unavailable".to_owned());
        }
        if rustc_vv.is_none() {
            warnings.push("build-time rustc -Vv output is unavailable".to_owned());
        }
        let mut provenance = Self {
            package_name: env!("CARGO_PKG_NAME").to_owned(),
            package_version: env!("CARGO_PKG_VERSION").to_owned(),
            source_revision,
            source_branch,
            source_tree_clean,
            source_status_digest,
            source_diff_digest,
            declared_toolchain: tracked_toolchain_spec(),
            compiler_toolchain,
            rustc_vv,
            toolchain_file_digest: manifest_digest("rust-toolchain.toml", RUST_TOOLCHAIN_BYTES),
            lockfile_digest: manifest_digest("Cargo.lock", CARGO_LOCK_BYTES),
            compiled_features,
            core: CoreBuildIdentityV0::current(),
            rustflags: option_env!("RUSTFLAGS").map(str::to_owned),
            rayon_num_threads: launch_environment.rayon_num_threads.clone(),
            scriptbots_max_threads: launch_environment.scriptbots_max_threads.clone(),
            provenance_complete: false,
            warnings,
        };
        provenance.provenance_complete = provenance.derived_provenance_complete();
        provenance
    }

    /// Recompute whether every build/source field required for a reproducibility claim exists.
    ///
    /// The serialized boolean is retained so readers can filter without reimplementing this
    /// contract, but constructors and storage projection verify that it equals this derivation.
    #[must_use]
    pub fn derived_provenance_complete(&self) -> bool {
        self.source_revision
            .as_deref()
            .is_some_and(is_nonblank_text)
            && self.source_tree_clean == Some(true)
            && self
                .source_status_digest
                .as_deref()
                .is_some_and(is_nonblank_text)
            && self
                .source_diff_digest
                .as_deref()
                .is_some_and(is_nonblank_text)
            && self.rustc_vv.as_deref().is_some_and(is_nonblank_text)
            && is_nonblank_text(&self.declared_toolchain)
            && is_nonblank_text(&self.toolchain_file_digest)
            && is_nonblank_text(&self.lockfile_digest)
    }
}

/// Temporary identity for the scenario construction used by a V0 characterization run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScenarioIdentityV0 {
    pub id: String,
    pub schema_version: u16,
    pub ordered_config_layer_digests: Vec<String>,
    pub population_recipe: String,
    pub bootstrap_ticks: u64,
}

impl ScenarioIdentityV0 {
    #[must_use]
    pub fn caller_seeded(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            schema_version: 0,
            ordered_config_layer_digests: Vec::new(),
            population_recipe: "caller_seeded_world_v0".to_owned(),
            bootstrap_ticks: 0,
        }
    }

    /// Append one configuration layer's content digest in application order.
    ///
    /// Each entry is prefixed with the layer kind's wire tag (`file:`, `environment:`,
    /// `cli:`), so a manifest can say not just WHAT the config was but WHICH KINDS of
    /// layer built it, in order. File layers digest their exact source bytes — those
    /// bytes ARE the layer — while the environment and CLI layers digest their
    /// canonical statement bytes.
    pub fn record_config_layer(&mut self, kind: precedence::ConfigLayerKind, bytes: &[u8]) {
        self.ordered_config_layer_digests.push(format!(
            "{}:{}",
            kind.wire_tag(),
            manifest_digest("config-layer-v0", bytes)
        ));
    }
}

/// Registered brain family recorded in stable key order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BrainRosterEntryV0 {
    pub registry_key: u64,
    pub kind: String,
}

/// Machine-readable statement of what V0 does and does not prove.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CharacterizationLimitationsV0 {
    pub purpose: String,
    pub agent_identity: String,
    pub source_identity: String,
    pub evaluator_state_covered: bool,
    pub rng_state_restorable: bool,
    pub checkpoint_replay_guarantee: bool,
    pub comparison_lane: String,
    pub superseded_by: String,
}

impl Default for CharacterizationLimitationsV0 {
    fn default() -> Self {
        Self {
            purpose: "characterization_only".to_owned(),
            agent_identity: "AgentState snapshots and persistence use stable AgentUid; RunManifestV3 records run identity and allocation counters; legacy CharacterizationDigestV0 still orders agents by transient AgentId and excludes AgentUid".to_owned(),
            source_identity:
                "commit plus status/tracked-diff digests; untracked file contents are not hashed"
                    .to_owned(),
            evaluator_state_covered: false,
            rng_state_restorable: true,
            checkpoint_replay_guarantee: false,
            comparison_lane: "same pinned toolchain, target, features, and thread lane".to_owned(),
            superseded_by: "WorldDigestV1".to_owned(),
        }
    }
}

/// The thread policy a run actually resolved to, and WHICH LAYER decided it.
///
/// `BuildProvenanceV0` already captures `RAYON_NUM_THREADS` and `SCRIPTBOTS_MAX_THREADS` — but
/// those are what the ENVIRONMENT said, not what the run DECIDED, and those are different facts.
/// A user who exported `SCRIPTBOTS_MAX_THREADS=16` and passed `--threads 8` RAN ON 8, while the
/// environment capture still says 16. A manifest carrying only the environment therefore describes
/// a run that did not happen — and it does so precisely in the case the precedence rules exist to
/// handle.
///
/// `source` matters as much as the number. Two runs that both used 8 threads — one because the
/// operator asked for 8, the other because the auto-tune probe measured its way there — have
/// different provenance, and a reader comparing them needs to tell which is which.
///
/// `overridden` names a layer whose suggestion was DECLINED because a more specific layer had
/// already spoken. That is the normal, correct outcome of the rules rather than a mistake — but it
/// must be visible: a user who passes `--low-power` alongside `--threads 16` deserves to learn
/// that low-power did not lower their thread count from the run's own record, not from a power
/// bill.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThreadPolicyV0 {
    /// The resolved cap. `None` means no layer named a value and Rayon's own default was left
    /// alone — which is itself a decision, recorded as one rather than as a silent absence.
    pub threads: Option<usize>,
    /// Which layer won: `cli-flag`, `environment`, `auto-tune`, `low-power-default`, or
    /// `builtin-default`.
    pub source: String,
    /// The layer whose suggestion was declined, if any.
    pub overridden: Option<String>,
}

/// Exact scientific boundaries proving an explicitly requested startup warmup.
///
/// Both digests are complete [`WorldDigestV1`] values rather than tick-only claims. The start
/// boundary is captured from the seeded launch state before any bootstrap transition, and the end
/// boundary is captured after `completed` persistence-session steps.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BootstrapEvidenceV0 {
    /// Number of bootstrap transitions requested by startup policy.
    pub requested: u64,
    /// Number of persistence-session transitions that completed successfully.
    pub completed: u64,
    /// Launch-state digest captured before the first bootstrap transition.
    pub start: WorldDigestV1,
    /// Final digest captured after the last completed bootstrap transition.
    pub end: WorldDigestV1,
}

/// Run-scoped identity and launch intent embedded in [`RunManifestV3`].
///
/// `started_at_unix_ms` is the launch boundary measured in milliseconds since the Unix epoch. It
/// is metadata, not a deterministic simulation input. Exactly one execution boundary must be
/// present. `requested_tick_budget` is the finite number of scientific transitions requested at
/// launch: `Some(0)` explicitly requests no transitions and `Some(u64::MAX)` is valid.
/// Alternatively, `live_run_policy` is a bounded policy identifier describing how an unbounded or
/// externally controlled run terminates or may be extended. Storage treats that policy as
/// provenance and does not infer behavior from the string.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunIdentityV1 {
    /// Globally stable run identifier, encoded as 32 lowercase hexadecimal characters.
    pub run_id: RunId,
    /// Optional experiment grouping key. When present it must be nonblank and bounded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment_id: Option<String>,
    /// Optional variant key within an experiment. When present it must be nonblank and bounded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant_id: Option<String>,
    /// Launch boundary in milliseconds since the Unix epoch.
    pub started_at_unix_ms: u64,
    /// Finite requested scientific transition count; mutually exclusive with `live_run_policy`.
    pub requested_tick_budget: Option<u64>,
    /// Unbounded-run policy identifier; mutually exclusive with `requested_tick_budget`.
    pub live_run_policy: Option<String>,
}

impl RunIdentityV1 {
    /// Construct an identity contract. Validation occurs when it is bound to a manifest.
    #[must_use]
    pub fn new(
        run_id: RunId,
        started_at_unix_ms: u64,
        requested_tick_budget: Option<u64>,
        live_run_policy: Option<String>,
    ) -> Self {
        Self {
            run_id,
            experiment_id: None,
            variant_id: None,
            started_at_unix_ms,
            requested_tick_budget,
            live_run_policy,
        }
    }

    /// Validate bounded human-supplied identity fields without changing their wire values.
    pub fn validate(&self) -> Result<(), RunManifestError> {
        if self.run_id.get() == 0 {
            return Err(RunManifestError::ZeroRunId);
        }
        match (
            self.requested_tick_budget.is_some(),
            self.live_run_policy.is_some(),
        ) {
            (false, false) => return Err(RunManifestError::MissingRunExecutionBoundary),
            (true, true) => return Err(RunManifestError::ConflictingRunExecutionBoundaries),
            (true, false) | (false, true) => {}
        }
        validate_run_identity_text(
            "experiment_id",
            self.experiment_id.as_deref(),
            MAX_RUN_IDENTITY_ID_BYTES,
        )?;
        validate_run_identity_text(
            "variant_id",
            self.variant_id.as_deref(),
            MAX_RUN_IDENTITY_ID_BYTES,
        )?;
        validate_run_identity_text(
            "live_run_policy",
            self.live_run_policy.as_deref(),
            MAX_LIVE_RUN_POLICY_BYTES,
        )
    }
}

/// Version-three record tying run identity, scenario construction, stable identity allocation,
/// domain-separated random-stream continuation, and normalized configuration to a build.
///
/// `reproducible` means the manifest has an explicit seed and complete clean-source provenance. It
/// does not override the characterization digest's exclusions or claim that replay can reconstruct
/// the world.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RunManifestV3 {
    pub schema: String,
    pub schema_version: u16,
    pub purpose: String,
    /// Required run-scoped identity and launch-intent contract.
    pub identity: RunIdentityV1,
    pub root_seed: u64,
    /// How many threads this run actually used, and which layer decided — see [`ThreadPolicyV0`].
    ///
    /// `None` only for a manifest built outside the binary (tests, tooling), where no policy was
    /// resolved. A real run always records one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_policy: Option<ThreadPolicyV0>,
    /// Cross-layer configuration displacements the composed config resolved — the config
    /// analogue of [`ThreadPolicyV0`]'s `overridden`.
    ///
    /// Empty when no explicit layer displaced another explicit layer's value (displacing a
    /// built-in default is configuration, not a displacement). A user whose scenario file
    /// said one thing and whose environment said another reads it here, in the run's own
    /// record, rather than discovering it from the results.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config_overrides: Vec<precedence::ConfigFieldOverride>,
    /// Explicit proof of the requested startup warmup, attached only after it completes.
    ///
    /// Attaching it through [`Self::with_bootstrap_evidence`] validates the two digests and
    /// upgrades the schema tag to [`RUN_MANIFEST_V3_BOOTSTRAP_SCHEMA`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_evidence: Option<BootstrapEvidenceV0>,
    pub random_streams: DomainStreamsCheckpoint,
    pub next_agent_uid: u64,
    pub next_spawn_ordinal: u64,
    pub next_birth_ordinal: u64,
    pub scenario: ScenarioIdentityV0,
    pub normalized_config: serde_json::Value,
    pub config_digest: String,
    pub config_digest_encoding: String,
    pub build: BuildProvenanceV0,
    pub brain_roster: Vec<BrainRosterEntryV0>,
    pub reproducible: bool,
    pub warnings: Vec<String>,
    pub limitations: CharacterizationLimitationsV0,
}

/// Errors returned while constructing a version-three run manifest.
#[derive(Debug, Error)]
pub enum RunManifestError {
    #[error("scenario identity must not be empty")]
    EmptyScenarioIdentity,
    /// A stable scenario identifier exceeded the durable storage boundary.
    #[error("scenario identity is {actual} bytes; maximum is {maximum}")]
    ScenarioIdentityTooLong {
        /// Actual UTF-8 byte length.
        actual: usize,
        /// Maximum permitted UTF-8 byte length.
        maximum: usize,
    },
    /// A stable scenario identifier contained a control character.
    #[error("scenario identity must not contain control characters")]
    ScenarioIdentityControlCharacter,
    /// Zero is the invalid/sentinel run identity and cannot name a durable run.
    #[error("run identity cannot use the zero RunId sentinel")]
    ZeroRunId,
    /// Neither finite nor live execution semantics were supplied.
    #[error("run identity requires exactly one of requested_tick_budget or live_run_policy")]
    MissingRunExecutionBoundary,
    /// Finite and live execution semantics were both supplied.
    #[error("run identity cannot combine requested_tick_budget with live_run_policy")]
    ConflictingRunExecutionBoundaries,
    /// A required or present identity field contained only whitespace.
    #[error("run identity field {field} must not be blank")]
    BlankRunIdentityField {
        /// Stable wire-field name.
        field: &'static str,
    },
    /// A human-supplied identity field exceeded its explicit UTF-8 byte bound.
    #[error("run identity field {field} is {actual} bytes; maximum is {maximum}")]
    RunIdentityFieldTooLong {
        /// Stable wire-field name.
        field: &'static str,
        /// Actual UTF-8 byte length.
        actual: usize,
        /// Maximum permitted UTF-8 byte length.
        maximum: usize,
    },
    /// A human-supplied identity field contained a control character.
    #[error("run identity field {field} must not contain control characters")]
    RunIdentityControlCharacter {
        /// Stable wire-field name.
        field: &'static str,
    },
    /// The serialized build completeness flag contradicted its required evidence.
    #[error(
        "build provenance_complete={recorded}, but the required embedded evidence derives {derived}"
    )]
    InconsistentBuildProvenance {
        /// Boolean carried by the build record.
        recorded: bool,
        /// Boolean derived from the build record's evidence.
        derived: bool,
    },
    /// The manifest-level reproduction claim contradicted its validated build provenance.
    #[error("run reproducible={recorded}, but validated build provenance derives {derived}")]
    InconsistentReproducibilityClaim {
        /// Boolean carried by the manifest.
        recorded: bool,
        /// Validated build-provenance result.
        derived: bool,
    },
    #[error("run manifest V3 requires an explicit rng_seed")]
    MissingExplicitSeed,
    /// The manifest root seed and the domain-stream checkpoint disagree.
    #[error(
        "run manifest root seed {manifest} does not match random-stream checkpoint root seed {checkpoint}"
    )]
    RandomStreamRootSeedMismatch {
        /// Root seed recorded by the manifest.
        manifest: u64,
        /// Root seed carried by the domain-stream checkpoint.
        checkpoint: u64,
    },
    #[error("failed to normalize ScriptBots configuration: {0}")]
    ConfigSerialization(#[source] serde_json::Error),
    #[error("failed to encode normalized ScriptBots configuration for its BLAKE3 digest: {0}")]
    ConfigDigestSerialization(String),
    /// A canonical JSON projection required by durable storage could not be encoded.
    #[error("failed to encode run-manifest storage field {field}: {source}")]
    StorageProjectionSerialization {
        /// Stable field name in [`RunManifestRecord`].
        field: &'static str,
        /// JSON serialization failure.
        #[source]
        source: serde_json::Error,
    },
    /// Bootstrap evidence was already attached and cannot be silently replaced.
    #[error("run manifest already carries bootstrap evidence")]
    BootstrapEvidenceAlreadyAttached,
    /// The evidence described a different request from the launch scenario.
    #[error(
        "bootstrap evidence requested {evidence_requested} ticks, but the scenario requested {scenario_requested}"
    )]
    BootstrapRequestMismatch {
        /// Bootstrap count recorded in the launch scenario.
        scenario_requested: u64,
        /// Bootstrap count claimed by the evidence.
        evidence_requested: u64,
    },
    /// A manifest may be written only after the requested session transitions all complete.
    #[error("bootstrap completed {completed} of {requested} requested ticks")]
    BootstrapCompletionMismatch {
        /// Bootstrap count requested at launch.
        requested: u64,
        /// Successfully completed persistence-session transitions.
        completed: u64,
    },
    /// Fresh-run bootstrap evidence must begin at the visible tick-zero launch boundary.
    #[error("bootstrap evidence starts at tick {found}, expected tick 0")]
    BootstrapStartTick {
        /// Unexpected starting tick.
        found: u64,
    },
    /// The requested transition count overflowed the tick domain.
    #[error("bootstrap tick arithmetic overflowed: start={start}, completed={completed}")]
    BootstrapTickOverflow {
        /// Starting scientific tick.
        start: u64,
        /// Number of completed transitions.
        completed: u64,
    },
    /// The end digest did not describe the boundary reached by the completed transition count.
    #[error("bootstrap evidence ends at tick {found}, expected tick {expected}")]
    BootstrapEndTick {
        /// Expected ending tick.
        expected: u64,
        /// Ending tick carried by the evidence.
        found: u64,
    },
    /// A zero-transition warmup changed scientific state despite executing no transition.
    #[error("zero-tick bootstrap evidence must carry identical start and end digests")]
    BootstrapZeroChanged,
    /// One of the embedded world digests violated the V1 contract.
    #[error("bootstrap {boundary} digest violates the WorldDigestV1 contract: {source}")]
    BootstrapDigest {
        /// Whether the invalid digest was the `start` or `end` boundary.
        boundary: &'static str,
        /// Typed world-digest contract failure.
        #[source]
        source: WorldDigestV1ContractError,
    },
}

impl RunManifestV3 {
    /// Project the complete manifest into the storage-owned queryable provenance contract.
    ///
    /// The full canonical manifest remains embedded, while frequently filtered provenance fields
    /// are duplicated into validated scalar columns. Storage registers this record in the same
    /// transaction as its feature set and initial persistence watermarks, before tick zero.
    pub fn to_storage_record(&self) -> Result<RunManifestRecord, RunManifestError> {
        self.identity.validate()?;
        validate_scenario_identity(&self.scenario.id)?;
        if self.random_streams.root_seed != self.root_seed {
            return Err(RunManifestError::RandomStreamRootSeedMismatch {
                manifest: self.root_seed,
                checkpoint: self.random_streams.root_seed,
            });
        }
        let derived_provenance = validate_build_provenance_claim(&self.build)?;
        if self.reproducible != derived_provenance {
            return Err(RunManifestError::InconsistentReproducibilityClaim {
                recorded: self.reproducible,
                derived: derived_provenance,
            });
        }
        let normalized_config_json =
            canonical_json_text(&self.normalized_config).map_err(|source| {
                RunManifestError::StorageProjectionSerialization {
                    field: "normalized_config_json",
                    source,
                }
            })?;
        let brain_roster_json = canonical_json_text(&self.brain_roster).map_err(|source| {
            RunManifestError::StorageProjectionSerialization {
                field: "brain_roster_json",
                source,
            }
        })?;
        let manifest_json = canonical_json_text(self).map_err(|source| {
            RunManifestError::StorageProjectionSerialization {
                field: "manifest_json",
                source,
            }
        })?;
        let target_triple = self
            .build
            .rustc_vv
            .as_deref()
            .and_then(|details| {
                details
                    .lines()
                    .find_map(|line| line.strip_prefix("host: ").map(str::to_owned))
            })
            .unwrap_or_else(|| {
                format!(
                    "{}-unknown-{}",
                    self.build.core.target_arch, self.build.core.target_os
                )
            });

        Ok(RunManifestRecord {
            run_id: self.identity.run_id,
            manifest_schema_version: self.schema_version,
            experiment_id: self.identity.experiment_id.clone(),
            variant_id: self.identity.variant_id.clone(),
            scenario_id: self.scenario.id.clone(),
            scenario_version: self.scenario.schema_version,
            normalized_config_json,
            config_digest: self.config_digest.clone(),
            root_seed: self.root_seed,
            rng_algorithm: self.random_streams.algorithm.clone(),
            rng_version: self.random_streams.version,
            brain_roster_json,
            source_revision: self.build.source_revision.clone(),
            source_tree_digest: self
                .build
                .source_diff_digest
                .clone()
                .or_else(|| self.build.source_status_digest.clone()),
            source_tree_dirty: self.build.source_tree_clean.map(|clean| !clean),
            source_bundle_digest: None,
            rust_toolchain: self
                .build
                .compiler_toolchain
                .clone()
                .unwrap_or_else(|| self.build.declared_toolchain.clone()),
            cargo_lock_digest: self.build.lockfile_digest.clone(),
            target_triple,
            started_at_unix_ms: self.identity.started_at_unix_ms,
            requested_tick_budget: self.identity.requested_tick_budget,
            live_run_policy: self.identity.live_run_policy.clone(),
            reproducible: self.reproducible,
            features: self.build.compiled_features.clone(),
            manifest_json,
        })
    }

    /// Record the thread policy the startup path actually resolved.
    ///
    /// This is the only way the policy reaches the manifest: `from_world` cannot know it, because
    /// a world does not know how it was launched. Without this call a real run's manifest would
    /// carry only `BuildProvenanceV0`'s environment capture — which reports what the ENVIRONMENT
    /// said rather than what the run DECIDED, and those differ precisely when a more specific
    /// layer overrode the environment. A manifest describing a run that did not happen is worse
    /// than one that says nothing.
    #[must_use]
    pub fn with_thread_policy(mut self, policy: ThreadPolicyV0) -> Self {
        self.thread_policy = Some(policy);
        self
    }

    /// Record the cross-layer configuration displacements the startup path resolved.
    ///
    /// Reaches the manifest the same way the thread policy does, and for the same reason:
    /// a world cannot know which configuration layers fought over its knobs, only the
    /// startup composition path can.
    #[must_use]
    pub fn with_config_overrides(
        mut self,
        overrides: Vec<precedence::ConfigFieldOverride>,
    ) -> Self {
        self.config_overrides = overrides;
        self
    }

    /// Validate and attach exact bootstrap execution evidence.
    ///
    /// This is deliberately fallible: the manifest must never claim a request different from its
    /// launch scenario, claim partial completion as success, start after a hidden warmup, or carry
    /// malformed digest values. A zero-tick request is represented explicitly by two identical
    /// tick-zero digests.
    pub fn with_bootstrap_evidence(
        mut self,
        evidence: BootstrapEvidenceV0,
    ) -> Result<Self, RunManifestError> {
        if self.bootstrap_evidence.is_some() {
            return Err(RunManifestError::BootstrapEvidenceAlreadyAttached);
        }
        if evidence.requested != self.scenario.bootstrap_ticks {
            return Err(RunManifestError::BootstrapRequestMismatch {
                scenario_requested: self.scenario.bootstrap_ticks,
                evidence_requested: evidence.requested,
            });
        }
        if evidence.completed != evidence.requested {
            return Err(RunManifestError::BootstrapCompletionMismatch {
                requested: evidence.requested,
                completed: evidence.completed,
            });
        }
        if evidence.start.tick.0 != 0 {
            return Err(RunManifestError::BootstrapStartTick {
                found: evidence.start.tick.0,
            });
        }
        let expected_end = evidence
            .start
            .tick
            .0
            .checked_add(evidence.completed)
            .ok_or(RunManifestError::BootstrapTickOverflow {
                start: evidence.start.tick.0,
                completed: evidence.completed,
            })?;
        if evidence.end.tick.0 != expected_end {
            return Err(RunManifestError::BootstrapEndTick {
                expected: expected_end,
                found: evidence.end.tick.0,
            });
        }
        if evidence.completed == 0 && evidence.start != evidence.end {
            return Err(RunManifestError::BootstrapZeroChanged);
        }
        evidence
            .start
            .validate_contract()
            .map_err(|source| RunManifestError::BootstrapDigest {
                boundary: "start",
                source,
            })?;
        evidence
            .end
            .validate_contract()
            .map_err(|source| RunManifestError::BootstrapDigest {
                boundary: "end",
                source,
            })?;

        self.schema = RUN_MANIFEST_V3_BOOTSTRAP_SCHEMA.to_owned();
        self.bootstrap_evidence = Some(evidence);
        Ok(self)
    }

    /// Capture a run manifest using provenance embedded in the current build.
    pub fn from_world(
        identity: RunIdentityV1,
        scenario_id: impl Into<String>,
        world: &WorldState,
    ) -> Result<Self, RunManifestError> {
        Self::from_world_with_provenance(
            identity,
            ScenarioIdentityV0::caller_seeded(scenario_id),
            world,
            BuildProvenanceV0::current(),
        )
    }

    /// Capture a run manifest using explicitly supplied build provenance.
    ///
    /// This constructor supports release tooling and tests that obtain source revision and tree
    /// cleanliness through a trusted path outside the library.
    pub fn from_world_with_provenance(
        identity: RunIdentityV1,
        mut scenario: ScenarioIdentityV0,
        world: &WorldState,
        build: BuildProvenanceV0,
    ) -> Result<Self, RunManifestError> {
        identity.validate()?;
        validate_scenario_identity(&scenario.id)?;
        scenario.id = scenario.id.trim().to_owned();
        let reproducible = validate_build_provenance_claim(&build)?;

        let normalized_config = normalized_config(world.config())?;
        let config_digest_bytes = canonical_json_bytes(&normalized_config)
            .map_err(|error| RunManifestError::ConfigDigestSerialization(error.to_string()))?;
        let config_digest = format!("blake3:{}", blake3::hash(&config_digest_bytes).to_hex());
        let root_seed = world
            .config()
            .rng_seed
            .ok_or(RunManifestError::MissingExplicitSeed)?;
        let brain_roster = world
            .brain_registry()
            .descriptors()
            .into_iter()
            .map(|(registry_key, kind)| BrainRosterEntryV0 { registry_key, kind })
            .collect();
        let warnings = build.warnings.clone();
        let random_streams = world.random_streams_checkpoint();
        if random_streams.root_seed != root_seed {
            return Err(RunManifestError::RandomStreamRootSeedMismatch {
                manifest: root_seed,
                checkpoint: random_streams.root_seed,
            });
        }
        let (next_agent_uid, next_spawn_ordinal, next_birth_ordinal) =
            world.identity_sequence_state();

        Ok(Self {
            schema: RUN_MANIFEST_V3_SCHEMA.to_owned(),
            schema_version: 3,
            purpose: "characterization_only".to_owned(),
            identity,
            root_seed,
            // Left empty here on purpose: a manifest built from a world alone has no way to know
            // what the STARTUP path decided about threads. The binary attaches it via
            // `with_thread_policy`, so a manifest that carries no policy is one that was built
            // outside a real run — which is a true statement, not a missing field.
            thread_policy: None,
            config_overrides: Vec::new(),
            bootstrap_evidence: None,
            random_streams,
            next_agent_uid,
            next_spawn_ordinal,
            next_birth_ordinal,
            scenario,
            normalized_config,
            config_digest,
            config_digest_encoding: "blake3-canonical-json-v1".to_owned(),
            build,
            brain_roster,
            reproducible,
            warnings,
            limitations: CharacterizationLimitationsV0::default(),
        })
    }

    /// Serialize the manifest to deterministic compact JSON bytes.
    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        canonical_json_bytes(self)
    }

    /// Serialize the manifest to deterministic compact JSON text.
    pub fn canonical_json(&self) -> Result<String, serde_json::Error> {
        self.canonical_json_bytes()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// One boundary captured in a V2 characterization trace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TracePointV0 {
    pub tick: u64,
    pub digest: CharacterizationDigestV0,
    pub tick_events: Option<TickEvents>,
}

/// Bounded fixed-seed characterization trace for the current implementation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CharacterizationTraceV2 {
    pub schema: String,
    pub schema_version: u16,
    pub digest_algorithm: String,
    pub manifest: RunManifestV3,
    pub manifest_digest: String,
    pub points: Vec<TracePointV0>,
}

/// Failures produced while capturing a V2 characterization trace.
#[derive(Debug, Error)]
pub enum CharacterizationTraceErrorV2 {
    #[error("requested {requested} ticks, but characterization V2 is capped at {maximum}")]
    ExcessiveTickCount { requested: u64, maximum: u64 },
    #[error(transparent)]
    Manifest(#[from] RunManifestError),
    #[error(transparent)]
    Characterization(#[from] CharacterizationError),
    #[error(transparent)]
    Step(#[from] scriptbots_core::WorldStepError),
    #[error("failed to encode characterization artifact: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl CharacterizationTraceV2 {
    /// Capture tick zero and every boundary through `ticks` from a persistence-disabled world.
    ///
    /// When `world.config().persistence_interval` is nonzero, use
    /// [`Self::capture_with_scenario_and_session`] with the world's bound admission session.
    pub fn capture(
        identity: RunIdentityV1,
        scenario_id: impl Into<String>,
        world: &mut WorldState,
        ticks: u64,
    ) -> Result<Self, CharacterizationTraceErrorV2> {
        Self::capture_with_scenario(
            identity,
            ScenarioIdentityV0::caller_seeded(scenario_id),
            world,
            ticks,
        )
    }

    /// Capture a trace with explicit temporary scenario construction metadata.
    ///
    /// This direct stepping entry point is persistence-disabled-only. Persistence-enabled worlds
    /// must use [`Self::capture_with_scenario_and_session`] so no completed batch is discarded.
    pub fn capture_with_scenario(
        identity: RunIdentityV1,
        scenario: ScenarioIdentityV0,
        world: &mut WorldState,
        ticks: u64,
    ) -> Result<Self, CharacterizationTraceErrorV2> {
        Self::capture_with_scenario_and_step(identity, scenario, world, ticks, |world| world.step())
    }

    /// Capture through the world's external persistence session without changing its config.
    pub fn capture_with_scenario_and_session(
        identity: RunIdentityV1,
        scenario: ScenarioIdentityV0,
        world: &mut WorldState,
        persistence: &mut PersistenceAdmissionSession,
        ticks: u64,
    ) -> Result<Self, CharacterizationTraceErrorV2> {
        Self::capture_with_scenario_and_step(identity, scenario, world, ticks, |world| {
            persistence.step(world)
        })
    }

    fn capture_with_scenario_and_step(
        identity: RunIdentityV1,
        scenario: ScenarioIdentityV0,
        world: &mut WorldState,
        ticks: u64,
        mut step: impl FnMut(&mut WorldState) -> Result<TickEvents, scriptbots_core::WorldStepError>,
    ) -> Result<Self, CharacterizationTraceErrorV2> {
        if ticks > MAX_CHARACTERIZATION_TICKS_V2 {
            return Err(CharacterizationTraceErrorV2::ExcessiveTickCount {
                requested: ticks,
                maximum: MAX_CHARACTERIZATION_TICKS_V2,
            });
        }

        let manifest = RunManifestV3::from_world_with_provenance(
            identity,
            scenario,
            world,
            BuildProvenanceV0::current(),
        )?;
        let manifest_bytes = manifest.canonical_json_bytes()?;
        let manifest_digest = manifest_digest("run-manifest-v3", &manifest_bytes);
        let mut points = Vec::with_capacity(usize::try_from(ticks).unwrap_or(0) + 1);
        let initial = world.characterization_digest_v0()?;
        points.push(TracePointV0 {
            tick: initial.tick.0,
            digest: initial,
            tick_events: None,
        });
        for _ in 0..ticks {
            let events = step(world)?;
            let digest = world.characterization_digest_v0()?;
            points.push(TracePointV0 {
                tick: digest.tick.0,
                digest,
                tick_events: Some(events),
            });
        }

        Ok(Self {
            schema: CHARACTERIZATION_TRACE_V2_SCHEMA.to_owned(),
            schema_version: 2,
            digest_algorithm: "fnv1a64-v0".to_owned(),
            manifest,
            manifest_digest,
            points,
        })
    }

    /// Serialize the trace to deterministic compact JSON bytes.
    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        canonical_json_bytes(self)
    }

    /// Serialize the trace to deterministic compact JSON text.
    pub fn canonical_json(&self) -> Result<String, serde_json::Error> {
        self.canonical_json_bytes()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
    }
}

fn validate_run_identity_text(
    field: &'static str,
    value: Option<&str>,
    maximum: usize,
) -> Result<(), RunManifestError> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.trim().is_empty() {
        return Err(RunManifestError::BlankRunIdentityField { field });
    }
    if value.len() > maximum {
        return Err(RunManifestError::RunIdentityFieldTooLong {
            field,
            actual: value.len(),
            maximum,
        });
    }
    if value.chars().any(char::is_control) {
        return Err(RunManifestError::RunIdentityControlCharacter { field });
    }
    Ok(())
}

fn validate_scenario_identity(value: &str) -> Result<(), RunManifestError> {
    if value.trim().is_empty() {
        return Err(RunManifestError::EmptyScenarioIdentity);
    }
    if value.len() > MAX_SCENARIO_ID_BYTES {
        return Err(RunManifestError::ScenarioIdentityTooLong {
            actual: value.len(),
            maximum: MAX_SCENARIO_ID_BYTES,
        });
    }
    if value.chars().any(char::is_control) {
        return Err(RunManifestError::ScenarioIdentityControlCharacter);
    }
    Ok(())
}

fn validate_build_provenance_claim(build: &BuildProvenanceV0) -> Result<bool, RunManifestError> {
    let derived = build.derived_provenance_complete();
    if build.provenance_complete != derived {
        return Err(RunManifestError::InconsistentBuildProvenance {
            recorded: build.provenance_complete,
            derived,
        });
    }
    Ok(derived)
}

fn normalized_config(config: &ScriptBotsConfig) -> Result<serde_json::Value, RunManifestError> {
    let mut value = serde_json::to_value(config).map_err(RunManifestError::ConfigSerialization)?;
    normalize_json_value(&mut value);
    Ok(value)
}

fn normalize_json_value(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                normalize_json_value(value);
            }
        }
        serde_json::Value::Object(map) => {
            let mut entries: Vec<_> = std::mem::take(map).into_iter().collect();
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            for (key, mut value) in entries {
                normalize_json_value(&mut value);
                map.insert(key, value);
            }
        }
        _ => {}
    }
}

fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, serde_json::Error> {
    let mut value = serde_json::to_value(value)?;
    normalize_json_value(&mut value);
    serde_json::to_vec(&value)
}

fn canonical_json_text<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    canonical_json_bytes(value).map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(test)]
fn canonical_json_value_bytes(value: &serde_json::Value) -> Result<Vec<u8>, serde_json::Error> {
    let mut value = value.clone();
    normalize_json_value(&mut value);
    serde_json::to_vec(&value)
}

fn parse_compile_time_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Some(true),
        "0" | "false" | "no" => Some(false),
        _ => None,
    }
}

fn compile_time_text(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn is_nonblank_text(value: &str) -> bool {
    !value.trim().is_empty()
}

fn tracked_toolchain_spec() -> String {
    RUST_TOOLCHAIN_TEXT
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("channel = \"")
                .and_then(|value| value.strip_suffix('"'))
        })
        .unwrap_or("unknown")
        .to_owned()
}

fn manifest_digest(domain: &str, bytes: &[u8]) -> String {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET_BASIS;
    for byte in domain
        .as_bytes()
        .iter()
        .copied()
        .chain(std::iter::once(0))
        .chain(bytes.iter().copied())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("fnv1a64:{hash:016x}")
}

#[cfg(test)]
mod characterization_tests {
    use super::*;
    use scriptbots_core::{NullPersistence, PersistenceSessionError, WorldStepError};

    fn test_world(seed: Option<u64>) -> WorldState {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 40,
            world_height: 40,
            food_cell_size: 10,
            initial_food: 0.25,
            food_respawn_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: seed,
            ..ScriptBotsConfig::default()
        })
        .expect("test world");
        for (x, y) in [(10.0, 10.0), (30.0, 30.0)] {
            let agent = scriptbots_core::AgentData {
                position: scriptbots_core::Position::new(x, y),
                ..scriptbots_core::AgentData::default()
            };
            world.try_spawn_agent(agent).expect("test agent is finite");
        }
        world
    }

    fn complete_test_build() -> BuildProvenanceV0 {
        BuildProvenanceV0 {
            package_name: "scriptbots-app".to_owned(),
            package_version: "0.1.0".to_owned(),
            source_revision: Some("0123456789abcdef".to_owned()),
            source_branch: Some("main".to_owned()),
            source_tree_clean: Some(true),
            source_status_digest: Some("fnv1a64:3333333333333333".to_owned()),
            source_diff_digest: Some("fnv1a64:4444444444444444".to_owned()),
            declared_toolchain: "nightly-2026-07-09".to_owned(),
            compiler_toolchain: Some("nightly-2026-07-09-test-target".to_owned()),
            rustc_vv: Some("rustc 1.99.0-nightly test".to_owned()),
            toolchain_file_digest: "fnv1a64:1111111111111111".to_owned(),
            lockfile_digest: "fnv1a64:2222222222222222".to_owned(),
            compiled_features: vec!["fast-alloc".to_owned(), "ml".to_owned()],
            core: CoreBuildIdentityV0 {
                parallel: true,
                simd_wide: true,
                rayon_threads: 8,
                target_arch: "x86_64".to_owned(),
                target_os: "linux".to_owned(),
                target_family: "unix".to_owned(),
                target_endian: "little".to_owned(),
                pointer_width: 64,
            },
            rustflags: None,
            rayon_num_threads: Some("1".to_owned()),
            scriptbots_max_threads: Some("1".to_owned()),
            provenance_complete: true,
            warnings: Vec::new(),
        }
    }

    fn test_run_identity(run_id: u128) -> RunIdentityV1 {
        RunIdentityV1::new(RunId::new(run_id), 1_752_515_200_000, Some(256), None)
    }

    #[test]
    fn current_build_provenance_uses_only_embedded_inputs_and_derives_its_claim() {
        let build = BuildProvenanceV0::current();

        assert_eq!(
            build.source_revision,
            compile_time_text(option_env!("SCRIPTBOTS_SOURCE_REVISION"))
        );
        assert_eq!(
            build.source_branch,
            compile_time_text(option_env!("SCRIPTBOTS_SOURCE_BRANCH"))
        );
        assert_eq!(
            build.source_status_digest,
            compile_time_text(option_env!("SCRIPTBOTS_SOURCE_STATUS_DIGEST"))
        );
        assert_eq!(
            build.source_diff_digest,
            compile_time_text(option_env!("SCRIPTBOTS_SOURCE_DIFF_DIGEST"))
        );
        assert_eq!(
            build.rustc_vv,
            compile_time_text(option_env!("SCRIPTBOTS_RUSTC_VV"))
        );
        assert_eq!(
            build.provenance_complete,
            build.derived_provenance_complete()
        );

        let mut expected_features = [
            (cfg!(feature = "bevy_render"), "bevy_render"),
            (cfg!(feature = "brain-ft"), "brain-ft"),
            (cfg!(feature = "fast-alloc"), "fast-alloc"),
            (cfg!(feature = "gui"), "gui"),
            (cfg!(feature = "llm-anthropic"), "llm-anthropic"),
            (cfg!(feature = "ml"), "ml"),
            (cfg!(feature = "neuro"), "neuro"),
        ]
        .into_iter()
        .filter_map(|(enabled, feature)| enabled.then_some(feature.to_owned()))
        .collect::<Vec<_>>();
        expected_features.sort_unstable();
        assert_eq!(build.compiled_features, expected_features);
    }

    #[test]
    fn brain_telemetry_round_trip_uses_stable_agent_uid() {
        let telemetry = scriptbots_brain::BrainTelemetry {
            agent: scriptbots_core::AgentUid(0xfeed_beef),
            tick: scriptbots_core::Tick(17),
            energy_spent: 0.25,
        };
        let encoded = serde_json::to_vec(&telemetry).expect("encode brain telemetry");
        let decoded: scriptbots_brain::BrainTelemetry =
            serde_json::from_slice(&encoded).expect("decode brain telemetry");
        assert_eq!(decoded, telemetry);
        let value: serde_json::Value =
            serde_json::from_slice(&encoded).expect("inspect brain telemetry");
        assert_eq!(value["agent"], 0xfeed_beef_u64);
    }

    #[test]
    fn canonical_json_sorts_nested_keys_and_round_trips_manifest() {
        let input = serde_json::json!({
            "z": {"beta": 2, "alpha": 1},
            "a": [{"right": true, "left": false}]
        });
        let bytes = canonical_json_value_bytes(&input).expect("canonical JSON");
        assert_eq!(
            String::from_utf8_lossy(&bytes),
            r#"{"a":[{"left":false,"right":true}],"z":{"alpha":1,"beta":2}}"#
        );

        let world = test_world(Some(17));
        let mut identity = test_run_identity(0x17);
        identity.experiment_id = Some("canonical-experiment".to_owned());
        identity.variant_id = Some("variant-a".to_owned());
        let manifest = RunManifestV3::from_world_with_provenance(
            identity,
            ScenarioIdentityV0::caller_seeded("canonical-test"),
            &world,
            complete_test_build(),
        )
        .expect("manifest");
        let normalized_config_bytes =
            canonical_json_bytes(&manifest.normalized_config).expect("canonical config bytes");
        assert_eq!(
            manifest.config_digest,
            format!("blake3:{}", blake3::hash(&normalized_config_bytes).to_hex())
        );
        assert_eq!(manifest.config_digest_encoding, "blake3-canonical-json-v1");
        assert_eq!(manifest.random_streams.root_seed, manifest.root_seed);
        let encoded = manifest.canonical_json_bytes().expect("manifest JSON");
        let storage_record = manifest.to_storage_record().expect("storage projection");
        assert_eq!(
            storage_record.rng_algorithm,
            manifest.random_streams.algorithm
        );
        assert_eq!(storage_record.rng_version, manifest.random_streams.version);
        assert_eq!(
            storage_record.manifest_json,
            manifest.canonical_json().expect("canonical manifest text")
        );
        assert_eq!(
            storage_record.normalized_config_json,
            canonical_json_text(&manifest.normalized_config).expect("canonical config text")
        );
        assert_eq!(
            storage_record.brain_roster_json,
            canonical_json_text(&manifest.brain_roster).expect("canonical roster text")
        );
        let decoded: RunManifestV3 = serde_json::from_slice(&encoded).expect("round trip");
        assert_eq!(manifest, decoded);
        assert_eq!(
            decoded.canonical_json_bytes().expect("re-encoded manifest"),
            encoded,
            "canonical bytes must survive the typed round trip"
        );
        let encoded_value: serde_json::Value =
            serde_json::from_slice(&encoded).expect("manifest schema");
        let encoded_object = encoded_value.as_object().expect("manifest object");
        assert_eq!(
            encoded_object
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            [
                "brain_roster",
                "build",
                "config_digest",
                "config_digest_encoding",
                "identity",
                "limitations",
                "next_agent_uid",
                "next_birth_ordinal",
                "next_spawn_ordinal",
                "normalized_config",
                "purpose",
                "random_streams",
                "reproducible",
                "root_seed",
                "scenario",
                "schema",
                "schema_version",
                "warnings",
            ]
        );
        assert_eq!(encoded_value["schema"], RUN_MANIFEST_V3_SCHEMA);
        assert_eq!(encoded_value["schema_version"], 3);
        assert_eq!(
            encoded_value["identity"]["experiment_id"],
            "canonical-experiment"
        );
        assert_eq!(encoded_value["identity"]["variant_id"], "variant-a");
        assert_eq!(
            encoded_value["identity"]["run_id"],
            "00000000000000000000000000000017"
        );
    }

    #[test]
    fn run_identity_preserves_max_budget_and_strict_run_id_wire_form() {
        let world = test_world(Some(18));
        let mut identity =
            RunIdentityV1::new(RunId::new(u128::MAX), u64::MAX, Some(u64::MAX), None);
        identity.experiment_id = Some("experiment-max".to_owned());
        identity.variant_id = Some("variant-max".to_owned());
        let manifest = RunManifestV3::from_world_with_provenance(
            identity,
            ScenarioIdentityV0::caller_seeded("identity-max"),
            &world,
            complete_test_build(),
        )
        .expect("maximum unsigned identity values are valid");
        let encoded = manifest.canonical_json_bytes().expect("identity JSON");
        let value: serde_json::Value = serde_json::from_slice(&encoded).expect("identity wire");

        assert_eq!(
            value["identity"]["run_id"],
            "ffffffffffffffffffffffffffffffff"
        );
        assert_eq!(value["identity"]["started_at_unix_ms"], u64::MAX);
        assert_eq!(value["identity"]["requested_tick_budget"], u64::MAX);
        assert_eq!(value["identity"]["experiment_id"], "experiment-max");
        assert_eq!(value["identity"]["variant_id"], "variant-max");
        assert!(value["identity"]["live_run_policy"].is_null());
    }

    #[test]
    fn run_identity_accepts_an_explicit_live_policy_without_a_finite_budget() {
        let world = test_world(Some(20));
        let identity = RunIdentityV1::new(
            RunId::new(0x20),
            1_752_515_200_020,
            None,
            Some("operator-controlled-until-stop-v1".to_owned()),
        );
        let manifest = RunManifestV3::from_world_with_provenance(
            identity,
            ScenarioIdentityV0::caller_seeded("live-identity"),
            &world,
            complete_test_build(),
        )
        .expect("an explicit live policy is a complete execution boundary");

        assert_eq!(manifest.identity.requested_tick_budget, None);
        assert_eq!(
            manifest.identity.live_run_policy.as_deref(),
            Some("operator-controlled-until-stop-v1")
        );
    }

    #[test]
    fn run_identity_rejects_blank_or_oversized_text_fields() {
        let world = test_world(Some(19));
        let manifest_error = |identity| {
            RunManifestV3::from_world_with_provenance(
                identity,
                ScenarioIdentityV0::caller_seeded("invalid-identity"),
                &world,
                complete_test_build(),
            )
            .expect_err("invalid identity text must be rejected")
        };

        let zero_run = RunIdentityV1::new(RunId::new(0), 0, Some(1), None);
        assert!(matches!(
            manifest_error(zero_run),
            RunManifestError::ZeroRunId
        ));

        let missing_boundary = RunIdentityV1::new(RunId::new(0x1900), 0, None, None);
        assert!(matches!(
            manifest_error(missing_boundary),
            RunManifestError::MissingRunExecutionBoundary
        ));

        let conflicting_boundaries = RunIdentityV1::new(
            RunId::new(0x1901),
            0,
            Some(10),
            Some("operator-controlled-v1".to_owned()),
        );
        assert!(matches!(
            manifest_error(conflicting_boundaries),
            RunManifestError::ConflictingRunExecutionBoundaries
        ));

        let mut blank_experiment = test_run_identity(0x1902);
        blank_experiment.experiment_id = Some(" \t ".to_owned());
        assert!(matches!(
            manifest_error(blank_experiment),
            RunManifestError::BlankRunIdentityField {
                field: "experiment_id"
            }
        ));

        let mut blank_variant = test_run_identity(0x1903);
        blank_variant.variant_id = Some("\n".to_owned());
        assert!(matches!(
            manifest_error(blank_variant),
            RunManifestError::BlankRunIdentityField {
                field: "variant_id"
            }
        ));

        let blank_policy = RunIdentityV1::new(RunId::new(0x1904), 0, None, Some("   ".to_owned()));
        assert!(matches!(
            manifest_error(blank_policy),
            RunManifestError::BlankRunIdentityField {
                field: "live_run_policy"
            }
        ));

        let mut oversized_variant = test_run_identity(0x1905);
        oversized_variant.variant_id = Some("v".repeat(MAX_RUN_IDENTITY_ID_BYTES + 1));
        assert!(matches!(
            manifest_error(oversized_variant),
            RunManifestError::RunIdentityFieldTooLong {
                field: "variant_id",
                actual,
                maximum: MAX_RUN_IDENTITY_ID_BYTES,
            } if actual == MAX_RUN_IDENTITY_ID_BYTES + 1
        ));

        let mut controlled_experiment = test_run_identity(0x1906);
        controlled_experiment.experiment_id = Some("experiment\u{0000}hidden".to_owned());
        assert!(matches!(
            manifest_error(controlled_experiment),
            RunManifestError::RunIdentityControlCharacter {
                field: "experiment_id"
            }
        ));

        let scenario_error = |id: String| {
            RunManifestV3::from_world_with_provenance(
                test_run_identity(0x1907),
                ScenarioIdentityV0::caller_seeded(id),
                &world,
                complete_test_build(),
            )
            .expect_err("invalid scenario identity must be rejected")
        };
        assert!(matches!(
            scenario_error("scenario\u{0000}hidden".to_owned()),
            RunManifestError::ScenarioIdentityControlCharacter
        ));
        assert!(matches!(
            scenario_error("s".repeat(MAX_SCENARIO_ID_BYTES + 1)),
            RunManifestError::ScenarioIdentityTooLong {
                actual,
                maximum: MAX_SCENARIO_ID_BYTES,
            } if actual == MAX_SCENARIO_ID_BYTES + 1
        ));
    }

    #[test]
    fn bootstrap_evidence_is_explicit_validated_and_schema_tagged() {
        let world = test_world(Some(0xB007_57A4));
        let start = world.world_digest_v1().expect("tick-zero start digest");
        let base = RunManifestV3::from_world_with_provenance(
            test_run_identity(0xB007_57A4),
            ScenarioIdentityV0::caller_seeded("zero-bootstrap"),
            &world,
            complete_test_build(),
        )
        .expect("base manifest");
        assert_eq!(base.schema, RUN_MANIFEST_V3_SCHEMA);
        assert!(base.bootstrap_evidence.is_none());

        let manifest = base
            .clone()
            .with_bootstrap_evidence(BootstrapEvidenceV0 {
                requested: 0,
                completed: 0,
                start: start.clone(),
                end: start.clone(),
            })
            .expect("zero bootstrap evidence");
        assert_eq!(manifest.schema, RUN_MANIFEST_V3_BOOTSTRAP_SCHEMA);
        let evidence = manifest
            .bootstrap_evidence
            .as_ref()
            .expect("attached bootstrap evidence");
        assert_eq!((evidence.requested, evidence.completed), (0, 0));
        assert_eq!((evidence.start.tick.0, evidence.end.tick.0), (0, 0));
        assert_eq!(evidence.start, evidence.end);

        let encoded = manifest.canonical_json_bytes().expect("evidence JSON");
        let decoded: RunManifestV3 = serde_json::from_slice(&encoded).expect("evidence round trip");
        assert_eq!(decoded, manifest);
        let value: serde_json::Value = serde_json::from_slice(&encoded).expect("evidence wire");
        assert_eq!(value["schema"], RUN_MANIFEST_V3_BOOTSTRAP_SCHEMA);
        assert_eq!(
            value["bootstrap_evidence"]
                .as_object()
                .expect("bootstrap evidence object")
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["completed", "end", "requested", "start"],
            "bootstrap evidence wire changes require another schema boundary"
        );

        let request_mismatch = base
            .clone()
            .with_bootstrap_evidence(BootstrapEvidenceV0 {
                requested: 1,
                completed: 1,
                start: start.clone(),
                end: start.clone(),
            })
            .expect_err("scenario request mismatch must fail");
        assert!(matches!(
            request_mismatch,
            RunManifestError::BootstrapRequestMismatch {
                scenario_requested: 0,
                evidence_requested: 1,
            }
        ));

        let completion_mismatch = base
            .clone()
            .with_bootstrap_evidence(BootstrapEvidenceV0 {
                requested: 0,
                completed: 1,
                start: start.clone(),
                end: start.clone(),
            })
            .expect_err("partial bootstrap evidence must fail");
        assert!(matches!(
            completion_mismatch,
            RunManifestError::BootstrapCompletionMismatch {
                requested: 0,
                completed: 1,
            }
        ));

        let different_zero = test_world(Some(0xB007_57A5))
            .world_digest_v1()
            .expect("different tick-zero digest");
        let zero_changed = base
            .with_bootstrap_evidence(BootstrapEvidenceV0 {
                requested: 0,
                completed: 0,
                start,
                end: different_zero,
            })
            .expect_err("zero bootstrap cannot change state");
        assert!(matches!(
            zero_changed,
            RunManifestError::BootstrapZeroChanged
        ));
    }

    #[test]
    fn bootstrap_evidence_counts_real_world_transitions() {
        let mut world = test_world(Some(0xB007_57A6));
        let start = world.world_digest_v1().expect("start digest");
        world.step().expect("one bootstrap transition");
        let end = world.world_digest_v1().expect("end digest");
        let mut scenario = ScenarioIdentityV0::caller_seeded("one-bootstrap");
        scenario.bootstrap_ticks = 1;
        let manifest = RunManifestV3::from_world_with_provenance(
            test_run_identity(0xB007_57A6),
            scenario,
            &test_world(Some(0xB007_57A6)),
            complete_test_build(),
        )
        .expect("launch-state manifest")
        .with_bootstrap_evidence(BootstrapEvidenceV0 {
            requested: 1,
            completed: 1,
            start,
            end,
        })
        .expect("one-tick evidence");
        let evidence = manifest
            .bootstrap_evidence
            .expect("attached one-tick evidence");
        assert_eq!((evidence.start.tick.0, evidence.end.tick.0), (0, 1));
        assert_ne!(evidence.start.overall, evidence.end.overall);
    }

    #[test]
    fn manifests_are_stable_and_lists_are_sorted() {
        let mut build = complete_test_build();
        build.compiled_features = vec!["neuro".to_owned(), "gui".to_owned()];
        build.compiled_features.sort_unstable();
        let world_a = test_world(Some(33));
        let world_b = test_world(Some(33));
        let scenario = ScenarioIdentityV0::caller_seeded("stable-test");
        let identity = test_run_identity(33);
        let manifest_a = RunManifestV3::from_world_with_provenance(
            identity.clone(),
            scenario.clone(),
            &world_a,
            build.clone(),
        )
        .expect("manifest A");
        let manifest_b =
            RunManifestV3::from_world_with_provenance(identity, scenario, &world_b, build)
                .expect("manifest B");
        assert_eq!(
            manifest_a.canonical_json_bytes().expect("manifest A JSON"),
            manifest_b.canonical_json_bytes().expect("manifest B JSON")
        );
        assert!(
            manifest_a
                .build
                .compiled_features
                .windows(2)
                .all(|pair| pair[0] <= pair[1])
        );
        assert!(
            manifest_a
                .brain_roster
                .windows(2)
                .all(|pair| pair[0].registry_key < pair[1].registry_key)
        );
    }

    #[test]
    fn manifest_and_audit_observe_closed_world_boundary_transitions() {
        let mut world = test_world(Some(34));
        world.step().expect("tick before policy transition");
        let build = complete_test_build();
        let scenario = ScenarioIdentityV0::caller_seeded("closed-policy-test");
        let identity = test_run_identity(34);
        let open_manifest = RunManifestV3::from_world_with_provenance(
            identity.clone(),
            scenario.clone(),
            &world,
            build.clone(),
        )
        .expect("open manifest");
        assert_eq!(open_manifest.normalized_config["closed"], false);

        world.set_closed(true).expect("close manifest world");
        let closed_manifest =
            RunManifestV3::from_world_with_provenance(identity, scenario, &world, build)
                .expect("closed manifest");

        assert!(world.is_closed());
        assert_eq!(world.config_revision(), 1);
        assert_eq!(closed_manifest.normalized_config["closed"], true);
        assert_ne!(open_manifest.config_digest, closed_manifest.config_digest);
        assert_eq!(
            world.config_audit(),
            [scriptbots_core::ConfigAuditEntry {
                tick: 1,
                patch: serde_json::json!({ "closed": true }),
            }]
        );
    }

    #[test]
    fn manifest_rejects_entropy_seed_and_marks_provenance_gaps() {
        let entropy_world = test_world(None);
        assert!(matches!(
            RunManifestV3::from_world_with_provenance(
                test_run_identity(1),
                ScenarioIdentityV0::caller_seeded("entropy"),
                &entropy_world,
                complete_test_build(),
            ),
            Err(RunManifestError::MissingExplicitSeed)
        ));

        let mut incomplete = complete_test_build();
        incomplete.source_revision = None;
        incomplete.source_tree_clean = None;
        incomplete.source_status_digest = None;
        incomplete.source_diff_digest = None;
        incomplete.rustc_vv = None;
        incomplete.provenance_complete = false;
        incomplete.warnings = vec!["provenance deliberately incomplete".to_owned()];
        let world = test_world(Some(5));
        let manifest = RunManifestV3::from_world_with_provenance(
            test_run_identity(5),
            ScenarioIdentityV0::caller_seeded("incomplete"),
            &world,
            incomplete,
        )
        .expect("incomplete manifest");
        assert!(!manifest.reproducible);
        assert_eq!(manifest.warnings, ["provenance deliberately incomplete"]);
    }

    #[test]
    fn inconsistent_provenance_and_reproducibility_claims_are_rejected() {
        let world = test_world(Some(0xC1A1_0001));
        let mut inconsistent_build = complete_test_build();
        inconsistent_build.rustc_vv = None;
        assert!(!inconsistent_build.derived_provenance_complete());
        assert!(inconsistent_build.provenance_complete);

        let error = RunManifestV3::from_world_with_provenance(
            test_run_identity(0xC1A1_0001),
            ScenarioIdentityV0::caller_seeded("inconsistent-build"),
            &world,
            inconsistent_build,
        )
        .expect_err("a stored completeness flag must not override missing evidence");
        assert!(matches!(
            error,
            RunManifestError::InconsistentBuildProvenance {
                recorded: true,
                derived: false,
            }
        ));

        let mut manifest = RunManifestV3::from_world_with_provenance(
            test_run_identity(0xC1A1_0002),
            ScenarioIdentityV0::caller_seeded("mutated-claim"),
            &world,
            complete_test_build(),
        )
        .expect("consistent manifest");
        let mut mismatched_streams = manifest.clone();
        mismatched_streams.random_streams.root_seed ^= 1;
        let error = mismatched_streams
            .to_storage_record()
            .expect_err("storage projection must reject a mismatched stream root seed");
        assert!(matches!(
            error,
            RunManifestError::RandomStreamRootSeedMismatch {
                manifest: 0xC1A1_0001,
                checkpoint,
            } if checkpoint == (0xC1A1_0001 ^ 1)
        ));

        manifest.reproducible = false;
        let error = manifest
            .to_storage_record()
            .expect_err("storage projection must revalidate a mutated public manifest");
        assert!(matches!(
            error,
            RunManifestError::InconsistentReproducibilityClaim {
                recorded: false,
                derived: true,
            }
        ));
    }

    #[test]
    fn traces_are_bounded_include_tick_zero_and_diverge_by_seed() {
        let trace_for = |seed: u64| {
            let mut world = test_world(Some(seed));
            CharacterizationTraceV2::capture(
                test_run_identity(u128::from(seed)),
                "trace-test",
                &mut world,
                4,
            )
            .expect("characterization trace")
        };
        let trace_a = trace_for(0xC0FFEE);
        let trace_b = trace_for(0xC0FFEE);
        let trace_c = trace_for(0xBAD5EED);
        assert_eq!(trace_a.schema, CHARACTERIZATION_TRACE_V2_SCHEMA);
        assert_eq!(trace_a.schema_version, 2);
        let sequence = |trace: &CharacterizationTraceV2| {
            trace
                .points
                .iter()
                .map(|point| point.digest.overall.clone())
                .collect::<Vec<_>>()
        };

        assert_eq!(trace_a.points.len(), 5);
        assert_eq!(trace_a.points[0].tick, 0);
        assert!(trace_a.points[0].tick_events.is_none());
        assert_eq!(trace_a.points[4].tick, 4);
        assert_eq!(sequence(&trace_a), sequence(&trace_b));
        assert_ne!(sequence(&trace_a), sequence(&trace_c));

        let mut trace_with_unknown_field =
            serde_json::to_value(&trace_a).expect("trace JSON value");
        trace_with_unknown_field["unknown_trace_field"] = serde_json::json!(true);
        assert!(
            serde_json::from_value::<CharacterizationTraceV2>(trace_with_unknown_field).is_err(),
            "the V2 trace contract must reject unknown top-level fields"
        );
        let mut manifest_with_unknown_field =
            serde_json::to_value(&trace_a.manifest).expect("manifest JSON value");
        manifest_with_unknown_field["unknown_manifest_field"] = serde_json::json!(true);
        assert!(
            serde_json::from_value::<RunManifestV3>(manifest_with_unknown_field).is_err(),
            "the V3 manifest contract must reject unknown top-level fields"
        );

        // These post-tick digests freeze the explicit energy-only ground-food
        // policy: eating changes energy and reproduction progress, not health.
        let observed_sequences: [[String; 5]; 2] = [
            sequence(&trace_a)
                .try_into()
                .expect("seed A trace has exactly five boundaries"),
            sequence(&trace_c)
                .try_into()
                .expect("seed C trace has exactly five boundaries"),
        ];
        assert_eq!(
            observed_sequences,
            [
                // Deliberately re-pinned in bd-2cd1 when the V0.1 probe expanded from one global
                // RNG stream to six named domain streams.
                // Re-pinned when the characterization digest was extended to cover active
                // interventions. The trajectory itself was unchanged: no intervention is active.
                [
                    "9170abeee25b6132",
                    "465d51c2d5baff86",
                    "35be0bef875fb7ce",
                    "e4ab9fb7f2c8e057",
                    "42f3e7cfd202e12b",
                ]
                .map(str::to_owned),
                // Re-pinned after sensory bearings moved from the host C math library to
                // deterministic pure-Rust `libm::atan2f`.
                [
                    "67db1d378a2a6d53",
                    "fa7277780a43affb",
                    "694b8a52675e93d2",
                    "403d0eb4df234203",
                    "be46adf860b9bfcd",
                ]
                .map(str::to_owned),
            ]
        );

        let mut world = test_world(Some(1));
        assert!(matches!(
            CharacterizationTraceV2::capture(
                test_run_identity(1),
                "too-long",
                &mut world,
                MAX_CHARACTERIZATION_TICKS_V2 + 1,
            ),
            Err(CharacterizationTraceErrorV2::ExcessiveTickCount { .. })
        ));
    }

    #[test]
    fn persistence_enabled_trace_requires_and_accepts_its_bound_session() {
        let (mut world, mut persistence) = WorldState::with_persistence(
            ScriptBotsConfig {
                world_width: 40,
                world_height: 40,
                food_cell_size: 10,
                initial_food: 0.25,
                food_respawn_interval: 0,
                population_minimum: 0,
                population_spawn_interval: 0,
                persistence_interval: 1,
                rng_seed: Some(0x05E5_510A),
                ..ScriptBotsConfig::default()
            },
            Box::new(NullPersistence),
        )
        .expect("persistence-enabled characterization world");
        world
            .try_spawn_agent(scriptbots_core::AgentData::default())
            .expect("characterization founder is finite");

        let identity = test_run_identity(0x05E5_510A);
        let direct_error = CharacterizationTraceV2::capture(
            identity.clone(),
            "direct-enabled-trace",
            &mut world,
            1,
        )
        .expect_err("direct capture must not bypass the bound persistence session");
        assert!(matches!(
            direct_error,
            CharacterizationTraceErrorV2::Step(WorldStepError::PersistenceSession(
                PersistenceSessionError::SessionRequired { tick: 1 }
            ))
        ));
        assert_eq!(
            world.tick().0,
            0,
            "rejected direct capture mutated the world"
        );
        assert_eq!(persistence.last_admitted_tick(), None);

        let trace = CharacterizationTraceV2::capture_with_scenario_and_session(
            identity,
            ScenarioIdentityV0::caller_seeded("session-enabled-trace"),
            &mut world,
            &mut persistence,
            2,
        )
        .expect("bound session capture");
        assert_eq!(
            trace
                .points
                .iter()
                .map(|point| point.tick)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(
            persistence.last_admitted_tick(),
            Some(scriptbots_core::Tick(2))
        );
        assert!(!persistence.has_pending_batch());
        assert!(persistence.fault().is_none());
    }
}

pub mod command;
pub mod control;
pub mod lab;
pub mod precedence;
pub mod servers;
pub mod terminal;

pub mod renderer {
    use anyhow::Result;

    use crate::{
        CommandDrain, CommandSubmit, ControlRuntime, SharedAnalytics, SharedWorld, WorldStepDriver,
    };

    /// Shared context passed to renderer implementations.
    pub struct RendererContext<'a> {
        pub world: SharedWorld,
        pub simulation_step: WorldStepDriver,
        pub analytics: SharedAnalytics,
        pub control_runtime: &'a ControlRuntime,
        pub command_drain: CommandDrain,
        pub command_submit: CommandSubmit,
    }

    pub trait Renderer {
        /// Stable identifier describing the renderer implementation (e.g., "gpui", "terminal").
        fn name(&self) -> &'static str;

        /// Launch the renderer; blocks until the rendering session completes.
        fn run(&self, ctx: RendererContext<'_>) -> Result<()>;
    }
}

pub use command::{
    CommandBusTelemetry, CommandDrain, CommandReceiver, CommandRecvError, CommandSendError,
    CommandSender, CommandSubmit, create_command_bus, drain_pending_commands, make_command_drain,
    make_command_submit,
};
pub use control::{
    ConfigSnapshot, ControlError, ControlHandle, HydrologySnapshot, KnobEntry, KnobKind, KnobUpdate,
};
pub use scriptbots_core::{ControlCommand, WorldStepDriver};
pub use servers::{
    ConfigPatchRequest, ControlRuntime, ControlRuntimeStatus, ControlServerConfig,
    ControlServerReservation, DEFAULT_CONTROL_MCP_HTTP_ADDRESS, DEFAULT_CONTROL_REST_ADDRESS,
    DEFAULT_CONTROL_SWAGGER_PATH, KnobApplyRequest, McpTransportConfig,
    default_control_rest_base_url,
};
