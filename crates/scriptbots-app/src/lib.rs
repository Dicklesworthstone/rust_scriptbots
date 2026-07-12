//! Shared application plumbing for ScriptBots control surfaces.

use std::process::Command;
use std::sync::{Arc, Mutex};

use scriptbots_core::{
    CharacterizationDigestV0, CharacterizationError, CoreBuildIdentityV0, ScriptBotsConfig,
    TickEvents, WorldState,
};
use scriptbots_storage::AnalyticsSnapshotProvider;
pub use scriptbots_storage::STORAGE_SIDECAR_SUFFIXES;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type SharedWorld = Arc<Mutex<WorldState>>;
pub type SharedAnalytics = AnalyticsSnapshotProvider;

/// Schema identifier for the temporary pre-redesign run manifest.
pub const RUN_MANIFEST_V0_SCHEMA: &str = "scriptbots.run-manifest";
/// Schema identifier for a sequence of V0 world characterization points.
pub const CHARACTERIZATION_TRACE_V0_SCHEMA: &str = "scriptbots.characterization-trace";
/// Safety bound for the temporary characterization runner.
pub const MAX_CHARACTERIZATION_TICKS_V0: u64 = 256;
const CARGO_LOCK_BYTES: &[u8] = include_bytes!("../../../Cargo.lock");
const RUST_TOOLCHAIN_BYTES: &[u8] = include_bytes!("../../../rust-toolchain.toml");
const RUST_TOOLCHAIN_TEXT: &str = include_str!("../../../rust-toolchain.toml");

/// Build identity captured alongside every version-zero run manifest.
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

impl BuildProvenanceV0 {
    /// Capture provenance embedded in the current `scriptbots-app` build.
    ///
    /// Reproducible builds should set `SCRIPTBOTS_SOURCE_REVISION` and
    /// `SCRIPTBOTS_SOURCE_TREE_CLEAN=true` before invoking Cargo. `GITHUB_SHA` is accepted as a
    /// revision fallback. Otherwise read-only Git and `rustc -Vv` probes are attempted. Missing
    /// values remain visibly unknown; provenance is never fabricated.
    #[must_use]
    pub fn current() -> Self {
        let runtime_git_status = command_output_bytes(
            "git",
            &["status", "--porcelain", "--untracked-files=normal"],
        );
        let runtime_git_diff =
            command_output_bytes("git", &["diff", "--binary", "--no-ext-diff", "HEAD"]);
        let source_revision = option_env!("SCRIPTBOTS_SOURCE_REVISION")
            .or(option_env!("GITHUB_SHA"))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .or_else(|| command_stdout("git", &["rev-parse", "HEAD"]));
        let source_branch = command_stdout("git", &["rev-parse", "--abbrev-ref", "HEAD"]);
        let source_tree_clean = option_env!("SCRIPTBOTS_SOURCE_TREE_CLEAN")
            .and_then(parse_compile_time_bool)
            .or_else(|| runtime_git_status.as_ref().map(Vec::is_empty));
        let source_status_digest = runtime_git_status
            .as_deref()
            .map(|bytes| manifest_digest("git-status-porcelain-v0", bytes));
        let source_diff_digest = runtime_git_diff
            .as_deref()
            .map(|bytes| manifest_digest("git-diff-binary-v0", bytes));
        let compiler_toolchain = option_env!("RUSTUP_TOOLCHAIN")
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let rustc_vv = command_stdout("rustc", &["-Vv"]);

        let mut compiled_features = Vec::new();
        for (enabled, name) in [
            (cfg!(feature = "bevy_render"), "bevy_render"),
            (cfg!(feature = "fast-alloc"), "fast-alloc"),
            (cfg!(feature = "gui"), "gui"),
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
        if rustc_vv.is_none() {
            warnings.push("rustc -Vv output is unavailable".to_owned());
        }
        let provenance_complete =
            source_revision.is_some() && source_tree_clean == Some(true) && rustc_vv.is_some();

        Self {
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
            rayon_num_threads: std::env::var("RAYON_NUM_THREADS").ok(),
            scriptbots_max_threads: std::env::var("SCRIPTBOTS_MAX_THREADS").ok(),
            provenance_complete,
            warnings,
        }
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
    pub fn record_config_layer(&mut self, bytes: &[u8]) {
        self.ordered_config_layer_digests
            .push(manifest_digest("config-layer-v0", bytes));
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
            agent_identity: "AgentId::data().as_ffi(), not stable AgentUid".to_owned(),
            source_identity:
                "commit plus status/tracked-diff digests; untracked file contents are not hashed"
                    .to_owned(),
            evaluator_state_covered: false,
            rng_state_restorable: false,
            checkpoint_replay_guarantee: false,
            comparison_lane: "same pinned toolchain, target, features, and thread lane".to_owned(),
            superseded_by: "WorldDigestV1".to_owned(),
        }
    }
}

/// Version-zero record tying scenario construction and normalized configuration to a build.
///
/// `reproducible` means the manifest has an explicit seed and complete clean-source provenance. It
/// does not override the characterization digest's exclusions or claim that replay can reconstruct
/// the world.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunManifestV0 {
    pub schema: String,
    pub schema_version: u16,
    pub purpose: String,
    pub root_seed: u64,
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

/// Errors returned while constructing a version-zero run manifest.
#[derive(Debug, Error)]
pub enum RunManifestError {
    #[error("scenario identity must not be empty")]
    EmptyScenarioIdentity,
    #[error("characterization V0 requires an explicit rng_seed")]
    MissingExplicitSeed,
    #[error("failed to normalize ScriptBots configuration: {0}")]
    ConfigSerialization(#[source] serde_json::Error),
    #[error("failed to encode ScriptBots configuration for its V0 digest: {0}")]
    ConfigDigestSerialization(String),
}

impl RunManifestV0 {
    /// Capture a run manifest using provenance embedded in the current build.
    pub fn from_world(
        scenario_id: impl Into<String>,
        world: &WorldState,
    ) -> Result<Self, RunManifestError> {
        Self::from_world_with_provenance(
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
        mut scenario: ScenarioIdentityV0,
        world: &WorldState,
        build: BuildProvenanceV0,
    ) -> Result<Self, RunManifestError> {
        scenario.id = scenario.id.trim().to_owned();
        if scenario.id.is_empty() {
            return Err(RunManifestError::EmptyScenarioIdentity);
        }

        let normalized_config = normalized_config(world.config())?;
        let config_digest_bytes = ron::ser::to_string(world.config())
            .map_err(|error| RunManifestError::ConfigDigestSerialization(error.to_string()))?;
        let config_digest =
            manifest_digest("scriptbots-config-ron-v0", config_digest_bytes.as_bytes());
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
        let reproducible = build.provenance_complete;
        let warnings = build.warnings.clone();

        Ok(Self {
            schema: RUN_MANIFEST_V0_SCHEMA.to_owned(),
            schema_version: 0,
            purpose: "characterization_only".to_owned(),
            root_seed,
            scenario,
            normalized_config,
            config_digest,
            config_digest_encoding: "ron-0.8-struct-order-v0".to_owned(),
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

/// One boundary captured in a V0 trace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TracePointV0 {
    pub tick: u64,
    pub digest: CharacterizationDigestV0,
    pub tick_events: Option<TickEvents>,
}

/// Bounded fixed-seed characterization trace for the current implementation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CharacterizationTraceV0 {
    pub schema: String,
    pub schema_version: u16,
    pub digest_algorithm: String,
    pub manifest: RunManifestV0,
    pub manifest_digest: String,
    pub points: Vec<TracePointV0>,
}

/// Failures produced while capturing a V0 characterization trace.
#[derive(Debug, Error)]
pub enum CharacterizationTraceErrorV0 {
    #[error("requested {requested} ticks, but characterization V0 is capped at {maximum}")]
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

impl CharacterizationTraceV0 {
    /// Capture tick zero and every boundary through `ticks` from an already constructed world.
    pub fn capture(
        scenario_id: impl Into<String>,
        world: &mut WorldState,
        ticks: u64,
    ) -> Result<Self, CharacterizationTraceErrorV0> {
        Self::capture_with_scenario(ScenarioIdentityV0::caller_seeded(scenario_id), world, ticks)
    }

    /// Capture a trace with explicit temporary scenario construction metadata.
    pub fn capture_with_scenario(
        scenario: ScenarioIdentityV0,
        world: &mut WorldState,
        ticks: u64,
    ) -> Result<Self, CharacterizationTraceErrorV0> {
        if ticks > MAX_CHARACTERIZATION_TICKS_V0 {
            return Err(CharacterizationTraceErrorV0::ExcessiveTickCount {
                requested: ticks,
                maximum: MAX_CHARACTERIZATION_TICKS_V0,
            });
        }

        let manifest = RunManifestV0::from_world_with_provenance(
            scenario,
            world,
            BuildProvenanceV0::current(),
        )?;
        let manifest_bytes = manifest.canonical_json_bytes()?;
        let manifest_digest = manifest_digest("run-manifest-v0", &manifest_bytes);
        let mut points = Vec::with_capacity(usize::try_from(ticks).unwrap_or(0) + 1);
        let initial = world.characterization_digest_v0()?;
        points.push(TracePointV0 {
            tick: initial.tick.0,
            digest: initial,
            tick_events: None,
        });
        for _ in 0..ticks {
            let events = world.step()?;
            let digest = world.characterization_digest_v0()?;
            points.push(TracePointV0 {
                tick: digest.tick.0,
                digest,
                tick_events: Some(events),
            });
        }

        Ok(Self {
            schema: CHARACTERIZATION_TRACE_V0_SCHEMA.to_owned(),
            schema_version: 0,
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

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    String::from_utf8(command_output_bytes(program, args)?)
        .ok()
        .map(|value| value.trim().to_owned())
}

fn command_output_bytes(program: &str, args: &[&str]) -> Option<Vec<u8>> {
    let output = Command::new(program).args(args).output().ok()?;
    output.status.success().then_some(output.stdout)
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
            world.spawn_agent(agent);
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
            core: CoreBuildIdentityV0::current(),
            rustflags: None,
            rayon_num_threads: Some("1".to_owned()),
            scriptbots_max_threads: Some("1".to_owned()),
            provenance_complete: true,
            warnings: Vec::new(),
        }
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
        let manifest = RunManifestV0::from_world_with_provenance(
            ScenarioIdentityV0::caller_seeded("canonical-test"),
            &world,
            complete_test_build(),
        )
        .expect("manifest");
        let encoded = manifest.canonical_json_bytes().expect("manifest JSON");
        let decoded: RunManifestV0 = serde_json::from_slice(&encoded).expect("round trip");
        assert_eq!(manifest, decoded);
    }

    #[test]
    fn manifests_are_stable_and_lists_are_sorted() {
        let mut build = complete_test_build();
        build.compiled_features = vec!["neuro".to_owned(), "gui".to_owned()];
        build.compiled_features.sort_unstable();
        let world_a = test_world(Some(33));
        let world_b = test_world(Some(33));
        let scenario = ScenarioIdentityV0::caller_seeded("stable-test");
        let manifest_a =
            RunManifestV0::from_world_with_provenance(scenario.clone(), &world_a, build.clone())
                .expect("manifest A");
        let manifest_b = RunManifestV0::from_world_with_provenance(scenario, &world_b, build)
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
    fn manifest_rejects_entropy_seed_and_marks_provenance_gaps() {
        let entropy_world = test_world(None);
        assert!(matches!(
            RunManifestV0::from_world_with_provenance(
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
        let manifest = RunManifestV0::from_world_with_provenance(
            ScenarioIdentityV0::caller_seeded("incomplete"),
            &world,
            incomplete,
        )
        .expect("incomplete manifest");
        assert!(!manifest.reproducible);
        assert_eq!(manifest.warnings, ["provenance deliberately incomplete"]);
    }

    #[test]
    fn traces_are_bounded_include_tick_zero_and_diverge_by_seed() {
        let trace_for = |seed| {
            let mut world = test_world(Some(seed));
            CharacterizationTraceV0::capture("trace-test", &mut world, 4)
                .expect("characterization trace")
        };
        let trace_a = trace_for(0xC0FFEE);
        let trace_b = trace_for(0xC0FFEE);
        let trace_c = trace_for(0xBAD5EED);
        let sequence = |trace: &CharacterizationTraceV0| {
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
        // These post-tick digests freeze the explicit energy-only ground-food
        // policy: eating changes energy and reproduction progress, not health.
        assert_eq!(
            sequence(&trace_a),
            [
                // Re-pinned when the characterization digest was extended to
                // cover ACTIVE INTERVENTIONS (bd-16g.10.1). A digest that ignored
                // a drought in force would certify two runs as identical while
                // one of them was in the middle of a famine, and every replay
                // proof built on it would be hollow. The trajectory itself is
                // unchanged: no intervention is active in this trace.
                "12ffd8903f2588f4",
                "57dadbd0dc0ef3bb",
                "093aada106046e77",
                "ccc3dc999e2dbd45",
                "77bf36ffad164f62",
            ]
        );
        assert_eq!(
            sequence(&trace_c),
            [
                "1daf2c9601805056",
                "34862afcc3524912",
                "efbb642bc81acede",
                "43e8f2f64a481ff5",
                "360f4818949324a7",
            ]
        );

        let mut world = test_world(Some(1));
        assert!(matches!(
            CharacterizationTraceV0::capture(
                "too-long",
                &mut world,
                MAX_CHARACTERIZATION_TICKS_V0 + 1,
            ),
            Err(CharacterizationTraceErrorV0::ExcessiveTickCount { .. })
        ));
    }
}

pub mod command;
pub mod control;
pub mod servers;
pub mod terminal;

pub mod renderer {
    use anyhow::Result;

    use crate::{CommandDrain, CommandSubmit, ControlRuntime, SharedAnalytics, SharedWorld};

    /// Shared context passed to renderer implementations.
    pub struct RendererContext<'a> {
        pub world: SharedWorld,
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
    CommandDrain, CommandReceiver, CommandSender, CommandSubmit, create_command_bus,
    drain_pending_commands, make_command_drain, make_command_submit,
};
pub use control::{
    ConfigSnapshot, ControlError, ControlHandle, HydrologySnapshot, KnobEntry, KnobKind, KnobUpdate,
};
pub use scriptbots_core::ControlCommand;
pub use servers::{
    ConfigPatchRequest, ControlRuntime, ControlRuntimeStatus, ControlServerConfig,
    ControlServerReservation, DEFAULT_CONTROL_MCP_HTTP_ADDRESS, DEFAULT_CONTROL_REST_ADDRESS,
    DEFAULT_CONTROL_SWAGGER_PATH, KnobApplyRequest, McpTransportConfig,
    default_control_rest_base_url,
};
