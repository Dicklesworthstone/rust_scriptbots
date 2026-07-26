//! Deterministic matched-seed experiment runner and batch execution engine (bd-2z0.5.5).

use scriptbots_core::{ScriptBotsConfig, WorldState, knob_range};
use scriptbots_runtime::RunId;
use scriptbots_storage::{RunManifestRecord, create_run_bundle_from_artifacts, verify_run_bundle};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Matched-seed cohort specifying the identical seed schedule shared across scenario variants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchedSeedCohort {
    pub cohort_id: String,
    pub seeds: Vec<u64>,
}

/// Scenario variant specification defining a brain family and configuration tuning overrides.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScenarioVariant {
    pub variant_id: String,
    pub brain_family: String,
    /// Canonical validated arm. JSON values preserve integer-vs-float types
    /// until strict `ScriptBotsConfig` deserialization.
    pub config_overrides: BTreeMap<String, serde_json::Value>,
}

/// Execution status of an individual experiment run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunState {
    Pending,
    Running,
    Completed,
    Failed,
}

/// Detailed run record tracking progress, final digest, and bundle location.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunRecord {
    pub run_id: String,
    pub variant_id: String,
    pub seed: u64,
    pub state: RunState,
    pub total_ticks: u64,
    pub final_digest: Option<String>,
    pub bundle_path: Option<String>,
    pub error_reason: Option<String>,
}

/// Complete experiment batch status report supporting atomic checkpointing and resume.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentBatchStatus {
    pub experiment_id: String,
    pub total_runs: usize,
    pub completed_runs: usize,
    pub failed_runs: usize,
    pub runs: Vec<RunRecord>,
}

impl ExperimentBatchStatus {
    pub fn is_finished(&self) -> bool {
        self.completed_runs + self.failed_runs == self.total_runs
    }
}

/// Deterministic matched-seed experiment runner.
pub struct MatchedSeedExperimentRunner {
    pub experiment_id: String,
    pub cohort: MatchedSeedCohort,
    pub variants: Vec<ScenarioVariant>,
    pub max_ticks: u64,
    pub max_concurrency: usize,
    pub output_dir: PathBuf,
}

impl MatchedSeedExperimentRunner {
    pub fn new(
        experiment_id: impl Into<String>,
        cohort: MatchedSeedCohort,
        variants: Vec<ScenarioVariant>,
        max_ticks: u64,
        max_concurrency: usize,
        output_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            experiment_id: experiment_id.into(),
            cohort,
            variants,
            max_ticks,
            max_concurrency: max_concurrency.max(1),
            output_dir: output_dir.into(),
        }
    }

    /// Generates initial pending run records for every variant and seed combination.
    pub fn plan_batch(&self) -> Vec<RunRecord> {
        let mut runs = Vec::new();
        for variant in &self.variants {
            for &seed in &self.cohort.seeds {
                let run_id = format!("{}-{}-seed{}", self.experiment_id, variant.variant_id, seed);
                runs.push(RunRecord {
                    run_id,
                    variant_id: variant.variant_id.clone(),
                    seed,
                    state: RunState::Pending,
                    total_ticks: 0,
                    final_digest: None,
                    bundle_path: None,
                    error_reason: None,
                });
            }
        }
        runs
    }

    /// Loads existing status file if present, or initializes a new batch plan.
    pub fn load_or_create_status(
        &self,
        state_file: &Path,
    ) -> std::io::Result<ExperimentBatchStatus> {
        if state_file.exists() {
            let bytes = fs::read(state_file)?;
            if let Ok(status) = serde_json::from_slice::<ExperimentBatchStatus>(&bytes) {
                return Ok(status);
            }
        }

        let planned_runs = self.plan_batch();
        let status = ExperimentBatchStatus {
            experiment_id: self.experiment_id.clone(),
            total_runs: planned_runs.len(),
            completed_runs: 0,
            failed_runs: 0,
            runs: planned_runs,
        };
        self.save_status(state_file, &status)?;
        Ok(status)
    }

    /// Atomically flushes status updates to disk.
    pub fn save_status(
        &self,
        state_file: &Path,
        status: &ExperimentBatchStatus,
    ) -> std::io::Result<()> {
        if let Some(parent) = state_file.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(status)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let tmp_path = state_file.with_extension("tmp");
        fs::write(&tmp_path, &json)?;
        fs::rename(tmp_path, state_file)?;
        Ok(())
    }

    /// Derive the bundle's stable `RunId` from the runner's human-readable run identity.
    ///
    /// The experiment runner keys runs by `<experiment>-<variant>-seed<N>`, while the run
    /// bundle manifest is keyed by `RunId`. BLAKE3 over that exact string keeps the
    /// mapping deterministic and collision-resistant, and the readable identity is still
    /// carried in the manifest's `experiment_id`, `variant_id`, and `scenario_id`.
    fn bundle_run_id(run_id: &str) -> RunId {
        let digest = blake3::hash(run_id.as_bytes());
        let mut leading = [0_u8; 16];
        leading.copy_from_slice(&digest.as_bytes()[..16]);
        RunId::new(u128::from_be_bytes(leading))
    }

    /// Executes a single run deterministically, creating and verifying its run bundle.
    pub fn execute_single_run(
        &self,
        variant: &ScenarioVariant,
        seed: u64,
    ) -> Result<RunRecord, String> {
        let run_id = format!("{}-{}-seed{}", self.experiment_id, variant.variant_id, seed);
        let config = config_for_run(&variant.config_overrides, seed)?;

        let mut world = WorldState::new(config).map_err(|e| e.to_string())?;

        for _ in 0..self.max_ticks {
            let _ = world.step();
        }

        let digest = world.world_digest_v1().map_err(|e| e.to_string())?;
        let digest_hex = digest.overall;
        let bundle_dir = self.output_dir.join(&run_id);

        // bd-4d9j: one run-bundle schema for the whole product. This runner steps a
        // persistence-disabled world and never opens a run database, so it uses the
        // artifact assembler — but it writes the same `scriptbots.run-bundle.v1`
        // manifest that `--create-bundle` writes and is read back by the same verifier.
        //
        // The provenance below records only what this runner actually knows. The fields
        // it cannot know keep `unattributed`'s explicit markers and `reproducible` stays
        // false, rather than the previous hardcoded "main"/"clean"/"nightly"/"lock"
        // placeholders that described a source tree nobody had inspected.
        let mut manifest = RunManifestRecord::unattributed(Self::bundle_run_id(&run_id));
        manifest.experiment_id = Some(self.experiment_id.clone());
        manifest.variant_id = Some(variant.variant_id.clone());
        manifest.scenario_id = run_id.clone();
        manifest.root_seed = seed;
        manifest.target_triple = std::env::consts::ARCH.to_owned();
        manifest.requested_tick_budget = Some(self.max_ticks);

        let summary_csv = format!(
            "tick,alive_agents,seed\n{},{},{}\n",
            self.max_ticks,
            world.agents().len(),
            seed
        );

        create_run_bundle_from_artifacts(
            &bundle_dir,
            manifest,
            self.max_ticks,
            &[("exports/summary.csv", "export", summary_csv.as_bytes())],
        )
        .map_err(|e| format!("Failed to assemble bundle: {e}"))?;

        verify_run_bundle(&bundle_dir).map_err(|e| format!("Bundle verification failed: {e}"))?;

        Ok(RunRecord {
            run_id,
            variant_id: variant.variant_id.clone(),
            seed,
            state: RunState::Completed,
            total_ticks: self.max_ticks,
            final_digest: Some(digest_hex),
            bundle_path: Some(bundle_dir.to_string_lossy().to_string()),
            error_reason: None,
        })
    }

    /// Executes the full batch with resume capabilities.
    pub fn execute_batch(&self, state_file: &Path) -> Result<ExperimentBatchStatus, String> {
        let mut status = self
            .load_or_create_status(state_file)
            .map_err(|e| format!("Failed loading status: {e}"))?;

        for i in 0..status.runs.len() {
            if status.runs[i].state == RunState::Completed {
                continue;
            }

            status.runs[i].state = RunState::Running;
            let _ = self.save_status(state_file, &status);

            let variant_id = &status.runs[i].variant_id;
            let seed = status.runs[i].seed;

            let variant = self
                .variants
                .iter()
                .find(|v| &v.variant_id == variant_id)
                .cloned()
                .ok_or_else(|| format!("Variant {variant_id} not found"))?;

            match self.execute_single_run(&variant, seed) {
                Ok(completed_record) => {
                    status.runs[i] = completed_record;
                    status.completed_runs += 1;
                }
                Err(err) => {
                    status.runs[i].state = RunState::Failed;
                    status.runs[i].error_reason = Some(err);
                    status.failed_runs += 1;
                }
            }

            let _ = self.save_status(state_file, &status);
        }

        Ok(status)
    }
}

/// Prove that one canonical arm becomes a valid fresh-world configuration
/// without creating a world or writing an artifact.
///
/// # Errors
///
/// Returns the exact unknown-path, type, decode, or config-invariant failure.
pub fn validate_scenario_arm(
    overrides: &BTreeMap<String, serde_json::Value>,
) -> Result<(), String> {
    config_for_run(overrides, 0).map(|_| ())
}

fn config_for_run(
    overrides: &BTreeMap<String, serde_json::Value>,
    seed: u64,
) -> Result<ScriptBotsConfig, String> {
    let mut encoded = serde_json::to_value(ScriptBotsConfig::default())
        .map_err(|error| format!("default config serialization failed: {error}"))?;
    let root = encoded
        .as_object_mut()
        .ok_or_else(|| "default config did not serialize as an object".to_owned())?;

    for (path, value) in overrides {
        if knob_range(path).is_none() {
            return Err(format!(
                "validated arm contains unknown or unsupported knob `{path}`"
            ));
        }
        insert_dotted_value(root, path, value.clone())?;
    }

    let mut config: ScriptBotsConfig = serde_json::from_value(encoded)
        .map_err(|error| format!("validated arm does not decode as ScriptBotsConfig: {error}"))?;
    // The matched-seed axis is owned by the cohort and can never be displaced by
    // a treatment arm.
    config.rng_seed = Some(seed);
    config
        .validate()
        .map_err(|error| format!("validated arm produces an invalid config: {error}"))?;
    Ok(config)
}

fn insert_dotted_value(
    root: &mut serde_json::Map<String, serde_json::Value>,
    path: &str,
    value: serde_json::Value,
) -> Result<(), String> {
    let segments = path.split('.').collect::<Vec<_>>();
    if segments.is_empty() || segments.iter().any(|segment| segment.is_empty()) {
        return Err(format!("invalid dotted knob path `{path}`"));
    }

    let mut object = root;
    for segment in &segments[..segments.len() - 1] {
        let entry = object
            .get_mut(*segment)
            .ok_or_else(|| format!("knob path `{path}` has unknown segment `{segment}`"))?;
        if entry.is_null() {
            *entry = serde_json::Value::Object(serde_json::Map::new());
        }
        object = entry
            .as_object_mut()
            .ok_or_else(|| format!("knob path `{path}` crosses non-object segment `{segment}`"))?;
    }
    let leaf = segments[segments.len() - 1];
    if !object.contains_key(leaf) {
        return Err(format!("knob path `{path}` has unknown leaf `{leaf}`"));
    }
    object.insert(leaf.to_owned(), value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_arm_application_is_strict_and_type_preserving() {
        let overrides = BTreeMap::from([
            ("world_width".to_owned(), serde_json::json!(5_000)),
            ("food_growth_rate".to_owned(), serde_json::json!(0.2)),
            ("food_transfer_rate".to_owned(), serde_json::json!(0.25)),
            ("population_minimum".to_owned(), serde_json::json!(2)),
        ]);
        let config = config_for_run(&overrides, 99).expect("canonical arm applies");
        assert_eq!(config.world_width, 5_000);
        assert!((config.food_growth_rate - 0.2).abs() < f32::EPSILON);
        assert!((config.food_transfer_rate - 0.25).abs() < f32::EPSILON);
        assert_eq!(config.population_minimum, 2);
        assert_eq!(config.rng_seed, Some(99));

        let unknown = BTreeMap::from([("food.regrowth".to_owned(), serde_json::json!(0.1))]);
        assert!(
            config_for_run(&unknown, 1)
                .expect_err("unknown overrides must never disappear")
                .contains("unknown or unsupported knob")
        );

        let default_json =
            serde_json::to_value(ScriptBotsConfig::default()).expect("default config schema");
        for range in scriptbots_core::KNOB_RANGES {
            if range.path.starts_with("mutation.") || range.path.starts_with("render.") {
                continue;
            }
            let pointer = format!("/{}", range.path.replace('.', "/"));
            assert!(
                default_json.pointer(&pointer).is_some(),
                "canonical validator accepts `{}`, but the strict runner config has no such field",
                range.path
            );
        }
    }

    #[test]
    fn test_experiment_runner_planning() {
        let cohort = MatchedSeedCohort {
            cohort_id: "cohort-1".into(),
            seeds: vec![100, 200, 300],
        };
        let variants = vec![
            ScenarioVariant {
                variant_id: "mlp_base".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
            },
            ScenarioVariant {
                variant_id: "dwraon_base".into(),
                brain_family: "dwraon".into(),
                config_overrides: BTreeMap::new(),
            },
        ];

        let runner =
            MatchedSeedExperimentRunner::new("exp-001", cohort, variants, 50, 2, "/tmp/exp_test");

        let plan = runner.plan_batch();
        assert_eq!(plan.len(), 6);
        assert_eq!(plan[0].run_id, "exp-001-mlp_base-seed100");
        assert_eq!(plan[3].run_id, "exp-001-dwraon_base-seed100");
    }

    #[test]
    fn test_execute_matched_seed_experiment_and_resume() {
        let temp_dir = tempfile::tempdir().unwrap();
        let state_file = temp_dir.path().join("experiment_state.json");
        let output_dir = temp_dir.path().join("bundles");

        let cohort = MatchedSeedCohort {
            cohort_id: "cohort-test".into(),
            seeds: vec![1001, 1002],
        };
        let variants = vec![
            ScenarioVariant {
                variant_id: "variant_a".into(),
                brain_family: "mlp".into(),
                config_overrides: BTreeMap::new(),
            },
            ScenarioVariant {
                variant_id: "variant_b".into(),
                brain_family: "dwraon".into(),
                config_overrides: BTreeMap::new(),
            },
        ];

        let runner =
            MatchedSeedExperimentRunner::new("exp-matched", cohort, variants, 20, 2, output_dir);

        let status = runner.execute_batch(&state_file).unwrap();
        assert_eq!(status.completed_runs, 4);
        assert_eq!(status.failed_runs, 0);
        assert!(status.is_finished());

        // Test resume behavior: running again on finished state returns 4 completed runs without duplicate execution
        let resumed_status = runner.execute_batch(&state_file).unwrap();
        assert_eq!(resumed_status.completed_runs, 4);
        assert_eq!(resumed_status.failed_runs, 0);
    }
}
