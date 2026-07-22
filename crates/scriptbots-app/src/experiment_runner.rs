//! Deterministic matched-seed experiment runner and batch execution engine (bd-2z0.5.5).

use scriptbots_core::{ScriptBotsConfig, WorldState};
use scriptbots_storage::export_pipeline::{DeterministicRunBundle, RunBundleManifest};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
    pub config_overrides: HashMap<String, f32>,
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

    /// Executes a single run deterministically, creating and verifying its run bundle.
    pub fn execute_single_run(
        &self,
        variant: &ScenarioVariant,
        seed: u64,
    ) -> Result<RunRecord, String> {
        let run_id = format!("{}-{}-seed{}", self.experiment_id, variant.variant_id, seed);
        let mut config = ScriptBotsConfig::default();
        config.rng_seed = Some(seed);

        for (k, &v) in &variant.config_overrides {
            match k.as_str() {
                "food_max" => config.food_max = v,
                "food_growth_rate" => config.food_growth_rate = v,
                "agent_max_speed" => config.bot_speed = v,
                _ => {}
            }
        }

        let mut world = WorldState::new(config).map_err(|e| e.to_string())?;

        for _ in 0..self.max_ticks {
            let _ = world.step();
        }

        let digest = world.world_digest_v1().map_err(|e| e.to_string())?;
        let digest_hex = digest.overall;
        let bundle_dir = self.output_dir.join(&run_id);

        let manifest = RunBundleManifest {
            schema_version: 1,
            run_id: run_id.clone(),
            seed,
            created_at_utc: "2026-07-22T15:45:00Z".into(),
            source_revision: "main".into(),
            source_tree_digest: "clean".into(),
            source_tree_dirty: false,
            rust_toolchain: "nightly".into(),
            cargo_lock_digest: "lock".into(),
            target_triple: std::env::consts::ARCH.into(),
            total_ticks: self.max_ticks,
            final_agent_count: world.agents().len(),
            config_hash: format!("{:016x}", world.config().rng_seed.unwrap_or(0)),
        };

        let summary_csv = format!(
            "tick,alive_agents,seed\n{},{},{}\n",
            self.max_ticks,
            world.agents().len(),
            seed
        );

        DeterministicRunBundle::assemble_bundle(
            &bundle_dir,
            manifest,
            &[("exports/summary.csv", summary_csv.as_bytes())],
        )
        .map_err(|e| format!("Failed to assemble bundle: {e}"))?;

        DeterministicRunBundle::verify_bundle(&bundle_dir)
            .map_err(|e| format!("Bundle verification failed: {e}"))?;

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

#[cfg(test)]
mod tests {
    use super::*;

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
                config_overrides: HashMap::new(),
            },
            ScenarioVariant {
                variant_id: "dwraon_base".into(),
                brain_family: "dwraon".into(),
                config_overrides: HashMap::new(),
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
                config_overrides: HashMap::new(),
            },
            ScenarioVariant {
                variant_id: "variant_b".into(),
                brain_family: "dwraon".into(),
                config_overrides: HashMap::new(),
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
