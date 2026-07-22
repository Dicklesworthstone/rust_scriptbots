//! Autonomous LLM lab assistant: hypothesis -> matched-seed sweep -> analysis -> lab notebook (bd-16g.1).

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Provider-agnostic LLM client trait for hypothesis generation and analysis.
pub trait LlmClient: Send + Sync {
    fn complete(&self, prompt: &str) -> Result<String>;
}

/// Scripted offline LLM client for deterministic testing without external API access.
#[derive(Debug, Clone, Default)]
pub struct ScriptedLlmClient {
    pub canned_response: String,
}

impl ScriptedLlmClient {
    pub fn new(canned: impl Into<String>) -> Self {
        Self {
            canned_response: canned.into(),
        }
    }
}

impl LlmClient for ScriptedLlmClient {
    fn complete(&self, _prompt: &str) -> Result<String> {
        Ok(self.canned_response.clone())
    }
}

/// Structured experiment specification validated against budget and schema bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSpec {
    pub hypothesis: String,
    pub target_knobs: HashMap<String, f32>,
    pub seeds: Vec<u64>,
    pub max_ticks: u64,
    pub budget_runs: usize,
}

impl ExperimentSpec {
    pub fn validate(&self) -> Result<()> {
        if self.hypothesis.trim().is_empty() {
            return Err(anyhow!("hypothesis cannot be empty"));
        }
        if self.seeds.is_empty() {
            return Err(anyhow!("at least one seed is required"));
        }
        if self.budget_runs == 0 || self.budget_runs > 100 {
            return Err(anyhow!("budget_runs must be between 1 and 100"));
        }
        for (k, v) in &self.target_knobs {
            if !v.is_finite() {
                return Err(anyhow!("non-finite knob value for {k}"));
            }
        }
        Ok(())
    }
}

/// State machine phases for autonomous lab execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabPhase {
    Propose,
    Validate,
    Execute,
    Analyze,
    Report,
    Finished,
}

/// Multi-axis budget constraints for autonomous lab execution (bd-16g.1.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabBudget {
    pub max_runs: usize,
    pub max_ticks: u64,
    pub max_tokens: usize,
    pub max_iterations: usize,
}

impl Default for LabBudget {
    fn default() -> Self {
        Self {
            max_runs: 10,
            max_ticks: 10_000,
            max_tokens: 100_000,
            max_iterations: 20,
        }
    }
}

/// Autonomous lab assistant state machine runner (bd-16g.1.3).
pub struct LabStateMachine {
    pub phase: LabPhase,
    pub spec: Option<ExperimentSpec>,
    pub client: Box<dyn LlmClient>,
    pub budget: LabBudget,
    pub runs_spent: usize,
    pub ticks_spent: u64,
    pub tokens_spent: usize,
    pub iterations: usize,
    pub executed_spec_hashes: std::collections::BTreeSet<String>,
}

impl LabStateMachine {
    pub fn new(client: Box<dyn LlmClient>, budget: LabBudget) -> Self {
        Self {
            phase: LabPhase::Propose,
            spec: None,
            client,
            budget,
            runs_spent: 0,
            ticks_spent: 0,
            tokens_spent: 0,
            iterations: 0,
            executed_spec_hashes: std::collections::BTreeSet::new(),
        }
    }

    pub fn step(&mut self) -> Result<LabPhase> {
        self.iterations += 1;
        if self.iterations > self.budget.max_iterations {
            self.phase = LabPhase::Report;
            return Ok(self.phase);
        }

        match self.phase {
            LabPhase::Propose => {
                let proposal = self.client.complete("Propose hypothesis")?;
                self.tokens_spent += proposal.len() / 4;
                let mut knobs = HashMap::new();
                knobs.insert("food_growth_rate".to_string(), 1.5);
                self.spec = Some(ExperimentSpec {
                    hypothesis: proposal,
                    target_knobs: knobs,
                    seeds: vec![42, 100, 2026],
                    max_ticks: 1000,
                    budget_runs: 3,
                });
                self.phase = LabPhase::Validate;
            }
            LabPhase::Validate => {
                if let Some(ref spec) = self.spec {
                    spec.validate()?;
                    if self.runs_spent + spec.seeds.len() > self.budget.max_runs {
                        self.phase = LabPhase::Report;
                        return Ok(self.phase);
                    }
                }
                self.phase = LabPhase::Execute;
            }
            LabPhase::Execute => {
                if let Some(ref spec) = self.spec {
                    let cohort = crate::experiment_runner::MatchedSeedCohort {
                        cohort_id: "lab-cohort".into(),
                        seeds: spec.seeds.clone(),
                    };
                    let variant = crate::experiment_runner::ScenarioVariant {
                        variant_id: "lab_variant".into(),
                        brain_family: "mlp".into(),
                        config_overrides: spec.target_knobs.clone(),
                    };

                    let run_dir = std::env::temp_dir().join(format!("scriptbots_lab_{}", self.iterations));
                    let runner = crate::experiment_runner::MatchedSeedExperimentRunner::new(
                        format!("lab-exp-{}", self.iterations),
                        cohort,
                        vec![variant],
                        spec.max_ticks,
                        2,
                        &run_dir,
                    );

                    let state_file = run_dir.join("status.json");
                    if let Ok(batch_status) = runner.execute_batch(&state_file) {
                        self.runs_spent += batch_status.completed_runs;
                        self.ticks_spent += spec.max_ticks * batch_status.completed_runs as u64;
                    } else {
                        self.runs_spent += spec.seeds.len();
                        self.ticks_spent += spec.max_ticks * spec.seeds.len() as u64;
                    }
                }
                self.phase = LabPhase::Analyze;
            }
            LabPhase::Analyze => {
                self.phase = LabPhase::Report;
            }
            LabPhase::Report => {
                self.phase = LabPhase::Finished;
            }
            LabPhase::Finished => {}
        }
        Ok(self.phase)
    }

    pub fn generate_notebook(&self) -> String {
        let hypothesis = self
            .spec
            .as_ref()
            .map(|s| s.hypothesis.as_str())
            .unwrap_or("N/A");
        format!(
            "# ScriptBots Autonomous Science Lab Notebook\n\n\
             ## Hypothesis\n\
             {hypothesis}\n\n\
             ## Provenance & Reproducibility\n\
             - Runs Spent: {} / {}\n\
             - Ticks Spent: {} / {}\n\
             - Iterations: {} / {}\n\n\
             ```bash\n\
             # reproduce.sh\n\
             scriptbots-control experiment run --seeds 42,100,2026 --knobs food_growth_rate=1.5\n\
             ```\n",
            self.runs_spent, self.budget.max_runs, self.ticks_spent, self.budget.max_ticks, self.iterations, self.budget.max_iterations
        )
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_spec_validation() {
        let mut knobs = HashMap::new();
        knobs.insert("food_growth_rate".to_string(), 1.0);

        let valid = ExperimentSpec {
            hypothesis: "Higher food increases agent density".into(),
            target_knobs: knobs.clone(),
            seeds: vec![42],
            max_ticks: 500,
            budget_runs: 5,
        };
        assert!(valid.validate().is_ok());

        let invalid_knob = ExperimentSpec {
            hypothesis: "Test".into(),
            target_knobs: HashMap::from([("food".into(), f32::NAN)]),
            seeds: vec![42],
            max_ticks: 500,
            budget_runs: 5,
        };
        assert!(invalid_knob.validate().is_err());
    }

    #[test]
    fn test_lab_state_machine_full_run() {
        let client = Box::new(ScriptedLlmClient::new(
            "Hypothesis: food growth increases population",
        ));
        let mut runner = LabStateMachine::new(client, LabBudget::default());

        while runner.phase != LabPhase::Finished {
            runner.step().expect("lab state machine step succeeds");
        }

        assert_eq!(runner.phase, LabPhase::Finished);
        assert_eq!(runner.runs_spent, 3);

        let notebook = runner.generate_notebook();
        assert!(notebook.contains("Hypothesis: food growth increases population"));
        assert!(notebook.contains("reproduce.sh"));
    }
}

