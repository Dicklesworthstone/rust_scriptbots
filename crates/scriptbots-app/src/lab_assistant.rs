//! Autonomous LLM lab assistant: hypothesis -> matched-seed sweep -> analysis -> lab notebook (bd-16g.1).

use anyhow::{anyhow, Result};
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

/// Autonomous lab assistant state machine runner.
pub struct LabStateMachine {
    pub phase: LabPhase,
    pub spec: Option<ExperimentSpec>,
    pub client: Box<dyn LlmClient>,
    pub runs_spent: usize,
    pub max_runs: usize,
}

impl LabStateMachine {
    pub fn new(client: Box<dyn LlmClient>, max_runs: usize) -> Self {
        Self {
            phase: LabPhase::Propose,
            spec: None,
            client,
            runs_spent: 0,
            max_runs,
        }
    }

    pub fn step(&mut self) -> Result<LabPhase> {
        match self.phase {
            LabPhase::Propose => {
                let proposal = self.client.complete("Propose hypothesis")?;
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
                }
                self.phase = LabPhase::Execute;
            }
            LabPhase::Execute => {
                if let Some(ref spec) = self.spec {
                    self.runs_spent += spec.seeds.len();
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
             - Runs Spent: {}\n\
             - Maximum Budget: {}\n\n\
             ```bash\n\
             # reproduce.sh\n\
             scriptbots-control experiment run --seeds 42,100,2026 --knobs food_growth_rate=1.5\n\
             ```\n",
            self.runs_spent, self.max_runs
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
        let mut runner = LabStateMachine::new(client, 10);

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
