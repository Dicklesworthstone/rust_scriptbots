//! Autonomous LLM lab assistant: canonical proposal -> validation -> matched-seed execution.

use crate::experiment_runner::{
    ExperimentBatchStatus, MatchedSeedCohort, MatchedSeedExperimentRunner, ScenarioVariant,
    validate_scenario_arm,
};
use crate::lab::llm::{
    LlmClient, LlmError, LlmMessage, LlmRequest, PROPOSE_EXPERIMENT_TOOL_NAME, StopReason,
    propose_experiment_tool,
};
use crate::lab::spec::{
    ExperimentSpec, SpecBudget, SpecError, ValidatedSpec, render_errors, validate_spec,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::PathBuf;

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

/// Successful execution accounting returned by an experiment executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionReceipt {
    /// Runs allocated by the executor.
    pub runs: usize,
    /// Ticks allocated across those runs.
    pub ticks: u64,
}

/// Side-effect boundary invoked only after canonical validation succeeds.
pub trait ExperimentExecutor: Send + Sync {
    /// Execute one validated deterministic experiment.
    ///
    /// # Errors
    ///
    /// Returns an actionable execution failure. Implementations must not accept
    /// an unvalidated proposal.
    fn execute(&self, spec: &ValidatedSpec) -> Result<ExecutionReceipt, String>;
}

#[derive(Debug)]
struct MatchedSeedExecutor {
    output_root: PathBuf,
}

impl MatchedSeedExecutor {
    fn new(output_root: impl Into<PathBuf>) -> Self {
        Self {
            output_root: output_root.into(),
        }
    }
}

impl ExperimentExecutor for MatchedSeedExecutor {
    fn execute(&self, spec: &ValidatedSpec) -> Result<ExecutionReceipt, String> {
        let cohort = MatchedSeedCohort {
            cohort_id: format!("{}-matched-seeds", spec.spec_id),
            seeds: spec.seeds.clone(),
        };
        let variants = spec
            .arms
            .iter()
            .enumerate()
            .map(|(index, arm)| ScenarioVariant {
                variant_id: format!("arm-{index:03}"),
                brain_family: "mlp".to_owned(),
                config_overrides: arm.clone(),
            })
            .collect();
        let output_dir = self.output_root.join(&spec.spec_id);
        let runner = MatchedSeedExperimentRunner::new(
            &spec.spec_id,
            cohort,
            variants,
            spec.spec.ticks_per_run,
            2,
            &output_dir,
        );
        let status = runner.execute_batch(&output_dir.join("status.json"))?;
        execution_receipt(spec, &status)
    }
}

fn execution_receipt(
    spec: &ValidatedSpec,
    status: &ExperimentBatchStatus,
) -> Result<ExecutionReceipt, String> {
    let cost = spec.cost();
    let expected_runs = usize::try_from(cost.runs)
        .map_err(|_| format!("validated run count {} does not fit usize", cost.runs))?;
    if status.total_runs != expected_runs || !status.is_finished() || status.failed_runs != 0 {
        return Err(format!(
            "experiment {} finished with {}/{} completed and {} failed; expected {} successful runs",
            spec.spec_id,
            status.completed_runs,
            status.total_runs,
            status.failed_runs,
            expected_runs
        ));
    }
    Ok(ExecutionReceipt {
        runs: expected_runs,
        ticks: cost.ticks,
    })
}

/// Typed state-machine failure. Validation failures retain the complete ordered
/// canonical error set for a repair turn.
#[derive(Debug, thiserror::Error)]
pub enum LabError {
    /// Provider failure.
    #[error(transparent)]
    Llm(#[from] LlmError),
    /// The provider did not make exactly one proposal call.
    #[error("expected exactly one `propose_experiment` call, received {found}")]
    ProposalToolCall {
        /// Calls returned by the model.
        found: usize,
    },
    /// The tool arguments do not deserialize as the canonical spec.
    #[error("proposal does not match the canonical ExperimentSpec: {0}")]
    InvalidProposal(String),
    /// The canonical validator rejected the proposal.
    #[error("experiment proposal failed validation:\n{rendered}")]
    Validation {
        /// Complete stable error list.
        errors: Vec<SpecError>,
        /// Actionable rendering for the next repair prompt.
        rendered: String,
    },
    /// The model consumed more tokens than the hard session ceiling.
    #[error("token budget exceeded: used {used}, allowed {allowed}")]
    TokenBudgetExceeded {
        /// Tokens actually reported by the provider.
        used: usize,
        /// Session limit.
        allowed: usize,
    },
    /// A validated arm cannot become a typed fresh-world configuration.
    #[error("validated arm {arm} cannot be executed: {reason}")]
    InvalidExecutionPlan {
        /// Arm index in deterministic expansion order.
        arm: usize,
        /// Strict typed-config failure.
        reason: String,
    },
    /// A state invariant was violated.
    #[error("lab state invariant violated: {0}")]
    State(String),
    /// The same validated experiment was submitted twice.
    #[error("experiment `{0}` was already executed")]
    DuplicateExperiment(String),
    /// Execution failed after the budget was allocated.
    #[error("experiment execution failed: {0}")]
    Execution(String),
    /// Executor accounting did not match the validated pre-flight cost.
    #[error(
        "executor accounting mismatch: expected {expected_runs} runs/{expected_ticks} ticks, \
         got {actual_runs} runs/{actual_ticks} ticks"
    )]
    AccountingMismatch {
        /// Canonical run count.
        expected_runs: usize,
        /// Canonical tick count.
        expected_ticks: u64,
        /// Executor-reported run count.
        actual_runs: usize,
        /// Executor-reported tick count.
        actual_ticks: u64,
    },
}

/// Autonomous lab assistant state machine runner (bd-16g.1.3).
pub struct LabStateMachine {
    pub phase: LabPhase,
    pub spec: Option<ExperimentSpec>,
    pub validated_spec: Option<ValidatedSpec>,
    pub validation_errors: Vec<SpecError>,
    pub proposal_id: Option<String>,
    pub budget: LabBudget,
    pub runs_spent: usize,
    pub ticks_spent: u64,
    pub tokens_spent: usize,
    pub iterations: usize,
    pub executed_spec_hashes: BTreeSet<String>,
    client: Box<dyn LlmClient>,
    executor: Box<dyn ExperimentExecutor>,
}

impl LabStateMachine {
    /// Build the production state machine. Merely constructing it performs no
    /// filesystem or world side effect.
    #[must_use]
    pub fn new(
        client: Box<dyn LlmClient>,
        budget: LabBudget,
        output_root: impl Into<PathBuf>,
    ) -> Self {
        Self::with_executor(
            client,
            budget,
            Box::new(MatchedSeedExecutor::new(output_root)),
        )
    }

    /// Build a state machine around an explicit execution boundary.
    #[must_use]
    pub fn with_executor(
        client: Box<dyn LlmClient>,
        budget: LabBudget,
        executor: Box<dyn ExperimentExecutor>,
    ) -> Self {
        Self {
            phase: LabPhase::Propose,
            spec: None,
            validated_spec: None,
            validation_errors: Vec::new(),
            proposal_id: None,
            budget,
            runs_spent: 0,
            ticks_spent: 0,
            tokens_spent: 0,
            iterations: 0,
            executed_spec_hashes: BTreeSet::new(),
            client,
            executor,
        }
    }

    /// Advance exactly one phase.
    ///
    /// # Errors
    ///
    /// Returns a typed provider, validation, planning, budget, or execution
    /// error. No executor call is possible before `Validate` succeeds.
    pub fn step(&mut self) -> Result<LabPhase, LabError> {
        self.iterations += 1;
        if self.iterations > self.budget.max_iterations
            && !matches!(self.phase, LabPhase::Report | LabPhase::Finished)
        {
            self.phase = LabPhase::Report;
            return Ok(self.phase);
        }

        match self.phase {
            LabPhase::Propose => self.propose()?,
            LabPhase::Validate => self.validate()?,
            LabPhase::Execute => self.execute()?,
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

    fn propose(&mut self) -> Result<(), LabError> {
        let remaining_tokens = self.budget.max_tokens.saturating_sub(self.tokens_spent);
        if remaining_tokens == 0 {
            self.phase = LabPhase::Report;
            return Err(LabError::TokenBudgetExceeded {
                used: self.tokens_spent,
                allowed: self.budget.max_tokens,
            });
        }
        let request = LlmRequest {
            system: "You propose bounded, falsifiable ScriptBots experiments. Call the offered \
                     tool exactly once; prose is not an experiment."
                .to_owned(),
            messages: vec![LlmMessage {
                role: "user".to_owned(),
                content: "Propose one matched-seed experiment within the supplied schema."
                    .to_owned(),
            }],
            tools: vec![propose_experiment_tool()?],
            max_tokens: u32::try_from(remaining_tokens.min(2_048)).unwrap_or(2_048),
        };
        let response = self.client.complete(&request)?;
        let used =
            usize::try_from(u64::from(response.usage.input) + u64::from(response.usage.output))
                .unwrap_or(usize::MAX);
        self.tokens_spent = self.tokens_spent.saturating_add(used);
        if self.tokens_spent > self.budget.max_tokens {
            self.phase = LabPhase::Report;
            return Err(LabError::TokenBudgetExceeded {
                used: self.tokens_spent,
                allowed: self.budget.max_tokens,
            });
        }
        if response.stop_reason != StopReason::ToolUse || response.tool_calls.len() != 1 {
            self.phase = LabPhase::Report;
            return Err(LabError::ProposalToolCall {
                found: response.tool_calls.len(),
            });
        }
        let call = response
            .tool_calls
            .into_iter()
            .next()
            .ok_or(LabError::ProposalToolCall { found: 0 })?;
        if call.name != PROPOSE_EXPERIMENT_TOOL_NAME {
            self.phase = LabPhase::Report;
            return Err(LabError::ProposalToolCall { found: 1 });
        }
        let proposal_id = stable_value_id(&call.arguments);
        let spec: ExperimentSpec = match serde_json::from_value(call.arguments) {
            Ok(spec) => spec,
            Err(error) => {
                tracing::warn!(
                    proposal_id = %proposal_id,
                    stage = "propose",
                    error_codes = ?["invalid_proposal"],
                    budget_decision = "not_allocated",
                    "lab proposal failed canonical schema decoding"
                );
                self.proposal_id = Some(proposal_id);
                self.phase = LabPhase::Report;
                return Err(LabError::InvalidProposal(error.to_string()));
            }
        };
        tracing::info!(
            proposal_id = %proposal_id,
            stage = "propose",
            model_id = self.client.model_id(),
            tokens_spent = self.tokens_spent,
            "lab proposal decoded through canonical tool schema"
        );
        self.proposal_id = Some(proposal_id);
        self.spec = Some(spec);
        self.phase = LabPhase::Validate;
        Ok(())
    }

    fn validate(&mut self) -> Result<(), LabError> {
        let spec = self
            .spec
            .as_ref()
            .ok_or_else(|| LabError::State("Validate phase has no proposal".to_owned()))?;
        let remaining_runs = self.budget.max_runs.saturating_sub(self.runs_spent);
        let operator_budget = SpecBudget {
            runs: u32::try_from(remaining_runs).unwrap_or(u32::MAX),
            ticks: self.budget.max_ticks.saturating_sub(self.ticks_spent),
        };
        let validated = match validate_spec(spec, operator_budget) {
            Ok(validated) => validated,
            Err(errors) => {
                let codes = errors.iter().map(SpecError::code).collect::<Vec<_>>();
                tracing::warn!(
                    proposal_id = self.proposal_id.as_deref().unwrap_or("unavailable"),
                    stage = "validate",
                    error_codes = ?codes,
                    budget_decision = "rejected",
                    "lab proposal rejected before execution"
                );
                self.validation_errors.clone_from(&errors);
                self.phase = LabPhase::Report;
                return Err(LabError::Validation {
                    rendered: render_errors(&errors),
                    errors,
                });
            }
        };
        for (arm, overrides) in validated.arms.iter().enumerate() {
            validate_scenario_arm(overrides).map_err(|reason| {
                self.phase = LabPhase::Report;
                LabError::InvalidExecutionPlan { arm, reason }
            })?;
        }
        if self.executed_spec_hashes.contains(&validated.spec_id) {
            self.phase = LabPhase::Report;
            return Err(LabError::DuplicateExperiment(validated.spec_id));
        }
        let cost = validated.cost();
        tracing::info!(
            proposal_id = self.proposal_id.as_deref().unwrap_or("unavailable"),
            spec_id = %validated.spec_id,
            stage = "validate",
            expanded_arms = validated.arms.len(),
            arm_assignments = ?validated.arms,
            seeds = ?validated.seeds,
            runs = cost.runs,
            ticks = cost.ticks,
            max_runs = operator_budget.runs,
            max_ticks = operator_budget.ticks,
            budget_decision = "accepted",
            "lab proposal passed canonical validation"
        );
        self.validation_errors.clear();
        self.validated_spec = Some(validated);
        self.phase = LabPhase::Execute;
        Ok(())
    }

    fn execute(&mut self) -> Result<(), LabError> {
        let validated = self
            .validated_spec
            .as_ref()
            .ok_or_else(|| LabError::State("Execute phase has no validated spec".to_owned()))?;
        let cost = validated.cost();
        let expected_runs = usize::try_from(cost.runs)
            .map_err(|_| LabError::State(format!("run count {} does not fit usize", cost.runs)))?;

        // Crossing this line allocates the complete preflight cost even if the
        // executor later reports a failed run. A failed experiment is not free.
        self.runs_spent = self.runs_spent.saturating_add(expected_runs);
        self.ticks_spent = self.ticks_spent.saturating_add(cost.ticks);
        let receipt = match self.executor.execute(validated) {
            Ok(receipt) => receipt,
            Err(error) => {
                self.phase = LabPhase::Report;
                return Err(LabError::Execution(error));
            }
        };
        if receipt.runs != expected_runs || receipt.ticks != cost.ticks {
            self.phase = LabPhase::Report;
            return Err(LabError::AccountingMismatch {
                expected_runs,
                expected_ticks: cost.ticks,
                actual_runs: receipt.runs,
                actual_ticks: receipt.ticks,
            });
        }
        self.executed_spec_hashes.insert(validated.spec_id.clone());
        self.phase = LabPhase::Analyze;
        Ok(())
    }

    /// Render a human-readable, reproducible notebook summary.
    #[must_use]
    pub fn generate_notebook(&self) -> String {
        let hypothesis = self
            .spec
            .as_ref()
            .map(|s| s.hypothesis.as_str())
            .unwrap_or("N/A");
        let spec_id = self
            .validated_spec
            .as_ref()
            .map_or("unvalidated", |spec| spec.spec_id.as_str());
        let seeds = self
            .validated_spec
            .as_ref()
            .map_or_else(|| "[]".to_owned(), |spec| format!("{:?}", spec.seeds));
        let arms = self.validated_spec.as_ref().map_or_else(
            || "[]".to_owned(),
            |spec| serde_json::to_string(&spec.arms).unwrap_or_else(|_| "[]".to_owned()),
        );
        format!(
            "# ScriptBots Autonomous Science Lab Notebook\n\n\
             ## Hypothesis\n\
             {hypothesis}\n\n\
             ## Provenance & Reproducibility\n\
             - Spec ID: {spec_id}\n\
             - Matched Seeds: {seeds}\n\
             - Ordered Arms: {arms}\n\
             - Runs Spent: {} / {}\n\
             - Ticks Spent: {} / {}\n\
             - Iterations: {} / {}\n\n\
             ```bash\n\
             # reproduce.sh\n\
             scriptbots-control experiment run --spec-id {spec_id}\n\
             ```\n",
            self.runs_spent,
            self.budget.max_runs,
            self.ticks_spent,
            self.budget.max_ticks,
            self.iterations,
            self.budget.max_iterations
        )
    }
}

fn stable_value_id(value: &serde_json::Value) -> String {
    let encoded = serde_json::to_vec(value).unwrap_or_default();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in encoded {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lab::llm::{ScriptedClient, ScriptedTurn};
    use std::sync::{Arc, Mutex};

    #[derive(Debug)]
    struct RecordingExecutor {
        calls: Arc<Mutex<Vec<ValidatedSpec>>>,
        marker: PathBuf,
    }

    impl ExperimentExecutor for RecordingExecutor {
        fn execute(&self, spec: &ValidatedSpec) -> Result<ExecutionReceipt, String> {
            std::fs::create_dir_all(&self.marker).map_err(|error| error.to_string())?;
            self.calls
                .lock()
                .map_err(|error| error.to_string())?
                .push(spec.clone());
            let cost = spec.cost();
            Ok(ExecutionReceipt {
                runs: usize::try_from(cost.runs)
                    .map_err(|_| "run count does not fit usize".to_owned())?,
                ticks: cost.ticks,
            })
        }
    }

    fn valid_input() -> serde_json::Value {
        serde_json::json!({
            "hypothesis": "faster food growth raises the final population",
            "falsifier": "matched seeds show no increase in alive agents",
            "factors": [{
                "knob_path": "food_growth_rate",
                "values": [0.01, 0.02]
            }],
            "seeds": {"base": 41, "count": 2},
            "ticks_per_run": 2,
            "metrics": ["alive_agents"],
            "budget": {"runs": 4, "ticks": 8}
        })
    }

    fn tool_turn(input: serde_json::Value) -> ScriptedTurn {
        ScriptedTurn {
            body: serde_json::json!({
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 4, "output_tokens": 6},
                "content": [{
                    "type": "tool_use",
                    "name": PROPOSE_EXPERIMENT_TOOL_NAME,
                    "input": input
                }]
            }),
        }
    }

    fn state_machine(
        input: serde_json::Value,
        marker: PathBuf,
        calls: Arc<Mutex<Vec<ValidatedSpec>>>,
    ) -> LabStateMachine {
        let client = ScriptedClient::new("offline-scripted", vec![tool_turn(input)]);
        LabStateMachine::with_executor(
            Box::new(client),
            LabBudget {
                max_runs: 64,
                max_ticks: 10_000,
                max_tokens: 1_000,
                max_iterations: 10,
            },
            Box::new(RecordingExecutor { calls, marker }),
        )
    }

    #[test]
    fn valid_tool_proposal_reaches_the_executor_as_the_canonical_plan() {
        let temp = tempfile::tempdir().expect("temp dir");
        let marker = temp.path().join("executor-called");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut runner = state_machine(valid_input(), marker.clone(), Arc::clone(&calls));

        assert_eq!(runner.step().expect("proposal"), LabPhase::Validate);
        assert_eq!(runner.tokens_spent, 10);
        assert!(runner.proposal_id.is_some());
        assert_eq!(runner.step().expect("validation"), LabPhase::Execute);
        let validated = runner
            .validated_spec
            .as_ref()
            .expect("validated spec")
            .clone();
        assert_eq!(validated.arms.len(), 2);
        assert_eq!(validated.seeds, [41, 42]);
        assert_eq!(validated.cost().runs, 4);

        assert_eq!(runner.step().expect("execution"), LabPhase::Analyze);
        assert_eq!(runner.runs_spent, 4);
        assert_eq!(runner.ticks_spent, 8);
        assert!(marker.is_dir());
        let recorded = calls.lock().expect("recorded calls");
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0], validated);
        drop(recorded);

        assert_eq!(runner.step().expect("analysis"), LabPhase::Report);
        assert_eq!(runner.step().expect("report"), LabPhase::Finished);
        let notebook = runner.generate_notebook();
        assert!(notebook.contains("faster food growth"));
        assert!(notebook.contains(&validated.spec_id));
        assert!(notebook.contains("Matched Seeds: [41, 42]"));
        assert!(notebook.contains("reproduce.sh"));
    }

    #[test]
    fn invalid_tool_proposals_report_all_codes_before_any_side_effect() {
        let temp = tempfile::tempdir().expect("temp dir");
        let marker = temp.path().join("must-not-exist");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let input = serde_json::json!({
            "hypothesis": "a malformed but decodable proposal",
            "falsifier": "",
            "factors": [
                {"knob_path": "unknown.knob", "values": [1]},
                {"knob_path": "food_growth_rate", "values": [2.0, 2.0, "NaN"]}
            ],
            "seeds": {"base": u64::MAX, "count": 2},
            "ticks_per_run": 100,
            "metrics": ["vibes"],
            "budget": {"runs": 1, "ticks": 1}
        });
        let mut runner = state_machine(input, marker.clone(), Arc::clone(&calls));
        assert_eq!(runner.step().expect("proposal"), LabPhase::Validate);
        let error = runner.step().expect_err("canonical validation must reject");
        let LabError::Validation { errors, .. } = error else {
            panic!("expected complete validation error set, got {error:?}");
        };
        assert_eq!(
            errors.iter().map(SpecError::code).collect::<Vec<_>>(),
            [
                "empty_falsifier",
                "unknown_metric",
                "unknown_knob",
                "out_of_range",
                "out_of_range",
                "non_finite",
                "duplicate_arm",
                "seed_overflow",
                "cost_exceeds_budget",
            ]
        );
        assert_eq!(runner.phase, LabPhase::Report);
        assert_eq!(runner.runs_spent, 0);
        assert_eq!(runner.ticks_spent, 0);
        assert!(runner.validated_spec.is_none());
        assert!(calls.lock().expect("calls").is_empty());
        assert!(!marker.exists());
        assert_eq!(
            runner.step().expect("report terminates"),
            LabPhase::Finished
        );
    }

    #[test]
    fn malformed_legacy_shape_never_reaches_validation_or_execution() {
        let temp = tempfile::tempdir().expect("temp dir");
        let marker = temp.path().join("must-not-exist");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut input = valid_input();
        input
            .as_object_mut()
            .expect("proposal object")
            .insert("target_knobs".to_owned(), serde_json::json!({}));
        let mut runner = state_machine(input, marker.clone(), Arc::clone(&calls));
        assert!(matches!(
            runner.step(),
            Err(LabError::InvalidProposal(message)) if message.contains("target_knobs")
        ));
        assert!(runner.proposal_id.is_some());
        assert_eq!(runner.phase, LabPhase::Report);
        assert_eq!(runner.runs_spent, 0);
        assert_eq!(runner.ticks_spent, 0);
        assert!(calls.lock().expect("calls").is_empty());
        assert!(!marker.exists());
    }
}
