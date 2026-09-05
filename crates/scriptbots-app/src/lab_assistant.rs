//! Autonomous LLM lab assistant: canonical proposal -> validation -> matched-seed execution.

use crate::experiment_runner::{
    ExperimentBatchStatus, MatchedSeedCohort, MatchedSeedExperimentRunner, ScenarioVariant,
    validate_scenario_arm,
};
use crate::lab::llm::{
    LlmClient, LlmError, LlmMessage, LlmRequest, PROPOSE_EXPERIMENT_TOOL_NAME, StopReason,
    propose_experiment_tool,
};
use crate::lab::notebook::{NotebookRenderError, NotebookRenderer, claims_from_analysis, run_refs};
use crate::lab::spec::{
    ExperimentSpec, SpecBudget, SpecError, ValidatedSpec, render_errors, validate_spec,
};
use crate::lab::stats::{
    AnalysisParams, MatchedSeedAnalysis, RunSummary, RunSummaryParts, StatsError,
    analyze_matched_seed_runs,
};
use scriptbots_storage::{
    RunBundleV1, bundle::RunBundleVerificationLimits, bundle::verify_run_bundle_bounded,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;

const MAX_ANALYSIS_SUMMARY_BYTES: usize = 4_096;
const MAX_ANALYSIS_BUNDLE_MANIFEST_BYTES: usize = 512 * 1_024;

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
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionReceipt {
    /// Runs allocated by the executor.
    pub runs: usize,
    /// Ticks allocated across those runs.
    pub ticks: u64,
    /// Verified per-run scientific summaries in canonical variant/seed order.
    pub summaries: Vec<RunSummary>,
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
pub struct MatchedSeedExecutor {
    output_root: PathBuf,
}

impl MatchedSeedExecutor {
    #[must_use]
    pub fn new(output_root: impl Into<PathBuf>) -> Self {
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
        let status = runner
            .execute_batch(&output_dir.join("status.json"))
            .map_err(|error| error.to_string())?;
        let summaries = completed_run_summaries(spec, &status)?;
        execution_receipt(spec, &status, summaries)
    }
}

fn execution_receipt(
    spec: &ValidatedSpec,
    status: &ExperimentBatchStatus,
    summaries: Vec<RunSummary>,
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
        summaries,
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperimentSummaryRow {
    tick: u64,
    alive_agents: usize,
    seed: u64,
    brain_family: String,
    final_digest: String,
}

fn read_bounded_analysis_file(path: &std::path::Path, limit: usize) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("inspect {}: {error}", path.display()))?;
    if !metadata.is_file() || metadata.len() > u64::try_from(limit).unwrap_or(u64::MAX) {
        return Err(format!(
            "{} must be a regular file no larger than {limit} bytes",
            path.display()
        ));
    }
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(limit).min(limit));
    file.take(u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() > limit {
        return Err(format!(
            "{} exceeded the {limit}-byte analysis bound while reading",
            path.display()
        ));
    }
    Ok(bytes)
}

fn completed_run_summaries(
    spec: &ValidatedSpec,
    status: &ExperimentBatchStatus,
) -> Result<Vec<RunSummary>, String> {
    let mut summaries = Vec::with_capacity(status.runs.len());
    for record in &status.runs {
        let arm_id = record
            .variant_id
            .strip_prefix("arm-")
            .ok_or_else(|| {
                format!(
                    "run {} has non-canonical variant_id {}",
                    record.run_id, record.variant_id
                )
            })?
            .parse::<u16>()
            .map_err(|error| {
                format!(
                    "run {} has invalid variant_id {}: {error}",
                    record.run_id, record.variant_id
                )
            })?;
        if spec.arms.get(usize::from(arm_id)).is_none() {
            return Err(format!(
                "run {} references arm {} outside validated arm count {}",
                record.run_id,
                arm_id,
                spec.arms.len()
            ));
        }
        let bundle_path = record.bundle_path.as_ref().ok_or_else(|| {
            format!(
                "completed experiment run {} has no verified bundle path",
                record.run_id
            )
        })?;
        let bundle_dir = PathBuf::from(bundle_path);
        let bundle_manifest_path = bundle_dir.join("bundle_manifest.json");
        let bundle_bytes =
            read_bounded_analysis_file(&bundle_manifest_path, MAX_ANALYSIS_BUNDLE_MANIFEST_BYTES)?;
        let bundle: RunBundleV1 = serde_json::from_slice(&bundle_bytes)
            .map_err(|error| format!("parse {}: {error}", bundle_manifest_path.display()))?;
        let summary_entries = bundle
            .artifacts
            .iter()
            .filter(|entry| {
                entry.relative_path == "exports/summary.csv"
                    && entry.artifact_type == "experiment-summary"
            })
            .collect::<Vec<_>>();
        let [summary_entry] = summary_entries.as_slice() else {
            return Err(format!(
                "bundle {} must index exports/summary.csv exactly once as experiment-summary",
                bundle_dir.display()
            ));
        };
        for (matches, field) in [
            (
                bundle.manifest.variant_id.as_deref() == Some(record.variant_id.as_str()),
                "variant_id",
            ),
            (bundle.manifest.root_seed == record.seed, "root_seed"),
            (
                bundle.manifest.requested_tick_budget == Some(record.total_ticks),
                "requested_tick_budget",
            ),
            (bundle.digests.max_tick == record.total_ticks, "max_tick"),
        ] {
            if !matches {
                return Err(format!(
                    "bundle {} disagrees with completed run {} on {field}",
                    bundle_dir.display(),
                    record.run_id
                ));
            }
        }

        let summary_path = bundle_dir.join("exports/summary.csv");
        let bytes = read_bounded_analysis_file(&summary_path, MAX_ANALYSIS_SUMMARY_BYTES)?;
        if summary_entry.bytes_len != u64::try_from(bytes.len()).unwrap_or(u64::MAX)
            || summary_entry.blake3_hex != blake3::hash(&bytes).to_hex().as_str()
        {
            return Err(format!(
                "summary {} no longer matches its verified bundle artifact entry",
                summary_path.display()
            ));
        }
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(bytes.as_slice());
        let mut rows = reader.deserialize::<ExperimentSummaryRow>();
        let row = rows
            .next()
            .ok_or_else(|| format!("summary {} has no data row", summary_path.display()))?
            .map_err(|error| format!("parse {}: {error}", summary_path.display()))?;
        if rows.next().is_some() {
            return Err(format!(
                "summary {} must contain exactly one data row",
                summary_path.display()
            ));
        }
        let final_digest = record.final_digest.as_ref().ok_or_else(|| {
            format!(
                "completed experiment run {} has no final digest",
                record.run_id
            )
        })?;
        for (matches, field) in [
            (row.tick == record.total_ticks, "tick"),
            (row.seed == record.seed, "seed"),
            (row.brain_family == record.brain_family, "brain_family"),
            (row.final_digest == *final_digest, "final_digest"),
        ] {
            if !matches {
                return Err(format!(
                    "summary {} disagrees with verified run {} on {field}",
                    summary_path.display(),
                    record.run_id
                ));
            }
        }
        let alive_agents = u32::try_from(row.alive_agents).map_err(|_| {
            format!(
                "summary {} alive_agents {} exceeds the bounded reporting representation",
                summary_path.display(),
                row.alive_agents
            )
        })?;
        verify_run_bundle_bounded(
            &bundle_dir,
            RunBundleVerificationLimits {
                max_manifest_bytes: u64::try_from(MAX_ANALYSIS_BUNDLE_MANIFEST_BYTES)
                    .unwrap_or(u64::MAX),
                max_artifacts: 32,
                max_artifact_bytes: u64::try_from(MAX_ANALYSIS_BUNDLE_MANIFEST_BYTES)
                    .unwrap_or(u64::MAX),
                max_total_artifact_bytes: 1_024 * 1_024,
            },
        )
        .map_err(|error| {
            format!(
                "bundle {} changed before analysis: {error}",
                bundle_dir.display()
            )
        })?;
        // The arm's actual overrides, not just its ordinal. An out-of-range arm is a hard
        // error rather than an empty map: empty overrides are a VALID config (the defaults),
        // so defaulting here would digest a broken run as a clean default-config run.
        let config_overrides = spec
            .arms
            .get(usize::from(arm_id))
            .ok_or_else(|| {
                format!(
                    "run {} names arm {arm_id} but the spec declares only {} arm(s)",
                    record.run_id,
                    spec.arms.len()
                )
            })?
            .clone();
        summaries.push(RunSummary::from_verified_parts(RunSummaryParts {
            run_id: record.run_id.clone(),
            arm_id,
            seed: row.seed,
            config_digest: bundle.manifest.config_digest.clone(),
            digest: row.final_digest,
            ticks: row.tick,
            metrics: BTreeMap::from([("alive_agents".to_owned(), f64::from(alive_agents))]),
            summary_artifact_digest: summary_entry.blake3_hex.clone(),
            summary_path: Some(summary_path.to_string_lossy().into_owned()),
            variant_id: record.variant_id.clone(),
            config_overrides,
        }));
    }
    Ok(summaries)
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
    /// The canonical statistics authority rejected completed run evidence.
    #[error("lab analysis failed: {0}")]
    Analysis(#[source] StatsError),
    /// The notebook rejected statistical or run provenance.
    #[error("lab notebook failed: {0}")]
    Notebook(#[source] NotebookRenderError),
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
    /// Verified completed summaries retained for the Analyze phase.
    pub run_summaries: Vec<RunSummary>,
    /// Canonical structured report produced by the Analyze phase.
    pub analysis: Option<MatchedSeedAnalysis>,
    /// Provenance-checked Markdown produced by the Report phase.
    pub rendered_notebook: Option<String>,
    /// Materialized notebook path for production runs.
    pub notebook_path: Option<PathBuf>,
    /// Exact terminal failure retained when a phase refuses to continue.
    pub failure_reason: Option<String>,
    client: Box<dyn LlmClient>,
    executor: Box<dyn ExperimentExecutor>,
    notebook_root: Option<PathBuf>,
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
        let output_root = output_root.into();
        Self::with_executor_and_notebook_root(
            client,
            budget,
            Box::new(MatchedSeedExecutor::new(output_root.clone())),
            Some(output_root),
        )
    }

    /// Build a state machine around an explicit execution boundary.
    #[must_use]
    pub fn with_executor(
        client: Box<dyn LlmClient>,
        budget: LabBudget,
        executor: Box<dyn ExperimentExecutor>,
    ) -> Self {
        Self::with_executor_and_notebook_root(client, budget, executor, None)
    }

    fn with_executor_and_notebook_root(
        client: Box<dyn LlmClient>,
        budget: LabBudget,
        executor: Box<dyn ExperimentExecutor>,
        notebook_root: Option<PathBuf>,
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
            run_summaries: Vec::new(),
            analysis: None,
            rendered_notebook: None,
            notebook_path: None,
            failure_reason: None,
            client,
            executor,
            notebook_root,
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
            self.failure_reason = Some(format!(
                "iteration budget exhausted after {} transitions (limit {})",
                self.iterations, self.budget.max_iterations
            ));
            self.phase = LabPhase::Report;
            return Ok(self.phase);
        }

        let transition = match self.phase {
            LabPhase::Propose => self.propose(),
            LabPhase::Validate => self.validate(),
            LabPhase::Execute => self.execute(),
            LabPhase::Analyze => self.analyze(),
            LabPhase::Report => self.report(),
            LabPhase::Finished => Ok(()),
        };
        if let Err(error) = &transition {
            self.failure_reason = Some(error.to_string());
        }
        transition?;
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
        if receipt.summaries.len() != expected_runs {
            self.phase = LabPhase::Report;
            return Err(LabError::Execution(format!(
                "experiment {} returned {} summaries for {expected_runs} completed runs",
                validated.spec_id,
                receipt.summaries.len()
            )));
        }
        self.run_summaries = receipt.summaries;
        self.executed_spec_hashes.insert(validated.spec_id.clone());
        self.phase = LabPhase::Analyze;
        Ok(())
    }

    fn analyze(&mut self) -> Result<(), LabError> {
        let validated = self
            .validated_spec
            .as_ref()
            .ok_or_else(|| LabError::State("Analyze phase has no validated spec".to_owned()))?;
        let started = std::time::Instant::now();
        let analysis = match analyze_matched_seed_runs(
            &self.run_summaries,
            &validated.spec.metrics,
            AnalysisParams::default(),
        ) {
            Ok(analysis) => analysis,
            Err(error) => {
                self.phase = LabPhase::Report;
                return Err(LabError::Analysis(error));
            }
        };
        tracing::info!(
            spec_id = %validated.spec_id,
            stage = "analyze",
            inputs = self.run_summaries.len(),
            effects = analysis.effects.len(),
            correction = analysis.params.correction.as_str(),
            alternative = analysis.params.alternative.as_str(),
            bootstrap_iterations = analysis.params.bootstrap_iterations,
            permutation_iterations = analysis.params.permutation_iterations,
            resampling_seed = analysis.params.resampling_seed,
            elapsed_micros = started.elapsed().as_micros(),
            "matched-seed summaries analyzed by the canonical statistics authority"
        );
        self.analysis = Some(analysis);
        self.phase = LabPhase::Report;
        Ok(())
    }

    fn report(&mut self) -> Result<(), LabError> {
        let (Some(validated), Some(analysis)) =
            (self.validated_spec.as_ref(), self.analysis.as_ref())
        else {
            let reason = self
                .failure_reason
                .as_deref()
                .unwrap_or("no validated statistical analysis was produced");
            let rendered = format!(
                "# ScriptBots Autonomous Science Lab Notebook\n\n\
                 ## Outcome\n\
                 No scientific result is available.\n\n\
                 ## Typed Refusal\n\
                 {reason}\n"
            );
            if let Some(root) = &self.notebook_root {
                let report_id = self
                    .proposal_id
                    .as_deref()
                    .unwrap_or("unvalidated-proposal");
                let directory = root.join(report_id).join("notebook");
                fs::create_dir_all(&directory).map_err(|error| {
                    LabError::Notebook(NotebookRenderError::Io(error.to_string()))
                })?;
                let path = directory.join("notebook.md");
                fs::write(&path, &rendered).map_err(|error| {
                    LabError::Notebook(NotebookRenderError::Io(error.to_string()))
                })?;
                self.notebook_path = Some(path);
            }
            self.rendered_notebook = Some(rendered);
            self.phase = LabPhase::Finished;
            return Ok(());
        };
        let known_runs = run_refs(&self.run_summaries);
        let claims = match claims_from_analysis(
            analysis,
            &self.run_summaries,
            &validated.spec.hypothesis,
            &validated.spec.falsifier,
        ) {
            Ok(claims) => claims,
            Err(error) => {
                self.phase = LabPhase::Finished;
                return Err(LabError::Notebook(error));
            }
        };
        let goal = format!(
            "{}\n\n- Validated Spec ID: {}",
            validated.spec.hypothesis, validated.spec_id
        );
        let rendered = match NotebookRenderer::render_markdown(&goal, &claims, &known_runs) {
            Ok(rendered) => rendered,
            Err(error) => {
                self.phase = LabPhase::Finished;
                return Err(LabError::Notebook(error));
            }
        };
        if let Some(root) = &self.notebook_root {
            let directory = root.join(&validated.spec_id).join("notebook");
            let path = NotebookRenderer::render_notebook(
                &validated.spec_id,
                &goal,
                &claims,
                &known_runs,
                &directory,
            )
            .map_err(LabError::Notebook)?;
            self.notebook_path = Some(path);
        }
        tracing::info!(
            spec_id = %validated.spec_id,
            stage = "report",
            effects = analysis.effects.len(),
            notebook_bytes = rendered.len(),
            "provenance-checked lab notebook rendered"
        );
        self.rendered_notebook = Some(rendered);
        self.phase = LabPhase::Finished;
        Ok(())
    }

    /// Return the completed report, or an explicitly provisional state summary.
    #[must_use]
    pub fn generate_notebook(&self) -> String {
        if let Some(rendered) = &self.rendered_notebook {
            return rendered.clone();
        }
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
             ## Provisional State\n\
             No completed statistical report has been rendered yet.\n\n\
             - Spec ID: {spec_id}\n\
             - Matched Seeds: {seeds}\n\
             - Ordered Arms: {arms}\n\
             - Runs Spent: {} / {}\n\
             - Ticks Spent: {} / {}\n\
             - Iterations: {} / {}\n\
             - Failure: {}\n",
            self.runs_spent,
            self.budget.max_runs,
            self.ticks_spent,
            self.budget.max_ticks,
            self.iterations,
            self.budget.max_iterations,
            self.failure_reason.as_deref().unwrap_or("none recorded")
        )
    }

    /// Return the written notebook path, if one was persisted.
    #[must_use]
    pub fn notebook_path(&self) -> Option<&PathBuf> {
        self.notebook_path.as_ref()
    }

    /// Advance the state machine repeatedly until reaching Finished or an unrecoverable error.
    pub fn run_to_completion(&mut self) -> Result<LabPhase, LabError> {
        while self.phase != LabPhase::Finished {
            let phase = self.step()?;
            if phase == LabPhase::Finished {
                break;
            }
        }
        Ok(self.phase)
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
            let mut summaries = Vec::new();
            for (arm_index, arm) in spec.arms.iter().enumerate() {
                let arm_id = u16::try_from(arm_index)
                    .map_err(|_| "arm index does not fit u16".to_owned())?;
                let config_digest =
                    blake3::hash(&serde_json::to_vec(arm).map_err(|error| error.to_string())?)
                        .to_hex()
                        .to_string();
                for &seed in &spec.seeds {
                    let seed_component = u32::try_from(seed % 17)
                        .map_err(|_| "bounded seed component does not fit u32".to_owned())?;
                    let arm_component = u32::from(arm_id)
                        .checked_mul(seed_component + 1)
                        .ok_or_else(|| "synthetic test outcome overflowed u32".to_owned())?;
                    let run_id = format!("{}-arm-{arm_id:03}-seed{seed}", spec.spec_id);
                    summaries.push(RunSummary::from_verified_parts(RunSummaryParts {
                        run_id: run_id.clone(),
                        arm_id,
                        seed,
                        config_digest: config_digest.clone(),
                        digest: format!("digest-{arm_id}-{seed}"),
                        ticks: spec.spec.ticks_per_run,
                        metrics: BTreeMap::from([(
                            "alive_agents".to_owned(),
                            f64::from(100 + seed_component + arm_component),
                        )]),
                        summary_artifact_digest: format!("summary-{run_id}"),
                        summary_path: None,
                        variant_id: format!("arm-{arm_id:03}"),
                        config_overrides: arm.clone(),
                    }));
                }
            }
            Ok(ExecutionReceipt {
                runs: usize::try_from(cost.runs)
                    .map_err(|_| "run count does not fit usize".to_owned())?,
                ticks: cost.ticks,
                summaries,
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

    fn three_arm_input() -> serde_json::Value {
        serde_json::json!({
            "hypothesis": "faster food growth changes the final population",
            "falsifier": "matched-seed final populations are unchanged",
            "factors": [{
                "knob_path": "food_growth_rate",
                "values": [0.01, 0.02, 0.03]
            }],
            "seeds": {"base": 101, "count": 3},
            "ticks_per_run": 2,
            "metrics": ["alive_agents"],
            "budget": {"runs": 9, "ticks": 18}
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
        let analysis = runner.analysis.as_ref().expect("analysis retained");
        assert_eq!(analysis.effects.len(), 1);
        assert_eq!(analysis.effects[0].metric, "alive_agents");
        assert_eq!(runner.run_summaries.len(), 4);
        assert_eq!(runner.step().expect("report"), LabPhase::Finished);
        let notebook = runner.generate_notebook();
        assert!(notebook.contains("faster food growth"));
        assert!(notebook.contains(&validated.spec_id));
        assert!(notebook.contains("Estimator**: paired_difference"));
        assert!(notebook.contains("Correction**: benjamini_hochberg"));
        assert!(notebook.contains("Matched Pairs**: 2"));
        assert!(notebook.contains("reproduce.sh"));
        assert_eq!(runner.rendered_notebook.as_deref(), Some(notebook.as_str()));
    }

    #[test]
    fn real_matched_seed_pipeline_is_byte_stable_and_data_responsive() {
        let temp = tempfile::tempdir().expect("temporary experiment root");
        let run_lab = |output_root: PathBuf| {
            // The provider response is an offline canonical tool fixture; validation, world
            // execution, bundle verification, statistics, and notebook materialization are real.
            let client =
                ScriptedClient::new("offline-scripted", vec![tool_turn(three_arm_input())]);
            let mut lab = LabStateMachine::new(
                Box::new(client),
                LabBudget {
                    max_runs: 9,
                    max_ticks: 18,
                    max_tokens: 1_000,
                    max_iterations: 10,
                },
                output_root,
            );
            for expected in [
                LabPhase::Validate,
                LabPhase::Execute,
                LabPhase::Analyze,
                LabPhase::Report,
                LabPhase::Finished,
            ] {
                assert_eq!(lab.step().expect("real lab transition"), expected);
            }
            lab
        };
        let first_lab = run_lab(temp.path().join("first"));
        assert_eq!(first_lab.runs_spent, 9);
        assert_eq!(first_lab.ticks_spent, 18);
        assert_eq!(first_lab.run_summaries.len(), 9);
        assert_eq!(
            first_lab
                .run_summaries
                .iter()
                .map(|summary| summary.config_digest.as_str())
                .collect::<BTreeSet<_>>()
                .len(),
            9,
            "the authoritative executed-config digest must include each matched seed"
        );
        assert_eq!(
            first_lab
                .analysis
                .as_ref()
                .expect("analysis retained")
                .effects
                .len(),
            2
        );
        let first = first_lab
            .rendered_notebook
            .as_ref()
            .expect("notebook retained");
        let notebook_path = first_lab
            .notebook_path
            .as_ref()
            .expect("production report materialized");
        assert_eq!(
            fs::read_to_string(notebook_path).expect("materialized notebook readable"),
            first.as_str()
        );
        let verifier = fs::read_to_string(
            notebook_path
                .parent()
                .expect("notebook parent")
                .join("reproduce.sh"),
        )
        .expect("retained evidence verifier");
        assert!(verifier.contains("b3sum"));
        // The verifier's SCOPE changed, and this assertion was pinning the old
        // limit. Until 5b64955ad7 ("reproduce.sh actually re-executes every arm x
        // seed") the script only checked retained artifact hashes, so it
        // truthfully said it did not re-run the simulation. It now re-executes
        // every arm x seed from the emitted config and compares each re-run world
        // digest against the cited one, exiting nonzero on any mismatch.
        //
        // Re-adding the old sentence to make this pass would have asserted a
        // limitation the product no longer has, and would have quietly weakened
        // the guarantee this test exists to defend. So it asserts the STRONGER
        // property instead: that the verifier really does re-execute and compare.
        assert!(
            verifier.contains("SCRIPTBOTS_DET_RUN=1"),
            "the verifier must actually re-execute each arm x seed, not only hash \
             retained artifacts"
        );
        assert!(
            verifier.contains("digest differs on re-execution"),
            "the verifier must compare each re-run digest against the cited one and \
             fail when they differ; hashing artifacts alone does not reproduce a run"
        );
        assert!(
            verifier.contains("did not reproduce"),
            "the verifier must exit nonzero when any arm fails to reproduce"
        );

        let repeated_lab = run_lab(temp.path().join("repeated"));
        assert_eq!(
            repeated_lab.rendered_notebook.as_ref(),
            Some(first),
            "independent executions of the same seeds must produce a byte-stable report"
        );

        let validated = first_lab
            .validated_spec
            .as_ref()
            .expect("validated spec retained");
        let render = |summaries: &[RunSummary]| {
            let analysis = analyze_matched_seed_runs(
                summaries,
                &validated.spec.metrics,
                AnalysisParams::default(),
            )
            .expect("verified summaries analyze");
            assert_eq!(analysis.effects.len(), 2);
            let claims = claims_from_analysis(
                &analysis,
                summaries,
                &validated.spec.hypothesis,
                &validated.spec.falsifier,
            )
            .expect("analysis retains exact run provenance");
            let goal = format!(
                "{}\n\n- Validated Spec ID: {}",
                validated.spec.hypothesis, validated.spec_id
            );
            NotebookRenderer::render_markdown(&goal, &claims, &run_refs(summaries))
                .expect("verified analysis renders")
        };
        let mut reordered = first_lab.run_summaries.clone();
        reordered.reverse();
        assert_eq!(
            render(&reordered),
            first.as_str(),
            "input order must not change the scientific report"
        );

        let mut changed = first_lab.run_summaries.clone();
        let treatment_index = changed
            .iter()
            .position(|summary| summary.arm_id == 1)
            .expect("treatment run");
        let treatment = &changed[treatment_index];
        let mut changed_metrics = treatment.metrics.clone();
        *changed_metrics
            .get_mut("alive_agents")
            .expect("reported metric") += 1.0;
        changed[treatment_index] = RunSummary::from_verified_parts(RunSummaryParts {
            run_id: treatment.run_id.clone(),
            arm_id: treatment.arm_id,
            seed: treatment.seed,
            config_digest: treatment.config_digest.clone(),
            digest: blake3::hash(b"independently changed world fixture")
                .to_hex()
                .to_string(),
            ticks: treatment.ticks,
            metrics: changed_metrics,
            summary_artifact_digest: blake3::hash(
                b"independently changed verified summary fixture",
            )
            .to_hex()
            .to_string(),
            summary_path: None,
            variant_id: treatment.variant_id.clone(),
            config_overrides: treatment.config_overrides.clone(),
        });
        assert_ne!(
            render(&changed),
            first.as_str(),
            "a changed observed outcome must change the report"
        );
    }

    #[test]
    fn analysis_ingestion_rejects_a_summary_changed_after_bundle_verification() {
        let proposed: ExperimentSpec =
            serde_json::from_value(valid_input()).expect("canonical proposal");
        let validated = validate_spec(&proposed, SpecBudget { runs: 4, ticks: 8 })
            .expect("bounded proposal validates");
        let temp = tempfile::tempdir().expect("temporary experiment root");
        MatchedSeedExecutor::new(temp.path())
            .execute(&validated)
            .expect("real bundle cohort");
        let status_path = temp.path().join(&validated.spec_id).join("status.json");
        let status: ExperimentBatchStatus = serde_json::from_slice(
            &fs::read(&status_path).expect("completed status remains readable"),
        )
        .expect("completed status schema");
        let summary_path = PathBuf::from(
            status.runs[0]
                .bundle_path
                .as_ref()
                .expect("completed run bundle"),
        )
        .join("exports/summary.csv");
        let summary = fs::read_to_string(&summary_path).expect("summary fixture");
        let mut lines = summary.lines();
        let header = lines.next().expect("summary header");
        let row = lines.next().expect("summary row");
        let mut cells = row.split(',').map(str::to_owned).collect::<Vec<_>>();
        let alive = cells[1].parse::<u32>().expect("alive count");
        cells[1] = alive.saturating_add(1).to_string();
        fs::write(&summary_path, format!("{header}\n{}\n", cells.join(",")))
            .expect("deliberate post-verification mutation");

        let error = completed_run_summaries(&validated, &status)
            .expect_err("artifact mutation must fail closed");
        assert!(
            error.contains("no longer matches its verified bundle artifact entry"),
            "unexpected refusal: {error}"
        );
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
        let report = runner
            .rendered_notebook
            .as_deref()
            .expect("typed refusal report retained");
        assert!(report.contains("No scientific result is available"));
        assert!(report.contains("experiment proposal failed validation"));
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
