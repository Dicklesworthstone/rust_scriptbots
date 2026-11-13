//! The lab's action space: what the model is allowed to ask for.
//!
//! The whole safety case for the autonomous lab rests on one sentence — the
//! model is "a proposer over a validated, bounded action space." This module IS
//! that action space, and it is what turns that sentence from a claim into a
//! mechanism.
//!
//! # The gap this closes
//!
//! `ScriptBotsConfig::validate` enforces ADMISSIBILITY, not PLAUSIBILITY: mostly
//! `>= 0.0`, `is_finite()`, and a handful of genuine intervals. `food_growth_rate
//! = 1e9` PASSES it. So "validate the model's proposal against the config
//! schema" is, on its own, a no-op for range checking: a maximally confused model
//! could burn an hour of runs on a degenerate world where every food cell is
//! saturated, and the notebook would confidently report statistics over garbage.
//!
//! The range layer that closes this lives in [`scriptbots_core::KNOB_RANGES`],
//! and this module is what enforces it at the proposal boundary — BEFORE any run
//! starts, rather than one rejected run at a time.
//!
//! # What makes a proposal an experiment rather than an opinion
//!
//! The spec type refuses to be built without a FALSIFIER, a MATCHED SEED COHORT,
//! and a DECLARED METRIC. Those three fields are the difference between a
//! hypothesis and a hunch.
//!
//! # Pure
//!
//! Types in, `Result<ValidatedSpec, Vec<SpecError>>` out. No world lock, no
//! storage, no network, and — deliberately — no logging: the errors ARE the
//! product surface, rendered as one actionable line each and fed verbatim back to
//! the model as a repair prompt.

use crate::ScenarioInterventionV1;
use crate::experiment_runner::{
    DEFAULT_EXPERIMENT_BRAIN_FAMILY, EXPERIMENT_BRAIN_FAMILIES,
    MAX_EXPERIMENT_INTERVENTION_PATCH_BYTES, MAX_EXPERIMENT_INTERVENTIONS, MatchedSeedCohort,
    MatchedSeedExperimentRunner, ScenarioVariant, canonical_experiment_brain_family,
};
use scriptbots_core::{KNOB_RANGES, knob_range};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use utoipa::{PartialSchema, ToSchema};

/// Most factors one spec may vary.
pub const MAX_FACTORS: usize = 4;
/// Most values one factor may take.
pub const MAX_VALUES_PER_FACTOR: usize = 8;
/// Most ordered intervention schedules one spec may compare.
pub const MAX_INTERVENTION_SCHEDULES: usize = 8;
/// Most arms (configuration factors times families times schedules) one spec may expand to.
pub const MAX_ARMS: usize = 32;
/// Most predeclared conditional interaction contrasts in one report.
pub const MAX_INTERACTION_CONTRASTS: usize = 32;
/// Most seeds in a cohort.
pub const MAX_SEEDS: u16 = 64;
/// Longest run, in ticks.
pub const MAX_TICKS_PER_RUN: u64 = 1_000_000;
/// Longest free-text field.
pub const MAX_STRING_LEN: usize = 4_096;

/// Final-state endpoints emitted by the matched-seed runner and valid for analysis.
/// Population means and maxima are unavailable after extinction; choosing one
/// requires it to be observed in every matched run, without zero imputation.
pub const KNOWN_METRICS: &[&str] = &crate::experiment_runner::EXPERIMENT_ENDPOINT_METRICS;

/// The knob that may NEVER be a factor.
///
/// `rng_seed` is the MATCHED-SEED AXIS. A spec that sweeps it as a factor is
/// confounded by construction — the arms would differ in both the treatment and
/// the noise, and no amount of downstream statistics can separate them again.
pub const SEED_KNOB: &str = "rng_seed";

/// One knob, varied over several values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Factor {
    /// Dotted config path, as `list_knobs` reports it.
    pub knob_path: String,
    /// The values to try, in declaration order.
    #[schema(value_type = Vec<f64>)]
    pub values: Vec<serde_json::Value>,
}

/// The seed cohort.
///
/// The SAME cohort runs under every arm. That is what makes the comparison
/// paired: the arms differ in the treatment and in nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct SeedPlan {
    /// First seed.
    pub base: u64,
    /// How many seeds; they run `base..base + count`.
    pub count: u16,
}

/// The ceiling the run must fit inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct SpecBudget {
    /// Total runs allowed.
    pub runs: u32,
    /// Total ticks allowed across all runs.
    pub ticks: u64,
}

/// What the model proposes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ExperimentSpec {
    /// What it thinks is true.
    pub hypothesis: String,
    /// What result would REFUTE it.
    ///
    /// Required and non-empty. A hypothesis nobody can refute is not a
    /// hypothesis, and a lab that accepted one would be generating confident
    /// prose rather than science.
    pub falsifier: String,
    /// The knobs to vary. May be empty when comparing at least two brain families
    /// or intervention schedules.
    pub factors: Vec<Factor>,
    /// Registered brain families to compare, in declaration order. Defaults to
    /// the runner's MLP family. Each configuration assignment runs every family
    /// with the same seed cohort; aliases normalize before arm identity checks.
    /// The first configuration, family and schedule form the baseline for every other
    /// arm. Conditional two-axis interactions can be requested separately.
    #[serde(default = "default_brain_families")]
    pub brain_families: Vec<String>,
    /// Ordered timed treatments to compare. Defaults to one empty schedule.
    /// Include an empty first schedule for an untreated control. Every config
    /// assignment and family runs every schedule with the same seed cohort.
    /// Patches use nested live numeric config fields; ticks are completed-tick
    /// boundaries before the run ends, with same-tick document order preserved.
    #[serde(default = "default_intervention_schedules")]
    pub intervention_schedules: Vec<Vec<ScenarioInterventionV1>>,
    /// The matched seed cohort.
    pub seeds: SeedPlan,
    /// How long each run lasts.
    pub ticks_per_run: u64,
    /// Which final-state endpoints to compare. Every requested metric must be
    /// available in every completed run. Population means and maxima are
    /// unavailable after extinction and are never silently replaced by zero.
    pub metrics: Vec<String>,
    /// Also estimate every two-axis difference-in-differences against the first
    /// declared levels, with all other axes held at their first levels. Includes
    /// configuration, brain-family and schedule axes. These conditional contrasts
    /// are not effects averaged over the other factors. All baseline comparisons
    /// and interaction tests share one multiple-comparison correction family.
    #[serde(default, skip_serializing_if = "interaction_analysis_disabled")]
    pub include_pairwise_interactions: bool,
    /// The ceiling this spec must fit inside.
    pub budget: SpecBudget,
}

fn default_brain_families() -> Vec<String> {
    vec![DEFAULT_EXPERIMENT_BRAIN_FAMILY.to_owned()]
}

fn default_intervention_schedules() -> Vec<Vec<ScenarioInterventionV1>> {
    vec![Vec::new()]
}

const fn interaction_analysis_disabled(value: &bool) -> bool {
    !*value
}

/// One complete treatment assignment, including its canonical registered family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    /// Canonical family name resolved against the adapters compiled into this build.
    pub brain_family: String,
    /// Configuration factors, kept separate from the family selection.
    pub config_overrides: BTreeMap<String, serde_json::Value>,
    /// Exact ordered timed treatment, including an empty untreated schedule.
    pub interventions: Vec<ScenarioInterventionV1>,
}

/// An independently declared treatment axis, never an outcome-selected grouping.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InteractionAxis {
    ConfigFactor { knob_path: String },
    BrainFamily,
    InterventionSchedule,
}

/// Exact levels whose conditional treatment effects are compared.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionLevels {
    pub axis: InteractionAxis,
    pub baseline: serde_json::Value,
    pub treatment: serde_json::Value,
}

/// One predeclared four-cell contrast, conditional on the other axes' first levels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionContrast {
    pub first: InteractionLevels,
    pub second: InteractionLevels,
    /// Ordered cells [baseline, first only, second only, both], with coefficients
    /// [+1, -1, -1, +1]. The estimate is not divided by two or four.
    pub arm_ids: [u16; 4],
}

fn interaction_axis_value(arm: &Arm, axis: &InteractionAxis) -> Result<serde_json::Value, String> {
    match axis {
        InteractionAxis::ConfigFactor { knob_path } => arm
            .config_overrides
            .get(knob_path)
            .cloned()
            .ok_or_else(|| format!("interaction axis {knob_path:?} is absent from an arm")),
        InteractionAxis::BrainFamily => Ok(serde_json::json!(arm.brain_family)),
        InteractionAxis::InterventionSchedule => {
            serde_json::to_value(&arm.interventions).map_err(|error| error.to_string())
        }
    }
}

fn set_interaction_axis(
    arm: &mut Arm,
    axis: &InteractionAxis,
    value: &serde_json::Value,
) -> Result<(), String> {
    match axis {
        InteractionAxis::ConfigFactor { knob_path } => {
            let current = arm
                .config_overrides
                .get_mut(knob_path)
                .ok_or_else(|| format!("interaction axis {knob_path:?} is absent from an arm"))?;
            *current = value.clone();
        }
        InteractionAxis::BrainFamily => {
            arm.brain_family = value
                .as_str()
                .ok_or_else(|| "interaction family level must be a string".to_owned())?
                .to_owned();
        }
        InteractionAxis::InterventionSchedule => {
            arm.interventions =
                serde_json::from_value(value.clone()).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

/// Derive the complete interaction family from treatment metadata alone.
///
/// Validation and reanalysis use this same authority. Axes are ordered by config
/// path, then family and schedule; levels retain their first appearance in arm
/// order. Equal metadata within an arm and complete seed pairing are checked by
/// the statistics consumer before it calls this function. No outcome is read.
pub(crate) fn derive_interaction_contrasts(
    arms: &[Arm],
) -> Result<Vec<InteractionContrast>, String> {
    if arms.is_empty() || arms.len() > MAX_ARMS {
        return Err(format!("interaction design requires 1..={MAX_ARMS} arms"));
    }
    let baseline = &arms[0];
    for (index, arm) in arms.iter().enumerate() {
        if arm
            .config_overrides
            .keys()
            .ne(baseline.config_overrides.keys())
        {
            return Err("interaction arms must declare the same configuration axes".to_owned());
        }
        if arms[..index].contains(arm) {
            return Err("interaction design contains duplicate treatment assignments".to_owned());
        }
    }
    let axes = baseline
        .config_overrides
        .keys()
        .cloned()
        .map(|knob_path| InteractionAxis::ConfigFactor { knob_path })
        .chain([
            InteractionAxis::BrainFamily,
            InteractionAxis::InterventionSchedule,
        ]);
    let mut varying = Vec::new();
    let mut combinations = 1_usize;
    for axis in axes {
        let mut levels = Vec::new();
        for arm in arms {
            let value = interaction_axis_value(arm, &axis)?;
            if !levels.contains(&value) {
                levels.push(value);
            }
        }
        combinations = combinations
            .checked_mul(levels.len())
            .ok_or_else(|| "interaction design cardinality overflow".to_owned())?;
        if levels.len() > 1 {
            varying.push((axis, levels));
        }
    }
    if combinations != arms.len() {
        return Err(
            "interaction analysis requires the complete Cartesian treatment grid".to_owned(),
        );
    }
    if varying.len() < 2 {
        return Err("interaction analysis requires at least two varying treatment axes".to_owned());
    }
    let locate = |arm: &Arm| -> Result<u16, String> {
        let index = arms
            .iter()
            .position(|candidate| candidate == arm)
            .ok_or_else(|| "interaction design is missing an observed cell".to_owned())?;
        u16::try_from(index).map_err(|error| error.to_string())
    };
    let mut contrasts = Vec::new();
    for (first_index, (first_axis, first_levels)) in varying.iter().enumerate() {
        for (second_axis, second_levels) in varying.iter().skip(first_index + 1) {
            for first_value in first_levels.iter().skip(1) {
                for second_value in second_levels.iter().skip(1) {
                    if contrasts.len() == MAX_INTERACTION_CONTRASTS {
                        return Err(format!(
                            "interaction contrast count exceeds {MAX_INTERACTION_CONTRASTS}"
                        ));
                    }
                    let mut first = baseline.clone();
                    set_interaction_axis(&mut first, first_axis, first_value)?;
                    let mut second = baseline.clone();
                    set_interaction_axis(&mut second, second_axis, second_value)?;
                    let mut both = first.clone();
                    set_interaction_axis(&mut both, second_axis, second_value)?;
                    contrasts.push(InteractionContrast {
                        first: InteractionLevels {
                            axis: first_axis.clone(),
                            baseline: first_levels[0].clone(),
                            treatment: first_value.clone(),
                        },
                        second: InteractionLevels {
                            axis: second_axis.clone(),
                            baseline: second_levels[0].clone(),
                            treatment: second_value.clone(),
                        },
                        arm_ids: [0, locate(&first)?, locate(&second)?, locate(&both)?],
                    });
                }
            }
        }
    }
    Ok(contrasts)
}

/// A spec that has been checked and expanded.
///
/// Two validations of the same spec produce a byte-identical `ValidatedSpec`, and
/// its `spec_id` IS the experiment's identity — arms expand in declaration order,
/// seeds ascend, and no `HashMap` appears anywhere in the expansion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidatedSpec {
    /// The proposal with brain-family aliases resolved to canonical names.
    pub spec: ExperimentSpec,
    /// Every configuration assignment in factor declaration order, with families
    /// then schedules in declaration order inside each assignment. The first arm is the control
    /// for every treatment comparison. Requested interactions are derived from
    /// this same complete grid before any run executes.
    pub arms: Vec<Arm>,
    /// Complete predeclared conditional interaction family; empty when disabled.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interaction_contrasts: Vec<InteractionContrast>,
    /// The cohort, ascending.
    pub seeds: Vec<u64>,
    /// Whether an arm changes the launch family or touches a knob that cannot
    /// be changed on a live world.
    ///
    /// Such a spec may still run — the lab starts fresh worlds — but it may NEVER
    /// be applied to a user's running simulation.
    pub fresh_world_only: bool,
    /// Content hash of the validated spec: the experiment's identity.
    pub spec_id: String,
}

impl ValidatedSpec {
    /// What this experiment will actually cost.
    ///
    /// Computed BEFORE the first run starts, not discovered at run 19 of 40.
    #[must_use]
    pub fn cost(&self) -> RunCost {
        let runs = (self.arms.len() as u64).saturating_mul(self.seeds.len() as u64);
        RunCost {
            runs,
            ticks: runs.saturating_mul(self.spec.ticks_per_run),
        }
    }
}

/// The pre-flight cost of an experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunCost {
    /// Arms x seeds.
    pub runs: u64,
    /// Runs x ticks each.
    pub ticks: u64,
}

/// Everything that can be wrong with a proposal.
///
/// `Display` renders one ACTIONABLE line per error, naming the path, the
/// offending value, and the allowed range. These strings are not debug output —
/// they go back to the model verbatim as the repair prompt, so each one must
/// answer, in a single line, "what exactly do I change to make this valid?".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpecError {
    /// No scientific claim was supplied.
    EmptyHypothesis,
    /// No treatment axis was supplied.
    EmptyFactors,
    /// No brain family was selected.
    EmptyBrainFamilies,
    /// No schedule was selected, including no untreated schedule.
    EmptyInterventionSchedules,
    /// The canonical runner cannot execute the expanded treatment plan.
    InvalidExecutionPlan {
        /// Canonical runner's actionable preflight refusal.
        reason: String,
    },
    /// A requested family has no executable adapter in this build.
    InvalidBrainFamily {
        /// Requested name or alias.
        name: String,
        /// Canonical resolver's actionable refusal.
        reason: String,
    },
    /// Two requested names resolve to the same family.
    DuplicateBrainFamily {
        /// Canonical registered family.
        name: String,
        /// First position in the family axis.
        first: usize,
        /// Repeated position in the family axis.
        second: usize,
    },
    /// No output was selected for comparison.
    EmptyMetrics,
    /// No such knob.
    UnknownKnob {
        /// The path the model asked for.
        path: String,
    },
    /// The knob exists but may not be swept.
    NotSweepable {
        /// The path.
        path: String,
        /// Why not.
        reason: String,
    },
    /// Sweeping the seed would confound the experiment.
    SeedAsFactor,
    /// Outside the experiment range.
    OutOfRange {
        /// The path.
        path: String,
        /// What was asked for.
        value: f64,
        /// Lowest allowed.
        min: f64,
        /// Highest allowed.
        max: f64,
    },
    /// NaN or an infinity, which can only arrive as a string.
    NonFinite {
        /// The path.
        path: String,
        /// The raw token.
        raw: String,
    },
    /// Survives f64 but not the f32 the config actually stores.
    NotRepresentableAsF32 {
        /// The path.
        path: String,
        /// The value that would become `inf` or lose its meaning.
        value: f64,
    },
    /// Wrong kind of value entirely.
    TypeMismatch {
        /// The path.
        path: String,
        /// What the knob wanted.
        expected: String,
        /// What arrived.
        got: String,
    },
    /// The cartesian product is too big.
    TooManyArms {
        /// How many arms the factors expand to.
        arms: usize,
        /// The cap.
        max: usize,
    },
    /// Two arms are the same, so one of them is a wasted run out of a hard budget.
    DuplicateArm {
        /// First index.
        first: usize,
        /// The duplicate.
        second: usize,
    },
    /// Two factor declarations target the same knob, which would overwrite an
    /// earlier treatment during arm expansion.
    DuplicateFactor {
        /// Repeated knob path.
        path: String,
        /// First factor index.
        first: usize,
        /// Repeated factor index.
        second: usize,
    },
    /// The experiment does not fit the budget.
    CostExceedsBudget {
        /// Runs required.
        runs: u64,
        /// Ticks required.
        ticks: u64,
        /// Runs allowed.
        max_runs: u32,
        /// Ticks allowed.
        max_ticks: u64,
    },
    /// A hypothesis nobody can refute.
    EmptyFalsifier,
    /// No such metric.
    UnknownMetric {
        /// The name asked for.
        name: String,
    },
    /// A cohort of nobody.
    ZeroSeeds,
    /// The ascending matched-seed cohort would wrap around `u64::MAX`.
    SeedOverflow {
        /// First seed.
        base: u64,
        /// Requested cohort size.
        count: u16,
    },
    /// A run that never advances cannot test a dynamic hypothesis.
    ZeroTicks,
    /// A structural limit was exceeded.
    Bounds {
        /// Which field.
        field: String,
        /// What was asked for.
        got: usize,
        /// The cap.
        max: usize,
    },
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHypothesis => write!(
                f,
                "`hypothesis` is required and must be non-empty: state the claim this experiment tests."
            ),
            Self::EmptyFactors => write!(
                f,
                "`factors` must contain a treatment axis unless `brain_families` or `intervention_schedules` selects at least two distinct treatments."
            ),
            Self::EmptyInterventionSchedules => write!(
                f,
                "`intervention_schedules` must contain at least one schedule; use `[[]]` for no timed interventions."
            ),
            Self::InvalidExecutionPlan { reason } => {
                write!(f, "the expanded treatment plan is not executable: {reason}")
            }
            Self::EmptyBrainFamilies => write!(
                f,
                "`brain_families` must select at least one executable family; omit it to use the default MLP family."
            ),
            Self::InvalidBrainFamily { name, reason } => write!(
                f,
                "`brain_families` contains unsupported `{name}`: {reason}. Choose a family from the tool schema's installed-family list."
            ),
            Self::DuplicateBrainFamily {
                name,
                first,
                second,
            } => write!(
                f,
                "brain families {first} and {second} both resolve to `{name}`. Select each family once so aliases do not duplicate treatment arms."
            ),
            Self::EmptyMetrics => write!(
                f,
                "`metrics` must name at least one reported metric to compare."
            ),
            Self::UnknownKnob { path } => write!(
                f,
                "`{path}` is not a knob. Call list_knobs and use a path exactly as it appears there."
            ),
            Self::NotSweepable { path, reason } => {
                write!(f, "`{path}` cannot be swept: {reason}")
            }
            Self::SeedAsFactor => write!(
                f,
                "`{SEED_KNOB}` may never be a factor: it is the matched-seed axis. Every arm runs \
                 the SAME seed cohort, so that the arms differ in the treatment and in nothing \
                 else. Sweeping the seed would confound the treatment with the noise, and no \
                 downstream statistics could separate them again. Put the seeds in `seeds` instead."
            ),
            Self::OutOfRange {
                path,
                value,
                min,
                max,
            } => write!(
                f,
                "`{path}` = {value} is outside the experiment range [{min}, {max}]. Choose a value \
                 inside that interval."
            ),
            Self::NonFinite { path, raw } => write!(
                f,
                "`{path}` = `{raw}` is not a finite number. NaN and infinities are not values a \
                 simulation can run; give a finite decimal."
            ),
            Self::NotRepresentableAsF32 { path, value } => write!(
                f,
                "`{path}` = {value} does not survive the conversion to the 32-bit float the config \
                 stores — it would silently become infinity. Use a value of smaller magnitude."
            ),
            Self::TypeMismatch {
                path,
                expected,
                got,
            } => write!(f, "`{path}` expects {expected}, but got {got}."),
            Self::TooManyArms { arms, max } => write!(
                f,
                "configuration factors, brain families and intervention schedules expand to {arms} arms, over the limit of {max}. \
                 Use fewer families, schedules, factors, or values per factor."
            ),
            Self::DuplicateArm { first, second } => write!(
                f,
                "arms {first} and {second} are identical, so one of them is a run spent learning \
                 nothing. Remove the duplicate value."
            ),
            Self::DuplicateFactor {
                path,
                first,
                second,
            } => write!(
                f,
                "factors {first} and {second} both target `{path}`. Combine their values into one \
                 factor so no treatment is silently overwritten."
            ),
            Self::CostExceedsBudget {
                runs,
                ticks,
                max_runs,
                max_ticks,
            } => write!(
                f,
                "this experiment needs {runs} runs and {ticks} ticks, over the budget of \
                 {max_runs} runs / {max_ticks} ticks. Reduce the seed count, the number of arms, \
                 or ticks_per_run."
            ),
            Self::EmptyFalsifier => write!(
                f,
                "`falsifier` is required and must be non-empty: state what result would REFUTE \
                 the hypothesis. A hypothesis nothing could refute is not a hypothesis."
            ),
            Self::UnknownMetric { name } => write!(
                f,
                "`{name}` is not a metric this run reports. Name a column the summary actually has."
            ),
            Self::ZeroSeeds => write!(
                f,
                "`seeds.count` must be at least 1: an experiment needs a cohort to run."
            ),
            Self::SeedOverflow { base, count } => write!(
                f,
                "`seeds` beginning at {base} with count {count} would wrap past u64::MAX. \
                 Choose a smaller base or cohort."
            ),
            Self::ZeroTicks => write!(
                f,
                "`ticks_per_run` must be at least 1: a zero-tick world cannot test a dynamic hypothesis."
            ),
            Self::Bounds { field, got, max } => {
                write!(f, "`{field}` has {got} entries, over the limit of {max}.")
            }
        }
    }
}

impl SpecError {
    /// Stable machine-readable code used in redacted lab audit events.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::EmptyHypothesis => "empty_hypothesis",
            Self::EmptyFactors => "empty_factors",
            Self::EmptyBrainFamilies => "empty_brain_families",
            Self::EmptyInterventionSchedules => "empty_intervention_schedules",
            Self::InvalidExecutionPlan { .. } => "invalid_execution_plan",
            Self::InvalidBrainFamily { .. } => "invalid_brain_family",
            Self::DuplicateBrainFamily { .. } => "duplicate_brain_family",
            Self::EmptyMetrics => "empty_metrics",
            Self::UnknownKnob { .. } => "unknown_knob",
            Self::NotSweepable { .. } => "not_sweepable",
            Self::SeedAsFactor => "seed_as_factor",
            Self::OutOfRange { .. } => "out_of_range",
            Self::NonFinite { .. } => "non_finite",
            Self::NotRepresentableAsF32 { .. } => "not_representable_as_f32",
            Self::TypeMismatch { .. } => "type_mismatch",
            Self::TooManyArms { .. } => "too_many_arms",
            Self::DuplicateArm { .. } => "duplicate_arm",
            Self::DuplicateFactor { .. } => "duplicate_factor",
            Self::CostExceedsBudget { .. } => "cost_exceeds_budget",
            Self::EmptyFalsifier => "empty_falsifier",
            Self::UnknownMetric { .. } => "unknown_metric",
            Self::ZeroSeeds => "zero_seeds",
            Self::SeedOverflow { .. } => "seed_overflow",
            Self::ZeroTicks => "zero_ticks",
            Self::Bounds { .. } => "bounds",
        }
    }
}

/// Render a whole error list as the repair prompt the model sees.
#[must_use]
pub fn render_errors(errors: &[SpecError]) -> String {
    errors
        .iter()
        .map(|error| format!("- {error}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the standalone tool-input schema from the canonical Rust type.
///
/// `utoipa` emits OpenAPI component references for nested structs. Keeping the
/// collected components beside the root makes those JSON pointers resolvable
/// when the schema is sent outside an OpenAPI document. The range and metric
/// registries are attached as deterministic vendor extensions; the validator
/// below remains authoritative for their dynamic constraints.
///
/// # Errors
///
/// Returns a serialization error if `utoipa` ever emits a schema value that
/// `serde_json` cannot encode.
pub fn tool_input_schema() -> Result<serde_json::Value, serde_json::Error> {
    let mut root = serde_json::to_value(<ExperimentSpec as PartialSchema>::schema())?;
    if let Some(items) = root
        .pointer_mut("/properties/metrics/items")
        .and_then(serde_json::Value::as_object_mut)
    {
        items.insert("enum".to_owned(), serde_json::json!(KNOWN_METRICS));
    }
    if let Some(families) = root
        .pointer_mut("/properties/brain_families")
        .and_then(serde_json::Value::as_object_mut)
    {
        families.insert(
            "items".to_owned(),
            serde_json::json!({"type": "string", "enum": EXPERIMENT_BRAIN_FAMILIES}),
        );
        families.insert("minItems".to_owned(), serde_json::json!(1));
        families.insert(
            "maxItems".to_owned(),
            serde_json::json!(EXPERIMENT_BRAIN_FAMILIES.len()),
        );
        families.insert("uniqueItems".to_owned(), serde_json::json!(true));
        families.insert(
            "default".to_owned(),
            serde_json::json!(default_brain_families()),
        );
    }
    if let Some(schedules) = root
        .pointer_mut("/properties/intervention_schedules")
        .and_then(serde_json::Value::as_object_mut)
    {
        schedules.insert("minItems".to_owned(), serde_json::json!(1));
        schedules.insert(
            "maxItems".to_owned(),
            serde_json::json!(MAX_INTERVENTION_SCHEDULES),
        );
        schedules.insert("uniqueItems".to_owned(), serde_json::json!(true));
        schedules.insert(
            "default".to_owned(),
            serde_json::json!(default_intervention_schedules()),
        );
        if let Some(schedule) = schedules
            .get_mut("items")
            .and_then(serde_json::Value::as_object_mut)
        {
            schedule.insert(
                "maxItems".to_owned(),
                serde_json::json!(MAX_EXPERIMENT_INTERVENTIONS),
            );
        }
    }
    let mut referenced = Vec::new();
    <ExperimentSpec as ToSchema>::schemas(&mut referenced);
    let schemas = referenced
        .into_iter()
        .map(|(name, schema)| {
            let mut schema = serde_json::to_value(schema)?;
            if matches!(
                name.as_str(),
                "Factor" | "SeedPlan" | "SpecBudget" | "ScenarioInterventionV1"
            ) && let Some(object) = schema.as_object_mut()
            {
                object.insert(
                    "additionalProperties".to_owned(),
                    serde_json::Value::Bool(false),
                );
            }
            Ok::<_, serde_json::Error>((name, schema))
        })
        .collect::<Result<serde_json::Map<String, serde_json::Value>, _>>()?;
    let ranges = KNOB_RANGES
        .iter()
        .map(|range| {
            serde_json::json!({
                "path": range.path,
                "min": range.min,
                "max": range.max,
                "fresh_world_only": range.fresh_world_only,
                "lab_sweepable": not_sweepable_reason(range.path).is_none(),
            })
        })
        .collect::<Vec<_>>();

    if let Some(object) = root.as_object_mut() {
        object.insert(
            "$schema".to_owned(),
            serde_json::Value::String("https://json-schema.org/draft/2020-12/schema".to_owned()),
        );
        object.insert(
            "additionalProperties".to_owned(),
            serde_json::Value::Bool(false),
        );
        object.insert(
            "components".to_owned(),
            serde_json::json!({"schemas": schemas}),
        );
        object.insert(
            "x-scriptbots-knob-ranges".to_owned(),
            serde_json::Value::Array(ranges),
        );
        object.insert(
            "x-scriptbots-known-metrics".to_owned(),
            serde_json::json!(KNOWN_METRICS),
        );
        object.insert(
            "x-scriptbots-brain-families".to_owned(),
            serde_json::json!(EXPERIMENT_BRAIN_FAMILIES),
        );
        object.insert(
            "x-scriptbots-validation-limits".to_owned(),
            serde_json::json!({
                "max_factors": MAX_FACTORS,
                "max_values_per_factor": MAX_VALUES_PER_FACTOR,
                "max_brain_families": EXPERIMENT_BRAIN_FAMILIES.len(),
                "max_intervention_schedules": MAX_INTERVENTION_SCHEDULES,
                "max_interventions_per_schedule": MAX_EXPERIMENT_INTERVENTIONS,
                "max_intervention_patch_bytes": MAX_EXPERIMENT_INTERVENTION_PATCH_BYTES,
                "max_arms": MAX_ARMS,
                "max_interaction_contrasts": MAX_INTERACTION_CONTRASTS,
                "max_seeds": MAX_SEEDS,
                "max_ticks_per_run": MAX_TICKS_PER_RUN,
                "max_string_length": MAX_STRING_LEN,
            }),
        );
        object.insert(
            "x-scriptbots-seed-policy".to_owned(),
            serde_json::json!({
                "factor_path": SEED_KNOB,
                "factor_sweepable": false,
                "allocation": "ascending_checked_range",
                "matching": "every arm receives the identical cohort",
            }),
        );
    }
    Ok(root)
}

/// Check and expand a proposal.
///
/// Returns ALL errors, not the first. The whole list goes back to the model as
/// one repair prompt: first-error-only doubles the number of turns and is the
/// most reliable way to burn a token budget on a single malformed proposal.
///
/// # Errors
///
/// Every way the spec can be wrong, as a list.
pub fn validate_spec(
    spec: &ExperimentSpec,
    operator_budget: SpecBudget,
) -> Result<ValidatedSpec, Vec<SpecError>> {
    let mut errors = Vec::new();

    if spec.hypothesis.trim().is_empty() {
        errors.push(SpecError::EmptyHypothesis);
    }
    if spec.falsifier.trim().is_empty() {
        errors.push(SpecError::EmptyFalsifier);
    }
    let families = resolve_brain_families(&spec.brain_families, &mut errors);
    if spec.factors.is_empty() && families.len() < 2 && spec.intervention_schedules.len() < 2 {
        errors.push(SpecError::EmptyFactors);
    }
    if spec.intervention_schedules.is_empty() {
        errors.push(SpecError::EmptyInterventionSchedules);
    }
    check_len(
        "intervention_schedules",
        spec.intervention_schedules.len(),
        MAX_INTERVENTION_SCHEDULES,
        &mut errors,
    );
    // Bound untrusted schedules before cloning them across the treatment product.
    // The runner's pure preflight below remains the authority for their meaning,
    // cumulative config validity, execution order and completion horizon.
    let mut schedules_bounded = !spec.intervention_schedules.is_empty()
        && spec.intervention_schedules.len() <= MAX_INTERVENTION_SCHEDULES;
    for (index, schedule) in spec
        .intervention_schedules
        .iter()
        .take(MAX_INTERVENTION_SCHEDULES)
        .enumerate()
    {
        check_len(
            &format!("intervention_schedules[{index}]"),
            schedule.len(),
            MAX_EXPERIMENT_INTERVENTIONS,
            &mut errors,
        );
        schedules_bounded &= schedule.len() <= MAX_EXPERIMENT_INTERVENTIONS;
        for (statement, intervention) in schedule
            .iter()
            .take(MAX_EXPERIMENT_INTERVENTIONS)
            .enumerate()
        {
            let bytes =
                serde_json::to_vec(&intervention.set).map_or(usize::MAX, |encoded| encoded.len());
            check_len(
                &format!("intervention_schedules[{index}][{statement}].set"),
                bytes,
                MAX_EXPERIMENT_INTERVENTION_PATCH_BYTES,
                &mut errors,
            );
            schedules_bounded &= bytes <= MAX_EXPERIMENT_INTERVENTION_PATCH_BYTES;
        }
    }
    if spec.metrics.is_empty() {
        errors.push(SpecError::EmptyMetrics);
    }
    check_len(
        "hypothesis",
        spec.hypothesis.len(),
        MAX_STRING_LEN,
        &mut errors,
    );
    check_len(
        "falsifier",
        spec.falsifier.len(),
        MAX_STRING_LEN,
        &mut errors,
    );
    check_len("factors", spec.factors.len(), MAX_FACTORS, &mut errors);
    check_len("metrics", spec.metrics.len(), MAX_FACTORS, &mut errors);

    if spec.seeds.count == 0 {
        errors.push(SpecError::ZeroSeeds);
    }
    if spec.ticks_per_run == 0 {
        errors.push(SpecError::ZeroTicks);
    }
    check_len(
        "seeds.count",
        spec.seeds.count as usize,
        MAX_SEEDS as usize,
        &mut errors,
    );
    check_len(
        "ticks_per_run",
        usize::try_from(spec.ticks_per_run).unwrap_or(usize::MAX),
        usize::try_from(MAX_TICKS_PER_RUN).unwrap_or(usize::MAX),
        &mut errors,
    );

    for metric in &spec.metrics {
        if !KNOWN_METRICS.contains(&metric.as_str()) {
            errors.push(SpecError::UnknownMetric {
                name: metric.clone(),
            });
        }
    }

    let mut fresh_world_only = families
        .iter()
        .any(|family| *family != DEFAULT_EXPERIMENT_BRAIN_FAMILY);
    let mut factor_positions = BTreeMap::new();
    for (factor_index, factor) in spec.factors.iter().enumerate() {
        if let Some(first) = factor_positions.get(&factor.knob_path).copied() {
            errors.push(SpecError::DuplicateFactor {
                path: factor.knob_path.clone(),
                first,
                second: factor_index,
            });
        } else {
            factor_positions.insert(factor.knob_path.clone(), factor_index);
        }
        check_len(
            &format!("factors[{}].values", factor.knob_path),
            factor.values.len(),
            MAX_VALUES_PER_FACTOR,
            &mut errors,
        );
        if factor.values.is_empty() {
            errors.push(SpecError::Bounds {
                field: format!("factors[{}].values", factor.knob_path),
                got: 0,
                max: MAX_VALUES_PER_FACTOR,
            });
        }
        if factor.knob_path == SEED_KNOB {
            errors.push(SpecError::SeedAsFactor);
            continue;
        }
        let Some(range) = knob_range(&factor.knob_path) else {
            errors.push(SpecError::UnknownKnob {
                path: factor.knob_path.clone(),
            });
            continue;
        };
        if let Some(reason) = not_sweepable_reason(&factor.knob_path) {
            errors.push(SpecError::NotSweepable {
                path: factor.knob_path.clone(),
                reason: reason.to_owned(),
            });
            continue;
        }
        if range.fresh_world_only {
            fresh_world_only = true;
        }
        for value in &factor.values {
            check_value(&factor.knob_path, value, range, &mut errors);
        }
    }

    let arm_count = spec
        .factors
        .iter()
        .try_fold(1_usize, |count, factor| {
            count.checked_mul(factor.values.len())
        })
        .and_then(|count| count.checked_mul(spec.brain_families.len()))
        .and_then(|count| count.checked_mul(spec.intervention_schedules.len()))
        .unwrap_or(usize::MAX);
    if arm_count > MAX_ARMS {
        errors.push(SpecError::TooManyArms {
            arms: arm_count,
            max: MAX_ARMS,
        });
    }
    let can_expand =
        (!spec.factors.is_empty() || families.len() >= 2 || spec.intervention_schedules.len() >= 2)
            && !families.is_empty()
            && families.len() == spec.brain_families.len()
            && families.len() <= EXPERIMENT_BRAIN_FAMILIES.len()
            && spec.factors.len() <= MAX_FACTORS
            && factor_positions.len() == spec.factors.len()
            && spec.factors.iter().all(|factor| {
                !factor.values.is_empty() && factor.values.len() <= MAX_VALUES_PER_FACTOR
            })
            && schedules_bounded
            && arm_count <= MAX_ARMS;
    let arms = if can_expand {
        expand(&spec.factors, &families, &spec.intervention_schedules)
    } else {
        Vec::new()
    };
    for (i, arm) in arms.iter().enumerate() {
        if let Some(j) = arms[..i].iter().position(|other| other == arm) {
            errors.push(SpecError::DuplicateArm {
                first: j,
                second: i,
            });
        }
    }

    let seeds = if spec.seeds.count == 0 {
        Vec::new()
    } else if spec
        .seeds
        .base
        .checked_add(u64::from(spec.seeds.count - 1))
        .is_none()
    {
        errors.push(SpecError::SeedOverflow {
            base: spec.seeds.base,
            count: spec.seeds.count,
        });
        Vec::new()
    } else {
        (0..spec.seeds.count)
            .map(|offset| spec.seeds.base + u64::from(offset))
            .collect()
    };

    let runs = u64::try_from(arm_count)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::from(spec.seeds.count));
    let ticks = runs.saturating_mul(spec.ticks_per_run);
    let effective_budget = SpecBudget {
        runs: spec.budget.runs.min(operator_budget.runs),
        ticks: spec.budget.ticks.min(operator_budget.ticks),
    };
    if runs > u64::from(effective_budget.runs) || ticks > effective_budget.ticks {
        errors.push(SpecError::CostExceedsBudget {
            runs,
            ticks,
            max_runs: effective_budget.runs,
            max_ticks: effective_budget.ticks,
        });
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // plan_batch constructs neither a world nor an artifact. Reuse the executable
    // runner's canonical checks rather than maintain a second schedule validator.
    // One actual matched seed is sufficient: configs and schedules are identical
    // across the cohort, while its seed is owned exclusively by the runner.
    let variants = arms
        .iter()
        .enumerate()
        .map(|(index, arm)| ScenarioVariant {
            variant_id: format!("arm-{index:03}"),
            brain_family: arm.brain_family.clone(),
            config_overrides: arm.config_overrides.clone(),
            interventions: arm.interventions.clone(),
        })
        .collect();
    MatchedSeedExperimentRunner::new(
        "lab-preflight",
        MatchedSeedCohort {
            cohort_id: "lab-preflight-matched-seeds".to_owned(),
            seeds: vec![seeds[0]],
        },
        variants,
        spec.ticks_per_run,
        2,
        std::path::PathBuf::new(),
    )
    .plan_batch()
    .map_err(|error| {
        vec![SpecError::InvalidExecutionPlan {
            reason: error.to_string(),
        }]
    })?;

    let mut canonical_spec = spec.clone();
    canonical_spec.brain_families = families.into_iter().map(str::to_owned).collect();
    let interaction_contrasts = if canonical_spec.include_pairwise_interactions {
        derive_interaction_contrasts(&arms)
            .map_err(|reason| vec![SpecError::InvalidExecutionPlan { reason }])?
    } else {
        Vec::new()
    };
    let spec_id = content_hash(&canonical_spec, &arms, &seeds);
    Ok(ValidatedSpec {
        spec: canonical_spec,
        arms,
        interaction_contrasts,
        seeds,
        fresh_world_only,
        spec_id,
    })
}

fn resolve_brain_families(requested: &[String], errors: &mut Vec<SpecError>) -> Vec<&'static str> {
    if requested.is_empty() {
        errors.push(SpecError::EmptyBrainFamilies);
    }
    check_len(
        "brain_families",
        requested.len(),
        EXPERIMENT_BRAIN_FAMILIES.len(),
        errors,
    );
    let mut families = Vec::with_capacity(requested.len().min(EXPERIMENT_BRAIN_FAMILIES.len()));
    let mut positions = BTreeMap::new();
    for (index, name) in requested.iter().enumerate() {
        if name.len() > MAX_STRING_LEN {
            check_len(
                &format!("brain_families[{index}]"),
                name.len(),
                MAX_STRING_LEN,
                errors,
            );
            continue;
        }
        match canonical_experiment_brain_family("lab-arm", name) {
            Ok(family) => {
                if let Some(&first) = positions.get(family) {
                    errors.push(SpecError::DuplicateBrainFamily {
                        name: family.to_owned(),
                        first,
                        second: index,
                    });
                } else {
                    positions.insert(family, index);
                    families.push(family);
                }
            }
            Err(error) => errors.push(SpecError::InvalidBrainFamily {
                name: name.clone(),
                reason: error.to_string(),
            }),
        }
    }
    families
}

fn check_len(field: &str, got: usize, max: usize, errors: &mut Vec<SpecError>) {
    if got > max {
        errors.push(SpecError::Bounds {
            field: field.to_owned(),
            got,
            max,
        });
    }
}

fn not_sweepable_reason(path: &str) -> Option<&'static str> {
    match path {
        "mutation.primary" | "mutation.secondary" => Some(
            "mutation rates are per-agent runtime state, not a fresh-world ScriptBotsConfig field",
        ),
        path if path.starts_with("render.") => {
            Some("render settings are presentation-only and cannot change a scientific run metric")
        }
        _ => None,
    }
}

fn integer_knob(path: &str) -> bool {
    matches!(
        path,
        "world_width"
            | "world_height"
            | "food_cell_size"
            | "population_minimum"
            | "population_spawn_interval"
            | "render.day_night.cycle_ticks"
    )
}

fn check_value(
    path: &str,
    value: &serde_json::Value,
    range: &scriptbots_core::KnobRange,
    errors: &mut Vec<SpecError>,
) {
    // A non-finite value cannot be a JSON number — serde_json::Number refuses to
    // hold NaN or Inf — so it can only arrive as a STRING. Reject it HERE, at the
    // spec boundary, rather than leaning on a downstream coercion check: a NaN
    // mutation rate that slipped through would poison the world and the notebook
    // would then report statistics over garbage.
    let number = match value {
        serde_json::Value::Number(number) => {
            if integer_knob(path) && number.as_u64().is_none() {
                errors.push(SpecError::TypeMismatch {
                    path: path.to_owned(),
                    expected: "a non-negative integer".to_owned(),
                    got: "a fractional number".to_owned(),
                });
                return;
            }
            number.as_f64()
        }
        serde_json::Value::String(raw) => {
            errors.push(SpecError::NonFinite {
                path: path.to_owned(),
                raw: raw.clone(),
            });
            return;
        }
        serde_json::Value::Bool(_) => {
            errors.push(SpecError::TypeMismatch {
                path: path.to_owned(),
                expected: "a number".to_owned(),
                got: "a boolean".to_owned(),
            });
            return;
        }
        other => {
            errors.push(SpecError::TypeMismatch {
                path: path.to_owned(),
                expected: "a number".to_owned(),
                got: describe(other),
            });
            return;
        }
    };

    let Some(number) = number else {
        errors.push(SpecError::NonFinite {
            path: path.to_owned(),
            raw: value.to_string(),
        });
        return;
    };
    if !number.is_finite() {
        errors.push(SpecError::NonFinite {
            path: path.to_owned(),
            raw: value.to_string(),
        });
        return;
    }

    // The config stores f32. A value that survives f64 but becomes `inf` on the
    // way in would be a silent corruption, so it is rejected by name.
    if !(number as f32).is_finite() {
        errors.push(SpecError::NotRepresentableAsF32 {
            path: path.to_owned(),
            value: number,
        });
        return;
    }

    // THE check this whole bead exists for. `food_growth_rate = 1e9` passes
    // ScriptBotsConfig::validate (which only asks for finite and non-negative)
    // and must still be refused: it would burn the run budget on a degenerate
    // world where every cell is saturated, and the notebook would report on it
    // with a straight face.
    if number < range.min || number > range.max {
        errors.push(SpecError::OutOfRange {
            path: path.to_owned(),
            value: number,
            min: range.min,
            max: range.max,
        });
    }
}

fn describe(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
    .to_owned()
}

/// The cartesian product of the configuration factors, with the family axis
/// inside each configuration assignment. The first arm is the common baseline.
///
/// `BTreeMap` and `Vec` only: a `HashMap` anywhere in here would make the arm
/// order — and therefore the spec id — depend on the hasher's seed.
fn expand(
    factors: &[Factor],
    families: &[&str],
    schedules: &[Vec<ScenarioInterventionV1>],
) -> Vec<Arm> {
    let mut assignments = vec![BTreeMap::new()];
    for factor in factors {
        let mut next = Vec::with_capacity(assignments.len() * factor.values.len().max(1));
        for arm in &assignments {
            for value in &factor.values {
                let mut extended = arm.clone();
                extended.insert(factor.knob_path.clone(), value.clone());
                next.push(extended);
            }
        }
        if !next.is_empty() {
            assignments = next;
        }
    }
    let mut arms = Vec::with_capacity(assignments.len() * families.len() * schedules.len());
    for config_overrides in assignments {
        for family in families {
            for interventions in schedules {
                arms.push(Arm {
                    brain_family: (*family).to_owned(),
                    config_overrides: config_overrides.clone(),
                    interventions: interventions.clone(),
                });
            }
        }
    }
    arms
}

/// FNV-1a over the canonical JSON encoding of the validated content.
///
/// Deviation from the brief, recorded rather than smuggled: the brief specified a
/// 32-byte hash, but the workspace has no cryptographic hash dependency and this
/// id is a DEDUPE KEY, not a security boundary — nobody is trying to forge a
/// colliding experiment. Reusing the same FNV-1a the characterization digest uses
/// keeps the dependency surface flat.
fn content_hash(spec: &ExperimentSpec, arms: &[Arm], seeds: &[u64]) -> String {
    let canonical = crate::precedence::canonical_value(&serde_json::json!({
        "hypothesis": spec.hypothesis,
        "falsifier": spec.falsifier,
        "ticks_per_run": spec.ticks_per_run,
        "metrics": spec.metrics,
        "arms": arms,
        "seeds": seeds,
    }));
    let encoded = serde_json::to_vec(&canonical).unwrap_or_default();
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

    const OPERATOR_BUDGET: SpecBudget = SpecBudget {
        runs: u32::MAX,
        ticks: u64::MAX,
    };

    fn spec() -> ExperimentSpec {
        ExperimentSpec {
            hypothesis: "faster food regrowth raises carrying capacity".to_owned(),
            falsifier: "population mean does not rise with growth rate".to_owned(),
            factors: vec![Factor {
                knob_path: "food_growth_rate".to_owned(),
                values: vec![
                    serde_json::json!(0.01),
                    serde_json::json!(0.05),
                    serde_json::json!(0.1),
                ],
            }],
            brain_families: default_brain_families(),
            intervention_schedules: default_intervention_schedules(),
            seeds: SeedPlan { base: 1, count: 4 },
            ticks_per_run: 2_000,
            metrics: vec!["alive_agents".to_owned()],
            include_pairwise_interactions: false,
            budget: SpecBudget {
                runs: 64,
                ticks: 10_000_000,
            },
        }
    }

    #[test]
    fn the_canonical_proposal_is_accepted_and_expands_deterministically() {
        let validated = validate_spec(&spec(), OPERATOR_BUDGET).expect("canonical spec is valid");
        assert_eq!(validated.arms.len(), 3);
        assert_eq!(validated.seeds, vec![1, 2, 3, 4]);
        assert_eq!(
            validated.cost(),
            RunCost {
                runs: 12,
                ticks: 24_000
            }
        );
        // Same spec, same identity — this id is what dedupes experiments.
        let again = validate_spec(&spec(), OPERATOR_BUDGET).expect("valid");
        assert_eq!(validated.spec_id, again.spec_id);
        assert_eq!(validated.arms, again.arms);
    }

    #[test]
    fn brain_family_axis_expands_after_config_factors_and_charges_every_run() {
        let mut proposed = spec();
        proposed.brain_families = vec!["mlp".to_owned(), "dwraon".to_owned()];
        let validated = validate_spec(&proposed, OPERATOR_BUDGET).unwrap();
        let expected = [
            ("mlp.baseline", 0.01),
            ("dwraon.baseline", 0.01),
            ("mlp.baseline", 0.05),
            ("dwraon.baseline", 0.05),
            ("mlp.baseline", 0.1),
            ("dwraon.baseline", 0.1),
        ];
        assert_eq!(validated.arms.len(), expected.len());
        for (arm, (family, growth)) in validated.arms.iter().zip(expected) {
            assert_eq!(arm.brain_family, family);
            assert_eq!(
                arm.config_overrides["food_growth_rate"],
                serde_json::json!(growth)
            );
        }
        assert!(validated.fresh_world_only);
        assert_eq!(
            validated.cost(),
            RunCost {
                runs: 24,
                ticks: 48_000,
            }
        );
        let errors = validate_spec(
            &proposed,
            SpecBudget {
                runs: 12,
                ticks: 48_000,
            },
        )
        .expect_err("family axis must count toward operator admission");
        assert!(errors.iter().any(|error| matches!(
            error,
            SpecError::CostExceedsBudget {
                runs: 24,
                ticks: 48_000,
                max_runs: 12,
                ..
            }
        )));

        proposed.factors.push(Factor {
            knob_path: "food_max".to_owned(),
            values: (1..=8)
                .map(|value| serde_json::json!(f64::from(value) * 0.1))
                .collect(),
        });
        let errors = validate_spec(&proposed, OPERATOR_BUDGET)
            .expect_err("3 configurations times 8 levels times 2 families exceeds the arm cap");
        assert!(errors.iter().any(|error| matches!(
            error,
            SpecError::TooManyArms {
                arms: 48,
                max: MAX_ARMS,
            }
        )));
    }

    #[test]
    fn interaction_opt_in_preserves_default_identity_and_predeclares_every_level() {
        let original = spec();
        let encoded = serde_json::to_value(&original).unwrap();
        assert!(encoded.get("include_pairwise_interactions").is_none());
        let mut explicit_off = encoded;
        explicit_off["include_pairwise_interactions"] = serde_json::json!(false);
        let explicit_off: ExperimentSpec = serde_json::from_value(explicit_off).unwrap();
        let baseline = validate_spec(&original, OPERATOR_BUDGET).unwrap();
        assert_eq!(
            baseline,
            validate_spec(&explicit_off, OPERATOR_BUDGET).unwrap()
        );
        assert!(baseline.interaction_contrasts.is_empty());

        let mut enabled = original;
        enabled.include_pairwise_interactions = true;
        let errors = validate_spec(&enabled, OPERATOR_BUDGET).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            SpecError::InvalidExecutionPlan { reason }
                if reason.contains("at least two varying")
        )));
        enabled.brain_families = vec!["mlp".to_owned(), "dwraon".to_owned()];
        let declared = validate_spec(&enabled, OPERATOR_BUDGET).unwrap();
        assert_eq!(declared.arms.len(), 6);
        assert_eq!(declared.interaction_contrasts.len(), 2);
        assert_eq!(declared.interaction_contrasts[0].arm_ids, [0, 2, 1, 3]);
        assert_eq!(declared.interaction_contrasts[1].arm_ids, [0, 4, 1, 5]);
        assert_eq!(
            declared.interaction_contrasts[0].second.baseline,
            serde_json::json!("mlp.baseline")
        );
        assert_eq!(
            declared.interaction_contrasts[0].second.treatment,
            serde_json::json!("dwraon.baseline")
        );
        enabled.include_pairwise_interactions = false;
        let without_interactions = validate_spec(&enabled, OPERATOR_BUDGET).unwrap();
        assert_eq!(declared.arms, without_interactions.arms);
        assert_eq!(declared.cost(), without_interactions.cost());
        assert_ne!(declared.spec_id, without_interactions.spec_id);
    }

    #[test]
    fn conditional_interactions_hold_other_family_and_schedule_axes_at_baseline() {
        let mut proposed = spec();
        proposed.factors[0].values.truncate(2);
        proposed.brain_families = vec!["mlp".to_owned(), "dwraon".to_owned()];
        proposed
            .intervention_schedules
            .push(vec![ScenarioInterventionV1 {
                tick: 1,
                set: serde_json::json!({"food_decay_rate": 0.001}),
            }]);
        proposed.include_pairwise_interactions = true;
        let validated = validate_spec(&proposed, OPERATOR_BUDGET).unwrap();
        assert_eq!(validated.arms.len(), 8);
        assert_eq!(validated.interaction_contrasts.len(), 3);
        let cells = validated
            .interaction_contrasts
            .iter()
            .map(|contrast| contrast.arm_ids)
            .collect::<Vec<_>>();
        assert_eq!(cells, [[0, 4, 2, 6], [0, 4, 1, 5], [0, 2, 1, 3]]);
        let schedule = &validated.interaction_contrasts[2].second;
        assert_eq!(schedule.axis, InteractionAxis::InterventionSchedule);
        assert_eq!(schedule.baseline, serde_json::json!([]));
        assert_eq!(
            schedule.treatment,
            serde_json::to_value(&proposed.intervention_schedules[1]).unwrap()
        );
        let mut missing = validated.arms.clone();
        missing.pop();
        assert!(
            derive_interaction_contrasts(&missing)
                .unwrap_err()
                .contains("complete Cartesian")
        );
        let mut duplicated = validated.arms.clone();
        duplicated[7] = duplicated[0].clone();
        assert!(
            derive_interaction_contrasts(&duplicated)
                .unwrap_err()
                .contains("duplicate")
        );
    }

    #[test]
    fn interaction_family_remains_bounded_at_the_largest_two_axis_grid() {
        let mut proposed = spec();
        proposed.factors = vec![
            Factor {
                knob_path: "food_growth_rate".to_owned(),
                values: (1..=8)
                    .map(|level| serde_json::json!(f64::from(level) * 0.01))
                    .collect(),
            },
            Factor {
                knob_path: "food_decay_rate".to_owned(),
                values: (0..4)
                    .map(|level| serde_json::json!(f64::from(level) * 0.01))
                    .collect(),
            },
        ];
        proposed.seeds.count = 1;
        proposed.include_pairwise_interactions = true;
        let validated = validate_spec(&proposed, OPERATOR_BUDGET).unwrap();
        assert_eq!(validated.arms.len(), MAX_ARMS);
        assert_eq!(validated.interaction_contrasts.len(), 7 * 3);
        assert!(validated.interaction_contrasts.len() <= MAX_INTERACTION_CONTRASTS);
    }

    #[test]
    fn family_only_studies_require_distinct_executable_families() {
        let mut proposed = spec();
        proposed.factors.clear();
        assert!(
            validate_spec(&proposed, OPERATOR_BUDGET)
                .unwrap_err()
                .contains(&SpecError::EmptyFactors)
        );

        proposed.brain_families = vec!["mlp".to_owned(), "dwraon.baseline".to_owned()];
        let validated = validate_spec(&proposed, OPERATOR_BUDGET).unwrap();
        assert_eq!(validated.arms.len(), 2);
        assert!(
            validated
                .arms
                .iter()
                .all(|arm| arm.config_overrides.is_empty())
        );
        assert_eq!(
            validated.cost(),
            RunCost {
                runs: 8,
                ticks: 16_000,
            }
        );

        proposed.brain_families = vec!["mlp".to_owned(), "mlp.baseline".to_owned()];
        let errors = validate_spec(&proposed, OPERATOR_BUDGET).unwrap_err();
        assert!(errors.contains(&SpecError::DuplicateBrainFamily {
            name: "mlp.baseline".to_owned(),
            first: 0,
            second: 1,
        }));
        assert!(errors.contains(&SpecError::EmptyFactors));
        proposed.brain_families.clear();
        assert!(
            validate_spec(&proposed, OPERATOR_BUDGET)
                .unwrap_err()
                .contains(&SpecError::EmptyBrainFamilies)
        );
    }

    #[test]
    fn intervention_schedule_axis_expands_after_families_and_charges_every_run() {
        let mut proposed = spec();
        proposed.brain_families = vec!["mlp".to_owned(), "dwraon".to_owned()];
        let schedule = vec![
            ScenarioInterventionV1 {
                tick: 0,
                set: serde_json::json!({"food_growth_rate": 0.02}),
            },
            ScenarioInterventionV1 {
                tick: 0,
                set: serde_json::json!({"food_growth_rate": 0.03}),
            },
        ];
        proposed.intervention_schedules = vec![Vec::new(), schedule.clone()];
        let validated = validate_spec(&proposed, OPERATOR_BUDGET).unwrap();
        assert_eq!(validated.arms.len(), 12);
        assert_eq!(validated.seeds, [1, 2, 3, 4]);
        for (index, arm) in validated.arms.iter().enumerate() {
            assert_eq!(
                arm.brain_family,
                ["mlp.baseline", "dwraon.baseline"][(index / 2) % 2]
            );
            assert_eq!(
                arm.interventions,
                proposed.intervention_schedules[index % 2]
            );
            assert_eq!(
                arm.config_overrides["food_growth_rate"],
                proposed.factors[0].values[index / 4]
            );
        }
        assert_eq!(
            validated.cost(),
            RunCost {
                runs: 48,
                ticks: 96_000
            }
        );
        let errors = validate_spec(
            &proposed,
            SpecBudget {
                runs: 24,
                ticks: 96_000,
            },
        )
        .expect_err("schedules must count toward operator admission");
        assert!(errors.iter().any(|error| matches!(
            error,
            SpecError::CostExceedsBudget {
                runs: 48,
                max_runs: 24,
                ..
            }
        )));

        proposed.intervention_schedules[1].swap(0, 1);
        assert_ne!(
            validated.spec_id,
            validate_spec(&proposed, OPERATOR_BUDGET).unwrap().spec_id,
            "same-tick statement order is part of the scientific treatment"
        );
        proposed.intervention_schedules[1] = schedule;
        proposed.intervention_schedules[1][1].tick = 1;
        assert_ne!(
            validated.spec_id,
            validate_spec(&proposed, OPERATOR_BUDGET).unwrap().spec_id,
            "intervention timing is part of the scientific treatment"
        );
    }

    #[test]
    fn schedule_only_studies_require_distinct_treatments_and_bound_the_full_product() {
        let mut proposed = spec();
        proposed.factors.clear();
        proposed.intervention_schedules = vec![
            Vec::new(),
            vec![ScenarioInterventionV1 {
                tick: 1,
                set: serde_json::json!({"metabolism_drain": 0.02}),
            }],
        ];
        let validated = validate_spec(&proposed, OPERATOR_BUDGET).unwrap();
        assert_eq!(validated.arms.len(), 2);
        assert!(validated.arms[0].interventions.is_empty());
        assert_eq!(
            validated.arms[1].interventions,
            proposed.intervention_schedules[1]
        );
        assert!(
            validated
                .arms
                .iter()
                .all(|arm| arm.config_overrides.is_empty())
        );
        proposed.intervention_schedules[1].clear();
        assert!(
            validate_spec(&proposed, OPERATOR_BUDGET)
                .unwrap_err()
                .contains(&SpecError::DuplicateArm {
                    first: 0,
                    second: 1
                })
        );
        proposed.intervention_schedules.clear();
        assert!(
            validate_spec(&proposed, OPERATOR_BUDGET)
                .unwrap_err()
                .contains(&SpecError::EmptyInterventionSchedules)
        );

        let mut product = spec();
        product.factors[0].values = (1..=8)
            .map(|value| serde_json::json!(f64::from(value) * 0.01))
            .collect();
        product.brain_families = vec!["mlp".to_owned(), "dwraon".to_owned()];
        product.intervention_schedules = vec![
            Vec::new(),
            vec![ScenarioInterventionV1 {
                tick: 1,
                set: serde_json::json!({"metabolism_drain": 0.02}),
            }],
        ];
        product.budget = OPERATOR_BUDGET;
        assert_eq!(
            validate_spec(&product, OPERATOR_BUDGET).unwrap().arms.len(),
            MAX_ARMS
        );
        product
            .intervention_schedules
            .push(vec![ScenarioInterventionV1 {
                tick: 2,
                set: serde_json::json!({"metabolism_drain": 0.02}),
            }]);
        assert!(
            validate_spec(&product, OPERATOR_BUDGET)
                .unwrap_err()
                .contains(&SpecError::TooManyArms {
                    arms: 48,
                    max: MAX_ARMS
                })
        );
    }

    #[test]
    fn scheduled_proposals_reuse_runner_timing_live_knob_and_cumulative_validation() {
        let at = |tick, set| ScenarioInterventionV1 { tick, set };
        let mut proposed = spec();
        proposed.factors = vec![Factor {
            knob_path: "food_max".to_owned(),
            values: vec![serde_json::json!(10.0)],
        }];
        proposed.ticks_per_run = 2;
        for (schedule, diagnostic) in [
            (
                vec![at(2, serde_json::json!({"food_growth_rate": 0.1}))],
                "must precede",
            ),
            (
                vec![
                    at(1, serde_json::json!({"food_growth_rate": 0.1})),
                    at(0, serde_json::json!({"food_growth_rate": 0.2})),
                ],
                "nondecreasing",
            ),
            (
                vec![at(0, serde_json::json!({"world_width": 80}))],
                "requires a fresh world",
            ),
            (
                vec![at(0, serde_json::json!({"rng_seed": 9}))],
                "unsupported knob",
            ),
            (
                vec![at(0, serde_json::json!({"food_growth_rate": true}))],
                "numeric value",
            ),
            (
                vec![at(0, serde_json::json!({"food_growth_rate": 1e9}))],
                "accepted range",
            ),
            (vec![at(0, serde_json::json!({}))], "must not be empty"),
        ] {
            proposed.intervention_schedules = vec![Vec::new(), schedule];
            let errors = validate_spec(&proposed, OPERATOR_BUDGET)
                .expect_err("invalid timed treatment must be refused before execution");
            assert!(
                matches!(errors.as_slice(), [SpecError::InvalidExecutionPlan { reason }]
                if reason.contains(diagnostic)),
                "{errors:?}"
            );
        }

        let fill = at(0, serde_json::json!({"initial_food": 2.0}));
        let lower_cap = at(1, serde_json::json!({"food_max": 1.0}));
        for independently_valid in [fill.clone(), lower_cap.clone()] {
            proposed.intervention_schedules = vec![Vec::new(), vec![independently_valid]];
            validate_spec(&proposed, OPERATOR_BUDGET).expect("each patch alone is admissible");
        }
        proposed.intervention_schedules = vec![Vec::new(), vec![fill, lower_cap]];
        let errors = validate_spec(&proposed, OPERATOR_BUDGET)
            .expect_err("the second patch is invalid after the first patch raises initial_food");
        assert!(
            matches!(errors.as_slice(), [SpecError::InvalidExecutionPlan { reason }]
            if reason.contains("initial_food cannot exceed food_max")),
            "{errors:?}"
        );
    }

    #[test]
    fn schedules_are_bounded_before_expansion_and_default_to_one_untreated_arm() {
        let mut proposed = spec();
        let statement = ScenarioInterventionV1 {
            tick: 0,
            set: serde_json::json!({"food_growth_rate": 0.01}),
        };
        proposed.intervention_schedules =
            vec![vec![statement.clone(); MAX_EXPERIMENT_INTERVENTIONS]];
        validate_spec(&proposed, OPERATOR_BUDGET).expect("exact statement cap is executable");
        proposed.intervention_schedules[0].push(statement.clone());
        assert!(
            validate_spec(&proposed, OPERATOR_BUDGET)
                .unwrap_err()
                .iter()
                .any(|error| matches!(
                    error, SpecError::Bounds { field, .. } if field == "intervention_schedules[0]"
                ))
        );
        proposed.intervention_schedules = vec![vec![statement]; MAX_INTERVENTION_SCHEDULES + 1];
        assert!(
            validate_spec(&proposed, OPERATOR_BUDGET)
                .unwrap_err()
                .iter()
                .any(|error| matches!(
                    error, SpecError::Bounds { field, .. } if field == "intervention_schedules"
                ))
        );
        proposed.intervention_schedules = vec![vec![ScenarioInterventionV1 {
            tick: 0,
            set: serde_json::json!({"food_growth_rate": "x".repeat(MAX_EXPERIMENT_INTERVENTION_PATCH_BYTES)}),
        }]];
        assert!(validate_spec(&proposed, OPERATOR_BUDGET).unwrap_err().iter().any(|error| matches!(
            error, SpecError::Bounds { field, .. } if field == "intervention_schedules[0][0].set"
        )));

        let baseline = spec();
        let mut encoded = serde_json::to_value(&baseline).unwrap();
        encoded
            .as_object_mut()
            .unwrap()
            .remove("intervention_schedules");
        let defaulted: ExperimentSpec = serde_json::from_value(encoded).unwrap();
        assert_eq!(
            defaulted.intervention_schedules,
            vec![Vec::<ScenarioInterventionV1>::new()]
        );
        assert_eq!(
            validate_spec(&defaulted, OPERATOR_BUDGET).unwrap(),
            validate_spec(&baseline, OPERATOR_BUDGET).unwrap()
        );
    }

    #[test]
    fn intervention_patch_object_key_order_does_not_split_experiment_identity() {
        let mut forward = spec();
        forward.intervention_schedules = vec![vec![ScenarioInterventionV1 {
            tick: 0,
            set: serde_json::json!({"food_growth_rate": 0.01, "food_decay_rate": 0.02}),
        }]];
        let mut reversed = forward.clone();
        reversed.intervention_schedules[0][0].set =
            serde_json::json!({"food_decay_rate": 0.02, "food_growth_rate": 0.01});
        assert_ne!(
            serde_json::to_vec(&forward).unwrap(),
            serde_json::to_vec(&reversed).unwrap()
        );
        assert_eq!(
            validate_spec(&forward, OPERATOR_BUDGET).unwrap().spec_id,
            validate_spec(&reversed, OPERATOR_BUDGET).unwrap().spec_id
        );
    }

    #[test]
    fn canonical_family_selection_participates_in_the_spec_identity() {
        let proposed = spec();
        let baseline = validate_spec(&proposed, OPERATOR_BUDGET).unwrap();
        let mut alias = proposed.clone();
        alias.brain_families = vec!["mlp".to_owned()];
        let alias = validate_spec(&alias, OPERATOR_BUDGET).unwrap();
        assert_eq!(baseline.spec_id, alias.spec_id);
        assert_eq!(baseline.arms, alias.arms);
        assert_eq!(baseline.spec, alias.spec);

        let mut changed = proposed.clone();
        changed.brain_families = vec!["dwraon".to_owned()];
        let changed = validate_spec(&changed, OPERATOR_BUDGET).unwrap();
        assert_ne!(baseline.spec_id, changed.spec_id);
        assert_eq!(baseline.cost(), changed.cost());
        assert_eq!(
            baseline.arms[0].config_overrides,
            changed.arms[0].config_overrides
        );
        assert_eq!(changed.spec.brain_families, ["dwraon.baseline"]);

        let mut json = serde_json::to_value(proposed).unwrap();
        json.as_object_mut().unwrap().remove("brain_families");
        let defaulted: ExperimentSpec = serde_json::from_value(json).unwrap();
        assert_eq!(defaulted.brain_families, [DEFAULT_EXPERIMENT_BRAIN_FAMILY]);
        assert_eq!(
            validate_spec(&defaulted, OPERATOR_BUDGET).unwrap().spec_id,
            baseline.spec_id
        );
    }

    #[test]
    fn brain_family_validation_uses_the_compiled_runner_catalog() {
        for requested in [
            "imaginary",
            #[cfg(not(feature = "neuro"))]
            "neuroflow",
        ] {
            let mut proposed = spec();
            proposed.brain_families = vec![requested.to_owned()];
            let errors = validate_spec(&proposed, OPERATOR_BUDGET).unwrap_err();
            assert!(errors.iter().any(|error| matches!(
                error,
                SpecError::InvalidBrainFamily { name, .. } if name == requested
            )));
        }
        for &family in EXPERIMENT_BRAIN_FAMILIES {
            let mut proposed = spec();
            proposed.brain_families = vec![family.to_owned()];
            let validated = validate_spec(&proposed, OPERATOR_BUDGET).unwrap();
            assert!(validated.arms.iter().all(|arm| arm.brain_family == family));
        }
        for alias in ["neuro", "neuroflow", "ml.neuroflow"] {
            let mut optional = spec();
            optional.brain_families = vec![alias.to_owned()];
            #[cfg(feature = "neuro")]
            assert!(
                validate_spec(&optional, OPERATOR_BUDGET)
                    .unwrap()
                    .arms
                    .iter()
                    .all(|arm| arm.brain_family == "ml.neuroflow")
            );
            #[cfg(not(feature = "neuro"))]
            assert!(
                validate_spec(&optional, OPERATOR_BUDGET)
                    .unwrap_err()
                    .iter()
                    .any(|error| matches!(
                        error,
                        SpecError::InvalidBrainFamily { name, reason }
                            if name == alias && reason.contains("neuro feature")
                    ))
            );
        }
        let mut optional = spec();
        optional.brain_families = vec!["frankentorch".to_owned()];
        #[cfg(feature = "brain-ft")]
        assert!(
            validate_spec(&optional, OPERATOR_BUDGET)
                .unwrap()
                .arms
                .iter()
                .all(|arm| arm.brain_family == "ft")
        );
        #[cfg(not(feature = "brain-ft"))]
        assert!(
            validate_spec(&optional, OPERATOR_BUDGET)
                .unwrap_err()
                .iter()
                .any(|error| matches!(
                    error,
                    SpecError::InvalidBrainFamily { reason, .. } if reason.contains("brain-ft")
                ))
        );
    }

    #[test]
    fn a_value_that_the_config_validator_happily_accepts_is_still_refused() {
        // THE marquee test. `food_growth_rate = 1e9` passes
        // ScriptBotsConfig::validate — which only asks for finite and
        // non-negative — and would burn the entire run budget on a degenerate
        // world where every food cell is saturated, while the notebook reported
        // on it with a straight face. The range layer is the only thing standing
        // between the lab and that outcome.
        let mut spec = spec();
        spec.factors[0].values = vec![serde_json::json!(1e9)];
        let errors = validate_spec(&spec, OPERATOR_BUDGET).expect_err("1e9 must be refused");
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, SpecError::OutOfRange { path, .. } if path == "food_growth_rate")),
            "expected OutOfRange, got {errors:?}"
        );
        // The message must tell the model exactly what to change.
        let rendered = render_errors(&errors);
        assert!(rendered.contains("food_growth_rate"), "{rendered}");
        assert!(
            rendered.contains("outside the experiment range"),
            "{rendered}"
        );
    }

    #[test]
    fn sweeping_the_seed_is_refused_with_an_explanation_that_teaches() {
        let mut spec = spec();
        spec.factors[0].knob_path = SEED_KNOB.to_owned();
        let errors =
            validate_spec(&spec, OPERATOR_BUDGET).expect_err("seed-as-factor must be refused");
        assert!(errors.contains(&SpecError::SeedAsFactor));
        // A model that made this mistake will make it again unless the error
        // explains the design, so the message is part of the contract.
        let rendered = render_errors(&errors);
        assert!(rendered.contains("matched-seed axis"), "{rendered}");
        assert!(rendered.contains("confound"), "{rendered}");
    }

    #[test]
    fn every_malformed_proposal_is_a_typed_error_and_nothing_panics() {
        /// One malformation, and the mutation that produces it.
        type Malformation = (&'static str, Box<dyn Fn(&mut ExperimentSpec)>);

        let cases: Vec<Malformation> = vec![
            (
                "unknown knob",
                Box::new(|s: &mut ExperimentSpec| {
                    s.factors[0].knob_path = "food.regrowth".to_owned()
                }),
            ),
            (
                "a boolean where a number belongs",
                Box::new(|s: &mut ExperimentSpec| {
                    s.factors[0].values = vec![serde_json::json!(true)]
                }),
            ),
            (
                "NaN, which can only arrive as a string",
                Box::new(|s: &mut ExperimentSpec| {
                    s.factors[0].values = vec![serde_json::json!("NaN")]
                }),
            ),
            (
                "1e40 — survives f64, becomes inf as f32",
                Box::new(|s: &mut ExperimentSpec| {
                    s.factors[0].values = vec![serde_json::json!(1e40)]
                }),
            ),
            (
                "an empty falsifier",
                Box::new(|s: &mut ExperimentSpec| s.falsifier = "   ".to_owned()),
            ),
            (
                "a cohort of nobody",
                Box::new(|s: &mut ExperimentSpec| s.seeds.count = 0),
            ),
            (
                "an unknown metric",
                Box::new(|s: &mut ExperimentSpec| s.metrics = vec!["vibes".to_owned()]),
            ),
            (
                "a duplicated value, which is a run spent learning nothing",
                Box::new(|s: &mut ExperimentSpec| {
                    s.factors[0].values = vec![serde_json::json!(0.05), serde_json::json!(0.05)];
                }),
            ),
            (
                "a cost over budget",
                Box::new(|s: &mut ExperimentSpec| {
                    s.budget = SpecBudget {
                        runs: 2,
                        ticks: 100,
                    }
                }),
            ),
            (
                "an absurd hypothesis length",
                Box::new(|s: &mut ExperimentSpec| s.hypothesis = "x".repeat(MAX_STRING_LEN + 1)),
            ),
            (
                "a path with empty segments",
                Box::new(|s: &mut ExperimentSpec| s.factors[0].knob_path = "a..b".to_owned()),
            ),
            (
                "an empty path",
                Box::new(|s: &mut ExperimentSpec| s.factors[0].knob_path = String::new()),
            ),
            (
                "a NUL byte in the path",
                Box::new(|s: &mut ExperimentSpec| s.factors[0].knob_path = "food\0rate".to_owned()),
            ),
        ];

        for (name, mutate) in cases {
            let mut spec = spec();
            mutate(&mut spec);
            let outcome = validate_spec(&spec, OPERATOR_BUDGET);
            assert!(
                outcome.is_err(),
                "`{name}` must be a typed error, but it validated"
            );
        }
    }

    #[test]
    fn the_validator_reports_every_error_at_once() {
        // First-error-only doubles the number of turns and is the most reliable
        // way to burn a token budget on one malformed proposal.
        let mut spec = spec();
        spec.falsifier = String::new();
        spec.seeds.count = 0;
        spec.metrics = vec!["vibes".to_owned()];
        spec.factors[0].knob_path = "nope".to_owned();

        let errors = validate_spec(&spec, OPERATOR_BUDGET).expect_err("four things are wrong");
        assert!(
            errors.len() >= 4,
            "the model must see all four problems in one turn, saw {}: {errors:?}",
            errors.len()
        );
        assert!(errors.contains(&SpecError::EmptyFalsifier));
        assert!(errors.contains(&SpecError::ZeroSeeds));
    }

    #[test]
    fn arm_expansion_is_ordered_and_the_cartesian_product_is_capped() {
        let mut spec = spec();
        spec.factors = vec![
            Factor {
                knob_path: "food_growth_rate".to_owned(),
                values: vec![serde_json::json!(0.01), serde_json::json!(0.02)],
            },
            Factor {
                knob_path: "food_max".to_owned(),
                values: vec![serde_json::json!(0.5), serde_json::json!(0.6)],
            },
        ];
        spec.budget = SpecBudget {
            runs: 1_000,
            ticks: 1_000_000_000,
        };
        let validated =
            validate_spec(&spec, OPERATOR_BUDGET).expect("two factors, two values each");
        assert_eq!(validated.arms.len(), 4);
        // Declaration order, every time.
        assert_eq!(
            validated.arms[0].config_overrides.get("food_growth_rate"),
            Some(&serde_json::json!(0.01))
        );
        assert_eq!(
            validated.arms[3].config_overrides.get("food_max"),
            Some(&serde_json::json!(0.6))
        );

        // Past the arm cap.
        let mut too_big = spec.clone();
        too_big.factors = (0..4)
            .map(|i| Factor {
                knob_path: [
                    "food_growth_rate",
                    "food_max",
                    "food_decay_rate",
                    "food_diffusion_rate",
                ][i]
                    .to_owned(),
                values: (0..3)
                    .map(|v| serde_json::json!(0.01 + f64::from(v) * 0.01))
                    .collect(),
            })
            .collect();
        // 3^4 = 81 > MAX_ARMS.
        let errors = validate_spec(&too_big, OPERATOR_BUDGET).expect_err("81 arms is over the cap");
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, SpecError::TooManyArms { .. })),
            "{errors:?}"
        );
    }

    #[test]
    fn a_world_dimension_sweep_is_allowed_but_flagged() {
        // The lab starts FRESH worlds, so it may sweep dimensions — but the
        // resulting spec must never be applied to a user's live simulation, and
        // the flag is how the caller knows.
        let mut spec = spec();
        spec.factors[0] = Factor {
            knob_path: "world_width".to_owned(),
            values: vec![serde_json::json!(200), serde_json::json!(400)],
        };
        let validated =
            validate_spec(&spec, OPERATOR_BUDGET).expect("fresh-world sweeps are legal in the lab");
        assert!(
            validated.fresh_world_only,
            "a dimension sweep must be flagged: applying it to a live world is refused by \
             apply_config_update, and an unflagged spec would be discovered one failed run at a time"
        );
    }

    #[test]
    fn structural_unknowns_seed_overflow_and_operator_budget_fail_closed() {
        let mut encoded = serde_json::to_value(spec()).expect("encode canonical spec");
        encoded
            .as_object_mut()
            .expect("spec is an object")
            .insert("target_knobs".to_owned(), serde_json::json!({}));
        let decode_error = serde_json::from_value::<ExperimentSpec>(encoded)
            .expect_err("unknown fields must not be ignored");
        assert!(decode_error.to_string().contains("target_knobs"));

        let mut overflowing = spec();
        overflowing.seeds = SeedPlan {
            base: u64::MAX,
            count: 2,
        };
        let errors = validate_spec(&overflowing, OPERATOR_BUDGET)
            .expect_err("an ascending cohort may not wrap");
        assert_eq!(
            errors.iter().map(SpecError::code).collect::<Vec<_>>(),
            ["seed_overflow"]
        );

        let errors = validate_spec(
            &spec(),
            SpecBudget {
                runs: 4,
                ticks: u64::MAX,
            },
        )
        .expect_err("the operator ceiling is independent of the model's budget");
        assert!(matches!(
            errors.as_slice(),
            [SpecError::CostExceedsBudget { max_runs: 4, .. }]
        ));
    }

    #[test]
    fn repeated_factor_paths_are_rejected_before_arm_expansion() {
        let mut repeated = spec();
        repeated.factors.push(Factor {
            knob_path: repeated.factors[0].knob_path.clone(),
            values: vec![serde_json::json!(0.5)],
        });
        let errors = validate_spec(&repeated, OPERATOR_BUDGET)
            .expect_err("a repeated path would silently overwrite an earlier treatment");
        assert!(matches!(
            errors.as_slice(),
            [SpecError::DuplicateFactor {
                path,
                first: 0,
                second: 1,
            }] if path == "food_growth_rate"
        ));
    }

    #[test]
    fn empty_and_maximal_arm_boundaries_have_stable_error_codes() {
        let mut empty = spec();
        empty.hypothesis = String::new();
        empty.falsifier = String::new();
        empty.factors.clear();
        empty.metrics.clear();
        empty.seeds.count = 0;
        empty.ticks_per_run = 0;
        let errors =
            validate_spec(&empty, OPERATOR_BUDGET).expect_err("empty experiment is invalid");
        assert_eq!(
            errors.iter().map(SpecError::code).collect::<Vec<_>>(),
            [
                "empty_hypothesis",
                "empty_falsifier",
                "empty_factors",
                "empty_metrics",
                "zero_seeds",
                "zero_ticks",
            ]
        );

        let paths = [
            "food_growth_rate",
            "food_max",
            "food_decay_rate",
            "food_diffusion_rate",
        ];
        let mut maximal = spec();
        maximal.factors = paths
            .into_iter()
            .enumerate()
            .map(|(index, path)| Factor {
                knob_path: path.to_owned(),
                values: if index == 0 {
                    vec![
                        serde_json::json!(0.01),
                        serde_json::json!(0.02),
                        serde_json::json!(0.03),
                        serde_json::json!(0.04),
                    ]
                } else if path == "food_max" {
                    // Keep both caps at least as large as the default respawn amount.
                    vec![serde_json::json!(0.5), serde_json::json!(0.6)]
                } else {
                    vec![serde_json::json!(0.01), serde_json::json!(0.02)]
                },
            })
            .collect();
        maximal.seeds.count = 1;
        maximal.ticks_per_run = 1;
        maximal.budget = SpecBudget {
            runs: u32::try_from(MAX_ARMS).expect("arm cap fits u32"),
            ticks: u64::try_from(MAX_ARMS).expect("arm cap fits u64"),
        };
        let validated =
            validate_spec(&maximal, OPERATOR_BUDGET).expect("the exact arm cap is admissible");
        assert_eq!(validated.arms.len(), MAX_ARMS);
    }

    #[test]
    fn tool_schema_is_canonical_and_carries_dynamic_registries() {
        let schema = tool_input_schema().expect("canonical schema serializes");
        assert_eq!(
            schema.pointer("/additionalProperties"),
            Some(&serde_json::Value::Bool(false))
        );
        let required = schema
            .pointer("/required")
            .and_then(serde_json::Value::as_array)
            .expect("root required fields");
        for field in [
            "hypothesis",
            "falsifier",
            "factors",
            "seeds",
            "ticks_per_run",
            "metrics",
            "budget",
        ] {
            assert!(
                required.iter().any(|entry| entry.as_str() == Some(field)),
                "schema drifted away from canonical field `{field}`: {schema}"
            );
        }
        assert!(
            schema.pointer("/components/schemas/Factor").is_some(),
            "nested canonical schemas must accompany their references: {schema}"
        );
        assert_eq!(
            schema.pointer("/x-scriptbots-known-metrics"),
            Some(&serde_json::json!(KNOWN_METRICS))
        );
        assert_eq!(
            schema.pointer("/properties/metrics/items/enum"),
            Some(&serde_json::json!(KNOWN_METRICS))
        );
        assert_eq!(
            schema.pointer("/properties/brain_families/items/enum"),
            Some(&serde_json::json!(EXPERIMENT_BRAIN_FAMILIES))
        );
        assert_eq!(
            schema.pointer("/properties/brain_families/default"),
            Some(&serde_json::json!([DEFAULT_EXPERIMENT_BRAIN_FAMILY]))
        );
        assert_eq!(
            schema.pointer("/x-scriptbots-brain-families"),
            Some(&serde_json::json!(EXPERIMENT_BRAIN_FAMILIES))
        );
        assert!(
            !required
                .iter()
                .any(|field| field.as_str() == Some("brain_families"))
        );
        assert!(
            !required
                .iter()
                .any(|field| field.as_str() == Some("intervention_schedules"))
        );
        assert!(
            !required
                .iter()
                .any(|field| field.as_str() == Some("include_pairwise_interactions"))
        );
        assert_eq!(
            schema.pointer("/properties/include_pairwise_interactions/type"),
            Some(&serde_json::json!("boolean"))
        );
        assert_eq!(
            schema.pointer("/properties/intervention_schedules/default"),
            Some(&serde_json::json!([[]]))
        );
        assert_eq!(
            schema.pointer("/properties/intervention_schedules/maxItems"),
            Some(&serde_json::json!(MAX_INTERVENTION_SCHEDULES))
        );
        assert_eq!(
            schema.pointer("/properties/intervention_schedules/items/maxItems"),
            Some(&serde_json::json!(MAX_EXPERIMENT_INTERVENTIONS))
        );
        assert_eq!(
            schema.pointer("/components/schemas/ScenarioInterventionV1/additionalProperties"),
            Some(&serde_json::json!(false))
        );
        assert!(
            schema
                .pointer("/x-scriptbots-knob-ranges")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|ranges| ranges.len() == KNOB_RANGES.len())
        );
        assert_eq!(
            schema.pointer("/x-scriptbots-validation-limits/max_arms"),
            Some(&serde_json::json!(MAX_ARMS))
        );
        assert_eq!(
            schema.pointer("/x-scriptbots-validation-limits/max_interaction_contrasts"),
            Some(&serde_json::json!(MAX_INTERACTION_CONTRASTS))
        );
        assert_eq!(
            schema.pointer("/x-scriptbots-seed-policy/factor_path"),
            Some(&serde_json::json!(SEED_KNOB))
        );
        assert_eq!(
            schema.pointer("/x-scriptbots-seed-policy/factor_sweepable"),
            Some(&serde_json::Value::Bool(false))
        );
    }
}
