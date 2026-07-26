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

use scriptbots_core::{KNOB_RANGES, knob_range};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use utoipa::{PartialSchema, ToSchema};

/// Most factors one spec may vary.
pub const MAX_FACTORS: usize = 4;
/// Most values one factor may take.
pub const MAX_VALUES_PER_FACTOR: usize = 8;
/// Most arms (the cartesian product of the factors) one spec may expand to.
pub const MAX_ARMS: usize = 32;
/// Most seeds in a cohort.
pub const MAX_SEEDS: u16 = 64;
/// Longest run, in ticks.
pub const MAX_TICKS_PER_RUN: u64 = 1_000_000;
/// Longest free-text field.
pub const MAX_STRING_LEN: usize = 4_096;

/// Metrics emitted by the matched-seed runner and therefore valid for analysis.
pub const KNOWN_METRICS: &[&str] = &["alive_agents"];

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
    /// The knobs to vary.
    pub factors: Vec<Factor>,
    /// The matched seed cohort.
    pub seeds: SeedPlan,
    /// How long each run lasts.
    pub ticks_per_run: u64,
    /// Which summary columns to compare. Must name metrics that exist.
    pub metrics: Vec<String>,
    /// The ceiling this spec must fit inside.
    pub budget: SpecBudget,
}

/// One arm of the experiment: a complete assignment of every factor.
pub type Arm = BTreeMap<String, serde_json::Value>;

/// A spec that has been checked and expanded.
///
/// Two validations of the same spec produce a byte-identical `ValidatedSpec`, and
/// its `spec_id` IS the experiment's identity — arms expand in declaration order,
/// seeds ascend, and no `HashMap` appears anywhere in the expansion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidatedSpec {
    /// The original proposal.
    pub spec: ExperimentSpec,
    /// Every arm, in declaration order.
    pub arms: Vec<Arm>,
    /// The cohort, ascending.
    pub seeds: Vec<u64>,
    /// Whether any arm touches a knob that cannot be changed on a live world.
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
                "`factors` must contain at least one treatment axis: an empty arm cannot compare a treatment."
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
                "these factors expand to {arms} arms, over the limit of {max}. Use fewer factors, \
                 or fewer values per factor."
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
    let mut referenced = Vec::new();
    <ExperimentSpec as ToSchema>::schemas(&mut referenced);
    let schemas = referenced
        .into_iter()
        .map(|(name, schema)| {
            let mut schema = serde_json::to_value(schema)?;
            if matches!(name.as_str(), "Factor" | "SeedPlan" | "SpecBudget")
                && let Some(object) = schema.as_object_mut()
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
            "x-scriptbots-validation-limits".to_owned(),
            serde_json::json!({
                "max_factors": MAX_FACTORS,
                "max_values_per_factor": MAX_VALUES_PER_FACTOR,
                "max_arms": MAX_ARMS,
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
    if spec.factors.is_empty() {
        errors.push(SpecError::EmptyFactors);
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

    let mut fresh_world_only = false;
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
        .unwrap_or(usize::MAX);
    if arm_count > MAX_ARMS {
        errors.push(SpecError::TooManyArms {
            arms: arm_count,
            max: MAX_ARMS,
        });
    }
    let can_expand = !spec.factors.is_empty()
        && spec.factors.len() <= MAX_FACTORS
        && factor_positions.len() == spec.factors.len()
        && spec.factors.iter().all(|factor| {
            !factor.values.is_empty() && factor.values.len() <= MAX_VALUES_PER_FACTOR
        })
        && arm_count <= MAX_ARMS;
    let arms = if can_expand {
        expand(&spec.factors)
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

    let spec_id = content_hash(spec, &arms, &seeds);
    Ok(ValidatedSpec {
        spec: spec.clone(),
        arms,
        seeds,
        fresh_world_only,
        spec_id,
    })
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

/// The cartesian product of the factors, in declaration order.
///
/// `BTreeMap` and `Vec` only: a `HashMap` anywhere in here would make the arm
/// order — and therefore the spec id — depend on the hasher's seed.
fn expand(factors: &[Factor]) -> Vec<Arm> {
    let mut arms: Vec<Arm> = vec![Arm::new()];
    for factor in factors {
        let mut next = Vec::with_capacity(arms.len() * factor.values.len().max(1));
        for arm in &arms {
            for value in &factor.values {
                let mut extended = arm.clone();
                extended.insert(factor.knob_path.clone(), value.clone());
                next.push(extended);
            }
        }
        if !next.is_empty() {
            arms = next;
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
    let canonical = serde_json::json!({
        "hypothesis": spec.hypothesis,
        "falsifier": spec.falsifier,
        "ticks_per_run": spec.ticks_per_run,
        "metrics": spec.metrics,
        "arms": arms,
        "seeds": seeds,
    });
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
            seeds: SeedPlan { base: 1, count: 4 },
            ticks_per_run: 2_000,
            metrics: vec!["alive_agents".to_owned()],
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
            validated.arms[0].get("food_growth_rate"),
            Some(&serde_json::json!(0.01))
        );
        assert_eq!(
            validated.arms[3].get("food_max"),
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
            schema.pointer("/x-scriptbots-known-metrics/0"),
            Some(&serde_json::json!("alive_agents"))
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
            schema.pointer("/x-scriptbots-seed-policy/factor_path"),
            Some(&serde_json::json!(SEED_KNOB))
        );
        assert_eq!(
            schema.pointer("/x-scriptbots-seed-policy/factor_sweepable"),
            Some(&serde_json::Value::Bool(false))
        );
    }
}
