//! Scenario-level budgets for complete, independently replayed saved narratives.
//!
//! Quiet budgets allow at most two records in any 10,000 consecutive ticks.
//! Curated budgets count event kinds within declared inclusive tick windows,
//! and count every record outside those windows against a separate allowance.
//!
//! Example budget for a recorded extinction scenario:
//! ```json
//! {"schema_version":1,"expected_ticks":320,"scenario_id":"extinction",
//!  "root_seed":42,"rule":{"type":"curated","max_unexpected":2,"expected":[
//!    {"kind":"extinction","first_tick":90,"last_tick":105,"min_count":1,"max_count":1}
//!  ]}}
//! ```

use scriptbots_core::narrative::{EventKind, NarrativeInputRecordV1};
use scriptbots_core::rng_domains::IslandId;
use scriptbots_storage::{IslandNarrativeInputBindingV1, StorageReader};
use serde::{Deserialize, Serialize};

use crate::{
    AnalyticsError, NarrativeComparisonScope, NarrativeVerificationV1,
    require_single_world_narrative, verify_narrative_inputs,
};

const QUIET_TICKS: u64 = 10_000;
const MAX_EXPECTATIONS: usize = 64;

/// Machine-readable specification supplied independently of the observed events.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NarrativeBudget {
    /// Version of the budget specification. Currently 1.
    pub schema_version: u32,
    /// Exact required completed tick count, including the final tick.
    pub expected_ticks: u64,
    /// Optional scenario identity required from an attributed manifest.
    pub scenario_id: Option<String>,
    /// Optional root seed required from an attributed manifest.
    pub root_seed: Option<u64>,
    /// Optional configured island whose detector is under evaluation.
    pub island_id: Option<IslandId>,
    /// Quiet-world or curated-event acceptance rule.
    pub rule: NarrativeBudgetRule,
}

/// Complete-run narrative budget.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum NarrativeBudgetRule {
    /// Occupied world with at most two events in every 10,000-tick window.
    Quiet,
    /// Expected events inside declared windows and a cap on everything else.
    Curated {
        /// Between one and 64 non-overlapping windows per event kind (64 total).
        expected: Vec<EventExpectation>,
        /// Maximum number of records outside all matching kind/window pairs.
        max_unexpected: u64,
    },
}

/// Expected count range for one kind, measured at canonical event ticks.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventExpectation {
    /// Canonical event kind.
    pub kind: EventKind,
    /// First included tick.
    pub first_tick: u64,
    /// Last included tick.
    pub last_tick: u64,
    /// Positive minimum required events.
    pub min_count: u64,
    /// Maximum accepted events.
    pub max_count: u64,
}

impl NarrativeBudget {
    fn validate(&self) -> Result<(), AnalyticsError> {
        let invalid = |reason: &str| AnalyticsError::BadParam {
            name: "narrative budget".to_owned(),
            reason: reason.to_owned(),
        };
        if self.schema_version != 1 || self.expected_ticks == 0 {
            return Err(invalid("expected schema_version 1 and positive expected_ticks"));
        }
        match &self.rule {
            NarrativeBudgetRule::Quiet if self.expected_ticks < QUIET_TICKS => {
                return Err(invalid("a quiet budget requires at least 10,000 completed ticks"));
            }
            NarrativeBudgetRule::Curated { expected, .. } => {
                if expected.is_empty() || expected.len() > MAX_EXPECTATIONS {
                    return Err(invalid("curated budgets require 1..=64 event windows"));
                }
                for (index, window) in expected.iter().enumerate() {
                    if window.first_tick == 0 || window.last_tick < window.first_tick
                        || window.last_tick > self.expected_ticks || window.min_count == 0
                        || window.max_count < window.min_count
                    {
                        return Err(invalid("event tick bounds or required count range are invalid"));
                    }
                    if expected[..index].iter().any(|previous| {
                        previous.kind == window.kind && previous.first_tick <= window.last_tick
                            && window.first_tick <= previous.last_tick
                    }) {
                        return Err(invalid("windows for the same event kind must be disjoint"));
                    }
                }
            }
            NarrativeBudgetRule::Quiet => {}
        }
        Ok(())
    }
}

/// Measured counts and first failures; memory is independent of run length.
#[derive(Debug, Default, Serialize)]
pub struct BudgetMeasurements {
    /// Exact number of input ticks included in at least one eligible detector pass.
    pub covered_ticks: u64,
    /// Number of inputs containing a nonempty population.
    pub occupied_ticks: u64,
    /// Number of eligible detector passes, including zero-event passes.
    pub detector_passes: u64,
    /// Counts in the same order as the specification's curated windows.
    pub counts: Vec<u64>,
    /// Number of events outside all curated kind/window pairs.
    pub unexpected_events: u64,
    /// First tick of an out-of-window curated event.
    pub first_unexpected_tick: Option<u64>,
    /// First three canonical event ticks violating a quiet rolling window.
    pub first_quiet_violation: Option<[u64; 3]>,
}

pub(crate) struct BudgetTally<'a> {
    budget: &'a NarrativeBudget,
    measurements: BudgetMeasurements,
    last_covered: u64,
    previous_events: [Option<u64>; 2],
}

impl<'a> BudgetTally<'a> {
    fn new(budget: &'a NarrativeBudget) -> Self {
        let count = match &budget.rule {
            NarrativeBudgetRule::Quiet => 0,
            NarrativeBudgetRule::Curated { expected, .. } => expected.len(),
        };
        Self {
            budget,
            measurements: BudgetMeasurements {
                counts: vec![0; count],
                ..BudgetMeasurements::default()
            },
            last_covered: 0,
            previous_events: [None, None],
        }
    }

    pub(crate) fn input(&mut self, record: NarrativeInputRecordV1, oldest: u64) {
        let tick = record.input.tick.0;
        if record.input.agent_count != 0 {
            self.measurements.occupied_ticks += 1;
        }
        // The production kernel requires eight retained measurements, an
        // enabled event ring, and a scheduled pass on the absolute tick.
        if record.event_capacity != 0 && record.narrative_interval != 0
            && tick.is_multiple_of(u64::from(record.narrative_interval))
            && tick - oldest >= 7
        {
            self.measurements.detector_passes += 1;
            self.measurements.covered_ticks += tick - self.last_covered.max(oldest - 1);
            self.last_covered = tick;
        }
    }

    pub(crate) fn event(&mut self, kind: EventKind, tick: u64) {
        let m = &mut self.measurements;
        match &self.budget.rule {
            NarrativeBudgetRule::Quiet => {
                if let [Some(first), Some(second)] = self.previous_events
                    && tick - first < QUIET_TICKS && m.first_quiet_violation.is_none()
                {
                    m.first_quiet_violation = Some([first, second, tick]);
                }
                self.previous_events = [self.previous_events[1], Some(tick)];
            }
            NarrativeBudgetRule::Curated { expected, .. } => {
                if let Some(index) = expected.iter().position(|w| {
                    kind == w.kind && (w.first_tick..=w.last_tick).contains(&tick)
                }) {
                    m.counts[index] += 1;
                } else {
                    m.unexpected_events += 1;
                    m.first_unexpected_tick.get_or_insert(tick);
                }
            }
        }
    }

    fn violations(&self) -> Vec<&'static str> {
        let mut failures = Vec::new();
        let m = &self.measurements;
        if m.covered_ticks != self.budget.expected_ticks {
            failures.push("incomplete_detector_coverage");
        }
        match &self.budget.rule {
            NarrativeBudgetRule::Quiet => {
                if m.occupied_ticks != self.budget.expected_ticks {
                    failures.push("quiet_world_has_unoccupied_ticks");
                }
                if m.first_quiet_violation.is_some() {
                    failures.push("quiet_rolling_limit_exceeded");
                }
            }
            NarrativeBudgetRule::Curated { expected, max_unexpected } => {
                if m.occupied_ticks == 0 {
                    failures.push("world_never_occupied");
                }
                if expected.iter().zip(&m.counts).any(|(w, n)| !(w.min_count..=w.max_count).contains(n)) {
                    failures.push("expected_event_count_out_of_range");
                }
                if m.unexpected_events > *max_unexpected {
                    failures.push("unexpected_event_limit_exceeded");
                }
            }
        }
        failures
    }
}

/// Full parity evidence and a separate budget verdict for this recorded detector.
#[derive(Debug, Serialize)]
pub struct NarrativeBudgetReport<'a> {
    /// Version of the report.
    pub schema_version: u32,
    /// Exact user-supplied budget.
    pub budget: &'a NarrativeBudget,
    /// Scenario and root seed reported by the run manifest, absent for unattributed runs.
    pub observed_scenario_id: Option<String>,
    /// Root seed from the same manifest (independent of an island's local seed).
    pub observed_root_seed: Option<u64>,
    /// Present for an explicitly selected configured island.
    pub island_binding: Option<IslandNarrativeInputBindingV1>,
    /// Source binding, stream hashes, counts and first persisted divergence.
    pub verification: NarrativeVerificationV1,
    /// Complete measurement totals on the reconstructed stream.
    pub measurements: BudgetMeasurements,
    /// Machine-readable failed conditions. Empty for a pass.
    pub violations: Vec<&'static str>,
    /// True only when parity, source identity, coverage and every budget check passed.
    pub passed: bool,
}

/// Replay the chosen detector and evaluate an explicit budget over its complete canonical events.
pub fn verify_narrative_budget<'a>(
    reader: &StorageReader,
    budget: &'a NarrativeBudget,
) -> Result<NarrativeBudgetReport<'a>, AnalyticsError> {
    budget.validate()?;
    let scope = if let Some(island) = budget.island_id {
        NarrativeComparisonScope::SelectedIsland(island)
    } else {
        require_single_world_narrative(reader)?;
        NarrativeComparisonScope::SingleWorld
    };
    let manifest = reader.run_manifest()?;
    let mut tally = BudgetTally::new(budget);
    let (verification, island_binding) = verify_narrative_inputs(reader, scope, Some(&mut tally))?;
    let mut violations = tally.violations();
    if !verification.equivalent {
        violations.push("saved_events_diverged");
    }
    if verification.inputs_replayed != budget.expected_ticks {
        violations.push("unexpected_run_length");
    }
    if budget.scenario_id.as_ref().is_some_and(|id| manifest.manifest_schema_version == 0 || id != &manifest.scenario_id) {
        violations.push("scenario_mismatch");
    }
    if budget.root_seed.is_some_and(|seed| manifest.manifest_schema_version == 0 || seed != manifest.root_seed) {
        violations.push("root_seed_mismatch");
    }
    Ok(NarrativeBudgetReport {
        schema_version: 1,
        budget,
        observed_scenario_id: (manifest.manifest_schema_version != 0).then_some(manifest.scenario_id),
        observed_root_seed: (manifest.manifest_schema_version != 0).then_some(manifest.root_seed),
        island_binding,
        verification,
        measurements: tally.measurements,
        passed: violations.is_empty(),
        violations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::Tick;
    use scriptbots_core::narrative::NarrativeInputV1;

    fn budget(rule: NarrativeBudgetRule) -> NarrativeBudget {
        NarrativeBudget {
            schema_version: 1, expected_ticks: QUIET_TICKS, scenario_id: None,
            root_seed: None, island_id: None, rule,
        }
    }

    fn input(tick: u64, interval: u32, event_capacity: u64) -> NarrativeInputRecordV1 {
        NarrativeInputRecordV1 {
            record_schema_version: 1,
            input: NarrativeInputV1::new(Tick(tick), 20, 1.0, 0).unwrap(),
            config_revision: 0, narrative_interval: interval, history_capacity: 64, event_capacity,
        }
    }

    #[test]
    fn quiet_counts_rolling_windows_and_refuses_missing_observation() {
        let spec = budget(NarrativeBudgetRule::Quiet);
        let mut tally = BudgetTally::new(&spec);
        for tick in 1..=QUIET_TICKS {
            tally.input(input(tick, 16, 1), tick.saturating_sub(63).max(1));
        }
        assert!(tally.violations().is_empty());
        assert_eq!(tally.measurements.detector_passes, 625);
        tally.event(EventKind::EnergyRecovery, 1);
        tally.event(EventKind::PopulationBoom, 10_000);
        tally.event(EventKind::Extinction, 10_001);
        assert!(tally.measurements.first_quiet_violation.is_none());
        tally.event(EventKind::EnergyCollapse, 10_001);
        assert_eq!(tally.measurements.first_quiet_violation, Some([10_000, 10_001, 10_001]));

        for (interval, cap, retained, expected) in [(0,1,64,0),(1,0,64,0),(1,1,7,0),(16,1,8,5000)] {
            let mut incomplete = BudgetTally::new(&spec);
            for tick in 1..=QUIET_TICKS {
                incomplete.input(input(tick, interval, cap), tick.saturating_sub(retained - 1).max(1));
            }
            assert_eq!(incomplete.measurements.covered_ticks, expected);
            assert!(incomplete.violations().contains(&"incomplete_detector_coverage"));
        }
    }

    #[test]
    fn curated_counts_exact_kinds_and_inclusive_disjoint_windows() {
        let window = |first_tick, last_tick| EventExpectation {
            kind: EventKind::PopulationBoom, first_tick, last_tick, min_count: 1, max_count: 2
        };
        let spec = budget(NarrativeBudgetRule::Curated {
            expected: vec![window(5, 7), window(9, 15)], max_unexpected: 1,
        });
        spec.validate().unwrap();
        let mut tally = BudgetTally::new(&spec);
        for (kind, tick) in [(EventKind::PopulationBoom, 5), (EventKind::PopulationBoom,7),
                             (EventKind::PopulationBoom,8), (EventKind::PopulationBoom,15),
                             (EventKind::Extinction,15)] {
            tally.event(kind,tick);
        }
        assert_eq!(tally.measurements.counts, [2,1]);
        assert_eq!(tally.measurements.unexpected_events,2);
        assert_eq!(tally.measurements.first_unexpected_tick,Some(8));
        assert!(tally.violations().contains(&"unexpected_event_limit_exceeded"));
        assert!(!tally.violations().contains(&"expected_event_count_out_of_range"));
        assert!(budget(NarrativeBudgetRule::Curated {
            expected: vec![window(5,10),window(10,15)], max_unexpected:0,
        }).validate().is_err());
    }
}
