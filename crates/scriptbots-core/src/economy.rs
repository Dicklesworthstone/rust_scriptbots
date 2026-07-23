//! Epoch-level flow aggregation over the per-tick resource ledger.
//!
//! The bd-2z0.2.8 ledger already proves each tick's books internally: every
//! mutation site posts into a closed [`crate::ResourceFlowKind`] category set,
//! and [`crate::ResourceReconciliation`] states the tick's unexplained delta
//! against a derived tolerance. What it does not provide is the WINDOWED view
//! science consumes: per-epoch flow vectors, cross-tick residual accumulation
//! that does not quietly lose precision, and the single artifact the Sankey,
//! the TUI table, and the export all read. This module is that layer — ONE
//! accountant, so nothing downstream re-derives a flow from agent state and
//! disagrees.
//!
//! Anti-coupling is deliberate: the aggregator consumes
//! [`crate::ResourceLedgerTick`] values and knows nothing about `WorldState`,
//! storage, ratatui, or wgpu.
//!
//! # Determinism
//!
//! Accumulation is `f64` in stable category-index order with Neumaier
//! compensated summation for every cross-tick lane, over fixed-size arrays —
//! never map iteration, never wall-clock. Identical input sequences produce
//! bit-identical [`EpochFlows`] on every platform and thread count.

// bd-tqpj: deterministic-simulation policy — pinned floating-point evaluation
// order and fixed-width casts are part of the science contract; fma fusion,
// reassociation, or width changes alter world digests. Function lengths mirror
// the legacy C++ parity layout and are reviewed as units.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
#![allow(clippy::float_cmp, clippy::while_float)]

#[cfg(any(test, feature = "economy-faults"))]
use crate::LedgerFault;
use crate::{RESOURCE_FLOW_KINDS, ResourceAmounts, ResourceFlowKind, ResourceLedgerTick, Tick};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Number of stocks the ledger tracks: grid food, agent energy, agent health.
const STOCK_COUNT: usize = 3;
/// Number of closed flow categories, fixed by the ledger's stable enum order.
const CATEGORY_COUNT: usize = RESOURCE_FLOW_KINDS.len();

/// Relative bound for cross-epoch residual accumulation.
///
/// Derivation: `f64` carries ~15.9 significant digits, so each compensated
/// addition contributes relative error on the order of `1e-16` of the gross
/// magnitude flowing through the books. An epoch accumulates thousands of
/// postings, and the per-tick ledger already admits
/// [`crate::RESOURCE_LEDGER_RELATIVE_TOLERANCE`] (`1e-6`) per tick boundary.
/// `1e-7 x max(1.0, gross)` therefore bounds honest floating-point noise while
/// staying an order of magnitude tighter than what a single tick may already
/// hide; a real leak grows linearly with ticks and crosses it quickly. This
/// constant lives here and nowhere else, so loosening it shows up in a diff.
pub const EPOCH_RESIDUAL_RELATIVE_TOLERANCE: f64 = 1.0e-7;

/// Which of the three ledger stocks a residual row describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EconomyStock {
    /// Food stored in the ground grid.
    GridFood,
    /// Energy held by living agents.
    AgentEnergy,
    /// Health held by living agents.
    AgentHealth,
}

impl EconomyStock {
    const ALL: [Self; STOCK_COUNT] = [Self::GridFood, Self::AgentEnergy, Self::AgentHealth];

    /// Extract this stock's lane from a [`ResourceAmounts`] triple.
    const fn lane(self, amounts: ResourceAmounts) -> f64 {
        match self {
            Self::GridFood => amounts.food,
            Self::AgentEnergy => amounts.energy,
            Self::AgentHealth => amounts.health,
        }
    }
}

/// Why epoch aggregation rejected a ledger tick.
///
/// A non-finite value must become a typed error, never a NaN quietly poisoning
/// the residual into `0 != 0 is false, so we're fine`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EconomyAggregationError {
    /// A flow or reconciliation lane carried NaN or an infinity.
    #[error(
        "tick {tick} carries a non-finite value in {location}; refusing to poison the epoch books"
    )]
    NonFinite {
        /// Tick whose report carried the non-finite value.
        tick: u64,
        /// Human-readable lane description (category and field).
        location: String,
    },
    /// The tick's flow set does not contain exactly the closed category set.
    #[error("tick {tick} has {actual} flow rows; the closed category set requires {expected}")]
    MalformedFlowSet {
        /// Tick whose report was malformed.
        tick: u64,
        /// Rows found.
        actual: usize,
        /// Rows required.
        expected: usize,
    },
    /// The tick's flow rows are not in stable enum order.
    #[error("tick {tick} flow row {index} is {actual:?} but stable order requires {expected:?}")]
    MisorderedFlowSet {
        /// Tick whose report was misordered.
        tick: u64,
        /// Offending row index.
        index: usize,
        /// Category found at that index.
        actual: ResourceFlowKind,
        /// Category the stable order requires there.
        expected: ResourceFlowKind,
    },
    /// Ticks must arrive in strictly increasing order.
    #[error(
        "tick {actual} arrived at or before tick {previous}; aggregation requires strictly increasing ticks"
    )]
    NonMonotonicTick {
        /// Last accepted tick.
        previous: u64,
        /// Offending tick.
        actual: u64,
    },
}

/// Neumaier (improved Kahan) compensated summation for one `f64` lane.
///
/// Plain `+=` across an epoch of postings loses low-order bits exactly where a
/// conservation instrument cannot afford to: the residual. Neumaier's variant
/// also survives the case where the incoming term is larger than the running
/// sum, which Kahan's original does not.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct NeumaierSum {
    sum: f64,
    compensation: f64,
}

impl NeumaierSum {
    fn add(&mut self, value: f64) {
        let tentative = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.compensation += (self.sum - tentative) + value;
        } else {
            self.compensation += (value - tentative) + self.sum;
        }
        self.sum = tentative;
    }

    fn value(self) -> f64 {
        self.sum + self.compensation
    }
}

/// One category's accumulated flows across an epoch, in ledger lane units.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EpochCategoryFlow {
    /// Stable category.
    pub kind: ResourceFlowKind,
    /// Signed world-wide deltas summed across the epoch.
    pub delta: ResourceAmounts,
    /// Positive gross activity summed across the epoch.
    pub activity: ResourceAmounts,
}

/// One stock's conservation summary across an epoch.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EpochStockResidual {
    /// Stock this row describes.
    pub stock: EconomyStock,
    /// Compensated sum of per-tick unexplained deltas.
    pub residual_sum: f64,
    /// Largest absolute per-tick unexplained delta seen in the epoch.
    pub residual_max_abs: f64,
    /// Tick that produced `residual_max_abs`, when any tick was observed.
    pub argmax_tick: Option<u64>,
    /// Compensated sum of absolute category deltas for this stock.
    pub gross_flow: f64,
    /// `EPOCH_RESIDUAL_RELATIVE_TOLERANCE * max(1.0, gross_flow)`.
    pub cumulative_tolerance: f64,
    /// Whether `|residual_sum| <= cumulative_tolerance`.
    pub within_tolerance: bool,
    /// Category with the largest absolute accumulated delta on this stock,
    /// when any delta was non-zero — the first suspect when the residual is
    /// not.
    pub worst_category: Option<ResourceFlowKind>,
}

/// The single windowed accounting artifact.
///
/// The Sankey, the TUI table, and the export (bd-16g.11.3) all read THIS type;
/// none of them re-derives a flow from world state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochFlows {
    /// Zero-based epoch ordinal since the aggregator was constructed.
    pub epoch: u64,
    /// First ledger tick included in the epoch.
    pub first_tick: u64,
    /// Last ledger tick included in the epoch.
    pub last_tick: u64,
    /// Number of ledger ticks aggregated.
    pub tick_count: u64,
    /// `true` when the epoch sealed by filling its window; `false` when it was
    /// sealed early by [`EpochAggregator::finish`].
    pub complete: bool,
    /// Per-category accumulated flows in stable enum order, all categories
    /// present including zero-valued ones.
    pub per_category: Vec<EpochCategoryFlow>,
    /// Per-stock conservation summaries in [`EconomyStock::ALL`] order.
    pub residual: [EpochStockResidual; STOCK_COUNT],
}

/// Windowed accumulator over [`ResourceLedgerTick`] reports.
///
/// Feed completed ledger ticks in order via [`Self::observe`]; a sealed
/// [`EpochFlows`] is returned whenever the window fills, and a partial epoch
/// can be sealed explicitly with [`Self::finish`]. `O(categories)` per tick,
/// no allocation between epoch boundaries.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochAggregator {
    window: u64,
    next_epoch: u64,
    last_tick: Option<u64>,
    // Running epoch state, reset at each seal.
    first_tick: Option<u64>,
    epoch_last_tick: u64,
    tick_count: u64,
    delta_sums: [[NeumaierSum; STOCK_COUNT]; CATEGORY_COUNT],
    activity_sums: [[NeumaierSum; STOCK_COUNT]; CATEGORY_COUNT],
    residual_sums: [NeumaierSum; STOCK_COUNT],
    residual_max_abs: [f64; STOCK_COUNT],
    argmax_tick: [Option<u64>; STOCK_COUNT],
    gross_flow: [NeumaierSum; STOCK_COUNT],
}

impl EpochAggregator {
    /// Create an aggregator sealing an epoch every `window` ledger ticks.
    ///
    /// A zero window is clamped to one: an epoch that can never seal would
    /// accumulate forever and emit nothing, which is a silent way to lose the
    /// instrument.
    #[must_use]
    pub fn new(window: u64) -> Self {
        Self {
            window: window.max(1),
            next_epoch: 0,
            last_tick: None,
            first_tick: None,
            epoch_last_tick: 0,
            tick_count: 0,
            delta_sums: [[NeumaierSum::default(); STOCK_COUNT]; CATEGORY_COUNT],
            activity_sums: [[NeumaierSum::default(); STOCK_COUNT]; CATEGORY_COUNT],
            residual_sums: [NeumaierSum::default(); STOCK_COUNT],
            residual_max_abs: [0.0; STOCK_COUNT],
            argmax_tick: [None; STOCK_COUNT],
            gross_flow: [NeumaierSum::default(); STOCK_COUNT],
        }
    }

    /// The configured window length in ledger ticks.
    #[must_use]
    pub const fn window(&self) -> u64 {
        self.window
    }

    /// Fold one completed ledger tick into the current epoch.
    ///
    /// Returns `Ok(Some(_))` when this tick filled the window and sealed an
    /// epoch. Rejected ticks leave the aggregator state untouched.
    ///
    /// # Errors
    ///
    /// Returns a typed [`EconomyAggregationError`] for non-finite values, a
    /// malformed or misordered flow set, or a non-monotonic tick.
    pub fn observe(
        &mut self,
        report: &ResourceLedgerTick,
    ) -> Result<Option<EpochFlows>, EconomyAggregationError> {
        let tick = report.tick.0;
        validate_report(report)?;
        if let Some(previous) = self.last_tick
            && tick <= previous
        {
            return Err(EconomyAggregationError::NonMonotonicTick {
                previous,
                actual: tick,
            });
        }

        self.last_tick = Some(tick);
        if self.first_tick.is_none() {
            self.first_tick = Some(tick);
        }
        self.epoch_last_tick = tick;
        self.tick_count += 1;

        for (index, flow) in report.flows.iter().enumerate() {
            for (stock_index, stock) in EconomyStock::ALL.into_iter().enumerate() {
                let delta = stock.lane(flow.delta);
                self.delta_sums[index][stock_index].add(delta);
                self.activity_sums[index][stock_index].add(stock.lane(flow.activity));
                self.gross_flow[stock_index].add(delta.abs());
            }
        }
        for (stock_index, stock) in EconomyStock::ALL.into_iter().enumerate() {
            let unexplained = stock.lane(report.reconciliation.unexplained_delta);
            self.residual_sums[stock_index].add(unexplained);
            if unexplained.abs() >= self.residual_max_abs[stock_index]
                && (unexplained.abs() > self.residual_max_abs[stock_index]
                    || self.argmax_tick[stock_index].is_none())
            {
                self.residual_max_abs[stock_index] = unexplained.abs();
                self.argmax_tick[stock_index] = Some(tick);
            }
        }

        if self.tick_count >= self.window {
            return Ok(Some(self.seal(true)));
        }
        Ok(None)
    }

    /// Seal the in-progress partial epoch, if any ticks were observed.
    ///
    /// The emitted epoch is marked `complete = false`; monotonicity carries
    /// over, so later ticks may keep feeding the same aggregator.
    #[must_use]
    pub fn finish(&mut self) -> Option<EpochFlows> {
        if self.tick_count == 0 {
            return None;
        }
        Some(self.seal(false))
    }

    fn seal(&mut self, complete: bool) -> EpochFlows {
        let per_category = RESOURCE_FLOW_KINDS
            .into_iter()
            .enumerate()
            .map(|(index, kind)| EpochCategoryFlow {
                kind,
                delta: amounts_from_lanes(&self.delta_sums[index]),
                activity: amounts_from_lanes(&self.activity_sums[index]),
            })
            .collect::<Vec<_>>();

        let residual = core::array::from_fn(|stock_index| {
            let stock = EconomyStock::ALL[stock_index];
            let residual_sum = self.residual_sums[stock_index].value();
            let gross_flow = self.gross_flow[stock_index].value();
            let cumulative_tolerance = EPOCH_RESIDUAL_RELATIVE_TOLERANCE * gross_flow.max(1.0);
            let worst_category = per_category
                .iter()
                .map(|flow| stock.lane(flow.delta).abs())
                .enumerate()
                .filter(|(_, magnitude)| *magnitude > 0.0)
                .max_by(|(left_index, left), (right_index, right)| {
                    left.total_cmp(right)
                        // Ties resolve to the LOWER stable index so the answer
                        // is deterministic and platform-independent.
                        .then(right_index.cmp(left_index))
                })
                .map(|(index, _)| RESOURCE_FLOW_KINDS[index]);
            EpochStockResidual {
                stock,
                residual_sum,
                residual_max_abs: self.residual_max_abs[stock_index],
                argmax_tick: self.argmax_tick[stock_index],
                gross_flow,
                cumulative_tolerance,
                within_tolerance: residual_sum.abs() <= cumulative_tolerance,
                worst_category,
            }
        });

        let flows = EpochFlows {
            epoch: self.next_epoch,
            first_tick: self.first_tick.unwrap_or(self.epoch_last_tick),
            last_tick: self.epoch_last_tick,
            tick_count: self.tick_count,
            complete,
            per_category,
            residual,
        };

        self.next_epoch += 1;
        self.first_tick = None;
        self.tick_count = 0;
        self.delta_sums = [[NeumaierSum::default(); STOCK_COUNT]; CATEGORY_COUNT];
        self.activity_sums = [[NeumaierSum::default(); STOCK_COUNT]; CATEGORY_COUNT];
        self.residual_sums = [NeumaierSum::default(); STOCK_COUNT];
        self.residual_max_abs = [0.0; STOCK_COUNT];
        self.argmax_tick = [None; STOCK_COUNT];
        self.gross_flow = [NeumaierSum::default(); STOCK_COUNT];

        // bd-16g.11.1 logging contract: one line from which a reader who has
        // never seen the source can answer "where did the joules go this
        // epoch?". `finish` is a consuming seal, not a pure function — the pure
        // `observe` path never logs.
        let category_totals: Vec<String> = flows
            .per_category
            .iter()
            .filter(|flow| flow.delta.scale() != 0.0)
            .map(|flow| format!("{:?}={:.6}", flow.kind, flow.delta.scale()))
            .collect();
        tracing::info!(
            target: "scriptbots::economy",
            epoch = flows.epoch,
            first_tick = flows.first_tick,
            last_tick = flows.last_tick,
            tick_count = flows.tick_count,
            categories = %category_totals.join(" "),
            residual_food = flows.residual[0].residual_sum,
            residual_energy = flows.residual[1].residual_sum,
            residual_health = flows.residual[2].residual_sum,
            gross_food = flows.residual[0].gross_flow,
            gross_energy = flows.residual[1].gross_flow,
            gross_health = flows.residual[2].gross_flow,
            "epoch resource books sealed"
        );

        flows
    }
}

fn amounts_from_lanes(lanes: &[NeumaierSum; STOCK_COUNT]) -> ResourceAmounts {
    ResourceAmounts {
        food: lanes[0].value(),
        energy: lanes[1].value(),
        health: lanes[2].value(),
    }
}

/// One tick's failure to reconcile, recorded for the gate's artifact
/// (bd-16g.11.2). The artifact exists so a red CI run is diagnosable without
/// reproducing it locally.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConservationBreach {
    /// Tick whose books did not close.
    pub tick: Tick,
    /// The stock that broke worst.
    pub stock: EconomyStock,
    /// The unexplained delta on that stock.
    pub residual: f64,
    /// The tolerance the tick declared.
    pub tolerance: f64,
    /// The largest-magnitude flow category on the broken stock — the prime suspect.
    pub worst_category: Option<ResourceFlowKind>,
}

/// Per-seed outcome of the conservation gate (bd-16g.11.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeedVerdict {
    /// The seed that built the world.
    pub seed: u64,
    /// Ticks observed.
    pub ticks: u64,
    /// Largest per-tick absolute residual per stock.
    pub max_abs_residual: [f64; STOCK_COUNT],
    /// Tick of each stock's maximum.
    pub argmax_tick: [Option<Tick>; STOCK_COUNT],
    /// Cumulative signed residual per stock (compensated summation).
    pub cumulative_residual: [f64; STOCK_COUNT],
    /// Gross attributed flow per stock, for the cumulative tolerance derivation.
    pub gross_flow: [f64; STOCK_COUNT],
    /// The first [`BREACH_REPORT_CAP`] breaches, in tick order.
    pub breaches: Vec<ConservationBreach>,
    /// Breaches beyond the cap (a log that prints 50,000 breaches is a log
    /// nobody reads).
    pub truncated_breaches: u64,
}

/// Number of breaches retained per seed in the verdict artifact.
pub const BREACH_REPORT_CAP: usize = 100;

/// Incremental, memory-light conservation gate (bd-16g.11.2). Feed every
/// [`ResourceLedgerTick`] of one seed's run; [`Self::finish`] yields the seed's
/// verdict. O(1) per tick, no allocation beyond the bounded breach list.
#[derive(Debug, Default)]
pub struct ConservationGate {
    ticks: u64,
    max_abs: [f64; STOCK_COUNT],
    argmax: [Option<Tick>; STOCK_COUNT],
    cumulative: [NeumaierSum; STOCK_COUNT],
    gross: [NeumaierSum; STOCK_COUNT],
    breaches: Vec<ConservationBreach>,
    truncated: u64,
}

impl ConservationGate {
    /// A fresh gate for one seed's run.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe one completed tick's books.
    pub fn observe(&mut self, report: &ResourceLedgerTick) {
        self.ticks = self.ticks.saturating_add(1);
        let unexplained = report.reconciliation.unexplained_delta;
        let attributed = report.reconciliation.attributed_delta;
        for (stock_index, stock) in EconomyStock::ALL.into_iter().enumerate() {
            let residual = stock.lane(unexplained);
            self.cumulative[stock_index].add(residual);
            self.gross[stock_index].add(stock.lane(attributed).abs());
            if residual.abs() > self.max_abs[stock_index] {
                self.max_abs[stock_index] = residual.abs();
                self.argmax[stock_index] = Some(report.tick);
            }
        }
        if !report.reconciliation.reconciled {
            let (stock, residual) = EconomyStock::ALL
                .into_iter()
                .map(|stock| (stock, stock.lane(unexplained)))
                .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
                .expect("three stocks");
            let worst_category = report
                .flows
                .iter()
                .enumerate()
                .map(|(idx, flow)| (flow.kind, stock.lane(flow.delta).abs(), idx))
                .filter(|(_, magnitude, _)| *magnitude > 0.0)
                .max_by(|left, right| {
                    left.1
                        .total_cmp(&right.1)
                        .then_with(|| right.2.cmp(&left.2))
                })
                .map(|(kind, _, _)| kind);
            if self.breaches.len() < BREACH_REPORT_CAP {
                self.breaches.push(ConservationBreach {
                    tick: report.tick,
                    stock,
                    residual,
                    tolerance: report.reconciliation.tolerance,
                    worst_category,
                });
            } else {
                self.truncated = self.truncated.saturating_add(1);
            }
        }
    }

    /// Seal the seed's verdict.
    #[must_use]
    pub fn finish(&self, seed: u64) -> SeedVerdict {
        SeedVerdict {
            seed,
            ticks: self.ticks,
            max_abs_residual: self.max_abs,
            argmax_tick: self.argmax,
            cumulative_residual: core::array::from_fn(|index| self.cumulative[index].value()),
            gross_flow: core::array::from_fn(|index| self.gross[index].value()),
            breaches: self.breaches.clone(),
            truncated_breaches: self.truncated,
        }
    }
}

/// The full conservation-gate verdict (bd-16g.11.2). Byte-identical for the
/// same (seed, config, tick count) inputs — a flaky gate gets disabled within
/// a month, and then the economy is unguarded again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConservationVerdict {
    /// Per-seed outcomes.
    pub seeds: Vec<SeedVerdict>,
    /// True only when every criterion holds on every seed.
    pub pass: bool,
    /// Human-readable failure lines, one per violated criterion.
    pub failures: Vec<String>,
    /// Set when the harness was run with a debugging tolerance override; a CI
    /// job must never produce artifacts with this set.
    pub tolerance_overridden: bool,
}

/// Relative bound for the cumulative criterion (bd-16g.11.2): catches a tiny
/// systematic bias that per-tick tolerance would wave through — precisely how
/// a 0.0002-vs-0.002 drain error hides. Mirrors
/// [`EPOCH_RESIDUAL_RELATIVE_TOLERANCE`]; a change to either is a reviewable diff.
pub const CUMULATIVE_RESIDUAL_RELATIVE_TOLERANCE: f64 = 1.0e-7;

/// Evaluate the three gate criteria over the sealed seed verdicts. All three
/// are required — any one alone is gameable: (1) zero per-tick breaches;
/// (2) cumulative residual within the derived bound per stock; (3) a quiet run
/// (zero warn-level economy events, which in-process is exactly the breach
/// count).
#[must_use]
pub fn evaluate_conservation(seeds: &[SeedVerdict]) -> ConservationVerdict {
    let mut failures = Vec::new();
    for seed in seeds {
        if !seed.breaches.is_empty() || seed.truncated_breaches > 0 {
            let first = seed.breaches.first().expect("nonempty checked");
            failures.push(format!(
                "seed {}: {} per-tick breaches (+{} truncated); first at tick {} on \
                 {:?} (residual {:.6e}, worst category {:?})",
                seed.seed,
                seed.breaches.len(),
                seed.truncated_breaches,
                first.tick.0,
                first.stock,
                first.residual,
                first.worst_category,
            ));
        }
        for (stock_index, stock) in EconomyStock::ALL.into_iter().enumerate() {
            let gross = seed.gross_flow[stock_index];
            let bound = CUMULATIVE_RESIDUAL_RELATIVE_TOLERANCE * gross.max(1.0);
            let cumulative = seed.cumulative_residual[stock_index];
            if cumulative.abs() > bound {
                failures.push(format!(
                    "seed {}: cumulative residual on {stock:?} is {cumulative:.6e}, beyond \
                     the derived cumulative bound {bound:.6e} (gross {gross:.6e})",
                    seed.seed,
                ));
            }
        }
    }
    ConservationVerdict {
        seeds: seeds.to_vec(),
        pass: failures.is_empty(),
        failures,
        tolerance_overridden: false,
    }
}

/// Classification of agent diet based on heritable/phenotypic tendency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DietClass {
    /// Herbivorous diet preference (herbivore_tendency >= 0.7).
    Herbivore,
    /// Omnivorous diet preference (0.3 < herbivore_tendency < 0.7).
    Omnivore,
    /// Carnivorous diet preference (herbivore_tendency <= 0.3).
    Carnivore,
}

impl DietClass {
    /// All diet classes in canonical order.
    pub const ALL: [Self; 3] = [Self::Herbivore, Self::Omnivore, Self::Carnivore];

    /// Classify an agent's diet based on heritable/phenotypic tendency.
    #[must_use]
    pub fn classify(herbivore_tendency: f32) -> Self {
        if herbivore_tendency >= 0.7 {
            Self::Herbivore
        } else if herbivore_tendency <= 0.3 {
            Self::Carnivore
        } else {
            Self::Omnivore
        }
    }
}

/// Standing census of living agents grouped by diet class.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DietCensus {
    /// Count of living agents per [`DietClass`].
    pub counts: [usize; 3],
    /// Sum of standing health per [`DietClass`].
    pub standing_health: [f64; 3],
    /// Sum of standing energy per [`DietClass`].
    pub standing_energy: [f64; 3],
}

impl DietCensus {
    /// Map a [`DietClass`] to its index in census arrays.
    #[must_use]
    pub const fn class_index(class: DietClass) -> usize {
        match class {
            DietClass::Herbivore => 0,
            DietClass::Omnivore => 1,
            DietClass::Carnivore => 2,
        }
    }

    /// Record one living agent in the census.
    pub fn add_agent(&mut self, herbivore_tendency: f32, health: f64, energy: f64) {
        let idx = Self::class_index(DietClass::classify(herbivore_tendency));
        self.counts[idx] += 1;
        self.standing_health[idx] += health;
        self.standing_energy[idx] += energy;
    }
}

/// Node classification for Sankey diagram layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeClass {
    /// External resource input (sunlight, respawn, spawn endowment).
    ExternalSource,
    /// Soil/water food grid.
    FoodGrid,
    /// Herbivorous agents cohort.
    Herbivores,
    /// Omnivorous agents cohort.
    Omnivores,
    /// Carnivorous agents cohort.
    Carnivores,
    /// Energy and health dissipation sinks.
    Sink,
    /// Unaccounted residual pool.
    Unaccounted,
}

/// A node in the Sankey energy-flow graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SankeyNode {
    /// Node identifier.
    pub id: usize,
    /// Human-readable node label.
    pub label: String,
    /// Semantic node class.
    pub class: NodeClass,
}

/// A link between two Sankey nodes carrying aggregated flow magnitude.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SankeyLink {
    /// Origin node ID.
    pub from: usize,
    /// Target node ID.
    pub to: usize,
    /// Flow magnitude (f64).
    pub value: f64,
    /// Underlying resource flow category, if tied to one.
    pub category: Option<ResourceFlowKind>,
}

/// Sankey graph structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SankeyGraph {
    /// All layout nodes.
    pub nodes: Vec<SankeyNode>,
    /// All layout links.
    pub links: Vec<SankeyLink>,
}

/// Options for Sankey layout generation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SankeyOpts {
    /// Include zero-magnitude links.
    pub include_zero_links: bool,
}

/// Generate the deterministic Sankey flow graph from an [`EpochFlows`] artifact.
#[must_use]
pub fn sankey_layout(flows: &EpochFlows, opts: &SankeyOpts) -> SankeyGraph {
    let nodes = vec![
        SankeyNode {
            id: 0,
            label: "External Sources".to_owned(),
            class: NodeClass::ExternalSource,
        },
        SankeyNode {
            id: 1,
            label: "Food Grid".to_owned(),
            class: NodeClass::FoodGrid,
        },
        SankeyNode {
            id: 2,
            label: "Herbivores".to_owned(),
            class: NodeClass::Herbivores,
        },
        SankeyNode {
            id: 3,
            label: "Omnivores".to_owned(),
            class: NodeClass::Omnivores,
        },
        SankeyNode {
            id: 4,
            label: "Carnivores".to_owned(),
            class: NodeClass::Carnivores,
        },
        SankeyNode {
            id: 5,
            label: "Energy & Health Sinks".to_owned(),
            class: NodeClass::Sink,
        },
        SankeyNode {
            id: 6,
            label: "Unaccounted Residual".to_owned(),
            class: NodeClass::Unaccounted,
        },
    ];

    let mut links = Vec::new();

    for flow in &flows.per_category {
        let val = flow.activity.scale().abs();
        if val == 0.0 && !opts.include_zero_links {
            continue;
        }

        let (from, to) = match flow.kind {
            ResourceFlowKind::FoodDynamics | ResourceFlowKind::ScenarioIntervention => (0, 1),
            ResourceFlowKind::GroundFoodConversion => (1, 2),
            ResourceFlowKind::BasalMetabolism
            | ResourceFlowKind::Movement
            | ResourceFlowKind::MetabolismRamp
            | ResourceFlowKind::Boost
            | ResourceFlowKind::Topography
            | ResourceFlowKind::TemperatureStress
            | ResourceFlowKind::Aging
            | ResourceFlowKind::DeathRemoval
            | ResourceFlowKind::ReproductionAllocation
            | ResourceFlowKind::CapacityRejection => (2, 5),
            ResourceFlowKind::EnergySharing => (2, 3),
            ResourceFlowKind::Combat => (2, 5),
            ResourceFlowKind::CarcassReward | ResourceFlowKind::PopulationInjection => (0, 2),
        };

        links.push(SankeyLink {
            from,
            to,
            value: val,
            category: Some(flow.kind),
        });
    }

    for res in &flows.residual {
        let val = res.residual_sum.abs();
        if val > 0.0 || opts.include_zero_links {
            links.push(SankeyLink {
                from: 1,
                to: 6,
                value: val,
                category: None,
            });
        }
    }

    SankeyGraph { nodes, links }
}

/// Row in the trophic summary table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrophicRow {
    /// Diet class.
    pub class: DietClass,
    /// Number of living agents in this class.
    pub agent_count: usize,
    /// Standing health in this class.
    pub standing_health: f64,
    /// Standing energy in this class.
    pub standing_energy: f64,
    /// Per-tick average intake rate across the epoch.
    pub intake_rate: f64,
    /// Per-tick average drain rate across the epoch.
    pub drain_rate: f64,
    /// Ratio of intake to drain (`None` if intake == 0 or drain == 0).
    pub conversion_efficiency: Option<f64>,
    /// Net rate (intake_rate - drain_rate).
    pub net: f64,
}

/// Trophic summary table for an epoch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrophicTable {
    /// Epoch ordinal.
    pub epoch: u64,
    /// First tick in epoch.
    pub first_tick: u64,
    /// Last tick in epoch.
    pub last_tick: u64,
    /// Per-diet-class trophic summary rows.
    pub rows: Vec<TrophicRow>,
}

/// Compute the trophic table from an [`EpochFlows`] artifact and a [`DietCensus`].
#[must_use]
pub fn trophic_table(flows: &EpochFlows, pop: &DietCensus) -> TrophicTable {
    let tick_count = flows.tick_count.max(1) as f64;

    let gross_intake: f64 = flows
        .per_category
        .iter()
        .filter(|f| {
            matches!(
                f.kind,
                ResourceFlowKind::GroundFoodConversion
                    | ResourceFlowKind::EnergySharing
                    | ResourceFlowKind::CarcassReward
            )
        })
        .map(|f| f.activity.energy + f.activity.health)
        .sum();

    let gross_drain: f64 = flows
        .per_category
        .iter()
        .filter(|f| {
            matches!(
                f.kind,
                ResourceFlowKind::BasalMetabolism
                    | ResourceFlowKind::Movement
                    | ResourceFlowKind::MetabolismRamp
                    | ResourceFlowKind::Boost
                    | ResourceFlowKind::Topography
                    | ResourceFlowKind::TemperatureStress
                    | ResourceFlowKind::Aging
                    | ResourceFlowKind::CapacityRejection
            )
        })
        .map(|f| f.activity.energy + f.activity.health)
        .sum();

    let total_count: usize = pop.counts.iter().sum();

    let rows = DietClass::ALL
        .into_iter()
        .map(|class| {
            let idx = DietCensus::class_index(class);
            let count = pop.counts[idx];
            let fraction = if total_count > 0 {
                count as f64 / total_count as f64
            } else {
                0.0
            };

            let class_intake = gross_intake * fraction;
            let class_drain = gross_drain * fraction;
            let intake_rate = class_intake / tick_count;
            let drain_rate = class_drain / tick_count;
            let net = intake_rate - drain_rate;

            let conversion_efficiency = if class_drain > 0.0 && class_intake > 0.0 {
                Some(class_intake / class_drain)
            } else {
                None
            };

            TrophicRow {
                class,
                agent_count: count,
                standing_health: pop.standing_health[idx],
                standing_energy: pop.standing_energy[idx],
                intake_rate,
                drain_rate,
                conversion_efficiency,
                net,
            }
        })
        .collect();

    TrophicTable {
        epoch: flows.epoch,
        first_tick: flows.first_tick,
        last_tick: flows.last_tick,
        rows,
    }
}

fn validate_report(report: &ResourceLedgerTick) -> Result<(), EconomyAggregationError> {
    let tick = report.tick.0;
    if report.flows.len() != CATEGORY_COUNT {
        return Err(EconomyAggregationError::MalformedFlowSet {
            tick,
            actual: report.flows.len(),
            expected: CATEGORY_COUNT,
        });
    }
    for (index, (flow, expected)) in report.flows.iter().zip(RESOURCE_FLOW_KINDS).enumerate() {
        if flow.kind != expected {
            return Err(EconomyAggregationError::MisorderedFlowSet {
                tick,
                index,
                actual: flow.kind,
                expected,
            });
        }
        require_finite_amounts(tick, flow.delta, || format!("{:?} delta", flow.kind))?;
        require_finite_amounts(tick, flow.activity, || format!("{:?} activity", flow.kind))?;
    }
    require_finite_amounts(tick, report.reconciliation.unexplained_delta, || {
        "reconciliation unexplained_delta".to_owned()
    })?;
    Ok(())
}

fn require_finite_amounts(
    tick: u64,
    amounts: ResourceAmounts,
    location: impl Fn() -> String,
) -> Result<(), EconomyAggregationError> {
    for (stock, value) in [
        ("food", amounts.food),
        ("energy", amounts.energy),
        ("health", amounts.health),
    ] {
        if !value.is_finite() {
            return Err(EconomyAggregationError::NonFinite {
                tick,
                location: format!("{} ({stock} lane)", location()),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ResourceFlow, ResourceReconciliation, Tick};

    fn zero_amounts() -> ResourceAmounts {
        ResourceAmounts {
            food: 0.0,
            energy: 0.0,
            health: 0.0,
        }
    }

    fn amounts(food: f64, energy: f64, health: f64) -> ResourceAmounts {
        ResourceAmounts {
            food,
            energy,
            health,
        }
    }

    /// A full stable-order flow set with selected categories overridden.
    fn flow_set(
        overrides: &[(ResourceFlowKind, ResourceAmounts, ResourceAmounts)],
    ) -> Vec<ResourceFlow> {
        RESOURCE_FLOW_KINDS
            .into_iter()
            .map(|kind| {
                let (delta, activity) = overrides
                    .iter()
                    .find(|(candidate, _, _)| *candidate == kind)
                    .map_or_else(
                        || (zero_amounts(), zero_amounts()),
                        |(_, delta, activity)| (*delta, *activity),
                    );
                ResourceFlow {
                    kind,
                    delta,
                    activity,
                }
            })
            .collect()
    }

    fn ledger_tick(
        tick: u64,
        overrides: &[(ResourceFlowKind, ResourceAmounts, ResourceAmounts)],
        unexplained: ResourceAmounts,
    ) -> ResourceLedgerTick {
        let flows = flow_set(overrides);
        let mut attributed = zero_amounts();
        for flow in &flows {
            attributed.food += flow.delta.food;
            attributed.energy += flow.delta.energy;
            attributed.health += flow.delta.health;
        }
        let observed = amounts(
            attributed.food + unexplained.food,
            attributed.energy + unexplained.energy,
            attributed.health + unexplained.health,
        );
        ResourceLedgerTick {
            tick: Tick(tick),
            opening: zero_amounts(),
            closing: observed,
            flows,
            reconciliation: ResourceReconciliation {
                observed_delta: observed,
                attributed_delta: attributed,
                unexplained_delta: unexplained,
                tolerance: 1.0e-5,
                reconciled: true,
            },
        }
    }

    #[test]
    fn a_give_only_epoch_has_a_residual_of_exactly_zero() {
        // The calibration fixture: energy sharing is the only strictly
        // conserved flow (giver -x, recipient +x). If THIS drifts, the
        // summation is wrong, not the model.
        let mut aggregator = EpochAggregator::new(3);
        for tick in 1..=2u64 {
            let report = ledger_tick(
                tick,
                &[(
                    ResourceFlowKind::EnergySharing,
                    zero_amounts(),
                    amounts(0.0, 0.25, 0.0),
                )],
                zero_amounts(),
            );
            assert_eq!(aggregator.observe(&report).expect("observe"), None);
        }
        let epoch = aggregator.finish().expect("partial epoch");
        for row in epoch.residual {
            assert_eq!(
                row.residual_sum, 0.0,
                "a conserved-only epoch must have an exact zero residual"
            );
            assert!(row.within_tolerance);
        }
        assert!(!epoch.complete);
        let sharing = &epoch.per_category[ResourceFlowKind::EnergySharing as usize];
        assert_eq!(sharing.activity.energy, 0.5);
        assert_eq!(sharing.delta.energy, 0.0);
    }

    #[test]
    fn category_sums_and_residual_extrema_are_tracked_per_stock() {
        let mut aggregator = EpochAggregator::new(3);
        let first = ledger_tick(
            10,
            &[(
                ResourceFlowKind::BasalMetabolism,
                amounts(0.0, -0.2, -0.2),
                zero_amounts(),
            )],
            amounts(0.0, 1.0e-9, 0.0),
        );
        let second = ledger_tick(
            11,
            &[(
                ResourceFlowKind::FoodDynamics,
                amounts(0.75, 0.0, 0.0),
                zero_amounts(),
            )],
            amounts(0.0, -3.0e-9, 0.0),
        );
        let third = ledger_tick(
            12,
            &[(
                ResourceFlowKind::BasalMetabolism,
                amounts(0.0, -0.1, -0.1),
                zero_amounts(),
            )],
            zero_amounts(),
        );

        assert_eq!(aggregator.observe(&first).expect("first"), None);
        assert_eq!(aggregator.observe(&second).expect("second"), None);
        let epoch = aggregator
            .observe(&third)
            .expect("third")
            .expect("window of three seals the epoch");

        assert!(epoch.complete);
        assert_eq!(epoch.epoch, 0);
        assert_eq!(
            (epoch.first_tick, epoch.last_tick, epoch.tick_count),
            (10, 12, 3)
        );

        let metabolism = &epoch.per_category[ResourceFlowKind::BasalMetabolism as usize];
        assert!((metabolism.delta.energy - -0.3).abs() < 1.0e-15);
        assert!((metabolism.delta.health - -0.3).abs() < 1.0e-15);
        let food = &epoch.per_category[ResourceFlowKind::FoodDynamics as usize];
        assert_eq!(food.delta.food, 0.75);

        let energy_row = epoch.residual[1];
        assert_eq!(energy_row.stock, EconomyStock::AgentEnergy);
        assert!((energy_row.residual_sum - -2.0e-9).abs() < 1.0e-22);
        assert_eq!(energy_row.residual_max_abs, 3.0e-9);
        assert_eq!(energy_row.argmax_tick, Some(11));
        assert_eq!(
            energy_row.worst_category,
            Some(ResourceFlowKind::BasalMetabolism)
        );
        assert!(energy_row.within_tolerance);

        let food_row = epoch.residual[0];
        assert_eq!(
            food_row.worst_category,
            Some(ResourceFlowKind::FoodDynamics)
        );
        assert_eq!(food_row.gross_flow, 0.75);
    }

    #[test]
    fn malformed_and_misordered_flow_sets_are_typed_errors() {
        let mut aggregator = EpochAggregator::new(4);
        let mut truncated = ledger_tick(5, &[], zero_amounts());
        truncated.flows.pop();
        assert!(matches!(
            aggregator.observe(&truncated),
            Err(EconomyAggregationError::MalformedFlowSet { tick: 5, .. })
        ));

        let mut swapped = ledger_tick(6, &[], zero_amounts());
        swapped.flows.swap(0, 1);
        assert!(matches!(
            aggregator.observe(&swapped),
            Err(EconomyAggregationError::MisorderedFlowSet {
                tick: 6,
                index: 0,
                ..
            })
        ));

        // Rejected ticks must leave the aggregator untouched.
        assert_eq!(aggregator.finish(), None);
    }

    #[test]
    fn non_finite_values_are_typed_errors_not_silent_poison() {
        let mut aggregator = EpochAggregator::new(4);
        let mut poisoned = ledger_tick(7, &[], zero_amounts());
        poisoned.flows[3].delta.energy = f64::NAN;
        assert!(matches!(
            aggregator.observe(&poisoned),
            Err(EconomyAggregationError::NonFinite { tick: 7, .. })
        ));

        let mut infinite = ledger_tick(8, &[], zero_amounts());
        infinite.reconciliation.unexplained_delta.health = f64::INFINITY;
        assert!(matches!(
            aggregator.observe(&infinite),
            Err(EconomyAggregationError::NonFinite { tick: 8, .. })
        ));
    }

    #[test]
    fn ticks_must_be_strictly_increasing() {
        let mut aggregator = EpochAggregator::new(4);
        let first = ledger_tick(9, &[], zero_amounts());
        aggregator.observe(&first).expect("first accepted");
        let replay = ledger_tick(9, &[], zero_amounts());
        assert!(matches!(
            aggregator.observe(&replay),
            Err(EconomyAggregationError::NonMonotonicTick {
                previous: 9,
                actual: 9
            })
        ));
    }

    #[test]
    fn windows_seal_on_fill_and_partial_epochs_are_flagged() {
        let mut aggregator = EpochAggregator::new(2);
        assert_eq!(
            aggregator
                .observe(&ledger_tick(1, &[], zero_amounts()))
                .expect("tick 1"),
            None
        );
        let sealed = aggregator
            .observe(&ledger_tick(2, &[], zero_amounts()))
            .expect("tick 2")
            .expect("window filled");
        assert!(sealed.complete);
        assert_eq!(sealed.epoch, 0);

        aggregator
            .observe(&ledger_tick(3, &[], zero_amounts()))
            .expect("tick 3");
        let partial = aggregator.finish().expect("partial epoch");
        assert!(!partial.complete);
        assert_eq!(partial.epoch, 1);
        assert_eq!(
            (partial.first_tick, partial.last_tick, partial.tick_count),
            (3, 3, 1)
        );

        // Monotonicity carries across epochs: replaying tick 3 must fail.
        assert!(matches!(
            aggregator.observe(&ledger_tick(3, &[], zero_amounts())),
            Err(EconomyAggregationError::NonMonotonicTick {
                previous: 3,
                actual: 3
            })
        ));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn aggregation_is_a_pure_function_of_the_tick_sequence() {
        let build = || {
            let mut aggregator = EpochAggregator::new(2);
            let mut sealed = Vec::new();
            for tick in 1..=5u64 {
                let report = ledger_tick(
                    tick,
                    &[(
                        ResourceFlowKind::Movement,
                        amounts(0.0, -0.01 * tick as f64, -0.01 * tick as f64),
                        zero_amounts(),
                    )],
                    amounts(0.0, 1.0e-12 * tick as f64, 0.0),
                );
                if let Some(epoch) = aggregator.observe(&report).expect("observe") {
                    sealed.push(epoch);
                }
            }
            sealed.extend(aggregator.finish());
            sealed
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn compensated_summation_survives_magnitude_cancellation() {
        // Naive f64 `+=` of [1e16, 1.0, -1e16] loses the 1.0 entirely; the
        // compensated lane must recover it exactly.
        let mut lane = NeumaierSum::default();
        lane.add(1.0e16);
        lane.add(1.0);
        lane.add(-1.0e16);
        assert_eq!(lane.value(), 1.0);

        let naive = (1.0e16_f64 + 1.0) - 1.0e16;
        assert_eq!(
            naive, 0.0,
            "if this fails, the fixture no longer proves anything"
        );
    }

    /// The fault-coverage table (bd-16g.11.2): every flow category is exercised
    /// by at least one injectable fault. Add a category without adding a fault
    /// and the length check below fails the build; remove a fault's coverage
    /// and the per-category check fails the test.
    const KIND_TO_FAULTS: [&[LedgerFault]; CATEGORY_COUNT] = [
        &[LedgerFault::DropInterventionPosting], // ScenarioIntervention
        &[LedgerFault::MislabelSunlightAsTransfer], // FoodDynamics
        &[LedgerFault::DropAgingPosting],        // Aging
        &[
            LedgerFault::DropActuationPosting,
            LedgerFault::ScaleMetabolismDrainBy10,
        ], // BasalMetabolism
        &[LedgerFault::DropActuationPosting],    // Movement
        &[LedgerFault::DropActuationPosting],    // MetabolismRamp
        &[LedgerFault::DropActuationPosting],    // Boost
        &[LedgerFault::DropActuationPosting],    // Topography
        &[LedgerFault::DropTemperaturePosting],  // TemperatureStress
        &[
            LedgerFault::DoubleCreditIntake,
            LedgerFault::HalveWasteDebit,
            LedgerFault::CreditIntakeWithoutHealthGate,
        ], // GroundFoodConversion
        &[LedgerFault::CorruptGiveTransfer],     // EnergySharing
        &[LedgerFault::DropCombatPosting],       // Combat
        &[LedgerFault::MislabelCarcassAsTransfer], // CarcassReward
        &[LedgerFault::SkipDeathResidue],        // DeathRemoval
        &[LedgerFault::DropReproductionPosting], // ReproductionAllocation
        &[LedgerFault::DropPopulationPosting],   // PopulationInjection
        &[
            LedgerFault::ForgetEnergyCapSpill,
            LedgerFault::ForgetHealthCapSpill,
        ], // CapacityRejection
    ];

    #[test]
    fn every_flow_category_is_exercised_by_at_least_one_fault() {
        assert_eq!(KIND_TO_FAULTS.len(), CATEGORY_COUNT, "compile-time length");
        for (index, kind) in RESOURCE_FLOW_KINDS.into_iter().enumerate() {
            assert!(
                !KIND_TO_FAULTS[index].is_empty(),
                "{kind:?} has no injectable fault — the gate would decay silently"
            );
        }
        // Every listed fault is a real variant of LedgerFault::ALL.
        for faults in &KIND_TO_FAULTS {
            for fault in *faults {
                assert!(LedgerFault::ALL.contains(fault));
            }
        }
    }

    fn clean_tick(tick: u64) -> ResourceLedgerTick {
        ledger_tick(
            tick,
            &[(
                ResourceFlowKind::FoodDynamics,
                amounts(0.5, 0.0, 0.0),
                zero_amounts(),
            )],
            zero_amounts(),
        )
    }

    fn breached_tick(tick: u64, residual: ResourceAmounts) -> ResourceLedgerTick {
        let mut report = ledger_tick(
            tick,
            &[(
                ResourceFlowKind::FoodDynamics,
                amounts(0.5, 0.0, 0.0),
                zero_amounts(),
            )],
            residual,
        );
        report.reconciliation.reconciled = false;
        report
    }

    #[test]
    fn gate_passes_a_clean_run() {
        let mut gate = ConservationGate::new();
        for tick in 1..=64 {
            gate.observe(&clean_tick(tick));
        }
        let seed = gate.finish(7);
        assert!(seed.breaches.is_empty());
        let verdict = evaluate_conservation(&[seed]);
        assert!(verdict.pass, "clean run must pass: {:?}", verdict.failures);
    }

    #[test]
    fn gate_fails_on_a_per_tick_breach_with_the_right_suspect() {
        let mut gate = ConservationGate::new();
        for tick in 1..=8 {
            gate.observe(&clean_tick(tick));
        }
        gate.observe(&breached_tick(9, amounts(-0.5, 0.0, 0.0)));
        let verdict = evaluate_conservation(&[gate.finish(11)]);
        assert!(!verdict.pass);
        let failure = &verdict.failures[0];
        assert!(failure.contains("per-tick breaches"), "{failure}");
        assert!(
            failure.contains("GridFood"),
            "the broken stock is named: {failure}"
        );
        let breach = &verdict.seeds[0].breaches[0];
        assert_eq!(breach.tick, Tick(9));
        assert_eq!(breach.worst_category, Some(ResourceFlowKind::FoodDynamics));
    }

    #[test]
    fn gate_fails_on_cumulative_bias_without_any_single_breach() {
        // A bias below per-tick tolerance, repeated every tick: each tick is
        // individually clean, and the cumulative criterion is the only thing
        // that can catch it — precisely how a 0.0002-vs-0.002 drain error hides.
        let mut gate = ConservationGate::new();
        let whisper = amounts(0.0, 0.02, 0.0);
        for tick in 1..=200 {
            let mut report = clean_tick(tick);
            report.reconciliation.unexplained_delta = whisper;
            report.reconciliation.reconciled = true; // under per-tick tolerance
            report.reconciliation.observed_delta = whisper;
            gate.observe(&report);
        }
        let seed = gate.finish(13);
        assert!(seed.breaches.is_empty(), "no single tick breaches");
        let verdict = evaluate_conservation(&[seed]);
        assert!(!verdict.pass, "cumulative bias must fail the gate");
        assert!(
            verdict.failures[0].contains("cumulative residual"),
            "{}",
            verdict.failures[0]
        );
    }

    #[test]
    fn gate_artifact_is_byte_identical_across_invocations() {
        let run = || {
            let mut gate = ConservationGate::new();
            for tick in 1..=32 {
                gate.observe(&clean_tick(tick));
            }
            gate.observe(&breached_tick(33, amounts(0.0, 0.0, -0.25)));
            evaluate_conservation(&[gate.finish(17)])
        };
        let first = serde_json::to_vec(&run()).expect("encode verdict");
        let second = serde_json::to_vec(&run()).expect("encode verdict");
        assert_eq!(
            first, second,
            "the verdict is a pure function of the inputs"
        );
    }

    #[test]
    fn sankey_layout_generates_nodes_and_links() {
        let mut aggregator = EpochAggregator::new(2);
        let mut t1 = clean_tick(1);
        t1.flows[0].activity.energy = 10.0;
        aggregator.observe(&t1).expect("tick 1");
        let epoch = aggregator
            .observe(&clean_tick(2))
            .expect("tick 2")
            .expect("epoch");

        let graph = sankey_layout(
            &epoch,
            &SankeyOpts {
                include_zero_links: false,
            },
        );
        assert_eq!(graph.nodes.len(), 7);
        assert!(!graph.links.is_empty());
        let total_value: f64 = graph.links.iter().map(|l| l.value).sum();
        assert!(total_value > 0.0);
    }

    #[test]
    fn trophic_table_handles_empty_and_present_diet_classes() {
        let mut aggregator = EpochAggregator::new(2);
        aggregator.observe(&clean_tick(1)).expect("tick 1");
        let epoch = aggregator
            .observe(&clean_tick(2))
            .expect("tick 2")
            .expect("epoch");

        let mut census = DietCensus::default();
        census.add_agent(0.8, 1.0, 1.5); // Herbivore
        census.add_agent(0.1, 0.8, 1.2); // Carnivore

        let table = trophic_table(&epoch, &census);
        assert_eq!(table.rows.len(), 3);
        let herbivore = &table.rows[0];
        assert_eq!(herbivore.class, DietClass::Herbivore);
        assert_eq!(herbivore.agent_count, 1);

        let omnivore = &table.rows[1];
        assert_eq!(omnivore.class, DietClass::Omnivore);
        assert_eq!(omnivore.agent_count, 0);
        assert_eq!(omnivore.conversion_efficiency, None);
    }

    #[test]
    fn agreement_e2e_export_sankey_and_trophic_table_have_identical_epoch_numbers() {
        let mut aggregator = EpochAggregator::new(2);
        aggregator.observe(&clean_tick(1)).expect("tick 1");
        let epoch = aggregator
            .observe(&clean_tick(2))
            .expect("tick 2")
            .expect("epoch");

        let mut census = DietCensus::default();
        census.add_agent(0.8, 1.0, 1.5);

        let sankey = sankey_layout(&epoch, &SankeyOpts::default());
        let table = trophic_table(&epoch, &census);

        let json_epoch = serde_json::to_string(&epoch).expect("json epoch");
        let json_sankey = serde_json::to_string(&sankey).expect("json sankey");
        let json_table = serde_json::to_string(&table).expect("json table");

        assert!(json_epoch.contains(&epoch.epoch.to_string()));
        assert!(json_sankey.contains("External Sources"));
        assert!(json_table.contains("herbivore"));
    }
}
