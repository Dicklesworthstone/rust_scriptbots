//! Archipelago determinism gate: identical science across Rayon thread counts
//! and island declaration order (bd-ydmr, bd-16g.5 acceptance (a)).
//!
//! # Why this test lives in `scriptbots-app`
//!
//! Because it is the only place it is not vacuous. `scriptbots-runtime` declares
//! `scriptbots-core` with `default-features = false`, so core's `parallel`
//! feature is OFF there and the tick pipeline never touches Rayon at all. A
//! thread-count determinism test written in that crate would vary a thread pool
//! nothing uses and pass unconditionally. `scriptbots-app` takes core with
//! default features, so `parallel` is on and the pools below are real.
//!
//! That trap is guarded rather than merely documented:
//! [`assert_pool_is_real`] asserts `rayon::current_num_threads()` inside each
//! pool, so if the feature is ever turned off, or the pool stops being honoured,
//! this gate fails instead of silently proving nothing.
//!
//! # What it catches that nothing else does
//!
//! Order-dependence. Every single-threaded test in this repository passes a
//! world that quietly depends on the order Rayon happened to finish work in;
//! this one does not. The cells are chosen for that: `islands > threads` forces
//! work stealing, and a non-divisor thread count breaks any chunking assumption.
//! Island-order permutation catches order-dependent seeding and UID allocation,
//! which no thread-count variation will ever surface.
//!
//! # What it deliberately does not claim
//!
//! `WorldDigestV1` is a regression oracle for ONE PINNED BUILD LANE. This gate
//! compares runs of the SAME BINARY across thread counts and declaration orders
//! — exactly the claim bd-16g.5 makes. It is not a cross-platform or
//! cross-toolchain reproducibility promise, and stretching it into one would
//! assert something the digest cannot support.

use scriptbots_core::ScriptBotsConfig;
use scriptbots_runtime::{
    Archipelago, ArchipelagoConfig, ArchipelagoMigration, HostCoreOptions, IslandId, IslandSpec,
    Topology, migrator::EmigrantSelectionRule,
};
use std::num::NonZeroU64;

const ISLANDS: u32 = 8;
const BARRIERS: usize = 4;
const BARRIER_INTERVAL: u64 = 25;
const MASTER_SEED: u64 = 0xDE7E_C7ED_5EED_0001;

/// One run's complete observable science: every island's digest, and every
/// migration that happened, in order.
#[derive(Debug, PartialEq)]
struct RunEvidence {
    /// `(island, digest.overall)` in ascending island order.
    digests: Vec<(IslandId, String)>,
    /// `(barrier_tick, from, to)` for every applied move, in application order.
    migrations: Vec<(u64, String, String)>,
}

fn island_config() -> ScriptBotsConfig {
    ScriptBotsConfig {
        world_width: 600,
        world_height: 300,
        food_cell_size: 50,
        rng_seed: None,
        persistence_interval: 0,
        population_minimum: 12,
        population_spawn_interval: 5,
        ..ScriptBotsConfig::default()
    }
}

/// Build the archipelago config with islands declared in `order`.
///
/// The order is a parameter precisely so a permuted declaration can be shown to
/// produce identical science; the archipelago sorts by id at construction, and
/// this is what proves that sort is load-bearing rather than incidental.
fn config_with_order(order: &[u32]) -> ArchipelagoConfig {
    ArchipelagoConfig {
        islands: order
            .iter()
            .map(|&id| IslandSpec {
                id: IslandId(id),
                label: format!("det-island-{id}"),
                config: island_config(),
            })
            .collect(),
        topology: Topology::Ring,
        barrier_interval: NonZeroU64::new(BARRIER_INTERVAL).expect("nonzero interval"),
        master_seed: MASTER_SEED,
        host_options: HostCoreOptions::default(),
        migration: Some(ArchipelagoMigration {
            interval_ticks: BARRIER_INTERVAL,
            emigrants_per_edge: 1,
            selection_rule: EmigrantSelectionRule::Fittest,
        }),
    }
}

/// Run one archipelago to completion and collect everything observable.
fn run(order: &[u32]) -> RunEvidence {
    let mut archipelago = Archipelago::new(config_with_order(order)).expect("valid archipelago");
    let mut migrations = Vec::new();
    for barrier in 0..BARRIERS {
        let stepped = archipelago.step_to_barrier();
        assert!(stepped.is_ok(), "barrier {barrier} must step: {stepped:?}");
        let report = stepped.expect("checked immediately above");
        if let Some(migration) = report.migration {
            for applied in &migration.moves {
                migrations.push((
                    migration.barrier_tick.0,
                    applied.from.to_string(),
                    applied.to.to_string(),
                ));
            }
        }
    }
    let digests = (0..ISLANDS)
        .map(IslandId)
        .map(|island| {
            let computed = archipelago.island_digest(island);
            assert!(computed.is_ok(), "island {island} digest: {computed:?}");
            let digest = computed.expect("checked immediately above");
            (island, digest.overall)
        })
        .collect();
    RunEvidence {
        digests,
        migrations,
    }
}

/// THE ANTI-VACUITY GUARD. A pool that is not honoured makes every assertion
/// below meaningless, and the failure would look exactly like success.
fn assert_pool_is_real(threads: usize) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("bounded Rayon pool");
    let observed = pool.install(rayon::current_num_threads);
    assert_eq!(
        observed, threads,
        "Rayon reported {observed} threads inside a {threads}-thread pool; either the \
         `parallel` feature is off in this crate or the pool is not being honoured, and \
         in either case this gate proves nothing"
    );
}

/// Run `order` inside a bounded Rayon pool of `threads`.
fn run_with_threads(order: &[u32], threads: usize) -> RunEvidence {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("bounded Rayon pool");
    pool.install(|| run(order))
}

/// Report the FIRST divergence by island rather than "digests differ", so a
/// failure names the place to look.
fn assert_same_science(baseline: &RunEvidence, candidate: &RunEvidence, cell: &str) {
    for ((left_island, left), (right_island, right)) in
        baseline.digests.iter().zip(&candidate.digests)
    {
        assert_eq!(
            left_island, right_island,
            "{cell}: island ordering in the evidence differs, which makes the digest \
             comparison meaningless"
        );
        assert_eq!(
            left, right,
            "{cell}: FIRST DIVERGENCE at {left_island}: baseline {left}, candidate {right}"
        );
    }
    assert_eq!(
        baseline.digests.len(),
        candidate.digests.len(),
        "{cell}: island count differs"
    );
    assert_eq!(
        baseline.migrations, candidate.migrations,
        "{cell}: the migration record diverged; per-island digests agreeing while the \
         migration order differs would mean the barrier is not deterministic even though \
         each island is"
    );
}

/// Islands must reach identical science at 1, 8, 4 and 3 Rayon threads, and
/// under a permuted declaration order.
#[test]
fn bd_ydmr_archipelago_science_is_identical_across_thread_counts_and_island_order() {
    for threads in [1usize, 8, 4, 3] {
        assert_pool_is_real(threads);
    }

    let ascending: Vec<u32> = (0..ISLANDS).collect();
    let baseline = run_with_threads(&ascending, 1);

    // Non-vacuity: the run must actually contain science to compare.
    assert_eq!(baseline.digests.len(), ISLANDS as usize);
    assert!(
        !baseline.migrations.is_empty(),
        "no organism migrated, so the migration half of this gate compares two empty lists"
    );
    let distinct: std::collections::BTreeSet<&String> =
        baseline.digests.iter().map(|(_, digest)| digest).collect();
    assert!(
        distinct.len() > 1,
        "every island produced the same digest, so this gate would pass even if the \
         digest ignored island state entirely"
    );

    // islands > threads forces work stealing; 3 is a non-divisor of 8 and breaks
    // any chunking assumption. Both are cells that a naive 1-vs-N check misses.
    for threads in [8usize, 4, 3] {
        let candidate = run_with_threads(&ascending, threads);
        assert_same_science(&baseline, &candidate, &format!("threads={threads}"));
    }

    // Declaration order must not reach the science. This catches order-dependent
    // seeding and UID allocation, which no thread-count variation surfaces.
    let permuted: Vec<u32> = (0..ISLANDS).rev().collect();
    let reordered = run_with_threads(&permuted, 4);
    assert_same_science(&baseline, &reordered, "island order reversed");
}

/// The gate must be able to FAIL: a different master seed must move the science.
///
/// Without this, every assertion above would also hold for a digest that ignored
/// the world, and the matrix would be proving that a constant equals itself.
#[test]
fn bd_ydmr_the_determinism_gate_can_detect_a_changed_run() {
    let ascending: Vec<u32> = (0..ISLANDS).collect();
    let baseline = run_with_threads(&ascending, 1);

    let mut altered = config_with_order(&ascending);
    altered.master_seed = MASTER_SEED ^ 0xFFFF_FFFF;
    let mut archipelago = Archipelago::new(altered).expect("valid archipelago");
    for _ in 0..BARRIERS {
        archipelago.step_to_barrier().expect("barrier steps");
    }
    let changed: Vec<(IslandId, String)> = (0..ISLANDS)
        .map(IslandId)
        .map(|island| {
            let digest = archipelago.island_digest(island).expect("digest");
            (island, digest.overall)
        })
        .collect();

    assert_ne!(
        baseline.digests, changed,
        "a different master seed produced identical per-island digests, so these digests \
         do not depend on the run and every comparison in this file is vacuous"
    );
}

/// Mutating `RAYON_NUM_THREADS` at runtime does NOT change the pool, which is why
/// the determinism matrix builds explicit bounded pools instead (bd-0dmc).
///
/// The archipelago det-check CLI originally varied its "thread matrix" by calling
/// `std::env::set_var("RAYON_NUM_THREADS", ...)` between cells. Rayon reads that
/// variable once, when the global pool is first initialized; every later write is
/// inert. A matrix built that way runs every cell at the same width while
/// printing that it covered [1, 4, 8, 3] — a success signal claiming more than it
/// observed, in a diagnostic tool.
///
/// This pins the underlying behaviour so nobody reintroduces that approach.
#[test]
fn bd_0dmc_setting_rayon_num_threads_at_runtime_does_not_change_the_pool() {
    // Force the global pool to exist before touching the variable.
    let initial = rayon::current_num_threads();
    assert!(initial > 0, "rayon must report a live global pool");

    for requested in ["1", "3", "8"] {
        // SAFETY: a well-formed Unicode value; this mirrors exactly what the old
        // det-check did, which is the point of the test.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("RAYON_NUM_THREADS", requested);
        }
        assert_eq!(
            rayon::current_num_threads(),
            initial,
            "setting RAYON_NUM_THREADS={requested} after the global pool exists changed \
             the reported width; if rayon ever gains that behaviour this guard should be \
             revisited, but until then a det-check must use explicit bounded pools"
        );
    }

    // And the supported way DOES work, which is what makes the matrix real.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("bounded pool");
    assert_eq!(pool.install(rayon::current_num_threads), 2);
}
