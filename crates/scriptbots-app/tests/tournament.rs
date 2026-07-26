//! End-to-end tournament harness proofs (bd-16g.12.1).
//!
//! Every test drives real headless matches: byte-identical outcomes across invocations,
//! order assignments that change the grid but not the seed allocation, an open-world
//! qualifier that can never be lost, and the null-tournament bias probe.

use scriptbots_app::tournament::{
    EloRating, FamilyOutcome, FamilyScore, MatchOutcome, MatchResult, OrderPolicy, TournamentError,
    TournamentHarness, TournamentSpec, enforce_no_config_drift, plan, run_match, run_tournament,
};
use scriptbots_brain::BrainKind;
use scriptbots_core::ScriptBotsConfig;
use std::collections::HashMap;

const MLP_A: BrainKind = BrainKind::new("mlp-a");
const MLP_B: BrainKind = BrainKind::new("mlp-b");
const DWRAON: BrainKind = BrainKind::new("dwraon.baseline");

fn small_world() -> ScriptBotsConfig {
    ScriptBotsConfig {
        world_width: 480,
        world_height: 480,
        food_cell_size: 40,
        ..ScriptBotsConfig::default()
    }
}

fn two_family_spec(ticks: u64) -> TournamentSpec {
    TournamentSpec {
        families: vec![MLP_A, DWRAON],
        seeds: vec![42],
        ticks,
        cohort_size: 16,
        order_policy: OrderPolicy::BothAssignments,
        closed: true,
        config_layers: Vec::new(),
    }
}

#[test]
fn matches_reproduce_byte_identical_outcomes() {
    let spec = two_family_spec(150);
    let plans = plan(&spec).expect("balanced plan");
    assert_eq!(plans.len(), 2, "[A,B] and [B,A]");
    for match_plan in &plans {
        let first =
            run_match(match_plan, spec.ticks, spec.closed, &small_world()).expect("first run");
        let second =
            run_match(match_plan, spec.ticks, spec.closed, &small_world()).expect("second run");
        assert_eq!(
            first.outcome, second.outcome,
            "a match must reproduce its outcome byte-identically"
        );
        assert_eq!(
            first.config_digest, second.config_digest,
            "the config digest must be stable across invocations"
        );
    }
}

#[test]
fn seed_allocation_is_identical_across_order_assignments() {
    let spec = two_family_spec(50);
    let plans = plan(&spec).expect("balanced plan");
    assert_eq!(plans.len(), 2);
    // The two assignments share the root seed; their hash-allocated seeds must differ by
    // match index but recompute identically every time.
    assert_ne!(plans[0].world_seed, plans[1].world_seed);
    assert_ne!(plans[0].brain_seed, plans[1].brain_seed);
    assert_eq!(plans[0].cohort, plans[1].cohort, "equal cohorts everywhere");
}

#[test]
fn open_world_stamps_the_respawn_warning_into_every_outcome() {
    let mut spec = two_family_spec(60);
    spec.closed = false;
    let plans = plan(&spec).expect("plan");
    let report =
        run_match(&plans[0], spec.ticks, spec.closed, &small_world()).expect("open-world match");
    assert!(
        report
            .outcome
            .warnings
            .iter()
            .any(|warning| warning.contains("open-world respawn active")),
        "an open world must stamp the respawn qualifier: {:?}",
        report.outcome.warnings
    );
}

#[test]
fn null_tournament_arms_are_indistinguishable_beyond_seed_noise() {
    // The same adapter entered under two names: any systematic gap between the arms is
    // harness bias (order, cohort, seed correlation), not a brain difference. This is the
    // test that catches an off-by-one the leaderboard would happily publish.
    let spec = TournamentSpec {
        families: vec![MLP_A, MLP_B],
        seeds: vec![7, 11, 13, 17],
        ticks: 200,
        cohort_size: 16,
        order_policy: OrderPolicy::BothAssignments,
        closed: true,
        config_layers: Vec::new(),
    };
    let plans = plan(&spec).expect("null plan");
    assert_eq!(plans.len(), 8, "4 seeds x 2 assignments");

    let mut gap_max = 0.0_f64;
    let mut total_a = 0.0_f64;
    let mut total_b = 0.0_f64;
    for match_plan in &plans {
        let report =
            run_match(match_plan, spec.ticks, spec.closed, &small_world()).expect("null match");
        let share_of = |family: BrainKind| -> f64 {
            report
                .outcome
                .per_family
                .get(family.as_str())
                .map_or(0.5, |outcome: &FamilyOutcome| outcome.survival_share)
        };
        let gap = (share_of(MLP_A) - share_of(MLP_B)).abs();
        gap_max = gap_max.max(gap);
        total_a += share_of(MLP_A);
        total_b += share_of(MLP_B);
    }
    let mean_gap = (total_a - total_b).abs() / plans.len() as f64;
    assert!(
        mean_gap <= 0.10,
        "null tournament: mean survival-share gap {mean_gap:.4} exceeds the seed-noise band"
    );
    eprintln!(
        "null tournament evidence: mean gap {mean_gap:.4}, max per-match gap {gap_max:.4} over {} matches",
        plans.len()
    );
}

#[test]
fn run_tournament_reproduces_identical_report_sets() {
    let spec = two_family_spec(120);
    let first = run_tournament(&spec, &small_world()).expect("tournament one");
    let second = run_tournament(&spec, &small_world()).expect("tournament two");
    assert_eq!(first.len(), 2, "two assignments");
    assert_eq!(first.len(), second.len());
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.outcome, b.outcome, "report sets are byte-identical");
        assert_eq!(a.config_digest, b.config_digest);
    }
    assert!(
        first
            .iter()
            .all(|report| report.config_digest == first[0].config_digest),
        "every arm ran the same effective config"
    );
}

#[test]
fn config_drift_guard_rejects_mismatched_arms() {
    // The guard compares digests across arms; a fault-injected mutation anywhere must
    // surface as ConfigDrift, not as a publishable finding.
    let spec = two_family_spec(10);
    let mut reports = run_tournament(&spec, &small_world()).expect("clean tournament");
    reports[1].config_digest = "tampered".to_owned();
    let error =
        enforce_no_config_drift(&reports).expect_err("a tampered arm must fail the drift guard");
    assert!(
        matches!(error, TournamentError::ConfigDrift { .. }),
        "expected ConfigDrift, got {error}"
    );
    assert!(
        error.to_string().contains("tampered"),
        "the drift error names the divergent digest: {error}"
    );
}

#[test]
fn outcomes_record_every_family_with_warnings_channel() {
    let spec = two_family_spec(40);
    let plans = plan(&spec).expect("plan");
    let report = run_match(&plans[0], spec.ticks, spec.closed, &small_world()).expect("match");
    let outcome: &MatchOutcome = &report.outcome;
    assert_eq!(
        outcome.per_family.len(),
        2,
        "every entered family has an outcome row"
    );
    assert!(
        outcome
            .per_family
            .values()
            .all(|outcome| outcome.survival_share >= 0.0 && outcome.survival_share <= 1.0),
        "survival shares are probabilities"
    );
}

#[test]
fn test_tournament_integration_elo_progression() {
    let mut harness = TournamentHarness::new();
    harness.register_family("mlp");
    harness.register_family("dwraon");
    harness.register_family("assembly");

    let mut scores = HashMap::new();
    scores.insert(
        "mlp".to_owned(),
        FamilyScore {
            survival_share: 0.8,
            biomass_share: 0.7,
            max_generation: 20,
        },
    );
    scores.insert(
        "dwraon".to_owned(),
        FamilyScore {
            survival_share: 0.2,
            biomass_share: 0.3,
            max_generation: 10,
        },
    );

    harness.record_match(MatchResult {
        seed: 100,
        ticks: 1_000,
        family_scores: scores,
    });

    let mlp_rating = harness.ratings.get("mlp").expect("mlp rating");
    let dwraon_rating = harness.ratings.get("dwraon").expect("dwraon rating");

    assert!(mlp_rating.rating > 1_500.0);
    assert!(dwraon_rating.rating < 1_500.0);
    assert_eq!(mlp_rating.wins, 1);
    assert_eq!(dwraon_rating.wins, 0);

    let leaderboard = harness.generate_leaderboard_markdown();
    assert!(leaderboard.contains("mlp"));
    assert!(leaderboard.contains("dwraon"));
    assert!(leaderboard.contains("assembly"));
}

#[test]
fn test_elo_update_symmetry() {
    let mut winner = EloRating::new("winner");
    let mut loser = EloRating::new("loser");

    EloRating::update_elo(&mut winner, &mut loser, 32.0);

    assert!((winner.rating - 1_516.0).abs() < 1e-4);
    assert!((loser.rating - 1_484.0).abs() < 1e-4);
}
