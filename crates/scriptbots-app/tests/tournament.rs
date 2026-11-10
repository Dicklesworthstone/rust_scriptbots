//! End-to-end tournament harness proofs (bd-16g.12.1).
//!
//! Every test drives real headless matches: byte-identical outcomes across invocations,
//! order assignments that change the grid but not the seed allocation, an open-world
//! qualifier that can never be lost, and the null-tournament bias probe.

use scriptbots_app::tournament::{
    EloRating, FamilyScore, MatchOutcome, MatchResult, OrderPolicy, RatingAxis, RatingError,
    RatingOptions, TournamentError, TournamentHarness, TournamentSpec, enforce_no_config_drift,
    plan, rate_reports, run_match, run_tournament, run_tournament_with_jobs,
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
fn seed_allocation_is_deterministic_across_order_assignments() {
    let spec = two_family_spec(50);
    let plans = plan(&spec).expect("balanced plan");
    let replayed = plan(&spec).expect("replayed balanced plan");
    assert_eq!(plans.len(), 2);
    assert_eq!(plans, replayed, "seed allocation is pure and repeatable");
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
                .unwrap_or_else(|| panic!("missing outcome row for {}", family.as_str()))
                .survival_share
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
    let second =
        run_tournament_with_jobs(&spec, &small_world(), 8).expect("parallel tournament two");
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
fn named_family_rows_preserve_the_exact_entered_arm_evidence() {
    use scriptbots_app::{
        BuildProvenanceV0,
        tournament::{
            leaderboard::{rows_to_match_outcomes, validate_result_matrix},
            reports_to_result_rows,
        },
    };

    let spec = two_family_spec(4);
    let reports = run_tournament(&spec, &small_world()).expect("real matches");
    let build = BuildProvenanceV0::current();
    let rows = reports_to_result_rows(
        &reports,
        "cpu_simd",
        "exact",
        true,
        "named-arm-test",
        &build,
    );
    assert_eq!(rows.len(), 4);
    validate_result_matrix(&rows, &spec, "named-arm-test", &build).expect("exact arm matrix");
    for reconstructed in rows_to_match_outcomes(&rows) {
        let original = reports
            .iter()
            .find(|report| report.outcome.match_id == reconstructed.match_id)
            .expect("original match");
        assert_eq!(reconstructed, original.outcome);
    }
    for row in rows {
        let report = reports
            .iter()
            .find(|report| report.outcome.match_id.0 == row.match_id)
            .expect("matching run");
        let entered = BrainKind::from(row.family);
        assert_eq!(
            row.initial_genome_digest,
            report.initial_genome_digests[&entered]
        );
        assert_eq!(
            row.initial_brain_source_scalars,
            report.initial_brain_source_scalars[&entered]
        );
        assert_eq!(
            row.initial_mutation_rates,
            report.initial_mutation_rates[&entered]
        );
        assert_eq!(
            row.agents_final,
            report.agents_final.get(&entered).copied().unwrap_or(0)
        );
    }
}

#[test]
fn neuroflow_tournament_reports_missing_protocol_adapter() {
    for name in ["neuro", "neuroflow", "ml.neuroflow"] {
        let mut spec = two_family_spec(1);
        spec.families[1] = BrainKind::from(name.to_owned());
        let error = run_tournament(&spec, &small_world()).expect_err("legacy-only family");
        assert!(matches!(
            error,
            TournamentError::UnavailableBrainFamily { ref family, ref reason }
                if family == name && reason.contains("no canonical genome/evaluator")
        ));
    }
}

#[cfg(not(feature = "brain-ft"))]
#[test]
fn ft_tournament_requires_the_feature_without_substitution() {
    for name in ["ft", "frankentorch", "ft-control", "ft.baseline"] {
        let mut spec = two_family_spec(1);
        spec.families[1] = BrainKind::from(name.to_owned());
        let error = run_tournament_with_jobs(&spec, &small_world(), 2)
            .expect_err("Ft without its compiled feature");
        assert!(matches!(
            error,
            TournamentError::UnavailableBrainFamily { ref family, ref reason }
                if family == name && reason.contains("brain-ft")
        ));
    }
}

#[test]
fn ft_spec_aliases_have_one_identity_and_keep_explicit_control_arms() {
    use scriptbots_app::tournament::{TournamentSpecFile, to_brain_kind};

    assert_eq!(to_brain_kind("frankentorch"), to_brain_kind("ft"));
    assert_ne!(to_brain_kind("ft-control"), to_brain_kind("ft"));
    let mut spec = TournamentSpecFile::from_toml_str(include_str!("../../../tournament/spec.toml"))
        .expect("existing spec");
    spec.apply_cli_overrides(Some("ft,frankentorch"), Some("1"), Some(1), Some(8))
        .expect("CLI overrides");
    assert!(matches!(
        plan(&spec.to_spec().unwrap()),
        Err(TournamentError::DuplicateFamilies { family }) if family == "ft"
    ));
}

#[test]
fn ft_comparison_preset_admits_bounded_mixed_architectures_explicitly() {
    use scriptbots_app::tournament::TournamentSpecFile;

    let spec_file =
        TournamentSpecFile::from_toml_str(include_str!("../../../tournament/ft_comparison.toml"))
            .expect("shipped Ft comparison preset");
    assert!(spec_file.parameters.is_none());
    assert!(
        spec_file
            .tournament
            .title
            .contains("architecture comparison")
    );
    assert!(
        spec_file
            .reproducibility
            .as_ref()
            .expect("explicit repeatability policy")
            .require_identical_rerun
    );
    for spec in [
        spec_file.to_spec().unwrap(),
        spec_file.to_smoke_spec().unwrap(),
    ] {
        assert_eq!(
            spec.families,
            vec![BrainKind::new("mlp"), BrainKind::new("ft")]
        );
        assert!(spec.closed);
        assert_eq!(spec.order_policy, OrderPolicy::BothAssignments);
        let plans = plan(&spec).expect("balanced bounded comparison");
        assert_eq!(plans.len(), spec.seeds.len() * 2);
        let repeated_tick_budget = plans.len() as u64 * spec.ticks * 2;
        assert!(repeated_tick_budget <= 4_096);
        assert!(plans.iter().all(|entry| {
            entry.cohort[&BrainKind::new("ft")] == entry.cohort[&BrainKind::new("mlp")]
        }));
    }
}

#[cfg(feature = "brain-ft")]
#[test]
fn ft_tournament_runs_real_genomes_with_repeatable_parallel_outcomes() {
    let mut spec = two_family_spec(16);
    spec.families[1] = BrainKind::new("frankentorch");
    spec.cohort_size = 8;
    spec.seeds = vec![17, 29];
    let serial = run_tournament(&spec, &small_world()).expect("Ft matches");
    let parallel = run_tournament_with_jobs(&spec, &small_world(), 2).expect("parallel Ft matches");
    assert_eq!(serial.len(), 4);
    for (first, repeated) in serial.iter().zip(parallel) {
        assert_eq!(first.outcome, repeated.outcome);
        assert_eq!(first.config_digest, repeated.config_digest);
        assert_eq!(
            first.initial_genome_digests,
            repeated.initial_genome_digests
        );
        assert_eq!(
            first.initial_mutation_rates,
            repeated.initial_mutation_rates
        );
        let ft = BrainKind::new("frankentorch");
        assert_eq!(first.outcome.ticks_run, spec.ticks);
        assert!(first.outcome.family(ft.clone()).is_some());
        assert!(first.initial_brain_source_scalars[&ft] > 0);
        assert_ne!(
            first.initial_genome_digests[&ft],
            first.initial_genome_digests[&MLP_A]
        );
        assert!(first.outcome.warnings.iter().any(|warning| {
            warning.contains("validated genome topology")
                && warning.contains("not recorded tick-zero activity")
        }));
    }

    let rows = scriptbots_app::tournament::reports_to_result_rows(
        &serial,
        "cpu_simd",
        "exact",
        true,
        "ft-alias-test",
        &scriptbots_app::BuildProvenanceV0::current(),
    );
    assert_eq!(
        rows.iter()
            .filter(|row| row.family == "frankentorch")
            .count(),
        4
    );
    assert!(rows.iter().all(|row| row.initial_genome_digest.len() == 64));
}

#[cfg(feature = "brain-ft")]
#[test]
fn ft_leaderboard_carries_topology_provenance_and_keeps_profile_constraints() {
    use scriptbots_app::{
        BuildProvenanceV0,
        tournament::{TournamentResultRow, TournamentSpecFile, execute_leaderboard_tournament},
    };

    let mut spec = TournamentSpecFile::from_toml_str(include_str!("../../../tournament/spec.toml"))
        .expect("existing spec");
    spec.apply_cli_overrides(Some("mlp,frankentorch"), Some("17,29"), Some(4), Some(8))
        .expect("explicit Ft CLI entry");
    let build = BuildProvenanceV0::current();
    let constrained = execute_leaderboard_tournament(
        &spec,
        None,
        true,
        &small_world(),
        &build,
        None,
        None,
        true,
        1,
    )
    .expect_err("the pinned 200-cell profile is not a dense Ft architecture");
    assert!(matches!(
        constrained,
        TournamentError::Publication { reason }
            if reason.contains("observed initial family topology differs")
    ));

    // The shipped opt-in experiment compares architectures without claiming equal cell counts.
    spec =
        TournamentSpecFile::from_toml_str(include_str!("../../../tournament/ft_comparison.toml"))
            .expect("shipped Ft comparison preset");
    spec.apply_cli_overrides(Some("mlp,frankentorch"), Some("17,29"), Some(4), Some(8))
        .expect("bounded explicit comparison");
    let outcome = execute_leaderboard_tournament(
        &spec,
        None,
        true,
        &small_world(),
        &build,
        None,
        None,
        true,
        2,
    )
    .expect("complete real Ft leaderboard and repeated-execution check");
    assert_eq!(outcome.total_matches, 4);
    assert_eq!(outcome.eligible_rows, 8);
    assert_eq!(outcome.excluded_rows, 0);
    let rows = TournamentResultRow::deserialize_jsonl(&outcome.results_jsonl).expect("result rows");
    let ft_rows: Vec<_> = rows.iter().filter(|row| row.family == "ft").collect();
    assert_eq!(ft_rows.len(), 4);
    assert!(ft_rows.iter().all(|row| {
        row.ticks_run == 4
            && row.initial_genome_digest.len() == 64
            && row
                .warnings
                .iter()
                .any(|warning| warning.contains("validated genome topology"))
    }));
    assert!(
        outcome
            .leaderboard_md
            .contains("Ft counts validated genome")
    );
    assert!(
        outcome
            .leaderboard_md
            .contains("without claiming tick-zero activity")
    );
    assert!(outcome.leaderboard_md.contains("ft-survival_share"));
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

#[test]
fn test_rating_multi_axis_clustered_bootstrap_and_markdown() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_test_writer()
        .try_init();

    let spec = TournamentSpec {
        families: vec![MLP_A, DWRAON],
        seeds: vec![101, 102, 103, 104, 105, 106, 107, 108],
        ticks: 200,
        cohort_size: 16,
        order_policy: OrderPolicy::BothAssignments,
        closed: true,
        config_layers: Vec::new(),
    };
    let reports = run_tournament(&spec, &small_world()).expect("run tournament");
    assert_eq!(reports.len(), 16); // 8 seeds * 2 assignments

    let options = RatingOptions {
        bootstrap_replicates: 50, // fast for integration test
        ..RatingOptions::default()
    };
    let rating_table = rate_reports(&reports, &options).expect("rate reports");

    // Verify rating axes
    let survival = rating_table
        .axis(RatingAxis::SurvivalShare)
        .or_else(|| {
            panic!(
                "survival share ratings not Rated; outcome: {:?}",
                rating_table.axes.get(&RatingAxis::SurvivalShare)
            )
        })
        .unwrap();
    assert_eq!(survival.n_matches, 16);
    assert_eq!(survival.n_seeds, 8);
    assert!(survival.converged);
    assert_eq!(survival.ratings.len(), 2);

    let mlp_rating = survival.ratings.get(&MLP_A).expect("mlp rating");
    let dwraon_rating = survival.ratings.get(&DWRAON).expect("dwraon rating");

    // Bradley-Terry theta zero-mean anchor check
    let mean_theta = (mlp_rating.theta + dwraon_rating.theta) / 2.0;
    assert!(
        mean_theta.abs() < 1e-6,
        "mean theta should be 0: {mean_theta}"
    );

    // Elo transform check: elo = 400 * theta / ln(10) + 1500
    let expected_mlp_elo = 400.0 * mlp_rating.theta / std::f64::consts::LN_10 + 1500.0;
    assert!((mlp_rating.elo - expected_mlp_elo).abs() < 1e-4);

    // Markdown leaderboard generation
    let md = rating_table.generate_markdown();
    assert!(md.contains("# ScriptBots Multi-Axis Brain Family Tournament Leaderboard"));
    assert!(md.contains("## Axis: survival_share (`survival_share`)"));
    assert!(md.contains("## Axis: biomass_share (`biomass_share`)"));
    assert!(md.contains("## Axis: mean_lineage_depth (`mean_lineage_depth`)"));
    assert!(md.contains("## Axis: time_to_extinction (`time_to_extinction`)"));
    assert!(md.contains("## Axis: aggregate_score (`aggregate_score`)"));
    assert!(md.contains("novelty_coverage data is absent or incomplete"));
}

#[test]
fn test_rate_outcomes_degenerate_inputs() {
    let options = RatingOptions::default();

    // 1. Empty matches -> error
    let empty_res = TournamentHarness::rate_outcomes(&[], &options);
    assert_eq!(empty_res, Err(RatingError::EmptyMatches));

    // 2. Single family -> error
    let mut outcome = MatchOutcome {
        match_id: scriptbots_app::tournament::MatchId(1),
        seed: 42,
        ticks_run: 100,
        spawn_order_index: 0,
        spawn_order: vec![MLP_A],
        per_family: std::collections::BTreeMap::new(),
        warnings: Vec::new(),
    };
    outcome.set_family(
        MLP_A,
        scriptbots_app::tournament::FamilyOutcome {
            survival_share: 1.0,
            biomass_share: 1.0,
            mean_lineage_depth: 1.0,
            max_lineage_depth: 1,
            extinct_at: None,
            novelty_coverage: None,
        },
    );
    let single_fam_res = TournamentHarness::rate_outcomes(&[outcome], &options);
    assert_eq!(
        single_fam_res,
        Err(RatingError::TooFewFamilies { families: 1 })
    );
}
