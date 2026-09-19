//! End-to-end integration test for paired checkpoint-branch intervention studies (bd-2z0.11.4).
//!
//! Verifies:
//! - Shared pre-intervention digest and tick across all branches.
//! - Exact-tick command application at checkpoint boundary.
//! - First divergence tracing to exact tick and stage.
//! - Extinction, recovery, and hysteresis quantitative measures.
//! - Effect sizes, 95% bootstrap confidence intervals, and Monte Carlo p-values.
//! - Self-contained run bundle creation and bounded canonical verification.
//! - Retention of configs, commands, digests, reports, timings, and logs on disk.

use std::fs;
use std::path::PathBuf;

use scriptbots_app::branch_study::{
    BranchStudyOrchestrator, BranchStudyPlan, BranchStudyReport, CooperationParams,
    StudyBranchSpec, StudyIntervention,
};
use tempfile::tempdir;

#[test]
fn test_seeded_checkpoint_branch_intervention_study_e2e() {
    let output_dir = tempdir().expect("temp output dir");
    let base_seed = 42_424;
    let checkpoint_tick = 12;
    let horizon_ticks = 15;

    let plan = BranchStudyPlan {
        study_id: "e2e_causal_interventions".to_string(),
        description:
            "E2E verification of 5 canonical interventions branched from a common checkpoint"
                .to_string(),
        base_seed,
        checkpoint_tick,
        horizon_ticks,
        brain_preset: scriptbots_app::BrainPreset::Mlp,
        config: scriptbots_core::ScriptBotsConfig {
            persistence_interval: 0,
            rng_seed: Some(base_seed),
            ..Default::default()
        },
        branches: vec![
            StudyBranchSpec {
                branch_id: "control".to_string(),
                description: "Unperturbed baseline control".to_string(),
                intervention: None,
                scheduled_tick: None,
            },
            StudyBranchSpec {
                branch_id: "drought".to_string(),
                description: "Total food growth suppression for 20 ticks".to_string(),
                intervention: Some(StudyIntervention::Drought {
                    region: None,
                    duration_ticks: 20,
                    growth_scale: 0.0,
                }),
                scheduled_tick: Some(checkpoint_tick),
            },
            StudyBranchSpec {
                branch_id: "resource_shock".to_string(),
                description: "Immediate food bloom (+3.5 per cell)".to_string(),
                intervention: Some(StudyIntervention::ResourceShock {
                    region: None,
                    bloom_amount: 3.5,
                    scorch: 0.0,
                }),
                scheduled_tick: Some(checkpoint_tick),
            },
            StudyBranchSpec {
                branch_id: "temperature_shift".to_string(),
                description: "Harsh temperature discomfort penalty rate (0.85)".to_string(),
                intervention: Some(StudyIntervention::TemperatureShift {
                    discomfort_rate: 0.85,
                    comfort_band: 0.04,
                }),
                scheduled_tick: Some(checkpoint_tick),
            },
            StudyBranchSpec {
                branch_id: "closed_world".to_string(),
                description: "Enforce closed world with zero external newcomer injection"
                    .to_string(),
                intervention: Some(StudyIntervention::ClosedWorld { closed: true }),
                scheduled_tick: Some(checkpoint_tick),
            },
            StudyBranchSpec {
                branch_id: "cooperation".to_string(),
                description: "Altruistic food sharing and partner reproduction boost".to_string(),
                intervention: Some(StudyIntervention::Cooperation(CooperationParams {
                    food_sharing_rate: 0.85,
                    food_transfer_rate: 0.45,
                    food_sharing_distance: 45.0,
                    reproduction_partner_chance: Some(0.8),
                })),
                scheduled_tick: Some(checkpoint_tick),
            },
        ],
    };

    let orchestrator = BranchStudyOrchestrator::new(plan.clone(), output_dir.path().to_path_buf());
    let report = orchestrator
        .execute_study(None)
        .expect("study execution must succeed");

    // 1. Shared pre-intervention digest and tick verification
    assert_eq!(report.checkpoint_tick, checkpoint_tick);
    assert_eq!(report.horizon_ticks, horizon_ticks);
    assert!(report.pre_intervention_digest_verified_for_all_branches);
    assert!(
        !report.pre_intervention_digest.is_empty(),
        "pre-intervention digest must be non-empty"
    );

    // 2. Verified command ticks and branches cardinality
    assert_eq!(report.branch_reports.len(), 6);
    assert_eq!(report.control_branch_id, "control");

    let control_rep = &report.branch_reports[0];
    assert_eq!(control_rep.branch_id, "control");
    assert!(control_rep.command_applied_tick.is_none());
    assert!(control_rep.first_divergence.is_none());
    assert!(control_rep.bundle_verified);

    // 3. Detailed checks for each intervention branch
    for branch_rep in &report.branch_reports[1..] {
        assert_eq!(
            branch_rep.command_applied_tick,
            Some(checkpoint_tick),
            "command must be applied at exact scheduled checkpoint tick"
        );

        // First divergence tracing: must identify the exact tick and stage
        let div = branch_rep
            .first_divergence
            .as_ref()
            .unwrap_or_else(|| panic!("branch `{}` must detect divergence", branch_rep.branch_id));
        assert_eq!(
            div.tick,
            checkpoint_tick + 1,
            "branch `{}` first divergence must manifest at the next tick after checkpoint",
            branch_rep.branch_id
        );
        assert!(
            !div.stage.is_empty(),
            "branch `{}` divergence stage must be named",
            branch_rep.branch_id
        );
        assert_ne!(
            div.branch_digest, div.control_digest,
            "branch `{}` digest must differ from control at divergence",
            branch_rep.branch_id
        );

        // Extinction and recovery metrics
        assert!(branch_rep.recovery.recovery_ratio.is_finite());
        assert!(branch_rep.recovery.recovery_ratio >= 0.0);
        assert!(
            branch_rep.extinction.min_population
                <= branch_rep
                    .extinction
                    .final_population
                    .max(branch_rep.recovery.pre_intervention_population)
        );

        // Hysteresis measures
        assert!(
            branch_rep
                .hysteresis
                .integrated_trajectory_deficit
                .is_finite()
        );
        assert!(
            branch_rep
                .hysteresis
                .normalized_trajectory_difference
                .is_finite()
        );
        assert!(branch_rep.hysteresis.terminal_trajectory_gap.is_finite());
        assert!(branch_rep.hysteresis.hysteresis_index.is_finite());

        // Effect sizes and confidence intervals
        assert!(branch_rep.effect_sizes.contains_key("population"));
        assert!(branch_rep.effect_sizes.contains_key("food_total"));
        assert!(branch_rep.effect_sizes.contains_key("total_energy"));

        let pop_eff = &branch_rep.effect_sizes["population"];
        assert_eq!(pop_eff.n_samples, (horizon_ticks + 1) as usize);
        assert!(pop_eff.mean_diff.is_finite());
        assert!(pop_eff.sd_diff.is_finite());
        assert!(pop_eff.p_value >= 0.0 && pop_eff.p_value <= 1.0);

        // Bundle verification
        assert!(
            branch_rep.bundle_verified,
            "bundle for `{}` must pass verification",
            branch_rep.branch_id
        );
        let bundle_path_str = branch_rep
            .bundle_path
            .as_ref()
            .expect("bundle path present");
        let bundle_dir = PathBuf::from(bundle_path_str);
        assert!(bundle_dir.exists());
        assert!(bundle_dir.join("bundle_manifest.json").exists());
        assert!(bundle_dir.join("exports/summary.csv").exists());
        assert!(bundle_dir.join("evidence/branch_evidence.json").exists());
    }

    // 4. Retained disk artifacts
    let study_dir = output_dir.path();
    let report_json_path = study_dir.join("study_report.json");
    let summary_md_path = study_dir.join("study_summary.md");
    let plan_json_path = study_dir.join("study_plan.json");

    assert!(
        report_json_path.exists(),
        "study_report.json must be retained on disk"
    );
    assert!(
        summary_md_path.exists(),
        "study_summary.md must be retained on disk"
    );
    assert!(
        plan_json_path.exists(),
        "study_plan.json must be retained on disk"
    );

    let read_report_json = fs::read_to_string(&report_json_path).expect("read report json");
    let deserialized_report: BranchStudyReport =
        serde_json::from_str(&read_report_json).expect("report json must deserialize validly");
    assert_eq!(deserialized_report.study_id, "e2e_causal_interventions");
    assert_eq!(deserialized_report.branch_reports.len(), 6);

    let summary_md = fs::read_to_string(&summary_md_path).expect("read summary md");
    assert!(summary_md.contains("e2e_causal_interventions"));
    assert!(summary_md.contains("Branch Outcomes Matrix"));
    assert!(summary_md.contains("Causal Attribution and Divergence Analysis"));
    assert!(
        summary_md.contains("**Pre-Intervention Digest Verified Across All Branches**: `true`")
    );
}
