//! End-to-end integration test for paired checkpoint-branch intervention studies (bd-2z0.11.4).
//!
//! Verifies:
//! - Shared pre-intervention digest and tick across all branches.
//! - Exact-tick command application at immediate and delayed boundaries.
//! - First divergence tracing to exact tick and stage.
//! - Extinction, recovery, and hysteresis quantitative measures.
//! - Descriptive single-seed comparisons without fabricated independent replicates or inference.
//! - Self-contained run bundle creation and bounded canonical verification.
//! - Retention of configs, commands, digests, reports, timings, and logs on disk.
//! - Real CLI execution, repeatable artifacts, and refusal without overwriting evidence.
//! - Saved-checkpoint plans continue real evolved worlds and retain their exact source bytes.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use scriptbots_app::branch_study::{
    BRANCH_STUDY_REPORT_SCHEMA_VERSION, BranchAnalysisBasis, BranchStudyOrchestrator,
    BranchStudyPlan, BranchStudyReport, BranchStudySource, CooperationParams,
    ReplicatedBranchStudyPlan, ReplicatedBranchStudyReport, StudyBranchSpec, StudyIntervention,
};
use scriptbots_app::experiment_runner::MatchedSeedExperimentRunner;
use scriptbots_storage::bundle::{RunBundleVerificationLimits, verify_run_bundle_bounded};
use serde::Deserialize;
use tempfile::tempdir;

#[derive(Debug, Deserialize)]
struct PersistedSample {
    tick: u64,
    population: usize,
    total_energy: f32,
    average_health: f32,
    digest: String,
}

fn read_samples(bundle: &str) -> Vec<PersistedSample> {
    csv::Reader::from_path(PathBuf::from(bundle).join("exports/summary.csv"))
        .expect("open persisted trajectory")
        .deserialize()
        .collect::<Result<Vec<_>, _>>()
        .expect("decode persisted trajectory")
}

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
        source_checkpoint: None,
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
                description: "Delayed food bloom (+3.5 per cell)".to_string(),
                intervention: Some(StudyIntervention::ResourceShock {
                    region: None,
                    bloom_amount: 3.5,
                    scorch: 0.0,
                }),
                scheduled_tick: Some(checkpoint_tick + 4),
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
    assert_eq!(control_rep.recovery.baseline_tick, checkpoint_tick);
    let control_samples = read_samples(control_rep.bundle_path.as_deref().unwrap());
    assert_eq!(control_samples.len(), (horizon_ticks + 1) as usize);
    assert!(control_samples[0].total_energy > 0.0);
    assert!(control_samples[0].average_health.is_finite());

    // 3. Detailed checks for each intervention branch
    for branch_rep in &report.branch_reports[1..] {
        let expected_due_tick = plan
            .branches
            .iter()
            .find(|branch| branch.branch_id == branch_rep.branch_id)
            .unwrap()
            .scheduled_tick
            .unwrap_or(checkpoint_tick);
        assert_eq!(
            branch_rep.command_applied_tick,
            Some(expected_due_tick),
            "command must be applied at the declared completed boundary"
        );

        // First divergence tracing: must identify the exact tick and stage
        let div = branch_rep
            .first_divergence
            .as_ref()
            .unwrap_or_else(|| panic!("branch `{}` must detect divergence", branch_rep.branch_id));
        assert_eq!(
            div.tick,
            expected_due_tick + 1,
            "branch `{}` first divergence must manifest on the transition after its due boundary",
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

        // Repeated ticks from one seed remain descriptive observations, not independent replicates.
        assert!(branch_rep.effect_sizes.contains_key("population"));
        assert!(branch_rep.effect_sizes.contains_key("food_total"));
        assert!(branch_rep.effect_sizes.contains_key("total_energy"));

        let pop_eff = &branch_rep.effect_sizes["population"];
        assert_eq!(pop_eff.n_samples, (horizon_ticks + 1) as usize);
        assert!(pop_eff.mean_diff.is_finite());
        assert!(pop_eff.sd_diff.is_finite());
        for effect in branch_rep.effect_sizes.values() {
            assert_eq!(
                effect.analysis_basis,
                BranchAnalysisBasis::SingleSeedTrajectory { seed: base_seed }
            );
            assert_eq!(effect.n_samples, (horizon_ticks + 1) as usize);
            assert!(effect.ci_95.is_none());
            assert!(effect.p_value.is_none());
            assert!(effect.cohens_d.is_none());
            assert!(effect.hedges_g.is_none());
        }

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

        let samples = read_samples(bundle_path_str);
        assert_eq!(samples.len(), control_samples.len());
        let baseline_index = (expected_due_tick - checkpoint_tick) as usize;
        for (control, branch) in control_samples
            .iter()
            .zip(&samples)
            .take(baseline_index + 1)
        {
            assert_eq!(control.tick, branch.tick);
            assert_eq!(
                control.digest, branch.digest,
                "{} changed before its declared boundary {}",
                branch_rep.branch_id, expected_due_tick
            );
        }
        assert_ne!(
            control_samples[baseline_index + 1].digest,
            samples[baseline_index + 1].digest
        );
        assert_eq!(branch_rep.recovery.baseline_tick, expected_due_tick);
        assert_eq!(
            branch_rep.recovery.pre_intervention_population,
            samples[baseline_index].population
        );

        let evidence: serde_json::Value = serde_json::from_slice(
            &fs::read(bundle_dir.join("evidence/branch_evidence.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            evidence["schema_version"],
            BRANCH_STUDY_REPORT_SCHEMA_VERSION
        );
        assert_eq!(evidence["command_applied_tick"], expected_due_tick);
        for effect in evidence["effect_sizes"].as_object().unwrap().values() {
            assert_eq!(effect["analysis_basis"]["kind"], "single_seed_trajectory");
            assert!(effect.get("ci_95").is_none());
            assert!(effect.get("p_value").is_none());
            assert!(effect.get("cohens_d").is_none());
            assert!(effect.get("hedges_g").is_none());
        }
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
    assert_eq!(
        deserialized_report.schema_version,
        BRANCH_STUDY_REPORT_SCHEMA_VERSION
    );
    assert_eq!(deserialized_report.branch_reports.len(), 6);

    let summary_md = fs::read_to_string(&summary_md_path).expect("read summary md");
    assert!(summary_md.contains("e2e_causal_interventions"));
    assert!(summary_md.contains("Branch Outcomes Matrix"));
    assert!(summary_md.contains("Causal Attribution and Divergence Analysis"));
    assert!(
        summary_md.contains("**Pre-Intervention Digest Verified Across All Branches**: `true`")
    );
    assert!(summary_md.contains("One independent matched seed pair"));
    assert!(summary_md.contains("Repeated ticks are not independent replicates"));
    assert!(!summary_md.contains("Pop Mean Diff (95% CI)"));
}

struct CliCapture {
    stdout: PathBuf,
    stderr: PathBuf,
}

fn diagnostic(path: &Path) -> String {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .expect("open CLI log")
        .take(64 * 1024)
        .read_to_end(&mut bytes)
        .expect("read bounded CLI log");
    String::from_utf8_lossy(&bytes).into_owned()
}

fn run_branch_cli(root: &Path, label: &str, args: &[&OsStr], succeed: bool) -> CliCapture {
    let capture = CliCapture {
        stdout: root.join(format!("{label}.stdout.log")),
        stderr: root.join(format!("{label}.stderr.log")),
    };
    let mut command = Command::new(env!("CARGO_BIN_EXE_scriptbots-app"));
    command
        .current_dir(root)
        .env("RAYON_NUM_THREADS", "1")
        .env("SCRIPTBOTS_MAX_THREADS", "1")
        .env("RUST_LOG", "warn")
        .arg("branch-study")
        .args(args)
        .stdout(fs::File::create(&capture.stdout).expect("capture CLI stdout"))
        .stderr(fs::File::create(&capture.stderr).expect("capture CLI stderr"));
    let result =
        MatchedSeedExperimentRunner::run_bounded_child(&mut command, Duration::from_secs(60));
    if let Err(error) = &result {
        assert!(
            error.kind() == std::io::ErrorKind::Other
                && error.to_string().contains("canonical child exit=")
                && error.to_string().contains("timeout=false"),
            "{label} did not reach an actual CLI exit: {error}; stdout={} stderr={}",
            diagnostic(&capture.stdout),
            diagnostic(&capture.stderr),
        );
    }
    assert_eq!(
        result.is_ok(),
        succeed,
        "{label}: {result:?}; stdout={} stderr={}",
        diagnostic(&capture.stdout),
        diagnostic(&capture.stderr),
    );
    capture
}

fn tiny_cli_plan(mut plan: BranchStudyPlan) -> BranchStudyPlan {
    plan.study_id = "cli-checkpoint-study".to_owned();
    plan.checkpoint_tick = 2;
    plan.horizon_ticks = 4;
    // Keep the application's real 4x4 founder recipe inside the world while
    // reducing the field to 144 cells and suppressing population growth.
    plan.config.world_width = 480;
    plan.config.world_height = 480;
    plan.config.food_cell_size = 40;
    plan.config.initial_food = plan.config.food_max;
    plan.config.population_minimum = 0;
    plan.config.population_spawn_interval = 0;
    plan.config.reproduction_attempt_chance = 0.0;
    let shock = plan
        .branches
        .iter_mut()
        .find(|branch| branch.branch_id == "resource_shock")
        .expect("example contains its resource-shock branch");
    shock.intervention = Some(StudyIntervention::ResourceShock {
        region: None,
        bloom_amount: 0.0,
        scorch: 0.5,
    });
    shock.scheduled_tick = Some(plan.checkpoint_tick + 2);
    plan
}

fn retained_study_bytes(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn collect(root: &Path, directory: &Path, files: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in fs::read_dir(directory).expect("read retained study") {
            let entry = entry.expect("study entry");
            let kind = entry.file_type().expect("study entry type");
            let path = entry.path();
            if kind.is_dir() {
                collect(root, &path, files);
            } else {
                assert!(kind.is_file(), "study artifact must be a regular file");
                files.insert(
                    path.strip_prefix(root)
                        .expect("contained artifact")
                        .to_owned(),
                    fs::read(&path).expect("retained artifact bytes"),
                );
            }
        }
    }
    let mut files = BTreeMap::new();
    collect(root, root, &mut files);
    files
}

fn verify_cli_study(output: &Path, plan: &BranchStudyPlan) -> BranchStudyReport {
    let persisted_plan: BranchStudyPlan = serde_json::from_slice(
        &fs::read(output.join("study_plan.json")).expect("retained CLI plan"),
    )
    .expect("typed persisted plan");
    assert_eq!(&persisted_plan, plan);
    let report: BranchStudyReport = serde_json::from_slice(
        &fs::read(output.join("study_report.json")).expect("retained CLI report"),
    )
    .expect("typed persisted report");
    assert_eq!(report.schema_version, BRANCH_STUDY_REPORT_SCHEMA_VERSION);
    assert_eq!(report.study_id, plan.study_id);
    assert_eq!(report.base_seed, plan.base_seed);
    assert_eq!(report.checkpoint_tick, plan.checkpoint_tick);
    assert_eq!(report.horizon_ticks, plan.horizon_ticks);
    assert_eq!(report.branch_reports.len(), plan.branches.len());
    for spec in &plan.branches {
        assert_eq!(
            report
                .branch_reports
                .iter()
                .filter(|branch| branch.branch_id == spec.branch_id)
                .count(),
            1,
            "each planned branch must be reported exactly once",
        );
    }
    assert_eq!(report.control_branch_id, "control");
    assert!(report.pre_intervention_digest_verified_for_all_branches);
    assert!(!report.pre_intervention_digest.is_empty());
    assert_eq!(
        fs::read_to_string(output.join("study_summary.md")).expect("retained CLI summary"),
        report.summary_markdown,
    );
    let retained_source: BranchStudySource = serde_json::from_slice(
        &fs::read(output.join("study_source.json")).expect("retained study source"),
    )
    .expect("typed source provenance");
    assert_eq!(retained_source, report.source);
    if plan.source_checkpoint.is_none() {
        assert_eq!(report.source, BranchStudySource::SeededWarmup);
    }

    let control = report
        .branch_reports
        .iter()
        .find(|branch| branch.branch_id == "control")
        .expect("reported control");
    assert_eq!(control.command_applied_tick, None);
    assert_eq!(control.first_divergence, None);
    let control_samples = read_samples(control.bundle_path.as_deref().expect("control bundle"));
    for branch in &report.branch_reports {
        let spec = plan
            .branches
            .iter()
            .find(|spec| spec.branch_id == branch.branch_id)
            .expect("reported branch belongs to requested plan");
        let expected_bundle = output.join(format!("bundle_{}", branch.branch_id));
        assert_eq!(
            Path::new(branch.bundle_path.as_deref().expect("reported bundle path")),
            expected_bundle,
        );
        assert!(branch.bundle_verified);
        let verified = verify_run_bundle_bounded(
            &expected_bundle,
            RunBundleVerificationLimits {
                max_manifest_bytes: 1024 * 1024,
                max_artifacts: 8,
                max_artifact_bytes: 1024 * 1024,
                max_total_artifact_bytes: 2 * 1024 * 1024,
            },
        )
        .expect("independently verify actual CLI bundle bytes");
        assert_eq!(verified.max_tick, plan.checkpoint_tick + plan.horizon_ticks);
        assert_eq!(verified.total_artifacts_verified, 2);

        let samples = read_samples(branch.bundle_path.as_deref().unwrap());
        assert_eq!(samples.len(), (plan.horizon_ticks + 1) as usize);
        assert_eq!(samples[0].tick, plan.checkpoint_tick);
        assert!(samples[0].population > 0);
        assert_eq!(samples[0].digest, report.pre_intervention_digest);
        assert_eq!(samples.last().unwrap().digest, branch.final_digest);
        assert_eq!(
            branch.terminal_sample.tick,
            plan.checkpoint_tick + plan.horizon_ticks
        );
        assert_eq!(
            branch.terminal_sample.population,
            samples.last().unwrap().population
        );
        assert_eq!(
            branch.terminal_sample.digest_overall,
            samples.last().unwrap().digest
        );
        assert_eq!(
            samples.iter().map(|sample| sample.tick).collect::<Vec<_>>(),
            (plan.checkpoint_tick..=plan.checkpoint_tick + plan.horizon_ticks).collect::<Vec<_>>(),
        );
        let evidence: serde_json::Value = serde_json::from_slice(
            &fs::read(expected_bundle.join("evidence/branch_evidence.json"))
                .expect("CLI branch evidence"),
        )
        .expect("typed branch evidence JSON");
        assert_eq!(evidence["schema_version"], report.schema_version);
        assert_eq!(evidence["source"], serde_json::to_value(&report.source).unwrap());
        assert_eq!(evidence["branch_id"], branch.branch_id);
        assert_eq!(evidence["base_seed"], plan.base_seed);
        assert_eq!(evidence["checkpoint_tick"], plan.checkpoint_tick);
        assert_eq!(
            evidence["terminal_sample"],
            serde_json::to_value(&branch.terminal_sample).unwrap()
        );
        assert_eq!(
            evidence["intervention"],
            serde_json::to_value(&spec.intervention).unwrap()
        );
        assert_eq!(
            evidence["command_applied_tick"],
            serde_json::to_value(branch.command_applied_tick).unwrap(),
        );

        if spec.intervention.is_some() {
            let due = spec.scheduled_tick.unwrap_or(plan.checkpoint_tick);
            assert_eq!(branch.command_applied_tick, Some(due));
            let first = branch
                .first_divergence
                .as_ref()
                .expect("observed intervention effect");
            assert_eq!(first.tick, due + 1);
            assert_eq!(first.command_applied_tick, due);
            assert_ne!(first.control_digest, first.branch_digest);
            for (observed, control) in samples.iter().zip(&control_samples) {
                if observed.tick <= due {
                    assert_eq!(observed.digest, control.digest);
                } else if observed.tick == due + 1 {
                    assert_ne!(observed.digest, control.digest);
                }
            }
            assert!(!branch.effect_sizes.is_empty());
            for effect in branch.effect_sizes.values() {
                assert_eq!(
                    effect.analysis_basis,
                    BranchAnalysisBasis::SingleSeedTrajectory {
                        seed: plan.base_seed
                    },
                );
                assert!(effect.ci_95.is_none() && effect.p_value.is_none());
            }
        }
    }
    report
}

#[test]
fn branch_study_cli_runs_its_example_reproducibly_and_preserves_occupied_output() {
    let root = tempdir().expect("CLI study test directory");
    let example = run_branch_cli(root.path(), "example", &[OsStr::new("--example")], true);
    let example_plan: BranchStudyPlan =
        serde_json::from_slice(&fs::read(example.stdout).expect("example JSON output"))
            .expect("--example writes only a complete editable plan");
    assert_eq!(example_plan.branches.len(), 6);
    assert!(!root.path().join("study_plan.json").exists());
    assert!(!root.path().join("study_report.json").exists());
    let plan = tiny_cli_plan(example_plan);
    let plan_path = root.path().join("tiny_plan.json");
    fs::write(&plan_path, serde_json::to_vec_pretty(&plan).unwrap()).expect("tiny editable plan");

    let first_dir = root.path().join("first");
    let first_run = run_branch_cli(
        root.path(),
        "first",
        &[
            OsStr::new("--plan"),
            plan_path.as_os_str(),
            OsStr::new("--out"),
            first_dir.as_os_str(),
        ],
        true,
    );
    let mut first = verify_cli_study(&first_dir, &plan);
    assert!(diagnostic(&first_run.stdout).contains("Report:"));

    let second_dir = root.path().join("second");
    fs::create_dir(&second_dir).expect("existing empty output directory");
    run_branch_cli(
        root.path(),
        "second",
        &[
            OsStr::new("--plan"),
            plan_path.as_os_str(),
            OsStr::new("--out"),
            second_dir.as_os_str(),
        ],
        true,
    );
    let mut second = verify_cli_study(&second_dir, &plan);
    for branch in &plan.branches {
        for artifact in ["exports/summary.csv", "evidence/branch_evidence.json"] {
            let relative = PathBuf::from(format!("bundle_{}", branch.branch_id)).join(artifact);
            assert_eq!(
                fs::read(first_dir.join(&relative)).expect("first scientific artifact"),
                fs::read(second_dir.join(relative)).expect("repeated scientific artifact"),
            );
        }
    }
    // Destination paths differ; all scientific report fields must remain exact.
    for branch in &mut first.branch_reports {
        branch.bundle_path = None;
    }
    for branch in &mut second.branch_reports {
        branch.bundle_path = None;
    }
    assert_eq!(first, second);

    let retained = retained_study_bytes(&first_dir);
    let refused = run_branch_cli(
        root.path(),
        "occupied",
        &[
            OsStr::new("--plan"),
            plan_path.as_os_str(),
            OsStr::new("--out"),
            first_dir.as_os_str(),
        ],
        false,
    );
    assert!(diagnostic(&refused.stderr).contains("already occupied"));
    assert_eq!(retained_study_bytes(&first_dir), retained);
}

#[test]
fn branch_study_cli_refuses_invalid_and_oversized_plans_before_creating_output() {
    let root = tempdir().expect("CLI refusal test directory");
    let valid = tiny_cli_plan(BranchStudyPlan::standard_five_branch(
        "invalid-input",
        42,
        2,
        4,
    ));
    let malformed = root.path().join("malformed.json");
    fs::write(&malformed, b"{\"study_id\":").expect("malformed plan fixture");

    let unknown = root.path().join("unknown.json");
    let mut unknown_value = serde_json::to_value(&valid).unwrap();
    unknown_value["horizon_tickss"] = serde_json::json!(4);
    fs::write(&unknown, serde_json::to_vec(&unknown_value).unwrap()).expect("unknown-field plan");

    let work_limit = root.path().join("work_limit.json");
    let mut excessive = valid;
    excessive.horizon_ticks = scriptbots_app::branch_study::MAX_STUDY_HORIZON_TICKS + 1;
    fs::write(&work_limit, serde_json::to_vec(&excessive).unwrap()).expect("bounded invalid plan");

    let oversized = root.path().join("oversized.json");
    fs::File::create(&oversized)
        .expect("sparse oversized plan")
        .set_len(2 * 1024 * 1024)
        .expect("oversized input without a large memory allocation");

    for (label, path, message) in [
        ("malformed", malformed, "invalid study plan"),
        ("unknown", unknown, "unknown field"),
        ("work_limit", work_limit, "horizon ticks"),
        ("oversized", oversized, "regular file of at most"),
    ] {
        let output = root.path().join(format!("{label}_output"));
        let capture = run_branch_cli(
            root.path(),
            label,
            &[
                OsStr::new("--plan"),
                path.as_os_str(),
                OsStr::new("--out"),
                output.as_os_str(),
            ],
            false,
        );
        let stderr = diagnostic(&capture.stderr);
        assert!(stderr.contains(message), "wrong refusal: {stderr}");
        assert!(
            !output.exists(),
            "invalid plans must not reserve study output"
        );
    }
}

#[test]
fn replicated_branch_study_cli_retains_independent_pairs_and_reproducible_endpoint_inference() {
    let root = tempdir().expect("replicated CLI study");
    let mut plan = tiny_cli_plan(BranchStudyPlan::standard_five_branch(
        "replicated-cli",
        19,
        2,
        4,
    ));
    plan.branches
        .retain(|branch| matches!(branch.branch_id.as_str(), "control" | "resource_shock"));
    let plan_path = root.path().join("plan.json");
    fs::write(&plan_path, serde_json::to_vec_pretty(&plan).unwrap()).unwrap();

    let mut reports = Vec::new();
    for label in ["replicated-first", "replicated-second"] {
        let output = root.path().join(label);
        let capture = run_branch_cli(
            root.path(),
            label,
            &[
                OsStr::new("--plan"),
                plan_path.as_os_str(),
                OsStr::new("--out"),
                output.as_os_str(),
                OsStr::new("--seeds"),
                OsStr::new("42,137"),
            ],
            true,
        );
        let report: ReplicatedBranchStudyReport =
            serde_json::from_slice(&fs::read(output.join("replicated_report.json")).unwrap())
                .unwrap();
        let retained_plan: ReplicatedBranchStudyPlan =
            serde_json::from_slice(&fs::read(output.join("study_plan.json")).unwrap()).unwrap();
        assert_eq!(retained_plan, report.plan);
        assert_eq!(report.plan.study, plan);
        assert_eq!(report.plan.seeds, [42, 137]);
        assert_eq!(
            report.replicates.len(),
            2,
            "resampling seed is not an extra replicate"
        );
        assert_eq!(report.total_transitions, 20);
        assert_eq!(report.endpoint_tick, 6);
        assert!(!output.join("study_report.json").exists());
        assert!(
            diagnostic(&capture.stdout).contains("2 independently initialized matched seed pairs")
        );
        assert_eq!(
            fs::read_to_string(output.join("replicated_summary.md")).unwrap(),
            report.summary_markdown
        );
        assert_ne!(
            report.replicates[0].report.pre_intervention_digest,
            report.replicates[1].report.pre_intervention_digest
        );
        let mut measured_differences = [0.0; 3];
        for replicate in &report.replicates {
            let mut seed_plan = plan.clone();
            seed_plan.base_seed = replicate.seed;
            seed_plan.config.rng_seed = Some(replicate.seed);
            let seed_output = output.join(format!("seed_{:016x}", replicate.seed));
            let independent = verify_cli_study(&seed_output, &seed_plan);
            assert_eq!(independent, replicate.report);
            assert_eq!(
                output.join(&replicate.report_path),
                seed_output.join("study_report.json")
            );
            let control = independent
                .branch_reports
                .iter()
                .find(|branch| branch.branch_id == "control")
                .unwrap();
            let shock = independent
                .branch_reports
                .iter()
                .find(|branch| branch.branch_id == "resource_shock")
                .unwrap();
            measured_differences[0] +=
                shock.terminal_sample.population as f64 - control.terminal_sample.population as f64;
            measured_differences[1] +=
                shock.terminal_sample.food_total - control.terminal_sample.food_total;
            measured_differences[2] += f64::from(shock.terminal_sample.total_energy)
                - f64::from(control.terminal_sample.total_energy);
        }
        for (index, metric) in ["population", "food_total", "total_energy"]
            .iter()
            .enumerate()
        {
            let effect = &report.endpoint_effects["resource_shock"][*metric];
            assert_eq!(effect.n_samples, 2);
            assert_eq!(
                effect.analysis_basis,
                BranchAnalysisBasis::IndependentPairedReplicates { pairs: 2 }
            );
            assert_eq!(effect.mean_diff, measured_differences[index] / 2.0);
            assert!(effect.ci_95.is_some() && effect.p_value.is_some());
        }
        reports.push(report);
    }
    assert_eq!(
        reports[0].endpoint_effects, reports[1].endpoint_effects,
        "seeded resampling and observations repeat exactly"
    );
    for (left, right) in reports[0].replicates.iter().zip(&reports[1].replicates) {
        assert_eq!(
            left.report.pre_intervention_digest,
            right.report.pre_intervention_digest
        );
        for (left, right) in left
            .report
            .branch_reports
            .iter()
            .zip(&right.report.branch_reports)
        {
            assert_eq!(left.terminal_sample, right.terminal_sample);
        }
    }

    let occupied = root.path().join("replicated-first");
    let before = retained_study_bytes(&occupied);
    let refusal = run_branch_cli(
        root.path(),
        "replicated-occupied",
        &[
            OsStr::new("--plan"),
            plan_path.as_os_str(),
            OsStr::new("--out"),
            occupied.as_os_str(),
            OsStr::new("--seeds"),
            OsStr::new("42,137"),
        ],
        false,
    );
    assert!(diagnostic(&refusal.stderr).contains("already occupied"));
    assert_eq!(retained_study_bytes(&occupied), before);
}

#[test]
fn replicated_branch_study_cli_refuses_duplicate_seeds_and_aggregate_work_before_output() {
    let root = tempdir().unwrap();
    let mut plan = tiny_cli_plan(BranchStudyPlan::standard_five_branch(
        "replicated-refusal",
        19,
        2,
        4,
    ));
    plan.branches
        .retain(|branch| matches!(branch.branch_id.as_str(), "control" | "resource_shock"));
    let valid_path = root.path().join("valid.json");
    fs::write(&valid_path, serde_json::to_vec_pretty(&plan).unwrap()).unwrap();
    let mut too_large = plan;
    too_large.checkpoint_tick = scriptbots_app::branch_study::MAX_STUDY_TOTAL_TRANSITIONS / 2;
    too_large.branches[1].scheduled_tick = None;
    let work_path = root.path().join("aggregate-limit.json");
    fs::write(&work_path, serde_json::to_vec_pretty(&too_large).unwrap()).unwrap();
    for (label, path, seeds, message) in [
        (
            "duplicate",
            &valid_path,
            "42,42",
            "duplicate replicate seed",
        ),
        (
            "aggregate-work",
            &work_path,
            "42,137",
            "replicated total transitions",
        ),
    ] {
        let output = root.path().join(format!("{label}-output"));
        let refusal = run_branch_cli(
            root.path(),
            label,
            &[
                OsStr::new("--plan"),
                path.as_os_str(),
                OsStr::new("--out"),
                output.as_os_str(),
                OsStr::new("--seeds"),
                OsStr::new(seeds),
            ],
            false,
        );
        assert!(diagnostic(&refusal.stderr).contains(message));
        assert!(
            !output.exists(),
            "refused replicated study must not claim an output"
        );
    }
}

fn evolved_import_fixture(
    preset: scriptbots_app::BrainPreset,
) -> (BranchStudyPlan, scriptbots_core::WorldState) {
    let mut plan = tiny_cli_plan(BranchStudyPlan::standard_five_branch(
        "saved-cut",
        42,
        2,
        4,
    ));
    plan.brain_preset = preset;
    plan.branches
        .retain(|branch| matches!(branch.branch_id.as_str(), "control" | "resource_shock"));
    if preset == scriptbots_app::BrainPreset::Neuro {
        plan.config.neuroflow.enabled = false;
        plan.config.neuroflow.hidden_layers = vec![7, 3];
        plan.config.neuroflow.activation = scriptbots_core::NeuroflowActivationKind::Sigmoid;
    }
    let mut world = scriptbots_core::WorldState::new(plan.config.clone()).unwrap();
    let installed = scriptbots_app::install_brains(&mut world, preset).unwrap();
    scriptbots_app::seed_founders(
        &mut world,
        installed.population(),
        scriptbots_app::FounderRecipe::CornerGrid,
    )
    .unwrap();
    world
        .enqueue_intervention(scriptbots_core::Intervention::Drought {
            region: scriptbots_core::Region::All,
            ticks: 7,
            growth_scale: 0.25,
        })
        .unwrap();
    for _ in 0..plan.checkpoint_tick {
        world.step().unwrap();
    }
    assert!(!world.active_effects().is_empty());
    assert!(world.agent_count() > 0);
    // Current configuration is not a recipe for repeating the world's original seed.
    // Import must retain this edit while verifying the actual restored RNG root.
    let mut edited = world.config().clone();
    edited.rng_seed = None;
    world.apply_config_update(edited.clone()).unwrap();
    assert_eq!(
        world.agent_substream_protocol_v1().root_seed(),
        plan.base_seed
    );
    plan.config = edited;
    (plan, world)
}

fn direct_import_trajectory(
    mut world: scriptbots_core::WorldState,
    horizon: u64,
    meteor_at: Option<u64>,
) -> Vec<(u64, String)> {
    let mut observed = vec![(world.tick().0, world.world_digest_v1().unwrap().overall)];
    for _ in 0..horizon {
        if meteor_at == Some(world.tick().0) {
            world
                .enqueue_intervention(scriptbots_core::Intervention::Meteor {
                    region: scriptbots_core::Region::All,
                    lethality: 0.0,
                    scorch: 0.5,
                })
                .unwrap();
        }
        world.step().unwrap();
        observed.push((world.tick().0, world.world_digest_v1().unwrap().overall));
    }
    observed
}

#[test]
fn saved_checkpoint_plan_cli_matches_original_control_and_direct_core_treatment() {
    let root = tempdir().unwrap();
    for preset in [
        scriptbots_app::BrainPreset::Mlp,
        scriptbots_app::BrainPreset::Dwraon,
        #[cfg(feature = "neuro")]
        scriptbots_app::BrainPreset::Neuro,
    ] {
        let label = preset.as_str();
        let (mut plan, original) = evolved_import_fixture(preset);
        let checkpoint = original.checkpoint_v1().unwrap();
        let source_bytes = checkpoint.encode().unwrap();
        let source_name = format!("source-{label}.bin");
        let source_path = root.path().join(&source_name);
        fs::write(&source_path, &source_bytes).unwrap();
        // The existing CLI runs in root; relative inputs have explicit invocation-cwd semantics.
        plan.source_checkpoint = Some(PathBuf::from(source_name));
        let plan_path = root.path().join(format!("import-{label}.json"));
        let plan_bytes = serde_json::to_vec_pretty(&plan).unwrap();
        fs::write(&plan_path, &plan_bytes).unwrap();

        let registry =
            scriptbots_app::create_brain_registry_for_config(preset, &plan.config).unwrap();
        let treatment =
            scriptbots_core::WorldState::restore_checkpoint_v1(&checkpoint, registry).unwrap();
        let due = plan.checkpoint_tick + 2;
        let expected_control = direct_import_trajectory(original, plan.horizon_ticks, None);
        let expected_treatment =
            direct_import_trajectory(treatment, plan.horizon_ticks, Some(due));
        assert_eq!(expected_control[0], expected_treatment[0]);
        assert_ne!(
            expected_control.last().unwrap().1,
            expected_treatment.last().unwrap().1,
            "the direct treatment oracle must have a real scientific effect"
        );

        let output = root.path().join(format!("import-{label}"));
        run_branch_cli(
            root.path(),
            &format!("import-{label}"),
            &[
                OsStr::new("--plan"),
                plan_path.as_os_str(),
                OsStr::new("--out"),
                output.as_os_str(),
            ],
            true,
        );
        let report = verify_cli_study(&output, &plan);
        assert_eq!(report.pre_intervention_digest, checkpoint.source_digest().overall);
        assert_eq!(
            report.source,
            BranchStudySource::SavedCheckpoint {
                artifact_path: "source_checkpoint.bin".to_owned(),
                byte_size: u64::try_from(source_bytes.len()).unwrap(),
                checksum_blake3: blake3::hash(&source_bytes).to_hex().to_string(),
            }
        );
        assert_eq!(
            fs::read(output.join("source_checkpoint.bin")).unwrap(),
            source_bytes
        );
        for (branch_id, expected) in [
            ("control", &expected_control),
            ("resource_shock", &expected_treatment),
        ] {
            let branch = report
                .branch_reports
                .iter()
                .find(|branch| branch.branch_id == branch_id)
                .unwrap();
            let observed: Vec<_> = read_samples(branch.bundle_path.as_deref().unwrap())
                .into_iter()
                .map(|sample| (sample.tick, sample.digest))
                .collect();
            assert_eq!(&observed, expected, "every saved-cut transition for {label}/{branch_id}");
        }

        let retained = retained_study_bytes(&output);
        let refusal = run_branch_cli(
            root.path(),
            &format!("import-occupied-{label}"),
            &[
                OsStr::new("--plan"),
                plan_path.as_os_str(),
                OsStr::new("--out"),
                output.as_os_str(),
            ],
            false,
        );
        assert!(diagnostic(&refusal.stderr).contains("already occupied"));
        assert_eq!(retained_study_bytes(&output), retained);
        assert_eq!(fs::read(&source_path).unwrap(), source_bytes);
        assert_eq!(fs::read(&plan_path).unwrap(), plan_bytes);
    }
}

#[test]
fn saved_checkpoint_plan_cli_refuses_mismatches_and_unsafe_inputs_without_output() {
    let root = tempdir().unwrap();
    let (mut plan, world) = evolved_import_fixture(scriptbots_app::BrainPreset::Mlp);
    let bytes = world.checkpoint_v1().unwrap().encode().unwrap();
    let source = root.path().join("valid-source.bin");
    fs::write(&source, &bytes).unwrap();
    plan.source_checkpoint = Some(source.clone());

    let mut wrong_tick = plan.clone();
    wrong_tick.checkpoint_tick += 1;
    let mut wrong_seed = plan.clone();
    wrong_seed.base_seed += 1;
    let mut wrong_config = plan.clone();
    wrong_config.config.closed = !wrong_config.config.closed;
    let mut wrong_family = plan.clone();
    wrong_family.brain_preset = scriptbots_app::BrainPreset::Dwraon;
    for (label, rejected) in [
        ("source-tick", wrong_tick),
        ("source-root", wrong_seed),
        ("source-config", wrong_config),
        ("source-registry", wrong_family),
    ] {
        let path = root.path().join(format!("{label}.json"));
        fs::write(&path, serde_json::to_vec(&rejected).unwrap()).unwrap();
        let output = root.path().join(format!("{label}-output"));
        run_branch_cli(
            root.path(),
            label,
            &[
                OsStr::new("--plan"),
                path.as_os_str(),
                OsStr::new("--out"),
                output.as_os_str(),
            ],
            false,
        );
        assert!(!output.exists(), "identity refusal must precede output reservation");
    }

    let malformed = root.path().join("malformed-source.bin");
    fs::write(&malformed, b"not a world checkpoint").unwrap();
    let corrupted = root.path().join("corrupted-source.bin");
    let mut changed = bytes.clone();
    let midpoint = changed.len() / 2;
    changed[midpoint] ^= 1;
    fs::write(&corrupted, &changed).unwrap();
    let trailing = root.path().join("trailing-source.bin");
    let mut extra = bytes.clone();
    extra.push(0);
    fs::write(&trailing, &extra).unwrap();
    let oversized = root.path().join("oversized-source.bin");
    fs::File::create(&oversized)
        .unwrap()
        .set_len(u64::try_from(scriptbots_core::MAX_WORLD_CHECKPOINT_BYTES).unwrap() + 1)
        .unwrap();
    let directory = root.path().join("source-directory");
    fs::create_dir(&directory).unwrap();
    let mut cases = vec![
        ("malformed-source", malformed),
        ("corrupted-source", corrupted),
        ("trailing-source", trailing),
        ("oversized-source", oversized),
        ("directory-source", directory),
        ("missing-source", root.path().join("missing-source.bin")),
    ];
    #[cfg(unix)]
    {
        let link = root.path().join("symlink-source.bin");
        std::os::unix::fs::symlink(&source, &link).unwrap();
        cases.push(("symlink-source", link));
    }
    for (label, input) in cases {
        let mut rejected = plan.clone();
        rejected.source_checkpoint = Some(input);
        let path = root.path().join(format!("{label}.json"));
        fs::write(&path, serde_json::to_vec(&rejected).unwrap()).unwrap();
        let output = root.path().join(format!("{label}-output"));
        run_branch_cli(
            root.path(),
            label,
            &[
                OsStr::new("--plan"),
                path.as_os_str(),
                OsStr::new("--out"),
                output.as_os_str(),
            ],
            false,
        );
        assert!(!output.exists(), "invalid source must not claim output");
    }

    let declared = root.path().join("declared-source.json");
    fs::write(&declared, serde_json::to_vec(&plan).unwrap()).unwrap();
    let output = root.path().join("false-replicates");
    let refusal = run_branch_cli(
        root.path(),
        "false-replicates",
        &[
            OsStr::new("--plan"),
            declared.as_os_str(),
            OsStr::new("--out"),
            output.as_os_str(),
            OsStr::new("--seeds"),
            OsStr::new("42,137"),
        ],
        false,
    );
    assert!(diagnostic(&refusal.stderr).contains("checkpoint"));
    assert!(!output.exists(), "one saved world cannot become independent founding replicates");
    assert_eq!(fs::read(&source).unwrap(), bytes);
}
