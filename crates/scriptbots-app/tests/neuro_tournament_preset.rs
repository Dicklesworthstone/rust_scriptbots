//! The shipped NeuroFlow comparison is a runnable, bounded architecture comparison.

use scriptbots_app::tournament::{OrderPolicy, TournamentSpecFile, plan};
use scriptbots_brain::BrainKind;

#[test]
fn shipped_neuroflow_preset_keeps_capacity_claims_and_work_bounded() {
    let file = TournamentSpecFile::from_toml_str(include_str!(
        "../../../tournament/neuro_comparison.toml"
    ))
    .expect("shipped NeuroFlow comparison");
    assert!(file.parameters.is_none());
    assert!(
        file.reproducibility
            .as_ref()
            .expect("explicit repeatability policy")
            .require_identical_rerun
    );
    for spec in [file.to_spec().unwrap(), file.to_smoke_spec().unwrap()] {
        assert_eq!(
            spec.families,
            vec![BrainKind::new("mlp"), BrainKind::new("neuro")]
        );
        assert!(spec.closed);
        assert_eq!(spec.order_policy, OrderPolicy::BothAssignments);
        let assignments = plan(&spec).expect("balanced comparison plan");
        assert_eq!(assignments.len(), spec.seeds.len() * 2);
        assert!(assignments.len() as u64 * spec.ticks * 2 <= 4096);
        assert!(assignments.iter().all(|entry| {
            entry.cohort[&BrainKind::new("mlp")] == entry.cohort[&BrainKind::new("neuro")]
        }));
    }
}

#[cfg(feature = "neuro")]
#[test]
fn shipped_neuroflow_smoke_runs_through_the_real_cli_and_retains_family_evidence() {
    use scriptbots_app::tournament::TournamentResultRow;
    use std::{fs, path::Path, process::Command};

    let root = tempfile::tempdir().unwrap();
    let spec = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tournament/neuro_comparison.toml");
    let output = root.path().join("comparison");
    let result = Command::new(env!("CARGO_BIN_EXE_scriptbots-app"))
        .current_dir(root.path())
        .arg("tournament")
        .arg("--spec")
        .arg(&spec)
        .arg("--smoke")
        .arg("--out")
        .arg(&output)
        .args(["--jobs", "1"])
        .env("RAYON_NUM_THREADS", "1")
        .env("SCRIPTBOTS_MAX_THREADS", "1")
        .output()
        .expect("launch shipped comparison through the application");
    assert!(
        result.status.success(),
        "NeuroFlow comparison CLI failed: stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr),
    );
    let results = fs::read_to_string(output.join("tournament_results.jsonl")).unwrap();
    let rows = TournamentResultRow::deserialize_jsonl(&results).unwrap();
    assert_eq!(rows.len(), 8, "two seeds, two orders, two families");
    let neuro: Vec<_> = rows.iter().filter(|row| row.family == "neuro").collect();
    assert_eq!(neuro.len(), 4);
    assert!(neuro.iter().all(|row| {
        row.ticks_run == 32
            && row.initial_genome_digest.len() == 64
            && row.initial_brain_source_scalars > 0
            && row.initial_mutation_rates.len() == 4
            && row.warnings.iter().any(|warning| {
                warning.contains("validated NeuroFlow genome topology")
                    && warning.contains("not recorded tick-zero activity")
            })
    }));
    assert!(
        rows.iter()
            .all(|row| matches!(row.family.as_str(), "mlp" | "neuro"))
    );
    let ratings: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("ratings.json")).unwrap()).unwrap();
    assert!(ratings.is_object() || ratings.is_array());
    let leaderboard = fs::read_to_string(output.join("leaderboard.md")).unwrap();
    assert!(leaderboard.contains("neuro-survival_share"));
    assert!(leaderboard.contains("without claiming tick-zero activity"));
}
