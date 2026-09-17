//! Retained-evidence verification and same-build canonical cohort re-execution.
use super::notebook::{NotebookContext, RunRef};
use super::stats::{AnalysisParams, MatchedSeedAnalysis, RunSummary, analyze_matched_seed_runs};
use crate::experiment_runner::{
    ExperimentBatchStatus, MatchedSeedCohort, MatchedSeedExperimentRunner, RunRecord, RunState,
    ScenarioVariant,
};
use crate::{BuildProvenanceV0, RunManifestV3};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

const SCHEMA: u32 = 1;
const MAX_INPUT_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, clap::Args)]
pub struct ReproduceArgs {
    /// Retained, immutable reproduction input file.
    #[arg(long)]
    pub input: PathBuf,
    /// BLAKE3 pinned by the emitted script, not read from the input itself.
    #[arg(long)]
    pub expected_digest: String,
    /// Internal bounded-worker invocation; output is allocated by the parent.
    #[arg(long, hide = true, requires = "run_index")]
    pub child_output: Option<PathBuf>,
    #[arg(long, hide = true, requires = "child_output")]
    pub run_index: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SerializedRun {
    reference: RunRef,
    metrics: BTreeMap<String, f64>,
}

impl SerializedRun {
    fn from_summary(summary: &RunSummary) -> Self {
        Self {
            reference: RunRef::from(summary),
            metrics: summary.metrics.clone(),
        }
    }
    fn scientific(&self) -> Self {
        let mut row = self.clone();
        row.reference.summary_path = None;
        row
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReproductionInput {
    schema: u32,
    session_id: String,
    experiment_id: String,
    cohort: MatchedSeedCohort,
    variants: Vec<ScenarioVariant>,
    ticks: u64,
    context: NotebookContext,
    runs: Vec<SerializedRun>,
    analysis: Option<MatchedSeedAnalysis>,
    configs: Vec<String>,
}

/// Reject lexical traversal and every pre-existing symlink, including ancestors.
/// Paths are locally derived; this also refuses a user-planted output symlink.
pub(crate) fn confined_directory(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut current = PathBuf::new();
    for component in absolute.components() {
        ensure!(
            !matches!(component, Component::ParentDir),
            "path traversal refused: {}",
            path.display()
        );
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) => ensure!(
                metadata.is_dir() && !metadata.file_type().is_symlink(),
                "non-directory or symlink refused: {}",
                current.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => fs::create_dir(&current)?,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(fs::canonicalize(current)?)
}

fn checked_file(path: &Path) -> Result<Vec<u8>> {
    let parent = path.parent().context("file has no parent")?;
    // Verification never creates an absent parent.
    ensure!(parent.exists(), "missing parent: {}", parent.display());
    let canonical = confined_directory(parent)?;
    let name = path.file_name().context("file has no name")?;
    let resolved = canonical.join(name);
    let metadata = fs::symlink_metadata(&resolved)?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "symlink or non-file refused: {}",
        resolved.display()
    );
    ensure!(
        metadata.len() <= MAX_INPUT_BYTES,
        "artifact exceeds read bound: {}",
        resolved.display()
    );
    let mut file = OpenOptions::new();
    file.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        file.custom_flags(libc::O_NOFOLLOW);
    }
    let mut bytes = Vec::new();
    file.open(&resolved)?
        .take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_INPUT_BYTES,
        "artifact grew beyond read bound"
    );
    Ok(bytes)
}

/// Immutable writes: identical rerenders are allowed; different retained bytes are never replaced.
pub(crate) fn retain_file(root: &Path, name: &str, bytes: &[u8]) -> Result<PathBuf> {
    ensure!(
        Path::new(name).components().count() == 1 && !matches!(name, "." | ".."),
        "invalid artifact name"
    );
    let root = confined_directory(root)?;
    let path = root.join(name);
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => ensure!(
            checked_file(&path)? == bytes,
            "retained artifact tampered; refusing overwrite: {}",
            path.display()
        ),
        Err(error) => return Err(error.into()),
    }
    ensure!(
        fs::canonicalize(&path)?.parent() == Some(root.as_path()),
        "artifact escaped output root"
    );
    Ok(path)
}

fn records(runs: &[RunRef]) -> Result<Vec<RunRecord>> {
    runs.iter()
        .map(|run| {
            let summary = Path::new(
                run.summary_path
                    .as_deref()
                    .context("missing retained summary path")?,
            );
            checked_file(summary)?;
            let bundle = summary
                .parent()
                .and_then(Path::parent)
                .context("invalid retained summary location")?;
            ensure!(
                summary.file_name().and_then(|s| s.to_str()) == Some("summary.csv")
                    && summary
                        .parent()
                        .and_then(Path::file_name)
                        .and_then(|s| s.to_str())
                        == Some("exports"),
                "noncanonical summary path"
            );
            Ok(RunRecord {
                run_id: run.run_id.clone(),
                variant_id: run.variant_id.clone(),
                brain_family: run.brain_family.clone(),
                seed: run.seed,
                state: RunState::Completed,
                total_ticks: run.total_ticks,
                final_digest: Some(run.digest.clone()),
                bundle_path: Some(bundle.to_string_lossy().into_owned()),
                error_reason: None,
            })
        })
        .collect()
}

/// Build identity with launch/pool thread settings masked: those are runtime
/// environment (the renderer already masks them), so reproduction compares the
/// rest byte-for-byte.
fn same_build_identity(recorded: &BuildProvenanceV0) -> bool {
    let mut expected = BuildProvenanceV0::current();
    expected.rayon_num_threads = recorded.rayon_num_threads.clone();
    expected.scriptbots_max_threads = recorded.scriptbots_max_threads.clone();
    expected.core.rayon_threads = recorded.core.rayon_threads;
    recorded == &expected
}

fn verify_rows(input: &ReproductionInput, run_records: Vec<RunRecord>) -> Result<Vec<RunSummary>> {
    let arms = input
        .variants
        .iter()
        .map(|v| v.config_overrides.clone())
        .collect::<Vec<_>>();
    for record in &run_records {
        let bundle = Path::new(record.bundle_path.as_deref().context("missing bundle")?);
        let wire: scriptbots_storage::RunBundleV1 =
            serde_json::from_slice(&checked_file(&bundle.join("bundle_manifest.json"))?)?;
        let manifest: RunManifestV3 = serde_json::from_str(&wire.manifest.manifest_json)?;
        ensure!(
            same_build_identity(&manifest.build),
            "build provenance mismatch run={} arm={} seed={}",
            record.run_id,
            record.variant_id,
            record.seed
        );
        ensure!(
            manifest.identity.experiment_id.as_deref() == Some(input.experiment_id.as_str()),
            "experiment identity mismatch"
        );
        // The manifest's normalized config must equal what THIS build derives from
        // the arm's cited overrides at the row's seed: a tampered override or a
        // bundle detached from its plan fails here, not only at digest compare.
        let arm_index = record
            .variant_id
            .strip_prefix("arm-")
            .and_then(|rest| rest.parse::<usize>().ok())
            .ok_or_else(|| anyhow::anyhow!("noncanonical variant_id {}", record.variant_id))?;
        let arm_overrides = arms.get(arm_index).ok_or_else(|| {
            anyhow::anyhow!("variant {} outside the cited plan", record.variant_id)
        })?;
        let expected_config = crate::experiment_runner::config_for_run(arm_overrides, record.seed)
            .map_err(anyhow::Error::msg)?;
        let expected_value = serde_json::to_value(&expected_config)?;
        ensure!(
            manifest.normalized_config == expected_value,
            "normalized config mismatch run={} variant={} seed={}",
            record.run_id,
            record.variant_id,
            record.seed
        );
        let evidence: serde_json::Value =
            serde_json::from_slice(&checked_file(&bundle.join("evidence/run.json"))?)?;
        ensure!(
            evidence["run_id"] == record.run_id
                && evidence["cohort_id"] == input.cohort.cohort_id
                && evidence["seed"] == record.seed
                && evidence["final_digest"].as_str() == record.final_digest.as_deref(),
            "evidence mismatch run={}",
            record.run_id
        );
    }
    let count = run_records.len();
    let status = ExperimentBatchStatus {
        schema_version: 2,
        generation: 0,
        plan_digest: String::new(),
        experiment_id: input.experiment_id.clone(),
        total_runs: count,
        completed_runs: count,
        failed_runs: 0,
        runs: run_records,
    };
    crate::lab_assistant::completed_run_summaries(&arms, &status).map_err(anyhow::Error::msg)
}

fn runner(input: &ReproductionInput, output: &Path) -> MatchedSeedExperimentRunner {
    MatchedSeedExperimentRunner::new(
        &input.experiment_id,
        input.cohort.clone(),
        input.variants.clone(),
        input.ticks,
        1,
        output,
    )
}

pub(crate) fn emit(
    session_id: &str,
    runs: &[RunRef],
    context: &NotebookContext,
    analysis_choices: Option<(Vec<String>, AnalysisParams)>,
    configs: Vec<String>,
    out: &Path,
) -> Result<String> {
    let mut input = ReproductionInput {
        schema: SCHEMA,
        session_id: session_id.to_owned(),
        experiment_id: session_id.to_owned(),
        cohort: MatchedSeedCohort {
            cohort_id: format!("{session_id}-matched-seeds"),
            seeds: Vec::new(),
        },
        variants: Vec::new(),
        ticks: 0,
        context: context.clone(),
        runs: Vec::new(),
        analysis: None,
        configs,
    };
    if let Some(first) = runs.first() {
        let suffix = format!("-{}-seed{}", first.variant_id, first.seed);
        input.experiment_id = first
            .run_id
            .strip_suffix(&suffix)
            .context("run identity is not canonical")?
            .to_owned();
        let recs = records(runs)?;
        let bundle = Path::new(recs[0].bundle_path.as_deref().context("missing bundle")?);
        let evidence: serde_json::Value =
            serde_json::from_slice(&checked_file(&bundle.join("evidence/run.json"))?)?;
        input.cohort.cohort_id = evidence["cohort_id"]
            .as_str()
            .context("missing cohort identity")?
            .to_owned();
        input.ticks = first.total_ticks;
        let mut variants = BTreeMap::new();
        let mut seeds = std::collections::BTreeSet::new();
        for run in runs {
            ensure!(
                run.total_ticks == input.ticks,
                "unequal tick horizons cannot form one matched-seed plan"
            );
            ensure!(
                run.variant_id == format!("arm-{:03}", run.arm_id),
                "noncanonical arm identity"
            );
            let variant = ScenarioVariant {
                variant_id: run.variant_id.clone(),
                brain_family: run.brain_family.clone(),
                config_overrides: run.config_overrides.clone(),
            };
            if let Some(old) = variants.insert(run.arm_id, variant.clone()) {
                ensure!(old == variant, "arm config differs between seeds");
            }
            seeds.insert(run.seed);
        }
        for (expected, (arm, variant)) in variants.into_iter().enumerate() {
            ensure!(usize::from(arm) == expected, "noncontiguous arm IDs");
            input.variants.push(variant);
        }
        input.cohort.seeds = seeds.into_iter().collect();
        // Canonicalize the cited order to the plan's own arm-then-seed order so a
        // shuffled caller is normalized, not rejected (bd-16g.1.7 ordering rule).
        let planned = runner(&input, out).plan_batch()?;
        ensure!(planned.len() == runs.len(), "missing arm/seed run");
        let mut by_key = BTreeMap::new();
        for run in runs {
            by_key.insert((run.arm_id, run.seed), run);
        }
        ensure!(
            by_key.len() == planned.len(),
            "duplicate run identities in the cited cohort"
        );
        for (planned, actual) in planned.iter().zip(by_key.values()) {
            ensure!(
                planned.run_id == actual.run_id && planned.seed == actual.seed,
                "plan does not match the cited cohort"
            );
        }
        let ordered: Vec<RunRef> = by_key.into_values().cloned().collect();
        let summaries = verify_rows(&input, recs)?;
        for (summary, reference) in summaries.iter().zip(ordered.iter()) {
            ensure!(
                RunRef::from(summary) == *reference,
                "retained summary/provenance mismatch run={}",
                reference.run_id
            );
        }
        let (metrics, params) = analysis_choices
            .unwrap_or_else(|| (vec!["alive_agents".to_owned()], AnalysisParams::default()));
        if input.variants.len() >= 2 {
            input.analysis = Some(analyze_matched_seed_runs(&summaries, &metrics, params)?);
        }
        input.runs = summaries.iter().map(SerializedRun::from_summary).collect();
    }
    let root = confined_directory(out)?;
    for (index, config) in input.configs.iter().enumerate() {
        retain_file(&root, &format!("config-{index:04}.toml"), config.as_bytes())?;
    }
    let bytes = serde_json::to_vec_pretty(&input)?;
    let digest = blake3::hash(&bytes).to_hex().to_string();
    retain_file(&root, "reproduction.json", &bytes)?;
    retain_file(
        &root,
        "summaries.json",
        &serde_json::to_vec_pretty(&input.runs)?,
    )?;
    retain_file(
        &root,
        "analysis.json",
        &serde_json::to_vec_pretty(&input.analysis)?,
    )?;
    Ok(digest)
}

fn compare_rows(expected: &[SerializedRun], actual: &[RunSummary]) -> Result<()> {
    ensure!(expected.len() == actual.len(), "summary row count mismatch");
    for (expected, actual) in expected.iter().zip(actual) {
        ensure!(
            expected.scientific() == SerializedRun::from_summary(actual).scientific(),
            "scientific summary mismatch run={} arm={} seed={} digest={} actual_digest={}",
            expected.reference.run_id,
            expected.reference.arm_id,
            expected.reference.seed,
            expected.reference.digest,
            actual.digest
        );
    }
    Ok(())
}

/// Execute a pinned reproduction, or one parent-allocated worker, without consulting a model.
///
/// # Errors
/// Refuses mismatched schemas, retained evidence, builds, seeds, scientific summaries or tables.
pub fn run(args: &ReproduceArgs) -> Result<()> {
    let bytes = checked_file(&args.input)?;
    ensure!(
        blake3::hash(&bytes).to_hex().as_str() == args.expected_digest,
        "reproduction input BLAKE3 mismatch"
    );
    let input: ReproductionInput = serde_json::from_slice(&bytes)?;
    ensure!(input.schema == SCHEMA, "reproduction schema mismatch");
    ensure!(
        same_build_identity(&input.context.build),
        "expected source/build provenance mismatch"
    );
    let root = confined_directory(args.input.parent().context("missing input directory")?)?;
    ensure!(
        !input.runs.is_empty(),
        "partial notebook has no completed cohort to reproduce"
    );
    ensure!(
        input.configs.len() == input.runs.len(),
        "config/run count mismatch"
    );
    for (index, config) in input.configs.iter().enumerate() {
        ensure!(
            checked_file(&root.join(format!("config-{index:04}.toml")))? == config.as_bytes(),
            "retained config tampered index={index}"
        );
    }
    let retained_runs: Vec<SerializedRun> =
        serde_json::from_slice(&checked_file(&root.join("summaries.json"))?)?;
    ensure!(
        retained_runs == input.runs,
        "retained summary table mismatch"
    );
    let retained_analysis: Option<MatchedSeedAnalysis> =
        serde_json::from_slice(&checked_file(&root.join("analysis.json"))?)?;
    ensure!(
        retained_analysis == input.analysis,
        "retained raw/adjusted analysis table mismatch"
    );
    if let (Some(output), Some(index)) = (&args.child_output, args.run_index) {
        let output = confined_directory(output)?;
        ensure!(
            output.starts_with(&root) && output != root,
            "worker output escapes reproduction directory"
        );
        let expected = input
            .runs
            .get(index)
            .context("worker run index outside plan")?;
        let variant = input
            .variants
            .get(usize::from(expected.reference.arm_id))
            .context("worker arm outside plan")?;
        let record =
            runner(&input, &output).execute_single_run(variant, expected.reference.seed)?;
        retain_file(
            &output,
            &format!("record-{index:04}.json"),
            &serde_json::to_vec(&record)?,
        )?;
        return Ok(());
    }
    let references = input
        .runs
        .iter()
        .map(|r| r.reference.clone())
        .collect::<Vec<_>>();
    let retained = verify_rows(&input, records(&references)?)?;
    compare_rows(&input.runs, &retained)?;
    if let Some(analysis) = &input.analysis {
        ensure!(
            analyze_matched_seed_runs(&retained, &analysis.metrics, analysis.params)? == *analysis,
            "retained canonical analysis differs"
        );
    }
    let output = allocate_output(&root)?;
    eprintln!(
        "[REPRODUCE] session={} output={} input_digest={} runs={} ticks_charged={} tokens_charged={}",
        input.session_id,
        output.display(),
        args.expected_digest,
        input.runs.len(),
        input.context.ticks_charged,
        input.context.tokens_charged
    );
    let mut completed = Vec::with_capacity(input.runs.len());
    for (index, row) in input.runs.iter().enumerate() {
        let mut command = std::process::Command::new(std::env::current_exe()?);
        command
            .arg("lab-reproduce")
            .arg("--input")
            .arg(fs::canonicalize(&args.input)?)
            .arg("--expected-digest")
            .arg(&args.expected_digest)
            .arg("--child-output")
            .arg(&output)
            .arg("--run-index")
            .arg(index.to_string());
        eprintln!(
            "[CHILD] session={} command_id={index} run={} arm={} seed={} config_digest={} expected_digest={}",
            input.session_id,
            row.reference.run_id,
            row.reference.arm_id,
            row.reference.seed,
            row.reference.config_digest,
            row.reference.digest
        );
        MatchedSeedExperimentRunner::run_bounded_child(
            &mut command,
            std::time::Duration::from_secs(300),
        )?;
        let record: RunRecord = serde_json::from_slice(&checked_file(
            &output.join(format!("record-{index:04}.json")),
        )?)?;
        ensure!(
            record.state == RunState::Completed && record.run_id == row.reference.run_id,
            "child record mismatch command_id={index}"
        );
        completed.push(record);
    }
    let regenerated = verify_rows(&input, completed)?;
    compare_rows(&input.runs, &regenerated)?;
    let analysis = input
        .analysis
        .as_ref()
        .map(|expected| analyze_matched_seed_runs(&regenerated, &expected.metrics, expected.params))
        .transpose()?;
    ensure!(
        analysis == input.analysis,
        "regenerated raw/adjusted tables differ from cited analysis"
    );
    let rows = regenerated
        .iter()
        .map(SerializedRun::from_summary)
        .collect::<Vec<_>>();
    retain_file(
        &output,
        "summaries.json",
        &serde_json::to_vec_pretty(&rows)?,
    )?;
    let table = serde_json::to_vec_pretty(&analysis)?;
    retain_file(&output, "analysis.json", &table)?;
    eprintln!(
        "[VERIFY] session={} output={} table_digest={} diff=equal completed={} timeout=false exit=0",
        input.session_id,
        output.display(),
        blake3::hash(&table).to_hex(),
        rows.len()
    );
    Ok(())
}

fn allocate_output(root: &Path) -> Result<PathBuf> {
    for index in 0..1_000_000_u32 {
        let output = root.join(format!("rerun-{index:06}"));
        match fs::create_dir(&output) {
            Ok(()) => return confined_directory(&output),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                ensure!(
                    !fs::symlink_metadata(&output)?.file_type().is_symlink(),
                    "rerun output symlink refused"
                );
            }
            Err(error) => return Err(error.into()),
        }
    }
    bail!("reproduction output namespace exhausted")
}
