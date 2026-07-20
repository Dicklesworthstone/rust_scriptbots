//! Cohort validation for the curated scenario catalog (bd-2z0.10.3).
//!
//! Every catalog document must run its full seed cohort through the real binary,
//! satisfy its declared measurable envelope on EVERY seed, and replay identically
//! on reruns — including its scheduled interventions. No attractive-seed selection,
//! no undocumented overrides: the document is the whole story.

use std::path::{Path, PathBuf};
use std::process::Command;

use scriptbots_app::{ScenarioDocumentV1, ScenarioEnvelopeV1};

const CATALOG: [&str; 6] = [
    "meadow",
    "drought_recovery",
    "predator_prey",
    "islands_closed",
    "cooperation_kinship",
    "brain_arena",
];

fn catalog_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scenarios")
}

fn binary() -> PathBuf {
    let mut path = std::env::current_exe().expect("test exe");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.join("scriptbots-app")
}

fn run_dir(label: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "scriptbots_catalog_{label}_{}_{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("run dir");
    dir
}

#[derive(Debug, serde::Deserialize)]
struct HeadlessSummaryDto {
    final_agent_count: usize,
    total_births: usize,
    total_deaths: usize,
    total_spike_hits: u64,
    ticks_simulated: u64,
}

#[derive(Debug, serde::Deserialize)]
struct HeadlessReportDto {
    summary: HeadlessSummaryDto,
}

struct CohortRun {
    summary: HeadlessSummaryDto,
    world_digest: String,
}

fn launch_scenario(
    document_path: &Path,
    seed: u64,
    frames: u64,
    report_path: &Path,
) -> std::process::Output {
    let mut command = Command::new(binary());
    command
        .env_remove("SSH_CONNECTION")
        .env_remove("SSH_CLIENT")
        .env_remove("SSH_TTY")
        .env_remove("SCRIPTBOTS_MAX_THREADS")
        .env("SCRIPTBOTS_RNG_SEED", seed.to_string())
        .env("SCRIPTBOTS_MODE", "terminal")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS", "1")
        .env("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES", frames.to_string())
        .env("SCRIPTBOTS_TERMINAL_HEADLESS_REPORT", report_path)
        .env("SCRIPTBOTS_CONTROL_REST_ENABLED", "0")
        .env("SCRIPTBOTS_CONTROL_MCP", "disabled")
        .env("RUST_LOG", "info")
        .env("RUST_LOG_STYLE", "never")
        .args([
            "--scenario",
            document_path.to_str().expect("utf-8 scenario path"),
            "--storage",
            "memory",
            "--threads",
            "2",
        ]);
    command.output().expect("the app binary runs")
}

fn parse_digest(stderr: &str) -> String {
    for line in stderr.lines() {
        if let Some(index) = line.find("world_digest=") {
            return line[index + "world_digest=".len()..]
                .split_whitespace()
                .next()
                .expect("digest token")
                .to_owned();
        }
    }
    panic!("no world digest found in run output:\n{stderr}");
}

fn run_cohort(document_path: &Path, seed: u64, frames: u64) -> CohortRun {
    let dir = run_dir(&format!("s{seed}"));
    let report_path = dir.join("report.json");
    let output = launch_scenario(document_path, seed, frames, &report_path);
    assert!(
        output.status.success(),
        "scenario run failed for seed {seed}:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: HeadlessReportDto =
        serde_json::from_slice(&std::fs::read(&report_path).expect("headless report exists"))
            .expect("headless report parses");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let world_digest = parse_digest(&stderr);
    let _ = std::fs::remove_dir_all(&dir);
    CohortRun {
        summary: report.summary,
        world_digest,
    }
}

fn assert_envelope(id: &str, seed: u64, envelope: &ScenarioEnvelopeV1, run: &CohortRun) {
    assert_eq!(
        run.summary.ticks_simulated, envelope.ticks,
        "{id}/seed {seed}: the run simulated fewer ticks than the envelope horizon"
    );
    if let Some(min) = envelope.population_min {
        assert!(
            run.summary.final_agent_count >= min as usize,
            "{id}/seed {seed}: final population {} fell below the envelope floor {min}",
            run.summary.final_agent_count
        );
    }
    if let Some(max) = envelope.population_max {
        assert!(
            run.summary.final_agent_count <= max as usize,
            "{id}/seed {seed}: final population {} exceeded the envelope ceiling {max}",
            run.summary.final_agent_count
        );
    }
    if let Some(min) = envelope.births_min {
        assert!(
            run.summary.total_births >= min as usize,
            "{id}/seed {seed}: total births {} below the envelope floor {min}",
            run.summary.total_births
        );
    }
    if let Some(min) = envelope.deaths_min {
        assert!(
            run.summary.total_deaths >= min as usize,
            "{id}/seed {seed}: total deaths {} below the envelope floor {min}",
            run.summary.total_deaths
        );
    }
    if let Some(min) = envelope.spike_events_min {
        assert!(
            run.summary.total_spike_hits >= min,
            "{id}/seed {seed}: spike hits {} below the envelope floor {min}",
            run.summary.total_spike_hits
        );
    }
}

#[test]
fn every_catalog_scenario_satisfies_its_envelope_on_every_cohort_seed_and_replays_identically() {
    let mut validated = 0usize;
    for name in CATALOG {
        let document_path = catalog_dir().join(format!("{name}.scenario.toml"));
        let bytes = std::fs::read(&document_path)
            .unwrap_or_else(|error| panic!("catalog document {name} unreadable: {error}"));
        let document = ScenarioDocumentV1::parse_toml(&bytes)
            .unwrap_or_else(|error| panic!("catalog document {name} must validate: {error}"));
        assert_eq!(
            document.id, name,
            "catalog file name and scenario id must agree"
        );
        assert!(
            !document.seeds.is_empty(),
            "{name}: a curated scenario must declare its cohort seed schedule"
        );
        assert!(
            document.hypothesis.is_some(),
            "{name}: a curated scenario must state its intended phenomenon"
        );
        let envelope = document.envelope.clone().unwrap_or_else(|| {
            panic!("{name}: a curated scenario must declare a measurable envelope")
        });

        for &seed in &document.seeds {
            let first = run_cohort(&document_path, seed, envelope.ticks);
            assert_envelope(name, seed, &envelope, &first);

            // Replay: the identical scenario + seed must produce bit-identical science,
            // scheduled interventions included.
            let second = run_cohort(&document_path, seed, envelope.ticks);
            assert_eq!(
                first.world_digest, second.world_digest,
                "{name}/seed {seed}: rerun diverged — the scenario does not replay identically"
            );
        }
        validated += 1;
    }
    assert_eq!(
        validated,
        CATALOG.len(),
        "every catalog scenario was validated"
    );
}
