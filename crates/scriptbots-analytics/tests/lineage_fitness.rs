//! Integration tests for the lineage fitness report (bd-2z0.11.10).
//!
//! Validates:
//! - Registration in `Registry::builtin()` and accessibility via `sb-analyze`.
//! - Closed accounting reconciliation: `total_arrivals == living_agents + total_deaths`.
//! - Explicit right-censored lifespans (`censored: true`) vs terminated lifespans (`censored: false`).
//! - Descendant lifespan distribution and bootstrap confidence intervals.
//! - Multi-generation dynamics and turnover rate calculations.
//! - Deterministic machine and markdown output across repeated runs.

use std::process::Command;

use scriptbots_analytics::{
    LINEAGE_FITNESS_SCHEMA_ID_V1, REPORT_SCHEMA_VERSION, ReaderCtx, Registry, ReportParams,
};
use scriptbots_core::{
    AgentUid, BirthOrigin, BirthRecord, DeathCause, DeathRecord, Generation, PersistenceBatch,
    PersistenceEvent, PersistenceEventKind, Position, Tick, TickSummary,
};
use scriptbots_storage::Storage;

fn batch(tick: u64, births_count: usize, deaths_count: usize) -> PersistenceBatch {
    let mut events = Vec::new();
    if births_count > 0 {
        events.push(PersistenceEvent::new(
            PersistenceEventKind::Births,
            births_count,
        ));
    }
    if deaths_count > 0 {
        events.push(PersistenceEvent::new(
            PersistenceEventKind::Deaths,
            deaths_count,
        ));
    }
    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(tick),
            agent_count: 0,
            births: births_count,
            deaths: deaths_count,
            total_energy: 0.0,
            average_energy: 0.0,
            average_health: 1.0,
            max_age: 0,
            spike_hits: 0,
        },
        epoch: 1,
        closed: false,
        metrics: Vec::new(),
        events,
        agents: Vec::new(),
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events: Vec::new(),
        narrative_events: Vec::new(),
        genomes: Vec::new(),
    }
}

fn birth_record(
    tick: u64,
    uid: u64,
    parent_a: Option<u64>,
    parent_b: Option<u64>,
    generation: u32,
    origin: BirthOrigin,
) -> BirthRecord {
    BirthRecord {
        tick: Tick(tick),
        agent_uid: AgentUid(uid),
        spawn_ordinal: uid,
        birth_ordinal: if origin == BirthOrigin::Born {
            Some(uid)
        } else {
            None
        },
        origin,
        parent_a: parent_a.map(AgentUid),
        parent_b: parent_b.map(AgentUid),
        brain_kind: Some("mlp".to_owned()),
        brain_key: Some(1),
        herbivore_tendency: 0.5,
        generation: Generation(generation),
        position: Position::new(10.0, 10.0),
        is_hybrid: parent_a.is_some() && parent_b.is_some(),
    }
}

fn death_record(tick: u64, uid: u64, age: u32, cause: DeathCause) -> DeathRecord {
    DeathRecord {
        tick: Tick(tick),
        agent_uid: AgentUid(uid),
        age,
        generation: Generation(0),
        herbivore_tendency: 0.5,
        brain_kind: Some("mlp".to_owned()),
        brain_key: Some(1),
        energy: 0.0,
        food_balance_total: 0.0,
        cause,
        was_hybrid: false,
        combat_flags: scriptbots_core::CombatEventFlags::default(),
    }
}

/// Creates a populated run database with 2 founders and 3 offspring across 20 ticks.
fn populated_fixture(dir: &tempfile::TempDir) -> String {
    let path = dir.path().join("lineage_run.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create run db");

    // Tick 0: Seed 2 founders (seeded arrivals do not count as demographic births)
    let mut b0 = batch(0, 0, 0);
    b0.births
        .push(birth_record(0, 100, None, None, 0, BirthOrigin::Seeded));
    b0.births
        .push(birth_record(0, 200, None, None, 0, BirthOrigin::Seeded));
    b0.summary.agent_count = 2;
    storage.persist(&b0).expect("persist tick 0");

    // Tick 4: Offspring 201 born to Founder 200
    let mut b4 = batch(4, 1, 0);
    b4.births
        .push(birth_record(4, 201, Some(200), None, 1, BirthOrigin::Born));
    b4.summary.agent_count = 3;
    storage.persist(&b4).expect("persist tick 4");

    // Tick 5: Offspring 101 born to Founder 100
    let mut b5 = batch(5, 1, 0);
    b5.births
        .push(birth_record(5, 101, Some(100), None, 1, BirthOrigin::Born));
    b5.summary.agent_count = 4;
    storage.persist(&b5).expect("persist tick 5");

    // Tick 8: Founder 200 dies
    let mut b8 = batch(8, 0, 1);
    b8.deaths.push(death_record(8, 200, 8, DeathCause::Aging));
    b8.summary.agent_count = 3;
    storage.persist(&b8).expect("persist tick 8");

    // Tick 9: Offspring 201 dies (Founder 200's lineage goes extinct)
    let mut b9 = batch(9, 0, 1);
    b9.deaths
        .push(death_record(9, 201, 5, DeathCause::Starvation));
    b9.summary.agent_count = 2;
    storage.persist(&b9).expect("persist tick 9");

    // Tick 10: Grandchild 102 born to Offspring 101
    let mut b10 = batch(10, 1, 0);
    b10.births
        .push(birth_record(10, 102, Some(101), None, 2, BirthOrigin::Born));
    b10.summary.agent_count = 3;
    storage.persist(&b10).expect("persist tick 10");

    // Tick 20: Run terminates with 3 living agents (100, 101, 102)
    let mut b20 = batch(20, 0, 0);
    b20.summary.agent_count = 3;
    storage.persist(&b20).expect("persist tick 20");

    storage.flush().expect("flush");
    storage.close().expect("close");
    path
}

#[test]
fn registry_exposes_lineage_fitness() {
    let registry = Registry::builtin();
    let reports = registry.list();
    let found = reports.iter().find(|(name, _)| *name == "lineage-fitness");
    assert!(
        found.is_some(),
        "lineage-fitness must be registered in Registry::builtin()"
    );
}

#[test]
fn empty_database_reconciles_cleanly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("empty.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&path).expect("create empty db");
    storage.flush().expect("flush");
    storage.close().expect("close");

    let cx = ReaderCtx::open(&path).expect("open reader");
    let out = Registry::builtin()
        .run("lineage-fitness", &cx, &ReportParams::default())
        .expect("run lineage-fitness");

    assert_eq!(out.schema_version, REPORT_SCHEMA_VERSION);
    assert_eq!(out.report, "lineage-fitness");
    assert_eq!(out.row_count, 0);

    let m = &out.machine;
    assert_eq!(m["schema"], LINEAGE_FITNESS_SCHEMA_ID_V1);
    assert_eq!(m["reconciliation"]["total_arrivals"], 0);
    assert_eq!(m["reconciliation"]["total_deaths"], 0);
    assert_eq!(m["reconciliation"]["living_agents"], 0);
    assert_eq!(m["reconciliation"]["arrivals_accounted"], true);
    assert_eq!(m["reconciliation"]["founder_count"], 0);
    assert_eq!(m["reconciliation"]["contribution_share_reconciled"], true);
    assert_eq!(m["evolutionary_change"]["turnover_rate"], 0.0);
    assert!(out.human_md.contains("No founder lineages recorded"));
}

#[test]
fn reconciled_lineage_fitness_on_persisted_run() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = populated_fixture(&dir);

    let cx = ReaderCtx::open(&path).expect("open reader");
    let out = Registry::builtin()
        .run("lineage-fitness", &cx, &ReportParams::default())
        .expect("run lineage-fitness");

    assert_eq!(out.schema_version, REPORT_SCHEMA_VERSION);
    assert_eq!(out.report, "lineage-fitness");
    assert_eq!(out.row_count, 2, "two founder lineages");

    let m = &out.machine;
    assert_eq!(m["latest_tick"], 20);

    // Closed accounting reconciliation: 5 arrivals == 3 living + 2 dead
    let rec = &m["reconciliation"];
    assert_eq!(rec["total_arrivals"], 5);
    assert_eq!(rec["total_deaths"], 2);
    assert_eq!(rec["living_agents"], 3);
    assert_eq!(rec["arrivals_accounted"], true);
    assert_eq!(rec["founder_count"], 2);
    assert_eq!(rec["living_founders"], 1);
    assert_eq!(rec["contribution_share_reconciled"], true);

    // Founder records
    let founders = m["founders"].as_array().expect("founders array");
    assert_eq!(founders.len(), 2);

    // Founder 100 (dominates living population)
    let f100 = &founders[0];
    assert_eq!(f100["founder_uid"], 100);
    assert_eq!(f100["origin"], "seeded");
    assert_eq!(f100["is_living"], true);
    assert_eq!(f100["censored"], true);
    assert_eq!(f100["lifespan_ticks"], 20); // 20 - 0
    assert_eq!(f100["direct_offspring_count"], 1); // 101
    assert_eq!(f100["total_descendants"], 2); // 101, 102
    assert_eq!(f100["living_descendants"], 2); // 101, 102
    assert_eq!(f100["living_lineage_members"], 3); // 100, 101, 102
    assert_eq!(f100["contribution_share"], 1.0); // 3 / 3
    assert_eq!(f100["max_generation_depth"], 2);

    // Founder 200 (extinct lineage)
    let f200 = &founders[1];
    assert_eq!(f200["founder_uid"], 200);
    assert_eq!(f200["origin"], "seeded");
    assert_eq!(f200["is_living"], false);
    assert_eq!(f200["censored"], false);
    assert_eq!(f200["lifespan_ticks"], 8); // 8 - 0
    assert_eq!(f200["direct_offspring_count"], 1); // 201
    assert_eq!(f200["total_descendants"], 1); // 201
    assert_eq!(f200["living_descendants"], 0);
    assert_eq!(f200["living_lineage_members"], 0);
    assert_eq!(f200["contribution_share"], 0.0);

    // Evolutionary change: 1 of 2 lineages extinct -> turnover = 0.5
    let evo = &m["evolutionary_change"];
    assert_eq!(evo["extinct_lineages"], 1);
    assert_eq!(evo["surviving_lineages"], 1);
    assert_eq!(evo["turnover_rate"], 0.5);
    assert_eq!(evo["max_founder_dominance"], 1.0);

    // Generations: 0, 1, 2
    let gens = evo["generations"].as_array().expect("generations array");
    assert_eq!(gens.len(), 3);
    assert_eq!(gens[0]["generation"], 0);
    assert_eq!(gens[0]["total_born"], 2);
    assert_eq!(gens[0]["living"], 1);
    assert_eq!(gens[0]["dead"], 1);

    assert_eq!(gens[1]["generation"], 1);
    assert_eq!(gens[1]["total_born"], 2);
    assert_eq!(gens[1]["living"], 1);
    assert_eq!(gens[1]["dead"], 1);

    assert_eq!(gens[2]["generation"], 2);
    assert_eq!(gens[2]["total_born"], 1);
    assert_eq!(gens[2]["living"], 1);
    assert_eq!(gens[2]["dead"], 0);

    // Human markdown contains key sections
    assert!(
        out.human_md
            .contains("# Lineage Fitness & Evolutionary Report")
    );
    assert!(out.human_md.contains("RECONCILED"));
    assert!(out.human_md.contains("## Founder Lineage Fitness"));
    assert!(out.human_md.contains("## Generation Dynamics"));
}

#[test]
fn censored_vs_uncensored_lifespan_semantics() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = populated_fixture(&dir);

    let cx = ReaderCtx::open(&path).expect("open reader");
    let out = Registry::builtin()
        .run("lineage-fitness", &cx, &ReportParams::default())
        .expect("run lineage-fitness");

    let founders = out.machine["founders"].as_array().expect("founders");
    for f in founders {
        let is_living = f["is_living"].as_bool().unwrap();
        let censored = f["censored"].as_bool().unwrap();
        assert_eq!(
            is_living, censored,
            "right-censored iff still living at run end"
        );
        if !is_living {
            assert!(
                !f["death_tick"].is_null(),
                "dead founder must have explicit death_tick"
            );
            assert!(
                !f["death_cause"].is_null(),
                "dead founder must have explicit death_cause"
            );
        }
    }
}

#[test]
fn descendant_lifespan_bootstrap_ci_uncertainty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = populated_fixture(&dir);

    let cx = ReaderCtx::open(&path).expect("open reader");
    let params = ReportParams::default();
    let out = Registry::builtin()
        .run("lineage-fitness", &cx, &params)
        .expect("run lineage-fitness");

    let f100 = &out.machine["founders"][0];
    let ls = &f100["descendant_lifespan"];
    assert!(
        ls.is_object(),
        "founder 100 has 2 descendants so lifespan summary exists"
    );
    assert_eq!(ls["count"], 2);
    assert_eq!(
        ls["censored_count"], 2,
        "both descendants 101 and 102 are alive at tick 20"
    );

    let mean = ls["mean"].as_f64().unwrap();
    let ci_low = ls["ci_low"].as_f64().unwrap();
    let ci_high = ls["ci_high"].as_f64().unwrap();
    assert!(
        ci_low <= ci_high,
        "ci_low <= ci_high: [{ci_low}, {ci_high}]"
    );
    assert!(
        ci_low <= mean + 1e-9 && mean <= ci_high + 1e-9,
        "mean within CI bounds: {mean} in [{ci_low}, {ci_high}]"
    );
}

#[test]
fn deterministic_output_guarantee() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = populated_fixture(&dir);

    let cx = ReaderCtx::open(&path).expect("open reader");
    let params = ReportParams::default();
    let run1 = Registry::builtin()
        .run("lineage-fitness", &cx, &params)
        .expect("run 1");
    let run2 = Registry::builtin()
        .run("lineage-fitness", &cx, &params)
        .expect("run 2");

    assert_eq!(
        run1.machine, run2.machine,
        "machine payload must be bit-identical across runs"
    );
    assert_eq!(
        run1.human_md, run2.human_md,
        "markdown rendering must be identical across runs"
    );
}

#[test]
fn cli_sb_analyze_invocation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = populated_fixture(&dir);

    let output = Command::new(env!("CARGO_BIN_EXE_sb-analyze"))
        .arg(&path)
        .arg("run")
        .arg("lineage-fitness")
        .arg("--params")
        .arg("top_founders=5")
        .arg("--params")
        .arg("resamples=100")
        .output()
        .expect("execute sb-analyze");

    assert!(
        output.status.success(),
        "sb-analyze must succeed, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("# Lineage Fitness & Evolutionary Report"));
    assert!(stdout.contains("RECONCILED"));
}
