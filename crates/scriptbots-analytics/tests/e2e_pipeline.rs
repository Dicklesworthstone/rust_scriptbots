//! End-to-end analytics pipeline integration test (`bd-2z0.11.9`).
//!
//! Validates the complete scientific analysis journey:
//! 1. A hand-built seeded fixture with a planted regime change (population crash) and a
//!    stationary null control, so statistical reports have a known ground truth.
//! 2. Execution of the complete report suite (all 12 built-in reports via CLI & Registry).
//! 3. Ground-truth invariant assertions across reports (significance under FDR, false-positive control,
//!    lineage component conservation, descendant birth accounting).
//! 4. Parquet export (Apache `parquet` writer) with exact SQL row count equality & round-trip verification.
//! 5. Graph export verification (lineage, dynasty, interaction in `GraphML` and Edge-List).
//! 6. FTS5 narrative event search verification.
//! 7. Structured MANIFEST.json artifact emission.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use scriptbots_analytics::dataframe::{
    AnalyticsExportFormat, export_database_table, read_parquet_batch,
};
use scriptbots_analytics::{ReaderCtx, Registry, ReportParams};
use scriptbots_core::narrative::{EventKind, EventRecord};
use scriptbots_core::{
    AgentData, AgentIdentity, AgentRuntime, AgentState, AgentUid, BirthOrigin, BirthRecord,
    BrainBinding, Generation, INTERACTION_EVENTS_OBSERVED_KIND, INTERACTION_EVENTS_PERSISTED_KIND,
    MetricSample, PersistenceBatch, PersistenceEvent, PersistenceEventKind, Position, ReplayEvent,
    ReplayEventKind, ReplayInteractionKind, Tick, TickSummary, Velocity,
};
use scriptbots_storage::{ExportTable, OpenFlags, RowExt, Storage, open_with_flags};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StageRecord {
    stage: String,
    duration_ms: u128,
    status: String,
    details: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InvariantRecord {
    invariant: String,
    expected: serde_json::Value,
    observed: serde_json::Value,
    status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct E2eManifest {
    schema: String,
    run_id: String,
    verdict: String,
    total_duration_ms: u128,
    stages: Vec<StageRecord>,
    invariants: Vec<InvariantRecord>,
}

fn make_agent(uid: u64, tick: u64, generation_num: u32, parent: Option<u64>) -> AgentState {
    let data = AgentData {
        position: Position::new(
            10.0 * f32::from(u16::try_from(uid).expect("uid fits u16")),
            20.0 * f32::from(u16::try_from(tick % 10).expect("tick fits u16")),
        ),
        velocity: Velocity::new(1.0, 0.0),
        heading: 0.0,
        health: 100.0,
        generation: Generation(generation_num),
        age: u32::try_from(tick).expect("tick fits u32"),
        ..AgentData::default()
    };
    let runtime = AgentRuntime {
        energy: 120.0,
        herbivore_tendency: 0.8,
        brain: BrainBinding::Legacy {
            runner: None,
            registry_key: None,
            kind: "mlp".to_string(),
        },
        lineage: [parent.map(AgentUid), None],
        hybrid: false,
        ..AgentRuntime::default()
    };
    AgentState {
        id: scriptbots_core::AgentId::default(),
        identity: AgentIdentity {
            uid: AgentUid(uid),
            spawn_ordinal: uid - 1,
            birth_ordinal: if parent.is_some() { Some(uid) } else { None },
        },
        data,
        runtime,
    }
}

fn make_birth(
    tick: u64,
    uid: u64,
    parent: Option<u64>,
    generation_num: u32,
    origin: BirthOrigin,
) -> BirthRecord {
    BirthRecord {
        tick: Tick(tick),
        agent_uid: AgentUid(uid),
        spawn_ordinal: uid.saturating_sub(1),
        birth_ordinal: if parent.is_some() { Some(uid) } else { None },
        origin,
        parent_a: parent.map(AgentUid),
        parent_b: None,
        brain_kind: Some("mlp".to_string()),
        brain_key: None,
        herbivore_tendency: 0.8,
        generation: Generation(generation_num),
        position: Position::new(10.0, 10.0),
        is_hybrid: false,
    }
}

fn empty_summary(tick: u64, pop: usize, births: usize) -> TickSummary {
    TickSummary {
        tick: Tick(tick),
        agent_count: pop,
        births,
        deaths: 0,
        total_energy: 120.0 * f32::from(u16::try_from(pop).expect("pop fits u16")),
        average_energy: 120.0,
        average_health: 100.0,
        max_age: u32::try_from(tick).expect("tick fits u32"),
        spike_hits: 0,
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "fixture populates 200 ticks with seeded regime shift and control"
)]
fn populate_seeded_fixture(dir: &Path) -> (String, String) {
    let run_path = dir.join("run.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&run_path).expect("create run db");

    let wobble = |tick: u64| -> f64 {
        let phase = u32::try_from(tick % 5).expect("fit");
        (f64::from(phase) - 2.0) * 0.2
    };

    let noise = |tick: u64| -> f64 {
        let phase = u32::try_from(tick % 7).expect("fit");
        (f64::from(phase) - 3.0) * 0.4
    };

    for tick in 0..=200u64 {
        let pop_val = if tick < 100 { 40.0 } else { 10.0 } + wobble(tick);
        let energy_val = 50.0 + noise(tick);

        let mut agents = Vec::new();
        let mut births = Vec::new();
        let mut replay_events = Vec::new();

        if tick == 0 {
            // Founders seeded at tick 0
            for uid in 1..=4u64 {
                agents.push(make_agent(uid, tick, 0, None));
                births.push(make_birth(tick, uid, None, 0, BirthOrigin::Seeded));
            }
        } else {
            // Founders alive throughout
            for uid in 1..=4u64 {
                agents.push(make_agent(uid, tick, 0, None));
            }

            // Offspring births at ticks 10, 20, 30
            if tick == 10 {
                births.push(make_birth(tick, 5, Some(1), 1, BirthOrigin::Born));
                agents.push(make_agent(5, tick, 1, Some(1)));
            } else if tick > 10 {
                agents.push(make_agent(5, tick, 1, Some(1)));
            }

            if tick == 20 {
                births.push(make_birth(tick, 6, Some(2), 1, BirthOrigin::Born));
                agents.push(make_agent(6, tick, 1, Some(2)));
            } else if tick > 20 {
                agents.push(make_agent(6, tick, 1, Some(2)));
            }

            if tick == 30 {
                births.push(make_birth(tick, 7, Some(3), 1, BirthOrigin::Born));
                agents.push(make_agent(7, tick, 1, Some(3)));
            } else if tick > 30 {
                agents.push(make_agent(7, tick, 1, Some(3)));
            }
        }

        // Interactions recorded periodically
        if tick > 0 && tick % 10 == 0 {
            replay_events.push(ReplayEvent {
                agent_uid: Some(AgentUid(1)),
                position: None,
                counterpart: Some(AgentUid(2)),
                counterpart_position: None,
                kind: ReplayEventKind::Interaction {
                    tick: Tick(tick),
                    ordinal: 0,
                    kind: ReplayInteractionKind::Combat,
                    magnitude: 5.0,
                },
            });
            replay_events.push(ReplayEvent {
                agent_uid: Some(AgentUid(2)),
                position: None,
                counterpart: Some(AgentUid(3)),
                counterpart_position: None,
                kind: ReplayEventKind::Interaction {
                    tick: Tick(tick),
                    ordinal: 1,
                    kind: ReplayInteractionKind::FoodShare,
                    magnitude: 3.0,
                },
            });
        }

        let born_count = births
            .iter()
            .filter(|b| b.origin == BirthOrigin::Born)
            .count();

        let mut events = Vec::new();
        if born_count > 0 {
            events.push(PersistenceEvent::new(
                PersistenceEventKind::Births,
                born_count,
            ));
        }

        let interactions_count = replay_events
            .iter()
            .filter(|e| matches!(e.kind, ReplayEventKind::Interaction { .. }))
            .count();

        if interactions_count > 0 {
            events.push(PersistenceEvent::new(
                PersistenceEventKind::Custom(std::borrow::Cow::Borrowed(
                    INTERACTION_EVENTS_OBSERVED_KIND,
                )),
                interactions_count,
            ));
            events.push(PersistenceEvent::new(
                PersistenceEventKind::Custom(std::borrow::Cow::Borrowed(
                    INTERACTION_EVENTS_PERSISTED_KIND,
                )),
                interactions_count,
            ));
        }

        let mut batch = PersistenceBatch {
            summary: empty_summary(tick, agents.len(), born_count),
            epoch: 1,
            closed: false,
            metrics: vec![
                MetricSample::new("population", pop_val),
                MetricSample::new("energy", energy_val),
            ],
            events,
            agents,
            births,
            deaths: Vec::new(),
            replay_events,
            narrative_events: Vec::new(),
            genomes: Vec::new(),
        };

        if tick == 100 {
            // Planted regime change event
            batch.narrative_events.push(EventRecord {
                schema_version: 1,
                tick: Tick(100),
                kind: EventKind::PopulationCrash,
                severity: 0.95,
                magnitude: 30.0,
                window: (70, 130),
                metric: "population".to_owned(),
                before: 40.0,
                after: 10.0,
                score: 15.0,
                subject: None,
                human_text: "drought triggered catastrophic population crash".to_owned(),
            });

            // Planted null-control noise event
            batch.narrative_events.push(EventRecord {
                schema_version: 1,
                tick: Tick(100),
                kind: EventKind::EnergyCollapse,
                severity: 0.2,
                magnitude: 0.0,
                window: (70, 130),
                metric: "energy".to_owned(),
                before: 50.0,
                after: 50.0,
                score: 0.05,
                subject: None,
                human_text: "stationary energy fluctuation".to_owned(),
            });
        }

        storage.persist(&batch).expect("persist batch");
    }

    storage.flush().expect("flush run db");
    storage.close().expect("close run db");

    // Paired control fixture (for compare-runs report)
    let ctrl_path = dir.join("control.sqlite").display().to_string();
    let mut ctrl_storage =
        Storage::create_unattributed_file(&ctrl_path).expect("create control db");
    for tick in 0..=200u64 {
        let pop_val = 40.0 + wobble(tick);
        let mut births = Vec::new();
        if tick == 0 {
            for uid in 1..=4u64 {
                births.push(make_birth(tick, uid, None, 0, BirthOrigin::Seeded));
            }
        }
        let batch = PersistenceBatch {
            summary: empty_summary(tick, 4, 0),
            epoch: 1,
            closed: false,
            metrics: vec![
                MetricSample::new("population", pop_val),
                MetricSample::new("energy", 50.0 + noise(tick)),
            ],
            events: Vec::new(),
            agents: (1..=4u64)
                .map(|uid| make_agent(uid, tick, 0, None))
                .collect(),
            births,
            deaths: Vec::new(),
            replay_events: Vec::new(),
            narrative_events: Vec::new(),
            genomes: Vec::new(),
        };
        ctrl_storage.persist(&batch).expect("persist control batch");
    }
    ctrl_storage.flush().expect("flush control db");
    ctrl_storage.close().expect("close control db");

    (run_path, ctrl_path)
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "comprehensive 7-stage end-to-end scientific analytics pipeline test"
)]
fn test_analytics_e2e_full_pipeline_and_invariants() {
    let start_total = Instant::now();
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let (run_db, ctrl_db) = populate_seeded_fixture(temp_dir.path());

    let mut stages = Vec::new();
    let mut invariants = Vec::new();

    // ------------------------------------------------------------------------
    // STAGE 1: Verify Simulation & Storage Populated
    // ------------------------------------------------------------------------
    let t0 = Instant::now();
    let cx = ReaderCtx::open(&run_db).expect("open reader context");
    let total_metrics = cx.reader.recent_metrics(4096).expect("read metrics").len();
    let total_events = cx.reader.recent_run_events(100).expect("read events").len();
    assert!(total_metrics >= 400, "metrics table must have >= 400 rows");
    assert_eq!(total_events, 2, "must have exactly 2 narrative events");

    stages.push(StageRecord {
        stage: "simulation_and_persistence".to_string(),
        duration_ms: t0.elapsed().as_millis(),
        status: "pass".to_string(),
        details: serde_json::json!({
            "metrics_rows": total_metrics,
            "events_rows": total_events,
            "run_db": run_db,
        }),
    });

    // ------------------------------------------------------------------------
    // STAGE 2: Execute Complete Report Suite (All 12 Reports)
    // ------------------------------------------------------------------------
    let t1 = Instant::now();
    let registry = Registry::builtin();
    let report_names: Vec<&str> = registry.list().into_iter().map(|(n, _)| n).collect();

    assert_eq!(
        report_names.len(),
        12,
        "registry must have exactly 12 built-in reports: {report_names:?}"
    );

    let mut report_machines = std::collections::BTreeMap::new();
    for name in &report_names {
        let t_rep = Instant::now();
        let params = if *name == "compare-runs" {
            ReportParams::from_pairs([format!("treatment_db={ctrl_db}")]).expect("compare params")
        } else if *name == "narrative-validate" {
            ReportParams::from_pairs([
                "window=30".to_string(),
                "fdr=0.05".to_string(),
                "resamples=100".to_string(),
                "permutations=100".to_string(),
            ])
            .expect("params")
        } else {
            ReportParams::default()
        };

        let output = registry.run(name, &cx, &params).unwrap_or_else(|e| {
            panic!("report '{name}' failed execution: {e}");
        });
        println!(
            "Report '{name}' executed in {} ms",
            t_rep.elapsed().as_millis()
        );
        assert!(
            !output.human_md.is_empty(),
            "report '{name}' markdown must not be empty"
        );
        report_machines.insert(name.to_string(), output.machine);
    }

    stages.push(StageRecord {
        stage: "report_suite_execution".to_string(),
        duration_ms: t1.elapsed().as_millis(),
        status: "pass".to_string(),
        details: serde_json::json!({
            "reports_executed": report_names,
        }),
    });

    // ------------------------------------------------------------------------
    // STAGE 3: Cross-Report Invariant Assertions
    // ------------------------------------------------------------------------
    let t2 = Instant::now();

    // Invariant 1: run-summary accounts for all 201 ticks and 3 birth records
    let run_summary = &report_machines["run-summary"];
    let ticks_persisted = run_summary["tick_count"].as_u64().unwrap_or(0);
    let births_persisted = run_summary["birth_records"].as_u64().unwrap_or(0);
    assert_eq!(
        ticks_persisted, 201,
        "run-summary must account for all 201 ticks"
    );
    assert_eq!(
        births_persisted, 3,
        "run-summary must account for 3 demographic births"
    );
    invariants.push(InvariantRecord {
        invariant: "run_summary_accounting_verified".to_string(),
        expected: serde_json::json!({ "tick_count": 201, "birth_records": 3 }),
        observed: serde_json::json!({ "tick_count": ticks_persisted, "birth_records": births_persisted }),
        status: "pass".to_string(),
    });

    // Invariant 2: narrative-timeline carries interaction replay events
    let timeline = &report_machines["narrative-timeline"];
    let timeline_events = timeline["events"].as_array().expect("timeline events");
    assert!(
        !timeline_events.is_empty(),
        "narrative-timeline must contain replay events"
    );
    invariants.push(InvariantRecord {
        invariant: "narrative_timeline_replay_events_present".to_string(),
        expected: serde_json::json!({ "events_non_empty": true }),
        observed: serde_json::json!({ "events_count": timeline_events.len() }),
        status: "pass".to_string(),
    });

    // Invariant 3: narrative-validate certifies planted shift under FDR (p < 0.05, CI < 0)
    let validate = &report_machines["narrative-validate"];
    let validate_events = validate["events"].as_array().expect("validate events");
    let pop_validation = validate_events
        .iter()
        .find(|ev| ev["metric"] == "population")
        .expect("population event validated");

    assert_eq!(pop_validation["tick"].as_u64(), Some(100));
    assert_eq!(pop_validation["kind"].as_str(), Some("PopulationCrash"));
    assert!(
        pop_validation["human_text"]
            .as_str()
            .is_some_and(|t| t.contains("drought")),
        "human_text must contain 'drought'"
    );

    let pop_sig = pop_validation["significant_fdr"].as_bool() == Some(true);
    let pop_p = pop_validation["p_value"].as_f64().unwrap_or(1.0);
    let pop_ci_upper = pop_validation["ci_upper"].as_f64().unwrap_or(0.0);
    let pop_valid = pop_sig && pop_p < 0.05 && pop_ci_upper < 0.0;

    invariants.push(InvariantRecord {
        invariant: "planted_shift_significant_fdr".to_string(),
        expected: serde_json::json!({ "significant_fdr": true, "p_value_max": 0.05, "ci_upper_max": 0.0, "tick": 100 }),
        observed: serde_json::json!({ "significant_fdr": pop_sig, "p_value": pop_p, "ci_upper": pop_ci_upper, "tick": pop_validation["tick"] }),
        status: if pop_valid { "pass" } else { "fail" }.to_string(),
    });
    assert!(
        pop_valid,
        "planted population shift must be certified significant with negative CI"
    );

    // Invariant 3: narrative-validate rejects null noise (false-positive control)
    let energy_validation = validate_events
        .iter()
        .find(|ev| ev["metric"] == "energy")
        .expect("energy event validated");

    let energy_sig = energy_validation["significant_fdr"].as_bool() == Some(false);
    invariants.push(InvariantRecord {
        invariant: "null_noise_not_significant".to_string(),
        expected: serde_json::json!({ "significant_fdr": false }),
        observed: serde_json::json!({ "significant_fdr": !energy_sig }),
        status: if energy_sig { "pass" } else { "fail" }.to_string(),
    });
    assert!(
        energy_sig,
        "stationary null energy fluctuation must not be certified significant"
    );

    // Invariant 4 & 5: lineage structure founder components and descendant conservation
    let lineage = &report_machines["lineage-structure"];
    let founder_components = lineage["weakly_connected_components_count"]
        .as_u64()
        .unwrap_or(0);
    let edge_count = lineage["edge_count"].as_u64().unwrap_or(0);
    let node_count = lineage["node_count"].as_u64().unwrap_or(0);

    invariants.push(InvariantRecord {
        invariant: "lineage_founder_components_conserved".to_string(),
        expected: serde_json::json!(4),
        observed: serde_json::json!(founder_components),
        status: if founder_components == 4 {
            "pass"
        } else {
            "fail"
        }
        .to_string(),
    });
    assert_eq!(
        founder_components, 4,
        "founder component count must strictly match initial 4 cohorts"
    );

    invariants.push(InvariantRecord {
        invariant: "lineage_descendants_match_births".to_string(),
        expected: serde_json::json!(3),
        observed: serde_json::json!(edge_count),
        status: if edge_count == 3 { "pass" } else { "fail" }.to_string(),
    });
    assert_eq!(
        edge_count, 3,
        "descendant counts must match the 3 recorded births"
    );
    assert_eq!(
        node_count, 7,
        "total lineage nodes must match 4 founders + 3 births"
    );

    stages.push(StageRecord {
        stage: "ground_truth_invariants".to_string(),
        duration_ms: t2.elapsed().as_millis(),
        status: "pass".to_string(),
        details: serde_json::json!({ "invariants_count": invariants.len() }),
    });

    // ------------------------------------------------------------------------
    // STAGE 4: FrankenPandas Export and SQL Count Verification
    // ------------------------------------------------------------------------
    let t3 = Instant::now();
    let export_dir = temp_dir.path().join("exports");
    fs::create_dir_all(&export_dir).expect("create export dir");

    let tables = [
        ExportTable::Run,
        ExportTable::Agent,
        ExportTable::Lineage,
        ExportTable::Event,
        ExportTable::Metric,
    ];

    let run_id = cx.reader.run_id().to_string();
    for table in tables {
        let path = export_database_table(
            &cx.reader,
            table,
            AnalyticsExportFormat::Parquet,
            &export_dir,
            true, // with round-trip verification
        )
        .expect("export table to parquet");
        assert!(path.exists(), "exported parquet file must exist");
    }
    drop(cx);

    // CLI invocation: sb-analyze export --format parquet --verify
    let cli_export_dir = temp_dir.path().join("cli_exports");
    let sb_analyze_bin = env!("CARGO_BIN_EXE_sb-analyze");
    let export_status = Command::new(sb_analyze_bin)
        .arg(&run_db)
        .arg("export")
        .arg("--format")
        .arg("parquet")
        .arg("--verify")
        .arg("--out-dir")
        .arg(&cli_export_dir)
        .status()
        .expect("run sb-analyze export");
    assert!(
        export_status.success(),
        "sb-analyze export CLI command failed"
    );

    // SQL row counts match DataFrame export row counts
    let conn =
        open_with_flags(&run_db, OpenFlags::SQLITE_OPEN_READ_ONLY).expect("open raw db connection");
    let sql_run_count: i64 = conn
        .query_row("SELECT count(*) FROM runs")
        .expect("query runs")
        .get_typed(0)
        .expect("get count");
    let sql_metric_count: i64 = conn
        .query_row("SELECT count(*) FROM metrics")
        .expect("query metrics")
        .get_typed(0)
        .expect("get count");
    let sql_agent_count: i64 = conn
        .query_row("SELECT count(*) FROM agents")
        .expect("query agents")
        .get_typed(0)
        .expect("get count");
    let sql_lineage_count: i64 = conn
        .query_row("SELECT count(*) FROM lineage_edges")
        .expect("query lineage_edges")
        .get_typed(0)
        .expect("get count");
    let sql_event_count: i64 = conn
        .query_row("SELECT count(*) FROM replay_events")
        .expect("query replay_events")
        .get_typed(0)
        .expect("get count");
    let _ = conn.close_without_checkpoint();

    let pq_run_rows = read_parquet_batch(&export_dir.join(format!("{run_id}_run.parquet")))
        .expect("read run pq")
        .num_rows();
    let pq_metric_rows = read_parquet_batch(&export_dir.join(format!("{run_id}_metric.parquet")))
        .expect("read metric pq")
        .num_rows();
    let pq_agent_rows = read_parquet_batch(&export_dir.join(format!("{run_id}_agent.parquet")))
        .expect("read agent pq")
        .num_rows();
    let pq_lineage_rows = read_parquet_batch(&export_dir.join(format!("{run_id}_lineage.parquet")))
        .expect("read lineage pq")
        .num_rows();
    let pq_event_rows = read_parquet_batch(&export_dir.join(format!("{run_id}_event.parquet")))
        .expect("read event pq")
        .num_rows();

    assert_eq!(
        pq_run_rows,
        usize::try_from(sql_run_count).unwrap(),
        "run parquet rows == sql"
    );
    assert_eq!(
        pq_metric_rows,
        usize::try_from(sql_metric_count).unwrap(),
        "metric parquet rows == sql"
    );
    assert_eq!(
        pq_agent_rows,
        usize::try_from(sql_agent_count).unwrap(),
        "agent parquet rows == sql"
    );
    assert_eq!(
        pq_lineage_rows,
        usize::try_from(sql_lineage_count).unwrap(),
        "lineage parquet rows == sql"
    );
    assert_eq!(
        pq_event_rows,
        usize::try_from(sql_event_count).unwrap(),
        "event parquet rows == sql"
    );

    invariants.push(InvariantRecord {
        invariant: "parquet_sql_row_count_equality".to_string(),
        expected: serde_json::json!({
            "sql_runs": sql_run_count,
            "sql_metrics": sql_metric_count,
            "sql_agents": sql_agent_count,
            "sql_lineage": sql_lineage_count,
            "sql_events": sql_event_count,
        }),
        observed: serde_json::json!({
            "pq_runs": pq_run_rows,
            "pq_metrics": pq_metric_rows,
            "pq_agents": pq_agent_rows,
            "pq_lineage": pq_lineage_rows,
            "pq_events": pq_event_rows,
        }),
        status: "pass".to_string(),
    });

    // Also run sb-analyze summarize
    let summary_dir = temp_dir.path().join("summaries");
    let summarize_status = Command::new(sb_analyze_bin)
        .arg(&run_db)
        .arg("summarize")
        .arg("--epoch-size")
        .arg("50")
        .arg("--rolling-window")
        .arg("10")
        .arg("--out-dir")
        .arg(&summary_dir)
        .status()
        .expect("run sb-analyze summarize");
    assert!(
        summarize_status.success(),
        "sb-analyze summarize CLI command failed"
    );
    assert!(
        summary_dir.join("summary.json").exists(),
        "summary.json must exist"
    );
    assert!(
        summary_dir.join("summary.md").exists(),
        "summary.md must exist"
    );

    stages.push(StageRecord {
        stage: "frankenpandas_export_and_summary".to_string(),
        duration_ms: t3.elapsed().as_millis(),
        status: "pass".to_string(),
        details: serde_json::json!({
            "parquet_tables_exported": tables.len(),
            "sql_metrics_count": sql_metric_count,
            "sql_runs_count": sql_run_count,
        }),
    });

    // ------------------------------------------------------------------------
    // STAGE 5: Graph Export Verification (Lineage, Dynasty, Interaction)
    // ------------------------------------------------------------------------
    let t4 = Instant::now();
    let graph_dir = temp_dir.path().join("graphs");
    fs::create_dir_all(&graph_dir).expect("create graph dir");

    let lineage_path = graph_dir.join("lineage.edgelist");
    let dynasty_path = graph_dir.join("dynasty.graphml");
    let interaction_path = graph_dir.join("interaction.edgelist");

    let g1 = Command::new(sb_analyze_bin)
        .arg(&run_db)
        .arg("export-graph")
        .arg("--graph")
        .arg("lineage")
        .arg("--format")
        .arg("edgelist")
        .arg("--out")
        .arg(&lineage_path)
        .status()
        .expect("export lineage graph");
    assert!(g1.success(), "export lineage graph failed");
    assert!(lineage_path.exists() && fs::metadata(&lineage_path).unwrap().len() > 0);

    let g2 = Command::new(sb_analyze_bin)
        .arg(&run_db)
        .arg("export-graph")
        .arg("--graph")
        .arg("dynasty")
        .arg("--format")
        .arg("graphml")
        .arg("--out")
        .arg(&dynasty_path)
        .status()
        .expect("export dynasty graph");
    assert!(g2.success(), "export dynasty graph failed");
    assert!(dynasty_path.exists() && fs::metadata(&dynasty_path).unwrap().len() > 0);

    let g3 = Command::new(sb_analyze_bin)
        .arg(&run_db)
        .arg("export-graph")
        .arg("--graph")
        .arg("interaction")
        .arg("--format")
        .arg("edgelist")
        .arg("--out")
        .arg(&interaction_path)
        .status()
        .expect("export interaction graph");
    assert!(g3.success(), "export interaction graph failed");
    assert!(interaction_path.exists() && fs::metadata(&interaction_path).unwrap().len() > 0);

    stages.push(StageRecord {
        stage: "graph_exports".to_string(),
        duration_ms: t4.elapsed().as_millis(),
        status: "pass".to_string(),
        details: serde_json::json!({
            "lineage_edgelist_bytes": fs::metadata(&lineage_path).unwrap().len(),
            "dynasty_graphml_bytes": fs::metadata(&dynasty_path).unwrap().len(),
            "interaction_edgelist_bytes": fs::metadata(&interaction_path).unwrap().len(),
        }),
    });

    // ------------------------------------------------------------------------
    // STAGE 6: FTS5 Narrative Search Verification
    // ------------------------------------------------------------------------
    let t5 = Instant::now();
    let cx = ReaderCtx::open(&run_db).expect("reopen reader context");
    let hits = cx
        .reader
        .search_narrative("drought", None, 10)
        .expect("search narrative for 'drought'");
    assert_eq!(hits.len(), 1, "drought search must find exactly 1 hit");
    assert_eq!(hits[0].tick(), Tick(100));
    assert!(hits[0].human_text().contains("drought"));

    invariants.push(InvariantRecord {
        invariant: "narrative_search_hit_confirmed".to_string(),
        expected: serde_json::json!({ "tick": 100, "query": "drought" }),
        observed: serde_json::json!({ "tick": hits[0].tick().0, "text": hits[0].human_text() }),
        status: "pass".to_string(),
    });

    stages.push(StageRecord {
        stage: "narrative_search_fts".to_string(),
        duration_ms: t5.elapsed().as_millis(),
        status: "pass".to_string(),
        details: serde_json::json!({ "hits_count": hits.len() }),
    });

    // ------------------------------------------------------------------------
    // STAGE 7: Emit Structured MANIFEST.json
    // ------------------------------------------------------------------------
    let total_duration = start_total.elapsed().as_millis();
    let manifest = E2eManifest {
        schema: "scriptbots.analytics.e2e-manifest.v1".to_string(),
        run_id: cx.reader.run_id().to_string(),
        verdict: "pass".to_string(),
        total_duration_ms: total_duration,
        stages,
        invariants,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest).expect("serialize manifest");
    let manifest_path = temp_dir.path().join("MANIFEST.json");
    fs::write(&manifest_path, &manifest_json).expect("write manifest");

    if let Some(target_manifest) = std::env::var_os("SCRIPTBOTS_E2E_ANALYTICS_MANIFEST") {
        fs::write(target_manifest, &manifest_json).expect("write manifest to target path");
    }

    // Machine-readable proof line for script/CI ingestion:
    println!(
        "E2E_ANALYTICS_MANIFEST: {}",
        serde_json::to_string(&manifest).expect("compact manifest")
    );
    println!(
        "MANIFEST successfully written to {}",
        manifest_path.display()
    );
}

/// Run a real seeded simulation into a run database, returning the database path and the
/// simulation's own ground truth: (ticks, born births, deaths, per-tick populations).
fn run_real_simulation(dir: &Path, name: &str, seed: u64) -> (String, u64, u64, u64, Vec<usize>) {
    use scriptbots_brain::mlp::{MlpBrain, MlpBrainFamily};
    use scriptbots_core::{ScriptBotsConfig, WorldState};
    use scriptbots_storage::StoragePipeline;

    const TICKS: u64 = 400;
    let run_path = dir.join(name).display().to_string();
    let mut pipeline = StoragePipeline::create_unattributed_file(&run_path).expect("create run db");
    let config = ScriptBotsConfig {
        rng_seed: Some(seed),
        persistence_interval: 1,
        reproduction_cooldown: 60,
        reproduction_attempt_chance: 0.2,
        ..ScriptBotsConfig::default()
    };
    let (width, height) = (config.world_width, config.world_height);
    let (mut world, mut session) =
        WorldState::with_persistence(config, Box::new(pipeline.sink())).expect("world");
    let brain = world
        .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
        .expect("register mlp");
    for index in 0..40_u16 {
        let mut agent = AgentData::default();
        agent.position = Position::new(
            f32::from(index % 8).mul_add(f32::from(u16::try_from(width / 8).expect("fit")), 17.0),
            f32::from(index / 8).mul_add(f32::from(u16::try_from(height / 5).expect("fit")), 23.0),
        );
        let id = world.try_spawn_agent(agent).expect("spawn founder");
        assert!(world.bind_agent_brain(id, brain).expect("bind brain"));
    }

    let (mut born, mut deaths) = (0_u64, 0_u64);
    let mut populations = Vec::new();
    for _ in 0..TICKS {
        let completion = session.step_outcome(&mut world).expect("step");
        let outcome = &completion.outcome;
        born += outcome
            .births
            .iter()
            .filter(|birth| birth.origin == BirthOrigin::Born)
            .count() as u64;
        deaths += outcome.deaths.len() as u64;
        populations.push(outcome.summary.agent_count);
        session.admit_pending(&mut world).expect("admit tick");
    }
    session.finalize(&mut world).expect("finalize persistence");
    drop(world);
    pipeline.shutdown().expect("storage shutdown");
    (run_path, TICKS, born, deaths, populations)
}

/// bd-2z0.11.9: the report suite on the output of an actual `WorldState` run (the test above
/// uses a hand-built fixture), with invariants checked against the simulation's own record.
#[test]
fn report_suite_on_a_real_seeded_simulation_matches_the_simulation_ground_truth() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let (run_db, ticks, born, deaths, populations) =
        run_real_simulation(temp_dir.path(), "real.sqlite", 7);
    let (treatment_db, ..) = run_real_simulation(temp_dir.path(), "real_treatment.sqlite", 8);
    assert!(
        born > 0,
        "the seeded run must produce natural births to be non-vacuous"
    );
    assert!(
        deaths > 0,
        "the seeded run must produce deaths to be non-vacuous"
    );

    let cx = ReaderCtx::open(&run_db).expect("open real run");
    let registry = Registry::builtin();
    let mut summary = None;
    for (name, _) in registry.list() {
        let params = match name {
            "compare-runs" => ReportParams::from_pairs([format!("treatment_db={treatment_db}")])
                .expect("compare params"),
            "narrative-validate" => ReportParams::from_pairs([
                "window=30".to_string(),
                "resamples=50".to_string(),
                "permutations=50".to_string(),
            ])
            .expect("narrative params"),
            _ => ReportParams::default(),
        };
        let output = registry
            .run(name, &cx, &params)
            .unwrap_or_else(|error| panic!("report '{name}' failed on a real run: {error}"));
        assert_eq!(
            output.latest_tick,
            Some(ticks),
            "report '{name}' latest tick"
        );
        if name == "run-summary" {
            summary = Some(output.machine);
        }
    }

    let summary = summary.expect("run-summary is a built-in report");
    let as_u64 = |key: &str| {
        summary[key]
            .as_u64()
            .unwrap_or_else(|| panic!("{key}: {summary}"))
    };
    assert_eq!(as_u64("tick_count"), ticks);
    assert_eq!(as_u64("birth_records"), born, "born births: {summary}");
    assert_eq!(as_u64("death_records"), deaths, "deaths: {summary}");
    assert_eq!(as_u64("population_first"), populations[0] as u64);
    assert_eq!(
        as_u64("population_last"),
        *populations.last().expect("ticks ran") as u64
    );
    assert_eq!(
        as_u64("population_min"),
        *populations.iter().min().expect("ticks ran") as u64
    );
    assert_eq!(
        as_u64("population_max"),
        *populations.iter().max().expect("ticks ran") as u64
    );
    #[allow(clippy::cast_precision_loss)]
    let mean = populations.iter().sum::<usize>() as f64 / populations.len() as f64;
    let reported_mean = summary["population_mean"].as_f64().expect("mean");
    assert!(
        (reported_mean - mean).abs() < 1e-9,
        "population mean {reported_mean} vs simulation {mean}"
    );
}
