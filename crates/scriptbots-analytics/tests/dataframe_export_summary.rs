//! Comprehensive integration test suite for `FrankenPandas` export and summary layer (bd-2z0.11.8).
//!
//! Verifies:
//! 1. `export_database_table` exports Run, Agent, Lineage, Event, and Metric tables
//!    to Parquet, Arrow, and CSV with verified bd-2z0.5.6 provenance headers.
//! 2. Round-trip verification mode (`verify = true`) asserts exact row count, schema,
//!    and bit-exact values upon re-reading.
//! 3. `summarize_run` produces per-epoch groupby aggregates (diet, brain kind), rolling-smoothed
//!    metrics, and lineage tree summaries with Markdown tables.
//! 4. SQL cross-check: independently computes population and metric averages via raw SQL
//!    and asserts exact agreement with DataFrame-derived summaries.
//! 5. CLI invocation: tests `sb-analyze export` and `sb-analyze summarize` subcommands end-to-end.

use std::fs;
use std::path::Path;
use std::process::Command;

use arrow::array::RecordBatch;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use scriptbots_analytics::{
    AnalyticsExportFormat, agent_rows_to_batch, event_rows_to_batch, export_database_table,
    lineage_rows_to_batch, metric_rows_to_batch, summarize_run, verify_record_batches,
};
use scriptbots_core::{
    AgentData, AgentIdentity, AgentRuntime, AgentState, AgentUid, BirthOrigin, BirthRecord,
    BrainBinding, Generation, MetricSample, PersistenceBatch, PersistenceEvent,
    PersistenceEventKind, Position, ReplayEvent, ReplayEventKind, ReplayRngScope, Tick,
    TickSummary, Velocity,
};
use scriptbots_storage::export_pipeline::{
    EXPORT_PAGE_ROWS, ExportTablePage, export_provenance, verify_storage_table_export,
    visit_export_pages,
};
use scriptbots_storage::{
    Connection, ExportTable, Storage, StorageError, StorageReader, verify_export_receipt,
};

fn make_agent(
    uid: u64,
    tick: u64,
    agent_gen: u32,
    parent: Option<u64>,
    ht: f32,
    brain_kind: &str,
) -> AgentState {
    let data = AgentData {
        position: Position::new(
            10.0 * f32::from(u16::try_from(uid).expect("fixture UID in 1..=4 fits u16")),
            20.0 * f32::from(u16::try_from(tick).expect("fixture tick in 0..=9 fits u16")),
        ),
        velocity: Velocity::new(1.0, 0.0),
        heading: 0.0,
        health: 100.0,
        generation: Generation(agent_gen),
        age: u32::try_from(tick).expect("fixture tick in 0..=9 fits u32"),
        ..AgentData::default()
    };
    #[expect(
        clippy::suboptimal_flops,
        reason = "preserve the separate multiplication and addition used by this serialized agent fixture"
    )]
    let runtime = AgentRuntime {
        energy: 100.0
            + (f32::from(u16::try_from(uid).expect("fixture UID in 1..=4 fits u16")) * 10.0),
        herbivore_tendency: ht,
        brain: BrainBinding::Legacy {
            runner: None,
            registry_key: None,
            kind: brain_kind.to_string(),
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
    birth_gen: u32,
    origin: BirthOrigin,
    ht: f32,
    brain_kind: &str,
) -> BirthRecord {
    BirthRecord {
        tick: Tick(tick),
        agent_uid: AgentUid(uid),
        spawn_ordinal: uid.saturating_sub(1),
        birth_ordinal: if parent.is_some() { Some(uid) } else { None },
        origin,
        parent_a: parent.map(AgentUid),
        parent_b: None,
        brain_kind: Some(brain_kind.to_string()),
        brain_key: None,
        herbivore_tendency: ht,
        generation: Generation(birth_gen),
        position: Position::new(0.0, 0.0),
        is_hybrid: false,
    }
}

#[expect(
    clippy::suboptimal_flops,
    reason = "preserve the separate multiplication and addition in the fixture's exported average_energy metric"
)]
fn fixture_batch(
    tick: u64,
    agent_count: usize,
    births_count: usize,
    agents: Vec<AgentState>,
    births: Vec<BirthRecord>,
) -> PersistenceBatch {
    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(tick),
            agent_count,
            births: births_count,
            deaths: 0,
            total_energy: 100.0
                * f32::from(
                    u16::try_from(agent_count).expect("fixture population in 2..=4 fits u16"),
                ),
            average_energy: 100.0,
            average_health: 95.0,
            max_age: u32::try_from(tick).expect("fixture tick in 0..=9 fits u32"),
            spike_hits: 0,
        },
        epoch: 1,
        closed: false,
        metrics: vec![
            MetricSample::new(
                "population",
                f64::from(
                    u16::try_from(agent_count).expect("fixture population in 2..=4 fits u16"),
                ),
            ),
            MetricSample::new(
                "average_energy",
                100.0
                    + (f64::from(u16::try_from(tick).expect("fixture tick in 0..=9 fits u16"))
                        * 2.5),
            ),
        ],
        events: if births_count > 0 {
            vec![PersistenceEvent::new(
                PersistenceEventKind::Births,
                births_count,
            )]
        } else {
            Vec::new()
        },
        agents,
        births,
        deaths: Vec::new(),
        replay_events: vec![ReplayEvent {
            agent_uid: Some(AgentUid(1)),
            position: Some(Position::new(10.0, 20.0)),
            counterpart: None,
            counterpart_position: None,
            kind: ReplayEventKind::RngSample {
                scope: ReplayRngScope::World,
                range_min: 0.0,
                range_max: 1.0,
                value: 0.42,
            },
        }],
        narrative_events: Vec::new(),
        genomes: Vec::new(),
    }
}

fn build_fixture_db(dir: &tempfile::TempDir) -> String {
    let db_path = dir.path().join("fixture_run.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&db_path).expect("create storage file");

    for tick in 0..=9 {
        let (agent_count, births_count, births, agents) = match tick {
            0 => (
                2,
                0,
                vec![
                    make_birth(0, 1, None, 0, BirthOrigin::Seeded, 0.9, "mlp"),
                    make_birth(0, 2, None, 0, BirthOrigin::Seeded, 0.1, "neuroflow"),
                ],
                vec![
                    make_agent(1, tick, 0, None, 0.9, "mlp"),
                    make_agent(2, tick, 0, None, 0.1, "neuroflow"),
                ],
            ),
            1 => (
                3,
                1,
                vec![make_birth(1, 3, Some(1), 1, BirthOrigin::Born, 0.5, "mlp")],
                vec![
                    make_agent(1, tick, 0, None, 0.9, "mlp"),
                    make_agent(2, tick, 0, None, 0.1, "neuroflow"),
                    make_agent(3, tick, 1, Some(1), 0.5, "mlp"),
                ],
            ),
            2 => (
                4,
                1,
                vec![make_birth(
                    2,
                    4,
                    Some(3),
                    2,
                    BirthOrigin::Born,
                    0.85,
                    "neuroflow",
                )],
                vec![
                    make_agent(1, tick, 0, None, 0.9, "mlp"),
                    make_agent(2, tick, 0, None, 0.1, "neuroflow"),
                    make_agent(3, tick, 1, Some(1), 0.5, "mlp"),
                    make_agent(4, tick, 2, Some(3), 0.85, "neuroflow"),
                ],
            ),
            _ => (
                4,
                0,
                Vec::new(),
                vec![
                    make_agent(1, tick, 0, None, 0.9, "mlp"),
                    make_agent(2, tick, 0, None, 0.1, "neuroflow"),
                    make_agent(3, tick, 1, Some(1), 0.5, "mlp"),
                    make_agent(4, tick, 2, Some(3), 0.85, "neuroflow"),
                ],
            ),
        };

        let batch = fixture_batch(tick, agent_count, births_count, agents, births);

        storage.persist(&batch).expect("persist batch");
    }

    storage.flush().expect("flush storage");
    storage.close().expect("close storage");
    db_path
}

/// Coherent synthetic lifecycle records cross every multi-row export cursor.
/// The lineage boundary falls between two parent edges for the same child.
fn build_multi_page_fixture_db(dir: &tempfile::TempDir) -> String {
    let db_path = dir.path().join("paged_run.sqlite").display().to_string();
    let mut storage = Storage::create_unattributed_file(&db_path).expect("create paged fixture");
    let population = EXPORT_PAGE_ROWS / 2 + 3;
    for tick in 0..=2 {
        let count = if tick == 0 { 2 } else { population };
        let agents: Vec<_> = (1..=count)
            .map(|index| {
                let uid = u64::try_from(index).expect("bounded fixture UID");
                let parent = (uid > 2).then_some(1);
                let mut agent =
                    make_agent(uid, tick, u32::from(parent.is_some()), parent, 0.5, "mlp");
                let birth_tick = u64::from(parent.is_some());
                agent.data.age = u32::try_from(tick - birth_tick).expect("bounded agent age");
                if uid > 3 {
                    agent.runtime.lineage[1] = Some(AgentUid(2));
                    agent.runtime.hybrid = true;
                }
                agent
            })
            .collect();
        let mut births = match tick {
            0 => vec![
                make_birth(0, 1, None, 0, BirthOrigin::Seeded, 0.5, "mlp"),
                make_birth(0, 2, None, 0, BirthOrigin::Seeded, 0.5, "mlp"),
            ],
            1 => (3..=count)
                .map(|index| {
                    let uid = u64::try_from(index).expect("bounded fixture UID");
                    let mut birth =
                        make_birth(1, uid, Some(1), 1, BirthOrigin::Born, 0.5, "mlp");
                    if uid > 3 {
                        birth.parent_b = Some(AgentUid(2));
                        birth.is_hybrid = true;
                    }
                    birth
                })
                .collect(),
            _ => Vec::new(),
        };
        for birth in &mut births {
            let index = usize::try_from(birth.agent_uid.0 - 1).expect("bounded birth UID");
            birth.position = agents[index].data.position;
        }
        let births_count = if tick == 1 { count - 2 } else { 0 };
        let mut batch = fixture_batch(tick, count, births_count, agents, births);
        batch.summary.total_energy = batch
            .agents
            .iter()
            .map(|agent| agent.runtime.energy)
            .sum();
        batch.summary.average_energy = batch.summary.total_energy
            / f32::from(u16::try_from(count).expect("bounded fixture population"));
        batch.summary.average_health = 100.0;
        batch.metrics.clear();
        batch.replay_events.clear();
        if tick != 0 {
            for index in 0..population {
                let name = if index == 0 {
                    "metric,\"quoted\"\nsecond line".to_owned()
                } else {
                    format!("metric-{index:05}")
                };
                let value = f64::from(u16::try_from(index).expect("bounded metric value"));
                batch.metrics.push(MetricSample::new(name, value + 0.25));
                batch.replay_events.push(ReplayEvent {
                    agent_uid: None,
                    position: None,
                    counterpart: None,
                    counterpart_position: None,
                    kind: ReplayEventKind::BrainOutputs {
                        outputs: vec![f32::from(u16::try_from(index).expect("bounded output"))],
                    },
                });
            }
        }
        storage.persist(&batch).expect("persist coherent page fixture");
    }
    storage.close().expect("durable paged fixture close");
    db_path
}

fn read_columnar_pages(path: &Path, format: AnalyticsExportFormat) -> Vec<RecordBatch> {
    let batches: Vec<_> = match format {
        AnalyticsExportFormat::Parquet => {
            let builder =
                ParquetRecordBatchReaderBuilder::try_new(fs::File::open(path).unwrap())
                    .expect("Parquet footer");
            let row_groups = builder.metadata().row_groups();
            assert!(row_groups.len() >= 2, "writer must seal multiple row groups");
            for group in row_groups {
                assert!(
                    usize::try_from(group.num_rows()).unwrap() <= EXPORT_PAGE_ROWS,
                    "Parquet writer must not retain a whole-table row group"
                );
            }
            builder
                .with_batch_size(EXPORT_PAGE_ROWS)
                .build()
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap()
        }
        AnalyticsExportFormat::Arrow => {
            ArrowFileReader::try_new(fs::File::open(path).unwrap(), None)
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap()
        }
        AnalyticsExportFormat::Csv => panic!("CSV has no columnar batches"),
    };
    assert!(
        batches.len() >= 2,
        "complete table must span multiple batches"
    );
    assert!(batches.iter().all(|batch| batch.num_rows() <= EXPORT_PAGE_ROWS));
    batches
}

#[test]
fn multi_page_exports_preserve_complete_values_order_and_legacy_lineage() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = build_multi_page_fixture_db(&dir);
    let reader = StorageReader::open_finished(&db_path).expect("finished source lease");
    // These all-row OFFSET loaders are independent of the new keyset traversal.
    let agents = reader.load_agent_export_rows().expect("agent SQL oracle");
    let lineage = reader.load_lineage_export_rows().expect("lineage SQL oracle");
    let events = reader.load_event_export_rows().expect("event SQL oracle");
    let metrics = reader.load_metric_export_rows().expect("metric SQL oracle");
    assert_eq!(agents.len(), EXPORT_PAGE_ROWS + 8);
    assert_eq!(lineage.len(), EXPORT_PAGE_ROWS + 1);
    assert_eq!(events.len(), EXPORT_PAGE_ROWS + 6);
    assert_eq!(metrics.len(), EXPORT_PAGE_ROWS + 6);
    assert_eq!(
        lineage[EXPORT_PAGE_ROWS - 1].child_agent_uid,
        lineage[EXPORT_PAGE_ROWS].child_agent_uid
    );
    assert_eq!(lineage[EXPORT_PAGE_ROWS - 1].parent_ordinal, 0);
    assert_eq!(lineage[EXPORT_PAGE_ROWS].parent_ordinal, 1);

    for table in [
        ExportTable::Agent,
        ExportTable::Lineage,
        ExportTable::Event,
        ExportTable::Metric,
    ] {
        for format in [AnalyticsExportFormat::Parquet, AnalyticsExportFormat::Arrow] {
            let path = export_database_table(&reader, table, format, dir.path(), true)
                .expect("bounded export and complete source verification");
            let pages = read_columnar_pages(&path, format);
            let schema = pages[0].schema();
            let mut prov = export_provenance(&reader, table).expect("source-bound provenance");
            // The artifact owns its export time; every identity field comes from the source.
            prov.exported_at_utc = schema.metadata()["provenance.exported_at_utc"].clone();
            let expected = match table {
                ExportTable::Agent => agent_rows_to_batch(&prov, &agents),
                ExportTable::Lineage => lineage_rows_to_batch(&prov, &lineage),
                ExportTable::Event => event_rows_to_batch(&prov, &events),
                ExportTable::Metric => metric_rows_to_batch(&prov, &metrics),
                ExportTable::Run => unreachable!("single-row manifest is covered separately"),
            }
            .unwrap();
            // The deliberately small test oracle can concatenate; production never does.
            let actual = arrow::compute::concat_batches(&schema, &pages).unwrap();
            verify_record_batches(&expected, &actual)
                .expect("all source values and nulls preserved");
        }
        let csv =
            export_database_table(&reader, table, AnalyticsExportFormat::Csv, dir.path(), true)
                .expect("CSV verifies quoted multiline fields without splitting lines");
        let content = fs::read_to_string(&csv).expect("small CSV test artifact");
        let receipt = verify_export_receipt(&content, table).expect("complete CSV receipt");
        let expected_rows = match table {
            ExportTable::Agent => agents.len(),
            ExportTable::Lineage => lineage.len(),
            ExportTable::Event => events.len(),
            ExportTable::Metric => metrics.len(),
            ExportTable::Run => unreachable!(),
        };
        assert_eq!(receipt.row_count, u64::try_from(expected_rows).unwrap());
        // CSV embeds stable provenance fields but no export timestamp. The original
        // completion time is retained in the parsed receipt for exact read-back.
        let prov = export_provenance(&reader, table).unwrap();
        verify_storage_table_export(&reader, &prov, &receipt, fs::File::open(csv).unwrap())
            .expect("CSV rows remain source-bound");
    }

    reader
        .close()
        .expect("release source before constructing legacy fixture");
    let connection = Connection::open_existing(&db_path).expect("legacy fixture writer");
    connection
        .execute("DELETE FROM lineage_edges")
        .expect("fixture retains parent births only");
    connection.close().expect("close legacy fixture writer");
    let legacy = StorageReader::open_finished(&db_path).expect("legacy source lease");
    let mut observed = Vec::new();
    let mut page_sizes = Vec::new();
    visit_export_pages::<StorageError>(&legacy, ExportTable::Lineage, |page| {
        page_sizes.push(page.len());
        if let ExportTablePage::Lineage(rows) = page {
            observed.extend(rows);
        } else {
            panic!("lineage table must contain lineage rows");
        }
        Ok(())
    })
    .expect("complete bounded birth-parent fallback");
    assert_eq!(page_sizes, vec![EXPORT_PAGE_ROWS - 1, 2]);
    assert_eq!(
        observed, lineage,
        "fallback preserves both parent ordinals and order"
    );
    for format in [
        AnalyticsExportFormat::Parquet,
        AnalyticsExportFormat::Arrow,
        AnalyticsExportFormat::Csv,
    ] {
        export_database_table(&legacy, ExportTable::Lineage, format, dir.path(), true)
            .expect("legacy parent rows use the same bounded verified pipeline");
    }
}

#[test]
fn empty_tables_export_their_complete_schema_in_every_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("empty.sqlite").display().to_string();
    Storage::create_unattributed_file(&db_path)
        .expect("create empty source")
        .close()
        .expect("finish empty source");
    let reader = StorageReader::open_finished(&db_path).expect("finished source lease");
    for table in [
        ExportTable::Agent,
        ExportTable::Lineage,
        ExportTable::Event,
        ExportTable::Metric,
    ] {
        let expected: Vec<_> = table
            .arrow_schema()
            .iter()
            .map(|column| column.name)
            .collect();
        for format in [AnalyticsExportFormat::Parquet, AnalyticsExportFormat::Arrow] {
            let path = export_database_table(&reader, table, format, dir.path(), true)
                .expect("empty columnar export verifies its schema");
            let batch = match format {
                AnalyticsExportFormat::Parquet => {
                    scriptbots_analytics::read_parquet_batch(&path)
                }
                AnalyticsExportFormat::Arrow => scriptbots_analytics::read_arrow_batch(&path),
                AnalyticsExportFormat::Csv => unreachable!(),
            }
            .unwrap();
            assert_eq!(batch.num_rows(), 0);
            let schema = batch.schema();
            let observed: Vec<_> = schema
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect();
            assert_eq!(observed, expected);
            assert_eq!(
                schema.metadata()["provenance.run_id"],
                reader.run_id().to_string()
            );
        }
        let csv =
            export_database_table(&reader, table, AnalyticsExportFormat::Csv, dir.path(), true)
                .expect("empty CSV export verifies its canonical header");
        let content = fs::read_to_string(csv).unwrap();
        assert_eq!(verify_export_receipt(&content, table).unwrap().row_count, 0);
        let headers: Vec<_> = content.lines().nth(1).unwrap().split(',').collect();
        assert_eq!(headers, expected);
    }
}

#[test]
fn test_bd_2z0_11_8_export_all_tables_and_verify_roundtrip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = build_fixture_db(&dir);
    let reader = StorageReader::open_finished(&db_path).expect("open finished storage");

    let formats = [
        AnalyticsExportFormat::Parquet,
        AnalyticsExportFormat::Arrow,
        AnalyticsExportFormat::Csv,
    ];

    for &format in &formats {
        for &table in &ExportTable::ALL {
            let out_path = export_database_table(&reader, table, format, dir.path(), true)
                .unwrap_or_else(|e| {
                    panic!("failed to export {table:?} as {format:?}: {e}");
                });

            assert!(out_path.exists(), "exported file must exist on disk");
            let metadata = fs::metadata(&out_path).expect("file metadata");
            assert!(
                metadata.len() > 0,
                "exported {format:?} file must not be empty"
            );
        }
    }
}

#[test]
fn test_bd_2z0_11_8_summarize_and_sql_conformance_agreement() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = build_fixture_db(&dir);
    let reader = StorageReader::open_finished(&db_path).expect("open finished storage");

    // 1. Run summarize via DataFrame pipeline
    let summary = summarize_run(&reader, 10, 5).expect("summarize run");
    assert_eq!(summary.epoch_size, 10);
    assert_eq!(summary.rolling_window, 5);

    // Verify Diet classes: 18 Herbivores, 10 Carnivores, 9 Omnivores across the 10 ticks
    let herbivore_summary = summary
        .diet_summaries
        .iter()
        .find(|d| d.diet_class == "Herbivore")
        .expect("herbivore summary row");
    assert_eq!(herbivore_summary.population, 18);

    let carnivore_summary = summary
        .diet_summaries
        .iter()
        .find(|d| d.diet_class == "Carnivore")
        .expect("carnivore summary row");
    assert_eq!(carnivore_summary.population, 10);

    let omnivore_summary = summary
        .diet_summaries
        .iter()
        .find(|d| d.diet_class == "Omnivore")
        .expect("omnivore summary row");
    assert_eq!(omnivore_summary.population, 9);

    // Verify Lineage founder subtree: Founder 1 has child 3 and grandchild 4 -> 2 descendants
    let founder1 = summary
        .founder_lineages
        .iter()
        .find(|f| f.founder_uid == 1)
        .expect("founder 1 row");
    assert_eq!(founder1.direct_offspring, 1);
    assert_eq!(founder1.total_descendants, 2);
    assert_eq!(founder1.max_generation_depth, 2);

    // Verify Markdown table contains all major sections
    assert!(summary.markdown_table.contains("# Run Summary Report:"));
    assert!(
        summary
            .markdown_table
            .contains("## Per-Epoch Diet Demographics")
    );
    assert!(
        summary
            .markdown_table
            .contains("## Per-Epoch Brain Architecture")
    );
    assert!(
        summary
            .markdown_table
            .contains("## Founder Lineage Contributions")
    );
    assert!(summary.markdown_table.contains("Herbivore"));
    assert!(summary.markdown_table.contains("Carnivore"));

    // 2. SQL cross-check for conformance evidence
    let metric_rows = reader.load_metric_export_rows().expect("load metrics");
    let pop_metrics: Vec<f64> = metric_rows
        .iter()
        .filter(|m| m.name == "population")
        .map(|m| m.value)
        .collect();
    let sql_mean_pop = pop_metrics.iter().sum::<f64>()
        / f64::from(u16::try_from(pop_metrics.len()).expect("ten fixture metric rows fit u16"));

    let pop_df_rows: Vec<f64> = summary
        .rolling_metrics
        .iter()
        .filter(|m| m.metric_name == "population")
        .map(|m| m.raw_value)
        .collect();
    let df_mean_pop = pop_df_rows.iter().sum::<f64>()
        / f64::from(u16::try_from(pop_df_rows.len()).expect("ten fixture metric rows fit u16"));

    assert_eq!(
        sql_mean_pop, df_mean_pop,
        "SQL population mean must match DataFrame raw metric mean"
    );
}

#[test]
fn test_bd_2z0_11_8_cli_export_and_summarize_subcommands() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = build_fixture_db(&dir);
    let export_dir = dir.path().join("cli_exports");

    // Test sb-analyze export --format parquet --verify
    let export_status = Command::new(env!("CARGO_BIN_EXE_sb-analyze"))
        .arg(&db_path)
        .arg("export")
        .arg("--format")
        .arg("parquet")
        .arg("--out-dir")
        .arg(&export_dir)
        .arg("--verify")
        .status()
        .expect("invoke sb-analyze export");

    assert!(export_status.success(), "sb-analyze export must exit 0");

    // Test sb-analyze summarize --json ... --md ...
    let json_out = dir.path().join("summary.json");
    let md_out = dir.path().join("summary.md");

    let summarize_status = Command::new(env!("CARGO_BIN_EXE_sb-analyze"))
        .arg(&db_path)
        .arg("summarize")
        .arg("--epoch-size")
        .arg("10")
        .arg("--rolling-window")
        .arg("5")
        .arg("--json")
        .arg(&json_out)
        .arg("--md")
        .arg(&md_out)
        .status()
        .expect("invoke sb-analyze summarize");

    assert!(
        summarize_status.success(),
        "sb-analyze summarize must exit 0"
    );
    assert!(json_out.exists(), "summary JSON output must exist");
    assert!(md_out.exists(), "summary MD output must exist");

    let json_content = fs::read_to_string(&json_out).expect("read summary json");
    assert!(json_content.contains("\"diet_summaries\""));
    assert!(json_content.contains("\"founder_lineages\""));

    let md_content = fs::read_to_string(&md_out).expect("read summary md");
    assert!(md_content.contains("# Run Summary Report:"));
}
