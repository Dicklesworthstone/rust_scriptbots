use fsqlite::{
    Connection,
    compat::{OpenFlags, RowExt, open_with_flags},
};
use scriptbots_core::{
    AgentData, AgentUid, BrainRunner, INPUT_SIZE, OUTPUT_SIZE, PersistenceBatch, Position,
    ReplayEvent, ReplayEventKind, ReplayInteractionKind, ScriptBotsConfig, Tick, TickSummary,
    WorldState,
    channels::OutputChannel,
    narrative::{EVENT_RECORD_SCHEMA_VERSION, EventKind, EventRecord, SubjectRef},
};
use scriptbots_runtime::RunId;
use scriptbots_storage::{
    ExportFormat, ExportTable, NarrativeQueryError, RunEventDecodeError, RunEventField,
    RunEventIdentity, RunManifestRecord, Storage, StorageDeadlines, StorageError, StoragePipeline,
    StorageReader, export_storage_table, verify_export_receipt,
};
use std::{
    fs,
    io::Write,
    process::Command,
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[test]
fn storage_persists_metrics_roundtrip() {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_micros();
    let path = std::env::temp_dir().join(format!(
        "scriptbots_storage_test_{}_{}.sqlite",
        std::process::id(),
        timestamp
    ));

    let path_str = path.to_str().expect("utf8 path");
    let mut pipeline =
        StoragePipeline::create_unattributed_file_with_thresholds(path_str, 1, 1, 1, 1)
            .expect("pipeline");
    let analytics = pipeline.analytics_provider();
    pipeline
        .submit(&PersistenceBatch {
            summary: TickSummary {
                tick: Tick(0),
                agent_count: 0,
                births: 0,
                deaths: 0,
                total_energy: 0.0,
                average_energy: 0.0,
                average_health: 0.0,
                max_age: 0,
                spike_hits: 0,
            },
            epoch: 0,
            closed: false,
            metrics: Vec::new(),
            events: Vec::new(),
            agents: Vec::new(),
            births: Vec::new(),
            deaths: Vec::new(),
            replay_events: vec![ReplayEvent {
                agent_uid: None,
                position: None,
                counterpart: None,
                counterpart_position: None,
                kind: ReplayEventKind::BrainOutputs {
                    outputs: vec![0.25, 0.75],
                },
            }],
            narrative_events: Vec::new(),
            genomes: Vec::new(),
        })
        .expect("explicit replay fixture should enter the bounded queue");

    let config = ScriptBotsConfig {
        world_width: 128,
        world_height: 128,
        food_cell_size: 16,
        initial_food: 0.25,
        food_max: 1.0,
        persistence_interval: 1,
        history_capacity: 32,
        ..ScriptBotsConfig::default()
    };

    {
        let (mut world, mut persistence) =
            WorldState::with_persistence(config, Box::new(pipeline.sink())).expect("world");
        world
            .try_spawn_agent(AgentData::default())
            .expect("default agent is finite");

        for _ in 0..5 {
            persistence
                .step(&mut world)
                .expect("file-backed persistence step");
        }
    }
    let shutdown = pipeline.shutdown().expect("durable pipeline shutdown");
    assert!(
        shutdown.committed_tick.is_some(),
        "expected a committed tick receipt"
    );
    assert_eq!(
        shutdown.guarantee,
        scriptbots_storage::PersistenceGuarantee::Durable
    );

    let snapshot = analytics.snapshot();
    assert!(
        !snapshot.readings.is_empty(),
        "expected published analytics readings"
    );
    assert!(
        snapshot.committed_tick.is_some(),
        "expected a committed analytics tick"
    );
    assert!(snapshot.stopped, "shutdown should be visible to readers");

    let storage = StorageReader::open(path_str).expect("open storage after pipeline shutdown");

    let predators = storage.top_predators(4).expect("top predators query");
    assert!(
        predators.len() <= 4,
        "top predators should not exceed requested limit"
    );

    let max_tick = storage.max_tick().expect("max tick");
    assert!(max_tick.is_some(), "expected ticks recorded");

    let replay_events = storage.load_replay_events().expect("replay events");
    assert!(
        !replay_events.is_empty(),
        "expected at least one replay event"
    );

    let counts = storage.replay_event_counts().expect("replay event counts");
    assert!(
        !counts.is_empty(),
        "expected replay event counts to be populated"
    );

    storage.close().expect("close storage reader explicitly");
    let _ = fs::remove_file(&path);
}

const NARRATIVE_CHILD_PATH: &str = "SCRIPTBOTS_NARRATIVE_CHILD_PATH";
const NARRATIVE_CHILD_SEED: &str = "SCRIPTBOTS_NARRATIVE_CHILD_SEED";
const NARRATIVE_CRASH_EXIT: i32 = 86;

#[derive(Clone)]
struct NarrativeLogBuffer(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for NarrativeLogBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .expect("narrative log buffer poisoned")
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn narrative_log_buffer() -> NarrativeLogBuffer {
    static BUFFER: LazyLock<NarrativeLogBuffer> = LazyLock::new(|| {
        let buffer = NarrativeLogBuffer(Arc::new(Mutex::new(Vec::new())));
        let writer = buffer.clone();
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .try_init();
        buffer
    });
    BUFFER.clone()
}

impl NarrativeLogBuffer {
    fn records_for(&self, needle: &str) -> Vec<String> {
        let text = String::from_utf8_lossy(&self.0.lock().expect("narrative log buffer poisoned"))
            .into_owned();
        text.lines()
            .filter(|line| line.contains(needle))
            .map(str::to_owned)
            .collect()
    }
}

fn narrative_test_path(label: &str, seed: u64) -> std::path::PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "scriptbots_narrative_{label}_{seed:016x}_{}_{}.sqlite",
        std::process::id(),
        timestamp
    ))
}

fn curated_narrative_events(seed: u64) -> Vec<EventRecord> {
    let base = 20 + seed % 31;
    vec![
        EventRecord {
            schema_version: EVENT_RECORD_SCHEMA_VERSION,
            tick: Tick(base),
            kind: EventKind::PopulationBoom,
            severity: 0.75,
            magnitude: 8.0 + (seed & 7) as f64,
            window: (base - 4, base),
            metric: "population".to_owned(),
            before: 24.0,
            after: 48.0,
            score: 4.5,
            subject: None,
            human_text: format!("seed {seed:016x}: population doubled"),
        },
        EventRecord {
            schema_version: EVENT_RECORD_SCHEMA_VERSION,
            tick: Tick(base + 3),
            kind: EventKind::DietShift,
            severity: 0.5,
            magnitude: 0.25,
            window: (base, base + 3),
            metric: "diet.mix".to_owned(),
            before: 0.25,
            after: 0.5,
            score: 3.0,
            subject: Some(SubjectRef::Species(seed & 0xff)),
            human_text: format!("seed {seed:016x}: diet shifted"),
        },
        EventRecord {
            schema_version: EVENT_RECORD_SCHEMA_VERSION,
            tick: Tick(base),
            kind: EventKind::EnergyRecovery,
            severity: 0.25,
            magnitude: 1.5,
            window: (base - 2, base),
            metric: "energy.mean".to_owned(),
            before: 2.0,
            after: 3.5,
            score: 2.25,
            subject: Some(SubjectRef::Agent(AgentUid(seed))),
            human_text: format!("seed {seed:016x}: energy recovered"),
        },
    ]
}

fn narrative_batch(events: Vec<EventRecord>) -> PersistenceBatch {
    let tick = events
        .iter()
        .map(|event| event.tick.0)
        .max()
        .expect("curated run has events");
    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(tick),
            agent_count: 0,
            births: 0,
            deaths: 0,
            total_energy: 0.0,
            average_energy: 0.0,
            average_health: 0.0,
            max_age: 0,
            spike_hits: 0,
        },
        epoch: 0,
        closed: false,
        metrics: Vec::new(),
        events: Vec::new(),
        agents: Vec::new(),
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events: Vec::new(),
        narrative_events: events,
        genomes: Vec::new(),
    }
}

fn expected_narrative_pairs(seed: u64) -> Vec<(RunEventIdentity, EventRecord)> {
    let mut expected = curated_narrative_events(seed)
        .into_iter()
        .map(|record| (RunEventIdentity::from_record(&record), record))
        .collect::<Vec<_>>();
    expected.sort_by(|left, right| left.0.cmp(&right.0));
    expected
}

/// Guarded child role: durably admit the exact narrative payload, then emulate a process crash
/// before the high-threshold worker can apply it. The parent owns all assertions.
#[test]
fn narrative_outbox_crash_child() {
    let (Ok(path), Ok(seed)) = (
        std::env::var(NARRATIVE_CHILD_PATH),
        std::env::var(NARRATIVE_CHILD_SEED),
    ) else {
        return;
    };
    let seed = seed.parse::<u64>().expect("child seed is u64");
    let run_id = RunId::new(u128::from(seed));
    let mut manifest = RunManifestRecord::unattributed(run_id);
    manifest.root_seed = seed;
    manifest.variant_id = Some(format!("curated-seed-{seed:016x}"));
    let pipeline = StoragePipeline::create_new_file_for_run_with_thresholds(
        &path, manifest, 10_000, 10_000, 10_000, 10_000,
    )
    .expect("child pipeline opens");
    let receipt = pipeline
        .submit_with_receipt(&narrative_batch(curated_narrative_events(seed)))
        .expect("child payload reaches the durable outbox");
    println!(
        "narrative-child: path={path} seed={seed:016x} batch={} admitted={:?}",
        receipt.batch_id.get(),
        receipt.watermarks.admitted
    );
    std::io::stdout().flush().expect("child stdout flushes");
    std::process::exit(NARRATIVE_CRASH_EXIT);
}

/// Three seeded, curated narrative runs survive a real admitted-before-apply process exit.
#[test]
fn narrative_events_recover_as_byte_identical_typed_records_with_stable_identities() {
    let logs = narrative_log_buffer();
    for seed in [0xCA1F_u64, 0x5EED, 0xD1E7] {
        let path = narrative_test_path("recovery", seed);
        let path_str = path.to_str().expect("utf8 path");
        let output = Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", "narrative_outbox_crash_child", "--nocapture"])
            .env(NARRATIVE_CHILD_PATH, path_str)
            .env(NARRATIVE_CHILD_SEED, seed.to_string())
            .output()
            .expect("narrative child launches");
        assert_eq!(
            output.status.code(),
            Some(NARRATIVE_CRASH_EXIT),
            "child must exit at the admitted-before-apply crash boundary: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let mut recovered =
            StoragePipeline::recover_existing(path_str).expect("outbox recovery opens");
        let shutdown = recovered.shutdown().expect("recovered run shuts down");
        assert_eq!(shutdown.watermarks.admitted, shutdown.watermarks.applied);
        assert_eq!(shutdown.watermarks.applied, shutdown.watermarks.durable);

        let expected = expected_narrative_pairs(seed);
        let reader = StorageReader::open(path_str).expect("typed reader opens");
        let first = reader
            .recent_run_events(64)
            .expect("complete narrative events decode")
            .into_iter()
            .map(|event| event.into_parts())
            .collect::<Vec<_>>();
        let recovered_search = reader
            .search_narrative("seed", None, 64)
            .expect("recovered events are indexed in the same applied transaction");
        assert_eq!(
            recovered_search.len(),
            expected.len(),
            "every recovered relational narrative row must be searchable"
        );
        reader.close().expect("typed reader closes");
        assert_eq!(
            serde_json::to_vec(&first).expect("first read serializes"),
            serde_json::to_vec(&expected).expect("expected events serialize"),
            "seed {seed:016x} did not reload byte-identical full EventRecord values"
        );

        let mut recovered_again =
            StoragePipeline::recover_existing(path_str).expect("fixed-point recovery opens");
        recovered_again
            .shutdown()
            .expect("fixed-point recovery shuts down");
        let reader = StorageReader::open(path_str).expect("second typed reader opens");
        let second = reader
            .recent_run_events(64)
            .expect("second complete narrative read")
            .into_iter()
            .map(|event| event.into_parts())
            .collect::<Vec<_>>();
        reader.close().expect("second typed reader closes");
        assert_eq!(
            serde_json::to_vec(&second).expect("second read serializes"),
            serde_json::to_vec(&first).expect("first read reserializes"),
            "seed {seed:016x} identities or evidence changed across repeated recovery"
        );
        assert_eq!(
            second
                .iter()
                .map(|(identity, _)| identity)
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|(identity, _)| identity)
                .collect::<Vec<_>>(),
            "seed {seed:016x} reader order is not canonical identity order"
        );

        let event_logs = logs
            .records_for(path_str)
            .into_iter()
            .filter(|line| line.contains("narrative event persisted"))
            .collect::<Vec<_>>();
        assert_eq!(
            event_logs.len(),
            expected.len(),
            "one structured info record is required per recovered event: {event_logs:?}"
        );
        for (identity, record) in &expected {
            let matching = event_logs
                .iter()
                .filter(|line| line.contains(&format!("event_identity={identity}")))
                .collect::<Vec<_>>();
            assert_eq!(
                matching.len(),
                1,
                "identity {identity} must have exactly one structured record: {event_logs:?}"
            );
            let line = matching[0];
            for field in [
                "event_tick=",
                "event_kind=",
                "event_metric=",
                "schema_version=",
                "severity=",
                "magnitude=",
                "window_start=",
                "window_end=",
                "before=",
                "after=",
                "score=",
                "subject_ref=",
                "human_text=",
            ] {
                assert!(
                    line.contains(field),
                    "structured record for {identity} omitted {field}: {line}"
                );
            }
            assert!(line.contains(" INFO "), "event record is not INFO: {line}");
            assert!(line.contains(record.kind.as_str()));
            assert!(line.contains(&record.metric));
            assert!(line.contains(&record.human_text));
            if let Some(subject) = record.subject {
                assert!(line.contains(&subject.to_db_string()));
            }
        }

        let _ = fs::remove_file(&path);
    }
}

#[test]
fn narrative_search_is_ranked_bounded_literal_and_run_scoped() {
    let logs = narrative_log_buffer();
    let seed = 0xF75_u64;
    let path = narrative_test_path("search", seed);
    let path_str = path.to_str().expect("utf8 path");
    let run_a = RunId::new(0xA);
    let run_b = RunId::new(0xB);

    let mut events = curated_narrative_events(seed);
    let base = events[0].tick.0;
    events[0].human_text = "drought drought caused population crash".to_owned();
    events[1].human_text = "drought caused a diet shift".to_owned();
    events[2].human_text = "clima café recovered energy".to_owned();
    let mut literal = events[0].clone();
    literal.tick = Tick(literal.tick.0 + 1);
    literal.metric = "quoted.operator".to_owned();
    literal.human_text = "literal OR quote \"danger\"".to_owned();
    events.push(literal);
    let mut tie_a = events[0].clone();
    tie_a.tick = Tick(base + 4);
    tie_a.metric = "tie.a".to_owned();
    tie_a.human_text = "equal tie token".to_owned();
    let mut tie_b = tie_a.clone();
    tie_b.tick = Tick(tie_a.tick.0 + 1);
    tie_b.metric = "tie.b".to_owned();
    events.extend([tie_a, tie_b]);

    let mut manifest_a = RunManifestRecord::unattributed(run_a);
    manifest_a.root_seed = seed;
    let mut pipeline =
        StoragePipeline::create_new_file_for_run_with_thresholds(path_str, manifest_a, 1, 1, 1, 1)
            .expect("search run A opens");
    pipeline
        .submit(&narrative_batch(events.clone()))
        .expect("search events admit");
    pipeline.shutdown().expect("search run A shuts down");

    let mut run_b_event = curated_narrative_events(seed + 1)
        .into_iter()
        .next()
        .expect("run B event");
    run_b_event.human_text = "drought belongs only to run B".to_owned();
    let mut manifest_b = RunManifestRecord::unattributed(run_b);
    manifest_b.root_seed = seed + 1;
    let mut pipeline =
        StoragePipeline::append_run(path_str, manifest_b).expect("search run B appends");
    pipeline
        .submit(&narrative_batch(vec![run_b_event]))
        .expect("run B search event admits");
    pipeline.shutdown().expect("search run B shuts down");

    let reader_a = StorageReader::open_for_run(path_str, run_a).expect("run A reader opens");
    let started = Instant::now();
    let drought = reader_a
        .search_narrative("drought", None, 16)
        .expect("literal drought search");
    let elapsed = started.elapsed();
    assert_eq!(drought.len(), 2, "run B must not leak into run A results");
    assert_eq!(
        drought[0].human_text(),
        "drought drought caused population crash",
        "the repeat-heavy document should receive the better BM25 score"
    );
    assert!(
        drought
            .windows(2)
            .all(|pair| pair[0].rank().expect("rank") <= pair[1].rank().expect("rank")),
        "BM25 results are not ordered best-first: {drought:?}"
    );
    let repeated = reader_a
        .search_narrative("drought", None, 16)
        .expect("repeat search");
    assert_eq!(
        drought
            .iter()
            .map(|hit| (hit.event().identity().clone(), hit.rank()))
            .collect::<Vec<_>>(),
        repeated
            .iter()
            .map(|hit| (hit.event().identity().clone(), hit.rank()))
            .collect::<Vec<_>>(),
        "BM25 scores and canonical tie-break order changed between identical reads"
    );
    assert_eq!(
        reader_a
            .search_narrative("drought", None, 1)
            .expect("limited search")
            .len(),
        1
    );
    assert!(
        reader_a
            .search_narrative("drought", None, 0)
            .expect("zero-result limit")
            .is_empty()
    );
    assert_eq!(
        reader_a
            .search_narrative("drought", Some((base + 2, base + 4)), 16)
            .expect("tick-filtered search")
            .len(),
        1
    );
    let unicode = reader_a
        .search_narrative("cafe", None, 16)
        .expect("unicode61 diacritic-normalized search");
    assert_eq!(unicode.len(), 1);
    assert_eq!(unicode[0].human_text(), "clima café recovered energy");
    assert_eq!(
        reader_a
            .search_narrative("café", None, 16)
            .expect("non-ASCII literal query")
            .len(),
        1
    );
    let ties = reader_a
        .search_narrative("tie token", None, 16)
        .expect("equal-score tie search");
    assert_eq!(ties.len(), 2);
    assert_eq!(ties[0].rank(), ties[1].rank());
    assert!(
        ties[0].event().identity() < ties[1].event().identity(),
        "equal BM25 scores must use canonical identity order"
    );
    let quoted = reader_a
        .search_narrative("literal OR quote \"danger\"", None, 16)
        .expect("quoted operator-like input remains literal");
    assert_eq!(quoted.len(), 1);
    assert!(
        reader_a
            .search_narrative("drought' OR 1=1 --", None, 16)
            .expect("SQL-shaped text remains a bound literal")
            .is_empty()
    );
    assert!(matches!(
        reader_a.search_narrative("   ", None, 16),
        Err(StorageError::NarrativeQuery(NarrativeQueryError::Empty))
    ));
    assert!(matches!(
        reader_a.search_narrative(&"x".repeat(1_025), None, 16),
        Err(StorageError::NarrativeQuery(
            NarrativeQueryError::TooLong { .. }
        ))
    ));
    assert!(matches!(
        reader_a.search_narrative(&format!("needle{}", " ".repeat(1_025)), None, 16),
        Err(StorageError::NarrativeQuery(
            NarrativeQueryError::TooLong { .. }
        ))
    ));
    assert!(matches!(
        reader_a.search_narrative("nul\0byte", None, 16),
        Err(StorageError::NarrativeQuery(
            NarrativeQueryError::ContainsNul
        ))
    ));
    assert!(matches!(
        reader_a.search_narrative("drought", Some((base, base)), 16),
        Err(StorageError::NarrativeQuery(
            NarrativeQueryError::InvalidTickRange { .. }
        ))
    ));
    assert!(matches!(
        reader_a.search_narrative("drought", Some((base + 1, base)), 16),
        Err(StorageError::NarrativeQuery(
            NarrativeQueryError::InvalidTickRange { .. }
        ))
    ));
    assert!(matches!(
        reader_a.search_narrative("drought", None, 4_097),
        Err(StorageError::NarrativeQuery(
            NarrativeQueryError::LimitTooLarge { .. }
        ))
    ));
    let around = reader_a
        .narrative_around_tick(base + 1, 1)
        .expect("bounded tick window");
    assert_eq!(around.len(), 3);
    assert_eq!(
        around.iter().map(|hit| hit.tick().0).collect::<Vec<_>>(),
        [base, base, base + 1],
        "tick-window results must preserve canonical chronological order"
    );
    assert!(
        around[0].event().identity() < around[1].event().identity(),
        "same-tick results must use the canonical identity tie-break"
    );
    assert!(around.iter().all(|hit| hit.rank().is_none()));
    assert_eq!(
        reader_a
            .narrative_around_tick(0, u64::MAX)
            .expect("saturated tick window")
            .len(),
        events.len(),
        "zero and i64::MAX window saturation must retain all in-range rows"
    );
    reader_a.close().expect("run A reader closes");

    let reader_b = StorageReader::open_for_run(path_str, run_b).expect("run B reader opens");
    let run_b_hits = reader_b
        .search_narrative("drought", None, 16)
        .expect("run B search");
    assert_eq!(run_b_hits.len(), 1);
    assert_eq!(run_b_hits[0].human_text(), "drought belongs only to run B");
    reader_b.close().expect("run B reader closes");

    eprintln!(
        "narrative-search measurement: rows={} elapsed_us={}",
        drought.len(),
        elapsed.as_micros()
    );
    let run_a_log_key = format!("run_id={run_a}");
    assert!(
        logs.records_for("completed bounded narrative search")
            .iter()
            .any(|line| {
                line.contains(&run_a_log_key)
                    && line.contains("row_count=2")
                    && line.contains("elapsed_micros=")
            }),
        "search tracing omitted run, row-count, or duration evidence"
    );
    let _ = fs::remove_file(&path);
}

#[test]
#[ignore = "10k-event observation lane; run explicitly through RCH"]
fn narrative_search_records_10k_event_latency_distribution() {
    const EVENT_COUNT: u64 = 10_000;
    const SAMPLE_COUNT: usize = 20;

    let logs = narrative_log_buffer();
    let seed = 10_000_u64;
    let path = narrative_test_path("10k-latency", seed);
    let path_str = path.to_str().expect("utf8 path");
    let run_id = RunId::new(10_000);
    let events = (0_u64..EVENT_COUNT)
        .map(|tick| EventRecord {
            schema_version: EVENT_RECORD_SCHEMA_VERSION,
            tick: Tick(tick),
            kind: EventKind::PopulationBoom,
            severity: 0.5,
            magnitude: 1.0,
            window: (tick.saturating_sub(1), tick),
            metric: format!("latency.{tick:05}"),
            before: 1.0,
            after: 2.0,
            score: 1.0,
            subject: None,
            human_text: if tick % 100 == 0 {
                format!("latency needle population event {tick}")
            } else {
                format!("background narrative population event {tick}")
            },
        })
        .collect::<Vec<_>>();

    let mut manifest = RunManifestRecord::unattributed(run_id);
    manifest.root_seed = seed;
    let mut pipeline =
        StoragePipeline::create_new_file_for_run_with_thresholds(path_str, manifest, 1, 1, 1, 1)
            .expect("10k measurement pipeline opens");
    pipeline
        .submit(&narrative_batch(events))
        .expect("10k narrative batch admits through the normal outbox path");
    pipeline
        .shutdown()
        .expect("10k measurement pipeline shuts down");

    let reader = StorageReader::open_for_run(path_str, run_id).expect("10k reader opens");
    let warmup = reader
        .search_narrative("latency needle", None, 256)
        .expect("10k search warmup");
    assert_eq!(warmup.len(), 100);
    let expected = warmup
        .iter()
        .map(|hit| (hit.event().identity().clone(), hit.rank()))
        .collect::<Vec<_>>();
    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    for sample in 0..SAMPLE_COUNT {
        let started = Instant::now();
        let hits = reader
            .search_narrative("latency needle", None, 256)
            .expect("10k timed search");
        let elapsed = started.elapsed();
        assert_eq!(
            hits.iter()
                .map(|hit| (hit.event().identity().clone(), hit.rank()))
                .collect::<Vec<_>>(),
            expected,
            "10k query identities or BM25 scores changed at sample {sample}"
        );
        eprintln!(
            "narrative-search-10k sample={} rows={} elapsed_us={}",
            sample + 1,
            hits.len(),
            elapsed.as_micros()
        );
        samples.push(elapsed);
    }
    samples.sort_unstable();
    let p50 = samples[SAMPLE_COUNT / 2];
    let p95 = samples[(SAMPLE_COUNT * 95).div_ceil(100) - 1];
    eprintln!(
        "narrative-search-10k summary events={EVENT_COUNT} matches={} samples={SAMPLE_COUNT} p50_us={} p95_us={} target_us=10000",
        expected.len(),
        p50.as_micros(),
        p95.as_micros()
    );
    assert!(matches!(
        reader.narrative_around_tick(EVENT_COUNT / 2, EVENT_COUNT),
        Err(StorageError::NarrativeQuery(
            NarrativeQueryError::AroundWindowTooDense { max: 4_096 }
        ))
    ));
    reader.close().expect("10k reader closes");

    let run_log_key = format!("run_id={run_id}");
    assert!(
        logs.records_for("completed bounded narrative search")
            .iter()
            .any(|line| {
                line.contains(&run_log_key)
                    && line.contains("row_count=100")
                    && line.contains("elapsed_micros=")
            }),
        "10k search tracing omitted run, row-count, or duration evidence"
    );
    let _ = fs::remove_file(&path);
}

fn create_narrative_fixture(path: &str, event: EventRecord) {
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(path, 1, 1, 1, 1)
        .expect("fixture pipeline opens");
    pipeline
        .submit(&narrative_batch(vec![event]))
        .expect("fixture event admits");
    pipeline.shutdown().expect("fixture pipeline shuts down");
}

fn mutate_narrative_fixture(path: &str, sql: &str) {
    let connection = Connection::open(path).expect("fixture writer opens");
    connection.execute(sql).expect("fixture mutation applies");
    connection.close().expect("fixture writer closes");
}

#[test]
fn narrative_reader_refuses_future_and_malformed_rows_without_partial_defaults() {
    let seed = 0xA11CE_u64;
    let path = narrative_test_path("malformed", seed);
    let path_str = path.to_str().expect("utf8 path");
    let event = curated_narrative_events(seed)
        .into_iter()
        .next()
        .expect("curated event");
    create_narrative_fixture(path_str, event);

    mutate_narrative_fixture(path_str, "UPDATE run_events SET schema_version = 2");
    let reader = StorageReader::open(path_str).expect("reader opens future row");
    let error = reader
        .recent_run_events(8)
        .expect_err("future schema must be refused");
    assert!(matches!(
        error,
        StorageError::RunEvent(RunEventDecodeError::UnsupportedSchemaVersion {
            found: 2,
            supported: EVENT_RECORD_SCHEMA_VERSION,
        })
    ));
    reader.close().expect("reader closes");
    mutate_narrative_fixture(path_str, "UPDATE run_events SET schema_version = 1");

    mutate_narrative_fixture(path_str, "UPDATE run_events SET kind = 'unknown_kind'");
    let reader = StorageReader::open(path_str).expect("reader opens malformed kind");
    let error = reader
        .recent_run_events(8)
        .expect_err("malformed identity kind must be refused");
    assert!(matches!(
        error,
        StorageError::RunEvent(RunEventDecodeError::InvalidField {
            field: RunEventField::Kind,
            ..
        })
    ));
    reader.close().expect("reader closes");
    mutate_narrative_fixture(
        path_str,
        "UPDATE run_events SET kind = 'population_boom', subject_ref = 'agent:not-a-number'",
    );

    let reader = StorageReader::open(path_str).expect("reader opens malformed subject");
    let error = reader
        .recent_run_events(8)
        .expect_err("malformed typed subject must be refused");
    assert!(matches!(
        error,
        StorageError::RunEvent(RunEventDecodeError::InvalidField {
            field: RunEventField::Subject,
            ..
        })
    ));
    reader.close().expect("reader closes");
    mutate_narrative_fixture(
        path_str,
        "UPDATE run_events SET subject_ref = NULL, severity = 0.1",
    );

    let reader = StorageReader::open(path_str).expect("reader opens malformed numeric");
    let error = reader
        .recent_run_events(8)
        .expect_err("precision-changing severity must be refused");
    assert!(matches!(
        error,
        StorageError::RunEvent(RunEventDecodeError::InvalidField {
            field: RunEventField::Severity,
            ..
        })
    ));
    reader.close().expect("reader closes");

    let _ = fs::remove_file(&path);
}

#[test]
fn narrative_admission_refuses_duplicate_and_conflicting_identities() {
    let seed = 0xD0B1E_u64;
    let event = curated_narrative_events(seed)
        .into_iter()
        .next()
        .expect("curated event");
    let mut storage = Storage::unattributed_memory().expect("memory storage opens");
    let duplicate = storage
        .persist(&narrative_batch(vec![event.clone(), event.clone()]))
        .expect_err("an identical identity cannot be admitted twice");
    assert!(matches!(
        duplicate,
        StorageError::RunEvent(RunEventDecodeError::DuplicateIdentity { .. })
    ));

    let mut changed = event.clone();
    changed.after += 1.0;
    changed.human_text.push_str(" conflicting");
    let conflict = storage
        .persist(&narrative_batch(vec![event, changed]))
        .expect_err("one identity cannot carry unstable evidence");
    assert!(matches!(
        conflict,
        StorageError::RunEvent(RunEventDecodeError::ConflictingIdentity { .. })
    ));
    storage.close().expect("memory storage closes");
}

/// Build a batch carrying exactly the supplied replay events at `tick`.
fn batch_with_replay_events(tick: u64, replay_events: Vec<ReplayEvent>) -> PersistenceBatch {
    let interaction_count = replay_events
        .iter()
        .filter(|event| matches!(event.kind, ReplayEventKind::Interaction { .. }))
        .count();
    let events = if interaction_count == 0 {
        Vec::new()
    } else {
        vec![
            scriptbots_core::PersistenceEvent::new(
                scriptbots_core::PersistenceEventKind::Custom(std::borrow::Cow::Borrowed(
                    scriptbots_core::INTERACTION_EVENTS_OBSERVED_KIND,
                )),
                interaction_count,
            ),
            scriptbots_core::PersistenceEvent::new(
                scriptbots_core::PersistenceEventKind::Custom(std::borrow::Cow::Borrowed(
                    scriptbots_core::INTERACTION_EVENTS_PERSISTED_KIND,
                )),
                interaction_count,
            ),
        ]
    };
    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(tick),
            agent_count: 2,
            births: 0,
            deaths: 0,
            total_energy: 0.0,
            average_energy: 0.0,
            average_health: 0.0,
            max_age: 0,
            spike_hits: 0,
        },
        epoch: 0,
        closed: false,
        metrics: Vec::new(),
        events,
        agents: Vec::new(),
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events,
        narrative_events: Vec::new(),
        genomes: Vec::new(),
    }
}

fn temp_run_path(label: &str) -> std::path::PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_micros();
    std::env::temp_dir().join(format!(
        "scriptbots_storage_{label}_{}_{}.sqlite",
        std::process::id(),
        timestamp
    ))
}

#[derive(Debug)]
struct GiveIntentBrain {
    give: bool,
}

impl BrainRunner for GiveIntentBrain {
    fn kind(&self) -> &'static str {
        if self.give {
            "test.storage.give"
        } else {
            "test.storage.receive"
        }
    }

    fn tick(&mut self, _inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        let mut outputs = [0.0; OUTPUT_SIZE];
        outputs[OutputChannel::GiveIntent.index()] = f32::from(u8::from(self.give));
        outputs
    }
}

/// A seeded 2k world must preserve the exact core interaction count through durable SQL.
#[test]
fn seeded_2k_world_interaction_count_matches_durable_rows() {
    const AGENTS: usize = 2_000;
    const PAIRS: usize = AGENTS / 2;

    let path = temp_run_path("seeded_2k_interactions");
    let path_str = path.to_str().expect("utf8 path");
    let proof_deadline = Duration::from_secs(10 * 60);
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds_and_deadlines(
        path_str,
        1,
        1,
        1,
        1,
        StorageDeadlines {
            flush_ack: proof_deadline,
            shutdown_ack: proof_deadline,
            ..StorageDeadlines::default()
        },
    )
    .expect("2k durable pipeline");
    let (mut world, mut persistence) = WorldState::with_persistence(
        ScriptBotsConfig {
            world_width: 5_100,
            world_height: 2_100,
            food_cell_size: 50,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_growth_rate: 0.0,
            food_decay_rate: 0.0,
            food_diffusion_rate: 0.0,
            food_intake_rate: 0.0,
            food_waste_rate: 0.0,
            food_transfer_rate: 0.01,
            food_sharing_distance: 2.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            temperature_discomfort_rate: 0.0,
            reproduction_attempt_chance: 0.0,
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 1,
            interaction_event_tick_cap: PAIRS,
            interaction_event_tick_stride: 1,
            rng_seed: Some(0x2000_5EED),
            ..ScriptBotsConfig::default()
        },
        Box::new(pipeline.sink()),
    )
    .expect("2k seeded world");

    let giver_brain = world
        .brain_registry_mut()
        .expect("giver brain registry")
        .register("test.storage.give", |_rng| {
            Ok(Box::new(GiveIntentBrain { give: true }))
        });
    let receiver_brain = world
        .brain_registry_mut()
        .expect("receiver brain registry")
        .register("test.storage.receive", |_rng| {
            Ok(Box::new(GiveIntentBrain { give: false }))
        });

    for pair in 0..PAIRS {
        let grid_x = u16::try_from(pair % 50).expect("2k grid x fits u16");
        let grid_y = u16::try_from(pair / 50).expect("2k grid y fits u16");
        let x = 50.0 + f32::from(grid_x) * 100.0;
        let y = 50.0 + f32::from(grid_y) * 100.0;
        let giver = world
            .try_spawn_agent(AgentData {
                position: Position::new(x, y),
                ..AgentData::default()
            })
            .expect("seed giver");
        let receiver = world
            .try_spawn_agent(AgentData {
                position: Position::new(x + 1.0, y),
                ..AgentData::default()
            })
            .expect("seed receiver");
        assert!(
            world
                .bind_agent_brain(giver, giver_brain)
                .expect("bind giver brain")
        );
        assert!(
            world
                .bind_agent_brain(receiver, receiver_brain)
                .expect("bind receiver brain")
        );
    }

    let completion = persistence
        .step_outcome(&mut world)
        .expect("run the seeded interaction tick");
    assert!(
        completion.fault.is_none(),
        "the seeded science boundary must complete without a contained fault"
    );
    let batch = persistence
        .pending_batch()
        .expect("the interval-one boundary stages a persistence batch");
    let core_edges = batch
        .replay_events
        .iter()
        .filter(|event| matches!(event.kind, ReplayEventKind::Interaction { .. }))
        .count();
    let counter = |kind: &str| {
        batch
            .events
            .iter()
            .find_map(|event| match &event.kind {
                scriptbots_core::PersistenceEventKind::Custom(name) if name == kind => {
                    Some(event.count)
                }
                _ => None,
            })
            .unwrap_or(0)
    };
    let core_observed = counter(scriptbots_core::INTERACTION_EVENTS_OBSERVED_KIND);
    let core_persisted = counter(scriptbots_core::INTERACTION_EVENTS_PERSISTED_KIND);
    let core_sampled_out = counter(scriptbots_core::INTERACTION_EVENTS_SAMPLED_OUT_KIND);
    let core_truncated = counter(scriptbots_core::INTERACTION_EVENTS_TRUNCATED_KIND);
    assert_eq!(batch.summary.agent_count, AGENTS);
    assert_eq!(core_edges, PAIRS);
    assert_eq!(
        (
            core_observed,
            core_persisted,
            core_sampled_out,
            core_truncated
        ),
        (PAIRS, PAIRS, 0, 0)
    );
    assert!(
        persistence
            .admit_pending(&mut world)
            .expect("admit the exact staged 2k batch")
    );
    pipeline
        .flush_and_wait()
        .expect("durably flush the 2k batch");
    pipeline.shutdown().expect("durable pipeline shutdown");

    let reader = open_with_flags(path_str, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("independent read-only reader opens");
    let durable = reader
        .query(
            "SELECT
               (SELECT COUNT(*) FROM interactions),
               COALESCE(SUM(CASE WHEN kind = 'interaction_events_observed' THEN count ELSE 0 END), 0),
               COALESCE(SUM(CASE WHEN kind = 'interaction_events_persisted' THEN count ELSE 0 END), 0)
             FROM events",
        )
        .expect("durable accounting query runs");
    let durable_edges: i64 = durable[0].get_typed(0).expect("edge count");
    let durable_observed: i64 = durable[0].get_typed(1).expect("observed count");
    let durable_persisted: i64 = durable[0].get_typed(2).expect("persisted count");
    assert_eq!(
        usize::try_from(durable_edges).expect("edge count is non-negative"),
        core_edges,
        "the durable interaction graph must contain every world-emitted edge exactly once"
    );
    assert_eq!(
        usize::try_from(durable_observed).expect("observed count is non-negative"),
        core_observed
    );
    assert_eq!(
        usize::try_from(durable_persisted).expect("persisted count is non-negative"),
        core_persisted
    );
    reader.close().expect("read-only reader closes");

    let _ = fs::remove_file(&path);
}

/// Typed combat and food-share facts must retain their source tick, stable participants,
/// positions, kind, and magnitude through the durable projection.
#[test]
fn typed_pairwise_events_persist_as_queryable_interaction_edges() {
    let path = temp_run_path("interaction_edge");
    let path_str = path.to_str().expect("utf8 path");

    let actor = scriptbots_core::AgentUid(7);
    let target = scriptbots_core::AgentUid(11);
    let actor_position = scriptbots_core::Position::new(12.5, -3.25);
    let target_position = scriptbots_core::Position::new(14.0, -2.5);

    {
        let mut pipeline =
            StoragePipeline::create_unattributed_file_with_thresholds(path_str, 1, 1, 1, 1)
                .expect("pipeline");
        pipeline
            .submit(&batch_with_replay_events(
                4,
                vec![
                    ReplayEvent {
                        agent_uid: Some(actor),
                        position: Some(actor_position),
                        counterpart: Some(target),
                        counterpart_position: Some(target_position),
                        kind: ReplayEventKind::Interaction {
                            tick: Tick(2),
                            ordinal: 0,
                            kind: ReplayInteractionKind::Combat,
                            magnitude: 0.375,
                        },
                    },
                    ReplayEvent {
                        agent_uid: Some(target),
                        position: Some(target_position),
                        counterpart: Some(actor),
                        counterpart_position: Some(actor_position),
                        kind: ReplayEventKind::Interaction {
                            tick: Tick(4),
                            ordinal: 0,
                            kind: ReplayInteractionKind::FoodShare,
                            magnitude: 0.125,
                        },
                    },
                    ReplayEvent {
                        agent_uid: Some(actor),
                        position: Some(actor_position),
                        counterpart: None,
                        counterpart_position: None,
                        kind: ReplayEventKind::BrainOutputs {
                            outputs: vec![0.5, 0.5],
                        },
                    },
                ],
            ))
            .expect("hand-built replay fixture enters the bounded queue");
        pipeline.flush_and_wait().expect("flush the staged batch");
        pipeline.shutdown().expect("durable pipeline shutdown");
    }

    let storage = StorageReader::open(path_str).expect("open storage after shutdown");

    // Premise: all events were actually written. Without this the exclusion assertion below
    // would be satisfied by a run that persisted nothing at all.
    let replayed = storage.load_replay_events().expect("replay events");
    assert_eq!(
        replayed.len(),
        3,
        "both interactions and the single-agent control must reach the database"
    );
    let edge_events = replayed
        .iter()
        .filter(|persisted| persisted.event.counterpart.is_some())
        .collect::<Vec<_>>();
    assert_eq!(edge_events.len(), 2);
    assert_eq!(edge_events[0].tick, 2);
    assert_eq!(edge_events[0].event.agent_uid, Some(actor));
    assert_eq!(edge_events[0].event.counterpart, Some(target));
    assert_eq!(
        edge_events[0].event.position,
        Some(actor_position),
        "the emission-time actor position must survive the write path"
    );
    assert_eq!(
        edge_events[0].event.counterpart_position,
        Some(target_position),
        "the emission-time counterpart position must survive the write path"
    );
    assert!(matches!(
        edge_events[0].event.kind,
        ReplayEventKind::Interaction {
            tick: Tick(2),
            ordinal: 0,
            kind: ReplayInteractionKind::Combat,
            magnitude,
        } if magnitude.to_bits() == 0.375_f32.to_bits()
    ));
    assert!(matches!(
        edge_events[1].event.kind,
        ReplayEventKind::Interaction {
            tick: Tick(4),
            ordinal: 0,
            kind: ReplayInteractionKind::FoodShare,
            magnitude,
        } if magnitude.to_bits() == 0.125_f32.to_bits()
    ));

    let interactions = storage.recent_interactions(16).expect("interaction edges");
    assert_eq!(
        interactions.len(),
        2,
        "exactly the pairwise events are interactions; the single-agent event is not: \
         {interactions:?}"
    );
    let combat = &interactions[0];
    assert_eq!(combat.tick, 2);
    assert_eq!(combat.actor, actor);
    assert_eq!(combat.target, target);
    assert_eq!(combat.kind, "combat");
    assert_eq!(combat.actor_position, Some(actor_position));
    assert_eq!(combat.target_position, Some(target_position));
    assert_eq!(combat.value, Some(0.375));
    let food_share = &interactions[1];
    assert_eq!(food_share.tick, 4);
    assert_eq!(food_share.actor, target);
    assert_eq!(food_share.target, actor);
    assert_eq!(food_share.kind, "food_share");
    assert_eq!(food_share.value, Some(0.125));

    storage.close().expect("close storage reader");

    // The edge is answerable in SQL by an offline consumer that never links this crate --
    // the property bd-2z0.5.9 was filed for, and the one a JSON payload could not provide.
    let reader = open_with_flags(path_str, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .expect("independent read-only reader opens");
    let rows = reader
        .query(
            "SELECT tick, actor_agent_uid, target_agent_uid, kind, value FROM interactions
             ORDER BY tick ASC, seq ASC",
        )
        .expect("interactions table is queryable");
    assert_eq!(rows.len(), 2, "exactly the two edges must be recorded");
    let source_tick: i64 = rows[0].get_typed(0).expect("tick is INTEGER");
    let actor_uid: i64 = rows[0].get_typed(1).expect("actor_agent_uid is INTEGER");
    let target_uid: i64 = rows[0].get_typed(2).expect("target_agent_uid is INTEGER");
    let kind: String = rows[0].get_typed(3).expect("kind is TEXT");
    let value: f64 = rows[0].get_typed(4).expect("value is REAL");
    assert_eq!(source_tick, 2);
    assert_eq!(actor_uid, 7);
    assert_eq!(target_uid, 11);
    assert_eq!(kind, "combat");
    assert_eq!(value, 0.375);

    let completeness = reader
        .query(&format!(
            "SELECT
               COALESCE(SUM(CASE WHEN kind = '{}' THEN count ELSE 0 END), 0),
               COALESCE(SUM(CASE WHEN kind = '{}' THEN count ELSE 0 END), 0),
               COALESCE(SUM(CASE WHEN kind = '{}' THEN count ELSE 0 END), 0),
               COALESCE(SUM(CASE WHEN kind = '{}' THEN count ELSE 0 END), 0)
             FROM events",
            scriptbots_core::INTERACTION_EVENTS_OBSERVED_KIND,
            scriptbots_core::INTERACTION_EVENTS_PERSISTED_KIND,
            scriptbots_core::INTERACTION_EVENTS_SAMPLED_OUT_KIND,
            scriptbots_core::INTERACTION_EVENTS_TRUNCATED_KIND,
        ))
        .expect("persisted interaction completeness counters are queryable");
    let observed: i64 = completeness[0].get_typed(0).expect("observed count");
    let projected: i64 = completeness[0].get_typed(1).expect("projected count");
    let sampled_out: i64 = completeness[0].get_typed(2).expect("sampled count");
    let truncated: i64 = completeness[0].get_typed(3).expect("truncated count");
    assert_eq!((observed, projected, sampled_out, truncated), (2, 2, 0, 0));

    // The accounting identity bd-2z0.5.9 asks for: an interaction row exists for exactly the
    // replay events that name two participants -- no edge without an event, no pairwise event
    // without an edge. Expressed as SQL over both tables rather than as two Rust counts, so it
    // also proves the shared (run_id, tick, seq) key really does join them.
    let orphans = reader
        .query(
            "SELECT
               (SELECT COUNT(*) FROM interactions i
                  LEFT JOIN replay_events e
                    ON e.run_id = i.run_id AND e.tick = i.tick AND e.seq = i.seq
                 WHERE e.run_id IS NULL),
               (SELECT COUNT(*) FROM replay_events e
                  LEFT JOIN interactions i
                    ON i.run_id = e.run_id AND i.tick = e.tick AND i.seq = e.seq
                 WHERE e.agent_uid IS NOT NULL
                   AND e.counterpart_uid IS NOT NULL
                   AND i.run_id IS NULL)",
        )
        .expect("accounting identity query runs");
    let edges_without_events: i64 = orphans[0].get_typed(0).expect("count is INTEGER");
    let pairwise_events_without_edges: i64 = orphans[0].get_typed(1).expect("count is INTEGER");
    assert_eq!(
        edges_without_events, 0,
        "an edge exists with no source event"
    );
    assert_eq!(
        pairwise_events_without_edges, 0,
        "a pairwise event was persisted without its interaction edge"
    );
    assert_eq!(
        observed,
        projected + sampled_out + truncated,
        "persisted completeness counters must account for every observed interaction"
    );
    assert_eq!(
        projected,
        i64::try_from(rows.len()).expect("row count fits i64"),
        "the projected counter must equal the durable SQL edge count"
    );

    reader.close().expect("read-only reader closes");

    let _ = fs::remove_file(&path);
}

/// A run whose events name no counterpart must yield an empty interaction set, not an error
/// and not a row with an invented participant.
///
/// The negative half of the guard above. Exercised in both directions because a writer with a
/// wrong pairwise filter -- say, one that required only an `agent_uid` -- would still satisfy
/// the positive test while turning every ordinary brain-output event into a fictional edge
/// between an agent and itself.
#[test]
fn events_without_a_counterpart_produce_no_interaction_edges() {
    let path = temp_run_path("no_interaction_edges");
    let path_str = path.to_str().expect("utf8 path");

    {
        let mut pipeline =
            StoragePipeline::create_unattributed_file_with_thresholds(path_str, 1, 1, 1, 1)
                .expect("pipeline");
        pipeline
            .submit(&batch_with_replay_events(
                1,
                vec![ReplayEvent {
                    agent_uid: Some(scriptbots_core::AgentUid(3)),
                    position: Some(scriptbots_core::Position::new(1.0, 2.0)),
                    counterpart: None,
                    counterpart_position: None,
                    kind: ReplayEventKind::BrainOutputs {
                        outputs: vec![0.1, 0.2],
                    },
                }],
            ))
            .expect("single-agent fixture enters the bounded queue");
        pipeline.flush_and_wait().expect("flush the staged batch");
        pipeline.shutdown().expect("durable pipeline shutdown");
    }

    let storage = StorageReader::open(path_str).expect("open storage after shutdown");
    assert_eq!(
        storage.load_replay_events().expect("replay events").len(),
        1,
        "premise: the event was persisted, so an empty interaction set is a real exclusion \
         rather than an empty database"
    );
    assert!(
        storage
            .recent_interactions(16)
            .expect("interaction edges")
            .is_empty(),
        "an event with no counterpart is not an interaction"
    );
    storage.close().expect("close storage reader");
    let _ = fs::remove_file(&path);
}

#[test]
fn test_bd_2z0_5_6_e2e_real_db_export_roundtrip() {
    let start_time = Instant::now();
    let path = temp_run_path("export_contracts_e2e");
    let path_str = path.to_str().expect("utf8 path");

    let mut pipeline =
        StoragePipeline::create_unattributed_file_with_thresholds(path_str, 1, 1, 1, 1)
            .expect("pipeline");

    let config = ScriptBotsConfig {
        world_width: 128,
        world_height: 128,
        food_cell_size: 16,
        initial_food: 0.25,
        food_max: 1.0,
        persistence_interval: 1,
        history_capacity: 32,
        ..ScriptBotsConfig::default()
    };

    {
        let (mut world, mut persistence) =
            WorldState::with_persistence(config, Box::new(pipeline.sink())).expect("world");
        world
            .try_spawn_agent(AgentData::default())
            .expect("spawn default agent");

        for _ in 0..5 {
            persistence.step(&mut world).expect("persistence step");
        }
    }

    let shutdown = pipeline.shutdown().expect("durable pipeline shutdown");
    assert!(shutdown.committed_tick.is_some());

    let storage = StorageReader::open(path_str).expect("open storage after shutdown");
    let manifest = storage.run_manifest().expect("load run manifest");

    eprintln!(
        "[bd-2z0.5.6] Exporting real database: run_id={}, manifest_schema={}, root_seed={}, started_at_unix_ms={}",
        manifest.run_id,
        manifest.manifest_schema_version,
        manifest.root_seed,
        manifest.started_at_unix_ms
    );

    for table in ExportTable::ALL {
        // Export to CSV
        let mut csv_buf = Vec::new();
        let csv_receipt = export_storage_table(&storage, table, ExportFormat::Csv, &mut csv_buf)
            .expect("export table to csv");
        let csv_str = String::from_utf8(csv_buf).expect("valid utf-8 csv");
        let csv_verified =
            verify_export_receipt(&csv_str, table).expect("csv receipt verification");
        assert_eq!(csv_verified.row_count, csv_receipt.row_count);
        assert_eq!(csv_verified.checksum_blake3, csv_receipt.checksum_blake3);

        // Export to JSONLines
        let mut jsonl_buf = Vec::new();
        let jsonl_receipt =
            export_storage_table(&storage, table, ExportFormat::JsonLines, &mut jsonl_buf)
                .expect("export table to jsonl");
        let jsonl_str = String::from_utf8(jsonl_buf).expect("valid utf-8 jsonl");
        let jsonl_verified =
            verify_export_receipt(&jsonl_str, table).expect("jsonl receipt verification");
        assert_eq!(jsonl_verified.row_count, jsonl_receipt.row_count);
        assert_eq!(
            jsonl_verified.checksum_blake3,
            jsonl_receipt.checksum_blake3
        );

        // Cross-format row count invariant
        assert_eq!(
            csv_receipt.row_count, jsonl_receipt.row_count,
            "row counts must match between CSV and JSONL for table {:?}",
            table
        );

        // Assert expected table occupancy from 5-step simulation
        match table {
            ExportTable::Run => assert_eq!(csv_receipt.row_count, 1),
            ExportTable::Agent => {
                assert!(csv_receipt.row_count >= 1, "at least one agent persisted")
            }
            ExportTable::Metric => {
                assert!(csv_receipt.row_count >= 1, "at least one metric persisted")
            }
            ExportTable::Lineage | ExportTable::Event => {} // Allowed to be 0 or more
        }

        eprintln!(
            "[bd-2z0.5.6] Table {:<8} -> {} rows, CSV blake3: {}, JSONL blake3: {}",
            table.as_str(),
            csv_receipt.row_count,
            csv_receipt.checksum_blake3,
            jsonl_receipt.checksum_blake3
        );
    }

    storage.close().expect("close storage reader");
    let _ = fs::remove_file(&path);
    eprintln!(
        "[bd-2z0.5.6] End-to-end real DB export verified in {:?}",
        start_time.elapsed()
    );
}
