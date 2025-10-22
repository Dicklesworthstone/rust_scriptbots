use scriptbots_core::{
    AgentData, AgentRuntime, AgentState, MetricSample, PersistenceBatch, PersistenceEvent,
    PersistenceEventKind, Position, Tick, TickSummary,
};
use scriptbots_storage::Storage;
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn temp_db_path(prefix: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    path.push(format!(
        "{prefix}-{}-{}.duckdb",
        std::process::id(),
        timestamp
    ));
    path
}

fn make_agent_state(energy: f32, position: (f32, f32)) -> AgentState {
    let data = AgentData {
        position: Position::new(position.0, position.1),
        health: energy,
        ..AgentData::default()
    };
    let runtime = AgentRuntime {
        energy,
        ..AgentRuntime::default()
    };
    AgentState {
        id: scriptbots_core::AgentId::default(),
        data,
        runtime,
    }
}

fn make_batch(
    tick: u64,
    agent_count: usize,
    births: usize,
    deaths: usize,
    total_energy: f32,
    agents: Vec<AgentState>,
    events: Vec<PersistenceEvent>,
) -> PersistenceBatch {
    let average_energy = if agent_count > 0 {
        total_energy / agent_count as f32
    } else {
        0.0
    };
    PersistenceBatch {
        summary: TickSummary {
            tick: Tick(tick),
            agent_count,
            births,
            deaths,
            total_energy,
            average_energy,
            average_health: 0.75,
        },
        epoch: 0,
        closed: false,
        metrics: vec![
            MetricSample::from_f32("population", agent_count as f32),
            MetricSample::from_f32("births", births as f32),
            MetricSample::from_f32("deaths", deaths as f32),
        ],
        events,
        agents,
    }
}

#[test]
fn golden_population_and_kill_queries_match_expectations() -> Result<(), Box<dyn std::error::Error>>
{
    let path = temp_db_path("storage-golden");
    let path_str = path.to_string_lossy().to_string();
    let mut storage = Storage::with_thresholds(&path_str, 1, 1, 1, 1)?;

    let batches = vec![
        make_batch(
            1,
            3,
            1,
            0,
            3.6,
            vec![
                make_agent_state(1.0, (10.0, 10.0)),
                make_agent_state(1.2, (15.0, 12.0)),
                make_agent_state(1.4, (20.0, 18.0)),
            ],
            vec![PersistenceEvent::new(PersistenceEventKind::Births, 1)],
        ),
        make_batch(
            2,
            4,
            2,
            1,
            4.8,
            vec![
                make_agent_state(1.3, (11.0, 11.0)),
                make_agent_state(1.1, (14.0, 16.0)),
                make_agent_state(1.0, (21.0, 19.0)),
                make_agent_state(1.4, (24.0, 22.0)),
            ],
            vec![
                PersistenceEvent::new(PersistenceEventKind::Births, 2),
                PersistenceEvent::new(PersistenceEventKind::Deaths, 1),
            ],
        ),
        make_batch(
            3,
            5,
            1,
            0,
            6.5,
            vec![
                make_agent_state(1.3, (13.0, 11.0)),
                make_agent_state(1.5, (17.0, 16.0)),
                make_agent_state(1.2, (23.0, 21.0)),
                make_agent_state(1.0, (25.0, 24.0)),
                make_agent_state(1.5, (28.0, 26.0)),
            ],
            vec![PersistenceEvent::new(PersistenceEventKind::Births, 1)],
        ),
    ];

    for batch in &batches {
        storage.persist(batch)?;
    }
    storage.flush()?;

    drop(storage);

    let connection = duckdb::Connection::open(&path_str)?;

    let mut tick_stmt = connection.prepare(
        "select tick, agent_count, births, deaths
         from ticks
         order by tick asc",
    )?;
    let mut tick_rows = tick_stmt.query([])?;
    let expected_ticks = vec![(1_i64, 3_i64, 1_i64, 0_i64), (2, 4, 2, 1), (3, 5, 1, 0)];
    for expected in expected_ticks {
        let row = tick_rows.next()?.expect("expected tick row");
        assert_eq!(row.get::<_, i64>(0)?, expected.0);
        assert_eq!(row.get::<_, i64>(1)?, expected.1);
        assert_eq!(row.get::<_, i64>(2)?, expected.2);
        assert_eq!(row.get::<_, i64>(3)?, expected.3);
    }
    assert!(tick_rows.next()?.is_none(), "unexpected extra tick rows");

    let mut event_stmt = connection.prepare(
        "select tick, kind, count
         from events
         order by tick asc, kind asc",
    )?;
    let mut event_rows = event_stmt.query([])?;
    let expected_events = vec![
        (1_i64, "births".to_string(), 1_i64),
        (2, "births".to_string(), 2_i64),
        (2, "deaths".to_string(), 1_i64),
        (3, "births".to_string(), 1_i64),
    ];
    for expected in expected_events {
        let row = event_rows.next()?.expect("expected event row");
        assert_eq!(row.get::<_, i64>(0)?, expected.0);
        assert_eq!(row.get::<_, String>(1)?, expected.1);
        assert_eq!(row.get::<_, i64>(2)?, expected.2);
    }
    assert!(event_rows.next()?.is_none(), "unexpected extra event rows");

    drop(connection);
    let _ = fs::remove_file(path);
    Ok(())
}
