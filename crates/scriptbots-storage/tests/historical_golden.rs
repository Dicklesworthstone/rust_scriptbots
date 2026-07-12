use fsqlite::{Connection, FrankenError, compat::RowExt};
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
        "{prefix}-{}-{}.sqlite",
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
            max_age: 0,
            spike_hits: 0,
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
        births: Vec::new(),
        deaths: Vec::new(),
        replay_events: Vec::new(),
    }
}

#[test]
fn golden_population_and_kill_queries_match_expectations() -> Result<(), Box<dyn std::error::Error>>
{
    let path = temp_db_path("storage-golden");
    let path_str = path.to_string_lossy().to_string();
    let mut storage = Storage::create_new_file_with_thresholds(&path_str, 1, 1, 1, 1)?;

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

    let connection = Connection::open(&path_str)?;

    let tick_rows = connection.query(
        "select tick, agent_count, births, deaths
         from ticks
         order by tick asc",
    )?;
    let expected_ticks = vec![(1_i64, 3_i64, 1_i64, 0_i64), (2, 4, 2, 1), (3, 5, 1, 0)];
    let actual_ticks = tick_rows
        .iter()
        .map(|row| -> Result<_, FrankenError> {
            Ok((
                row.get_typed::<i64>(0)?,
                row.get_typed::<i64>(1)?,
                row.get_typed::<i64>(2)?,
                row.get_typed::<i64>(3)?,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(actual_ticks, expected_ticks);

    let event_rows = connection.query(
        "select tick, kind, count
         from events
         order by tick asc, kind asc",
    )?;
    let expected_events = vec![
        (1_i64, "births".to_string(), 1_i64),
        (2, "births".to_string(), 2_i64),
        (2, "deaths".to_string(), 1_i64),
        (3, "births".to_string(), 1_i64),
    ];
    let actual_events = event_rows
        .iter()
        .map(|row| -> Result<_, FrankenError> {
            Ok((
                row.get_typed::<i64>(0)?,
                row.get_typed::<String>(1)?,
                row.get_typed::<i64>(2)?,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(actual_events, expected_events);

    connection.close()?;
    let _ = fs::remove_file(path);
    Ok(())
}
