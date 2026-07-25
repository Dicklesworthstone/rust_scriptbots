use fsqlite::{Connection, FrankenError, compat::RowExt};
use scriptbots_core::{
    AgentData, AgentIdentity, AgentRuntime, AgentState, AgentUid, BirthOrigin, BirthRecord,
    CombatEventFlags, DeathCause, DeathRecord, Generation, MetricSample, PersistenceBatch,
    PersistenceEvent, PersistenceEventKind, Position, Tick, TickSummary,
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

fn make_agent_state(
    uid: u64,
    generation: u32,
    parents: Option<(u64, u64)>,
    energy: f32,
    position: (f32, f32),
) -> AgentState {
    let data = AgentData {
        position: Position::new(position.0, position.1),
        health: energy,
        generation: Generation(generation),
        ..AgentData::default()
    };
    let (lineage, hybrid) = parents.map_or(([None, None], false), |(parent_a, parent_b)| {
        ([Some(AgentUid(parent_a)), Some(AgentUid(parent_b))], true)
    });
    let runtime = AgentRuntime {
        energy,
        hybrid,
        lineage,
        ..AgentRuntime::default()
    };
    AgentState {
        id: scriptbots_core::AgentId::default(),
        identity: AgentIdentity {
            uid: AgentUid(uid),
            spawn_ordinal: uid - 1,
            birth_ordinal: uid.checked_sub(3),
        },
        data,
        runtime,
    }
}

fn make_seeded(tick: u64, uid: u64) -> BirthRecord {
    BirthRecord {
        tick: Tick(tick),
        agent_uid: AgentUid(uid),
        spawn_ordinal: uid.saturating_sub(1),
        birth_ordinal: None,
        origin: BirthOrigin::Seeded,
        parent_a: None,
        parent_b: None,
        brain_kind: None,
        brain_key: None,
        herbivore_tendency: 0.5,
        generation: Generation::default(),
        position: Position::new(0.0, 0.0),
        is_hybrid: false,
    }
}

fn make_born(
    tick: u64,
    uid: u64,
    birth_ordinal: u64,
    parent_a: u64,
    parent_b: u64,
    generation: u32,
) -> BirthRecord {
    BirthRecord {
        tick: Tick(tick),
        agent_uid: AgentUid(uid),
        spawn_ordinal: uid.saturating_sub(1),
        birth_ordinal: Some(birth_ordinal),
        origin: BirthOrigin::Born,
        parent_a: Some(AgentUid(parent_a)),
        parent_b: Some(AgentUid(parent_b)),
        brain_kind: None,
        brain_key: None,
        herbivore_tendency: 0.5,
        generation: Generation(generation),
        position: Position::new(0.0, 0.0),
        is_hybrid: true,
    }
}

fn make_death(tick: u64, uid: u64) -> DeathRecord {
    DeathRecord {
        tick: Tick(tick),
        agent_uid: AgentUid(uid),
        age: 2,
        generation: Generation::default(),
        herbivore_tendency: 0.5,
        brain_kind: None,
        brain_key: None,
        energy: 0.0,
        food_balance_total: 0.0,
        cause: DeathCause::Unknown,
        was_hybrid: false,
        combat_flags: CombatEventFlags::default(),
    }
}

fn make_batch(
    tick: u64,
    agent_count: usize,
    total_energy: f32,
    agents: Vec<AgentState>,
    events: Vec<PersistenceEvent>,
    birth_records: Vec<BirthRecord>,
    death_records: Vec<DeathRecord>,
) -> PersistenceBatch {
    let births = birth_records
        .iter()
        .filter(|record| record.origin == BirthOrigin::Born)
        .count();
    let deaths = death_records.len();
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
        births: birth_records,
        deaths: death_records,
        replay_events: Vec::new(),
        narrative_events: Vec::new(),
    }
}

#[test]
fn golden_population_and_kill_queries_match_expectations() -> Result<(), Box<dyn std::error::Error>>
{
    let path = temp_db_path("storage-golden");
    let path_str = path.to_string_lossy().to_string();
    let mut storage = Storage::create_unattributed_file_with_thresholds(&path_str, 1, 1, 1, 1)?;

    let batches = vec![
        make_batch(
            1,
            3,
            3.6,
            vec![
                make_agent_state(1, 0, None, 1.0, (10.0, 10.0)),
                make_agent_state(2, 0, None, 1.2, (15.0, 12.0)),
                make_agent_state(3, 1, Some((1, 2)), 1.4, (20.0, 18.0)),
            ],
            vec![PersistenceEvent::new(PersistenceEventKind::Births, 1)],
            vec![
                make_seeded(0, 1),
                make_seeded(0, 2),
                make_born(1, 3, 0, 1, 2, 1),
            ],
            Vec::new(),
        ),
        make_batch(
            2,
            4,
            4.8,
            vec![
                make_agent_state(1, 0, None, 1.3, (11.0, 11.0)),
                make_agent_state(3, 1, Some((1, 2)), 1.1, (14.0, 16.0)),
                make_agent_state(4, 2, Some((1, 3)), 1.0, (21.0, 19.0)),
                make_agent_state(5, 2, Some((2, 3)), 1.4, (24.0, 22.0)),
            ],
            vec![
                PersistenceEvent::new(PersistenceEventKind::Births, 2),
                PersistenceEvent::new(PersistenceEventKind::Deaths, 1),
            ],
            vec![make_born(2, 4, 1, 1, 3, 2), make_born(2, 5, 2, 2, 3, 2)],
            vec![make_death(2, 2)],
        ),
        make_batch(
            3,
            5,
            6.5,
            vec![
                make_agent_state(1, 0, None, 1.3, (13.0, 11.0)),
                make_agent_state(3, 1, Some((1, 2)), 1.5, (17.0, 16.0)),
                make_agent_state(4, 2, Some((1, 3)), 1.2, (23.0, 21.0)),
                make_agent_state(5, 2, Some((2, 3)), 1.0, (25.0, 24.0)),
                make_agent_state(6, 3, Some((3, 4)), 1.5, (28.0, 26.0)),
            ],
            vec![PersistenceEvent::new(PersistenceEventKind::Births, 1)],
            vec![make_born(3, 6, 3, 3, 4, 3)],
            Vec::new(),
        ),
    ];

    for batch in &batches {
        storage.persist(batch)?;
    }
    storage.flush()?;
    let run_id = storage.run_id().to_string();

    drop(storage);

    let connection = Connection::open(&path_str)?;

    let tick_rows = connection.query_with_params(
        "select tick, agent_count, births, deaths
         from tick_summaries
         where run_id = ?1
         order by tick asc",
        &[run_id.as_str().into()],
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

    let event_rows = connection.query_with_params(
        "select tick, kind, count
         from events
         where run_id = ?1
         order by tick asc, kind asc",
        &[run_id.as_str().into()],
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
