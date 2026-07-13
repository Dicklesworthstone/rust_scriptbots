use fsqlite::{Connection, SqliteValue, compat::RowExt};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

const FIRST_TICK: i64 = 41;
const SECOND_TICK: i64 = 42;
const ROLLED_BACK_TICK: i64 = 9_001;

const SCHEMA_SQL: &str = r#"
    CREATE TABLE ticks (
        tick INTEGER PRIMARY KEY,
        epoch INTEGER,
        closed INTEGER,
        agent_count INTEGER,
        births INTEGER,
        deaths INTEGER,
        total_energy REAL,
        average_energy REAL,
        average_health REAL
    );
    CREATE TABLE metrics (
        tick INTEGER,
        name TEXT,
        value REAL,
        PRIMARY KEY (tick, name)
    );
    CREATE TABLE events (
        tick INTEGER,
        kind TEXT,
        count INTEGER,
        PRIMARY KEY (tick, kind)
    );
    CREATE TABLE replay_events (
        tick INTEGER,
        seq INTEGER,
        agent_uid INTEGER,
        scope TEXT,
        event_type TEXT,
        payload TEXT,
        PRIMARY KEY (tick, seq)
    );
    CREATE TABLE agents (
        tick INTEGER,
        agent_uid INTEGER,
        generation INTEGER,
        age INTEGER,
        position_x REAL,
        position_y REAL,
        velocity_x REAL,
        velocity_y REAL,
        heading REAL,
        health REAL,
        energy REAL,
        color_r REAL,
        color_g REAL,
        color_b REAL,
        spike_length REAL,
        boost INTEGER,
        herbivore_tendency REAL,
        sound_multiplier REAL,
        reproduction_counter REAL,
        mutation_rate_primary REAL,
        mutation_rate_secondary REAL,
        trait_smell REAL,
        trait_sound REAL,
        trait_hearing REAL,
        trait_eye REAL,
        trait_blood REAL,
        give_intent REAL,
        brain_binding TEXT,
        brain_key INTEGER,
        food_delta REAL,
        spiked INTEGER,
        hybrid INTEGER,
        sound_output REAL,
        spike_attacker INTEGER,
        spike_victim INTEGER,
        hit_carnivore INTEGER,
        hit_herbivore INTEGER,
        hit_by_carnivore INTEGER,
        hit_by_herbivore INTEGER,
        PRIMARY KEY (tick, agent_uid)
    );
    CREATE TABLE births (
        tick INTEGER,
        agent_uid INTEGER,
        spawn_ordinal INTEGER,
        birth_ordinal INTEGER,
        parent_a INTEGER,
        parent_b INTEGER,
        brain_kind TEXT,
        brain_key INTEGER,
        herbivore_tendency REAL,
        generation INTEGER,
        position_x REAL,
        position_y REAL,
        is_hybrid INTEGER,
        PRIMARY KEY (tick, agent_uid)
    );
    CREATE TABLE deaths (
        tick INTEGER,
        agent_uid INTEGER,
        age INTEGER,
        generation INTEGER,
        herbivore_tendency REAL,
        brain_kind TEXT,
        brain_key INTEGER,
        energy REAL,
        food_balance_total REAL,
        cause TEXT,
        was_hybrid INTEGER,
        spike_attacker INTEGER,
        spike_victim INTEGER,
        hit_carnivore INTEGER,
        hit_herbivore INTEGER,
        hit_by_carnivore INTEGER,
        hit_by_herbivore INTEGER,
        PRIMARY KEY (tick, agent_uid)
    );
"#;

const AGENT_INSERT_SQL: &str = r#"
    INSERT OR REPLACE INTO agents (
        tick, agent_uid, generation, age,
        position_x, position_y, velocity_x, velocity_y, heading,
        health, energy, color_r, color_g, color_b, spike_length, boost,
        herbivore_tendency, sound_multiplier, reproduction_counter,
        mutation_rate_primary, mutation_rate_secondary,
        trait_smell, trait_sound, trait_hearing, trait_eye, trait_blood,
        give_intent, brain_binding, brain_key, food_delta, spiked, hybrid,
        sound_output, spike_attacker, spike_victim, hit_carnivore,
        hit_herbivore, hit_by_carnivore, hit_by_herbivore
    ) VALUES (
        ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
        ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20,
        ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30,
        ?31, ?32, ?33, ?34, ?35, ?36, ?37, ?38, ?39
    )
"#;

fn agent_values(tick: i64, agent_uid: i64) -> [SqliteValue; 39] {
    [
        tick.into(),
        agent_uid.into(),
        3_i64.into(),
        17_i64.into(),
        12.25_f64.into(),
        18.75_f64.into(),
        0.5_f64.into(),
        (-0.25_f64).into(),
        1.5_f64.into(),
        0.9_f64.into(),
        14.5_f64.into(),
        0.2_f64.into(),
        0.4_f64.into(),
        0.8_f64.into(),
        0.3_f64.into(),
        1_i64.into(),
        0.65_f64.into(),
        0.75_f64.into(),
        4.0_f64.into(),
        0.02_f64.into(),
        0.01_f64.into(),
        0.11_f64.into(),
        0.22_f64.into(),
        0.33_f64.into(),
        0.44_f64.into(),
        0.55_f64.into(),
        0.66_f64.into(),
        "mlp".into(),
        SqliteValue::Null,
        (-0.5_f64).into(),
        1_i64.into(),
        0_i64.into(),
        0.42_f64.into(),
        1_i64.into(),
        0_i64.into(),
        1_i64.into(),
        0_i64.into(),
        1_i64.into(),
        0_i64.into(),
    ]
}

fn create_schema(connection: &Connection) {
    connection
        .execute_batch(SCHEMA_SQL)
        .expect("FrankenSQLite should create all seven ScriptBots tables");
}

fn insert_table_group(connection: &Connection, tick: i64, agent_uid: i64) {
    connection
        .execute_with_params(
            "INSERT OR REPLACE INTO ticks VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            &[
                tick.into(),
                2_i64.into(),
                0_i64.into(),
                3_i64.into(),
                1_i64.into(),
                1_i64.into(),
                14.5_f64.into(),
                4.5_f64.into(),
                0.85_f64.into(),
            ],
        )
        .expect("ticks insert should accept nine numbered parameters");
    connection
        .execute_with_params(
            "INSERT OR REPLACE INTO metrics VALUES (?1, ?2, ?3)",
            &[tick.into(), "energy".into(), 14.5_f64.into()],
        )
        .expect("metrics insert should accept a REAL value");
    connection
        .execute_with_params(
            "INSERT OR REPLACE INTO events VALUES (?1, ?2, ?3)",
            &[tick.into(), "births".into(), 1_i64.into()],
        )
        .expect("events insert should preserve an integer count");
    connection
        .execute_with_params(
            "INSERT OR REPLACE INTO replay_events VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            &[
                tick.into(),
                0_i64.into(),
                SqliteValue::Null,
                "world".into(),
                "agent_action".into(),
                r#"{"action":"move","distance":1.25}"#.into(),
            ],
        )
        .expect("replay insert should preserve nullable ids and JSON encoded as TEXT");
    connection
        .execute_with_params(AGENT_INSERT_SQL, &agent_values(tick, agent_uid))
        .expect("agent insert should bind all 39 numbered parameters");
    connection
        .execute_with_params(
            "INSERT OR REPLACE INTO births VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            &[
                tick.into(),
                agent_uid.into(),
                (agent_uid - 1).into(),
                (agent_uid - 1).into(),
                (agent_uid - 1).into(),
                SqliteValue::Null,
                "mlp".into(),
                SqliteValue::Null,
                0.65_f64.into(),
                3_i64.into(),
                12.25_f64.into(),
                18.75_f64.into(),
                0_i64.into(),
            ],
        )
        .expect("birth insert should preserve nullable parent and brain keys");
    connection
        .execute_with_params(
            "INSERT OR REPLACE INTO deaths VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            &[
                tick.into(),
                agent_uid.into(),
                17_i64.into(),
                3_i64.into(),
                0.65_f64.into(),
                SqliteValue::Null,
                SqliteValue::Null,
                0.0_f64.into(),
                (-1.25_f64).into(),
                "starvation".into(),
                0_i64.into(),
                1_i64.into(),
                0_i64.into(),
                1_i64.into(),
                0_i64.into(),
                1_i64.into(),
                0_i64.into(),
            ],
        )
        .expect("death insert should preserve nullable brain metadata");
}

fn insert_committed_workload(connection: &Connection) {
    connection
        .begin_transaction()
        .expect("committed workload transaction should begin");
    insert_table_group(connection, FIRST_TICK, 7);
    insert_table_group(connection, SECOND_TICK, 8);

    for (tick, population) in [(FIRST_TICK, 3.0_f64), (SECOND_TICK, 4.0_f64)] {
        connection
            .execute_with_params(
                "INSERT OR REPLACE INTO metrics VALUES (?1, ?2, ?3)",
                &[tick.into(), "population".into(), population.into()],
            )
            .expect("population metric insert should succeed");
    }

    connection
        .execute_with_params(
            "INSERT OR REPLACE INTO ticks VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            &[
                FIRST_TICK.into(),
                2_i64.into(),
                1_i64.into(),
                5_i64.into(),
                2_i64.into(),
                1_i64.into(),
                20.0_f64.into(),
                4.0_f64.into(),
                0.9_f64.into(),
            ],
        )
        .expect("INSERT OR REPLACE should update the existing tick snapshot");
    connection
        .commit_transaction()
        .expect("committed workload transaction should commit");
}

fn prove_failed_transaction_rolls_back_every_table(connection: &Connection) {
    connection
        .begin_transaction()
        .expect("rollback workload transaction should begin");
    insert_table_group(connection, ROLLED_BACK_TICK, 99);

    let expected_failure = connection.execute_with_params(
        "INSERT INTO ticks (tick, missing_column) VALUES (?1, ?2)",
        &[ROLLED_BACK_TICK.into(), 1_i64.into()],
    );
    assert!(
        expected_failure.is_err(),
        "the deliberately invalid statement must fail before rollback"
    );
    connection
        .rollback_transaction()
        .expect("the failed transaction should remain explicitly rollbackable");

    for table in [
        "ticks",
        "metrics",
        "events",
        "replay_events",
        "agents",
        "births",
        "deaths",
    ] {
        let sql = format!("SELECT COUNT(*) FROM {table} WHERE tick = ?1");
        let row = connection
            .query_row_with_params(&sql, &[ROLLED_BACK_TICK.into()])
            .expect("rollback count query should succeed");
        let count = row
            .get_typed::<i64>(0)
            .expect("rollback count should be an integer");
        assert_eq!(count, 0, "rollback left partial rows in {table}");
    }
}

fn verify_tick_queries(connection: &Connection) {
    let tick_rows = connection
        .query_with_params(
            "SELECT tick, closed, agent_count FROM ticks ORDER BY tick DESC LIMIT ?1",
            &[2_i64.into()],
        )
        .expect("ordered tick query with a bound LIMIT should succeed");
    assert_eq!(tick_rows.len(), 2, "bound LIMIT should return two ticks");
    assert_eq!(
        tick_rows[0]
            .get_typed::<i64>(0)
            .expect("newest tick should be an integer"),
        SECOND_TICK,
        "ticks should be ordered newest first"
    );
    assert_eq!(
        tick_rows[1]
            .get_typed::<i64>(2)
            .expect("replacement agent_count should be an integer"),
        5,
        "INSERT OR REPLACE should expose the replacement tick row"
    );
    assert!(
        tick_rows[1]
            .get_typed::<bool>(1)
            .expect("replacement closed flag should be boolean-compatible"),
        "replacement tick should be marked closed"
    );

    let max_tick = connection
        .query_row("SELECT MAX(tick) FROM ticks")
        .expect("MAX(tick) should return exactly one aggregate row")
        .get_typed::<i64>(0)
        .expect("MAX(tick) should decode as an integer");
    assert_eq!(
        max_tick, SECOND_TICK,
        "MAX(tick) should identify the newest committed snapshot"
    );
}

fn assert_float(actual: f64, expected: f64, context: &str) {
    assert!(
        (actual - expected).abs() < f64::EPSILON,
        "{context}: expected {expected}, got {actual}"
    );
}

fn verify_metric_aggregates(connection: &Connection) {
    let aggregate_rows = connection
        .query_with_params(
            "SELECT name, COUNT(*), SUM(value), AVG(value) \
             FROM metrics WHERE tick >= ?1 \
             GROUP BY name ORDER BY name ASC LIMIT ?2",
            &[FIRST_TICK.into(), 2_i64.into()],
        )
        .expect("grouped aggregate query with a bound LIMIT should succeed");
    assert_eq!(
        aggregate_rows.len(),
        2,
        "both metric groups should be returned"
    );
    assert_eq!(
        aggregate_rows[0]
            .get_typed::<String>(0)
            .expect("first aggregate group name should be TEXT"),
        "energy"
    );
    assert_eq!(
        aggregate_rows[0]
            .get_typed::<i64>(1)
            .expect("energy aggregate count should be INTEGER"),
        2
    );
    assert_float(
        aggregate_rows[0]
            .get_typed::<f64>(2)
            .expect("energy aggregate sum should be REAL"),
        29.0,
        "energy SUM(value)",
    );
    assert_float(
        aggregate_rows[0]
            .get_typed::<f64>(3)
            .expect("energy aggregate average should be REAL"),
        14.5,
        "energy AVG(value)",
    );
    assert_eq!(
        aggregate_rows[1]
            .get_typed::<String>(0)
            .expect("second aggregate group name should be TEXT"),
        "population"
    );
    assert_float(
        aggregate_rows[1]
            .get_typed::<f64>(2)
            .expect("population aggregate sum should be REAL"),
        7.0,
        "population SUM(value)",
    );
}

fn verify_nullable_payloads_and_agent_row(connection: &Connection) {
    let replay = connection
        .query_row_with_params(
            "SELECT agent_uid, payload FROM replay_events WHERE tick = ?1 AND seq = ?2",
            &[FIRST_TICK.into(), 0_i64.into()],
        )
        .expect("replay row should be queryable with numbered parameters");
    assert_eq!(
        replay
            .get_typed::<Option<i64>>(0)
            .expect("nullable replay agent uid should decode"),
        None
    );
    assert_eq!(
        replay
            .get_typed::<String>(1)
            .expect("replay JSON payload should decode from TEXT"),
        r#"{"action":"move","distance":1.25}"#
    );

    let agent = connection
        .query_row_with_params(
            "SELECT brain_binding, brain_key, spiked, sound_output, hit_by_herbivore \
             FROM agents WHERE tick = ?1 AND agent_uid = ?2",
            &[FIRST_TICK.into(), 7_i64.into()],
        )
        .expect("39-column agent row should be queryable by its composite key");
    assert_eq!(
        agent
            .get_typed::<String>(0)
            .expect("agent brain binding should be TEXT"),
        "mlp"
    );
    assert_eq!(
        agent
            .get_typed::<Option<i64>>(1)
            .expect("nullable agent brain key should decode"),
        None
    );
    assert!(
        agent
            .get_typed::<bool>(2)
            .expect("agent spiked flag should be boolean-compatible")
    );
    assert_float(
        agent
            .get_typed::<f64>(3)
            .expect("agent sound output should be REAL"),
        0.42,
        "agent sound output",
    );
    assert!(
        !agent
            .get_typed::<bool>(4)
            .expect("agent hit_by_herbivore flag should be boolean-compatible")
    );
}

fn verify_integrity(connection: &Connection) {
    let integrity = connection
        .query_row("PRAGMA integrity_check")
        .expect("PRAGMA integrity_check should return one row")
        .get_typed::<String>(0)
        .expect("PRAGMA integrity_check result should be TEXT");
    assert_eq!(integrity, "ok", "FrankenSQLite integrity check failed");
}

fn verify_committed_workload(connection: &Connection) {
    verify_tick_queries(connection);
    verify_metric_aggregates(connection);
    verify_nullable_payloads_and_agent_row(connection);
    verify_integrity(connection);
}

fn exercise_workload(connection: &Connection) {
    create_schema(connection);
    insert_committed_workload(connection);
    prove_failed_transaction_rolls_back_every_table(connection);
    verify_committed_workload(connection);
}

fn file_backed_test_path() -> PathBuf {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"));
    fs::create_dir_all(&target_dir).expect("Cargo target directory should be creatable");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after the Unix epoch")
        .as_nanos();
    target_dir.join(format!(
        "scriptbots-fsqlite-conformance-{}-{nonce}.sqlite3",
        std::process::id()
    ))
}

#[test]
fn frankensqlite_in_memory_matches_scriptbots_storage_workload() {
    let connection = Connection::open(":memory:")
        .expect("FrankenSQLite should open an in-memory ScriptBots database");
    exercise_workload(&connection);
    connection
        .close()
        .expect("in-memory FrankenSQLite connection should close cleanly");
}

#[test]
fn frankensqlite_file_backed_round_trips_scriptbots_storage_workload() {
    let path = file_backed_test_path();
    let path = path
        .to_str()
        .expect("Cargo target path should be valid UTF-8");

    let connection = Connection::open(path)
        .expect("FrankenSQLite should open a file-backed ScriptBots database");
    exercise_workload(&connection);
    connection
        .close()
        .expect("file-backed connection should checkpoint and close cleanly");

    let reopened = Connection::open(path)
        .expect("FrankenSQLite should reopen the committed ScriptBots database");
    verify_committed_workload(&reopened);
    reopened
        .close()
        .expect("reopened FrankenSQLite connection should close cleanly");
}
