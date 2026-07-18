//! BEGIN CONCURRENT vs per-island-DB writer benchmark (bd-2z0.8.9.13).
//!
//! Measures the real ScriptBots batch workload shape — one transaction per tick carrying
//! tick_summaries + metrics + events + agents rows plus a per-run progress row (the
//! watermark row whose page is the shared-DB conflict hot spot) — in two topologies:
//!
//! * `per-island`: N writer threads, each with its own database file (today's default:
//!   one writer per file, trivial recovery, zero cross-island conflict handling);
//! * `shared`: N writer threads, each with its own `Connection` on ONE shared database
//!   file. With `concurrent_mode_default` on, every BEGIN auto-promotes to BEGIN
//!   CONCURRENT and disjoint pages commit in parallel; contended pages surface transient
//!   conflicts that the harness counts and retries with the production policy
//!   (transient-only, bounded at 4 attempts, fully rolled back).
//!
//! This harness deliberately bypasses `StoragePipeline` to qualify engine topology
//! behavior. Its insert SQL remains private to this binary and is not a supported
//! application write path. Run it through the pinned DSR profile; results feed the
//! archipelago persistence decision memo in bd-2z0.8.9.13.

use fsqlite::{Connection, FrankenError, compat::RowExt};
use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const MAX_CONFLICT_RETRIES: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    PerIsland,
    Shared,
}

#[derive(Debug)]
struct Config {
    mode: Mode,
    writers: u32,
    batches_per_writer: u32,
    agents_per_batch: u32,
    out_dir: PathBuf,
}

#[derive(Debug, Default)]
struct WriterReport {
    committed_batches: u64,
    conflict_retries: u64,
    conflict_failures: u64,
    hard_failures: u64,
    wall: Duration,
}

fn parse_args() -> Result<Config, String> {
    let mut mode = None;
    let mut writers = None;
    let mut batches = None;
    let mut agents = None;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let Some((flag, value)) = arg.split_once('=') else {
            return Err(format!("expected --flag=value, got {arg:?}"));
        };
        match flag {
            "--mode" => {
                mode = Some(match value {
                    "per-island" => Mode::PerIsland,
                    "shared" => Mode::Shared,
                    other => return Err(format!("unknown mode {other:?}")),
                });
            }
            "--writers" => writers = Some(value.parse::<u32>().map_err(|e| e.to_string())?),
            "--batches" => batches = Some(value.parse::<u32>().map_err(|e| e.to_string())?),
            "--agents-per-batch" => {
                agents = Some(value.parse::<u32>().map_err(|e| e.to_string())?);
            }
            "--out" => out = Some(PathBuf::from(value)),
            other => return Err(format!("unknown flag {other:?}")),
        }
    }
    let mode = mode.ok_or("missing --mode=per-island|shared")?;
    let writers = writers.unwrap_or(4);
    let batches = batches.unwrap_or(200);
    let agents = agents.unwrap_or(16);
    if writers == 0 || batches == 0 || agents == 0 {
        return Err("writers, batches, and agents-per-batch must all be nonzero".to_owned());
    }
    let out = out.unwrap_or_else(|| {
        std::env::temp_dir().join(format!(
            "mvcc-writer-bench-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ))
    });
    Ok(Config {
        mode,
        writers,
        batches_per_writer: batches,
        agents_per_batch: agents,
        out_dir: out,
    })
}

fn create_schema(connection: &Connection, context: &str) {
    connection
        .execute_batch(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, label TEXT NOT NULL);
             CREATE TABLE tick_summaries (
                 run_id TEXT NOT NULL, tick INTEGER NOT NULL,
                 agent_count INTEGER NOT NULL, births INTEGER NOT NULL,
                 deaths INTEGER NOT NULL, total_energy REAL NOT NULL,
                 PRIMARY KEY (run_id, tick)
             );
             CREATE TABLE metrics (
                 run_id TEXT NOT NULL, tick INTEGER NOT NULL,
                 name TEXT NOT NULL, value REAL NOT NULL,
                 PRIMARY KEY (run_id, tick, name)
             );
             CREATE TABLE events (
                 run_id TEXT NOT NULL, tick INTEGER NOT NULL,
                 kind TEXT NOT NULL, count INTEGER NOT NULL CHECK (count >= 0),
                 PRIMARY KEY (run_id, tick, kind)
             );
             CREATE TABLE agents (
                 run_id TEXT NOT NULL, tick INTEGER NOT NULL, agent_uid INTEGER NOT NULL,
                 energy REAL NOT NULL, spike_length REAL NOT NULL,
                 PRIMARY KEY (run_id, tick, agent_uid)
             );
             CREATE TABLE progress (
                 run_id TEXT PRIMARY KEY,
                 committed_batches INTEGER NOT NULL,
                 updated_unix_ms INTEGER NOT NULL
             );",
        )
        .expect("bench schema batch must apply");
}

fn unix_ms() -> i64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    i64::try_from(nanos / 1_000_000).unwrap_or(i64::MAX)
}

/// One production-shaped batch transaction for `run_id` at `tick`. Returns the number of
/// conflict retries consumed (transient-only, bounded, rolled back — the production
/// `should_retry_transaction` policy) or an error after the bound.
fn apply_batch(
    connection: &Connection,
    run_id: &str,
    tick: i64,
    agents_per_batch: u32,
) -> Result<u64, (u64, FrankenError)> {
    let mut attempt = 1_u8;
    let mut conflicts = 0_u64;
    loop {
        let outcome = (|| -> Result<(), FrankenError> {
            connection.begin_transaction()?;
            connection.execute_with_params(
                "INSERT OR REPLACE INTO tick_summaries VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                &[
                    run_id.into(),
                    tick.into(),
                    i64::from(agents_per_batch).into(),
                    1_i64.into(),
                    0_i64.into(),
                    (f64::from(agents_per_batch) * 0.75).into(),
                ],
            )?;
            for metric in ["energy", "population"] {
                connection.execute_with_params(
                    "INSERT OR REPLACE INTO metrics VALUES (?1, ?2, ?3, ?4)",
                    &[
                        run_id.into(),
                        tick.into(),
                        metric.into(),
                        (tick as f64 * 1.5).into(),
                    ],
                )?;
            }
            connection.execute_with_params(
                "INSERT OR REPLACE INTO events VALUES (?1, ?2, ?3, ?4)",
                &[run_id.into(), tick.into(), "births".into(), 1_i64.into()],
            )?;
            for agent in 0..agents_per_batch {
                connection.execute_with_params(
                    "INSERT OR REPLACE INTO agents VALUES (?1, ?2, ?3, ?4, ?5)",
                    &[
                        run_id.into(),
                        tick.into(),
                        i64::from(agent).into(),
                        (f64::from(agent) * 0.5).into(),
                        0.0_f64.into(),
                    ],
                )?;
            }
            connection.execute_with_params(
                "INSERT INTO progress (run_id, committed_batches, updated_unix_ms)
                 VALUES (?1, 1, ?2)
                 ON CONFLICT (run_id) DO UPDATE SET
                     committed_batches = committed_batches + 1,
                     updated_unix_ms = excluded.updated_unix_ms",
                &[run_id.into(), unix_ms().into()],
            )?;
            connection.commit_transaction()
        })();
        match outcome {
            Ok(()) => return Ok(conflicts),
            Err(error) => {
                let _ = connection.rollback_transaction();
                if error.is_transient() && attempt < MAX_CONFLICT_RETRIES {
                    conflicts += 1;
                    attempt += 1;
                    continue;
                }
                return Err((conflicts, error));
            }
        }
    }
}

fn run_writer(config: &Config, writer_index: u32, barrier: Arc<Barrier>) -> WriterReport {
    let mut report = WriterReport::default();
    let (db_path, run_id) = match config.mode {
        Mode::PerIsland => (
            config.out_dir.join(format!("island-{writer_index}.sqlite")),
            "island".to_owned(),
        ),
        Mode::Shared => (
            config.out_dir.join("shared.sqlite"),
            format!("island-{writer_index}"),
        ),
    };
    let db_path_string = db_path.to_string_lossy().to_string();
    let connection = match &config.mode {
        Mode::PerIsland => Connection::open(&db_path_string),
        Mode::Shared => {
            let mut opened = None;
            for attempt in 1..=MAX_CONFLICT_RETRIES {
                match Connection::open(&db_path_string) {
                    Ok(connection) => {
                        opened = Some(connection);
                        break;
                    }
                    Err(error) if error.is_transient() && attempt < MAX_CONFLICT_RETRIES => {
                        thread::sleep(Duration::from_millis(u64::from(attempt)));
                    }
                    Err(error) => {
                        report.hard_failures += 1;
                        eprintln!("shared-mode open failed for writer {writer_index}: {error}");
                        return report;
                    }
                }
            }
            match opened {
                Some(connection) => Ok(connection),
                None => {
                    report.hard_failures += 1;
                    eprintln!(
                        "shared-mode open for writer {writer_index} exhausted the transient bound"
                    );
                    return report;
                }
            }
        }
    }
    .expect("bench writer opens its database");
    if config.mode == Mode::PerIsland {
        create_schema(&connection, &format!("writer {writer_index}"));
        connection
            .execute_with_params(
                "INSERT INTO runs (run_id, label) VALUES (?1, ?2)",
                &[run_id.clone().into(), "bench".into()],
            )
            .expect("bench writer registers its run");
    }

    barrier.wait();
    let started = Instant::now();
    for batch in 0..config.batches_per_writer {
        let tick = i64::from(batch) * 16 + i64::try_from(writer_index).unwrap_or(0) + 1;
        match apply_batch(&connection, &run_id, tick, config.agents_per_batch) {
            Ok(conflicts) => {
                report.committed_batches += 1;
                report.conflict_retries += conflicts;
            }
            Err((conflicts, error)) => {
                report.conflict_retries += conflicts;
                report.conflict_failures += 1;
                eprintln!(
                    "writer {writer_index} batch {batch}: terminal conflict failure after bound: {error}"
                );
                break;
            }
        }
    }
    report.wall = started.elapsed();
    connection
        .close()
        .expect("bench writer closes its connection");
    report
}

fn integrity_check(path: &Path) -> String {
    let connection = Connection::open(path.to_string_lossy().as_ref())
        .expect("integrity reader opens the bench database");
    let result: String = connection
        .query_row("PRAGMA integrity_check")
        .expect("integrity_check runs")
        .get_typed(0)
        .expect("integrity_check result is text");
    connection.close().expect("integrity reader closes");
    result
}

fn main() {
    let config = parse_args().unwrap_or_else(|error| {
        eprintln!("mvcc-writer-bench: {error}");
        std::process::exit(2);
    });
    fs::create_dir_all(&config.out_dir).expect("bench output directory");
    if config.mode == Mode::Shared {
        let shared = config.out_dir.join("shared.sqlite");
        let connection =
            Connection::open(shared.to_string_lossy().as_ref()).expect("shared schema host opens");
        create_schema(&connection, "shared schema host");
        for writer_index in 0..config.writers {
            connection
                .execute_with_params(
                    "INSERT INTO runs (run_id, label) VALUES (?1, ?2)",
                    &[format!("island-{writer_index}").into(), "bench".into()],
                )
                .expect("shared schema host registers run");
        }
        connection.close().expect("shared schema host closes");
    }

    let barrier = Arc::new(Barrier::new(usize::try_from(config.writers).unwrap_or(1)));
    let bench_started = Instant::now();
    let mut handles = Vec::new();
    for writer_index in 0..config.writers {
        let barrier = Arc::clone(&barrier);
        let config_ref = Config {
            mode: config.mode,
            writers: config.writers,
            batches_per_writer: config.batches_per_writer,
            agents_per_batch: config.agents_per_batch,
            out_dir: config.out_dir.clone(),
        };
        handles.push(thread::spawn(move || {
            run_writer(&config_ref, writer_index, barrier)
        }));
    }
    let mut reports = Vec::new();
    for handle in handles {
        reports.push(handle.join().expect("writer thread joins"));
    }
    let wall = bench_started.elapsed();

    let committed: u64 = reports.iter().map(|report| report.committed_batches).sum();
    let retries: u64 = reports.iter().map(|report| report.conflict_retries).sum();
    let failures: u64 = reports.iter().map(|report| report.conflict_failures).sum();
    let slowest = reports
        .iter()
        .map(|report| report.wall)
        .max()
        .unwrap_or_default();
    let mode_label = match config.mode {
        Mode::PerIsland => "per-island",
        Mode::Shared => "shared",
    };
    let integrity = if config.mode == Mode::Shared {
        integrity_check(&config.out_dir.join("shared.sqlite"))
    } else {
        "n/a (per-file)".to_owned()
    };
    let per_island_integrity: Vec<String> = if config.mode == Mode::PerIsland {
        (0..config.writers)
            .map(|writer_index| {
                integrity_check(&config.out_dir.join(format!("island-{writer_index}.sqlite")))
            })
            .collect()
    } else {
        Vec::new()
    };

    let throughput = committed as f64 / wall.as_secs_f64().max(1e-9);
    let report_json = format!(
        "{{\n  \"mode\": \"{mode_label}\",
  \"writers\": {},
  \"batches_per_writer\": {},
  \"agents_per_batch\": {},
  \"committed_batches\": {committed},
  \"wall_ms\": {},
  \"slowest_writer_ms\": {},
  \"batches_per_second\": {:.1},
  \"conflict_retries\": {retries},
  \"conflict_failures\": {failures},
  \"integrity\": \"{integrity}\",
  \"per_island_integrity\": {per_island_integrity:?}\n}}",
        config.writers,
        config.batches_per_writer,
        config.agents_per_batch,
        wall.as_millis(),
        slowest.as_millis(),
        throughput
    );
    println!("{report_json}");
    let report_path = config.out_dir.join("bench_report.json");
    fs::write(&report_path, &report_json).expect("write bench report");

    if failures > 0 {
        eprintln!("mvcc-writer-bench: {failures} conflict failures exhausted the retry bound");
        std::process::exit(1);
    }
    if integrity != "ok" && config.mode == Mode::Shared {
        eprintln!("mvcc-writer-bench: shared database integrity_check = {integrity}");
        std::process::exit(1);
    }
    if per_island_integrity.iter().any(|result| result != "ok") {
        eprintln!("mvcc-writer-bench: per-island integrity failure: {per_island_integrity:?}");
        std::process::exit(1);
    }
}
