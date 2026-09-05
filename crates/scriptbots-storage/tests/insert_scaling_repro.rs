//! Standalone reproducer for bd-w1oi's engine lane: does single-statement
//! INSERT cost grow with cumulative table row count in the pinned
//! frankensqlite (`fsqlite =0.1.16`)? Measured on `:memory:` with one indexed
//! table, two modes:
//!
//! - `autocommit`: each INSERT is its own implicit transaction — isolated
//!   committed-state growth.
//! - `bigtx`: one transaction around all INSERTs — within-transaction
//!   accumulation.
//!
//! Run with:
//! ```text
//! cargo test -p scriptbots-storage --test insert_scaling_repro -- --ignored --nocapture
//! ```
//!
//! 2026-09-04 baseline (rch worker, release): see bd-w1oi for the recorded
//! curves. Kept `#[ignore]` because it is a timing diagnostic, not a gate.

use fsqlite::SqliteValue;
use std::time::Instant;

const N: usize = 4000;
const WINDOW: usize = 500;

fn measure_mode(conn: &fsqlite::Connection, mode: &str) {
    conn.execute("DELETE FROM t").expect("delete");
    let mut times: Vec<u128> = Vec::with_capacity(N);
    if mode == "bigtx" {
        conn.execute("BEGIN").expect("begin");
    }
    for i in 0..N {
        let start = Instant::now();
        conn.execute_with_params(
            "INSERT INTO t (id, a, b) VALUES (?1, ?2, ?3)",
            &[
                (i as i64).into(),
                ((i % 97) as i64).into(),
                format!("row-{i}").into(),
            ],
        )
        .expect("insert");
        times.push(start.elapsed().as_micros());
    }
    if mode.starts_with("bigtx") {
        conn.execute("COMMIT").expect("commit");
    }
    println!("mode={mode}");
    for w in (0..N).step_by(WINDOW) {
        let slice = &times[w..(w + WINDOW).min(N)];
        let mean = slice.iter().sum::<u128>() / slice.len() as u128;
        println!("  rows {w:>5}..{}: mean_insert={mean}us", w + slice.len());
    }
}

#[test]
#[ignore = "timing diagnostic for bd-w1oi; run explicitly with --ignored --nocapture"]
fn insert_latency_scaling_repro() {
    let conn = fsqlite::Connection::open(":memory:").expect("open");
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER NOT NULL, b TEXT NOT NULL)")
        .expect("create");
    conn.execute("CREATE INDEX t_a ON t(a)").expect("index");
    // A/B the concurrent-write coordinator (bd-w1oi): if per-insert growth
    // flattens with concurrent_mode=OFF, the linear term lives in the BEGIN
    // CONCURRENT machinery and a single-writer connection can opt out.
    for mode in [
        "autocommit",
        "bigtx-default",
        "bigtx-concurrent-on",
        "bigtx-concurrent-off",
    ] {
        match mode {
            "bigtx-concurrent-on" => {
                conn.execute("PRAGMA fsqlite.concurrent_mode=ON")
                    .expect("pragma on");
            }
            "bigtx-concurrent-off" => {
                conn.execute("PRAGMA fsqlite.concurrent_mode=OFF")
                    .expect("pragma off");
            }
            _ => {}
        }
        measure_mode(&conn, mode);
    }
}
