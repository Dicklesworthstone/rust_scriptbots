//! Proofs for the fail-closed hard-bounded SQL refusal (`bd-91lr`).

use scriptbots_storage::{StorageError, async_lane::AsyncReadLane};
use std::{
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

fn test_path(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "scriptbots-async-lane-{tag}-{}-{nanos}.sqlite",
        std::process::id()
    ))
}

#[test]
fn refusal_is_typed_and_records_the_exact_pin() {
    let requested = Duration::from_millis(17);
    let error = match AsyncReadLane::open("this path is deliberately ignored", requested) {
        Ok(_) => panic!("a hard deadline must be refused, not advertised as enforced"),
        Err(error) => error,
    };

    match error {
        StorageError::ReadExecutionBoundUnavailable {
            operation,
            requested,
            engine_version,
            engine_revision,
        } => {
            assert_eq!(operation, "async_read_lane.open");
            assert_eq!(requested, Duration::from_millis(17));
            assert_eq!(engine_version, "=0.1.16");
            assert_eq!(engine_revision, "e536d7f8ca102b3eb5236bef48514582379f9346");
        }
        other => panic!("expected typed hard-bound refusal, got {other:?}"),
    }
}

#[test]
fn refusal_does_not_create_the_database_path() {
    let path = test_path("deadline-refusal");
    let path_string = path.to_string_lossy().to_string();
    assert!(!path.exists(), "test starts without a database path");

    let error = AsyncReadLane::open(&path_string, Duration::from_millis(1))
        .expect_err("hard-bounded SQL must be refused");
    assert!(
        matches!(
            &error,
            StorageError::ReadExecutionBoundUnavailable {
                operation: "async_read_lane.open",
                ..
            }
        ),
        "path-safe refusal remains typed: {error}"
    );
    assert!(
        !path.exists(),
        "refusal must happen before opening or creating the database"
    );
}

#[test]
fn refusal_names_safe_caller_alternatives() {
    let error = AsyncReadLane::open("ignored", Duration::from_secs(1))
        .expect_err("hard-bounded SQL must be refused");
    let message = error.to_string();
    assert!(
        message.contains("AnalyticsSnapshotProvider::snapshot()"),
        "refusal names the lock-free frontend alternative: {message}"
    );
    assert!(
        message.contains("StorageReader"),
        "refusal names the explicitly unbounded offline/reporting alternative: {message}"
    );
}
