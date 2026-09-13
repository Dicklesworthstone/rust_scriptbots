//! Fail-closed refusal surface for hard-bounded SQL reads (`bd-91lr`).
//!
//! Updating the pinned engine does not establish a hard wall-clock guarantee
//! across statement execution, blocking I/O, database open, or connection close.
//! Cooperative cancellation checks do not justify enabling this lane without
//! qualifying the complete connection-owner execution path.
//!
//! A caller-side timeout is not an honest substitute: returning while the owner
//! worker continues executing would bound only the wait, not the database work.
//! This module therefore opens no connection and exposes no query method.
//! Frontends use `AnalyticsSnapshotProvider::snapshot()` for lock-free latest
//! state. Offline reporting that can accept unbounded SQL uses `StorageReader`.

use std::time::Duration;

use crate::StorageError;

const FSQLITE_PINNED_VERSION: &str = "=0.4.0";
const FSQLITE_PINNED_REVISION: &str = "a855a15399a1994943c81e15f284bed780b4f86b";

/// Uninhabited marker for the unavailable hard-bounded async SQL lane.
///
/// There are deliberately no variants and therefore no constructible lane.
#[derive(Debug)]
pub enum AsyncReadLane {}

impl AsyncReadLane {
    /// Refuse a requested hard execution bound before touching `path`.
    ///
    /// The `_path` name is intentional: this fail-closed branch must not inspect,
    /// open, canonicalize, or otherwise access the supplied database path.
    pub fn open(_path: &str, requested: Duration) -> Result<Self, StorageError> {
        Err(StorageError::ReadExecutionBoundUnavailable {
            operation: "async_read_lane.open",
            requested,
            engine_version: FSQLITE_PINNED_VERSION,
            engine_revision: FSQLITE_PINNED_REVISION,
        })
    }
}
