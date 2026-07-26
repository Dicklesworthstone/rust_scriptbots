//! Fail-closed refusal surface for hard-bounded SQL reads (`bd-91lr`).
//!
//! The pinned `fsqlite =0.1.16` async query methods check a caller `Cx` before
//! dispatch, but that context does not reach the statement executing on the
//! connection-owner worker. The engine has cooperative cancellation checks in
//! parts of VDBE execution, but they cannot provide a hard wall-clock guarantee
//! across statement execution, blocking I/O, database open, or connection close.
//!
//! A caller-side timeout is not an honest substitute: returning while the owner
//! worker continues executing would bound only the wait, not the database work.
//! This module therefore opens no connection and exposes no query method.
//! Frontends use `AnalyticsSnapshotProvider::snapshot()` for lock-free latest
//! state. Offline reporting that can accept unbounded SQL uses `StorageReader`.

use std::time::Duration;

use crate::StorageError;

const FSQLITE_PINNED_VERSION: &str = "=0.1.16";
const FSQLITE_PINNED_REVISION: &str = "e536d7f8ca102b3eb5236bef48514582379f9346";

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
