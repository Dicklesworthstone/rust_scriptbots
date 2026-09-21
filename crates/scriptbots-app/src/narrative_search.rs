//! Unified narrative search reader facade and contracts (bd-16g.2.7).
//!
//! Exposes bounded full-text (FTS5 + BM25) search and chronological tick window
//! retrieval across all scriptbots-app surfaces: CLI, REST, FastMCP, and the TUI rail.

use std::path::Path;

use scriptbots_core::narrative::EventRecord;
use scriptbots_storage::{NarrativeHit, StorageError, StorageReader};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use utoipa::{IntoParams, ToSchema};

/// Hard upper bound on search query length in bytes.
pub const MAX_NARRATIVE_QUERY_BYTES: usize = 1024;
/// Hard upper bound on returned search or around page rows.
pub const MAX_NARRATIVE_PAGE_LIMIT: usize = 4096;
/// Default page size when limit is unspecified.
pub const DEFAULT_NARRATIVE_PAGE_LIMIT: usize = 32;
/// Hard upper bound on the chronological half-window size in ticks.
pub const MAX_NARRATIVE_AROUND_WINDOW: u64 = 10_000;
/// Default chronological half-window size when unspecified.
pub const DEFAULT_NARRATIVE_AROUND_WINDOW: u64 = 100;

/// Input parameters for a full-text narrative search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema, IntoParams)]
pub struct NarrativeSearchQuery {
    /// Full-text search query string (e.g. "population", "extinction", "drought").
    pub query: String,
    /// Optional inclusive lower tick bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_tick: Option<u64>,
    /// Optional exclusive upper tick bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_tick: Option<u64>,
    /// Optional maximum number of hits (capped at 4096, default 32).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

/// Input parameters for a chronological narrative window query around a tick.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema, IntoParams)]
pub struct NarrativeAroundQuery {
    /// Center tick for the chronological event window.
    pub tick: u64,
    /// Half-window size in ticks (capped at 10000, default 100).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<u64>,
}

/// Standardized event hit DTO returned uniformly by REST, MCP, and CLI surfaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct NarrativeSearchHitDto {
    /// The simulation tick the event occurred at.
    pub tick: u64,
    /// Categorical event kind string (e.g. "population_crash", "extinction").
    pub kind: String,
    /// Detector severity score in [0.0, 1.0].
    pub severity: f32,
    /// Deterministic templated narrative prose describing the event.
    pub human_text: String,
    /// BM25 ranking score for full-text search (lower is more relevant; None for around queries).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

impl From<NarrativeHit> for NarrativeSearchHitDto {
    fn from(hit: NarrativeHit) -> Self {
        Self {
            tick: hit.tick().0,
            kind: hit.kind().as_str().to_string(),
            severity: hit.severity(),
            human_text: hit.human_text().to_string(),
            score: hit.rank(),
        }
    }
}

impl From<&EventRecord> for NarrativeSearchHitDto {
    fn from(record: &EventRecord) -> Self {
        Self {
            tick: record.tick.0,
            kind: record.kind.as_str().to_string(),
            severity: record.severity,
            human_text: record.human_text.clone(),
            score: None,
        }
    }
}

/// Typed validation and storage errors for narrative query processing.
#[derive(Debug, Error)]
pub enum NarrativeSearchError {
    #[error("narrative query must not be empty")]
    EmptyQuery,

    #[error("narrative query must contain non-whitespace text")]
    WhitespaceQuery,

    #[error("narrative query length {length} exceeds maximum {max} bytes")]
    QueryTooLong { length: usize, max: usize },

    #[error("narrative query contains forbidden NUL byte")]
    NulByte,

    #[error("from_tick ({from}) must be less than or equal to to_tick ({to})")]
    InvalidTickRange { from: u64, to: u64 },

    #[error("requested limit {limit} exceeds maximum allowable {max}")]
    LimitTooLarge { limit: usize, max: usize },

    #[error("requested window {window} exceeds maximum allowable {max}")]
    WindowTooLarge { window: u64, max: u64 },

    #[error("storage unavailable for narrative search: {0}")]
    StorageUnavailable(String),

    #[error("storage query failed: {0}")]
    Storage(#[from] StorageError),
}

/// Validated search query components: (clean_query, optional_tick_range, limit).
pub type ValidatedSearchQuery = (String, Option<(u64, u64)>, usize);

/// Validate search query parameters before hitting storage.
pub fn validate_search_query(
    query: &NarrativeSearchQuery,
) -> Result<ValidatedSearchQuery, NarrativeSearchError> {
    if query.query.is_empty() {
        return Err(NarrativeSearchError::EmptyQuery);
    }
    let trimmed = query.query.trim();
    if trimmed.is_empty() {
        return Err(NarrativeSearchError::WhitespaceQuery);
    }
    if query.query.len() > MAX_NARRATIVE_QUERY_BYTES {
        return Err(NarrativeSearchError::QueryTooLong {
            length: query.query.len(),
            max: MAX_NARRATIVE_QUERY_BYTES,
        });
    }
    if query.query.contains('\0') {
        return Err(NarrativeSearchError::NulByte);
    }

    let tick_range = match (query.from_tick, query.to_tick) {
        (Some(from), Some(to)) => {
            if from > to {
                return Err(NarrativeSearchError::InvalidTickRange { from, to });
            }
            Some((from, to))
        }
        (Some(from), None) => Some((from, i64::MAX as u64)),
        (None, Some(to)) => Some((0, to)),
        (None, None) => None,
    };

    let limit = match query.limit {
        Some(l) if l > MAX_NARRATIVE_PAGE_LIMIT => {
            return Err(NarrativeSearchError::LimitTooLarge {
                limit: l,
                max: MAX_NARRATIVE_PAGE_LIMIT,
            });
        }
        Some(l) => l,
        None => DEFAULT_NARRATIVE_PAGE_LIMIT,
    };

    Ok((trimmed.to_string(), tick_range, limit))
}

/// Validate around query parameters before hitting storage.
pub fn validate_around_query(
    query: &NarrativeAroundQuery,
) -> Result<(u64, u64), NarrativeSearchError> {
    let window = match query.window {
        Some(w) if w > MAX_NARRATIVE_AROUND_WINDOW => {
            return Err(NarrativeSearchError::WindowTooLarge {
                window: w,
                max: MAX_NARRATIVE_AROUND_WINDOW,
            });
        }
        Some(w) => w,
        None => DEFAULT_NARRATIVE_AROUND_WINDOW,
    };

    Ok((query.tick, window))
}

/// Execute a full-text search against the storage reader or in-memory fallback.
pub fn execute_narrative_search(
    database_path: Option<&Path>,
    snapshot_events: Option<&[EventRecord]>,
    query: NarrativeSearchQuery,
) -> Result<Vec<NarrativeSearchHitDto>, NarrativeSearchError> {
    let (clean_query, tick_range, limit) = validate_search_query(&query)?;
    if limit == 0 {
        return Ok(Vec::new());
    }

    if let Some(db_path) = database_path.filter(|p| p.exists()) {
        let path_str = db_path.to_str().ok_or_else(|| {
            NarrativeSearchError::StorageUnavailable(format!(
                "database path {} is not valid UTF-8",
                db_path.display()
            ))
        })?;
        let reader = StorageReader::open(path_str)?;
        let hits = reader.search_narrative(&clean_query, tick_range, limit)?;
        let dtos = hits.into_iter().map(NarrativeSearchHitDto::from).collect();
        return Ok(dtos);
    }

    if let Some(events) = snapshot_events {
        let needle = clean_query.to_lowercase();
        let (start, end) = tick_range.unwrap_or((0, u64::MAX));
        let mut matching: Vec<NarrativeSearchHitDto> = events
            .iter()
            .filter(|e| e.tick.0 >= start && e.tick.0 < end)
            .filter(|e| e.human_text.to_lowercase().contains(&needle))
            .take(limit)
            .map(|e| NarrativeSearchHitDto {
                tick: e.tick.0,
                kind: e.kind.as_str().to_string(),
                severity: e.severity,
                human_text: e.human_text.clone(),
                score: Some(1.0),
            })
            .collect();
        matching.sort_by(|a, b| a.tick.cmp(&b.tick).then_with(|| a.kind.cmp(&b.kind)));
        return Ok(matching);
    }

    Err(NarrativeSearchError::StorageUnavailable(
        "no accessible FrankenSQLite database file or in-memory snapshot was available".into(),
    ))
}

/// Execute a chronological around-tick retrieval against the storage reader or in-memory fallback.
pub fn execute_narrative_around(
    database_path: Option<&Path>,
    snapshot_events: Option<&[EventRecord]>,
    query: NarrativeAroundQuery,
) -> Result<Vec<NarrativeSearchHitDto>, NarrativeSearchError> {
    let (center_tick, window) = validate_around_query(&query)?;

    if let Some(db_path) = database_path.filter(|p| p.exists()) {
        let path_str = db_path.to_str().ok_or_else(|| {
            NarrativeSearchError::StorageUnavailable(format!(
                "database path {} is not valid UTF-8",
                db_path.display()
            ))
        })?;
        let reader = StorageReader::open(path_str)?;
        let hits = reader.narrative_around_tick(center_tick, window)?;
        let dtos = hits.into_iter().map(NarrativeSearchHitDto::from).collect();
        return Ok(dtos);
    }

    if let Some(events) = snapshot_events {
        let start = center_tick.saturating_sub(window);
        let end = center_tick.saturating_add(window);
        let mut matching: Vec<NarrativeSearchHitDto> = events
            .iter()
            .filter(|e| e.tick.0 >= start && e.tick.0 <= end)
            .map(|e| NarrativeSearchHitDto {
                tick: e.tick.0,
                kind: e.kind.as_str().to_string(),
                severity: e.severity,
                human_text: e.human_text.clone(),
                score: None,
            })
            .collect();
        matching.sort_by(|a, b| a.tick.cmp(&b.tick).then_with(|| a.kind.cmp(&b.kind)));
        return Ok(matching);
    }

    Err(NarrativeSearchError::StorageUnavailable(
        "no accessible FrankenSQLite database file or in-memory snapshot was available".into(),
    ))
}

/// Format hits as an aligned human-readable table.
pub fn format_hits_table(hits: &[NarrativeSearchHitDto]) -> String {
    if hits.is_empty() {
        return "No narrative events found matching query.\n".to_string();
    }
    let mut out = String::new();
    out.push_str(&format!(
        "{:<10} {:<20} {:<10} {:<8} {}\n",
        "TICK", "KIND", "SEVERITY", "SCORE", "TEXT"
    ));
    out.push_str(&format!(
        "{:<10} {:<20} {:<10} {:<8} {}\n",
        "----", "----", "--------", "-----", "----"
    ));
    for hit in hits {
        let score_str = hit.score.map_or("-".to_string(), |s| format!("{s:.3}"));
        out.push_str(&format!(
            "{:<10} {:<20} {:<10.2} {:<8} {}\n",
            hit.tick, hit.kind, hit.severity, score_str, hit.human_text
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::Tick;
    use scriptbots_core::narrative::EventKind;

    #[test]
    fn test_validate_search_query_empty_and_whitespace() {
        let empty = NarrativeSearchQuery {
            query: String::new(),
            from_tick: None,
            to_tick: None,
            limit: None,
        };
        assert!(matches!(
            validate_search_query(&empty),
            Err(NarrativeSearchError::EmptyQuery)
        ));

        let ws = NarrativeSearchQuery {
            query: "   \t\n  ".into(),
            from_tick: None,
            to_tick: None,
            limit: None,
        };
        assert!(matches!(
            validate_search_query(&ws),
            Err(NarrativeSearchError::WhitespaceQuery)
        ));
    }

    #[test]
    fn test_validate_search_query_too_long() {
        let overlong = NarrativeSearchQuery {
            query: "x".repeat(MAX_NARRATIVE_QUERY_BYTES + 1),
            from_tick: None,
            to_tick: None,
            limit: None,
        };
        assert!(matches!(
            validate_search_query(&overlong),
            Err(NarrativeSearchError::QueryTooLong { .. })
        ));
    }

    #[test]
    fn test_validate_search_query_nul_byte() {
        let nul = NarrativeSearchQuery {
            query: "hello\0world".into(),
            from_tick: None,
            to_tick: None,
            limit: None,
        };
        assert!(matches!(
            validate_search_query(&nul),
            Err(NarrativeSearchError::NulByte)
        ));
    }

    #[test]
    fn test_validate_search_query_invalid_tick_range() {
        let invalid = NarrativeSearchQuery {
            query: "population".into(),
            from_tick: Some(100),
            to_tick: Some(50),
            limit: None,
        };
        assert!(matches!(
            validate_search_query(&invalid),
            Err(NarrativeSearchError::InvalidTickRange { from: 100, to: 50 })
        ));
    }

    #[test]
    fn test_validate_search_query_limit_bounds() {
        let oversized = NarrativeSearchQuery {
            query: "population".into(),
            from_tick: None,
            to_tick: None,
            limit: Some(MAX_NARRATIVE_PAGE_LIMIT + 1),
        };
        assert!(matches!(
            validate_search_query(&oversized),
            Err(NarrativeSearchError::LimitTooLarge { .. })
        ));

        let normal = NarrativeSearchQuery {
            query: "population".into(),
            from_tick: None,
            to_tick: None,
            limit: Some(10),
        };
        let (q, range, limit) = validate_search_query(&normal).expect("valid query");
        assert_eq!(q, "population");
        assert_eq!(range, None);
        assert_eq!(limit, 10);
    }

    #[test]
    fn test_validate_around_query_window_bounds() {
        let oversized = NarrativeAroundQuery {
            tick: 50,
            window: Some(MAX_NARRATIVE_AROUND_WINDOW + 1),
        };
        assert!(matches!(
            validate_around_query(&oversized),
            Err(NarrativeSearchError::WindowTooLarge { .. })
        ));

        let normal = NarrativeAroundQuery {
            tick: 100,
            window: Some(50),
        };
        let (center, win) = validate_around_query(&normal).expect("valid around query");
        assert_eq!(center, 100);
        assert_eq!(win, 50);
    }

    #[test]
    fn test_execute_in_memory_search_and_around() {
        let events = vec![
            EventRecord {
                schema_version: 1,
                tick: Tick(10),
                kind: EventKind::PopulationBoom,
                severity: 0.7,
                magnitude: 20.0,
                window: (0, 10),
                metric: "population".into(),
                before: 10.0,
                after: 30.0,
                score: 5.0,
                subject: None,
                human_text: "population rose 200% (10 -> 30)".into(),
            },
            EventRecord {
                schema_version: 1,
                tick: Tick(25),
                kind: EventKind::PopulationCrash,
                severity: 0.9,
                magnitude: 25.0,
                window: (15, 25),
                metric: "population".into(),
                before: 30.0,
                after: 5.0,
                score: 8.0,
                subject: None,
                human_text: "population fell 83% (30 -> 5)".into(),
            },
            EventRecord {
                schema_version: 1,
                tick: Tick(50),
                kind: EventKind::Extinction,
                severity: 1.0,
                magnitude: 5.0,
                window: (40, 50),
                metric: "population".into(),
                before: 5.0,
                after: 0.0,
                score: 10.0,
                subject: None,
                human_text: "population reached zero".into(),
            },
        ];

        // Search for "rose"
        let search_res = execute_narrative_search(
            None,
            Some(&events),
            NarrativeSearchQuery {
                query: "rose".into(),
                from_tick: None,
                to_tick: None,
                limit: None,
            },
        )
        .expect("in-memory search");
        assert_eq!(search_res.len(), 1);
        assert_eq!(search_res[0].tick, 10);
        assert_eq!(search_res[0].kind, "population_boom");

        // Search for "population" with tick range [20, 60)
        let search_range = execute_narrative_search(
            None,
            Some(&events),
            NarrativeSearchQuery {
                query: "population".into(),
                from_tick: Some(20),
                to_tick: Some(60),
                limit: Some(10),
            },
        )
        .expect("in-memory search with range");
        assert_eq!(search_range.len(), 2);
        assert_eq!(search_range[0].tick, 25);
        assert_eq!(search_range[1].tick, 50);

        // Around tick 20 with window 10 -> covers ticks [10, 30]
        let around_res = execute_narrative_around(
            None,
            Some(&events),
            NarrativeAroundQuery {
                tick: 20,
                window: Some(10),
            },
        )
        .expect("in-memory around");
        assert_eq!(around_res.len(), 2);
        assert_eq!(around_res[0].tick, 10);
        assert_eq!(around_res[1].tick, 25);
        assert!(around_res[0].score.is_none());

        // Format table check
        let table = format_hits_table(&search_res);
        assert!(table.contains("TICK"));
        assert!(table.contains("population_boom"));
        assert!(table.contains("population rose 200%"));
    }
}
