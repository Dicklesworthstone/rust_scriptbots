use serde::{Deserialize, Serialize};
use std::io::Write;
use std::time::SystemTime;

use crate::{StorageError, StorageReader, checked_u64, decode as decode_column, sqlite_run_id};

// ============================================================================
// SCHEMA EVOLUTION AND DIGEST COMPARABILITY POLICY (bd-2z0.5.6)
//
// This is the compatibility policy the acceptance requires, made executable
// rather than left as prose, because it is the decision that several downstream
// beads were each about to make separately and differently.
//
// THE PROBLEM. Every export artifact carries an integrity digest. When a table
// gains a provenance field -- and it will, because provenance is exactly the
// thing that grows -- there appear to be only two options, and both are bad:
//
//   (a) Fold the new field into the digest. Old artifacts now fail verification
//       even though nothing about them changed or is wrong.
//   (b) Carry the new field outside the digest. Old artifacts stay valid, but
//       the new field is not covered by the integrity guarantee -- it is
//       reproducible in principle and unverified in fact.
//
// THE DECISION: NEITHER. The dilemma is an artifact of treating a digest as a
// global truth. A digest is a claim about bytes UNDER A STATED SCHEMA. Comparing
// digests across schema versions is a category error, not a mismatch.
//
// So the policy is:
//   1. Schema version is PER TABLE, not one global number for the whole export.
//      Adding a field to the run table must not invalidate metric artifacts.
//   2. A new field is folded INTO the digest and bumps that table's version.
//      Provenance that the integrity digest does not cover is not provenance.
//   3. Readers accept older versions (forward-compatible reads), but a digest is
//      only comparable to another digest of the SAME (table, version).
//   4. A comparison across versions reports NOT COMPARABLE, which is a distinct
//      outcome from MISMATCH. Collapsing those two is what makes teams choose
//      (b): they see old artifacts "failing" and conclude the digest must not
//      cover new fields. It was never a failure -- it was a question nobody was
//      allowed to answer as "that is not the same question".
//
// WHY THIS UNBLOCKS bd-16g.1.7. Its reproduce.sh cannot rerun an arm because the
// lab's RunSummary/RunRef keep `arm_id: u16` plus a config digest, and a digest
// verifies a config without being able to recreate it. Adding the arm identity
// looked like a forced choice between (a) and (b). Under this policy it is
// neither: bump the run table's version, cover the new field, and let older
// artifacts report NotComparable instead of failing.
// ============================================================================

/// Canonical export table families (bd-2z0.5.6).
///
/// Typed rather than a `&str` deliberately. The generic `write_row(&str, ...)`
/// envelope on [`CoreTableExportWriter`] is what let an arbitrary table name
/// masquerade as schema coverage: passing the strings "run", "agent" and
/// "lineage" to a writer that validates neither made three canonical tables
/// look delivered when none existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportTable {
    /// One row per run: identity, seed, config, provenance.
    Run,
    /// One row per agent observation.
    Agent,
    /// One row per lineage edge.
    Lineage,
    /// One row per narrative/domain event.
    Event,
    /// One row per (tick, metric) sample.
    Metric,
}

impl ExportTable {
    /// Stable snake_case name used in artifacts and filenames.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Agent => "agent",
            Self::Lineage => "lineage",
            Self::Event => "event",
            Self::Metric => "metric",
        }
    }

    /// Current schema version for this table.
    ///
    /// PER TABLE by policy: adding a field to `run` must not invalidate every
    /// retained `metric` artifact. Bump exactly the table you changed.
    #[must_use]
    pub const fn current_version(self) -> u32 {
        match self {
            // All families start at 1. When a table gains a field, bump ONLY its
            // arm here and say why in the commit -- the version number is the
            // only thing standing between a real mismatch and a false one.
            Self::Run | Self::Agent | Self::Lineage | Self::Event | Self::Metric => 1,
        }
    }

    /// Canonical Arrow/Parquet schema specification for this table.
    #[must_use]
    pub const fn arrow_schema(self) -> &'static [ArrowColumnSpec] {
        match self {
            Self::Run => RUN_ARROW_SCHEMA,
            Self::Agent => AGENT_ARROW_SCHEMA,
            Self::Lineage => LINEAGE_ARROW_SCHEMA,
            Self::Event => EVENT_ARROW_SCHEMA,
            Self::Metric => METRIC_ARROW_SCHEMA,
        }
    }

    /// Every canonical family, for exhaustive iteration in tests and tooling.
    pub const ALL: [Self; 5] = [
        Self::Run,
        Self::Agent,
        Self::Lineage,
        Self::Event,
        Self::Metric,
    ];
}

/// Identity a digest is scoped to. A digest means nothing without one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaId {
    /// Which canonical table.
    pub table: ExportTable,
    /// That table's schema version at write time.
    pub version: u32,
}

impl SchemaId {
    /// Schema identity for a table at its current version.
    #[must_use]
    pub const fn current(table: ExportTable) -> Self {
        Self {
            table,
            version: table.current_version(),
        }
    }
}

/// Whether two artifacts' digests may be compared at all.
///
/// The three outcomes are deliberately distinct. A verifier that cannot say
/// "not comparable" is forced to report a schema change as data corruption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigestComparability {
    /// Same table, same version: digests are directly comparable.
    Comparable,
    /// Same table, different versions. NOT a mismatch -- a different question.
    DifferentVersion {
        /// Version the verifier expected.
        expected: u32,
        /// Version the artifact was written at.
        found: u32,
    },
    /// Different tables entirely: comparing them is meaningless.
    DifferentTable {
        /// Table the verifier expected.
        expected: ExportTable,
        /// Table the artifact actually holds.
        found: ExportTable,
    },
}

impl DigestComparability {
    /// True only when a digest equality check is a meaningful question.
    #[must_use]
    pub const fn is_comparable(self) -> bool {
        matches!(self, Self::Comparable)
    }
}

/// Decide whether two schema identities admit a digest comparison.
///
/// Call this BEFORE comparing digests. Comparing first and interpreting the
/// result afterwards is how a schema bump gets reported as corruption.
#[must_use]
pub const fn compare_schema(expected: SchemaId, found: SchemaId) -> DigestComparability {
    if expected.table as u8 != found.table as u8 {
        return DigestComparability::DifferentTable {
            expected: expected.table,
            found: found.table,
        };
    }
    if expected.version != found.version {
        return DigestComparability::DifferentVersion {
            expected: expected.version,
            found: found.version,
        };
    }
    DigestComparability::Comparable
}

/// Export format for run analytics data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    Csv,
    JsonLines,
}

/// Provenance metadata embedded in exported artifacts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportProvenance {
    pub run_id: String,
    pub seed: u64,
    /// Digest of the normalized run configuration (the conservation verdict's
    /// `config_digest` is the canonical source for this value).
    pub config_digest: String,
    /// Exact source revision/tree that produced the run or rendered artifact.
    pub source_revision: String,
    pub source_tree_digest: String,
    /// Decisions that affect interpretation of the exported rows (for example,
    /// authority/recovery outcomes or conservation verdict status).
    pub authority_decisions: Vec<String>,
    /// Serialized tolerance policy when a conservation verdict was part of the run.
    pub conservation_tolerances: Option<serde_json::Value>,
    pub schema_version: u32,
    pub exported_at_utc: String,
}

// ============================================================================
// CANONICAL TYPED ROW SCHEMAS (bd-2z0.5.6)
// ============================================================================

/// Canonical row for the `run` export table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunExportRow {
    pub run_id: String,
    pub manifest_schema_version: u32,
    pub scenario_id: String,
    pub scenario_version: u32,
    pub config_digest: String,
    pub root_seed_hex: String,
    pub source_revision: Option<String>,
    pub source_tree_digest: Option<String>,
    pub started_at_unix_ms: u64,
    pub reproducible: bool,
}

/// Canonical row for the `agent` export table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentExportRow {
    pub run_id: String,
    pub tick: u64,
    pub agent_uid: u64,
    pub generation: u32,
    pub age: u32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub vel_x: f32,
    pub vel_y: f32,
    pub heading: f32,
    pub health: f32,
    pub energy: f32,
    pub herbivore_tendency: f32,
    pub brain_binding: String,
}

/// Canonical row for the `lineage` export table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageExportRow {
    pub run_id: String,
    pub child_agent_uid: u64,
    pub parent_agent_uid: u64,
    pub parent_ordinal: u32,
    pub relationship: String,
    pub birth_tick: u64,
}

/// Canonical row for the `event` export table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventExportRow {
    pub run_id: String,
    pub tick: u64,
    pub seq: u64,
    pub event_id: String,
    pub kind: String,
    pub payload: String,
}

/// Canonical row for the `metric` export table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricExportRow {
    pub run_id: String,
    pub tick: u64,
    pub name: String,
    pub value: f64,
}

// ============================================================================
// ARROW / PARQUET LOGICAL TYPE MAPPING (consumed by bd-2z0.11.8)
// ============================================================================

/// Logical data type for canonical Arrow/Parquet table export mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArrowDataType {
    Utf8,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Boolean,
}

/// Specification for a single column in an Arrow/Parquet export table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrowColumnSpec {
    pub name: &'static str,
    pub data_type: ArrowDataType,
    pub nullable: bool,
    pub units: &'static str,
    pub doc: &'static str,
}

/// Arrow column layout for the canonical `run` table.
pub const RUN_ARROW_SCHEMA: &[ArrowColumnSpec] = &[
    ArrowColumnSpec {
        name: "run_id",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "id",
        doc: "Unique identifier for the simulation run",
    },
    ArrowColumnSpec {
        name: "manifest_schema_version",
        data_type: ArrowDataType::UInt32,
        nullable: false,
        units: "version",
        doc: "Schema version of run manifest record",
    },
    ArrowColumnSpec {
        name: "scenario_id",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "id",
        doc: "Scenario preset identifier",
    },
    ArrowColumnSpec {
        name: "scenario_version",
        data_type: ArrowDataType::UInt32,
        nullable: false,
        units: "version",
        doc: "Version number of the scenario preset",
    },
    ArrowColumnSpec {
        name: "config_digest",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "hash",
        doc: "Blake3 hash of canonical configuration json",
    },
    ArrowColumnSpec {
        name: "root_seed_hex",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "hex",
        doc: "Hex-encoded root RNG seed",
    },
    ArrowColumnSpec {
        name: "source_revision",
        data_type: ArrowDataType::Utf8,
        nullable: true,
        units: "git-sha",
        doc: "Git commit hash of simulation source",
    },
    ArrowColumnSpec {
        name: "source_tree_digest",
        data_type: ArrowDataType::Utf8,
        nullable: true,
        units: "hash",
        doc: "Blake3 tree hash of source workspace",
    },
    ArrowColumnSpec {
        name: "started_at_unix_ms",
        data_type: ArrowDataType::UInt64,
        nullable: false,
        units: "milliseconds",
        doc: "Unix timestamp in milliseconds when run began",
    },
    ArrowColumnSpec {
        name: "reproducible",
        data_type: ArrowDataType::Boolean,
        nullable: false,
        units: "boolean",
        doc: "Whether execution followed deterministic reproducibility",
    },
];

/// Arrow column layout for the canonical `agent` table.
pub const AGENT_ARROW_SCHEMA: &[ArrowColumnSpec] = &[
    ArrowColumnSpec {
        name: "run_id",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "id",
        doc: "Run identifier",
    },
    ArrowColumnSpec {
        name: "tick",
        data_type: ArrowDataType::UInt64,
        nullable: false,
        units: "ticks",
        doc: "Simulation step when observation occurred",
    },
    ArrowColumnSpec {
        name: "agent_uid",
        data_type: ArrowDataType::UInt64,
        nullable: false,
        units: "uid",
        doc: "Monotonic unique agent identifier",
    },
    ArrowColumnSpec {
        name: "generation",
        data_type: ArrowDataType::UInt32,
        nullable: false,
        units: "count",
        doc: "Generational depth from founder ancestors",
    },
    ArrowColumnSpec {
        name: "age",
        data_type: ArrowDataType::UInt32,
        nullable: false,
        units: "ticks",
        doc: "Agent lifetime in simulation ticks",
    },
    ArrowColumnSpec {
        name: "pos_x",
        data_type: ArrowDataType::Float32,
        nullable: false,
        units: "world_units",
        doc: "2D spatial position X coordinate",
    },
    ArrowColumnSpec {
        name: "pos_y",
        data_type: ArrowDataType::Float32,
        nullable: false,
        units: "world_units",
        doc: "2D spatial position Y coordinate",
    },
    ArrowColumnSpec {
        name: "vel_x",
        data_type: ArrowDataType::Float32,
        nullable: false,
        units: "units/tick",
        doc: "Linear velocity X component",
    },
    ArrowColumnSpec {
        name: "vel_y",
        data_type: ArrowDataType::Float32,
        nullable: false,
        units: "units/tick",
        doc: "Linear velocity Y component",
    },
    ArrowColumnSpec {
        name: "heading",
        data_type: ArrowDataType::Float32,
        nullable: false,
        units: "radians",
        doc: "Orientation angle in radians [-pi, pi]",
    },
    ArrowColumnSpec {
        name: "health",
        data_type: ArrowDataType::Float32,
        nullable: false,
        units: "points",
        doc: "Current health points [0.0, 1.0]",
    },
    ArrowColumnSpec {
        name: "energy",
        data_type: ArrowDataType::Float32,
        nullable: false,
        units: "joules",
        doc: "Internal metabolic energy reserves",
    },
    ArrowColumnSpec {
        name: "herbivore_tendency",
        data_type: ArrowDataType::Float32,
        nullable: false,
        units: "ratio",
        doc: "Dietary phenotype [0.0 = carnivore, 1.0 = herbivore]",
    },
    ArrowColumnSpec {
        name: "brain_binding",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "kind",
        doc: "Brain architecture family name (e.g. mlp, dwraon, neuro)",
    },
];

/// Arrow column layout for the canonical `lineage` table.
pub const LINEAGE_ARROW_SCHEMA: &[ArrowColumnSpec] = &[
    ArrowColumnSpec {
        name: "run_id",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "id",
        doc: "Run identifier",
    },
    ArrowColumnSpec {
        name: "child_agent_uid",
        data_type: ArrowDataType::UInt64,
        nullable: false,
        units: "uid",
        doc: "Child agent monotonic unique identifier",
    },
    ArrowColumnSpec {
        name: "parent_agent_uid",
        data_type: ArrowDataType::UInt64,
        nullable: false,
        units: "uid",
        doc: "Parent agent monotonic unique identifier",
    },
    ArrowColumnSpec {
        name: "parent_ordinal",
        data_type: ArrowDataType::UInt32,
        nullable: false,
        units: "ordinal",
        doc: "0 for primary parent, 1 for secondary crossover parent",
    },
    ArrowColumnSpec {
        name: "relationship",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "enum",
        doc: "Type of reproduction: asexual_clone, sexual_parent_a, sexual_parent_b",
    },
    ArrowColumnSpec {
        name: "birth_tick",
        data_type: ArrowDataType::UInt64,
        nullable: false,
        units: "ticks",
        doc: "Simulation tick when child was spawned",
    },
];

/// Arrow column layout for the canonical `event` table.
pub const EVENT_ARROW_SCHEMA: &[ArrowColumnSpec] = &[
    ArrowColumnSpec {
        name: "run_id",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "id",
        doc: "Run identifier",
    },
    ArrowColumnSpec {
        name: "tick",
        data_type: ArrowDataType::UInt64,
        nullable: false,
        units: "ticks",
        doc: "Simulation step when event occurred",
    },
    ArrowColumnSpec {
        name: "seq",
        data_type: ArrowDataType::UInt64,
        nullable: false,
        units: "sequence",
        doc: "Sequence number within tick",
    },
    ArrowColumnSpec {
        name: "event_id",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "id",
        doc: "Unique event identifier",
    },
    ArrowColumnSpec {
        name: "kind",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "category",
        doc: "Domain event kind (e.g. speciation, combat, extinction)",
    },
    ArrowColumnSpec {
        name: "payload",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "json",
        doc: "Serialized event details / narrative description",
    },
];

/// Arrow column layout for the canonical `metric` table.
pub const METRIC_ARROW_SCHEMA: &[ArrowColumnSpec] = &[
    ArrowColumnSpec {
        name: "run_id",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "id",
        doc: "Run identifier",
    },
    ArrowColumnSpec {
        name: "tick",
        data_type: ArrowDataType::UInt64,
        nullable: false,
        units: "ticks",
        doc: "Simulation step when metric was sampled",
    },
    ArrowColumnSpec {
        name: "name",
        data_type: ArrowDataType::Utf8,
        nullable: false,
        units: "name",
        doc: "Metric identifier name",
    },
    ArrowColumnSpec {
        name: "value",
        data_type: ArrowDataType::Float64,
        nullable: false,
        units: "dimensionless/SI",
        doc: "Sampled scalar metric reading",
    },
];

// ============================================================================
// FORMATTING, ESCAPING, AND RECEIPT VERIFICATION HELPERS
// ============================================================================

/// RFC 4180 CSV escaping: quotes fields containing commas, double-quotes, or newlines.
pub fn csv_escape(s: &str) -> String {
    if s.contains(['"', ',', '\r', '\n']) {
        let mut out = String::with_capacity(s.len() + 4);
        out.push('"');
        for ch in s.chars() {
            if ch == '"' {
                out.push('"');
                out.push('"');
            } else {
                out.push(ch);
            }
        }
        out.push('"');
        out
    } else {
        s.to_string()
    }
}

/// Simple RFC 4180 CSV line parser handling double-quoted fields.
pub fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == ',' {
            fields.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    fields.push(current);
    fields
}

/// Format f32 for CSV under the explicit IEEE-754 policy (NaN, Infinity, -Infinity).
pub fn format_f32_csv(val: f32) -> String {
    if val.is_nan() {
        "NaN".to_string()
    } else if val.is_infinite() {
        if val.is_sign_positive() {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        }
    } else {
        val.to_string()
    }
}

/// Format f64 for CSV under the explicit IEEE-754 policy.
pub fn format_f64_csv(val: f64) -> String {
    if val.is_nan() {
        "NaN".to_string()
    } else if val.is_infinite() {
        if val.is_sign_positive() {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        }
    } else {
        val.to_string()
    }
}

/// Parse f32 from CSV adhering to the documented float policy.
pub fn parse_f32_csv(s: &str) -> Result<f32, std::num::ParseFloatError> {
    match s.trim() {
        "NaN" | "nan" => Ok(f32::NAN),
        "Infinity" | "+Infinity" | "inf" | "+inf" => Ok(f32::INFINITY),
        "-Infinity" | "-inf" => Ok(f32::NEG_INFINITY),
        other => other.parse::<f32>(),
    }
}

/// Parse f64 from CSV adhering to the documented float policy.
pub fn parse_f64_csv(s: &str) -> Result<f64, std::num::ParseFloatError> {
    match s.trim() {
        "NaN" | "nan" => Ok(f64::NAN),
        "Infinity" | "+Infinity" | "inf" | "+inf" => Ok(f64::INFINITY),
        "-Infinity" | "-inf" => Ok(f64::NEG_INFINITY),
        other => other.parse::<f64>(),
    }
}

/// Deterministic calendar conversion for ISO 8601 UTC timestamp string without third-party crates.
pub fn format_unix_timestamp_iso8601(secs: u64) -> String {
    let days = (secs / 86400) as i64;
    let rem_secs = (secs % 86400) as u32;
    let hours = rem_secs / 3600;
    let mins = (rem_secs % 3600) / 60;
    let s = rem_secs % 60;

    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1020 + doe / 1460 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{hours:02}:{mins:02}:{s:02}Z")
}

/// Current UTC ISO 8601 string from system time.
pub fn current_utc_iso8601() -> String {
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_unix_timestamp_iso8601(secs)
}

// ============================================================================
// ATOMIC RECEIPT & VERIFICATION (bd-2z0.5.6)
// ============================================================================

/// Proof of atomic completion stamped at the end of an export artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportReceipt {
    pub table: ExportTable,
    pub format: ExportFormat,
    pub row_count: u64,
    pub checksum_blake3: String,
    pub completed_at_utc: String,
}

/// Failures detected when verifying an exported artifact's integrity receipt.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExportVerificationError {
    #[error(
        "export artifact is missing the trailing completion receipt (interrupted or truncated stream)"
    )]
    MissingReceipt,
    #[error("export artifact receipt table {found:?} does not match expected {expected:?}")]
    TableMismatch {
        expected: ExportTable,
        found: ExportTable,
    },
    #[error("export artifact reported {reported} rows but contained {counted} rows")]
    RowCountMismatch { reported: u64, counted: u64 },
    #[error("export artifact reported checksum {reported} but computed checksum was {computed}")]
    ChecksumMismatch { reported: String, computed: String },
    #[error("malformed receipt format: {0}")]
    MalformedReceipt(String),
}

/// Verify an exported stream or string buffer end-to-end against its trailing receipt.
pub fn verify_export_receipt(
    content: &str,
    expected_table: ExportTable,
) -> Result<ExportReceipt, ExportVerificationError> {
    let mut receipt_opt: Option<ExportReceipt> = None;
    let mut row_count = 0u64;
    let mut hasher = blake3::Hasher::new();
    let mut in_quotes = false;
    let mut pending_record = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if !in_quotes {
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with("# PROVENANCE:") || trimmed.starts_with("{\"provenance\":") {
                continue;
            }
            if trimmed.starts_with("# RECEIPT:") {
                let receipt_str = trimmed.trim_start_matches("# RECEIPT:").trim();
                receipt_opt = Some(parse_csv_receipt(receipt_str)?);
                continue;
            }
            if trimmed.starts_with("{\"_receipt\":") {
                let val: serde_json::Value = serde_json::from_str(trimmed)
                    .map_err(|e| ExportVerificationError::MalformedReceipt(e.to_string()))?;
                let receipt: ExportReceipt = serde_json::from_value(val["_receipt"].clone())
                    .map_err(|e| ExportVerificationError::MalformedReceipt(e.to_string()))?;
                receipt_opt = Some(receipt);
                continue;
            }
            // Skip CSV header line
            if trimmed.starts_with("run_id,")
                || trimmed.starts_with("tick,")
                || trimmed.starts_with("event_id,")
            {
                continue;
            }
        }

        // Check for double-quotes across the line to detect multi-line CSV fields
        let mut chars = line.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '"' {
                if in_quotes && chars.peek() == Some(&'"') {
                    chars.next(); // skip escaped quote pair ""
                } else {
                    in_quotes = !in_quotes;
                }
            }
        }

        if !pending_record.is_empty() {
            pending_record.push('\n');
        }
        pending_record.push_str(line);

        if !in_quotes {
            row_count += 1;
            hasher.update(pending_record.as_bytes());
            hasher.update(b"\n");
            pending_record.clear();
        }
    }

    let receipt = receipt_opt.ok_or(ExportVerificationError::MissingReceipt)?;
    if receipt.table != expected_table {
        return Err(ExportVerificationError::TableMismatch {
            expected: expected_table,
            found: receipt.table,
        });
    }
    if receipt.row_count != row_count {
        return Err(ExportVerificationError::RowCountMismatch {
            reported: receipt.row_count,
            counted: row_count,
        });
    }
    let computed_hash = format!("blake3:{}", hasher.finalize().to_hex());
    if receipt.checksum_blake3 != computed_hash {
        return Err(ExportVerificationError::ChecksumMismatch {
            reported: receipt.checksum_blake3,
            computed: computed_hash,
        });
    }

    Ok(receipt)
}

fn parse_csv_receipt(s: &str) -> Result<ExportReceipt, ExportVerificationError> {
    let mut table = None;
    let mut format = None;
    let mut row_count = None;
    let mut checksum_blake3 = None;
    let mut completed_at_utc = None;

    for part in s.split(',') {
        let mut kv = part.splitn(2, '=');
        let key = kv.next().unwrap_or("").trim();
        let val = kv.next().unwrap_or("").trim();
        match key {
            "table" => {
                let t = match val {
                    "run" => ExportTable::Run,
                    "agent" => ExportTable::Agent,
                    "lineage" => ExportTable::Lineage,
                    "event" => ExportTable::Event,
                    "metric" => ExportTable::Metric,
                    _ => {
                        return Err(ExportVerificationError::MalformedReceipt(format!(
                            "unknown table {val}"
                        )));
                    }
                };
                table = Some(t);
            }
            "format" => {
                let f = match val {
                    "csv" => ExportFormat::Csv,
                    "json_lines" => ExportFormat::JsonLines,
                    _ => {
                        return Err(ExportVerificationError::MalformedReceipt(format!(
                            "unknown format {val}"
                        )));
                    }
                };
                format = Some(f);
            }
            "row_count" => {
                let c = val
                    .parse::<u64>()
                    .map_err(|e| ExportVerificationError::MalformedReceipt(e.to_string()))?;
                row_count = Some(c);
            }
            "checksum_blake3" => checksum_blake3 = Some(val.to_string()),
            "completed_at_utc" => completed_at_utc = Some(val.to_string()),
            _ => {}
        }
    }

    Ok(ExportReceipt {
        table: table
            .ok_or_else(|| ExportVerificationError::MalformedReceipt("missing table".into()))?,
        format: format
            .ok_or_else(|| ExportVerificationError::MalformedReceipt("missing format".into()))?,
        row_count: row_count
            .ok_or_else(|| ExportVerificationError::MalformedReceipt("missing row_count".into()))?,
        checksum_blake3: checksum_blake3
            .ok_or_else(|| ExportVerificationError::MalformedReceipt("missing checksum".into()))?,
        completed_at_utc: completed_at_utc
            .ok_or_else(|| ExportVerificationError::MalformedReceipt("missing timestamp".into()))?,
    })
}

// ============================================================================
// STREAMING CORE TABLE EXPORT WRITER (bd-2z0.5.6)
// ============================================================================

/// Canonical streaming export writer supporting bounded CSV and JSONLines
/// output across all 5 core tables with atomic integrity receipts.
pub struct CoreTableExportWriter<W: Write> {
    writer: W,
    format: ExportFormat,
    header_written: bool,
    row_count: u64,
    hasher: blake3::Hasher,
}

impl<W: Write> CoreTableExportWriter<W> {
    /// Create a new writer defaulting to JSONLines format for backward compatibility.
    pub fn new(writer: W) -> Self {
        Self::new_with_format(writer, ExportFormat::JsonLines)
    }

    /// Create a new writer with explicit format.
    pub fn new_with_format(writer: W, format: ExportFormat) -> Self {
        Self {
            writer,
            format,
            header_written: false,
            row_count: 0,
            hasher: blake3::Hasher::new(),
        }
    }

    /// Access the configured export format.
    #[must_use]
    pub const fn format(&self) -> ExportFormat {
        self.format
    }

    /// Access the current count of successfully written data rows.
    #[must_use]
    pub const fn row_count(&self) -> u64 {
        self.row_count
    }

    /// Stream canonical provenance metadata header into the output.
    pub fn write_provenance(&mut self, prov: &ExportProvenance) -> std::io::Result<()> {
        let json = serde_json::to_string(prov).unwrap_or_default();
        match self.format {
            ExportFormat::JsonLines => writeln!(self.writer, "{{\"provenance\":{json}}}"),
            ExportFormat::Csv => {
                writeln!(
                    self.writer,
                    "# PROVENANCE: run_id={},seed={},config_digest={},source_revision={},source_tree_digest={},schema_version={},authority_decisions={},conservation_tolerances={}",
                    csv_escape(&prov.run_id),
                    prov.seed,
                    csv_escape(&prov.config_digest),
                    csv_escape(&prov.source_revision),
                    csv_escape(&prov.source_tree_digest),
                    prov.schema_version,
                    csv_escape(
                        &serde_json::to_string(&prov.authority_decisions).unwrap_or_default()
                    ),
                    csv_escape(
                        &serde_json::to_string(&prov.conservation_tolerances).unwrap_or_default()
                    )
                )
            }
        }
    }

    /// Write one canonical `RunExportRow`.
    pub fn write_run_row(&mut self, row: &RunExportRow) -> std::io::Result<()> {
        match self.format {
            ExportFormat::Csv => {
                if !self.header_written {
                    writeln!(
                        self.writer,
                        "run_id,manifest_schema_version,scenario_id,scenario_version,config_digest,root_seed_hex,source_revision,source_tree_digest,started_at_unix_ms,reproducible"
                    )?;
                    self.header_written = true;
                }
                let line = format!(
                    "{},{},{},{},{},{},{},{},{},{}",
                    csv_escape(&row.run_id),
                    row.manifest_schema_version,
                    csv_escape(&row.scenario_id),
                    row.scenario_version,
                    csv_escape(&row.config_digest),
                    csv_escape(&row.root_seed_hex),
                    row.source_revision
                        .as_deref()
                        .map(csv_escape)
                        .unwrap_or_default(),
                    row.source_tree_digest
                        .as_deref()
                        .map(csv_escape)
                        .unwrap_or_default(),
                    row.started_at_unix_ms,
                    row.reproducible
                );
                writeln!(self.writer, "{line}")?;
                self.hasher.update(line.as_bytes());
                self.hasher.update(b"\n");
                self.row_count += 1;
                Ok(())
            }
            ExportFormat::JsonLines => {
                let json = serde_json::to_string(row).unwrap_or_default();
                let line = format!("{{\"table\":\"run\",\"row\":{json}}}");
                writeln!(self.writer, "{line}")?;
                self.hasher.update(line.as_bytes());
                self.hasher.update(b"\n");
                self.row_count += 1;
                Ok(())
            }
        }
    }

    /// Write one canonical `AgentExportRow`.
    pub fn write_agent_row(&mut self, row: &AgentExportRow) -> std::io::Result<()> {
        match self.format {
            ExportFormat::Csv => {
                if !self.header_written {
                    writeln!(
                        self.writer,
                        "run_id,tick,agent_uid,generation,age,pos_x,pos_y,vel_x,vel_y,heading,health,energy,herbivore_tendency,brain_binding"
                    )?;
                    self.header_written = true;
                }
                let line = format!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    csv_escape(&row.run_id),
                    row.tick,
                    row.agent_uid,
                    row.generation,
                    row.age,
                    format_f32_csv(row.pos_x),
                    format_f32_csv(row.pos_y),
                    format_f32_csv(row.vel_x),
                    format_f32_csv(row.vel_y),
                    format_f32_csv(row.heading),
                    format_f32_csv(row.health),
                    format_f32_csv(row.energy),
                    format_f32_csv(row.herbivore_tendency),
                    csv_escape(&row.brain_binding)
                );
                writeln!(self.writer, "{line}")?;
                self.hasher.update(line.as_bytes());
                self.hasher.update(b"\n");
                self.row_count += 1;
                Ok(())
            }
            ExportFormat::JsonLines => {
                let json = serde_json::to_string(row).unwrap_or_default();
                let line = format!("{{\"table\":\"agent\",\"row\":{json}}}");
                writeln!(self.writer, "{line}")?;
                self.hasher.update(line.as_bytes());
                self.hasher.update(b"\n");
                self.row_count += 1;
                Ok(())
            }
        }
    }

    /// Write one canonical `LineageExportRow`.
    pub fn write_lineage_row(&mut self, row: &LineageExportRow) -> std::io::Result<()> {
        match self.format {
            ExportFormat::Csv => {
                if !self.header_written {
                    writeln!(
                        self.writer,
                        "run_id,child_agent_uid,parent_agent_uid,parent_ordinal,relationship,birth_tick"
                    )?;
                    self.header_written = true;
                }
                let line = format!(
                    "{},{},{},{},{},{}",
                    csv_escape(&row.run_id),
                    row.child_agent_uid,
                    row.parent_agent_uid,
                    row.parent_ordinal,
                    csv_escape(&row.relationship),
                    row.birth_tick
                );
                writeln!(self.writer, "{line}")?;
                self.hasher.update(line.as_bytes());
                self.hasher.update(b"\n");
                self.row_count += 1;
                Ok(())
            }
            ExportFormat::JsonLines => {
                let json = serde_json::to_string(row).unwrap_or_default();
                let line = format!("{{\"table\":\"lineage\",\"row\":{json}}}");
                writeln!(self.writer, "{line}")?;
                self.hasher.update(line.as_bytes());
                self.hasher.update(b"\n");
                self.row_count += 1;
                Ok(())
            }
        }
    }

    /// Write one canonical `EventExportRow`.
    pub fn write_event_row(&mut self, row: &EventExportRow) -> std::io::Result<()> {
        match self.format {
            ExportFormat::Csv => {
                if !self.header_written {
                    writeln!(self.writer, "run_id,tick,seq,event_id,kind,payload")?;
                    self.header_written = true;
                }
                let line = format!(
                    "{},{},{},{},{},{}",
                    csv_escape(&row.run_id),
                    row.tick,
                    row.seq,
                    csv_escape(&row.event_id),
                    csv_escape(&row.kind),
                    csv_escape(&row.payload)
                );
                writeln!(self.writer, "{line}")?;
                self.hasher.update(line.as_bytes());
                self.hasher.update(b"\n");
                self.row_count += 1;
                Ok(())
            }
            ExportFormat::JsonLines => {
                let json = serde_json::to_string(row).unwrap_or_default();
                let line = format!("{{\"table\":\"event\",\"row\":{json}}}");
                writeln!(self.writer, "{line}")?;
                self.hasher.update(line.as_bytes());
                self.hasher.update(b"\n");
                self.row_count += 1;
                Ok(())
            }
        }
    }

    /// Write one canonical `MetricExportRow`.
    pub fn write_metric_row(&mut self, row: &MetricExportRow) -> std::io::Result<()> {
        match self.format {
            ExportFormat::Csv => {
                if !self.header_written {
                    writeln!(self.writer, "run_id,tick,name,value")?;
                    self.header_written = true;
                }
                let line = format!(
                    "{},{},{},{}",
                    csv_escape(&row.run_id),
                    row.tick,
                    csv_escape(&row.name),
                    format_f64_csv(row.value)
                );
                writeln!(self.writer, "{line}")?;
                self.hasher.update(line.as_bytes());
                self.hasher.update(b"\n");
                self.row_count += 1;
                Ok(())
            }
            ExportFormat::JsonLines => {
                let json = serde_json::to_string(row).unwrap_or_default();
                let line = format!("{{\"table\":\"metric\",\"row\":{json}}}");
                writeln!(self.writer, "{line}")?;
                self.hasher.update(line.as_bytes());
                self.hasher.update(b"\n");
                self.row_count += 1;
                Ok(())
            }
        }
    }

    /// Backward-compatible generic JSON row writer.
    pub fn write_row(&mut self, table: &str, row: &serde_json::Value) -> std::io::Result<()> {
        let mut envelope = serde_json::Map::new();
        envelope.insert("table".into(), serde_json::Value::String(table.into()));
        envelope.insert("row".into(), row.clone());
        let line = serde_json::to_string(&serde_json::Value::Object(envelope)).unwrap_or_default();
        writeln!(self.writer, "{line}")?;
        self.hasher.update(line.as_bytes());
        self.hasher.update(b"\n");
        self.row_count += 1;
        Ok(())
    }

    /// Atomically seal the export artifact by stamping the trailing completion receipt.
    pub fn finish(mut self, table: ExportTable) -> std::io::Result<ExportReceipt> {
        let checksum_hex = self.hasher.finalize().to_hex().to_string();
        let receipt = ExportReceipt {
            table,
            format: self.format,
            row_count: self.row_count,
            checksum_blake3: format!("blake3:{checksum_hex}"),
            completed_at_utc: current_utc_iso8601(),
        };

        match self.format {
            ExportFormat::Csv => {
                writeln!(
                    self.writer,
                    "# RECEIPT: table={},format=csv,row_count={},checksum_blake3={},completed_at_utc={}",
                    receipt.table.as_str(),
                    receipt.row_count,
                    receipt.checksum_blake3,
                    receipt.completed_at_utc
                )?;
            }
            ExportFormat::JsonLines => {
                let json = serde_json::to_string(&receipt).unwrap_or_default();
                writeln!(self.writer, "{{\"_receipt\":{json}}}")?;
            }
        }
        self.writer.flush()?;
        Ok(receipt)
    }
}

// ============================================================================
// LEGACY FORMAT WRITERS (PRESERVED FOR BACKWARD COMPATIBILITY)
// ============================================================================

/// Legacy streaming writer for the event table.
pub struct EventExportWriter<W: Write> {
    writer: W,
    format: ExportFormat,
}

impl<W: Write> EventExportWriter<W> {
    pub fn new(writer: W, format: ExportFormat) -> Self {
        Self { writer, format }
    }

    pub fn write_provenance(&mut self, prov: &ExportProvenance) -> std::io::Result<()> {
        let json = serde_json::to_string(prov).unwrap_or_default();
        match self.format {
            ExportFormat::JsonLines => writeln!(self.writer, "{{\"provenance\":{json}}}"),
            ExportFormat::Csv => writeln!(self.writer, "# PROVENANCE: {json}"),
        }
    }

    pub fn write_event_row(
        &mut self,
        event_id: &str,
        tick: u64,
        kind: &str,
        payload: &str,
    ) -> std::io::Result<()> {
        match self.format {
            ExportFormat::JsonLines => writeln!(
                self.writer,
                "{{\"event_id\":{},\"tick\":{tick},\"kind\":{},\"payload\":{}}}",
                serde_json::to_string(event_id).unwrap_or_default(),
                serde_json::to_string(kind).unwrap_or_default(),
                serde_json::to_string(payload).unwrap_or_default()
            ),
            ExportFormat::Csv => {
                writeln!(self.writer, "event_id,tick,kind,payload")?;
                writeln!(self.writer, "{event_id},{tick},{kind},{payload}")
            }
        }
    }
}

/// Legacy export writer for streaming CSV and JSONLines metrics tables.
pub struct MetricExportWriter<W: Write> {
    writer: W,
    format: ExportFormat,
    header_written: bool,
}

impl<W: Write> MetricExportWriter<W> {
    pub fn new(writer: W, format: ExportFormat) -> Self {
        Self {
            writer,
            format,
            header_written: false,
        }
    }

    pub fn write_provenance(&mut self, prov: &ExportProvenance) -> std::io::Result<()> {
        match self.format {
            ExportFormat::Csv => {
                writeln!(
                    self.writer,
                    "# PROVENANCE: run_id={},seed={},config_digest={},source_revision={},source_tree_digest={},schema_version={},authority_decisions={},conservation_tolerances={}",
                    prov.run_id,
                    prov.seed,
                    prov.config_digest,
                    prov.source_revision,
                    prov.source_tree_digest,
                    prov.schema_version,
                    serde_json::to_string(&prov.authority_decisions).unwrap_or_default(),
                    serde_json::to_string(&prov.conservation_tolerances).unwrap_or_default()
                )
            }
            ExportFormat::JsonLines => {
                let json = serde_json::to_string(&prov).unwrap_or_default();
                writeln!(self.writer, "{{\"provenance\":{json}}}")
            }
        }
    }

    pub fn write_metric_row(
        &mut self,
        tick: u64,
        metric_name: &str,
        value: f64,
    ) -> std::io::Result<()> {
        match self.format {
            ExportFormat::Csv => {
                if !self.header_written {
                    writeln!(self.writer, "tick,metric_name,value")?;
                    self.header_written = true;
                }
                writeln!(self.writer, "{tick},{metric_name},{value}")
            }
            ExportFormat::JsonLines => {
                writeln!(
                    self.writer,
                    "{{\"tick\":{tick},\"metric\":\"{metric_name}\",\"value\":{value}}}"
                )
            }
        }
    }
}

// ============================================================================
// DATABASE STREAMING EXPORTER (bd-2z0.5.6)
// ============================================================================

/// Stream rows from a finished run database into a `CoreTableExportWriter` in
/// bounded pages (O(1) memory), stamping an atomic completion receipt.
pub fn export_storage_table<W: Write>(
    reader: &StorageReader,
    table: ExportTable,
    format: ExportFormat,
    writer: W,
) -> Result<ExportReceipt, StorageError> {
    let mut export_writer = CoreTableExportWriter::new_with_format(writer, format);
    let manifest = reader.run_manifest()?;
    let run_id = reader.run_id();

    // Embed canonical provenance header
    let prov = ExportProvenance {
        run_id: run_id.to_string(),
        seed: manifest.root_seed,
        config_digest: manifest.config_digest.clone(),
        source_revision: manifest
            .source_revision
            .clone()
            .unwrap_or_else(|| "unknown".into()),
        source_tree_digest: manifest
            .source_tree_digest
            .clone()
            .unwrap_or_else(|| "unknown".into()),
        authority_decisions: vec!["persisted_storage_export".into()],
        conservation_tolerances: None,
        schema_version: table.current_version(),
        exported_at_utc: current_utc_iso8601(),
    };
    export_writer
        .write_provenance(&prov)
        .map_err(|e| StorageError::InvalidData {
            context: "export_storage_table.write_provenance",
            reason: e.to_string(),
        })?;

    match table {
        ExportTable::Run => {
            let row = RunExportRow {
                run_id: manifest.run_id.to_string(),
                manifest_schema_version: manifest.manifest_schema_version as u32,
                scenario_id: manifest.scenario_id,
                scenario_version: manifest.scenario_version as u32,
                config_digest: manifest.config_digest,
                root_seed_hex: format!("{:016x}", manifest.root_seed),
                source_revision: manifest.source_revision,
                source_tree_digest: manifest.source_tree_digest,
                started_at_unix_ms: manifest.started_at_unix_ms,
                reproducible: manifest.reproducible,
            };
            export_writer
                .write_run_row(&row)
                .map_err(|e| StorageError::InvalidData {
                    context: "export_storage_table.write_run_row",
                    reason: e.to_string(),
                })?;
        }
        ExportTable::Agent => {
            let limit = 1024;
            let mut offset = 0;
            loop {
                let rows = reader.connection()?.query_with_params(
                    "SELECT run_id, tick, agent_uid, generation, age, position_x, position_y, velocity_x, velocity_y, heading, health, energy, herbivore_tendency, brain_binding
                     FROM agents
                     WHERE run_id = ?1
                     ORDER BY tick ASC, agent_uid ASC
                     LIMIT ?2 OFFSET ?3",
                    &[sqlite_run_id(run_id), (limit as i64).into(), (offset as i64).into()],
                )?;
                if rows.is_empty() {
                    break;
                }
                for r in &rows {
                    let row = AgentExportRow {
                        run_id: decode_column(r, 0, "agents.run_id")?,
                        tick: checked_u64("agents.tick", decode_column(r, 1, "agents.tick")?)?,
                        agent_uid: checked_u64(
                            "agents.agent_uid",
                            decode_column(r, 2, "agents.agent_uid")?,
                        )?,
                        generation: decode_column::<i64>(r, 3, "agents.generation")? as u32,
                        age: decode_column::<i64>(r, 4, "agents.age")? as u32,
                        pos_x: decode_column::<f64>(r, 5, "agents.position_x")? as f32,
                        pos_y: decode_column::<f64>(r, 6, "agents.position_y")? as f32,
                        vel_x: decode_column::<f64>(r, 7, "agents.velocity_x")? as f32,
                        vel_y: decode_column::<f64>(r, 8, "agents.velocity_y")? as f32,
                        heading: decode_column::<f64>(r, 9, "agents.heading")? as f32,
                        health: decode_column::<f64>(r, 10, "agents.health")? as f32,
                        energy: decode_column::<f64>(r, 11, "agents.energy")? as f32,
                        herbivore_tendency: decode_column::<f64>(
                            r,
                            12,
                            "agents.herbivore_tendency",
                        )? as f32,
                        brain_binding: decode_column(r, 13, "agents.brain_binding")?,
                    };
                    export_writer
                        .write_agent_row(&row)
                        .map_err(|e| StorageError::InvalidData {
                            context: "export_storage_table.write_agent_row",
                            reason: e.to_string(),
                        })?;
                }
                offset += rows.len();
                if rows.len() < limit {
                    break;
                }
            }
        }
        ExportTable::Lineage => {
            let limit = 1024;
            let mut offset = 0;
            loop {
                let rows = reader.connection()?.query_with_params(
                    "SELECT run_id, child_agent_uid, parent_agent_uid, parent_ordinal, relationship, birth_tick
                     FROM lineage_edges
                     WHERE run_id = ?1
                     ORDER BY birth_tick ASC, child_agent_uid ASC, parent_ordinal ASC
                     LIMIT ?2 OFFSET ?3",
                    &[sqlite_run_id(run_id), (limit as i64).into(), (offset as i64).into()],
                )?;
                if rows.is_empty() {
                    break;
                }
                for r in &rows {
                    let row = LineageExportRow {
                        run_id: decode_column(r, 0, "lineage_edges.run_id")?,
                        child_agent_uid: checked_u64(
                            "lineage_edges.child_agent_uid",
                            decode_column(r, 1, "lineage_edges.child_agent_uid")?,
                        )?,
                        parent_agent_uid: checked_u64(
                            "lineage_edges.parent_agent_uid",
                            decode_column(r, 2, "lineage_edges.parent_agent_uid")?,
                        )?,
                        parent_ordinal: decode_column::<i64>(r, 3, "lineage_edges.parent_ordinal")?
                            as u32,
                        relationship: decode_column(r, 4, "lineage_edges.relationship")?,
                        birth_tick: checked_u64(
                            "lineage_edges.birth_tick",
                            decode_column(r, 5, "lineage_edges.birth_tick")?,
                        )?,
                    };
                    export_writer.write_lineage_row(&row).map_err(|e| {
                        StorageError::InvalidData {
                            context: "export_storage_table.write_lineage_row",
                            reason: e.to_string(),
                        }
                    })?;
                }
                offset += rows.len();
                if rows.len() < limit {
                    break;
                }
            }
        }
        ExportTable::Event => {
            let limit = 1024;
            let mut offset = 0;
            loop {
                let rows = reader.connection()?.query_with_params(
                    "SELECT run_id, tick, seq, scope, event_type, payload
                     FROM replay_events
                     WHERE run_id = ?1
                     ORDER BY tick ASC, seq ASC
                     LIMIT ?2 OFFSET ?3",
                    &[
                        sqlite_run_id(run_id),
                        (limit as i64).into(),
                        (offset as i64).into(),
                    ],
                )?;
                if rows.is_empty() {
                    break;
                }
                for r in &rows {
                    let seq: i64 = decode_column(r, 2, "replay_events.seq")?;
                    let scope: String = decode_column(r, 3, "replay_events.scope")?;
                    let event_type: String = decode_column(r, 4, "replay_events.event_type")?;
                    let row = EventExportRow {
                        run_id: decode_column(r, 0, "replay_events.run_id")?,
                        tick: checked_u64(
                            "replay_events.tick",
                            decode_column(r, 1, "replay_events.tick")?,
                        )?,
                        seq: checked_u64("replay_events.seq", seq)?,
                        event_id: format!("evt-{seq}"),
                        kind: format!("{scope}:{event_type}"),
                        payload: decode_column(r, 5, "replay_events.payload")?,
                    };
                    export_writer
                        .write_event_row(&row)
                        .map_err(|e| StorageError::InvalidData {
                            context: "export_storage_table.write_event_row",
                            reason: e.to_string(),
                        })?;
                }
                offset += rows.len();
                if rows.len() < limit {
                    break;
                }
            }
        }
        ExportTable::Metric => {
            let limit = 1024;
            let mut offset = 0;
            loop {
                let rows = reader.connection()?.query_with_params(
                    "SELECT run_id, tick, name, value
                     FROM metrics
                     WHERE run_id = ?1
                     ORDER BY tick ASC, name ASC
                     LIMIT ?2 OFFSET ?3",
                    &[
                        sqlite_run_id(run_id),
                        (limit as i64).into(),
                        (offset as i64).into(),
                    ],
                )?;
                if rows.is_empty() {
                    break;
                }
                for r in &rows {
                    let row = MetricExportRow {
                        run_id: decode_column(r, 0, "metrics.run_id")?,
                        tick: checked_u64("metrics.tick", decode_column(r, 1, "metrics.tick")?)?,
                        name: decode_column(r, 2, "metrics.name")?,
                        value: decode_column::<f64>(r, 3, "metrics.value")?,
                    };
                    export_writer.write_metric_row(&row).map_err(|e| {
                        StorageError::InvalidData {
                            context: "export_storage_table.write_metric_row",
                            reason: e.to_string(),
                        }
                    })?;
                }
                offset += rows.len();
                if rows.len() < limit {
                    break;
                }
            }
        }
    }

    let receipt = export_writer
        .finish(table)
        .map_err(|e| StorageError::InvalidData {
            context: "export_storage_table.finish",
            reason: e.to_string(),
        })?;

    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_metric_export_pipeline() {
        let mut buf = Vec::new();
        let mut writer = MetricExportWriter::new(&mut buf, ExportFormat::Csv);

        let prov = ExportProvenance {
            run_id: "run-42".into(),
            seed: 2026,
            config_digest: "blake3:abc123hash".into(),
            source_revision: "git:deadbeef".into(),
            source_tree_digest: "blake3:tree".into(),
            authority_decisions: vec!["conservation:pass".into(), "durable_watermark:42".into()],
            conservation_tolerances: Some(serde_json::json!({
                "per_tick_relative": 1.0e-6,
                "cumulative_relative": 1.0e-5,
            })),
            schema_version: 1,
            exported_at_utc: "2026-07-22T15:00:00Z".into(),
        };

        writer.write_provenance(&prov).unwrap();
        writer.write_metric_row(100, "population", 42.0).unwrap();
        writer.write_metric_row(101, "population", 45.0).unwrap();

        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("# PROVENANCE: run_id=run-42"));
        assert!(output.contains("source_tree_digest=blake3:tree"));
        assert!(output.contains("conservation:pass"));
        assert!(output.contains("tick,metric_name,value"));
        assert!(output.contains("100,population,42"));
    }

    #[test]
    fn jsonl_metric_export_round_trips_provenance_and_row() {
        let mut buf = Vec::new();
        let mut writer = MetricExportWriter::new(&mut buf, ExportFormat::JsonLines);
        let prov = ExportProvenance {
            run_id: "run-round-trip".into(),
            seed: 7,
            config_digest: "blake3:config".into(),
            source_revision: "git:abc123".into(),
            source_tree_digest: "blake3:tree".into(),
            authority_decisions: vec!["conservation:pass".into(), "durable_watermark:9".into()],
            conservation_tolerances: Some(serde_json::json!({"per_tick_relative": 1.0e-6})),
            schema_version: 1,
            exported_at_utc: "2026-07-27T00:00:00Z".into(),
        };
        writer.write_provenance(&prov).unwrap();
        writer.write_metric_row(9, "population", 12.5).unwrap();

        let output = String::from_utf8(buf).unwrap();
        let mut lines = output.lines();
        let envelope: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        let exported = &envelope["provenance"];
        assert_eq!(exported["run_id"], "run-round-trip");
        assert_eq!(exported["config_digest"], "blake3:config");
        assert_eq!(exported["source_tree_digest"], "blake3:tree");
        assert_eq!(exported["authority_decisions"][0], "conservation:pass");
        assert_eq!(
            exported["conservation_tolerances"]["per_tick_relative"],
            1.0e-6
        );

        let row: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(row["tick"], 9);
        assert_eq!(row["metric"], "population");
        assert_eq!(row["value"], 12.5);
    }

    #[test]
    fn jsonl_event_export_reuses_provenance_contract() {
        let mut buf = Vec::new();
        let mut writer = EventExportWriter::new(&mut buf, ExportFormat::JsonLines);
        let prov = ExportProvenance {
            run_id: "run-events".into(),
            seed: 11,
            config_digest: "blake3:config-events".into(),
            source_revision: "git:def456".into(),
            source_tree_digest: "blake3:tree-events".into(),
            authority_decisions: vec!["durable_watermark:4".into()],
            conservation_tolerances: None,
            schema_version: 1,
            exported_at_utc: "2026-07-27T00:00:00Z".into(),
        };
        writer.write_provenance(&prov).unwrap();
        writer
            .write_event_row("evt-1", 4, "combat", "unicode: π")
            .unwrap();

        let output = String::from_utf8(buf).unwrap();
        let mut lines = output.lines();
        let envelope: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(envelope["provenance"]["run_id"], "run-events");
        assert_eq!(
            envelope["provenance"]["config_digest"],
            "blake3:config-events"
        );
        assert_eq!(
            envelope["provenance"]["source_tree_digest"],
            "blake3:tree-events"
        );
        let row: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(row["event_id"], "evt-1");
        assert_eq!(row["kind"], "combat");
        assert_eq!(row["payload"], "unicode: π");
    }

    #[test]
    fn jsonl_core_tables_round_trip_shared_provenance_envelope() {
        let mut buf = Vec::new();
        let mut writer = CoreTableExportWriter::new(&mut buf);
        let prov = ExportProvenance {
            run_id: "run-core".into(),
            seed: 19,
            config_digest: "blake3:core-config".into(),
            source_revision: "git:core".into(),
            source_tree_digest: "blake3:core-tree".into(),
            authority_decisions: vec!["durable_watermark:12".into()],
            conservation_tolerances: None,
            schema_version: 1,
            exported_at_utc: "2026-07-27T00:00:00Z".into(),
        };
        writer.write_provenance(&prov).unwrap();
        writer
            .write_row("run", &serde_json::json!({"run_id":"run-core","tick":12}))
            .unwrap();
        writer
            .write_row(
                "agent",
                &serde_json::json!({"agent_id":"a-1","run_id":"run-core"}),
            )
            .unwrap();
        writer
            .write_row(
                "lineage",
                &serde_json::json!({"child":"a-1","parent":"a-0"}),
            )
            .unwrap();

        let output = String::from_utf8(buf).unwrap();
        let mut lines = output.lines();
        let provenance: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
        assert_eq!(
            provenance["provenance"]["source_tree_digest"],
            "blake3:core-tree"
        );
        for (table, key, value) in [
            ("run", "run_id", "run-core"),
            ("agent", "agent_id", "a-1"),
            ("lineage", "child", "a-1"),
        ] {
            let exported: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
            assert_eq!(exported["table"], table);
            assert_eq!(exported["row"][key], value);
        }
    }

    /// A schema bump must read as "different question", never as corruption.
    #[test]
    fn bd_2z0_5_6_a_version_bump_is_not_a_digest_mismatch() {
        let v1 = SchemaId {
            table: ExportTable::Run,
            version: 1,
        };
        let v2 = SchemaId {
            table: ExportTable::Run,
            version: 2,
        };
        let verdict = compare_schema(v1, v2);
        assert_eq!(
            verdict,
            DigestComparability::DifferentVersion {
                expected: 1,
                found: 2
            }
        );
        assert!(
            !verdict.is_comparable(),
            "a cross-version digest comparison is not a meaningful question"
        );
        assert_ne!(
            verdict,
            DigestComparability::DifferentTable {
                expected: ExportTable::Run,
                found: ExportTable::Run
            },
            "a version bump must not be reported as a table mix-up"
        );
    }

    /// Comparing two different tables is meaningless regardless of version.
    #[test]
    fn bd_2z0_5_6_different_tables_never_compare() {
        let run = SchemaId::current(ExportTable::Run);
        let metric = SchemaId::current(ExportTable::Metric);
        assert_eq!(run.version, metric.version, "both start at 1 today");
        assert_eq!(
            compare_schema(run, metric),
            DigestComparability::DifferentTable {
                expected: ExportTable::Run,
                found: ExportTable::Metric
            },
            "equal version numbers must not make two different tables comparable"
        );
    }

    /// The same table at the same version is the ONLY comparable case.
    #[test]
    fn bd_2z0_5_6_only_identical_schema_ids_are_comparable() {
        for table in ExportTable::ALL {
            let id = SchemaId::current(table);
            assert!(
                compare_schema(id, id).is_comparable(),
                "{} must compare with itself",
                table.as_str()
            );
            for other in ExportTable::ALL {
                if other != table {
                    assert!(
                        !compare_schema(id, SchemaId::current(other)).is_comparable(),
                        "{} must not compare with {}",
                        table.as_str(),
                        other.as_str()
                    );
                }
            }
        }
    }

    /// Versions are PER TABLE: bumping one must not disturb the others.
    #[test]
    fn bd_2z0_5_6_schema_versions_are_scoped_per_table() {
        // The whole point of per-table versioning is that a `run` change leaves
        // retained `metric` artifacts comparable. Simulate the bump.
        let metric_before = SchemaId::current(ExportTable::Metric);
        let bumped_run = SchemaId {
            table: ExportTable::Run,
            version: ExportTable::Run.current_version() + 1,
        };
        assert!(
            compare_schema(metric_before, SchemaId::current(ExportTable::Metric)).is_comparable(),
            "bumping the run table must not invalidate metric artifacts"
        );
        assert!(!compare_schema(SchemaId::current(ExportTable::Run), bumped_run).is_comparable());
    }

    /// Table names are stable identifiers and must not collide.
    #[test]
    fn bd_2z0_5_6_table_names_are_stable_and_distinct() {
        let mut names: Vec<&str> = ExportTable::ALL.iter().map(|t| t.as_str()).collect();
        assert_eq!(
            names,
            ["run", "agent", "lineage", "event", "metric"],
            "these strings appear in retained artifacts; changing one is a schema break"
        );
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            ExportTable::ALL.len(),
            "names must be distinct"
        );
    }

    // ========================================================================
    // COMPREHENSIVE NEW TESTS FOR bd-2z0.5.6 ACCEPTANCE
    // ========================================================================

    fn sample_provenance() -> ExportProvenance {
        ExportProvenance {
            run_id: "run-test-e2e-001".into(),
            seed: 4242,
            config_digest: "blake3:c0ffee1234".into(),
            source_revision: "git:abcd7890".into(),
            source_tree_digest: "blake3:treefeed".into(),
            authority_decisions: vec!["authority:accepted".into()],
            conservation_tolerances: Some(serde_json::json!({"per_tick": 1e-6})),
            schema_version: 1,
            exported_at_utc: "2026-09-03T00:00:00Z".into(),
        }
    }

    #[test]
    fn test_all_five_tables_csv_roundtrip_with_receipt() {
        let prov = sample_provenance();

        // 1. Run Table
        {
            let mut buf = Vec::new();
            let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
            writer.write_provenance(&prov).unwrap();
            writer
                .write_run_row(&RunExportRow {
                    run_id: "run-test-e2e-001".into(),
                    manifest_schema_version: 1,
                    scenario_id: "foraging-primary".into(),
                    scenario_version: 2,
                    config_digest: "blake3:c0ffee1234".into(),
                    root_seed_hex: "0000000000001092".into(),
                    source_revision: Some("git:abcd7890".into()),
                    source_tree_digest: Some("blake3:treefeed".into()),
                    started_at_unix_ms: 1770000000000,
                    reproducible: true,
                })
                .unwrap();
            let receipt = writer.finish(ExportTable::Run).unwrap();
            assert_eq!(receipt.row_count, 1);
            let content = String::from_utf8(buf).unwrap();
            let verified = verify_export_receipt(&content, ExportTable::Run).unwrap();
            assert_eq!(verified.row_count, 1);
            assert_eq!(verified.checksum_blake3, receipt.checksum_blake3);
        }

        // 2. Agent Table
        {
            let mut buf = Vec::new();
            let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
            writer.write_provenance(&prov).unwrap();
            writer
                .write_agent_row(&AgentExportRow {
                    run_id: "run-test-e2e-001".into(),
                    tick: 50,
                    agent_uid: 101,
                    generation: 3,
                    age: 24,
                    pos_x: 12.5,
                    pos_y: 33.25,
                    vel_x: 0.1,
                    vel_y: -0.2,
                    heading: 1.57,
                    health: 0.95,
                    energy: 150.0,
                    herbivore_tendency: 0.85,
                    brain_binding: "mlp".into(),
                })
                .unwrap();
            let receipt = writer.finish(ExportTable::Agent).unwrap();
            assert_eq!(receipt.row_count, 1);
            let content = String::from_utf8(buf).unwrap();
            let verified = verify_export_receipt(&content, ExportTable::Agent).unwrap();
            assert_eq!(verified.row_count, 1);
        }

        // 3. Lineage Table
        {
            let mut buf = Vec::new();
            let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
            writer.write_provenance(&prov).unwrap();
            writer
                .write_lineage_row(&LineageExportRow {
                    run_id: "run-test-e2e-001".into(),
                    child_agent_uid: 101,
                    parent_agent_uid: 42,
                    parent_ordinal: 0,
                    relationship: "asexual_clone".into(),
                    birth_tick: 26,
                })
                .unwrap();
            let receipt = writer.finish(ExportTable::Lineage).unwrap();
            assert_eq!(receipt.row_count, 1);
            let content = String::from_utf8(buf).unwrap();
            let verified = verify_export_receipt(&content, ExportTable::Lineage).unwrap();
            assert_eq!(verified.row_count, 1);
        }

        // 4. Event Table
        {
            let mut buf = Vec::new();
            let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
            writer.write_provenance(&prov).unwrap();
            writer
                .write_event_row(&EventExportRow {
                    run_id: "run-test-e2e-001".into(),
                    tick: 30,
                    seq: 1,
                    event_id: "evt-1".into(),
                    kind: "speciation:detected".into(),
                    payload: "{\"cluster_id\":2}".into(),
                })
                .unwrap();
            let receipt = writer.finish(ExportTable::Event).unwrap();
            assert_eq!(receipt.row_count, 1);
            let content = String::from_utf8(buf).unwrap();
            let verified = verify_export_receipt(&content, ExportTable::Event).unwrap();
            assert_eq!(verified.row_count, 1);
        }

        // 5. Metric Table
        {
            let mut buf = Vec::new();
            let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
            writer.write_provenance(&prov).unwrap();
            writer
                .write_metric_row(&MetricExportRow {
                    run_id: "run-test-e2e-001".into(),
                    tick: 50,
                    name: "population".into(),
                    value: 42.0,
                })
                .unwrap();
            let receipt = writer.finish(ExportTable::Metric).unwrap();
            assert_eq!(receipt.row_count, 1);
            let content = String::from_utf8(buf).unwrap();
            let verified = verify_export_receipt(&content, ExportTable::Metric).unwrap();
            assert_eq!(verified.row_count, 1);
        }
    }

    #[test]
    fn test_all_five_tables_jsonl_roundtrip_with_receipt() {
        let prov = sample_provenance();

        for table in ExportTable::ALL {
            let mut buf = Vec::new();
            let mut writer =
                CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::JsonLines);
            writer.write_provenance(&prov).unwrap();

            match table {
                ExportTable::Run => {
                    writer
                        .write_run_row(&RunExportRow {
                            run_id: "run-test-e2e-001".into(),
                            manifest_schema_version: 1,
                            scenario_id: "foraging-primary".into(),
                            scenario_version: 2,
                            config_digest: "blake3:c0ffee1234".into(),
                            root_seed_hex: "0000000000001092".into(),
                            source_revision: None,
                            source_tree_digest: None,
                            started_at_unix_ms: 1770000000000,
                            reproducible: true,
                        })
                        .unwrap();
                }
                ExportTable::Agent => {
                    writer
                        .write_agent_row(&AgentExportRow {
                            run_id: "run-test-e2e-001".into(),
                            tick: 1,
                            agent_uid: 1,
                            generation: 0,
                            age: 1,
                            pos_x: 0.0,
                            pos_y: 0.0,
                            vel_x: 0.0,
                            vel_y: 0.0,
                            heading: 0.0,
                            health: 1.0,
                            energy: 100.0,
                            herbivore_tendency: 0.5,
                            brain_binding: "neuro".into(),
                        })
                        .unwrap();
                }
                ExportTable::Lineage => {
                    writer
                        .write_lineage_row(&LineageExportRow {
                            run_id: "run-test-e2e-001".into(),
                            child_agent_uid: 2,
                            parent_agent_uid: 1,
                            parent_ordinal: 0,
                            relationship: "clone".into(),
                            birth_tick: 1,
                        })
                        .unwrap();
                }
                ExportTable::Event => {
                    writer
                        .write_event_row(&EventExportRow {
                            run_id: "run-test-e2e-001".into(),
                            tick: 1,
                            seq: 0,
                            event_id: "evt-0".into(),
                            kind: "spawn:founder".into(),
                            payload: "{}".into(),
                        })
                        .unwrap();
                }
                ExportTable::Metric => {
                    writer
                        .write_metric_row(&MetricExportRow {
                            run_id: "run-test-e2e-001".into(),
                            tick: 1,
                            name: "total_energy".into(),
                            value: 500.0,
                        })
                        .unwrap();
                }
            }

            let receipt = writer.finish(table).unwrap();
            assert_eq!(receipt.row_count, 1);
            let content = String::from_utf8(buf).unwrap();
            let verified = verify_export_receipt(&content, table).unwrap();
            assert_eq!(verified.row_count, 1);
            assert_eq!(verified.checksum_blake3, receipt.checksum_blake3);
        }
    }

    #[test]
    fn test_interrupted_write_missing_receipt_detection() {
        let prov = sample_provenance();
        let mut buf = Vec::new();
        let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
        writer.write_provenance(&prov).unwrap();
        writer
            .write_metric_row(&MetricExportRow {
                run_id: "run-interrupted".into(),
                tick: 1,
                name: "pop".into(),
                value: 10.0,
            })
            .unwrap();
        // Crucially: writer.finish() is NEVER called (simulating power failure / crash / truncated output)
        drop(writer);

        let content = String::from_utf8(buf).unwrap();
        let err = verify_export_receipt(&content, ExportTable::Metric).unwrap_err();
        assert_eq!(err, ExportVerificationError::MissingReceipt);
    }

    #[test]
    fn test_row_count_tampering_detection() {
        let prov = sample_provenance();
        let mut buf = Vec::new();
        let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
        writer.write_provenance(&prov).unwrap();
        writer
            .write_metric_row(&MetricExportRow {
                run_id: "run-tamper".into(),
                tick: 1,
                name: "pop".into(),
                value: 10.0,
            })
            .unwrap();
        writer.finish(ExportTable::Metric).unwrap();

        let content = String::from_utf8(buf).unwrap();
        // Tamper with receipt row_count
        let tampered = content.replace("row_count=1", "row_count=2");
        let err = verify_export_receipt(&tampered, ExportTable::Metric).unwrap_err();
        assert_eq!(
            err,
            ExportVerificationError::RowCountMismatch {
                reported: 2,
                counted: 1,
            }
        );
    }

    #[test]
    fn test_data_corruption_checksum_detection() {
        let prov = sample_provenance();
        let mut buf = Vec::new();
        let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
        writer.write_provenance(&prov).unwrap();
        writer
            .write_metric_row(&MetricExportRow {
                run_id: "run-corrupt".into(),
                tick: 1,
                name: "pop".into(),
                value: 10.0,
            })
            .unwrap();
        writer.finish(ExportTable::Metric).unwrap();

        let content = String::from_utf8(buf).unwrap();
        // Silently alter data payload
        let corrupted = content.replace("10", "11");
        let err = verify_export_receipt(&corrupted, ExportTable::Metric).unwrap_err();
        assert!(matches!(
            err,
            ExportVerificationError::ChecksumMismatch { .. }
        ));
    }

    #[test]
    fn test_table_mismatch_detection() {
        let prov = sample_provenance();
        let mut buf = Vec::new();
        let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
        writer.write_provenance(&prov).unwrap();
        writer
            .write_metric_row(&MetricExportRow {
                run_id: "run-mismatch".into(),
                tick: 1,
                name: "pop".into(),
                value: 10.0,
            })
            .unwrap();
        writer.finish(ExportTable::Metric).unwrap();

        let content = String::from_utf8(buf).unwrap();
        let err = verify_export_receipt(&content, ExportTable::Agent).unwrap_err();
        assert_eq!(
            err,
            ExportVerificationError::TableMismatch {
                expected: ExportTable::Agent,
                found: ExportTable::Metric,
            }
        );
    }

    #[test]
    fn test_non_finite_float_policy_csv_and_jsonl() {
        // Test format_f32_csv & parse_f32_csv
        assert_eq!(format_f32_csv(f32::NAN), "NaN");
        assert_eq!(format_f32_csv(f32::INFINITY), "Infinity");
        assert_eq!(format_f32_csv(f32::NEG_INFINITY), "-Infinity");
        assert_eq!(format_f32_csv(12.345), "12.345");

        assert!(parse_f32_csv("NaN").unwrap().is_nan());
        assert_eq!(parse_f32_csv("Infinity").unwrap(), f32::INFINITY);
        assert_eq!(parse_f32_csv("-Infinity").unwrap(), f32::NEG_INFINITY);

        // Test format_f64_csv & parse_f64_csv
        assert_eq!(format_f64_csv(f64::NAN), "NaN");
        assert_eq!(format_f64_csv(f64::INFINITY), "Infinity");
        assert_eq!(format_f64_csv(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(format_f64_csv(67.89), "67.89");

        assert!(parse_f64_csv("NaN").unwrap().is_nan());
        assert_eq!(parse_f64_csv("Infinity").unwrap(), f64::INFINITY);
        assert_eq!(parse_f64_csv("-Infinity").unwrap(), f64::NEG_INFINITY);

        // In CSV export stream
        let prov = sample_provenance();
        let mut buf = Vec::new();
        let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
        writer.write_provenance(&prov).unwrap();
        writer
            .write_metric_row(&MetricExportRow {
                run_id: "run-nan".into(),
                tick: 1,
                name: "diverged_loss".into(),
                value: f64::NAN,
            })
            .unwrap();
        let receipt = writer.finish(ExportTable::Metric).unwrap();

        let content = String::from_utf8(buf).unwrap();
        assert!(content.contains("run-nan,1,diverged_loss,NaN"));
        let verified = verify_export_receipt(&content, ExportTable::Metric).unwrap();
        assert_eq!(verified.row_count, receipt.row_count);
    }

    #[test]
    fn test_unicode_and_rfc4180_escaping() {
        assert_eq!(csv_escape("simple"), "simple");
        assert_eq!(csv_escape("hello, world"), "\"hello, world\"");
        assert_eq!(csv_escape("he said \"yes\""), "\"he said \"\"yes\"\"\"");
        assert_eq!(csv_escape("line1\nline2"), "\"line1\nline2\"");
        assert_eq!(csv_escape("emoji 🧬 organisms"), "emoji 🧬 organisms");
        assert_eq!(csv_escape("中文测试"), "中文测试");

        let fields = parse_csv_line("a,\"b,c\",\"d\"\"e\",f");
        assert_eq!(fields, vec!["a", "b,c", "d\"e", "f"]);

        // Verify that event with complex payload round-trips through writer and receipt
        let prov = sample_provenance();
        let mut buf = Vec::new();
        let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
        writer.write_provenance(&prov).unwrap();
        writer
            .write_event_row(&EventExportRow {
                run_id: "run-unicode".into(),
                tick: 10,
                seq: 1,
                event_id: "evt-unicode".into(),
                kind: "narrative:citation".into(),
                payload: "Agent 🧬 encountered, \"quoted\", and said: 'hello\nworld' in 北京"
                    .into(),
            })
            .unwrap();
        let receipt = writer.finish(ExportTable::Event).unwrap();

        let content = String::from_utf8(buf).unwrap();
        let verified = verify_export_receipt(&content, ExportTable::Event).unwrap();
        assert_eq!(verified.row_count, receipt.row_count);
    }

    #[test]
    fn test_boundary_and_extreme_values() {
        let prov = sample_provenance();
        let mut buf = Vec::new();
        let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
        writer.write_provenance(&prov).unwrap();

        writer
            .write_agent_row(&AgentExportRow {
                run_id: "run-boundary".into(),
                tick: u64::MAX,
                agent_uid: u64::MAX,
                generation: u32::MAX,
                age: u32::MAX,
                pos_x: f32::MIN_POSITIVE,
                pos_y: -1.0e30,
                vel_x: f32::MAX,
                vel_y: f32::MIN,
                heading: -std::f32::consts::PI,
                health: 0.0,
                energy: 0.0,
                herbivore_tendency: 1.0,
                brain_binding: "".into(),
            })
            .unwrap();

        let receipt = writer.finish(ExportTable::Agent).unwrap();
        assert_eq!(receipt.row_count, 1);
        let content = String::from_utf8(buf).unwrap();
        let verified = verify_export_receipt(&content, ExportTable::Agent).unwrap();
        assert_eq!(verified.row_count, 1);
    }

    #[test]
    fn test_empty_tables_produce_valid_receipt() {
        let prov = sample_provenance();
        for table in ExportTable::ALL {
            let mut buf = Vec::new();
            let mut writer = CoreTableExportWriter::new_with_format(&mut buf, ExportFormat::Csv);
            writer.write_provenance(&prov).unwrap();
            let receipt = writer.finish(table).unwrap();
            assert_eq!(receipt.row_count, 0);

            let content = String::from_utf8(buf).unwrap();
            let verified = verify_export_receipt(&content, table).unwrap();
            assert_eq!(verified.row_count, 0);
            assert_eq!(verified.table, table);
        }
    }

    #[test]
    fn test_arrow_parquet_mapping_contract_exhaustiveness() {
        for table in ExportTable::ALL {
            let schema = table.arrow_schema();
            assert!(
                !schema.is_empty(),
                "table {} must have non-empty Arrow schema",
                table.as_str()
            );

            for col in schema {
                assert!(!col.name.is_empty(), "column name must not be empty");
                assert!(
                    !col.units.is_empty(),
                    "column {} must document its units",
                    col.name
                );
                assert!(
                    !col.doc.is_empty(),
                    "column {} must document its purpose",
                    col.name
                );

                // Column name must be valid snake_case identifier
                assert!(
                    col.name.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                    "column {} in table {} is not valid snake_case",
                    col.name,
                    table.as_str()
                );
            }
        }
    }
}
