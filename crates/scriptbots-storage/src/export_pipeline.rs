//! Provenance-rich CSV, Arrow, and Parquet export pipeline (bd-2z0.5.6).

use serde::{Deserialize, Serialize};
use std::io::Write;

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

/// Export writer for streaming CSV and JSONLines metrics tables.
pub struct MetricExportWriter<W: Write> {
    writer: W,
    format: ExportFormat,
    header_written: bool,
}

/// Streaming writer for the canonical event table, sharing the metric table's
/// provenance envelope and schema-versioning rules.
pub struct EventExportWriter<W: Write> {
    writer: W,
    format: ExportFormat,
}

/// Shared bounded JSONL writer for the remaining identity-oriented core tables.
pub struct CoreTableExportWriter<W: Write> {
    writer: W,
}

impl<W: Write> CoreTableExportWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub fn write_provenance(&mut self, prov: &ExportProvenance) -> std::io::Result<()> {
        let json = serde_json::to_string(prov).unwrap_or_default();
        writeln!(self.writer, "{{\"provenance\":{json}}}")
    }

    pub fn write_row(&mut self, table: &str, row: &serde_json::Value) -> std::io::Result<()> {
        let mut envelope = serde_json::Map::new();
        envelope.insert("table".into(), serde_json::Value::String(table.into()));
        envelope.insert("row".into(), row.clone());
        writeln!(self.writer, "{}", serde_json::Value::Object(envelope))
    }
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

// Run-bundle assembly and verification deliberately do not live here (`bd-4d9j`).
//
// This module previously carried a second, parallel implementation —
// `DeterministicRunBundle` with `BUNDLE_SCHEMA_VERSION: u32 = 1`, its own
// `RunBundleManifest`, `ArtifactIndex`, and `BundleVerificationReport` — doing the same
// job as `crate::bundle` with an incompatible schema, a different manifest filename, and
// a different verification report. Two producers could each emit something called a "run
// bundle" that the other could not read.
//
// The surviving implementation is `crate::bundle`: `RunBundleV1` under the
// `scriptbots.run-bundle.v1` schema tag, which is the one wired into the shipped binary's
// `--create-bundle` / `--verify-bundle` surface. Producers that have a run database call
// `bundle::create_run_bundle`; producers that only have in-memory artifacts call
// `bundle::create_run_bundle_from_artifacts`. Both are verified by `bundle::verify_run_bundle`.
//
// What remains below is the genuinely distinct job this module is named for: streaming
// provenance-tagged metric tables (bd-2z0.5.6).

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
}
