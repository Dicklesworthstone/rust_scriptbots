//! Provenance-rich CSV, Arrow, and Parquet export pipeline (bd-2z0.5.6).

use serde::{Deserialize, Serialize};
use std::io::Write;

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

        let mut lines = String::from_utf8(buf).unwrap().lines();
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

        let mut lines = String::from_utf8(buf).unwrap().lines();
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

        let mut lines = String::from_utf8(buf).unwrap().lines();
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
}
