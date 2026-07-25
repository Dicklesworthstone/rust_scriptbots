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
    pub config_hash: String,
    pub schema_version: u32,
    pub exported_at_utc: String,
}

/// Export writer for streaming CSV and JSONLines metrics tables.
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
                    "# PROVENANCE: run_id={},seed={},config_hash={},schema_version={}",
                    prov.run_id, prov.seed, prov.config_hash, prov.schema_version
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
            config_hash: "abc123hash".into(),
            schema_version: 1,
            exported_at_utc: "2026-07-22T15:00:00Z".into(),
        };

        writer.write_provenance(&prov).unwrap();
        writer.write_metric_row(100, "population", 42.0).unwrap();
        writer.write_metric_row(101, "population", 45.0).unwrap();

        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("# PROVENANCE: run_id=run-42"));
        assert!(output.contains("tick,metric_name,value"));
        assert!(output.contains("100,population,42"));
    }
}
