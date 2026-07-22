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

/// Schema version for deterministic run bundles.
pub const BUNDLE_SCHEMA_VERSION: u32 = 1;

/// Package manifest describing run provenance and environment identity (bd-2z0.5.4).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunBundleManifest {
    pub schema_version: u32,
    pub run_id: String,
    pub seed: u64,
    pub created_at_utc: String,
    pub source_revision: String,
    pub source_tree_digest: String,
    pub source_tree_dirty: bool,
    pub rust_toolchain: String,
    pub cargo_lock_digest: String,
    pub target_triple: String,
    pub total_ticks: u64,
    pub final_agent_count: usize,
    pub config_hash: String,
}

/// Individual artifact record with relative path, file size, and BLAKE3 checksum.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactEntry {
    pub relative_path: String,
    pub file_size_bytes: u64,
    pub blake3_hex: String,
}

/// Index of all files included in the run bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactIndex {
    pub schema_version: u32,
    pub run_id: String,
    pub artifacts: Vec<ArtifactEntry>,
}

/// Verification summary report for an imported run bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleVerificationReport {
    pub run_id: String,
    pub verified_files: usize,
    pub total_bytes_verified: u64,
    pub is_valid: bool,
}

/// Detailed verification errors for invalid or corrupted bundles.
#[derive(Debug, thiserror::Error)]
pub enum BundleVerificationError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Missing expected file: {0}")]
    MissingFile(String),
    #[error("Checksum mismatch for {path}: expected {expected}, got {computed}")]
    ChecksumMismatch {
        path: String,
        expected: String,
        computed: String,
    },
    #[error("Schema version mismatch: expected {expected}, got {found}")]
    SchemaVersionMismatch { expected: u32, found: u32 },
}

/// Manager for assembling and verifying portable deterministic run bundles.
pub struct DeterministicRunBundle;

impl DeterministicRunBundle {
    /// Assembles a portable run bundle directory containing manifest, artifact index, and payload files.
    pub fn assemble_bundle(
        output_dir: &std::path::Path,
        manifest: RunBundleManifest,
        artifact_files: &[(&str, &[u8])],
    ) -> Result<ArtifactIndex, std::io::Error> {
        std::fs::create_dir_all(output_dir)?;

        let mut entries = Vec::new();

        for (rel_path, bytes) in artifact_files {
            let full_path = output_dir.join(rel_path);
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&full_path, bytes)?;

            let blake3_hex = blake3::hash(bytes).to_hex().to_string();
            entries.push(ArtifactEntry {
                relative_path: rel_path.to_string(),
                file_size_bytes: bytes.len() as u64,
                blake3_hex,
            });
        }

        let index = ArtifactIndex {
            schema_version: BUNDLE_SCHEMA_VERSION,
            run_id: manifest.run_id.clone(),
            artifacts: entries,
        };

        // Write manifest.json
        let manifest_path = output_dir.join("manifest.json");
        let manifest_str = serde_json::to_string_pretty(&manifest)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        std::fs::write(manifest_path, manifest_str)?;

        // Write artifact_index.json
        let index_path = output_dir.join("artifact_index.json");
        let index_str = serde_json::to_string_pretty(&index)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        std::fs::write(index_path, index_str)?;

        Ok(index)
    }

    /// Verifies a run bundle directory against its manifest and artifact index.
    pub fn verify_bundle(
        bundle_dir: &std::path::Path,
    ) -> Result<BundleVerificationReport, BundleVerificationError> {
        let manifest_path = bundle_dir.join("manifest.json");
        if !manifest_path.exists() {
            return Err(BundleVerificationError::MissingFile("manifest.json".into()));
        }

        let manifest_bytes = std::fs::read(&manifest_path)?;
        let manifest: RunBundleManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| BundleVerificationError::Serialization(e.to_string()))?;

        if manifest.schema_version != BUNDLE_SCHEMA_VERSION {
            return Err(BundleVerificationError::SchemaVersionMismatch {
                expected: BUNDLE_SCHEMA_VERSION,
                found: manifest.schema_version,
            });
        }

        let index_path = bundle_dir.join("artifact_index.json");
        if !index_path.exists() {
            return Err(BundleVerificationError::MissingFile(
                "artifact_index.json".into(),
            ));
        }

        let index_bytes = std::fs::read(&index_path)?;
        let index: ArtifactIndex = serde_json::from_slice(&index_bytes)
            .map_err(|e| BundleVerificationError::Serialization(e.to_string()))?;

        let mut verified_files = 0;
        let mut total_bytes_verified = 0;

        for entry in &index.artifacts {
            let file_path = bundle_dir.join(&entry.relative_path);
            if !file_path.exists() {
                return Err(BundleVerificationError::MissingFile(
                    entry.relative_path.clone(),
                ));
            }

            let bytes = std::fs::read(&file_path)?;
            let computed_hex = blake3::hash(&bytes).to_hex().to_string();

            if computed_hex != entry.blake3_hex {
                return Err(BundleVerificationError::ChecksumMismatch {
                    path: entry.relative_path.clone(),
                    expected: entry.blake3_hex.clone(),
                    computed: computed_hex,
                });
            }

            verified_files += 1;
            total_bytes_verified += bytes.len() as u64;
        }

        Ok(BundleVerificationReport {
            run_id: manifest.run_id,
            verified_files,
            total_bytes_verified,
            is_valid: true,
        })
    }
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

    #[test]
    fn test_assemble_and_verify_bundle_success() {
        let temp_dir = tempfile::tempdir().unwrap();
        let bundle_path = temp_dir.path().join("bundle_run_42");

        let manifest = RunBundleManifest {
            schema_version: 1,
            run_id: "run-42".into(),
            seed: 12345,
            created_at_utc: "2026-07-22T15:30:00Z".into(),
            source_revision: "abc123commit".into(),
            source_tree_digest: "def456digest".into(),
            source_tree_dirty: false,
            rust_toolchain: "nightly-2026-07-09".into(),
            cargo_lock_digest: "lock123digest".into(),
            target_triple: "aarch64-apple-darwin".into(),
            total_ticks: 1000,
            final_agent_count: 50,
            config_hash: "conf789hash".into(),
        };

        let db_bytes = b"SQLITE_FORMAT_TEST_DB_DATA";
        let csv_bytes = b"tick,metric,value\n1,pop,50\n2,pop,52\n";

        let index = DeterministicRunBundle::assemble_bundle(
            &bundle_path,
            manifest.clone(),
            &[("db.sqlite", db_bytes), ("exports/metrics.csv", csv_bytes)],
        )
        .unwrap();

        assert_eq!(index.artifacts.len(), 2);
        assert_eq!(index.artifacts[0].relative_path, "db.sqlite");

        let report = DeterministicRunBundle::verify_bundle(&bundle_path).unwrap();
        assert_eq!(report.run_id, "run-42");
        assert_eq!(report.verified_files, 2);
        assert!(report.is_valid);
    }

    #[test]
    fn test_verify_bundle_tampered_content() {
        let temp_dir = tempfile::tempdir().unwrap();
        let bundle_path = temp_dir.path().join("tampered_bundle");

        let manifest = RunBundleManifest {
            schema_version: 1,
            run_id: "run-tampered".into(),
            seed: 99,
            created_at_utc: "2026-07-22T15:30:00Z".into(),
            source_revision: "rev".into(),
            source_tree_digest: "digest".into(),
            source_tree_dirty: false,
            rust_toolchain: "nightly".into(),
            cargo_lock_digest: "lock".into(),
            target_triple: "aarch64-apple-darwin".into(),
            total_ticks: 10,
            final_agent_count: 5,
            config_hash: "conf".into(),
        };

        DeterministicRunBundle::assemble_bundle(
            &bundle_path,
            manifest,
            &[("data.txt", b"original content")],
        )
        .unwrap();

        // Tamper with data.txt
        std::fs::write(bundle_path.join("data.txt"), b"tampered content").unwrap();

        let err = DeterministicRunBundle::verify_bundle(&bundle_path).unwrap_err();
        match err {
            BundleVerificationError::ChecksumMismatch { path, .. } => {
                assert_eq!(path, "data.txt");
            }
            other => panic!("Expected ChecksumMismatch, got {:?}", other),
        }
    }
}
