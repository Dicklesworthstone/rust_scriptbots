//! Portable deterministic run bundle assembly and verification.

use crate::{RunManifestRecord, StorageError, StorageReader};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Schema version tag for portable run bundles.
pub const RUN_BUNDLE_SCHEMA_VERSION: &str = "scriptbots.run-bundle.v1";

fn hash_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn current_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

#[derive(Debug, Error)]
pub enum BundleError {
    #[error("I/O error at {path}: {error}")]
    Io {
        path: PathBuf,
        #[source]
        error: std::io::Error,
    },
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
    #[error("Serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Invalid bundle path (must be relative and portable): {0}")]
    NonPortablePath(PathBuf),
    #[error("Bundle manifest missing or corrupted at {0}")]
    InvalidManifest(PathBuf),
    #[error("Artifact hash mismatch for {path}: expected {expected}, calculated {actual}")]
    HashMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("Missing expected artifact in bundle: {0}")]
    MissingArtifact(PathBuf),
    #[error("Run ID mismatch: manifest run_id {manifest_run_id} != database run_id {db_run_id}")]
    RunIdMismatch {
        manifest_run_id: String,
        db_run_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunBundleArtifactEntry {
    pub relative_path: String,
    pub sha256_hex: String,
    pub bytes_len: u64,
    pub artifact_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunBundleDigests {
    pub source_revision: Option<String>,
    pub lockfile_digest: Option<String>,
    pub run_id: String,
    pub max_tick: u64,
    pub event_count: u64,
    pub checkpoint_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunBundleV1 {
    pub bundle_version: String,
    pub created_at_utc: String,
    pub manifest: RunManifestRecord,
    pub digests: RunBundleDigests,
    pub artifacts: Vec<RunBundleArtifactEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunBundleVerificationResult {
    pub bundle_version: String,
    pub run_id: String,
    pub max_tick: u64,
    pub total_artifacts_verified: usize,
    pub total_bytes_verified: u64,
    pub reproducible: bool,
    pub verified_at_utc: String,
}

/// Create a portable run bundle directory from an existing FrankenSQLite run database.
pub fn create_run_bundle(
    run_db_path: &Path,
    output_bundle_dir: &Path,
) -> Result<RunBundleV1, BundleError> {
    if !run_db_path.exists() {
        return Err(BundleError::Io {
            path: run_db_path.to_path_buf(),
            error: std::io::Error::new(std::io::ErrorKind::NotFound, "Run DB not found"),
        });
    }

    fs::create_dir_all(output_bundle_dir).map_err(|error| BundleError::Io {
        path: output_bundle_dir.to_path_buf(),
        error,
    })?;

    let reader = StorageReader::open(&run_db_path.to_string_lossy())?;
    let manifest = reader.run_manifest()?;
    let max_tick = reader.max_tick()?.unwrap_or(0);
    let events = reader.load_replay_events()?;
    let checkpoints = reader.load_checkpoints()?;
    let run_id = manifest.run_id.to_string();
    reader.close()?;

    // 1. Copy run.db into bundle
    let db_dst = output_bundle_dir.join("run.db");
    fs::copy(run_db_path, &db_dst).map_err(|error| BundleError::Io {
        path: db_dst.clone(),
        error,
    })?;

    let db_bytes = fs::read(&db_dst).map_err(|error| BundleError::Io {
        path: db_dst.clone(),
        error,
    })?;
    let db_hash = hash_hex(&db_bytes);

    let mut artifacts = vec![RunBundleArtifactEntry {
        relative_path: "run.db".to_owned(),
        sha256_hex: db_hash,
        bytes_len: db_bytes.len() as u64,
        artifact_type: "database".to_owned(),
    }];

    // 2. Export events.json
    let events_json = serde_json::to_string_pretty(&events)?;
    let events_dst = output_bundle_dir.join("events.json");
    fs::write(&events_dst, &events_json).map_err(|error| BundleError::Io {
        path: events_dst.clone(),
        error,
    })?;
    let events_hash = hash_hex(events_json.as_bytes());
    artifacts.push(RunBundleArtifactEntry {
        relative_path: "events.json".to_owned(),
        sha256_hex: events_hash,
        bytes_len: events_json.len() as u64,
        artifact_type: "events".to_owned(),
    });

    // 3. Export checkpoints.json
    let checkpoints_json = serde_json::to_string_pretty(&checkpoints)?;
    let cp_dst = output_bundle_dir.join("checkpoints.json");
    fs::write(&cp_dst, &checkpoints_json).map_err(|error| BundleError::Io {
        path: cp_dst.clone(),
        error,
    })?;
    let cp_hash = hash_hex(checkpoints_json.as_bytes());
    artifacts.push(RunBundleArtifactEntry {
        relative_path: "checkpoints.json".to_owned(),
        sha256_hex: cp_hash,
        bytes_len: checkpoints_json.len() as u64,
        artifact_type: "checkpoints".to_owned(),
    });

    let bundle_digests = RunBundleDigests {
        source_revision: manifest.source_revision.clone(),
        lockfile_digest: Some(manifest.cargo_lock_digest.clone()),
        run_id: run_id.clone(),
        max_tick,
        event_count: events.len() as u64,
        checkpoint_count: checkpoints.len() as u64,
    };

    let bundle = RunBundleV1 {
        bundle_version: RUN_BUNDLE_SCHEMA_VERSION.to_owned(),
        created_at_utc: current_timestamp(),
        manifest,
        digests: bundle_digests,
        artifacts,
    };

    let manifest_dst = output_bundle_dir.join("bundle_manifest.json");
    let bundle_json = serde_json::to_string_pretty(&bundle)?;
    fs::write(&manifest_dst, &bundle_json).map_err(|error| BundleError::Io {
        path: manifest_dst,
        error,
    })?;

    Ok(bundle)
}

/// Verify the integrity, schema, and portability of a run bundle directory.
pub fn verify_run_bundle(bundle_dir: &Path) -> Result<RunBundleVerificationResult, BundleError> {
    let manifest_path = bundle_dir.join("bundle_manifest.json");
    if !manifest_path.exists() {
        return Err(BundleError::InvalidManifest(manifest_path));
    }

    let manifest_data = fs::read_to_string(&manifest_path).map_err(|error| BundleError::Io {
        path: manifest_path.clone(),
        error,
    })?;

    let bundle: RunBundleV1 = serde_json::from_str(&manifest_data)?;

    if bundle.bundle_version != RUN_BUNDLE_SCHEMA_VERSION {
        return Err(BundleError::InvalidManifest(manifest_path));
    }

    let mut total_bytes = 0u64;

    for entry in &bundle.artifacts {
        let rel_path = Path::new(&entry.relative_path);
        if rel_path.is_absolute() || entry.relative_path.contains("..") {
            return Err(BundleError::NonPortablePath(rel_path.to_path_buf()));
        }

        let full_path = bundle_dir.join(rel_path);
        if !full_path.exists() {
            return Err(BundleError::MissingArtifact(rel_path.to_path_buf()));
        }

        let bytes = fs::read(&full_path).map_err(|error| BundleError::Io {
            path: full_path.clone(),
            error,
        })?;

        if bytes.len() as u64 != entry.bytes_len {
            return Err(BundleError::HashMismatch {
                path: rel_path.to_path_buf(),
                expected: format!("{} bytes", entry.bytes_len),
                actual: format!("{} bytes", bytes.len()),
            });
        }

        let actual_hash = hash_hex(&bytes);
        if actual_hash != entry.sha256_hex {
            return Err(BundleError::HashMismatch {
                path: rel_path.to_path_buf(),
                expected: entry.sha256_hex.clone(),
                actual: actual_hash,
            });
        }

        total_bytes += bytes.len() as u64;
    }

    // Also verify database run_id matches bundle run_id
    let db_path = bundle_dir.join("run.db");
    if db_path.exists() {
        let reader = StorageReader::open(&db_path.to_string_lossy())?;
        let db_manifest = reader.run_manifest()?;
        if db_manifest.run_id != bundle.manifest.run_id {
            return Err(BundleError::RunIdMismatch {
                manifest_run_id: bundle.manifest.run_id.to_string(),
                db_run_id: db_manifest.run_id.to_string(),
            });
        }
        reader.close()?;
    }

    Ok(RunBundleVerificationResult {
        bundle_version: bundle.bundle_version,
        run_id: bundle.manifest.run_id.to_string(),
        max_tick: bundle.digests.max_tick,
        total_artifacts_verified: bundle.artifacts.len(),
        total_bytes_verified: total_bytes,
        reproducible: bundle.manifest.reproducible,
        verified_at_utc: current_timestamp(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Storage;

    fn temp_db_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let rnd = rand::random::<u64>();
        path.push(format!("scriptbots-{name}-{rnd}.db"));
        path
    }

    fn temp_bundle_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let rnd = rand::random::<u64>();
        path.push(format!("scriptbots-bundle-{name}-{rnd}"));
        path
    }

    #[test]
    fn bundle_creation_and_verification_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let db_path = temp_db_path("bundle-test");
        let bundle_dir = temp_bundle_dir("bundle-out");
        let db_path_str = db_path.to_string_lossy().to_string();

        let mut storage =
            Storage::create_unattributed_file_with_thresholds(&db_path_str, 64, 4096, 1024, 1024)?;
        storage.record_checkpoint(
            "cp-001",
            50,
            0,
            "scriptbots.world-checkpoint.v1.3+postcard_hex",
            "aabbcc",
            "digest1",
            "{}",
        )?;
        storage.flush()?;
        storage.close()?;

        let bundle = create_run_bundle(&db_path, &bundle_dir)?;
        assert_eq!(bundle.bundle_version, RUN_BUNDLE_SCHEMA_VERSION);
        assert_eq!(bundle.artifacts.len(), 3);

        let verification = verify_run_bundle(&bundle_dir)?;
        assert_eq!(verification.bundle_version, RUN_BUNDLE_SCHEMA_VERSION);
        assert_eq!(verification.total_artifacts_verified, 3);
        assert!(verification.total_bytes_verified > 0);

        // Tamper test: modify one artifact and verify it fails with HashMismatch
        let db_dst = bundle_dir.join("run.db");
        fs::write(&db_dst, b"tampered data")?;
        let res = verify_run_bundle(&bundle_dir);
        assert!(res.is_err());
        match res.unwrap_err() {
            BundleError::HashMismatch { .. } => {}
            other => panic!("Expected HashMismatch error, got {other:?}"),
        }

        let _ = fs::remove_file(db_path);
        let _ = fs::remove_dir_all(bundle_dir);
        Ok(())
    }
}
