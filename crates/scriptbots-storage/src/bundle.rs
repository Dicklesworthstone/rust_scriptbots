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
    #[serde(alias = "sha256_hex")]
    pub blake3_hex: String,
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

/// Reject any artifact path that is not relative and contained by the bundle directory.
///
/// `verify_run_bundle` has always refused absolute and `..`-bearing entries on read.
/// Applying the identical rule at write time is what makes it safe to accept
/// caller-supplied relative paths in `create_run_bundle_from_artifacts`: without it a
/// caller could name `../../elsewhere` and the assembler would happily write outside the
/// bundle it claims to be building.
fn validate_relative_path(relative_path: &str) -> Result<&Path, BundleError> {
    let path = Path::new(relative_path);
    let escapes = path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        || relative_path.is_empty();
    if escapes {
        return Err(BundleError::NonPortablePath(path.to_path_buf()));
    }
    Ok(path)
}

/// Write one artifact into the bundle and return its manifest entry.
///
/// Every artifact in every bundle — database, JSON export, or caller-supplied payload —
/// goes through this one function, so path validation, BLAKE3 hashing, and byte accounting
/// cannot drift between the database-backed and database-free assemblers.
fn stage_artifact(
    bundle_dir: &Path,
    relative_path: &str,
    artifact_type: &str,
    bytes: &[u8],
) -> Result<RunBundleArtifactEntry, BundleError> {
    let validated = validate_relative_path(relative_path)?;
    let destination = bundle_dir.join(validated);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| BundleError::Io {
            path: parent.to_path_buf(),
            error,
        })?;
    }
    fs::write(&destination, bytes).map_err(|error| BundleError::Io {
        path: destination,
        error,
    })?;
    Ok(RunBundleArtifactEntry {
        relative_path: relative_path.to_owned(),
        blake3_hex: hash_hex(bytes),
        bytes_len: bytes.len() as u64,
        artifact_type: artifact_type.to_owned(),
    })
}

/// Serialize the assembled bundle to the canonical `bundle_manifest.json`.
fn write_bundle_manifest(bundle_dir: &Path, bundle: &RunBundleV1) -> Result<(), BundleError> {
    let manifest_path = bundle_dir.join("bundle_manifest.json");
    let bundle_json = serde_json::to_string_pretty(bundle)?;
    fs::write(&manifest_path, &bundle_json).map_err(|error| BundleError::Io {
        path: manifest_path,
        error,
    })
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

    let db_bytes = fs::read(run_db_path).map_err(|error| BundleError::Io {
        path: run_db_path.to_path_buf(),
        error,
    })?;
    let events_json = serde_json::to_string_pretty(&events)?;
    let checkpoints_json = serde_json::to_string_pretty(&checkpoints)?;

    let artifacts = vec![
        stage_artifact(output_bundle_dir, "run.db", "database", &db_bytes)?,
        stage_artifact(
            output_bundle_dir,
            "events.json",
            "events",
            events_json.as_bytes(),
        )?,
        stage_artifact(
            output_bundle_dir,
            "checkpoints.json",
            "checkpoints",
            checkpoints_json.as_bytes(),
        )?,
    ];

    let bundle = RunBundleV1 {
        bundle_version: RUN_BUNDLE_SCHEMA_VERSION.to_owned(),
        created_at_utc: current_timestamp(),
        digests: RunBundleDigests {
            source_revision: manifest.source_revision.clone(),
            lockfile_digest: Some(manifest.cargo_lock_digest.clone()),
            run_id,
            max_tick,
            event_count: events.len() as u64,
            checkpoint_count: checkpoints.len() as u64,
        },
        manifest,
        artifacts,
    };

    write_bundle_manifest(output_bundle_dir, &bundle)?;
    Ok(bundle)
}

/// Create a portable run bundle from caller-supplied artifact bytes, with no run database.
///
/// This is the assembler for producers that never opened a `Storage` — the experiment
/// runner steps a persistence-disabled world and has only in-memory exports to package.
/// It emits the same `scriptbots.run-bundle.v1` `bundle_manifest.json` as
/// `create_run_bundle` and is verified by the same `verify_run_bundle`, so a bundle's
/// provenance does not depend on which producer built it.
///
/// `event_count` and `checkpoint_count` are recorded as zero because a database-free
/// bundle genuinely has no persisted replay or checkpoint rows; the caller supplies the
/// tick budget it actually ran.
pub fn create_run_bundle_from_artifacts(
    output_bundle_dir: &Path,
    manifest: RunManifestRecord,
    max_tick: u64,
    artifact_files: &[(&str, &str, &[u8])],
) -> Result<RunBundleV1, BundleError> {
    fs::create_dir_all(output_bundle_dir).map_err(|error| BundleError::Io {
        path: output_bundle_dir.to_path_buf(),
        error,
    })?;

    let mut artifacts = Vec::with_capacity(artifact_files.len());
    for (relative_path, artifact_type, bytes) in artifact_files {
        artifacts.push(stage_artifact(
            output_bundle_dir,
            relative_path,
            artifact_type,
            bytes,
        )?);
    }

    let bundle = RunBundleV1 {
        bundle_version: RUN_BUNDLE_SCHEMA_VERSION.to_owned(),
        created_at_utc: current_timestamp(),
        digests: RunBundleDigests {
            source_revision: manifest.source_revision.clone(),
            lockfile_digest: Some(manifest.cargo_lock_digest.clone()),
            run_id: manifest.run_id.to_string(),
            max_tick,
            event_count: 0,
            checkpoint_count: 0,
        },
        manifest,
        artifacts,
    };

    write_bundle_manifest(output_bundle_dir, &bundle)?;
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
        // Exactly the rule the assembler applies, so read and write cannot disagree.
        let rel_path = validate_relative_path(&entry.relative_path)?;

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
        if actual_hash != entry.blake3_hex {
            return Err(BundleError::HashMismatch {
                path: rel_path.to_path_buf(),
                expected: entry.blake3_hex.clone(),
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
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!("scriptbots-{name}-{nanos}.sqlite"));
        path
    }

    fn temp_bundle_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!("scriptbots-bundle-{name}-{nanos}"));
        path
    }

    #[test]
    fn bundle_creation_and_verification_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let db_path = temp_db_path("bundle-test");
        let bundle_dir = temp_bundle_dir("bundle-out");
        let db_path_str = db_path.to_string_lossy().to_string();

        let manifest = crate::RunManifestRecord::unattributed(scriptbots_runtime::RunId::new(1));
        let mut storage = Storage::create_new_file_for_run_with_thresholds(
            &db_path_str,
            manifest,
            64,
            4096,
            1024,
            1024,
        )?;
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

    /// `bd-4d9j`: the database-free assembler that replaced
    /// `export_pipeline::DeterministicRunBundle` emits the same `scriptbots.run-bundle.v1`
    /// manifest and is read back by the same verifier, including nested relative paths.
    #[test]
    fn artifact_bundle_round_trips_through_the_single_verifier()
    -> Result<(), Box<dyn std::error::Error>> {
        let bundle_dir = temp_bundle_dir("artifact-bundle");
        let manifest = crate::RunManifestRecord::unattributed(scriptbots_runtime::RunId::new(42));
        let run_id = manifest.run_id.to_string();

        let summary_csv = b"tick,metric,value\n1,pop,50\n2,pop,52\n";
        let notes = b"free-form producer payload";
        let bundle = create_run_bundle_from_artifacts(
            &bundle_dir,
            manifest,
            1_000,
            &[
                ("exports/summary.csv", "export", summary_csv),
                ("notes.txt", "export", notes),
            ],
        )?;

        assert_eq!(bundle.bundle_version, RUN_BUNDLE_SCHEMA_VERSION);
        assert_eq!(bundle.artifacts.len(), 2);
        assert_eq!(bundle.artifacts[0].relative_path, "exports/summary.csv");
        assert_eq!(bundle.digests.max_tick, 1_000);
        assert_eq!(bundle.digests.run_id, run_id);
        // A database-free bundle honestly reports no persisted replay or checkpoint rows.
        assert_eq!(bundle.digests.event_count, 0);
        assert_eq!(bundle.digests.checkpoint_count, 0);
        assert!(bundle_dir.join("bundle_manifest.json").exists());
        assert!(bundle_dir.join("exports/summary.csv").exists());

        let verification = verify_run_bundle(&bundle_dir)?;
        assert_eq!(verification.run_id, run_id);
        assert_eq!(verification.total_artifacts_verified, 2);
        assert_eq!(
            verification.total_bytes_verified,
            (summary_csv.len() + notes.len()) as u64
        );

        // Tampering with a nested artifact is caught by the same checksum loop.
        fs::write(bundle_dir.join("exports/summary.csv"), b"tampered")?;
        let tampered = verify_run_bundle(&bundle_dir);
        assert!(
            matches!(tampered, Err(BundleError::HashMismatch { .. })),
            "expected a hash mismatch for the tampered artifact, got {tampered:?}"
        );

        let _ = fs::remove_dir_all(bundle_dir);
        Ok(())
    }

    /// `bd-4d9j`: the replaced assembler accepted any caller-supplied relative path and
    /// would write outside the bundle directory. Assembly now applies the same portability
    /// rule the verifier always applied, and nothing is written before it is checked.
    #[test]
    fn an_escaping_artifact_path_is_refused_before_anything_is_written()
    -> Result<(), Box<dyn std::error::Error>> {
        let bundle_dir = temp_bundle_dir("escaping-artifact");
        // Name the escape target after this bundle directory: its parent is the shared
        // temp directory, so a fixed name could collide with a stale file from an earlier
        // run and make the containment check pass or fail for the wrong reason.
        let sibling = format!(
            "{}-escaped.txt",
            bundle_dir
                .file_name()
                .expect("the bundle directory has a file name")
                .to_string_lossy()
        );
        for escaping in [format!("../{sibling}"), format!("nested/../../{sibling}")] {
            let outcome = create_run_bundle_from_artifacts(
                &bundle_dir,
                crate::RunManifestRecord::unattributed(scriptbots_runtime::RunId::new(7)),
                0,
                &[(escaping.as_str(), "export", b"payload")],
            );
            assert!(
                matches!(outcome, Err(BundleError::NonPortablePath(_))),
                "expected {escaping} to be refused, got {outcome:?}"
            );
        }
        assert!(
            !bundle_dir.join("bundle_manifest.json").exists(),
            "a refused assembly still wrote a bundle manifest"
        );
        let escaped = bundle_dir
            .parent()
            .expect("the bundle directory has a parent")
            .join(&sibling);
        assert!(
            !escaped.exists(),
            "a refused assembly wrote outside its bundle directory to {}",
            escaped.display()
        );

        let _ = fs::remove_dir_all(bundle_dir);
        Ok(())
    }
}
