//! Portable deterministic run bundle assembly and verification.

use crate::{PersistenceWatermarks, RunManifestRecord, StorageError, StorageReader};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
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
    #[error(
        "bounded bundle verification limit `{resource}` exceeded at {path}: observed {observed}, limit {limit}"
    )]
    VerificationLimitExceeded {
        resource: &'static str,
        path: PathBuf,
        observed: u64,
        limit: u64,
    },
    #[error("bounded bundle verification requires a regular file at {0}")]
    NonRegularFile(PathBuf),
    #[error(
        "bounded semantic verification of database-backed bundle {0} is unavailable: the pinned FrankenSQLite engine cannot interrupt a running statement; verify it in an isolated trusted workflow instead"
    )]
    BoundedDatabaseVerificationUnavailable(PathBuf),
    #[error("bundle manifest projects run_id {projected_run_id}, but embeds {manifest_run_id}")]
    RunIdProjectionMismatch {
        projected_run_id: String,
        manifest_run_id: String,
    },
    #[error("bundle manifest lists artifact path more than once: {0}")]
    DuplicateArtifactPath(PathBuf),
    #[error("bundle database disagrees with exported {0}")]
    DatabaseProjectionMismatch(&'static str),
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persistence_watermarks: Option<RunBundleWatermarks>,
}

/// Batch prefixes independently read from a database-backed bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunBundleWatermarks {
    pub admitted: Option<u64>,
    pub applied: Option<u64>,
    pub durable: Option<u64>,
}

impl From<PersistenceWatermarks> for RunBundleWatermarks {
    fn from(value: PersistenceWatermarks) -> Self {
        Self {
            admitted: value.admitted.map(crate::PersistenceBatchId::get),
            applied: value.applied.map(crate::PersistenceBatchId::get),
            durable: value.durable.map(crate::PersistenceBatchId::get),
        }
    }
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

/// Hard byte and cardinality caps for untrusted artifact-only bundle verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunBundleVerificationLimits {
    pub max_manifest_bytes: u64,
    pub max_artifacts: usize,
    pub max_artifact_bytes: u64,
    pub max_total_artifact_bytes: u64,
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
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
        || relative_path.is_empty();
    if escapes {
        return Err(BundleError::NonPortablePath(path.to_path_buf()));
    }
    Ok(path)
}

/// Write one artifact into the bundle and return its manifest entry.
///
/// JSON exports and caller-supplied payloads share path validation, hashing and
/// byte accounting here. The database image is copied from its leased descriptor.
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
    let write = || -> std::io::Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&destination)?;
        file.write_all(bytes)?;
        file.sync_all()
    };
    write().map_err(|error| BundleError::Io {
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
    let write = || -> std::io::Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&manifest_path)?;
        file.write_all(bundle_json.as_bytes())?;
        file.sync_all()
    };
    write().map_err(|error| BundleError::Io {
        path: manifest_path,
        error,
    })
}

/// Export a finished, checkpointed run under one identity-bound lease.
/// Live writers, nonempty WAL/journal files and existing output directories are refused.
/// Source files are never checkpointed or rewritten by this operation.
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

    let reader = StorageReader::open_finished(&run_db_path.to_string_lossy())?;
    reader.require_finished_bundle_state()?;
    let manifest = reader.run_manifest()?;
    let max_tick = reader.max_tick()?.unwrap_or(0);
    let events = reader.load_replay_events()?;
    let checkpoints = reader.load_checkpoints()?;
    let persistence_watermarks = reader.persistence_watermarks()?.into();
    let run_id = manifest.run_id.to_string();

    if let Some(parent) = output_bundle_dir.parent() {
        fs::create_dir_all(parent).map_err(|error| BundleError::Io {
            path: parent.to_path_buf(),
            error,
        })?;
    }
    fs::create_dir(output_bundle_dir).map_err(|error| BundleError::Io {
        path: output_bundle_dir.to_path_buf(),
        error,
    })?;
    let db_path = output_bundle_dir.join("run.db");
    reader.materialize_finished_database(run_db_path, &db_path)?;
    let db_bytes = fs::read(&db_path).map_err(|error| BundleError::Io {
        path: db_path,
        error,
    })?;
    let events_json = serde_json::to_string_pretty(&events)?;
    let checkpoints_json = serde_json::to_string_pretty(&checkpoints)?;

    let artifacts = vec![
        RunBundleArtifactEntry {
            relative_path: "run.db".to_owned(),
            blake3_hex: hash_hex(&db_bytes),
            bytes_len: db_bytes.len() as u64,
            artifact_type: "database".to_owned(),
        },
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
            persistence_watermarks: Some(persistence_watermarks),
        },
        manifest,
        artifacts,
    };

    verify_database_projection(output_bundle_dir, &bundle)?;
    reader.close()?;
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
            persistence_watermarks: None,
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
    let mut artifact_paths = std::collections::BTreeSet::new();

    if bundle.digests.run_id != bundle.manifest.run_id.to_string() {
        return Err(BundleError::RunIdProjectionMismatch {
            projected_run_id: bundle.digests.run_id,
            manifest_run_id: bundle.manifest.run_id.to_string(),
        });
    }

    for entry in &bundle.artifacts {
        // Exactly the rule the assembler applies, so read and write cannot disagree.
        let rel_path = validate_relative_path(&entry.relative_path)?;
        if !artifact_paths.insert(rel_path.to_path_buf()) {
            return Err(BundleError::DuplicateArtifactPath(rel_path.to_path_buf()));
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
        if actual_hash != entry.blake3_hex {
            return Err(BundleError::HashMismatch {
                path: rel_path.to_path_buf(),
                expected: entry.blake3_hex.clone(),
                actual: actual_hash,
            });
        }

        total_bytes += bytes.len() as u64;
    }

    let db_path = bundle_dir.join("run.db");
    if db_path.exists() {
        verify_database_projection(bundle_dir, &bundle)?;
    } else if bundle
        .artifacts
        .iter()
        .any(|entry| entry.artifact_type == "database")
        || bundle.digests.persistence_watermarks.is_some()
        || bundle.digests.event_count != 0
        || bundle.digests.checkpoint_count != 0
    {
        return Err(BundleError::MissingArtifact(PathBuf::from("run.db")));
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

/// Reopen the materialized database and compare logical state, not just file hashes.
fn verify_database_projection(bundle_dir: &Path, bundle: &RunBundleV1) -> Result<(), BundleError> {
    for (path, kind) in [
        ("run.db", "database"),
        ("events.json", "events"),
        ("checkpoints.json", "checkpoints"),
    ] {
        if !bundle
            .artifacts
            .iter()
            .any(|entry| entry.relative_path == path && entry.artifact_type == kind)
        {
            return Err(BundleError::MissingArtifact(PathBuf::from(path)));
        }
    }
    let reader = StorageReader::open_finished(&bundle_dir.join("run.db").to_string_lossy())?;
    reader.require_finished_bundle_state()?;
    let manifest = reader.run_manifest()?;
    if manifest.run_id != bundle.manifest.run_id {
        return Err(BundleError::RunIdMismatch {
            manifest_run_id: bundle.manifest.run_id.to_string(),
            db_run_id: manifest.run_id.to_string(),
        });
    }
    if manifest != bundle.manifest
        || bundle.digests.source_revision != manifest.source_revision
        || bundle.digests.lockfile_digest.as_deref() != Some(manifest.cargo_lock_digest.as_str())
    {
        return Err(BundleError::DatabaseProjectionMismatch("run manifest"));
    }
    if reader.max_tick()?.unwrap_or(0) != bundle.digests.max_tick {
        return Err(BundleError::DatabaseProjectionMismatch("max tick"));
    }
    if bundle.digests.persistence_watermarks != Some(reader.persistence_watermarks()?.into()) {
        return Err(BundleError::DatabaseProjectionMismatch(
            "persistence watermarks",
        ));
    }
    let events = reader.load_replay_events()?;
    let checkpoints = reader.load_checkpoints()?;
    for (path, actual, expected_count) in [
        (
            "events.json",
            serde_json::to_value(&events)?,
            bundle.digests.event_count,
        ),
        (
            "checkpoints.json",
            serde_json::to_value(&checkpoints)?,
            bundle.digests.checkpoint_count,
        ),
    ] {
        let full_path = bundle_dir.join(path);
        let bytes = fs::read(&full_path).map_err(|error| BundleError::Io {
            path: full_path,
            error,
        })?;
        let exported: serde_json::Value = serde_json::from_slice(&bytes)?;
        if exported != actual
            || actual.as_array().map(|rows| rows.len() as u64) != Some(expected_count)
        {
            return Err(BundleError::DatabaseProjectionMismatch(path));
        }
    }
    reader.close()?;
    Ok(())
}

/// Verify an artifact-only bundle while enforcing hard read and cardinality caps.
///
/// This entry point opens each file once, validates the opened handle, and streams
/// at most the declared byte count plus one byte through BLAKE3. It never allocates
/// from an artifact's declared size. Database-backed bundles are refused explicitly:
/// the pinned FrankenSQLite engine cannot interrupt an already-running statement,
/// so advertising that semantic database verification as bounded would be false.
pub fn verify_run_bundle_bounded(
    bundle_dir: &Path,
    limits: RunBundleVerificationLimits,
) -> Result<RunBundleVerificationResult, BundleError> {
    let manifest_path = bundle_dir.join("bundle_manifest.json");
    let manifest_data =
        read_file_bounded(&manifest_path, limits.max_manifest_bytes, "manifest_bytes")?;
    let bundle: RunBundleV1 = serde_json::from_slice(&manifest_data)?;
    if bundle.bundle_version != RUN_BUNDLE_SCHEMA_VERSION {
        return Err(BundleError::InvalidManifest(manifest_path));
    }
    let manifest_run_id = bundle.manifest.run_id.to_string();
    if bundle.digests.run_id != manifest_run_id {
        return Err(BundleError::RunIdProjectionMismatch {
            projected_run_id: bundle.digests.run_id,
            manifest_run_id,
        });
    }
    if bundle.artifacts.len() > limits.max_artifacts {
        return Err(BundleError::VerificationLimitExceeded {
            resource: "artifact_count",
            path: bundle_dir.to_path_buf(),
            observed: u64::try_from(bundle.artifacts.len()).unwrap_or(u64::MAX),
            limit: u64::try_from(limits.max_artifacts).unwrap_or(u64::MAX),
        });
    }

    let db_path = bundle_dir.join("run.db");
    match fs::symlink_metadata(&db_path) {
        Ok(_) => {
            return Err(BundleError::BoundedDatabaseVerificationUnavailable(db_path));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(BundleError::Io {
                path: db_path,
                error,
            });
        }
    }
    if bundle.digests.persistence_watermarks.is_some()
        || bundle.digests.event_count != 0
        || bundle.digests.checkpoint_count != 0
    {
        return Err(BundleError::MissingArtifact(PathBuf::from("run.db")));
    }

    let mut total_bytes = 0_u64;
    let mut artifact_paths = std::collections::BTreeSet::new();
    for entry in &bundle.artifacts {
        let rel_path = validate_relative_path(&entry.relative_path)?;
        let normalized_path = rel_path.components().collect::<PathBuf>();
        if !artifact_paths.insert(normalized_path.clone()) {
            return Err(BundleError::DuplicateArtifactPath(normalized_path));
        }
        if is_database_artifact(entry, &normalized_path) {
            return Err(BundleError::BoundedDatabaseVerificationUnavailable(
                bundle_dir.join(normalized_path),
            ));
        }
        let full_path = bundle_dir.join(&normalized_path);
        if entry.bytes_len > limits.max_artifact_bytes {
            return Err(BundleError::VerificationLimitExceeded {
                resource: "artifact_bytes",
                path: full_path,
                observed: entry.bytes_len,
                limit: limits.max_artifact_bytes,
            });
        }
        total_bytes = total_bytes.checked_add(entry.bytes_len).ok_or_else(|| {
            BundleError::VerificationLimitExceeded {
                resource: "total_artifact_bytes",
                path: bundle_dir.to_path_buf(),
                observed: u64::MAX,
                limit: limits.max_total_artifact_bytes,
            }
        })?;
        if total_bytes > limits.max_total_artifact_bytes {
            return Err(BundleError::VerificationLimitExceeded {
                resource: "total_artifact_bytes",
                path: bundle_dir.to_path_buf(),
                observed: total_bytes,
                limit: limits.max_total_artifact_bytes,
            });
        }
        let actual_hash = hash_file_exact_bounded(&full_path, entry.bytes_len)?;
        if actual_hash != entry.blake3_hex {
            return Err(BundleError::HashMismatch {
                path: rel_path.to_path_buf(),
                expected: entry.blake3_hex.clone(),
                actual: actual_hash,
            });
        }
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

fn is_database_artifact(entry: &RunBundleArtifactEntry, path: &Path) -> bool {
    let artifact_type = entry.artifact_type.to_ascii_lowercase();
    matches!(
        artifact_type.as_str(),
        "database" | "sqlite" | "frankensqlite"
    ) || path.extension().is_some_and(|extension| {
        extension.eq_ignore_ascii_case("db")
            || extension.eq_ignore_ascii_case("sqlite")
            || extension.eq_ignore_ascii_case("sqlite3")
    })
}

fn read_file_bounded(
    path: &Path,
    max_bytes: u64,
    resource: &'static str,
) -> Result<Vec<u8>, BundleError> {
    let file = File::open(path).map_err(|error| BundleError::Io {
        path: path.to_path_buf(),
        error,
    })?;
    let metadata = file.metadata().map_err(|error| BundleError::Io {
        path: path.to_path_buf(),
        error,
    })?;
    if !metadata.is_file() {
        return Err(BundleError::NonRegularFile(path.to_path_buf()));
    }
    if metadata.len() > max_bytes {
        return Err(BundleError::VerificationLimitExceeded {
            resource,
            path: path.to_path_buf(),
            observed: metadata.len(),
            limit: max_bytes,
        });
    }
    let mut bytes =
        Vec::with_capacity(usize::try_from(metadata.len().min(max_bytes)).unwrap_or(usize::MAX));
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| BundleError::Io {
            path: path.to_path_buf(),
            error,
        })?;
    let observed = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if observed > max_bytes {
        return Err(BundleError::VerificationLimitExceeded {
            resource,
            path: path.to_path_buf(),
            observed,
            limit: max_bytes,
        });
    }
    Ok(bytes)
}

fn hash_file_exact_bounded(path: &Path, expected_bytes: u64) -> Result<String, BundleError> {
    let file = File::open(path).map_err(|error| BundleError::Io {
        path: path.to_path_buf(),
        error,
    })?;
    let metadata = file.metadata().map_err(|error| BundleError::Io {
        path: path.to_path_buf(),
        error,
    })?;
    if !metadata.is_file() {
        return Err(BundleError::NonRegularFile(path.to_path_buf()));
    }
    if metadata.len() != expected_bytes {
        return Err(BundleError::HashMismatch {
            path: path.to_path_buf(),
            expected: format!("{expected_bytes} bytes"),
            actual: format!("{} bytes", metadata.len()),
        });
    }

    let mut reader = file.take(expected_bytes.saturating_add(1));
    let mut hasher = blake3::Hasher::new();
    let mut observed = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer).map_err(|error| BundleError::Io {
            path: path.to_path_buf(),
            error,
        })?;
        if read == 0 {
            break;
        }
        observed = observed.saturating_add(u64::try_from(read).unwrap_or(u64::MAX));
        if observed > expected_bytes {
            return Err(BundleError::HashMismatch {
                path: path.to_path_buf(),
                expected: format!("{expected_bytes} bytes"),
                actual: format!("more than {expected_bytes} bytes"),
            });
        }
        hasher.update(&buffer[..read]);
    }
    if observed != expected_bytes {
        return Err(BundleError::HashMismatch {
            path: path.to_path_buf(),
            expected: format!("{expected_bytes} bytes"),
            actual: format!("{observed} bytes"),
        });
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Storage;

    fn replace_test_manifest(bundle_dir: &Path, bundle: &RunBundleV1) -> Result<(), BundleError> {
        let path = bundle_dir.join("bundle_manifest.json");
        fs::write(&path, serde_json::to_vec_pretty(bundle)?)
            .map_err(|error| BundleError::Io { path, error })
    }

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
    fn bundle_refuses_committed_wal_without_rewriting_source()
    -> Result<(), Box<dyn std::error::Error>> {
        let db_path = temp_db_path("uncheckpointed-bundle");
        let mut storage = Storage::create_new_file_for_run(
            &db_path.to_string_lossy(),
            RunManifestRecord::unattributed(scriptbots_runtime::RunId::new(7)),
        )?;
        storage
            .connection()?
            .execute("PRAGMA wal_autocheckpoint = 0")?;
        let world = scriptbots_core::WorldState::new(scriptbots_core::ScriptBotsConfig {
            world_width: 40,
            world_height: 40,
            food_cell_size: 10,
            persistence_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(42),
            ..scriptbots_core::ScriptBotsConfig::default()
        })?;
        storage.record_checkpoint(
            "wal-checkpoint",
            0,
            &world.checkpoint_v1()?,
            &serde_json::json!({}),
        )?;
        storage
            .conn
            .take()
            .expect("fixture connection")
            .close_without_checkpoint()?;
        drop(storage);
        let wal_path = PathBuf::from(format!("{}-wal", db_path.display()));
        let original_db = fs::read(&db_path)?;
        let original_wal = fs::read(&wal_path)?;
        assert!(
            !original_wal.is_empty(),
            "fixture must actually contain committed WAL data"
        );
        let output = temp_bundle_dir("wal-refusal");
        assert!(matches!(
            create_run_bundle(&db_path, &output),
            Err(BundleError::Storage(StorageError::InvalidTarget { reason, .. })) if reason.contains("checkpointed storage")
        ));
        assert!(!output.join("bundle_manifest.json").exists());
        assert_eq!(fs::read(&db_path)?, original_db);
        assert_eq!(fs::read(&wal_path)?, original_wal);
        Ok(())
    }

    #[test]
    fn bundle_copy_holds_writer_lease_and_refuses_changed_source_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let db_path = temp_db_path("leased-bundle");
        let storage = Storage::create_new_file_for_run(
            &db_path.to_string_lossy(),
            RunManifestRecord::unattributed(scriptbots_runtime::RunId::new(8)),
        )?;
        storage.close()?;
        let reader = StorageReader::open_finished(&db_path.to_string_lossy())?;
        assert!(matches!(
            crate::StoragePipeline::recover_existing(&db_path.to_string_lossy()),
            Err(StorageError::InvalidData {
                context: "storage.path_lease",
                ..
            })
        ));
        let copy_path = temp_db_path("leased-copy");
        reader.materialize_finished_database(&db_path, &copy_path)?;
        let before = fs::read(&copy_path)?;
        assert!(matches!(
            reader.materialize_finished_database(&db_path, &copy_path),
            Err(StorageError::Filesystem { source, .. }) if source.kind() == std::io::ErrorKind::AlreadyExists
        ));
        assert_eq!(fs::read(&copy_path)?, before);
        let moved = temp_db_path("retained-original");
        fs::rename(&db_path, &moved)?;
        fs::copy(&moved, &db_path)?;
        let refused_copy = temp_db_path("changed-source-copy");
        let error = reader
            .materialize_finished_database(&db_path, &refused_copy)
            .expect_err("changed path identity must fail");
        assert!(error.to_string().contains("DIFFERENT FILE"), "{error}");
        assert!(!refused_copy.exists());
        reader.close()?;
        Ok(())
    }

    #[test]
    fn bundle_refuses_unapplied_outbox_and_ambiguous_runs() -> Result<(), Box<dyn std::error::Error>>
    {
        let pending = temp_db_path("pending-bundle");
        let mut storage = Storage::create_new_file_for_run(
            &pending.to_string_lossy(),
            RunManifestRecord::unattributed(scriptbots_runtime::RunId::new(9)),
        )?;
        let (receipt, _) = storage.stage_outbox(0, &crate::StorageBuffer::default())?;
        let progress = storage.persistence_watermarks()?;
        assert_eq!(progress.admitted, Some(receipt.batch_id));
        assert_eq!(progress.applied, None);
        assert_eq!(progress.durable, None);
        storage
            .connection()?
            .query("PRAGMA wal_checkpoint(TRUNCATE)")?;
        storage
            .conn
            .take()
            .expect("pending fixture connection")
            .close()?;
        drop(storage);
        let before = fs::read(&pending)?;
        let output = temp_bundle_dir("pending-refusal");
        assert!(matches!(
            create_run_bundle(&pending, &output),
            Err(BundleError::Storage(StorageError::InvalidData { reason, .. })) if reason.contains("not fully durable")
        ));
        assert!(!output.exists());
        assert_eq!(fs::read(&pending)?, before);

        let multiple = temp_db_path("multiple-run-bundle");
        Storage::create_new_file_for_run(
            &multiple.to_string_lossy(),
            RunManifestRecord::unattributed(scriptbots_runtime::RunId::new(10)),
        )?
        .close()?;
        Storage::append_run(
            &multiple.to_string_lossy(),
            RunManifestRecord::unattributed(scriptbots_runtime::RunId::new(11)),
        )?
        .close()?;
        let before = fs::read(&multiple)?;
        let output = temp_bundle_dir("multiple-run-refusal");
        assert!(matches!(
            create_run_bundle(&multiple, &output),
            Err(BundleError::Storage(StorageError::InvalidData { reason, .. })) if reason.contains("multiple runs")
        ));
        assert!(!output.exists());
        assert_eq!(fs::read(&multiple)?, before);
        Ok(())
    }

    #[test]
    fn bundle_refuses_truncated_source_and_failed_staging_without_success_manifest()
    -> Result<(), Box<dyn std::error::Error>> {
        let truncated = temp_db_path("truncated-bundle");
        let bytes = b"SQLite format 3\0";
        let mut source = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&truncated)?;
        source.write_all(bytes)?;
        source.sync_all()?;
        drop(source);
        let output = temp_bundle_dir("truncated-refusal");
        assert!(matches!(
            create_run_bundle(&truncated, &output),
            Err(BundleError::Storage(_))
        ));
        assert!(!output.exists());
        assert_eq!(fs::read(&truncated)?, bytes);

        let output = temp_bundle_dir("staging-refusal");
        fs::create_dir_all(output.join("events.json"))?;
        assert!(matches!(
            stage_artifact(&output, "events.json", "events", b"[]"),
            Err(BundleError::Io { .. })
        ));
        assert!(output.join("events.json").is_dir());
        assert!(!output.join("bundle_manifest.json").exists());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn bundle_refuses_symlink_and_hard_link_source_aliases()
    -> Result<(), Box<dyn std::error::Error>> {
        let original = temp_db_path("alias-bundle");
        Storage::create_new_file_for_run(
            &original.to_string_lossy(),
            RunManifestRecord::unattributed(scriptbots_runtime::RunId::new(12)),
        )?
        .close()?;
        let before = fs::read(&original)?;
        let alias = temp_db_path("symlink-bundle");
        std::os::unix::fs::symlink(&original, &alias)?;
        let output = temp_bundle_dir("symlink-refusal");
        assert!(matches!(
            create_run_bundle(&alias, &output),
            Err(BundleError::Storage(StorageError::InvalidData { reason, .. })) if reason.contains("non-symlink")
        ));
        assert!(!output.exists());
        let hard_link = temp_db_path("hard-link-bundle");
        fs::hard_link(&original, &hard_link)?;
        for source in [&original, &hard_link] {
            let output = temp_bundle_dir("hard-link-refusal");
            assert!(matches!(
                create_run_bundle(source, &output),
                Err(BundleError::Storage(StorageError::InvalidData { reason, .. })) if reason.contains("multiply linked")
            ));
            assert!(!output.exists());
        }
        assert_eq!(fs::read(&original)?, before);
        Ok(())
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
        let world = scriptbots_core::WorldState::new(scriptbots_core::ScriptBotsConfig {
            persistence_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(42),
            ..scriptbots_core::ScriptBotsConfig::default()
        })?;
        storage.record_checkpoint("cp-001", 0, &world.checkpoint_v1()?, &serde_json::json!({}))?;
        storage.flush()?;
        let live_output = temp_bundle_dir("live-refusal");
        assert!(matches!(
            create_run_bundle(&db_path, &live_output),
            Err(BundleError::Storage(StorageError::InvalidData {
                context: "storage.path_lease",
                ..
            }))
        ));
        assert!(!live_output.exists());
        storage.close()?;

        let bundle = create_run_bundle(&db_path, &bundle_dir)?;
        assert_eq!(bundle.bundle_version, RUN_BUNDLE_SCHEMA_VERSION);
        assert_eq!(bundle.artifacts.len(), 3);

        let verification = verify_run_bundle(&bundle_dir)?;
        assert_eq!(verification.bundle_version, RUN_BUNDLE_SCHEMA_VERSION);
        assert_eq!(verification.total_artifacts_verified, 3);
        assert!(verification.total_bytes_verified > 0);
        let original_manifest = fs::read(bundle_dir.join("bundle_manifest.json"))?;
        assert!(
            matches!(create_run_bundle(&db_path, &bundle_dir), Err(BundleError::Io { error, .. }) if error.kind() == std::io::ErrorKind::AlreadyExists)
        );
        assert_eq!(
            fs::read(bundle_dir.join("bundle_manifest.json"))?,
            original_manifest
        );

        // A hash-valid export with a missing logical checkpoint must still be refused.
        let mut contradictory = bundle.clone();
        let empty = b"[]";
        fs::write(bundle_dir.join("checkpoints.json"), empty)?;
        let checkpoint_entry = contradictory
            .artifacts
            .iter_mut()
            .find(|entry| entry.artifact_type == "checkpoints")
            .expect("checkpoint artifact");
        checkpoint_entry.blake3_hex = hash_hex(empty);
        checkpoint_entry.bytes_len = empty.len() as u64;
        contradictory.digests.checkpoint_count = 0;
        replace_test_manifest(&bundle_dir, &contradictory)?;
        assert!(matches!(
            verify_run_bundle(&bundle_dir),
            Err(BundleError::DatabaseProjectionMismatch("checkpoints.json"))
        ));
        let reader = StorageReader::open_finished(&bundle_dir.join("run.db").to_string_lossy())?;
        let checkpoints = serde_json::to_vec_pretty(&reader.load_checkpoints()?)?;
        reader.close()?;
        fs::write(bundle_dir.join("checkpoints.json"), checkpoints)?;
        replace_test_manifest(&bundle_dir, &bundle)?;
        let bounded = verify_run_bundle_bounded(
            &bundle_dir,
            RunBundleVerificationLimits {
                max_manifest_bytes: 1024 * 1024,
                max_artifacts: 8,
                max_artifact_bytes: 1024 * 1024,
                max_total_artifact_bytes: 4 * 1024 * 1024,
            },
        );
        assert!(matches!(
            bounded,
            Err(BundleError::BoundedDatabaseVerificationUnavailable(_))
        ));

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
        let bounded = verify_run_bundle_bounded(
            &bundle_dir,
            RunBundleVerificationLimits {
                max_manifest_bytes: 1024 * 1024,
                max_artifacts: 4,
                max_artifact_bytes: 1024,
                max_total_artifact_bytes: 2048,
            },
        )?;
        assert_eq!(bounded.total_artifacts_verified, 2);
        assert_eq!(
            bounded.total_bytes_verified,
            verification.total_bytes_verified
        );

        let capped = verify_run_bundle_bounded(
            &bundle_dir,
            RunBundleVerificationLimits {
                max_manifest_bytes: 1024 * 1024,
                max_artifacts: 4,
                max_artifact_bytes: 1,
                max_total_artifact_bytes: 2048,
            },
        );
        assert!(matches!(
            capped,
            Err(BundleError::VerificationLimitExceeded {
                resource: "artifact_bytes",
                ..
            })
        ));

        for claim in 0..3 {
            let mut contradictory = bundle.clone();
            match claim {
                0 => contradictory.digests.event_count = 1,
                1 => contradictory.digests.checkpoint_count = 1,
                _ => {
                    contradictory.digests.persistence_watermarks = Some(RunBundleWatermarks {
                        admitted: Some(1),
                        applied: Some(1),
                        durable: Some(1),
                    });
                }
            }
            replace_test_manifest(&bundle_dir, &contradictory)?;
            assert!(matches!(
                verify_run_bundle(&bundle_dir),
                Err(BundleError::MissingArtifact(path)) if path == Path::new("run.db")
            ));
            assert!(matches!(
                verify_run_bundle_bounded(
                    &bundle_dir,
                    RunBundleVerificationLimits {
                        max_manifest_bytes: 1024 * 1024,
                        max_artifacts: 4,
                        max_artifact_bytes: 1024,
                        max_total_artifact_bytes: 2048,
                    },
                ),
                Err(BundleError::MissingArtifact(path)) if path == Path::new("run.db")
            ));
        }

        let mut database_alias = bundle.clone();
        database_alias.artifacts[0].relative_path = "nested/renamed.sqlite".to_owned();
        database_alias.artifacts[0].artifact_type = "database".to_owned();
        fs::create_dir_all(bundle_dir.join("nested"))?;
        fs::write(bundle_dir.join("nested/renamed.sqlite"), summary_csv)?;
        database_alias.artifacts[0].blake3_hex = hash_hex(summary_csv);
        database_alias.artifacts[0].bytes_len = summary_csv.len() as u64;
        replace_test_manifest(&bundle_dir, &database_alias)?;
        assert!(matches!(
            verify_run_bundle_bounded(
                &bundle_dir,
                RunBundleVerificationLimits {
                    max_manifest_bytes: 1024 * 1024,
                    max_artifacts: 4,
                    max_artifact_bytes: 1024,
                    max_total_artifact_bytes: 2048,
                },
            ),
            Err(BundleError::BoundedDatabaseVerificationUnavailable(_))
        ));

        let mut duplicate = bundle.clone();
        duplicate.artifacts.push(duplicate.artifacts[0].clone());
        replace_test_manifest(&bundle_dir, &duplicate)?;
        assert!(matches!(
            verify_run_bundle_bounded(
                &bundle_dir,
                RunBundleVerificationLimits {
                    max_manifest_bytes: 1024 * 1024,
                    max_artifacts: 4,
                    max_artifact_bytes: 1024,
                    max_total_artifact_bytes: 2048,
                },
            ),
            Err(BundleError::DuplicateArtifactPath(_))
        ));

        let mut contradictory = bundle.clone();
        contradictory.digests.run_id = "different-run".to_owned();
        replace_test_manifest(&bundle_dir, &contradictory)?;
        assert!(matches!(
            verify_run_bundle_bounded(
                &bundle_dir,
                RunBundleVerificationLimits {
                    max_manifest_bytes: 1024 * 1024,
                    max_artifacts: 4,
                    max_artifact_bytes: 1024,
                    max_total_artifact_bytes: 2048,
                },
            ),
            Err(BundleError::RunIdProjectionMismatch { .. })
        ));
        replace_test_manifest(&bundle_dir, &bundle)?;

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
