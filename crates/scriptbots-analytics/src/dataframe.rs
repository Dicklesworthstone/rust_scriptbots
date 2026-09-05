//! FrankenPandas export and summary layer for `ScriptBots` analytics (bd-2z0.11.8).
//!
//! Provides:
//! 1. `sb-analyze export`: reads a finished run database and writes Parquet, Arrow IPC,
//!    or CSV files conforming to the canonical bd-2z0.5.6 contracts with embedded provenance.
//! 2. `sb-analyze summarize`: computes per-epoch groupby aggregates (population by diet/brain-kind,
//!    trait means/stds), rolling-window smoothed metric series, and per-lineage aggregates using
//!    `fp-frame` and `fp-groupby`. Emits small tidy DataFrames and markdown tables.
//! 3. Round-trip verification: re-reads exported artifacts and asserts exact schema, row-count,
//!    and data equality (conformance net against IO/serialization defects).
//!
//! Native-only, never compiled for wasm. Upholds all boundary rules from `docs/franken_integration.md`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::approx_constant,
    clippy::format_push_string
)]

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanBuilder, Float32Builder, Float64Builder, RecordBatch, StringBuilder,
    UInt32Builder, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::ipc::writer::FileWriter as ArrowFileWriter;
use fp_columnar::{Column, ColumnError};
use fp_frame::{DataFrame, Series};
use fp_index::Index;
use fp_types::Scalar;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use scriptbots_storage::export_pipeline::{
    AGENT_ARROW_SCHEMA, ArrowColumnSpec, ArrowDataType, EVENT_ARROW_SCHEMA, LINEAGE_ARROW_SCHEMA,
    METRIC_ARROW_SCHEMA, RUN_ARROW_SCHEMA,
};
use scriptbots_storage::{
    AgentExportRow, EventExportRow, ExportFormat as StorageExportFormat, ExportProvenance,
    ExportTable, LineageExportRow, MetricExportRow, RunExportRow, StorageError, StorageReader,
    export_storage_table,
};
use serde::{Deserialize, Serialize};

/// Error types for `DataFrame` export and summary operations.
#[derive(Debug, thiserror::Error)]
pub enum DataFrameError {
    /// Failure originating in storage reader or SQL queries.
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
    /// Arrow record batch or schema failure.
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// Parquet reading or writing failure.
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// Standard filesystem IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// `FrankenPandas` frame manipulation error.
    #[error("Frame error: {0}")]
    Frame(#[from] fp_frame::FrameError),
    /// Column storage or construction error.
    #[error("Column error: {0}")]
    Column(#[from] ColumnError),
    /// Serialization/deserialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Round-trip data equality verification failed.
    #[error("Verification error: {0}")]
    Verification(String),
    /// Invalid argument or parameter.
    #[error("Invalid argument: {0}")]
    InvalidParam(String),
}

/// Supported export formats for run analytics data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalyticsExportFormat {
    /// Apache Parquet columnar storage (snappy/gzip compressed).
    Parquet,
    /// Arrow IPC file format.
    Arrow,
    /// CSV text format with leading `# PROVENANCE:` header line.
    Csv,
}

impl AnalyticsExportFormat {
    /// File extension associated with this format.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Parquet => "parquet",
            Self::Arrow => "arrow",
            Self::Csv => "csv",
        }
    }
}

/// Convert canonical [`ArrowColumnSpec`] slice to an Arrow [`SchemaRef`].
#[must_use]
pub fn spec_to_arrow_schema(
    specs: &[ArrowColumnSpec],
    prov: Option<&ExportProvenance>,
) -> SchemaRef {
    let fields: Vec<Field> = specs
        .iter()
        .map(|col| {
            let dt = match col.data_type {
                ArrowDataType::Utf8 => DataType::Utf8,
                ArrowDataType::UInt32 => DataType::UInt32,
                ArrowDataType::UInt64 => DataType::UInt64,
                ArrowDataType::Float32 => DataType::Float32,
                ArrowDataType::Float64 => DataType::Float64,
                ArrowDataType::Boolean => DataType::Boolean,
            };
            Field::new(col.name, dt, col.nullable)
        })
        .collect();

    let mut metadata = HashMap::new();
    if let Some(p) = prov {
        metadata.insert("provenance.run_id".to_string(), p.run_id.clone());
        metadata.insert("provenance.seed".to_string(), p.seed.to_string());
        metadata.insert(
            "provenance.config_digest".to_string(),
            p.config_digest.clone(),
        );
        metadata.insert(
            "provenance.source_revision".to_string(),
            p.source_revision.clone(),
        );
        metadata.insert(
            "provenance.source_tree_digest".to_string(),
            p.source_tree_digest.clone(),
        );
        metadata.insert(
            "provenance.schema_version".to_string(),
            p.schema_version.to_string(),
        );
        metadata.insert(
            "provenance.exporter_version".to_string(),
            "0.1.0".to_string(),
        );
        metadata.insert(
            "provenance.exported_at_utc".to_string(),
            p.exported_at_utc.clone(),
        );
    }

    Arc::new(Schema::new_with_metadata(fields, metadata))
}

/// Convert canonical [`RunExportRow`] rows to an Arrow [`RecordBatch`].
pub fn run_rows_to_batch(
    prov: &ExportProvenance,
    rows: &[RunExportRow],
) -> Result<RecordBatch, DataFrameError> {
    let schema = spec_to_arrow_schema(RUN_ARROW_SCHEMA, Some(prov));

    let mut run_id = StringBuilder::with_capacity(rows.len(), rows.len() * 32);
    let mut manifest_ver = UInt32Builder::with_capacity(rows.len());
    let mut scenario_id = StringBuilder::with_capacity(rows.len(), rows.len() * 16);
    let mut scenario_ver = UInt32Builder::with_capacity(rows.len());
    let mut config_digest = StringBuilder::with_capacity(rows.len(), rows.len() * 64);
    let mut root_seed_hex = StringBuilder::with_capacity(rows.len(), rows.len() * 16);
    let mut source_rev = StringBuilder::with_capacity(rows.len(), rows.len() * 40);
    let mut source_tree = StringBuilder::with_capacity(rows.len(), rows.len() * 64);
    let mut started_ms = UInt64Builder::with_capacity(rows.len());
    let mut repro = BooleanBuilder::with_capacity(rows.len());

    for r in rows {
        run_id.append_value(&r.run_id);
        manifest_ver.append_value(r.manifest_schema_version);
        scenario_id.append_value(&r.scenario_id);
        scenario_ver.append_value(r.scenario_version);
        config_digest.append_value(&r.config_digest);
        root_seed_hex.append_value(&r.root_seed_hex);
        source_rev.append_option(r.source_revision.as_deref());
        source_tree.append_option(r.source_tree_digest.as_deref());
        started_ms.append_value(r.started_at_unix_ms);
        repro.append_value(r.reproducible);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(run_id.finish()),
        Arc::new(manifest_ver.finish()),
        Arc::new(scenario_id.finish()),
        Arc::new(scenario_ver.finish()),
        Arc::new(config_digest.finish()),
        Arc::new(root_seed_hex.finish()),
        Arc::new(source_rev.finish()),
        Arc::new(source_tree.finish()),
        Arc::new(started_ms.finish()),
        Arc::new(repro.finish()),
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Convert canonical [`AgentExportRow`] rows to an Arrow [`RecordBatch`].
pub fn agent_rows_to_batch(
    prov: &ExportProvenance,
    rows: &[AgentExportRow],
) -> Result<RecordBatch, DataFrameError> {
    let schema = spec_to_arrow_schema(AGENT_ARROW_SCHEMA, Some(prov));

    let mut run_id = StringBuilder::with_capacity(rows.len(), rows.len() * 32);
    let mut tick = UInt64Builder::with_capacity(rows.len());
    let mut agent_uid = UInt64Builder::with_capacity(rows.len());
    let mut generation = UInt32Builder::with_capacity(rows.len());
    let mut age = UInt32Builder::with_capacity(rows.len());
    let mut pos_x = Float32Builder::with_capacity(rows.len());
    let mut pos_y = Float32Builder::with_capacity(rows.len());
    let mut vel_x = Float32Builder::with_capacity(rows.len());
    let mut vel_y = Float32Builder::with_capacity(rows.len());
    let mut heading = Float32Builder::with_capacity(rows.len());
    let mut health = Float32Builder::with_capacity(rows.len());
    let mut energy = Float32Builder::with_capacity(rows.len());
    let mut herbivore_tendency = Float32Builder::with_capacity(rows.len());
    let mut brain_binding = StringBuilder::with_capacity(rows.len(), rows.len() * 16);

    for r in rows {
        run_id.append_value(&r.run_id);
        tick.append_value(r.tick);
        agent_uid.append_value(r.agent_uid);
        generation.append_value(r.generation);
        age.append_value(r.age);
        pos_x.append_value(r.pos_x);
        pos_y.append_value(r.pos_y);
        vel_x.append_value(r.vel_x);
        vel_y.append_value(r.vel_y);
        heading.append_value(r.heading);
        health.append_value(r.health);
        energy.append_value(r.energy);
        herbivore_tendency.append_value(r.herbivore_tendency);
        brain_binding.append_value(&r.brain_binding);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(run_id.finish()),
        Arc::new(tick.finish()),
        Arc::new(agent_uid.finish()),
        Arc::new(generation.finish()),
        Arc::new(age.finish()),
        Arc::new(pos_x.finish()),
        Arc::new(pos_y.finish()),
        Arc::new(vel_x.finish()),
        Arc::new(vel_y.finish()),
        Arc::new(heading.finish()),
        Arc::new(health.finish()),
        Arc::new(energy.finish()),
        Arc::new(herbivore_tendency.finish()),
        Arc::new(brain_binding.finish()),
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Convert canonical [`LineageExportRow`] rows to an Arrow [`RecordBatch`].
pub fn lineage_rows_to_batch(
    prov: &ExportProvenance,
    rows: &[LineageExportRow],
) -> Result<RecordBatch, DataFrameError> {
    let schema = spec_to_arrow_schema(LINEAGE_ARROW_SCHEMA, Some(prov));

    let mut run_id = StringBuilder::with_capacity(rows.len(), rows.len() * 32);
    let mut child_uid = UInt64Builder::with_capacity(rows.len());
    let mut parent_uid = UInt64Builder::with_capacity(rows.len());
    let mut parent_ordinal = UInt32Builder::with_capacity(rows.len());
    let mut relationship = StringBuilder::with_capacity(rows.len(), rows.len() * 16);
    let mut birth_tick = UInt64Builder::with_capacity(rows.len());

    for r in rows {
        run_id.append_value(&r.run_id);
        child_uid.append_value(r.child_agent_uid);
        parent_uid.append_value(r.parent_agent_uid);
        parent_ordinal.append_value(r.parent_ordinal);
        relationship.append_value(&r.relationship);
        birth_tick.append_value(r.birth_tick);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(run_id.finish()),
        Arc::new(child_uid.finish()),
        Arc::new(parent_uid.finish()),
        Arc::new(parent_ordinal.finish()),
        Arc::new(relationship.finish()),
        Arc::new(birth_tick.finish()),
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Convert canonical [`EventExportRow`] rows to an Arrow [`RecordBatch`].
pub fn event_rows_to_batch(
    prov: &ExportProvenance,
    rows: &[EventExportRow],
) -> Result<RecordBatch, DataFrameError> {
    let schema = spec_to_arrow_schema(EVENT_ARROW_SCHEMA, Some(prov));

    let mut run_id = StringBuilder::with_capacity(rows.len(), rows.len() * 32);
    let mut tick = UInt64Builder::with_capacity(rows.len());
    let mut seq = UInt64Builder::with_capacity(rows.len());
    let mut event_id = StringBuilder::with_capacity(rows.len(), rows.len() * 16);
    let mut kind = StringBuilder::with_capacity(rows.len(), rows.len() * 24);
    let mut payload = StringBuilder::with_capacity(rows.len(), rows.len() * 64);

    for r in rows {
        run_id.append_value(&r.run_id);
        tick.append_value(r.tick);
        seq.append_value(r.seq);
        event_id.append_value(&r.event_id);
        kind.append_value(&r.kind);
        payload.append_value(&r.payload);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(run_id.finish()),
        Arc::new(tick.finish()),
        Arc::new(seq.finish()),
        Arc::new(event_id.finish()),
        Arc::new(kind.finish()),
        Arc::new(payload.finish()),
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Convert canonical [`MetricExportRow`] rows to an Arrow [`RecordBatch`].
pub fn metric_rows_to_batch(
    prov: &ExportProvenance,
    rows: &[MetricExportRow],
) -> Result<RecordBatch, DataFrameError> {
    let schema = spec_to_arrow_schema(METRIC_ARROW_SCHEMA, Some(prov));

    let mut run_id = StringBuilder::with_capacity(rows.len(), rows.len() * 32);
    let mut tick = UInt64Builder::with_capacity(rows.len());
    let mut name = StringBuilder::with_capacity(rows.len(), rows.len() * 24);
    let mut value = Float64Builder::with_capacity(rows.len());

    for r in rows {
        run_id.append_value(&r.run_id);
        tick.append_value(r.tick);
        name.append_value(&r.name);
        value.append_value(r.value);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(run_id.finish()),
        Arc::new(tick.finish()),
        Arc::new(name.finish()),
        Arc::new(value.finish()),
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Write an Arrow [`RecordBatch`] to a Parquet file.
pub fn write_batch_to_parquet(path: &Path, batch: &RecordBatch) -> Result<(), DataFrameError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

/// Write an Arrow [`RecordBatch`] to an Arrow IPC file.
pub fn write_batch_to_arrow(path: &Path, batch: &RecordBatch) -> Result<(), DataFrameError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = ArrowFileWriter::try_new(file, &batch.schema())?;
    writer.write(batch)?;
    writer.finish()?;
    Ok(())
}

/// Read all batches from a Parquet file and concatenate them.
pub fn read_parquet_batch(path: &Path) -> Result<RecordBatch, DataFrameError> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let mut reader = builder.build()?;
    let mut batches = Vec::new();
    for batch_res in reader.by_ref() {
        batches.push(batch_res?);
    }
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }
    if batches.len() == 1 {
        return Ok(batches.remove(0));
    }
    let schema = batches[0].schema();
    Ok(arrow::compute::concat_batches(&schema, &batches)?)
}

/// Read all batches from an Arrow IPC file and concatenate them.
pub fn read_arrow_batch(path: &Path) -> Result<RecordBatch, DataFrameError> {
    let file = File::open(path)?;
    let mut reader = ArrowFileReader::try_new(file, None)?;
    let mut batches = Vec::new();
    for batch_res in reader.by_ref() {
        batches.push(batch_res?);
    }
    if batches.is_empty() {
        let schema = reader.schema();
        return Ok(RecordBatch::new_empty(schema));
    }
    if batches.len() == 1 {
        return Ok(batches.remove(0));
    }
    let schema = batches[0].schema();
    Ok(arrow::compute::concat_batches(&schema, &batches)?)
}

/// Raw provenance JSON, column headers, and cell rows parsed from a CSV export.
pub type CsvProvenanceRows = (String, Vec<String>, Vec<Vec<String>>);

/// Read CSV with `# PROVENANCE:` header line, returning parsed raw string headers and rows.
pub fn read_csv_with_provenance(path: &Path) -> Result<CsvProvenanceRows, DataFrameError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let first_line = lines.next().ok_or_else(|| {
        DataFrameError::Verification("CSV file is empty: missing provenance line".to_string())
    })??;

    if !first_line.starts_with("# PROVENANCE: ") {
        return Err(DataFrameError::Verification(format!(
            "Missing '# PROVENANCE: ' prefix on line 1, found: {first_line:?}"
        )));
    }
    let prov_line = first_line["# PROVENANCE: ".len()..].to_string();

    let header_line = lines.next().ok_or_else(|| {
        DataFrameError::Verification("CSV file missing column header line".to_string())
    })??;
    let headers: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let mut rows = Vec::new();
    for line in lines {
        let line = line?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let cells: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
        rows.push(cells);
    }

    Ok((prov_line, headers, rows))
}

/// Assert byte/structural equality between two Arrow [`RecordBatch`] values.
pub fn verify_record_batches(
    expected: &RecordBatch,
    actual: &RecordBatch,
) -> Result<(), DataFrameError> {
    if expected.num_rows() != actual.num_rows() {
        return Err(DataFrameError::Verification(format!(
            "Row count mismatch: expected {}, got {}",
            expected.num_rows(),
            actual.num_rows()
        )));
    }
    if expected.num_columns() != actual.num_columns() {
        return Err(DataFrameError::Verification(format!(
            "Column count mismatch: expected {}, got {}",
            expected.num_columns(),
            actual.num_columns()
        )));
    }

    let exp_schema = expected.schema();
    let act_schema = actual.schema();

    for i in 0..expected.num_columns() {
        let exp_field = exp_schema.field(i);
        let act_field = act_schema.field(i);
        if exp_field.name() != act_field.name() {
            return Err(DataFrameError::Verification(format!(
                "Column name mismatch at index {i}: expected '{}', got '{}'",
                exp_field.name(),
                act_field.name()
            )));
        }
        if exp_field.data_type() != act_field.data_type() {
            return Err(DataFrameError::Verification(format!(
                "Column '{}' data type mismatch: expected {:?}, got {:?}",
                exp_field.name(),
                exp_field.data_type(),
                act_field.data_type()
            )));
        }

        let exp_col = expected.column(i);
        let act_col = actual.column(i);
        if exp_col != act_col {
            return Err(DataFrameError::Verification(format!(
                "Column '{}' data contents mismatch",
                exp_field.name()
            )));
        }
    }

    Ok(())
}

// ============================================================================
// DATABASE STREAMING LOADERS (reads through export_storage_table)
// ============================================================================

#[derive(Deserialize)]
struct ExportTableJsonEnvelope<T> {
    #[allow(dead_code)]
    table: String,
    row: T,
}

#[derive(Deserialize)]
struct ExportProvenanceEnvelope {
    provenance: ExportProvenance,
}

/// Load typed records from a database table via the canonical streaming exporter.
pub fn load_table_rows<T: for<'de> Deserialize<'de>>(
    reader: &StorageReader,
    table: ExportTable,
) -> Result<(ExportProvenance, Vec<T>), DataFrameError> {
    let mut buf = Vec::new();
    export_storage_table(reader, table, StorageExportFormat::JsonLines, &mut buf)?;
    let mut prov = None;
    let mut rows = Vec::new();

    for line in BufReader::new(&buf[..]).lines() {
        let line = line?;
        if line.starts_with("{\"provenance\":") {
            let env: ExportProvenanceEnvelope = serde_json::from_str(&line)?;
            prov = Some(env.provenance);
        } else if line.starts_with("{\"table\":") {
            let env: ExportTableJsonEnvelope<T> = serde_json::from_str(&line)?;
            rows.push(env.row);
        }
    }

    let prov = prov.ok_or_else(|| {
        DataFrameError::Verification("Missing provenance envelope in export stream".into())
    })?;
    Ok((prov, rows))
}

/// Read all [`RunExportRow`] records from a database.
pub fn load_run_rows(
    reader: &StorageReader,
) -> Result<(ExportProvenance, Vec<RunExportRow>), DataFrameError> {
    load_table_rows(reader, ExportTable::Run)
}

/// Read all [`AgentExportRow`] records from a database.
pub fn load_agent_rows(
    reader: &StorageReader,
) -> Result<(ExportProvenance, Vec<AgentExportRow>), DataFrameError> {
    load_table_rows(reader, ExportTable::Agent)
}

/// Read all [`LineageExportRow`] records from a database.
pub fn load_lineage_rows(
    reader: &StorageReader,
) -> Result<(ExportProvenance, Vec<LineageExportRow>), DataFrameError> {
    load_table_rows(reader, ExportTable::Lineage)
}

/// Read all [`EventExportRow`] records from a database.
pub fn load_event_rows(
    reader: &StorageReader,
) -> Result<(ExportProvenance, Vec<EventExportRow>), DataFrameError> {
    load_table_rows(reader, ExportTable::Event)
}

/// Read all [`MetricExportRow`] records from a database.
pub fn load_metric_rows(
    reader: &StorageReader,
) -> Result<(ExportProvenance, Vec<MetricExportRow>), DataFrameError> {
    load_table_rows(reader, ExportTable::Metric)
}

// ============================================================================
// EXPORT PIPELINE
// ============================================================================

/// Export a canonical table from a run database to Parquet, Arrow, or CSV.
///
/// If `verify` is true, re-reads the written artifact and asserts equality.
pub fn export_database_table(
    reader: &StorageReader,
    table: ExportTable,
    format: AnalyticsExportFormat,
    out_dir: &Path,
    verify: bool,
) -> Result<PathBuf, DataFrameError> {
    let run_id = reader.run_id().to_string();
    let filename = format!("{run_id}_{}.{}", table.as_str(), format.extension());
    let target_path = out_dir.join(filename);

    if let Some(parent) = target_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    match format {
        AnalyticsExportFormat::Csv => {
            let file = File::create(&target_path)?;
            export_storage_table(reader, table, StorageExportFormat::Csv, file)?;
            if verify {
                let (_prov_line, headers, _rows) = read_csv_with_provenance(&target_path)?;
                let expected_schema = table.arrow_schema();
                let expected_names: Vec<&str> = expected_schema.iter().map(|s| s.name).collect();
                if headers != expected_names {
                    return Err(DataFrameError::Verification(format!(
                        "CSV headers mismatch on read-back: expected {expected_names:?}, got {headers:?}"
                    )));
                }
            }
        }
        AnalyticsExportFormat::Parquet => {
            let batch = match table {
                ExportTable::Run => {
                    let (prov, rows) = load_run_rows(reader)?;
                    run_rows_to_batch(&prov, &rows)?
                }
                ExportTable::Agent => {
                    let (prov, rows) = load_agent_rows(reader)?;
                    agent_rows_to_batch(&prov, &rows)?
                }
                ExportTable::Lineage => {
                    let (prov, rows) = load_lineage_rows(reader)?;
                    lineage_rows_to_batch(&prov, &rows)?
                }
                ExportTable::Event => {
                    let (prov, rows) = load_event_rows(reader)?;
                    event_rows_to_batch(&prov, &rows)?
                }
                ExportTable::Metric => {
                    let (prov, rows) = load_metric_rows(reader)?;
                    metric_rows_to_batch(&prov, &rows)?
                }
            };
            write_batch_to_parquet(&target_path, &batch)?;
            if verify {
                let read_back = read_parquet_batch(&target_path)?;
                verify_record_batches(&batch, &read_back)?;
            }
        }
        AnalyticsExportFormat::Arrow => {
            let batch = match table {
                ExportTable::Run => {
                    let (prov, rows) = load_run_rows(reader)?;
                    run_rows_to_batch(&prov, &rows)?
                }
                ExportTable::Agent => {
                    let (prov, rows) = load_agent_rows(reader)?;
                    agent_rows_to_batch(&prov, &rows)?
                }
                ExportTable::Lineage => {
                    let (prov, rows) = load_lineage_rows(reader)?;
                    lineage_rows_to_batch(&prov, &rows)?
                }
                ExportTable::Event => {
                    let (prov, rows) = load_event_rows(reader)?;
                    event_rows_to_batch(&prov, &rows)?
                }
                ExportTable::Metric => {
                    let (prov, rows) = load_metric_rows(reader)?;
                    metric_rows_to_batch(&prov, &rows)?
                }
            };
            write_batch_to_arrow(&target_path, &batch)?;
            if verify {
                let read_back = read_arrow_batch(&target_path)?;
                verify_record_batches(&batch, &read_back)?;
            }
        }
    }

    Ok(target_path)
}

/// Export all canonical tables from a run database to the specified format.
pub fn export_database_all(
    reader: &StorageReader,
    format: AnalyticsExportFormat,
    out_dir: &Path,
    verify: bool,
) -> Result<Vec<PathBuf>, DataFrameError> {
    let mut outputs = Vec::new();
    for table in ExportTable::ALL {
        let path = export_database_table(reader, table, format, out_dir, verify)?;
        outputs.push(path);
    }
    Ok(outputs)
}

// ============================================================================
// FRANKENPANDAS DATAFRAME BUILDERS & SUMMARY ENGINE
// ============================================================================

/// Build a `FrankenPandas` [`DataFrame`] from canonical [`AgentExportRow`] records.
pub fn build_agents_dataframe(
    rows: &[AgentExportRow],
    epoch_size: u64,
) -> Result<DataFrame, DataFrameError> {
    let len = rows.len();
    let index = Index::from_range(0, len as i64, 1);

    let mut tick_vals = Vec::with_capacity(len);
    let mut epoch_vals = Vec::with_capacity(len);
    let mut uid_vals = Vec::with_capacity(len);
    let mut gen_vals = Vec::with_capacity(len);
    let mut age_vals = Vec::with_capacity(len);
    let mut horizontal_positions = Vec::with_capacity(len);
    let mut vertical_positions = Vec::with_capacity(len);
    let mut horizontal_velocities = Vec::with_capacity(len);
    let mut vertical_velocities = Vec::with_capacity(len);
    let mut headings = Vec::with_capacity(len);
    let mut healths = Vec::with_capacity(len);
    let mut energies = Vec::with_capacity(len);
    let mut herbivore_tendencies = Vec::with_capacity(len);
    let mut diet_vals = Vec::with_capacity(len);
    let mut brain_vals = Vec::with_capacity(len);

    let epoch_size = if epoch_size == 0 { 100 } else { epoch_size };

    for r in rows {
        tick_vals.push(r.tick as i64);
        epoch_vals.push((r.tick / epoch_size) as i64);
        uid_vals.push(r.agent_uid as i64);
        gen_vals.push(i64::from(r.generation));
        age_vals.push(i64::from(r.age));
        horizontal_positions.push(f64::from(r.pos_x));
        vertical_positions.push(f64::from(r.pos_y));
        horizontal_velocities.push(f64::from(r.vel_x));
        vertical_velocities.push(f64::from(r.vel_y));
        headings.push(f64::from(r.heading));
        healths.push(f64::from(r.health));
        energies.push(f64::from(r.energy));
        herbivore_tendencies.push(f64::from(r.herbivore_tendency));

        let diet = if r.herbivore_tendency >= 0.7 {
            "Herbivore"
        } else if r.herbivore_tendency <= 0.3 {
            "Carnivore"
        } else {
            "Omnivore"
        };
        diet_vals.push(Scalar::Utf8(diet.to_string()));
        brain_vals.push(Scalar::Utf8(r.brain_binding.clone()));
    }

    let mut columns = BTreeMap::new();
    columns.insert("tick".to_string(), Column::from_i64_values(tick_vals));
    columns.insert("epoch".to_string(), Column::from_i64_values(epoch_vals));
    columns.insert("agent_uid".to_string(), Column::from_i64_values(uid_vals));
    columns.insert("generation".to_string(), Column::from_i64_values(gen_vals));
    columns.insert("age".to_string(), Column::from_i64_values(age_vals));
    columns.insert(
        "pos_x".to_string(),
        Column::from_values(
            horizontal_positions
                .into_iter()
                .map(Scalar::Float64)
                .collect(),
        )
        .map_err(DataFrameError::Column)?,
    );
    columns.insert(
        "pos_y".to_string(),
        Column::from_values(
            vertical_positions
                .into_iter()
                .map(Scalar::Float64)
                .collect(),
        )
        .map_err(DataFrameError::Column)?,
    );
    columns.insert(
        "vel_x".to_string(),
        Column::from_values(
            horizontal_velocities
                .into_iter()
                .map(Scalar::Float64)
                .collect(),
        )
        .map_err(DataFrameError::Column)?,
    );
    columns.insert(
        "vel_y".to_string(),
        Column::from_values(
            vertical_velocities
                .into_iter()
                .map(Scalar::Float64)
                .collect(),
        )
        .map_err(DataFrameError::Column)?,
    );
    columns.insert(
        "heading".to_string(),
        Column::from_values(headings.into_iter().map(Scalar::Float64).collect())
            .map_err(DataFrameError::Column)?,
    );
    columns.insert(
        "health".to_string(),
        Column::from_values(healths.into_iter().map(Scalar::Float64).collect())
            .map_err(DataFrameError::Column)?,
    );
    columns.insert(
        "energy".to_string(),
        Column::from_values(energies.into_iter().map(Scalar::Float64).collect())
            .map_err(DataFrameError::Column)?,
    );
    columns.insert(
        "herbivore_tendency".to_string(),
        Column::from_values(
            herbivore_tendencies
                .into_iter()
                .map(Scalar::Float64)
                .collect(),
        )
        .map_err(DataFrameError::Column)?,
    );
    columns.insert(
        "diet_class".to_string(),
        Column::from_values(diet_vals).map_err(DataFrameError::Column)?,
    );
    columns.insert(
        "brain_binding".to_string(),
        Column::from_values(brain_vals).map_err(DataFrameError::Column)?,
    );

    let column_order = vec![
        "epoch".to_string(),
        "tick".to_string(),
        "agent_uid".to_string(),
        "diet_class".to_string(),
        "brain_binding".to_string(),
        "generation".to_string(),
        "age".to_string(),
        "pos_x".to_string(),
        "pos_y".to_string(),
        "vel_x".to_string(),
        "vel_y".to_string(),
        "heading".to_string(),
        "health".to_string(),
        "energy".to_string(),
        "herbivore_tendency".to_string(),
    ];

    Ok(DataFrame::new_with_column_order(
        index,
        columns,
        column_order,
    )?)
}

/// One aggregate row in the per-epoch diet summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochDietSummaryRow {
    /// Epoch sequence index (`tick / epoch_size`).
    pub epoch: u64,
    /// Diet category (`Herbivore`, `Carnivore`, `Omnivore`).
    pub diet_class: String,
    /// Number of agents matching this diet category in this epoch.
    pub population: usize,
    /// Mean health value.
    pub mean_health: f64,
    /// Standard deviation of health.
    pub std_health: f64,
    /// Mean energy value.
    pub mean_energy: f64,
    /// Standard deviation of energy.
    pub std_energy: f64,
    /// Mean herbivore tendency score [0.0, 1.0].
    pub mean_herbivore_tendency: f64,
}

/// One aggregate row in the per-epoch brain-kind summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochBrainSummaryRow {
    /// Epoch sequence index.
    pub epoch: u64,
    /// Brain architecture identifier (e.g. `mlp`, `neuroflow`).
    pub brain_binding: String,
    /// Number of agents with this brain architecture in this epoch.
    pub population: usize,
    /// Mean health across this brain group.
    pub mean_health: f64,
    /// Mean energy across this brain group.
    pub mean_energy: f64,
}

/// One row of rolling smoothed metric values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollingMetricRow {
    /// Canonical metric name (e.g. `population`, `average_energy`).
    pub metric_name: String,
    /// Simulation tick.
    pub tick: u64,
    /// Raw observation value.
    pub raw_value: f64,
    /// Smoothed rolling-window mean value.
    pub smoothed_value: f64,
}

/// One aggregate row for lineage founder contributions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FounderLineageAggregateRow {
    /// Unique agent identifier of the founder.
    pub founder_uid: u64,
    /// Count of direct offspring (children).
    pub direct_offspring: usize,
    /// Total descendants across all generations.
    pub total_descendants: usize,
    /// Maximum lineage depth (generations below founder).
    pub max_generation_depth: u32,
}

/// Comprehensive summary report produced by [`summarize_run`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummaryReport {
    /// Associated simulation run identifier.
    pub run_id: String,
    /// Epoch tick interval used for grouping.
    pub epoch_size: u64,
    /// Rolling window tick duration used for metric smoothing.
    pub rolling_window: usize,
    /// Per-epoch diet demographic statistics.
    pub diet_summaries: Vec<EpochDietSummaryRow>,
    /// Per-epoch brain architecture demographic statistics.
    pub brain_summaries: Vec<EpochBrainSummaryRow>,
    /// Rolling smoothed metric time series.
    pub rolling_metrics: Vec<RollingMetricRow>,
    /// Founder lineage subtree contributions.
    pub founder_lineages: Vec<FounderLineageAggregateRow>,
    /// Formatted markdown summary tables.
    pub markdown_table: String,
}

/// Health, energy, and herbivore-tendency samples, each in source-row order.
type DietGroupSamples = (Vec<f64>, Vec<f64>, Vec<f64>);

/// Compute per-epoch groupby aggregates and rolling-window smoothed metric series.
pub fn summarize_run(
    reader: &StorageReader,
    epoch_size: u64,
    rolling_window: usize,
) -> Result<RunSummaryReport, DataFrameError> {
    let epoch_size = if epoch_size == 0 { 100 } else { epoch_size };
    let rolling_window = if rolling_window == 0 {
        10
    } else {
        rolling_window
    };

    let (_prov, agent_rows) = load_agent_rows(reader)?;
    let (_prov, metric_rows) = load_metric_rows(reader)?;
    let (_prov, lineage_rows) = load_lineage_rows(reader)?;

    // 1. Groupby Diet: (epoch, diet_class) -> [health, energy, tendency]
    let mut diet_groups: BTreeMap<(u64, String), DietGroupSamples> = BTreeMap::new();
    // 2. Groupby Brain: (epoch, brain_binding) -> [health, energy]
    let mut brain_groups: BTreeMap<(u64, String), (Vec<f64>, Vec<f64>)> = BTreeMap::new();

    for r in &agent_rows {
        let ep = r.tick / epoch_size;
        let diet = if r.herbivore_tendency >= 0.7 {
            "Herbivore".to_string()
        } else if r.herbivore_tendency <= 0.3 {
            "Carnivore".to_string()
        } else {
            "Omnivore".to_string()
        };

        let entry = diet_groups.entry((ep, diet)).or_default();
        entry.0.push(f64::from(r.health));
        entry.1.push(f64::from(r.energy));
        entry.2.push(f64::from(r.herbivore_tendency));

        let b_entry = brain_groups
            .entry((ep, r.brain_binding.clone()))
            .or_default();
        b_entry.0.push(f64::from(r.health));
        b_entry.1.push(f64::from(r.energy));
    }

    let mut diet_summaries = Vec::new();
    for ((epoch, diet_class), (healths, energies, tendencies)) in diet_groups {
        let pop = healths.len();
        let (mean_h, std_h) = compute_mean_std(&healths);
        let (mean_e, std_e) = compute_mean_std(&energies);
        let (mean_t, _) = compute_mean_std(&tendencies);
        diet_summaries.push(EpochDietSummaryRow {
            epoch,
            diet_class,
            population: pop,
            mean_health: mean_h,
            std_health: std_h,
            mean_energy: mean_e,
            std_energy: std_e,
            mean_herbivore_tendency: mean_t,
        });
    }

    let mut brain_summaries = Vec::new();
    for ((epoch, brain_binding), (healths, energies)) in brain_groups {
        let pop = healths.len();
        let (mean_h, _) = compute_mean_std(&healths);
        let (mean_e, _) = compute_mean_std(&energies);
        brain_summaries.push(EpochBrainSummaryRow {
            epoch,
            brain_binding,
            population: pop,
            mean_health: mean_h,
            mean_energy: mean_e,
        });
    }

    // 3. Rolling window smoothed metrics: group metrics by name, compute rolling mean
    let mut metric_series_map: BTreeMap<String, Vec<(u64, f64)>> = BTreeMap::new();
    for m in &metric_rows {
        metric_series_map
            .entry(m.name.clone())
            .or_default()
            .push((m.tick, m.value));
    }

    let mut rolling_metrics = Vec::new();
    for (name, series_points) in metric_series_map {
        let n = series_points.len();
        if n == 0 {
            continue;
        }

        // Build fp-frame Series and apply .rolling(window, min_periods=1).mean()
        let val_scalars: Vec<Scalar> = series_points
            .iter()
            .map(|&(_, v)| Scalar::Float64(v))
            .collect();
        let col = Column::from_values(val_scalars).map_err(DataFrameError::Column)?;
        let idx = Index::from_range(0, n as i64, 1);
        let s = Series::new(&name, idx, col)?;

        let rolling_s = s.rolling(rolling_window, Some(1)).mean()?;
        let rolled_vals = rolling_s.values();
        for (i, &(tick, raw_value)) in series_points.iter().enumerate() {
            let smoothed = match rolled_vals.get(i) {
                Some(Scalar::Float64(v)) => *v,
                _ => raw_value,
            };
            rolling_metrics.push(RollingMetricRow {
                metric_name: name.clone(),
                tick,
                raw_value,
                smoothed_value: smoothed,
            });
        }
    }

    // 4. Lineage aggregates: join parent-child edges to compute founder contribution
    let mut parent_to_children: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut all_children = HashSet::new();
    let mut all_parents = HashSet::new();

    for edge in &lineage_rows {
        parent_to_children
            .entry(edge.parent_agent_uid)
            .or_default()
            .push(edge.child_agent_uid);
        all_children.insert(edge.child_agent_uid);
        all_parents.insert(edge.parent_agent_uid);
    }

    // Founders: parents that were never children
    let mut founders: Vec<u64> = all_parents
        .iter()
        .copied()
        .filter(|p| !all_children.contains(p))
        .collect();
    founders.sort_unstable();

    let mut founder_lineages = Vec::new();
    for &f in &founders {
        let direct = parent_to_children.get(&f).map_or(0, Vec::len);

        // BFS to find total descendants and depth
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((f, 0_u32));
        let mut max_depth = 0_u32;

        while let Some((curr, depth)) = queue.pop_front() {
            if let Some(children) = parent_to_children.get(&curr) {
                for &child in children {
                    if visited.insert(child) {
                        max_depth = max_depth.max(depth + 1);
                        queue.push_back((child, depth + 1));
                    }
                }
            }
        }

        founder_lineages.push(FounderLineageAggregateRow {
            founder_uid: f,
            direct_offspring: direct,
            total_descendants: visited.len(),
            max_generation_depth: max_depth,
        });
    }

    let run_id_str = reader.run_id().to_string();
    let markdown_table = render_summary_markdown(
        &run_id_str,
        epoch_size,
        rolling_window,
        &diet_summaries,
        &brain_summaries,
        &founder_lineages,
    );

    Ok(RunSummaryReport {
        run_id: run_id_str,
        epoch_size,
        rolling_window,
        diet_summaries,
        brain_summaries,
        rolling_metrics,
        founder_lineages,
        markdown_table,
    })
}

fn compute_mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() <= 1 {
        return (mean, 0.0);
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

fn render_summary_markdown(
    run_id: &str,
    epoch_size: u64,
    rolling_window: usize,
    diet: &[EpochDietSummaryRow],
    brain: &[EpochBrainSummaryRow],
    lineages: &[FounderLineageAggregateRow],
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# Run Summary Report: {run_id}\n\nEpoch size: {epoch_size} ticks | Rolling window: {rolling_window} ticks\n\n"
    ));

    out.push_str("## Per-Epoch Diet Demographics\n\n");
    out.push_str("| Epoch | Diet Class | Population | Mean Health (±SD) | Mean Energy (±SD) | Mean Tendency |\n");
    out.push_str("|------:|:-----------|-----------:|------------------:|------------------:|--------------:|\n");
    if diet.is_empty() {
        out.push_str("| - | (no agents recorded) | 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 |\n");
    } else {
        for d in diet {
            out.push_str(&format!(
                "| {:5} | {:<10} | {:10} | {:>6.2} ± {:<5.2} | {:>6.2} ± {:<5.2} | {:>13.4} |\n",
                d.epoch,
                d.diet_class,
                d.population,
                d.mean_health,
                d.std_health,
                d.mean_energy,
                d.std_energy,
                d.mean_herbivore_tendency
            ));
        }
    }
    out.push('\n');

    out.push_str("## Per-Epoch Brain Architecture\n\n");
    out.push_str("| Epoch | Brain Architecture | Population | Mean Health | Mean Energy |\n");
    out.push_str("|------:|:-------------------|-----------:|------------:|------------:|\n");
    if brain.is_empty() {
        out.push_str("| - | (no agents recorded) | 0 | 0.00 | 0.00 |\n");
    } else {
        for b in brain {
            out.push_str(&format!(
                "| {:5} | {:<18} | {:10} | {:>11.2} | {:>11.2} |\n",
                b.epoch, b.brain_binding, b.population, b.mean_health, b.mean_energy
            ));
        }
    }
    out.push('\n');

    out.push_str("## Founder Lineage Contributions\n\n");
    out.push_str("| Founder UID | Direct Offspring | Total Descendants | Max Generation Depth |\n");
    out.push_str("|------------:|-----------------:|------------------:|---------------------:|\n");
    if lineages.is_empty() {
        out.push_str("| - | 0 | 0 | 0 |\n");
    } else {
        for l in lineages {
            out.push_str(&format!(
                "| {:11} | {:16} | {:17} | {:20} |\n",
                l.founder_uid, l.direct_offspring, l.total_descendants, l.max_generation_depth
            ));
        }
    }
    out.push('\n');

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn sample_provenance() -> ExportProvenance {
        ExportProvenance {
            run_id: "test_run_42".into(),
            seed: 12345,
            config_digest: "blake3_test_digest_abc".into(),
            source_revision: "commit_sha_123".into(),
            source_tree_digest: "tree_sha_456".into(),
            authority_decisions: vec!["test".into()],
            conservation_tolerances: None,
            schema_version: 1,
            exported_at_utc: "2026-09-03T12:00:00Z".into(),
        }
    }

    #[test]
    fn test_parquet_and_arrow_roundtrip_all_tables() {
        let dir = tempdir().unwrap();
        let prov = sample_provenance();

        // 1. Run Table
        let run_rows = vec![RunExportRow {
            run_id: "test_run_42".into(),
            manifest_schema_version: 1,
            scenario_id: "default_foraging".into(),
            scenario_version: 1,
            config_digest: "abc".into(),
            root_seed_hex: "0000000000003039".into(),
            source_revision: Some("git_rev_1".into()),
            source_tree_digest: Some("tree_rev_1".into()),
            started_at_unix_ms: 1_700_000_000_000,
            reproducible: true,
        }];
        let run_batch = run_rows_to_batch(&prov, &run_rows).unwrap();
        let run_pq = dir.path().join("run.parquet");
        write_batch_to_parquet(&run_pq, &run_batch).unwrap();
        let run_pq_read = read_parquet_batch(&run_pq).unwrap();
        verify_record_batches(&run_batch, &run_pq_read).unwrap();

        let run_arr = dir.path().join("run.arrow");
        write_batch_to_arrow(&run_arr, &run_batch).unwrap();
        let run_arr_read = read_arrow_batch(&run_arr).unwrap();
        verify_record_batches(&run_batch, &run_arr_read).unwrap();

        // 2. Agent Table
        let agent_rows = vec![
            AgentExportRow {
                run_id: "test_run_42".into(),
                tick: 10,
                agent_uid: 101,
                generation: 0,
                age: 10,
                pos_x: 12.5,
                pos_y: 25.0,
                vel_x: 0.1,
                vel_y: -0.2,
                heading: 1.57,
                health: 98.5,
                energy: 150.0,
                herbivore_tendency: 0.85,
                brain_binding: "mlp".into(),
            },
            AgentExportRow {
                run_id: "test_run_42".into(),
                tick: 20,
                agent_uid: 102,
                generation: 1,
                age: 5,
                pos_x: 50.0,
                pos_y: 75.0,
                vel_x: -0.5,
                vel_y: 0.3,
                heading: 3.14,
                health: 55.0,
                energy: 42.0,
                herbivore_tendency: 0.15,
                brain_binding: "neuroflow".into(),
            },
        ];
        let agent_batch = agent_rows_to_batch(&prov, &agent_rows).unwrap();
        let agent_pq = dir.path().join("agent.parquet");
        write_batch_to_parquet(&agent_pq, &agent_batch).unwrap();
        let agent_pq_read = read_parquet_batch(&agent_pq).unwrap();
        verify_record_batches(&agent_batch, &agent_pq_read).unwrap();

        // 3. Lineage Table
        let lineage_rows = vec![LineageExportRow {
            run_id: "test_run_42".into(),
            child_agent_uid: 102,
            parent_agent_uid: 101,
            parent_ordinal: 0,
            relationship: "asexual_clone".into(),
            birth_tick: 15,
        }];
        let lineage_batch = lineage_rows_to_batch(&prov, &lineage_rows).unwrap();
        let lineage_pq = dir.path().join("lineage.parquet");
        write_batch_to_parquet(&lineage_pq, &lineage_batch).unwrap();
        let lineage_pq_read = read_parquet_batch(&lineage_pq).unwrap();
        verify_record_batches(&lineage_batch, &lineage_pq_read).unwrap();

        // 4. Event Table
        let event_rows = vec![EventExportRow {
            run_id: "test_run_42".into(),
            tick: 15,
            seq: 1,
            event_id: "evt-1".into(),
            kind: "domain:birth".into(),
            payload: "{\"agent_uid\":102}".into(),
        }];
        let event_batch = event_rows_to_batch(&prov, &event_rows).unwrap();
        let event_pq = dir.path().join("event.parquet");
        write_batch_to_parquet(&event_pq, &event_batch).unwrap();
        let event_pq_read = read_parquet_batch(&event_pq).unwrap();
        verify_record_batches(&event_batch, &event_pq_read).unwrap();

        // 5. Metric Table
        let metric_rows = vec![
            MetricExportRow {
                run_id: "test_run_42".into(),
                tick: 0,
                name: "population".into(),
                value: 100.0,
            },
            MetricExportRow {
                run_id: "test_run_42".into(),
                tick: 10,
                name: "population".into(),
                value: 105.0,
            },
        ];
        let metric_batch = metric_rows_to_batch(&prov, &metric_rows).unwrap();
        let metric_pq = dir.path().join("metric.parquet");
        write_batch_to_parquet(&metric_pq, &metric_batch).unwrap();
        let metric_pq_read = read_parquet_batch(&metric_pq).unwrap();
        verify_record_batches(&metric_batch, &metric_pq_read).unwrap();
    }

    #[test]
    fn test_csv_with_provenance_roundtrip() {
        let dir = tempdir().unwrap();
        let prov = sample_provenance();
        let csv_path = dir.path().join("test.csv");

        let headers = ["run_id", "tick", "metric", "value"];
        let rows: Vec<Vec<String>> = vec![
            vec!["test_run_42".into(), "10".into(), "pop".into(), "50".into()],
            vec!["test_run_42".into(), "20".into(), "pop".into(), "55".into()],
        ];

        let mut file = File::create(&csv_path).unwrap();
        let prov_json = serde_json::to_string(&prov).unwrap();
        writeln!(file, "# PROVENANCE: {prov_json}").unwrap();
        writeln!(file, "{}", headers.join(",")).unwrap();
        for r in &rows {
            writeln!(file, "{}", r.join(",")).unwrap();
        }

        let (read_prov_line, read_headers, read_rows) =
            read_csv_with_provenance(&csv_path).unwrap();
        assert!(read_prov_line.contains("test_run_42"));
        assert_eq!(read_headers, vec!["run_id", "tick", "metric", "value"]);
        assert_eq!(read_rows, rows);
    }

    #[test]
    fn test_frankenpandas_agents_dataframe_and_rolling() {
        let agent_rows = vec![
            AgentExportRow {
                run_id: "run_test".into(),
                tick: 5,
                agent_uid: 1,
                generation: 0,
                age: 5,
                pos_x: 10.0,
                pos_y: 20.0,
                vel_x: 1.0,
                vel_y: 0.0,
                heading: 0.0,
                health: 100.0,
                energy: 200.0,
                herbivore_tendency: 0.9,
                brain_binding: "mlp".into(),
            },
            AgentExportRow {
                run_id: "run_test".into(),
                tick: 10,
                agent_uid: 2,
                generation: 0,
                age: 10,
                pos_x: 15.0,
                pos_y: 25.0,
                vel_x: 0.0,
                vel_y: 1.0,
                heading: 1.57,
                health: 80.0,
                energy: 150.0,
                herbivore_tendency: 0.1,
                brain_binding: "neuroflow".into(),
            },
        ];

        let df = build_agents_dataframe(&agent_rows, 100).unwrap();
        assert_eq!(df.shape(), (2, 15));

        let diet_col = df.get_column("diet_class");
        let diet_vals = diet_col.values();
        assert_eq!(diet_vals[0], Scalar::Utf8("Herbivore".into()));
        assert_eq!(diet_vals[1], Scalar::Utf8("Carnivore".into()));

        // Test rolling mean on a series
        let val_scalars = vec![
            Scalar::Float64(10.0),
            Scalar::Float64(20.0),
            Scalar::Float64(30.0),
        ];
        let col = Column::from_values(val_scalars).unwrap();
        let idx = Index::from_range(0, 3, 1);
        let s = Series::new("test_series", idx, col).unwrap();
        let rolled = s.rolling(2, Some(1)).mean().unwrap();
        let rolled_vals = rolled.values();

        assert_eq!(rolled_vals[0], Scalar::Float64(10.0));
        assert_eq!(rolled_vals[1], Scalar::Float64(15.0)); // (10+20)/2
        assert_eq!(rolled_vals[2], Scalar::Float64(25.0)); // (20+30)/2
    }
}
