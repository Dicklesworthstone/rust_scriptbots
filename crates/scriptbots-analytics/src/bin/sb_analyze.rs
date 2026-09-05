//! `sb-analyze` — offline report CLI over finished `ScriptBots` run databases.
//!
//! bd-2z0.11.5 (program bd-2js6). Read-only by construction: this binary can
//! only open a [`scriptbots_analytics::ReaderCtx`] or a read-only [`StorageReader`].
//!
//! Examples:
//!   sb-analyze runs/scriptbots-123.sqlite list
//!   sb-analyze runs/scriptbots-123.sqlite run run-summary --md summary.md
//!   sb-analyze runs/scriptbots-123.sqlite run narrative-timeline --params limit=50 --json out.json -v
//!   sb-analyze runs/scriptbots-123.sqlite export --format parquet --out-dir ./exports --verify
//!   sb-analyze runs/scriptbots-123.sqlite summarize --epoch-size 100 --rolling-window 10 --md summary.md

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use scriptbots_analytics::dataframe::{
    AnalyticsExportFormat, export_database_table, summarize_run,
};
use scriptbots_analytics::{AnalyticsError, ReaderCtx, Registry, ReportParams};
use scriptbots_storage::{ExportTable, StorageReader};

#[derive(Parser)]
#[command(
    name = "sb-analyze",
    about = "Offline reports over finished ScriptBots run databases (read-only)"
)]
struct Cli {
    /// Path to the finished run database (never opened writable).
    db: PathBuf,

    /// Select the canonical run ID in a multi-run database.
    #[arg(long, global = true)]
    run_id: Option<scriptbots_runtime::RunId>,

    /// Increase log verbosity (-v info, -vv debug); `RUST_LOG` overrides.
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List registered reports.
    List,
    /// Run a report by name.
    Run {
        /// Registered report name (see `list`).
        report: String,
        /// Report parameters as key=value (repeatable).
        #[arg(long = "params", value_name = "K=V")]
        params: Vec<String>,
        /// Write the machine-readable JSON payload to this path.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Write the human-readable markdown to this path (stdout regardless).
        #[arg(long)]
        md: Option<PathBuf>,
    },
    /// Export run tables to Parquet, Arrow, or CSV (bd-2z0.11.8).
    Export {
        /// Destination directory for exported files (default: current dir).
        #[arg(long, default_value = ".")]
        out_dir: PathBuf,
        /// Export format: parquet, arrow, or csv.
        #[arg(long, value_enum, default_value_t = AnalyticsExportFormat::Parquet)]
        format: AnalyticsExportFormat,
        /// Specific table(s) to export: run, agent, lineage, event, metric (default: all).
        #[arg(long, value_delimiter = ',')]
        tables: Vec<String>,
        /// Verify written files by re-reading and asserting exact equality.
        #[arg(long)]
        verify: bool,
    },
    /// Generate per-epoch `FrankenPandas` summary and rolling metrics report (bd-2z0.11.8).
    Summarize {
        /// Epoch tick interval for demographic grouping (default: 100).
        #[arg(long, default_value_t = 100)]
        epoch_size: u64,
        /// Rolling window tick duration for metric smoothing (default: 10).
        #[arg(long, default_value_t = 10)]
        rolling_window: usize,
        /// Optional directory to write summary.json and summary.md.
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Machine-readable JSON output path.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Human-readable Markdown output path.
        #[arg(long)]
        md: Option<PathBuf>,
    },
    /// Export graph representations to `GraphML` or Edge-List (bd-2z0.11.7).
    ExportGraph {
        /// Which graph to export: lineage, dynasty, or interaction.
        #[arg(long, value_enum, default_value_t = ExportGraphKind::Lineage)]
        graph: ExportGraphKind,
        /// Export format: graphml or edgelist.
        #[arg(long, value_enum, default_value_t = ExportGraphFormat::Graphml)]
        format: ExportGraphFormat,
        /// Interaction selection/budgets as key=value (same parameters as the report).
        #[arg(long = "params", value_name = "K=V")]
        params: Vec<String>,
        /// Optional destination file path (stdout if omitted).
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

/// Target graph representation for export.
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum ExportGraphKind {
    /// Directed ancestry/phylogeny graph (parent -> child).
    Lineage,
    /// Undirected mating/ancestry graph for dynasty community clustering.
    Dynasty,
    /// Directed pairwise interaction graph (actor -> target).
    Interaction,
}

/// Serialization format for exported graphs.
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum ExportGraphFormat {
    /// Standard XML-based `GraphML` format.
    Graphml,
    /// Space-delimited edge list format.
    Edgelist,
}

fn init_tracing(verbose: u8) {
    let default = match verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err @ AnalyticsError::UnknownReport(_)) => {
            eprintln!("error: {err}");
            ExitCode::from(2)
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: &Cli) -> Result<(), AnalyticsError> {
    let registry = Registry::builtin();
    let db = cli.db.display().to_string();
    let open_reader = || {
        cli.run_id.map_or_else(
            || StorageReader::open_finished(&db),
            |run_id| StorageReader::open_finished_for_run(&db, run_id),
        )
    };
    let open_context = || {
        cli.run_id.map_or_else(
            || ReaderCtx::open(&db),
            |run_id| ReaderCtx::open_for_run(&db, run_id),
        )
    };

    match &cli.command {
        Command::List => {
            println!("available reports:");
            for (name, desc) in registry.list() {
                println!("  {name:<22} {desc}");
            }
            Ok(())
        }
        Command::Run {
            report,
            params,
            json,
            md,
        } => {
            let cx = open_context()?;
            let params = ReportParams::from_pairs(params.iter().cloned())?;
            let output = registry.run(report, &cx, &params)?;

            if let Some(path) = json {
                std::fs::write(path, serde_json::to_vec_pretty(&output)?)?;
                tracing::info!(path = %path.display(), "machine payload written");
            }
            if let Some(path) = md {
                std::fs::write(path, &output.human_md)?;
                tracing::info!(path = %path.display(), "markdown written");
            }
            println!("{}", output.human_md);
            Ok(())
        }
        Command::Export {
            out_dir,
            format,
            tables,
            verify,
        } => {
            let reader = open_reader()?;
            export_tables(&reader, *format, tables, out_dir, *verify)
        }
        Command::Summarize {
            epoch_size,
            rolling_window,
            out_dir,
            json,
            md,
        } => {
            let reader = open_reader()?;
            let summary = summarize_run(&reader, *epoch_size, *rolling_window)?;

            if let Some(dir) = out_dir {
                std::fs::create_dir_all(dir)?;
                let json_path = dir.join("summary.json");
                let md_path = dir.join("summary.md");
                std::fs::write(&json_path, serde_json::to_vec_pretty(&summary)?)?;
                std::fs::write(&md_path, &summary.markdown_table)?;
                tracing::info!(
                    json = %json_path.display(),
                    md = %md_path.display(),
                    "summary written to directory"
                );
            }

            if let Some(path) = json {
                std::fs::write(path, serde_json::to_vec_pretty(&summary)?)?;
                tracing::info!(path = %path.display(), "machine payload written");
            }

            if let Some(path) = md {
                std::fs::write(path, &summary.markdown_table)?;
                tracing::info!(path = %path.display(), "markdown written");
            }

            println!("{}", summary.markdown_table);
            Ok(())
        }
        Command::ExportGraph {
            graph,
            format,
            params,
            out,
        } => {
            let cx = open_context()?;
            let text = graph_export_text(&cx, *graph, *format, params)?;

            if let Some(path) = out {
                std::fs::write(path, &text)?;
                tracing::info!(path = %path.display(), "graph exported");
            } else {
                print!("{text}");
            }
            Ok(())
        }
    }
}

fn export_tables(
    reader: &StorageReader,
    format: AnalyticsExportFormat,
    tables: &[String],
    out_dir: &Path,
    verify: bool,
) -> Result<(), AnalyticsError> {
    let target_tables = if tables.is_empty() {
        ExportTable::ALL.to_vec()
    } else {
        let mut parsed = Vec::new();
        for t in tables {
            match t.to_lowercase().as_str() {
                "run" => parsed.push(ExportTable::Run),
                "agent" | "agents" => parsed.push(ExportTable::Agent),
                "lineage" | "lineage_edges" => parsed.push(ExportTable::Lineage),
                "event" | "events" | "replay_events" => parsed.push(ExportTable::Event),
                "metric" | "metrics" => parsed.push(ExportTable::Metric),
                other => {
                    return Err(AnalyticsError::BadParam {
                        name: "tables".into(),
                        reason: format!("Unknown table '{other}'"),
                    });
                }
            }
        }
        parsed
    };

    std::fs::create_dir_all(out_dir)?;
    println!(
        "exporting {} table(s) to {:?} format in {}",
        target_tables.len(),
        format,
        out_dir.display()
    );

    for table in target_tables {
        let path = export_database_table(reader, table, format, out_dir, verify)?;
        println!("  ✓ exported {} -> {}", table.as_str(), path.display());
    }

    if verify {
        println!("all exported files verified successfully.");
    }
    Ok(())
}

fn graph_export_text(
    cx: &ReaderCtx,
    graph: ExportGraphKind,
    format: ExportGraphFormat,
    params: &[String],
) -> Result<String, AnalyticsError> {
    if graph != ExportGraphKind::Interaction && !params.is_empty() {
        return Err(AnalyticsError::BadParam {
            name: "params".to_owned(),
            reason: "graph selection parameters apply to interaction exports".to_owned(),
        });
    }
    match (graph, format) {
        (ExportGraphKind::Lineage, ExportGraphFormat::Graphml) => {
            let births = cx.reader.load_ancestry_births()?;
            let (digraph, _) = scriptbots_analytics::build_lineage_digraph(&births);
            scriptbots_analytics::export_digraph_graphml(&digraph)
        }
        (ExportGraphKind::Lineage, ExportGraphFormat::Edgelist) => {
            let births = cx.reader.load_ancestry_births()?;
            let (digraph, _) = scriptbots_analytics::build_lineage_digraph(&births);
            scriptbots_analytics::export_digraph_edgelist(&digraph)
        }
        (ExportGraphKind::Dynasty, ExportGraphFormat::Graphml) => {
            let births = cx.reader.load_ancestry_births()?;
            let (digraph, _) = scriptbots_analytics::build_lineage_digraph(&births);
            let undirected = digraph.to_undirected();
            scriptbots_analytics::export_graph_graphml(&undirected)
        }
        (ExportGraphKind::Dynasty, ExportGraphFormat::Edgelist) => {
            let births = cx.reader.load_ancestry_births()?;
            let (digraph, _) = scriptbots_analytics::build_lineage_digraph(&births);
            let undirected = digraph.to_undirected();
            scriptbots_analytics::export_graph_edgelist(&undirected)
        }
        (ExportGraphKind::Interaction, format) => {
            let params = ReportParams::from_pairs(params.iter().cloned())?;
            let (interactions, evidence) =
                scriptbots_analytics::load_interaction_graph_input(cx, &params)?;
            let (digraph, _, _) = scriptbots_analytics::build_interaction_digraph(&interactions)?;
            let format = match format {
                ExportGraphFormat::Graphml => scriptbots_analytics::InteractionGraphFormat::GraphMl,
                ExportGraphFormat::Edgelist => {
                    scriptbots_analytics::InteractionGraphFormat::EdgeList
                }
            };
            scriptbots_analytics::export_interaction_graph(&digraph, &evidence, format)
        }
    }
}
