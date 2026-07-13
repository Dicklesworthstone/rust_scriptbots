//! `sb-analyze` — offline report CLI over finished `ScriptBots` run databases.
//!
//! bd-2z0.11.5 (program bd-2js6). Read-only by construction: this binary can
//! only open a [`scriptbots_analytics::ReaderCtx`], which has no write path.
//!
//! Examples:
//!   sb-analyze runs/scriptbots-123.sqlite list
//!   sb-analyze runs/scriptbots-123.sqlite run run-summary --md summary.md
//!   sb-analyze runs/scriptbots-123.sqlite run narrative-timeline --params limit=50 --json out.json -v

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use scriptbots_analytics::{AnalyticsError, ReaderCtx, Registry, ReportParams};

#[derive(Parser)]
#[command(
    name = "sb-analyze",
    about = "Offline reports over finished ScriptBots run databases (read-only)"
)]
struct Cli {
    /// Path to the finished run database (never opened writable).
    db: PathBuf,

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
            let cx = ReaderCtx::open(&db)?;
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
    }
}
