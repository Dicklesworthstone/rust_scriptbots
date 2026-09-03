//! Archipelago report generation and conservation auditing CLI runner (bd-16g.5.5.5).

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use scriptbots_storage::StorageReader;

/// CLI arguments for the `report-archipelago` subcommand or mode.
#[derive(Args, Debug, Clone, PartialEq, Eq)]
pub struct ReportArchipelagoArgs {
    /// Path to the SQLite run database to inspect.
    #[arg(value_name = "DB")]
    pub db: PathBuf,

    /// Optional path to write report JSON.
    #[arg(long, value_name = "JSON_PATH")]
    pub json: Option<PathBuf>,

    /// Require per-island population conservation to close (exit nonzero on violation).
    #[arg(long, default_value_t = false)]
    pub verify_conservation: bool,
}

/// Generate a human-readable text report and optional JSON for an archipelago run database.
pub fn run_archipelago_report(args: &ReportArchipelagoArgs) -> Result<bool> {
    let path_str = args.db.to_string_lossy().to_string();
    let reader = StorageReader::open(&path_str)
        .with_context(|| format!("failed to open run database at {}", args.db.display()))?;
    let report = reader
        .archipelago_report()
        .context("failed to reconstruct archipelago report from database")?;

    println!("=== ARCHIPELAGO OFFLINE RECONSTRUCTION REPORT ===");
    println!("Database:   {}", args.db.display());
    println!("Run ID:     {}", report.run_id);
    println!("Islands:    {}", report.islands.len());
    println!("Migrations: {}", report.migrations.len());
    println!();

    println!("--- CONFIGURED ISLANDS ---");
    for island in &report.islands {
        let hist_len = report
            .histories
            .get(&island.island_id)
            .map(|h| h.points.len())
            .unwrap_or(0);
        println!(
            "  Island #{:<2} [label: {:<12}] config_hash: 0x{:016x} ({} history points)",
            island.island_id, island.label, island.config_hash, hist_len
        );
    }
    println!();

    println!("--- MIGRATION MULTIGRAPH EDGES ---");
    if report.migration_graph.is_empty() {
        println!("  (No migrations recorded)");
    } else {
        for edge in &report.migration_graph {
            println!(
                "  Island {} -> Island {}: {} emigrant(s)",
                edge.from, edge.to, edge.count
            );
        }
    }
    println!();

    println!("--- POPULATION CONSERVATION AUDIT ---");
    println!("  Passed:          {}", report.conservation_audit.passed);
    println!(
        "  Islands Checked: {}",
        report.conservation_audit.total_islands_checked
    );
    println!(
        "  Ticks Checked:   {}",
        report.conservation_audit.total_ticks_checked
    );
    println!(
        "  Breaches:        {}",
        report.conservation_audit.breaches.len()
    );

    if !report.conservation_audit.breaches.is_empty() {
        eprintln!();
        eprintln!("CONSERVATION BREACHES DETECTED:");
        for breach in &report.conservation_audit.breaches {
            eprintln!(
                "  Island {} at tick {}: expected {}, recorded {} (births: {}, deaths: {}, immigrations: {}, emigrations: {})",
                breach.island_id,
                breach.tick,
                breach.expected_population,
                breach.recorded_population,
                breach.births,
                breach.deaths,
                breach.immigrations,
                breach.emigrations
            );
        }
    }

    if let Some(json_path) = &args.json {
        let json_str = serde_json::to_string_pretty(&report)?;
        if let Some(parent) = json_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(json_path, json_str)?;
        println!();
        println!("Wrote archipelago report JSON to {}", json_path.display());
    }

    if args.verify_conservation && !report.conservation_audit.passed {
        return Ok(false);
    }

    Ok(true)
}
