//! Economy conservation audit subcommand (bd-9sg6 / bd-16g.11.2).
//!
//! Provides a headless multi-seed runner verifying physical conservation laws
//! (mass/energy/food balance) across simulation ticks. Streams per-tick, per-stock
//! residual timeseries into `residual_<seed>.csv` and emits deterministic
//! `ConservationVerdict` JSON to the specified output path.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use scriptbots_core::economy::{
    ConservationGate, ConservationTolerances, ConservationVerdict, EconomyStock,
    evaluate_conservation,
};
use scriptbots_core::{Position, ResourceAmounts, ScriptBotsConfig, WorldState};
use tracing::{error, info};

/// Arguments for the `economy-audit` CLI subcommand.
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct EconomyAuditArgs {
    /// Number of seeds to evaluate, or comma/space-separated list of explicit seeds.
    #[arg(long = "seeds", default_value = "3")]
    pub seeds: String,

    /// Number of simulation ticks per seed.
    #[arg(long = "ticks", default_value = "50000")]
    pub ticks: u64,

    /// Output directory or file path for verdict and residual artifacts.
    #[arg(long = "out")]
    pub out: PathBuf,

    /// Debugging tolerance override. Sets `tolerance_overridden` on the verdict.
    /// CI jobs must never pass this flag.
    #[arg(long = "tolerance")]
    pub tolerance: Option<f64>,
}

/// Parse seed specifications: either an integer count `N` generating canonical
/// seeds `0x6A7E_0001..+N`, or a comma/whitespace-separated list of `u64` integers.
pub fn parse_seeds(spec: &str) -> Result<Vec<u64>> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        anyhow::bail!("empty seeds specification");
    }

    if let Ok(count) = trimmed.parse::<usize>() {
        if count == 0 {
            anyhow::bail!("seeds count must be at least 1");
        }
        return Ok((0..count).map(|i| 0x6A7E_0001_u64 + (i as u64)).collect());
    }

    let mut parsed = Vec::new();
    for part in trimmed.split([',', ' ', '\t', '\n']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let seed = if let Some(hex) = part.strip_prefix("0x").or_else(|| part.strip_prefix("0X")) {
            u64::from_str_radix(hex, 16)
                .with_context(|| format!("invalid hexadecimal seed: {part}"))?
        } else {
            part.parse::<u64>()
                .with_context(|| format!("invalid decimal seed: {part}"))?
        };
        parsed.push(seed);
    }

    if parsed.is_empty() {
        anyhow::bail!("no valid seeds found in specification: {spec}");
    }
    Ok(parsed)
}

fn stock_name(stock: EconomyStock) -> &'static str {
    match stock {
        EconomyStock::GridFood => "GridFood",
        EconomyStock::AgentEnergy => "AgentEnergy",
        EconomyStock::AgentHealth => "AgentHealth",
    }
}

fn stock_lane(stock: EconomyStock, amounts: &ResourceAmounts) -> f64 {
    match stock {
        EconomyStock::GridFood => amounts.food,
        EconomyStock::AgentEnergy => amounts.energy,
        EconomyStock::AgentHealth => amounts.health,
    }
}

/// Compute a deterministic Blake3 config digest for the world.
pub fn compute_world_config_digest(world: &WorldState) -> Result<String> {
    let config_val =
        serde_json::to_value(world.config()).context("failed to serialize world config to JSON")?;
    let canonical = crate::canonical_json_value_bytes(&config_val)
        .context("failed to canonically encode world config JSON")?;
    Ok(format!("blake3:{}", blake3::hash(&canonical).to_hex()))
}

/// Build a deterministic world instance for the conservation gate.
pub fn create_audit_world(seed: u64) -> Result<WorldState> {
    let config = ScriptBotsConfig {
        reproduction_energy_threshold: 0.8,
        reproduction_energy_cost: 0.8,
        reproduction_cooldown: 8,
        reproduction_attempt_interval: 16,
        reproduction_attempt_chance: 0.25,
        reproduction_child_energy: 0.4,
        closed: false,
        population_minimum: 8,
        population_spawn_interval: 5,
        max_agents: 32,
        aging_tick_interval: 1,
        aging_health_decay_rate: 0.005,
        aging_health_decay_max: 0.05,
        persistence_interval: 0,
        chart_flush_interval: 0,
        rng_seed: Some(seed),
        ..ScriptBotsConfig::default()
    };

    let mut world = WorldState::new(config).context("failed to construct audit world")?;
    for index in 0..6_u32 {
        let agent = world
            .try_spawn_agent(scriptbots_core::AgentData {
                position: Position::new(40.0 + f32::from(index as u16) * 6.0, 60.0),
                ..scriptbots_core::AgentData::default()
            })
            .context("failed to spawn initial audit agent")?;
        world
            .try_update_agent_runtime(agent, |runtime| {
                runtime.energy = 1.5;
                runtime.reproduction_counter = 10.0;
            })
            .context("failed to configure initial audit agent runtime")?;
    }
    world
        .enqueue_intervention(scriptbots_core::Intervention::Bloom {
            region: scriptbots_core::Region::All,
            amount: 0.1,
        })
        .context("failed to enqueue audit bloom intervention")?;

    world.set_resource_ledger_enabled(true);
    Ok(world)
}

/// Execute the economy audit across all seeds and stream residual csvs.
pub fn run_economy_audit(args: &EconomyAuditArgs) -> Result<bool> {
    let seeds = parse_seeds(&args.seeds)?;
    info!(
        seeds_count = seeds.len(),
        ticks_per_seed = args.ticks,
        out = %args.out.display(),
        tolerance = ?args.tolerance,
        "starting economy conservation audit"
    );

    let (artifact_dir, verdict_path) =
        if args.out.extension().and_then(|s| s.to_str()) == Some("json") {
            let parent = args.out.parent().unwrap_or_else(|| Path::new("."));
            (parent.to_path_buf(), args.out.clone())
        } else {
            (args.out.clone(), args.out.join("verdict.json"))
        };

    std::fs::create_dir_all(&artifact_dir).with_context(|| {
        format!(
            "failed to create artifact directory: {}",
            artifact_dir.display()
        )
    })?;

    let stocks = [
        EconomyStock::GridFood,
        EconomyStock::AgentEnergy,
        EconomyStock::AgentHealth,
    ];

    let mut seed_verdicts = Vec::with_capacity(seeds.len());
    let mut last_config_digest: Option<String> = None;

    for &seed in &seeds {
        let mut world = create_audit_world(seed)?;
        if last_config_digest.is_none() {
            if let Ok(digest) = compute_world_config_digest(&world) {
                last_config_digest = Some(digest);
            }
        }

        let csv_path = artifact_dir.join(format!("residual_{seed}.csv"));
        let csv_file = File::create(&csv_path)
            .with_context(|| format!("failed to create CSV file: {}", csv_path.display()))?;
        let mut csv_writer = BufWriter::new(csv_file);

        writeln!(
            csv_writer,
            "tick,stock,inflow,outflow,delta_stock,residual,tolerance,gross_flow,worst_category"
        )?;

        let mut gate = ConservationGate::new();

        for _ in 0..args.ticks {
            world.step().context("world step during audit")?;
            let report_opt = world.resource_ledger().latest.clone();
            if let Some(ref report) = report_opt {
                gate.observe(report);

                let tick = report.tick.0;
                let tolerance = report.reconciliation.tolerance;

                for &stock in &stocks {
                    let s_name = stock_name(stock);
                    let mut inflow = 0.0_f64;
                    let mut outflow = 0.0_f64;
                    let mut gross_flow = 0.0_f64;
                    let mut max_flow_mag = 0.0_f64;
                    let mut worst_cat: Option<String> = None;

                    for flow in &report.flows {
                        let delta = stock_lane(stock, &flow.delta);
                        if delta > 0.0 {
                            inflow += delta;
                        } else if delta < 0.0 {
                            outflow += -delta;
                        }
                        let abs_delta = delta.abs();
                        gross_flow += abs_delta;
                        if abs_delta > max_flow_mag {
                            max_flow_mag = abs_delta;
                            worst_cat = Some(format!("{:?}", flow.kind));
                        }
                    }

                    let delta_stock = stock_lane(stock, &report.reconciliation.observed_delta);
                    let residual = stock_lane(stock, &report.reconciliation.unexplained_delta);
                    let worst_category_str = worst_cat.as_deref().unwrap_or("none");

                    writeln!(
                        csv_writer,
                        "{tick},{s_name},{inflow:.6e},{outflow:.6e},{delta_stock:.6e},{residual:.6e},{tolerance:.6e},{gross_flow:.6e},{worst_category_str}"
                    )?;
                }
            }
        }

        csv_writer.flush()?;
        seed_verdicts.push(gate.finish(seed));
    }

    let mut verdict: ConservationVerdict = evaluate_conservation(&seed_verdicts);

    if let Some(digest) = last_config_digest {
        verdict = verdict.with_config_digest(digest);
    }

    if let Some(tol) = args.tolerance {
        verdict.tolerance_overridden = true;
        verdict.tolerances = ConservationTolerances {
            per_tick_relative: tol,
            cumulative_relative: tol,
        };
        // Re-evaluate pass status based on overridden tolerances
        let mut overridden_failures = Vec::new();
        for seed in &verdict.seeds {
            for &stock in &stocks {
                let stock_idx = match stock {
                    EconomyStock::GridFood => 0,
                    EconomyStock::AgentEnergy => 1,
                    EconomyStock::AgentHealth => 2,
                };
                let gross = seed.gross_flow[stock_idx];
                let bound = tol * gross.max(1.0);
                let cumulative = seed.cumulative_residual[stock_idx];
                if cumulative.abs() > bound {
                    overridden_failures.push(format!(
                        "seed {}: cumulative residual on {stock:?} is {cumulative:.6e}, beyond overridden bound {bound:.6e}",
                        seed.seed
                    ));
                }
            }
        }
        verdict.pass = overridden_failures.is_empty();
        verdict.failures = overridden_failures;
    }

    let verdict_json = serde_json::to_string_pretty(&verdict)
        .context("failed to serialize ConservationVerdict to JSON")?;
    std::fs::write(&verdict_path, verdict_json.as_bytes())
        .with_context(|| format!("failed to write verdict JSON: {}", verdict_path.display()))?;

    let verdict_str = verdict_path.display().to_string();
    verdict.log_summary(Some(&verdict_str));

    let summary = verdict.summary_line(Some(&verdict_str));
    println!("{summary}");

    if verdict.pass {
        info!(artifact = %verdict_str, "conservation audit passed");
    } else {
        error!(
            artifact = %verdict_str,
            failures = ?verdict.failures,
            "conservation audit failed"
        );
    }

    Ok(verdict.pass)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_seeds_formats() {
        assert_eq!(
            parse_seeds("3").unwrap(),
            vec![0x6A7E_0001, 0x6A7E_0002, 0x6A7E_0003]
        );
        assert_eq!(parse_seeds("1").unwrap(), vec![0x6A7E_0001]);
        assert_eq!(parse_seeds("100, 200, 300").unwrap(), vec![100, 200, 300]);
        assert_eq!(parse_seeds("0x10, 0x20").unwrap(), vec![16, 32]);
        assert!(parse_seeds("0").is_err());
        assert!(parse_seeds("").is_err());
        assert!(parse_seeds("abc").is_err());
    }

    #[test]
    fn small_run_produces_artifacts_and_passes() {
        let temp_dir =
            std::env::temp_dir().join(format!("economy_audit_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);

        let args = EconomyAuditArgs {
            seeds: "1".to_string(),
            ticks: 16,
            out: temp_dir.clone(),
            tolerance: None,
        };

        let pass = run_economy_audit(&args).expect("audit runs");
        assert!(pass, "short unmutated audit must pass");

        assert!(temp_dir.join("verdict.json").exists());
        assert!(temp_dir.join("residual_1786642433.csv").exists());

        // Verify CSV has header and rows
        let content = std::fs::read_to_string(temp_dir.join("residual_1786642433.csv")).unwrap();
        assert!(content.starts_with(
            "tick,stock,inflow,outflow,delta_stock,residual,tolerance,gross_flow,worst_category"
        ));
        let lines: Vec<&str> = content.lines().collect();
        // 16 ticks * 3 stocks + 1 header = 49 lines
        assert_eq!(lines.len(), 49);

        // Verify verdict.json is valid JSON
        let verdict_content = std::fs::read_to_string(temp_dir.join("verdict.json")).unwrap();
        let parsed: ConservationVerdict = serde_json::from_str(&verdict_content).unwrap();
        assert!(parsed.pass);
        assert_eq!(parsed.seeds.len(), 1);
        assert!(!parsed.tolerance_overridden);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
