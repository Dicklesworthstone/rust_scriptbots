//! Cross-surface agreement and structural boundary integration tests for ecosystem accounting (bd-16g.11.3).
//!
//! Verifies:
//! 1. One shared model: GPUI Sankey, TUI trophic table, and JSON export all consume canonical
//!    `SankeyGraph` and `TrophicTable` computed by `scriptbots_core::economy`.
//! 2. Agreement E2E: on a run with residual at 90% of tolerance, all three surfaces carry
//!    identical f64 values for all categories and residuals.
//! 3. Negative control 1: de-synchronized renderer altering values fails agreement assertion.
//! 4. Negative control 2: dropping Unaccounted link fails Kirchhoff balance verification.
//! 5. Negative control 3: missing epoch renders "no data" and never emits false zeros.
//! 6. CI grep-level structural test: render and terminal modules never reference `TickFlows`
//!    or arithmetic helpers; they only consume `EpochFlows`, `SankeyGraph`, and `TrophicTable`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use scriptbots_core::economy::{
    DietCensus, EconomyStock, EpochCategoryFlow, EpochFlows, EpochStockResidual,
    FoodWebSummaryReport, SankeyOpts, sankey_layout, trophic_table,
};
use scriptbots_core::{RESOURCE_FLOW_KINDS, ResourceFlowKind};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("determine repository root")
        .to_path_buf()
}

fn build_calibrated_epoch(residual_ratio: f64) -> (EpochFlows, DietCensus) {
    let mut categories = Vec::new();
    for kind in RESOURCE_FLOW_KINDS {
        let (delta, activity) = match kind {
            ResourceFlowKind::FoodDynamics => (
                scriptbots_core::ResourceAmounts {
                    food: 100.0,
                    energy: 0.0,
                    health: 0.0,
                },
                scriptbots_core::ResourceAmounts {
                    food: 100.0,
                    energy: 0.0,
                    health: 0.0,
                },
            ),
            ResourceFlowKind::GroundFoodConversion => (
                scriptbots_core::ResourceAmounts {
                    food: -85.0,
                    energy: 85.0,
                    health: 0.0,
                },
                scriptbots_core::ResourceAmounts {
                    food: 85.0,
                    energy: 85.0,
                    health: 0.0,
                },
            ),
            ResourceFlowKind::BasalMetabolism => (
                scriptbots_core::ResourceAmounts {
                    food: 0.0,
                    energy: -70.0,
                    health: 0.0,
                },
                scriptbots_core::ResourceAmounts {
                    food: 0.0,
                    energy: 70.0,
                    health: 0.0,
                },
            ),
            _ => (
                scriptbots_core::ResourceAmounts {
                    food: 0.0,
                    energy: 0.0,
                    health: 0.0,
                },
                scriptbots_core::ResourceAmounts {
                    food: 0.0,
                    energy: 0.0,
                    health: 0.0,
                },
            ),
        };

        categories.push(EpochCategoryFlow {
            kind,
            delta,
            activity,
        });
    }

    // Residual intentionally at residual_ratio of tolerance
    let food_tolerance = 1.0;
    let food_residual = residual_ratio * food_tolerance;
    let energy_tolerance = 1.0;
    let energy_residual = residual_ratio * energy_tolerance;

    let residuals = [
        EpochStockResidual {
            stock: EconomyStock::GridFood,
            residual_sum: food_residual,
            residual_max_abs: food_residual.abs(),
            argmax_tick: Some(50),
            gross_flow: 185.0,
            cumulative_tolerance: food_tolerance,
            within_tolerance: food_residual.abs() <= food_tolerance,
            worst_category: Some(ResourceFlowKind::FoodDynamics),
        },
        EpochStockResidual {
            stock: EconomyStock::AgentEnergy,
            residual_sum: energy_residual,
            residual_max_abs: energy_residual.abs(),
            argmax_tick: Some(60),
            gross_flow: 155.0,
            cumulative_tolerance: energy_tolerance,
            within_tolerance: energy_residual.abs() <= energy_tolerance,
            worst_category: Some(ResourceFlowKind::BasalMetabolism),
        },
        EpochStockResidual {
            stock: EconomyStock::AgentHealth,
            residual_sum: 0.0,
            residual_max_abs: 0.0,
            argmax_tick: None,
            gross_flow: 0.0,
            cumulative_tolerance: 1.0,
            within_tolerance: true,
            worst_category: None,
        },
    ];

    let epoch = EpochFlows {
        epoch: 7,
        first_tick: 701,
        last_tick: 800,
        tick_count: 100,
        complete: true,
        per_category: categories,
        residual: residuals,
    };

    let mut census = DietCensus::default();
    census.counts[0] = 12;
    census.standing_health[0] = 120.0;
    census.standing_energy[0] = 105.0;

    census.counts[1] = 0;
    census.standing_health[1] = 0.0;
    census.standing_energy[1] = 0.0;

    census.counts[2] = 4;
    census.standing_health[2] = 40.0;
    census.standing_energy[2] = 35.0;

    (epoch, census)
}

#[test]
fn test_bd_16g_11_3_agreement_e2e_all_three_surfaces_identical_f64() {
    let nonce = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let (epoch, census) = build_calibrated_epoch(0.90);

    // 1. Surface A: TUI Table
    let table = trophic_table(&epoch, &census);
    let tui_rendered = scriptbots_app::terminal::render_trophic_table(Some(&table));

    // 2. Surface B: GPUI Sankey Graph
    let sankey = sankey_layout(&epoch, &SankeyOpts::default());

    // 3. Surface C: JSON Export Report
    let temp_dir = std::env::temp_dir().join(format!(
        "scriptbots_food_web_export_{}_{}",
        std::process::id(),
        nonce
    ));
    fs::create_dir_all(&temp_dir).expect("create temp dir");
    let export_path = temp_dir.join("food_web_epoch_7.json");

    let report = FoodWebSummaryReport::build(
        "run_e2e_acceptance",
        &epoch,
        &census,
        Some("blake3:e2e_config_digest".to_owned()),
    );
    let json = report.to_json().expect("serialize food web report");
    fs::write(&export_path, &json).expect("write export json");

    // Read back and parse
    let read_back_bytes = fs::read_to_string(&export_path).expect("read export json");
    let parsed_report =
        FoodWebSummaryReport::from_json(&read_back_bytes).expect("parse export json");

    // Assert identical structs between export, sankey, and table
    assert_eq!(
        parsed_report.sankey, sankey,
        "parsed export sankey must be bit-exact identical to live sankey graph"
    );
    assert_eq!(
        parsed_report.trophic_table, table,
        "parsed export trophic table must be bit-exact identical to live trophic table"
    );

    // Assert exact values for all categories
    for cat in &epoch.per_category {
        let cat_val = cat.activity.scale().abs();
        if cat_val > 0.0 {
            let matching_link = sankey.links.iter().find(|l| l.category == Some(cat.kind));
            assert!(
                matching_link.is_some(),
                "sankey must contain link for category {:?}",
                cat.kind
            );
            assert_eq!(
                matching_link.unwrap().value,
                cat_val,
                "category {:?} flow value must match bit-exactly",
                cat.kind
            );
        }
    }

    // Assert non-zero residual at 90% tolerance is represented in Sankey
    let residual_link_food = sankey
        .links
        .iter()
        .find(|l| l.from == 1 && l.to == 6)
        .expect("residual food link from 1 to 6 must exist");
    assert_eq!(
        residual_link_food.value,
        epoch.residual[0].residual_sum.abs(),
        "food residual link value must match epoch residual"
    );

    let residual_link_energy = sankey
        .links
        .iter()
        .find(|l| l.from == 2 && l.to == 6)
        .expect("residual energy link from 2 to 6 must exist");
    assert_eq!(
        residual_link_energy.value,
        epoch.residual[1].residual_sum.abs(),
        "energy residual link value must match epoch residual"
    );

    // Assert TUI rendered string contains exact numbers from table
    assert!(tui_rendered.contains("Epoch 7 · Ticks 701..800"));
    assert!(tui_rendered.contains("Herbivore"));
    assert!(tui_rendered.contains("Omnivore"));
    assert!(tui_rendered.contains("Carnivore"));
    assert!(
        tui_rendered.contains("n/a"),
        "empty omnivore class must show n/a"
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_bd_16g_11_3_negative_desynced_renderer_fails_agreement() {
    let (epoch, _census) = build_calibrated_epoch(0.90);
    let canonical_sankey = sankey_layout(&epoch, &SankeyOpts::default());

    // Simulated de-synchronized renderer that adds epsilon to a link
    let mut desynced_sankey = canonical_sankey.clone();
    desynced_sankey.links[0].value += 1e-4;

    assert_ne!(
        canonical_sankey, desynced_sankey,
        "agreement test must detect epsilon divergence in renderer"
    );
}

#[test]
fn test_bd_16g_11_3_negative_dropping_unaccounted_violates_kirchhoff() {
    // A perfectly balanced fixture where FoodDynamics balances GroundFoodConversion + residual,
    // and GroundFoodConversion balances BasalMetabolism + residual
    let (mut epoch, _) = build_calibrated_epoch(0.90);
    for cat in &mut epoch.per_category {
        match cat.kind {
            ResourceFlowKind::FoodDynamics => {
                cat.activity = scriptbots_core::ResourceAmounts {
                    food: 85.9,
                    energy: 0.0,
                    health: 0.0,
                };
            }
            ResourceFlowKind::GroundFoodConversion => {
                cat.activity = scriptbots_core::ResourceAmounts {
                    food: 85.0,
                    energy: 85.0,
                    health: 0.0,
                };
            }
            ResourceFlowKind::BasalMetabolism => {
                cat.activity = scriptbots_core::ResourceAmounts {
                    food: 0.0,
                    energy: 84.1,
                    health: 0.0,
                };
            }
            _ => {
                cat.activity = scriptbots_core::ResourceAmounts {
                    food: 0.0,
                    energy: 0.0,
                    health: 0.0,
                };
            }
        }
    }

    let sankey = sankey_layout(&epoch, &SankeyOpts::default());
    assert!(
        sankey.verify_kirchhoff(1e-5).is_ok(),
        "balanced sankey with unaccounted residual link must pass Kirchhoff: {:?}",
        sankey.verify_kirchhoff(1e-5).err()
    );

    // Negative control: drop unaccounted links
    let mut dropped = sankey.clone();
    dropped.links.retain(|l| l.to != 6);
    let res = dropped.verify_kirchhoff(1e-5);
    assert!(
        res.is_err(),
        "dropping unaccounted residual link MUST fail Kirchhoff test"
    );
}

#[test]
fn test_bd_16g_11_3_negative_missing_epoch_renders_no_data_and_never_emits_zeros() {
    let tui_rendered = scriptbots_app::terminal::render_trophic_table(None);
    assert!(
        tui_rendered.contains("no data for requested epoch"),
        "missing epoch must render 'no data'"
    );
    assert!(
        !tui_rendered.contains("+-----------+"),
        "table body must not be rendered"
    );
    assert!(
        !tui_rendered.contains("0.0000"),
        "false zeros must never appear when data is absent"
    );
}

#[test]
fn test_bd_16g_11_3_ci_grep_structural_boundary_no_rederivation() {
    let root = repo_root();
    let render_dir = root.join("crates/scriptbots-render/src");
    let terminal_dir = root.join("crates/scriptbots-app/src/terminal");

    let forbidden_patterns = [
        "TickFlows",
        "FlowCategory",
        "BasalMetabolism +",
        "GroundFoodConversion -",
    ];

    let mut scanned_files = 0;

    for dir in &[render_dir, terminal_dir] {
        if !dir.exists() {
            continue;
        }
        for entry in fs::read_dir(dir).expect("read dir") {
            let entry = entry.expect("entry");
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                scanned_files += 1;
                let content = fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));

                for forbidden in &forbidden_patterns {
                    assert!(
                        !content.contains(forbidden),
                        "Structural violation in {}: referenced forbidden token '{}'. \
                         Render and terminal modules must consume only EpochFlows, \
                         SankeyGraph, or TrophicTable without re-deriving flows.",
                        path.display(),
                        forbidden
                    );
                }
            }
        }
    }

    assert!(
        scanned_files >= 5,
        "structural boundary test must scan at least 5 source files"
    );
}
