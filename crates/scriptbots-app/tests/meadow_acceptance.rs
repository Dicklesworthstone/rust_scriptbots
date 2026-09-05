//! Meadow cohort checks using the terminal TestBackend and CPU PNG renderer.
//! These helper paths do not prove production GUI startup or real PTY behavior;
//! that acceptance remains with bd-2z0.10.5.
//!
//! Completion-proof debt from bd-2z0.10.2:
//! 1. The checked-in meadow scenario (`scenarios/meadow.scenario.toml`) executes its full declared
//!    cohort schedule (`seeds = [42, 137, 20260717]`, 300 ticks).
//! 2. Satisfies all declared envelope criteria on every seed (population in [10, 250], births >= 5, deaths >= 1).
//! 3. Scientific parity between these two named helper paths (bit-exact WorldDigestV1 match).
//! 4. Balanced ledger: Resource ledger enabled, reconciles at every tick, and evaluates to zero breaches
//!    under `ConservationGate` and `evaluate_conservation`.
//! 5. Negative controls: Divergent seeds break parity; injected ledger breach fails conservation gate.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use scriptbots_app::{
    BrainPreset, ScenarioDocumentV1, ScenarioEnvelopeV1, ScenarioIdentityV0, install_brains,
    precedence::{ConfigLayerKind, ConfigLayerStatement, resolve_config_layers},
    renderer::RendererContext,
    seed_founding_population,
    terminal::TerminalRenderer,
};
use scriptbots_core::economy::{
    ConservationGate, ConservationVerdict, SeedVerdict, evaluate_conservation,
};
use scriptbots_core::{
    ResourceAmounts, ResourceFlow, ResourceFlowKind, ResourceLedgerTick, ScriptBotsConfig, Tick,
    WorldDigestV1, WorldState,
};
use scriptbots_render::render_png_offscreen;
use scriptbots_storage::AnalyticsSnapshotProvider;

fn meadow_scenario_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("scenarios/meadow.scenario.toml")
}

fn load_meadow_scenario() -> ScenarioDocumentV1 {
    let path = meadow_scenario_path();
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|err| panic!("failed to read scenario file {}: {err}", path.display()));
    let document = ScenarioDocumentV1::parse_toml(&bytes)
        .unwrap_or_else(|err| panic!("failed to parse meadow scenario: {err}"));
    assert_eq!(document.id, "meadow", "scenario id must be meadow");
    assert_eq!(
        document.seeds,
        vec![42, 137, 20260717],
        "declared cohort seeds must match frozen schedule"
    );
    document
}

fn build_meadow_world(document: &ScenarioDocumentV1, seed: u64) -> WorldState {
    let defaults = ScriptBotsConfig {
        persistence_interval: 0,
        history_capacity: 600,
        rng_seed: Some(seed),
        ..ScriptBotsConfig::default()
    };
    let resolved = resolve_config_layers(
        &serde_json::to_value(defaults).expect("serialize defaults"),
        &[
            ConfigLayerStatement {
                kind: ConfigLayerKind::File,
                label: document.id.clone(),
                fields: document.config.clone(),
            },
            ConfigLayerStatement {
                kind: ConfigLayerKind::Cli,
                label: "cohort seed".into(),
                fields: serde_json::json!({"rng_seed": seed}),
            },
        ],
    );
    let config: ScriptBotsConfig =
        serde_json::from_value(resolved.merged).expect("decode complete scenario config");
    let mut world = WorldState::new(config).expect("create meadow world");
    world.set_resource_ledger_enabled(true);

    let brain_keys = install_brains(&mut world, BrainPreset::Mixed)
        .expect("install brains")
        .population;
    seed_founding_population(&mut world, &brain_keys).expect("seed founders");

    world
}

fn assert_envelope(id: &str, seed: u64, envelope: &ScenarioEnvelopeV1, world: &WorldState) {
    assert_eq!(
        world.tick().0,
        envelope.ticks,
        "{id}/seed {seed}: simulation tick count mismatch"
    );
    let agent_count = world.agent_count();
    if let Some(min) = envelope.population_min {
        assert!(
            agent_count >= min as usize,
            "{id}/seed {seed}: population {agent_count} fell below floor {min}"
        );
    }
    if let Some(max) = envelope.population_max {
        assert!(
            agent_count <= max as usize,
            "{id}/seed {seed}: population {agent_count} exceeded ceiling {max}"
        );
    }
    let total_births = world.history().map(|h| h.births as usize).sum::<usize>();
    let total_deaths = world.history().map(|h| h.deaths as usize).sum::<usize>();
    if let Some(min) = envelope.births_min {
        assert!(
            total_births >= min as usize,
            "{id}/seed {seed}: total births {total_births} fell below floor {min}"
        );
    }
    if let Some(min) = envelope.deaths_min {
        assert!(
            total_deaths >= min as usize,
            "{id}/seed {seed}: total deaths {total_deaths} fell below floor {min}"
        );
    }
}

struct TuiRunResult {
    world: WorldState,
    digest: WorldDigestV1,
    gate: ConservationGate,
    rendered_frames: usize,
}

fn run_tui_path(document: &ScenarioDocumentV1, seed: u64, ticks: u64) -> TuiRunResult {
    let world = build_meadow_world(document, seed);
    let gate = Arc::new(Mutex::new(ConservationGate::new()));
    let shared_world = Arc::new(Mutex::new(world));
    let renderer = TerminalRenderer::default();
    let step_world = Arc::clone(&shared_world);
    let step_gate = Arc::clone(&gate);
    let simulation_step = Arc::new(move || {
        let mut world = step_world.lock().expect("world lock");
        let summary = world.step()?;
        let report = world
            .resource_ledger()
            .latest
            .as_ref()
            .expect("enabled ledger produced tick report");
        assert!(
            report.reconciliation.reconciled,
            "seed {seed} tick {}: resource reconciliation breach: {:?}",
            world.tick().0,
            report.reconciliation
        );
        step_gate.lock().expect("gate lock").observe(report);
        Ok(summary)
    });

    let (control, command_drain, command_submit) = {
        let config = scriptbots_app::ControlServerConfig {
            rest_enabled: false,
            mcp_transport: scriptbots_app::McpTransportConfig::Disabled,
            ..scriptbots_app::ControlServerConfig::default()
        };
        let reservation =
            scriptbots_app::ControlServerReservation::prepare(config).expect("prepare control");
        reservation
            .launch(
                Arc::clone(&shared_world),
                scriptbots_app::control::empty_latest_summary(),
            )
            .expect("launch control")
    };

    let context = RendererContext {
        world: Arc::clone(&shared_world),
        simulation_step,
        analytics: AnalyticsSnapshotProvider::empty(),
        control_runtime: &control,
        command_drain,
        command_submit,
        scenario: Arc::new(ScenarioIdentityV0::caller_seeded("meadow-tui")),
    };

    let report = renderer
        .run_headless_frames(context, usize::try_from(ticks).unwrap())
        .expect("render actual TestBackend frames");
    let report = serde_json::to_value(report).expect("serialize observed frames");
    let rendered_frames = report["frames"].as_array().expect("observed frames").len();
    assert_eq!(u64::try_from(rendered_frames).unwrap(), ticks);
    control
        .shutdown()
        .expect("join control runtime before releasing world");
    let world = Arc::try_unwrap(shared_world)
        .expect("unwrap shared world")
        .into_inner()
        .expect("into inner world");

    let digest = world.world_digest_v1().expect("world digest v1");
    let gate = Arc::try_unwrap(gate)
        .expect("step closure released gate")
        .into_inner()
        .expect("gate lock");
    TuiRunResult {
        world,
        digest,
        gate,
        rendered_frames,
    }
}

struct GuiRunResult {
    world: WorldState,
    digest: WorldDigestV1,
    gate: ConservationGate,
    last_png_bytes: Vec<u8>,
}

fn run_gui_path(document: &ScenarioDocumentV1, seed: u64, ticks: u64) -> GuiRunResult {
    let mut world = build_meadow_world(document, seed);
    let mut gate = ConservationGate::new();
    let mut last_png_bytes = Vec::new();

    for tick in 1..=ticks {
        if tick == ticks {
            world.request_replay_world_digest();
        }

        world.step().expect("step world in GUI path");

        // Offscreen GUI rendering at select cadence and final tick
        if tick % 50 == 0 || tick == ticks {
            let png = render_png_offscreen(&world, 800, 450);
            assert!(
                !png.is_empty() && png.starts_with(b"\x89PNG\r\n\x1a\n"),
                "seed {seed} tick {tick}: GUI offscreen raster must emit valid PNG header"
            );
            last_png_bytes = png;
        }

        if let Some(ref report) = world.resource_ledger().latest {
            assert!(
                report.reconciliation.reconciled,
                "seed {seed} tick {tick}: GUI resource reconciliation breach! unexplained={:?}, tol={}",
                report.reconciliation.unexplained_delta, report.reconciliation.tolerance
            );
            gate.observe(report);
        }
    }

    let digest = world.world_digest_v1().expect("world digest v1");

    GuiRunResult {
        world,
        digest,
        gate,
        last_png_bytes,
    }
}

#[test]
fn meadow_testbackend_cpu_png_parity_and_balanced_ledger_cohort() {
    let document = load_meadow_scenario();
    let envelope = document
        .envelope
        .clone()
        .expect("envelope exists on meadow scenario");
    let ticks = envelope.ticks;

    println!("================================================================================");
    println!(
        "MEADOW HELPER COHORT: TESTBACKEND/CPU PNG PARITY & BALANCED LEDGER (bd-2z0.10.5 remains open)"
    );
    println!(
        "Cohort seeds: {:?}, Horizon: {} ticks",
        document.seeds, ticks
    );
    println!(
        "Envelope: pop=[{:?}, {:?}], births_min={:?}, deaths_min={:?}",
        envelope.population_min, envelope.population_max, envelope.births_min, envelope.deaths_min
    );
    println!("================================================================================");

    let mut seed_verdicts: Vec<SeedVerdict> = Vec::with_capacity(document.seeds.len());
    let mut cohort_evidence = Vec::new();

    for &seed in &document.seeds {
        println!("\n--- Executing Seed {seed} ---");

        // 1. Run TUI path (terminal headless drawing frames into test buffer)
        let tui_result = run_tui_path(&document, seed, ticks);
        assert_eq!(u64::try_from(tui_result.rendered_frames).unwrap(), ticks);
        assert_envelope("meadow-tui", seed, &envelope, &tui_result.world);
        println!(
            "  [TUI] Finished {} ticks, pop={}, births={}, deaths={}, digest={}",
            ticks,
            tui_result.world.agent_count(),
            tui_result
                .world
                .history()
                .map(|h| h.births as usize)
                .sum::<usize>(),
            tui_result
                .world
                .history()
                .map(|h| h.deaths as usize)
                .sum::<usize>(),
            tui_result.digest.overall
        );

        // 2. Run GUI path (stepping simulation & rendering offscreen PNG canvas)
        let gui_result = run_gui_path(&document, seed, ticks);
        assert_envelope("meadow-gui", seed, &envelope, &gui_result.world);
        println!(
            "  [GUI] Finished {} ticks, pop={}, births={}, deaths={}, digest={}",
            ticks,
            gui_result.world.agent_count(),
            gui_result
                .world
                .history()
                .map(|h| h.births as usize)
                .sum::<usize>(),
            gui_result
                .world
                .history()
                .map(|h| h.deaths as usize)
                .sum::<usize>(),
            gui_result.digest.overall
        );

        // 3. SCIENTIFIC PARITY ASSERTION
        assert_eq!(
            tui_result.digest.overall, gui_result.digest.overall,
            "seed {seed}: GUI and TUI scientific digests diverged! TUI={} GUI={}",
            tui_result.digest.overall, gui_result.digest.overall
        );
        println!(
            "  [PARITY] EXACT BIT-FOR-BIT MATCH between GUI and TUI: {}",
            tui_result.digest.overall
        );

        // Verify PNG canvas output was produced and valid
        assert!(
            gui_result.last_png_bytes.len() > 1000,
            "seed {seed}: GUI offscreen canvas PNG size must be substantial"
        );

        // 4. BALANCED LEDGER: seal the seed verdict
        let seed_verdict = tui_result.gate.finish(seed);
        let gui_verdict = gui_result.gate.finish(seed);
        assert!(
            gui_verdict.breaches.is_empty() && gui_verdict.truncated_breaches == 0,
            "CPU PNG path must also conserve resources"
        );
        assert!(
            seed_verdict.breaches.is_empty() && seed_verdict.truncated_breaches == 0,
            "seed {seed}: expected 0 ledger breaches, got {} (+{})",
            seed_verdict.breaches.len(),
            seed_verdict.truncated_breaches
        );
        println!(
            "  [LEDGER] Seed {seed} 0 breaches; gross flows: food={:.2e}, energy={:.2e}, health={:.2e}",
            seed_verdict.gross_flow[0], seed_verdict.gross_flow[1], seed_verdict.gross_flow[2]
        );

        cohort_evidence.push(serde_json::json!({
            "seed": seed,
            "ticks": ticks,
            "tui_digest": tui_result.digest.overall,
            "gui_digest": gui_result.digest.overall,
            "parity": true,
            "final_population": tui_result.world.agent_count(),
            "gross_flow_food": seed_verdict.gross_flow[0],
            "gross_flow_energy": seed_verdict.gross_flow[1],
            "gross_flow_health": seed_verdict.gross_flow[2],
            "cumulative_residual_food": seed_verdict.cumulative_residual[0],
            "cumulative_residual_energy": seed_verdict.cumulative_residual[1],
            "cumulative_residual_health": seed_verdict.cumulative_residual[2],
        }));

        seed_verdicts.push(seed_verdict);
    }

    // 5. COHORT CONSERVATION EVALUATION
    let verdict: ConservationVerdict = evaluate_conservation(&seed_verdicts);
    println!("\n--- Economy Conservation Audit Verdict ---");
    println!("  pass: {}", verdict.pass);
    println!("  total_breaches: {}", verdict.total_breaches());
    println!("  failures: {:?}", verdict.failures);
    println!("  summary: {}", verdict.summary_line(None));

    assert!(
        verdict.pass,
        "Economy conservation audit must PASS across the entire meadow cohort"
    );
    assert_eq!(
        verdict.total_breaches(),
        0,
        "Economy audit must have exactly ZERO breaches"
    );
    assert!(verdict.failures.is_empty(), "Failures list must be empty");
    assert!(
        !verdict.tolerance_overridden,
        "Tolerance override must never be set in production proof"
    );

    // 6. INJECTED NEGATIVE CONTROLS
    println!("\n--- Running Injected Negative Controls ---");

    // Negative Control 1: Parity divergence on differing seed
    {
        let tui_run = run_tui_path(&document, 42, 50);
        let gui_run = run_gui_path(&document, 43, 50);
        assert_ne!(
            tui_run.digest.overall, gui_run.digest.overall,
            "Negative control 1 failed: differing seeds (42 vs 43) must produce differing digests!"
        );
        println!("  [NEG-1 PASSED] Seed divergence produces distinct digests (seed 42 != seed 43)");
    }

    // Negative Control 2: Conservation gate catches injected leak
    {
        let mut fault_gate = ConservationGate::new();
        // Feed an unreconciled report with an artificial energy loss
        let breach_report = ResourceLedgerTick {
            tick: Tick(1),
            opening: ResourceAmounts {
                food: 100.0,
                energy: 100.0,
                health: 100.0,
            },
            closing: ResourceAmounts {
                food: 100.0,
                energy: 95.0,
                health: 100.0,
            }, // 5.0 missing!
            flows: vec![ResourceFlow {
                kind: ResourceFlowKind::FoodDynamics,
                delta: ResourceAmounts::default(),
                activity: ResourceAmounts::default(),
            }],
            reconciliation: scriptbots_core::ResourceReconciliation {
                observed_delta: ResourceAmounts {
                    food: 0.0,
                    energy: -5.0,
                    health: 0.0,
                },
                attributed_delta: ResourceAmounts::default(),
                unexplained_delta: ResourceAmounts {
                    food: 0.0,
                    energy: -5.0,
                    health: 0.0,
                },
                tolerance: 1e-6,
                reconciled: false, // Breach!
            },
        };
        fault_gate.observe(&breach_report);
        let fault_seed_verdict = fault_gate.finish(999);
        let fault_verdict = evaluate_conservation(&[fault_seed_verdict]);
        assert!(
            !fault_verdict.pass,
            "Negative control 2 failed: injected breach report must cause evaluate_conservation to FAIL"
        );
        assert_eq!(
            fault_verdict.total_breaches(),
            1,
            "Negative control 2: exactly 1 breach must be reported"
        );
        println!(
            "  [NEG-2 PASSED] Conservation gate detected artificial breach (pass=false, breaches=1)"
        );
    }

    // 7. EMIT MACHINE-PARSEABLE STRUCTURED JSON EVIDENCE
    let evidence_json = serde_json::json!({
        "schema": "scriptbots.meadow-acceptance.v1",
        "terminal_backend": "ratatui_testbackend",
        "image_backend": "cpu_png_offscreen",
        "production_gui_and_pty_verified": false,
        "scenario": "meadow",
        "seeds": document.seeds,
        "ticks": ticks,
        "parity_verified": true,
        "ledger_balanced": true,
        "total_breaches": verdict.total_breaches(),
        "envelope_verified": true,
        "negative_controls_verified": true,
        "cohort": cohort_evidence,
    });

    println!("\nEVIDENCE_START");
    println!(
        "{}",
        serde_json::to_string(&evidence_json).expect("serialize evidence JSON")
    );
    println!("EVIDENCE_END");
    println!(
        "\nMeadow helper cohort checks passed; production GUI/PTY acceptance remains unverified."
    );
}
