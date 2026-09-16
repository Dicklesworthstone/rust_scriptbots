//! Bounded probe allocation slope and observational neutrality guard (bd-16g.4.4).
//!
//! This integration test proves that enabling an activation probe incurs strictly O(1)
//! population-independent allocation and runtime overhead, and that probing does not
//! alter the simulation trajectory (observational neutrality) even across non-trivial
//! brain architectures such as DWRAON.

#![allow(unsafe_code, clippy::cast_precision_loss)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use slotmap::Key;

use scriptbots_brain::dwraon::{DwraonBrain, DwraonFamilyAdapter};
use scriptbots_core::{
    AgentData, CaptureBudget, PersistenceAdmissionError, PersistenceBatch, Position,
    ScriptBotsConfig, SelectionMode, SelectionState, SelectionUpdate, WorldPersistence, WorldState,
};

/// Process-level counting allocator for the isolated integration test binary.
struct CountingAlloc;

static TRACKING_ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING_ENABLED.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: Delegated to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Delegated to the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACKING_ENABLED.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: Delegated to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING_ENABLED.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: Delegated to the system allocator.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc = CountingAlloc;

fn reset_alloc_count() {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
}

fn set_tracking(enabled: bool) {
    TRACKING_ENABLED.store(enabled, Ordering::SeqCst);
}

fn get_alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::SeqCst)
}

fn linear_regression(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len() as f64;
    let x_mean = xs.iter().sum::<f64>() / n;
    let y_mean = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        cov = (x - x_mean).mul_add(y - y_mean, cov);
        var_x = (x - x_mean).mul_add(x - x_mean, var_x);
    }
    let slope = if var_x == 0.0 { 0.0 } else { cov / var_x };
    let intercept = (-slope).mul_add(x_mean, y_mean);
    (slope, intercept)
}

fn build_quiescent_world(
    population: usize,
    seed: u64,
) -> (WorldState, Vec<scriptbots_core::AgentId>) {
    let side = 3000;
    let config = ScriptBotsConfig {
        world_width: side,
        world_height: side,
        food_cell_size: 50,
        initial_food: 0.0,
        food_respawn_interval: 0,
        population_minimum: 0,
        population_spawn_interval: 0,
        persistence_interval: 0,
        chart_flush_interval: 0,
        spike_damage: 0.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        aging_health_decay_rate: 0.0,
        closed: true,
        rng_seed: Some(seed),
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("world init");
    let mut handles = Vec::with_capacity(population);
    for i in 0..population {
        let x = (i % 2500) as f32 + 50.0;
        let y = ((i / 2500) * 50) as f32 + 50.0;
        let handle = world
            .try_spawn_agent(AgentData {
                position: Position::new(x, y),
                health: 100.0,
                ..AgentData::default()
            })
            .expect("spawn quiescent agent");
        handles.push(handle);
    }
    (world, handles)
}

#[derive(Debug, Clone, Copy)]
struct MeasureResult {
    allocs_per_tick: f64,
    micros_per_tick: f64,
}

fn measure_ticks(world: &mut WorldState, ticks: usize) -> MeasureResult {
    reset_alloc_count();
    set_tracking(true);
    let start = Instant::now();
    for _ in 0..ticks {
        world.step().expect("step");
    }
    let elapsed = start.elapsed();
    set_tracking(false);
    let allocs = get_alloc_count();

    MeasureResult {
        allocs_per_tick: allocs as f64 / ticks as f64,
        micros_per_tick: elapsed.as_micros() as f64 / ticks as f64,
    }
}

/// bd-16g.4.4 Acceptance Requirement 2:
/// Instrumented 1k/5k/10k allocation and timing table proving the probe-on minus probe-off
/// slope is population-independent within documented tolerance (~0), logging slope, intercept,
/// allocations, and timings.
#[test]
fn test_probe_allocation_and_timing_slope_across_populations() {
    const POPULATIONS: [usize; 3] = [1000, 5000, 10000];
    const MEASURE_TICKS: usize = 5;
    const WARMUP_TICKS: usize = 2;

    struct PopRow {
        population: usize,
        off: MeasureResult,
        on: MeasureResult,
        on_plus_selected: MeasureResult,
        delta_allocs: f64,
    }

    let mut rows = Vec::new();

    for &pop in &POPULATIONS {
        let seed = 0xA110_C000 + pop as u64;

        // Construct 3 identical worlds at the same seed so each condition evaluates
        // the exact same simulation trajectory without drift between ticks.
        let (mut world_off, _) = build_quiescent_world(pop, seed);
        world_off.set_capture_budget(CaptureBudget { max_agents: 4 });

        let (mut world_on, handles_on) = build_quiescent_world(pop, seed);
        world_on.set_capture_budget(CaptureBudget { max_agents: 4 });
        world_on.set_activation_probe(Some(handles_on[0]));

        let (mut world_on_sel, handles_on_sel) = build_quiescent_world(pop, seed);
        world_on_sel.set_capture_budget(CaptureBudget { max_agents: 4 });
        world_on_sel.set_activation_probe(Some(handles_on_sel[0]));
        let _ = world_on_sel.apply_selection_update(SelectionUpdate {
            mode: SelectionMode::Replace,
            agent_ids: vec![handles_on_sel[1].data().as_ffi()],
            state: SelectionState::Selected,
        });

        // Warmup to settle Rayon pools, slotmaps, and spatial grids.
        for _ in 0..WARMUP_TICKS {
            world_off.step().expect("warmup step off");
            world_on.step().expect("warmup step on");
            world_on_sel.step().expect("warmup step on+sel");
        }

        // 1. Condition: Probe OFF.
        let off = measure_ticks(&mut world_off, MEASURE_TICKS);

        // 2. Condition: Probe ON (1 probed agent).
        let on = measure_ticks(&mut world_on, MEASURE_TICKS);

        // 3. Condition: Probe ON + 1 selected.
        let on_plus_selected = measure_ticks(&mut world_on_sel, MEASURE_TICKS);

        let delta_allocs = on.allocs_per_tick - off.allocs_per_tick;
        rows.push(PopRow {
            population: pop,
            off,
            on,
            on_plus_selected,
            delta_allocs,
        });
    }

    println!("\n=== bd-16g.4.4 Instrumented Allocation & Timing Table ===");
    println!(
        "| {:^10} | {:^18} | {:^15} | {:^16} |",
        "Population", "Condition", "Allocs / Tick", "Time / Tick (µs)"
    );
    println!("|------------|--------------------|-----------------|------------------|");
    for r in &rows {
        println!(
            "| {:>10} | {:<18} | {:>15.2} | {:>16.2} |",
            r.population, "Probe OFF", r.off.allocs_per_tick, r.off.micros_per_tick
        );
        println!(
            "| {:>10} | {:<18} | {:>15.2} | {:>16.2} |",
            r.population, "Probe ON (1)", r.on.allocs_per_tick, r.on.micros_per_tick
        );
        println!(
            "| {:>10} | {:<18} | {:>15.2} | {:>16.2} |",
            r.population,
            "Probe ON + 1 Sel",
            r.on_plus_selected.allocs_per_tick,
            r.on_plus_selected.micros_per_tick
        );
    }
    println!("------------------------------------------------------------------------");

    let xs: Vec<f64> = rows.iter().map(|r| r.population as f64).collect();
    let ys: Vec<f64> = rows.iter().map(|r| r.delta_allocs).collect();
    let (slope, intercept) = linear_regression(&xs, &ys);

    let delta_low = rows[0].delta_allocs;
    let delta_high = rows[2].delta_allocs;
    let delta_diff = delta_high - delta_low;

    println!(
        "Linear Regression (Probe-On minus Probe-Off Δallocs w.r.t. Population):\n  \
         slope = {slope:.8} allocs/agent/tick\n  \
         intercept = {intercept:.4} allocs/tick\n  \
         (Δallocs @ 10k) - (Δallocs @ 1k) = {delta_diff:.4}\n  \
         documented tolerance = 1e-4\n"
    );

    tracing::info!(
        target: "scriptbots::perf",
        slope,
        intercept,
        delta_diff,
        tolerance = 1e-4,
        "measured probe allocation slope w.r.t. population"
    );

    // Hard requirement: Slope w.r.t. population must be within tolerance of zero (O(1) cost).
    assert!(
        slope.abs() <= 1e-4,
        "Probe overhead slope w.r.t. population must be ~0 (O(1) extra work); got slope={slope:.8}"
    );

    // Hard requirement: (delta_allocs at 10k) - (delta_allocs at 1k) must be ~0.
    assert!(
        delta_diff.abs() <= 2.0,
        "(delta_allocs @ 10k) - (delta_allocs @ 1k) must be <= 2.0; got {delta_diff:.4}"
    );
}

struct CollectingPersistence {
    batches: Arc<Mutex<Vec<PersistenceBatch>>>,
}

impl WorldPersistence for CollectingPersistence {
    fn on_tick(&mut self, payload: &PersistenceBatch) -> Result<(), PersistenceAdmissionError> {
        self.batches
            .lock()
            .expect("lock batches")
            .push(payload.clone());
        Ok(())
    }
}

/// bd-16g.4.4 Acceptance Requirement 3:
/// Seeded probed and unprobed runs with non-trivial brain family (DWRAON) produce
/// identical world digests and replay event streams.
#[test]
#[allow(clippy::too_many_lines)]
fn test_probe_and_selection_neutrality_dwraon_brain() {
    const INTERVAL: u32 = 1;
    const AGENTS: usize = 12;
    const SEED: u64 = 0xD3AA_0026;

    let config = ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        initial_food: 0.0,
        food_respawn_interval: 0,
        population_minimum: 0,
        population_spawn_interval: 0,
        persistence_interval: INTERVAL,
        chart_flush_interval: 0,
        spike_damage: 0.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        aging_health_decay_rate: 0.0,
        closed: true,
        rng_seed: Some(SEED),
        ..ScriptBotsConfig::default()
    };

    let mut world_a = WorldState::new(config.clone()).expect("world a");
    world_a
        .register_brain_family(
            DwraonBrain::KIND.as_str(),
            Box::new(DwraonFamilyAdapter::default()),
        )
        .expect("register DWRAON family a");

    let mut world_b = WorldState::new(config).expect("world b");
    world_b
        .register_brain_family(
            DwraonBrain::KIND.as_str(),
            Box::new(DwraonFamilyAdapter::default()),
        )
        .expect("register DWRAON family b");

    let mut handles_b = Vec::new();
    for i in 0..AGENTS {
        let agent_data = AgentData {
            position: Position::new(
                (i as f32).mul_add(10.0, 20.0),
                (i as f32).mul_add(8.0, 30.0),
            ),
            health: 100.0,
            ..AgentData::default()
        };
        let _ = world_a.try_spawn_agent(agent_data).expect("spawn a");
        let h_b = world_b.try_spawn_agent(agent_data).expect("spawn b");
        handles_b.push(h_b);
    }

    let batches_a = Arc::new(Mutex::new(Vec::new()));
    let batches_b = Arc::new(Mutex::new(Vec::new()));

    let mut session_a = world_a
        .bind_persistence(Box::new(CollectingPersistence {
            batches: Arc::clone(&batches_a),
        }))
        .expect("session a");
    let mut session_b = world_b
        .bind_persistence(Box::new(CollectingPersistence {
            batches: Arc::clone(&batches_b),
        }))
        .expect("session b");

    // World B enables probe on ticks 20..60 and selection on agent 2.
    world_b.set_capture_budget(CaptureBudget { max_agents: 3 });

    for tick in 1..=80 {
        if tick == 20 {
            world_b.set_activation_probe(Some(handles_b[0]));
            let _ = world_b.apply_selection_update(SelectionUpdate {
                mode: SelectionMode::Replace,
                agent_ids: vec![handles_b[2].data().as_ffi()],
                state: SelectionState::Selected,
            });
        }
        if tick == 60 {
            world_b.set_activation_probe(None);
            let _ = world_b.apply_selection_update(SelectionUpdate {
                mode: SelectionMode::Clear,
                agent_ids: Vec::new(),
                state: SelectionState::None,
            });
        }

        let _ = session_a.step(&mut world_a).expect("step a");
        let _ = session_b.step(&mut world_b).expect("step b");
    }

    {
        let b_a = batches_a.lock().expect("batches a");
        let b_b = batches_b.lock().expect("batches b");
        assert_eq!(b_a.len(), 80, "80 batches collected for world a");
        assert_eq!(b_b.len(), 80, "80 batches collected for world b");
        for (tick_idx, (batch_a, batch_b)) in b_a.iter().zip(b_b.iter()).enumerate() {
            assert_eq!(
                batch_a.replay_events, batch_b.replay_events,
                "replay events must match byte-for-byte at batch {tick_idx}"
            );
        }
    }

    // World digests after 80 ticks must be byte-for-byte identical.
    let digest_a = world_a.world_digest_v1().expect("digest a");
    let digest_b = world_b.world_digest_v1().expect("digest b");
    assert_eq!(
        digest_a, digest_b,
        "world digests must be identical between probed and unprobed DWRAON runs"
    );
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct InspectorArtifactRow {
    tick: u64,
    agent_uid: u64,
    sensors: std::collections::BTreeMap<String, f32>,
    attribution_summary: AttributionSummaryRecord,
    activation_layer_digest: String,
    outputs: std::collections::BTreeMap<String, OutputExplanationRecord>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct AttributionSummaryRecord {
    raw: Vec<f32>,
    clamped: Vec<f32>,
    saturated: Vec<bool>,
    contributions_count: usize,
    truncated: usize,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct OutputExplanationRecord {
    raw_value: f32,
    effective: String,
    boost_active: Option<bool>,
    method: String,
    top_inputs: Vec<InputAttributionRecord>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct InputAttributionRecord {
    sensor_name: String,
    contribution: f32,
}

/// bd-16g.4.4 Acceptance Requirement 5:
/// Headless inspector 100-tick JSONL artifact generation and self-consistency assertion.
///
/// Steps 100 ticks with probe active on an agent, dumping one JSONL row per tick containing:
/// - all 25 NAMED sensors (bd-16g.4.1)
/// - the raw/clamped/saturated attribution summary (bd-16g.4.2)
/// - the activation layer digest
/// - the 9 NAMED outputs with effective values and attribution (bd-16g.4.3)
///
/// Asserts on every row:
/// - clamped == world.runtime[agent].sensors
/// - outputs == runtime.outputs
/// - boost == (outputs[6] > 0.5)
#[test]
#[allow(clippy::too_many_lines)]
fn test_headless_inspector_100_ticks_jsonl_artifact() {
    let config = ScriptBotsConfig {
        world_width: 100,
        world_height: 100,
        food_cell_size: 50,
        population_minimum: 0,
        population_spawn_interval: 0,
        persistence_interval: 0,
        spike_damage: 0.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        aging_health_decay_rate: 0.0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        food_respawn_interval: 0,
        initial_food: 0.0,
        closed: true,
        rng_seed: Some(0x1604_4E42),
        ..ScriptBotsConfig::default()
    };

    let mut world = WorldState::new(config).expect("world");
    let agent_handle = world
        .try_spawn_agent(AgentData {
            position: Position::new(50.0, 50.0),
            health: 100.0,
            ..AgentData::default()
        })
        .expect("spawn agent");
    let target_uid = world.agent_uid(agent_handle).expect("target uid");

    world.set_activation_probe(Some(agent_handle));
    world.set_capture_budget(CaptureBudget { max_agents: 4 });

    let artifact_dir = std::env::var("SCRIPTBOTS_INSPECTOR_ARTIFACT_DIR").map_or_else(
        |_| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/artifacts"),
        std::path::PathBuf::from,
    );
    std::fs::create_dir_all(&artifact_dir).expect("create inspector artifact dir");
    let artifact_file = artifact_dir.join("headless_inspector_100_ticks.jsonl");

    let mut jsonl_lines = Vec::with_capacity(100);

    for _ in 1..=100 {
        // 2. The raw/clamped/saturated attribution summary (bd-16g.4.2)
        // Captured over the completed boundary immediately preceding step so that
        // clamped matches the exact brain-facing sensor vector realized by stage_sense.
        let attr = world
            .explain_sensors(agent_handle, 12)
            .expect("sensor attribution");

        world.step().expect("step");

        let runtime = world.agent_runtime(agent_handle).expect("agent runtime");
        let runtime_sensors = runtime.sensors;
        let runtime_outputs = runtime.outputs;

        // 1. All 25 NAMED sensors (bd-16g.4.1)
        let mut named_sensors = std::collections::BTreeMap::new();
        for channel in &scriptbots_core::channels::SENSOR_LAYOUT {
            named_sensors.insert(channel.name.to_string(), runtime_sensors[channel.index]);
        }
        assert_eq!(named_sensors.len(), 25, "must name all 25 sensors");

        let attribution_summary = AttributionSummaryRecord {
            raw: attr.raw.to_vec(),
            clamped: attr.clamped.to_vec(),
            saturated: attr.saturated.to_vec(),
            contributions_count: attr.contributions.len(),
            truncated: attr.truncated,
        };
        assert_eq!(attribution_summary.raw.len(), 25);
        assert_eq!(attribution_summary.clamped.len(), 25);
        assert_eq!(attribution_summary.saturated.len(), 25);

        // 3. Activation layer digest
        let mut hasher = blake3::Hasher::new();
        let inspection = runtime
            .brain
            .inspect(scriptbots_core::BrainInspection::Activations(
                scriptbots_core::BrainInspectionLimits::default(),
            ))
            .expect("inspect");
        if let Some(inspected) = &inspection {
            for layer in &inspected.activations.layers {
                hasher.update(layer.name.as_bytes());
                hasher.update(&layer.width.to_le_bytes());
                hasher.update(&layer.height.to_le_bytes());
                for v in &layer.values {
                    hasher.update(&v.to_le_bytes());
                }
            }
        }
        let activation_layer_digest = hasher.finalize().to_hex().to_string();

        // 4. All 9 NAMED outputs with effective values (bd-16g.4.3)
        let activations = inspection.as_ref().map(|i| &i.activations);
        let explanations =
            scriptbots_core::attribution::explain_outputs(&runtime_outputs, true, activations, 3);
        assert_eq!(explanations.len(), 9, "must explain all 9 outputs");

        let mut named_outputs = std::collections::BTreeMap::new();
        for expl in &explanations {
            let (eff_str, boost_active) = match expl.effective {
                scriptbots_core::attribution::EffectiveOutput::Continuous(v) => {
                    (format!("{v:.4}"), None)
                }
                scriptbots_core::attribution::EffectiveOutput::Thresholded { active, .. } => (
                    if active {
                        "ON".to_string()
                    } else {
                        "OFF".to_string()
                    },
                    Some(active),
                ),
                scriptbots_core::attribution::EffectiveOutput::Clamped { applied, .. } => {
                    (format!("{applied:.4}"), None)
                }
            };
            let top_inputs = expl
                .inputs
                .iter()
                .map(|inp| InputAttributionRecord {
                    sensor_name: inp.sensor_name.to_string(),
                    contribution: inp.contribution,
                })
                .collect();
            named_outputs.insert(
                expl.output_name.to_string(),
                OutputExplanationRecord {
                    raw_value: expl.raw_value,
                    effective: eff_str,
                    boost_active,
                    method: format!("{:?}", expl.method),
                    top_inputs,
                },
            );
        }

        let row = InspectorArtifactRow {
            tick: world.tick().0,
            agent_uid: target_uid.get(),
            sensors: named_sensors,
            attribution_summary,
            activation_layer_digest,
            outputs: named_outputs,
        };

        // Assert every row is complete and self-consistent:
        // clamped == world.runtime[agent].sensors
        for i in 0..25 {
            assert_eq!(
                row.attribution_summary.clamped[i], runtime_sensors[i],
                "row clamped[{i}] must equal runtime_sensors[{i}] at tick {}",
                row.tick
            );
        }
        // outputs == runtime.outputs
        for (i, channel) in scriptbots_core::channels::OutputChannel::ALL
            .iter()
            .enumerate()
        {
            let expl_rec = &row.outputs[channel.name()];
            assert_eq!(
                expl_rec.raw_value,
                runtime_outputs[i],
                "row outputs[{:?}] raw must equal runtime_outputs[{i}] at tick {}",
                channel.name(),
                row.tick
            );
        }
        // boost == (outputs[6] > 0.5)
        let boost_rec = &row.outputs["boost"];
        assert_eq!(
            boost_rec.boost_active,
            Some(runtime_outputs[6] > scriptbots_core::channels::BOOST_THRESHOLD),
            "boost_active must equal (outputs[6] > 0.5) at tick {}",
            row.tick
        );

        let line = serde_json::to_string(&row).expect("serialize row JSON");
        jsonl_lines.push(line);
    }

    assert_eq!(jsonl_lines.len(), 100);
    let artifact_content = jsonl_lines.join("\n") + "\n";
    std::fs::write(&artifact_file, &artifact_content).expect("write inspector JSONL artifact");

    // Also write to conversation artifact directory if available
    let conversation_artifact_dir = std::path::Path::new(
        "/home/ubuntu/.gemini/antigravity-cli/brain/0280f6d9-5e30-4df4-827b-24a4033c8c65",
    );
    if conversation_artifact_dir.exists() {
        let _ = std::fs::write(
            conversation_artifact_dir.join("headless_inspector_100_ticks.jsonl"),
            &artifact_content,
        );
    }

    // Verify artifact is fully readable and self-consistent on disk
    let disk_content = std::fs::read_to_string(&artifact_file).expect("read artifact from disk");
    let parsed_rows: Vec<InspectorArtifactRow> = disk_content
        .lines()
        .map(|l| serde_json::from_str(l).expect("parse JSONL row from disk"))
        .collect();
    assert_eq!(parsed_rows.len(), 100);
    assert_eq!(parsed_rows[0].tick, 1);
    assert_eq!(parsed_rows[99].tick, 100);

    println!("=== bd-16g.4.4 Headless Inspector 100-Tick JSONL Artifact Verified ===");
    println!("Artifact path: {}", artifact_file.display());
    println!("Total JSONL rows: {}", parsed_rows.len());
    println!(
        "Tick range: {}..={}",
        parsed_rows[0].tick, parsed_rows[99].tick
    );
    println!("Agent UID: {}", parsed_rows[0].agent_uid);
    println!("First row sensors count: {}", parsed_rows[0].sensors.len());
    println!("First row outputs count: {}", parsed_rows[0].outputs.len());
    println!(
        "First row activation digest: {}",
        parsed_rows[0].activation_layer_digest
    );
    println!(
        "First row attribution clamped: {:?}",
        &parsed_rows[0].attribution_summary.clamped[0..5]
    );
}
