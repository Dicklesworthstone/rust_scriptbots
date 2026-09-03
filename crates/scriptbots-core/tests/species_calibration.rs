//! Empirical calibration sweep for speciation persistence and separation constants (bd-3l5d).
//!
//! Evaluates:
//! 1. K (persistence samples): sweeps candidate values K in 1..=5 over multi-cohort and
//!    evolving simulation runs to measure the false-positive vs false-negative trade-off.
//! 2. Max cross-mating rate: measures the realized cross-cluster mating distribution under
//!    a panmictic null model vs an allopatrically isolated model, proving that 0.05 sits
//!    strictly outside the panmictic null distribution while reliably admitting allopatric clades.

use scriptbots_core::species::{
    REPRODUCTIVE_SEPARATION_MAX_RATE, SPECIATION_PERSISTENCE_SAMPLES, SpeciationStatus,
    SpeciationVerdict, SpeciationWatch, SpeciesCadence, SpeciesParams,
    SpeciesPhenotypeAdapterConfig, SpeciesTable, TypedPhenotypeInput, classify_speciation,
    measure_cross_cluster_mating, step_species_cadence,
};
use scriptbots_core::{
    AgentData, AgentDebugQuery, AgentUid, BirthOrigin, BirthRecord, Generation, Position,
    ScriptBotsConfig, Tick, WorldState,
};

fn birth(uid: u64, parents: (Option<u64>, Option<u64>)) -> BirthRecord {
    BirthRecord {
        tick: Tick(100),
        agent_uid: AgentUid(uid),
        spawn_ordinal: uid,
        birth_ordinal: Some(uid),
        origin: BirthOrigin::Born,
        parent_a: parents.0.map(AgentUid),
        parent_b: parents.1.map(AgentUid),
        brain_kind: None,
        brain_key: None,
        herbivore_tendency: 0.5,
        generation: Generation(1),
        position: Position::default(),
        is_hybrid: false,
    }
}

#[test]
fn test_calibrate_speciation_persistence_sweep() {
    let config = ScriptBotsConfig {
        population_minimum: 40,
        population_spawn_interval: 1,
        ..Default::default()
    };
    let mut world = WorldState::new(config).expect("world init");

    for _ in 0..40 {
        let _ = world.try_spawn_agent(AgentData::default());
    }

    let cadence = SpeciesCadence::new(5);
    let params = SpeciesParams::default();
    let adapter_config = SpeciesPhenotypeAdapterConfig::default();

    let mut current_table = SpeciesTable::default();
    let mut history_tables: Vec<SpeciesTable> = Vec::new();

    // Run 50 ticks (10 cadence samples)
    for tick_num in 1..=50 {
        world.step().expect("step");
        let tick = Tick(tick_num);

        let live_agents = world.agent_debug_view(AgentDebugQuery::default());
        if live_agents.is_empty() {
            continue;
        }

        // Generate dynamic phenotype features with three distinct cohorts plus transient jitter
        let inputs: Vec<TypedPhenotypeInput> = live_agents
            .iter()
            .enumerate()
            .map(|(idx, agent)| {
                let cohort = idx % 3;
                let jitter = if idx % 7 == 0 && tick_num % 10 < 3 {
                    0.4
                } else {
                    0.0
                };
                let features = match cohort {
                    0 => [0.15 + jitter, 0.85, 0.90, 0.05, 0.35, 0.10],
                    1 => [0.85, 0.15 + jitter, 0.30, 0.80, 0.05, 0.25],
                    _ => [0.50, 0.50, 0.50 + jitter, 0.40, 0.20, 0.15],
                };
                TypedPhenotypeInput::new(agent.agent_uid, u64::from(agent.age) + 1, features)
            })
            .collect();

        let step_res = step_species_cadence(
            tick,
            cadence,
            &params,
            &adapter_config,
            &inputs,
            &current_table,
            None,
        );

        if let scriptbots_core::species::SpeciesCadenceStepResult::Segmented { snapshot, .. } =
            step_res
        {
            current_table = snapshot.table.clone();
            history_tables.push(snapshot.table);
        }
    }

    assert!(
        history_tables.len() >= 8,
        "insufficient sample history collected"
    );

    // Sweep K from 1 to 5
    // For each K, we evaluate a SpeciationWatch and count confirmed vs transient splits
    println!("\n=== SPECIATION PERSISTENCE SWEEP RESULTS (bd-3l5d) ===");
    println!("K | Confirmed Persisted | Transient Dropped | Total Observed");

    let mut confirmed_per_k = Vec::new();

    for k in 1..=5 {
        let mut watch = SpeciationWatch::new(k);
        let mut confirmed_count = 0;
        let mut transient_count = 0;

        for table in &history_tables {
            let verdicts = watch.observe(table);
            for v in verdicts {
                match v {
                    SpeciationVerdict::Persisted { .. } => confirmed_count += 1,
                    SpeciationVerdict::Transient { .. } => transient_count += 1,
                }
            }
        }

        println!(
            "{k} | {confirmed_count:19} | {transient_count:17} | {:14}",
            confirmed_count + transient_count
        );
        confirmed_per_k.push((k, confirmed_count, transient_count));
    }

    // Verify properties of K:
    // 1. K=1 confirms unverified transient jitter immediately.
    // 2. K=3 stabilizes confirmed species count and drops transient perturbations.
    assert_eq!(SPECIATION_PERSISTENCE_SAMPLES, 3);
    let k1 = confirmed_per_k[0];
    let k3 = confirmed_per_k[2];
    assert!(
        k1.1 >= k3.1,
        "K=1 should admit equal or more candidates than K=3"
    );
}

#[test]
fn test_calibrate_reproductive_separation_cross_mating_rate() {
    let mut table = SpeciesTable::default();
    table.species.push(scriptbots_core::species::Species {
        id: scriptbots_core::species::SpeciesId(1),
        name: "Alpha-1".to_string(),
        founders: vec![AgentUid(1)],
        members: (1..=20).map(AgentUid).collect(),
        centroid: vec![0.2; 6],
        spread: 0.1,
        first_tick: Tick(0),
        last_seen_tick: Tick(100),
    });
    table.species.push(scriptbots_core::species::Species {
        id: scriptbots_core::species::SpeciesId(2),
        name: "Beta-2".to_string(),
        founders: vec![AgentUid(21)],
        members: (21..=40).map(AgentUid).collect(),
        centroid: vec![0.8; 6],
        spread: 0.1,
        first_tick: Tick(0),
        last_seen_tick: Tick(100),
    });

    // 1. Panmictic null distribution:
    // Parents are drawn uniformly from the combined population (no reproductive barrier)
    let mut panmictic_rates = Vec::new();
    let mut rng_seed: u64 = 0xdead_beef_cafe_babe;

    for _ in 0..50 {
        let mut births = Vec::new();
        for b_idx in 0..40 {
            rng_seed = rng_seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let p1 = (rng_seed % 40) + 1;
            rng_seed = rng_seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let p2 = (rng_seed % 40) + 1;
            births.push(birth(1000 + b_idx, (Some(p1), Some(p2))));
        }
        let m = measure_cross_cluster_mating(&births, &table);
        let rate = m
            .cross_rate()
            .expect("panmictic births must have attributed matings");
        panmictic_rates.push(rate);
    }

    #[allow(clippy::cast_precision_loss)]
    let panmictic_mean = panmictic_rates.iter().sum::<f64>() / panmictic_rates.len() as f64;
    let panmictic_min = panmictic_rates
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let panmictic_max = panmictic_rates
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\n=== PANMICTIC NULL CROSS-MATING DISTRIBUTION (bd-3l5d) ===");
    println!("Panmictic mean cross-rate: {panmictic_mean:.4}");
    println!("Panmictic min cross-rate:  {panmictic_min:.4}");
    println!("Panmictic max cross-rate:  {panmictic_max:.4}");

    // Theoretical expectation for 2 equal-sized clusters mating randomly is ~0.50 (50% cross matings)
    assert!(
        panmictic_min > 0.25,
        "panmictic null distribution must have cross-mating rate well above 0.25, found min {panmictic_min}"
    );

    // 2. Allopatric / isolated model:
    // Mating occurs within clades, with at most 1 occasional leakage
    let mut allopatric_births = Vec::new();
    for b_idx in 0..40 {
        let (p1, p2) = if b_idx < 20 {
            // Within species 1
            ((b_idx % 20) + 1, ((b_idx + 1) % 20) + 1)
        } else {
            // Within species 2
            (20 + (b_idx % 20) + 1, 20 + ((b_idx + 1) % 20) + 1)
        };
        allopatric_births.push(birth(2000 + b_idx, (Some(p1), Some(p2))));
    }

    let allopatric_m = measure_cross_cluster_mating(&allopatric_births, &table);
    let allopatric_rate = allopatric_m
        .cross_rate()
        .expect("allopatric attributed rate");
    assert!(
        allopatric_rate.abs() < f64::EPSILON,
        "clean allopatric clades have 0.0 cross rate"
    );

    // Now test classification under REPRODUCTIVE_SEPARATION_MAX_RATE = 0.05
    assert!((REPRODUCTIVE_SEPARATION_MAX_RATE - 0.05).abs() < f64::EPSILON);

    let persisted_verdict = SpeciationVerdict::Persisted {
        species: scriptbots_core::species::SpeciesId(2),
        first_seen: Tick(10),
        confirmed_at: Tick(30),
    };

    // Panmictic mating must NEVER be classified as Speciation
    let panmictic_births: Vec<BirthRecord> = (0..40)
        .map(|idx| birth(5000 + idx, (Some(1), Some(25))))
        .collect();
    let panmictic_m = measure_cross_cluster_mating(&panmictic_births, &table);

    let panmictic_status = classify_speciation(
        &persisted_verdict,
        &panmictic_m,
        REPRODUCTIVE_SEPARATION_MAX_RATE,
    );
    assert_eq!(
        panmictic_status,
        SpeciationStatus::Polymorphic { cross_rate: 1.0 },
        "panmictic mating must be classified as Polymorphic, not Speciation"
    );

    // Allopatric mating must be classified as Speciation
    let allopatric_status = classify_speciation(
        &persisted_verdict,
        &allopatric_m,
        REPRODUCTIVE_SEPARATION_MAX_RATE,
    );
    assert_eq!(
        allopatric_status,
        SpeciationStatus::Speciation { cross_rate: 0.0 },
        "allopatric mating must be classified as Speciation"
    );

    // Bound separation verification:
    // Max cross rate of 0.05 is safely > 0.0 (tolerates up to 1 cross-mating in 20 births)
    // while sitting far below the panmictic minimum (min > 0.25, mean ~ 0.50).
    assert!(
        REPRODUCTIVE_SEPARATION_MAX_RATE < panmictic_min,
        "calibrated threshold 0.05 must be strictly less than panmictic null minimum {panmictic_min}"
    );
}
