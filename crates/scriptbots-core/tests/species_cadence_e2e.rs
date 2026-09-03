//! End-to-end multi-cohort species cadence and offline reconstruction test (bd-16g.3.6).
//!
//! Acceptance Criteria 3 & 4:
//! - Runs a multi-cohort world through at least two cadence boundaries.
//! - Executes live cadence segmentation, publishes species snapshots, and persists species tables.
//! - Reconstructs species tables offline from the exact adapted inputs.
//! - Compares byte-identical live vs offline table digests.
//! - Exercises fault injection to verify agreement gate failure.

use scriptbots_core::species::{
    SpeciesAdapterFault, SpeciesCadence, SpeciesCadenceStepResult, SpeciesParams,
    SpeciesPhenotypeAdapterConfig, SpeciesSnapshotProvider, SpeciesTable, TypedPhenotypeInput,
    adapt_phenotype_samples, reconstruct_species_table_offline, step_species_cadence,
};
use scriptbots_core::{AgentData, AgentDebugQuery, ScriptBotsConfig, Tick, WorldState};

#[test]
fn test_species_cadence_multi_cohort_e2e_reconstruction_and_fault_gate() {
    let config = ScriptBotsConfig {
        population_minimum: 20,
        population_spawn_interval: 1,
        ..Default::default()
    };
    let mut world = WorldState::new(config).expect("world initialization");

    // Spawn 20 initial agents at tick 0
    for _ in 0..20 {
        let _ = world.try_spawn_agent(AgentData::default());
    }

    let cadence = SpeciesCadence::new(10); // Boundary every 10 ticks
    let params = SpeciesParams::default();
    let clean_adapter_config = SpeciesPhenotypeAdapterConfig::default();
    let publisher = SpeciesSnapshotProvider::new();

    let mut current_table = SpeciesTable::default();
    let mut boundary_snapshots = Vec::new();
    let mut captured_inputs_per_boundary = Vec::new();

    // Step world through at least 2 cadence boundaries: ticks 10 and 20
    for tick_num in 1..=25 {
        world.step().expect("world step");
        let tick = Tick(tick_num);

        // Derive phenotype inputs from living agents with divergent cohort traits
        let live_agents = world.agent_debug_view(AgentDebugQuery::default());
        assert!(!live_agents.is_empty(), "living agents must be present");

        let inputs: Vec<TypedPhenotypeInput> = live_agents
            .iter()
            .enumerate()
            .map(|(idx, agent)| {
                // Synthesize two distinct trait cohorts:
                // Cohort A (even idx): fast, herbivore, low aggression
                // Cohort B (odd idx): slow, carnivore, high aggression
                let is_cohort_a = idx % 2 == 0;
                let features = if is_cohort_a {
                    [0.2, 0.9, 0.8, 0.05, 0.4, 0.1]
                } else {
                    [0.8, 0.1, 0.3, 0.85, 0.05, 0.2]
                };
                TypedPhenotypeInput::new(agent.agent_uid, u64::from(agent.age) + 1, features)
            })
            .collect();

        let step_result = step_species_cadence(
            tick,
            cadence,
            &params,
            &clean_adapter_config,
            &inputs,
            &current_table,
            Some(&publisher),
        );

        if cadence.should_segment(tick) {
            match step_result {
                SpeciesCadenceStepResult::Segmented { snapshot, .. } => {
                    current_table = snapshot.table.clone();
                    boundary_snapshots.push(snapshot);
                    captured_inputs_per_boundary.push((tick, inputs));
                }
                other => panic!("expected segmented result at tick {tick_num}, got {other:?}"),
            }
        } else {
            assert_eq!(step_result, SpeciesCadenceStepResult::OffCadence);
        }
    }

    // Must have crossed at least two cadence boundaries (ticks 10 and 20)
    assert!(
        boundary_snapshots.len() >= 2,
        "must cross at least 2 cadence boundaries, crossed {}",
        boundary_snapshots.len()
    );

    // Verify offline reconstruction for each crossed boundary
    let mut prev_offline = SpeciesTable::default();
    for (i, (tick, inputs)) in captured_inputs_per_boundary.iter().enumerate() {
        let live_snap = &boundary_snapshots[i];
        assert_eq!(*tick, live_snap.tick);

        // Offline adaptation & reconstruction
        let adapted = adapt_phenotype_samples(inputs, &clean_adapter_config)
            .expect("clean offline adaptation");
        let (offline_table, offline_report, offline_digest) =
            reconstruct_species_table_offline(*tick, &adapted, &prev_offline, &params);

        // Assert BYTE-IDENTICAL live vs offline parity
        assert_eq!(
            live_snap.table_digest, offline_digest,
            "live vs offline table digest mismatch at tick {tick:?}"
        );
        assert_eq!(live_snap.table, offline_table);
        assert_eq!(live_snap.report, offline_report);

        // Verify serializability and reload
        let serialized = serde_json::to_string(&live_snap.table).expect("serialize table");
        let reloaded: SpeciesTable =
            serde_json::from_str(&serialized).expect("reload species table");
        assert_eq!(reloaded.canonical_digest(), live_snap.table_digest);

        // Advance previous table for next boundary
        prev_offline = offline_table;
    }

    // Criterion 4: Verify agreement gate FAILS on fault injection
    let first_inputs = &captured_inputs_per_boundary[0].1;
    let mut fault_config = clean_adapter_config;
    fault_config.fault = SpeciesAdapterFault::PerturbFeatures;
    let fault_samples = adapt_phenotype_samples(first_inputs, &fault_config)
        .expect("perturbed adaptation succeeds");
    let (_, _, fault_digest) = reconstruct_species_table_offline(
        captured_inputs_per_boundary[0].0,
        &fault_samples,
        &SpeciesTable::default(),
        &params,
    );
    assert_ne!(
        boundary_snapshots[0].table_digest, fault_digest,
        "agreement gate must fail when features are perturbed"
    );
}
