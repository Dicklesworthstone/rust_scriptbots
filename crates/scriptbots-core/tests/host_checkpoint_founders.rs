//! Real founder and recording-boundary continuity through full-owner checkpoints.

use scriptbots_brain::MlpBrain;
use scriptbots_brain::mlp::MlpBrainFamily;
use scriptbots_core::{
    AgentData, AgentUid, BrainRegistry, NullPersistence, PersistedGenome, ScriptBotsConfig, Tick,
    WorldCheckpointError, WorldHostCheckpointV1, WorldState,
};

fn world(persistence_interval: u32, reproduction: bool) -> (WorldState, u64) {
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 80,
        world_height: 80,
        food_cell_size: 20,
        initial_food: 0.0,
        food_respawn_interval: 0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        food_intake_rate: 0.0,
        food_sharing_rate: 0.0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        bot_speed: 0.0,
        spike_damage: 0.0,
        spike_energy_cost: 0.0,
        aging_health_decay_rate: 0.0,
        population_minimum: 0,
        population_spawn_interval: 0,
        reproduction_energy_threshold: if reproduction { 0.5 } else { f32::MAX },
        reproduction_energy_cost: 0.0,
        reproduction_cooldown: 1,
        reproduction_attempt_interval: 1,
        reproduction_attempt_chance: 1.0,
        reproduction_child_energy: 1.0,
        reproduction_spawn_jitter: 0.0,
        reproduction_color_jitter: 0.0,
        reproduction_spawn_back_distance: 0.0,
        reproduction_partner_chance: 0.0,
        reproduction_meta_mutation_chance: 0.0,
        reproduction_meta_mutation_scale: 0.0,
        persistence_interval,
        chart_flush_interval: 0,
        history_capacity: 1,
        narrative_interval: 0,
        narrative_capacity: 0,
        rng_seed: Some(0xC4EC_9A71),
        ..ScriptBotsConfig::default()
    })
    .unwrap();
    let key = world
        .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
        .unwrap();
    (world, key)
}

fn registry() -> BrainRegistry {
    let mut registry = BrainRegistry::new();
    assert_eq!(
        registry
            .register_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
            .unwrap(),
        0
    );
    registry
}

fn declarations(continuity: &WorldHostCheckpointV1) -> Vec<PersistedGenome> {
    let json = serde_json::to_value(continuity).unwrap();
    serde_json::from_value(json["pending_genomes"].clone()).unwrap()
}

fn continuation_bytes(world: &WorldState) -> Vec<u8> {
    let science = world.capture_checkpoint_quiescent().unwrap();
    serde_json::to_vec(&world.capture_host_continuity_v1(&science).unwrap()).unwrap()
}

#[test]
fn fresh_founders_keep_exact_declarations_and_their_first_recording_boundary() {
    for persistence_interval in [0, 1] {
        let (mut original, key) = world(persistence_interval, false);
        let founders: Vec<_> = (0..3)
            .map(|_| original.try_spawn_agent(AgentData::default()).unwrap())
            .collect();
        for founder in founders.into_iter().rev() {
            assert!(original.bind_agent_brain(founder, key).unwrap());
        }
        let before = original.world_digest_v1().unwrap();
        let science = original.capture_checkpoint_quiescent().unwrap();
        let continuity = original.capture_host_continuity_v1(&science).unwrap();
        let pending = declarations(&continuity);
        assert_eq!(pending.len(), 3);
        assert!(
            pending
                .windows(2)
                .all(|pair| pair[0].agent_uid > pair[1].agent_uid)
        );
        let bytes = serde_json::to_vec(&continuity).unwrap();
        let decoded = serde_json::from_slice(&bytes).unwrap();
        let mut restored =
            WorldState::restore_host_checkpoint_v1(&science, &decoded, registry()).unwrap();
        assert_eq!(original.tick(), Tick::zero());
        assert_eq!(original.world_digest_v1().unwrap(), before);
        assert_eq!(restored.world_digest_v1().unwrap(), before);
        assert_eq!(
            serde_json::to_vec(&restored.capture_host_continuity_v1(&science).unwrap()).unwrap(),
            bytes
        );

        let scientific_import = WorldState::restore_checkpoint_v1(&science, registry()).unwrap();
        let imported = declarations(
            &scientific_import
                .capture_host_continuity_v1(&science)
                .unwrap(),
        );
        assert_eq!(imported.len(), pending.len());
        assert!(
            pending
                .iter()
                .all(|declaration| imported.contains(declaration))
        );

        let mut source_session = original
            .bind_persistence(Box::new(NullPersistence))
            .unwrap();
        let mut restored_session = restored
            .bind_persistence(Box::new(NullPersistence))
            .unwrap();
        let source_completion = source_session.step_outcome(&mut original).unwrap();
        let restored_completion = restored_session.step_outcome(&mut restored).unwrap();
        assert!(source_completion.fault.is_none());
        assert!(restored_completion.fault.is_none());
        assert!(source_completion.outcome.births.is_empty());
        assert_eq!(source_completion.outcome.summary.births, 0);
        assert_eq!(
            source_completion.outcome.births,
            restored_completion.outcome.births
        );
        assert_eq!(
            source_session.pending_batch().is_some(),
            persistence_interval == 1
        );
        if let Some(batch) = source_session.pending_batch() {
            assert_eq!(batch.births.len(), 3);
            assert_eq!(batch.genomes, pending);
        }
        assert_eq!(
            source_session.pending_batch().map(|batch| (
                &batch.summary,
                &batch.births,
                &batch.genomes
            )),
            restored_session.pending_batch().map(|batch| (
                &batch.summary,
                &batch.births,
                &batch.genomes
            ))
        );
        source_session.admit_pending(&mut original).unwrap();
        restored_session.admit_pending(&mut restored).unwrap();
        assert_eq!(
            source_session.last_admitted_tick(),
            restored_session.last_admitted_tick()
        );
        assert_eq!(
            original.world_digest_v1().unwrap(),
            restored.world_digest_v1().unwrap()
        );
        assert_eq!(continuation_bytes(&original), continuation_bytes(&restored));
    }
}

#[test]
fn missing_changed_or_duplicate_founder_declarations_are_refused() {
    let (mut original, key) = world(0, false);
    for _ in 0..2 {
        let founder = original.try_spawn_agent(AgentData::default()).unwrap();
        assert!(original.bind_agent_brain(founder, key).unwrap());
    }
    let science = original.capture_checkpoint_quiescent().unwrap();
    let continuity = original.capture_host_continuity_v1(&science).unwrap();
    let json = serde_json::to_value(&continuity).unwrap();
    for defect in ["missing", "duplicate", "identity", "tick", "genome"] {
        let mut changed = json.clone();
        let records = changed["pending_genomes"].as_array_mut().unwrap();
        match defect {
            "missing" => {
                records.pop();
            }
            "duplicate" => records[1] = records[0].clone(),
            "identity" => records[0]["agent_uid"] = serde_json::json!(AgentUid(u64::MAX)),
            "tick" => records[0]["created_at_tick"] = serde_json::json!(Tick(1)),
            "genome" => records[0]["envelope"]["schema_version"] = serde_json::json!(u32::MAX),
            _ => unreachable!(),
        }
        let changed: WorldHostCheckpointV1 = serde_json::from_value(changed).unwrap();
        assert!(changed.validate(&science).is_err(), "{defect}");
        assert!(
            WorldState::restore_host_checkpoint_v1(&science, &changed, registry()).is_err(),
            "{defect}"
        );
    }
    assert_eq!(
        serde_json::to_value(original.capture_host_continuity_v1(&science).unwrap()).unwrap(),
        json
    );
}

#[test]
fn completed_birth_history_survives_while_later_arrivals_still_block_capture() {
    for persistence_interval in [0, 1] {
        let (mut original, key) = world(persistence_interval, true);
        let founder = original.try_spawn_agent(AgentData::default()).unwrap();
        assert!(original.bind_agent_brain(founder, key).unwrap());
        assert!(
            original
                .try_update_agent_runtime(founder, |runtime| {
                    runtime.energy = 5.0;
                    runtime.reproduction_counter = f32::MAX;
                })
                .unwrap()
        );
        let mut session = original
            .bind_persistence(Box::new(NullPersistence))
            .unwrap();
        let completion = session.step_outcome(&mut original).unwrap();
        assert!(completion.fault.is_none());
        assert!(completion.outcome.summary.births > 0);
        session.admit_pending(&mut original).unwrap();
        let science = original.capture_checkpoint_quiescent().unwrap();
        let continuity = original.capture_host_continuity_v1(&science).unwrap();
        assert_eq!(
            serde_json::to_value(&continuity).unwrap()["history"][0]["births"],
            serde_json::json!(completion.outcome.summary.births)
        );
        let restored =
            WorldState::restore_host_checkpoint_v1(&science, &continuity, registry()).unwrap();
        assert_eq!(
            restored.world_digest_v1().unwrap(),
            original.world_digest_v1().unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&restored.capture_host_continuity_v1(&science).unwrap()).unwrap(),
            serde_json::to_vec(&continuity).unwrap()
        );
        if persistence_interval == 0 {
            original.try_inject_agent(AgentData::default()).unwrap();
            let later_science = original.capture_checkpoint_quiescent().unwrap();
            assert!(matches!(
                original.capture_host_continuity_v1(&later_science),
                Err(WorldCheckpointError::DeferredHostOutput {
                    field: "host.persistence_tail"
                })
            ));
        }
    }
}

#[test]
fn pending_replay_anchors_still_block_fresh_owner_capture() {
    for persistence_interval in [0, 1] {
        let (mut original, _) = world(persistence_interval, false);
        original.request_replay_world_digest();
        let science = original.capture_checkpoint_quiescent().unwrap();
        assert!(matches!(
            original.capture_host_continuity_v1(&science),
            Err(WorldCheckpointError::DeferredHostOutput {
                field: "host.persistence_tail"
            })
        ));
    }
}
