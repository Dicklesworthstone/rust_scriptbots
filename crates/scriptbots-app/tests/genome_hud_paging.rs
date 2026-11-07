//! Real-family owner-channel evidence for bounded genome HUD paging.

use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use scriptbots_brain::mlp::{MlpBrain, MlpBrainFamily};
use scriptbots_core::{
    AgentData, AgentUid, BrainFamilyAdapter, BrainGenomeDerivation, BrainProvenance, MutationRates,
    ScriptBotsConfig, SmallRngStream, Tick, WorldState, genome_browser::MutationDiffStatus,
};
use scriptbots_runtime::{
    CommandEnvelope, CommandId, FixedDeadlineHost, HostCommand, HostCore, HostCoreOptions,
    HostPort, HostSessionId, ManualInstant, PlaybackSnapshot,
    channel::{ChannelHostDriver, ChannelHostOptions, ChannelHostPort, INSPECTOR_GENOME_PAGE_SIZE},
};

struct Owner {
    port: ChannelHostPort,
    worker: Option<thread::JoinHandle<()>>,
}

impl Drop for Owner {
    fn drop(&mut self) {
        let _ = self.port.submit(CommandEnvelope::new(
            CommandId::from_client_sequence(991, 1),
            HostCommand::Shutdown,
        ));
        if let Some(worker) = self.worker.take() {
            let result = worker.join();
            if !thread::panicking() {
                assert!(result.is_ok(), "owner worker failed");
            }
        }
    }
}

#[test]
fn real_mlp_genome_slices_are_bounded_distinct_and_source_consistent() {
    let (send, receive) = mpsc::sync_channel(1);
    let worker = thread::spawn(move || {
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(918),
            world_width: 80,
            world_height: 80,
            food_cell_size: 40,
            population_minimum: 0,
            persistence_interval: 0,
            ..Default::default()
        })
        .unwrap();
        let family = world
            .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
            .unwrap();
        let agent = world.try_inject_agent(AgentData::default()).unwrap();
        assert!(world.bind_agent_brain(agent, family).unwrap());
        let uid = world.agent_uid(agent).unwrap();
        let core = HostCore::new(
            HostSessionId::new(991),
            world,
            HostCoreOptions {
                initial_playback: PlaybackSnapshot {
                    paused: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let (mut driver, port) = ChannelHostDriver::new(
            FixedDeadlineHost::new(core),
            ChannelHostOptions {
                maintenance_period: Duration::from_millis(1),
                ..Default::default()
            },
        )
        .unwrap();
        send.send((port, uid)).unwrap();
        let start = Instant::now();
        driver
            .run(move || {
                ManualInstant::from_nanos(u64::try_from(start.elapsed().as_nanos()).unwrap())
            })
            .unwrap();
    });
    let (port, uid) = receive
        .recv_timeout(Duration::from_secs(10))
        .expect("owner setup");
    let owner = Owner {
        port,
        worker: Some(worker),
    };
    let read = |offset| {
        owner
            .port
            .inspect_agent_for_hud_page_before(uid, offset, Instant::now() + Duration::from_secs(3))
            .unwrap()
            .unwrap()
    };
    let first = read(0);
    let second = read(INSPECTOR_GENOME_PAGE_SIZE);
    assert_eq!(first.source, second.source);
    let a = first.genome_browser.unwrap();
    let b = second.genome_browser.unwrap();
    assert!(a.paging.total_nodes > INSPECTOR_GENOME_PAGE_SIZE);
    assert_eq!(a.paging.page_offset, 0);
    assert_eq!(b.paging.page_offset, INSPECTOR_GENOME_PAGE_SIZE);
    assert_eq!(a.nodes.len(), INSPECTOR_GENOME_PAGE_SIZE);
    assert!(b.nodes.len() <= INSPECTOR_GENOME_PAGE_SIZE && !b.nodes.is_empty());
    assert_eq!(a.genome_digest, b.genome_digest);
    assert!(a.nodes.last().unwrap().node_index < b.nodes.first().unwrap().node_index);
    let final_offset =
        (a.paging.total_nodes - 1) / INSPECTOR_GENOME_PAGE_SIZE * INSPECTOR_GENOME_PAGE_SIZE;
    let last = read(final_offset).genome_browser.unwrap();
    assert_eq!(last.paging.page_offset, final_offset);
    assert_eq!(last.nodes.len(), a.paging.total_nodes - final_offset);
    assert!(!last.paging.is_truncated);
    let beyond = read(final_offset + INSPECTOR_GENOME_PAGE_SIZE)
        .genome_browser
        .unwrap();
    assert!(beyond.nodes.is_empty());
}

#[test]
fn real_mlp_parent_diff_and_missing_parent_are_reported_truthfully() {
    let (send, receive) = mpsc::sync_channel(1);
    let worker = thread::spawn(move || {
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(919),
            world_width: 80,
            world_height: 80,
            food_cell_size: 40,
            population_minimum: 0,
            persistence_interval: 0,
            ..Default::default()
        })
        .unwrap();
        let key = world
            .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
            .unwrap();
        let parent = world.try_inject_agent(AgentData::default()).unwrap();
        assert!(world.bind_agent_brain(parent, key).unwrap());
        let parent_uid = world.agent_uid(parent).unwrap();
        let parent_genome = world.agent_brain_genome(parent).unwrap().clone();
        let provenance = |uid, derivation| BrainProvenance {
            parents: [Some(uid), None],
            parent_genome_hashes: [Some(parent_genome.material_hash()), None],
            created_at: Tick(0),
            derivation,
        };
        let family = MlpBrainFamily::new();
        let mut rng = SmallRngStream::seed_from_u64(0x1234);
        let child_genome = family
            .mutate_genome(
                &parent_genome,
                MutationRates {
                    primary: 1.0,
                    secondary: 0.5,
                },
                provenance(parent_uid, BrainGenomeDerivation::MutationOnly),
                &mut rng,
            )
            .unwrap();
        let child = world.try_inject_agent(AgentData::default()).unwrap();
        assert!(world.bind_agent_brain_genome(child, &child_genome).unwrap());
        let child_uid = world.agent_uid(child).unwrap();
        let missing_uid = AgentUid(1_000_000);
        let orphan_genome = family
            .clone_genome(
                &parent_genome,
                provenance(missing_uid, BrainGenomeDerivation::Clone),
            )
            .unwrap();
        let orphan = world.try_inject_agent(AgentData::default()).unwrap();
        assert!(
            world
                .bind_agent_brain_genome(orphan, &orphan_genome)
                .unwrap()
        );
        let orphan_uid = world.agent_uid(orphan).unwrap();
        let rebound_parent = world.try_inject_agent(AgentData::default()).unwrap();
        assert!(world.bind_agent_brain(rebound_parent, key).unwrap());
        let rebound_uid = world.agent_uid(rebound_parent).unwrap();
        let original = world.agent_brain_genome(rebound_parent).unwrap().clone();
        let rebound_child_genome = family
            .clone_genome(
                &original,
                BrainProvenance {
                    parents: [Some(rebound_uid), None],
                    parent_genome_hashes: [Some(original.material_hash()), None],
                    created_at: Tick(0),
                    derivation: BrainGenomeDerivation::Clone,
                },
            )
            .unwrap();
        let rebound_child = world.try_inject_agent(AgentData::default()).unwrap();
        assert!(
            world
                .bind_agent_brain_genome(rebound_child, &rebound_child_genome)
                .unwrap()
        );
        let rebound_child_uid = world.agent_uid(rebound_child).unwrap();
        assert!(world.bind_agent_brain(rebound_parent, key).unwrap());
        assert_ne!(
            world
                .agent_brain_genome(rebound_parent)
                .unwrap()
                .material_hash(),
            original.material_hash()
        );
        let core = HostCore::new(
            HostSessionId::new(992),
            world,
            HostCoreOptions {
                initial_playback: PlaybackSnapshot {
                    paused: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let (mut driver, port) = ChannelHostDriver::new(
            FixedDeadlineHost::new(core),
            ChannelHostOptions {
                maintenance_period: Duration::from_millis(1),
                ..Default::default()
            },
        )
        .unwrap();
        send.send((port, parent_uid, child_uid, orphan_uid, rebound_child_uid))
            .unwrap();
        let start = Instant::now();
        driver
            .run(move || {
                ManualInstant::from_nanos(u64::try_from(start.elapsed().as_nanos()).unwrap())
            })
            .unwrap();
    });
    let (port, parent_uid, child_uid, orphan_uid, rebound_child_uid) = receive
        .recv_timeout(Duration::from_secs(10))
        .expect("owner setup");
    let owner = Owner {
        port,
        worker: Some(worker),
    };
    let read = |uid| {
        owner
            .port
            .inspect_agent_for_hud_page_before(uid, 0, Instant::now() + Duration::from_secs(3))
            .unwrap()
            .unwrap()
            .genome_browser
            .unwrap()
    };
    let child = read(child_uid);
    match child.mutation_diff {
        MutationDiffStatus::Computed {
            parent_uid: observed,
            total_deltas,
            ..
        } => {
            assert_eq!(observed, parent_uid);
            assert!(total_deltas > 0);
            assert_eq!(child.deltas.len(), total_deltas);
        }
        other => panic!("expected real parent diff: {other:?}"),
    }
    let orphan = read(orphan_uid);
    assert!(
        matches!(orphan.mutation_diff, MutationDiffStatus::Unavailable { ref reason } if reason == "primary parent genome is unavailable")
    );
    assert!(orphan.deltas.is_empty());
    let rebound_child = read(rebound_child_uid);
    assert!(
        matches!(rebound_child.mutation_diff, MutationDiffStatus::Unavailable { ref reason } if reason == "primary parent genome does not match recorded material")
    );
    assert!(rebound_child.deltas.is_empty());
}
