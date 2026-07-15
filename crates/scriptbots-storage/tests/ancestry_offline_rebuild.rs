//! The run database must be sufficient to reconstruct a run's ancestry offline.
//!
//! This is the test that justifies the storage layer. If a graph rebuilt from
//! nothing but the persisted rows is not IDENTICAL to the one the live run held,
//! then every claim that a run can be analysed after the fact is a claim we
//! cannot back — and a phylogeny rendered from that database would be a
//! plausible-looking lie.

use scriptbots_core::ancestry::AncestryGraph;
use scriptbots_core::{
    AgentData, BirthOrigin, PersistenceAdmissionError, PersistenceBatch, ScientificStateError,
    ScriptBotsConfig, WorldPersistence, WorldState,
};
use scriptbots_storage::{StoragePipeline, StorageReader, rebuild_ancestry};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// A sink that records exactly what storage successfully admitted.
///
/// Admission happens first so a definite rejection cannot mutate the live oracle
/// and then duplicate its rows when the admission session retries the retained batch.
/// If the test recorded a different stream from storage, a mismatch between live
/// and rebuilt could be blamed on the test rather than on the database.
struct TeeSink {
    inner: Box<dyn WorldPersistence>,
    seen: Arc<Mutex<AncestryGraph>>,
}

impl WorldPersistence for TeeSink {
    fn on_tick(&mut self, payload: &PersistenceBatch) -> Result<(), PersistenceAdmissionError> {
        self.inner.on_tick(payload)?;
        {
            let mut graph = self.seen.lock().expect("tee graph lock");
            for birth in &payload.births {
                graph.apply_birth(birth).expect("live birth is well formed");
            }
            for death in &payload.deaths {
                graph.apply_death(death).expect("live death is well formed");
            }
        }
        Ok(())
    }
}

fn temp_db(label: &str) -> String {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir()
        .join(format!(
            "scriptbots_ancestry_{label}_{}_{nonce}.sqlite",
            std::process::id()
        ))
        .to_str()
        .expect("utf8 path")
        .to_owned()
}

#[test]
fn the_run_database_alone_rebuilds_the_identical_ancestry_graph() {
    let path = temp_db("rebuild");

    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");

    let config = ScriptBotsConfig {
        world_width: 400,
        world_height: 400,
        food_cell_size: 20,
        initial_food: 0.4,
        food_max: 1.0,
        persistence_interval: 1,
        rng_seed: Some(4242),
        ..ScriptBotsConfig::default()
    };

    // Run a world long enough that agents are actually born and actually die —
    // a rebuild proven only on an empty graph proves nothing at all.
    let seen = Arc::new(Mutex::new(AncestryGraph::new()));
    let identity_sequence = {
        let tee = TeeSink {
            inner: Box::new(pipeline.sink()),
            seen: Arc::clone(&seen),
        };
        let (mut world, mut persistence) =
            WorldState::with_persistence(config, Box::new(tee)).expect("world");

        // Every agent that enters the world now emits an origin record. The tee
        // must observe these roots from the production stream; synthesizing them
        // in the test would conceal the exact defect this test guards against.
        for _ in 0..16 {
            world
                .try_spawn_agent(AgentData::default())
                .expect("seed agent");
        }

        for _ in 0..600 {
            persistence.step(&mut world).expect("step");
        }
        world.identity_sequence_state()
    };
    let live_graph = seen.lock().expect("tee graph lock").clone();
    let allocated_agent_count = usize::try_from(identity_sequence.1)
        .expect("the allocated-agent count fits in usize on the test host");
    assert_eq!(
        identity_sequence.0,
        identity_sequence.1 + 1,
        "uid and spawn-ordinal identity counters diverged"
    );

    let shutdown = pipeline.shutdown().expect("durable shutdown");
    assert!(
        shutdown.committed_tick.is_some(),
        "the run must have committed something to disk, or this test is vacuous"
    );

    // A rebuild that found nothing would trivially "match" an empty live graph.
    // Refuse to draw any conclusion from an empty run.
    assert!(
        live_graph.len() > 16,
        "the run produced no births at all ({} nodes) — an offline-rebuild test \
         over an empty graph proves nothing",
        live_graph.len()
    );

    // Reopen the database READ-ONLY and rebuild from the rows alone. No world, no
    // in-memory state, nothing but what was persisted — which is precisely the
    // situation an offline analyst is in.
    let reader = StorageReader::open(&path).expect("reopen the run database");
    let births = reader.load_ancestry_births().expect("load births");
    let deaths = reader.load_ancestry_deaths().expect("load deaths");
    assert!(
        !births.is_empty(),
        "the persisted origin stream is empty — an empty rebuild proves nothing"
    );
    assert!(
        births.iter().any(|birth| birth.origin == BirthOrigin::Born),
        "the run produced no demographic birth, so parent-edge reconstruction was not exercised"
    );
    assert!(
        births
            .iter()
            .any(|birth| birth.origin == BirthOrigin::Seeded),
        "the run persisted no seeded root, so bootstrap completeness was not exercised"
    );
    assert!(
        births
            .iter()
            .any(|birth| birth.origin == BirthOrigin::Injected),
        "the run persisted no injected root, so population-policy completeness was not exercised"
    );
    assert!(
        !deaths.is_empty(),
        "the run produced no death, so death replay was not exercised"
    );
    assert_eq!(
        births.len(),
        allocated_agent_count,
        "durable origin-row count differs from the core identity allocator's independent count"
    );
    assert_eq!(
        births.len(),
        live_graph.len(),
        "the database must contain exactly one origin row for every agent the live stream saw"
    );
    let rebuilt = rebuild_ancestry(&births, &deaths).expect("the persisted log is well formed");

    // THE ACCEPTANCE CRITERION: a graph rebuilt from nothing but persisted origin
    // and death rows is exactly the graph the live run held. There is no agents-
    // table founder inference and no test-only repair path.
    assert!(
        rebuilt.len() > 16,
        "the complete stream never advanced beyond its initial roots ({} nodes)",
        rebuilt.len()
    );
    assert_eq!(
        rebuilt.len(),
        live_graph.len(),
        "the rebuilt graph has a different node count from the live graph"
    );
    assert_eq!(
        rebuilt.canonical_digest(),
        live_graph.canonical_digest(),
        "the graph rebuilt from the run database differs from the live graph"
    );

    // And it is DETERMINISTIC: two rebuilds of the same database agree bit for
    // bit. That is the property an offline analyst actually depends on.
    let again = rebuild_ancestry(&births, &deaths).expect("second rebuild");
    assert_eq!(
        again.canonical_digest(),
        rebuilt.canonical_digest(),
        "two rebuilds of the same database disagreed — the run database is not a \
         reliable basis for offline science"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn tick_zero_origins_finalize_once_and_seal_the_real_storage_boundary() {
    let path = temp_db("tick-zero-origins");
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");
    let (mut world, mut persistence) = WorldState::with_persistence(
        ScriptBotsConfig {
            persistence_interval: 1,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(0x71C0_0A10),
            ..ScriptBotsConfig::default()
        },
        Box::new(pipeline.sink()),
    )
    .expect("world");

    world
        .try_spawn_agent(AgentData::default())
        .expect("tick-zero founder");
    assert!(
        persistence
            .finalize(&mut world)
            .expect("tick-zero admission")
    );
    assert!(
        !persistence
            .finalize(&mut world)
            .expect("idempotent tick-zero finalization")
    );

    let callback_ran = std::cell::Cell::new(false);
    let error = world
        .try_inject_agent_with(AgentData::default(), |_| callback_ran.set(true))
        .expect_err("an admitted tick-zero boundary must reject a conflicting arrival");
    assert_eq!(
        error,
        ScientificStateError::PersistenceBoundarySealed {
            path: "agent".to_owned(),
            tick: 0,
        }
    );
    assert!(!callback_ran.get(), "sealed ingress invoked its callback");
    drop(world);
    drop(persistence);

    let shutdown = pipeline.shutdown().expect("durable shutdown");
    assert_eq!(shutdown.committed_tick, Some(0));
    let reader = StorageReader::open(&path).expect("reopen tick-zero run");
    let births = reader
        .load_ancestry_births()
        .expect("load tick-zero origins");
    assert_eq!(births.len(), 1);
    assert_eq!(births[0].tick.0, 0);
    assert_eq!(births[0].origin, BirthOrigin::Seeded);
    reader.close().expect("close reader");
}

#[test]
fn a_rebuild_from_an_empty_database_is_empty_rather_than_wrong() {
    // The degenerate case must be honest: no rows means no graph, not a graph of
    // invented roots.
    let path = temp_db("empty");
    let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(&path, 1, 1, 1, 1)
        .expect("pipeline");
    pipeline.shutdown().expect("shutdown");

    let reader = StorageReader::open(&path).expect("reopen");
    let births = reader.load_ancestry_births().expect("load births");
    let deaths = reader.load_ancestry_deaths().expect("load deaths");
    assert!(births.is_empty() && deaths.is_empty());

    let rebuilt = rebuild_ancestry(&births, &deaths).expect("an empty log is a valid log");
    assert!(rebuilt.is_empty());
    assert_eq!(
        rebuilt.canonical_digest(),
        AncestryGraph::new().canonical_digest()
    );

    let _ = std::fs::remove_file(&path);
}
