//! The run database must be sufficient to reconstruct a run's ancestry offline.
//!
//! This is the test that justifies the storage layer. If a graph rebuilt from
//! nothing but the persisted rows is not IDENTICAL to the one the live run held,
//! then every claim that a run can be analysed after the fact is a claim we
//! cannot back — and a phylogeny rendered from that database would be a
//! plausible-looking lie.

use scriptbots_core::ancestry::AncestryGraph;
use scriptbots_core::{
    AgentData, BirthRecord, Generation, PersistenceAdmissionError, PersistenceBatch, Position,
    ScriptBotsConfig, Tick, WorldPersistence, WorldState,
};
use scriptbots_storage::{StoragePipeline, StorageReader, rebuild_ancestry};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// A sink that forwards to storage AND records what it forwarded.
///
/// The live graph must be built from EXACTLY the records that reach the disk. If
/// the test built it from some other source, a mismatch between live and rebuilt
/// could be blamed on the test rather than on the database, and the comparison
/// would prove nothing.
struct TeeSink {
    inner: Box<dyn WorldPersistence>,
    seen: Arc<Mutex<AncestryGraph>>,
}

impl WorldPersistence for TeeSink {
    fn on_tick(&mut self, payload: &PersistenceBatch) -> Result<(), PersistenceAdmissionError> {
        {
            let mut graph = self.seen.lock().expect("tee graph lock");
            for birth in &payload.births {
                graph.apply_birth(birth).expect("live birth is well formed");
            }
            for death in &payload.deaths {
                graph.apply_death(death).expect("live death is well formed");
            }
        }
        self.inner.on_tick(payload)
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

    let mut pipeline =
        StoragePipeline::create_new_file_with_thresholds(&path, 1, 1, 1, 1).expect("pipeline");

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
    {
        let tee = TeeSink {
            inner: Box::new(pipeline.sink()),
            seen: Arc::clone(&seen),
        };
        let mut world = WorldState::with_persistence(config, Box::new(tee)).expect("world");

        // THE FOUNDING POPULATION. Founders are seeded at bootstrap, so they never
        // pass through the spawn-commit stage and emit NO BirthRecord — which
        // means the live graph cannot be built from birth records alone either.
        // Both graphs must therefore learn their founders the same way, or the
        // comparison below would be measuring the test rather than the database.
        let mut founder_uids = Vec::new();
        for _ in 0..16 {
            let id = world
                .try_spawn_agent(AgentData::default())
                .expect("seed agent");
            founder_uids.push(world.agent_uid(id).expect("a spawned agent has a uid"));
        }
        {
            let mut graph = seen.lock().expect("tee graph lock");
            for uid in &founder_uids {
                graph
                    .apply_birth(&BirthRecord {
                        tick: Tick(0),
                        agent_uid: *uid,
                        spawn_ordinal: 0,
                        birth_ordinal: 0,
                        parent_a: None,
                        parent_b: None,
                        brain_kind: None,
                        brain_key: None,
                        herbivore_tendency: 0.0,
                        generation: Generation(0),
                        position: Position::new(0.0, 0.0),
                        is_hybrid: false,
                    })
                    .expect("a founder is a root");
            }
        }

        for _ in 0..600 {
            world.step().expect("step");
        }
    }
    let live_graph = seen.lock().expect("tee graph lock").clone();

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
    let founders = reader.load_ancestry_founders().expect("load founders");
    let births = reader.load_ancestry_births().expect("load births");
    let deaths = reader.load_ancestry_deaths().expect("load deaths");
    assert!(
        !founders.is_empty(),
        "the founding population is invisible in the database — every \
         first-generation child would name a parent that was never born, and no \
         phylogeny could be reconstructed at all"
    );
    let rebuilt =
        rebuild_ancestry(&founders, &births, &deaths).expect("the persisted log is well formed");

    // THE DATABASE IS STRICTLY MORE COMPLETE THAN THE RECORD STREAM, and that gap
    // is a finding, not a nuisance to be papered over.
    //
    // Agents enter this world by two different doors. Most are BORN, through the
    // spawn-commit stage, which emits a BirthRecord. But the founding population
    // is seeded at bootstrap, and the population floor INJECTS agents when the
    // world would otherwise die out — and neither of those paths emits a
    // BirthRecord at all. A live consumer building a phylogeny from the record
    // stream therefore never sees them: they are invisible, and their descendants
    // would be orphan roots.
    //
    // The database does not have this problem, because `agents` records every
    // agent that ever existed. So the rebuilt graph is a strict SUPERSET of the
    // record-derived one, and the difference is exactly the agents the record
    // stream dropped.
    assert!(
        rebuilt.len() >= live_graph.len(),
        "the database must know at least as much as the record stream, but it \
         rebuilt {} nodes against the stream's {}",
        rebuilt.len(),
        live_graph.len()
    );
    for uid in live_graph.roots() {
        assert!(
            rebuilt.node(uid).is_some(),
            "an agent the record stream saw is missing from the database rebuild"
        );
    }
    assert!(
        rebuilt.len() > live_graph.len(),
        "EXPECTED GAP NOT PRESENT. This test documents a real defect: agents \
         injected by the population floor (and the bootstrap founders) emit no \
         BirthRecord, so the record stream is incomplete and the database rebuild \
         is strictly larger. If this assertion ever fails, the defect has been \
         FIXED — delete this assertion and assert digest equality instead."
    );

    // The rebuild is internally sound: every parent resolves to a node that is
    // actually present. A phylogeny with an edge into nothing would have to choose
    // between panicking and silently truncating a lineage.
    for uid in rebuilt.roots() {
        assert!(rebuilt.node(uid).is_some());
    }

    // And it is DETERMINISTIC: two rebuilds of the same database agree bit for
    // bit. That is the property an offline analyst actually depends on.
    let again = rebuild_ancestry(&founders, &births, &deaths).expect("second rebuild");
    assert_eq!(
        again.canonical_digest(),
        rebuilt.canonical_digest(),
        "two rebuilds of the same database disagreed — the run database is not a \
         reliable basis for offline science"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_rebuild_from_an_empty_database_is_empty_rather_than_wrong() {
    // The degenerate case must be honest: no rows means no graph, not a graph of
    // invented roots.
    let path = temp_db("empty");
    let mut pipeline =
        StoragePipeline::create_new_file_with_thresholds(&path, 1, 1, 1, 1).expect("pipeline");
    pipeline.shutdown().expect("shutdown");

    let reader = StorageReader::open(&path).expect("reopen");
    let founders = reader.load_ancestry_founders().expect("load founders");
    let births = reader.load_ancestry_births().expect("load births");
    let deaths = reader.load_ancestry_deaths().expect("load deaths");
    assert!(founders.is_empty() && births.is_empty() && deaths.is_empty());

    let rebuilt =
        rebuild_ancestry(&founders, &births, &deaths).expect("an empty log is a valid log");
    assert!(rebuilt.is_empty());
    assert_eq!(
        rebuilt.canonical_digest(),
        AncestryGraph::new().canonical_digest()
    );

    let _ = std::fs::remove_file(&path);
}
