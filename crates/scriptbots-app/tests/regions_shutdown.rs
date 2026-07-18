//! Structured-shutdown integration test (bd-2z0.4.13): real services, ordered
//! budgeted teardown, per-region outcomes, and durable-watermark proof.

use asupersync::types::{Budget, Outcome};
use scriptbots_app::{
    ControlServerConfig, McpTransportConfig, SharedWorld,
    control::empty_latest_summary,
    regions::{AppRoot, ServiceRegion},
    servers::ControlRuntime,
};
use scriptbots_core::{ScriptBotsConfig, WorldState};
use scriptbots_storage::StorageReader;
use std::{fs, sync::Mutex, time::SystemTime};

fn temp_db(tag: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_nanos();
    std::env::temp_dir()
        .join(format!(
            "scriptbots-regions-{tag}-{}-{nanos}.sqlite",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

fn cleanup(path: &str) {
    let _ = fs::remove_file(path);
    for suffix in ["-wal", "-shm", "-journal", "-wal-fec", ".lock", "-lock"] {
        let _ = fs::remove_file(format!("{path}{suffix}"));
    }
}

#[test]
fn structured_shutdown_reports_every_region_outcome_and_drains_storage() {
    let db = temp_db("shutdown");
    let mut pipeline =
        scriptbots_storage::StoragePipeline::create_unattributed_file_with_thresholds(
            &db, 1, 1, 1, 1,
        )
        .expect("file-backed storage pipeline");

    let config = ScriptBotsConfig {
        rng_seed: Some(7),
        persistence_interval: 1,
        ..ScriptBotsConfig::default()
    };
    let (mut world, mut persistence) =
        WorldState::with_persistence(config, Box::new(pipeline.sink()))
            .expect("world with persistence");
    // Inject load: one science tick admits a real batch through the outbox.
    persistence.step(&mut world).expect("science tick");

    let world_shared: SharedWorld = std::sync::Arc::new(Mutex::new(world));
    let control_config = ControlServerConfig {
        rest_address: "127.0.0.1:0".parse().expect("ephemeral REST address"),
        mcp_transport: McpTransportConfig::Http {
            bind_address: "127.0.0.1:0".parse().expect("ephemeral MCP address"),
        },
        ..ControlServerConfig::default()
    };
    let (control_runtime, _drain, _submit) =
        ControlRuntime::launch(world_shared.clone(), empty_latest_summary(), control_config)
            .expect("control runtime launches with every listener bound");

    let persistence_shared = std::sync::Arc::new(Mutex::new(persistence));
    let mut root = AppRoot::new();
    let world_for_storage = std::sync::Arc::clone(&world_shared);
    let persistence_for_storage = std::sync::Arc::clone(&persistence_shared);
    root.register(ServiceRegion::new(
        "storage-pipeline",
        Budget::new().with_deadline_at_secs(30),
        move |_budget| {
            let finalize = (|| -> anyhow::Result<()> {
                let mut world = world_for_storage
                    .lock()
                    .map_err(|error| anyhow::anyhow!("world mutex poisoned: {error}"))?;
                persistence_for_storage
                    .lock()
                    .map_err(|error| anyhow::anyhow!("persistence mutex poisoned: {error}"))?
                    .finalize(&mut world)?;
                Ok(())
            })();
            match finalize.and_then(|()| {
                pipeline
                    .shutdown()
                    .map(|_receipt| ())
                    .map_err(|error| anyhow::anyhow!("{error:#}"))
            }) {
                Ok(()) => Outcome::ok("storage drained to the durable watermark".to_owned()),
                Err(error) => Outcome::Err(format!("{error:#}")),
            }
        },
    ));
    root.register(ServiceRegion::new(
        "control-server",
        Budget::new().with_deadline_at_secs(15),
        move |_budget| match control_runtime.shutdown() {
            Ok(()) => Outcome::ok("control runtime shut down".to_owned()),
            Err(error) => Outcome::Err(format!("{error:#}")),
        },
    ));

    let outcomes = root.close();
    assert_eq!(
        outcomes.len(),
        2,
        "every registered region must report an outcome"
    );
    assert_eq!(
        outcomes[0].name, "control-server",
        "control closes before storage drains"
    );
    assert_eq!(outcomes[1].name, "storage-pipeline");
    assert!(
        matches!(outcomes[0].outcome, Outcome::Ok(_)),
        "control region must close cleanly: {:?}",
        outcomes[0].outcome
    );
    assert!(
        matches!(outcomes[1].outcome, Outcome::Ok(_)),
        "storage region must drain cleanly: {:?}",
        outcomes[1].outcome
    );

    // The durable watermark must equal the last admitted batch after teardown.
    let reader = StorageReader::open(&db).expect("open run database after teardown");
    let watermarks = reader
        .persistence_watermarks()
        .expect("watermark query");
    assert!(
        watermarks.admitted.is_some()
            && watermarks.admitted == watermarks.applied
            && watermarks.applied == watermarks.durable,
        "teardown must converge admitted/applied/durable watermarks: {watermarks:?}"
    );
    assert_eq!(
        reader.max_tick().expect("max tick"),
        Some(1),
        "the admitted science tick must be durably committed"
    );
    reader.close().expect("reader closes");
    cleanup(&db);
}
