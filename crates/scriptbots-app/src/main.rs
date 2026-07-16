use anyhow::{Context, Result, bail};
use clap::{ArgAction, Parser, ValueEnum};
use owo_colors::OwoColorize;
use ron::ser::PrettyConfig as RonPrettyConfig;
use scriptbots_app::{
    BootstrapEvidenceV0, CharacterizationTraceV2, ControlServerConfig, ControlServerReservation,
    RunIdentityV1, RunManifestV3, ScenarioIdentityV0, SharedAnalytics, SharedWorld, ThreadPolicyV0,
    WorldStepDriver,
    precedence::{
        ConfigFieldOverride, ConfigLayerKind, ConfigLayerStatement, ThreadPolicy, ThreadSource,
        canonical_layer_bytes, resolve_config_layers, resolve_thread_policy,
    },
    renderer::{Renderer, RendererContext},
    terminal::TerminalRenderer,
};
#[cfg(feature = "bevy_render")]
use scriptbots_bevy::{BevyRendererContext, render_png_offscreen as render_bevy_png};
use scriptbots_brain::{
    AssemblyBrain, DwraonBrain, MlpBrain, assembly::AssemblyFamilyAdapter,
    dwraon::DwraonFamilyAdapter, mlp::MlpBrainFamily,
};
#[cfg(feature = "brain-ft")]
use scriptbots_brain_ml::{FT_BRAIN_KIND, FtBrainFamily};
use scriptbots_core::{
    AgentData, NeuroflowActivationKind, NullPersistence, PersistenceAdmissionSession,
    PersistenceSessionError, RenderTonemapMode, ReplayEventKind, ScriptBotsConfig, TickSummary,
    WorldDigestV1, WorldPersistence, WorldState,
};
#[cfg(feature = "gui")]
use scriptbots_render::{render_png_offscreen, run_demo};
use scriptbots_runtime::RunId;
use scriptbots_storage::{
    PersistedReplayEvent, PersistenceGuarantee, ShutdownReceipt, StoragePipeline, StorageReader,
};
use serde_json::{self, Value as JsonValue};
use std::process::{Command, Stdio};
use std::{
    collections::HashMap,
    env, fmt, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tracing::{debug, info, warn};

#[cfg(feature = "fast-alloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DEFAULT_BOOTSTRAP_TICKS: u64 = 0;
const LIVE_RUN_POLICY: &str = "operator-controlled-until-stop-v1";

type SharedPersistenceAdmission = Arc<Mutex<PersistenceAdmissionSession>>;

#[derive(Clone, Copy)]
struct SenseRunSummary {
    tick: u64,
    saturations_total: u64,
}

impl SenseRunSummary {
    fn capture(world: &WorldState) -> Self {
        Self {
            tick: world.tick().0,
            saturations_total: world.sense_saturations_total(),
        }
    }
}

fn emit_sense_startup_contract() {
    info!(
        target: "scriptbots::sense",
        sense_kernel = "fixed_point",
        frac_bits = scriptbots_core::sense_fixed::SENSE_FRAC_BITS,
        max_neighbors_assumed = scriptbots_core::sense_fixed::MAX_NEIGHBORS_ASSUMED,
        headroom_bits = scriptbots_core::sense_fixed::SENSE_HEADROOM_BITS,
        geometry = scriptbots_core::sense_fixed::SENSE_GEOMETRY,
        poly_max_err = scriptbots_core::sense_fixed::ACOS_MAX_ERROR,
        "sense numeric contract"
    );
}

fn emit_sense_run_end(summary: SenseRunSummary, completed: bool) {
    let suspect = summary.saturations_total != 0;
    let status = if completed { "completed" } else { "error" };
    if suspect || !completed {
        warn!(
            target: "scriptbots::sense",
            tick = summary.tick,
            saturations_total = summary.saturations_total,
            suspect,
            status,
            "sense run ended"
        );
    } else {
        info!(
            target: "scriptbots::sense",
            tick = summary.tick,
            saturations_total = summary.saturations_total,
            suspect,
            status,
            "sense run ended"
        );
    }
}

fn capture_shared_sense_run_summary(world: &SharedWorld) -> SenseRunSummary {
    match world.lock() {
        Ok(world) => SenseRunSummary::capture(&world),
        Err(poisoned) => SenseRunSummary::capture(&poisoned.into_inner()),
    }
}

fn persistence_step_driver(
    world: &SharedWorld,
    session: &SharedPersistenceAdmission,
) -> WorldStepDriver {
    let world = Arc::clone(world);
    let session = Arc::clone(session);
    Arc::new(move || {
        let mut world = world
            .lock()
            .map_err(|error| PersistenceSessionError::Unavailable {
                detail: format!("world mutex poisoned while stepping: {error}"),
            })?;
        let mut session = session
            .lock()
            .map_err(|error| PersistenceSessionError::Unavailable {
                detail: format!("session mutex poisoned while stepping: {error}"),
            })?;
        session.step(&mut world)
    })
}

fn main() -> Result<()> {
    // FIRST, before anything can mutate the process environment: pin the launch
    // environment so build provenance records what the user exported, not what
    // startup's thread-policy set_var smeared over it (bd-3p7i).
    let _launch_environment = scriptbots_app::LaunchEnvironmentV0::pin();
    let cli = AppCli::parse();
    init_tracing();

    if let Some(path) = cli.recover_storage.as_deref() {
        recover_storage(path)?;
        return Ok(());
    }

    // Determinism check child mode: run headless and emit JSON, then exit.
    if let Ok(flag) = env::var("SCRIPTBOTS_DET_RUN")
        && matches!(parse_bool(&flag), Some(true))
    {
        let ticks_env = env::var("SCRIPTBOTS_DET_TICKS").ok();
        let tick_limit = ticks_env.and_then(|s| s.parse::<u64>().ok()).unwrap_or(500);
        let config = compose_config(&cli)?;
        run_det_child(&config, tick_limit, cli.brain)?;
        return Ok(());
    }
    let (config, launch_scenario, config_overrides) = compose_config_with_scenario(&cli)?;

    if let Some(ticks) = cli.characterize_v0 {
        run_characterization_v0(&cli, config, launch_scenario, config_overrides, ticks)?;
        return Ok(());
    }

    // Validate any path that will eventually create a world/storage runtime
    // before configuration writes, auto-tuning sweeps, thread-priority
    // changes, or database reservation. Explicitly non-interactive commands
    // do not need a renderer and retain their independent contracts.
    let storage_owning_startup = storage_owning_startup_requested(&cli);
    let resolved_renderer = if storage_owning_startup {
        preflight_renderer_startup(&cli)?
    } else {
        None
    };
    let control_reservation = if resolved_renderer.is_some() {
        let control_config = ControlServerConfig::try_from_env()?;
        Some(ControlServerReservation::prepare(control_config)?)
    } else {
        None
    };

    if let Some(outcome) = maybe_emit_config(&cli, &config)?
        && matches!(outcome, ConfigEmitOutcome::Exit)
    {
        return Ok(());
    }

    if cli.replay_db.is_some() {
        run_replay_cli(&cli, &config)?;
        return Ok(());
    }

    if let Some(ticks) = cli.det_check {
        run_det_check(&cli, ticks)?;
        return Ok(());
    }

    // Optional: profiling runs (headless). Execute and exit if specified.
    let thresholds = thresholds_from_cli(&cli);
    if cli.profile_steps.is_some() || cli.profile_storage_steps.is_some() {
        let env_threads = std::env::var("SCRIPTBOTS_MAX_THREADS")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|threads| *threads > 0);
        let profile_thread_policy =
            resolve_thread_policy(cli.threads, env_threads, None, cli.low_power);
        if profile_thread_policy.source != ThreadSource::Environment
            && let Some(threads) = profile_thread_policy.threads
        {
            // SAFETY: Profile dispatch happens before constructing a world or starting worker
            // threads, so the resolved cap cannot race an environment reader.
            unsafe {
                std::env::set_var("SCRIPTBOTS_MAX_THREADS", threads.to_string());
            }
        }
        if let Some(ticks) = cli.profile_steps {
            profile_world_steps(&config, ticks, cli.brain)?;
        }
        if let Some(ticks) = cli.profile_storage_steps {
            profile_world_steps_with_storage(
                &config,
                ticks,
                cli.brain,
                cli.storage,
                thresholds,
                profile_thread_policy,
                launch_scenario.clone(),
                config_overrides.clone(),
            )?;
        }
        return Ok(());
    }

    // Automated profiling sweep (child-process based)
    if let Some(ticks) = cli.profile_sweep {
        run_profile_sweep(&config, ticks, &cli)?;
        return Ok(());
    }

    if !storage_owning_startup {
        bail!("internal error: non-interactive command reached runtime startup");
    }

    // Auto-tune: run a quick sweep for the chosen storage mode, apply best settings, then continue.
    let mut thresholds = thresholds;
    let mut auto_tune_threads: Option<usize> = None;
    if let Some(ticks) = cli.auto_tune
        && let Some(best) = pick_best_for_storage(
            &config,
            ticks,
            cli.brain,
            cli.storage,
            cli.threads,
            cli.low_power,
        )?
    {
        // The probe RECOMMENDS; it does not decide. Whether the recommendation is
        // taken is settled by resolve_thread_policy below, which is what keeps a
        // probe child's tuning from leaking into a run whose operator had already
        // made the decision themselves.
        auto_tune_threads = Some(best.threads);
        // Apply thresholds if not provided via CLI
        if cli.storage_thresholds.is_none() {
            thresholds = ThresholdsOverride {
                tick: Some(best.tick),
                agent: Some(best.agent),
                event: Some(best.event),
                metric: Some(best.metric),
            };
        }
        println!(
            "{} Auto-tune selected: threads={} storage={} thresholds={},{},{},{} ({:.0} tps)",
            "✔".green().bold(),
            best.threads,
            match cli.storage {
                StorageMode::File => "file",
                StorageMode::Memory => "memory",
            },
            best.tick,
            best.agent,
            best.event,
            best.metric,
            best.tps
        );
    }

    // Configure the thread budget before world creation so the Rayon pool is
    // capped. Precedence is resolved in ONE place, by a pure function, rather than
    // by whichever branch of a chain happened to run last — see
    // `scriptbots_app::precedence`. The bug that arrangement used to hide: a user
    // who exported SCRIPTBOTS_MAX_THREADS=16 and passed --low-power had their
    // explicit value silently replaced with 2.
    let env_threads = std::env::var("SCRIPTBOTS_MAX_THREADS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|threads| *threads > 0);
    let policy = resolve_thread_policy(cli.threads, env_threads, auto_tune_threads, cli.low_power);
    tracing::info!(
        threads = ?policy.threads,
        source = %policy.source,
        overridden = ?policy.overridden.map(|source| source.to_string()),
        "resolved thread policy"
    );
    if let Some(overridden) = policy.overridden {
        // Visible, not silent. This is the correct outcome of the precedence
        // rules, but the user must learn it from the program rather than from a
        // surprising performance number.
        tracing::info!(
            winner = %policy.source,
            declined = %overridden,
            "a less specific configuration layer was declined"
        );
    }
    // Only write the variable when a layer other than the environment decided;
    // rewriting it with the value it already holds is noise, and rewriting it
    // with a DIFFERENT value is the clobber this whole module exists to prevent.
    if policy.source != ThreadSource::Environment
        && let Some(threads) = policy.threads
    {
        unsafe {
            std::env::set_var("SCRIPTBOTS_MAX_THREADS", threads.to_string());
        }
    }

    // Apply OS-level priority niceness where supported.
    apply_process_niceness(cli.low_power)?;

    // Prefer high-performance adapter on Windows for wgpu
    #[cfg(windows)]
    unsafe {
        if std::env::var("WGPU_POWER_PREFERENCE").is_err() {
            std::env::set_var("WGPU_POWER_PREFERENCE", "high_performance");
        }
    }

    // Renderer debug toggles
    if cli.debug_watermark {
        unsafe {
            std::env::set_var("SCRIPTBOTS_RENDER_WATERMARK", "1");
        }
    }
    if cli.renderer_safe || cli.low_power {
        unsafe {
            std::env::set_var("SCRIPTBOTS_RENDER_SAFE", "1");
        }
    }
    let (world, persistence, analytics, mut storage_pipeline) = bootstrap_world(
        config,
        cli.brain,
        cli.storage,
        thresholds,
        cli.bootstrap_ticks,
        policy,
        launch_scenario,
        config_overrides,
    )?;
    let simulation_step = persistence_step_driver(&world, &persistence);

    // Capture every ordinary post-bootstrap exit so the exact retained tail is
    // finalized and the worker is acknowledged before this function returns.
    let runtime_result = (|| -> Result<()> {
        // Optional: dump a PNG snapshot and exit (no UI launched).
        if let Some(path) = cli.dump_png.as_ref() {
            #[cfg(feature = "gui")]
            {
                let (w, h) = cli
                    .png_size
                    .as_deref()
                    .and_then(parse_png_size)
                    .unwrap_or((1600, 900));

                let bytes = {
                    let guard = world.lock().map_err(|error| {
                        anyhow::anyhow!("world mutex poisoned while rendering PNG: {error}")
                    })?;
                    // Prefer wgpu compositor path if requested via env; otherwise fallback CPU raster
                    if matches!(
                        std::env::var("SB_WGPU_DUMP").ok().as_deref(),
                        Some("1" | "true" | "yes" | "on")
                    ) {
                        scriptbots_render::world_compositor::render_wgpu_png_offscreen(&guard, w, h)
                    } else {
                        render_png_offscreen(&guard, w, h)
                    }
                };
                if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                    fs::create_dir_all(parent)?;
                }
                fs::write(path, &bytes)?;
                println!(
                    "{} Wrote snapshot {} ({}x{})",
                    "✔".green().bold(),
                    path.display(),
                    w,
                    h
                );
                return Ok(());
            }
            #[cfg(not(feature = "gui"))]
            {
                // Avoid unused-variable warning when GUI is not enabled.
                let _ = path;
                bail!("--dump-png requires GUI feature; recompile with --features gui");
            }
        }
        #[cfg(feature = "bevy_render")]
        if let Some(path) = cli.dump_bevy_png.as_ref() {
            let (w, h) = cli
                .png_size
                .as_deref()
                .and_then(parse_png_size)
                .unwrap_or((1600, 900));
            let bytes = {
                let guard = world.lock().map_err(|error| {
                    anyhow::anyhow!("world mutex poisoned while rendering Bevy PNG: {error}")
                })?;
                render_bevy_png(&guard, w, h)?
            };
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, &bytes)?;
            println!(
                "{} Wrote Bevy snapshot {} ({}x{})",
                "✔".green().bold(),
                path.display(),
                w,
                h
            );
            return Ok(());
        }

        let (active_mode, renderer) = resolved_renderer.ok_or_else(|| {
            anyhow::anyhow!("interactive renderer was not resolved before startup")
        })?;
        let control_reservation = control_reservation.ok_or_else(|| {
            anyhow::anyhow!("control listeners were not reserved before runtime startup")
        })?;
        let (control_runtime, command_drain, command_submit) =
            control_reservation.launch(world.clone())?;
        info!(
            requested_mode = cli.mode.as_str(),
            active_mode = active_mode.as_str(),
            renderer = renderer.name(),
            "Starting ScriptBots simulation shell"
        );
        let context = RendererContext {
            world: Arc::clone(&world),
            analytics: analytics.clone(),
            simulation_step: Arc::clone(&simulation_step),
            control_runtime: &control_runtime,
            command_drain,
            command_submit,
        };
        let render_result = renderer.run(context);
        let control_result = control_runtime.shutdown();
        match (render_result, control_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(render_error), Ok(())) => Err(render_error),
            (Ok(()), Err(control_error)) => Err(control_error),
            (Err(render_error), Err(control_error)) => Err(render_error).context(format!(
                "control runtime shutdown also failed: {control_error:#}"
            )),
        }
    })();
    let result = finish_with_storage(runtime_result, "runtime", || {
        finalize_and_shutdown_storage(&world, &persistence, &mut storage_pipeline)
    });
    emit_sense_run_end(capture_shared_sense_run_summary(&world), result.is_ok());
    result
}

fn prefer_storage_failure<T>(
    operation: Result<T>,
    storage: Result<()>,
    operation_name: &str,
) -> Result<T> {
    match (operation, storage) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(operation_error), Ok(())) => Err(operation_error),
        (Ok(_), Err(storage_error)) => Err(storage_error),
        (Err(operation_error), Err(storage_error)) => {
            Err(storage_error).context(format!("{operation_name} also failed: {operation_error:#}"))
        }
    }
}

fn finish_with_storage<T>(
    operation: Result<T>,
    operation_name: &str,
    finish: impl FnOnce() -> Result<()>,
) -> Result<T> {
    // Invoke cleanup unconditionally before inspecting the operation result.
    // This is the ordinary-error boundary for every storage-owning host path.
    let storage = finish();
    prefer_storage_failure(operation, storage, operation_name)
}

fn shutdown_storage(pipeline: &mut StoragePipeline) -> Result<scriptbots_storage::ShutdownReceipt> {
    pipeline
        .shutdown()
        .context("FrankenSQLite worker failed during acknowledged shutdown")
}

#[derive(Debug, Clone, Copy)]
struct StorageFinalization {
    admitted_tail: bool,
    required_tick: Option<u64>,
}

fn finalize_world_persistence(
    world: &mut WorldState,
    persistence: &mut PersistenceAdmissionSession,
) -> Result<StorageFinalization> {
    let mut admitted_tail = false;
    if persistence.fault().is_some() {
        admitted_tail |= persistence
            .retry_pending(world)
            .context("failed to re-admit the retained persistence batch before shutdown")?;
    }

    match persistence.finalize(world) {
        Ok(admitted) => admitted_tail |= admitted,
        Err(first_error) if persistence.has_pending_batch() => {
            // Finalization itself can create the retained batch. Retry that
            // exact batch before closing worker admission; never reconstruct it.
            let retried = persistence.retry_pending(world).with_context(|| {
                format!(
                    "failed to re-admit the final partial persistence batch after its first rejection: {first_error}"
                )
            })?;
            if !retried {
                bail!(
                    "final persistence admission failed without retaining an exact retry batch: {first_error}"
                );
            }
            admitted_tail = true;
        }
        Err(error) => return Err(error).context("failed to finalize persistence boundary"),
    }

    if persistence.fault().is_some() || persistence.has_pending_batch() {
        bail!("storage finalization left an unresolved retained persistence batch");
    }

    Ok(StorageFinalization {
        admitted_tail,
        // Admission history, rather than the current cadence setting, is the
        // shutdown contract. A runtime update may disable persistence after a
        // batch was admitted, and bootstrap origins can admit a real tick-zero
        // batch even though an untouched tick-zero world has no batch at all.
        required_tick: persistence.last_admitted_tick().map(|tick| tick.0),
    })
}

fn finalize_and_shutdown_storage(
    world: &Arc<Mutex<WorldState>>,
    persistence: &SharedPersistenceAdmission,
    pipeline: &mut StoragePipeline,
) -> Result<()> {
    let finalization = (|| -> Result<StorageFinalization> {
        let mut world = world
            .lock()
            .map_err(|error| anyhow::anyhow!("world mutex poisoned during shutdown: {error}"))?;
        let mut persistence = persistence.lock().map_err(|error| {
            anyhow::anyhow!("persistence session mutex poisoned during shutdown: {error}")
        })?;
        finalize_world_persistence(&mut world, &mut persistence)
    })();
    finalize_then_shutdown_storage(finalization, pipeline)
}

fn finalize_then_shutdown_storage(
    finalization: Result<StorageFinalization>,
    pipeline: &mut StoragePipeline,
) -> Result<()> {
    let shutdown = shutdown_storage(pipeline);

    match (finalization, shutdown) {
        (Ok(finalization), Ok(receipt)) => {
            let receipt = validate_shutdown_receipt(finalization, receipt)?;
            if finalization.admitted_tail {
                info!("Admitted and committed final partial persistence batch");
            }
            info!(
                committed_tick = ?receipt.committed_tick,
                guarantee = ?receipt.guarantee,
                admitted_batch_id = ?receipt.watermarks.admitted.map(|batch_id| batch_id.get()),
                applied_batch_id = ?receipt.watermarks.applied.map(|batch_id| batch_id.get()),
                durable_batch_id = ?receipt.watermarks.durable.map(|batch_id| batch_id.get()),
                analytics_revision = receipt.analytics_revision,
                "FrankenSQLite worker shut down with an explicit persistence receipt"
            );
            Ok(())
        }
        (Err(finalization_error), Ok(_)) => Err(finalization_error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error),
        (Err(finalization_error), Err(shutdown_error)) => Err(finalization_error).context(format!(
            "FrankenSQLite worker shutdown also failed: {shutdown_error:#}"
        )),
    }
}

fn validate_shutdown_receipt(
    finalization: StorageFinalization,
    receipt: ShutdownReceipt,
) -> Result<ShutdownReceipt> {
    let watermarks = receipt.watermarks;
    let closed_watermark_prefix = match receipt.guarantee {
        PersistenceGuarantee::Durable => {
            watermarks.admitted == watermarks.applied && watermarks.applied == watermarks.durable
        }
        PersistenceGuarantee::CommittedVolatile => {
            watermarks.admitted == watermarks.applied && watermarks.durable.is_none()
        }
    };
    if receipt.committed_tick != finalization.required_tick || !closed_watermark_prefix {
        let expected = match receipt.guarantee {
            PersistenceGuarantee::Durable => "admitted == applied == durable",
            PersistenceGuarantee::CommittedVolatile => "admitted == applied and durable == None",
        };
        bail!(
            "invalid FrankenSQLite shutdown receipt: guarantee={:?}, committed_tick={:?}, required_tick={:?}, admitted={:?}, applied={:?}, durable={:?}; expected {expected}",
            receipt.guarantee,
            receipt.committed_tick,
            finalization.required_tick,
            watermarks.admitted.map(|batch_id| batch_id.get()),
            watermarks.applied.map(|batch_id| batch_id.get()),
            watermarks.durable.map(|batch_id| batch_id.get()),
        );
    }
    Ok(receipt)
}

#[cfg(unix)]
fn apply_process_niceness(low_power: bool) -> Result<()> {
    use libc::{PRIO_PROCESS, id_t, setpriority};
    // Always reduce CPU priority a bit when low_power; otherwise keep default niceness.
    if low_power {
        unsafe {
            // niceness +10 (lower priority); ignore errors on restricted environments
            let _ = setpriority(PRIO_PROCESS as _, 0 as id_t, 10);
        }
    }
    // Best-effort I/O niceness via ionice class 3 (idle) where available.
    // There is no stable libc wrapper; attempt calling the ionice syscall number is fragile.
    // We intentionally skip ionice here and rely on OS tools if needed.
    Ok(())
}

#[cfg(windows)]
fn apply_process_niceness(low_power: bool) -> Result<()> {
    use windows_sys::Win32::System::Threading::{
        BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, SetPriorityClass,
    };
    unsafe {
        let handle = GetCurrentProcess();
        let class = if low_power {
            BELOW_NORMAL_PRIORITY_CLASS
        } else {
            0
        };
        if class != 0 {
            let _ = SetPriorityClass(handle, class);
        }
    }
    Ok(())
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}

fn run_started_at_unix_ms() -> Result<u64> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch; cannot identify this run")?;
    u64::try_from(elapsed.as_millis())
        .context("current Unix timestamp does not fit the run-manifest u64 contract")
}

fn materialize_run_seed(config: &mut ScriptBotsConfig) -> (u64, bool) {
    match config.rng_seed {
        Some(seed) => (seed, false),
        None => {
            let seed = rand::random::<u64>();
            config.rng_seed = Some(seed);
            (seed, true)
        }
    }
}

fn allocate_run_id() -> RunId {
    // A run can outlive its originating process and be merged into a database produced on another
    // host, so process-local counters and wall-clock namespaces are not durable identity. Draw the
    // complete 128-bit identifier from the operating system-seeded RNG. Zero is reserved as an
    // invalid/sentinel identity; it is the only value that requires a retry.
    loop {
        let candidate = rand::random::<u128>();
        if candidate != 0 {
            return RunId::new(candidate);
        }
    }
}

fn run_characterization_v0(
    cli: &AppCli,
    mut config: ScriptBotsConfig,
    mut scenario: ScenarioIdentityV0,
    config_overrides: Vec<ConfigFieldOverride>,
    ticks: u64,
) -> Result<()> {
    let (root_seed, generated_seed) = materialize_run_seed(&mut config);
    if generated_seed {
        info!(
            root_seed,
            "Generated and pinned the characterization run's previously unspecified scientific seed"
        );
    }
    let (mut world, mut persistence) =
        WorldState::with_persistence(config, Box::new(NullPersistence))?;
    let brain_keys = install_brains(&mut world, cli.brain)?.population;
    seed_agents(&mut world, &brain_keys)?;

    scenario.bootstrap_ticks = 0;

    let started_at_unix_ms = run_started_at_unix_ms()?;
    let identity = RunIdentityV1::new(allocate_run_id(), started_at_unix_ms, Some(ticks), None);
    identity
        .validate()
        .context("invalid characterization run identity")?;
    emit_sense_startup_contract();
    let trace = CharacterizationTraceV2::capture_with_scenario_and_session(
        identity,
        scenario,
        config_overrides,
        &mut world,
        &mut persistence,
        ticks,
    );
    emit_sense_run_end(SenseRunSummary::capture(&world), trace.is_ok());
    let mut bytes = trace?.canonical_json_bytes()?;
    bytes.push(b'\n');

    if let Some(path) = cli.characterization_out.as_ref() {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create characterization output directory {}",
                    parent.display()
                )
            })?;
        }
        fs::write(path, bytes).with_context(|| {
            format!(
                "failed to write V2 characterization trace to {}",
                path.display()
            )
        })?;
    } else {
        let stdout = io::stdout();
        let mut lock = stdout.lock();
        lock.write_all(&bytes)
            .context("failed to write V2 characterization trace to stdout")?;
        lock.flush()
            .context("failed to flush V2 characterization trace to stdout")?;
    }

    Ok(())
}

fn run_det_child(
    config: &ScriptBotsConfig,
    tick_limit: u64,
    brain_preset: BrainPreset,
) -> Result<()> {
    let run = run_headless_simulation(config, tick_limit, brain_preset)?;
    #[derive(serde::Serialize)]
    struct DetOut {
        events: usize,
        ticks: u64,
        last_tick: u64,
        summaries: Vec<TickSummary>,
    }
    let last_tick = run.summaries.last().map(|s| s.tick.0).unwrap_or(0);
    let out = DetOut {
        events: run.events.len(),
        ticks: run.simulated_ticks,
        last_tick,
        summaries: run.summaries,
    };
    let json = serde_json::to_string(&out)?;
    println!("{}", json);
    Ok(())
}

fn run_det_check(cli: &AppCli, ticks: u64) -> Result<()> {
    let exe = std::env::current_exe().context("failed to get current exe path")?;

    // Both children must replay the identical scenario: compose the parent's
    // effective config, pin a shared seed, and hand the whole thing over as a
    // config layer (env-only forwarding loses `--config`/`--rng-seed` flags).
    let mut config = compose_config(cli)?;
    if let Ok(seed) = std::env::var("SCRIPTBOTS_DET_SEED") {
        let parsed = seed
            .trim()
            .parse::<u64>()
            .context("SCRIPTBOTS_DET_SEED must be a u64")?;
        config.rng_seed = Some(parsed);
    }
    let seed = match config.rng_seed {
        Some(seed) => seed,
        None => {
            let generated = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x0D57_C0FF_EE00);
            config.rng_seed = Some(generated);
            generated
        }
    };
    let layer = toml::to_string_pretty(&config).context("failed to serialize det-check config")?;
    let layer_path = std::env::temp_dir().join(format!(
        "scriptbots-det-check-{}-{seed}.toml",
        std::process::id()
    ));
    fs::write(&layer_path, layer).with_context(|| {
        format!(
            "failed to write det-check config layer {}",
            layer_path.display()
        )
    })?;

    let spawn_child = |threads: Option<&str>| -> Result<std::process::Child> {
        let mut child = Command::new(&exe);
        child.arg("--config-only"); // avoid launching UI
        child.arg("--config");
        child.arg(&layer_path);
        child.arg("--brain");
        child.arg(cli.brain.as_str());
        child.env("SCRIPTBOTS_DET_RUN", "1");
        child.env("SCRIPTBOTS_DET_TICKS", ticks.to_string());
        child.env("SCRIPTBOTS_RNG_SEED", seed.to_string());
        child.env("RUST_LOG", "error");
        if let Some(threads) = threads {
            child.env("RAYON_NUM_THREADS", threads);
        }
        child.stdout(Stdio::piped());
        child.stderr(Stdio::null());
        child.spawn().context("failed to spawn det child")
    };

    // Child 1: single-thread; child N: default thread budget.
    let handle1 = spawn_child(Some("1"))?;
    let handlen = spawn_child(None)?;

    // Wait for both to complete (they run concurrently)
    let out1 = handle1
        .wait_with_output()
        .context("failed to wait for det child 1")?;
    let outn = handlen
        .wait_with_output()
        .context("failed to wait for det child N")?;
    if !out1.status.success() {
        bail!("det child 1 failed: status {:?}", out1.status);
    }
    if !outn.status.success() {
        bail!("det child N failed: status {:?}", outn.status);
    }

    #[derive(serde::Deserialize)]
    struct DetOutIn {
        events: usize,
        ticks: u64,
        last_tick: u64,
        summaries: Vec<TickSummary>,
    }
    let left: DetOutIn =
        serde_json::from_slice(&out1.stdout).context("failed to parse child 1 JSON output")?;
    let right: DetOutIn =
        serde_json::from_slice(&outn.stdout).context("failed to parse child N JSON output")?;

    if left.ticks != right.ticks || left.last_tick != right.last_tick {
        bail!(
            "tick count/last tick mismatch: 1t {:?} vs Nt {:?}",
            left.ticks,
            right.ticks
        );
    }
    for (idx, (a, b)) in left
        .summaries
        .iter()
        .zip(right.summaries.iter())
        .enumerate()
    {
        if a != b {
            println!(
                "{} divergence at idx {} tick {}",
                "✖".red().bold(),
                idx,
                a.tick.0
            );
            println!(
                "    1t: agents={} births={} deaths={} avgE={:.4}",
                a.agent_count, a.births, a.deaths, a.average_energy
            );
            println!(
                "    Nt: agents={} births={} deaths={} avgE={:.4}",
                b.agent_count, b.births, b.deaths, b.average_energy
            );
            bail!("determinism self-check failed");
        }
    }
    if left.events != right.events {
        bail!(
            "event count mismatch: 1t={} vs Nt={}",
            left.events,
            right.events
        );
    }
    let _ = fs::remove_file(&layer_path);
    println!(
        "{} Determinism self-check passed for {} ticks (seed {}, events: 1t={}, Nt={})",
        "✔".green().bold(),
        ticks,
        seed,
        left.events,
        right.events
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct ThresholdsOverride {
    tick: Option<usize>,
    agent: Option<usize>,
    event: Option<usize>,
    metric: Option<usize>,
}

fn thresholds_from_cli(cli: &AppCli) -> ThresholdsOverride {
    if let Some(raw) = cli.storage_thresholds.as_ref() {
        let mut parts = raw.split(',').map(|s| s.trim());
        let tick = parts.next().and_then(|p| p.parse::<usize>().ok());
        let agent = parts.next().and_then(|p| p.parse::<usize>().ok());
        let event = parts.next().and_then(|p| p.parse::<usize>().ok());
        let metric = parts.next().and_then(|p| p.parse::<usize>().ok());
        return ThresholdsOverride {
            tick,
            agent,
            event,
            metric,
        };
    }
    ThresholdsOverride::default()
}

fn storage_path_from_env() -> Result<String> {
    match env::var("SCRIPTBOTS_STORAGE_PATH") {
        Ok(path) => Ok(path),
        Err(env::VarError::NotPresent) => {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_millis());
            Ok(format!(
                "runs/scriptbots-{timestamp}-{}.sqlite",
                std::process::id()
            ))
        }
        Err(env::VarError::NotUnicode(_)) => {
            bail!(
                "SCRIPTBOTS_STORAGE_PATH is not valid Unicode; refusing to choose a different run database silently"
            )
        }
    }
}

fn recover_storage(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path).with_context(|| {
        format!(
            "cannot recover FrankenSQLite storage {} because it does not exist",
            path.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!(
            "refusing to recover non-regular FrankenSQLite storage path {}",
            path.display()
        );
    }
    let path = path.to_str().with_context(|| {
        format!(
            "storage recovery path is not valid UTF-8: {}",
            path.display()
        )
    })?;
    let mut pipeline = StoragePipeline::recover_existing(path)
        .with_context(|| format!("failed to recover FrankenSQLite storage at {path}"))?;
    let receipt = pipeline
        .shutdown()
        .context("recovered FrankenSQLite worker failed during acknowledged shutdown")?;
    println!(
        "{} Recovered storage {} (tick={:?}, admitted={:?}, applied={:?}, durable={:?})",
        "✔".green().bold(),
        path.cyan(),
        receipt.committed_tick,
        receipt.watermarks.admitted.map(|batch_id| batch_id.get()),
        receipt.watermarks.applied.map(|batch_id| batch_id.get()),
        receipt.watermarks.durable.map(|batch_id| batch_id.get()),
    );
    Ok(())
}

/// Build the canonical provenance record registered durably before tick zero.
fn build_run_manifest(
    world: &WorldState,
    identity: RunIdentityV1,
    scenario: ScenarioIdentityV0,
    thread_policy: ThreadPolicy,
    config_overrides: Vec<ConfigFieldOverride>,
) -> std::result::Result<RunManifestV3, scriptbots_app::RunManifestError> {
    RunManifestV3::from_world_with_provenance(
        identity,
        scenario,
        world,
        scriptbots_app::BuildProvenanceV0::current(),
    )
    .map(|manifest| {
        // Record what the run DECIDED, not merely what the environment said. Build provenance
        // captures environment declarations, which may have lost to a more specific policy layer.
        // The same rule covers the config itself: cross-layer displacements are part of how this
        // exact configuration came to be, so they ride beside the thread policy.
        manifest
            .with_thread_policy(ThreadPolicyV0 {
                threads: thread_policy.threads,
                source: thread_policy.source.wire_tag().to_owned(),
                overridden: thread_policy
                    .overridden
                    .map(|declined| declined.wire_tag().to_owned()),
            })
            .with_config_overrides(config_overrides)
    })
}

struct PendingRunManifest {
    path: PathBuf,
    manifest: RunManifestV3,
    start: WorldDigestV1,
    requested: u64,
    thread_policy: ThreadPolicy,
}

fn prepare_run_manifest(
    world: &WorldState,
    storage_path: Option<&str>,
    manifest: RunManifestV3,
    thread_policy: ThreadPolicy,
    bootstrap_ticks: u64,
) -> Option<PendingRunManifest> {
    let Some(storage_path) = storage_path else {
        // The in-memory database already contains the complete launch manifest; it simply has no
        // durable directory beside which a supplemental post-bootstrap JSON artifact can live.
        return None;
    };
    let manifest_path = std::path::Path::new(storage_path).with_extension("manifest.json");

    let start = match world.world_digest_v1() {
        Ok(digest) => digest,
        Err(error) => {
            warn!(
                error = %error,
                path = %manifest_path.display(),
                "could not capture the launch-state WorldDigestV1; the database provenance is \
                 durable, but the supplemental bootstrap sidecar cannot be emitted"
            );
            return None;
        }
    };

    Some(PendingRunManifest {
        path: manifest_path,
        manifest,
        start,
        requested: bootstrap_ticks,
        thread_policy,
    })
}

fn emit_run_manifest(world: &WorldState, pending: Option<PendingRunManifest>, completed: u64) {
    let Some(pending) = pending else {
        return;
    };
    let PendingRunManifest {
        path,
        manifest,
        start,
        requested,
        thread_policy,
    } = pending;
    let end = match world.world_digest_v1() {
        Ok(digest) => digest,
        Err(error) => {
            warn!(
                error = %error,
                path = %path.display(),
                completed,
                "could not capture the post-bootstrap WorldDigestV1; the database provenance is \
                 durable, but the supplemental bootstrap sidecar cannot be emitted"
            );
            return;
        }
    };
    let start_tick = start.tick.0;
    let end_tick = end.tick.0;
    let manifest = match manifest.with_bootstrap_evidence(BootstrapEvidenceV0 {
        requested,
        completed,
        start,
        end,
    }) {
        Ok(manifest) => manifest,
        Err(error) => {
            warn!(
                error = %error,
                path = %path.display(),
                "bootstrap evidence did not satisfy the run-manifest contract; the database \
                 provenance is durable, but the supplemental sidecar cannot be emitted"
            );
            return;
        }
    };

    match manifest.canonical_json_bytes() {
        Ok(encoded) => {
            if let Err(error) = std::fs::write(&path, &encoded) {
                warn!(
                    error = %error,
                    path = %path.display(),
                    "could not write the supplemental run-manifest sidecar; database provenance \
                     remains durable"
                );
                return;
            }
            info!(
                path = %path.display(),
                config_digest = %manifest.config_digest,
                root_seed = manifest.root_seed,
                reproducible = manifest.reproducible,
                warnings = manifest.warnings.len(),
                bootstrap_requested = requested,
                bootstrap_completed = completed,
                bootstrap_start_tick = start_tick,
                bootstrap_end_tick = end_tick,
                threads = ?thread_policy.threads,
                thread_source = %thread_policy.source,
                "wrote run manifest"
            );
            if !manifest.reproducible {
                // Say it out loud. A manifest that claims reproducibility it does
                // not have is worse than no manifest at all.
                warn!(
                    warnings = ?manifest.warnings,
                    "this run is NOT reproducible; the manifest records why"
                );
            }
        }
        Err(error) => warn!(
            error = %error,
            "could not encode the supplemental run-manifest sidecar; database provenance remains durable"
        ),
    }
}

fn bootstrap_world(
    mut config: ScriptBotsConfig,
    brain_preset: BrainPreset,
    storage_mode: StorageMode,
    thresholds: ThresholdsOverride,
    bootstrap_ticks: u64,
    thread_policy: ThreadPolicy,
    mut scenario: ScenarioIdentityV0,
    config_overrides: Vec<ConfigFieldOverride>,
) -> Result<(
    SharedWorld,
    SharedPersistenceAdmission,
    SharedAnalytics,
    StoragePipeline,
)> {
    // This function owns bootstrap execution, so it also owns the authoritative requested count.
    // Never trust a caller-populated scenario field that could disagree with the work done here.
    scenario.bootstrap_ticks = bootstrap_ticks;
    let (root_seed, generated_seed) = materialize_run_seed(&mut config);
    if generated_seed {
        info!(
            root_seed,
            "Generated and pinned the run's previously unspecified scientific seed"
        );
        println!(
            "{} Run seed: {}",
            "◆".bright_blue().bold(),
            root_seed.to_string().cyan()
        );
    }

    #[cfg(feature = "neuro")]
    if brain_preset == BrainPreset::Mixed {
        let _ = validated_neuroflow_config(&config)?;
    }

    // Establish the complete scientific launch state before opening a database. This keeps
    // invalid brain/seed configuration from reserving a run path and lets the manifest describe
    // the actual initial roster rather than a planned approximation.
    let mut world =
        WorldState::new(config).context("failed to construct world before tick zero")?;
    let brain_keys = install_brains(&mut world, brain_preset)?.population;
    seed_agents(&mut world, &brain_keys)?;

    let started_at_unix_ms = run_started_at_unix_ms()?;
    let identity = RunIdentityV1::new(
        allocate_run_id(),
        started_at_unix_ms,
        None,
        Some(LIVE_RUN_POLICY.to_owned()),
    );
    identity
        .validate()
        .context("invalid live-run identity before storage registration")?;
    let manifest = build_run_manifest(&world, identity, scenario, thread_policy, config_overrides)
        .context("failed to build durable run provenance before tick zero")?;
    let storage_record = manifest
        .to_storage_record()
        .context("failed to project durable run provenance before tick zero")?;

    // Pipeline startup atomically registers `storage_record`. A registration failure is fatal:
    // no sink is bound and no bootstrap transition can run without queryable database provenance.
    let (mut pipeline, storage_path) = match storage_mode {
        StorageMode::File => {
            let storage_path = storage_path_from_env()?;
            let pipeline = match (
                thresholds.tick,
                thresholds.agent,
                thresholds.event,
                thresholds.metric,
            ) {
                (Some(t), Some(a), Some(e), Some(m)) => {
                    StoragePipeline::create_new_file_for_run_with_thresholds(
                        &storage_path,
                        storage_record,
                        t,
                        a,
                        e,
                        m,
                    )
                }
                _ => StoragePipeline::create_new_file_for_run(&storage_path, storage_record),
            }
            .with_context(|| {
                format!(
                    "failed to register run provenance and initialize FrankenSQLite storage at {storage_path}"
                )
            })?;
            (pipeline, Some(storage_path))
        }
        StorageMode::Memory => (
            StoragePipeline::memory_for_run_with_thresholds(
                storage_record,
                thresholds.tick.unwrap_or(64),
                thresholds.agent.unwrap_or(2048),
                thresholds.event.unwrap_or(512),
                thresholds.metric.unwrap_or(512),
            )
            .context("failed to register run provenance in volatile FrankenSQLite storage")?,
            None,
        ),
    };
    let run_id = pipeline.run_id();
    if let Some(storage_path) = storage_path.as_deref() {
        info!(path = %storage_path, "Selected unique FrankenSQLite run database");
        println!(
            "{} Run {} database: {}",
            "◆".bright_blue().bold(),
            run_id.to_string().cyan(),
            storage_path.cyan()
        );
    } else {
        info!(%run_id, "Selected volatile in-memory FrankenSQLite storage");
    }

    let pending_manifest = prepare_run_manifest(
        &world,
        storage_path.as_deref(),
        manifest,
        thread_policy,
        bootstrap_ticks,
    );
    let analytics = pipeline.analytics_provider();
    let mut persistence = match world.bind_persistence(Box::new(pipeline.sink())) {
        Ok(persistence) => persistence,
        Err(error) => {
            return finish_with_storage(Err(error.into()), "world persistence binding", || {
                shutdown_storage(&mut pipeline).map(|_| ())
            });
        }
    };
    emit_sense_startup_contract();
    let bootstrap_result = (|| -> Result<()> {
        for _ in 0..bootstrap_ticks {
            persistence.step(&mut world)?;
        }
        let completed_bootstrap_ticks = bootstrap_ticks;
        emit_run_manifest(&world, pending_manifest, completed_bootstrap_ticks);

        if let Some(summary) = world.history().last() {
            info!(
                tick = summary.tick.0,
                agents = summary.agent_count,
                births = summary.births,
                deaths = summary.deaths,
                avg_energy = summary.average_energy,
                bootstrap_ticks,
                "Primed world and persisted initial summary",
            );
        } else if bootstrap_ticks == 0 {
            info!("Initialized seeded world at tick zero without bootstrap advancement");
        } else {
            warn!("World bootstrap completed without persistence summaries");
        }
        Ok(())
    })();
    if let Err(error) = bootstrap_result {
        let result = finish_with_storage(Err(error), "world bootstrap", || {
            let finalization = finalize_world_persistence(&mut world, &mut persistence);
            finalize_then_shutdown_storage(finalization, &mut pipeline)
        });
        emit_sense_run_end(SenseRunSummary::capture(&world), result.is_ok());
        return result;
    }

    Ok((
        Arc::new(Mutex::new(world)),
        Arc::new(Mutex::new(persistence)),
        analytics,
        pipeline,
    ))
}

fn compose_config(cli: &AppCli) -> Result<ScriptBotsConfig> {
    compose_config_with_scenario(cli).map(|(config, _scenario, _overrides)| config)
}

/// Compose the effective configuration, its exact ordered layer provenance, and every
/// cross-layer displacement in one pass.
///
/// Layer order, most general first: built-in defaults -> configuration files (in
/// order) -> environment -> CLI. Statement GATHERING (file reads, environment reads,
/// flag interpretation) happens here; the merge itself is the pure
/// [`resolve_config_layers`], so the precedence rules are testable without a process.
/// Reading each file once is load-bearing: re-reading files after composition could
/// digest bytes different from those that actually configured the run.
fn compose_config_with_scenario(
    cli: &AppCli,
) -> Result<(
    ScriptBotsConfig,
    ScenarioIdentityV0,
    Vec<ConfigFieldOverride>,
)> {
    let defaults = ScriptBotsConfig {
        persistence_interval: 60,
        history_capacity: 600,
        ..ScriptBotsConfig::default()
    };
    let scenario_id = if cli.config_layers.is_empty() {
        "scriptbots-app-default-v1"
    } else {
        "scriptbots-app-layered-v1"
    };
    let mut scenario = ScenarioIdentityV0::caller_seeded(scenario_id);
    scenario.population_recipe = format!(
        "fixed-4x4-registered-brain-grid-v1;brain={}",
        cli.brain.as_str()
    );

    let defaults_value =
        serde_json::to_value(&defaults).context("failed to serialize base config")?;
    // The defaults are a layer too: "every applied layer appends its digest" includes
    // the layer every run starts from, so even a default-only run's manifest names
    // what built it instead of carrying an empty provenance list.
    scenario.record_config_layer(
        ConfigLayerKind::Defaults,
        &canonical_layer_bytes(&defaults_value),
    );

    let mut statements: Vec<ConfigLayerStatement> = Vec::new();
    for path in &cli.config_layers {
        let (fields, source_bytes) = load_config_layer_with_source(path)?;
        info!(layer = %path.display(), "Applying configuration layer");
        scenario.record_config_layer(ConfigLayerKind::File, &source_bytes);
        statements.push(ConfigLayerStatement {
            kind: ConfigLayerKind::File,
            label: format!("file:{}", path.display()),
            fields,
        });
    }

    // Whether anything before the environment layer already spoke for
    // `render.auto_exposure`. When nothing did, a speed-only environment override
    // must state the block's fresh default (`enabled: true`) itself — the merged
    // block replaces a `null` wholesale and `enabled` is a required field.
    let auto_exposure_already_spoken = defaults.render.auto_exposure.is_some()
        || statements.iter().any(|statement| {
            statement
                .fields
                .pointer("/render/auto_exposure")
                .is_some_and(|value| !value.is_null())
        });
    for statement in gather_env_statements(auto_exposure_already_spoken, &defaults_value)? {
        scenario.record_config_layer(
            ConfigLayerKind::Environment,
            &canonical_layer_bytes(&statement.fields),
        );
        statements.push(statement);
    }
    for statement in gather_cli_statements(cli, &defaults_value)? {
        scenario.record_config_layer(
            ConfigLayerKind::Cli,
            &canonical_layer_bytes(&statement.fields),
        );
        statements.push(statement);
    }

    let resolved = resolve_config_layers(&defaults_value, &statements);
    for displaced in &resolved.overrides {
        info!(
            path = %displaced.path,
            losing_layer = %displaced.losing_layer,
            winning_layer = %displaced.winning_layer,
            "Configuration layer displaced an earlier layer's value"
        );
    }

    let config = deserialize_merged_config(&resolved.merged)?;
    config
        .validate()
        .context("invalid composed ScriptBots configuration")?;
    Ok((config, scenario, resolved.overrides))
}

/// Decode the merged configuration tree, naming the exact field on failure.
fn deserialize_merged_config(merged: &JsonValue) -> Result<ScriptBotsConfig> {
    let json = serde_json::to_string(merged).context("failed to encode merged configuration")?;
    let mut deserializer = serde_json::Deserializer::from_str(&json);
    serde_path_to_error::deserialize::<_, ScriptBotsConfig>(&mut deserializer).map_err(
        |error: serde_path_to_error::Error<serde_json::Error>| {
            anyhow::anyhow!(
                "failed to deserialize merged configuration at {}: {}",
                error.path(),
                error.inner()
            )
        },
    )
}

/// Gather what the environment said about the configuration, without mutating anything.
///
/// Two statements, most general first:
/// 1. `SCRIPTBOTS_CONFIG_OVERRIDES` — one inline TOML document able to speak for any
///    knob. Malformed content fails closed, like every other control-environment value.
/// 2. The typed `SCRIPTBOTS_*` variables — each names exactly one knob, so together
///    they are more specific than the catch-all document and are applied after it.
///
/// The values are PASSED to the resolver, never written back into the process
/// environment: startup `set_var` smearing is exactly what made the thread-count
/// environment capture lie about what the user actually exported.
fn gather_env_statements(
    auto_exposure_already_spoken: bool,
    defaults_value: &JsonValue,
) -> Result<Vec<ConfigLayerStatement>> {
    let mut statements = Vec::new();
    if let Ok(raw) = env::var("SCRIPTBOTS_CONFIG_OVERRIDES") {
        let fields: JsonValue = toml::from_str(&raw)
            .context("failed to parse SCRIPTBOTS_CONFIG_OVERRIDES as a TOML document")?;
        if fields.as_object().is_some_and(|map| !map.is_empty()) {
            reject_unknown_paths(defaults_value, &fields, "SCRIPTBOTS_CONFIG_OVERRIDES")?;
            statements.push(ConfigLayerStatement {
                kind: ConfigLayerKind::Environment,
                label: "env:SCRIPTBOTS_CONFIG_OVERRIDES".to_owned(),
                fields,
            });
        }
    }
    let typed = typed_env_fields(auto_exposure_already_spoken)?;
    if typed.as_object().is_some_and(|map| !map.is_empty()) {
        statements.push(ConfigLayerStatement {
            kind: ConfigLayerKind::Environment,
            label: "env:SCRIPTBOTS_*".to_owned(),
            fields: typed,
        });
    }
    Ok(statements)
}

/// The typed `SCRIPTBOTS_*` environment variables, decoded into the partial
/// configuration they state.
///
/// The pre-existing NeuroFlow/render variables keep their historical warn-and-skip
/// semantics for invalid values; the variables clap used to parse
/// (`SCRIPTBOTS_RNG_SEED` and the auto-pause family) keep their historical
/// fail-closed semantics.
#[allow(clippy::too_many_lines)]
fn typed_env_fields(auto_exposure_already_spoken: bool) -> Result<JsonValue> {
    let mut root = serde_json::Map::new();

    let mut neuroflow = serde_json::Map::new();
    if let Ok(value) = env::var("SCRIPTBOTS_NEUROFLOW_ENABLED") {
        match parse_bool(&value) {
            Some(flag) => {
                neuroflow.insert("enabled".to_owned(), JsonValue::Bool(flag));
            }
            None => {
                warn!(value = %value, "Invalid SCRIPTBOTS_NEUROFLOW_ENABLED value; expected true/false")
            }
        }
    }
    if let Ok(value) = env::var("SCRIPTBOTS_NEUROFLOW_HIDDEN") {
        match parse_layers(&value) {
            Some(layers) => {
                neuroflow.insert(
                    "hidden_layers".to_owned(),
                    serde_json::to_value(layers)
                        .context("failed to encode SCRIPTBOTS_NEUROFLOW_HIDDEN")?,
                );
            }
            None => {
                warn!(value = %value, "Invalid SCRIPTBOTS_NEUROFLOW_HIDDEN value; expected comma-separated integers")
            }
        }
    }
    if let Ok(value) = env::var("SCRIPTBOTS_NEUROFLOW_ACTIVATION") {
        match parse_activation(&value) {
            Some(activation) => {
                neuroflow.insert(
                    "activation".to_owned(),
                    serde_json::to_value(activation)
                        .context("failed to encode SCRIPTBOTS_NEUROFLOW_ACTIVATION")?,
                );
            }
            None => {
                warn!(value = %value, "Invalid SCRIPTBOTS_NEUROFLOW_ACTIVATION value; expected tanh|sigmoid|relu")
            }
        }
    }
    if !neuroflow.is_empty() {
        root.insert("neuroflow".to_owned(), JsonValue::Object(neuroflow));
    }

    let mut render = serde_json::Map::new();
    if let Ok(value) = env::var("SCRIPTBOTS_RENDER_TONEMAP") {
        match parse_tonemap(&value) {
            Some(mode) => {
                render.insert(
                    "tonemap_mode".to_owned(),
                    serde_json::to_value(mode)
                        .context("failed to encode SCRIPTBOTS_RENDER_TONEMAP")?,
                );
            }
            None => {
                warn!(value = %value, "Invalid SCRIPTBOTS_RENDER_TONEMAP value; expected aces|agx|tony")
            }
        }
    }
    if let Ok(value) = env::var("SCRIPTBOTS_RENDER_TONEMAP_BIAS") {
        match value.trim().parse::<f32>() {
            Ok(bias) if bias.is_finite() => {
                render.insert(
                    "tonemap_exposure_bias".to_owned(),
                    JsonValue::from(f64::from(bias)),
                );
            }
            _ => {
                warn!(value = %value, "Invalid SCRIPTBOTS_RENDER_TONEMAP_BIAS value; expected finite f32")
            }
        }
    }
    let mut auto_exposure = serde_json::Map::new();
    if let Ok(value) = env::var("SCRIPTBOTS_RENDER_AUTO_EXPOSURE") {
        match parse_bool(&value) {
            Some(enabled) => {
                auto_exposure.insert("enabled".to_owned(), JsonValue::Bool(enabled));
            }
            None => {
                warn!(value = %value, "Invalid SCRIPTBOTS_RENDER_AUTO_EXPOSURE value; expected true/false")
            }
        }
    }
    if let Ok(value) = env::var("SCRIPTBOTS_RENDER_AUTO_EXPOSURE_SPEED_BRIGHTEN") {
        match value.trim().parse::<f32>() {
            Ok(speed) if speed.is_finite() && speed >= 0.0 => {
                auto_exposure.insert(
                    "speed_brighten".to_owned(),
                    JsonValue::from(f64::from(speed)),
                );
            }
            _ => {
                warn!(value = %value, "Invalid SCRIPTBOTS_RENDER_AUTO_EXPOSURE_SPEED_BRIGHTEN value; expected non-negative f32")
            }
        }
    }
    if let Ok(value) = env::var("SCRIPTBOTS_RENDER_AUTO_EXPOSURE_SPEED_DARKEN") {
        match value.trim().parse::<f32>() {
            Ok(speed) if speed.is_finite() && speed >= 0.0 => {
                auto_exposure.insert("speed_darken".to_owned(), JsonValue::from(f64::from(speed)));
            }
            _ => {
                warn!(value = %value, "Invalid SCRIPTBOTS_RENDER_AUTO_EXPOSURE_SPEED_DARKEN value; expected non-negative f32")
            }
        }
    }
    if !auto_exposure.is_empty() {
        // A speed-only override creating a FRESH auto-exposure block states the
        // block's historical fresh default (`enabled: true`) itself, because the
        // merged block replaces a `null` wholesale and `enabled` is required. When an
        // earlier layer already spoke for the block, deep-merge preserves that
        // layer's `enabled` — exactly as the old take-or-init mutation did.
        if !auto_exposure.contains_key("enabled") && !auto_exposure_already_spoken {
            auto_exposure.insert("enabled".to_owned(), JsonValue::Bool(true));
        }
        render.insert("auto_exposure".to_owned(), JsonValue::Object(auto_exposure));
    }
    if !render.is_empty() {
        root.insert("render".to_owned(), JsonValue::Object(render));
    }

    if let Ok(value) = env::var("SCRIPTBOTS_RNG_SEED") {
        let seed: u64 = value.trim().parse().with_context(|| {
            format!("invalid SCRIPTBOTS_RNG_SEED value `{value}`; expected an unsigned 64-bit seed")
        })?;
        root.insert("rng_seed".to_owned(), JsonValue::from(seed));
    }
    let mut control = serde_json::Map::new();
    if let Ok(value) = env::var("SCRIPTBOTS_AUTO_PAUSE_BELOW") {
        let count: u32 = value.trim().parse().with_context(|| {
            format!("invalid SCRIPTBOTS_AUTO_PAUSE_BELOW value `{value}`; expected u32")
        })?;
        control.insert(
            "auto_pause_population_below".to_owned(),
            JsonValue::from(count),
        );
    }
    if let Ok(value) = env::var("SCRIPTBOTS_AUTO_PAUSE_AGE_ABOVE") {
        let age: u32 = value.trim().parse().with_context(|| {
            format!("invalid SCRIPTBOTS_AUTO_PAUSE_AGE_ABOVE value `{value}`; expected u32")
        })?;
        control.insert("auto_pause_age_above".to_owned(), JsonValue::from(age));
    }
    if let Ok(value) = env::var("SCRIPTBOTS_AUTO_PAUSE_ON_SPIKE") {
        let flag = parse_bool(&value).with_context(|| {
            format!("invalid SCRIPTBOTS_AUTO_PAUSE_ON_SPIKE value `{value}`; expected true/false")
        })?;
        // An explicit `false` is a statement too: it must be able to displace a file
        // layer's `true` and be evidenced in the override record, exactly like any
        // other explicitly stated value.
        control.insert("auto_pause_on_spike_hit".to_owned(), JsonValue::Bool(flag));
    }
    if !control.is_empty() {
        root.insert("control".to_owned(), JsonValue::Object(control));
    }

    Ok(JsonValue::Object(root))
}

/// Gather what the command line said about the configuration.
///
/// Generic `--set` entries come first, each as its own statement so a later entry
/// displacing an earlier one is attributable to the exact flag text; the typed flags
/// (`--rng-seed`, the auto-pause family) name single knobs and are applied last.
fn gather_cli_statements(
    cli: &AppCli,
    defaults_value: &JsonValue,
) -> Result<Vec<ConfigLayerStatement>> {
    let mut statements = Vec::new();
    for entry in &cli.set_overrides {
        let fields: JsonValue = toml::from_str(entry).with_context(|| {
            format!(
                "failed to parse --set {entry} as TOML (expected PATH=VALUE; string values \
                 use TOML quotes, e.g. --set 'label=\"dunes\"')"
            )
        })?;
        if fields.as_object().is_none_or(serde_json::Map::is_empty) {
            bail!("--set {entry} names no configuration field (expected PATH=VALUE)");
        }
        reject_unknown_paths(defaults_value, &fields, &format!("--set {entry}"))?;
        statements.push(ConfigLayerStatement {
            kind: ConfigLayerKind::Cli,
            label: format!("cli:--set {entry}"),
            fields,
        });
    }
    let typed = typed_cli_fields(cli);
    if typed.as_object().is_some_and(|map| !map.is_empty()) {
        statements.push(ConfigLayerStatement {
            kind: ConfigLayerKind::Cli,
            label: "cli:flags".to_owned(),
            fields: typed,
        });
    }
    Ok(statements)
}

/// Reject generic-override paths the configuration schema does not contain.
///
/// The generic surfaces (`--set`, `SCRIPTBOTS_CONFIG_OVERRIDES`) can name any dotted
/// path, and serde deserialization silently IGNORES unknown keys — so a typo like
/// `world_widht=800` would merge, vanish, and leave the run configured differently
/// than its operator believes. Unknown keys fail closed instead, checked against the
/// serialized defaults tree (the complete schema: every config field is present in
/// it, and no config field is map-valued). Two deliberate limits:
/// - below a `null` (an unset `Option` block) the JSON shape is not introspectable,
///   so deeper keys are admitted and the typed deserializer owns their validation;
/// - type mismatches are not checked here because serde already fails them loudly
///   with an exact field path — only the SILENT failure class lives in this guard.
fn reject_unknown_paths(defaults: &JsonValue, incoming: &JsonValue, label: &str) -> Result<()> {
    fn walk(
        defaults: &JsonValue,
        incoming: &JsonValue,
        path: &mut Vec<String>,
        label: &str,
    ) -> Result<()> {
        let (JsonValue::Object(default_map), JsonValue::Object(incoming_map)) =
            (defaults, incoming)
        else {
            return Ok(());
        };
        for (key, value) in incoming_map {
            let Some(deeper) = default_map.get(key) else {
                path.push(key.clone());
                bail!(
                    "{label} names `{}`, which is not a configuration field",
                    path.join(".")
                );
            };
            path.push(key.clone());
            walk(deeper, value, path, label)?;
            path.pop();
        }
        Ok(())
    }
    let mut path = Vec::new();
    walk(defaults, incoming, &mut path, label)
}

/// The typed configuration-affecting CLI flags, as the partial they state.
fn typed_cli_fields(cli: &AppCli) -> JsonValue {
    let mut root = serde_json::Map::new();
    if let Some(seed) = cli.rng_seed {
        root.insert("rng_seed".to_owned(), JsonValue::from(seed));
    }
    let mut control = serde_json::Map::new();
    if let Some(limit) = cli.auto_pause_below {
        control.insert(
            "auto_pause_population_below".to_owned(),
            JsonValue::from(limit),
        );
    }
    if let Some(age) = cli.auto_pause_age_above {
        control.insert("auto_pause_age_above".to_owned(), JsonValue::from(age));
    }
    if cli.auto_pause_on_spike {
        control.insert("auto_pause_on_spike_hit".to_owned(), JsonValue::Bool(true));
    }
    if !control.is_empty() {
        root.insert("control".to_owned(), JsonValue::Object(control));
    }
    JsonValue::Object(root)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConfigEmitOutcome {
    Continue,
    Exit,
}

fn maybe_emit_config(cli: &AppCli, config: &ScriptBotsConfig) -> Result<Option<ConfigEmitOutcome>> {
    if !cli.print_config && cli.write_config.is_none() {
        return Ok(None);
    }

    let rendered = match cli.config_format {
        ConfigFormat::Json => serde_json::to_string_pretty(config)?,
        ConfigFormat::Toml => toml::to_string_pretty(config)?,
        ConfigFormat::Ron => ron::ser::to_string_pretty(config, RonPrettyConfig::new())?,
    };

    if cli.print_config {
        println!("{}", rendered);
    }

    if let Some(path) = cli.write_config.as_ref() {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, rendered.as_bytes())?;
        info!(path = %path.display(), "Wrote composed configuration to file");
    }

    let outcome = if cli.config_only {
        ConfigEmitOutcome::Exit
    } else {
        ConfigEmitOutcome::Continue
    };

    Ok(Some(outcome))
}

#[derive(Parser, Debug)]
#[command(
    name = "scriptbots-app",
    version,
    about = "ScriptBots simulation shell"
)]
struct AppCli {
    /// Rendering mode (auto selects only compiled backends and otherwise uses the terminal).
    #[arg(
        long,
        value_enum,
        env = "SCRIPTBOTS_MODE",
        default_value_t = RendererMode::Auto
    )]
    mode: RendererMode,
    /// Brain-family preset used for registration, founders, and later population injection.
    #[arg(
        long,
        value_enum,
        env = "SCRIPTBOTS_BRAIN",
        default_value_t = BrainPreset::Mixed
    )]
    brain: BrainPreset,
    /// Layered configuration files (TOML or RON) applied in order.
    #[arg(
        long = "config",
        value_name = "FILE",
        action = ArgAction::Append,
        env = "SCRIPTBOTS_CONFIG",
        value_delimiter = ';'
    )]
    config_layers: Vec<PathBuf>,
    /// RNG seed override for deterministic runs.
    ///
    /// `SCRIPTBOTS_RNG_SEED` supplies the same knob through the environment layer,
    /// which this flag outranks — the two are separate layers with separate
    /// provenance, not one flag with two spellings.
    #[arg(long = "rng-seed", value_name = "SEED")]
    rng_seed: Option<u64>,
    /// Explicit number of simulation ticks to run before launching the selected frontend.
    #[arg(
        long = "bootstrap-ticks",
        value_name = "TICKS",
        env = "SCRIPTBOTS_BOOTSTRAP_TICKS",
        default_value_t = DEFAULT_BOOTSTRAP_TICKS
    )]
    bootstrap_ticks: u64,
    /// Path to a FrankenSQLite run to verify via headless deterministic replay.
    #[arg(long = "replay-db", value_name = "FILE", env = "SCRIPTBOTS_REPLAY_DB")]
    replay_db: Option<PathBuf>,
    /// Optional comparison database for divergence analysis.
    #[arg(long = "compare-db", value_name = "FILE", requires = "replay_db")]
    compare_db: Option<PathBuf>,
    /// Limit the number of ticks simulated during replay verification.
    #[arg(long = "tick-limit", value_name = "TICKS", requires = "replay_db")]
    tick_limit: Option<u64>,
    /// Auto-pause when population is at or below this count.
    /// (`SCRIPTBOTS_AUTO_PAUSE_BELOW` supplies the environment-layer equivalent.)
    #[arg(long = "auto-pause-below", value_name = "COUNT")]
    auto_pause_below: Option<u32>,
    /// Auto-pause when any agent's age meets or exceeds this value.
    /// (`SCRIPTBOTS_AUTO_PAUSE_AGE_ABOVE` supplies the environment-layer equivalent.)
    #[arg(long = "auto-pause-age-above", value_name = "AGE")]
    auto_pause_age_above: Option<u32>,
    /// Auto-pause after a spike hit is recorded.
    /// (`SCRIPTBOTS_AUTO_PAUSE_ON_SPIKE` supplies the environment-layer equivalent.)
    #[arg(long = "auto-pause-on-spike", action = ArgAction::SetTrue)]
    auto_pause_on_spike: bool,
    /// Dotted-path configuration override in TOML syntax, repeatable and applied after
    /// configuration files and environment variables (e.g., `--set world_width=800`,
    /// `--set neuroflow.enabled=true`). String values use TOML quoting.
    #[arg(long = "set", value_name = "PATH=VALUE", action = ArgAction::Append)]
    set_overrides: Vec<String>,
    /// Print the composed configuration in the selected format.
    #[arg(long = "print-config", action = ArgAction::SetTrue)]
    print_config: bool,
    /// Write the composed configuration to the provided path (directories created as needed).
    #[arg(long = "write-config", value_name = "FILE")]
    write_config: Option<PathBuf>,
    /// Output format for `--print-config` / `--write-config`.
    #[arg(long = "config-format", value_enum, default_value_t = ConfigFormat::Json)]
    config_format: ConfigFormat,
    /// Exit immediately after emitting configuration output.
    #[arg(long = "config-only", action = ArgAction::SetTrue)]
    config_only: bool,
    /// Capture the bounded characterization trace from tick zero and exit (maximum 256 ticks).
    #[arg(long = "characterize-v0", value_name = "TICKS")]
    characterize_v0: Option<u64>,
    /// Write the characterization trace to this file instead of standard output.
    #[arg(
        long = "characterization-out",
        value_name = "FILE",
        requires = "characterize_v0"
    )]
    characterization_out: Option<PathBuf>,
    /// Run determinism self-check comparing 1-thread vs N-threads for the given number of ticks.
    #[arg(long = "det-check", value_name = "TICKS")]
    det_check: Option<u64>,
    /// Overlay a tiny debug watermark in the render canvas (diagnostics).
    #[arg(long = "debug-watermark", action = ArgAction::SetTrue)]
    debug_watermark: bool,
    /// Force a conservative canvas paint path (diagnostics on Windows black canvas).
    #[arg(long = "renderer-safe", action = ArgAction::SetTrue)]
    renderer_safe: bool,
    /// Cap simulation worker threads (overrides low-power default).
    #[arg(long = "threads", value_name = "N")]
    threads: Option<usize>,
    /// Prefer lower CPU usage (equivalent to --threads 2 unless --threads is provided).
    #[arg(long = "low-power", action = ArgAction::SetTrue)]
    low_power: bool,
    /// Write an offscreen PNG snapshot and exit (no UI).
    #[arg(long = "dump-png", value_name = "FILE")]
    dump_png: Option<PathBuf>,
    /// Write a Bevy offscreen PNG (requires bevy_render feature) and exit (no UI).
    #[cfg(feature = "bevy_render")]
    #[arg(long = "dump-bevy-png", value_name = "FILE")]
    dump_bevy_png: Option<PathBuf>,
    /// Snapshot size for --dump-png, formatted as WIDTHxHEIGHT (e.g., 1280x720).
    #[arg(long = "png-size", value_name = "WxH")]
    png_size: Option<String>,
    /// Storage target: file (default) or memory (same engine, no durable file).
    #[arg(long = "storage", value_enum, default_value_t = StorageMode::File)]
    storage: StorageMode,
    /// Recover and finalize an existing FrankenSQLite outbox, then exit without starting a run.
    #[arg(
        long = "recover-storage",
        value_name = "FILE",
        env = "SCRIPTBOTS_RECOVER_STORAGE",
        exclusive = true
    )]
    recover_storage: Option<PathBuf>,
    /// Profile headless `world.step()` without persistence for N ticks, then exit.
    #[arg(long = "profile-steps", value_name = "TICKS")]
    profile_steps: Option<u64>,
    /// Profile headless `world.step()` with selected storage mode for N ticks, then exit.
    #[arg(long = "profile-storage-steps", value_name = "TICKS")]
    profile_storage_steps: Option<u64>,
    /// Override storage flush thresholds: tick,agent,event,metric (e.g., 64,4096,1024,1024).
    #[arg(long = "storage-thresholds", value_name = "t,a,e,m")]
    storage_thresholds: Option<String>,
    /// Automated profiling sweep: runs multiple configurations for N ticks and summarizes.
    #[arg(long = "profile-sweep", value_name = "TICKS")]
    profile_sweep: Option<u64>,
    /// Auto-tune: quick sweep to pick threads/thresholds for current storage, then continue.
    #[arg(long = "auto-tune", value_name = "TICKS")]
    auto_tune: Option<u64>,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum BrainPreset {
    Mixed,
    Mlp,
    Dwraon,
    Assembly,
    Ft,
}

impl BrainPreset {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Mixed => "mixed",
            Self::Mlp => "mlp",
            Self::Dwraon => "dwraon",
            Self::Assembly => "assembly",
            Self::Ft => "ft",
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum StorageMode {
    File,
    Memory,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum ConfigFormat {
    Json,
    Toml,
    Ron,
}

#[cfg(test)]
fn apply_config_layers(base: ScriptBotsConfig, layers: &[PathBuf]) -> Result<ScriptBotsConfig> {
    let defaults_value = serde_json::to_value(&base).context("failed to serialize base config")?;
    let mut statements = Vec::new();
    for path in layers {
        let (fields, _source_bytes) = load_config_layer_with_source(path)?;
        statements.push(ConfigLayerStatement {
            kind: ConfigLayerKind::File,
            label: format!("file:{}", path.display()),
            fields,
        });
    }
    deserialize_merged_config(&resolve_config_layers(&defaults_value, &statements).merged)
}

fn load_config_layer_with_source(path: &Path) -> Result<(JsonValue, Vec<u8>)> {
    let source_bytes = fs::read(path)
        .with_context(|| format!("failed to read configuration layer {}", path.display()))?;
    let contents = std::str::from_utf8(&source_bytes)
        .with_context(|| format!("configuration layer {} is not valid UTF-8", path.display()))?;

    let value = match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("ron") => ron::from_str(contents)
            .with_context(|| format!("failed to parse RON config layer {}", path.display())),
        _ => toml::from_str(contents)
            .with_context(|| format!("failed to parse TOML config layer {}", path.display())),
    }?;
    Ok((value, source_bytes))
}

fn snapshot_exit_requested(cli: &AppCli) -> bool {
    if cli.dump_png.is_some() {
        return true;
    }
    #[cfg(feature = "bevy_render")]
    if cli.dump_bevy_png.is_some() {
        return true;
    }
    false
}

fn storage_owning_startup_requested(cli: &AppCli) -> bool {
    !cli.config_only
        && cli.replay_db.is_none()
        && cli.det_check.is_none()
        && cli.profile_steps.is_none()
        && cli.profile_storage_steps.is_none()
        && cli.profile_sweep.is_none()
}

fn preflight_renderer_startup(cli: &AppCli) -> Result<Option<(RendererMode, Box<dyn Renderer>)>> {
    #[cfg(not(feature = "gui"))]
    if cli.dump_png.is_some() {
        bail!("--dump-png requires GUI feature; recompile with --features gui");
    }

    if snapshot_exit_requested(cli) {
        Ok(None)
    } else {
        resolve_renderer(cli.mode, cli.low_power).map(Some)
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum RendererMode {
    Auto,
    Gui,
    Bevy,
    Terminal,
}

impl RendererMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gui => "gui",
            Self::Bevy => "bevy",
            Self::Terminal => "terminal",
        }
    }
}

impl fmt::Display for RendererMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RendererAvailability {
    gui: bool,
    bevy: bool,
}

impl RendererAvailability {
    const fn compiled() -> Self {
        Self {
            gui: cfg!(feature = "gui"),
            bevy: cfg!(feature = "bevy_render"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RendererEnvironment {
    force_terminal: bool,
    force_gui: bool,
    graphical_session: bool,
}

impl RendererEnvironment {
    fn from_process(prefer_terminal: bool) -> Result<Self> {
        let force_terminal = renderer_env_flag("SCRIPTBOTS_FORCE_TERMINAL")?;
        let force_gui = renderer_env_flag("SCRIPTBOTS_FORCE_GUI")?;
        Ok(Self {
            force_terminal: force_terminal || (prefer_terminal && !force_gui),
            force_gui,
            graphical_session: native_graphical_session_available(),
        })
    }
}

fn renderer_env_flag(name: &'static str) -> Result<bool> {
    let Some(value) = env::var_os(name) else {
        return Ok(false);
    };
    let value = value
        .to_str()
        .with_context(|| format!("{name} is not valid Unicode"))?;
    parse_bool(value).with_context(|| {
        format!("{name} must be one of true/false, yes/no, on/off, or 1/0; got {value:?}")
    })
}

#[cfg(all(target_family = "unix", not(target_os = "macos")))]
fn native_graphical_session_available() -> bool {
    display_environment_available(env::var_os("DISPLAY"), env::var_os("WAYLAND_DISPLAY"))
}

#[cfg(any(test, all(target_family = "unix", not(target_os = "macos"))))]
fn display_environment_available(
    display: Option<std::ffi::OsString>,
    wayland_display: Option<std::ffi::OsString>,
) -> bool {
    [display, wayland_display]
        .into_iter()
        .flatten()
        .any(|value| !value.is_empty())
}

#[cfg(target_os = "macos")]
fn native_graphical_session_available() -> bool {
    let remote_shell = remote_shell_session_detected([
        env::var_os("SSH_CONNECTION"),
        env::var_os("SSH_CLIENT"),
        env::var_os("SSH_TTY"),
    ]);
    macos_auto_graphical_session_available(
        macos_graphical_session::quartz_session_available(),
        remote_shell,
    )
}

#[cfg(target_os = "macos")]
fn remote_shell_session_detected(values: [Option<std::ffi::OsString>; 3]) -> bool {
    values.into_iter().flatten().any(|value| !value.is_empty())
}

#[cfg(target_os = "macos")]
const fn macos_auto_graphical_session_available(
    quartz_session_available: bool,
    remote_shell: bool,
) -> bool {
    quartz_session_available && !remote_shell
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod macos_graphical_session {
    use std::ffi::c_void;

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {
        fn CGSessionCopyCurrentDictionary() -> *const c_void;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFRelease(value: *const c_void);
    }

    /// Ask Quartz whether this process belongs to a live WindowServer session.
    ///
    /// Apple specifies a retained dictionary on success and null when the
    /// caller is outside a Quartz GUI session or WindowServer is disabled.
    #[allow(unsafe_code)]
    pub(super) fn quartz_session_available() -> bool {
        // SAFETY: both declarations exactly match the CoreGraphics/CoreFoundation
        // C APIs. A non-null Copy result is owned by this call and released once.
        let session = unsafe { CGSessionCopyCurrentDictionary() };
        if session.is_null() {
            false
        } else {
            // SAFETY: `session` is the non-null retained object returned above.
            unsafe { CFRelease(session) };
            true
        }
    }
}

#[cfg(windows)]
fn native_graphical_session_available() -> bool {
    windows_graphical_session::process_window_station_visible()
}

#[cfg(any(test, windows))]
const fn window_station_is_visible(flags: u32) -> bool {
    const WSF_VISIBLE: u32 = 0x0001;
    flags & WSF_VISIBLE != 0
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod windows_graphical_session {
    use std::{ffi::c_void, mem::size_of};

    const UOI_FLAGS: i32 = 1;

    #[repr(C)]
    #[derive(Default)]
    struct UserObjectFlags {
        inherit: i32,
        reserved: i32,
        flags: u32,
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(super) struct WindowStationEvidence {
        pub station_present: bool,
        pub query_succeeded: bool,
        pub flags: u32,
        pub buffer_size: u32,
        pub bytes_needed: u32,
        pub visible: bool,
    }

    #[link(name = "User32")]
    unsafe extern "system" {
        fn GetProcessWindowStation() -> *mut c_void;
        fn GetUserObjectInformationW(
            object: *mut c_void,
            index: i32,
            info: *mut c_void,
            info_len: u32,
            bytes_needed: *mut u32,
        ) -> i32;
    }

    pub(super) fn process_window_station_evidence() -> WindowStationEvidence {
        // SAFETY: `GetProcessWindowStation` takes no arguments and returns a
        // borrowed process-owned handle which Microsoft says must not be closed.
        let station = unsafe { GetProcessWindowStation() };
        if station.is_null() {
            return WindowStationEvidence::default();
        }

        let mut flags = UserObjectFlags::default();
        let mut bytes_needed = 0;
        let buffer_size = size_of::<UserObjectFlags>() as u32;
        // SAFETY: `flags` is a correctly laid-out writable USEROBJECTFLAGS
        // buffer for UOI_FLAGS and both sizes match the Win32 declarations.
        let succeeded = unsafe {
            GetUserObjectInformationW(
                station,
                UOI_FLAGS,
                (&raw mut flags).cast(),
                buffer_size,
                &raw mut bytes_needed,
            )
        } != 0;

        WindowStationEvidence {
            station_present: true,
            query_succeeded: succeeded,
            flags: flags.flags,
            buffer_size,
            bytes_needed,
            visible: succeeded && super::window_station_is_visible(flags.flags),
        }
    }

    pub(super) fn process_window_station_visible() -> bool {
        process_window_station_evidence().visible
    }
}

#[cfg(all(not(target_family = "unix"), not(windows)))]
const fn native_graphical_session_available() -> bool {
    false
}

fn select_renderer_mode(
    requested: RendererMode,
    available: RendererAvailability,
    environment: RendererEnvironment,
) -> Result<RendererMode> {
    match requested {
        RendererMode::Terminal => Ok(RendererMode::Terminal),
        RendererMode::Gui if available.gui => Ok(RendererMode::Gui),
        RendererMode::Gui => {
            bail!("--mode gui requires a binary built with --features gui")
        }
        RendererMode::Bevy if available.bevy => Ok(RendererMode::Bevy),
        RendererMode::Bevy => {
            bail!("--mode bevy requires a binary built with --features bevy_render")
        }
        RendererMode::Auto => {
            if environment.force_terminal && environment.force_gui {
                bail!("SCRIPTBOTS_FORCE_TERMINAL and SCRIPTBOTS_FORCE_GUI cannot both be enabled");
            }
            if environment.force_terminal {
                return Ok(RendererMode::Terminal);
            }
            if environment.force_gui {
                if available.gui {
                    return Ok(RendererMode::Gui);
                }
                bail!(
                    "SCRIPTBOTS_FORCE_GUI requires a binary built with --features gui; refusing to substitute another renderer"
                );
            }
            if environment.graphical_session {
                if available.gui {
                    return Ok(RendererMode::Gui);
                }
                if available.bevy {
                    return Ok(RendererMode::Bevy);
                }
            }
            Ok(RendererMode::Terminal)
        }
    }
}

fn resolve_renderer(
    mode: RendererMode,
    prefer_terminal: bool,
) -> Result<(RendererMode, Box<dyn Renderer>)> {
    let environment = if mode == RendererMode::Auto {
        RendererEnvironment::from_process(prefer_terminal)?
    } else {
        RendererEnvironment::default()
    };
    let active = select_renderer_mode(mode, RendererAvailability::compiled(), environment)?;
    let renderer: Box<dyn Renderer> = match active {
        RendererMode::Terminal => Box::new(TerminalRenderer::default()),
        RendererMode::Gui => {
            #[cfg(feature = "gui")]
            {
                Box::new(GuiRenderer)
            }
            #[cfg(not(feature = "gui"))]
            {
                bail!("internal error: selected GPUI without the gui feature")
            }
        }
        RendererMode::Bevy => {
            #[cfg(feature = "bevy_render")]
            {
                Box::new(BevyRenderer)
            }
            #[cfg(not(feature = "bevy_render"))]
            {
                bail!("internal error: selected Bevy without the bevy_render feature")
            }
        }
        RendererMode::Auto => bail!("internal error: unresolved automatic renderer mode"),
    };
    debug!(
        requested_mode = mode.as_str(),
        active_mode = active.as_str(),
        "Resolved renderer mode"
    );
    Ok((active, renderer))
}

#[cfg(feature = "gui")]
#[derive(Default)]
struct GuiRenderer;

#[cfg(feature = "gui")]
impl Renderer for GuiRenderer {
    fn name(&self) -> &'static str {
        "gpui"
    }

    fn run(&self, ctx: RendererContext<'_>) -> Result<()> {
        prepare_linux_gui_backend();
        let control_health: scriptbots_render::GuiHealthProbe =
            Arc::new(ctx.control_runtime.health_probe());
        run_demo(
            Arc::clone(&ctx.world),
            Arc::clone(&ctx.simulation_step),
            ctx.analytics.clone(),
            Arc::clone(&ctx.command_drain),
            Arc::clone(&ctx.command_submit),
            control_health,
        )
        .map_err(anyhow::Error::new)
    }
}

#[cfg(feature = "bevy_render")]
#[derive(Default)]
struct BevyRenderer;

#[cfg(feature = "bevy_render")]
impl Renderer for BevyRenderer {
    fn name(&self) -> &'static str {
        "bevy"
    }

    fn run(&self, ctx: RendererContext<'_>) -> Result<()> {
        prepare_linux_gui_backend();
        let control_health: scriptbots_bevy::ControlHealthFn =
            Arc::new(ctx.control_runtime.health_probe());
        let bevy_ctx = BevyRendererContext {
            world: Arc::clone(&ctx.world),
            simulation_step: Arc::clone(&ctx.simulation_step),
            command_submit: Arc::clone(&ctx.command_submit),
            command_drain: Arc::clone(&ctx.command_drain),
            control_health: Some(control_health),
        };
        scriptbots_bevy::run_renderer(bevy_ctx)
    }
}

#[cfg(all(target_os = "linux", any(feature = "gui", feature = "bevy_render")))]
fn prepare_linux_gui_backend() {
    use std::sync::Once;

    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if let Err(error) = maybe_force_x11_for_legacy_wayland() {
            tracing::debug!(%error, "Wayland backend probe failed; leaving backend selection unchanged");
        }
    });
}

#[cfg(all(
    not(target_os = "linux"),
    any(feature = "gui", feature = "bevy_render")
))]
fn prepare_linux_gui_backend() {}

#[cfg(all(target_os = "linux", any(feature = "gui", feature = "bevy_render")))]
fn maybe_force_x11_for_legacy_wayland() -> Result<()> {
    use std::env;
    use wayland_client::{
        Connection, Dispatch, Proxy, QueueHandle,
        globals::{GlobalListContents, registry_queue_init},
        protocol::{wl_compositor, wl_registry, wl_surface},
    };

    struct RegistryProbe;

    impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for RegistryProbe {
        fn event(
            _state: &mut Self,
            _proxy: &wl_registry::WlRegistry,
            _event: wl_registry::Event,
            _data: &GlobalListContents,
            _conn: &Connection,
            _qh: &QueueHandle<Self>,
        ) {
        }
    }

    // Respect explicit backend overrides or GUI forcing.
    if env::var_os("WINIT_UNIX_BACKEND").is_some() {
        return Ok(());
    }
    if matches!(
        env::var("SCRIPTBOTS_FORCE_GUI")
            .ok()
            .and_then(|value| parse_bool(&value)),
        Some(true)
    ) {
        return Ok(());
    }

    let Some(display) = env::var_os("WAYLAND_DISPLAY") else {
        return Ok(());
    };
    if display.is_empty() {
        return Ok(());
    }

    let connection = Connection::connect_to_env()?;
    let (globals, _queue) = registry_queue_init::<RegistryProbe>(&connection)?;
    let required = wl_surface::REQ_SET_BUFFER_SCALE_SINCE;

    let compositor_version = globals.contents().with_list(|globals| {
        globals
            .iter()
            .find(|global| global.interface == wl_compositor::WlCompositor::interface().name)
            .map(|global| global.version)
    });

    if let Some(version) = compositor_version
        && version < required
    {
        // SAFETY: Modifying the process environment before spawning GUI worker threads.
        unsafe {
            env::set_var("WINIT_UNIX_BACKEND", "x11");
        }
        tracing::warn!(
            version,
            required,
            "Wayland compositor version too old (v{version}); forcing WINIT_UNIX_BACKEND=x11. Set SCRIPTBOTS_FORCE_GUI=1 to override."
        );
    }

    Ok(())
}

fn run_replay_cli(cli: &AppCli, config: &ScriptBotsConfig) -> Result<()> {
    let db_path = cli
        .replay_db
        .as_ref()
        .expect("replay_db required to enter replay mode");
    let db_display = db_path.display().to_string();

    let storage = StorageReader::open(&db_display)
        .with_context(|| format!("failed to open replay database {db_display}"))?;
    let recorded_max_tick = storage.max_tick()?.unwrap_or(0);
    let persisted_events = storage.load_replay_events()?;
    let recorded_counts = storage.replay_event_counts()?;
    storage.close()?;

    let events_max_tick = persisted_events.iter().map(|e| e.tick).max().unwrap_or(0);
    let tick_limit = cli
        .tick_limit
        .unwrap_or(recorded_max_tick.max(events_max_tick));

    if config.rng_seed.is_none() {
        warn!(
            "config_rng_seed" = false,
            "Replay config has no rng_seed; deterministic verification may fail"
        );
    }

    let replay_run = run_headless_simulation(config, tick_limit, cli.brain)?;
    let simulated_tick_count = replay_run.simulated_ticks;
    debug!(
        simulated_ticks = simulated_tick_count,
        simulated_events = replay_run.events.len(),
        "Headless replay complete"
    );
    if simulated_tick_count != tick_limit {
        warn!(
            requested_ticks = tick_limit,
            simulated_ticks = simulated_tick_count,
            "Simulated tick count differs from requested limit"
        );
    }
    // The current run database has no canonical replay-digest stream yet. Pass
    // that absence explicitly so event equality cannot be promoted into a
    // replay-verification success without the later digest instrumentation.
    require_non_vacuous_replay(
        tick_limit,
        persisted_events.len(),
        replay_run.events.len(),
        0,
        0,
    )?;
    let diff = diff_event_stream(&persisted_events, &replay_run.events);

    let recorded_map = recorded_counts
        .into_iter()
        .map(|entry| (entry.event_type, entry.count))
        .collect::<HashMap<_, _>>();
    let simulated_counts = count_event_kinds(&replay_run.events)
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect::<HashMap<_, _>>();

    // Deterministic key ordering for printed output
    let mut recorded_sorted: Vec<_> = recorded_map.iter().collect();
    recorded_sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut simulated_sorted: Vec<_> = simulated_counts.iter().collect();
    simulated_sorted.sort_by(|a, b| a.0.cmp(b.0));

    println!(
        "{} Replaying {} ticks ({} recorded events) against {}",
        "▶".bright_blue().bold(),
        tick_limit,
        persisted_events.len(),
        db_display.cyan()
    );
    print_event_counts("recorded", &recorded_map, None);
    print_event_counts("simulated", &simulated_counts, Some(&recorded_map));

    if let Some(divergence) = diff {
        report_divergence("recorded", "simulated", divergence)?;
    } else {
        println!(
            "{} Replay matched {} events across {} ticks",
            "✔".green().bold(),
            replay_run.events.len().green(),
            simulated_tick_count.green()
        );
    }

    if let Some(compare_path) = cli.compare_db.as_ref() {
        let compare_display = compare_path.display().to_string();
        let other = StorageReader::open(&compare_display)
            .with_context(|| format!("failed to open comparison database {compare_display}"))?;
        let other_events = other.load_replay_events()?;
        let other_counts = other.replay_event_counts()?;
        other.close()?;

        println!(
            "{} Comparing {} against {}",
            "▶".bright_blue().bold(),
            db_display.cyan(),
            compare_display.cyan()
        );
        let compare_diff = diff_event_stream(&persisted_events, &other_events);
        let other_map = other_counts
            .into_iter()
            .map(|entry| (entry.event_type, entry.count))
            .collect::<HashMap<_, _>>();
        print_event_counts("baseline", &recorded_map, None);
        print_event_counts("candidate", &other_map, Some(&recorded_map));

        if let Some(divergence) = compare_diff {
            report_divergence("baseline", "candidate", divergence)?;
        } else {
            println!(
                "{} Event streams are identical ({} events)",
                "✔".green().bold(),
                other_events.len().green()
            );
        }
    }

    Ok(())
}

fn require_non_vacuous_replay(
    tick_limit: u64,
    recorded_events: usize,
    simulated_events: usize,
    recorded_digests: usize,
    simulated_digests: usize,
) -> Result<()> {
    if tick_limit > 0
        && (recorded_events == 0
            || simulated_events == 0
            || recorded_digests == 0
            || simulated_digests == 0)
    {
        bail!(
            "replay verification refused a vacuous nonzero run: events recorded={recorded_events} simulated={simulated_events}; digests recorded={recorded_digests} simulated={simulated_digests}; ticks={tick_limit}. Production replay event and canonical digest instrumentation must both provide nonempty evidence"
        );
    }
    Ok(())
}

struct ReplayCollector {
    ticks: Arc<Mutex<Vec<ReplayTickRecord>>>,
}

impl ReplayCollector {
    fn with_capacity(capacity: usize) -> (Self, Arc<Mutex<Vec<ReplayTickRecord>>>) {
        let ticks = Arc::new(Mutex::new(Vec::with_capacity(capacity)));
        (
            Self {
                ticks: Arc::clone(&ticks),
            },
            ticks,
        )
    }

    fn new() -> (Self, Arc<Mutex<Vec<ReplayTickRecord>>>) {
        Self::with_capacity(0)
    }
}

impl WorldPersistence for ReplayCollector {
    fn on_tick(
        &mut self,
        payload: &scriptbots_core::PersistenceBatch,
    ) -> Result<(), scriptbots_core::PersistenceAdmissionError> {
        let mut guard = self.ticks.lock().map_err(|error| {
            scriptbots_core::PersistenceAdmissionError::new(
                payload.summary.tick.0,
                format!("replay collector lock poisoned: {error}"),
            )
        })?;
        guard.push(ReplayTickRecord {
            tick: payload.summary.tick.0,
            events: payload.replay_events.clone(),
            summary: payload.summary.clone(),
        });
        Ok(())
    }
}

struct ReplayTickRecord {
    tick: u64,
    events: Vec<scriptbots_core::ReplayEvent>,
    summary: TickSummary,
}

struct ReplayRun {
    events: Vec<PersistedReplayEvent>,
    summaries: Vec<TickSummary>,
    simulated_ticks: u64,
}

fn run_headless_simulation(
    config: &ScriptBotsConfig,
    tick_limit: u64,
    brain_preset: BrainPreset,
) -> Result<ReplayRun> {
    let (collector, handle) = ReplayCollector::with_capacity(tick_limit as usize);
    let (mut world, mut persistence) =
        WorldState::with_persistence(config.clone(), Box::new(collector))?;
    let brain_keys = install_brains(&mut world, brain_preset)?.population;
    seed_agents(&mut world, &brain_keys)?;

    emit_sense_startup_contract();
    let simulation_result = (|| -> Result<()> {
        for _ in 0..tick_limit {
            persistence.step(&mut world)?;
        }
        persistence
            .finalize(&mut world)
            .context("failed to admit the final partial replay batch")?;
        Ok(())
    })();
    let sense_summary = SenseRunSummary::capture(&world);
    emit_sense_run_end(sense_summary, simulation_result.is_ok());
    simulation_result?;
    drop(world);
    drop(persistence);

    let records = Arc::try_unwrap(handle)
        .map_err(|_| anyhow::anyhow!("replay collector still in use"))?
        .into_inner()
        .map_err(|err| anyhow::anyhow!("replay collector poisoned: {err}"))?;

    let mut events = Vec::new();
    let mut summaries = Vec::with_capacity(records.len());
    for record in records {
        summaries.push(record.summary);
        for (seq, event) in record.events.into_iter().enumerate() {
            events.push(PersistedReplayEvent {
                tick: record.tick,
                seq: seq as u64,
                event,
            });
        }
    }

    Ok(ReplayRun {
        events,
        summaries,
        simulated_ticks: tick_limit,
    })
}

fn profile_world_steps(
    config: &ScriptBotsConfig,
    tick_limit: u64,
    brain_preset: BrainPreset,
) -> Result<()> {
    let (collector, _handle) = ReplayCollector::new();
    let (mut world, mut persistence) =
        WorldState::with_persistence(config.clone(), Box::new(collector))?;
    let brain_keys = install_brains(&mut world, brain_preset)?.population;
    seed_agents(&mut world, &brain_keys)?;

    emit_sense_startup_contract();
    let start = Instant::now();
    let result = (|| -> Result<()> {
        for _ in 0..tick_limit {
            persistence.step(&mut world)?;
        }
        Ok(())
    })();
    emit_sense_run_end(SenseRunSummary::capture(&world), result.is_ok());
    result?;
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64().max(1e-9);
    let tps = tick_limit as f64 / secs;
    println!(
        "{} Headless no-storage: {} ticks in {:.3}s ({:.0} tps)",
        "✔".green().bold(),
        tick_limit,
        secs,
        tps
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn profile_world_steps_with_storage(
    config: &ScriptBotsConfig,
    tick_limit: u64,
    brain_preset: BrainPreset,
    storage_mode: StorageMode,
    thresholds: ThresholdsOverride,
    thread_policy: ThreadPolicy,
    mut scenario: ScenarioIdentityV0,
    config_overrides: Vec<ConfigFieldOverride>,
) -> Result<()> {
    // Storage profiling performs only the requested measured steps; it has no startup warmup.
    scenario.bootstrap_ticks = 0;
    let mut run_config = config.clone();
    let (root_seed, generated_seed) = materialize_run_seed(&mut run_config);
    if generated_seed {
        info!(
            root_seed,
            "Generated and pinned the storage profile's previously unspecified scientific seed"
        );
    }

    #[cfg(feature = "neuro")]
    if brain_preset == BrainPreset::Mixed {
        let _ = validated_neuroflow_config(&run_config)?;
    }

    // Materialize the complete launch state before opening storage. The database registration
    // below therefore describes the exact seeded roster that will produce tick one, and a
    // configuration/brain error cannot leave a reserved run file behind.
    let mut world = WorldState::new(run_config)
        .context("failed to construct storage profile before tick zero")?;
    let brain_keys = install_brains(&mut world, brain_preset)?.population;
    seed_agents(&mut world, &brain_keys)?;

    let started_at_unix_ms = run_started_at_unix_ms()?;
    let identity = RunIdentityV1::new(
        allocate_run_id(),
        started_at_unix_ms,
        Some(tick_limit),
        None,
    );
    identity
        .validate()
        .context("invalid finite storage-profile identity before registration")?;
    let manifest = build_run_manifest(&world, identity, scenario, thread_policy, config_overrides)
        .context("failed to build storage-profile provenance before tick zero")?;
    let storage_record = manifest
        .to_storage_record()
        .context("failed to project storage-profile provenance before tick zero")?;

    let mut pipeline = match storage_mode {
        StorageMode::File => {
            let storage_path = storage_path_from_env()?;
            match (
                thresholds.tick,
                thresholds.agent,
                thresholds.event,
                thresholds.metric,
            ) {
                (Some(t), Some(a), Some(e), Some(m)) => {
                    StoragePipeline::create_new_file_for_run_with_thresholds(
                        &storage_path,
                        storage_record,
                        t,
                        a,
                        e,
                        m,
                    )
                }
                _ => StoragePipeline::create_new_file_for_run(&storage_path, storage_record),
            }
            .with_context(|| {
                format!(
                    "failed to register storage-profile provenance and initialize FrankenSQLite storage at {storage_path}"
                )
            })?
        }
        StorageMode::Memory => StoragePipeline::memory_for_run_with_thresholds(
            storage_record,
            thresholds.tick.unwrap_or(64),
            thresholds.agent.unwrap_or(2048),
            thresholds.event.unwrap_or(512),
            thresholds.metric.unwrap_or(512),
        )
        .context("failed to register storage-profile provenance in volatile storage")?,
    };

    let mut persistence = match world.bind_persistence(Box::new(pipeline.sink())) {
        Ok(persistence) => persistence,
        Err(error) => {
            return finish_with_storage(
                Err(error.into()),
                "profile world persistence binding",
                || shutdown_storage(&mut pipeline).map(|_| ()),
            );
        }
    };
    emit_sense_startup_contract();
    let start = Instant::now();
    let profile_result = (|| -> Result<()> {
        for _ in 0..tick_limit {
            persistence.step(&mut world)?;
        }
        Ok(())
    })();
    let result = finish_with_storage(profile_result, "storage profiling", || {
        let finalization = finalize_world_persistence(&mut world, &mut persistence);
        finalize_then_shutdown_storage(finalization, &mut pipeline)
    });
    emit_sense_run_end(SenseRunSummary::capture(&world), result.is_ok());
    result?;
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64().max(1e-9);
    let tps = tick_limit as f64 / secs;
    println!(
        "{} Headless with-storage({}): {} ticks in {:.3}s ({:.0} tps)",
        "✔".green().bold(),
        match storage_mode {
            StorageMode::File => "file",
            StorageMode::Memory => "memory",
        },
        tick_limit,
        secs,
        tps
    );
    Ok(())
}

fn parse_tps_from_stdout(stdout: &[u8]) -> Option<f64> {
    let s = std::str::from_utf8(stdout).ok()?;
    // Expect a line ending with "(NNN tps)"; grab the last number before " tps)"
    for line in s.lines().rev() {
        if let Some(idx) = line.rfind(" tps)") {
            let start = line[..idx].rfind('(')? + 1;
            let num_str = &line[start..idx];
            if let Ok(val) = num_str.trim().parse::<f64>() {
                return Some(val);
            }
        }
    }
    None
}

fn configure_profile_child_storage(command: &mut Command, storage: StorageMode) {
    match storage {
        StorageMode::File => {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos());
            let path_nonce = allocate_run_id();
            let path = env::temp_dir().join(format!(
                "scriptbots-profile-{}-{timestamp}-{path_nonce}.sqlite",
                std::process::id()
            ));
            command.env("SCRIPTBOTS_STORAGE_PATH", path);
        }
        StorageMode::Memory => {
            command.env_remove("SCRIPTBOTS_STORAGE_PATH");
        }
    }
}

fn run_profile_sweep(_config: &ScriptBotsConfig, ticks: u64, cli: &AppCli) -> Result<()> {
    let exe = std::env::current_exe().context("failed to get current exe path")?;

    // Candidate configurations
    let thread_candidates: Vec<usize> = if let Some(t) = cli.threads {
        vec![t]
    } else {
        vec![1, 2, 4, 8]
    };
    let storage_candidates = [StorageMode::Memory, StorageMode::File];
    let threshold_candidates: Vec<&str> = vec![
        "64,2048,512,512",
        "128,4096,1024,1024",
        "256,4096,2048,1024",
    ];

    #[derive(Clone)]
    struct ResultRow {
        threads: usize,
        storage: StorageMode,
        thresholds: &'static str,
        tps: f64,
    }

    let mut results: Vec<ResultRow> = Vec::new();

    for threads in thread_candidates {
        for &storage in &storage_candidates {
            let threshold_list: Vec<&str> = match storage {
                StorageMode::Memory => threshold_candidates.clone(),
                StorageMode::File => threshold_candidates.clone(),
            };
            for thresholds in threshold_list {
                let storage_label = match storage {
                    StorageMode::File => "file",
                    StorageMode::Memory => "memory",
                };
                let mut cmd = Command::new(&exe);
                cmd.env("SCRIPTBOTS_DET_RUN", "0");
                cmd.env("RUST_LOG", "error");
                configure_profile_child_storage(&mut cmd, storage);
                cmd.arg("--profile-storage-steps").arg(ticks.to_string());
                cmd.arg("--brain").arg(cli.brain.as_str());
                cmd.arg("--storage").arg(storage_label);
                cmd.arg("--storage-thresholds").arg(thresholds);
                cmd.arg("--threads").arg(threads.to_string());
                if cli.low_power {
                    cmd.arg("--low-power");
                }
                cmd.stdout(Stdio::piped());
                cmd.stderr(Stdio::null());
                let out = cmd.output().with_context(|| {
                    format!(
                        "sweep run failed (thr={threads}, storage={storage_label}, thres={thresholds})"
                    )
                })?;
                if !out.status.success() {
                    continue;
                }
                if let Some(tps) = parse_tps_from_stdout(&out.stdout) {
                    // thresholds is &'static str via const literals
                    let thresholds_static: &'static str = match thresholds {
                        "64,2048,512,512" => "64,2048,512,512",
                        "128,4096,1024,1024" => "128,4096,1024,1024",
                        _ => "256,4096,2048,1024",
                    };
                    results.push(ResultRow {
                        threads,
                        storage,
                        thresholds: thresholds_static,
                        tps,
                    });
                }
            }
        }
    }

    results.sort_by(|a, b| {
        b.tps
            .partial_cmp(&a.tps)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!(
        "{} Automated profile sweep ({} ticks):",
        "▶".bright_blue().bold(),
        ticks
    );
    for row in results.iter().take(8) {
        println!(
            "    threads={:<2} storage={:<6} thresholds={:<20} {:>8.0} tps",
            row.threads,
            match row.storage {
                StorageMode::File => "file",
                StorageMode::Memory => "memory",
            },
            row.thresholds,
            row.tps
        );
    }

    if let Some(best) = results.first() {
        println!(
            "{} Best: threads={} storage={} thresholds={} ({:.0} tps)",
            "✔".green().bold(),
            best.threads,
            match best.storage {
                StorageMode::File => "file",
                StorageMode::Memory => "memory",
            },
            best.thresholds,
            best.tps
        );
    } else {
        println!("{} No successful sweep results", "✖".red().bold());
    }
    Ok(())
}

struct BestPick {
    threads: usize,
    tick: usize,
    agent: usize,
    event: usize,
    metric: usize,
    tps: f64,
}

fn pick_best_for_storage(
    _config: &ScriptBotsConfig,
    ticks: u64,
    brain_preset: BrainPreset,
    storage: StorageMode,
    pinned_threads: Option<usize>,
    low_power: bool,
) -> Result<Option<BestPick>> {
    let exe = std::env::current_exe().context("failed to get current exe path")?;

    let thread_candidates: Vec<usize> = if let Some(t) = pinned_threads {
        vec![t]
    } else {
        vec![1, 2, 4, 8]
    };
    let threshold_candidates: Vec<&str> = vec![
        "64,2048,512,512",
        "128,4096,1024,1024",
        "256,4096,2048,1024",
    ];
    let mut best: Option<BestPick> = None;

    for threads in thread_candidates {
        for thresholds in &threshold_candidates {
            let mut cmd = Command::new(&exe);
            cmd.env("SCRIPTBOTS_DET_RUN", "0");
            cmd.env("RUST_LOG", "error");
            configure_profile_child_storage(&mut cmd, storage);
            cmd.arg("--profile-storage-steps").arg(ticks.to_string());
            cmd.arg("--brain").arg(brain_preset.as_str());
            cmd.arg("--storage").arg(match storage {
                StorageMode::File => "file",
                StorageMode::Memory => "memory",
            });
            cmd.arg("--storage-thresholds").arg(thresholds);
            cmd.arg("--threads").arg(threads.to_string());
            if low_power {
                cmd.arg("--low-power");
            }
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());
            let out = cmd.output()?;
            if !out.status.success() {
                continue;
            }
            if let Some(tps) = parse_tps_from_stdout(&out.stdout) {
                let parts: Vec<_> = thresholds.split(',').collect();
                if parts.len() == 4
                    && let (Ok(tk), Ok(ag), Ok(ev), Ok(me)) = (
                        parts[0].parse(),
                        parts[1].parse(),
                        parts[2].parse(),
                        parts[3].parse(),
                    )
                {
                    let candidate = BestPick {
                        threads,
                        tick: tk,
                        agent: ag,
                        event: ev,
                        metric: me,
                        tps,
                    };
                    match &best {
                        Some(b) if b.tps >= candidate.tps => {}
                        _ => {
                            best = Some(candidate);
                        }
                    }
                }
            }
        }
    }
    Ok(best)
}

#[derive(Debug)]
struct Divergence {
    kind: DivergenceKind,
    expected: Option<PersistedReplayEvent>,
    actual: Option<PersistedReplayEvent>,
}

#[derive(Debug)]
enum DivergenceKind {
    TickMismatch,
    SequenceMismatch,
    EventMismatch,
    MissingActual,
    ExtraActual,
}

fn diff_event_stream(
    expected: &[PersistedReplayEvent],
    actual: &[PersistedReplayEvent],
) -> Option<Divergence> {
    let mut idx = 0;
    loop {
        match (expected.get(idx), actual.get(idx)) {
            (Some(left), Some(right)) => {
                if left.tick != right.tick {
                    return Some(Divergence {
                        kind: DivergenceKind::TickMismatch,
                        expected: Some(left.clone()),
                        actual: Some(right.clone()),
                    });
                }
                if left.seq != right.seq {
                    return Some(Divergence {
                        kind: DivergenceKind::SequenceMismatch,
                        expected: Some(left.clone()),
                        actual: Some(right.clone()),
                    });
                }
                if left.event != right.event {
                    return Some(Divergence {
                        kind: DivergenceKind::EventMismatch,
                        expected: Some(left.clone()),
                        actual: Some(right.clone()),
                    });
                }
            }
            (Some(left), None) => {
                return Some(Divergence {
                    kind: DivergenceKind::MissingActual,
                    expected: Some(left.clone()),
                    actual: None,
                });
            }
            (None, Some(right)) => {
                return Some(Divergence {
                    kind: DivergenceKind::ExtraActual,
                    expected: None,
                    actual: Some(right.clone()),
                });
            }
            (None, None) => return None,
        }
        idx += 1;
    }
}

fn count_event_kinds(events: &[PersistedReplayEvent]) -> HashMap<&'static str, u64> {
    let mut counts = HashMap::new();
    for entry in events {
        let key = match entry.event.kind {
            ReplayEventKind::BrainOutputs { .. } => "brain_outputs",
            ReplayEventKind::Action { .. } => "action",
            ReplayEventKind::RngSample { .. } => "rng_sample",
        };
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn report_divergence(left_label: &str, right_label: &str, divergence: Divergence) -> Result<()> {
    let marker = format!("{}", "✖".red().bold());
    match divergence.kind {
        DivergenceKind::TickMismatch => {
            if let (Some(exp), Some(act)) = (&divergence.expected, &divergence.actual) {
                println!(
                    "{marker} Tick mismatch: {left_label} tick {} vs {right_label} tick {}",
                    exp.tick.red(),
                    act.tick.red()
                );
            }
        }
        DivergenceKind::SequenceMismatch => {
            if let (Some(exp), Some(act)) = (&divergence.expected, &divergence.actual) {
                println!(
                    "{marker} Sequence mismatch at tick {}: {left_label} seq {} vs {right_label} seq {}",
                    exp.tick.red(),
                    exp.seq.red(),
                    act.seq.red()
                );
            }
        }
        DivergenceKind::EventMismatch => {
            if let (Some(exp), Some(act)) = (&divergence.expected, &divergence.actual) {
                println!(
                    "{marker} Event mismatch at tick {} seq {}",
                    exp.tick.red(),
                    exp.seq.red()
                );
                println!("    expected: {}", format_replay_event(&exp.event).yellow());
                println!("    actual:   {}", format_replay_event(&act.event).yellow());
            }
        }
        DivergenceKind::MissingActual => {
            if let Some(exp) = divergence.expected {
                println!(
                    "{marker} {right_label} stream ended before event tick {} seq {}",
                    exp.tick.red(),
                    exp.seq.red()
                );
            }
        }
        DivergenceKind::ExtraActual => {
            if let Some(act) = divergence.actual {
                println!(
                    "{marker} {right_label} has extra event at tick {} seq {}",
                    act.tick.red(),
                    act.seq.red()
                );
            }
        }
    }

    bail!("replay divergence detected")
}

fn format_replay_event(event: &scriptbots_core::ReplayEvent) -> String {
    match &event.kind {
        ReplayEventKind::BrainOutputs { outputs } => format!(
            "BrainOutputs(agent={:?}, len={})",
            event.agent_uid,
            outputs.len()
        ),
        ReplayEventKind::Action {
            left_wheel,
            right_wheel,
            boost,
            spike_target,
            sound_level,
            give_intent,
        } => format!(
            "Action(agent={:?}, lw={:.3}, rw={:.3}, boost={}, spike={:?}, sound={:.3}, give={:.3})",
            event.agent_uid, left_wheel, right_wheel, boost, spike_target, sound_level, give_intent
        ),
        ReplayEventKind::RngSample {
            scope,
            range_min,
            range_max,
            value,
        } => format!(
            "RngSample(scope={:?}, min={:.3}, max={:.3}, value={:.3})",
            scope, range_min, range_max, value
        ),
    }
}

fn print_event_counts(
    label: &str,
    counts: &HashMap<String, u64>,
    reference: Option<&HashMap<String, u64>>,
) {
    let keys = ["brain_outputs", "action", "rng_sample"];
    println!("  {}", label.cyan().bold());
    for key in keys {
        let value = counts.get(key).copied().unwrap_or(0);
        if let Some(baseline) = reference {
            let baseline_value = baseline.get(key).copied().unwrap_or(0);
            let delta = value as i64 - baseline_value as i64;
            let delta_fmt = format!("Δ {delta:+}");
            let delta_colored = if delta == 0 {
                format!("{}", delta_fmt.yellow())
            } else if delta > 0 {
                format!("{}", delta_fmt.green())
            } else {
                format!("{}", delta_fmt.red())
            };
            println!("    {:<14} {:>8} ({delta_colored})", key, value);
        } else {
            println!("    {:<14} {:>8}", key, value);
        }
    }
}

#[cfg(feature = "neuro")]
fn validated_neuroflow_config(
    config: &ScriptBotsConfig,
) -> Result<Option<scriptbots_brain_neuro::NeuroflowBrainConfig>> {
    use scriptbots_brain_neuro::NeuroflowBrainConfig;

    if !config.neuroflow.enabled {
        return Ok(None);
    }
    let adapter = NeuroflowBrainConfig::from_settings(&config.neuroflow);
    adapter
        .validate()
        .context("failed to validate configured NeuroFlow brain")?;
    Ok(Some(adapter))
}

impl InstalledBrains {
    /// How many families are REGISTERED — eligible plus withheld.
    ///
    /// Registration and population-eligibility are now different questions, and conflating them
    /// is what this bead is about. A withheld family is still registered: it can be bound
    /// explicitly by an experiment that genuinely wants it. It simply may not found a
    /// population until it implements the versioned genome and evaluator-state protocol.
    fn registered(&self) -> usize {
        self.population.len() + self.withheld.len()
    }
}

/// Registered brain families, split by whether they may found a population.
#[derive(Debug)]
struct InstalledBrains {
    /// Versioned protocol families admitted to seed the founding population.
    population: Vec<u64>,
    /// Families registered for explicit selection but withheld from default populations,
    /// with the reason, so the exclusion is inspectable rather than folklore.
    withheld: Vec<(String, u64)>,
}

fn install_brains(world: &mut WorldState, preset: BrainPreset) -> Result<InstalledBrains> {
    #[cfg(feature = "neuro")]
    let neuro_config = if preset == BrainPreset::Mixed {
        validated_neuroflow_config(world.config())?
    } else {
        None
    };

    let mut withheld = Vec::new();
    let mut population = Vec::new();

    // Founding-population admission is structural: every eligible entry must own a versioned
    // genome codec, evaluator-state codec, offspring-state policy, and evaluator constructor.
    // There is no legacy runner beside these entries that could become a second hereditary truth.
    let register_mlp = |world: &mut WorldState| {
        world
            .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
            .context("failed to register the versioned MLP brain family")
    };
    let register_dwraon = |world: &mut WorldState| {
        world
            .register_brain_family(
                DwraonBrain::KIND.as_str(),
                Box::new(DwraonFamilyAdapter::default()),
            )
            .context("failed to register the versioned DWRAON brain family")
    };
    let register_assembly = |world: &mut WorldState| {
        let assembly = AssemblyFamilyAdapter::new()
            .context("failed to construct the versioned Assembly brain family")?;
        world
            .register_brain_family(AssemblyBrain::KIND.as_str(), Box::new(assembly))
            .context("failed to register the versioned Assembly brain family")
    };
    #[cfg(feature = "brain-ft")]
    let register_ft = |world: &mut WorldState| {
        world
            .register_brain_family(FT_BRAIN_KIND, Box::new(FtBrainFamily::default()))
            .context("failed to register the versioned Frankentorch brain family")
    };

    match preset {
        BrainPreset::Mixed => {
            population.push(register_mlp(world)?);
            population.push(register_dwraon(world)?);
            population.push(register_assembly(world)?);
            #[cfg(feature = "brain-ft")]
            population.push(register_ft(world)?);
        }
        BrainPreset::Mlp => population.push(register_mlp(world)?),
        BrainPreset::Dwraon => population.push(register_dwraon(world)?),
        BrainPreset::Assembly => population.push(register_assembly(world)?),
        BrainPreset::Ft => {
            #[cfg(feature = "brain-ft")]
            population.push(register_ft(world)?);
            #[cfg(not(feature = "brain-ft"))]
            bail!(
                "brain preset `ft` requires a scriptbots-app build with the non-default \
                 `brain-ft` feature"
            );
        }
    }

    #[cfg(feature = "neuro")]
    if preset == BrainPreset::Mixed {
        use scriptbots_brain_neuro::NeuroflowBrain;
        if let Some(config) = neuro_config {
            let key = NeuroflowBrain::register(world, config)
                .context("failed to register configured NeuroFlow brain")?;
            let label = world
                .brain_registry()
                .kind(key)
                .unwrap_or("neuroflow")
                .to_owned();
            warn!(
                brain = %label,
                key,
                "NeuroFlow remains available as an explicitly selected legacy runner, but it \
                 has no versioned genome/evaluator-state protocol codec and is WITHHELD from \
                 the founding population. Admitting it would reintroduce an opaque hereditary \
                 state beside the canonical protocol families."
            );
            withheld.push((label, key));
        }
    }

    let installed = InstalledBrains {
        population,
        withheld,
    };

    // One consolidated line, so the composition of the founding population is a matter of
    // record rather than something a reader has to reconstruct from scattered warnings.
    // A run whose brain roster differs from what the operator assumed is a run whose
    // results mean something other than what they think.
    if installed.withheld.is_empty() {
        info!(
            registered = installed.registered(),
            eligible = installed.population.len(),
            "every registered brain family implements the versioned genome/evaluator protocol; all are eligible to found the population"
        );
    } else {
        let withheld_labels: Vec<&str> = installed
            .withheld
            .iter()
            .map(|(label, _)| label.as_str())
            .collect();
        warn!(
            registered = installed.registered(),
            eligible = installed.population.len(),
            withheld = ?withheld_labels,
            "SOME BRAIN FAMILIES ARE WITHHELD FROM THE FOUNDING POPULATION because they do \
             not implement the versioned genome/evaluator-state protocol. They remain registered \
             and can still be selected explicitly, but no founder will be seeded with an opaque \
             legacy hereditary state."
        );
    }

    Ok(installed)
}

fn parse_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_layers(raw: &str) -> Option<Vec<usize>> {
    let mut layers = Vec::new();
    for token in raw.split(',') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed.parse::<usize>() {
            Ok(value) if value > 0 => layers.push(value),
            _ => return None,
        }
    }
    Some(layers)
}

fn parse_activation(raw: &str) -> Option<NeuroflowActivationKind> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "tanh" => Some(NeuroflowActivationKind::Tanh),
        "sigmoid" => Some(NeuroflowActivationKind::Sigmoid),
        "relu" => Some(NeuroflowActivationKind::Relu),
        _ => None,
    }
}

fn parse_tonemap(raw: &str) -> Option<RenderTonemapMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "aces" => Some(RenderTonemapMode::Aces),
        "agx" => Some(RenderTonemapMode::Agx),
        "tony" | "tonymcmapface" => Some(RenderTonemapMode::Tony),
        _ => None,
    }
}

#[cfg(any(feature = "gui", feature = "bevy_render"))]
fn parse_png_size(raw: &str) -> Option<(u32, u32)> {
    let lower = raw.trim().to_ascii_lowercase();
    let mut parts = lower.split('x');
    let w = parts.next()?.trim().parse::<u32>().ok()?;
    let h = parts.next()?.trim().parse::<u32>().ok()?;
    if parts.next().is_some() || w == 0 || h == 0 {
        return None;
    }
    Some((w, h))
}

fn seed_agents(world: &mut WorldState, brain_keys: &[u64]) -> Result<()> {
    if brain_keys.is_empty() {
        bail!("cannot seed the scenario without at least one registered brain");
    }
    let mut agent = AgentData::default();
    let spacing = 120.0;
    for row in 0..4 {
        for col in 0..4 {
            agent.position.x = col as f32 * spacing + spacing * 0.5;
            agent.position.y = row as f32 * spacing + spacing * 0.5;
            agent.heading = 0.0;
            agent.spike_length = 10.0;
            let id = world
                .try_spawn_agent(agent)
                .context("seeded agent must be finite")?;
            let Some(&key) = brain_keys.get((row * 4 + col) % brain_keys.len()) else {
                bail!("registered-brain selection invariant failed while seeding agent {id:?}");
            };
            let bound = world.bind_agent_brain(id, key).with_context(|| {
                format!("failed to construct registered brain {key} for seeded agent {id:?}")
            })?;
            if !bound {
                bail!(
                    "registered brain {key} disappeared while binding seeded agent {id:?}; refusing an unbound fallback"
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::BirthOrigin;
    use scriptbots_storage::{StoragePipeline, StorageReader};
    use serial_test::serial;
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    /// Exercise the shipped startup function, not a protocol-only fixture.
    #[test]
    fn install_brains_seeds_only_versioned_protocol_families() {
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(11),
            ..ScriptBotsConfig::default()
        })
        .expect("world");

        let installed = install_brains(&mut world, BrainPreset::Mixed).expect("brains install");

        let expected = [
            (MlpBrain::KIND.as_str(), "mlp-baseline"),
            (DwraonBrain::KIND.as_str(), "dwraon-baseline"),
            (AssemblyBrain::KIND.as_str(), "assembly"),
        ];
        assert_eq!(
            installed.population.len(),
            expected.len() + usize::from(cfg!(feature = "brain-ft")),
            "mixed startup must admit every compiled, implemented protocol family"
        );

        let registry = world.brain_registry();
        for (key, (expected_kind, expected_family_id)) in installed.population.iter().zip(expected)
        {
            assert_eq!(registry.kind(*key), Some(expected_kind));
            assert!(
                registry.is_protocol_family(*key),
                "founding family `{expected_kind}` must be backed by the versioned protocol"
            );
            let family = registry
                .family(*key)
                .expect("a protocol registry key must expose its family adapter");
            assert_eq!(family.family_id().as_str(), expected_family_id);
        }
        #[cfg(feature = "brain-ft")]
        {
            let ft_key = installed.population[expected.len()];
            assert_eq!(registry.kind(ft_key), Some(FT_BRAIN_KIND));
            assert!(registry.is_protocol_family(ft_key));
            assert!(registry.family(ft_key).is_some());
        }

        for (label, key) in &installed.withheld {
            assert!(
                !registry.is_protocol_family(*key),
                "withheld legacy family `{label}` must not masquerade as a protocol family"
            );
            assert!(registry.family(*key).is_none());
        }

        assert!(
            registry
                .descriptors()
                .iter()
                .all(|(_, kind)| !kind.contains("placeholder")),
            "`ml.placeholder` has no protocol codec and must not be registered at startup"
        );
    }

    #[test]
    fn single_family_presets_register_only_the_requested_protocol_family() {
        for (preset, expected_kind, expected_family_id) in [
            (BrainPreset::Mlp, MlpBrain::KIND.as_str(), "mlp-baseline"),
            (
                BrainPreset::Dwraon,
                DwraonBrain::KIND.as_str(),
                "dwraon-baseline",
            ),
            (
                BrainPreset::Assembly,
                AssemblyBrain::KIND.as_str(),
                "assembly",
            ),
        ] {
            let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
            let installed =
                install_brains(&mut world, preset).expect("single-family brain install");

            assert_eq!(installed.registered(), 1);
            assert_eq!(installed.population.len(), 1);
            assert!(installed.withheld.is_empty());
            let key = installed.population[0];
            assert_eq!(
                world.brain_registry().descriptors(),
                vec![(key, expected_kind.to_owned())]
            );
            assert_eq!(
                world
                    .brain_registry()
                    .family(key)
                    .expect("single-family preset must register a protocol adapter")
                    .family_id()
                    .as_str(),
                expected_family_id
            );
        }
    }

    #[cfg(feature = "brain-ft")]
    #[test]
    fn ft_preset_registers_only_frankentorch() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let installed =
            install_brains(&mut world, BrainPreset::Ft).expect("Frankentorch brain install");

        assert_eq!(installed.registered(), 1);
        assert_eq!(installed.population.len(), 1);
        assert!(installed.withheld.is_empty());
        let key = installed.population[0];
        assert_eq!(
            world.brain_registry().descriptors(),
            vec![(key, FT_BRAIN_KIND.to_owned())]
        );
        assert!(world.brain_registry().is_protocol_family(key));
    }

    #[cfg(not(feature = "brain-ft"))]
    #[test]
    fn ft_preset_without_feature_fails_before_registering_any_family() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let error = install_brains(&mut world, BrainPreset::Ft)
            .expect_err("uncompiled Frankentorch preset must fail");

        assert!(
            error.to_string().contains("requires") && error.to_string().contains("brain-ft"),
            "feature refusal must name the required app feature: {error:#}"
        );
        assert!(world.brain_registry().descriptors().is_empty());
    }

    #[test]
    fn seeded_birth_records_capture_the_post_bind_brain_identity() {
        struct BirthCapture {
            batches: Arc<Mutex<Vec<scriptbots_core::PersistenceBatch>>>,
        }

        impl WorldPersistence for BirthCapture {
            fn on_tick(
                &mut self,
                payload: &scriptbots_core::PersistenceBatch,
            ) -> std::result::Result<(), scriptbots_core::PersistenceAdmissionError> {
                self.batches
                    .lock()
                    .expect("birth capture lock")
                    .push(payload.clone());
                Ok(())
            }
        }

        let batches = Arc::new(Mutex::new(Vec::new()));
        let (mut world, mut persistence) = WorldState::with_persistence(
            ScriptBotsConfig {
                persistence_interval: 1,
                population_minimum: 0,
                population_spawn_interval: 0,
                reproduction_attempt_chance: 0.0,
                rng_seed: Some(0x5EED_B17A),
                ..ScriptBotsConfig::default()
            },
            Box::new(BirthCapture {
                batches: Arc::clone(&batches),
            }),
        )
        .expect("world");
        let installed =
            install_brains(&mut world, BrainPreset::Mixed).expect("install seed brains");
        seed_agents(&mut world, &installed.population).expect("seed founding population");

        persistence
            .step(&mut world)
            .expect("persist seeded lifecycle records");

        let batches = batches.lock().expect("birth capture lock");
        let batch = batches.last().expect("first persistence cadence batch");
        assert_eq!(batch.births.len(), 16, "every founder must be recorded");
        for (index, birth) in batch.births.iter().enumerate() {
            let expected_key = installed.population[index % installed.population.len()];
            let expected_kind = world
                .brain_registry()
                .kind(expected_key)
                .expect("seed brain remains registered");
            assert_eq!(birth.origin, BirthOrigin::Seeded);
            assert_eq!(birth.brain_key, Some(expected_key));
            assert_eq!(birth.brain_kind.as_deref(), Some(expected_kind));
            assert_eq!((birth.parent_a, birth.parent_b), (None, None));
        }
    }

    fn default_cli() -> AppCli {
        AppCli::parse_from(["scriptbots-app"])
    }

    #[test]
    #[serial]
    fn brain_preset_cli_defaults_to_mixed_and_parses_ft() {
        with_clean_config_env(|| {
            assert_eq!(default_cli().brain, BrainPreset::Mixed);
            let cli = AppCli::parse_from(["scriptbots-app", "--brain", "ft"]);
            assert_eq!(cli.brain, BrainPreset::Ft);
            let (_, scenario, _) =
                compose_config_with_scenario(&cli).expect("compose FT scenario provenance");
            assert_eq!(
                scenario.population_recipe,
                "fixed-4x4-registered-brain-grid-v1;brain=ft"
            );
        });
    }

    #[test]
    #[serial]
    fn startup_defaults_begin_at_tick_zero_and_preserve_explicit_bootstrap() {
        with_env_lock(|| {
            let previous_mode = std::env::var("SCRIPTBOTS_MODE").ok();
            let previous_bootstrap = std::env::var("SCRIPTBOTS_BOOTSTRAP_TICKS").ok();
            unsafe {
                std::env::remove_var("SCRIPTBOTS_MODE");
                std::env::remove_var("SCRIPTBOTS_BOOTSTRAP_TICKS");
            }

            let cli = default_cli();
            assert_eq!(cli.mode, RendererMode::Auto);
            assert_eq!(DEFAULT_BOOTSTRAP_TICKS, 0);
            assert_eq!(cli.bootstrap_ticks, DEFAULT_BOOTSTRAP_TICKS);

            let cli = AppCli::parse_from([
                "scriptbots-app",
                "--mode",
                "terminal",
                "--bootstrap-ticks",
                "37",
            ]);
            assert_eq!(cli.mode, RendererMode::Terminal);
            assert_eq!(cli.bootstrap_ticks, 37);

            restore_env("SCRIPTBOTS_MODE", previous_mode);
            restore_env("SCRIPTBOTS_BOOTSTRAP_TICKS", previous_bootstrap);
        });
    }

    #[test]
    fn run_seed_materialization_is_one_time_and_preserves_explicit_values() {
        let mut generated = ScriptBotsConfig::default();
        let (seed, was_generated) = materialize_run_seed(&mut generated);
        assert!(was_generated);
        assert_eq!(generated.rng_seed, Some(seed));

        let (same_seed, was_generated_again) = materialize_run_seed(&mut generated);
        assert!(!was_generated_again);
        assert_eq!(same_seed, seed);

        let mut explicit = ScriptBotsConfig {
            rng_seed: Some(u64::MAX),
            ..ScriptBotsConfig::default()
        };
        assert_eq!(materialize_run_seed(&mut explicit), (u64::MAX, false));
        assert_eq!(explicit.rng_seed, Some(u64::MAX));
    }

    #[test]
    fn run_id_allocator_returns_canonical_nonzero_distinct_ids() {
        let first = allocate_run_id();
        let second = allocate_run_id();

        assert_ne!(first.get(), 0);
        assert_ne!(second.get(), 0);
        assert_ne!(first, second);
        assert_eq!(first.to_string(), format!("{:032x}", first.get()));
        assert_eq!(second.to_string(), format!("{:032x}", second.get()));
    }

    #[test]
    fn run_manifest_records_requested_bootstrap_before_tick_zero() {
        let world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(0xB007_57A4),
            ..ScriptBotsConfig::default()
        })
        .expect("world");
        let identity = RunIdentityV1::new(
            RunId::new(0xB007_57A4),
            1_752_515_200_000,
            None,
            Some(LIVE_RUN_POLICY.to_owned()),
        );
        let thread_policy = resolve_thread_policy(None, None, None, false);

        let mut scenario = ScenarioIdentityV0::caller_seeded("manifest-test");
        scenario.bootstrap_ticks = 37;
        let manifest = build_run_manifest(&world, identity, scenario, thread_policy, Vec::new())
            .expect("manifest");

        assert_eq!(
            world.tick().0,
            0,
            "manifest emission must not advance science"
        );
        assert_eq!(manifest.identity.run_id, RunId::new(0xB007_57A4));
        assert_eq!(manifest.identity.requested_tick_budget, None);
        assert_eq!(
            manifest.identity.live_run_policy.as_deref(),
            Some(LIVE_RUN_POLICY)
        );
        assert_eq!(manifest.scenario.bootstrap_ticks, 37);
        assert!(manifest.bootstrap_evidence.is_none());
        assert_eq!(manifest.schema, scriptbots_app::RUN_MANIFEST_V3_SCHEMA);
        assert_eq!(
            manifest
                .thread_policy
                .as_ref()
                .map(|policy| policy.source.as_str()),
            Some("builtin-default")
        );
    }

    #[test]
    fn explicit_renderer_modes_never_fall_back() {
        let unavailable = RendererAvailability {
            gui: false,
            bevy: false,
        };
        let hostile_auto_environment = RendererEnvironment {
            force_terminal: true,
            force_gui: false,
            graphical_session: false,
        };

        assert_eq!(
            select_renderer_mode(
                RendererMode::Terminal,
                unavailable,
                hostile_auto_environment,
            )
            .expect("terminal is always compiled"),
            RendererMode::Terminal
        );
        let gui_error =
            select_renderer_mode(RendererMode::Gui, unavailable, hostile_auto_environment)
                .expect_err("an explicit unavailable GPUI request must fail");
        assert!(gui_error.to_string().contains("--features gui"));
        let bevy_error =
            select_renderer_mode(RendererMode::Bevy, unavailable, hostile_auto_environment)
                .expect_err("an explicit unavailable Bevy request must fail");
        assert!(bevy_error.to_string().contains("--features bevy_render"));
    }

    #[test]
    fn automatic_renderer_uses_only_compiled_backends() {
        let graphical = RendererEnvironment {
            graphical_session: true,
            ..RendererEnvironment::default()
        };
        let headless = RendererEnvironment::default();

        assert_eq!(
            select_renderer_mode(
                RendererMode::Auto,
                RendererAvailability {
                    gui: true,
                    bevy: true,
                },
                graphical,
            )
            .expect("GPUI is the preferred compiled native renderer"),
            RendererMode::Gui
        );
        assert_eq!(
            select_renderer_mode(
                RendererMode::Auto,
                RendererAvailability {
                    gui: false,
                    bevy: true,
                },
                graphical,
            )
            .expect("Bevy is the next compiled graphical renderer"),
            RendererMode::Bevy
        );
        assert_eq!(
            select_renderer_mode(
                RendererMode::Auto,
                RendererAvailability {
                    gui: false,
                    bevy: false,
                },
                graphical,
            )
            .expect("a non-GUI build must remain usable"),
            RendererMode::Terminal
        );
        assert_eq!(
            select_renderer_mode(
                RendererMode::Auto,
                RendererAvailability {
                    gui: true,
                    bevy: true,
                },
                headless,
            )
            .expect("headless auto mode must select the terminal"),
            RendererMode::Terminal
        );
    }

    #[test]
    fn automatic_renderer_force_contract_is_fail_closed() {
        let available = RendererAvailability {
            gui: true,
            bevy: true,
        };
        let conflict = RendererEnvironment {
            force_terminal: true,
            force_gui: true,
            graphical_session: true,
        };
        assert!(
            select_renderer_mode(RendererMode::Auto, available, conflict)
                .expect_err("conflicting overrides must not depend on branch order")
                .to_string()
                .contains("cannot both be enabled")
        );

        let unavailable_gui = RendererEnvironment {
            force_gui: true,
            graphical_session: true,
            ..RendererEnvironment::default()
        };
        assert!(
            select_renderer_mode(
                RendererMode::Auto,
                RendererAvailability {
                    gui: false,
                    bevy: true,
                },
                unavailable_gui,
            )
            .expect_err("force-GUI must not silently substitute Bevy")
            .to_string()
            .contains("refusing to substitute")
        );
    }

    #[test]
    fn compiled_renderer_availability_matches_cargo_features() {
        let available = RendererAvailability::compiled();
        assert_eq!(available.gui, cfg!(feature = "gui"));
        assert_eq!(available.bevy, cfg!(feature = "bevy_render"));
    }

    #[test]
    fn display_environment_matrix_rejects_missing_and_empty_values() {
        use std::ffi::OsString;

        assert!(!display_environment_available(None, None));
        assert!(!display_environment_available(
            Some(OsString::new()),
            Some(OsString::new()),
        ));
        assert!(display_environment_available(
            Some(OsString::from(":0")),
            None,
        ));
        assert!(display_environment_available(
            None,
            Some(OsString::from("wayland-0")),
        ));
    }

    #[test]
    fn windows_auto_mode_requires_a_visible_window_station() {
        assert!(!window_station_is_visible(0));
        assert!(window_station_is_visible(0x0001));
        assert!(window_station_is_visible(0x0001 | 0x4000));
        assert!(!window_station_is_visible(0x4000));
    }

    #[cfg(windows)]
    #[test]
    fn windows_auto_mode_queries_the_actual_process_window_station() {
        let evidence = windows_graphical_session::process_window_station_evidence();
        assert_eq!(
            evidence.visible,
            evidence.query_succeeded && window_station_is_visible(evidence.flags),
            "the production User32 result must drive the same visibility policy as Auto mode"
        );
        assert_eq!(
            evidence.buffer_size, 12,
            "Win32 USEROBJECTFLAGS must remain three 32-bit fields: {evidence:?}"
        );
        if evidence.query_succeeded {
            assert!(
                evidence.station_present,
                "GetUserObjectInformationW cannot succeed without a process window station"
            );
            assert!(
                evidence.bytes_needed == evidence.buffer_size,
                "User32 must report the exact USEROBJECTFLAGS buffer size on a successful \
                 UOI_FLAGS query: {evidence:?}"
            );
        }

        let selected = select_renderer_mode(
            RendererMode::Auto,
            RendererAvailability {
                gui: true,
                bevy: true,
            },
            RendererEnvironment {
                graphical_session: evidence.visible,
                ..RendererEnvironment::default()
            },
        )
        .expect("actual Windows session evidence must select a compiled renderer");
        assert_eq!(
            selected,
            if evidence.visible {
                RendererMode::Gui
            } else {
                RendererMode::Terminal
            }
        );
        eprintln!(
            "SCRIPTBOTS_WINDOWS_SESSION_EVIDENCE station_present={} query_succeeded={} \
             flags=0x{:08x} buffer_size={} bytes_needed={} visible={} selected={}",
            evidence.station_present,
            evidence.query_succeeded,
            evidence.flags,
            evidence.buffer_size,
            evidence.bytes_needed,
            evidence.visible,
            selected
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_auto_mode_rejects_remote_shell_sessions_without_x11_assumptions() {
        use std::ffi::OsString;

        assert!(macos_auto_graphical_session_available(true, false));
        assert!(!macos_auto_graphical_session_available(false, false));
        assert!(!macos_auto_graphical_session_available(true, true));
        assert!(!remote_shell_session_detected([None, None, None]));
        assert!(!remote_shell_session_detected([
            Some(OsString::new()),
            None,
            None,
        ]));
        assert!(remote_shell_session_detected([
            Some(OsString::from("host 22 host 50000")),
            None,
            None,
        ]));
    }

    #[cfg(not(feature = "gui"))]
    #[test]
    fn unavailable_explicit_gpui_resolution_returns_an_error() {
        let error = resolve_renderer(RendererMode::Gui, false)
            .err()
            .expect("default application build must reject unavailable GPUI");
        assert!(error.to_string().contains("--features gui"));
    }

    #[cfg(not(feature = "bevy_render"))]
    #[test]
    fn unavailable_explicit_bevy_resolution_returns_an_error() {
        let error = resolve_renderer(RendererMode::Bevy, false)
            .err()
            .expect("default application build must reject unavailable Bevy");
        assert!(error.to_string().contains("--features bevy_render"));
    }

    fn persistence_batch_at(tick: u64) -> scriptbots_core::PersistenceBatch {
        scriptbots_core::PersistenceBatch {
            summary: TickSummary {
                tick: scriptbots_core::Tick(tick),
                agent_count: 0,
                births: 0,
                deaths: 0,
                total_energy: 0.0,
                average_energy: 0.0,
                average_health: 0.0,
                max_age: 0,
                spike_hits: 0,
            },
            epoch: 0,
            closed: false,
            metrics: Vec::new(),
            events: Vec::new(),
            agents: Vec::new(),
            births: Vec::new(),
            deaths: Vec::new(),
            replay_events: Vec::new(),
        }
    }

    #[test]
    fn shutdown_finalization_requires_committed_tick_zero_for_bootstrap_origins() {
        let config = ScriptBotsConfig {
            persistence_interval: 5,
            population_minimum: 0,
            population_spawn_interval: 0,
            ..ScriptBotsConfig::default()
        };
        let mut pipeline = StoragePipeline::unattributed_memory().expect("volatile pipeline");
        let (mut world, mut persistence) =
            WorldState::with_persistence(config, Box::new(pipeline.sink()))
                .expect("world with persistence");
        world
            .try_spawn_agent(AgentData::default())
            .expect("seeded bootstrap arrival");

        let finalization = finalize_world_persistence(&mut world, &mut persistence)
            .expect("tick-zero origin finalization");
        assert!(finalization.admitted_tail);
        assert_eq!(finalization.required_tick, Some(0));

        let receipt = pipeline.shutdown().expect("volatile shutdown");
        assert_eq!(receipt.committed_tick, Some(0));
        validate_shutdown_receipt(finalization, receipt)
            .expect("tick-zero origin batch must close the admitted prefix");
    }

    #[test]
    fn shutdown_receipt_tracks_admitted_history_after_persistence_is_disabled() {
        let config = ScriptBotsConfig {
            persistence_interval: 1,
            population_minimum: 0,
            population_spawn_interval: 0,
            ..ScriptBotsConfig::default()
        };
        let mut pipeline = StoragePipeline::unattributed_memory().expect("volatile pipeline");
        let (mut world, mut persistence) =
            WorldState::with_persistence(config, Box::new(pipeline.sink()))
                .expect("world with persistence");
        persistence.step(&mut world).expect("cadence admission");
        assert_eq!(
            persistence.last_admitted_tick(),
            Some(scriptbots_core::Tick(1))
        );

        let mut disabled = world.config().clone();
        disabled.persistence_interval = 0;
        world
            .apply_config_update(disabled)
            .expect("disable persistence after admission");

        let finalization = finalize_world_persistence(&mut world, &mut persistence)
            .expect("shutdown after disabling persistence");
        assert!(!finalization.admitted_tail);
        assert_eq!(finalization.required_tick, Some(1));

        let receipt = pipeline.shutdown().expect("volatile shutdown");
        assert_eq!(receipt.committed_tick, Some(1));
        validate_shutdown_receipt(finalization, receipt)
            .expect("shutdown must validate the historically admitted tick");
    }

    #[test]
    fn shutdown_receipt_validation_requires_closed_prefixes_for_both_guarantees() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("durable-receipt.sqlite");
        let mut durable_pipeline =
            StoragePipeline::create_unattributed_file(path.to_str().expect("utf8 test path"))
                .expect("durable pipeline");
        let durable_admission = durable_pipeline
            .submit_with_receipt(&persistence_batch_at(11))
            .expect("durable admission");
        let durable_receipt = durable_pipeline.shutdown().expect("durable shutdown");
        assert_eq!(durable_receipt.guarantee, PersistenceGuarantee::Durable);
        validate_shutdown_receipt(
            StorageFinalization {
                admitted_tail: false,
                required_tick: Some(11),
            },
            durable_receipt,
        )
        .expect("closed durable prefix");

        let mut incomplete_durable = durable_receipt;
        incomplete_durable.watermarks.durable = None;
        let error = validate_shutdown_receipt(
            StorageFinalization {
                admitted_tail: false,
                required_tick: Some(11),
            },
            incomplete_durable,
        )
        .expect_err("durable shutdown must close the durable prefix");
        let rendered = error.to_string();
        assert!(rendered.contains("admitted=Some"));
        assert!(rendered.contains("applied=Some"));
        assert!(rendered.contains("durable=None"));

        let mut volatile_pipeline =
            StoragePipeline::unattributed_memory().expect("volatile pipeline");
        let volatile_admission = volatile_pipeline
            .submit_with_receipt(&persistence_batch_at(12))
            .expect("volatile admission");
        let volatile_receipt = volatile_pipeline.shutdown().expect("volatile shutdown");
        assert_eq!(
            volatile_receipt.guarantee,
            PersistenceGuarantee::CommittedVolatile
        );
        validate_shutdown_receipt(
            StorageFinalization {
                admitted_tail: false,
                required_tick: Some(12),
            },
            volatile_receipt,
        )
        .expect("closed volatile prefix");

        let mut invalid_volatile = volatile_receipt;
        invalid_volatile.watermarks.durable = Some(volatile_admission.batch_id);
        let error = validate_shutdown_receipt(
            StorageFinalization {
                admitted_tail: false,
                required_tick: Some(12),
            },
            invalid_volatile,
        )
        .expect_err("volatile shutdown cannot claim a durable prefix");
        let rendered = error.to_string();
        assert!(rendered.contains("guarantee=CommittedVolatile"));
        assert!(rendered.contains("durable=Some"));

        assert_eq!(
            durable_receipt.watermarks.admitted,
            Some(durable_admission.batch_id)
        );
    }

    #[test]
    fn new_run_reservation_refuses_existing_main_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("run.sqlite");
        fs::write(&path, b"existing-run").expect("create existing run fixture");

        let error =
            StoragePipeline::create_unattributed_file(path.to_str().expect("utf8 test path"))
                .err()
                .expect("existing run path must be rejected");
        assert!(error.to_string().contains("refusing to reuse"));
        assert_eq!(fs::read(&path).expect("read existing run"), b"existing-run");
    }

    #[test]
    fn new_run_reservation_refuses_orphaned_wal_sidecar() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("run.sqlite");
        let wal = PathBuf::from(format!("{}-wal", path.display()));
        fs::write(&wal, b"stale-wal").expect("write stale WAL fixture");

        let error =
            StoragePipeline::create_unattributed_file(path.to_str().expect("utf8 test path"))
                .err()
                .expect("orphaned WAL must prevent run reuse");
        assert!(error.to_string().contains("stale FrankenSQLite sidecar"));
        assert!(!path.exists(), "reservation must not create the main file");
        assert_eq!(fs::read(wal).expect("read stale WAL"), b"stale-wal");
    }

    #[test]
    fn recovery_cli_parses_an_explicit_existing_database_path() {
        let cli = AppCli::parse_from([
            "scriptbots-app",
            "--recover-storage",
            "runs/interrupted.sqlite",
        ]);
        assert_eq!(
            cli.recover_storage,
            Some(PathBuf::from("runs/interrupted.sqlite"))
        );
        assert!(
            AppCli::try_parse_from([
                "scriptbots-app",
                "--recover-storage",
                "runs/interrupted.sqlite",
                "--profile-steps",
                "1",
            ])
            .is_err(),
            "repair mode must not silently ignore run-mode arguments"
        );
    }

    #[test]
    fn explicit_storage_recovery_reopens_and_preserves_durable_watermarks() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("interrupted.sqlite");
        let path_string = path.to_string_lossy().to_string();
        let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(
            &path_string,
            64,
            4096,
            1024,
            1024,
        )
        .expect("create recovery fixture");
        let admission = pipeline
            .submit_with_receipt(&scriptbots_core::PersistenceBatch {
                summary: TickSummary {
                    tick: scriptbots_core::Tick(7),
                    agent_count: 0,
                    births: 0,
                    deaths: 0,
                    total_energy: 0.0,
                    average_energy: 0.0,
                    average_health: 0.0,
                    max_age: 0,
                    spike_hits: 0,
                },
                epoch: 0,
                closed: false,
                metrics: Vec::new(),
                events: Vec::new(),
                agents: Vec::new(),
                births: Vec::new(),
                deaths: Vec::new(),
                replay_events: Vec::new(),
            })
            .expect("admit recovery fixture");
        pipeline.shutdown().expect("finalize recovery fixture");

        recover_storage(&path).expect("explicit recovery mode");
        let reader = StorageReader::open(&path_string).expect("open recovered database");
        let watermarks = reader
            .persistence_watermarks()
            .expect("read recovered watermarks");
        assert_eq!(watermarks.admitted, Some(admission.batch_id));
        assert_eq!(watermarks.applied, Some(admission.batch_id));
        assert_eq!(watermarks.durable, Some(admission.batch_id));
        reader.close().expect("close recovery reader");
    }

    #[test]
    fn explicit_storage_recovery_refuses_a_missing_path() {
        let dir = tempdir().expect("tempdir");
        let missing = dir.path().join("missing.sqlite");
        let error = recover_storage(&missing).expect_err("recovery must not create a new run");
        assert!(error.to_string().contains("does not exist"));
        assert!(!missing.exists());
    }

    #[test]
    fn explicit_storage_recovery_refuses_an_unrecognized_file_without_mutation() {
        let dir = tempdir().expect("tempdir");
        let unrelated = dir.path().join("unrelated.sqlite");
        fs::write(&unrelated, b"not-a-scriptbots-database").expect("write unrelated fixture");
        let error = recover_storage(&unrelated).expect_err("unrecognized file must be refused");
        assert!(error.to_string().contains("failed to recover"));
        assert_eq!(
            fs::read(&unrelated).expect("read unrelated fixture"),
            b"not-a-scriptbots-database"
        );
    }

    #[test]
    fn new_run_reservation_rejects_volatile_and_uri_path_shapes() {
        let error = StoragePipeline::create_unattributed_file(":memory:")
            .err()
            .expect("file mode must reject the in-memory engine path");
        assert!(
            error
                .to_string()
                .contains("explicit Storage or StoragePipeline memory constructors")
        );
        assert!(
            !Path::new(":memory:").exists(),
            "reservation must not create a literal :memory: file"
        );

        let error = StoragePipeline::create_unattributed_file("")
            .err()
            .expect("file mode must reject an empty path");
        assert!(error.to_string().contains("non-empty path"));

        let error = StoragePipeline::create_unattributed_file("   \t")
            .err()
            .expect("file mode must reject a whitespace-only path");
        assert!(error.to_string().contains("non-empty path"));

        let error = StoragePipeline::create_unattributed_file("file:run.sqlite?mode=memory")
            .err()
            .expect("file mode must reject file: URI paths");
        assert!(error.to_string().contains("file: URI"));
    }

    #[test]
    fn profile_children_never_consume_the_requested_run_database() {
        fn storage_override(command: &Command) -> Option<Option<PathBuf>> {
            command
                .get_envs()
                .find(|(key, _)| *key == std::ffi::OsStr::new("SCRIPTBOTS_STORAGE_PATH"))
                .map(|(_, value)| value.map(PathBuf::from))
        }

        let requested = PathBuf::from("requested-final-run.sqlite");
        let mut first = Command::new("scriptbots-app");
        first.env("SCRIPTBOTS_STORAGE_PATH", &requested);
        configure_profile_child_storage(&mut first, StorageMode::File);
        let first_path = storage_override(&first)
            .flatten()
            .expect("file profile child path");
        assert_ne!(first_path, requested);
        assert_eq!(first_path.extension(), Some(std::ffi::OsStr::new("sqlite")));

        let mut second = Command::new("scriptbots-app");
        second.env("SCRIPTBOTS_STORAGE_PATH", &requested);
        configure_profile_child_storage(&mut second, StorageMode::File);
        let second_path = storage_override(&second)
            .flatten()
            .expect("second file profile child path");
        assert_ne!(second_path, requested);
        assert_ne!(first_path, second_path);

        let mut memory = Command::new("scriptbots-app");
        memory.env("SCRIPTBOTS_STORAGE_PATH", &requested);
        configure_profile_child_storage(&mut memory, StorageMode::Memory);
        assert_eq!(storage_override(&memory), Some(None));
    }

    #[test]
    fn characterization_cli_parses_ticks_and_output() {
        let cli = AppCli::parse_from([
            "scriptbots-app",
            "--rng-seed",
            "42",
            "--characterize-v0",
            "16",
            "--characterization-out",
            "trace.json",
        ]);

        assert_eq!(cli.rng_seed, Some(42));
        assert_eq!(cli.characterize_v0, Some(16));
        assert_eq!(cli.characterization_out, Some(PathBuf::from("trace.json")));
    }

    #[test]
    fn characterization_output_requires_trace_mode() {
        let result =
            AppCli::try_parse_from(["scriptbots-app", "--characterization-out", "trace.json"]);
        assert!(result.is_err());
    }

    #[test]
    fn characterization_materializes_and_records_an_unspecified_seed() {
        let dir = tempdir().expect("tempdir");
        let output = dir.path().join("trace.json");
        let mut cli = default_cli();
        cli.characterization_out = Some(output.clone());
        let config = ScriptBotsConfig {
            rng_seed: None,
            ..ScriptBotsConfig::default()
        };
        let scenario = ScenarioIdentityV0::caller_seeded("characterization-seed-test");

        run_characterization_v0(&cli, config, scenario, Vec::new(), 0)
            .expect("capture zero-tick characterization");
        let trace: CharacterizationTraceV2 =
            serde_json::from_slice(&fs::read(output).expect("read characterization artifact"))
                .expect("decode characterization trace");
        assert_eq!(
            trace.schema,
            scriptbots_app::CHARACTERIZATION_TRACE_V2_SCHEMA
        );
        assert_eq!(trace.schema_version, 2);
        let root_seed = trace.manifest.root_seed;
        assert_eq!(
            trace.manifest.normalized_config["rng_seed"],
            serde_json::json!(root_seed),
            "the manifest must carry the exact materialized seed used by the world"
        );
    }

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn with_env_lock<F: FnOnce()>(f: F) {
        let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock.lock().expect("env mutex poisoned");
        f();
    }

    #[test]
    #[serial]
    fn layered_configs_apply_in_order() {
        let dir = tempdir().expect("tempdir");
        let base_path = dir.path().join("base.toml");
        fs::write(
            &base_path,
            r#"
persistence_interval = 120
rng_seed = 1337

[neuroflow]
enabled = true
hidden_layers = [64, 32]
activation = "Tanh"
"#,
        )
        .expect("write base layer");

        let overlay_path = dir.path().join("overlay.toml");
        fs::write(
            &overlay_path,
            r#"
history_capacity = 1024
world_width = 2048

[neuroflow]
hidden_layers = [8, 4]
activation = "Sigmoid"
"#,
        )
        .expect("write overlay layer");

        let base_config = ScriptBotsConfig {
            persistence_interval: 60,
            history_capacity: 600,
            ..ScriptBotsConfig::default()
        };

        let layered = apply_config_layers(base_config, &[base_path, overlay_path])
            .expect("apply config layers");

        assert_eq!(layered.persistence_interval, 120);
        assert_eq!(layered.history_capacity, 1024);
        assert_eq!(layered.world_width, 2048);
        assert_eq!(layered.rng_seed, Some(1337));
        assert!(layered.neuroflow.enabled);
        assert_eq!(layered.neuroflow.hidden_layers, vec![8, 4]);
        assert_eq!(
            layered.neuroflow.activation,
            NeuroflowActivationKind::Sigmoid
        );
    }

    #[test]
    #[serial]
    fn composed_scenario_digests_the_exact_ordered_layer_bytes() {
        with_clean_config_env(|| {
            let dir = tempdir().expect("tempdir");
            let first_path = dir.path().join("first.toml");
            let second_path = dir.path().join("second.toml");
            let first = b"rng_seed = 41\nworld_width = 800\n";
            let second = b"rng_seed = 42\nworld_height = 600\n";
            fs::write(&first_path, first).expect("write first layer");
            fs::write(&second_path, second).expect("write second layer");

            let mut cli = default_cli();
            cli.config_layers = vec![first_path, second_path];
            let (config, scenario, overrides) =
                compose_config_with_scenario(&cli).expect("compose scenario provenance");

            let mut expected = ScenarioIdentityV0::caller_seeded("scriptbots-app-layered-v1");
            expected.population_recipe =
                "fixed-4x4-registered-brain-grid-v1;brain=mixed".to_owned();
            let defaults_value = serde_json::to_value(ScriptBotsConfig {
                persistence_interval: 60,
                history_capacity: 600,
                ..ScriptBotsConfig::default()
            })
            .expect("serialize composed defaults");
            expected.record_config_layer(
                ConfigLayerKind::Defaults,
                &canonical_layer_bytes(&defaults_value),
            );
            expected.record_config_layer(ConfigLayerKind::File, first);
            expected.record_config_layer(ConfigLayerKind::File, second);
            assert_eq!(scenario, expected);
            assert_eq!(config.rng_seed, Some(42));
            assert_eq!(config.world_width, 800);
            assert_eq!(config.world_height, 600);
            // Two files disagreeing about rng_seed is itself a reportable displacement.
            assert_eq!(overrides.len(), 1, "unexpected overrides: {overrides:?}");
            assert_eq!(overrides[0].path, "rng_seed");
        });
    }

    #[test]
    #[serial]
    fn cli_config_layer_rejects_invalid_finite_float_with_field_path() {
        let dir = tempdir().expect("tempdir");
        let layer = dir.path().join("invalid.toml");
        fs::write(&layer, "food_growth_rate = -1.0\n").expect("write config layer");
        let mut cli = default_cli();
        cli.config_layers.push(layer);
        cli.config_only = true;

        let error = compose_config(&cli).expect_err("invalid config-only input must fail");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("food_growth_rate"),
            "CLI error did not identify field: {rendered}"
        );
    }

    #[test]
    #[serial]
    fn cli_config_layer_rejects_float_outside_f32_domain_with_field_path() {
        let dir = tempdir().expect("tempdir");
        let layer = dir.path().join("unrepresentable.toml");
        fs::write(&layer, "food_growth_rate = 1e40\n").expect("write config layer");
        let mut cli = default_cli();
        cli.config_layers.push(layer);

        let error = compose_config(&cli).expect_err("unrepresentable f32 input must fail");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("food_growth_rate"),
            "CLI deserialization error did not identify field: {rendered}"
        );
    }

    #[test]
    #[serial]
    fn headless_replay_finalizes_non_aligned_persistence_tail() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("replay.sqlite");
        let db_str = db_path.to_string_lossy().to_string();

        let config = ScriptBotsConfig {
            world_width: 600,
            world_height: 600,
            food_cell_size: 60,
            persistence_interval: 5,
            history_capacity: 128,
            rng_seed: Some(0xA1B2C3D4),
            ..ScriptBotsConfig::default()
        };

        {
            let mut pipeline =
                StoragePipeline::create_unattributed_file_with_thresholds(&db_str, 1, 1, 1, 1)
                    .expect("pipeline");
            let (mut world, mut persistence) =
                WorldState::with_persistence(config.clone(), Box::new(pipeline.sink()))
                    .expect("world");
            let keys = install_brains(&mut world, BrainPreset::Mixed)
                .expect("install replay-fixture brains");
            seed_agents(&mut world, &keys.population).expect("seed replay-fixture brains");
            for _ in 0..16 {
                persistence
                    .step(&mut world)
                    .expect("durable replay fixture step");
            }
            let finalization = finalize_world_persistence(&mut world, &mut persistence);
            finalize_then_shutdown_storage(finalization, &mut pipeline)
                .expect("durable replay fixture finalization and shutdown");
        }

        let storage = StorageReader::open(&db_str).expect("open storage read-only");
        let recorded_events = storage.load_replay_events().expect("load events");
        let max_tick = storage.max_tick().expect("max tick").unwrap_or(0);
        storage.close().expect("close storage reader");
        assert_eq!(max_tick, 16, "fixture must persist its partial final tail");

        let replay =
            run_headless_simulation(&config, max_tick, BrainPreset::Mixed).expect("replay run");
        assert_eq!(replay.simulated_ticks, max_tick);
        assert_eq!(
            replay
                .summaries
                .iter()
                .map(|summary| summary.tick.0)
                .collect::<Vec<_>>(),
            [5, 10, 15, 16]
        );
        assert!(
            recorded_events.is_empty() && replay.events.is_empty(),
            "this cadence test must not masquerade as meaningful replay instrumentation"
        );
        let diff = diff_event_stream(&recorded_events, &replay.events);
        assert!(
            diff.is_none(),
            "empty event-stream plumbing should remain stable"
        );
    }

    #[test]
    fn runtime_error_still_commits_exact_partial_tail() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("runtime-error.sqlite");
        let db_str = db_path.to_string_lossy().to_string();
        let mut pipeline = StoragePipeline::create_unattributed_file_with_thresholds(
            &db_str, 64, 4096, 1024, 1024,
        )
        .expect("pipeline");
        let config = ScriptBotsConfig {
            persistence_interval: 5,
            population_minimum: 0,
            population_spawn_interval: 0,
            rng_seed: Some(0xC105_E0A1),
            ..ScriptBotsConfig::default()
        };
        let (mut initial_world, mut initial_persistence) =
            WorldState::with_persistence(config, Box::new(pipeline.sink())).expect("world");
        initial_persistence
            .step(&mut initial_world)
            .expect("non-boundary tick");
        let world = Arc::new(Mutex::new(initial_world));
        let persistence = Arc::new(Mutex::new(initial_persistence));

        let operation: Result<()> = Err(anyhow::anyhow!("injected renderer failure"));
        let error = finish_with_storage(operation, "runtime", || {
            finalize_and_shutdown_storage(&world, &persistence, &mut pipeline)
        })
        .expect_err("runtime error must survive acknowledged storage cleanup");
        assert!(format!("{error:#}").contains("injected renderer failure"));

        drop(world);
        let reader = StorageReader::open(&db_str).expect("open finalized database");
        assert_eq!(reader.max_tick().expect("read final tick"), Some(1));
        reader.close().expect("close reader");
    }

    #[test]
    fn shutdown_finalization_retries_a_newly_rejected_exact_tail() {
        struct RejectingPersistence {
            remaining_rejections: Arc<std::sync::atomic::AtomicUsize>,
            batches: Arc<Mutex<Vec<scriptbots_core::PersistenceBatch>>>,
        }

        impl WorldPersistence for RejectingPersistence {
            fn on_tick(
                &mut self,
                payload: &scriptbots_core::PersistenceBatch,
            ) -> std::result::Result<(), scriptbots_core::PersistenceAdmissionError> {
                self.batches
                    .lock()
                    .expect("batch log")
                    .push(payload.clone());
                let remaining = self
                    .remaining_rejections
                    .load(std::sync::atomic::Ordering::SeqCst);
                if remaining > 0 {
                    self.remaining_rejections
                        .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    Err(scriptbots_core::PersistenceAdmissionError::new(
                        payload.summary.tick.0,
                        "injected final-tail rejection",
                    ))
                } else {
                    Ok(())
                }
            }
        }

        let make_world = |rejections, batches: Arc<Mutex<Vec<_>>>| {
            let config = ScriptBotsConfig {
                persistence_interval: 5,
                population_minimum: 0,
                population_spawn_interval: 0,
                rng_seed: Some(0xD00D),
                ..ScriptBotsConfig::default()
            };
            WorldState::with_persistence(
                config,
                Box::new(RejectingPersistence {
                    remaining_rejections: Arc::new(std::sync::atomic::AtomicUsize::new(rejections)),
                    batches,
                }),
            )
            .expect("world")
        };

        let accepted_batches = Arc::new(Mutex::new(Vec::new()));
        let (mut accepted, mut accepted_persistence) = make_world(1, Arc::clone(&accepted_batches));
        accepted_persistence
            .step(&mut accepted)
            .expect("non-boundary tick");
        accepted_persistence
            .finalize(&mut accepted)
            .expect_err("first tail admission is rejected and retained");
        let mut disabled = accepted.config().clone();
        disabled.persistence_interval = 0;
        accepted
            .apply_config_update(disabled)
            .expect("disable persistence while the exact tail is retained");
        let finalization = finalize_world_persistence(&mut accepted, &mut accepted_persistence)
            .expect("exact retained final tail should succeed on bounded retry");
        assert!(finalization.admitted_tail);
        assert_eq!(finalization.required_tick, Some(1));
        assert!(!accepted_persistence.has_pending_batch());
        assert!(accepted_persistence.fault().is_none());
        let batches = accepted_batches.lock().expect("accepted batch log");
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].epoch, batches[1].epoch);
        assert_eq!(batches[0].closed, batches[1].closed);
        assert_eq!(batches[0].summary, batches[1].summary);
        assert_eq!(batches[0].metrics, batches[1].metrics);
        assert_eq!(batches[0].events, batches[1].events);
        assert!(batches.iter().all(|batch| batch.agents.is_empty()));
        assert_eq!(batches[0].births, batches[1].births);
        assert_eq!(batches[0].deaths, batches[1].deaths);
        assert_eq!(batches[0].replay_events, batches[1].replay_events);
        drop(batches);

        let rejected_batches = Arc::new(Mutex::new(Vec::new()));
        let (mut rejected, mut rejected_persistence) = make_world(2, Arc::clone(&rejected_batches));
        rejected_persistence
            .step(&mut rejected)
            .expect("non-boundary tick");
        let mut pipeline = StoragePipeline::unattributed_memory().expect("cleanup pipeline");
        let operation: Result<()> = Err(anyhow::anyhow!("injected runtime failure"));
        let error = finish_with_storage(operation, "runtime", || {
            let finalization = finalize_world_persistence(&mut rejected, &mut rejected_persistence);
            finalize_then_shutdown_storage(finalization, &mut pipeline)
        })
        .expect_err("second rejection and runtime error must both remain observable");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("failed to re-admit"));
        assert!(rendered.contains("injected runtime failure"));
        assert!(
            pipeline.shutdown().is_err(),
            "failed finalization must still close the storage worker"
        );
        assert!(rejected_persistence.has_pending_batch());
        assert!(rejected_persistence.fault().is_some());
        let batches = rejected_batches.lock().expect("rejected batch log");
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].epoch, batches[1].epoch);
        assert_eq!(batches[0].closed, batches[1].closed);
        assert_eq!(batches[0].summary, batches[1].summary);
        assert_eq!(batches[0].metrics, batches[1].metrics);
        assert_eq!(batches[0].events, batches[1].events);
        assert!(batches.iter().all(|batch| batch.agents.is_empty()));
        assert_eq!(batches[0].births, batches[1].births);
        assert_eq!(batches[0].deaths, batches[1].deaths);
        assert_eq!(batches[0].replay_events, batches[1].replay_events);
    }

    #[test]
    fn replay_verification_requires_nonempty_event_and_digest_evidence() {
        let error = require_non_vacuous_replay(16, 0, 0, 0, 0)
            .expect_err("empty nonzero replay must not be reported as verified");
        assert!(error.to_string().contains("refused a vacuous nonzero run"));
        assert!(error.to_string().contains("events recorded=0 simulated=0"));
        assert!(error.to_string().contains("digests recorded=0 simulated=0"));

        let missing_digests = require_non_vacuous_replay(16, 1, 1, 0, 0)
            .expect_err("event equality without digest evidence must fail closed");
        assert!(
            missing_digests
                .to_string()
                .contains("digests recorded=0 simulated=0")
        );

        let missing_events = require_non_vacuous_replay(16, 0, 0, 1, 1)
            .expect_err("digest equality without event evidence must fail closed");
        assert!(
            missing_events
                .to_string()
                .contains("events recorded=0 simulated=0")
        );

        require_non_vacuous_replay(16, 1, 1, 1, 1)
            .expect("nonempty event and digest counts satisfy the evidence gate");
        require_non_vacuous_replay(0, 0, 0, 0, 0).expect("a zero-tick replay is not vacuous");
    }

    #[test]
    #[serial]
    fn write_config_honors_format_and_exit_flag() {
        let dir = tempdir().expect("tempdir");
        let output = dir.path().join("effective.toml");
        let mut cli = default_cli();
        cli.write_config = Some(output.clone());
        cli.config_format = ConfigFormat::Toml;
        cli.config_only = true;

        let config = ScriptBotsConfig {
            world_width: 1234,
            world_height: 5678,
            rng_seed: Some(42),
            ..ScriptBotsConfig::default()
        };

        let outcome = maybe_emit_config(&cli, &config)
            .expect("emit config")
            .expect("expected emit outcome");
        assert_eq!(outcome, ConfigEmitOutcome::Exit);

        let written = fs::read_to_string(&output).expect("read output");
        assert!(written.contains("world_width = 1234"));
        assert!(written.contains("rng_seed = 42"));
    }

    #[test]
    #[serial]
    fn rng_seed_cli_override_applies() {
        let mut cli = default_cli();
        cli.rng_seed = Some(2025);
        let config = compose_config(&cli).expect("compose config");
        assert_eq!(config.rng_seed, Some(2025));
    }

    #[test]
    #[serial]
    fn emit_config_continue_when_not_config_only() {
        let dir = tempdir().expect("tempdir");
        let output = dir.path().join("effective.ron");
        let mut cli = default_cli();
        cli.write_config = Some(output.clone());
        cli.config_format = ConfigFormat::Ron;
        cli.config_only = false;

        let config = ScriptBotsConfig::default();

        let outcome = maybe_emit_config(&cli, &config)
            .expect("emit config")
            .expect("expected emit outcome");
        assert_eq!(outcome, ConfigEmitOutcome::Continue);

        let written = fs::read_to_string(&output).expect("read output");
        assert!(written.contains("world_width"));
    }

    fn restore_env(var: &str, previous: Option<String>) {
        unsafe {
            if let Some(value) = previous {
                std::env::set_var(var, value);
            } else {
                std::env::remove_var(var);
            }
        }
    }

    /// Every environment variable the CLI/configuration gatherers read while composing a
    /// scenario. Tests that need deterministic provenance clear all of them: an ambient brain
    /// preset changes the population recipe, while an ambient configuration value adds a
    /// digest-recorded layer statement.
    const CONFIG_ENV_VARS: [&str; 14] = [
        "SCRIPTBOTS_BRAIN",
        "SCRIPTBOTS_CONFIG_OVERRIDES",
        "SCRIPTBOTS_NEUROFLOW_ENABLED",
        "SCRIPTBOTS_NEUROFLOW_HIDDEN",
        "SCRIPTBOTS_NEUROFLOW_ACTIVATION",
        "SCRIPTBOTS_RENDER_TONEMAP",
        "SCRIPTBOTS_RENDER_TONEMAP_BIAS",
        "SCRIPTBOTS_RENDER_AUTO_EXPOSURE",
        "SCRIPTBOTS_RENDER_AUTO_EXPOSURE_SPEED_BRIGHTEN",
        "SCRIPTBOTS_RENDER_AUTO_EXPOSURE_SPEED_DARKEN",
        "SCRIPTBOTS_RNG_SEED",
        "SCRIPTBOTS_AUTO_PAUSE_BELOW",
        "SCRIPTBOTS_AUTO_PAUSE_AGE_ABOVE",
        "SCRIPTBOTS_AUTO_PAUSE_ON_SPIKE",
    ];

    fn with_clean_config_env<F: FnOnce()>(f: F) {
        with_env_lock(|| {
            let saved: Vec<(&str, Option<String>)> = CONFIG_ENV_VARS
                .iter()
                .map(|var| (*var, std::env::var(var).ok()))
                .collect();
            for (var, _) in &saved {
                unsafe { std::env::remove_var(var) };
            }
            f();
            for (var, previous) in saved {
                restore_env(var, previous);
            }
        });
    }

    #[test]
    #[serial]
    fn env_overrides_apply_expected_settings() {
        with_clean_config_env(|| {
            unsafe {
                std::env::set_var("SCRIPTBOTS_NEUROFLOW_ENABLED", "true");
                std::env::set_var("SCRIPTBOTS_NEUROFLOW_HIDDEN", "64, 32 ,16");
                std::env::set_var("SCRIPTBOTS_NEUROFLOW_ACTIVATION", "relu");
            }

            let config = compose_config(&default_cli()).expect("compose with environment layer");

            assert!(config.neuroflow.enabled);
            assert_eq!(config.neuroflow.hidden_layers, vec![64, 32, 16]);
            assert_eq!(config.neuroflow.activation, NeuroflowActivationKind::Relu);
        });
    }

    #[test]
    #[serial]
    fn a_file_loses_to_the_environment_which_loses_to_the_cli() {
        // The bead's acceptance chain driven through the real composition path: a
        // scenario file that sets `world_width` loses to an environment variable that
        // sets it, which loses to a CLI flag — and every displacement is on the record.
        with_clean_config_env(|| {
            let dir = tempdir().expect("tempdir");
            let file_path = dir.path().join("scenario.toml");
            fs::write(&file_path, b"world_width = 2000\n").expect("write scenario layer");
            unsafe {
                std::env::set_var("SCRIPTBOTS_CONFIG_OVERRIDES", "world_width = 1000");
            }

            let mut cli = default_cli();
            cli.config_layers = vec![file_path];
            cli.set_overrides = vec!["world_width=500".to_owned()];
            let (config, scenario, overrides) =
                compose_config_with_scenario(&cli).expect("compose layered config");

            assert_eq!(
                config.world_width, 500,
                "the CLI names the value for THIS invocation and must win"
            );
            assert_eq!(
                overrides.len(),
                2,
                "both displacements must be on the record: {overrides:?}"
            );
            assert_eq!(overrides[0].path, "world_width");
            assert_eq!(overrides[0].losing_kind, ConfigLayerKind::File);
            assert_eq!(overrides[0].winning_kind, ConfigLayerKind::Environment);
            assert_eq!(overrides[1].path, "world_width");
            assert_eq!(overrides[1].losing_kind, ConfigLayerKind::Environment);
            assert_eq!(overrides[1].winning_kind, ConfigLayerKind::Cli);

            // Every layer that spoke appended a kind-tagged digest, in application
            // order — including the defaults every run starts from.
            let kinds: Vec<&str> = scenario
                .ordered_config_layer_digests
                .iter()
                .map(|entry| entry.split(':').next().unwrap_or(""))
                .collect();
            assert_eq!(kinds, vec!["defaults", "file", "environment", "cli"]);
        });
    }

    #[test]
    #[serial]
    fn a_default_only_run_still_records_the_defaults_digest() {
        with_clean_config_env(|| {
            let (_config, scenario, overrides) =
                compose_config_with_scenario(&default_cli()).expect("compose defaults only");
            assert!(overrides.is_empty(), "nothing spoke, nothing displaced");
            let kinds: Vec<&str> = scenario
                .ordered_config_layer_digests
                .iter()
                .map(|entry| entry.split(':').next().unwrap_or(""))
                .collect();
            assert_eq!(
                kinds,
                vec!["defaults"],
                "a run configured by nothing but the defaults must still name its one layer"
            );
        });
    }

    #[test]
    #[serial]
    fn an_explicit_env_false_displaces_a_file_layer_true() {
        // `SCRIPTBOTS_AUTO_PAUSE_ON_SPIKE=false` is a statement, not an absence: it
        // must displace a scenario file's `true` and appear in the override record.
        with_clean_config_env(|| {
            let dir = tempdir().expect("tempdir");
            let file_path = dir.path().join("spike.toml");
            fs::write(&file_path, b"[control]\nauto_pause_on_spike_hit = true\n")
                .expect("write spike layer");
            unsafe {
                std::env::set_var("SCRIPTBOTS_AUTO_PAUSE_ON_SPIKE", "false");
            }

            let mut cli = default_cli();
            cli.config_layers = vec![file_path];
            let (config, _scenario, overrides) =
                compose_config_with_scenario(&cli).expect("compose spike layers");

            assert!(
                !config.control.auto_pause_on_spike_hit,
                "the explicit environment false must win over the file's true"
            );
            assert_eq!(overrides.len(), 1, "unexpected overrides: {overrides:?}");
            assert_eq!(overrides[0].path, "control.auto_pause_on_spike_hit");
            assert_eq!(overrides[0].losing_kind, ConfigLayerKind::File);
            assert_eq!(overrides[0].winning_kind, ConfigLayerKind::Environment);
        });
    }

    #[test]
    #[serial]
    fn unknown_generic_override_paths_fail_closed() {
        with_clean_config_env(|| {
            // A top-level typo through --set.
            let mut cli = default_cli();
            cli.set_overrides = vec!["world_widht=800".to_owned()];
            let error = compose_config(&cli).expect_err("a typo'd field must not vanish");
            let rendered = format!("{error:#}");
            assert!(
                rendered.contains("world_widht") && rendered.contains("not a configuration field"),
                "the error must name the unknown path: {rendered}"
            );

            // A nested typo through the environment document.
            unsafe {
                std::env::set_var("SCRIPTBOTS_CONFIG_OVERRIDES", "neuroflow.enabld = true");
            }
            let error =
                compose_config(&default_cli()).expect_err("a nested typo'd field must not vanish");
            let rendered = format!("{error:#}");
            assert!(
                rendered.contains("neuroflow.enabld")
                    && rendered.contains("not a configuration field"),
                "the error must name the nested unknown path: {rendered}"
            );
        });
    }

    #[test]
    #[serial]
    fn set_override_rejects_entries_that_name_no_field() {
        with_clean_config_env(|| {
            let mut cli = default_cli();
            cli.set_overrides = vec!["world_width".to_owned()];
            let error = compose_config(&cli).expect_err("PATH without VALUE must fail");
            assert!(
                format!("{error:#}").contains("--set"),
                "the error must name the flag: {error:#}"
            );
        });
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn neuroflow_installation_propagates_typed_configuration_error() {
        let mut config = ScriptBotsConfig::default();
        config.neuroflow.enabled = true;
        config.neuroflow.hidden_layers = vec![4, 0, 2];
        let mut world = WorldState::new(config).expect("world accepts adapter-owned settings");
        let before = world.brain_registry().descriptors();

        let error = install_brains(&mut world, BrainPreset::Mixed)
            .expect_err("invalid NeuroFlow dimensions must fail scenario registration");
        let source = error
            .downcast_ref::<scriptbots_brain_neuro::NeuroflowBrainError>()
            .expect("typed NeuroFlow error must remain in the startup error chain");
        assert_eq!(source.field(), Some("hidden_layers[1]"));
        assert!(format!("{error:#}").contains("failed to validate configured"));
        assert_eq!(world.brain_registry().descriptors(), before);
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn single_family_preset_does_not_register_or_validate_neuroflow() {
        let mut config = ScriptBotsConfig::default();
        config.neuroflow.enabled = true;
        config.neuroflow.hidden_layers = vec![4, 0, 2];
        let mut world = WorldState::new(config).expect("world accepts adapter-owned settings");

        let installed = install_brains(&mut world, BrainPreset::Mlp)
            .expect("an explicit MLP-only scenario must not activate NeuroFlow");
        assert_eq!(installed.registered(), 1);
        assert!(installed.withheld.is_empty());
        assert_eq!(
            world.brain_registry().descriptors(),
            vec![(installed.population[0], MlpBrain::KIND.as_str().to_owned())]
        );
    }

    #[cfg(feature = "neuro")]
    #[test]
    #[serial]
    fn invalid_neuroflow_configuration_precedes_storage_reservation() {
        with_env_lock(|| {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("must-not-be-reserved.sqlite");
            let previous = std::env::var("SCRIPTBOTS_STORAGE_PATH").ok();
            unsafe {
                std::env::set_var("SCRIPTBOTS_STORAGE_PATH", &path);
            }

            let mut config = ScriptBotsConfig::default();
            config.neuroflow.enabled = true;
            config.neuroflow.hidden_layers = vec![4, 0, 2];
            let error = bootstrap_world(
                config,
                BrainPreset::Mixed,
                StorageMode::File,
                ThresholdsOverride::default(),
                DEFAULT_BOOTSTRAP_TICKS,
                resolve_thread_policy(None, None, None, false),
                ScenarioIdentityV0::caller_seeded("invalid-neuroflow-test"),
                Vec::new(),
            )
            .err()
            .expect("adapter validation must fail before storage setup");
            let path_exists = path.exists();
            restore_env("SCRIPTBOTS_STORAGE_PATH", previous);

            assert!(
                error
                    .downcast_ref::<scriptbots_brain_neuro::NeuroflowBrainError>()
                    .is_some(),
                "typed NeuroFlow error must survive bootstrap context: {error:#}"
            );
            assert!(
                !path_exists,
                "invalid adapter configuration must not reserve the requested run database"
            );
        });
    }

    #[cfg(feature = "neuro")]
    #[test]
    #[serial]
    fn invalid_neuroflow_configuration_precedes_profile_storage_reservation() {
        with_env_lock(|| {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("profile-must-not-be-reserved.sqlite");
            let previous = std::env::var("SCRIPTBOTS_STORAGE_PATH").ok();
            unsafe {
                std::env::set_var("SCRIPTBOTS_STORAGE_PATH", &path);
            }

            let mut config = ScriptBotsConfig::default();
            config.neuroflow.enabled = true;
            config.neuroflow.hidden_layers = vec![4, 0, 2];
            let error = profile_world_steps_with_storage(
                &config,
                1,
                BrainPreset::Mixed,
                StorageMode::File,
                ThresholdsOverride::default(),
                resolve_thread_policy(None, None, None, false),
                ScenarioIdentityV0::caller_seeded("invalid-neuroflow-profile-test"),
                Vec::new(),
            )
            .expect_err("adapter validation must fail before profiling storage setup");
            let path_exists = path.exists();
            restore_env("SCRIPTBOTS_STORAGE_PATH", previous);

            assert!(
                error
                    .downcast_ref::<scriptbots_brain_neuro::NeuroflowBrainError>()
                    .is_some(),
                "typed NeuroFlow error must survive profiling context: {error:#}"
            );
            assert!(
                !path_exists,
                "invalid adapter configuration must not reserve a profiling run database"
            );
        });
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn neuroflow_installation_respects_toggle() {
        let expected_protocol_families = 3 + usize::from(cfg!(feature = "brain-ft"));
        let mut config = ScriptBotsConfig::default();
        config.neuroflow.enabled = false;
        let mut world = WorldState::new(config).expect("world");
        let keys = install_brains(&mut world, BrainPreset::Mixed).expect("install baseline brains");
        assert_eq!(
            keys.registered(),
            expected_protocol_families,
            "NeuroFlow brain should not register when disabled"
        );
        assert_eq!(keys.population.len(), expected_protocol_families);
        assert!(keys.withheld.is_empty());
        for key in &keys.population {
            assert!(world.brain_registry().is_protocol_family(*key));
            assert!(world.brain_registry().family(*key).is_some());
        }

        let mut config_enabled = ScriptBotsConfig::default();
        config_enabled.neuroflow.enabled = true;
        config_enabled.neuroflow.hidden_layers = vec![12, 6];
        config_enabled.neuroflow.activation = NeuroflowActivationKind::Sigmoid;
        config_enabled.rng_seed = Some(99);
        let mut world_enabled = WorldState::new(config_enabled.clone()).expect("world");
        let keys_enabled = install_brains(&mut world_enabled, BrainPreset::Mixed)
            .expect("install enabled NeuroFlow brain");
        assert_eq!(
            keys_enabled.registered(),
            expected_protocol_families + 1,
            "expected every compiled protocol family plus explicitly selectable NeuroFlow"
        );
        assert_eq!(
            keys_enabled.population.len(),
            expected_protocol_families,
            "NeuroFlow has no protocol codec and must not enter the founding population"
        );
        for key in &keys_enabled.population {
            assert!(world_enabled.brain_registry().is_protocol_family(*key));
            assert!(world_enabled.brain_registry().family(*key).is_some());
        }

        let (neuro_label, neuro_key) = keys_enabled
            .withheld
            .iter()
            .find(|(label, _)| label.contains("neuroflow"))
            .expect("enabled NeuroFlow must be registered but withheld");
        let neuro_key = *neuro_key;
        assert_eq!(
            neuro_label.as_str(),
            scriptbots_brain_neuro::NeuroflowBrain::KIND.as_str()
        );
        assert!(!world_enabled.brain_registry().is_protocol_family(neuro_key));
        assert!(world_enabled.brain_registry().family(neuro_key).is_none());
        let agent_id = world_enabled
            .try_spawn_agent(AgentData::default())
            .expect("default agent is finite");
        assert!(
            world_enabled
                .bind_agent_brain(agent_id, neuro_key)
                .expect("construct enabled NeuroFlow runner")
        );
        world_enabled
            .step()
            .expect("enabled NeuroFlow simulation step");
        let outputs_one = world_enabled.agent_runtime(agent_id).unwrap().outputs;

        let mut world_repeat = WorldState::new(config_enabled).expect("world");
        let keys_repeat = install_brains(&mut world_repeat, BrainPreset::Mixed)
            .expect("install repeat NeuroFlow brain");
        assert_eq!(keys_repeat.registered(), expected_protocol_families + 1);
        assert_eq!(keys_repeat.population.len(), expected_protocol_families);
        let neuro_repeat = keys_repeat
            .withheld
            .iter()
            .find(|(label, _)| label.contains("neuroflow"))
            .map(|(_, key)| *key)
            .expect("repeat NeuroFlow registration must remain withheld");
        assert!(
            !world_repeat
                .brain_registry()
                .is_protocol_family(neuro_repeat)
        );
        assert!(world_repeat.brain_registry().family(neuro_repeat).is_none());
        let agent_repeat = world_repeat
            .try_spawn_agent(AgentData::default())
            .expect("default agent is finite");
        assert!(
            world_repeat
                .bind_agent_brain(agent_repeat, neuro_repeat)
                .expect("construct repeat NeuroFlow runner")
        );
        world_repeat
            .step()
            .expect("repeat NeuroFlow simulation step");
        let outputs_two = world_repeat.agent_runtime(agent_repeat).unwrap().outputs;

        assert_eq!(
            outputs_one, outputs_two,
            "NeuroFlow outputs should be deterministic for same seed"
        );
    }
}
