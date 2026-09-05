use anyhow::{Context, Result, anyhow, bail};
use asupersync::types::{Budget, Outcome};
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use owo_colors::OwoColorize;
use ron::ser::PrettyConfig as RonPrettyConfig;
use scriptbots_app::archipelago_report::{self, ReportArchipelagoArgs};
use scriptbots_app::economy_audit::{self, EconomyAuditArgs};
#[cfg(feature = "neuro")]
use scriptbots_app::validated_neuroflow_config;
use scriptbots_app::{
    BootstrapEvidenceV0, BrainPreset, CharacterizationTraceV2, ControlServerConfig,
    ControlServerReservation, RunIdentityV1, RunManifestV3, ScenarioDocumentV1, ScenarioIdentityV0,
    SharedAnalytics, SharedWorld, ThreadPolicyV0, WorldStepDriver, install_brains,
    precedence::{
        ConfigFieldOverride, ConfigLayerKind, ConfigLayerStatement, ThreadPolicy, ThreadSource,
        canonical_layer_bytes, resolve_config_layers, resolve_thread_policy,
    },
    regions::{AppRoot, RegionOutcome, ServiceRegion},
    renderer::{Renderer, RendererContext},
    terminal::TerminalRenderer,
    write_atomic_manifest_sidecar,
};
#[cfg(feature = "bevy_render")]
use scriptbots_bevy::{BevyRendererContext, render_png_offscreen as render_bevy_png};
#[cfg(test)]
use scriptbots_brain::{AssemblyBrain, DwraonBrain, MlpBrain};
use scriptbots_core::{
    LEGACY_RENDER_ENV_NAMES, NeuroflowActivationKind, NullPersistence, PersistenceAdmissionSession,
    PersistenceSessionError, RenderQuality, RenderTonemapMode, ReplayEventKind,
    ReplayInteractionKind, ScriptBotsConfig, TickSummary, WorldDigestV1, WorldPersistence,
    WorldState, map_legacy_render_env, parse_render_quality,
};
#[cfg(feature = "gui")]
use scriptbots_render::{render_png_offscreen, run_demo};
use scriptbots_runtime::RunId;
use scriptbots_storage::{
    INTERACTION_REPLAY_SEQ_BASE, NARRATIVE_INPUT_REPLAY_SEQ, PersistedReplayEvent,
    PersistenceGuarantee, ShutdownReceipt, StoragePipeline, StorageReader,
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
    latest_summary: &scriptbots_app::control::SharedLatestSummary,
) -> WorldStepDriver {
    let world = Arc::clone(world);
    let session = Arc::clone(session);
    let latest_summary = Arc::clone(latest_summary);
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
        let outcome = session.step(&mut world)?;
        // Publish the completed summary outside the mutex protocol (bd-134):
        // control surfaces read this slot wait-free instead of contending on
        // the world lock the next tick will hold.
        if let Some(summary) = world.history().next_back() {
            latest_summary.store(Some(std::sync::Arc::new(summary.clone())));
        }
        Ok(outcome)
    })
}

fn main() -> Result<()> {
    // FIRST, before anything can mutate the process environment: pin the launch
    // environment so build provenance records what the user exported, not what
    // startup's thread-policy set_var smeared over it (bd-3p7i).
    let _launch_environment = scriptbots_app::LaunchEnvironmentV0::pin();
    let cli = AppCli::parse();
    init_tracing();

    if let Some(AppSubcommand::EconomyAudit(ref audit_args)) = cli.subcommand {
        let pass = economy_audit::run_economy_audit(audit_args)?;
        if pass {
            std::process::exit(0);
        } else {
            std::process::exit(1);
        }
    }

    if let Some(AppSubcommand::ReportArchipelago(ref report_args)) = cli.subcommand {
        let pass = archipelago_report::run_archipelago_report(report_args)?;
        if pass {
            std::process::exit(0);
        } else {
            std::process::exit(1);
        }
    }

    if let Some(ref db_path) = cli.report_archipelago {
        let report_args = ReportArchipelagoArgs {
            db: db_path.clone(),
            json: None,
            verify_conservation: false,
        };
        let pass = archipelago_report::run_archipelago_report(&report_args)?;
        if pass {
            std::process::exit(0);
        } else {
            std::process::exit(1);
        }
    }

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
    let (config, mut launch_scenario, config_overrides) = compose_config_with_scenario(&cli)?;
    // Resolve the effective bootstrap policy once: an explicit CLI/env value outranks
    // the scenario document's policy; the document outranks the built-in zero default.
    // The manifest, REST scenario view, and TUI header all read this same value.
    let effective_bootstrap_ticks = if cli_bootstrap_explicit(&cli) {
        cli.bootstrap_ticks
    } else {
        launch_scenario.bootstrap_ticks
    };
    launch_scenario.bootstrap_ticks = effective_bootstrap_ticks;
    let launch_scenario_shared = Arc::new(launch_scenario.clone());

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
        let mut control_config = ControlServerConfig::try_from_env()?;
        control_config.scenario = Some(Arc::clone(&launch_scenario_shared));
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

    if let Some(islands) = cli.run_archipelago {
        ensure_recorded_archipelago_scenario(&launch_scenario)?;
        let path = cli
            .archipelago_db
            .as_ref()
            .context("--archipelago-db is required")?;
        let path = path
            .to_str()
            .context("archipelago database path is not UTF-8")?;
        let ticks = cli.archipelago_ticks;
        let env_threads = std::env::var("SCRIPTBOTS_MAX_THREADS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|threads| *threads > 0);
        let mut policy = resolve_thread_policy(cli.threads, env_threads, None, cli.low_power);
        let threads = policy.threads.unwrap_or(1);
        if threads == 0 {
            bail!("recorded archipelago requires a positive thread count");
        }
        policy.threads = Some(threads);
        let identity = RunIdentityV1::new(
            allocate_run_id(),
            run_started_at_unix_ms()?,
            Some(ticks),
            None,
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        let result = pool.install(|| {
            archipelago_report::run_recorded_archipelago(
                config,
                islands,
                ticks,
                cli.brain,
                path,
                |world| {
                    let manifest = build_run_manifest(
                        world,
                        identity,
                        launch_scenario,
                        policy,
                        config_overrides,
                    )?;
                    Ok(manifest.to_storage_record()?)
                },
            )
        })?;
        println!("{}", serde_json::to_string(&result)?);
        return Ok(());
    }

    if let Some(ref run_db) = cli.create_bundle {
        let output_dir = cli
            .bundle_output
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("{}-bundle", run_db.display())));
        run_create_bundle_cli(run_db, &output_dir)?;
        return Ok(());
    }

    if let Some(ref bundle_path) = cli.verify_bundle {
        run_verify_bundle_cli(bundle_path)?;
        return Ok(());
    }

    if let Some(ref goal) = cli.lab_goal {
        run_lab_cli(&cli, goal)?;
        return Ok(());
    }

    if let Some(ticks) = cli.det_check {
        run_det_check(&cli, ticks)?;
        return Ok(());
    }
    if let Some(ticks) = cli.det_check_archipelago {
        run_archipelago_det_check(&cli, ticks)?;
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
        // Profile dispatch is the other pre-thread path, and it publishes only the
        // thread cap: it never reaches a renderer, so the render toggles and the GPU
        // adapter preference have no meaning here (bd-o0cq).
        PreThreadEnvironment {
            max_threads: profile_thread_policy
                .threads
                .filter(|_| profile_thread_policy.source != ThreadSource::Environment),
            render_watermark: false,
            render_safe: false,
            prefer_high_performance_gpu: false,
        }
        .publish();
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
    tracing::debug!(
        sensor_layout_digest = scriptbots_core::SENSOR_LAYOUT_DIGEST,
        sensor_count = scriptbots_core::INPUT_SIZE,
        "canonical sensor layout loaded"
    );
    // One audited environment publication for the whole interactive startup, rather
    // than four scattered `unsafe` blocks (bd-o0cq).
    PreThreadEnvironment {
        // Only write the variable when a layer other than the environment decided;
        // rewriting it with the value it already holds is noise, and rewriting it
        // with a DIFFERENT value is the clobber `precedence` exists to prevent.
        max_threads: policy
            .threads
            .filter(|_| policy.source != ThreadSource::Environment),
        render_watermark: cli.debug_watermark,
        render_safe: cli.renderer_safe || cli.low_power,
        prefer_high_performance_gpu: true,
    }
    .publish();

    // Apply OS-level priority niceness where supported.
    apply_process_niceness(cli.low_power)?;
    let (bootstrapped_world, persistence, analytics, mut storage_pipeline) = bootstrap_world(
        config,
        BootstrapRequest {
            brain_preset: cli.brain,
            storage_mode: cli.storage,
            thresholds,
            bootstrap_ticks: effective_bootstrap_ticks,
            thread_policy: policy,
            scenario: launch_scenario,
            config_overrides,
        },
    )?;
    // The wrap happens HERE now, not inside bootstrap_world. This is the line
    // bd-pcfj replaces with a move into the host thread.
    let world: SharedWorld = Arc::new(Mutex::new(bootstrapped_world));
    let latest_summary = scriptbots_app::control::empty_latest_summary();
    let simulation_step = persistence_step_driver(&world, &persistence, &latest_summary);

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
                            .map_err(|error| anyhow::anyhow!("wgpu snapshot failed: {error}"))?
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
        if let Some(path) = cli.dump_semantic_png.as_ref() {
            // Semantic projection only: this CPU rasterizer does NOT exercise the
            // GPU pipeline (bd-2z0.14.3.4 renamed it to make the dishonesty
            // impossible to miss); --dump-scene-png owns real GPU captures.
            //
            // Deliberately NO probe_gpu_capability() call here. It used to log a
            // capability report, which meant a command documented as CPU-only
            // built a wgpu instance and requested an adapter — real GPU work,
            // and a report describing hardware this path never uses. That made
            // the semantic reference lane unable to prove it is GPU-free, and
            // on a headless host it charged adapter enumeration for a raster
            // that cannot use one (bd-2z0.14.3.4 round-1 audit). The
            // no_gpu_touch alarm below pins this.
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
                "{} Wrote semantic projection {} ({}x{}; CPU reference raster, NOT GPU-rendered)",
                "\u{2714}".green().bold(),
                path.display(),
                w,
                h
            );
            return Ok(());
        }
        #[cfg(feature = "bevy_render")]
        if let Some(scene_path) = cli.dump_scene_png.as_ref() {
            return run_scene_capture_cli(scene_path);
        }

        let (active_mode, renderer) = resolved_renderer.ok_or_else(|| {
            anyhow::anyhow!("interactive renderer was not resolved before startup")
        })?;
        let control_reservation = control_reservation.ok_or_else(|| {
            anyhow::anyhow!("control listeners were not reserved before runtime startup")
        })?;
        let (control_runtime, command_drain, command_submit) =
            control_reservation.launch(world.clone(), latest_summary.clone())?;
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
            scenario: Arc::clone(&launch_scenario_shared),
        };
        let render_result = renderer.run(context);
        // bd-2z0.4.13: the app entrypoint owns an AppRoot whose regions tear down in
        // reverse dependency order with explicit budgets and per-region outcomes.
        // Control closes first (stop accepting), storage drains last (every producer
        // quiesced before the durable watermark advances).
        let mut root = AppRoot::new();
        let world_for_storage = Arc::clone(&world);
        let persistence_for_storage = Arc::clone(&persistence);
        root.register(ServiceRegion::new(
            "storage-pipeline",
            Budget::with_deadline_at_secs(30),
            move |_budget| match finalize_and_shutdown_storage(
                &world_for_storage,
                &persistence_for_storage,
                &mut storage_pipeline,
            ) {
                Ok(()) => Outcome::ok("storage drained to the durable watermark".to_owned()),
                Err(error) => Outcome::Err(format!("{error:#}")),
            },
        ));
        root.register(ServiceRegion::new(
            "control-server",
            Budget::with_deadline_at_secs(15),
            move |_budget| match control_runtime.shutdown() {
                Ok(()) => Outcome::ok("control runtime shut down".to_owned()),
                Err(error) => Outcome::Err(format!("{error:#}")),
            },
        ));
        let outcomes = root.close();
        let control_result = region_result(&outcomes, "control-server");
        let storage_result = region_result(&outcomes, "storage-pipeline");
        prefer_storage_failure(
            match (render_result, control_result) {
                (Ok(()), Ok(())) => Ok(()),
                (Err(render_error), Ok(())) => Err(render_error),
                (Ok(()), Err(control_error)) => Err(control_error),
                (Err(render_error), Err(control_error)) => Err(render_error).context(format!(
                    "control runtime shutdown also failed: {control_error:#}"
                )),
            },
            storage_result,
            "runtime",
        )
    })();
    let result = runtime_result;
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

/// Map one region's recorded outcome back into the `Result` contract the caller
/// already had, preserving the typed error in the message.
fn region_result(outcomes: &[RegionOutcome], name: &str) -> Result<()> {
    let Some(region) = outcomes.iter().find(|outcome| outcome.name == name) else {
        return Err(anyhow!("region {name} reported no teardown outcome"));
    };
    match &region.outcome {
        Outcome::Ok(_) => Ok(()),
        Outcome::Err(error) => Err(anyhow!(error.clone())),
        Outcome::Cancelled(reason) => Err(anyhow!(
            "region {name} exhausted its teardown budget: {reason:?}"
        )),
        Outcome::Panicked(payload) => Err(anyhow!("region {name} finalizer panicked: {payload}")),
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

/// Every process-environment value startup publishes for a later reader.
///
/// These four variables are consumed through the environment because their readers
/// require it, not by preference: `SCRIPTBOTS_MAX_THREADS` is read by
/// `scriptbots-core` when it caps Rayon, `SCRIPTBOTS_RENDER_SAFE` and
/// `SCRIPTBOTS_RENDER_WATERMARK` by `scriptbots-render`, and
/// `WGPU_POWER_PREFERENCE` by `wgpu` itself, which offers no programmatic
/// equivalent. Collecting them in one value keeps the set closed and reviewable and
/// reduces startup from four separate `unsafe` blocks to the single audited write
/// below (bd-o0cq).
///
/// A `None`/`false` field means "do not speak for this variable", which is not the
/// same as clearing it: an operator's exported value must survive a layer that
/// declined to override it.
struct PreThreadEnvironment {
    /// Resolved worker-thread cap, when a layer other than the environment won.
    max_threads: Option<usize>,
    /// Overlay the diagnostics watermark in the render canvas.
    render_watermark: bool,
    /// Force the conservative paint path.
    render_safe: bool,
    /// Ask wgpu for the high-performance adapter (Windows only, and only if the
    /// operator has not already expressed a preference).
    prefer_high_performance_gpu: bool,
}

impl PreThreadEnvironment {
    /// Publish the requested values into the process environment.
    ///
    /// Must be called only from the startup path, before any world, Rayon pool,
    /// renderer, or control-runtime thread exists.
    fn publish(&self) {
        // SAFETY: `std::env::set_var` is unsound only when it races another thread
        // touching the environment. This is the one audited write for the whole
        // binary and every caller is on the single-threaded startup path: profile
        // dispatch runs before any world is constructed, and the interactive path
        // runs before `bootstrap_world`, the Rayon pool, the control runtime, and
        // every renderer. `LaunchEnvironmentV0::pin()` has already captured the
        // operator's pre-publication environment for build provenance (bd-3p7i), so
        // these writes cannot retroactively rewrite what the manifest reports.
        unsafe {
            if let Some(threads) = self.max_threads {
                std::env::set_var("SCRIPTBOTS_MAX_THREADS", threads.to_string());
            }
            if self.render_watermark {
                std::env::set_var("SCRIPTBOTS_RENDER_WATERMARK", "1");
            }
            if self.render_safe {
                std::env::set_var("SCRIPTBOTS_RENDER_SAFE", "1");
            }
            #[cfg(windows)]
            if self.prefer_high_performance_gpu && std::env::var("WGPU_POWER_PREFERENCE").is_err() {
                std::env::set_var("WGPU_POWER_PREFERENCE", "high_performance");
            }
        }
        // The adapter preference exists only on Windows; reference the field
        // elsewhere so a non-Windows build does not warn about one it cannot use.
        #[cfg(not(windows))]
        let _ = self.prefer_high_performance_gpu;
    }
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
    let raw_env = std::env::var("RUST_LOG").unwrap_or_default();
    let mut filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    if !raw_env.contains("fsqlite") {
        for target in [
            "fsqlite=warn",
            "fsqlite_mvcc=warn",
            "fsqlite_vdbe=warn",
            "fsqlite_core=warn",
            "fsqlite_planner=warn",
        ] {
            if let Ok(directive) = target.parse() {
                filter = filter.add_directive(directive);
            }
        }
    }
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
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
        world_digest: WorldDigestV1,
    }
    let last_tick = run.summaries.last().map(|s| s.tick.0).unwrap_or(0);
    let out = DetOut {
        events: run.events.len(),
        ticks: run.simulated_ticks,
        last_tick,
        summaries: run.summaries,
        world_digest: run.final_digest,
    };
    let json = serde_json::to_string(&out)?;
    println!("{}", json);
    Ok(())
}

fn run_lab_cli(cli: &AppCli, goal: &str) -> Result<()> {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    #[cfg(feature = "llm-anthropic")]
    use scriptbots_app::lab::llm::AnthropicClient;
    use scriptbots_app::lab::llm::{LlmClient, ScriptedClient};
    use scriptbots_app::lab_assistant::{LabBudget, LabStateMachine};

    let output_root = cli
        .lab_out
        .clone()
        .unwrap_or_else(|| PathBuf::from("runs/lab"));
    fs::create_dir_all(&output_root)
        .with_context(|| format!("create lab output directory: {}", output_root.display()))?;

    let budget = LabBudget {
        max_runs: cli.lab_runs,
        max_ticks: cli.lab_ticks,
        max_tokens: 100_000,
        max_iterations: 20,
    };

    let rng_seed = cli.rng_seed.unwrap_or(42);
    let rng = SmallRng::seed_from_u64(rng_seed);

    let client: Box<dyn LlmClient> = if let Some(ref fixture_path) = cli.lab_fixture {
        let bytes = fs::read(fixture_path)
            .with_context(|| format!("read lab fixture from {}", fixture_path.display()))?;
        let fixture_client = ScriptedClient::from_fixture("offline-fixture", &bytes)
            .map_err(|err| anyhow!("invalid lab fixture: {err}"))?;
        Box::new(fixture_client)
    } else {
        #[cfg(feature = "llm-anthropic")]
        {
            let anthropic = AnthropicClient::from_env("claude-3-5-sonnet-20241022", rng)
                .map_err(|err| anyhow!("could not initialize Anthropic client: {err}"))?;
            Box::new(anthropic)
        }
        #[cfg(not(feature = "llm-anthropic"))]
        {
            let _ = rng;
            bail!(
                "Autonomous lab assistant requires either an offline scripted fixture (--lab-fixture <FILE>) \
                 or the binary compiled with --features llm-anthropic and ANTHROPIC_API_KEY exported."
            );
        }
    };

    let mut state_machine = LabStateMachine::new(client, budget, output_root);

    info!(goal = %goal, "starting autonomous lab assistant loop");
    let phase = state_machine
        .run_to_completion()
        .map_err(|err| anyhow!("lab assistant execution failure: {err}"))?;

    println!("Autonomous Lab Assistant completed with final phase: {phase:?}");
    if let Some(path) = state_machine.notebook_path() {
        println!("Rendered lab notebook: {}", path.display());
    }

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
        world_digest: WorldDigestV1,
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
    if left.world_digest != right.world_digest {
        bail!(
            "WorldDigestV1 mismatch: 1t={} vs Nt={}",
            left.world_digest.overall,
            right.world_digest.overall
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

fn run_archipelago_det_check(cli: &AppCli, ticks: u64) -> Result<()> {
    use scriptbots_runtime::archipelago::{
        Archipelago, ArchipelagoConfig, ArchipelagoMigration, IslandId, IslandSpec,
    };
    use scriptbots_runtime::migrator::EmigrantSelectionRule;

    println!(
        "{} Starting archipelago determinism self-check matrix for {} ticks...",
        "ℹ".blue().bold(),
        ticks
    );

    let master_seed = cli.rng_seed.unwrap_or(987_654_321);
    let mut base_config = compose_config(cli)?;
    base_config.rng_seed = Some(master_seed);

    const ISLANDS: u32 = 8;
    let barriers = 4u32;
    let barrier_interval = (ticks / u64::from(barriers)).max(1);

    // MIGRATION IS ON. It used to be disabled here, with the comment that it
    // "would couple the islands it compares" -- but coupling is not the problem
    // a determinism gate has. The requirement is that the SAME coupling
    // reproduces, and the migrator is the component most likely to carry an
    // order dependence, so excluding it removed exactly what most needed gating.
    let build = |order: &[u32]| -> Result<Archipelago> {
        let specs: Vec<IslandSpec> = order
            .iter()
            .map(|&id| IslandSpec {
                id: IslandId(id),
                label: format!("island-{id}"),
                config: base_config.clone(),
            })
            .collect();
        Archipelago::new(ArchipelagoConfig {
            islands: specs,
            topology: scriptbots_runtime::Topology::Ring,
            barrier_interval: std::num::NonZeroU64::new(barrier_interval)
                .expect("barrier interval is at least one"),
            master_seed,
            host_options: scriptbots_runtime::HostCoreOptions::default(),
            migration: Some(ArchipelagoMigration {
                interval_ticks: barrier_interval,
                emigrants_per_edge: 1,
                selection_rule: EmigrantSelectionRule::Fittest,
            }),
        })
        .context("failed to build archipelago")
    };

    // Per-island digests plus the migration record, which is the half a
    // per-island comparison cannot see: every island can be individually
    // reproducible while the barrier that moves organisms between them is not.
    let run = |order: &[u32]| -> Result<(Vec<String>, Vec<String>)> {
        let mut arch = build(order)?;
        let mut migrations = Vec::new();
        for _ in 0..barriers {
            let report = arch.step_to_barrier().context("step to barrier")?;
            if let Some(migration) = report.migration {
                for applied in &migration.moves {
                    migrations.push(format!(
                        "t{} {} -> {}",
                        migration.barrier_tick.0, applied.from, applied.to
                    ));
                }
            }
        }
        let digests = (0..ISLANDS)
            .map(|id| {
                arch.island_digest(IslandId(id))
                    .context("island digest")
                    .map(|digest| digest.overall)
            })
            .collect::<Result<Vec<String>>>()?;
        Ok((digests, migrations))
    };

    // EXPLICIT BOUNDED POOLS, NOT `RAYON_NUM_THREADS`. This matrix previously
    // varied thread counts by writing that variable between cells. Rayon reads
    // it once, when the global pool is first built, so every later write is
    // inert and all four cells ran at one width while the summary line claimed
    // it had covered [1, 4, 8, 3]. That is a diagnostic tool reporting coverage
    // it never had. `bd_0dmc_setting_rayon_num_threads_at_runtime_does_not_
    // change_the_pool` in tests/archipelago_determinism.rs pins that behaviour
    // so the approach cannot come back.
    let run_on = |order: &[u32], threads: usize| -> Result<(Vec<String>, Vec<String>)> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .context("failed to build a bounded Rayon pool for the determinism matrix")?;
        let observed = pool.install(rayon::current_num_threads);
        if observed != threads {
            bail!(
                "Rayon reported {observed} threads inside a {threads}-thread pool; this \
                 matrix would compare identical configurations and prove nothing"
            );
        }
        pool.install(|| run(order))
    };

    let ascending: Vec<u32> = (0..ISLANDS).collect();
    let (baseline_digests, baseline_migrations) = run_on(&ascending, 1)?;

    if baseline_migrations.is_empty() {
        bail!(
            "no organism migrated during the baseline run, so the migration half of this \
             check compares two empty lists; raise --det-check-archipelago ticks"
        );
    }
    if baseline_digests
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
        < 2
    {
        bail!(
            "every island produced an identical digest, so this check would pass even if \
             the digest ignored island state entirely"
        );
    }

    // 4 threads puts islands > threads and forces work stealing; 3 is a
    // non-divisor of 8 and breaks chunking assumptions. Both are cells a plain
    // 1-vs-N comparison misses.
    let mut cells: Vec<(String, Vec<u32>, usize)> = vec![
        ("threads=8".to_owned(), ascending.clone(), 8),
        ("threads=4".to_owned(), ascending.clone(), 4),
        ("threads=3".to_owned(), ascending.clone(), 3),
    ];
    // Declaration order must not reach the science: this catches order-dependent
    // seeding and UID allocation, which no thread-count variation surfaces.
    cells.push((
        "island order reversed".to_owned(),
        (0..ISLANDS).rev().collect(),
        4,
    ));

    for (label, order, threads) in cells {
        let (digests, migrations) = run_on(&order, threads)?;
        for (island, (left, right)) in baseline_digests.iter().zip(&digests).enumerate() {
            if left != right {
                println!(
                    "{} FIRST DIVERGENCE at island {island} under {label}: baseline {left}, \
                     candidate {right}",
                    "✖".red().bold()
                );
                bail!("archipelago determinism check failed");
            }
        }
        if migrations != baseline_migrations {
            println!(
                "{} migration record diverged under {label}: {} baseline moves vs {} \
                 candidate moves",
                "✖".red().bold(),
                baseline_migrations.len(),
                migrations.len()
            );
            bail!("archipelago determinism check failed");
        }
    }

    println!(
        "{} Archipelago determinism self-check passed: {} islands x {} ticks, migration on, \
         across bounded pools [1, 8, 4, 3] and a reversed island order ({} moves compared).",
        "✔".green().bold(),
        ISLANDS,
        ticks,
        baseline_migrations.len()
    );
    println!(
        "{} Scope: this compares runs of THIS binary. WorldDigestV1 is a regression oracle \
         for one pinned build lane, not a cross-platform reproducibility promise.",
        "ℹ".blue().bold()
    );
    Ok(())
}

fn ensure_recorded_archipelago_scenario(scenario: &ScenarioIdentityV0) -> Result<()> {
    if scenario.bootstrap_ticks != 0 || !scenario.interventions.is_empty() {
        bail!(
            "recorded isolated-island runs require zero bootstrap ticks and no scheduled scenario interventions"
        );
    }
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

    // Emit the initial canonical sidecar BEFORE scientific advancement.
    // If the run is interrupted, fails during bootstrap, or process is killed,
    // this initial parseable sidecar remains next to the run database.
    match manifest.canonical_json_bytes() {
        Ok(encoded) => {
            if let Err(error) = write_atomic_manifest_sidecar(&manifest_path, &encoded) {
                warn!(
                    error = %error,
                    path = %manifest_path.display(),
                    "could not write initial run-manifest sidecar; database provenance remains durable"
                );
            } else {
                info!(
                    path = %manifest_path.display(),
                    config_digest = %manifest.config_digest,
                    root_seed = manifest.root_seed,
                    reproducible = manifest.reproducible,
                    warnings = manifest.warnings.len(),
                    "wrote initial run manifest sidecar"
                );
            }
        }
        Err(error) => {
            warn!(
                error = %error,
                path = %manifest_path.display(),
                "could not serialize initial run-manifest sidecar; database provenance remains durable"
            );
        }
    }

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
            if let Err(error) = write_atomic_manifest_sidecar(&path, &encoded) {
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

struct BootstrapRequest {
    brain_preset: BrainPreset,
    storage_mode: StorageMode,
    thresholds: ThresholdsOverride,
    bootstrap_ticks: u64,
    thread_policy: ThreadPolicy,
    scenario: ScenarioIdentityV0,
    config_overrides: Vec<ConfigFieldOverride>,
}

/// Bootstrap the world and return it BY VALUE, unwrapped.
///
/// This used to hand back `Arc<Mutex<WorldState>>`, which meant the single point
/// where sole ownership is given away sat inside this function rather than at
/// the call site. bd-pcfj moves the world into a HostCore on a dedicated owner
/// thread, and `HostCore` is `!Send` - its admission state is
/// `Rc<RefCell<SharedHostState>>` - so the world has to be handed to that thread
/// as a plain value and the host constructed there. Returning the value keeps
/// that handover at one visible line instead of buried behind a wrap this
/// function performed for the caller's convenience.
///
/// Callers still wrap it today. Nothing else changes yet.
fn bootstrap_world(
    mut config: ScriptBotsConfig,
    request: BootstrapRequest,
) -> Result<(
    WorldState,
    SharedPersistenceAdmission,
    SharedAnalytics,
    StoragePipeline,
)> {
    let BootstrapRequest {
        brain_preset,
        storage_mode,
        thresholds,
        bootstrap_ticks,
        thread_policy,
        mut scenario,
        config_overrides,
    } = request;
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
        world,
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
/// Load and validate a versioned scenario document, returning it with its exact
/// source bytes (those bytes ARE the provenance — digests must cover what was read,
/// not a re-serialization).
fn load_scenario_document(path: &std::path::Path) -> Result<(ScenarioDocumentV1, Vec<u8>)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read scenario document {}", path.display()))?;
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let document = match extension.as_str() {
        "toml" => ScenarioDocumentV1::parse_toml(&bytes),
        "ron" => ScenarioDocumentV1::parse_ron(&bytes),
        other => anyhow::bail!(
            "unsupported scenario document format '.{other}' for {}; expected .toml or .ron",
            path.display()
        ),
    }
    .with_context(|| format!("invalid scenario document {}", path.display()))?;
    Ok((document, bytes))
}

/// Whether the operator explicitly named a bootstrap count on the CLI or in the
/// environment. An explicit value outranks a scenario document's bootstrap policy;
/// an unset flag lets the document (if any) speak.
fn cli_bootstrap_explicit(cli: &AppCli) -> bool {
    cli.bootstrap_ticks != DEFAULT_BOOTSTRAP_TICKS
        || std::env::var_os("SCRIPTBOTS_BOOTSTRAP_TICKS").is_some()
}

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
    let scenario_document = match &cli.scenario {
        Some(path) => Some(load_scenario_document(path)?),
        None => None,
    };
    let mut scenario = match &scenario_document {
        Some((document, _bytes)) => document.to_identity(),
        None => {
            let scenario_id = if cli.config_layers.is_empty() {
                "scriptbots-app-default-v1"
            } else {
                "scriptbots-app-layered-v1"
            };
            ScenarioIdentityV0::caller_seeded(scenario_id)
        }
    };
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
    // A scenario document speaks first among file layers: its identity is the run's,
    // its config body is the most general file statement, and `--config` files, the
    // environment, and the CLI can all still displace it later in the order.
    if let Some((document, source_bytes)) = &scenario_document {
        info!(id = %document.id, "Applying scenario document");
        scenario.record_config_layer(ConfigLayerKind::File, source_bytes);
        statements.push(ConfigLayerStatement {
            kind: ConfigLayerKind::File,
            label: format!(
                "scenario:{} ({})",
                cli.scenario
                    .as_ref()
                    .expect("scenario document implies --scenario")
                    .display(),
                document.id
            ),
            fields: document.config.clone(),
        });
    }
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

/// Insert `value` at a dotted path inside a JSON object, creating and
/// merging intermediate objects. Leaf values overwrite; object values merge
/// recursively so `render.post.bloom.enabled` and `render.tonemap_mode` can
/// arrive from different sources without clobbering each other.
fn insert_dotted_path(root: &mut serde_json::Map<String, JsonValue>, path: &str, value: JsonValue) {
    let mut segments = path.split('.');
    let Some(first) = segments.next() else {
        return;
    };
    // Our render paths always start below the `render` root handled by the caller.
    let mut current = root;
    let mut last = first;
    for segment in segments {
        let entry = current
            .entry(last.to_owned())
            .or_insert_with(|| JsonValue::Object(serde_json::Map::new()));
        if !entry.is_object() {
            *entry = JsonValue::Object(serde_json::Map::new());
        }
        let Some(next) = entry.as_object_mut() else {
            return;
        };
        current = next;
        last = segment;
    }
    current.insert(last.to_owned(), value);
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
    // Legacy `SB_WGPU_*` / `SCRIPTBOTS_TERMINAL_PALETTE` variables map onto the
    // typed RenderSettings v2 schema FIRST, so the canonical typed
    // `SCRIPTBOTS_RENDER_*` variables inserted below always outrank them.
    // Each applied mapping is logged once so operators learn the new knob names.
    for env_name in LEGACY_RENDER_ENV_NAMES {
        let Ok(value) = env::var(env_name) else {
            continue;
        };
        match map_legacy_render_env(env_name, &value) {
            Some(mapping) => {
                let paths: Vec<&str> = mapping
                    .assignments
                    .iter()
                    .map(|assignment| assignment.path)
                    .collect();
                for assignment in mapping.assignments {
                    // Mapping paths are config-rooted (`render.post...`); the
                    // env layer builds the `render` object itself, so insert
                    // below it.
                    let sub_path = assignment
                        .path
                        .strip_prefix("render.")
                        .unwrap_or(assignment.path);
                    insert_dotted_path(&mut render, sub_path, assignment.value);
                }
                info!(
                    env_name = mapping.env_name,
                    paths = ?paths,
                    note = mapping.note,
                    "Mapped legacy render environment variable onto typed render schema"
                );
            }
            None => {
                warn!(value = %value, "Invalid {env_name} value; ignoring legacy render override");
            }
        }
    }
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
    if let Some(quality) = cli.quality {
        let mut render = serde_json::Map::new();
        if let Ok(value) = serde_json::to_value(quality) {
            render.insert("quality".to_owned(), value);
        }
        root.insert("render".to_owned(), JsonValue::Object(render));
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

#[derive(Subcommand, Debug, Clone, PartialEq)]
enum AppSubcommand {
    /// Economy conservation audit (bd-9sg6 / bd-16g.11.2)
    EconomyAudit(EconomyAuditArgs),
    /// Offline archipelago reconstruction report and population conservation audit (bd-16g.5.5.5)
    ReportArchipelago(ReportArchipelagoArgs),
}

#[derive(Parser, Debug)]
#[command(
    name = "scriptbots-app",
    version,
    about = "ScriptBots simulation shell",
    subcommand_negates_reqs = true,
    args_conflicts_with_subcommands = true
)]
struct AppCli {
    #[command(subcommand)]
    subcommand: Option<AppSubcommand>,

    /// Generate an archipelago report and run population conservation audit from a DB (bd-16g.5.5.5).
    #[arg(long = "report-archipelago", value_name = "DB")]
    report_archipelago: Option<PathBuf>,

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
    /// Versioned scenario document (TOML or RON) — a stable, schema-tagged scenario
    /// identity whose config body applies as the first file layer (after defaults,
    /// before every `--config` file) and whose id/schema_version/bootstrap policy are
    /// recorded in the run manifest. `--bootstrap-ticks` overrides the document's
    /// bootstrap policy.
    #[arg(long = "scenario", value_name = "FILE")]
    scenario: Option<PathBuf>,
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
    /// Path to a run database from which to create a portable deterministic run bundle.
    #[arg(long = "create-bundle", value_name = "RUN_DB")]
    create_bundle: Option<PathBuf>,
    /// Target path/directory for bundle output when using --create-bundle.
    #[arg(
        long = "bundle-output",
        value_name = "PATH",
        requires = "create_bundle"
    )]
    bundle_output: Option<PathBuf>,
    /// Path to a run bundle directory to verify reproducibility.
    #[arg(long = "verify-bundle", value_name = "BUNDLE_PATH")]
    verify_bundle: Option<PathBuf>,
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
    /// Visual quality tier shortcut (auto|potato|low|medium|high|ultra); equivalent to
    /// `--set render.quality=TIER` but validated at parse time and recorded as its own
    /// CLI configuration layer. (`SCRIPTBOTS_RENDER_QUALITY` is the env equivalent.)
    #[arg(long = "quality", value_name = "TIER", value_parser = parse_quality_clap)]
    quality: Option<RenderQuality>,
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
    /// Run archipelago determinism self-check across thread budgets for N ticks.
    #[arg(long = "det-check-archipelago", value_name = "TICKS")]
    det_check_archipelago: Option<u64>,

    /// Record N isolated populations in one database (sequential islands, no migration).
    #[arg(long, value_name = "ISLANDS", requires = "archipelago_db",
        conflicts_with_all = ["replay_db", "det_check", "det_check_archipelago", "profile_steps", "profile_storage_steps", "profile_sweep", "create_bundle", "verify_bundle", "lab_goal", "characterize_v0"])]
    run_archipelago: Option<u32>,
    /// Exclusive new database for --run-archipelago.
    #[arg(long, value_name = "FILE", requires = "run_archipelago")]
    archipelago_db: Option<PathBuf>,
    /// Number of complete recorded island ticks (requires persistence_interval=1).
    #[arg(long, default_value_t = 100, requires = "run_archipelago")]
    archipelago_ticks: u64,
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
    /// Write a Bevy SEMANTIC PNG projection (CPU reference raster, requires bevy_render
    /// feature) and exit (no UI). This is a semantic reference only: it does NOT
    /// exercise the GPU pipeline. For real offscreen GPU captures use --dump-scene-png.
    #[cfg(feature = "bevy_render")]
    #[arg(long = "dump-semantic-png", value_name = "FILE")]
    dump_semantic_png: Option<PathBuf>,
    /// Render a scene manifest offscreen through the REAL Bevy GPU pipeline and write
    /// capture PNGs + provenance JSON + scene log, honoring the golden workflow
    /// (RUST_REGEN_GOLDEN=1 blesses; missing golden = explicit failure).
    #[cfg(feature = "bevy_render")]
    #[arg(long = "dump-scene-png", value_name = "SCENE.toml")]
    dump_scene_png: Option<PathBuf>,
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
    /// Autonomous LLM lab goal: run an automated hypothesis-driven scientific sweep (bd-16g.1).
    #[arg(long = "lab-goal", value_name = "GOAL")]
    lab_goal: Option<String>,
    /// Offline scripted fixture JSON path for autonomous lab testing.
    #[arg(long = "lab-fixture", value_name = "FILE")]
    lab_fixture: Option<PathBuf>,
    /// Maximum number of experiment runs budgeted for the lab assistant.
    #[arg(long = "lab-runs", value_name = "N", default_value_t = 10)]
    lab_runs: usize,
    /// Maximum simulation ticks per experiment run budgeted for the lab assistant.
    #[arg(long = "lab-ticks", value_name = "TICKS", default_value_t = 10_000)]
    lab_ticks: u64,
    /// Output directory for the lab notebook and reproduction scripts.
    #[arg(long = "lab-out", value_name = "DIR")]
    lab_out: Option<PathBuf>,
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
    if cli.dump_semantic_png.is_some() || cli.dump_scene_png.is_some() {
        return true;
    }
    false
}

/// `--dump-scene-png SCENE.toml`: render a scene manifest through the REAL
/// offscreen Bevy GPU pipeline, write capture PNGs + provenance JSON +
/// scene log under `captures/<scene>/`, then run the golden workflow
/// (RUST_REGEN_GOLDEN=1 blesses; missing golden = explicit failure, never
/// an auto-bless). Exit non-zero on any golden mismatch or missing golden.
#[cfg(feature = "bevy_render")]
fn run_scene_capture_cli(scene_path: &Path) -> Result<()> {
    use scriptbots_app::scene::{
        BevyOffscreenDriver, GoldenOutcome, SceneManifest, process_golden, run_scene,
        write_scene_log,
    };
    use scriptbots_bevy::capture::{CaptureProvenance, CapturedFrame, decode_png};

    let manifest = SceneManifest::load(scene_path)
        .map_err(|error| anyhow!("load scene manifest {}: {error}", scene_path.display()))?;
    let artifacts_dir = PathBuf::from("captures").join(&manifest.name);
    let goldens_dir =
        PathBuf::from("crates/scriptbots-app/tests/scenes/goldens").join(&manifest.name);
    let regen = env::var("RUST_REGEN_GOLDEN").ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    });

    let mut driver = BevyOffscreenDriver {
        artifacts_dir: Some(artifacts_dir.clone()),
        ..BevyOffscreenDriver::default()
    };
    let log = run_scene(&manifest, &mut driver).map_err(|error| anyhow!("scene run: {error}"))?;

    // Publish the structured per-scene evidence README promises alongside the PNGs
    // and provenance. Written before the golden workflow so a golden mismatch still
    // leaves the run's own log on disk for the reviewer diagnosing it
    // (bd-2z0.14.3.5.1).
    let log_path = write_scene_log(&artifacts_dir, &log)
        .map_err(|error| anyhow!("write scene log: {error}"))?;
    info!(
        scene = %log.name,
        path = %log_path.display(),
        phases = log.timings_ms.len(),
        "wrote structured scene log"
    );

    // Golden workflow per capture, re-reading the artifacts from disk so
    // the comparison path exercises exactly what a reviewer receives.
    let mut failures: Vec<String> = Vec::new();
    let mut capture_outcomes = Vec::new();
    for capture in &manifest.captures {
        let png_path = artifacts_dir.join(format!("{}.png", capture.name));
        let png_bytes =
            fs::read(&png_path).with_context(|| format!("read capture {}", png_path.display()))?;
        let (width, height, rgba8) = decode_png(&png_bytes)
            .with_context(|| format!("decode capture {}", png_path.display()))?;
        let provenance_path = artifacts_dir.join(format!("{}.provenance.json", capture.name));
        let provenance: CaptureProvenance = serde_json::from_str(
            &fs::read_to_string(&provenance_path)
                .with_context(|| format!("read {}", provenance_path.display()))?,
        )
        .with_context(|| format!("parse {}", provenance_path.display()))?;
        let frame = CapturedFrame {
            width,
            height,
            rgba8,
            provenance,
        };
        let golden_path = goldens_dir.join(format!("{}.png", capture.name));
        let outcome = process_golden(&frame, &golden_path, regen)
            .map_err(|error| anyhow!("golden workflow for {}: {error}", capture.name))?;
        match &outcome {
            GoldenOutcome::Pass {
                differing_ratio,
                mean_abs_diff,
                ..
            } => info!(
                capture = %capture.name,
                differing_ratio, mean_abs_diff, "golden comparison pass"
            ),
            GoldenOutcome::Mismatch {
                heatmap,
                differing_ratio,
                max_channel_diff,
                ..
            } => {
                failures.push(format!(
                    "capture `{}` mismatched its golden (differing ratio {differing_ratio:.6}, \
                     max channel diff {max_channel_diff}); diff heatmap: {}",
                    capture.name,
                    heatmap.display()
                ));
            }
            GoldenOutcome::MissingGolden { instructions, .. } => {
                failures.push(format!("capture `{}`: {instructions}", capture.name));
            }
            GoldenOutcome::Regenerated { golden } => info!(
                capture = %capture.name,
                golden = %golden.display(),
                "golden regenerated (review before committing)"
            ),
        }
        capture_outcomes.push(serde_json::json!({
            "name": capture.name,
            "tick": capture.tick,
            "outcome": outcome,
        }));
    }

    let summary = serde_json::json!({
        "scene": log.name,
        "frontend": log.frontend,
        "seed": log.seed,
        "ticks_executed": log.ticks_executed,
        "world_digest": log.world_digest,
        "captures": capture_outcomes,
        "expectations": log.expectations,
        "regen_mode": regen,
        "scene_log": log_path,
        "timings_ms": log.timings_ms,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).context("serialize scene summary")?
    );
    if failures.is_empty() {
        println!(
            "{} Scene {} captured and verified ({} captures)",
            "\u{2714}".green().bold(),
            manifest.name,
            manifest.captures.len()
        );
        Ok(())
    } else {
        for failure in &failures {
            eprintln!("{} {failure}", "golden failure:".red().bold());
        }
        bail!("{} golden failure(s)", failures.len())
    }
}

fn storage_owning_startup_requested(cli: &AppCli) -> bool {
    !cli.config_only
        && cli.replay_db.is_none()
        && cli.run_archipelago.is_none()
        && cli.det_check_archipelago.is_none()
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
    Server,
}

impl RendererMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gui => "gui",
            Self::Bevy => "bevy",
            Self::Terminal => "terminal",
            Self::Server => "server",
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
        RendererMode::Server => Ok(RendererMode::Server),
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
        RendererMode::Server => Box::new(ServerRenderer::default()),
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

#[derive(Default)]
struct ServerRenderer;

impl Renderer for ServerRenderer {
    fn name(&self) -> &'static str {
        "server"
    }

    fn run(&self, ctx: RendererContext<'_>) -> Result<()> {
        info!("ScriptBots server mode starting (headless background simulation with REST/MCP API)");
        let target_interval = std::time::Duration::from_millis(16);
        let reporter = ctx.control_runtime.command_reporter();
        let mut server_paused = false;
        let mut force_step = false;
        while !matches!(
            ctx.control_runtime.status(),
            scriptbots_app::servers::ControlRuntimeStatus::Failed(_)
        ) {
            let start = std::time::Instant::now();
            // Server mode APPLIES the commands it admits. This loop previously
            // stepped the world and never drained the bus, so every command a
            // REST, MCP, CLI or websocket client submitted was admitted and then
            // ignored: receipts stayed at `admitted` forever, and once the
            // bounded queue filled at capacity everything after it was rejected
            // as queue-full. The one frontend with no UI to hide it was the one
            // that never applied anything (bd-88yj).
            match ctx.world.lock() {
                Ok(mut world) => {
                    for bus in (ctx.command_drain.as_ref())() {
                        let outcome = match scriptbots_core::apply_control_command(
                            &mut world,
                            bus.command,
                        ) {
                            Ok(scriptbots_core::ControlDisposition::WorldApplied) => {
                                scriptbots_app::CommandOutcome::Applied
                            }
                            Ok(scriptbots_core::ControlDisposition::Playback(cmd)) => {
                                if let Some(paused) = cmd.paused {
                                    server_paused = paused;
                                }
                                if cmd.step_once {
                                    force_step = true;
                                    server_paused = true;
                                }
                                scriptbots_app::CommandOutcome::Applied
                            }
                            Err(error) => {
                                warn!(%error, receipt = %bus.id, "server rejected a drained control command");
                                scriptbots_app::CommandOutcome::Rejected
                            }
                        };
                        reporter(&bus.id, outcome);
                    }
                }
                Err(error) => {
                    warn!(%error, "world mutex poisoned in server mode; stopping loop");
                    break;
                }
            }
            let should_step = !server_paused || force_step;
            force_step = false;
            if should_step {
                if let Err(error) = (ctx.simulation_step)() {
                    warn!(%error, "Simulation step failed in server mode; stopping loop");
                    break;
                }
            }
            let elapsed = start.elapsed();
            if elapsed < target_interval {
                std::thread::sleep(target_interval - elapsed);
            }
        }
        info!("Server mode exiting");
        Ok(())
    }
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
        // The renderer does not depend on scriptbots-app, so the bus envelope
        // is adapted here rather than leaking the type across the boundary: the
        // id travels as a plain String and the renderer's outcome is mapped
        // back onto the ledger's (bd-tgfz).
        let drain = Arc::clone(&ctx.command_drain);
        let gui_drain: scriptbots_render::GuiCommandDrain = Arc::new(move || {
            (drain)()
                .into_iter()
                .map(|bus| (bus.id, bus.command))
                .collect()
        });
        let reporter = ctx.control_runtime.command_reporter();
        let gui_reporter: scriptbots_render::GuiCommandReporter =
            Arc::new(move |command_id, outcome| {
                let outcome = match outcome {
                    scriptbots_render::GuiCommandOutcome::Applied => {
                        scriptbots_app::CommandOutcome::Applied
                    }
                    scriptbots_render::GuiCommandOutcome::Rejected => {
                        scriptbots_app::CommandOutcome::Rejected
                    }
                };
                reporter(command_id, outcome);
            });
        run_demo(
            Arc::clone(&ctx.world),
            Arc::clone(&ctx.simulation_step),
            ctx.analytics.clone(),
            gui_drain,
            gui_reporter,
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

fn run_create_bundle_cli(run_db: &Path, output_dir: &Path) -> Result<()> {
    info!(
        run_db = %run_db.display(),
        output_dir = %output_dir.display(),
        "Creating portable deterministic run bundle"
    );
    let bundle = scriptbots_storage::create_run_bundle(run_db, output_dir)?;
    println!(
        "✓ Created run bundle V1 for run {} at {}",
        bundle.manifest.run_id,
        output_dir.display()
    );
    println!("  Artifacts: {} files", bundle.artifacts.len());
    println!("  Max Tick: {}", bundle.digests.max_tick);
    println!("  Reproducible: {}", bundle.manifest.reproducible);
    Ok(())
}

fn run_verify_bundle_cli(bundle_path: &Path) -> Result<()> {
    info!(
        bundle_path = %bundle_path.display(),
        "Verifying portable deterministic run bundle"
    );
    let result = scriptbots_storage::verify_run_bundle(bundle_path)?;
    println!(
        "✓ Verified run bundle V1 for run {} at {}",
        result.run_id,
        bundle_path.display()
    );
    println!(
        "  Verified Artifacts: {} files",
        result.total_artifacts_verified
    );
    println!("  Total Bytes Verified: {}", result.total_bytes_verified);
    println!("  Max Tick: {}", result.max_tick);
    println!("  Reproducible: {}", result.reproducible);
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
    let mut persisted_events = storage.load_replay_events()?;
    canonicalize_replay_event_order(&mut persisted_events)
        .context("recorded replay stream contains an invalid or duplicate identity")?;
    let recorded_counts = storage.replay_event_counts()?;
    let latest_checkpoint = storage.load_latest_checkpoint()?;
    if let Some(ref cp) = latest_checkpoint {
        info!(
            checkpoint_id = %cp.checkpoint_id,
            tick = cp.tick,
            format = %cp.format,
            "Discovered persisted world checkpoint in replay database"
        );
    }
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
    // The replay-digest stream is real: drivers record one `WorldDigest` event per clean
    // boundary, and both sides must carry it for verification to be non-vacuous.
    let recorded_digest_events = persisted_events
        .iter()
        .filter(|entry| matches!(entry.event.kind, ReplayEventKind::WorldDigest { .. }))
        .count();
    let simulated_digest_events = replay_run
        .events
        .iter()
        .filter(|entry| matches!(entry.event.kind, ReplayEventKind::WorldDigest { .. }))
        .count();
    require_non_vacuous_replay(
        tick_limit,
        persisted_events.len(),
        replay_run.events.len(),
        recorded_digest_events,
        simulated_digest_events,
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
        "{} Replaying {} ticks ({} recorded events) against {} (seed {})",
        "▶".bright_blue().bold(),
        tick_limit,
        persisted_events.len(),
        db_display.cyan(),
        config
            .rng_seed
            .map_or_else(|| "<unset>".to_owned(), |seed| seed.to_string())
    );
    print_event_counts("recorded", &recorded_map, None);
    print_event_counts("simulated", &simulated_counts, Some(&recorded_map));

    if let Some(divergence) = diff {
        report_divergence("recorded", "simulated", divergence)?;
    } else {
        println!(
            "{} Replay matched {} events across {} ticks (final digest {})",
            "✔".green().bold(),
            replay_run.events.len().green(),
            simulated_tick_count.green(),
            replay_run.final_digest.overall.cyan()
        );
    }

    if let Some(compare_path) = cli.compare_db.as_ref() {
        let compare_display = compare_path.display().to_string();
        let other = StorageReader::open(&compare_display)
            .with_context(|| format!("failed to open comparison database {compare_display}"))?;
        let mut other_events = other.load_replay_events()?;
        canonicalize_replay_event_order(&mut other_events)
            .context("comparison replay stream contains an invalid or duplicate identity")?;
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
    final_digest: WorldDigestV1,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
enum ReplayIdentityError {
    #[error(
        "ordinary replay event at enclosing tick {tick} uses sequence {seq}, entering the reserved narrative namespace beginning at {reserved_from}"
    )]
    OrdinarySequenceReserved {
        tick: u64,
        seq: u64,
        reserved_from: u64,
    },
    #[error("interaction replay identity overflow at tick {tick}, ordinal {ordinal}")]
    InteractionSequenceOverflow { tick: u64, ordinal: u64 },
    #[error(
        "interaction replay identity at tick {tick}, ordinal {ordinal} produces sequence {seq}, exceeding the durable SQLite identity limit {max}"
    )]
    InteractionSequenceOutOfRange {
        tick: u64,
        ordinal: u64,
        seq: u64,
        max: u64,
    },
    #[error(
        "{kind} replay source tick {source_tick} exceeds its enclosing persistence tick {enclosing_tick}"
    )]
    SourceTickAfterEnclosing {
        kind: &'static str,
        source_tick: u64,
        enclosing_tick: u64,
    },
    #[error(
        "narrative replay identity at tick {tick} is invalid (record_schema_version={record_schema_version}, input_schema_version={input_schema_version}): {reason}"
    )]
    InvalidNarrativeRecord {
        tick: u64,
        record_schema_version: u32,
        input_schema_version: u32,
        reason: String,
    },
    #[error(
        "replay row identity ({actual_tick}, {actual_seq}) aliases event identity ({expected_tick}, {expected_seq}): {event}"
    )]
    IdentityMismatch {
        actual_tick: u64,
        actual_seq: u64,
        expected_tick: u64,
        expected_seq: u64,
        event: String,
    },
    #[error(
        "duplicate replay identity at tick {tick}, sequence {seq}: first={first}; duplicate={duplicate}"
    )]
    Duplicate {
        tick: u64,
        seq: u64,
        first: String,
        duplicate: String,
    },
}

fn replay_event_identity(
    enclosing_tick: u64,
    fallback_seq: u64,
    event: &scriptbots_core::ReplayEvent,
) -> std::result::Result<(u64, u64), ReplayIdentityError> {
    match &event.kind {
        ReplayEventKind::Interaction { tick, ordinal, .. } => {
            if tick.0 > enclosing_tick {
                return Err(ReplayIdentityError::SourceTickAfterEnclosing {
                    kind: "interaction",
                    source_tick: tick.0,
                    enclosing_tick,
                });
            }
            let seq = INTERACTION_REPLAY_SEQ_BASE.checked_add(*ordinal).ok_or(
                ReplayIdentityError::InteractionSequenceOverflow {
                    tick: tick.0,
                    ordinal: *ordinal,
                },
            )?;
            let max = i64::MAX as u64;
            if seq > max {
                return Err(ReplayIdentityError::InteractionSequenceOutOfRange {
                    tick: tick.0,
                    ordinal: *ordinal,
                    seq,
                    max,
                });
            }
            Ok((tick.0, seq))
        }
        ReplayEventKind::NarrativeInputV1 { record } => {
            record
                .validate()
                .map_err(|error| ReplayIdentityError::InvalidNarrativeRecord {
                    tick: record.input.tick.0,
                    record_schema_version: record.record_schema_version,
                    input_schema_version: record.input.schema_version,
                    reason: error.to_string(),
                })?;
            if record.input.tick.0 > enclosing_tick {
                return Err(ReplayIdentityError::SourceTickAfterEnclosing {
                    kind: "narrative",
                    source_tick: record.input.tick.0,
                    enclosing_tick,
                });
            }
            Ok((record.input.tick.0, NARRATIVE_INPUT_REPLAY_SEQ))
        }
        _ if fallback_seq >= NARRATIVE_INPUT_REPLAY_SEQ => {
            Err(ReplayIdentityError::OrdinarySequenceReserved {
                tick: enclosing_tick,
                seq: fallback_seq,
                reserved_from: NARRATIVE_INPUT_REPLAY_SEQ,
            })
        }
        _ => Ok((enclosing_tick, fallback_seq)),
    }
}

fn canonicalize_replay_event_order(
    events: &mut [PersistedReplayEvent],
) -> std::result::Result<(), ReplayIdentityError> {
    for entry in events.iter() {
        let expected = replay_event_identity(entry.tick, entry.seq, &entry.event)?;
        if expected != (entry.tick, entry.seq) {
            return Err(ReplayIdentityError::IdentityMismatch {
                actual_tick: entry.tick,
                actual_seq: entry.seq,
                expected_tick: expected.0,
                expected_seq: expected.1,
                event: format_replay_event(&entry.event),
            });
        }
    }

    // Stable ordering keeps duplicate diagnostics deterministic.
    events.sort_by_key(|entry| (entry.tick, entry.seq));
    if let Some(pair) = events
        .windows(2)
        .find(|pair| (pair[0].tick, pair[0].seq) == (pair[1].tick, pair[1].seq))
    {
        return Err(ReplayIdentityError::Duplicate {
            tick: pair[0].tick,
            seq: pair[0].seq,
            first: format_replay_event(&pair[0].event),
            duplicate: format_replay_event(&pair[1].event),
        });
    }
    Ok(())
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
    let simulation_result = (|| -> Result<WorldDigestV1> {
        for index in 0..tick_limit {
            // The final tick's batch carries the canonical world digest so the simulated
            // stream stays structurally aligned with a recorded one.
            if index + 1 == tick_limit {
                world.request_replay_world_digest();
            }
            persistence.step(&mut world)?;
        }
        let final_digest = world
            .world_digest_v1()
            .context("failed to capture the final headless WorldDigestV1")?;
        persistence
            .finalize(&mut world)
            .context("failed to admit the final partial replay batch")?;
        Ok(final_digest)
    })();
    let sense_summary = SenseRunSummary::capture(&world);
    emit_sense_run_end(sense_summary, simulation_result.is_ok());
    let final_digest = simulation_result?;
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
        for (fallback_seq, event) in record.events.into_iter().enumerate() {
            let fallback_seq = u64::try_from(fallback_seq)
                .context("replay enumeration exceeds the u64 identity domain")?;
            let (tick, seq) = replay_event_identity(record.tick, fallback_seq, &event)?;
            events.push(PersistedReplayEvent { tick, seq, event });
        }
    }
    canonicalize_replay_event_order(&mut events)?;

    Ok(ReplayRun {
        events,
        summaries,
        simulated_ticks: tick_limit,
        final_digest,
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
            ReplayEventKind::WorldDigest { .. } => "world_digest",
            ReplayEventKind::Interaction {
                kind: ReplayInteractionKind::Combat,
                ..
            } => "combat",
            ReplayEventKind::Interaction {
                kind: ReplayInteractionKind::FoodShare,
                ..
            } => "food_share",
            ReplayEventKind::NarrativeInputV1 { .. } => "narrative_input_v1",
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
            "RngSample(scope={scope:?}, min={range_min:.3}, max={range_max:.3}, value={value:.3})"
        ),
        ReplayEventKind::WorldDigest { overall } => {
            format!(
                "WorldDigest(agent={:?}, overall={overall})",
                event.agent_uid
            )
        }
        ReplayEventKind::Interaction {
            tick,
            ordinal,
            kind,
            magnitude,
        } => format!(
            "Interaction(tick={}, ordinal={ordinal}, kind={kind:?}, actor={:?}, target={:?}, magnitude={magnitude:.6})",
            tick.0, event.agent_uid, event.counterpart
        ),
        ReplayEventKind::NarrativeInputV1 { record } => format!(
            "NarrativeInputV1(record_schema_version={}, input_schema_version={}, tick={}, agents={}, average_energy_bits={:08x}, spike_hits={}, config_revision={}, interval={}, history_capacity={}, event_capacity={})",
            record.record_schema_version,
            record.input.schema_version,
            record.input.tick.0,
            record.input.agent_count,
            record.input.average_energy.to_bits(),
            record.input.spike_hits,
            record.config_revision,
            record.narrative_interval,
            record.history_capacity,
            record.event_capacity
        ),
    }
}

fn print_event_counts(
    label: &str,
    counts: &HashMap<String, u64>,
    reference: Option<&HashMap<String, u64>>,
) {
    let keys = [
        "brain_outputs",
        "action",
        "rng_sample",
        "world_digest",
        "combat",
        "food_share",
        "narrative_input_v1",
    ];
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

/// Clap value parser for `--quality`: fail closed at parse time with the
/// valid tier vocabulary instead of accepting a typo into the run.
fn parse_quality_clap(raw: &str) -> Result<RenderQuality, String> {
    parse_render_quality(raw).ok_or_else(|| {
        format!("invalid quality tier `{raw}`; expected auto|potato|low|medium|high|ultra")
    })
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
    scriptbots_app::seed_founding_population(world, brain_keys)
        .map_err(|error| anyhow::anyhow!("{error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{AgentData, BirthOrigin};
    use scriptbots_storage::{StoragePipeline, StorageReader};
    use serial_test::serial;
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    /// bd-2z0.14.3.4: the semantic projection path must not touch the GPU.
    ///
    /// `--dump-semantic-png` is the CPU reference raster. It used to call
    /// `probe_gpu_capability()` to log a capability report, so a command
    /// documented as CPU-only built a wgpu instance and enumerated adapters —
    /// describing hardware it never uses, and charging that cost on headless
    /// hosts for a raster that cannot use one.
    ///
    /// This inspects the source because the alternative is asserting on a real
    /// adapter enumeration, which is exactly the side effect under test. The
    /// scan is scoped to the `dump_semantic_png` block so an unrelated probe
    /// elsewhere in `main.rs` — `--dump-scene-png` legitimately needs one —
    /// cannot trip or mask it.
    #[test]
    fn semantic_projection_path_never_probes_the_gpu() {
        let source = include_str!("main.rs");
        let after = source
            .split_once("if let Some(path) = cli.dump_semantic_png.as_ref() {")
            .expect("semantic png branch")
            .1;
        // Stop at the next top-level CLI branch so only this arm is inspected.
        let block = after
            .split_once("\n        #[cfg(feature")
            .map_or(after, |(before, _)| before);
        // Comments are stripped before scanning. The block deliberately explains
        // in prose why it does NOT probe, and naming the function there must not
        // read as calling it — a guard that cannot tell a call from a comment
        // about a call fails on the very code it is meant to bless.
        let code: String = block
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !code.contains("probe_gpu_capability"),
            "the CPU-only semantic projection path must not probe the GPU; \
             --dump-scene-png owns real GPU captures"
        );
        assert!(
            code.contains("render_bevy_png"),
            "scan anchored to the wrong block: the semantic arm must still \
             render through the CPU rasterizer"
        );
    }

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

    #[cfg(feature = "brain-ft")]
    #[test]
    fn ft_headless_run_exposes_a_complete_world_digest() {
        let run = run_headless_simulation(
            &ScriptBotsConfig {
                rng_seed: Some(424_242),
                persistence_interval: 1,
                population_minimum: 0,
                population_spawn_interval: 0,
                reproduction_attempt_chance: 0.0,
                ..ScriptBotsConfig::default()
            },
            2,
            BrainPreset::Ft,
        )
        .expect("headless Ft run");

        run.final_digest
            .validate_contract()
            .expect("complete Ft world digest contract");
        assert!(run.final_digest.evaluator_state_covered);
        assert!(run.final_digest.uncovered_families.is_empty());
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
            narrative_events: Vec::new(),
            genomes: Vec::new(),
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
                narrative_events: Vec::new(),
                genomes: Vec::new(),
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
    fn canonical_replay_order_places_narrative_before_interactions() {
        let tick = scriptbots_core::Tick(7);
        let ordinary = PersistedReplayEvent {
            tick: tick.0,
            seq: 0,
            event: scriptbots_core::ReplayEvent {
                agent_uid: None,
                position: None,
                counterpart: None,
                counterpart_position: None,
                kind: ReplayEventKind::WorldDigest {
                    overall: "1111111111111111".to_owned(),
                },
            },
        };
        let narrative_record = scriptbots_core::narrative::NarrativeInputRecordV1::from_summary(
            &TickSummary {
                tick,
                agent_count: 2,
                births: 0,
                deaths: 0,
                total_energy: 2.0,
                average_energy: 1.0,
                average_health: 1.0,
                max_age: 0,
                spike_hits: 1,
            },
            3,
            &ScriptBotsConfig::default(),
        )
        .expect("canonical narrative record");
        let narrative = PersistedReplayEvent {
            tick: tick.0,
            seq: NARRATIVE_INPUT_REPLAY_SEQ,
            event: scriptbots_core::ReplayEvent {
                agent_uid: None,
                position: None,
                counterpart: None,
                counterpart_position: None,
                kind: ReplayEventKind::NarrativeInputV1 {
                    record: narrative_record,
                },
            },
        };
        let interaction = PersistedReplayEvent {
            tick: tick.0,
            seq: INTERACTION_REPLAY_SEQ_BASE,
            event: scriptbots_core::ReplayEvent {
                agent_uid: Some(scriptbots_core::AgentUid(1)),
                position: None,
                counterpart: Some(scriptbots_core::AgentUid(2)),
                counterpart_position: None,
                kind: ReplayEventKind::Interaction {
                    tick,
                    ordinal: 0,
                    kind: ReplayInteractionKind::Combat,
                    magnitude: 0.5,
                },
            },
        };
        let mut events = vec![interaction, narrative, ordinary];

        canonicalize_replay_event_order(&mut events).expect("unique replay identities");

        assert_eq!(
            events
                .iter()
                .map(|entry| (entry.tick, entry.seq))
                .collect::<Vec<_>>(),
            [
                (tick.0, 0),
                (tick.0, NARRATIVE_INPUT_REPLAY_SEQ),
                (tick.0, INTERACTION_REPLAY_SEQ_BASE),
            ]
        );
    }

    #[test]
    fn canonical_replay_order_rejects_duplicate_narrative_identities_and_aliases() {
        let tick = scriptbots_core::Tick(7);
        let event = |config_revision| {
            let record = scriptbots_core::narrative::NarrativeInputRecordV1::from_summary(
                &TickSummary {
                    tick,
                    agent_count: 2,
                    births: 0,
                    deaths: 0,
                    total_energy: 2.0,
                    average_energy: 1.0,
                    average_health: 1.0,
                    max_age: 0,
                    spike_hits: 1,
                },
                config_revision,
                &ScriptBotsConfig::default(),
            )
            .expect("canonical narrative record");
            PersistedReplayEvent {
                tick: tick.0,
                seq: NARRATIVE_INPUT_REPLAY_SEQ,
                event: scriptbots_core::ReplayEvent {
                    agent_uid: None,
                    position: None,
                    counterpart: None,
                    counterpart_position: None,
                    kind: ReplayEventKind::NarrativeInputV1 { record },
                },
            }
        };
        let mut events = vec![event(1), event(2)];

        let error = canonicalize_replay_event_order(&mut events)
            .expect_err("duplicate replay identity must be rejected");

        assert!(matches!(
            &error,
            ReplayIdentityError::Duplicate { tick, seq, .. }
                if *tick == 7 && *seq == NARRATIVE_INPUT_REPLAY_SEQ
        ));
        let rendered = error.to_string();
        assert!(rendered.contains("config_revision=1"), "{rendered}");
        assert!(rendered.contains("config_revision=2"), "{rendered}");
        assert!(
            rendered.contains("record_schema_version=1")
                && rendered.contains("input_schema_version=1"),
            "{rendered}"
        );

        let mut aliased = vec![event(3)];
        aliased[0].tick = 8;
        let error = canonicalize_replay_event_order(&mut aliased)
            .expect_err("row tick must not alias the embedded narrative tick");
        assert!(matches!(
            error,
            ReplayIdentityError::IdentityMismatch {
                actual_tick: 8,
                actual_seq: NARRATIVE_INPUT_REPLAY_SEQ,
                expected_tick: 7,
                expected_seq: NARRATIVE_INPUT_REPLAY_SEQ,
                ..
            }
        ));
    }

    #[test]
    fn replay_identity_guards_reserved_sequences_and_reports_narrative_versions() {
        let tick = scriptbots_core::Tick(7);
        let ordinary = scriptbots_core::ReplayEvent {
            agent_uid: None,
            position: None,
            counterpart: None,
            counterpart_position: None,
            kind: ReplayEventKind::WorldDigest {
                overall: "1111111111111111".to_owned(),
            },
        };
        assert!(matches!(
            replay_event_identity(tick.0, NARRATIVE_INPUT_REPLAY_SEQ, &ordinary),
            Err(ReplayIdentityError::OrdinarySequenceReserved {
                tick: 7,
                seq: NARRATIVE_INPUT_REPLAY_SEQ,
                ..
            })
        ));

        let mut record = scriptbots_core::narrative::NarrativeInputRecordV1::from_summary(
            &TickSummary {
                tick,
                agent_count: 2,
                births: 0,
                deaths: 0,
                total_energy: 2.0,
                average_energy: 1.0,
                average_health: 1.0,
                max_age: 0,
                spike_hits: 1,
            },
            3,
            &ScriptBotsConfig::default(),
        )
        .expect("canonical narrative record");
        let valid_narrative = scriptbots_core::ReplayEvent {
            agent_uid: None,
            position: None,
            counterpart: None,
            counterpart_position: None,
            kind: ReplayEventKind::NarrativeInputV1 { record },
        };
        assert!(matches!(
            replay_event_identity(tick.0 - 1, 0, &valid_narrative),
            Err(ReplayIdentityError::SourceTickAfterEnclosing {
                kind: "narrative",
                source_tick: 7,
                enclosing_tick: 6,
            })
        ));

        record.input.schema_version = 3;
        let narrative = scriptbots_core::ReplayEvent {
            agent_uid: None,
            position: None,
            counterpart: None,
            counterpart_position: None,
            kind: ReplayEventKind::NarrativeInputV1 { record },
        };

        let error = replay_event_identity(tick.0, 0, &narrative)
            .expect_err("unknown narrative versions must be rejected");
        let rendered = error.to_string();
        assert!(rendered.contains("record_schema_version=1"), "{rendered}");
        assert!(rendered.contains("input_schema_version=3"), "{rendered}");
    }

    #[test]
    fn interaction_identity_matches_the_durable_sqlite_domain() {
        let tick = scriptbots_core::Tick(7);
        let interaction = |ordinal| scriptbots_core::ReplayEvent {
            agent_uid: Some(scriptbots_core::AgentUid(1)),
            position: None,
            counterpart: Some(scriptbots_core::AgentUid(2)),
            counterpart_position: None,
            kind: ReplayEventKind::Interaction {
                tick,
                ordinal,
                kind: ReplayInteractionKind::Combat,
                magnitude: 0.5,
            },
        };
        let max_ordinal = (i64::MAX as u64) - INTERACTION_REPLAY_SEQ_BASE;
        assert_eq!(
            replay_event_identity(tick.0, 0, &interaction(max_ordinal))
                .expect("maximum durable interaction identity"),
            (tick.0, i64::MAX as u64)
        );
        assert!(matches!(
            replay_event_identity(tick.0 - 1, 0, &interaction(0)),
            Err(ReplayIdentityError::SourceTickAfterEnclosing {
                kind: "interaction",
                source_tick: 7,
                enclosing_tick: 6,
            })
        ));
        assert!(matches!(
            replay_event_identity(tick.0, 0, &interaction(max_ordinal + 1)),
            Err(ReplayIdentityError::InteractionSequenceOutOfRange {
                tick: 7,
                ordinal,
                seq,
                max,
            }) if ordinal == max_ordinal + 1
                && seq == (i64::MAX as u64) + 1
                && max == i64::MAX as u64
        ));
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
            for index in 0..16 {
                if index + 1 == 16 {
                    world.request_replay_world_digest();
                }
                persistence
                    .step(&mut world)
                    .expect("durable replay fixture step");
            }
            let finalization = finalize_world_persistence(&mut world, &mut persistence);
            finalize_then_shutdown_storage(finalization, &mut pipeline)
                .expect("durable replay fixture finalization and shutdown");
        }

        let storage = StorageReader::open(&db_str).expect("open storage read-only");
        let mut recorded_events = storage.load_replay_events().expect("load events");
        let max_tick = storage.max_tick().expect("max tick").unwrap_or(0);
        storage.close().expect("close storage reader");
        canonicalize_replay_event_order(&mut recorded_events)
            .expect("durable replay rows must have unique canonical identities");
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
        let narrative_identities = |events: &[PersistedReplayEvent]| {
            events
                .iter()
                .filter_map(|entry| match &entry.event.kind {
                    ReplayEventKind::NarrativeInputV1 { record } => Some((
                        entry.tick,
                        entry.seq,
                        record.input.tick.0,
                        record.record_schema_version,
                        record.input.schema_version,
                    )),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let expected_identities = (1..=16)
            .map(|tick| {
                (
                    tick,
                    NARRATIVE_INPUT_REPLAY_SEQ,
                    tick,
                    scriptbots_core::narrative::NARRATIVE_INPUT_RECORD_V1_SCHEMA_VERSION,
                    scriptbots_core::narrative::NARRATIVE_INPUT_V1_SCHEMA_VERSION,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            narrative_identities(&recorded_events),
            expected_identities,
            "production persistence must retain one exact narrative identity per tick"
        );
        assert_eq!(
            narrative_identities(&replay.events),
            expected_identities,
            "production replay must reconstruct the same narrative identities"
        );
        assert!(
            replay
                .events
                .windows(2)
                .all(|pair| (pair[0].tick, pair[0].seq) <= (pair[1].tick, pair[1].seq)),
            "simulated events must use the same canonical order as durable replay rows"
        );
        let diff = diff_event_stream(&recorded_events, &replay.events);
        assert!(
            diff.is_none(),
            "non-aligned finalization must preserve the complete replay stream: {diff:#?}"
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
    const CONFIG_ENV_VARS: [&str; 25] = [
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
        "SB_WGPU_TONEMAP",
        "SB_WGPU_EXPOSURE",
        "SB_WGPU_BLOOM",
        "SB_WGPU_BLOOM_THRESH",
        "SB_WGPU_BLOOM_INTENSITY",
        "SB_WGPU_VIGNETTE",
        "SB_WGPU_FOG",
        "SB_WGPU_FOG_COLOR",
        "SB_WGPU_FXAA",
        "SCRIPTBOTS_TERMINAL_PALETTE",
        "SCRIPTBOTS_RENDER_QUALITY",
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
    fn legacy_render_env_maps_onto_typed_schema() {
        with_clean_config_env(|| {
            unsafe {
                std::env::set_var("SB_WGPU_BLOOM", "off");
                std::env::set_var("SB_WGPU_FOG", "med");
                std::env::set_var("SB_WGPU_VIGNETTE", "0.5");
                std::env::set_var("SB_WGPU_FXAA", "on");
                std::env::set_var("SCRIPTBOTS_TERMINAL_PALETTE", "deuteranopia");
            }
            let config = compose_config(&default_cli()).expect("compose with legacy render env");
            let post = config.render.post.expect("post settings materialized");
            assert_eq!(
                post.bloom.map(|bloom| bloom.enabled),
                Some(false),
                "SB_WGPU_BLOOM=off must map onto render.post.bloom.enabled=false"
            );
            assert_eq!(
                post.fog.and_then(|fog| fog.mode),
                Some(scriptbots_core::RenderFogMode::Medium),
            );
            let vignette = post.vignette.expect("vignette materialized");
            assert!(vignette.enabled);
            assert_eq!(vignette.intensity, Some(0.5));
            assert_eq!(
                post.anti_aliasing,
                Some(scriptbots_core::RenderAntiAliasing::Fxaa),
            );
            assert_eq!(
                config.render.palette,
                Some(scriptbots_core::AccessibilityPalette::Deuteranopia),
            );
        });
    }

    #[test]
    #[serial]
    fn typed_render_env_outranks_legacy() {
        with_clean_config_env(|| {
            unsafe {
                std::env::set_var("SB_WGPU_TONEMAP", "agx");
                std::env::set_var("SCRIPTBOTS_RENDER_TONEMAP", "aces");
                std::env::set_var("SB_WGPU_BLOOM_THRESH", "1.5");
            }
            let config = compose_config(&default_cli()).expect("compose with competing env");
            assert_eq!(
                config.render.tonemap_mode,
                Some(RenderTonemapMode::Aces),
                "typed SCRIPTBOTS_RENDER_TONEMAP must outrank legacy SB_WGPU_TONEMAP"
            );
            let bloom = config
                .render
                .post
                .and_then(|post| post.bloom)
                .expect("legacy bloom threshold lands");
            assert_eq!(bloom.threshold, Some(1.5));
        });
    }

    #[test]
    #[serial]
    fn invalid_legacy_render_env_warns_and_skips() {
        with_clean_config_env(|| {
            unsafe {
                std::env::set_var("SB_WGPU_FOG", "soup");
                std::env::set_var("SB_WGPU_BLOOM", "sometimes");
            }
            let config = compose_config(&default_cli()).expect("compose tolerates bad legacy env");
            assert!(
                config.render.post.is_none(),
                "invalid legacy values must be skipped, not merged: {:?}",
                config.render.post
            );
        });
    }

    #[test]
    #[serial]
    fn quality_flag_and_env_precedence() {
        with_clean_config_env(|| {
            unsafe {
                std::env::set_var("SCRIPTBOTS_RENDER_QUALITY", "low");
            }
            let env_only = compose_config(&default_cli()).expect("env quality composes");
            assert_eq!(
                env_only.render.quality,
                Some(scriptbots_core::RenderQuality::Low),
            );

            let cli = AppCli::parse_from(["scriptbots-app", "--quality", "ultra"]);
            let cli_config = compose_config(&cli).expect("cli quality composes");
            assert_eq!(
                cli_config.render.quality,
                Some(scriptbots_core::RenderQuality::Ultra),
                "--quality (CLI layer) must outrank SCRIPTBOTS_RENDER_QUALITY (env layer)"
            );
        });
    }

    #[test]
    fn quality_flag_rejects_unknown_tier() {
        let parsed = AppCli::try_parse_from(["scriptbots-app", "--quality", "ludicrous"]);
        assert!(
            parsed.is_err(),
            "--quality must fail closed on unknown tiers"
        );
    }

    #[test]
    fn insert_dotted_path_merges_nested_objects() {
        let mut map = serde_json::Map::new();
        insert_dotted_path(&mut map, "post.bloom.enabled", JsonValue::Bool(false));
        insert_dotted_path(&mut map, "post.bloom.threshold", serde_json::json!(1.5));
        insert_dotted_path(&mut map, "tonemap_mode", serde_json::json!("aces"));
        let post = map["post"].as_object().expect("post object");
        let bloom = post["bloom"].as_object().expect("bloom object");
        assert_eq!(bloom["enabled"], JsonValue::Bool(false));
        assert_eq!(bloom["threshold"], serde_json::json!(1.5));
        assert_eq!(map["tonemap_mode"], serde_json::json!("aces"));

        // A later insert into the same nested object merges instead of replacing.
        insert_dotted_path(&mut map, "post.fog.mode", serde_json::json!("high"));
        let post = map["post"].as_object().expect("post object persists");
        assert!(post.contains_key("bloom"), "sibling object must survive");
        assert_eq!(post["fog"]["mode"], serde_json::json!("high"));
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
                BootstrapRequest {
                    brain_preset: BrainPreset::Mixed,
                    storage_mode: StorageMode::File,
                    thresholds: ThresholdsOverride::default(),
                    bootstrap_ticks: DEFAULT_BOOTSTRAP_TICKS,
                    thread_policy: resolve_thread_policy(None, None, None, false),
                    scenario: ScenarioIdentityV0::caller_seeded("invalid-neuroflow-test"),
                    config_overrides: Vec::new(),
                },
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
