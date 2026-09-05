//! Bevy renderer integration for ScriptBots.

// Capture queries require deeper auto-trait evaluation through Bevy/wgpu attachment types.
#![recursion_limit = "256"]

pub mod capture;
pub mod creature_meshes;
pub mod particles;

use anyhow::{Context, Result, anyhow};
use bevy::app::AppExit;
use bevy::asset::RenderAssetUsages;
use bevy::camera::RenderTarget;
use bevy::camera::prelude::*;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::ecs::system::NonSendMut;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::light::{
    CascadeShadowConfig, CascadeShadowConfigBuilder, DirectionalLightShadowMap,
    EnvironmentMapLight, LightProbe,
};
use bevy::math::primitives::{Capsule3d, Cone, Rectangle, Sphere, Torus};
use bevy::pbr::prelude::*;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::render::view::{ColorGrading, Hdr};
use bevy::ui::{BorderColor, BorderRadius};
use bevy::window::{PresentMode, PrimaryWindow, WindowPlugin, WindowRef};
use bevy_mesh::{Indices, Mesh};
use bevy_post_process::auto_exposure::{AutoExposure, AutoExposurePlugin};
use bevy_post_process::bloom::Bloom;
use image::{ImageBuffer, Rgba as ImgRgba};
use scriptbots_core::{
    AccessibilityPalette, AgentId, BrainInspectionClientId, BrainInspectionLimits,
    BrainInspectionRequest, BrainInspectionRevision, ControlCommand, ControlDisposition, GpuClass,
    GpuInfo, IndicatorState, NUM_EYES, OutputChannel, OutputsExt, RenderGovernor, RenderQuality,
    RenderSettings, RenderTonemapMode, SelectedBrainTelemetryOutcome, SelectionMode,
    SelectionState, SelectionUpdate, SimulationCommand, TerrainKind, TickSummary, TierFeatures,
    TraitModifiers, WorldState, WorldStepDriver, apply_control_command, initial_tier_for,
    tier_features, toroidal_delta,
    visual::{
        self, AgentVisualInput, AgentVisualParams, SplatInput, TerrainSurfaceInput, VisualSelection,
    },
};
use slotmap::Key;
use std::{
    collections::{HashMap, HashSet},
    env,
    io::Cursor,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use tracing::{error, info, warn};

/// Launch context supplied by the ScriptBots application shell.
/// Submit a command and receive the identity it was admitted under.
///
/// `None` is a refusal. `Some(id)` names the submission, which is what a bare
/// `bool` could never do — and why every acknowledgement fix in this file could
/// only ever reach `admitted` (bd-k7nq).
pub type CommandSubmitFn = Arc<dyn Fn(ControlCommand) -> Option<String> + Send + Sync>;
pub type CommandDrainFn = Arc<dyn Fn() -> Vec<ControlCommand> + Send + Sync>;
pub type ControlHealthFn = Arc<dyn Fn() -> std::result::Result<(), String> + Send + Sync>;

pub struct BevyRendererContext {
    pub world: Arc<Mutex<WorldState>>,
    pub simulation_step: WorldStepDriver,
    pub command_submit: CommandSubmitFn,
    pub command_drain: CommandDrainFn,
    pub control_health: Option<ControlHealthFn>,
}

type BevyWorker = thread::JoinHandle<Result<()>>;

#[derive(Debug)]
struct BevyLifecycleFailure {
    component: &'static str,
    detail: String,
}

struct BevyLifecycleFailureInbox {
    receiver: mpsc::Receiver<BevyLifecycleFailure>,
    first_failure: Arc<Mutex<Option<String>>>,
}

#[derive(Resource)]
struct ControlHealthMonitor {
    check: Option<ControlHealthFn>,
    failures: mpsc::Sender<BevyLifecycleFailure>,
    running: Arc<AtomicBool>,
    failure_reported: bool,
}

struct BevyWorkerGroup {
    running: Arc<AtomicBool>,
    snapshot: Option<BevyWorker>,
    simulation: Option<BevyWorker>,
}

impl BevyWorkerGroup {
    fn stop_and_join(mut self) -> Result<()> {
        self.running.store(false, Ordering::Release);
        let simulation = join_bevy_worker("simulation", self.simulation.take());
        let snapshot = join_bevy_worker("snapshot", self.snapshot.take());
        combine_bevy_results(simulation, snapshot, "snapshot worker also failed")
    }
}

impl Drop for BevyWorkerGroup {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Err(error) = join_bevy_worker("simulation", self.simulation.take()) {
            warn!(%error, "Bevy simulation worker failed during emergency cleanup");
        }
        if let Err(error) = join_bevy_worker("snapshot", self.snapshot.take()) {
            warn!(%error, "Bevy snapshot worker failed during emergency cleanup");
        }
    }
}

fn join_bevy_worker(role: &str, worker: Option<BevyWorker>) -> Result<()> {
    let Some(worker) = worker else {
        return Ok(());
    };
    worker
        .join()
        .map_err(|panic| {
            anyhow!(
                "Bevy {role} worker panicked: {}",
                panic_detail(panic.as_ref())
            )
        })?
        .with_context(|| format!("Bevy {role} worker terminated with an error"))
}

fn panic_detail(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn run_reported_worker(
    component: &'static str,
    failures: &mpsc::Sender<BevyLifecycleFailure>,
    running: &AtomicBool,
    worker: impl FnOnce() -> Result<()>,
) -> Result<()> {
    // The debug/test profiles unwind and can convert a panic into a reported
    // lifecycle error. The shipped release profile uses `panic = "abort"`, so
    // it deliberately cannot promise panic recovery or destructor cleanup.
    #[cfg(panic = "unwind")]
    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(worker)) {
        Ok(result) => result,
        Err(panic) => Err(anyhow!(
            "Bevy {component} panicked: {}",
            panic_detail(panic.as_ref())
        )),
    };
    #[cfg(panic = "abort")]
    let result = worker();
    if let Err(error) = &result {
        // A renderer worker group is one structured lifetime: failure of either
        // child cancels its sibling immediately, without waiting for the next
        // Bevy update to observe the failure message.
        running.store(false, Ordering::Release);
        let _ = failures.send(BevyLifecycleFailure {
            component,
            detail: format!("{error:#}"),
        });
    }
    result
}

fn poll_control_health(mut monitor: ResMut<ControlHealthMonitor>) {
    if monitor.failure_reported {
        return;
    }
    let Some(check) = monitor.check.as_ref() else {
        return;
    };
    #[cfg(panic = "unwind")]
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check()));
    #[cfg(panic = "abort")]
    let result: std::thread::Result<std::result::Result<(), String>> = Ok(check());
    let failure = match result {
        Ok(Ok(())) => return,
        Ok(Err(detail)) => detail,
        Err(panic) => format!("health callback panicked: {}", panic_detail(panic.as_ref())),
    };
    monitor.failure_reported = true;
    monitor.running.store(false, Ordering::Release);
    let _ = monitor.failures.send(BevyLifecycleFailure {
        component: "control plane",
        detail: failure,
    });
}

fn poll_bevy_lifecycle_failures(
    inbox: NonSendMut<BevyLifecycleFailureInbox>,
    mut exit_events: MessageWriter<AppExit>,
) {
    while let Ok(failure) = inbox.receiver.try_recv() {
        let rendered = format!("Bevy {} failed: {}", failure.component, failure.detail);
        error!(component = failure.component, detail = %failure.detail, "Bevy lifecycle dependency failed; stopping renderer");
        match inbox.first_failure.lock() {
            Ok(mut first) => {
                if first.is_none() {
                    *first = Some(rendered);
                }
            }
            Err(poisoned) => {
                let mut first = poisoned.into_inner();
                if first.is_none() {
                    *first = Some(rendered);
                }
            }
        }
        exit_events.write(AppExit::error());
    }
}

fn combine_bevy_results(
    primary: Result<()>,
    cleanup: Result<()>,
    cleanup_context: &str,
) -> Result<()> {
    match (primary, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(primary), Err(cleanup)) => {
            Err(primary).context(format!("{cleanup_context}: {cleanup:#}"))
        }
    }
}

fn app_exit_result(exit: AppExit) -> Result<()> {
    match exit {
        AppExit::Success => Ok(()),
        AppExit::Error(code) => Err(anyhow!(
            "Bevy application exited with error code {}",
            code.get()
        )),
    }
}

/// Entry point for the Bevy renderer; blocks until the window closes.
/// Resolved render configuration for a Bevy launch (bd-2z0.14.3.3).
///
/// The operator's requested tier (`render.quality`, `Auto` by default) is
/// resolved against the probed adapter into one concrete tier plus the
/// canonical feature matrix row. Downstream systems read this resource
/// instead of re-deriving capability decisions.
#[derive(Resource, Debug, Clone)]
pub struct EffectiveRenderSettings {
    /// Concrete tier after Auto-resolution.
    pub tier: RenderQuality,
    /// Canonical per-tier feature matrix row.
    pub features: TierFeatures,
    /// Adapter probe (None when probing found no adapter).
    pub gpu: Option<GpuInfo>,
    /// The tier the operator actually asked for, retained after Auto-resolution.
    ///
    /// `tier` alone cannot answer "may this adapt?": once `Auto` resolves to a
    /// concrete rung it is indistinguishable from an explicitly requested one.
    /// The adaptive governor is only permitted to run when this is
    /// [`RenderQuality::Auto`], which is the bead's "explicit quality disables
    /// adaptation" clause (bd-2z0.14.3.3).
    pub requested: RenderQuality,
}

/// Probe the default high-performance GPU adapter and classify it.
///
/// Returns `None` when no adapter is available. GPU frontends must surface
/// that as an unavailable-backend error; they must not silently claim that a
/// software path rendered a frame. VRAM is not reliably exposed by the
/// bevy-pinned wgpu 26 line, so `vram_bytes` stays `None` until a
/// backend-specific lane proves otherwise (documented in bd-2z0.14.3.3).
#[must_use]
pub fn probe_gpu_capability() -> Option<GpuInfo> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let info = adapter.get_info();
    let class = gpu_class_from_device_type(info.device_type);
    let limits = adapter.limits();
    let features = adapter.features();
    // Absent rather than zero/empty (bd-2z0.14.3.3): wgpu reports 0 for an
    // unknown vendor or device id and an empty string for an unreported driver.
    // Storing those verbatim would turn "the backend told us nothing" into a
    // positive claim about vendor 0 or a driver named "", which is exactly the
    // kind of confident-looking non-evidence this bead family keeps removing.
    let vendor_id = (info.vendor != 0).then_some(info.vendor);
    let device_id = (info.device != 0).then_some(info.device);
    let driver = (!info.driver.is_empty()).then(|| info.driver.clone());
    let driver_info = (!info.driver_info.is_empty()).then(|| info.driver_info.clone());
    Some(GpuInfo {
        name: info.name,
        backend: format!("{:?}", info.backend),
        class,
        vram_bytes: None,
        max_texture_2d: Some(limits.max_texture_dimension_2d),
        timestamp_queries: features.contains(wgpu::Features::TIMESTAMP_QUERY),
        vendor_id,
        device_id,
        driver,
        driver_info,
    })
}

pub(crate) const fn gpu_class_from_device_type(device_type: wgpu::DeviceType) -> GpuClass {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => GpuClass::Discrete,
        wgpu::DeviceType::IntegratedGpu => GpuClass::Integrated,
        wgpu::DeviceType::VirtualGpu => GpuClass::Virtual,
        wgpu::DeviceType::Cpu => GpuClass::Software,
        _ => GpuClass::Unknown,
    }
}

/// Resolve the effective render settings for a launch: probe the adapter,
/// map `Auto` onto the canonical initial tier, and emit the structured
/// capability report the C3 acceptance requires (one startup INFO block,
/// never per-frame).
#[must_use]
pub fn resolve_effective_render_settings(settings: &RenderSettings) -> EffectiveRenderSettings {
    let gpu = probe_gpu_capability();
    resolve_effective_render_settings_for_gpu(settings, gpu)
}

pub(crate) fn resolve_effective_render_settings_for_gpu(
    settings: &RenderSettings,
    gpu: Option<GpuInfo>,
) -> EffectiveRenderSettings {
    let requested = settings.requested_quality();
    let tier = match requested {
        RenderQuality::Auto => gpu
            .as_ref()
            .map(|info| initial_tier_for(info.class, info.vram_bytes))
            .unwrap_or(RenderQuality::Medium),
        explicit => explicit,
    };
    // Software rasterizers run at Potato and nowhere else (bd-2z0.14.3.3).
    //
    // This OVERRIDES an explicit request, which is deliberate and is why it
    // warns rather than logging quietly: an operator who asked for Ultra on
    // llvmpipe is going to get Potato, and discovering that from a frame rate
    // instead of a log line is the dishonesty this bead is about. The clamp is
    // applied after resolution so the warning can name both what was asked for
    // and what was imposed.
    //
    // `requested` is deliberately NOT rewritten to Potato. It is what
    // distinguishes "Auto happened to resolve here" from "the operator chose
    // this", and AdaptiveQualityGovernor::for_launch keys on it. Rewriting it
    // would silently disable adaptation for Auto launches by making them look
    // explicit. On a software adapter the governor is inert anyway — ceiling
    // equals initial equals Potato, the bottom rung — but it should be inert
    // for the honest reason.
    let tier = if gpu
        .as_ref()
        .is_some_and(|info| info.class == GpuClass::Software)
    {
        if tier != RenderQuality::Potato {
            warn!(
                requested = ?requested,
                resolved = ?tier,
                imposed = ?RenderQuality::Potato,
                adapter = gpu.as_ref().map_or("<unknown>", |info| info.name.as_str()),
                "software rasterizer detected: forcing Potato. This lane is not \
                 GPU-accelerated and must not be read as representative performance \
                 or as pixel evidence"
            );
        }
        RenderQuality::Potato
    } else {
        tier
    };
    let features = tier_features(tier);
    match &gpu {
        Some(info) => {
            info!(
                adapter = %info.name,
                backend = %info.backend,
                class = ?info.class,
                vendor_id = ?info.vendor_id,
                device_id = ?info.device_id,
                driver = info.driver.as_deref().unwrap_or("<unreported>"),
                driver_info = info.driver_info.as_deref().unwrap_or("<unreported>"),
                max_texture_2d = ?info.max_texture_2d,
                timestamp_queries = info.timestamp_queries,
                "GPU capability report"
            );
        }
        None => {
            warn!("no GPU adapter detected; the Bevy renderer is unavailable");
        }
    }
    info!(
        requested = ?requested,
        effective = ?tier,
        shadows = features.shadows,
        shadow_cascades = features.shadow_cascades,
        bloom = features.bloom,
        ssao = features.ssao,
        aa = ?features.anti_aliasing,
        particles_max = features.particles_max,
        "render quality tier resolved"
    );
    EffectiveRenderSettings {
        tier,
        features,
        gpu,
        requested,
    }
}

/// Frame budget the adaptive governor holds the renderer to: 60 fps.
const ADAPTIVE_FRAME_BUDGET_MS: f32 = 16.6;

/// Live adaptive quality governor for the production Bevy renderer (bd-2z0.14.3.3).
///
/// The [`RenderGovernor`] itself is a pure, already-tested closed loop in
/// `scriptbots-core`; before this it had no production consumer at all, so the
/// tier was startup-only and no measured frame time could ever change it. This
/// resource is the frontend half: it owns the governor when — and only when —
/// adaptation is permitted.
///
/// `None` means "never adapt", and it is deliberately not recoverable. Two
/// distinct situations produce it: the operator requested an explicit tier, or
/// the operator took manual control at runtime via the quality-tier button. In
/// both cases a later automatic transition would be the renderer overriding a
/// human, so the governor is dropped rather than paused.
#[derive(Resource, Debug, Default)]
pub struct AdaptiveQualityGovernor {
    governor: Option<RenderGovernor>,
}

impl AdaptiveQualityGovernor {
    /// Build the governor for a resolved launch, or an inert one when the
    /// operator pinned a tier.
    ///
    /// The ceiling is the Auto-resolved tier: adaptation may fall back from what
    /// the adapter was judged capable of, and may climb back to it, but never
    /// above it. That keeps the capability mapping authoritative.
    #[must_use]
    pub fn for_launch(effective: &EffectiveRenderSettings) -> Self {
        if effective.requested != RenderQuality::Auto {
            info!(
                requested = ?effective.requested,
                "explicit render quality requested; adaptive governor disabled"
            );
            return Self { governor: None };
        }
        info!(
            initial = ?effective.tier,
            ceiling = ?effective.tier,
            budget_ms = ADAPTIVE_FRAME_BUDGET_MS,
            "adaptive render governor engaged"
        );
        Self {
            governor: Some(RenderGovernor::new(
                effective.tier,
                effective.tier,
                ADAPTIVE_FRAME_BUDGET_MS,
            )),
        }
    }

    /// Whether automatic tier transitions can still happen.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.governor.is_some()
    }

    /// Permanently hand the tier back to the operator.
    pub fn relinquish_to_operator(&mut self) {
        if self.governor.take().is_some() {
            info!("operator set the quality tier manually; adaptive governor disengaged");
        }
    }
}

/// Feed measured frame times to the governor and apply any tier it decides on.
///
/// This is the closed loop the bead calls for: observe, decide, and reconfigure
/// through [`EffectiveRenderSettings`] so every existing tier consumer picks the
/// change up. `EffectiveRenderSettings` is only written when the tier actually
/// moves — a blind write every frame would mark the resource changed forever and
/// make `Res::is_changed` useless to downstream systems like
/// `apply_tier_to_sun_light`.
fn drive_adaptive_quality(
    time: Res<Time>,
    mut adaptive: ResMut<AdaptiveQualityGovernor>,
    mut effective: ResMut<EffectiveRenderSettings>,
) {
    let Some(governor) = adaptive.governor.as_mut() else {
        return;
    };

    // Sampled before `observe`, deliberately. `observe` clears the window as
    // soon as it evaluates one, so reading p95 afterwards reports the empty
    // successor window (0.0) rather than the evidence that drove the decision.
    // This is the window one sample short of the one just judged, which is the
    // closest honest statistic the governor exposes.
    let window_p95_ms = governor.window_p95();

    governor.observe(time.delta_secs() * 1_000.0);

    let decided = governor.current_tier();
    if decided == effective.tier {
        return;
    }

    // Reported from the frame-time regime rather than by comparing tiers:
    // `RenderQuality` is deliberately unordered (no `PartialOrd`), and the
    // governor steps down only after windows over budget and up only after
    // windows under 60% of it, so p95 names the cause directly.
    let cause = if window_p95_ms > ADAPTIVE_FRAME_BUDGET_MS {
        "frame budget exceeded"
    } else {
        "sustained headroom"
    };

    let previous = effective.tier;
    effective.tier = decided;
    effective.features = tier_features(decided);
    info!(
        previous = ?previous,
        tier = ?decided,
        cause,
        window_p95_ms,
        budget_ms = ADAPTIVE_FRAME_BUDGET_MS,
        adapter = effective.gpu.as_ref().map_or("<none>", |info| info.name.as_str()),
        shadows = effective.features.shadows,
        ssao = effective.features.ssao,
        bloom = effective.features.bloom,
        "adaptive render governor changed the quality tier"
    );
}

pub fn run_renderer(ctx: BevyRendererContext) -> Result<()> {
    info!("Launching Bevy renderer (Phase 1: static world visuals)");

    let BevyRendererContext {
        world,
        simulation_step,
        command_submit,
        command_drain,
        control_health,
    } = ctx;

    let initial_render_settings = {
        let guard = world.lock().map_err(|error| {
            anyhow!("world mutex poisoned while reading render settings: {error}")
        })?;
        guard.config().render.clone()
    };
    let effective_render_settings = resolve_effective_render_settings(&initial_render_settings);
    if effective_render_settings.gpu.is_none() {
        return Err(anyhow!(
            "no GPU adapter is available for the Bevy renderer; choose a non-GPU frontend"
        ));
    }

    let (tx, rx) = mpsc::channel::<WorldSnapshot>();
    let (failure_tx, failure_rx) = mpsc::channel::<BevyLifecycleFailure>();
    let first_worker_failure = Arc::new(Mutex::new(None));
    let running = Arc::new(AtomicBool::new(true));
    let worker_flag = Arc::clone(&running);
    let world_for_worker = Arc::clone(&world);
    let submitter_resource = CommandSubmitter {
        submit: command_submit.clone(),
    };
    let controls_resource = SimulationControl::new();
    let controls_for_thread = controls_resource.clone();
    let drain_for_thread = Arc::clone(&command_drain);
    let world_for_sim = Arc::clone(&world);
    let running_sim = Arc::clone(&running);
    let snapshot_failures = failure_tx.clone();

    let snapshot_worker = thread::Builder::new()
        .name("scriptbots-bevy-snapshot".into())
        .spawn(move || {
            run_reported_worker("snapshot worker", &snapshot_failures, &worker_flag, || {
                let mut last_snapshot: Option<WorldSnapshot> = None;
                let mut next_revision = 1_u64;
                // Client identity and revision for brain inspection are owned
                // here, not in `from_world`, because the revision must be
                // monotonic per client across snapshots (bd-2z0.14.1.15).
                let brain_client_id = BrainInspectionClientId::new(BRAIN_OVERLAY_CLIENT_ID);
                let mut next_brain_revision = 0_u64;
                while worker_flag.load(Ordering::Acquire) {
                    let mut snapshot = {
                        let guard = world_for_worker.lock().map_err(|error| {
                            anyhow!("world mutex poisoned in Bevy snapshot worker: {error}")
                        })?;
                        let built = WorldSnapshot::from_world(&guard);
                        // Captured under the same lock, so the activations and
                        // the rest of the snapshot describe one world state.
                        built.map(|mut snap| {
                            snap.brain = BrainOverlay::capture(
                                &guard,
                                &snap.agents,
                                brain_client_id,
                                &mut next_brain_revision,
                            );
                            snap
                        })
                    }
                    .ok_or_else(|| {
                        anyhow!(
                            "Bevy snapshot worker rejected non-positive or invalid world dimensions"
                        )
                    })?;

                    if assign_presentation_revision(
                        &mut snapshot,
                        last_snapshot.as_ref(),
                        &mut next_revision,
                    )? {
                        last_snapshot = Some(snapshot.clone());
                        if tx.send(snapshot).is_err() {
                            break;
                        }
                    }

                    thread::sleep(Duration::from_millis(30));
                }
                Ok(())
            })
        })
        .context("failed to spawn Bevy snapshot worker")?;

    let simulation_worker = match spawn_simulation_driver(
        world_for_sim,
        simulation_step,
        drain_for_thread,
        controls_for_thread.clone(),
        Arc::clone(&running_sim),
        failure_tx.clone(),
    ) {
        Ok(worker) => worker,
        Err(error) => {
            running.store(false, Ordering::Release);
            let snapshot_cleanup = join_bevy_worker("snapshot", Some(snapshot_worker));
            return match snapshot_cleanup {
                Ok(()) => Err(error),
                Err(cleanup) => {
                    Err(error).context(format!("snapshot worker cleanup also failed: {cleanup:#}"))
                }
            };
        }
    };
    let workers = BevyWorkerGroup {
        running: Arc::clone(&running),
        snapshot: Some(snapshot_worker),
        simulation: Some(simulation_worker),
    };

    let mut app = App::new();
    let diagnostics_enabled = diagnostics_enabled();
    // bd-2z0.14.3.3: probe the adapter and resolve the effective quality tier
    // BEFORE the render app exists so the capability report is honest even
    // when plugin init later fails.
    app.insert_resource(AmbientLight {
        color: Color::srgb(0.45, 0.52, 0.65),
        brightness: 800.0,
        affects_lightmapped_meshes: true,
    })
    .insert_resource(submitter_resource)
    .insert_resource(controls_resource)
    // Built from the resolved settings, so it engages only for Auto launches
    // (bd-2z0.14.3.3).
    .insert_resource(AdaptiveQualityGovernor::for_launch(
        &effective_render_settings,
    ))
    .insert_non_send_resource(SnapshotInbox { receiver: rx })
    .insert_non_send_resource(BevyLifecycleFailureInbox {
        receiver: failure_rx,
        first_failure: Arc::clone(&first_worker_failure),
    })
    .insert_resource(ControlHealthMonitor {
        check: control_health,
        failures: failure_tx,
        running: Arc::clone(&running),
        failure_reported: false,
    })
    .insert_resource(SnapshotState::default())
    .insert_resource(AgentRegistry::default())
    .insert_resource(AccessibilityState::new())
    .insert_resource(TonemappingState::from_render_settings(
        &initial_render_settings,
    ))
    .insert_resource(effective_render_settings)
    .add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "ScriptBots • Bevy Renderer".to_string(),
            present_mode: PresentMode::AutoVsync,
            ..Default::default()
        }),
        ..Default::default()
    }))
    .add_plugins(AutoExposurePlugin)
    .add_systems(Startup, setup_scene)
    .add_systems(
        Update,
        (
            poll_snapshots,
            sync_world,
            handle_playback_shortcuts,
            handle_playback_buttons,
            handle_tonemap_mode_buttons,
            // Nested so the outer tuple stays within Bevy's 20-system arity
            // limit; the inner .chain() preserves strict ordering, so the
            // button still runs before the system that applies its effect.
            //
            // The governor sits between them on purpose: the button may
            // disengage it this frame, and whichever of the two sets the tier
            // must land before `apply_tier_to_sun_light` reads `is_changed`.
            (
                handle_quality_tier_button,
                drive_adaptive_quality,
                apply_tier_to_sun_light,
                apply_tier_to_bloom,
            )
                .chain(),
            handle_auto_exposure_toggle,
            handle_exposure_adjust_buttons,
            handle_palette_shortcuts,
            handle_selection_input,
            handle_follow_button_interactions,
            handle_clear_selection_button,
            update_playback_button_colors,
            update_follow_button_colors,
            update_tonemap_button_colors,
            update_auto_exposure_button_colors,
            update_exposure_button_colors,
            control_camera,
            sync_camera_tonemapping,
            update_hud,
        )
            .chain(),
    )
    .add_systems(
        Update,
        (
            poll_control_health,
            poll_bevy_lifecycle_failures,
            close_on_esc,
        )
            .chain(),
    );

    if diagnostics_enabled {
        app.insert_resource(DiagnosticsTicker::new(DIAGNOSTIC_REPORT_INTERVAL))
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Update, report_frame_metrics);
    }

    let app_exit = app.run();
    let worker_failure = match first_worker_failure.lock() {
        Ok(mut failure) => failure.take(),
        Err(poisoned) => poisoned.into_inner().take(),
    };
    let app_result = worker_failure.map_or_else(
        || app_exit_result(app_exit),
        |failure| Err(anyhow!(failure)),
    );
    let worker_result = workers.stop_and_join();
    combine_bevy_results(
        app_result,
        worker_result,
        "Bevy worker shutdown also failed",
    )
}

fn diagnostics_enabled() -> bool {
    env::var("SB_DIAGNOSTICS")
        .ok()
        .and_then(|value| parse_env_flag(&value))
        .unwrap_or(false)
}

fn parse_env_flag(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

#[derive(Component)]
struct PrimaryCamera;

struct SnapshotInbox {
    receiver: mpsc::Receiver<WorldSnapshot>,
}

#[derive(Default, Resource)]
pub(crate) struct SnapshotState {
    latest: Option<WorldSnapshot>,
    last_applied_revision: u64,
    last_applied_tick: u64,
    last_applied_palette: Option<ColorPaletteMode>,
    last_reported_tick: u64,
    focus_point: Vec2,
    world_size: Vec2,
    world_center: Vec2,
    selection_center: Option<Vec2>,
    selection_bounds: Option<(Vec2, Vec2)>,
    oldest_position: Option<Vec2>,
    first_agent_position: Option<Vec2>,
    hud_prev_tick: u64,
    hud_prev_time: f64,
    sim_rate: f32,
}

#[derive(Default, Resource)]
pub(crate) struct AgentRegistry {
    records: HashMap<AgentId, AgentRecord>,
}

#[derive(Resource, Clone, Default)]
pub(crate) struct ReflectionProbeAssets {
    diffuse: Handle<Image>,
    specular: Handle<Image>,
}

struct PartRef {
    entity: Entity,
    material: Option<Handle<StandardMaterial>>,
}

struct EyePart {
    sclera: PartRef,
    pupil: PartRef,
}

struct AgentRecord {
    root: Entity,
    body: PartRef,
    stripe: PartRef,
    wheel_left: PartRef,
    wheel_right: PartRef,
    mouth: PartRef,
    nose: PartRef,
    spike: PartRef,
    boost: PartRef,
    ear_left: PartRef,
    ear_right: PartRef,
    selection: PartRef,
    indicator: PartRef,
    sound_inner: PartRef,
    sound_outer: PartRef,
    eyes: Vec<EyePart>,
}

#[derive(Resource, Default)]
pub(crate) struct AgentMeshes {
    base_radius: f32,
    body: Handle<Mesh>,
    wheel: Handle<Mesh>,
    spike: Handle<Mesh>,
    sphere: Handle<Mesh>,
    quad: Handle<Mesh>,
    ring: Handle<Mesh>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ColorPaletteMode {
    #[default]
    Natural,
    Deuteranopia,
    Protanopia,
    Tritanopia,
    HighContrast,
}

impl ColorPaletteMode {
    fn next(self) -> Self {
        match self {
            Self::Natural => Self::Deuteranopia,
            Self::Deuteranopia => Self::Protanopia,
            Self::Protanopia => Self::Tritanopia,
            Self::Tritanopia => Self::HighContrast,
            Self::HighContrast => Self::Natural,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Natural => "Palette: Natural",
            Self::Deuteranopia => "Palette: Deuteranopia",
            Self::Protanopia => "Palette: Protanopia",
            Self::Tritanopia => "Palette: Tritanopia",
            Self::HighContrast => "Palette: High Contrast",
        }
    }

    const fn accessibility(self) -> AccessibilityPalette {
        match self {
            Self::Natural => AccessibilityPalette::Natural,
            Self::Deuteranopia => AccessibilityPalette::Deuteranopia,
            Self::Protanopia => AccessibilityPalette::Protanopia,
            Self::Tritanopia => AccessibilityPalette::Tritanopia,
            Self::HighContrast => AccessibilityPalette::HighContrast,
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct AccessibilityState {
    palette: ColorPaletteMode,
}

impl AccessibilityState {
    fn new() -> Self {
        Self {
            palette: ColorPaletteMode::Natural,
        }
    }

    fn cycle(&mut self) {
        self.palette = self.palette.next();
    }

    fn palette(&self) -> ColorPaletteMode {
        self.palette
    }
}

fn make_material(
    materials: &mut Assets<StandardMaterial>,
    base_color: Color,
    emissive: Color,
    alpha_mode: AlphaMode,
    unlit: bool,
    double_sided: bool,
) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color,
        emissive: emissive.into(),
        alpha_mode,
        unlit,
        double_sided,
        ..Default::default()
    })
}

#[allow(clippy::too_many_arguments)]
fn spawn_part(
    commands: &mut Commands,
    mesh_handle: &Handle<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    base_color: Color,
    emissive: Color,
    alpha_mode: AlphaMode,
    unlit: bool,
    double_sided: bool,
    transform: Transform,
) -> PartRef {
    let material = make_material(
        materials,
        base_color,
        emissive,
        alpha_mode,
        unlit,
        double_sided,
    );
    let entity = commands
        .spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(material.clone()),
            transform,
            GlobalTransform::default(),
            Visibility::default(),
            InheritedVisibility::default(),
        ))
        .id();
    PartRef {
        entity,
        material: Some(material),
    }
}

fn update_part_transform(commands: &mut Commands, part: &PartRef, transform: Transform) {
    commands.entity(part.entity).insert(transform);
}

fn update_part_colors(
    materials: &mut Assets<StandardMaterial>,
    part: &PartRef,
    base: Color,
    emissive: Color,
) {
    if let Some(handle) = part.material.as_ref()
        && let Some(mat) = materials.get_mut(handle)
    {
        mat.base_color = base;
        mat.emissive = emissive.into();
    }
}

fn set_part_visibility(commands: &mut Commands, part: &PartRef, visible: bool) {
    let visibility = if visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    commands.entity(part.entity).insert(visibility);
}

fn apply_palette_rgb(rgb: Vec3, palette: ColorPaletteMode) -> Vec3 {
    Vec3::from_array(visual::apply_accessibility_palette(
        rgb.to_array(),
        palette.accessibility(),
    ))
}

fn srgb_from_vec_with_palette(rgb: Vec3, alpha: f32, palette: ColorPaletteMode) -> Color {
    let mapped = apply_palette_rgb(rgb, palette);
    srgb_from_vec(mapped, alpha)
}

fn palette_emissive_from_vec(rgb: Vec3, palette: ColorPaletteMode) -> Color {
    let mapped = apply_palette_rgb(rgb, palette);
    Color::linear_rgb(mapped.x, mapped.y, mapped.z)
}

fn palette_hdr_emissive_from_srgb(rgb: [f32; 3], gain: f32, palette: ColorPaletteMode) -> Color {
    let mapped = apply_palette_rgb(Vec3::from_array(rgb), palette);
    let linear = srgb_to_linear_rgb(mapped.to_array());
    Color::linear_rgb(linear[0] * gain, linear[1] * gain, linear[2] * gain)
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn srgb_from_vec(rgb: Vec3, alpha: f32) -> Color {
    let mut color = Color::srgb(
        rgb.x.clamp(0.0, 1.0),
        rgb.y.clamp(0.0, 1.0),
        rgb.z.clamp(0.0, 1.0),
    );
    color.set_alpha(alpha.clamp(0.0, 1.0));
    color
}

fn mix_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

fn cleanup_agent_materials(materials: &mut Assets<StandardMaterial>, record: &AgentRecord) {
    fn remove(materials: &mut Assets<StandardMaterial>, part: &PartRef) {
        if let Some(handle) = part.material.as_ref() {
            materials.remove(handle);
        }
    }

    remove(materials, &record.body);
    remove(materials, &record.stripe);
    remove(materials, &record.wheel_left);
    remove(materials, &record.wheel_right);
    remove(materials, &record.mouth);
    remove(materials, &record.nose);
    remove(materials, &record.spike);
    remove(materials, &record.boost);
    remove(materials, &record.ear_left);
    remove(materials, &record.ear_right);
    remove(materials, &record.selection);
    remove(materials, &record.indicator);
    remove(materials, &record.sound_inner);
    remove(materials, &record.sound_outer);
    for eye in &record.eyes {
        remove(materials, &eye.sclera);
        remove(materials, &eye.pupil);
    }
}

fn despawn_agent_entities(record: AgentRecord, commands: &mut Commands) {
    fn despawn(commands: &mut Commands, part: PartRef) {
        commands.entity(part.entity).despawn();
    }

    let AgentRecord {
        root,
        body,
        stripe,
        wheel_left,
        wheel_right,
        mouth,
        nose,
        spike,
        boost,
        ear_left,
        ear_right,
        selection,
        indicator,
        sound_inner,
        sound_outer,
        eyes,
    } = record;

    for eye in eyes {
        despawn(commands, eye.sclera);
        despawn(commands, eye.pupil);
    }

    despawn(commands, body);
    despawn(commands, stripe);
    despawn(commands, wheel_left);
    despawn(commands, wheel_right);
    despawn(commands, mouth);
    despawn(commands, nose);
    despawn(commands, spike);
    despawn(commands, boost);
    despawn(commands, ear_left);
    despawn(commands, ear_right);
    despawn(commands, selection);
    despawn(commands, indicator);
    despawn(commands, sound_inner);
    despawn(commands, sound_outer);

    commands.entity(root).despawn();
}

#[derive(Resource, Clone)]
struct CommandSubmitter {
    submit: CommandSubmitFn,
}

const SIM_TICK_INTERVAL: f32 = 1.0 / 60.0;
const MAX_SIM_STEPS_PER_FRAME: usize = 8;
const SPEED_STEP: f32 = 0.5;
const MIN_SPEED: f32 = 0.0;
const MAX_SPEED: f32 = 8.0;

#[derive(Clone, Debug)]
struct SimControlData {
    paused: bool,
    speed_multiplier: f32,
    pending_steps: u64,
    auto_pause_reason: Option<String>,
}

impl Default for SimControlData {
    fn default() -> Self {
        Self {
            paused: false,
            speed_multiplier: 1.0,
            pending_steps: 0,
            auto_pause_reason: None,
        }
    }
}

#[derive(Clone, Debug)]
struct SimControlSnapshot {
    paused: bool,
    speed_multiplier: f32,
    auto_pause_reason: Option<String>,
}

#[derive(Resource, Clone)]
struct SimulationControl(Arc<Mutex<SimControlData>>);

impl SimulationControl {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(SimControlData::default())))
    }

    fn snapshot(&self) -> SimControlSnapshot {
        let data = match self.0.lock() {
            Ok(data) => data.clone(),
            Err(poisoned) => {
                let mut data = poisoned.into_inner();
                apply_auto_pause_to_state(
                    &mut data,
                    "Bevy simulation control mutex poisoned; science driver stopped",
                );
                data.clone()
            }
        };
        SimControlSnapshot {
            paused: data.paused,
            speed_multiplier: data.speed_multiplier,
            auto_pause_reason: data.auto_pause_reason.clone(),
        }
    }

    fn update<F>(&self, f: F) -> bool
    where
        F: FnOnce(&mut SimControlData),
    {
        match self.0.lock() {
            Ok(mut data) => {
                f(&mut data);
                true
            }
            Err(poisoned) => {
                let mut data = poisoned.into_inner();
                apply_auto_pause_to_state(
                    &mut data,
                    "Bevy simulation control mutex poisoned; science driver stopped",
                );
                false
            }
        }
    }
}

impl Default for SimulationControl {
    fn default() -> Self {
        Self::new()
    }
}

fn apply_simulation_command_to_state(state: &mut SimControlData, command: &SimulationCommand) {
    if let Some(paused) = command.paused {
        state.paused = paused;
        if paused {
            state.auto_pause_reason = None;
        }
    }
    if let Some(speed) = command.speed_multiplier {
        state.speed_multiplier = speed.clamp(0.0, MAX_SPEED);
        if state.speed_multiplier <= MIN_SPEED {
            state.paused = true;
        }
    }
    if command.step_once {
        enqueue_step_request(state);
        state.paused = true;
    }
}

fn enqueue_step_request(state: &mut SimControlData) {
    if let Some(pending_steps) = state.pending_steps.checked_add(1) {
        state.pending_steps = pending_steps;
    } else {
        state.paused = true;
        state.auto_pause_reason = Some("Bevy step queue exhausted its u64 capacity".to_owned());
    }
}

fn apply_auto_pause_to_state(state: &mut SimControlData, reason: &str) {
    state.paused = true;
    state.auto_pause_reason = Some(reason.to_owned());
    state.pending_steps = 0;
}

fn submit_simulation_command(submitter: &CommandSubmitter, command: SimulationCommand) -> bool {
    // The identity is logged rather than returned: every caller here gates on a
    // yes/no, and widening them all is the receipt-plumbing slice, not this one.
    // Naming it in the log is still strictly more than the bool carried before
    // (bd-k7nq).
    match (submitter.submit)(ControlCommand::UpdateSimulation(command)) {
        Some(command_id) => {
            debug!(%command_id, "simulation control command enqueued");
            true
        }
        None => {
            warn!("failed to enqueue simulation control command");
            false
        }
    }
}

/// Submit a playback command and reconcile local state with the answer.
///
/// The playback handlers edit local UI state first and submit second. That is
/// only sound if the submission succeeds. When it is refused, the HUD keeps
/// showing a pause state and speed the simulation never agreed to, and the two
/// stay diverged until something unrelated happens to overwrite them — the
/// operator reads "running at 2.0x" off a host that is still paused at 1.0x.
/// `submit_simulation_command` already warns, but a warning in the log does not
/// un-lie the number on screen (bd-2z0.7.14).
///
/// So a refusal rolls the optimistic edit back. Restoring from
/// [`SimControlSnapshot`] is exactly right rather than merely convenient: it
/// carries the three fields these handlers set and deliberately does NOT carry
/// `pending_steps`, which the science driver consumes concurrently and which a
/// wholesale restore would clobber.
///
/// This returns nothing on purpose. Both outcomes are fully handled here, so
/// there is no result for a caller to drop on the floor — which is the failure
/// this function exists to remove.
fn submit_playback_command(
    submitter: &CommandSubmitter,
    command: SimulationCommand,
    controls: &SimulationControl,
    previous: &SimControlSnapshot,
) {
    if submit_simulation_command(submitter, command) {
        return;
    }
    controls.update(|state| {
        state.paused = previous.paused;
        state.speed_multiplier = previous.speed_multiplier;
        state.auto_pause_reason = previous.auto_pause_reason.clone();
    });
    warn!(
        paused = previous.paused,
        speed_multiplier = previous.speed_multiplier,
        "playback command refused; local playback state rolled back so the HUD keeps \
         matching the simulation"
    );
}

const DIAGNOSTIC_REPORT_INTERVAL: u32 = 300;
const CAMERA_MIN_DISTANCE: f32 = 300.0;
const CAMERA_MAX_DISTANCE: f32 = 6000.0;
const CAMERA_SMOOTHING_LERP: f32 = 8.0;
const FIT_WORLD_FACTOR: f32 = 0.38;
const FIT_SELECTION_FACTOR: f32 = 0.55;

#[derive(Resource, Debug)]
struct DiagnosticsTicker {
    interval: u32,
    frames_since_report: u32,
}

impl DiagnosticsTicker {
    fn new(interval: u32) -> Self {
        Self {
            interval,
            frames_since_report: 0,
        }
    }

    fn tick(&mut self) -> bool {
        self.frames_since_report = self.frames_since_report.saturating_add(1);
        if self.frames_since_report >= self.interval {
            self.frames_since_report = 0;
            true
        } else {
            false
        }
    }
}

fn report_frame_metrics(mut ticker: ResMut<DiagnosticsTicker>, diagnostics: Res<DiagnosticsStore>) {
    if !ticker.tick() {
        return;
    }

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|diag| diag.smoothed())
        .unwrap_or_default();
    let frame_time_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|diag| diag.smoothed())
        .unwrap_or_default();

    info!(
        "Frame metrics: {:>6.1} fps • {:>6.3} ms per frame",
        fps, frame_time_ms
    );
}

fn follow_idle_color() -> Color {
    Color::srgba(0.16, 0.22, 0.33, 0.92)
}

fn follow_hover_color() -> Color {
    Color::srgba(0.22, 0.30, 0.46, 0.95)
}

fn follow_active_color() -> Color {
    Color::srgba(0.34, 0.26, 0.64, 0.95)
}
pub(crate) const TERRAIN_CHUNK_SIZE: u32 = 64;
pub(crate) const TERRAIN_HEIGHT_SCALE: f32 = 180.0;

fn bounds_extent(bounds: (Vec2, Vec2)) -> Vec2 {
    let size = bounds.1 - bounds.0;
    Vec2::new(size.x.abs().max(1.0), size.y.abs().max(1.0))
}

fn fit_distance_for_extent(extent: Vec2, factor: f32) -> f32 {
    let max_extent = extent.max_element().max(200.0);
    (max_extent * factor).clamp(CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE)
}

fn encode_agent_id(id: AgentId) -> u64 {
    id.data().as_ffi()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FollowMode {
    Off,
    Selected,
    Oldest,
}

impl FollowMode {
    fn label(self) -> &'static str {
        match self {
            FollowMode::Off => "Off",
            FollowMode::Selected => "Selected",
            FollowMode::Oldest => "Oldest",
        }
    }

    fn cycle(self) -> Self {
        match self {
            FollowMode::Off => FollowMode::Selected,
            FollowMode::Selected => FollowMode::Oldest,
            FollowMode::Oldest => FollowMode::Off,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FitCommand {
    World,
    Selection,
}

#[derive(Resource)]
struct CameraRig {
    yaw: f32,
    pitch: f32,
    distance: f32,
    distance_smoothed: f32,
    pan: Vec2,
    focus_smoothed: Vec2,
    follow_mode: FollowMode,
    pending_fit: Option<FitCommand>,
    recenter_now: bool,
}

impl Default for CameraRig {
    fn default() -> Self {
        Self {
            yaw: -0.6,
            pitch: -0.45,
            distance: 2200.0,
            distance_smoothed: 2200.0,
            pan: Vec2::ZERO,
            focus_smoothed: Vec2::ZERO,
            follow_mode: FollowMode::Selected,
            pending_fit: None,
            recenter_now: true,
        }
    }
}

impl CameraRig {
    fn toggle_follow_mode(&mut self, mode: FollowMode) {
        if self.follow_mode == mode {
            self.follow_mode = FollowMode::Off;
        } else {
            self.follow_mode = mode;
            self.pan = Vec2::ZERO;
            self.recenter_now = true;
        }
    }

    fn cycle_follow_mode(&mut self) {
        self.follow_mode = self.follow_mode.cycle();
        if self.follow_mode != FollowMode::Off {
            self.pan = Vec2::ZERO;
            self.recenter_now = true;
        }
    }

    fn queue_fit(&mut self, command: FitCommand) {
        self.pending_fit = Some(command);
        self.pan = Vec2::ZERO;
        self.recenter_now = true;
    }
}

#[derive(Resource)]
struct HudElements {
    tick: Entity,
    agents: Entity,
    selection: Entity,
    follow: Entity,
    camera: Entity,
    playback: Entity,
    fps: Entity,
    world: Entity,
    tonemap: Entity,
    palette: Entity,
    events: Entity,
    inspector: Entity,
    history: Entity,
    brain: Entity,
}

/// The sun light whose shadow state follows the resolved quality tier.
///
/// Marked so [`apply_tier_to_sun_light`] can re-apply on a runtime tier change
/// rather than the tier only mattering at startup (bd-2z0.14.1.17).
#[derive(Component)]
struct TierDrivenSunLight;

/// Marks the persistent SOFTWARE RENDERER banner (bd-2z0.14.3.3 item 4).
///
/// Exists so the watermark is addressable — a test can assert it is present on
/// a software adapter and absent on hardware — rather than being an anonymous
/// node that nothing can check for.
#[derive(Component)]
struct SoftwareRendererWatermark;

/// Cycles the render quality tier at runtime.
#[derive(Component)]
struct QualityTierButton;

#[derive(Component)]
struct FollowButton {
    mode: FollowMode,
}

#[derive(Component)]
struct PlaybackButton {
    action: PlaybackAction,
}

#[derive(Clone, Copy)]
enum PlaybackAction {
    Play,
    Pause,
    Step,
    SpeedDown,
    SpeedUp,
}

#[derive(Component)]
struct ClearSelectionButton;

#[derive(Clone, Copy, PartialEq, Eq)]
enum TonemappingMode {
    Aces,
    Agx,
    Tony,
}

impl TonemappingMode {
    fn label(self) -> &'static str {
        match self {
            TonemappingMode::Aces => "ACES",
            TonemappingMode::Agx => "AgX",
            TonemappingMode::Tony => "TonyMcMapface",
        }
    }

    fn to_component(self) -> Tonemapping {
        match self {
            TonemappingMode::Aces => Tonemapping::AcesFitted,
            TonemappingMode::Agx => Tonemapping::AgX,
            TonemappingMode::Tony => Tonemapping::TonyMcMapface,
        }
    }

    fn from_config(mode: RenderTonemapMode) -> Self {
        match mode {
            RenderTonemapMode::Aces => TonemappingMode::Aces,
            RenderTonemapMode::Agx => TonemappingMode::Agx,
            RenderTonemapMode::Tony => TonemappingMode::Tony,
        }
    }
}

const DEFAULT_AUTO_EXPOSURE_BRIGHTEN: f32 = 3.0;
const DEFAULT_AUTO_EXPOSURE_DARKEN: f32 = 1.0;

#[derive(Resource)]
struct TonemappingState {
    mode: TonemappingMode,
    auto_exposure_enabled: bool,
    exposure_bias: f32,
    auto_exposure_speed_brighten: f32,
    auto_exposure_speed_darken: f32,
    dirty: bool,
}

impl Default for TonemappingState {
    fn default() -> Self {
        Self {
            mode: TonemappingMode::Aces,
            auto_exposure_enabled: false,
            exposure_bias: 0.0,
            auto_exposure_speed_brighten: DEFAULT_AUTO_EXPOSURE_BRIGHTEN,
            auto_exposure_speed_darken: DEFAULT_AUTO_EXPOSURE_DARKEN,
            dirty: true,
        }
    }
}

impl TonemappingState {
    fn from_render_settings(settings: &RenderSettings) -> Self {
        let mut state = Self::default();

        if let Some(mode) = settings.tonemap_mode {
            state.mode = TonemappingMode::from_config(mode);
        }
        if let Some(bias) = settings.tonemap_exposure_bias {
            state.exposure_bias = bias;
        }
        if let Some(auto) = &settings.auto_exposure {
            state.auto_exposure_enabled = auto.enabled;
            if let Some(speed) = auto.speed_brighten
                && speed.is_finite()
                && speed >= 0.0
            {
                state.auto_exposure_speed_brighten = speed;
            }
            if let Some(speed) = auto.speed_darken
                && speed.is_finite()
                && speed >= 0.0
            {
                state.auto_exposure_speed_darken = speed;
            }
        }

        state.dirty = true;
        state
    }
}

#[derive(Component)]
struct TonemapButton {
    mode: TonemappingMode,
}

#[derive(Component)]
struct AutoExposureToggleButton;

#[derive(Component)]
struct ExposureAdjustButton {
    delta: f32,
}

type ChangedButtonFilter = (Changed<Interaction>, With<Button>);

#[derive(Clone, PartialEq)]
struct TerrainColorMap {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

#[derive(Clone, PartialEq)]
struct TerrainHeightSnapshot {
    dims: UVec2,
    cell_size: u32,
    elevation: Vec<f32>,
    moisture: Vec<f32>,
    accent: Vec<f32>,
    water_depth: Vec<f32>,
    fertility: Vec<f32>,
    temperature: Vec<f32>,
    kinds: Vec<TerrainKind>,
    daylight: f32,
}

impl TerrainHeightSnapshot {
    fn new(
        layer: &scriptbots_core::TerrainLayer,
        water_depth: Option<&[f32]>,
        daylight: f32,
    ) -> Option<Self> {
        let dims = UVec2::new(layer.width(), layer.height());
        let total = (dims.x as usize) * (dims.y as usize);
        if layer.tiles().len() != total
            || water_depth.is_some_and(|depth| depth.len() != total)
            || !daylight.is_finite()
        {
            return None;
        }
        let mut elevation = Vec::with_capacity(total);
        let mut moisture = Vec::with_capacity(total);
        let mut accent = Vec::with_capacity(total);
        let mut fertility = Vec::with_capacity(total);
        let mut temperature = Vec::with_capacity(total);
        let mut kinds = Vec::with_capacity(total);
        for tile in layer.tiles() {
            elevation.push(tile.elevation);
            moisture.push(tile.moisture);
            accent.push(tile.accent);
            fertility.push(tile.fertility_bias);
            temperature.push(tile.temperature_bias);
            kinds.push(tile.kind);
        }
        Some(Self {
            dims,
            cell_size: layer.cell_size(),
            elevation,
            moisture,
            accent,
            water_depth: water_depth.map_or_else(|| vec![0.0; total], |depth| depth.to_vec()),
            fertility,
            temperature,
            kinds,
            daylight,
        })
    }

    fn index(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.dims.x as usize) + (x as usize)
    }

    fn sample_tile(&self, x: u32, y: u32) -> TerrainTileSample {
        let clamped_x = x.min(self.dims.x.saturating_sub(1));
        let clamped_y = y.min(self.dims.y.saturating_sub(1));
        let idx = self.index(clamped_x, clamped_y);
        TerrainTileSample {
            kind: self.kinds[idx],
            elevation: self.elevation[idx],
            moisture: self.moisture[idx],
            accent: self.accent[idx],
            water_depth: self.water_depth[idx],
            _fertility: self.fertility[idx],
            _temperature: self.temperature[idx],
        }
    }
}

#[derive(Clone, Copy)]
struct TerrainTileSample {
    kind: TerrainKind,
    elevation: f32,
    moisture: f32,
    accent: f32,
    water_depth: f32,
    _fertility: f32,
    _temperature: f32,
}

#[derive(Default, Resource)]
struct TerrainChunkRegistry {
    chunks: HashMap<TerrainChunkKey, TerrainChunkRecord>,
    chunk_size: u32,
    height_scale: f32,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct TerrainChunkKey {
    x: u32,
    y: u32,
}

struct TerrainChunkRecord {
    entity: Entity,
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    bounds: TerrainChunkBounds,
    signature: TerrainChunkSignature,
    last_tick: u64,
    probe: Option<Entity>,
    stats: TerrainChunkStats,
    palette: ColorPaletteMode,
}

#[derive(Clone, Copy)]
struct TerrainChunkBounds {
    origin: UVec2,
    size: UVec2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TerrainChunkSignature([u8; 32]);

impl TerrainChunkSignature {
    fn from_render_inputs(
        positions: &[[f32; 3]],
        colors: &[[f32; 4]],
        material_inputs: &[f32],
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"scriptbots.bevy-terrain-chunk.v1");
        hasher.update(&(positions.len() as u64).to_le_bytes());
        for position in positions {
            for channel in position {
                hasher.update(&channel.to_bits().to_le_bytes());
            }
        }
        hasher.update(&(colors.len() as u64).to_le_bytes());
        for color in colors {
            for channel in color {
                hasher.update(&channel.to_bits().to_le_bytes());
            }
        }
        hasher.update(&(material_inputs.len() as u64).to_le_bytes());
        for input in material_inputs {
            hasher.update(&input.to_bits().to_le_bytes());
        }
        Self(*hasher.finalize().as_bytes())
    }
}

#[derive(Clone, PartialEq)]
pub(crate) struct WorldSnapshot {
    revision: u64,
    tick: u64,
    world_size: Vec2,
    agent_radius: f32,
    terrain_color: TerrainColorMap,
    terrain_height: TerrainHeightSnapshot,
    agents: Vec<AgentVisual>,
    /// Decimated population/birth/death series for the HUD sparklines
    /// (bd-2z0.14.1.16). Bounded by [`HUD_SPARKLINE_SAMPLES`] at construction.
    history: HudHistory,
    /// Bounded brain activations for the selected agent (bd-2z0.14.1.15).
    ///
    /// Populated by the snapshot worker, which owns the request revision;
    /// `from_world` deliberately does not issue inspection requests.
    brain: BrainOverlay,
    /// Newest-first recent world events for the HUD feed (bd-2z0.14.1.13).
    ///
    /// Bounded by [`HUD_EVENT_FEED_CAPACITY`] at construction, so a long run
    /// or a births/deaths storm cannot grow the snapshot without limit.
    events: Vec<HudEvent>,
}

/// How many recent events the HUD feed retains and renders.
///
/// The bound is applied while deriving from world history, not after, so the
/// vector is never transiently large.
const HUD_EVENT_FEED_CAPACITY: usize = 6;

/// Layers the brain overlay will render, and values shown per layer
/// (bd-2z0.14.1.15). Core already bounds the payload it hands back; these are
/// the presentation budgets on top of that.
const BRAIN_OVERLAY_MAX_LAYERS: usize = 4;
const BRAIN_OVERLAY_MAX_VALUES: usize = 12;

/// Stable client identity for this frontend's inspection requests, so core can
/// distinguish Bevy's requests from GPUI's and the TUI's.
const BRAIN_OVERLAY_CLIENT_ID: u64 = 0x6265_7679; // "bevy"

/// Why the brain overlay is showing what it is showing.
///
/// `NotRequested` is the important one: with nothing selected the overlay
/// issues no inspection request at all, so no brain is inspected. Core's
/// contract guarantees digest-neutrality only for the no-request case, and an
/// overlay that polled every frame regardless of selection would quietly turn
/// a read-only projection into a per-tick side effect.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum BrainOverlayStatus {
    /// No selection, therefore no request was issued.
    #[default]
    NotRequested,
    /// Core returned a payload.
    Ready,
    /// Core typed-refused this target.
    Unavailable,
    /// The selected agent has no stable UID (already despawned).
    NoStableIdentity,
}

/// One rendered activation layer, already clipped to the presentation budget.
#[derive(Debug, Clone, PartialEq)]
struct BrainOverlayLayer {
    name: String,
    shown: Vec<f32>,
    total: usize,
}

/// Bounded projection of the selected agent's brain activations.
#[derive(Debug, Clone, Default, PartialEq)]
struct BrainOverlay {
    status: BrainOverlayStatus,
    source_tick: u64,
    layers: Vec<BrainOverlayLayer>,
    /// Set when core clipped the payload, or when this projection clipped it
    /// further. An inspector showing a truncated view must say so.
    truncated: bool,
}

/// Sample budget for each HUD sparkline (bd-2z0.14.1.16).
///
/// GPUI's `HistoryChartData::from_entries` uses the same stride-decimation
/// policy with a budget of 120, sized for a 220px polyline. A text sparkline
/// is one glyph per sample, so the budget differs while the policy does not.
const HUD_SPARKLINE_SAMPLES: usize = 32;

/// Ramp used to draw a text sparkline, lowest to highest.
const SPARK_GLYPHS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

/// Decimated population/birth/death series behind the HUD sparklines.
///
/// Mirrors GPUI's chart rather than deriving a second history projection:
/// same three series, same `< 2` guard, same stride decimation, and the same
/// per-series scaling by that series' own maximum (bd-2z0.14.1.16).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct HudHistory {
    agents: Vec<u32>,
    births: Vec<u32>,
    deaths: Vec<u32>,
}

impl HudHistory {
    /// Decimate retained tick summaries down to the sparkline budget.
    ///
    /// Fewer than two samples yields an empty history, matching GPUI's
    /// `from_entries` returning `None`: a single point is not a trend and
    /// drawing it would imply one.
    fn from_history<'a>(history: impl DoubleEndedIterator<Item = &'a TickSummary>) -> Self {
        let entries: Vec<&TickSummary> = history.collect();
        if entries.len() < 2 {
            return Self::default();
        }
        let stride = entries.len().div_ceil(HUD_SPARKLINE_SAMPLES).max(1);
        let mut out = Self::default();
        for summary in entries.iter().step_by(stride) {
            out.agents
                .push(u32::try_from(summary.agent_count).unwrap_or(u32::MAX));
            out.births
                .push(u32::try_from(summary.births).unwrap_or(u32::MAX));
            out.deaths
                .push(u32::try_from(summary.deaths).unwrap_or(u32::MAX));
        }
        out
    }

    fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
}

impl BrainOverlay {
    /// Capture the primary selection's activations, or issue no request at all.
    ///
    /// Returns early with `NotRequested` when nothing is selected, BEFORE
    /// touching `inspect_brains`. That ordering is the contract: no request
    /// means no brain is inspected, which is what keeps the projection
    /// digest-neutral.
    fn capture(
        world: &WorldState,
        agents: &[AgentVisual],
        client_id: BrainInspectionClientId,
        next_revision: &mut u64,
    ) -> Self {
        let Some(selected) = agents
            .iter()
            .find(|a| matches!(a.selection, SelectionState::Selected))
        else {
            return Self::default();
        };
        let Some(uid) = world.agent_uid(selected.id) else {
            return Self {
                status: BrainOverlayStatus::NoStableIdentity,
                ..Self::default()
            };
        };

        let revision = next_revision.saturating_add(1);
        *next_revision = revision;
        let request = BrainInspectionRequest {
            client_id,
            revision: BrainInspectionRevision::new(revision),
            targets: vec![uid],
            limits: BrainInspectionLimits::default(),
        };
        let Ok(response) = world.inspect_brains(&request) else {
            return Self {
                status: BrainOverlayStatus::Unavailable,
                ..Self::default()
            };
        };

        let source_tick = response.source_tick.0;
        match response.telemetry.into_iter().next() {
            Some(SelectedBrainTelemetryOutcome::Ready { telemetry }) => {
                let activations = &telemetry.inspection.activations;
                let mut truncated = activations.truncated;
                let mut layers = Vec::with_capacity(BRAIN_OVERLAY_MAX_LAYERS);
                for layer in activations.layers.iter().take(BRAIN_OVERLAY_MAX_LAYERS) {
                    let total = layer.values.len();
                    if total > BRAIN_OVERLAY_MAX_VALUES {
                        truncated = true;
                    }
                    layers.push(BrainOverlayLayer {
                        name: layer.name.clone(),
                        shown: layer
                            .values
                            .iter()
                            .take(BRAIN_OVERLAY_MAX_VALUES)
                            .copied()
                            .collect(),
                        total,
                    });
                }
                if activations.layers.len() > BRAIN_OVERLAY_MAX_LAYERS {
                    truncated = true;
                }
                Self {
                    status: BrainOverlayStatus::Ready,
                    source_tick,
                    layers,
                    truncated,
                }
            }
            _ => Self {
                status: BrainOverlayStatus::Unavailable,
                source_tick,
                ..Self::default()
            },
        }
    }
}

/// Render the brain overlay as a multi-line HUD block (bd-2z0.14.1.15).
fn format_brain_overlay(overlay: &BrainOverlay) -> String {
    match overlay.status {
        BrainOverlayStatus::NotRequested => "Brain: select an agent".to_string(),
        BrainOverlayStatus::NoStableIdentity => {
            "Brain: selection has no stable identity".to_string()
        }
        BrainOverlayStatus::Unavailable => {
            format!("Brain: unavailable (tick {})", overlay.source_tick)
        }
        BrainOverlayStatus::Ready => {
            let mut out = format!("Brain @ tick {}", overlay.source_tick);
            if overlay.truncated {
                out.push_str(" (clipped)");
            }
            for layer in &overlay.layers {
                out.push_str(&format!(
                    "\n  {} [{}/{}] {}",
                    layer.name,
                    layer.shown.len(),
                    layer.total,
                    format_sparkline_signed(&layer.shown)
                ));
            }
            out
        }
    }
}

/// Draw activations, which are signed, on the same block ramp.
///
/// Scaled by peak absolute magnitude so a mostly-negative layer still shows
/// shape rather than collapsing onto the floor.
fn format_sparkline_signed(values: &[f32]) -> String {
    let peak = values.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
    if peak <= f32::EPSILON {
        return SPARK_GLYPHS[0].repeat(values.len());
    }
    values
        .iter()
        .map(|v| {
            let level = ((v.abs() / peak) * 7.0).round().clamp(0.0, 7.0) as usize;
            SPARK_GLYPHS[level.min(SPARK_GLYPHS.len() - 1)]
        })
        .collect()
}

/// Draw one series as a text sparkline, scaled by its own maximum.
///
/// Per-series scaling matches GPUI: births and deaths are small numbers beside
/// a population in the hundreds, so a shared scale would flatten them into a
/// dead line and hide exactly the events worth seeing.
fn format_sparkline(series: &[u32]) -> String {
    let Some(&max) = series.iter().max() else {
        return String::new();
    };
    if max == 0 {
        return SPARK_GLYPHS[0].repeat(series.len());
    }
    series
        .iter()
        .map(|&v| {
            let level = (u64::from(v) * 7 / u64::from(max)) as usize;
            SPARK_GLYPHS[level.min(SPARK_GLYPHS.len() - 1)]
        })
        .collect()
}

/// Render the three history sparklines as a multi-line HUD block.
fn format_history_panel(history: &HudHistory) -> String {
    if history.is_empty() {
        return "History: collecting…".to_string();
    }
    let peak = |s: &[u32]| s.iter().max().copied().unwrap_or(0);
    format!(
        "History ({} samples)\n  Agents {} peak {}\n  Births {} peak {}\n  Deaths {} peak {}",
        history.agents.len(),
        format_sparkline(&history.agents),
        peak(&history.agents),
        format_sparkline(&history.births),
        peak(&history.births),
        format_sparkline(&history.deaths),
        peak(&history.deaths),
    )
}

/// One entry in the HUD event feed.
///
/// Counts are per completed tick, taken straight from [`TickSummary`], rather
/// than one entry per individual birth or death: a busy tick would otherwise
/// flood the feed and push everything else out within a single frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HudEvent {
    tick: u64,
    kind: HudEventKind,
    count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HudEventKind {
    Birth,
    Death,
    SpikeHit,
}

impl HudEventKind {
    /// Glyph shown beside the entry. Kept to widely available symbols so a
    /// missing font renders a box rather than nothing at all.
    const fn glyph(self) -> &'static str {
        match self {
            Self::Birth => "✚",
            Self::Death => "✖",
            Self::SpikeHit => "⚔",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Birth => "born",
            Self::Death => "died",
            Self::SpikeHit => "spiked",
        }
    }
}

impl HudEvent {
    /// Derive the newest-first feed from retained tick summaries.
    ///
    /// `history` is oldest-first, so it is walked in reverse. Within one tick
    /// the order is births, deaths, then spike hits, so a tick that produced
    /// several kinds always reads the same way.
    fn recent_from_history<'a>(
        history: impl DoubleEndedIterator<Item = &'a TickSummary>,
    ) -> Vec<Self> {
        let mut events = Vec::with_capacity(HUD_EVENT_FEED_CAPACITY);
        for summary in history.rev() {
            let tick = summary.tick.0;
            let candidates = [
                (
                    HudEventKind::Birth,
                    u32::try_from(summary.births).unwrap_or(u32::MAX),
                ),
                (
                    HudEventKind::Death,
                    u32::try_from(summary.deaths).unwrap_or(u32::MAX),
                ),
                (HudEventKind::SpikeHit, summary.spike_hits),
            ];
            for (kind, count) in candidates {
                if count == 0 {
                    continue;
                }
                events.push(Self { tick, kind, count });
                if events.len() == HUD_EVENT_FEED_CAPACITY {
                    return events;
                }
            }
        }
        events
    }
}

#[derive(Clone, PartialEq)]
struct AgentVisual {
    id: AgentId,
    position: Vec2,
    heading: f32,
    color: [f32; 3],
    selection: SelectionState,
    health: f32,
    /// Current energy reserve, for the inspector overlay (bd-2z0.14.1.14).
    ///
    /// Sourced from `AgentRuntime::energy`, the same value GPUI's inspector
    /// reports as `detail.energy`, so the two frontends cannot disagree.
    energy: f32,
    age: u32,
    /// Lineage depth, matching GPUI's `detail.generation` (bd-2z0.14.1.14).
    generation: u32,
    reference_age_ticks: u64,
    spike_length: f32,
    boost: f32,
    wheel_left: f32,
    wheel_right: f32,
    herbivore_tendency: f32,
    temperature_preference: f32,
    food_delta: f32,
    sound_level: f32,
    sound_output: f32,
    sound_multiplier: f32,
    trait_modifiers: TraitModifiers,
    eye_dirs: [f32; NUM_EYES],
    eye_fov: [f32; NUM_EYES],
    indicator: IndicatorState,
    reproduction_intent: f32,
    spiked: bool,
}

impl WorldSnapshot {
    fn same_render_content(&self, other: &Self) -> bool {
        self.tick == other.tick
            && self.world_size == other.world_size
            && self.agent_radius == other.agent_radius
            && self.terrain_color == other.terrain_color
            && self.terrain_height == other.terrain_height
            && self.agents == other.agents
            && self.events == other.events
            && self.history == other.history
            && self.brain == other.brain
    }

    fn from_world(world: &WorldState) -> Option<Self> {
        let config = world.config();
        let width = config.world_width as f32;
        let height = config.world_height as f32;
        if width <= 0.0 || height <= 0.0 {
            return None;
        }

        let terrain_layer = world.terrain();
        let terrain_w = terrain_layer.width();
        let terrain_h = terrain_layer.height();
        let (cycle_ticks, start_phase) = config.render.resolved_day_night();
        let daylight = visual::daylight_factor(world.tick().0, cycle_ticks, start_phase);
        let terrain_height = TerrainHeightSnapshot::new(
            terrain_layer,
            world
                .hydrology()
                .map(scriptbots_core::HydrologyState::water_depth),
            daylight,
        )?;

        let arena = world.agents();
        let columns = arena.columns();
        let positions = columns.positions();
        let colors = columns.colors();
        let healths = columns.health();
        let ages = columns.ages();
        let headings = columns.headings();
        let spikes = columns.spike_lengths();
        let boosts = columns.boosts();
        let generations = columns.generations();
        let runtime = world.runtime();

        let mut agents = Vec::with_capacity(arena.len());
        for (idx, agent_id) in arena.iter_handles().enumerate() {
            let runtime_entry = runtime.get(agent_id);
            // Read separately rather than growing the already 15-wide tuple
            // below; `Option<&AgentRuntime>` is `Copy`, so this does not
            // disturb that destructuring.
            let energy = runtime_entry.map_or(0.0, |rt| rt.energy);
            let (
                selection,
                wheel_left,
                wheel_right,
                herbivore_tendency,
                temperature_preference,
                food_delta,
                sound_level,
                sound_output,
                sound_multiplier,
                trait_modifiers,
                eye_dirs,
                eye_fov,
                indicator,
                reproduction_intent,
                spiked,
            ) = runtime_entry
                .map(|rt| {
                    let mut eye_dirs = [0.0_f32; NUM_EYES];
                    let mut eye_fov = [0.0_f32; NUM_EYES];
                    eye_dirs.copy_from_slice(&rt.eye_direction);
                    eye_fov.copy_from_slice(&rt.eye_fov);
                    (
                        rt.selection,
                        rt.outputs.channel(OutputChannel::WheelLeft),
                        rt.outputs.channel(OutputChannel::WheelRight),
                        rt.herbivore_tendency,
                        rt.temperature_preference,
                        rt.food_delta,
                        rt.outputs.channel(OutputChannel::SoundLevel),
                        rt.sound_output,
                        rt.sound_multiplier,
                        TraitModifiers {
                            smell: rt.trait_modifiers.smell,
                            sound: rt.trait_modifiers.sound,
                            hearing: rt.trait_modifiers.hearing,
                            eye: rt.trait_modifiers.eye,
                            blood: rt.trait_modifiers.blood,
                        },
                        eye_dirs,
                        eye_fov,
                        rt.indicator,
                        rt.give_intent,
                        rt.spiked,
                    )
                })
                .unwrap_or_else(|| {
                    (
                        SelectionState::None,
                        0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        TraitModifiers::default(),
                        [0.0; NUM_EYES],
                        [1.0; NUM_EYES],
                        IndicatorState::default(),
                        0.0,
                        false,
                    )
                });
            agents.push(AgentVisual {
                id: agent_id,
                position: Vec2::new(positions[idx].x, positions[idx].y),
                heading: headings[idx],
                color: colors[idx],
                spike_length: spikes[idx],
                boost: if boosts[idx] { 1.0 } else { 0.0 },
                wheel_left,
                wheel_right,
                herbivore_tendency,
                temperature_preference,
                food_delta,
                sound_level,
                sound_output,
                sound_multiplier,
                trait_modifiers,
                eye_dirs,
                eye_fov,
                selection,
                health: healths[idx],
                energy,
                age: ages[idx],
                generation: generations[idx].0,
                reference_age_ticks: u64::from(config.aging_health_decay_start.max(1)),
                indicator,
                reproduction_intent,
                spiked,
            });
        }

        let mut terrain_pixels = Vec::with_capacity((terrain_w * terrain_h * 4) as usize);
        for tile in terrain_layer.tiles() {
            let base = terrain_kind_color(tile.kind);
            terrain_pixels.push((base[0] * 255.0).round().clamp(0.0, 255.0) as u8);
            terrain_pixels.push((base[1] * 255.0).round().clamp(0.0, 255.0) as u8);
            terrain_pixels.push((base[2] * 255.0).round().clamp(0.0, 255.0) as u8);
            terrain_pixels.push(255);
        }

        Some(Self {
            revision: 1,
            tick: world.tick().0,
            world_size: Vec2::new(width, height),
            agent_radius: config.bot_radius.max(1.0),
            terrain_color: TerrainColorMap {
                width: terrain_w,
                height: terrain_h,
                pixels: terrain_pixels,
            },
            terrain_height,
            agents,
            events: HudEvent::recent_from_history(world.history()),
            history: HudHistory::from_history(world.history()),
            // Left empty here on purpose: issuing an inspection request needs a
            // client-owned monotonic revision, which the snapshot worker holds.
            brain: BrainOverlay::default(),
        })
    }
}

fn assign_presentation_revision(
    snapshot: &mut WorldSnapshot,
    previous: Option<&WorldSnapshot>,
    next_revision: &mut u64,
) -> Result<bool> {
    if previous.is_some_and(|last| snapshot.same_render_content(last)) {
        return Ok(false);
    }
    snapshot.revision = *next_revision;
    *next_revision = next_revision
        .checked_add(1)
        .ok_or_else(|| anyhow!("Bevy presentation revision overflow"))?;
    Ok(true)
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    effective: Res<EffectiveRenderSettings>,
) {
    let camera_transform = Transform::from_xyz(0.0, 1800.0, 1400.0).looking_at(Vec3::ZERO, Vec3::Y);
    let camera = commands
        .spawn((
            Camera3d::default(),
            Camera {
                clear_color: ClearColorConfig::Custom(Color::srgb(0.03, 0.05, 0.09)),
                ..default()
            },
            camera_transform,
            GlobalTransform::default(),
            Visibility::default(),
            InheritedVisibility::default(),
            Tonemapping::AcesFitted,
            ColorGrading::default(),
            Hdr,
            PrimaryCamera,
        ))
        .id();
    // Inserted separately rather than in the tuple because it is CONDITIONAL:
    // the component's presence is the on/off switch, so an unconditional insert
    // would make every tier bloom (bd-2z0.14.3.3).
    if effective.features.bloom {
        commands.entity(camera).insert(Bloom::NATURAL);
    }

    let light_transform =
        Transform::from_xyz(-1200.0, 1800.0, 900.0).looking_at(Vec3::ZERO, Vec3::Y);
    // bd-2z0.14.1.17: the FIRST live consumer of the resolved quality tier.
    // Before this the tier was probed against a real GPU, logged, inserted as a
    // resource, and read by nobody — Potato and Ultra rendered identically.
    commands.spawn((
        DirectionalLight {
            illuminance: 9000.0,
            shadows_enabled: effective.features.shadows,
            ..default()
        },
        light_transform,
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        TierDrivenSunLight,
    ));

    let body_mesh = meshes.add(Mesh::from(Capsule3d::new(0.5, 1.6)));
    let wheel_mesh = meshes.add(Mesh::from(Torus::new(0.3, 0.6)));
    let spike_mesh = meshes.add(Mesh::from(Cone {
        radius: 0.45,
        height: 1.0,
    }));
    let sphere_mesh = meshes.add(Mesh::from(Sphere::new(0.5)));
    let quad_mesh = meshes.add(Mesh::from(Rectangle::new(1.0, 1.0)));
    let ring_mesh = meshes.add(Mesh::from(Torus::new(0.7, 1.0)));
    commands.insert_resource(AgentMeshes {
        base_radius: 1.0,
        body: body_mesh,
        wheel: wheel_mesh,
        spike: spike_mesh,
        sphere: sphere_mesh,
        quad: quad_mesh,
        ring: ring_mesh,
    });
    commands.insert_resource(TerrainChunkRegistry {
        chunk_size: TERRAIN_CHUNK_SIZE,
        height_scale: TERRAIN_HEIGHT_SCALE,
        ..default()
    });
    commands.insert_resource(CameraRig::default());

    commands.insert_resource(ReflectionProbeAssets {
        diffuse: Handle::default(),
        specular: Handle::default(),
    });

    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            ..Default::default()
        },
        Transform::default(),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
    ));

    let primary_text_color = Color::WHITE;
    let secondary_text_color = Color::srgb(0.74, 0.82, 0.94);
    let primary_font = TextFont::from_font_size(18.0);
    let secondary_font = TextFont::from_font_size(15.0);

    let button_node = Node {
        padding: UiRect::axes(Val::Px(12.0), Val::Px(8.0)),
        border: UiRect::all(Val::Px(1.0)),
        align_items: AlignItems::Center,
        justify_content: JustifyContent::Center,
        min_width: Val::Px(120.0),
        ..default()
    };
    let button_row_node = Node {
        flex_direction: FlexDirection::Row,
        column_gap: Val::Px(8.0),
        row_gap: Val::Px(8.0),
        margin: UiRect::axes(Val::Px(0.0), Val::Px(8.0)),
        ..default()
    };
    let button_border_color = Color::srgba(0.32, 0.38, 0.58, 1.0);

    // Persistent SOFTWARE RENDERER watermark (bd-2z0.14.3.3 item 4).
    //
    // The startup warning is a log line, and a log line scrolls away — someone
    // reading a screenshot, a screen recording, or a bug report has no way to
    // know the frame came from llvmpipe rather than a GPU. The tier is already
    // forced to Potato in that case (f243255fd51), so the picture is genuinely
    // not representative, and the frame itself has to say so.
    //
    // Spawned once and never removed: the adapter cannot change mid-run, so
    // there is no state to keep in sync and nothing that can silently drop it.
    // Anchored top-right so it does not overlap the top-left HUD panel.
    if effective
        .gpu
        .as_ref()
        .is_some_and(|info| info.class == GpuClass::Software)
    {
        commands.spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                right: Val::Px(12.0),
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.35, 0.10, 0.10, 0.82)),
            Text::new("SOFTWARE RENDERER — not GPU accelerated, not representative"),
            TextFont::from_font_size(15.0),
            TextColor(Color::srgb(1.0, 0.85, 0.85)),
            SoftwareRendererWatermark,
        ));
    }

    let hud_root = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                padding: UiRect::all(Val::Px(10.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.07, 0.11, 0.18, 0.72)),
        ))
        .id();

    let mut tick = Entity::PLACEHOLDER;
    let mut agents = Entity::PLACEHOLDER;
    let mut selection = Entity::PLACEHOLDER;
    let mut follow = Entity::PLACEHOLDER;
    let mut camera = Entity::PLACEHOLDER;
    let mut playback = Entity::PLACEHOLDER;
    let mut fps = Entity::PLACEHOLDER;
    let mut world = Entity::PLACEHOLDER;
    let mut tonemap = Entity::PLACEHOLDER;
    let mut palette = Entity::PLACEHOLDER;
    let mut events = Entity::PLACEHOLDER;
    let mut inspector = Entity::PLACEHOLDER;
    let mut history = Entity::PLACEHOLDER;
    let mut brain = Entity::PLACEHOLDER;

    commands.entity(hud_root).with_children(|parent| {
        tick = parent
            .spawn((
                Text::new("Tick: --"),
                primary_font.clone(),
                TextColor(primary_text_color),
            ))
            .id();
        agents = parent
            .spawn((
                Text::new("Agents: --"),
                primary_font.clone(),
                TextColor(primary_text_color),
            ))
            .id();
        selection = parent
            .spawn((
                Text::new("Selection: --"),
                secondary_font.clone(),
                TextColor(secondary_text_color),
            ))
            .id();
        follow = parent
            .spawn((
                Text::new("Follow: --"),
                primary_font.clone(),
                TextColor(primary_text_color),
            ))
            .id();
        camera = parent
            .spawn((
                Text::new("Camera: --"),
                primary_font.clone(),
                TextColor(primary_text_color),
            ))
            .id();
        playback = parent
            .spawn((
                Text::new("Playback: --"),
                secondary_font.clone(),
                TextColor(secondary_text_color),
            ))
            .id();
        fps = parent
            .spawn((
                Text::new("FPS: --"),
                secondary_font.clone(),
                TextColor(secondary_text_color),
            ))
            .id();
        world = parent
            .spawn((
                Text::new("World: --"),
                secondary_font.clone(),
                TextColor(secondary_text_color),
            ))
            .id();
        tonemap = parent
            .spawn((
                Text::new("Tone: ACES • AutoExp Off • Bias +0.0"),
                secondary_font.clone(),
                TextColor(secondary_text_color),
            ))
            .id();
        palette = parent
            .spawn((
                Text::new("Palette: Natural • press C to cycle"),
                secondary_font.clone(),
                TextColor(secondary_text_color),
            ))
            .id();
        events = parent
            .spawn((
                Text::new("Events: --"),
                secondary_font.clone(),
                TextColor(secondary_text_color),
            ))
            .id();
        inspector = parent
            .spawn((
                Text::new("Inspector: no selection"),
                secondary_font.clone(),
                TextColor(secondary_text_color),
            ))
            .id();
        history = parent
            .spawn((
                Text::new("History: collecting…"),
                secondary_font.clone(),
                TextColor(secondary_text_color),
            ))
            .id();
        brain = parent
            .spawn((
                Text::new("Brain: select an agent"),
                secondary_font.clone(),
                TextColor(secondary_text_color),
            ))
            .id();

        let playback_buttons = [
            (PlaybackAction::Play, "▶ Run (Space)"),
            (PlaybackAction::Pause, "⏸ Pause"),
            (PlaybackAction::Step, "⏭ Step (N)"),
            (PlaybackAction::SpeedDown, "➖ Speed (−)"),
            (PlaybackAction::SpeedUp, "➕ Speed (+)"),
        ];

        parent
            .spawn((button_row_node.clone(),))
            .with_children(|row| {
                for (action, label) in playback_buttons {
                    row.spawn((
                        Button,
                        button_node.clone(),
                        BackgroundColor(follow_idle_color()),
                        BorderRadius::all(Val::Px(6.0)),
                        BorderColor::all(button_border_color),
                        PlaybackButton { action },
                    ))
                    .with_children(|btn| {
                        btn.spawn((
                            Text::new(label),
                            secondary_font.clone(),
                            TextColor(secondary_text_color),
                        ));
                    });
                }
            });

        let follow_buttons = [
            (FollowMode::Off, "🛑 Follow off (F)"),
            (FollowMode::Selected, "🎯 Follow selected (Ctrl+S)"),
            (FollowMode::Oldest, "📜 Follow oldest (Ctrl+O)"),
        ];

        parent
            .spawn((button_row_node.clone(),))
            .with_children(|row| {
                for (mode, label) in follow_buttons {
                    row.spawn((
                        Button,
                        button_node.clone(),
                        BackgroundColor(follow_idle_color()),
                        BorderRadius::all(Val::Px(6.0)),
                        BorderColor::all(button_border_color),
                        FollowButton { mode },
                    ))
                    .with_children(|btn| {
                        btn.spawn((
                            Text::new(label),
                            secondary_font.clone(),
                            TextColor(secondary_text_color),
                        ));
                    });
                }

                row.spawn((
                    Button,
                    button_node.clone(),
                    BackgroundColor(follow_idle_color()),
                    BorderRadius::all(Val::Px(6.0)),
                    BorderColor::all(button_border_color),
                    ClearSelectionButton,
                ))
                .with_children(|btn| {
                    btn.spawn((
                        Text::new("✖ Clear selection (Esc)"),
                        secondary_font.clone(),
                        TextColor(secondary_text_color),
                    ));
                });
            });

        let tonemap_modes = [
            TonemappingMode::Aces,
            TonemappingMode::Agx,
            TonemappingMode::Tony,
        ];

        parent
            .spawn((button_row_node.clone(),))
            .with_children(|row| {
                for mode in tonemap_modes {
                    row.spawn((
                        Button,
                        button_node.clone(),
                        BackgroundColor(follow_idle_color()),
                        BorderRadius::all(Val::Px(6.0)),
                        BorderColor::all(button_border_color),
                        TonemapButton { mode },
                    ))
                    .with_children(|btn| {
                        btn.spawn((
                            Text::new(mode.label()),
                            secondary_font.clone(),
                            TextColor(secondary_text_color),
                        ));
                    });
                }
            });

        parent
            .spawn((button_row_node.clone(),))
            .with_children(|row| {
                row.spawn((
                    Button,
                    button_node.clone(),
                    BackgroundColor(follow_idle_color()),
                    BorderRadius::all(Val::Px(6.0)),
                    BorderColor::all(button_border_color),
                    AutoExposureToggleButton,
                ))
                .with_children(|btn| {
                    btn.spawn((
                        Text::new("Auto Exposure"),
                        secondary_font.clone(),
                        TextColor(secondary_text_color),
                    ));
                });

                row.spawn((
                    Button,
                    button_node.clone(),
                    BackgroundColor(follow_idle_color()),
                    BorderRadius::all(Val::Px(6.0)),
                    BorderColor::all(button_border_color),
                    QualityTierButton,
                ))
                .with_children(|btn| {
                    btn.spawn((
                        Text::new("Quality"),
                        secondary_font.clone(),
                        TextColor(secondary_text_color),
                    ));
                });

                row.spawn((
                    Button,
                    button_node.clone(),
                    BackgroundColor(follow_idle_color()),
                    BorderRadius::all(Val::Px(6.0)),
                    BorderColor::all(button_border_color),
                    ExposureAdjustButton { delta: -0.5 },
                ))
                .with_children(|btn| {
                    btn.spawn((
                        Text::new("Exposure –"),
                        secondary_font.clone(),
                        TextColor(secondary_text_color),
                    ));
                });

                row.spawn((
                    Button,
                    button_node.clone(),
                    BackgroundColor(follow_idle_color()),
                    BorderRadius::all(Val::Px(6.0)),
                    BorderColor::all(button_border_color),
                    ExposureAdjustButton { delta: 0.5 },
                ))
                .with_children(|btn| {
                    btn.spawn((
                        Text::new("Exposure +"),
                        secondary_font.clone(),
                        TextColor(secondary_text_color),
                    ));
                });
            });
    });

    commands.insert_resource(HudElements {
        tick,
        agents,
        selection,
        follow,
        camera,
        playback,
        fps,
        world,
        tonemap,
        palette,
        events,
        inspector,
        history,
        brain,
    });
}

fn poll_snapshots(inbox: NonSendMut<SnapshotInbox>, mut state: ResMut<SnapshotState>) {
    let receiver = &inbox.receiver;
    while let Ok(snapshot) = receiver.try_recv() {
        state.latest = Some(snapshot);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sync_world(
    mut commands: Commands,
    mut state: ResMut<SnapshotState>,
    mut registry: ResMut<AgentRegistry>,
    mut terrain_registry: ResMut<TerrainChunkRegistry>,
    agent_meshes: Res<AgentMeshes>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    probe_assets: Res<ReflectionProbeAssets>,
    accessibility: Res<AccessibilityState>,
) {
    let Some(snapshot) = state.latest.as_ref() else {
        return;
    };

    let palette = accessibility.palette();
    if state.last_applied_revision == snapshot.revision
        && state.last_applied_palette == Some(palette)
    {
        return;
    }

    let snapshot_revision = snapshot.revision;
    let snapshot_tick = snapshot.tick;
    let world_size = snapshot.world_size;
    let world_center = Vec2::new(world_size.x * 0.5, world_size.y * 0.5);

    let mut selection_min = Vec2::splat(f32::INFINITY);
    let mut selection_max = Vec2::splat(f32::NEG_INFINITY);
    let mut has_selection = false;
    let mut oldest: Option<(Vec2, u32)> = None;
    let mut first_agent = None;

    for agent in &snapshot.agents {
        if first_agent.is_none() {
            first_agent = Some(agent.position);
        }
        if matches!(agent.selection, SelectionState::Selected) {
            selection_min = selection_min.min(agent.position);
            selection_max = selection_max.max(agent.position);
            has_selection = true;
        }
        match oldest {
            None => oldest = Some((agent.position, agent.age)),
            Some((_, age)) if agent.age > age => oldest = Some((agent.position, agent.age)),
            _ => {}
        }
    }

    let selection_bounds = has_selection.then_some((selection_min, selection_max));
    let selection_center = selection_bounds.map(|(min, max)| (min + max) * 0.5);
    let focus_point = selection_center.or(first_agent).unwrap_or(world_center);

    sync_terrain(
        snapshot,
        &mut commands,
        &mut terrain_registry,
        meshes.as_mut(),
        materials.as_mut(),
        probe_assets.as_ref(),
        palette,
    );
    sync_agents(
        snapshot,
        &mut commands,
        &mut registry,
        agent_meshes.as_ref(),
        materials.as_mut(),
        palette,
    );

    state.last_applied_revision = snapshot_revision;
    state.last_applied_tick = snapshot_tick;
    state.last_applied_palette = Some(palette);
    state.focus_point = focus_point;
    state.world_size = world_size;
    state.world_center = world_center;
    state.selection_bounds = selection_bounds;
    state.selection_center = selection_center;
    state.oldest_position = oldest.map(|(pos, _)| pos);
    state.first_agent_position = first_agent;
}

#[allow(clippy::too_many_arguments)]
fn update_hud(
    mut state: ResMut<SnapshotState>,
    rig: Res<CameraRig>,
    hud: Option<Res<HudElements>>,
    time: Res<Time>,
    controls: Res<SimulationControl>,
    tonemap_state: Res<TonemappingState>,
    accessibility: Res<AccessibilityState>,
    mut texts: Query<&mut Text>,
) {
    let Some(_) = state.latest.as_ref() else {
        return;
    };

    let (
        tick,
        agent_count,
        world_size,
        agent_radius,
        selected_count,
        primary_selection,
        event_feed,
        inspector_text,
        history_panel,
        brain_panel,
    ) = {
        let snapshot = state.latest.as_ref().expect("snapshot available");
        let tick = snapshot.tick;
        let agent_count = snapshot.agents.len();
        let world_size = snapshot.world_size;
        let agent_radius = snapshot.agent_radius;
        let mut selected_count = 0usize;
        let mut primary: Option<(AgentId, u32, f32)> = None;
        let mut primary_agent: Option<&AgentVisual> = None;
        for agent in &snapshot.agents {
            if matches!(agent.selection, SelectionState::Selected) {
                selected_count += 1;
                if primary.is_none() {
                    primary = Some((agent.id, agent.age, agent.health));
                    primary_agent = Some(agent);
                }
            }
        }
        // Formatted here so the snapshot borrow ends with the rest of the
        // extraction rather than being held across the text writes below.
        let event_feed = format_event_feed(&snapshot.events);
        let history_panel = format_history_panel(&snapshot.history);
        let brain_panel = format_brain_overlay(&snapshot.brain);
        let inspector_text = format_inspector(
            primary_agent
                .map(|agent| InspectorDetail::from_agent(agent, selected_count.saturating_sub(1))),
        );
        (
            tick,
            agent_count,
            world_size,
            agent_radius,
            selected_count,
            primary,
            event_feed,
            inspector_text,
            history_panel,
            brain_panel,
        )
    };

    if tick != state.last_reported_tick && tick % 120 == 0 {
        info!(tick, agents = agent_count, "Bevy world snapshot applied");
    }
    state.last_reported_tick = tick;

    let now = time.elapsed_secs_f64();
    if state.hud_prev_tick == 0 {
        state.hud_prev_tick = tick;
        state.hud_prev_time = now;
    }
    if tick > state.hud_prev_tick {
        let delta_tick = (tick - state.hud_prev_tick) as f64;
        let delta_time = (now - state.hud_prev_time).max(1e-4);
        state.sim_rate = (delta_tick / delta_time) as f32;
        state.hud_prev_tick = tick;
        state.hud_prev_time = now;
    }
    let control_snapshot = controls.snapshot();
    let playback_status = if control_snapshot.paused {
        "Paused"
    } else {
        "Running"
    };

    let selection_text = if let Some((id, age, health)) = primary_selection {
        let extra = if selected_count > 1 {
            format!(" • +{}", selected_count - 1)
        } else {
            String::new()
        };
        format!(
            "Selection: {:?} • age {:>4} • health {:>5.1}{}",
            id, age, health, extra
        )
    } else {
        "Selection: none".to_string()
    };

    if let Some(hud_elements) = hud {
        if let Ok(mut text) = texts.get_mut(hud_elements.tick) {
            **text = format!("Tick: {}", tick);
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.agents) {
            **text = format!(
                "Agents: {:>4} (selected {:>2})",
                agent_count, selected_count
            );
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.selection) {
            **text = selection_text.clone();
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.follow) {
            **text = format!(
                "Follow: {} • F cycle • Ctrl+S sel • Ctrl+O oldest",
                rig.follow_mode.label()
            );
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.camera) {
            let yaw_deg = rig.yaw.to_degrees();
            let pitch_deg = rig.pitch.to_degrees();
            **text = format!(
                "Camera: dist {:>5.0} yaw {:>6.1}° pitch {:>5.1}° • Ctrl+F fit selection • Ctrl+W fit world",
                rig.distance, yaw_deg, pitch_deg
            );
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.playback) {
            let speed = control_snapshot.speed_multiplier.clamp(0.0, 999.9);
            let mut message = if speed > 0.0 {
                format!("Playback: {} • x{speed:>4.1}", playback_status)
            } else {
                format!("Playback: {} • x0.0", playback_status)
            };
            if let Some(reason) = control_snapshot.auto_pause_reason {
                message.push_str(" • ");
                message.push_str(&reason);
            }
            **text = message;
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.fps) {
            let delta_seconds = time.delta_secs();
            let fps = if delta_seconds > f32::EPSILON {
                1.0 / delta_seconds
            } else {
                0.0
            };
            **text = format!("FPS: {:>5.1}", fps);
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.world) {
            **text = format!(
                "World: {:>4}×{:>4} • r {:>4.1}",
                world_size.x as i32, world_size.y as i32, agent_radius
            );
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.tonemap) {
            let mode_label = tonemap_state.mode.label();
            let auto_label = if tonemap_state.auto_exposure_enabled {
                "AutoExp On"
            } else {
                "AutoExp Off"
            };
            **text = format!(
                "Tone: {} • {} • Bias {:+.1}",
                mode_label, auto_label, tonemap_state.exposure_bias
            );
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.palette) {
            **text = format!("{} • press C to cycle", accessibility.palette().label());
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.events) {
            **text = event_feed;
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.inspector) {
            **text = inspector_text;
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.history) {
            **text = history_panel;
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.brain) {
            **text = brain_panel;
        }
    }
}

#[cfg(test)]
mod hud_inspector_tests {
    use super::*;

    fn sample_agent_visual_for_inspector() -> AgentVisual {
        AgentVisual {
            id: AgentId::null(),
            position: Vec2::new(10.0, 20.0),
            heading: 0.0,
            color: [0.2, 0.4, 0.7],
            selection: SelectionState::Selected,
            health: 1.4,
            energy: 0.62,
            age: 120,
            generation: 7,
            reference_age_ticks: 1,
            spike_length: 4.0,
            boost: 1.0,
            wheel_left: 0.25,
            wheel_right: -0.5,
            herbivore_tendency: 0.8,
            temperature_preference: 0.3,
            food_delta: 0.4,
            sound_level: 0.2,
            sound_output: 0.1,
            sound_multiplier: 1.0,
            trait_modifiers: TraitModifiers::default(),
            eye_dirs: [0.0; NUM_EYES],
            eye_fov: [1.0; NUM_EYES],
            indicator: IndicatorState::default(),
            reproduction_intent: 0.25,
            spiked: false,
        }
    }

    /// Built directly rather than borrowing the sibling test module's
    /// `sample_agent_visual` helper, which is private to that module.
    /// Energy and health are deliberately distinct values so a transposition
    /// in the format string cannot pass.
    fn detail() -> InspectorDetail {
        InspectorDetail {
            id: AgentId::null(),
            energy: 0.62,
            health: 1.40,
            age: 120,
            generation: 7,
            herbivore_tendency: 0.80,
            temperature_preference: 0.30,
            spike_length: 4.00,
            boost: 1.00,
            reproduction_intent: 0.25,
            spiked: false,
            trait_modifiers: TraitModifiers::default(),
            also_selected: 0,
        }
    }

    /// The whole point of the bead: the overlay must carry more than the
    /// id/age/health the old one-line readout showed. Energy and generation
    /// in particular were absent from the snapshot entirely before this.
    #[test]
    fn inspector_reports_vitals_the_one_line_readout_omitted() {
        let line = format_inspector(Some(detail()));
        for expected in ["Energy", "0.62", "Gen 7", "Health", "1.40", "Age", "120"] {
            assert!(line.contains(expected), "missing {expected} in: {line}");
        }
    }

    /// `from_agent` is a plain field copy, which is exactly where a
    /// transposition hides. Pin the mapping against a real `AgentVisual`.
    #[test]
    fn from_agent_maps_each_field_to_its_own_slot() {
        let mut agent = sample_agent_visual_for_inspector();
        agent.energy = 0.11;
        agent.health = 0.22;
        agent.age = 33;
        agent.generation = 44;
        let d = InspectorDetail::from_agent(&agent, 2);
        assert!((d.energy - 0.11).abs() < f32::EPSILON, "energy mismapped");
        assert!((d.health - 0.22).abs() < f32::EPSILON, "health mismapped");
        assert_eq!(d.age, 33, "age mismapped");
        assert_eq!(d.generation, 44, "generation mismapped");
        assert_eq!(d.also_selected, 2);
    }

    /// Diet wording is derived from the same herbivore tendency the renderer
    /// colours by, so the label and the colour cannot disagree.
    #[test]
    fn diet_label_follows_herbivore_tendency() {
        let mut d = detail();
        d.herbivore_tendency = 0.9;
        assert_eq!(d.diet(), "herbivore");
        d.herbivore_tendency = 0.5;
        assert_eq!(d.diet(), "omnivore");
        d.herbivore_tendency = 0.1;
        assert_eq!(d.diet(), "carnivore");
    }

    /// A spike that landed a hit must be visibly distinct from one merely
    /// extended, since that is the difference between threat and posture.
    #[test]
    fn spike_hit_is_called_out() {
        let mut d = detail();
        d.spiked = true;
        assert!(format_inspector(Some(d)).contains("HIT"));
        d.spiked = false;
        assert!(!format_inspector(Some(d)).contains("HIT"));
    }

    /// Multi-select reports the remainder, and a lone selection says nothing
    /// about extras.
    #[test]
    fn multi_selection_reports_the_remainder() {
        let mut d = detail();
        d.also_selected = 3;
        assert!(format_inspector(Some(d)).contains("+3 more selected"));
        d.also_selected = 0;
        assert!(!format_inspector(Some(d)).contains("more selected"));
    }

    /// With nothing selected the panel keeps a stable presence rather than
    /// disappearing — the same mounting lesson as bd-rzy3.
    #[test]
    fn empty_selection_renders_a_stable_placeholder() {
        assert_eq!(format_inspector(None), "Inspector: no selection");
    }

    /// Trait modifiers are surfaced, matching GPUI's inspector vocabulary.
    #[test]
    fn trait_modifiers_are_surfaced() {
        let line = format_inspector(Some(detail()));
        for expected in ["smell", "sound", "hearing", "eye", "blood"] {
            assert!(line.contains(expected), "missing {expected} in: {line}");
        }
    }
}

/// bd-ikts.1: every computed visual authority must have a live consumer, or an
/// explicit exemption naming the bead that will wire it.
///
/// bd-ikts diagnosed the root cause of the visual layer looking flat: 1,610
/// lines of renderer-neutral appearance semantics in `scriptbots-core::visual`
/// that no renderer called. That has since been largely wired by hand, bead by
/// bead — and then the SAME failure recurred one layer up, where
/// `EffectiveRenderSettings` is resolved against a real GPU probe, inserted as
/// a Bevy resource, and read by nobody.
///
/// The defect is not any single unwired item. It is that nothing detects the
/// difference between "wired" and "computed then ignored" — both compile, both
/// pass tests, and the gap is invisible until somebody greps. This module is
/// that detector.
#[cfg(test)]
mod visual_authority_consumer_guard {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    /// Authorities that are deliberately not consumed yet, each with the bead
    /// that owns wiring or retiring it.
    ///
    /// An entry here is a TRACKED DECISION, not a silence. Adding one without a
    /// bead id is the thing this guard exists to prevent.
    ///
    /// `cell_phase` was on this list and should not have been: it is called by
    /// `shimmer`, which frontends do consume. The first version of this guard
    /// could not see that, because it excluded `visual.rs` wholesale and so read
    /// every internal helper as dead. An audit that cannot tell "dead" from
    /// "helper of a live function" produces deletion advice for working code,
    /// which is the failure mode bd-ikts.2 warns about — so the guard now counts
    /// production call sites inside `visual.rs` as well.
    /// `terrain_lushness` was also listed here and also should not have been:
    /// the TUI calls it in production (`app/src/terminal/mod.rs:3535`). My
    /// manual survey reported it unconsumed; the survey was wrong and this
    /// guard caught it. That is the argument for having the check at all —
    /// a hand grep is a snapshot of one person's shell history, and this runs
    /// on every build.
    const EXEMPT: &[(&str, &str)] = &[("bake_biome_atlas", "bd-ikts.2")];

    fn workspace_root() -> PathBuf {
        // crates/scriptbots-bevy -> crates -> workspace root
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root above crates/scriptbots-bevy")
            .to_path_buf()
    }

    /// Text of this guard module is cut from every file before scanning.
    ///
    /// Without this the guard reads its own source and its `EXEMPT` string
    /// literals count as consumers, so every exemption immediately looks stale.
    /// That is not hypothetical — it is exactly how this test first failed.
    /// An allowlist entry naming an authority is not a use of it.
    const GUARD_MODULE_MARKER: &str = "mod visual_authority_consumer_guard";

    /// Every `.rs` file under `crates/`, excluding `visual.rs` itself and this
    /// guard module: a definition is not a consumer, and neither is a mention
    /// inside the detector.
    ///
    /// Sweeping the WHOLE workspace matters. Measuring only the two GPU
    /// frontends makes `resolve_day_night` and `terrain_normal_light_factor`
    /// look unconsumed when they are in fact used by core and by the terminal
    /// frontend; a narrower guard would invite deleting live code.
    fn consumer_sources(root: &Path) -> Vec<String> {
        fn walk(dir: &Path, out: &mut Vec<String>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if path.file_name().is_some_and(|n| n == "target") {
                        continue;
                    }
                    walk(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs")
                    && !path.ends_with("scriptbots-core/src/visual.rs")
                    && let Ok(text) = std::fs::read_to_string(&path)
                {
                    let before_guard = text
                        .split_once(GUARD_MODULE_MARKER)
                        .map_or(text.as_str(), |(before, _)| before);
                    out.push(before_guard.to_string());
                }
            }
        }
        let mut out = Vec::new();
        walk(&root.join("crates"), &mut out);
        out
    }

    /// `visual.rs`'s own production half, with definitions and comments removed.
    ///
    /// An authority used only by a sibling in `visual.rs` is NOT dead when that
    /// sibling is itself consumed — `cell_phase` is exactly this, reached
    /// through `shimmer`. Excluding the file wholesale, as the first version of
    /// this guard did, reports such helpers as unconsumed and invites deleting
    /// live code. Test call sites are excluded because a function whose only
    /// callers are its own tests IS dead in production, which is a real finding
    /// rather than a consumer.
    fn visual_internal_production_uses(root: &Path) -> String {
        let source = std::fs::read_to_string(root.join("crates/scriptbots-core/src/visual.rs"))
            .expect("visual.rs readable");
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source.as_str(), |(before, _)| before);
        production
            .lines()
            .filter(|line| {
                let t = line.trim_start();
                !t.starts_with("//") && !t.starts_with("pub fn ") && !t.starts_with("fn ")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Files that own a canonical authority, and therefore may define it.
    ///
    /// Each entry is (source path, importable module path). The module path is
    /// what a failure message tells the reader to call, so it must be the real
    /// import route and not just a filename.
    const AUTHORITY_OWNERS: &[(&str, &str)] = &[
        ("crates/scriptbots-core/src/lib.rs", "scriptbots_core"),
        (
            "crates/scriptbots-core/src/visual.rs",
            "scriptbots_core::visual",
        ),
    ];

    /// Crates that consume authorities and must never re-define one.
    const CONSUMER_CRATE_DIRS: &[&str] = &[
        "crates/scriptbots-bevy/src",
        "crates/scriptbots-render/src",
        "crates/scriptbots-app/src",
        "crates/scriptbots-world-gfx/src",
    ];

    /// Every `pub fn` name owned by core, paired with the module that owns it.
    ///
    /// bd-ikts.5: the consumer guard above catches an authority with NO
    /// consumer. This catches the opposite failure — an authority with too
    /// MANY implementations. bd-ikts.4 found `toroidal_delta` defined three
    /// times, the copies missing core's non-finite and non-positive-extent
    /// guards and disagreeing with it at exactly half a world, and nothing in
    /// the build noticed.
    fn core_authority_owners(root: &Path) -> BTreeMap<String, &'static str> {
        let mut owners = BTreeMap::new();
        for (path, module) in AUTHORITY_OWNERS {
            let Ok(source) = std::fs::read_to_string(root.join(path)) else {
                continue;
            };
            for line in source.lines() {
                if let Some(rest) = line.strip_prefix("pub fn ")
                    && let Some(name) = rest.split(['(', '<']).next()
                {
                    let name = name.trim();
                    if !name.is_empty() {
                        owners.insert(name.to_string(), *module);
                    }
                }
            }
        }
        owners
    }

    /// Function definitions in the consumer crates, as (name, file, is_test).
    ///
    /// Comments are stripped so prose naming a function is not read as
    /// defining it, and this guard module is cut so its own text cannot
    /// trigger it — both mistakes were made and caught earlier today.
    fn consumer_fn_definitions(root: &Path) -> Vec<(String, String)> {
        let mut found = Vec::new();
        for dir in CONSUMER_CRATE_DIRS {
            collect_fn_definitions(&root.join(dir), root, &mut found);
        }
        found
    }

    fn collect_fn_definitions(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_fn_definitions(&path, root, out);
                continue;
            }
            if !path.extension().is_some_and(|e| e == "rs") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            // Cut this guard module so its own literals cannot trip it.
            let text = text
                .split_once(GUARD_MODULE_MARKER)
                .map_or(text.as_str(), |(before, _)| before);
            let display = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            for line in text.lines() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("//") {
                    continue;
                }
                let after = trimmed
                    .strip_prefix("pub fn ")
                    .or_else(|| trimmed.strip_prefix("fn "))
                    .or_else(|| trimmed.strip_prefix("pub(crate) fn "));
                if let Some(rest) = after
                    && let Some(name) = rest.split(['(', '<']).next()
                {
                    let name = name.trim();
                    if !name.is_empty() {
                        out.push((name.to_string(), display.clone()));
                    }
                }
            }
        }
    }

    /// A consumer crate must never re-implement a core authority.
    ///
    /// The message names the CANONICAL OWNER rather than merely reporting a
    /// duplicate, because "X is duplicated" sends the reader hunting while
    /// "X is owned by scriptbots_core, call it" tells them what to do. That is
    /// the difference between a guard that gets obeyed and one that gets
    /// suppressed.
    #[test]
    fn no_consumer_crate_reimplements_a_core_authority() {
        let root = workspace_root();
        let owners = core_authority_owners(&root);
        assert!(
            owners.len() > 10,
            "only {} core authorities found; the extraction is broken, not the codebase",
            owners.len()
        );

        let definitions = consumer_fn_definitions(&root);
        assert!(
            definitions.len() > 50,
            "only {} consumer fn definitions found; the sweep is broken",
            definitions.len()
        );

        let mut offences = Vec::new();
        for (name, file) in &definitions {
            if let Some(module) = owners.get(name.as_str()) {
                offences.push(format!(
                    "{file} defines `{name}`, which is owned by `{module}` — \
                     call `{module}::{name}` instead of re-implementing it"
                ));
            }
        }
        assert!(
            offences.is_empty(),
            "a core authority was re-implemented in a consumer crate:\n  {}\n\
             bd-ikts.4 found toroidal_delta defined three times, the copies \
             silently disagreeing with core at exactly half a world. If a \
             same-named private helper is genuinely unrelated, rename it.",
            offences.join("\n  ")
        );
    }

    /// The duplication guard must actually bite.
    ///
    /// A synthetic definition of a real authority, in a path the sweep covers,
    /// must be recognised as an offence. Without this the test above is
    /// indistinguishable from one whose extraction silently returns nothing.
    #[test]
    fn duplication_guard_detects_a_synthetic_reimplementation() {
        let root = workspace_root();
        let owners = core_authority_owners(&root);
        let authority = owners
            .keys()
            .next()
            .cloned()
            .expect("core exposes at least one authority");
        let synthetic = [(
            authority.clone(),
            "crates/scriptbots-bevy/src/fake.rs".to_string(),
        )];
        let offences: Vec<_> = synthetic
            .iter()
            .filter(|(name, _)| owners.contains_key(name.as_str()))
            .collect();
        assert_eq!(
            offences.len(),
            1,
            "a synthetic re-implementation of `{authority}` must be flagged; \
             if it is not, the owner table or the matching is broken"
        );
    }

    fn visual_authorities(root: &Path) -> Vec<String> {
        let source = std::fs::read_to_string(root.join("crates/scriptbots-core/src/visual.rs"))
            .expect("visual.rs readable");
        source
            .lines()
            .filter_map(|line| line.strip_prefix("pub fn "))
            .filter_map(|rest| rest.split(['(', '<']).next())
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// The guard. A new public authority in `visual.rs` fails this test until it
    /// is either consumed somewhere in the workspace or exempted with a bead id.
    #[test]
    fn every_visual_authority_is_consumed_or_explicitly_exempt() {
        let root = workspace_root();
        let sources = consumer_sources(&root);
        assert!(
            sources.len() > 10,
            "consumer sweep found only {} files; the walk is broken, not the codebase",
            sources.len()
        );
        let exempt: BTreeMap<&str, &str> = EXEMPT.iter().copied().collect();
        let internal = visual_internal_production_uses(&root);

        let mut unconsumed = Vec::new();
        let mut exempt_but_actually_consumed = Vec::new();
        for name in visual_authorities(&root) {
            let consumed =
                sources.iter().any(|text| text.contains(&name)) || internal.contains(&name);
            match (consumed, exempt.get(name.as_str())) {
                (false, None) => unconsumed.push(name),
                (true, Some(bead)) => exempt_but_actually_consumed.push(format!("{name} ({bead})")),
                _ => {}
            }
        }

        assert!(
            unconsumed.is_empty(),
            "these visual authorities are computed but consumed by nobody: {unconsumed:?}. \
             Either wire a consumer, or add an EXEMPT entry naming the bead that will. \
             Leaving it unlisted is how bd-ikts happened."
        );
        assert!(
            exempt_but_actually_consumed.is_empty(),
            "these are exempt but now have real consumers, so the exemption is stale \
             and should be removed along with its bead: {exempt_but_actually_consumed:?}"
        );
    }

    /// A helper reached only from a sibling in `visual.rs` is alive when that
    /// sibling is consumed, and must NOT be reported as dead.
    ///
    /// `cell_phase` is the concrete case: nothing outside `visual.rs` calls it,
    /// but `shimmer` does, and frontends call `shimmer`. The first version of
    /// this guard exempted it as unconsumed, which was wrong and would have led
    /// someone to delete a live function.
    #[test]
    fn internal_helpers_of_consumed_functions_are_not_dead() {
        let root = workspace_root();
        let internal = visual_internal_production_uses(&root);
        assert!(
            internal.contains("cell_phase"),
            "cell_phase is called by shimmer in visual.rs production code; if this \
             fails the internal sweep is not seeing intra-file call sites"
        );
        let sources = consumer_sources(&root);
        assert!(
            sources.iter().any(|t| t.contains("shimmer")),
            "shimmer must have an external consumer for this case to hold"
        );
        // Definitions and comments must not count as uses, or everything looks alive.
        assert!(
            !internal.contains("pub fn cell_phase"),
            "definition lines must be stripped from the internal sweep"
        );
    }

    /// Test-only call sites are not consumers. A function whose only callers are
    /// its own tests is dead in production, and saying so is the point.
    ///
    /// If this ever stopped holding, the guard would go permanently blind to
    /// the exact class it exists to catch: every dead function has tests, so
    /// counting test callers would mark all of them alive.
    #[test]
    fn test_only_call_sites_do_not_count_as_consumers() {
        let internal = visual_internal_production_uses(&workspace_root());
        assert!(
            !internal.contains("bake_biome_atlas"),
            "bake_biome_atlas is called only from tests, so it must not appear in \
             the production-only sweep; if it does, the #[cfg(test)] split is wrong"
        );
    }

    /// An exemption without a bead id is an excuse. Reject it at the guard.
    #[test]
    fn every_exemption_names_a_bead() {
        for (name, bead) in EXEMPT {
            assert!(
                bead.starts_with("bd-"),
                "exemption for {name} must name a bead, got {bead:?}"
            );
        }
    }

    /// The guard must actually bite. A name that exists nowhere stands in for a
    /// newly added, unwired authority.
    #[test]
    fn guard_detects_an_unconsumed_authority() {
        let sources = consumer_sources(&workspace_root());
        let invented = "totally_unwired_visual_authority_bd_ikts_1";
        assert!(
            !sources.iter().any(|text| text.contains(invented)),
            "the detection this guard relies on is broken if an invented name appears to be consumed"
        );
    }
}

#[cfg(test)]
mod adaptive_governor_tests {
    use super::*;

    /// A launch that is permitted to adapt, with a known tier, built by hand so
    /// these tests never depend on the host having a particular GPU.
    fn auto_launch(tier: RenderQuality) -> EffectiveRenderSettings {
        EffectiveRenderSettings {
            tier,
            features: tier_features(tier),
            gpu: None,
            requested: RenderQuality::Auto,
        }
    }

    /// The real system, driven by a deterministic clock rather than wall time.
    fn governed_app(tier: RenderQuality) -> App {
        let effective = auto_launch(tier);
        let mut app = App::new();
        app.insert_resource(Time::<()>::default())
            .insert_resource(AdaptiveQualityGovernor::for_launch(&effective))
            .insert_resource(effective)
            .add_systems(Update, drive_adaptive_quality);
        app
    }

    fn run_frames(app: &mut App, frames: usize, frame_time: std::time::Duration) {
        for _ in 0..frames {
            app.world_mut()
                .resource_mut::<Time>()
                .advance_by(frame_time);
            app.update();
        }
    }

    fn tier_of(app: &App) -> RenderQuality {
        app.world().resource::<EffectiveRenderSettings>().tier
    }

    const WINDOW: usize = 120;
    const HEALTHY: std::time::Duration = std::time::Duration::from_millis(8);

    /// A single catastrophic frame must not cost the user a quality tier.
    ///
    /// Window drags, display reconfiguration, resuming from sleep and debugger
    /// pauses all produce one enormous delta. If the governor reacted to the
    /// worst frame it would downgrade the renderer for something that already
    /// finished, and the user would see quality collapse for no visible reason.
    ///
    /// It survives because the decision is a p95 over a 120-sample window, and
    /// `((120-1)*95 + 50)/100` selects sorted index 113 — a single outlier sits
    /// at index 119 and cannot reach the statistic. This pins that arithmetic
    /// against a future change to the window size or the percentile.
    #[test]
    fn one_catastrophic_frame_per_window_never_downgrades_quality() {
        let mut app = governed_app(RenderQuality::High);
        assert_eq!(tier_of(&app), RenderQuality::High);

        // Three full windows, each carrying one ten-second stall.
        for _ in 0..3 {
            run_frames(&mut app, WINDOW - 1, HEALTHY);
            run_frames(&mut app, 1, std::time::Duration::from_secs(10));
        }

        assert_eq!(
            tier_of(&app),
            RenderQuality::High,
            "a lone stall per window must be absorbed by the p95, not acted on"
        );
    }

    /// The positive control, without which the test above proves nothing: a
    /// governor that never moves would also pass it.
    ///
    /// Sustained overload — every frame over the 16.6ms budget — must step the
    /// tier down once the required consecutive blowout windows have elapsed.
    #[test]
    fn sustained_overload_does_step_the_tier_down() {
        let mut app = governed_app(RenderQuality::High);
        let overloaded = std::time::Duration::from_millis(40);

        run_frames(
            &mut app,
            WINDOW * (RenderGovernor::BLOWOUT_WINDOWS_REQUIRED as usize + 1),
            overloaded,
        );

        assert_ne!(
            tier_of(&app),
            RenderQuality::High,
            "sustained frames at 40ms against a 16.6ms budget must downgrade"
        );
    }

    /// An explicitly requested tier is never touched, however bad frames get.
    #[test]
    fn an_explicit_tier_is_never_downgraded_by_sustained_overload() {
        let pinned = EffectiveRenderSettings {
            tier: RenderQuality::High,
            features: tier_features(RenderQuality::High),
            gpu: None,
            requested: RenderQuality::High,
        };
        let mut app = App::new();
        app.insert_resource(Time::<()>::default())
            .insert_resource(AdaptiveQualityGovernor::for_launch(&pinned))
            .insert_resource(pinned)
            .add_systems(Update, drive_adaptive_quality);

        run_frames(&mut app, WINDOW * 5, std::time::Duration::from_millis(40));

        assert_eq!(
            tier_of(&app),
            RenderQuality::High,
            "explicit quality disables adaptation; the operator's choice stands"
        );
    }

    /// The frame time this frontend feeds the governor is ALWAYS finite.
    ///
    /// The bead asks for NaN and outlier coverage. Outliers are covered above;
    /// NaN is covered by being unreachable from here rather than by a test that
    /// pretends otherwise. `Time::delta_secs` is derived from a `Duration`,
    /// which cannot represent NaN or a negative span, so `drive_adaptive_quality`
    /// cannot deliver a non-finite sample no matter how pathological the clock.
    /// `RenderGovernor::observe` still defends against one for other callers.
    #[test]
    fn the_frontend_cannot_deliver_a_non_finite_frame_time() {
        let mut app = governed_app(RenderQuality::High);
        for extreme in [
            std::time::Duration::ZERO,
            std::time::Duration::from_nanos(1),
            std::time::Duration::from_secs(86_400),
        ] {
            app.world_mut().resource_mut::<Time>().advance_by(extreme);
            let delta = app.world().resource::<Time>().delta_secs() * 1_000.0;
            assert!(
                delta.is_finite() && delta >= 0.0,
                "a Duration of {extreme:?} produced a non-finite frame time: {delta}"
            );
            app.update();
        }
    }
}

#[cfg(test)]
mod quality_tier_consumer_tests {
    use super::*;

    /// A fixed adapter so tier-consumer tests never depend on the host's GPU.
    fn gpu_info(name: &str, class: GpuClass) -> GpuInfo {
        GpuInfo {
            name: name.to_string(),
            backend: "Vulkan".to_string(),
            class,
            vram_bytes: None,
            max_texture_2d: Some(16_384),
            timestamp_queries: true,
            vendor_id: None,
            device_id: None,
            driver: None,
            driver_info: None,
        }
    }

    /// The defect this bead names, as an assertion: the resolved tier must be
    /// READ by a live system, not merely resolved, logged and inserted.
    ///
    /// Before bd-2z0.14.1.17 there was no `Res<EffectiveRenderSettings>`
    /// anywhere in the workspace, so Potato and Ultra rendered identically and
    /// nothing detected it.
    #[test]
    fn the_resolved_tier_has_a_live_reader() {
        let source = include_str!("lib.rs");
        let production = source
            .split_once("#[cfg(test)]\nmod quality_tier_consumer_tests")
            .map_or(source, |(before, _)| before);
        assert!(
            production.contains("Res<EffectiveRenderSettings>")
                || production.contains("ResMut<EffectiveRenderSettings>"),
            "the quality tier must be read by a live system; if this fails the \
             tier has gone inert again and Potato renders like Ultra"
        );
        assert!(
            production.contains("shadows_enabled: effective.features.shadows"),
            "the sun light must take its shadow state from the tier, not a literal"
        );
    }

    /// Build a control in a known non-default state, then apply the optimistic
    /// edit the playback handlers perform before submitting.
    ///
    /// The pre-state is deliberately NOT the default: if it were, a rollback
    /// test could pass because the restore happened to coincide with
    /// `SimControlData::default()` rather than because anything was restored.
    fn optimistically_edited_controls() -> (SimulationControl, SimControlSnapshot) {
        let controls = SimulationControl::new();
        controls.update(|state| {
            state.paused = true;
            state.speed_multiplier = 2.0;
            state.auto_pause_reason = Some("host stalled".to_string());
        });
        let before = controls.snapshot();
        controls.update(|state| {
            state.paused = false;
            state.speed_multiplier = 4.0;
            state.auto_pause_reason = None;
        });
        (controls, before)
    }

    fn submitter_answering(accepted: bool) -> CommandSubmitter {
        CommandSubmitter {
            submit: Arc::new(move |_| accepted.then(|| "test-cmd".to_owned())),
        }
    }

    /// A refused playback command must not leave the HUD showing it as applied.
    ///
    /// This drives the real submit path with a real refusing submitter rather
    /// than scanning source, so it fails if the rollback stops working for any
    /// reason — not only if the call is deleted.
    #[test]
    fn a_refused_playback_command_rolls_the_local_state_back() {
        let (controls, before) = optimistically_edited_controls();

        submit_playback_command(
            &submitter_answering(false),
            SimulationCommand::default(),
            &controls,
            &before,
        );

        let after = controls.snapshot();
        assert!(
            after.paused,
            "a refused command must not leave the HUD showing a running simulation"
        );
        assert!(
            (after.speed_multiplier - 2.0).abs() < f32::EPSILON,
            "speed must return to what the simulation actually has, got {}",
            after.speed_multiplier
        );
        assert_eq!(
            after.auto_pause_reason.as_deref(),
            Some("host stalled"),
            "the auto-pause reason still applies: the host never received the command \
             that would have cleared it"
        );
    }

    /// Positive control: an ACCEPTED command must keep the optimistic edit.
    ///
    /// Without this, the rollback test above would still pass if the code
    /// rolled back unconditionally — which would be a different bug wearing the
    /// same green checkmark.
    #[test]
    fn an_accepted_playback_command_keeps_the_local_edit() {
        let (controls, before) = optimistically_edited_controls();

        submit_playback_command(
            &submitter_answering(true),
            SimulationCommand::default(),
            &controls,
            &before,
        );

        let after = controls.snapshot();
        assert!(
            !after.paused,
            "an accepted command must keep the edit the operator asked for"
        );
        assert!(
            (after.speed_multiplier - 4.0).abs() < f32::EPSILON,
            "accepted speed change must stand, got {}",
            after.speed_multiplier
        );
        assert_eq!(
            after.auto_pause_reason, None,
            "an accepted resume clears the auto-pause reason"
        );
    }

    /// A rollback must not disturb `pending_steps`.
    ///
    /// The science driver consumes `pending_steps` concurrently, so a step
    /// banked between the snapshot and the refusal is not ours to erase.
    /// `SimControlSnapshot` omits the field for exactly this reason; this test
    /// pins that omission as load-bearing rather than incidental.
    #[test]
    fn rolling_back_leaves_concurrently_banked_steps_alone() {
        let (controls, before) = optimistically_edited_controls();
        // Stand in for the driver banking a step after `before` was taken.
        controls.update(|state| state.pending_steps = 3);

        submit_playback_command(
            &submitter_answering(false),
            SimulationCommand::default(),
            &controls,
            &before,
        );

        let pending = controls
            .0
            .lock()
            .expect("control mutex should not be poisoned in this test")
            .pending_steps;
        assert_eq!(
            pending, 3,
            "rollback erased a step the driver had already banked"
        );
    }

    /// Pointer input must resolve to the window its own camera renders to.
    ///
    /// This is the per-window half of bd-2z0.7.14. The failure it prevents is
    /// silent: with two windows, reading the cursor from `PrimaryWindow` while
    /// projecting through that cursor's unrelated camera picks an agent the
    /// user never clicked on, and every individual step succeeds, so nothing
    /// reports anything.
    #[test]
    fn pointer_input_resolves_to_the_cameras_own_window() {
        let primary = Entity::from_raw_u32(1).expect("entity index 1 is valid");
        let secondary = Entity::from_raw_u32(2).expect("entity index 2 is valid");

        assert_eq!(
            camera_target_window(&RenderTarget::Window(WindowRef::Primary), Some(primary)),
            Some(primary),
            "a primary-ref camera must follow whichever entity currently holds PrimaryWindow"
        );

        // THE REGRESSION THIS EXISTS TO CATCH: an explicit window ref must not
        // fall back to the primary window. That fallback is exactly what
        // projected a click in the second window through the first window's
        // cursor.
        assert_eq!(
            camera_target_window(
                &RenderTarget::Window(WindowRef::Entity(secondary)),
                Some(primary)
            ),
            Some(secondary),
            "a camera rendering to a secondary window must read that window's cursor, \
             not the primary window's"
        );

        // Absent is absent, not a guess.
        assert_eq!(
            camera_target_window(&RenderTarget::Window(WindowRef::Primary), None),
            None,
            "with no primary window there is no pointer window to fall back to"
        );
    }

    /// A camera that renders off-screen has no pointer window at all.
    ///
    /// The positive control is the test above: these same inputs DO resolve for
    /// window targets, so `None` here is a decision about off-screen targets
    /// rather than the function failing to resolve anything.
    #[test]
    fn offscreen_cameras_have_no_pointer_window() {
        let primary = Entity::from_raw_u32(1).expect("entity index 1 is valid");
        for target in [
            RenderTarget::TextureView(bevy::camera::ManualTextureViewHandle(0)),
            RenderTarget::None {
                size: UVec2::splat(4),
            },
        ] {
            assert_eq!(
                camera_target_window(&target, Some(primary)),
                None,
                "an off-screen target has no cursor, so it must not borrow the \
                 primary window's: {target:?}"
            );
        }
    }

    /// Every selection path must gate its camera commit on the enqueue result.
    ///
    /// The submitter returns a bool meaning ENQUEUED, not applied — a weaker
    /// fact than the UI was treating it as (bd-2z0.7.14). Three of the four
    /// selection paths gated on it; the clear-selection button DISCARDED it and
    /// committed the camera anyway, so a full command queue silently dropped the
    /// clear while the camera stopped following regardless. The user sees the
    /// camera obey a request the simulation never received.
    ///
    /// This asserts the discarding form is gone. It does NOT claim the remaining
    /// commits are correct: gating on enqueue is still optimistic, because a
    /// queued command can be rejected later, and making the UI wait for real
    /// application needs the receipt path this bead also asks for and which is
    /// not built yet. Truthful about a weaker guarantee beats silent about a
    /// false one.
    #[test]
    fn no_selection_path_commits_the_camera_on_a_discarded_enqueue() {
        let source = include_str!("lib.rs");
        // Assembled at runtime so this test cannot match its own literal — the
        // self-satisfying-needle trap that made the bloom guard vacuous.
        let discarded = format!("{}{}", "(submitter.submit)(command)", ";");
        assert!(
            !source.contains(&discarded),
            "a selection path is discarding the enqueue result and committing the \
             camera regardless; gate on it or propagate the failure"
        );
        // The gating form moved when the submitter began returning an
        // identity instead of a bool (bd-k7nq): a refusal is now `None`, so the
        // guard looks for the binding form rather than the negated one.
        assert!(
            source
                .matches("let Some(command_id) = (submitter.submit)(command)")
                .count()
                >= 1,
            "at least one path must refuse to commit when the enqueue fails"
        );
    }

    /// Selection logs must claim enqueueing, not application.
    ///
    /// "selection cleared" reads as a completed state change. All the code knows
    /// is that a command entered a queue, and a log that overstates what
    /// happened is how a dropped command becomes an unexplainable UI bug later.
    #[test]
    fn selection_logs_do_not_claim_application() {
        let source = include_str!("lib.rs");
        // Assembled, not written literally: a source-scanning test that spells
        // out its own needle finds itself and passes forever. Third time this
        // trap has come up in this file, so it is worth naming every time.
        let overclaim = format!("Bevy selection {} via", "cleared");
        assert!(
            !source.contains(&overclaim),
            "a log message claims the selection was applied when the code only \
             knows it was enqueued"
        );

        // The agent-selection path had the same defect one step earlier: these
        // fired during command construction, before the submitter was even
        // asked. Announcing a selection you have not yet tried to enqueue is a
        // stronger claim than announcing one you have.
        for premature in [
            format!("Bevy selection {}", "replace"),
            format!("Bevy selection {} -> add", "toggle"),
            format!("Bevy selection {} -> clear", "toggle"),
        ] {
            assert!(
                !source.contains(&premature),
                "a selection log fires before the submitter is asked: {premature:?}"
            );
        }
        assert!(
            source.contains("clear-selection enqueued via"),
            "the corrected wording must be present, or this test passes against \
             deleted logging rather than honest logging"
        );
    }

    /// A tier transition must not panic when the shadow-map resource is absent.
    ///
    /// This is the transition-FAILURE path. `DirectionalLightShadowMap` only
    /// exists where `PbrPlugin` is added, so a headless or partially-built app
    /// can reach `apply_tier_to_sun_light` without it. A plain `ResMut` would
    /// panic the whole schedule for a resource the renderer does not strictly
    /// need — turning a missing optional into a crash — so the system takes
    /// `Option<ResMut<..>>` and this pins that the rest of the transition still
    /// happens: the light is still reconfigured.
    #[test]
    fn a_tier_transition_survives_a_missing_shadow_map_resource() {
        let effective = EffectiveRenderSettings {
            tier: RenderQuality::Ultra,
            features: tier_features(RenderQuality::Ultra),
            gpu: None,
            requested: RenderQuality::Ultra,
        };
        let mut app = App::new();
        app.insert_resource(effective)
            .add_systems(Update, apply_tier_to_sun_light);
        // Deliberately NO DirectionalLightShadowMap inserted.
        let light = app
            .world_mut()
            .spawn((
                DirectionalLight {
                    shadows_enabled: false,
                    ..default()
                },
                TierDrivenSunLight,
            ))
            .id();

        app.update();

        let shadows_enabled = app
            .world()
            .get::<DirectionalLight>(light)
            .expect("the light must survive the transition")
            .shadows_enabled;
        assert!(
            shadows_enabled,
            "the Ultra tier must still switch shadows on even though the shadow-map \
             resource was absent; a missing optional must degrade, not abort"
        );
    }

    /// Resolution is restart-stable: the same inputs resolve identically.
    ///
    /// A "restart" for this renderer is a fresh resolve of the same settings
    /// against the same adapter. If that were not stable, a tier could differ
    /// between two launches of an unchanged configuration, and every golden and
    /// every performance comparison across runs would be silently untrustworthy.
    #[test]
    fn resolution_is_stable_across_restarts() {
        for (requested, class) in [
            (RenderQuality::Auto, GpuClass::Discrete),
            (RenderQuality::Ultra, GpuClass::Discrete),
            (RenderQuality::Auto, GpuClass::Integrated),
            (RenderQuality::Ultra, GpuClass::Software),
            (RenderQuality::Auto, GpuClass::Software),
        ] {
            let settings = RenderSettings {
                quality: Some(requested),
                ..RenderSettings::default()
            };
            let first = resolve_effective_render_settings_for_gpu(
                &settings,
                Some(gpu_info("adapter", class)),
            );
            let second = resolve_effective_render_settings_for_gpu(
                &settings,
                Some(gpu_info("adapter", class)),
            );

            assert_eq!(
                first.tier, second.tier,
                "requested {requested:?} on {class:?} must resolve to the same tier twice"
            );
            assert_eq!(first.requested, second.requested);
            assert_eq!(
                first.features.shadow_resolution, second.features.shadow_resolution,
                "the feature row must be stable too, not just the tier label"
            );
            assert_eq!(
                first.features.shadow_cascades,
                second.features.shadow_cascades
            );
            assert_eq!(first.features.bloom, second.features.bloom);
        }
    }

    /// The watermark must PERSIST across tier changes.
    ///
    /// It is spawned once from the adapter class and nothing may take it away,
    /// because the adapter cannot change mid-run — so a frame that ever deserved
    /// the banner deserves it for the whole session. The tier systems touch
    /// lights, the shadow map and the camera bloom; none of them may despawn or
    /// hide it, or a quality change would silently un-label a software render.
    #[test]
    fn no_tier_system_removes_the_software_renderer_watermark() {
        let source = include_str!("lib.rs");
        // Searched in CALL form, with the leading dot, precisely so this test
        // cannot match the names written in its own message below. Asserting on
        // the bare type name would find this function and pass forever — the
        // self-satisfying-needle trap that made the bloom guard vacuous until
        // 3b02a3cf246f.
        // The needle is ASSEMBLED at runtime so the complete string never
        // appears as a literal in this file. Writing it out — even in call form
        // with a leading dot — makes the test find itself and pass forever,
        // which is exactly how the bloom guard was vacuous until 3b02a3cf246f.
        // Two failed attempts here proved the point; only construction is safe
        // when a source-scanning test lives in the source it scans.
        let marker = "SoftwareRendererWatermark";
        for verb in [".despawn::<", ".remove::<"] {
            let forbidden = format!("{verb}{marker}>");
            assert!(
                !source.contains(&forbidden),
                "nothing may remove the watermark; the adapter cannot change mid-run, \
                 so removing it would un-label a software render that is still software"
            );
        }
    }

    /// The software-renderer watermark must be spawned, marked, and conditional.
    ///
    /// A startup warning is a log line, and a log line scrolls away — someone
    /// reading a screenshot, a recording or a bug report otherwise has no way to
    /// know the frame came from llvmpipe. Since the tier is already forced to
    /// Potato in that case, the picture is genuinely not representative and the
    /// FRAME has to carry that, not just the console.
    #[test]
    fn the_software_renderer_watermark_is_spawned_only_on_a_software_adapter() {
        let source = include_str!("lib.rs");
        let sites = |needle: &str| source.matches(needle).count();

        assert!(
            sites("SoftwareRendererWatermark") > 2,
            "the watermark needs a marker component, a spawn site, and this assertion"
        );
        assert!(
            sites("SOFTWARE RENDERER") > 1,
            "the banner text must exist in production, not only in this test"
        );
        assert!(
            source.contains(
                "info.class == GpuClass::Software\n        })\n    {\n        commands.spawn(("
            ) || source.contains("is_some_and(|info| info.class == GpuClass::Software)"),
            "the spawn must be gated on a software adapter; an unconditional banner \
             would libel every hardware run as unaccelerated"
        );
    }

    /// The watermark and the forced tier must agree.
    ///
    /// They are two expressions of one fact. If a future edit forced Potato
    /// without the banner, or banners without forcing, the renderer would be
    /// telling the operator two different stories about the same adapter.
    #[test]
    fn the_watermark_condition_matches_the_forced_potato_condition() {
        let software = gpu_info("llvmpipe", GpuClass::Software);
        let effective = resolve_effective_render_settings_for_gpu(
            &RenderSettings::default(),
            Some(software.clone()),
        );
        assert_eq!(
            effective.tier,
            RenderQuality::Potato,
            "the same adapter that earns a watermark must also be forced to Potato"
        );
        assert!(
            effective
                .gpu
                .as_ref()
                .is_some_and(|info| info.class == GpuClass::Software),
            "the resolved settings must retain the adapter class the watermark keys on"
        );

        let hardware = resolve_effective_render_settings_for_gpu(
            &RenderSettings::default(),
            Some(gpu_info("Apple M4", GpuClass::Discrete)),
        );
        assert!(
            hardware
                .gpu
                .as_ref()
                .is_some_and(|info| info.class != GpuClass::Software),
            "hardware must not satisfy the watermark condition"
        );
    }

    /// `features.shadow_cascades` must reach the cascade config.
    ///
    /// Cascade count controls perspective aliasing — near-camera shadows
    /// getting far fewer texels than distant ones — so a tier that moved
    /// resolution but not cascades still rendered Low and Ultra with the same
    /// blocky near field. Resolution and cascades are separate axes and both
    /// have to be live for the tier to mean what it says.
    #[test]
    fn the_shadow_cascade_tier_feature_has_a_live_consumer() {
        let source = include_str!("lib.rs");
        let sites = |needle: &str| source.matches(needle).count();
        assert!(
            sites("effective.features.shadow_cascades") > 1,
            "the cascade count must be read by production code, not only logged"
        );
        assert!(
            sites("CascadeShadowConfigBuilder {") > 1,
            "reading it is not enough; it must rebuild the CascadeShadowConfig"
        );
        assert!(
            sites("cascades.bounds.len() != num_cascades") > 1,
            "the rebuild must be conditional, or every tier evaluation churns the \
             cascade config for no change"
        );
    }

    /// The cascade ladder must vary, or the live consumer wires a constant.
    #[test]
    fn the_shadow_cascade_ladder_is_not_flat() {
        let cascades: Vec<u8> = [
            RenderQuality::Potato,
            RenderQuality::Low,
            RenderQuality::Medium,
            RenderQuality::High,
            RenderQuality::Ultra,
        ]
        .into_iter()
        .map(|tier| tier_features(tier).shadow_cascades)
        .collect();

        assert_eq!(cascades[0], 0, "Potato has no shadows, so no cascades");
        assert!(
            cascades.iter().skip(1).all(|count| *count > 0),
            "every shadowed tier must ask for at least one cascade, or the rebuild \
             is skipped and the config silently keeps the previous tier's count"
        );
        assert!(
            cascades.iter().skip(1).any(|count| *count != cascades[1]),
            "cascade count must vary across shadowed tiers, or Low and Ultra render \
             the same near-field aliasing"
        );
    }

    /// `features.shadow_resolution` must have a real consumer too.
    ///
    /// Shadows could already switch on and off with the tier, but every tier
    /// that had them rendered into the same 2048px map, so the tier changed
    /// shadow PRESENCE and never shadow COST. This pins that the resolution is
    /// read, that it reaches bevy's `DirectionalLightShadowMap`, and that the
    /// zero Potato reports is refused rather than written.
    #[test]
    fn the_shadow_resolution_tier_feature_has_a_live_consumer() {
        let source = include_str!("lib.rs");
        let sites = |needle: &str| source.matches(needle).count();
        assert!(
            sites("effective.features.shadow_resolution") > 1,
            "the shadow resolution must be read by production code, not only logged"
        );
        assert!(
            sites("shadow_map.size = requested;") > 1,
            "reading it is not enough; it must reach DirectionalLightShadowMap"
        );
        assert!(
            sites("if !effective.features.shadows {") > 1,
            "resolution must not be applied when shadows are off, or Potato's zero \
             reaches bevy and gets silently rounded to a power of two"
        );
    }

    /// The tiers must actually differ in shadow cost, or the consumer above is
    /// wiring a constant.
    ///
    /// A live consumer of a value that never changes still leaves Potato and
    /// Ultra rendering identical shadows, which is the defect one layer up.
    #[test]
    fn the_shadow_resolution_ladder_is_not_flat() {
        let resolutions: Vec<u32> = [
            RenderQuality::Potato,
            RenderQuality::Low,
            RenderQuality::Medium,
            RenderQuality::High,
            RenderQuality::Ultra,
        ]
        .into_iter()
        .map(|tier| tier_features(tier).shadow_resolution)
        .collect();

        assert_eq!(resolutions[0], 0, "Potato has no shadows, so no shadow map");
        for (tier, resolution) in resolutions.iter().enumerate().skip(1) {
            assert!(
                resolution.is_power_of_two(),
                "tier {tier} resolution {resolution} must be a power of two; bevy \
                 rounds anything else and the tier would not get what it asked for"
            );
        }
        assert!(
            resolutions.iter().skip(1).any(|r| *r != resolutions[1]),
            "shadow resolution must vary across tiers, or a live consumer still \
             renders Potato and Ultra identically"
        );
    }

    /// `features.bloom` must have a real consumer, not just a log line.
    ///
    /// Before bd-2z0.14.3.3 this crate never attached a `Bloom` component to any
    /// camera. The field was logged at startup and on every tier change, so the
    /// governor could announce Potato -> Ultra while the post stack stayed
    /// identical — a tier that reports moving without moving anything. Shadows
    /// were the only feature with a live consumer.
    #[test]
    fn the_bloom_tier_feature_has_a_live_consumer() {
        let source = include_str!("lib.rs");
        // Deliberately NOT the split-at-the-test-module trick the sibling test
        // uses. Production code in this file continues well past that module —
        // `apply_tier_to_bloom` lives ~1000 lines after it — so slicing at the
        // boundary silently hides the very consumer this asserts on, and the
        // test fails while the code is correct.
        //
        // Instead match the CALL form — leading dot, trailing semicolon — and
        // require MORE THAN ONE occurrence. Each needle necessarily matches the
        // string literal in the assertion that names it, so `contains` alone
        // would still hold after the production code was deleted: the test would
        // be quoting itself and calling that proof. Counting makes the extra
        // occurrence, the real call site, the thing being asserted.
        let sites = |needle: &str| source.matches(needle).count();
        assert!(
            sites("effective.features.bloom") > 1,
            "the bloom tier feature must be read by production code, not only logged"
        );
        assert!(
            sites(".insert(Bloom::NATURAL);") > 1,
            "reading the flag is not enough; a Bloom component must actually be attached"
        );
        assert!(
            sites(".remove::<Bloom>();") > 1,
            "a tier step DOWN must be able to take bloom away again, or the effect \
             is one-way and the tier still does not describe the frame"
        );
    }

    /// Cycling must always move, and must never land on Auto.
    ///
    /// Auto is a REQUEST meaning resolve-against-the-adapter, not a tier.
    /// Cycling into it would mean re-probing the GPU from an input handler.
    #[test]
    fn tier_cycle_covers_every_concrete_tier_and_never_yields_auto() {
        let mut seen = Vec::new();
        let mut tier = RenderQuality::Potato;
        for _ in 0..5 {
            seen.push(tier);
            assert_ne!(
                next_quality_tier(tier),
                RenderQuality::Auto,
                "cycling must never produce Auto"
            );
            tier = next_quality_tier(tier);
        }
        assert_eq!(tier, RenderQuality::Potato, "the cycle must wrap");
        for expected in [
            RenderQuality::Potato,
            RenderQuality::Low,
            RenderQuality::Medium,
            RenderQuality::High,
            RenderQuality::Ultra,
        ] {
            assert!(seen.contains(&expected), "cycle skipped {expected:?}");
        }
        // Auto is an accepted INPUT, and resolves somewhere concrete.
        assert_eq!(
            next_quality_tier(RenderQuality::Auto),
            RenderQuality::Potato
        );
    }

    /// The tier must actually change what the frame looks like, or the toggle
    /// is a label. Potato and Ultra must disagree about shadows.
    #[test]
    fn tier_change_visibly_changes_shadow_state() {
        assert!(
            !tier_features(RenderQuality::Potato).shadows,
            "Potato must disable shadows, else the wiring proves nothing"
        );
        assert!(
            tier_features(RenderQuality::Ultra).shadows,
            "Ultra must enable shadows"
        );
    }
}

#[cfg(test)]
mod hud_brain_overlay_tests {
    use super::*;
    use scriptbots_core::ScriptBotsConfig;

    fn ready_overlay() -> BrainOverlay {
        BrainOverlay {
            status: BrainOverlayStatus::Ready,
            source_tick: 42,
            layers: vec![
                BrainOverlayLayer {
                    name: "input".to_string(),
                    shown: vec![0.0, 0.5, 1.0],
                    total: 3,
                },
                BrainOverlayLayer {
                    name: "hidden".to_string(),
                    shown: vec![-1.0, 0.0],
                    total: 40,
                },
            ],
            truncated: true,
        }
    }

    /// THE contract for this bead: with nothing selected the overlay must
    /// report NotRequested, which is the state reached by returning BEFORE
    /// `inspect_brains` is ever called. No request means no brain inspected,
    /// which is what keeps the projection digest-neutral.
    #[test]
    fn no_selection_means_no_request_was_issued() {
        let overlay = BrainOverlay::default();
        assert_eq!(overlay.status, BrainOverlayStatus::NotRequested);
        assert!(overlay.layers.is_empty());
        assert_eq!(format_brain_overlay(&overlay), "Brain: select an agent");
    }

    /// A capture attempt must not consume a revision when there is nothing to
    /// ask about; otherwise an idle session would burn the revision space.
    #[test]
    fn revision_is_untouched_when_nothing_is_selected() {
        let world = WorldState::new(ScriptBotsConfig::default()).expect("world init");
        let mut revision = 7_u64;
        let overlay = BrainOverlay::capture(
            &world,
            &[],
            BrainInspectionClientId::new(BRAIN_OVERLAY_CLIENT_ID),
            &mut revision,
        );
        assert_eq!(overlay.status, BrainOverlayStatus::NotRequested);
        assert_eq!(revision, 7, "no request should consume no revision");
    }

    /// Clipping must be visible. A truncated view that does not say so lets a
    /// reader conclude a brain has no deep structure when we simply refused to
    /// copy it — core documents that same reasoning on `BrainActivations`.
    #[test]
    fn clipped_payload_is_announced() {
        assert!(format_brain_overlay(&ready_overlay()).contains("clipped"));
        let mut clean = ready_overlay();
        clean.truncated = false;
        assert!(!format_brain_overlay(&clean).contains("clipped"));
    }

    /// Each layer reports shown-vs-total so a reader can tell the panel is a
    /// window onto a larger layer, not the whole thing.
    #[test]
    fn layers_report_shown_and_total_counts() {
        let panel = format_brain_overlay(&ready_overlay());
        assert!(panel.contains("input [3/3]"), "panel was {panel}");
        assert!(panel.contains("hidden [2/40]"), "panel was {panel}");
        assert!(panel.contains("tick 42"), "panel was {panel}");
    }

    /// Typed refusals and missing identities are surfaced rather than being
    /// rendered as an empty brain.
    #[test]
    fn refusals_are_surfaced_not_blanked() {
        let unavailable = BrainOverlay {
            status: BrainOverlayStatus::Unavailable,
            source_tick: 9,
            ..BrainOverlay::default()
        };
        assert!(format_brain_overlay(&unavailable).contains("unavailable"));
        let no_uid = BrainOverlay {
            status: BrainOverlayStatus::NoStableIdentity,
            ..BrainOverlay::default()
        };
        assert!(format_brain_overlay(&no_uid).contains("no stable identity"));
    }

    /// Activations are signed; scaling by peak absolute magnitude keeps a
    /// mostly-negative layer legible instead of collapsing it onto the floor.
    #[test]
    fn signed_activations_scale_by_absolute_peak() {
        let line = format_sparkline_signed(&[0.0, -1.0, 0.5]);
        assert_eq!(line.chars().count(), 3);
        assert!(
            line.starts_with('▁'),
            "zero should sit at the floor: {line}"
        );
        assert!(
            line.contains('█'),
            "peak magnitude should reach full height even when negative: {line}"
        );
        assert_eq!(format_sparkline_signed(&[0.0, 0.0]), "▁▁");
        assert_eq!(format_sparkline_signed(&[]), "");
    }
}

#[cfg(test)]
mod hud_history_sparkline_tests {
    use super::*;

    fn summary(tick: u64, agents: usize, births: usize, deaths: usize) -> TickSummary {
        TickSummary {
            tick: scriptbots_core::Tick(tick),
            agent_count: agents,
            births,
            deaths,
            total_energy: 0.0,
            average_energy: 0.0,
            average_health: 0.0,
            max_age: 0,
            spike_hits: 0,
        }
    }

    /// Mirrors GPUI's `from_entries` returning `None` below two entries: one
    /// point is not a trend, and drawing it would imply one.
    #[test]
    fn fewer_than_two_samples_yields_no_history() {
        assert!(HudHistory::from_history([].iter()).is_empty());
        let one = [summary(1, 10, 0, 0)];
        assert!(HudHistory::from_history(one.iter()).is_empty());
        let two = [summary(1, 10, 0, 0), summary(2, 11, 1, 0)];
        assert!(!HudHistory::from_history(two.iter()).is_empty());
    }

    /// Stride decimation caps every series at the sample budget regardless of
    /// how long the run has been going.
    #[test]
    fn long_history_is_decimated_to_the_budget() {
        let long: Vec<TickSummary> = (0..5000).map(|t| summary(t, 100, 1, 1)).collect();
        let h = HudHistory::from_history(long.iter());
        assert!(
            h.agents.len() <= HUD_SPARKLINE_SAMPLES,
            "agents series was {}",
            h.agents.len()
        );
        assert_eq!(h.agents.len(), h.births.len());
        assert_eq!(h.agents.len(), h.deaths.len());
    }

    /// Each series scales by its OWN maximum. Births in single digits beside a
    /// population in the hundreds must still show shape, not a flat line.
    #[test]
    fn series_scale_independently() {
        let entries: Vec<TickSummary> = vec![
            summary(1, 500, 0, 0),
            summary(2, 500, 4, 0),
            summary(3, 500, 8, 0),
        ];
        let h = HudHistory::from_history(entries.iter());
        let births = format_sparkline(&h.births);
        assert_eq!(births.chars().count(), 3);
        assert!(
            births.contains('█'),
            "peak births must reach full height, got {births}"
        );
        assert!(
            births.starts_with('▁'),
            "zero births must sit at the floor, got {births}"
        );
    }

    /// A flat series renders at the floor rather than dividing by zero.
    #[test]
    fn all_zero_series_renders_at_the_floor() {
        assert_eq!(format_sparkline(&[0, 0, 0]), "▁▁▁");
        assert_eq!(format_sparkline(&[]), "");
    }

    /// The panel names each series and reports its peak, so a reader can tell
    /// what the glyph heights are relative to.
    #[test]
    fn panel_labels_series_and_peaks() {
        let entries: Vec<TickSummary> = vec![summary(1, 10, 1, 2), summary(2, 20, 3, 4)];
        let panel = format_history_panel(&HudHistory::from_history(entries.iter()));
        for expected in ["Agents", "Births", "Deaths", "peak 20", "peak 3", "peak 4"] {
            assert!(panel.contains(expected), "missing {expected} in: {panel}");
        }
    }

    /// With nothing to draw the panel keeps a stable placeholder rather than
    /// unmounting — same lesson as bd-rzy3.
    #[test]
    fn empty_history_renders_a_stable_placeholder() {
        assert_eq!(
            format_history_panel(&HudHistory::default()),
            "History: collecting…"
        );
    }
}

#[cfg(test)]
mod hud_event_feed_tests {
    use super::*;

    fn summary(tick: u64, births: usize, deaths: usize, spike_hits: u32) -> TickSummary {
        TickSummary {
            tick: scriptbots_core::Tick(tick),
            agent_count: 10,
            births,
            deaths,
            total_energy: 0.0,
            average_energy: 0.0,
            average_health: 0.0,
            max_age: 0,
            spike_hits,
        }
    }

    /// History is oldest-first; the feed must read newest-first so the most
    /// recent thing that happened is the first thing shown.
    #[test]
    fn feed_is_newest_first() {
        let history = [
            summary(1, 1, 0, 0),
            summary(2, 0, 1, 0),
            summary(3, 0, 0, 2),
        ];
        let events = HudEvent::recent_from_history(history.iter());
        let ticks: Vec<u64> = events.iter().map(|e| e.tick).collect();
        assert_eq!(ticks, vec![3, 2, 1], "newest tick must come first");
        assert_eq!(events[0].kind, HudEventKind::SpikeHit);
        assert_eq!(events[0].count, 2);
    }

    /// A tick with no births, deaths or spike hits contributes nothing, so
    /// quiet ticks cannot push real events out of a bounded feed.
    #[test]
    fn quiet_ticks_produce_no_entries() {
        let history = [summary(1, 0, 0, 0), summary(2, 0, 0, 0)];
        assert!(HudEvent::recent_from_history(history.iter()).is_empty());
    }

    /// Within one tick the order is births, deaths, then spike hits, so a busy
    /// tick always reads the same way.
    #[test]
    fn one_tick_orders_births_deaths_then_spikes() {
        let history = [summary(7, 3, 2, 1)];
        let events = HudEvent::recent_from_history(history.iter());
        let kinds: Vec<HudEventKind> = events.iter().map(|e| e.kind).collect();
        assert_eq!(
            kinds,
            vec![
                HudEventKind::Birth,
                HudEventKind::Death,
                HudEventKind::SpikeHit
            ]
        );
        assert!(events.iter().all(|e| e.tick == 7));
    }

    /// The bound is applied while deriving, not after: a births/deaths storm
    /// over a long history must not build a large vector and then trim it.
    #[test]
    fn feed_is_bounded_under_a_storm() {
        let history: Vec<TickSummary> = (0..500).map(|t| summary(t, 9, 9, 9)).collect();
        let events = HudEvent::recent_from_history(history.iter());
        assert_eq!(events.len(), HUD_EVENT_FEED_CAPACITY);
        assert_eq!(
            events[0].tick, 499,
            "the newest tick must survive the bound"
        );
    }

    /// An empty feed says so rather than rendering a bare label.
    #[test]
    fn empty_feed_renders_a_placeholder() {
        assert_eq!(format_event_feed(&[]), "Events: none yet");
    }

    /// Every entry reaches the rendered line with its glyph, count and tick.
    #[test]
    fn formatted_line_carries_glyph_count_and_tick() {
        let events = HudEvent::recent_from_history([summary(12, 2, 1, 0)].iter());
        let line = format_event_feed(&events);
        assert!(line.starts_with("Events:"), "line was {line}");
        assert!(line.contains("✚ 2×born@12"), "line was {line}");
        assert!(line.contains("✖ 1×died@12"), "line was {line}");
    }
}

/// The primary selection's vitals, projected for the inspector overlay.
///
/// Field vocabulary deliberately mirrors GPUI's inspector detail (`energy`,
/// `health`, `age`, `generation`, `spike_length`, trait modifiers) so the two
/// frontends describe the same agent the same way rather than inventing a
/// second vocabulary (bd-2z0.14.1.14).
#[derive(Debug, Clone, Copy, PartialEq)]
struct InspectorDetail {
    id: AgentId,
    energy: f32,
    health: f32,
    age: u32,
    generation: u32,
    herbivore_tendency: f32,
    temperature_preference: f32,
    spike_length: f32,
    boost: f32,
    reproduction_intent: f32,
    spiked: bool,
    trait_modifiers: TraitModifiers,
    also_selected: usize,
}

impl InspectorDetail {
    fn from_agent(agent: &AgentVisual, also_selected: usize) -> Self {
        Self {
            id: agent.id,
            energy: agent.energy,
            health: agent.health,
            age: agent.age,
            generation: agent.generation,
            herbivore_tendency: agent.herbivore_tendency,
            temperature_preference: agent.temperature_preference,
            spike_length: agent.spike_length,
            boost: agent.boost,
            reproduction_intent: agent.reproduction_intent,
            spiked: agent.spiked,
            trait_modifiers: agent.trait_modifiers,
            also_selected,
        }
    }

    /// Diet label from the same herbivore/carnivore tendency the renderer
    /// colours by, so the word and the colour cannot disagree.
    const fn diet(&self) -> &'static str {
        if self.herbivore_tendency >= 0.66 {
            "herbivore"
        } else if self.herbivore_tendency <= 0.33 {
            "carnivore"
        } else {
            "omnivore"
        }
    }
}

/// Render the inspector overlay as a multi-line block (bd-2z0.14.1.14).
///
/// Returns the placeholder when nothing is selected, so the panel keeps a
/// stable presence in the HUD instead of appearing and vanishing.
fn format_inspector(detail: Option<InspectorDetail>) -> String {
    let Some(d) = detail else {
        return "Inspector: no selection".to_string();
    };
    let extra = if d.also_selected > 0 {
        format!(" (+{} more selected)", d.also_selected)
    } else {
        String::new()
    };
    let spike = if d.spiked {
        format!("{:.2} HIT", d.spike_length)
    } else {
        format!("{:.2}", d.spike_length)
    };
    let t = &d.trait_modifiers;
    format!(
        "Inspector: {:?}{}\n  Energy {:>5.2} • Health {:>5.2} • Age {:>5} • Gen {}\n  \
         Diet {} ({:.2}) • Temp pref {:.2} • Repro {:.2}\n  \
         Spike {} • Boost {:.2}\n  \
         Traits smell {:.2} sound {:.2} hearing {:.2} eye {:.2} blood {:.2}",
        d.id,
        extra,
        d.energy,
        d.health,
        d.age,
        d.generation,
        d.diet(),
        d.herbivore_tendency,
        d.temperature_preference,
        d.reproduction_intent,
        spike,
        d.boost,
        t.smell,
        t.sound,
        t.hearing,
        t.eye,
        t.blood
    )
}

/// Render the newest-first event feed as one HUD line (bd-2z0.14.1.13).
///
/// Entries are already bounded by [`HUD_EVENT_FEED_CAPACITY`] at derivation,
/// so this only formats; it never truncates a list it was handed.
fn format_event_feed(events: &[HudEvent]) -> String {
    if events.is_empty() {
        return "Events: none yet".to_string();
    }
    let mut line = String::from("Events:");
    for event in events {
        line.push_str(&format!(
            " {} {}×{}@{}",
            event.kind.glyph(),
            event.count,
            event.kind.label(),
            event.tick
        ));
    }
    line
}

/// Resolve which window a camera's pointer input belongs to.
///
/// Pointer-driven selection used to read the cursor from whichever window
/// carried [`PrimaryWindow`] while projecting that cursor through
/// [`PrimaryCamera`]. With a single window those are the same window and the
/// bug is invisible. With two, a click in the second window is projected
/// through the FIRST window's cursor position, so the pick lands somewhere the
/// user never clicked — and silently, because every step succeeds
/// (bd-2z0.7.14).
///
/// Returns `None` when the camera does not render to a window at all. An image
/// target, a manual texture view and a disabled target have no cursor, so the
/// honest answer is "no window" rather than a fallback to the primary one.
/// Falling back is precisely what produced the wrong-window pick.
fn camera_target_window(target: &RenderTarget, primary: Option<Entity>) -> Option<Entity> {
    match target {
        RenderTarget::Window(WindowRef::Primary) => primary,
        RenderTarget::Window(WindowRef::Entity(window)) => Some(*window),
        RenderTarget::Image(_) | RenderTarget::TextureView(_) | RenderTarget::None { .. } => None,
    }
}

fn handle_selection_input(
    (buttons, keys): (Res<ButtonInput<MouseButton>>, Res<ButtonInput<KeyCode>>),
    windows: Query<&Window>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<PrimaryCamera>>,
    state: Res<SnapshotState>,
    submitter: Option<Res<CommandSubmitter>>,
    mut rig: ResMut<CameraRig>,
) {
    let Some(submitter) = submitter else {
        return;
    };

    if keys.just_pressed(KeyCode::Escape) {
        let command = ControlCommand::UpdateSelection(SelectionUpdate {
            mode: SelectionMode::Clear,
            agent_ids: Vec::new(),
            state: SelectionState::Selected,
        });
        if let Some(command_id) = (submitter.submit)(command) {
            info!(%command_id, "Bevy clear-selection enqueued via Escape");
            rig.follow_mode = FollowMode::Off;
            rig.pan = Vec2::ZERO;
            rig.recenter_now = true;
        }
        return;
    }

    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let snapshot = match state.latest.as_ref() {
        Some(snapshot) => snapshot,
        None => return,
    };

    let Ok((camera, transform)) = camera_query.single() else {
        return;
    };

    // Read the cursor from the window this camera actually renders to, not from
    // whichever window happens to hold `PrimaryWindow` (bd-2z0.7.14). The camera
    // is resolved FIRST because it is what names the window; doing it the other
    // way round is what let the two drift apart.
    let Some(target_window) = camera_target_window(&camera.target, primary_window.single().ok())
    else {
        return;
    };
    let Some(cursor_pos) = windows
        .get(target_window)
        .ok()
        .and_then(|window| window.cursor_position())
    else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(transform, cursor_pos) else {
        return;
    };

    let dir_y = ray.direction.y;
    if dir_y.abs() <= f32::EPSILON {
        return;
    }
    let distance = -ray.origin.y / dir_y;
    if distance <= 0.0 {
        return;
    }
    let impact = ray.origin + ray.direction * distance;

    let world_size = state.world_size;
    if world_size.x <= 0.0 || world_size.y <= 0.0 {
        return;
    }

    let world_point = Vec2::new(impact.x + world_size.x * 0.5, world_size.y * 0.5 - impact.z);

    let selection_radius = (snapshot.agent_radius * 3.0).max(24.0);
    let radius_sq = selection_radius * selection_radius;

    let mut best: Option<&AgentVisual> = None;
    let mut best_dist = f32::MAX;

    for agent in &snapshot.agents {
        // Argument order swapped versus the removed private copy: that computed
        // target - origin, core computes a - b. Both feed dist_sq below, so the
        // sign is unused, but the swap keeps the value identical rather than
        // relying on that (bd-ikts.4).
        let dx = toroidal_delta(agent.position.x, world_point.x, world_size.x);
        let dy = toroidal_delta(agent.position.y, world_point.y, world_size.y);
        let dist_sq = dx.mul_add(dx, dy * dy);
        if dist_sq <= radius_sq && dist_sq < best_dist {
            best_dist = dist_sq;
            best = Some(agent);
        }
    }

    let extend = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    if let Some(agent) = best {
        let agent_id = encode_agent_id(agent.id);
        // The intent is NAMED here but not announced. These logs used to fire
        // during command construction, before the submitter had been asked, so
        // the log claimed a selection change that a full queue then dropped
        // (bd-2z0.7.14). Deciding and reporting are now separate steps.
        let (mode, intent) = if extend {
            if matches!(agent.selection, SelectionState::Selected) {
                (SelectionMode::Clear, "toggle -> clear")
            } else {
                (SelectionMode::Add, "toggle -> add")
            }
        } else {
            (SelectionMode::Replace, "replace")
        };
        let command = ControlCommand::UpdateSelection(SelectionUpdate {
            mode,
            agent_ids: vec![agent_id],
            state: SelectionState::Selected,
        });

        let Some(command_id) = (submitter.submit)(command) else {
            // Previously this failure was entirely silent: the result was
            // consumed by `&& !extend`, so a refused shift-click reported
            // nothing at all and a refused plain click only skipped the camera.
            warn!(
                agent_id,
                intent, "selection command could not be enqueued; selection unchanged"
            );
            return;
        };
        info!(agent_id, intent, %command_id, "Bevy selection enqueued");
        if !extend {
            rig.follow_mode = FollowMode::Selected;
            rig.pan = Vec2::ZERO;
            rig.recenter_now = true;
        }
    } else if !extend {
        let command = ControlCommand::UpdateSelection(SelectionUpdate {
            mode: SelectionMode::Clear,
            agent_ids: Vec::new(),
            state: SelectionState::Selected,
        });
        if let Some(command_id) = (submitter.submit)(command) {
            info!(%command_id, "Bevy clear-selection enqueued via empty click");
            rig.follow_mode = FollowMode::Off;
            rig.pan = Vec2::ZERO;
            rig.recenter_now = true;
        }
    }
}

fn handle_playback_buttons(
    controls: Res<SimulationControl>,
    submitter: Option<Res<CommandSubmitter>>,
    mut query: Query<(&PlaybackButton, &Interaction), ChangedButtonFilter>,
) {
    let command_is_authoritative = submitter.is_some();
    for (button, interaction) in query.iter_mut() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        // Captured before the optimistic edit below, so a refusal has something
        // truthful to restore.
        let before = controls.snapshot();
        let mut command_to_send: Option<SimulationCommand> = None;
        controls.update(|state| {
            let mut command = SimulationCommand::default();
            match button.action {
                PlaybackAction::Play => {
                    state.paused = false;
                    if state.speed_multiplier <= MIN_SPEED {
                        state.speed_multiplier = 1.0;
                    }
                    state.pending_steps = 0;
                    state.auto_pause_reason = None;
                    command.paused = Some(false);
                    command.speed_multiplier = Some(state.speed_multiplier);
                    info!("Bevy playback: resume");
                }
                PlaybackAction::Pause => {
                    state.paused = true;
                    state.pending_steps = 0;
                    command.paused = Some(true);
                    info!("Bevy playback: pause");
                }
                PlaybackAction::Step => {
                    // With a submitter, the drained command is the sole science
                    // authority. Also setting the local edge made Step advance
                    // once or twice depending on queue/driver interleaving.
                    if !command_is_authoritative {
                        enqueue_step_request(state);
                    }
                    state.auto_pause_reason = None;
                    state.paused = true;
                    command.paused = Some(true);
                    command.step_once = true;
                    info!("Bevy playback: step once");
                }
                PlaybackAction::SpeedDown => {
                    state.speed_multiplier = (state.speed_multiplier - SPEED_STEP).max(MIN_SPEED);
                    if state.speed_multiplier <= MIN_SPEED {
                        state.speed_multiplier = 0.0;
                        state.paused = true;
                        info!("Bevy playback: speed set to 0.0 (paused)");
                    } else {
                        state.paused = false;
                        info!(
                            "Bevy playback: speed decreased to {:.1}",
                            state.speed_multiplier
                        );
                    }
                    state.auto_pause_reason = None;
                    command.speed_multiplier = Some(state.speed_multiplier);
                    command.paused = Some(state.paused);
                }
                PlaybackAction::SpeedUp => {
                    state.speed_multiplier =
                        (state.speed_multiplier + SPEED_STEP).clamp(SPEED_STEP, MAX_SPEED);
                    state.paused = false;
                    state.auto_pause_reason = None;
                    command.speed_multiplier = Some(state.speed_multiplier);
                    command.paused = Some(false);
                    info!(
                        "Bevy playback: speed increased to {:.1}",
                        state.speed_multiplier
                    );
                }
            }
            command_to_send = Some(command);
        });

        if let (Some(submitter), Some(command)) = (submitter.as_ref(), command_to_send) {
            if command.step_once {
                // Step keeps its own repair and is deliberately NOT rolled back.
                // A rejected step cannot simply disappear: it falls back to the
                // local driver edge, and `paused = true` is part of stepping
                // rather than an optimistic edit to undo.
                if !submit_simulation_command(submitter, command) {
                    controls.update(enqueue_step_request);
                }
            } else {
                submit_playback_command(submitter, command, &controls, &before);
            }
        }
    }
}

fn handle_playback_shortcuts(
    keys: Res<ButtonInput<KeyCode>>,
    controls: Res<SimulationControl>,
    submitter: Option<Res<CommandSubmitter>>,
) {
    let command_is_authoritative = submitter.is_some();
    if keys.just_pressed(KeyCode::Space) {
        let before = controls.snapshot();
        let mut command = SimulationCommand::default();
        controls.update(|state| {
            state.paused = !state.paused;
            if !state.paused && state.speed_multiplier <= MIN_SPEED {
                state.speed_multiplier = 1.0;
            }
            state.pending_steps = 0;
            state.auto_pause_reason = None;
            info!(paused = state.paused, "Bevy playback toggled via Space");
            command.paused = Some(state.paused);
            command.speed_multiplier = Some(state.speed_multiplier);
        });
        if let Some(submitter) = submitter.as_ref() {
            submit_playback_command(submitter, command, &controls, &before);
        }
    }

    if keys.just_pressed(KeyCode::KeyN) {
        let mut command = SimulationCommand::default();
        controls.update(|state| {
            if !command_is_authoritative {
                enqueue_step_request(state);
            }
            state.paused = true;
            state.auto_pause_reason = None;
            info!("Bevy playback: step requested via keyboard");
            command.paused = Some(true);
            command.step_once = true;
        });
        if let (Some(submitter), Some(command)) = (submitter.as_ref(), Some(command))
            && !submit_simulation_command(submitter, command)
        {
            controls.update(enqueue_step_request);
        }
    }

    if keys.just_pressed(KeyCode::Equal) || keys.just_pressed(KeyCode::NumpadAdd) {
        let before = controls.snapshot();
        let mut command = SimulationCommand::default();
        controls.update(|state| {
            state.speed_multiplier =
                (state.speed_multiplier + SPEED_STEP).clamp(SPEED_STEP, MAX_SPEED);
            state.paused = false;
            state.auto_pause_reason = None;
            info!(
                "Bevy playback: speed increased to {:.1} via keyboard",
                state.speed_multiplier
            );
            command.speed_multiplier = Some(state.speed_multiplier);
            command.paused = Some(false);
        });
        if let Some(submitter) = submitter.as_ref() {
            submit_playback_command(submitter, command, &controls, &before);
        }
    }

    if keys.just_pressed(KeyCode::Minus) || keys.just_pressed(KeyCode::NumpadSubtract) {
        let before = controls.snapshot();
        let mut command = SimulationCommand::default();
        controls.update(|state| {
            state.speed_multiplier = (state.speed_multiplier - SPEED_STEP).max(MIN_SPEED);
            if state.speed_multiplier <= MIN_SPEED {
                state.speed_multiplier = 0.0;
                state.paused = true;
                info!("Bevy playback: speed decreased to 0.0 (paused) via keyboard");
            } else {
                state.paused = false;
                info!(
                    "Bevy playback: speed decreased to {:.1} via keyboard",
                    state.speed_multiplier
                );
            }
            state.auto_pause_reason = None;
            command.speed_multiplier = Some(state.speed_multiplier);
            command.paused = Some(state.paused);
        });
        if let Some(submitter) = submitter.as_ref() {
            submit_playback_command(submitter, command, &controls, &before);
        }
    }
}

fn update_playback_button_colors(
    controls: Res<SimulationControl>,
    mut query: Query<(&PlaybackButton, &Interaction, &mut BackgroundColor)>,
) {
    let snapshot = controls.snapshot();
    for (button, interaction, mut color) in query.iter_mut() {
        let highlight = match button.action {
            PlaybackAction::Play => !snapshot.paused,
            PlaybackAction::Pause => snapshot.paused,
            _ => false,
        };

        let target = if highlight {
            follow_active_color()
        } else if matches!(interaction, Interaction::Hovered | Interaction::Pressed) {
            follow_hover_color()
        } else {
            follow_idle_color()
        };
        *color = target.into();
    }
}
fn handle_follow_button_interactions(
    mut rig: ResMut<CameraRig>,
    mut query: Query<(&FollowButton, &Interaction), ChangedButtonFilter>,
) {
    for (button, interaction) in query.iter_mut() {
        if *interaction == Interaction::Pressed {
            info!(mode = ?button.mode, "Bevy follow button pressed");
            rig.toggle_follow_mode(button.mode);
        }
    }
}

fn handle_clear_selection_button(
    submitter: Option<Res<CommandSubmitter>>,
    mut rig: ResMut<CameraRig>,
    mut buttons: Query<&Interaction, (Changed<Interaction>, With<ClearSelectionButton>)>,
) {
    let Some(submitter) = submitter else {
        return;
    };
    for interaction in buttons.iter_mut() {
        if *interaction == Interaction::Pressed {
            let command = ControlCommand::UpdateSelection(SelectionUpdate {
                mode: SelectionMode::Clear,
                agent_ids: Vec::new(),
                state: SelectionState::Selected,
            });
            // The return value is CHECKED here (bd-2z0.7.14). It used to be
            // discarded, so a full command queue silently dropped the clear
            // while the camera below still stopped following — the UI acting on
            // a request the simulation never received. The other three
            // selection paths already gated on it; this one did not, which made
            // the failure mode inconsistent as well as invisible.
            // `continue`, not `return`: the loop body is the unit of work. This
            // bead is making interaction per-window, so more than one clear
            // button will exist, and bailing out of the whole system would drop
            // the other windows' presses on the floor — the same class of silent
            // loss this fix exists to remove.
            let Some(command_id) = (submitter.submit)(command) else {
                warn!("clear-selection command could not be enqueued; leaving the camera as it is");
                continue;
            };
            info!(%command_id, "Bevy clear selection enqueued");
            rig.follow_mode = FollowMode::Off;
            rig.pan = Vec2::ZERO;
            rig.recenter_now = true;
        }
    }
}

/// Advance the quality tier through the concrete tiers, skipping Auto.
///
/// Auto is a REQUEST, not a tier: it means "resolve against the probed
/// adapter". Cycling into it at runtime would mean re-probing the GPU from an
/// input handler, so the cycle stays over concrete tiers only (bd-2z0.14.1.17).
const fn next_quality_tier(tier: RenderQuality) -> RenderQuality {
    match tier {
        RenderQuality::Potato => RenderQuality::Low,
        RenderQuality::Low => RenderQuality::Medium,
        RenderQuality::Medium => RenderQuality::High,
        RenderQuality::High => RenderQuality::Ultra,
        // Ultra wraps, and Auto resolves to the low end so a click always moves.
        RenderQuality::Ultra | RenderQuality::Auto => RenderQuality::Potato,
    }
}

fn handle_quality_tier_button(
    mut effective: ResMut<EffectiveRenderSettings>,
    mut adaptive: ResMut<AdaptiveQualityGovernor>,
    mut query: Query<&Interaction, (Changed<Interaction>, With<QualityTierButton>)>,
) {
    for interaction in &mut query {
        if *interaction == Interaction::Pressed {
            // Taking the tier by hand is an explicit choice, so the governor
            // stands down permanently. Without this the next completed window
            // would silently overwrite what the operator just picked.
            adaptive.relinquish_to_operator();
            let next = next_quality_tier(effective.tier);
            effective.tier = next;
            effective.features = tier_features(next);
            info!(
                tier = ?next,
                shadows = effective.features.shadows,
                ssao = effective.features.ssao,
                bloom = effective.features.bloom,
                "Bevy quality tier changed at runtime"
            );
        }
    }
}

/// Re-apply the tier to the sun light whenever the tier changes.
///
/// This is what makes the toggle more than a label: without it the tier would
/// still be startup-only, which is the defect this bead names.
fn apply_tier_to_sun_light(
    effective: Res<EffectiveRenderSettings>,
    shadow_map: Option<ResMut<DirectionalLightShadowMap>>,
    mut lights: Query<(&mut DirectionalLight, &mut CascadeShadowConfig), With<TierDrivenSunLight>>,
) {
    if !effective.is_changed() {
        return;
    }
    for (mut light, mut cascades) in &mut lights {
        light.shadows_enabled = effective.features.shadows;

        // `shadow_cascades` was the third logged-but-unconsumed tier feature.
        // Cascade COUNT is what controls perspective aliasing — shadows near
        // the camera getting far fewer texels than distant ones — so a tier
        // that moved resolution but not cascades still rendered Low and Ultra
        // with the same blocky near-field (bd-2z0.14.3.3 item 2).
        //
        // Only `num_cascades` is overridden. `CascadeShadowConfig::default()`
        // IS `CascadeShadowConfigBuilder::default().into()`, so rebuilding from
        // the builder with everything else defaulted changes the cascade count
        // and nothing about the near/far distances — this must not quietly
        // become a shadow-range change while claiming to be a quality one.
        if effective.features.shadows && effective.features.shadow_cascades > 0 {
            let num_cascades = usize::from(effective.features.shadow_cascades);
            if cascades.bounds.len() != num_cascades {
                *cascades = CascadeShadowConfigBuilder {
                    num_cascades,
                    ..default()
                }
                .build();
            }
        }
    }

    // `shadow_resolution` was the second tier feature that existed only in logs
    // (bd-2z0.14.3.3 item 2). Shadows could switch on and off with the tier, but
    // Potato and Ultra rendered them into the same 2048px map, so the tier
    // changed shadow PRESENCE and never shadow COST — which is most of what the
    // setting is for.
    //
    // Skipped entirely when shadows are off: the tier reports resolution 0 for
    // Potato, and bevy requires a power of two, so writing 0 would trip its own
    // validator and get silently rounded rather than respected.
    let Some(mut shadow_map) = shadow_map else {
        // Absent only in test apps built without PbrPlugin. Missing the resource
        // is not a reason to panic a renderer that is otherwise fine.
        return;
    };
    if !effective.features.shadows {
        return;
    }
    let requested = effective.features.shadow_resolution as usize;
    if requested == 0 || !requested.is_power_of_two() {
        warn!(
            requested,
            tier = ?effective.tier,
            "tier shadow resolution is not a usable shadow-map size; leaving the \
             current map untouched rather than letting bevy silently round it"
        );
        return;
    }
    // Written only on an actual change: a blind write per tier evaluation would
    // mark the resource changed forever and force shadow-map reallocation work
    // that nothing asked for.
    if shadow_map.size != requested {
        info!(
            previous = shadow_map.size,
            size = requested,
            tier = ?effective.tier,
            "directional shadow map resized to match the quality tier"
        );
        shadow_map.size = requested;
    }
}

/// Re-apply the tier's bloom decision to the primary camera whenever the tier
/// changes.
///
/// `features.bloom` was pure decoration before this: it was logged at startup
/// and again on every tier change, and NOTHING consumed it — no `Bloom`
/// component was ever attached to any camera in this crate. So the governor
/// (and the manual tier button) could report moving between Potato and Ultra
/// while the post stack stayed exactly the same. That is the bd-ikts complaint
/// "the post stack is implemented but invisible", one layer up: here it was not
/// merely invisible, it was absent.
///
/// Presence of the component IS the switch, so this inserts and removes rather
/// than mutating a field. `Bloom` requires `Hdr`, which the primary camera
/// already carries, and `BloomPlugin` ships inside `PostProcessPlugin` via
/// `DefaultPlugins`, so the inserted component is genuinely rendered rather
/// than silently inert.
fn apply_tier_to_bloom(
    mut commands: Commands,
    effective: Res<EffectiveRenderSettings>,
    cameras: Query<(Entity, Option<&Bloom>), With<PrimaryCamera>>,
) {
    if !effective.is_changed() {
        return;
    }
    for (entity, current) in &cameras {
        match (effective.features.bloom, current.is_some()) {
            (true, false) => {
                commands.entity(entity).insert(Bloom::NATURAL);
            }
            (false, true) => {
                commands.entity(entity).remove::<Bloom>();
            }
            // Already in the requested state; touching it would churn the
            // render world for no reason.
            _ => {}
        }
    }
}

fn handle_tonemap_mode_buttons(
    mut state: ResMut<TonemappingState>,
    mut query: Query<(&TonemapButton, &Interaction), ChangedButtonFilter>,
) {
    for (button, interaction) in &mut query {
        if *interaction == Interaction::Pressed && state.mode != button.mode {
            state.mode = button.mode;
            state.dirty = true;
        }
    }
}

fn handle_auto_exposure_toggle(
    mut state: ResMut<TonemappingState>,
    mut query: Query<&Interaction, (Changed<Interaction>, With<AutoExposureToggleButton>)>,
) {
    for interaction in &mut query {
        if *interaction == Interaction::Pressed {
            state.auto_exposure_enabled = !state.auto_exposure_enabled;
            state.dirty = true;
        }
    }
}

fn handle_exposure_adjust_buttons(
    mut state: ResMut<TonemappingState>,
    mut query: Query<(&ExposureAdjustButton, &Interaction), ChangedButtonFilter>,
) {
    for (button, interaction) in &mut query {
        if *interaction == Interaction::Pressed {
            state.exposure_bias = (state.exposure_bias + button.delta).clamp(-5.0, 5.0);
            state.dirty = true;
        }
    }
}

fn handle_palette_shortcuts(
    keys: Res<ButtonInput<KeyCode>>,
    mut accessibility: ResMut<AccessibilityState>,
) {
    if keys.just_pressed(KeyCode::KeyC) {
        accessibility.cycle();
        info!("Bevy palette cycled to {:?}", accessibility.palette());
    }
}

fn update_tonemap_button_colors(
    state: Res<TonemappingState>,
    mut query: Query<(&TonemapButton, &Interaction, &mut BackgroundColor)>,
) {
    for (button, interaction, mut color) in &mut query {
        let target = if state.mode == button.mode {
            follow_active_color()
        } else if matches!(interaction, Interaction::Hovered | Interaction::Pressed) {
            follow_hover_color()
        } else {
            follow_idle_color()
        };
        *color = target.into();
    }
}

fn update_auto_exposure_button_colors(
    state: Res<TonemappingState>,
    mut query: Query<(&Interaction, &mut BackgroundColor), With<AutoExposureToggleButton>>,
) {
    for (interaction, mut color) in &mut query {
        let target = if state.auto_exposure_enabled {
            follow_active_color()
        } else if matches!(interaction, Interaction::Hovered | Interaction::Pressed) {
            follow_hover_color()
        } else {
            follow_idle_color()
        };
        *color = target.into();
    }
}

fn update_exposure_button_colors(
    mut query: Query<(&Interaction, &mut BackgroundColor), With<ExposureAdjustButton>>,
) {
    for (interaction, mut color) in &mut query {
        let target = if matches!(interaction, Interaction::Pressed) {
            follow_active_color()
        } else if matches!(interaction, Interaction::Hovered) {
            follow_hover_color()
        } else {
            follow_idle_color()
        };
        *color = target.into();
    }
}

fn sync_camera_tonemapping(
    mut commands: Commands,
    mut state: ResMut<TonemappingState>,
    mut cameras: Query<
        (
            Entity,
            &mut Tonemapping,
            &mut ColorGrading,
            Option<&mut AutoExposure>,
        ),
        With<PrimaryCamera>,
    >,
) {
    if !state.dirty {
        return;
    }

    if let Ok((entity, mut tonemap, mut grading, auto_exposure)) = cameras.single_mut() {
        *tonemap = state.mode.to_component();
        grading.global.exposure = state.exposure_bias;

        match (state.auto_exposure_enabled, auto_exposure) {
            (true, None) => {
                commands.entity(entity).insert(AutoExposure {
                    speed_brighten: state.auto_exposure_speed_brighten,
                    speed_darken: state.auto_exposure_speed_darken,
                    ..Default::default()
                });
            }
            (true, Some(mut existing)) => {
                existing.speed_brighten = state.auto_exposure_speed_brighten;
                existing.speed_darken = state.auto_exposure_speed_darken;
            }
            (false, Some(_)) => {
                commands.entity(entity).remove::<AutoExposure>();
            }
            (false, None) => {}
        }
    }

    state.dirty = false;
}

fn update_follow_button_colors(
    rig: Res<CameraRig>,
    mut query: Query<(&FollowButton, &Interaction, &mut BackgroundColor)>,
) {
    for (button, interaction, mut color) in query.iter_mut() {
        let target = if rig.follow_mode == button.mode {
            follow_active_color()
        } else if matches!(interaction, Interaction::Hovered | Interaction::Pressed) {
            follow_hover_color()
        } else {
            follow_idle_color()
        };
        *color = target.into();
    }
}

#[allow(clippy::too_many_arguments)]
fn control_camera(
    time: Res<Time>,
    mut rig: ResMut<CameraRig>,
    state: Res<SnapshotState>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    mut camera_query: Query<&mut Transform, With<PrimaryCamera>>,
) {
    let Ok(mut transform) = camera_query.single_mut() else {
        mouse_motion.clear();
        mouse_wheel.clear();
        return;
    };

    let ctrl_held = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    let shift_held = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let alt_held = keys.pressed(KeyCode::AltLeft) || keys.pressed(KeyCode::AltRight);

    for wheel in mouse_wheel.read() {
        rig.distance *= (1.0 - wheel.y * 0.1).clamp(0.2, 5.0);
    }
    rig.distance = rig.distance.clamp(CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE);

    if keys.just_pressed(KeyCode::KeyF) {
        if ctrl_held {
            rig.queue_fit(FitCommand::Selection);
        } else if !shift_held {
            rig.cycle_follow_mode();
        }
    }

    if ctrl_held && keys.just_pressed(KeyCode::KeyW) {
        rig.queue_fit(FitCommand::World);
    }

    if ctrl_held && keys.just_pressed(KeyCode::KeyS) {
        rig.toggle_follow_mode(FollowMode::Selected);
    }

    if ctrl_held && keys.just_pressed(KeyCode::KeyO) {
        rig.toggle_follow_mode(FollowMode::Oldest);
    }

    if keys.pressed(KeyCode::KeyQ) {
        rig.yaw += time.delta_secs() * 1.2;
    }
    if keys.pressed(KeyCode::KeyE) {
        rig.yaw -= time.delta_secs() * 1.2;
    }
    if keys.pressed(KeyCode::PageUp) {
        rig.pitch += time.delta_secs() * 0.8;
    }
    if keys.pressed(KeyCode::PageDown) {
        rig.pitch -= time.delta_secs() * 0.8;
    }

    if buttons.pressed(MouseButton::Right) {
        for ev in mouse_motion.read() {
            rig.yaw -= ev.delta.x * 0.005;
            rig.pitch = (rig.pitch - ev.delta.y * 0.005).clamp(-1.45, -0.05);
        }
    } else {
        mouse_motion.clear();
    }

    let mut pan_input = Vec2::ZERO;
    let allow_pan = !ctrl_held && !alt_held;
    if allow_pan && keys.pressed(KeyCode::KeyW) {
        pan_input.y += 1.0;
    }
    if allow_pan && keys.pressed(KeyCode::KeyS) {
        pan_input.y -= 1.0;
    }
    if allow_pan && keys.pressed(KeyCode::KeyA) {
        pan_input.x -= 1.0;
    }
    if allow_pan && keys.pressed(KeyCode::KeyD) {
        pan_input.x += 1.0;
    }

    if pan_input.length_squared() > 0.0 {
        let forward = Vec2::new(rig.yaw.cos(), rig.yaw.sin());
        let right = Vec2::new(-forward.y, forward.x);
        let delta = (right * pan_input.x + forward * pan_input.y) * 600.0 * time.delta_secs();
        if rig.follow_mode != FollowMode::Off {
            rig.follow_mode = FollowMode::Off;
        }
        rig.pan += delta;
    }

    let mut focus_override = None;
    if state.latest.is_some()
        && let Some(command) = rig.pending_fit
    {
        match command {
            FitCommand::World => {
                focus_override = Some(state.world_center);
                let distance = fit_distance_for_extent(state.world_size, FIT_WORLD_FACTOR);
                rig.distance = distance;
                rig.distance_smoothed = distance;
            }
            FitCommand::Selection => {
                if let Some(bounds) = state.selection_bounds {
                    let center = state
                        .selection_center
                        .unwrap_or_else(|| (bounds.0 + bounds.1) * 0.5);
                    focus_override = Some(center);
                    let extent = bounds_extent(bounds);
                    let distance = fit_distance_for_extent(extent, FIT_SELECTION_FACTOR);
                    rig.distance = distance;
                    rig.distance_smoothed = distance;
                } else if let Some(selected) = state.first_agent_position {
                    focus_override = Some(selected);
                    let distance =
                        fit_distance_for_extent(Vec2::splat(400.0), FIT_SELECTION_FACTOR);
                    rig.distance = distance;
                    rig.distance_smoothed = distance;
                } else {
                    focus_override = Some(state.world_center);
                    let distance = fit_distance_for_extent(state.world_size, FIT_SELECTION_FACTOR);
                    rig.distance = distance;
                    rig.distance_smoothed = distance;
                }
            }
        }
        rig.pending_fit = None;
    }

    let follow_target = match rig.follow_mode {
        FollowMode::Off => None,
        FollowMode::Selected => state.selection_center.or(state.first_agent_position),
        FollowMode::Oldest => state
            .oldest_position
            .or(state.selection_center)
            .or(state.first_agent_position),
    };

    let mut target_focus = focus_override
        .or(follow_target)
        .unwrap_or(state.focus_point);
    if rig.follow_mode == FollowMode::Off && focus_override.is_none() {
        target_focus += rig.pan;
    }

    let world_size = state.world_size;
    if world_size.x > 0.0 && world_size.y > 0.0 {
        target_focus.x = target_focus.x.clamp(0.0, world_size.x);
        target_focus.y = target_focus.y.clamp(0.0, world_size.y);
    }

    if rig.recenter_now || rig.focus_smoothed.length_squared() == 0.0 {
        rig.focus_smoothed = target_focus;
        rig.distance_smoothed = rig.distance;
        rig.recenter_now = false;
    }

    let smoothing = 1.0 - (-time.delta_secs() * CAMERA_SMOOTHING_LERP).exp();
    rig.focus_smoothed = rig.focus_smoothed.lerp(target_focus, smoothing);
    rig.distance_smoothed += (rig.distance - rig.distance_smoothed) * smoothing;

    rig.pitch = rig.pitch.clamp(-1.45, -0.05);
    let yaw = rig.yaw;
    let pitch = rig.pitch;
    let distance = rig
        .distance_smoothed
        .clamp(CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE);

    let center = Vec3::new(
        rig.focus_smoothed.x - world_size.x * 0.5,
        0.0,
        world_size.y * 0.5 - rig.focus_smoothed.y,
    );

    let dir = Vec3::new(
        yaw.cos() * pitch.cos(),
        pitch.sin(),
        yaw.sin() * pitch.cos(),
    );
    transform.translation = center + dir * distance;
    transform.look_at(center, Vec3::Y);
}

fn sync_terrain(
    snapshot: &WorldSnapshot,
    commands: &mut Commands,
    registry: &mut TerrainChunkRegistry,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    probe_assets: &ReflectionProbeAssets,
    palette: ColorPaletteMode,
) {
    let dims = snapshot.terrain_height.dims;
    if dims.x == 0 || dims.y == 0 {
        return;
    }

    let chunk_size = registry.chunk_size.max(1);
    let chunks_x = dims.x.div_ceil(chunk_size);
    let chunks_y = dims.y.div_ceil(chunk_size);

    let mut seen: HashSet<TerrainChunkKey> = HashSet::with_capacity((chunks_x * chunks_y) as usize);

    for chunk_y in 0..chunks_y {
        for chunk_x in 0..chunks_x {
            let key = TerrainChunkKey {
                x: chunk_x,
                y: chunk_y,
            };

            let bounds = TerrainChunkBounds {
                origin: UVec2::new(chunk_x * chunk_size, chunk_y * chunk_size),
                size: UVec2::new(
                    (chunk_size).min(dims.x.saturating_sub(chunk_x * chunk_size)),
                    (chunk_size).min(dims.y.saturating_sub(chunk_y * chunk_size)),
                ),
            };

            if bounds.size.x == 0 || bounds.size.y == 0 {
                continue;
            }

            seen.insert(key);

            let built = build_chunk_mesh(snapshot, bounds, registry.height_scale, palette);

            match registry.chunks.get_mut(&key) {
                Some(record) => {
                    if record.palette != palette || record.signature != built.stats.signature {
                        if let Some(existing) = meshes.get_mut(&record.mesh) {
                            *existing = built.mesh;
                        } else {
                            let mesh_handle = meshes.add(built.mesh);
                            record.mesh = mesh_handle.clone();
                            commands
                                .entity(record.entity)
                                .insert(Mesh3d(mesh_handle.clone()));
                        }
                        update_chunk_material(materials, &record.material, &built.stats);
                        record.signature = built.stats.signature;
                        record.bounds = bounds;
                        record.stats = built.stats;
                        record.palette = palette;
                    }
                    sync_reflection_probe(
                        commands,
                        probe_assets,
                        record,
                        bounds,
                        &built.stats,
                        snapshot,
                    );
                    record.last_tick = snapshot.tick;
                }
                None => {
                    let mesh_handle = meshes.add(built.mesh);
                    let material_handle = materials.add(create_chunk_material(&built.stats));
                    let entity = commands
                        .spawn((
                            Mesh3d(mesh_handle.clone()),
                            MeshMaterial3d(material_handle.clone()),
                            Transform::default(),
                            GlobalTransform::default(),
                            Visibility::default(),
                            InheritedVisibility::default(),
                        ))
                        .id();
                    let probe = spawn_reflection_probe(
                        commands,
                        probe_assets,
                        bounds,
                        &built.stats,
                        snapshot,
                    );
                    registry.chunks.insert(
                        key,
                        TerrainChunkRecord {
                            entity,
                            mesh: mesh_handle,
                            material: material_handle,
                            bounds,
                            signature: built.stats.signature,
                            last_tick: snapshot.tick,
                            probe: Some(probe),
                            stats: built.stats,
                            palette,
                        },
                    );
                }
            }
        }
    }

    let stale: Vec<_> = registry
        .chunks
        .keys()
        .copied()
        .filter(|key| !seen.contains(key))
        .collect();

    for key in stale {
        if let Some(mut record) = registry.chunks.remove(&key) {
            commands.entity(record.entity).despawn();
            meshes.remove(&record.mesh);
            materials.remove(&record.material);
            if let Some(probe_entity) = record.probe.take() {
                commands.entity(probe_entity).despawn();
            }
        }
    }
}

fn create_chunk_material(stats: &TerrainChunkStats) -> StandardMaterial {
    let roughness = (0.45 + stats.mean_moisture * 0.4).clamp(0.1, 0.95);
    let metallic = (stats.mean_slope * 0.35).clamp(0.0, 0.5);
    let emissive_intensity =
        (stats.mean_moisture * 0.12 + stats.height_factor * 0.05).clamp(0.0, 0.35);
    let emissive = Color::linear_rgb(
        emissive_intensity * 0.6,
        emissive_intensity,
        emissive_intensity * 0.8,
    );
    StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: roughness,
        metallic,
        reflectance: (0.04 + stats.height_factor * 0.02).clamp(0.02, 0.08),
        emissive: emissive.into(),
        ..Default::default()
    }
}

fn update_chunk_material(
    materials: &mut Assets<StandardMaterial>,
    handle: &Handle<StandardMaterial>,
    stats: &TerrainChunkStats,
) {
    if let Some(material) = materials.get_mut(handle) {
        let roughness = (0.45 + stats.mean_moisture * 0.4).clamp(0.1, 0.95);
        let metallic = (stats.mean_slope * 0.35).clamp(0.0, 0.5);
        let emissive_intensity =
            (stats.mean_moisture * 0.12 + stats.height_factor * 0.05).clamp(0.0, 0.35);
        material.perceptual_roughness = roughness;
        material.metallic = metallic;
        material.reflectance = (0.04 + stats.height_factor * 0.02).clamp(0.02, 0.08);
        material.emissive = Color::linear_rgb(
            emissive_intensity * 0.6,
            emissive_intensity,
            emissive_intensity * 0.8,
        )
        .into();
    }
}

fn sync_reflection_probe(
    commands: &mut Commands,
    assets: &ReflectionProbeAssets,
    record: &mut TerrainChunkRecord,
    bounds: TerrainChunkBounds,
    stats: &TerrainChunkStats,
    snapshot: &WorldSnapshot,
) {
    if let Some(entity) = record.probe {
        let transform = chunk_probe_transform(bounds, stats, snapshot);
        commands.entity(entity).insert(transform);
    } else {
        let probe = spawn_reflection_probe(commands, assets, bounds, stats, snapshot);
        record.probe = Some(probe);
    }
}

fn spawn_reflection_probe(
    commands: &mut Commands,
    assets: &ReflectionProbeAssets,
    bounds: TerrainChunkBounds,
    stats: &TerrainChunkStats,
    snapshot: &WorldSnapshot,
) -> Entity {
    let transform = chunk_probe_transform(bounds, stats, snapshot);
    commands
        .spawn((
            LightProbe::new(),
            EnvironmentMapLight {
                diffuse_map: assets.diffuse.clone(),
                specular_map: assets.specular.clone(),
                intensity: 3500.0,
                rotation: Quat::IDENTITY,
                affects_lightmapped_mesh_diffuse: true,
            },
            transform,
            GlobalTransform::default(),
            Visibility::default(),
            InheritedVisibility::default(),
        ))
        .id()
}

fn chunk_probe_transform(
    bounds: TerrainChunkBounds,
    stats: &TerrainChunkStats,
    snapshot: &WorldSnapshot,
) -> Transform {
    let cell = snapshot.terrain_height.cell_size.max(1) as f32;
    let half = snapshot.world_size * 0.5;

    let min_x = bounds.origin.x as f32 * cell - half.x;
    let max_x = (bounds.origin.x + bounds.size.x) as f32 * cell - half.x;
    let max_z = half.y - bounds.origin.y as f32 * cell;
    let min_z = half.y - (bounds.origin.y + bounds.size.y) as f32 * cell;

    let center_x = (min_x + max_x) * 0.5;
    let center_z = (min_z + max_z) * 0.5;
    let width = stats.world_extent.x.max(cell);
    let depth = stats.world_extent.y.max(cell);
    let height = stats.max_height.max(20.0);

    Transform::from_translation(Vec3::new(center_x, height * 0.5, center_z))
        .with_scale(Vec3::new(width, height, depth))
}

struct BuiltChunk {
    mesh: Mesh,
    stats: TerrainChunkStats,
}

#[derive(Clone, Copy)]
struct TerrainChunkStats {
    mean_moisture: f32,
    mean_slope: f32,
    height_factor: f32,
    max_height: f32,
    world_extent: Vec2,
    signature: TerrainChunkSignature,
}

fn build_chunk_mesh(
    snapshot: &WorldSnapshot,
    bounds: TerrainChunkBounds,
    height_scale: f32,
    palette: ColorPaletteMode,
) -> BuiltChunk {
    let terrain = &snapshot.terrain_height;
    let cell_size = terrain.cell_size as f32;
    let half = snapshot.world_size * 0.5;

    let verts_x = bounds.size.x + 1;
    let verts_z = bounds.size.y + 1;
    let vertex_count = (verts_x * verts_z) as usize;

    let mut positions = Vec::with_capacity(vertex_count);
    let mut normals = vec![Vec3::ZERO; vertex_count];
    let mut uvs = Vec::with_capacity(vertex_count);
    let mut colors = Vec::with_capacity(vertex_count);
    let mut sum_moisture = 0.0f64;
    let mut sum_slope = 0.0f64;
    let mut max_height = f32::MIN;

    for vz in 0..verts_z {
        for vx in 0..verts_x {
            let global_x = bounds.origin.x + vx;
            let global_z = bounds.origin.y + vz;
            let height =
                sample_height_linear(terrain, global_x as f32, global_z as f32, height_scale);
            let world_x = global_x as f32 * cell_size - half.x;
            let world_z = half.y - global_z as f32 * cell_size;
            positions.push([world_x, height, world_z]);
            max_height = max_height.max(height);

            let uv_x = global_x as f32 / terrain.dims.x.max(1) as f32;
            let uv_z = global_z as f32 / terrain.dims.y.max(1) as f32;
            uvs.push([uv_x, uv_z]);

            let color = terrain_vertex_color(terrain, global_x, global_z, palette);
            colors.push(color);

            let sample = terrain.sample_tile(global_x, global_z);
            sum_moisture += sample.moisture as f64;
            let slope = compute_tile_slope(terrain, global_x, global_z);
            sum_slope += slope as f64;
        }
    }

    let mut indices = Vec::with_capacity((bounds.size.x * bounds.size.y * 6) as usize);
    let stride = verts_x;
    for z in 0..bounds.size.y {
        for x in 0..bounds.size.x {
            let i0 = z * stride + x;
            let i1 = i0 + 1;
            let i2 = i0 + stride;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    for &[ia, ib, ic] in indices.as_chunks::<3>().0 {
        let ia = ia as usize;
        let ib = ib as usize;
        let ic = ic as usize;
        let a = Vec3::from_array(positions[ia]);
        let b = Vec3::from_array(positions[ib]);
        let c = Vec3::from_array(positions[ic]);
        let normal = (b - a).cross(c - a);
        normals[ia] += normal;
        normals[ib] += normal;
        normals[ic] += normal;
    }

    let normals: Vec<[f32; 3]> = normals
        .into_iter()
        .map(|n| {
            let n = if n.length_squared() > 1e-6 {
                n.normalize()
            } else {
                Vec3::Y
            };
            [n.x, n.y, n.z]
        })
        .collect();

    let vertex_total = vertex_count as f64;
    let mean_moisture = (sum_moisture / vertex_total) as f32;
    let mean_slope = (sum_slope / vertex_total) as f32;
    let height_factor = (max_height / height_scale).clamp(0.0, 1.0);
    let world_extent = Vec2::new(
        bounds.size.x.max(1) as f32 * cell_size,
        bounds.size.y.max(1) as f32 * cell_size,
    );
    let signature = TerrainChunkSignature::from_render_inputs(
        &positions,
        &colors,
        &[
            mean_moisture,
            mean_slope,
            height_factor,
            max_height,
            world_extent.x,
            world_extent.y,
        ],
    );
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));

    let stats = TerrainChunkStats {
        mean_moisture,
        mean_slope,
        height_factor,
        max_height,
        world_extent,
        signature,
    };

    BuiltChunk { mesh, stats }
}

fn sample_height_linear(terrain: &TerrainHeightSnapshot, x: f32, z: f32, height_scale: f32) -> f32 {
    if terrain.dims.x == 0 || terrain.dims.y == 0 {
        return 0.0;
    }
    let max_x = (terrain.dims.x - 1) as f32;
    let max_z = (terrain.dims.y - 1) as f32;
    let fx = x.clamp(0.0, max_x);
    let fz = z.clamp(0.0, max_z);
    let x0 = fx.floor() as u32;
    let x1 = (x0 + 1).min(terrain.dims.x - 1);
    let z0 = fz.floor() as u32;
    let z1 = (z0 + 1).min(terrain.dims.y - 1);
    let tx = fx - x0 as f32;
    let tz = fz - z0 as f32;

    let h00 = terrain.elevation[terrain.index(x0, z0)];
    let h10 = terrain.elevation[terrain.index(x1, z0)];
    let h01 = terrain.elevation[terrain.index(x0, z1)];
    let h11 = terrain.elevation[terrain.index(x1, z1)];

    let h0 = h00 + (h10 - h00) * tx;
    let h1 = h01 + (h11 - h01) * tx;
    let h = h0 + (h1 - h0) * tz;
    h * height_scale
}

fn sample_height_world(terrain: &TerrainHeightSnapshot, position: Vec2, height_scale: f32) -> f32 {
    if terrain.dims.x == 0 || terrain.dims.y == 0 {
        return 0.0;
    }
    let cell = terrain.cell_size.max(1) as f32;
    let grid_x = (position.x / cell).clamp(0.0, (terrain.dims.x - 1) as f32);
    let grid_z = (position.y / cell).clamp(0.0, (terrain.dims.y - 1) as f32);
    sample_height_linear(terrain, grid_x, grid_z, height_scale)
}

fn terrain_vertex_color(
    terrain: &TerrainHeightSnapshot,
    x: u32,
    z: u32,
    palette: ColorPaletteMode,
) -> [f32; 4] {
    let sample = terrain.sample_tile(x, z);
    let slope = compute_tile_slope(terrain, x, z);
    let weights = visual::splat_weights(&SplatInput {
        kind: sample.kind,
        elevation: sample.elevation,
        slope,
        water_depth: sample.water_depth,
    });
    let mapped = visual::terrain_surface_srgb(&TerrainSurfaceInput {
        splat_weights: weights,
        moisture: sample.moisture,
        elevation: sample.elevation,
        slope,
        accent: sample.accent,
        daylight: terrain.daylight,
        accessibility: palette.accessibility(),
    });
    let linear = srgb_to_linear_rgb(mapped);
    [linear[0], linear[1], linear[2], 1.0]
}

fn srgb_to_linear_rgb(rgb: [f32; 3]) -> [f32; 3] {
    [
        srgb_to_linear_component(rgb[0]),
        srgb_to_linear_component(rgb[1]),
        srgb_to_linear_component(rgb[2]),
    ]
}

fn srgb_to_linear_component(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod terrain_tests {
    use super::*;
    use bevy_mesh::VertexAttributeValues;
    use scriptbots_core::{
        TerrainKind, TerrainLayer, TerrainTile,
        visual::{self, AgentVisualInput, VisualSelection},
    };
    use slotmap::KeyData;

    const TEST_REFERENCE_AGE_TICKS: u64 = 500;

    fn terrain_kinds() -> [TerrainKind; 6] {
        [
            TerrainKind::DeepWater,
            TerrainKind::ShallowWater,
            TerrainKind::Sand,
            TerrainKind::Grass,
            TerrainKind::Bloom,
            TerrainKind::Rock,
        ]
    }

    fn sample_agent_visual() -> AgentVisual {
        AgentVisual {
            id: AgentId::from(KeyData::from_ffi(1)),
            position: Vec2::new(50.0, 50.0),
            heading: 0.75,
            color: [0.2, 0.4, 0.7],
            selection: SelectionState::Selected,
            health: 1.4,
            energy: 0.62,
            age: 120,
            generation: 7,
            reference_age_ticks: TEST_REFERENCE_AGE_TICKS,
            spike_length: 4.0,
            boost: 1.0,
            wheel_left: 0.25,
            wheel_right: -0.5,
            herbivore_tendency: 0.8,
            temperature_preference: 0.3,
            food_delta: 0.4,
            sound_level: 0.2,
            sound_output: 0.6,
            sound_multiplier: 1.1,
            trait_modifiers: TraitModifiers::default(),
            eye_dirs: [0.0; NUM_EYES],
            eye_fov: [1.0; NUM_EYES],
            indicator: IndicatorState::default(),
            reproduction_intent: 0.0,
            spiked: true,
        }
    }

    fn sample_layer() -> TerrainLayer {
        let tiles = vec![
            TerrainTile {
                kind: TerrainKind::Grass,
                elevation: 0.0,
                moisture: 0.3,
                accent: 0.1,
                fertility_bias: 0.0,
                temperature_bias: 0.0,
                palette_index: 0,
            },
            TerrainTile {
                kind: TerrainKind::Sand,
                elevation: 0.5,
                moisture: 0.1,
                accent: 0.2,
                fertility_bias: 0.0,
                temperature_bias: 0.0,
                palette_index: 0,
            },
            TerrainTile {
                kind: TerrainKind::Rock,
                elevation: 1.0,
                moisture: 0.05,
                accent: 0.4,
                fertility_bias: 0.0,
                temperature_bias: 0.0,
                palette_index: 0,
            },
            TerrainTile {
                kind: TerrainKind::Bloom,
                elevation: 0.25,
                moisture: 0.8,
                accent: 0.6,
                fertility_bias: 0.0,
                temperature_bias: 0.0,
                palette_index: 0,
            },
        ];
        TerrainLayer::from_tiles(2, 2, 50, tiles).expect("construct terrain layer")
    }

    fn sample_world_snapshot() -> WorldSnapshot {
        let layer = sample_layer();
        let height =
            TerrainHeightSnapshot::new(&layer, None, visual::DAYLIGHT_STATIC).expect("snapshot");
        let dims = height.dims;
        let cell = layer.cell_size() as f32;
        let world_size = Vec2::new(dims.x as f32 * cell, dims.y as f32 * cell);
        let color = TerrainColorMap {
            width: dims.x,
            height: dims.y,
            pixels: vec![255; (dims.x * dims.y * 4) as usize],
        };
        WorldSnapshot {
            revision: 1,
            tick: 42,
            world_size,
            agent_radius: 12.0,
            terrain_color: color,
            terrain_height: height,
            agents: Vec::new(),
            events: Vec::new(),
            history: HudHistory::default(),
            brain: BrainOverlay::default(),
        }
    }

    fn backend_agreement_snapshot() -> WorldSnapshot {
        let mut tiles = vec![
            TerrainTile {
                kind: TerrainKind::Grass,
                elevation: 0.1,
                moisture: 0.2,
                accent: 0.1,
                fertility_bias: 0.0,
                temperature_bias: 0.0,
                palette_index: 0,
            };
            9
        ];
        tiles[4] = TerrainTile {
            kind: TerrainKind::Grass,
            elevation: 0.9,
            moisture: 0.37,
            accent: 0.63,
            fertility_bias: 0.0,
            temperature_bias: 0.0,
            palette_index: 0,
        };
        let layer =
            TerrainLayer::from_tiles(3, 3, 50, tiles).expect("backend agreement terrain layer");
        let mut water_depth = vec![0.0; 9];
        water_depth[4] = 1.5;
        let height = TerrainHeightSnapshot::new(&layer, Some(&water_depth), 0.35)
            .expect("backend agreement height snapshot");
        WorldSnapshot {
            revision: 1,
            tick: 42,
            world_size: Vec2::splat(150.0),
            agent_radius: 12.0,
            terrain_color: TerrainColorMap {
                width: 3,
                height: 3,
                pixels: vec![0; 3 * 3 * 4],
            },
            terrain_height: height,
            agents: Vec::new(),
            events: Vec::new(),
            history: HudHistory::default(),
            brain: BrainOverlay::default(),
        }
    }

    #[test]
    fn bevy_terrain_palette_matches_the_core_visual_authority() {
        for kind in terrain_kinds() {
            assert_eq!(
                terrain_kind_color(kind),
                visual::terrain_kind_base_color(kind),
                "Bevy must not own a competing {kind:?} base color"
            );
        }
    }

    #[test]
    fn terrain_accessibility_palette_is_applied_by_the_core_authority() {
        let snapshot = sample_world_snapshot();
        let natural =
            terrain_vertex_color(&snapshot.terrain_height, 0, 0, ColorPaletteMode::Natural);
        let high_contrast = terrain_vertex_color(
            &snapshot.terrain_height,
            0,
            0,
            ColorPaletteMode::HighContrast,
        );
        assert_ne!(
            natural.map(f32::to_bits),
            high_contrast.map(f32::to_bits),
            "palette cycling must recolor terrain as well as agents and HUD labels"
        );
    }

    #[test]
    fn terrain_projection_consumes_canonical_hydrology_and_daylight_inputs() {
        let dry_noon = sample_world_snapshot();
        let dry_color =
            terrain_vertex_color(&dry_noon.terrain_height, 0, 0, ColorPaletteMode::Natural);

        let mut flooded = dry_noon.clone();
        flooded.terrain_height.water_depth[0] = 1.0;
        let flooded_color =
            terrain_vertex_color(&flooded.terrain_height, 0, 0, ColorPaletteMode::Natural);
        assert_ne!(
            dry_color.map(f32::to_bits),
            flooded_color.map(f32::to_bits),
            "hydrology must feed the same splat-weight authority as GPUI/wgpu"
        );

        let mut night = dry_noon;
        night.terrain_height.daylight = visual::DAYLIGHT_NIGHT_FLOOR;
        let night_color =
            terrain_vertex_color(&night.terrain_height, 0, 0, ColorPaletteMode::Natural);
        assert_ne!(
            dry_color.map(f32::to_bits),
            night_color.map(f32::to_bits),
            "tick-derived daylight must feed the same shading authority as GPUI/wgpu"
        );
    }

    #[test]
    fn bevy_mesh_terrain_color_matches_the_shared_oracle_for_every_palette() {
        let snapshot = backend_agreement_snapshot();
        let terrain = &snapshot.terrain_height;
        let expected_slope = 0.8_f32;
        assert!(
            (compute_tile_slope(terrain, 1, 1) - expected_slope).abs() <= 1.0e-6,
            "agreement fixture must independently pin its symmetric center slope"
        );

        let sample = terrain.sample_tile(1, 1);
        let weights = visual::splat_weights(&SplatInput {
            kind: sample.kind,
            elevation: sample.elevation,
            slope: expected_slope,
            water_depth: sample.water_depth,
        });
        for palette in [
            ColorPaletteMode::Natural,
            ColorPaletteMode::Deuteranopia,
            ColorPaletteMode::Protanopia,
            ColorPaletteMode::Tritanopia,
            ColorPaletteMode::HighContrast,
        ] {
            let expected_srgb = visual::terrain_surface_srgb(&TerrainSurfaceInput {
                splat_weights: weights,
                moisture: sample.moisture,
                elevation: sample.elevation,
                slope: expected_slope,
                accent: sample.accent,
                daylight: terrain.daylight,
                accessibility: palette.accessibility(),
            });
            let expected_linear =
                Color::srgb(expected_srgb[0], expected_srgb[1], expected_srgb[2]).to_linear();
            let chunk = build_chunk_mesh(
                &snapshot,
                TerrainChunkBounds {
                    origin: UVec2::ZERO,
                    size: UVec2::splat(3),
                },
                TERRAIN_HEIGHT_SCALE,
                palette,
            );
            let VertexAttributeValues::Float32x4(colors) = chunk
                .mesh
                .attribute(Mesh::ATTRIBUTE_COLOR)
                .expect("terrain mesh must carry vertex colors")
            else {
                panic!("terrain mesh colors must be Float32x4");
            };
            let actual = colors[5];
            for (channel, (actual, expected)) in actual[..3]
                .iter()
                .copied()
                .zip([
                    expected_linear.red,
                    expected_linear.green,
                    expected_linear.blue,
                ])
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 1.0e-6,
                    "{palette:?} linear terrain channel {channel}: expected {expected}, got \
                     {actual}"
                );
            }
            assert_eq!(
                actual[3].to_bits(),
                1.0_f32.to_bits(),
                "{palette:?} terrain vertex alpha must stay opaque"
            );
            assert!(
                actual[..3]
                    .iter()
                    .copied()
                    .zip(expected_srgb)
                    .any(|(linear, srgb)| (linear - srgb).abs() > 1.0e-3),
                "fixture must detect feeding {palette:?} semantic sRGB directly into linear PBR"
            );
        }
        assert_eq!(
            terrain.fertility[4].to_bits(),
            0.0_f32.to_bits(),
            "agreement fixture deliberately avoids blessing the shared unwired fertility input"
        );
        assert_eq!(
            terrain.daylight.to_bits(),
            0.35_f32.to_bits(),
            "agreement fixture must exercise non-default daylight"
        );
        assert_eq!(
            terrain.water_depth[4].to_bits(),
            1.5_f32.to_bits(),
            "agreement fixture must exercise hydrology-driven splat weights"
        );
    }

    #[test]
    fn duplicate_outer_vertex_reuses_the_canonical_boundary_cell_color() {
        let snapshot = sample_world_snapshot();
        let terrain = &snapshot.terrain_height;
        let boundary = terrain_vertex_color(
            terrain,
            terrain.dims.x - 1,
            terrain.dims.y - 1,
            ColorPaletteMode::Natural,
        );
        let duplicate = terrain_vertex_color(
            terrain,
            terrain.dims.x,
            terrain.dims.y,
            ColorPaletteMode::Natural,
        );
        assert_eq!(
            duplicate.map(f32::to_bits),
            boundary.map(f32::to_bits),
            "the mesh-closing duplicate vertex must not invent a non-canonical edge slope"
        );
    }

    #[test]
    fn terrain_chunk_signature_rejects_equal_aggregate_spatial_rearrangements() {
        let positions_a = [[0.0, 1.0, 0.0], [1.0, 3.0, 0.0]];
        let positions_b = [[0.0, 3.0, 0.0], [1.0, 1.0, 0.0]];
        let colors_a = [[0.1, 0.8, 0.2, 1.0], [0.8, 0.7, 0.2, 1.0]];
        let colors_b = [colors_a[1], colors_a[0]];

        assert_ne!(
            TerrainChunkSignature::from_render_inputs(&positions_a, &colors_a, &[0.2]),
            TerrainChunkSignature::from_render_inputs(&positions_b, &colors_b, &[0.2]),
            "equal height/color aggregates must not preserve a stale spatial mesh"
        );
    }

    #[test]
    fn terrain_chunk_signature_covers_material_only_changes() {
        let positions = [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        let colors = [[0.8, 0.7, 0.2, 1.0]; 2];
        assert_ne!(
            TerrainChunkSignature::from_render_inputs(&positions, &colors, &[0.1, 0.0]),
            TerrainChunkSignature::from_render_inputs(&positions, &colors, &[0.9, 0.0]),
            "moisture/slope material changes must not preserve a stale PBR material"
        );
    }

    #[test]
    fn bevy_agent_base_color_matches_the_core_visual_authority() {
        let agent = sample_agent_visual();
        let expected = visual::agent_visual_params(&AgentVisualInput {
            genome_color: agent.color,
            health: agent.health,
            age_ticks: u64::from(agent.age),
            reference_age_ticks: agent.reference_age_ticks,
            herbivore_tendency: agent.herbivore_tendency,
            temperature_preference: agent.temperature_preference,
            wheel_left: agent.wheel_left,
            wheel_right: agent.wheel_right,
            heading: agent.heading,
            spike_extended: agent.spiked,
            spike_length: agent.spike_length,
            boosting: agent.boost > 0.05,
            sound_output: agent.sound_output,
            sound_multiplier: agent.sound_multiplier,
            sound_level: agent.sound_level,
            food_delta: agent.food_delta,
            trait_smell: agent.trait_modifiers.smell,
            trait_hearing: agent.trait_modifiers.hearing,
            selection: VisualSelection::Selected,
        });
        let (base, _) = agent_colors(&agent, ColorPaletteMode::Natural);
        let actual = base.to_srgba();
        for (channel, (actual, expected)) in [
            ("red", (actual.red, expected.body_color[0])),
            ("green", (actual.green, expected.body_color[1])),
            ("blue", (actual.blue, expected.body_color[2])),
        ] {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "Bevy {channel} channel {actual} disagrees with core authority {expected}"
            );
        }
    }

    #[test]
    fn presentation_revision_covers_tick_zero_and_same_tick_visual_changes() {
        let mut next_revision = 1;
        let mut initial = sample_world_snapshot();
        initial.tick = 0;
        initial.agents.push(sample_agent_visual());
        assert!(
            assign_presentation_revision(&mut initial, None, &mut next_revision)
                .expect("initial revision")
        );
        assert_eq!(initial.revision, 1, "tick zero must publish");

        let previous = initial.clone();
        let mut unchanged = initial.clone();
        assert!(
            !assign_presentation_revision(&mut unchanged, Some(&previous), &mut next_revision,)
                .expect("unchanged revision check"),
            "an identical snapshot must not produce presentation churn"
        );

        let mut selection_changed = initial;
        selection_changed.agents[0].selection = SelectionState::Hovered;
        assert!(
            assign_presentation_revision(
                &mut selection_changed,
                Some(&previous),
                &mut next_revision,
            )
            .expect("same-tick selection revision"),
            "same-tick presentation changes must not be dropped"
        );
        assert_eq!(selection_changed.tick, 0);
        assert_eq!(selection_changed.revision, 2);
    }

    #[test]
    fn chunk_mesh_positions_match_heightfield() {
        let snapshot = sample_world_snapshot();
        let bounds = TerrainChunkBounds {
            origin: UVec2::ZERO,
            size: snapshot.terrain_height.dims,
        };
        let built = build_chunk_mesh(&snapshot, bounds, 100.0, ColorPaletteMode::Natural);

        let positions = match built.mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(values)) => values.clone(),
            other => panic!("unexpected position attribute: {:?}", other),
        };
        assert_eq!(positions.len(), 9, "expected 3x3 vertex grid");

        let world_half = snapshot.world_size * 0.5;
        // Top-left vertex (0,0)
        let p0 = Vec3::from_array(positions[0]);
        assert!(
            (p0.x + world_half.x).abs() < 1e-3,
            "x mismatch for vertex 0"
        );
        assert!(
            (p0.z - world_half.y).abs() < 1e-3,
            "z mismatch for vertex 0"
        );
        assert!((p0.y - 0.0).abs() < 1e-3, "height mismatch for vertex 0");

        // Center vertex (global 1,1) should match bilinear height sample
        let center = Vec3::from_array(positions[4]);
        let expected_center = sample_height_linear(&snapshot.terrain_height, 1.0, 1.0, 100.0);
        assert!(
            (center.y - expected_center).abs() < 1e-3,
            "center height incorrect: {} vs {}",
            center.y,
            expected_center
        );

        // Bottom-right vertex corresponds to the far corner height sample
        let last = Vec3::from_array(positions[8]);
        let expected_last = sample_height_linear(
            &snapshot.terrain_height,
            snapshot.terrain_height.dims.x as f32,
            snapshot.terrain_height.dims.y as f32,
            100.0,
        );
        assert!(
            (last.y - expected_last).abs() < 1e-3,
            "bottom-right height incorrect: {} vs {}",
            last.y,
            expected_last
        );

        let indices = match built.mesh.indices() {
            Some(Indices::U32(idx)) => idx.clone(),
            other => panic!("unexpected index buffer: {:?}", other),
        };
        assert_eq!(
            indices.len(),
            24,
            "expected 2x2 quads => 24 indices (two tris per cell)"
        );

        assert!(built.stats.mean_moisture > 0.0);
        assert!(built.stats.max_height > 0.0);
    }

    #[test]
    fn agent_translation_respects_terrain_height() {
        let mut snapshot = sample_world_snapshot();
        snapshot.agents.push(sample_agent_visual());

        let translation = agent_translation(&snapshot, &snapshot.agents[0]);
        let terrain_height = sample_height_world(
            &snapshot.terrain_height,
            snapshot.agents[0].position,
            TERRAIN_HEIGHT_SCALE,
        );
        let expected = terrain_height + snapshot.agent_radius * 0.35;
        assert!((translation.y - expected).abs() < 1e-3);
    }
}

fn compute_tile_slope(terrain: &TerrainHeightSnapshot, x: u32, z: u32) -> f32 {
    if terrain.dims.x == 0 || terrain.dims.y == 0 {
        return 0.0;
    }
    // Chunk meshes duplicate the last tile at their outer vertex. Clamp the
    // coordinate before choosing neighbours so that duplicate inherits the
    // exact boundary-cell color instead of acquiring a one-sided slope that
    // no canonical terrain cell has.
    let x = x.min(terrain.dims.x - 1);
    let z = z.min(terrain.dims.y - 1);
    let center = terrain.sample_tile(x, z).elevation;
    let left = terrain.sample_tile(x.saturating_sub(1), z).elevation;
    let right = terrain
        .sample_tile((x + 1).min(terrain.dims.x.saturating_sub(1)), z)
        .elevation;
    let up = terrain.sample_tile(x, z.saturating_sub(1)).elevation;
    let down = terrain
        .sample_tile(x, (z + 1).min(terrain.dims.y.saturating_sub(1)))
        .elevation;
    ((center - left).abs() + (center - right).abs() + (center - up).abs() + (center - down).abs())
        * 0.25
}

fn spawn_agent_entity(
    agent: &AgentVisual,
    snapshot: &WorldSnapshot,
    commands: &mut Commands,
    meshes: &AgentMeshes,
    materials: &mut Assets<StandardMaterial>,
    palette: ColorPaletteMode,
) -> AgentRecord {
    let root_transform = Transform::from_translation(agent_translation(snapshot, agent))
        .with_rotation(Quat::from_rotation_y(agent.heading));
    let root = commands
        .spawn((
            root_transform,
            GlobalTransform::default(),
            Visibility::default(),
            InheritedVisibility::default(),
        ))
        .id();

    let (body_color, body_emissive) = agent_colors(agent, palette);
    let body = spawn_part(
        commands,
        &meshes.body,
        materials,
        body_color,
        body_emissive,
        AlphaMode::Opaque,
        false,
        false,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(body.entity);

    let stripe = spawn_part(
        commands,
        &meshes.quad,
        materials,
        Color::srgb(0.4, 0.62, 0.2),
        Color::linear_rgb(0.2, 0.4, 0.12),
        AlphaMode::Opaque,
        false,
        true,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(stripe.entity);

    let wheel_left = spawn_part(
        commands,
        &meshes.wheel,
        materials,
        Color::srgb(0.12, 0.14, 0.2),
        Color::linear_rgb(0.1, 0.12, 0.18),
        AlphaMode::Opaque,
        false,
        false,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(wheel_left.entity);

    let wheel_right = spawn_part(
        commands,
        &meshes.wheel,
        materials,
        Color::srgb(0.12, 0.14, 0.2),
        Color::linear_rgb(0.1, 0.12, 0.18),
        AlphaMode::Opaque,
        false,
        false,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(wheel_right.entity);

    let mouth = spawn_part(
        commands,
        &meshes.quad,
        materials,
        Color::srgb(0.72, 0.2, 0.16),
        Color::linear_rgb(0.3, 0.08, 0.06),
        AlphaMode::Blend,
        true,
        true,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(mouth.entity);

    let nose = spawn_part(
        commands,
        &meshes.sphere,
        materials,
        Color::srgb(0.95, 0.86, 0.66),
        Color::linear_rgb(0.4, 0.28, 0.18),
        AlphaMode::Opaque,
        false,
        false,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(nose.entity);

    let spike = spawn_part(
        commands,
        &meshes.spike,
        materials,
        Color::srgb(0.86, 0.34, 0.2),
        Color::linear_rgb(0.6, 0.1, 0.08),
        AlphaMode::Opaque,
        false,
        false,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(spike.entity);

    let boost = spawn_part(
        commands,
        &meshes.quad,
        materials,
        Color::srgb(0.2, 0.36, 0.95),
        Color::linear_rgb(0.25, 0.5, 1.18),
        AlphaMode::Add,
        true,
        true,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(boost.entity);

    let ear_left = spawn_part(
        commands,
        &meshes.sphere,
        materials,
        Color::srgb(0.82, 0.78, 0.58),
        Color::linear_rgb(0.22, 0.24, 0.12),
        AlphaMode::Opaque,
        false,
        false,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(ear_left.entity);

    let ear_right = spawn_part(
        commands,
        &meshes.sphere,
        materials,
        Color::srgb(0.82, 0.78, 0.58),
        Color::linear_rgb(0.22, 0.24, 0.12),
        AlphaMode::Opaque,
        false,
        false,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(ear_right.entity);

    let selection = spawn_part(
        commands,
        &meshes.ring,
        materials,
        Color::srgb(0.24, 0.52, 1.0),
        Color::linear_rgb(0.22, 0.58, 1.2),
        AlphaMode::Add,
        true,
        true,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(selection.entity);

    let indicator = spawn_part(
        commands,
        &meshes.quad,
        materials,
        Color::srgb(0.48, 0.82, 0.36),
        Color::linear_rgb(0.32, 0.7, 0.26),
        AlphaMode::Add,
        true,
        true,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(indicator.entity);

    let sound_inner = spawn_part(
        commands,
        &meshes.quad,
        materials,
        Color::srgba(0.3, 0.55, 0.95, 0.0),
        Color::linear_rgb(0.0, 0.0, 0.0),
        AlphaMode::Add,
        true,
        true,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(sound_inner.entity);

    let sound_outer = spawn_part(
        commands,
        &meshes.quad,
        materials,
        Color::srgba(0.15, 0.45, 0.95, 0.0),
        Color::linear_rgb(0.0, 0.0, 0.0),
        AlphaMode::Add,
        true,
        true,
        Transform::IDENTITY,
    );
    commands.entity(root).add_child(sound_outer.entity);

    let mut eyes = Vec::with_capacity(NUM_EYES);
    for _ in 0..NUM_EYES {
        let sclera = spawn_part(
            commands,
            &meshes.sphere,
            materials,
            Color::srgb(0.92, 0.95, 1.0),
            Color::linear_rgb(0.18, 0.2, 0.24),
            AlphaMode::Opaque,
            false,
            false,
            Transform::IDENTITY,
        );
        let pupil = spawn_part(
            commands,
            &meshes.sphere,
            materials,
            Color::srgb(0.08, 0.09, 0.12),
            Color::linear_rgb(0.1, 0.14, 0.2),
            AlphaMode::Opaque,
            false,
            false,
            Transform::IDENTITY,
        );
        commands.entity(root).add_child(sclera.entity);
        commands.entity(root).add_child(pupil.entity);
        eyes.push(EyePart { sclera, pupil });
    }

    let record = AgentRecord {
        root,
        body,
        stripe,
        wheel_left,
        wheel_right,
        mouth,
        nose,
        spike,
        boost,
        ear_left,
        ear_right,
        selection,
        indicator,
        sound_inner,
        sound_outer,
        eyes,
    };

    apply_agent_visuals(
        &record, agent, snapshot, commands, materials, meshes, palette,
    );
    record
}

fn update_agent_entity(
    record: &AgentRecord,
    agent: &AgentVisual,
    snapshot: &WorldSnapshot,
    commands: &mut Commands,
    materials: &mut Assets<StandardMaterial>,
    meshes: &AgentMeshes,
    palette: ColorPaletteMode,
) {
    apply_agent_visuals(
        record, agent, snapshot, commands, materials, meshes, palette,
    );
}

fn apply_agent_visuals(
    record: &AgentRecord,
    agent: &AgentVisual,
    snapshot: &WorldSnapshot,
    commands: &mut Commands,
    materials: &mut Assets<StandardMaterial>,
    meshes: &AgentMeshes,
    palette: ColorPaletteMode,
) {
    use std::f32::consts::FRAC_PI_2;

    let visuals = canonical_agent_visual_params(agent);
    let translation = agent_translation(snapshot, agent);
    let rotation = Quat::from_rotation_y(agent.heading);
    commands
        .entity(record.root)
        .insert(Transform::from_translation(translation).with_rotation(rotation));

    let scale_factor = (snapshot.agent_radius / meshes.base_radius).clamp(0.2, 1024.0);
    let body_length = scale_factor * 2.35;
    let body_radius = scale_factor * 0.88;

    let body_transform = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::from_rotation_z(FRAC_PI_2),
        scale: Vec3::new(
            body_length.max(0.1),
            body_radius.max(0.1),
            body_radius.max(0.1),
        ),
    };
    let (body_color, body_emissive) = agent_colors_from_params(&visuals, palette);
    update_part_transform(commands, &record.body, body_transform);
    update_part_colors(materials, &record.body, body_color, body_emissive);

    let stripe_rgb = Vec3::from_array(visuals.stripe_color);
    let stripe_color = srgb_from_vec_with_palette(stripe_rgb, 0.9, palette);
    let stripe_emissive = palette_hdr_emissive_from_srgb(
        visuals.stripe_emissive,
        visuals.body_emissive_gain,
        palette,
    );
    let stripe_transform = Transform {
        translation: Vec3::new(0.0, body_radius * 0.16, 0.0),
        rotation: Quat::from_rotation_y(FRAC_PI_2) * Quat::from_rotation_z(FRAC_PI_2),
        scale: Vec3::new(
            (body_length * 1.04).max(0.05),
            (body_radius * 0.3).max(0.05),
            (body_radius * 0.3).max(0.05),
        ),
    };
    update_part_transform(commands, &record.stripe, stripe_transform);
    update_part_colors(materials, &record.stripe, stripe_color, stripe_emissive);

    let wheel_offset = body_radius * 1.12;
    let wheel_vertical = -body_radius * 0.38;
    let wheel_scale = Vec3::new(
        (scale_factor * 0.75).max(0.05),
        (scale_factor * 0.75).max(0.05),
        (scale_factor * 0.4).max(0.05),
    );
    let left_wheel_transform = Transform {
        translation: Vec3::new(0.0, wheel_vertical, wheel_offset),
        rotation: Quat::from_rotation_x(FRAC_PI_2),
        scale: wheel_scale,
    };
    let right_wheel_transform = Transform {
        translation: Vec3::new(0.0, wheel_vertical, -wheel_offset),
        rotation: Quat::from_rotation_x(FRAC_PI_2),
        scale: wheel_scale,
    };
    update_part_transform(commands, &record.wheel_left, left_wheel_transform);
    update_part_transform(commands, &record.wheel_right, right_wheel_transform);

    let left_rgb = Vec3::from_array(visuals.wheel_colors[0]);
    let right_rgb = Vec3::from_array(visuals.wheel_colors[1]);
    let left_color = srgb_from_vec_with_palette(left_rgb, 1.0, palette);
    let right_color = srgb_from_vec_with_palette(right_rgb, 1.0, palette);
    let left_emissive = palette_hdr_emissive_from_srgb(visuals.wheel_emissives[0], 1.0, palette);
    let right_emissive = palette_hdr_emissive_from_srgb(visuals.wheel_emissives[1], 1.0, palette);
    update_part_colors(materials, &record.wheel_left, left_color, left_emissive);
    update_part_colors(materials, &record.wheel_right, right_color, right_emissive);

    let vocal_energy = clamp01(agent.sound_output.abs() * agent.sound_multiplier.max(0.1));
    let mouth_activity = visuals.mouth_activity;
    let mouth_height = scale_factor * (0.25 + 0.6 * mouth_activity);
    let mouth_depth = scale_factor * 0.12;
    let mouth_width = body_radius * 0.95;
    let mouth_transform = Transform {
        translation: Vec3::new(body_length * 0.58, scale_factor * 0.04, 0.0),
        rotation: Quat::from_rotation_y(FRAC_PI_2),
        scale: Vec3::new(
            mouth_depth.max(0.02),
            mouth_height.max(0.05),
            mouth_width.max(0.05),
        ),
    };
    update_part_transform(commands, &record.mouth, mouth_transform);
    let mouth_rgb = Vec3::from_array(visuals.mouth_color);
    let mouth_color = srgb_from_vec_with_palette(mouth_rgb, 0.9, palette);
    let mouth_emissive = palette_emissive_from_vec(
        Vec3::new(
            mouth_rgb.x * mouth_activity * 0.8,
            mouth_rgb.y * mouth_activity * 0.4,
            mouth_rgb.z * mouth_activity * 0.3,
        ),
        palette,
    );
    update_part_colors(materials, &record.mouth, mouth_color, mouth_emissive);

    let nose_transform = Transform {
        translation: Vec3::new(body_length * 0.63, scale_factor * 0.24, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::splat((scale_factor * 0.34).max(0.05)),
    };
    update_part_transform(commands, &record.nose, nose_transform);
    let nose_rgb = Vec3::from_array(visuals.nose_color);
    let nose_color = srgb_from_vec_with_palette(nose_rgb, 1.0, palette);
    let nose_emissive = palette_emissive_from_vec(
        Vec3::new(nose_rgb.x * 0.25, nose_rgb.y * 0.2, nose_rgb.z * 0.15),
        palette,
    );
    update_part_colors(materials, &record.nose, nose_color, nose_emissive);

    let spike_ready = visuals.spike_readiness;
    let spike_length = scale_factor * (0.65 + agent.spike_length.max(0.0));
    let spike_transform = Transform {
        translation: Vec3::new(
            body_length * 0.7 + spike_length * 0.5,
            scale_factor * 0.06,
            0.0,
        ),
        rotation: Quat::from_rotation_z(-FRAC_PI_2),
        scale: Vec3::new(
            spike_length.max(0.06),
            (scale_factor * 0.48).max(0.04),
            (scale_factor * 0.48).max(0.04),
        ),
    };
    update_part_transform(commands, &record.spike, spike_transform);
    let spike_rgb = Vec3::from_array(visuals.spike_color);
    let spike_color = srgb_from_vec_with_palette(spike_rgb, 1.0, palette);
    let spike_emissive = palette_hdr_emissive_from_srgb(
        visuals.spike_color,
        visuals.spike_emissive_gain * spike_ready,
        palette,
    );
    update_part_colors(materials, &record.spike, spike_color, spike_emissive);

    let boost_strength = clamp01(agent.boost);
    let boost_transform = Transform {
        translation: Vec3::new(-body_length * 0.62, -scale_factor * 0.05, 0.0),
        rotation: Quat::from_rotation_z(FRAC_PI_2),
        scale: Vec3::new(
            (scale_factor * (0.6 + boost_strength * 0.9)).max(0.05),
            (scale_factor * 0.18).max(0.03),
            (scale_factor * 0.18).max(0.03),
        ),
    };
    update_part_transform(commands, &record.boost, boost_transform);
    let boost_rgb = Vec3::new(
        0.22 + boost_strength * 0.25,
        0.48 + boost_strength * 0.45,
        1.0 + boost_strength * 0.55,
    );
    let boost_color = srgb_from_vec_with_palette(boost_rgb, 0.45 + boost_strength * 0.4, palette);
    let boost_emissive = palette_emissive_from_vec(
        Vec3::new(
            boost_rgb.x * boost_strength * 1.3,
            boost_rgb.y * boost_strength * 1.45,
            boost_rgb.z * boost_strength * 1.6,
        ),
        palette,
    );
    update_part_colors(materials, &record.boost, boost_color, boost_emissive);
    set_part_visibility(commands, &record.boost, boost_strength > 0.02);

    let hearing = clamp01(agent.trait_modifiers.hearing);
    let ear_scale = Vec3::new(
        (scale_factor * (0.3 + hearing * 0.35)).max(0.04),
        (scale_factor * (0.48 + hearing * 0.4)).max(0.05),
        (scale_factor * (0.32 + hearing * 0.2)).max(0.04),
    );
    let ear_height = body_radius * (0.58 + hearing * 0.1);
    let ear_offset = body_radius * 0.92;
    let ear_left_transform = Transform {
        translation: Vec3::new(-scale_factor * 0.12, ear_height, ear_offset),
        rotation: Quat::IDENTITY,
        scale: ear_scale,
    };
    let ear_right_transform = Transform {
        translation: Vec3::new(-scale_factor * 0.12, ear_height, -ear_offset),
        rotation: Quat::IDENTITY,
        scale: ear_scale,
    };
    update_part_transform(commands, &record.ear_left, ear_left_transform);
    update_part_transform(commands, &record.ear_right, ear_right_transform);
    let ear_rgb = Vec3::new(0.82, 0.75 + hearing * 0.18, 0.54);
    let ear_color = srgb_from_vec_with_palette(ear_rgb, 1.0, palette);
    let ear_emissive = palette_emissive_from_vec(
        Vec3::new(ear_rgb.x * 0.18, ear_rgb.y * 0.2, ear_rgb.z * 0.15),
        palette,
    );
    update_part_colors(materials, &record.ear_left, ear_color, ear_emissive);
    update_part_colors(materials, &record.ear_right, ear_color, ear_emissive);

    let ring_radius_scale = Vec3::splat((body_radius * 1.45).max(0.1));
    let ring_transform = Transform {
        translation: Vec3::new(0.0, -body_radius * 0.82, 0.0),
        rotation: Quat::from_rotation_x(FRAC_PI_2),
        scale: ring_radius_scale,
    };
    update_part_transform(commands, &record.selection, ring_transform);
    let (ring_alpha, ring_emissive_scale) = match agent.selection {
        SelectionState::None => (0.0, 0.0),
        SelectionState::Hovered => (0.35, 0.65),
        SelectionState::Selected => (0.65, 0.95),
    };
    let ring_rgb = Vec3::from_array(visuals.selection_rim_color);
    let ring_color = srgb_from_vec_with_palette(ring_rgb, ring_alpha, palette);
    let ring_emissive = palette_emissive_from_vec(
        Vec3::new(
            ring_rgb.x * ring_emissive_scale,
            ring_rgb.y * ring_emissive_scale,
            ring_rgb.z * ring_emissive_scale,
        ),
        palette,
    );
    update_part_colors(materials, &record.selection, ring_color, ring_emissive);
    set_part_visibility(commands, &record.selection, ring_alpha > 0.02);

    let indicator_intensity = clamp01(agent.indicator.intensity);
    let indicator_rgb = Vec3::from_array(agent.indicator.color);
    let indicator_alpha =
        0.35 + indicator_intensity * 0.4 + agent.reproduction_intent.clamp(0.0, 1.0) * 0.2;
    let indicator_color = srgb_from_vec_with_palette(indicator_rgb, indicator_alpha, palette);
    let indicator_emissive = palette_emissive_from_vec(
        Vec3::new(
            indicator_rgb.x * indicator_intensity * 1.3,
            indicator_rgb.y * indicator_intensity * 1.3,
            indicator_rgb.z * indicator_intensity * 1.3,
        ),
        palette,
    );
    let indicator_transform = Transform {
        translation: Vec3::new(
            0.0,
            body_radius * (1.75 + agent.reproduction_intent.clamp(0.0, 1.0) * 0.45),
            0.0,
        ),
        rotation: Quat::from_rotation_y(FRAC_PI_2),
        scale: Vec3::new(
            (scale_factor * 0.38).max(0.05),
            (scale_factor * 0.38).max(0.05),
            (scale_factor * (0.62 + indicator_intensity * 0.45)).max(0.05),
        ),
    };
    update_part_transform(commands, &record.indicator, indicator_transform);
    update_part_colors(
        materials,
        &record.indicator,
        indicator_color,
        indicator_emissive,
    );
    set_part_visibility(commands, &record.indicator, indicator_alpha > 0.05);

    let ambient_sound = clamp01(agent.sound_level);
    let arc_strength = (vocal_energy * 0.8 + ambient_sound * 0.4).clamp(0.0, 1.0);
    let arc_base_translation = Vec3::new(
        body_length * (0.75 + arc_strength * 0.25),
        scale_factor * 0.05,
        0.0,
    );
    let arc_rotation = Quat::from_rotation_y(FRAC_PI_2);

    let inner_visible = arc_strength > 0.02;
    set_part_visibility(commands, &record.sound_inner, inner_visible);
    if inner_visible {
        let inner_scale = Vec3::new(
            (scale_factor * 0.08).max(0.01),
            (scale_factor * (0.35 + arc_strength * 0.55)).max(0.05),
            (scale_factor * 0.05).max(0.01),
        );
        let inner_transform = Transform {
            translation: arc_base_translation + Vec3::new(0.0, scale_factor * 0.04, 0.0),
            rotation: arc_rotation,
            scale: inner_scale,
        };
        update_part_transform(commands, &record.sound_inner, inner_transform);
        let inner_rgb = Vec3::new(0.32, 0.6, 1.0);
        let inner_alpha = 0.15 + arc_strength * 0.55;
        let inner_color = srgb_from_vec_with_palette(inner_rgb, inner_alpha, palette);
        let inner_emissive = palette_emissive_from_vec(
            Vec3::new(
                inner_rgb.x * (0.6 + arc_strength * 0.8),
                inner_rgb.y * (0.6 + arc_strength * 0.8),
                inner_rgb.z * (0.9 + arc_strength * 0.9),
            ),
            palette,
        );
        update_part_colors(materials, &record.sound_inner, inner_color, inner_emissive);
    }

    let outer_visible = arc_strength > 0.04;
    set_part_visibility(commands, &record.sound_outer, outer_visible);
    if outer_visible {
        let outer_scale = Vec3::new(
            (scale_factor * 0.12).max(0.02),
            (scale_factor * (0.55 + arc_strength * 0.85)).max(0.08),
            (scale_factor * 0.05).max(0.01),
        );
        let outer_transform = Transform {
            translation: arc_base_translation + Vec3::new(0.0, scale_factor * 0.02, 0.0),
            rotation: arc_rotation,
            scale: outer_scale,
        };
        update_part_transform(commands, &record.sound_outer, outer_transform);
        let outer_rgb = Vec3::new(0.18, 0.45, 0.95);
        let outer_alpha = 0.08 + arc_strength * 0.45;
        let outer_color = srgb_from_vec_with_palette(outer_rgb, outer_alpha, palette);
        let outer_emissive = palette_emissive_from_vec(
            Vec3::new(
                outer_rgb.x * (0.4 + arc_strength * 0.7),
                outer_rgb.y * (0.4 + arc_strength * 0.7),
                outer_rgb.z * (0.6 + arc_strength * 0.8),
            ),
            palette,
        );
        update_part_colors(materials, &record.sound_outer, outer_color, outer_emissive);
    }

    let eye_base = scale_factor * (0.22 + clamp01(agent.trait_modifiers.eye) * 0.15);
    let eye_vertical = body_radius * 0.35;
    let eye_forward = body_length * 0.42;
    let pupil_scale = eye_base * 0.45;

    for (idx, eye) in record.eyes.iter().enumerate() {
        let rel_dir = agent.eye_dirs[idx];
        let fov = agent.eye_fov[idx];
        let fov_scale = clamp01(fov / std::f32::consts::PI);

        let lateral = Vec3::new(0.0, 0.0, -rel_dir.sin() * body_radius * 0.42);
        let forward_bias = Vec3::new(rel_dir.cos().max(0.0) * body_radius * 0.15, 0.0, 0.0);
        let sclera_translation = Vec3::new(eye_forward, eye_vertical, 0.0) + lateral + forward_bias;
        let sclera_scale = Vec3::splat((eye_base * (0.88 + fov_scale * 0.4)).max(0.03));
        let sclera_transform = Transform {
            translation: sclera_translation,
            rotation: Quat::IDENTITY,
            scale: sclera_scale,
        };
        update_part_transform(commands, &eye.sclera, sclera_transform);

        let look_dir = Quat::from_rotation_y(rel_dir)
            .mul_vec3(Vec3::X)
            .normalize_or_zero();
        let pupil_translation = sclera_translation + look_dir * (eye_base * 0.35);
        let pupil_transform = Transform {
            translation: pupil_translation,
            rotation: Quat::IDENTITY,
            scale: Vec3::splat(pupil_scale.max(0.015)),
        };
        update_part_transform(commands, &eye.pupil, pupil_transform);

        let sclera_rgb = mix_vec3(
            Vec3::new(0.92, 0.94, 1.0),
            Vec3::new(0.88, 0.93, 1.05),
            clamp01(agent.trait_modifiers.eye * 0.3),
        );
        let sclera_color = srgb_from_vec_with_palette(sclera_rgb, 1.0, palette);
        let sclera_emissive = palette_emissive_from_vec(
            Vec3::new(sclera_rgb.x * 0.18, sclera_rgb.y * 0.2, sclera_rgb.z * 0.24),
            palette,
        );
        update_part_colors(materials, &eye.sclera, sclera_color, sclera_emissive);

        let pupil_rgb =
            Vec3::new(0.08, 0.09, 0.12) * (1.0 + clamp01(agent.sound_multiplier - 1.0) * 0.25);
        let pupil_color = srgb_from_vec_with_palette(pupil_rgb, 1.0, palette);
        let pupil_emissive = palette_emissive_from_vec(
            Vec3::new(
                pupil_rgb.x * vocal_energy * 0.6,
                pupil_rgb.y * vocal_energy * 0.5,
                pupil_rgb.z * vocal_energy * 0.9,
            ),
            palette,
        );
        update_part_colors(materials, &eye.pupil, pupil_color, pupil_emissive);
    }
}

fn sync_agents(
    snapshot: &WorldSnapshot,
    commands: &mut Commands,
    registry: &mut AgentRegistry,
    meshes: &AgentMeshes,
    materials: &mut Assets<StandardMaterial>,
    palette: ColorPaletteMode,
) {
    let mut seen: HashSet<AgentId> = HashSet::with_capacity(snapshot.agents.len());
    for agent in &snapshot.agents {
        seen.insert(agent.id);
        if let Some(record) = registry.records.get_mut(&agent.id) {
            update_agent_entity(
                record, agent, snapshot, commands, materials, meshes, palette,
            );
        } else {
            let record = spawn_agent_entity(agent, snapshot, commands, meshes, materials, palette);
            registry.records.insert(agent.id, record);
        }
    }

    let stale: Vec<AgentId> = registry
        .records
        .keys()
        .copied()
        .filter(|id| !seen.contains(id))
        .collect();

    for id in stale {
        if let Some(record) = registry.records.remove(&id) {
            cleanup_agent_materials(materials, &record);
            despawn_agent_entities(record, commands);
        }
    }
}

fn agent_translation(snapshot: &WorldSnapshot, agent: &AgentVisual) -> Vec3 {
    let half = snapshot.world_size * 0.5;
    let terrain_height = sample_height_world(
        &snapshot.terrain_height,
        agent.position,
        TERRAIN_HEIGHT_SCALE,
    );
    let x = agent.position.x - half.x;
    let z = half.y - agent.position.y;
    Vec3::new(x, terrain_height + snapshot.agent_radius * 0.35, z)
}
fn terrain_kind_color(kind: TerrainKind) -> [f32; 3] {
    visual::terrain_kind_base_color(kind)
}

const fn visual_selection(selection: SelectionState) -> VisualSelection {
    match selection {
        SelectionState::None => VisualSelection::None,
        SelectionState::Hovered => VisualSelection::Hovered,
        SelectionState::Selected => VisualSelection::Selected,
    }
}

fn canonical_agent_visual_params(agent: &AgentVisual) -> AgentVisualParams {
    visual::agent_visual_params(&AgentVisualInput {
        genome_color: agent.color,
        health: agent.health,
        age_ticks: u64::from(agent.age),
        reference_age_ticks: agent.reference_age_ticks,
        herbivore_tendency: agent.herbivore_tendency,
        temperature_preference: agent.temperature_preference,
        wheel_left: agent.wheel_left,
        wheel_right: agent.wheel_right,
        heading: agent.heading,
        spike_extended: agent.spiked,
        spike_length: agent.spike_length,
        boosting: agent.boost > 0.05,
        sound_output: agent.sound_output,
        sound_multiplier: agent.sound_multiplier,
        sound_level: agent.sound_level,
        food_delta: agent.food_delta,
        trait_smell: agent.trait_modifiers.smell,
        trait_hearing: agent.trait_modifiers.hearing,
        selection: visual_selection(agent.selection),
    })
}

fn agent_colors_from_params(
    params: &AgentVisualParams,
    palette: ColorPaletteMode,
) -> (Color, Color) {
    let base = srgb_from_vec_with_palette(Vec3::from_array(params.body_color), 1.0, palette);
    let emissive =
        palette_hdr_emissive_from_srgb(params.body_emissive, params.body_emissive_gain, palette);
    (base, emissive)
}

fn agent_colors(agent: &AgentVisual, palette: ColorPaletteMode) -> (Color, Color) {
    agent_colors_from_params(&canonical_agent_visual_params(agent), palette)
}

fn close_on_esc(
    mut exit_events: MessageWriter<AppExit>,
    keyboard: Res<ButtonInput<KeyCode>>,
    state: Res<SnapshotState>,
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        // Escape first clears an active selection (handled by
        // `handle_selection_input`); only exit when nothing is selected.
        if state.selection_center.is_some() {
            return;
        }
        exit_events.write(AppExit::Success);
    }
}

pub fn render_png_offscreen(world: &WorldState, width: u32, height: u32) -> Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(anyhow!(
            "zero-sized Bevy CPU projection {width}x{height} is rejected"
        ));
    }
    let snapshot = WorldSnapshot::from_world(world)
        .ok_or_else(|| anyhow!("unable to build world snapshot for Bevy render"))?;

    let mut image = ImageBuffer::<ImgRgba<u8>, Vec<u8>>::new(width, height);

    let terrain_w = snapshot.terrain_color.width.max(1);
    let terrain_h = snapshot.terrain_color.height.max(1);

    for y in 0..height {
        let tile_y = (terrain_h as u64 - 1)
            .saturating_sub(((y as u64) * terrain_h as u64) / height as u64)
            as u32;
        for x in 0..width {
            let tile_x =
                ((x as u64) * terrain_w as u64 / width as u64).min((terrain_w - 1) as u64) as u32;
            let idx = ((tile_y * terrain_w) + tile_x) as usize * 4;
            let px = ImgRgba([
                snapshot.terrain_color.pixels[idx],
                snapshot.terrain_color.pixels[idx + 1],
                snapshot.terrain_color.pixels[idx + 2],
                255,
            ]);
            image.put_pixel(x, y, px);
        }
    }

    let scale_x = width as f32 / snapshot.world_size.x.max(1.0);
    let scale_y = height as f32 / snapshot.world_size.y.max(1.0);
    let radius_px = snapshot.agent_radius * scale_x.min(scale_y);

    for agent in &snapshot.agents {
        let center_x = (agent.position.x * scale_x).round() as i32;
        let center_y = ((snapshot.world_size.y - agent.position.y) * scale_y).round() as i32;
        let radius = radius_px.ceil() as i32;
        let (base_color, _) = agent_colors(agent, ColorPaletteMode::Natural);
        let rgba = color_to_rgba(base_color);
        for dy in -radius..=radius {
            let py = center_y + dy;
            if py < 0 || py >= height as i32 {
                continue;
            }
            for dx in -radius..=radius {
                let px = center_x + dx;
                if px < 0 || px >= width as i32 {
                    continue;
                }
                let dist = ((dx as f32).powi(2) + (dy as f32).powi(2)).sqrt();
                if dist <= radius_px {
                    image.put_pixel(
                        px as u32,
                        py as u32,
                        ImgRgba([rgba[0], rgba[1], rgba[2], rgba[3]]),
                    );
                }
            }
        }
    }

    let mut bytes = Vec::new();
    {
        let mut cursor = Cursor::new(&mut bytes);
        image.write_to(&mut cursor, image::ImageFormat::Png)?;
    }
    Ok(bytes)
}

fn spawn_simulation_driver(
    world: Arc<Mutex<WorldState>>,
    simulation_step: WorldStepDriver,
    command_drain: CommandDrainFn,
    controls: SimulationControl,
    running: Arc<AtomicBool>,
    worker_failures: mpsc::Sender<BevyLifecycleFailure>,
) -> Result<BevyWorker> {
    thread::Builder::new()
        .name("scriptbots-bevy-simulation".into())
        .spawn(move || {
            run_reported_worker("simulation worker", &worker_failures, &running, || {
                let mut last = Instant::now();
                let mut accumulator = 0.0f32;

                while running.load(Ordering::Acquire) {
                    let now = Instant::now();
                    let mut dt = (now - last).as_secs_f32();
                    last = now;
                    if !dt.is_finite() || dt > 0.25 {
                        dt = 0.25;
                    }

                    let mut latched_step_failure = None;
                    {
                        let mut world_guard = world.lock().map_err(|error| {
                            anyhow!("world mutex poisoned in Bevy simulation worker: {error}")
                        })?;
                        if let Some(error) = world_guard.latched_step_error() {
                            latched_step_failure = Some(format!(
                                "Simulation stopped after a terminal step failure: {error}"
                            ));
                        } else {
                            for command in (command_drain.as_ref())() {
                                match apply_control_command(&mut world_guard, command) {
                                    Ok(ControlDisposition::WorldApplied) => {}
                                    Ok(ControlDisposition::Playback(command)) => {
                                        controls.update(|state| {
                                            apply_simulation_command_to_state(state, &command)
                                        });
                                    }
                                    Err(error) => {
                                        warn!(%error, "Bevy rejected a drained control command");
                                    }
                                }
                            }
                        }
                    }
                    if let Some(reason) = latched_step_failure {
                        controls.update(|state| {
                            apply_auto_pause_to_state(state, &reason);
                        });
                        accumulator = 0.0;
                        thread::sleep(Duration::from_millis(4));
                        continue;
                    }

                    let (paused, speed, step_once) = {
                        let mut paused = false;
                        let mut speed = 1.0;
                        let mut step_once = false;
                        let control_available = controls.update(|state| {
                            paused = state.paused;
                            speed = state.speed_multiplier.clamp(MIN_SPEED, MAX_SPEED);
                            if state.pending_steps > 0 {
                                step_once = true;
                                state.pending_steps -= 1;
                                state.paused = true;
                                state.auto_pause_reason = None;
                            }
                        });
                        if !control_available {
                            paused = true;
                            speed = 0.0;
                            step_once = false;
                        }
                        (paused, speed, step_once)
                    };

                    if paused && !step_once {
                        thread::sleep(Duration::from_millis(4));
                        continue;
                    }

                    if !step_once {
                        accumulator += dt * speed.max(0.0);
                        let max_accumulator = SIM_TICK_INTERVAL * MAX_SIM_STEPS_PER_FRAME as f32;
                        accumulator = accumulator.min(max_accumulator);
                    }

                    let mut steps = if step_once {
                        accumulator = 0.0;
                        1
                    } else {
                        let mut queued = 0usize;
                        while accumulator >= SIM_TICK_INTERVAL && queued < MAX_SIM_STEPS_PER_FRAME {
                            accumulator -= SIM_TICK_INTERVAL;
                            queued += 1;
                        }
                        queued
                    };

                    if steps == 0 && !step_once && speed <= MIN_SPEED {
                        thread::sleep(Duration::from_millis(4));
                        continue;
                    }

                    if steps == 0 && step_once {
                        steps = 1;
                    }

                    let mut step_failure = None;
                    for _ in 0..steps {
                        if let Err(error) = (simulation_step)() {
                            step_failure = Some(format!(
                                "Simulation stopped after a terminal step failure: {error}"
                            ));
                            break;
                        }
                    }

                    let (control, agent_count, max_age, spike_hits) = {
                        let world_guard = world.lock().map_err(|error| {
                            anyhow!("world mutex poisoned in Bevy simulation worker: {error}")
                        })?;
                        (
                            world_guard.config().control.clone(),
                            world_guard.agent_count(),
                            world_guard.last_max_age(),
                            world_guard.last_spike_hits(),
                        )
                    };

                    let step_failed = step_failure.is_some();
                    let mut reason = step_failure;
                    if reason.is_none() {
                        if control.auto_pause_on_spike_hit && spike_hits > 0 {
                            reason = Some(format!("Spike hits detected ({spike_hits})"));
                        } else if let Some(age_limit) = control.auto_pause_age_above {
                            if max_age >= age_limit {
                                reason = Some(format!("Max age {max_age} ≥ {age_limit}"));
                            }
                        } else if let Some(limit) = control.auto_pause_population_below
                            && agent_count as u32 <= limit
                        {
                            reason = Some(format!("Population {agent_count} ≤ {limit}"));
                        }
                    }

                    if let Some(reason) = reason {
                        controls.update(|state| {
                            apply_auto_pause_to_state(state, &reason);
                        });
                        if step_failed {
                            warn!(%reason, "Bevy simulation paused after terminal step failure");
                        } else {
                            info!(%reason, "Bevy simulation auto-paused");
                        }
                    } else if steps > 0 {
                        controls.update(|state| {
                            state.auto_pause_reason = None;
                        });
                    }

                    if steps == 0 {
                        thread::sleep(Duration::from_millis(2));
                    }
                }
                Ok(())
            })
        })
        .context("failed to spawn Bevy simulation worker")
}

fn color_to_rgba(color: Color) -> [u8; 4] {
    let srgba = color.to_srgba();
    [
        (srgba.red * 255.0).round().clamp(0.0, 255.0) as u8,
        (srgba.green * 255.0).round().clamp(0.0, 255.0) as u8,
        (srgba.blue * 255.0).round().clamp(0.0, 255.0) as u8,
        (srgba.alpha * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{MinimalPlugins, prelude::Messages};
    use scriptbots_core::ScriptBotsConfig;
    use std::sync::{Arc, Mutex};

    fn worker_group(snapshot: BevyWorker, simulation: BevyWorker) -> BevyWorkerGroup {
        BevyWorkerGroup {
            running: Arc::new(AtomicBool::new(true)),
            snapshot: Some(snapshot),
            simulation: Some(simulation),
        }
    }

    #[test]
    fn gpu_probe_reports_sane_fields_or_none() {
        // On GPU-less CI the probe may legitimately return None; when an
        // adapter exists the report must be complete and coherent.
        if let Some(report) = probe_gpu_capability() {
            assert!(!report.name.is_empty(), "adapter name must be non-empty");
            assert!(!report.backend.is_empty());
            assert!(report.max_texture_2d.unwrap_or(0) >= 2048);
        }
    }

    /// Explicit tiers are honoured and Auto lands on the ladder.
    ///
    /// Driven through the pure `_for_gpu` form against a fixed HARDWARE adapter.
    /// It previously called `resolve_effective_render_settings`, which probes
    /// the real GPU, so its result depended on the host: on a software-rasterizer
    /// worker the software clamp (bd-2z0.14.3.3 item 4) correctly turns an
    /// explicit Low into Potato and the assertion failed. That was the test
    /// being environment-dependent rather than the clamp being wrong — the
    /// property it means to check belongs to the resolution logic, not to
    /// whatever adapter the machine happens to have. The live probe itself stays
    /// covered by the capability-report test above.
    #[test]
    fn effective_settings_honor_explicit_tiers_and_auto() {
        let hardware = gpu_info("Apple M4", GpuClass::Discrete);

        let explicit = resolve_effective_render_settings_for_gpu(
            &RenderSettings {
                quality: Some(RenderQuality::Low),
                ..RenderSettings::default()
            },
            Some(hardware.clone()),
        );
        assert_eq!(explicit.tier, RenderQuality::Low);
        assert!(!explicit.features.ssao, "Low has no SSAO");

        let auto =
            resolve_effective_render_settings_for_gpu(&RenderSettings::default(), Some(hardware));
        assert!(
            matches!(
                auto.tier,
                RenderQuality::Potato
                    | RenderQuality::Low
                    | RenderQuality::Medium
                    | RenderQuality::High
            ),
            "auto resolves onto the ladder, never Ultra"
        );
    }

    /// A fixed adapter so governor tests never depend on the host's real GPU.
    fn discrete_gpu_fixture() -> GpuInfo {
        GpuInfo {
            name: "ScriptBots Test Adapter".to_string(),
            backend: "Metal".to_string(),
            class: GpuClass::Discrete,
            vram_bytes: None,
            max_texture_2d: Some(16_384),
            timestamp_queries: true,
            vendor_id: None,
            device_id: None,
            driver: None,
            driver_info: None,
        }
    }

    /// The resolved tier cannot answer "was this asked for, or inferred?".
    /// Auto resolves to a concrete rung and then looks exactly like an explicit
    /// request, which is why the governor needs `requested` retained (bd-2z0.14.3.3).
    #[test]
    fn resolved_settings_record_the_requested_tier_not_just_the_effective_one() {
        let explicit = resolve_effective_render_settings_for_gpu(
            &RenderSettings {
                quality: Some(RenderQuality::Low),
                ..RenderSettings::default()
            },
            Some(discrete_gpu_fixture()),
        );
        assert_eq!(explicit.requested, RenderQuality::Low);
        assert_eq!(explicit.tier, RenderQuality::Low);

        let auto = resolve_effective_render_settings_for_gpu(
            &RenderSettings::default(),
            Some(discrete_gpu_fixture()),
        );
        assert_eq!(
            auto.requested,
            RenderQuality::Auto,
            "Auto must survive resolution; the effective tier alone loses it"
        );
        assert_ne!(
            auto.tier,
            RenderQuality::Auto,
            "Auto must still resolve to a concrete rung"
        );
    }

    /// A fixed adapter so tier-clamp tests never depend on the host's real GPU.
    fn gpu_info(name: &str, class: GpuClass) -> GpuInfo {
        GpuInfo {
            name: name.to_string(),
            backend: "Vulkan".to_string(),
            class,
            vram_bytes: None,
            max_texture_2d: Some(16_384),
            timestamp_queries: true,
            vendor_id: None,
            device_id: None,
            driver: None,
            driver_info: None,
        }
    }

    /// A software rasterizer is forced to Potato, even over an explicit request.
    ///
    /// Overriding the operator is the point: llvmpipe cannot deliver Ultra, and
    /// letting the tier claim otherwise is exactly the false-capability problem
    /// this bead names. The override is loud rather than quiet.
    #[test]
    fn a_software_adapter_is_forced_to_potato_whatever_was_requested() {
        let software = gpu_info("llvmpipe (LLVM 21.1.8, 256 bits)", GpuClass::Software);
        for requested in [
            RenderQuality::Auto,
            RenderQuality::Ultra,
            RenderQuality::High,
            RenderQuality::Medium,
            RenderQuality::Low,
            RenderQuality::Potato,
        ] {
            let settings = RenderSettings {
                quality: Some(requested),
                ..RenderSettings::default()
            };
            let effective =
                resolve_effective_render_settings_for_gpu(&settings, Some(software.clone()));
            assert_eq!(
                effective.tier,
                RenderQuality::Potato,
                "requested {requested:?} on a software rasterizer must resolve to Potato"
            );
            assert!(
                !effective.features.bloom && !effective.features.ssao,
                "the imposed tier must carry Potato's feature row, not the requested one"
            );
        }
    }

    /// The clamp must not rewrite `requested`.
    ///
    /// `requested` is what separates "Auto resolved here" from "the operator
    /// chose this", and the governor keys on it. Rewriting it to Potato would
    /// silently disable adaptation for Auto launches by making them look
    /// explicit — a bug that would only show as a renderer that never adapts.
    #[test]
    fn forcing_potato_preserves_what_the_operator_actually_requested() {
        let software = gpu_info("llvmpipe", GpuClass::Software);

        let auto = resolve_effective_render_settings_for_gpu(
            &RenderSettings::default(),
            Some(software.clone()),
        );
        assert_eq!(auto.tier, RenderQuality::Potato);
        assert_eq!(
            auto.requested,
            RenderQuality::Auto,
            "an Auto launch must still read as Auto after the clamp"
        );

        let explicit = resolve_effective_render_settings_for_gpu(
            &RenderSettings {
                quality: Some(RenderQuality::Ultra),
                ..RenderSettings::default()
            },
            Some(software),
        );
        assert_eq!(explicit.tier, RenderQuality::Potato);
        assert_eq!(
            explicit.requested,
            RenderQuality::Ultra,
            "the operator's rejected choice must remain visible, not be rewritten \
             into agreement with what was imposed"
        );
    }

    /// Real hardware is untouched by the clamp, so it narrows nothing that works.
    #[test]
    fn hardware_adapters_keep_the_tier_they_resolved_to() {
        for class in [GpuClass::Discrete, GpuClass::Integrated] {
            let effective = resolve_effective_render_settings_for_gpu(
                &RenderSettings {
                    quality: Some(RenderQuality::Ultra),
                    ..RenderSettings::default()
                },
                Some(gpu_info("Apple M4", class)),
            );
            assert_eq!(
                effective.tier,
                RenderQuality::Ultra,
                "{class:?} must keep an explicitly requested Ultra"
            );
        }
    }

    #[test]
    fn adaptive_governor_engages_only_for_auto_quality() {
        let auto = resolve_effective_render_settings_for_gpu(
            &RenderSettings::default(),
            Some(discrete_gpu_fixture()),
        );
        assert!(
            AdaptiveQualityGovernor::for_launch(&auto).is_active(),
            "an Auto launch must adapt to measured frame times"
        );

        for pinned in [
            RenderQuality::Potato,
            RenderQuality::Low,
            RenderQuality::Medium,
            RenderQuality::High,
            RenderQuality::Ultra,
        ] {
            let explicit = resolve_effective_render_settings_for_gpu(
                &RenderSettings {
                    quality: Some(pinned),
                    ..RenderSettings::default()
                },
                Some(discrete_gpu_fixture()),
            );
            assert!(
                !AdaptiveQualityGovernor::for_launch(&explicit).is_active(),
                "explicit {pinned:?} must never be overridden by the governor"
            );
        }
    }

    #[test]
    fn manual_tier_override_permanently_disengages_the_governor() {
        let auto = resolve_effective_render_settings_for_gpu(
            &RenderSettings::default(),
            Some(discrete_gpu_fixture()),
        );
        let mut governor = AdaptiveQualityGovernor::for_launch(&auto);
        assert!(governor.is_active());

        governor.relinquish_to_operator();
        assert!(
            !governor.is_active(),
            "the operator's manual pick must not be silently overwritten later"
        );

        // Idempotent: releasing twice is not an error and does not re-engage.
        governor.relinquish_to_operator();
        assert!(!governor.is_active());
    }

    /// Guards the seam this bead exists to close: before it, the governor had
    /// no production consumer, so the tier was startup-only no matter how badly
    /// frames ran. If the wiring is deleted, this fails.
    #[test]
    fn production_update_schedule_drives_the_adaptive_governor() {
        let production = include_str!("lib.rs");
        assert!(
            production.contains("drive_adaptive_quality,"),
            "drive_adaptive_quality must stay registered in the Update schedule"
        );
        assert!(
            production.contains("AdaptiveQualityGovernor::for_launch("),
            "run_renderer must construct the governor resource"
        );
    }

    #[test]
    fn bevy_error_exit_is_not_reported_as_success() {
        let error = app_exit_result(AppExit::error()).expect_err("error exit must propagate");
        assert!(error.to_string().contains("error code 1"));
        app_exit_result(AppExit::Success).expect("success exit");
    }

    #[test]
    fn post_start_worker_failure_cancels_siblings_and_requests_bevy_error_exit() {
        let (failure_tx, failure_rx) = mpsc::channel();
        let first_failure = Arc::new(Mutex::new(None));
        let running = Arc::new(AtomicBool::new(true));
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_non_send_resource(BevyLifecycleFailureInbox {
                receiver: failure_rx,
                first_failure: Arc::clone(&first_failure),
            })
            .add_systems(Update, poll_bevy_lifecycle_failures);

        app.update();
        assert!(app.should_exit().is_none());

        let worker_running = Arc::clone(&running);
        let worker = thread::spawn(move || {
            run_reported_worker("snapshot worker", &failure_tx, &worker_running, || {
                Err(anyhow!("injected post-startup failure"))
            })
        });
        worker
            .join()
            .expect("reported worker must not panic")
            .expect_err("injected worker failure must propagate");
        app.update();

        assert!(!running.load(Ordering::Acquire));
        assert!(matches!(app.should_exit(), Some(AppExit::Error(_))));
        assert_eq!(
            first_failure.lock().expect("first failure").as_deref(),
            Some("Bevy snapshot worker failed: injected post-startup failure")
        );
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn worker_panic_is_converted_into_a_reported_failure() {
        let (failure_tx, failure_rx) = mpsc::channel();
        let running = Arc::new(AtomicBool::new(true));
        let worker_running = Arc::clone(&running);
        let worker = thread::spawn(move || {
            run_reported_worker(
                "simulation worker",
                &failure_tx,
                &worker_running,
                || -> Result<()> { panic!("injected supervised panic") },
            )
        });

        let error = worker
            .join()
            .expect("panic must be caught at worker boundary")
            .expect_err("caught panic must remain an error");
        assert!(error.to_string().contains("injected supervised panic"));
        assert!(!running.load(Ordering::Acquire));
        let failure = failure_rx.recv().expect("reported worker failure");
        assert_eq!(failure.component, "simulation worker");
        assert!(failure.detail.contains("injected supervised panic"));
    }

    #[test]
    fn control_health_transition_reports_once_and_requests_bevy_error_exit() {
        let (failure_tx, failure_rx) = mpsc::channel();
        let first_failure = Arc::new(Mutex::new(None));
        let checks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let running = Arc::new(AtomicBool::new(true));
        let health_checks = Arc::clone(&checks);
        let health: ControlHealthFn = Arc::new(move || {
            if health_checks.fetch_add(1, Ordering::AcqRel) == 0 {
                Ok(())
            } else {
                Err("control runtime stopped".to_string())
            }
        });
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_non_send_resource(BevyLifecycleFailureInbox {
                receiver: failure_rx,
                first_failure: Arc::clone(&first_failure),
            })
            .insert_resource(ControlHealthMonitor {
                check: Some(health),
                failures: failure_tx,
                running: Arc::clone(&running),
                failure_reported: false,
            })
            .add_systems(
                Update,
                (poll_control_health, poll_bevy_lifecycle_failures).chain(),
            );

        app.update();
        assert!(app.should_exit().is_none());
        app.update();

        assert!(matches!(app.should_exit(), Some(AppExit::Error(_))));
        assert!(!running.load(Ordering::Acquire));
        assert_eq!(
            first_failure.lock().expect("first failure").as_deref(),
            Some("Bevy control plane failed: control runtime stopped")
        );
        app.update();
        assert_eq!(checks.load(Ordering::Acquire), 2);
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn simulation_worker_panic_is_propagated() {
        let snapshot = thread::spawn(|| -> Result<()> { Ok(()) });
        let simulation =
            thread::spawn(|| -> Result<()> { panic!("injected simulation-worker panic") });

        let error = worker_group(snapshot, simulation)
            .stop_and_join()
            .expect_err("simulation panic must propagate");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("simulation worker panicked"),
            "{rendered}"
        );
        assert!(
            rendered.contains("injected simulation-worker panic"),
            "{rendered}"
        );
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn snapshot_worker_panic_is_propagated() {
        let snapshot = thread::spawn(|| -> Result<()> { panic!("injected snapshot-worker panic") });
        let simulation = thread::spawn(|| -> Result<()> { Ok(()) });

        let error = worker_group(snapshot, simulation)
            .stop_and_join()
            .expect_err("snapshot panic must propagate");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("snapshot worker panicked"), "{rendered}");
        assert!(
            rendered.contains("injected snapshot-worker panic"),
            "{rendered}"
        );
    }

    #[test]
    fn worker_shutdown_clears_the_shared_running_flag_before_join() {
        let running = Arc::new(AtomicBool::new(true));
        let snapshot_flag = Arc::clone(&running);
        let simulation_flag = Arc::clone(&running);
        let snapshot = thread::spawn(move || -> Result<()> {
            while snapshot_flag.load(Ordering::Acquire) {
                thread::yield_now();
            }
            Ok(())
        });
        let simulation = thread::spawn(move || -> Result<()> {
            while simulation_flag.load(Ordering::Acquire) {
                thread::yield_now();
            }
            Ok(())
        });
        let workers = BevyWorkerGroup {
            running: Arc::clone(&running),
            snapshot: Some(snapshot),
            simulation: Some(simulation),
        };

        workers.stop_and_join().expect("clean worker shutdown");
        assert!(!running.load(Ordering::Acquire));
    }

    #[test]
    fn bevy_scene_cpu_surrogate_raster_produces_png() -> Result<()> {
        let config = ScriptBotsConfig::default();
        let mut world = WorldState::new(config).expect("world initialization");
        for _ in 0..32 {
            world
                .step()
                .expect("offscreen-render test world should accept each step");
        }
        for (width, height) in [(640, 360), (1920, 1080), (2560, 1440), (3840, 2160)] {
            let png = render_png_offscreen(&world, width, height)?;
            assert!(
                png.len() > 4096,
                "expected non-trivial PNG output for {}x{}",
                width,
                height
            );
            assert_eq!(
                &png[0..8],
                b"\x89PNG\r\n\x1a\n",
                "invalid PNG header for {}x{} capture",
                width,
                height
            );
        }
        Ok(())
    }

    #[test]
    fn follow_button_toggles_mode() {
        let mut app = App::new();
        app.add_systems(
            Update,
            (
                handle_follow_button_interactions,
                update_follow_button_colors,
            ),
        );
        app.insert_resource(CameraRig::default());
        app.world_mut().resource_mut::<CameraRig>().follow_mode = FollowMode::Off;

        let button = app
            .world_mut()
            .spawn((
                Button,
                FollowButton {
                    mode: FollowMode::Selected,
                },
                Interaction::Pressed,
            ))
            .id();

        app.update();

        let rig = app.world().resource::<CameraRig>();
        assert_eq!(rig.follow_mode, FollowMode::Selected);

        app.world_mut().entity_mut(button).insert(Interaction::None);
        app.update();
    }

    #[test]
    fn clear_selection_button_submits_command() {
        let mut app = App::new();
        app.add_systems(Update, handle_clear_selection_button);

        let logs: Arc<Mutex<Vec<SelectionMode>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = logs.clone();
        app.insert_resource(CommandSubmitter {
            submit: Arc::new(move |command| {
                if let ControlCommand::UpdateSelection(update) = command {
                    sink.lock().unwrap().push(update.mode);
                }
                Some("test-cmd".to_owned())
            }),
        });
        app.insert_resource(CameraRig::default());

        app.world_mut()
            .spawn((Button, ClearSelectionButton, Interaction::Pressed));

        app.update();

        let entries = logs.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], SelectionMode::Clear);

        println!("Captured command log entries: {:?}", *entries);
    }

    /// The camera state this bead cares about: what it follows, where it is
    /// panned, and whether it was told to recentre.
    const STARTING_PAN: Vec2 = Vec2::new(37.0, -19.0);

    /// Drive the clear-selection button with a submitter that answers `accepted`
    /// and report the camera state afterwards.
    ///
    /// `pan` and `recenter_now` are set to values `CameraRig::default()` does
    /// NOT produce, so "unchanged" is a real observation rather than the
    /// absence of a default. That distinction is not theoretical: the default
    /// has `recenter_now: true`, and letting `..default()` supply it made the
    /// first version of this test assert against the value it had silently
    /// inherited. `follow_mode` does coincide with the default, because
    /// `Selected` is the only honest starting point for clearing a selection;
    /// the accepted-case control below is what proves the handler moves it.
    ///
    /// Fields are returned individually because `CameraRig` is not `Clone`, and
    /// a production type should not grow a derive to suit a test.
    fn clear_selection_button_camera_after(accepted: bool) -> (FollowMode, Vec2, bool) {
        let mut app = App::new();
        app.add_systems(Update, handle_clear_selection_button);
        app.insert_resource(CommandSubmitter {
            submit: Arc::new(move |_| accepted.then(|| "test-cmd".to_owned())),
        });
        app.insert_resource(CameraRig {
            follow_mode: FollowMode::Selected,
            pan: STARTING_PAN,
            recenter_now: false,
            ..CameraRig::default()
        });
        app.world_mut()
            .spawn((Button, ClearSelectionButton, Interaction::Pressed));
        app.update();
        let rig = app.world().resource::<CameraRig>();
        (rig.follow_mode, rig.pan, rig.recenter_now)
    }

    /// A refused clear must leave the camera exactly where it was.
    ///
    /// This is the behavioural form of the guard that slice 1 could only assert
    /// by scanning source. It runs the real system through a real refusing
    /// submitter, so it fails if the gate stops working for any reason rather
    /// than only if the `if !` is textually deleted.
    #[test]
    fn a_refused_clear_selection_does_not_move_the_camera() {
        let (follow_mode, pan, recenter_now) = clear_selection_button_camera_after(false);
        assert_eq!(
            follow_mode,
            FollowMode::Selected,
            "the simulation never received the clear, so the camera must keep following"
        );
        assert_eq!(
            pan, STARTING_PAN,
            "a refused clear must not reset the operator's pan"
        );
        assert!(!recenter_now, "a refused clear must not trigger a recentre");
    }

    /// Positive control: an ACCEPTED clear does move the camera.
    ///
    /// Without this, the test above would pass equally well against a handler
    /// that had simply stopped touching the camera at all.
    #[test]
    fn an_accepted_clear_selection_releases_the_camera() {
        let (follow_mode, pan, recenter_now) = clear_selection_button_camera_after(true);
        assert_eq!(
            follow_mode,
            FollowMode::Off,
            "an accepted clear stops following the now-cleared selection"
        );
        assert_eq!(pan, Vec2::ZERO, "an accepted clear resets the pan");
        assert!(recenter_now, "an accepted clear recentres");
    }

    #[test]
    fn playback_button_updates_controls() {
        let mut app = App::new();
        app.add_systems(Update, handle_playback_buttons);
        let controls = SimulationControl::new();
        app.insert_resource(controls.clone());

        let button = app
            .world_mut()
            .spawn((
                Button,
                PlaybackButton {
                    action: PlaybackAction::SpeedUp,
                },
                Interaction::Pressed,
            ))
            .id();

        app.update();

        let snapshot = controls.snapshot();
        assert!(
            snapshot.speed_multiplier > 1.0,
            "speed should accelerate after speed-up button"
        );

        app.world_mut().entity_mut(button).insert(Interaction::None);
        app.update();
    }

    #[test]
    fn playback_shortcuts_toggle_pause() {
        let mut app = App::new();
        app.add_systems(Update, handle_playback_shortcuts);
        let controls = SimulationControl::new();
        app.insert_resource(controls.clone());
        app.insert_resource(ButtonInput::<KeyCode>::default());

        {
            let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            keys.press(KeyCode::Space);
        }

        app.update();

        {
            let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            keys.release(KeyCode::Space);
        }

        let snapshot = controls.snapshot();
        assert!(
            snapshot.paused,
            "spacebar shortcut should toggle pause state to true"
        );
    }

    #[test]
    fn poisoned_simulation_controls_latch_a_fail_closed_pause() {
        let controls = SimulationControl::new();
        let poison_target = controls.clone();
        let panic = std::panic::catch_unwind(move || {
            let _guard = poison_target
                .0
                .lock()
                .expect("control mutex starts healthy");
            panic!("deliberately poison the simulation-control mutex");
        });
        assert!(panic.is_err(), "test must actually poison the mutex");

        let updated = controls.update(|state| {
            state.paused = false;
            state.pending_steps = 1;
        });
        assert!(
            !updated,
            "a poisoned control plane must reject updates instead of silently losing them"
        );
        let snapshot = controls.snapshot();
        assert!(
            snapshot.paused,
            "the science driver must fail closed after control-plane poisoning"
        );
        assert_eq!(
            snapshot.auto_pause_reason.as_deref(),
            Some("Bevy simulation control mutex poisoned; science driver stopped")
        );
    }

    #[test]
    fn cpu_projection_rejects_zero_dimensions() {
        let world = WorldState::new(ScriptBotsConfig::default()).expect("world init");
        let error = render_png_offscreen(&world, 0, 64)
            .expect_err("zero-width projection must not be silently widened to one pixel");
        assert!(error.to_string().contains("zero-sized"));
    }

    #[test]
    fn auto_pause_preserves_speed_and_records_the_trigger() {
        let mut state = SimControlData {
            paused: false,
            speed_multiplier: 3.5,
            pending_steps: 1,
            auto_pause_reason: Some("stale reason".to_owned()),
        };

        apply_auto_pause_to_state(&mut state, "Spike hits detected (2)");

        assert!(state.paused);
        assert_eq!(state.speed_multiplier, 3.5);
        assert_eq!(state.pending_steps, 0);
        assert_eq!(
            state.auto_pause_reason.as_deref(),
            Some("Spike hits detected (2)")
        );
    }

    fn consume_driver_step_request(state: &mut SimControlData) -> usize {
        if state.pending_steps > 0 {
            state.pending_steps -= 1;
            state.paused = true;
            state.auto_pause_reason = None;
            1
        } else {
            0
        }
    }

    fn current_bevy_step_count(queued_command_arrives_before_driver: bool) -> usize {
        let mut state = SimControlData {
            paused: true,
            // Production has a CommandSubmitter, so the UI does not also set
            // the local edge. The drained command is the single authority.
            pending_steps: 0,
            ..SimControlData::default()
        };
        let queued = SimulationCommand {
            paused: Some(true),
            speed_multiplier: None,
            step_once: true,
        };

        if queued_command_arrives_before_driver {
            apply_simulation_command_to_state(&mut state, &queued);
        }
        let mut steps = consume_driver_step_request(&mut state);
        if !queued_command_arrives_before_driver {
            apply_simulation_command_to_state(&mut state, &queued);
            steps += consume_driver_step_request(&mut state);
        }
        steps
    }

    #[test]
    fn target_bevy_step_is_exactly_once_for_every_queue_interleaving() {
        let observed = [
            current_bevy_step_count(true),
            current_bevy_step_count(false),
        ];
        assert_eq!(
            observed,
            [1, 1],
            "the queued step command must be the sole authority for every interleaving"
        );
    }

    fn run_step_button_with_submitter(accepted: bool, pending_steps: u64) -> SimControlData {
        let mut app = App::new();
        app.add_systems(Update, handle_playback_buttons);
        let controls = SimulationControl::new();
        controls.update(|state| {
            state.paused = true;
            state.pending_steps = pending_steps;
        });
        app.insert_resource(controls.clone());
        app.insert_resource(CommandSubmitter {
            submit: Arc::new(move |command| {
                let ControlCommand::UpdateSimulation(command) = command else {
                    panic!("step button submitted the wrong command kind");
                };
                assert!(command.step_once, "step command lost its edge");
                accepted.then(|| "test-cmd".to_owned())
            }),
        });
        app.world_mut().spawn((
            Button,
            PlaybackButton {
                action: PlaybackAction::Step,
            },
            Interaction::Pressed,
        ));
        app.update();
        controls.0.lock().expect("simulation controls").clone()
    }

    #[test]
    fn step_button_preserves_pending_work_and_falls_back_on_enqueue_rejection() {
        let accepted = run_step_button_with_submitter(true, 0);
        assert_eq!(
            accepted.pending_steps, 0,
            "an accepted queued step must remain the sole local authority"
        );

        let pending = run_step_button_with_submitter(true, 1);
        assert_eq!(
            pending.pending_steps, 1,
            "submitting another step must not erase a drained but unconsumed step"
        );

        let rejected = run_step_button_with_submitter(false, 0);
        assert_eq!(
            rejected.pending_steps, 1,
            "a rejected queued step must fall back to the local driver edge"
        );
    }

    #[test]
    fn two_queued_step_edges_produce_two_driver_steps() {
        let mut state = SimControlData {
            paused: true,
            ..SimControlData::default()
        };
        let queued = SimulationCommand {
            paused: Some(true),
            speed_multiplier: None,
            step_once: true,
        };
        apply_simulation_command_to_state(&mut state, &queued);
        apply_simulation_command_to_state(&mut state, &queued);
        assert_eq!(state.pending_steps, 2, "step edges must not coalesce");
        assert_eq!(consume_driver_step_request(&mut state), 1);
        assert_eq!(consume_driver_step_request(&mut state), 1);
        assert_eq!(consume_driver_step_request(&mut state), 0);
    }

    #[test]
    fn hud_overlay_populates_metrics() -> Result<()> {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_systems(Update, update_hud);

        let controls = SimulationControl::new();
        controls.update(|state| {
            state.paused = true;
            state.speed_multiplier = 0.0;
            state.auto_pause_reason = Some("Spike hits detected (3)".to_string());
        });
        app.insert_resource(controls);
        app.insert_resource(CameraRig::default());
        app.insert_resource(TonemappingState::default());
        app.insert_resource(AccessibilityState::new());

        let config = ScriptBotsConfig::default();
        let mut world = WorldState::new(config).expect("world initialization");
        for _ in 0..48 {
            world
                .step()
                .expect("HUD test world should accept each simulation step");
        }
        let snapshot = WorldSnapshot::from_world(&world).expect("world snapshot");

        let state = SnapshotState {
            latest: Some(snapshot.clone()),
            last_applied_revision: snapshot.revision,
            last_applied_tick: snapshot.tick,
            last_applied_palette: Some(ColorPaletteMode::Natural),
            last_reported_tick: 0,
            focus_point: Vec2::new(snapshot.world_size.x * 0.5, snapshot.world_size.y * 0.5),
            world_size: snapshot.world_size,
            world_center: Vec2::new(snapshot.world_size.x * 0.5, snapshot.world_size.y * 0.5),
            selection_center: None,
            selection_bounds: None,
            oldest_position: None,
            first_agent_position: snapshot.agents.first().map(|agent| agent.position),
            hud_prev_tick: 0,
            hud_prev_time: 0.0,
            sim_rate: 0.0,
        };
        app.insert_resource(state);

        fn spawn_label(app: &mut App) -> Entity {
            app.world_mut().spawn(Text::default()).id()
        }

        let hud = HudElements {
            tick: spawn_label(&mut app),
            agents: spawn_label(&mut app),
            selection: spawn_label(&mut app),
            follow: spawn_label(&mut app),
            camera: spawn_label(&mut app),
            playback: spawn_label(&mut app),
            fps: spawn_label(&mut app),
            world: spawn_label(&mut app),
            tonemap: spawn_label(&mut app),
            palette: spawn_label(&mut app),
            events: spawn_label(&mut app),
            inspector: spawn_label(&mut app),
            history: spawn_label(&mut app),
            brain: spawn_label(&mut app),
        };
        app.insert_resource(hud);

        let hud_ids = {
            let hud_ref = app.world().resource::<HudElements>();
            (
                hud_ref.tick,
                hud_ref.agents,
                hud_ref.selection,
                hud_ref.follow,
                hud_ref.camera,
                hud_ref.playback,
                hud_ref.fps,
                hud_ref.world,
                hud_ref.tonemap,
                hud_ref.palette,
            )
        };

        app.update();

        let world = app.world();

        let tick_text = world
            .get::<Text>(hud_ids.0)
            .expect("tick text exists")
            .as_str()
            .to_string();
        assert_eq!(tick_text, format!("Tick: {}", snapshot.tick));

        let agents_text = world
            .get::<Text>(hud_ids.1)
            .expect("agents text exists")
            .as_str()
            .to_string();
        assert!(
            agents_text.starts_with("Agents: "),
            "agents text missing prefix: {agents_text}"
        );

        let selection_text = world
            .get::<Text>(hud_ids.2)
            .expect("selection text exists")
            .as_str()
            .to_string();
        assert!(
            selection_text.starts_with("Selection:"),
            "selection text missing prefix: {selection_text}"
        );

        let follow_text = world
            .get::<Text>(hud_ids.3)
            .expect("follow text exists")
            .as_str()
            .to_string();
        assert!(
            follow_text.contains("Ctrl+S sel"),
            "follow text missing shortcut hint: {follow_text}"
        );

        let camera_text = world
            .get::<Text>(hud_ids.4)
            .expect("camera text exists")
            .as_str()
            .to_string();
        assert!(
            camera_text.contains("Ctrl+W fit world"),
            "camera text missing fit-world hint: {camera_text}"
        );

        let playback_text = world
            .get::<Text>(hud_ids.5)
            .expect("playback text exists")
            .as_str()
            .to_string();
        assert!(
            playback_text.contains("Spike hits detected (3)"),
            "playback text missing auto-pause reason: {playback_text}"
        );

        let fps_text = world
            .get::<Text>(hud_ids.6)
            .expect("fps text exists")
            .as_str()
            .to_string();
        assert!(
            fps_text.starts_with("FPS:"),
            "fps text missing prefix: {fps_text}"
        );

        let world_text = world
            .get::<Text>(hud_ids.7)
            .expect("world size text exists")
            .as_str()
            .to_string();
        assert!(
            world_text.starts_with("World:"),
            "world text missing prefix: {world_text}"
        );

        let tonemap_text = world
            .get::<Text>(hud_ids.8)
            .expect("tonemap text exists")
            .as_str()
            .to_string();
        assert!(
            tonemap_text.starts_with("Tone:"),
            "tonemap text missing prefix: {tonemap_text}"
        );

        let palette_text = world
            .get::<Text>(hud_ids.9)
            .expect("palette text exists")
            .as_str()
            .to_string();
        assert!(
            palette_text.contains("press C to cycle"),
            "palette text missing cycle hint: {palette_text}"
        );

        Ok(())
    }

    #[test]
    fn follow_mode_keeps_selection_centered() -> Result<()> {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_systems(Update, control_camera);

        app.insert_resource(Time::<bevy::time::Real>::default());
        app.insert_resource(ButtonInput::<MouseButton>::default());
        app.insert_resource(ButtonInput::<KeyCode>::default());
        app.insert_resource(Messages::<MouseMotion>::default());
        app.insert_resource(Messages::<MouseWheel>::default());

        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world init");
        for _ in 0..48 {
            world
                .step()
                .expect("follow-mode test world should accept each simulation step");
        }
        let snapshot = WorldSnapshot::from_world(&world).expect("snapshot generation");
        let selection_center = Vec2::new(snapshot.world_size.x * 0.4, snapshot.world_size.y * 0.6);
        let selection_bounds = (
            selection_center - Vec2::splat(32.0),
            selection_center + Vec2::splat(32.0),
        );

        app.insert_resource(SnapshotState {
            latest: Some(snapshot.clone()),
            last_applied_revision: snapshot.revision,
            last_applied_tick: snapshot.tick,
            last_applied_palette: Some(ColorPaletteMode::Natural),
            last_reported_tick: snapshot.tick,
            focus_point: selection_center,
            world_size: snapshot.world_size,
            world_center: Vec2::new(snapshot.world_size.x * 0.5, snapshot.world_size.y * 0.5),
            selection_center: Some(selection_center),
            selection_bounds: Some(selection_bounds),
            oldest_position: Some(selection_center),
            first_agent_position: Some(selection_center),
            hud_prev_tick: snapshot.tick,
            hud_prev_time: 0.0,
            sim_rate: 0.0,
        });

        let rig = CameraRig {
            follow_mode: FollowMode::Selected,
            recenter_now: true,
            ..Default::default()
        };
        app.insert_resource(rig);

        let camera_entity = app
            .world_mut()
            .spawn((
                Transform::default(),
                GlobalTransform::default(),
                PrimaryCamera,
            ))
            .id();

        app.update();

        let rig = app.world().resource::<CameraRig>();
        let focus_delta = rig.focus_smoothed.distance(selection_center);
        let tolerance = selection_center.length().max(snapshot.world_size.length()) * 0.03;
        assert!(
            focus_delta <= tolerance,
            "follow mode should keep focus within tolerance (delta {focus_delta}, limit {tolerance})"
        );
        assert!(
            (rig.follow_mode == FollowMode::Selected),
            "follow mode should remain Selected"
        );

        let transform = app
            .world()
            .entity(camera_entity)
            .get::<Transform>()
            .expect("camera transform");
        let expected_center = Vec3::new(
            selection_center.x - snapshot.world_size.x * 0.5,
            0.0,
            snapshot.world_size.y * 0.5 - selection_center.y,
        );
        let distance_expected = transform.translation.distance(expected_center);
        assert!(
            (distance_expected - rig.distance_smoothed).abs() < 1.0,
            "camera distance should match rig distance ({distance_expected} vs {})",
            rig.distance_smoothed
        );

        let forward = transform.forward().normalize_or_zero();
        let toward_center = (expected_center - transform.translation).normalize_or_zero();
        assert!(
            forward.dot(toward_center) > 0.99,
            "camera should look at focus center (dot {})",
            forward.dot(toward_center)
        );

        Ok(())
    }

    #[test]
    fn test_bevy_presentation_snapshot_decoupling() -> Result<()> {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);

        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world init");
        for _ in 0..42 {
            let _ = world.step();
        }
        let snapshot = WorldSnapshot::from_world(&world).expect("snapshot generation");
        let snapshot_state = SnapshotState {
            latest: Some(snapshot),
            ..SnapshotState::default()
        };
        app.insert_resource(snapshot_state);
        app.insert_resource(AgentRegistry::default());
        app.insert_resource(TerrainChunkRegistry::default());
        app.insert_resource(AgentMeshes::default());
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());
        app.insert_resource(ReflectionProbeAssets::default());
        app.insert_resource(AccessibilityState::default());

        app.add_systems(Update, sync_world);

        // First presentation frame update
        app.update();
        let state = app.world().resource::<SnapshotState>();
        assert_eq!(state.last_applied_revision, 1);
        assert_eq!(state.last_applied_tick, 42);

        // Multiple presentation repaints on unchanged snapshot tick do not advance tick
        app.update();
        app.update();
        let state = app.world().resource::<SnapshotState>();
        assert_eq!(state.last_applied_revision, 1);
        assert_eq!(
            state.last_applied_tick, 42,
            "presentation repaints must not alter scientific tick count"
        );

        app.world_mut().resource_mut::<AccessibilityState>().cycle();
        app.update();
        let state = app.world().resource::<SnapshotState>();
        assert_eq!(state.last_applied_revision, 1);
        assert_eq!(
            state.last_applied_palette,
            Some(ColorPaletteMode::Deuteranopia),
            "a paused scene must be re-projected when its presentation palette changes"
        );
        assert_eq!(
            state.last_applied_tick, 42,
            "presentation-only palette changes must not advance science"
        );

        Ok(())
    }

    #[test]
    fn test_toroidal_picking_across_wrap_seams() {
        let extent = 1000.0;

        // Picking near origin (x=5) against an agent near wrap boundary (x=995).
        // Argument order follows core's a - b convention (bd-ikts.4).
        let dx = toroidal_delta(995.0, 5.0, extent);
        assert_eq!(
            dx.abs(),
            10.0,
            "toroidal_delta across wrap seam must compute minimum distance"
        );

        // Distance across opposite seam (origin=990, target=10)
        let dx_reverse = toroidal_delta(10.0, 990.0, extent);
        assert_eq!(
            dx_reverse.abs(),
            20.0,
            "toroidal_delta across opposite wrap seam must compute minimum distance"
        );
    }
}

/// Workspace guard for the acknowledgement class (bd-d6gv).
///
/// THE CLASS: a UI or client asserting an outcome the host never acknowledged.
/// It was fixed in seventeen sites across five surfaces — Bevy, the control
/// layer, REST, MCP and the CLI — under bd-2z0.7.14 and bd-2z0.4.9. It kept
/// regenerating because a boolean is a PLAUSIBLE-LOOKING receipt: `true` reads
/// like success, so each surface invented its own and no reviewer saw a lie.
///
/// Three guards landed with those fixes, but each was scoped to a single file,
/// so an eighth instance in a new file or crate was caught by none of them.
/// This one is workspace-wide, which is the whole point.
///
/// SCOPE IS DETECTION, AND DELIBERATELY NARROW. It catches a submitted command
/// whose answer is thrown away, and the two receipt literals that were really
/// fabricated. It does NOT catch the log-ordering form — announcing an outcome
/// before the submitter is even asked, which was `426f4083a9` — because that is
/// not a discard and a rule loose enough to catch it would fire on ordinary
/// logging. A noisy guard gets suppressed, which is worse than none.
#[cfg(test)]
mod acknowledgement_guard {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::{Path, PathBuf};

    /// This module's own text is cut from every file before scanning.
    ///
    /// Three separate tests during bd-2z0.7.14 and bd-2z0.4.9 matched their own
    /// literals and passed forever. A source-scanning guard that can read
    /// itself is not a guard.
    const GUARD_MODULE_MARKER: &str = "mod acknowledgement_guard";

    /// Crates carrying a user- or client-facing surface.
    const SCANNED_CRATE_DIRS: &[&str] = &[
        "crates/scriptbots-app/src",
        "crates/scriptbots-bevy/src",
        "crates/scriptbots-render/src",
        "crates/scriptbots-runtime/src",
        "crates/scriptbots-web/src",
        "crates/scriptbots-world-gfx/src",
    ];

    /// Crates deliberately NOT scanned, each with the bead that decided it.
    ///
    /// The point of this list is that it does not exist to shrink the work — it
    /// exists so that a crate is never merely FORGOTTEN. Before it, the scanned
    /// set was an explicit list and a newly added crate was silently outside
    /// the guard while every test stayed green, which is the failure mode where
    /// a guard is most dangerous because it still reads as protection. A new
    /// crate now fails the build until someone writes down which of the two
    /// things it is (bd-hhsl).
    ///
    /// Each entry states why the crate has no command surface. These are not
    /// permanent: `an_exempt_crate_must_still_have_no_command_surface` below
    /// re-derives that claim on every build, so an exemption cannot become a
    /// hiding place for a submitter added later.
    const EXEMPT_CRATE_DIRS: &[(&str, &str, &str)] = &[
        (
            "crates/scriptbots-core/src",
            "bd-hhsl",
            "defines and APPLIES ControlCommand; it is the target of commands, not a submitter",
        ),
        (
            "crates/scriptbots-storage/src",
            "bd-hhsl",
            "persistence; no control surface",
        ),
        (
            "crates/scriptbots-analytics/src",
            "bd-hhsl",
            "offline analysis over persisted data; no control surface",
        ),
        (
            "crates/scriptbots-index/src",
            "bd-hhsl",
            "spatial index; no control surface",
        ),
        (
            "crates/scriptbots-brain/src",
            "bd-hhsl",
            "neural substrate; no control surface",
        ),
        (
            "crates/scriptbots-brain-ml/src",
            "bd-hhsl",
            "neural substrate; no control surface",
        ),
        (
            "crates/scriptbots-brain-neuro/src",
            "bd-hhsl",
            "neural substrate; no control surface",
        ),
    ];

    /// Calls that submit a command and hand back an answer about its fate.
    ///
    /// Each entry is (call fragment, what the answer actually means, what to do
    /// with it). The third field is the reason this guard is actionable rather
    /// than merely red: bd-ikts.5 established that a message naming the fix
    /// gets obeyed while one naming the smell gets suppressed.
    const COMMAND_SUBMITTERS: &[(&str, &str, &str)] = &[
        (
            "(submitter.submit)",
            "a bool meaning ENQUEUED, not applied",
            "gate the UI change on it and report the refusal instead of committing anyway",
        ),
        (
            "submit_simulation_command(",
            "a bool meaning ENQUEUED, not applied",
            "call submit_playback_command so a refusal rolls the local state back",
        ),
        (
            "run_control(move || state.handle.",
            "a CommandStatusDto receipt",
            "return the receipt so the client has a command id it can poll",
        ),
        (
            "run_control_mcp_sync(move || handle.",
            "a CommandStatusDto receipt",
            "return the receipt instead of echoing the request back as the result",
        ),
        // Added after this guard MISSED a live instance. The GPUI render layer
        // has its own submitter returning a bool, and `submit_config_update`
        // dropped it with `let _ =` while every rule here scored zero. A guard
        // is only as wide as its registry, which is why bd-hhsl also asks
        // whether the registry can be derived rather than listed.
        (
            "self.submit_control_command(",
            "a bool meaning ENQUEUED, not applied",
            "return or gate on it so the caller learns the edit never reached the simulation",
        ),
    ];

    /// Literals that fabricate a receipt out of nothing.
    ///
    /// Both were real: `queued: true` was the entire body of POST
    /// /api/selection, and `success: true` answered four REST control
    /// endpoints. Neither was derived from anything the host said.
    const FABRICATED_RECEIPTS: &[(&str, &str)] = &[
        (
            "success: true",
            "return the CommandStatusDto the control call already produces",
        ),
        (
            "queued: true",
            "return the CommandStatusDto the control call already produces",
        ),
    ];

    /// Calls that unambiguously reach the command bus.
    ///
    /// These three are the ONLY hand-written entries left, and they are seeds
    /// rather than a registry: everything else is derived from them by
    /// following the call graph. bd-hhsl was raised to P1 because a
    /// hand-maintained registry missed two live defects in one session -
    /// render's `submit_config_update` and the websocket's plain
    /// `handle.pause()` - and a guard that misses what it exists to catch is
    /// worse than none, because it manufactures confidence.
    const BUS_SEEDS: &[&str] = &[
        "(submitter.submit)(",
        "(self.command_submit.as_ref())(",
        "self.commands.try_send(",
    ];

    /// The file whose methods a `handle.` / `state.handle.` receiver names.
    ///
    /// Names are not identities. `apply_preset` exists in both the REST layer
    /// and the GPUI layer, and `step` means one thing on `ControlHandle` and
    /// another on `WorldState`. Qualifying by receiver AND by defining file is
    /// what keeps the derived set from flagging `world.step()`, which an
    /// earlier draft of this derivation did.
    const CONTROL_HANDLE_FILE: &str = "crates/scriptbots-app/src/control.rs";

    /// Calls that submit a command from inside a control method.
    const SUBMITTING_CALLS: &[&str] = &["self.enqueue(", "self.submit_control_command("];

    /// The receipt type a submitting method must hand back.
    const RECEIPT_TYPE: &str = "CommandStatusDto";

    /// One detected offence, carrying enough to act on without hunting.
    #[derive(Debug)]
    struct Offence {
        file: String,
        line_no: usize,
        line: String,
        problem: String,
        fix: String,
    }

    impl std::fmt::Display for Offence {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "\n  {}:{}\n    {}\n    problem: {}\n    fix: {}",
                self.file, self.line_no, self.line, self.problem, self.fix
            )
        }
    }

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root above crates/scriptbots-bevy")
            .to_path_buf()
    }

    /// Every scanned source as (repo-relative path, text with this guard cut).
    fn scanned_sources() -> Vec<(String, String)> {
        let root = workspace_root();
        let mut out = Vec::new();
        for dir in SCANNED_CRATE_DIRS {
            collect_sources(&root.join(dir), &root, &mut out);
        }
        out
    }

    fn collect_sources(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_sources(&path, root, out);
                continue;
            }
            if !path.extension().is_some_and(|ext| ext == "rs") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let text = text
                .split_once(GUARD_MODULE_MARKER)
                .map_or(text.as_str(), |(before, _)| before)
                .to_owned();
            let display = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            out.push((display, text));
        }
    }

    /// Submissions whose answer is bound to nothing.
    ///
    /// A control call whose line both opens the statement and ends it with a
    /// semicolon bound no result, so the answer went nowhere. A bound call
    /// continues into an expression and does not match. `let _ =` is the
    /// explicit form of the same thing.
    fn discarded_submissions(sources: &[(String, String)]) -> Vec<Offence> {
        let mut out = Vec::new();
        for (file, text) in sources {
            let mut previous = String::new();
            for (index, raw) in text.lines().enumerate() {
                let line = raw.trim();
                if line.starts_with("//") || line.is_empty() {
                    continue;
                }
                // rustfmt wraps a long bound call onto its own line, so a line
                // can both start the call and end the statement while still
                // being assigned to something. The binding is on the previous
                // line, which ends with `=`. Measured, not guessed: without
                // this, the rule reported a false offence on the very fix that
                // 0a63c630c4 landed.
                let bound_by_wrap = previous.ends_with('=');
                for (fragment, meaning, fix) in COMMAND_SUBMITTERS {
                    let opens_and_closes =
                        line.starts_with(fragment) && line.ends_with(';') && !bound_by_wrap;
                    let explicitly_dropped = line.contains(&format!("let _ = {fragment}"));
                    if opens_and_closes || explicitly_dropped {
                        out.push(Offence {
                            file: file.clone(),
                            line_no: index + 1,
                            line: line.to_owned(),
                            problem: format!("discards {meaning}"),
                            fix: (*fix).to_owned(),
                        });
                    }
                }
                previous = line.to_owned();
            }
        }
        out
    }

    /// Receipt literals invented rather than observed.
    fn fabricated_receipts(sources: &[(String, String)]) -> Vec<Offence> {
        let mut out = Vec::new();
        for (file, text) in sources {
            for (index, raw) in text.lines().enumerate() {
                let line = raw.trim();
                if line.starts_with("//") {
                    continue;
                }
                for (literal, fix) in FABRICATED_RECEIPTS {
                    if line.contains(literal) {
                        out.push(Offence {
                            file: file.clone(),
                            line_no: index + 1,
                            line: line.to_owned(),
                            problem: format!(
                                "answers with the hardcoded literal `{literal}`, which is not \
                                 derived from anything the host said"
                            ),
                            fix: (*fix).to_owned(),
                        });
                    }
                }
            }
        }
        out
    }

    /// One function as the derivation sees it.
    struct DerivedFn {
        file: String,
        name: String,
        signature: String,
        body: String,
        is_test: bool,
    }

    /// Split a source file into functions, signatures accumulated to the brace.
    ///
    /// String literals are blanked first so a test that quotes a call cannot be
    /// read as making one — the self-reference trap that made three earlier
    /// guards vacuous.
    fn derived_functions(file: &str, text: &str) -> Vec<DerivedFn> {
        let mut out = Vec::new();
        let mut current: Option<(String, String, Vec<String>)> = None;
        let mut collecting = false;
        let mut previous = String::new();
        let mut is_test = false;
        for raw in text.lines() {
            let line = blank_string_literals(raw.trim());
            if line.starts_with("//") {
                continue;
            }
            if let Some(name) = declared_function_name(&line) {
                if let Some((n, sig, body)) = current.take() {
                    out.push(DerivedFn {
                        file: file.to_owned(),
                        name: n,
                        signature: sig,
                        body: body.join("\n"),
                        is_test,
                    });
                }
                is_test = previous.contains("#[test]");
                collecting = !line.contains('{');
                current = Some((name, line.clone(), Vec::new()));
                previous = line;
                continue;
            }
            if collecting && let Some((_, sig, _)) = current.as_mut() {
                sig.push(' ');
                sig.push_str(&line);
                collecting = !line.contains('{');
                previous = line;
                continue;
            }
            if let Some((_, _, body)) = current.as_mut() {
                body.push(line.clone());
            }
            previous = line;
        }
        if let Some((n, sig, body)) = current {
            out.push(DerivedFn {
                file: file.to_owned(),
                name: n,
                signature: sig,
                body: body.join("\n"),
                is_test,
            });
        }
        out
    }

    /// Replace `"..."` contents so quoted code is not read as code.
    fn blank_string_literals(line: &str) -> String {
        let mut out = String::with_capacity(line.len());
        let mut inside = false;
        let mut escaped = false;
        for ch in line.chars() {
            match (inside, ch) {
                (false, '"') => {
                    inside = true;
                    out.push('"');
                }
                (false, _) => out.push(ch),
                (true, _) if escaped => escaped = false,
                (true, '\\') => escaped = true,
                (true, '"') => {
                    inside = false;
                    out.push('"');
                }
                (true, _) => {}
            }
        }
        out
    }

    /// Does this signature return something a caller could drop on the floor?
    ///
    /// A function returning unit, or `Result<(), _>`, has no receipt to
    /// discard — `enqueue(cmd)?` is correct code. Conflating "reaches the bus"
    /// with "hands back an answer" made an earlier draft flag it.
    fn yields_a_receipt(signature: &str) -> bool {
        if !signature.contains("-> ") || signature.contains("Result<(), ") {
            return false;
        }
        // `Option<String>` is the submit-boundary receipt: an id on admission,
        // `None` on refusal. It joins the list because the bus envelope made it
        // the answer the Bevy and GPUI closures now hand back (bd-k7nq).
        signature.contains(RECEIPT_TYPE)
            || signature.contains("-> bool")
            || signature.contains("-> Option<String>")
    }

    /// DERIVE the submitter set instead of listing it.
    ///
    /// Seeded from the three calls that unambiguously reach the bus, then
    /// closed over the call graph. Returns the functions that both reach the
    /// bus AND hand back an answer, mapped to the files that define them so a
    /// caller can be matched by receiver rather than by bare name.
    fn derived_submitters(sources: &[(String, String)]) -> BTreeMap<String, BTreeSet<String>> {
        let all: Vec<DerivedFn> = sources
            .iter()
            .flat_map(|(file, text)| derived_functions(file, text))
            .collect();

        let mut reaches_bus: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for f in all.iter().filter(|f| !f.is_test) {
            if BUS_SEEDS.iter().any(|seed| f.body.contains(seed)) {
                reaches_bus
                    .entry(f.name.clone())
                    .or_default()
                    .insert(f.file.clone());
            }
        }
        // Transitive closure. Five passes is far beyond the observed depth
        // (submit -> enqueue is two), and a fixed bound cannot hang the build.
        for _ in 0..5 {
            let known: BTreeSet<String> = reaches_bus.keys().cloned().collect();
            for f in all.iter().filter(|f| !f.is_test) {
                if known.iter().any(|s| f.body.contains(&format!(".{s}("))) {
                    reaches_bus
                        .entry(f.name.clone())
                        .or_default()
                        .insert(f.file.clone());
                }
            }
        }

        reaches_bus
            .into_iter()
            .filter(|(name, files)| {
                all.iter().any(|f| {
                    &f.name == name && files.contains(&f.file) && yields_a_receipt(&f.signature)
                })
            })
            .collect()
    }

    /// `fn name` at the start of a declaration, ignoring calls and types.
    fn declared_function_name(raw: &str) -> Option<String> {
        let line = raw.trim_start();
        let rest = line
            .strip_prefix("pub async fn ")
            .or_else(|| line.strip_prefix("pub(crate) async fn "))
            .or_else(|| line.strip_prefix("async fn "))
            .or_else(|| line.strip_prefix("pub fn "))
            .or_else(|| line.strip_prefix("pub(crate) fn "))
            .or_else(|| line.strip_prefix("pub(super) fn "))
            .or_else(|| line.strip_prefix("fn "))?;
        let name = rest.split(['(', '<']).next()?.trim();
        (!name.is_empty()).then(|| name.to_owned())
    }

    /// Methods that submit a command but hand back something else.
    ///
    /// THE THIRD SHAPE, and the one that got past the first two rules. When
    /// `apply_patch` projected unapplied config it discarded no answer and
    /// invented no literal: it built a `ConfigSnapshot` from the REQUESTED
    /// config and returned that. The value looked richly correct — real config
    /// data, correctly shaped — but it was assembled from the request rather
    /// than from anything the host said, and stamped with the tick at which
    /// those values were not yet in effect. That is the most dangerous of the
    /// three, because a fabricated literal looks thin while a fabricated
    /// structure looks like evidence.
    ///
    /// PROVEN AGAINST HISTORY BEFORE BEING WRITTEN: this rule reports exactly
    /// one offence at `0a63c630c4^` — apply_patch, at its real line — and zero
    /// afterwards.
    ///
    /// Multi-line signatures are accumulated up to the opening brace; reading
    /// only the first header line produced two false positives in the
    /// prototype, because the return type had not been seen yet.
    ///
    /// SCOPED TO FILES THAT KNOW THE RECEIPT TYPE, deliberately. `CommandStatusDto`
    /// is owned by scriptbots-app; the GPUI render layer has its own submitter
    /// returning a bool and never mentions the type. Applying this rule there
    /// produced three false offences in the first run, demanding a type that
    /// crate cannot name. Render's weaker contract is covered by the discard
    /// rule instead, which is what actually caught its live defect.
    fn submitters_returning_a_projection(sources: &[(String, String)]) -> Vec<Offence> {
        let mut out = Vec::new();
        for (file, text) in sources {
            if !text.contains(RECEIPT_TYPE) {
                continue;
            }
            let mut current: Option<(String, usize)> = None;
            let mut signature = String::new();
            let mut collecting = false;
            for (index, raw) in text.lines().enumerate() {
                let line = raw.trim();
                if line.starts_with("//") {
                    continue;
                }
                if let Some(name) = declared_function_name(raw) {
                    current = Some((name, index + 1));
                    signature = line.to_owned();
                    collecting = !line.contains('{');
                    continue;
                }
                if collecting {
                    signature.push(' ');
                    signature.push_str(line);
                    collecting = !line.contains('{');
                    continue;
                }
                let Some((name, declared_at)) = current.as_ref() else {
                    continue;
                };
                if SUBMITTING_CALLS.iter().any(|call| line.contains(call))
                    && !signature.contains(RECEIPT_TYPE)
                {
                    out.push(Offence {
                        file: file.clone(),
                        line_no: index + 1,
                        line: line.to_owned(),
                        problem: format!(
                            "`{name}` (declared at line {declared_at}) submits a command but does \
                             not return {RECEIPT_TYPE}, so whatever it returns was assembled from \
                             the request rather than from the host's answer"
                        ),
                        fix: format!(
                            "return the {RECEIPT_TYPE} that submit_control_command produces, and \
                             let callers read applied state back through a read endpoint"
                        ),
                    });
                }
            }
        }
        out
    }

    /// Discarded calls to any DERIVED submitter, matched by receiver.
    ///
    /// `self.NAME(` counts only where NAME is defined in that same file, and
    /// `handle.NAME(` / `state.handle.NAME(` only where NAME is a
    /// `ControlHandle` method. Without that qualification the set flags
    /// `world.step()`, because `step` is also a `ControlHandle` method — names
    /// are not identities.
    fn discarded_calls_to_derived_submitters(sources: &[(String, String)]) -> Vec<Offence> {
        let submitters = derived_submitters(sources);
        let mut out = Vec::new();
        for (file, text) in sources {
            let mut previous = String::new();
            for (index, raw) in text.lines().enumerate() {
                let line = raw.trim();
                if line.starts_with("//") || line.is_empty() {
                    continue;
                }
                for (name, definitions) in &submitters {
                    let mut receivers = Vec::new();
                    if definitions.contains(file) {
                        receivers.push(format!("self.{name}("));
                    }
                    if definitions.iter().any(|d| d == CONTROL_HANDLE_FILE) {
                        receivers.push(format!("handle.{name}("));
                        receivers.push(format!("state.handle.{name}("));
                    }
                    let dropped = receivers.iter().any(|call| {
                        line.contains(&format!("let _ = {call}"))
                            || (line.starts_with(call.as_str())
                                && line.ends_with(';')
                                && !previous.ends_with('='))
                    });
                    if dropped {
                        out.push(Offence {
                            file: file.clone(),
                            line_no: index + 1,
                            line: line.to_owned(),
                            problem: format!(
                                "`{name}` reaches the command bus and returns an answer, and this \
                                 call drops it"
                            ),
                            fix: "bind the result and act on it, or propagate it to a caller that \
                                  can"
                            .to_owned(),
                        });
                        break;
                    }
                }
                previous = line.to_owned();
            }
        }
        out
    }

    /// Is this crate directory accounted for, either way?
    ///
    /// Extracted so the negative case is a permanent test rather than a claim.
    fn crate_dir_is_accounted(dir: &str) -> bool {
        SCANNED_CRATE_DIRS.contains(&dir) || EXEMPT_CRATE_DIRS.iter().any(|(d, _, _)| *d == dir)
    }

    /// The accounting must reject a crate nobody has decided about.
    ///
    /// A gate whose failing case was never exercised is a gate nobody has
    /// tested. This is the whole mechanism in miniature: a name that is in
    /// neither list must come back unaccounted.
    #[test]
    fn a_crate_in_neither_list_is_unaccounted() {
        assert!(
            !crate_dir_is_accounted("crates/scriptbots-newly-added/src"),
            "a crate in neither list was reported as accounted for, so adding one would slip \
             past this gate silently"
        );
        assert!(
            crate_dir_is_accounted("crates/scriptbots-app/src"),
            "a scanned crate must be accounted for"
        );
        assert!(
            crate_dir_is_accounted("crates/scriptbots-core/src"),
            "an exempt crate must be accounted for"
        );
    }

    /// Every crate in the workspace is either scanned or explicitly exempt.
    ///
    /// This closes the gap that mattered most. The crate-reached anchor catches
    /// a RENAMED crate — a declared directory that yields nothing — but nothing
    /// caught an ADDED one, and a new crate is exactly how a submitter enters
    /// this tree unnoticed once nobody is watching. Enumerating the filesystem
    /// and demanding an answer for each converts silence into a decision, which
    /// is the whole point of deriving the registry in the first place
    /// (bd-hhsl).
    #[test]
    fn every_crate_is_scanned_or_exempt_with_a_reason() {
        let root = workspace_root();
        let entries = std::fs::read_dir(root.join("crates")).expect("crates/ is readable");
        let mut unaccounted = Vec::new();
        let mut seen = 0usize;
        for entry in entries.flatten() {
            if !entry.path().join("src").is_dir() {
                continue;
            }
            seen += 1;
            let dir = format!("crates/{}/src", entry.file_name().to_string_lossy());
            if !crate_dir_is_accounted(&dir) {
                unaccounted.push(dir);
            }
        }

        assert!(
            seen >= SCANNED_CRATE_DIRS.len(),
            "only {seen} crates were found on disk; the enumeration is broken and this test \
             is checking nothing"
        );
        assert!(
            unaccounted.is_empty(),
            "these crates are neither scanned nor exempt, so a command submitter added to one \
             would be invisible to this guard. Add each to SCANNED_CRATE_DIRS, or to \
             EXEMPT_CRATE_DIRS with the bead that decided it and why: {unaccounted:#?}"
        );

        // An exemption that names a directory which no longer exists is a lie
        // the next reader would trust.
        for (dir, bead, _) in EXEMPT_CRATE_DIRS {
            assert!(
                root.join(dir).is_dir(),
                "exemption for {dir} ({bead}) points at a directory that does not exist; \
                 remove the entry rather than leaving a stale claim"
            );
        }
    }

    /// An exempt crate must still have no command surface.
    ///
    /// Without this, an exemption is a permanent hiding place: someone adds a
    /// submitter to an exempt crate and the guard stays silent forever. The
    /// claim written in the exemption is re-derived from the source on every
    /// build, so it either remains true or the build says so.
    #[test]
    fn an_exempt_crate_must_still_have_no_command_surface() {
        let root = workspace_root();
        let mut offenders = Vec::new();
        for (dir, bead, why) in EXEMPT_CRATE_DIRS {
            let mut sources = Vec::new();
            collect_sources(&root.join(dir), &root, &mut sources);
            let reaches_bus = sources.iter().any(|(_, text)| {
                BUS_SEEDS.iter().any(|seed| {
                    text.lines()
                        .map(str::trim)
                        .any(|line| !line.starts_with("//") && line.contains(seed))
                })
            });
            if reaches_bus {
                offenders.push(format!(
                    "{dir} is exempt under {bead} because it \"{why}\", but it now reaches the \
                     command bus"
                ));
            }
        }
        assert!(
            offenders.is_empty(),
            "an exemption has gone stale; scan the crate instead of exempting it:{}",
            offenders
                .iter()
                .map(|o| format!("\n  {o}"))
                .collect::<String>()
        );
    }

    /// No call to a derived submitter may drop its answer.
    ///
    /// This is the rule the hand-written registry could not be: it finds
    /// submitters by following the call graph from three bus seeds, so a
    /// helper added tomorrow is covered the moment it reaches the bus.
    #[test]
    fn no_derived_submitter_call_discards_its_answer() {
        let sources = scanned_sources();
        let submitters = derived_submitters(&sources);

        // Anchor: derivation must actually find the known submitters. An empty
        // or tiny set would make this test pass while checking nothing.
        for expected in ["pause", "resume", "set_speed", "submit_config_update"] {
            assert!(
                submitters.contains_key(expected),
                "derivation lost `{expected}`; the seeds or the closure are wrong, and this \
                 guard is checking almost nothing. Derived: {:?}",
                submitters.keys().collect::<Vec<_>>()
            );
        }

        let offences = discarded_calls_to_derived_submitters(&sources);
        assert!(
            offences.is_empty(),
            "a call to a command submitter drops the answer it was handed (bd-hhsl):{}",
            offences
                .iter()
                .map(Offence::to_string)
                .collect::<Vec<_>>()
                .join("")
        );
    }

    /// A method that submits a command must return the receipt, not a projection.
    #[test]
    fn no_submitting_method_returns_a_projection() {
        let offences = submitters_returning_a_projection(&scanned_sources());
        assert!(
            offences.is_empty(),
            "a control method answers with a value built from the request instead of the \
             host's receipt (bd-hhsl):{}",
            offences
                .iter()
                .map(Offence::to_string)
                .collect::<Vec<_>>()
                .join("")
        );
    }

    /// No surface may throw away the answer to a command it submitted.
    #[test]
    fn no_surface_discards_a_command_submission_result() {
        let sources = scanned_sources();

        // POSITIVE ANCHOR. A scan that matches nothing is indistinguishable
        // from a clean tree, and one run during bd-2z0.4.9 reported
        // "running 0 tests / test result: ok" and read exactly like a pass.
        // If the submitters have vanished from the scan, the scope is wrong.
        let submitter_mentions = sources
            .iter()
            .flat_map(|(_, text)| text.lines())
            .filter(|line| {
                COMMAND_SUBMITTERS
                    .iter()
                    .any(|(fragment, _, _)| line.contains(fragment))
            })
            .count();
        assert!(
            submitter_mentions >= 8,
            "only {submitter_mentions} command-submitter call sites were scanned; the scan is \
             mis-scoped and this guard is checking nothing"
        );

        // Every declared crate must actually have been reached. Without this,
        // renaming or moving a crate directory would silently drop it from
        // coverage while the guard stayed green — the failure mode where a
        // guard is most dangerous, because it still reads as protection.
        for dir in SCANNED_CRATE_DIRS {
            assert!(
                sources.iter().any(|(file, _)| file.starts_with(dir)),
                "no sources were scanned under {dir}; the path is stale, so that crate's \
                 surfaces are unguarded while this test still passes"
            );
        }

        let offences = discarded_submissions(&sources);
        assert!(
            offences.is_empty(),
            "a surface discards the answer to a command it submitted, so the caller is acting \
             on an outcome the host never acknowledged (bd-d6gv):{}",
            offences
                .iter()
                .map(Offence::to_string)
                .collect::<Vec<_>>()
                .join("")
        );
    }

    /// No surface may answer with a receipt it made up.
    #[test]
    fn no_surface_fabricates_a_receipt_literal() {
        let offences = fabricated_receipts(&scanned_sources());
        assert!(
            offences.is_empty(),
            "a surface reports success from a literal rather than from the host's answer \
             (bd-d6gv):{}",
            offences
                .iter()
                .map(Offence::to_string)
                .collect::<Vec<_>>()
                .join("")
        );
    }

    /// The guard must fire on a known offence.
    ///
    /// Proven against the REAL pre-fix code rather than an invented sample: the
    /// three lines below are transcribed from `a3e820fb9c^` and
    /// `eb5312e58f^`, at which revisions this detection finds 4 and 8 offences
    /// respectively. A guard whose negative case was never proven is a guard
    /// nobody has tested.
    #[test]
    fn the_guard_fires_on_the_offences_it_was_built_from() {
        let historical = vec![(
            "crates/scriptbots-app/src/servers.rs".to_owned(),
            [
                "async fn post_pause() {",
                "    run_control(move || state.handle.pause()).await?;",
                "    Ok(Json(CommandAcknowledge {",
                "        success: true,",
                "    }))",
                "}",
                "fn clear() {",
                "    let _ = submit_simulation_command(submitter, command);",
                "}",
            ]
            .join("\n"),
        )];

        let discards = discarded_submissions(&historical);
        assert_eq!(
            discards.len(),
            2,
            "the discard rule stopped detecting the code it was built from: {discards:#?}"
        );
        assert!(
            discards.iter().any(|o| o.fix.contains("command id")),
            "the message must name the fix, not just the smell"
        );

        let fabrications = fabricated_receipts(&historical);
        assert_eq!(
            fabrications.len(),
            1,
            "the literal rule stopped detecting the code it was built from: {fabrications:#?}"
        );

        // And it must stay quiet on the corrected form, or it would fire on
        // every fix it just demanded.
        let corrected = vec![(
            "corrected.rs".to_owned(),
            "    let status = run_control(move || state.handle.pause()).await?;".to_owned(),
        )];
        assert!(
            discarded_submissions(&corrected).is_empty(),
            "the guard fires on the corrected form, so it would forbid its own fix"
        );
    }

    /// The derived rule must catch both defects the hand registry missed.
    ///
    /// This is the justification for bd-hhsl's P1 bump, kept executable. The
    /// listed registry matched call fragments like
    /// `run_control(move || state.handle.`, so it scored zero on BOTH of these
    /// while they were live:
    ///   render's `submit_config_update` dropping its enqueue result, and
    ///   the websocket channel's plain `let _ = handle.pause()`.
    /// Derivation finds them because it follows the call graph from the bus
    /// rather than from a list someone has to remember to update.
    #[test]
    fn derivation_catches_both_defects_the_listed_registry_missed() {
        // A minimal two-file world: the handle defines the submitters, the
        // surfaces call them. Paths matter — the receiver rule keys on them.
        let sources = vec![
            (
                CONTROL_HANDLE_FILE.to_owned(),
                [
                    "fn enqueue(&self, command: ControlCommand) -> Result<(), ControlError> {",
                    "    self.commands.try_send(command)",
                    "}",
                    "pub fn pause(&self) -> Result<CommandStatusDto, ControlError> {",
                    "    self.submit_control_command(ControlCommand::Pause)",
                    "}",
                    "fn submit_control_command(&self, cmd: ControlCommand) -> Result<CommandStatusDto, ControlError> {",
                    "    self.enqueue(cmd)?;",
                    "}",
                ]
                .join("\n"),
            ),
            (
                "crates/scriptbots-app/src/servers.rs".to_owned(),
                "async fn ws() {\n    let _ = handle.pause();\n}".to_owned(),
            ),
            (
                "crates/scriptbots-render/src/lib.rs".to_owned(),
                [
                    "fn submit_control_command(&self, command: ControlCommand) -> bool {",
                    "    (self.command_submit.as_ref())(command)",
                    "}",
                    "fn submit_config_update(&self) -> bool {",
                    "    let _ = self.submit_control_command(ControlCommand::UpdateConfig(c));",
                    "    true",
                    "}",
                ]
                .join("\n"),
            ),
        ];

        let submitters = derived_submitters(&sources);
        assert!(
            submitters.contains_key("pause"),
            "`pause` must be derived through submit_control_command -> enqueue -> bus; \
             the chain broke. Derived: {:?}",
            submitters.keys().collect::<Vec<_>>()
        );
        assert!(
            !submitters.contains_key("enqueue"),
            "`enqueue` returns Result<(), _> and has no answer to drop; flagging it would \
             condemn `self.enqueue(cmd)?`, which is correct code"
        );

        let offences = discarded_calls_to_derived_submitters(&sources);
        let files: Vec<&str> = offences.iter().map(|o| o.file.as_str()).collect();
        assert!(
            files.contains(&"crates/scriptbots-app/src/servers.rs"),
            "the websocket miss went uncaught again: {offences:#?}"
        );
        assert!(
            files.contains(&"crates/scriptbots-render/src/lib.rs"),
            "the render miss went uncaught again: {offences:#?}"
        );

        // And it must stay quiet on a call that binds the answer.
        let corrected = vec![(
            "crates/scriptbots-app/src/servers.rs".to_owned(),
            "async fn ws() {\n    let status = handle.pause()?;\n}".to_owned(),
        )];
        assert!(
            discarded_calls_to_derived_submitters(&corrected).is_empty(),
            "the derived rule fires on a bound call, so it would forbid its own fix"
        );
    }

    /// The projection rule must fire on the config defect that got past the
    /// other two.
    ///
    /// This case exists because the first two rules BOTH scored zero on
    /// `crates/scriptbots-app/src/control.rs` at `0a63c630c4^` while the
    /// projection defect was live there. A guard that missed a real instance is
    /// worth a permanent test of the miss, not just a patch.
    ///
    /// The lines are transcribed from that revision.
    #[test]
    fn the_projection_rule_fires_on_the_defect_the_other_rules_missed() {
        let offending = vec![(
            "crates/scriptbots-app/src/control.rs".to_owned(),
            [
                // The real file declares the receipt type, which is what brings
                // it into this rule's scope. Without this line the fixture is
                // skipped as a crate that cannot name the type — a difference
                // between the sample and the real file that this canary caught
                // on its first run.
                "    pub fn pause(&self) -> Result<CommandStatusDto, ControlError> {",
                "        self.submit_control_command(ControlCommand::Pause)",
                "    }",
                "    pub fn apply_patch(&self, patch: Value) -> Result<ConfigSnapshot, ControlError> {",
                "        let snapshot = ConfigSnapshot::from_config(new_config.clone(), current_tick)?;",
                "        drop(world);",
                "        self.enqueue(ControlCommand::UpdateConfig(Box::new(new_config)))?;",
                "        Ok(snapshot)",
                "    }",
            ]
            .join("\n"),
        )];

        // The other two rules are blind to it — that is the point.
        assert!(
            discarded_submissions(&offending).is_empty(),
            "the discard rule was never the one that could catch this"
        );
        assert!(
            fabricated_receipts(&offending).is_empty(),
            "the literal rule was never the one that could catch this"
        );

        let offences = submitters_returning_a_projection(&offending);
        assert_eq!(
            offences.len(),
            1,
            "the projection rule stopped detecting the defect it was written for: {offences:#?}"
        );
        assert!(
            offences[0].problem.contains("apply_patch"),
            "the message must name the offending method: {}",
            offences[0].problem
        );

        // Quiet on the corrected form, or it forbids its own fix.
        let corrected = vec![(
            "corrected.rs".to_owned(),
            [
                "    pub fn apply_patch(&self, patch: Value) -> Result<CommandStatusDto, ControlError> {",
                "        let status = self.submit_control_command(ControlCommand::UpdateConfig(c))?;",
                "        Ok(status)",
                "    }",
            ]
            .join("\n"),
        )];
        assert!(
            submitters_returning_a_projection(&corrected).is_empty(),
            "the projection rule fires on the corrected form"
        );

        // A crate that cannot name the receipt type is out of scope rather than
        // permanently guilty. This is the render layer's shape, and demanding
        // CommandStatusDto there produced three false offences before the rule
        // was scoped.
        let foreign = vec![(
            "crates/scriptbots-render/src/lib.rs".to_owned(),
            [
                "    fn submit_simulation_command(&self, command: SimulationCommand) -> bool {",
                "        self.submit_control_command(ControlCommand::UpdateSimulation(command))",
                "    }",
            ]
            .join("\n"),
        )];
        assert!(
            submitters_returning_a_projection(&foreign).is_empty(),
            "the projection rule demands a type this crate cannot name"
        );
    }
}
