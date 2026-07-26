//! Bevy renderer integration for ScriptBots.

pub mod capture;
pub mod creature_meshes;
pub mod particles;

use anyhow::{Context, Result, anyhow};
use bevy::app::AppExit;
use bevy::asset::RenderAssetUsages;
use bevy::camera::prelude::*;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::ecs::system::NonSendMut;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::light::{EnvironmentMapLight, LightProbe};
use bevy::math::primitives::{Capsule3d, Cone, Rectangle, Sphere, Torus};
use bevy::pbr::prelude::*;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::render::view::{ColorGrading, Hdr};
use bevy::ui::{BorderColor, BorderRadius};
use bevy::window::{PresentMode, PrimaryWindow, WindowPlugin};
use bevy_mesh::{Indices, Mesh};
use bevy_post_process::auto_exposure::{AutoExposure, AutoExposurePlugin};
use image::{ImageBuffer, Rgba as ImgRgba};
use scriptbots_core::{
    AccessibilityPalette, AgentId, ControlCommand, ControlDisposition, GpuClass, GpuInfo,
    IndicatorState, NUM_EYES, OutputChannel, OutputsExt, RenderQuality, RenderSettings,
    RenderTonemapMode, SelectionMode, SelectionState, SelectionUpdate, SimulationCommand,
    TerrainKind, TickSummary, TierFeatures, TraitModifiers, WorldState, WorldStepDriver,
    apply_control_command, initial_tier_for, tier_features,
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
pub type CommandSubmitFn = Arc<dyn Fn(ControlCommand) -> bool + Send + Sync>;
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
    Some(GpuInfo {
        name: info.name,
        backend: format!("{:?}", info.backend),
        class,
        vram_bytes: None,
        max_texture_2d: Some(limits.max_texture_dimension_2d),
        timestamp_queries: features.contains(wgpu::Features::TIMESTAMP_QUERY),
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
    let features = tier_features(tier);
    match &gpu {
        Some(info) => {
            info!(
                adapter = %info.name,
                backend = %info.backend,
                class = ?info.class,
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
    }
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
                while worker_flag.load(Ordering::Acquire) {
                    let mut snapshot = {
                        let guard = world_for_worker.lock().map_err(|error| {
                            anyhow!("world mutex poisoned in Bevy snapshot worker: {error}")
                        })?;
                        WorldSnapshot::from_world(&guard)
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
    let accepted = (submitter.submit)(ControlCommand::UpdateSimulation(command));
    if !accepted {
        warn!("failed to enqueue simulation control command");
    }
    accepted
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

fn toroidal_delta(origin: f32, target: f32, extent: f32) -> f32 {
    let mut delta = target - origin;
    let half = extent * 0.5;
    if delta > half {
        delta -= extent;
    } else if delta < -half {
        delta += extent;
    }
    delta
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
}

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

fn setup_scene(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let camera_transform = Transform::from_xyz(0.0, 1800.0, 1400.0).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((
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
    ));

    let light_transform =
        Transform::from_xyz(-1200.0, 1800.0, 900.0).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((
        DirectionalLight {
            illuminance: 9000.0,
            shadows_enabled: true,
            ..default()
        },
        light_transform,
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
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
        let history = vec![
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
        let history = vec![summary(1, 0, 0, 0), summary(2, 0, 0, 0)];
        assert!(HudEvent::recent_from_history(history.iter()).is_empty());
    }

    /// Within one tick the order is births, deaths, then spike hits, so a busy
    /// tick always reads the same way.
    #[test]
    fn one_tick_orders_births_deaths_then_spikes() {
        let history = vec![summary(7, 3, 2, 1)];
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

fn handle_selection_input(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
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
        if (submitter.submit)(command) {
            info!("Bevy selection cleared via Escape");
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

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };

    let Ok((camera, transform)) = camera_query.single() else {
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
        let dx = toroidal_delta(world_point.x, agent.position.x, world_size.x);
        let dy = toroidal_delta(world_point.y, agent.position.y, world_size.y);
        let dist_sq = dx.mul_add(dx, dy * dy);
        if dist_sq <= radius_sq && dist_sq < best_dist {
            best_dist = dist_sq;
            best = Some(agent);
        }
    }

    let extend = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    if let Some(agent) = best {
        let agent_id = encode_agent_id(agent.id);
        let command = if extend {
            if matches!(agent.selection, SelectionState::Selected) {
                info!(agent_id, "Bevy selection toggle -> clear");
                ControlCommand::UpdateSelection(SelectionUpdate {
                    mode: SelectionMode::Clear,
                    agent_ids: vec![agent_id],
                    state: SelectionState::Selected,
                })
            } else {
                info!(agent_id, "Bevy selection toggle -> add");
                ControlCommand::UpdateSelection(SelectionUpdate {
                    mode: SelectionMode::Add,
                    agent_ids: vec![agent_id],
                    state: SelectionState::Selected,
                })
            }
        } else {
            info!(agent_id, "Bevy selection replace");
            ControlCommand::UpdateSelection(SelectionUpdate {
                mode: SelectionMode::Replace,
                agent_ids: vec![agent_id],
                state: SelectionState::Selected,
            })
        };

        if (submitter.submit)(command) && !extend {
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
        if (submitter.submit)(command) {
            info!("Bevy selection cleared via empty click");
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
            let step_once = command.step_once;
            if !submit_simulation_command(submitter, command) && step_once {
                // A rejected command cannot disappear. Fall back to the local
                // driver edge, while preserving any pending unconsumed step.
                controls.update(enqueue_step_request);
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
        if let (Some(submitter), Some(command)) = (submitter.as_ref(), Some(command)) {
            let _ = submit_simulation_command(submitter, command);
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
        if let (Some(submitter), Some(command)) = (submitter.as_ref(), Some(command)) {
            if !submit_simulation_command(submitter, command) {
                controls.update(enqueue_step_request);
            }
        }
    }

    if keys.just_pressed(KeyCode::Equal) || keys.just_pressed(KeyCode::NumpadAdd) {
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
        if let (Some(submitter), Some(command)) = (submitter.as_ref(), Some(command)) {
            let _ = submit_simulation_command(submitter, command);
        }
    }

    if keys.just_pressed(KeyCode::Minus) || keys.just_pressed(KeyCode::NumpadSubtract) {
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
        if let (Some(submitter), Some(command)) = (submitter.as_ref(), Some(command)) {
            let _ = submit_simulation_command(submitter, command);
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
            (submitter.submit)(command);
            info!("Bevy clear selection button pressed");
            rig.follow_mode = FollowMode::Off;
            rig.pan = Vec2::ZERO;
            rig.recenter_now = true;
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

    #[test]
    fn effective_settings_honor_explicit_tiers_and_auto() {
        let explicit = resolve_effective_render_settings(&RenderSettings {
            quality: Some(RenderQuality::Low),
            ..RenderSettings::default()
        });
        assert_eq!(explicit.tier, RenderQuality::Low);
        assert!(!explicit.features.ssao, "Low has no SSAO");

        let auto = resolve_effective_render_settings(&RenderSettings::default());
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
                true
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
                accepted
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
        let state = controls.0.lock().expect("simulation controls").clone();
        state
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
        let mut snapshot_state = SnapshotState::default();
        snapshot_state.latest = Some(snapshot);
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

        // Picking near origin (x=5) against an agent near wrap boundary (x=995)
        let dx = toroidal_delta(5.0, 995.0, extent);
        assert_eq!(
            dx.abs(),
            10.0,
            "toroidal_delta across wrap seam must compute minimum distance"
        );

        // Distance across opposite seam (origin=990, target=10)
        let dx_reverse = toroidal_delta(990.0, 10.0, extent);
        assert_eq!(
            dx_reverse.abs(),
            20.0,
            "toroidal_delta across opposite wrap seam must compute minimum distance"
        );
    }
}
