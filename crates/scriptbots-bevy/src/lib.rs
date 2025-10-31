//! Bevy renderer integration for ScriptBots.

use anyhow::{anyhow, Result};
use bevy::app::AppExit;
use bevy::asset::RenderAssetUsages;
use bevy::camera::prelude::*;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::ecs::system::NonSendMut;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::math::primitives::Sphere;
use bevy::pbr::prelude::*;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::ui::prelude::*;
use bevy::ui::{BorderColor, BorderRadius};
use bevy::window::{PresentMode, PrimaryWindow, WindowPlugin};
use bevy_mesh::{Indices, Mesh};
use owo_colors::OwoColorize;
use image::{ImageBuffer, Rgba as ImgRgba};
use scriptbots_core::{
    AgentId, ControlCommand, SelectionMode, SelectionState, SelectionUpdate, SimulationCommand,
    TerrainKind, WorldState,
};
use slotmap::Key;
use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    env,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use tracing::{info, warn};

/// Launch context supplied by the ScriptBots application shell.
pub type CommandSubmitFn = Arc<dyn Fn(ControlCommand) -> bool + Send + Sync>;
pub type CommandDrainFn = Arc<dyn Fn(&mut WorldState) + Send + Sync>;

pub struct BevyRendererContext {
    pub world: Arc<Mutex<WorldState>>,
    pub command_submit: CommandSubmitFn,
    pub command_drain: CommandDrainFn,
}

/// Entry point for the Bevy renderer; blocks until the window closes.
pub fn run_renderer(ctx: BevyRendererContext) -> Result<()> {
    info!("Launching Bevy renderer (Phase 1: static world visuals)");

    let BevyRendererContext {
        world,
        command_submit,
        command_drain,
    } = ctx;

    let (tx, rx) = mpsc::channel::<WorldSnapshot>();
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

    let worker = thread::spawn(move || {
        let mut last_tick = 0u64;
        while worker_flag.load(Ordering::Relaxed) {
            let snapshot = {
                let guard = world_for_worker.lock().expect("world mutex poisoned");
                WorldSnapshot::from_world(&guard)
            };

            if let Some(snapshot) = snapshot {
                if snapshot.tick != last_tick {
                    last_tick = snapshot.tick;
                    if tx.send(snapshot).is_err() {
                        break;
                    }
                }
            }

            thread::sleep(Duration::from_millis(30));
        }
    });

    let simulation_thread = spawn_simulation_driver(
        world_for_sim,
        drain_for_thread,
        controls_for_thread.clone(),
        Arc::clone(&running_sim),
    );

    let mut app = App::new();
    let diagnostics_enabled = diagnostics_enabled();
    app.insert_resource(AmbientLight {
            color: Color::srgb(0.45, 0.52, 0.65),
            brightness: 800.0,
            affects_lightmapped_meshes: true,
        })
        .insert_resource(submitter_resource)
        .insert_resource(controls_resource)
        .insert_non_send_resource(SnapshotInbox { receiver: rx })
        .insert_resource(SnapshotState::default())
        .insert_resource(AgentRegistry::default())
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "ScriptBots • Bevy Renderer".to_string(),
                present_mode: PresentMode::AutoVsync,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (
                poll_snapshots,
                sync_world,
                handle_playback_shortcuts,
                handle_playback_buttons,
                handle_selection_input,
                handle_follow_button_interactions,
                handle_clear_selection_button,
                update_playback_button_colors,
                update_follow_button_colors,
                control_camera,
                update_hud,
            )
                .chain(),
        )
        .add_systems(Update, close_on_esc);

    if diagnostics_enabled {
        app.insert_resource(DiagnosticsTicker::new(DIAGNOSTIC_REPORT_INTERVAL))
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Update, report_frame_metrics);
    }

    app.run();

    running.store(false, Ordering::Relaxed);
    let _ = simulation_thread.join();
    let _ = worker.join();
    Ok(())
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
struct SnapshotState {
    latest: Option<WorldSnapshot>,
    last_applied_tick: u64,
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
struct AgentRegistry {
    records: HashMap<AgentId, AgentRecord>,
}

struct AgentRecord {
    entity: Entity,
    material: Handle<StandardMaterial>,
}

#[derive(Resource)]
struct AgentMeshAsset {
    mesh: Handle<Mesh>,
    base_radius: f32,
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
    step_requested: bool,
    auto_pause_reason: Option<String>,
}

impl Default for SimControlData {
    fn default() -> Self {
        Self {
            paused: false,
            speed_multiplier: 1.0,
            step_requested: false,
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
        let data = self.0.lock().expect("simulation control poisoned").clone();
        SimControlSnapshot {
            paused: data.paused,
            speed_multiplier: data.speed_multiplier,
            auto_pause_reason: data.auto_pause_reason.clone(),
        }
    }

    fn update<F>(&self, f: F)
    where
        F: FnOnce(&mut SimControlData),
    {
        if let Ok(mut data) = self.0.lock() {
            f(&mut data);
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
        state.step_requested = true;
        state.paused = true;
    }
}

fn submit_simulation_command(submitter: &CommandSubmitter, command: SimulationCommand) {
    if !(submitter.submit)(ControlCommand::UpdateSimulation(command)) {
        warn!("failed to enqueue simulation control command");
    }
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

fn follow_idle_color() -> Color {
    Color::srgba(0.16, 0.22, 0.33, 0.92)
}

fn follow_hover_color() -> Color {
    Color::srgba(0.22, 0.30, 0.46, 0.95)
}

fn follow_active_color() -> Color {
    Color::srgba(0.34, 0.26, 0.64, 0.95)
}
const TERRAIN_CHUNK_SIZE: u32 = 64;
const TERRAIN_HEIGHT_SCALE: f32 = 180.0;

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

#[derive(Clone)]
struct TerrainColorMap {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

#[derive(Clone)]
struct TerrainHeightSnapshot {
    dims: UVec2,
    cell_size: u32,
    elevation: Vec<f32>,
    moisture: Vec<f32>,
    accent: Vec<f32>,
    fertility: Vec<f32>,
    temperature: Vec<f32>,
    kinds: Vec<TerrainKind>,
}

impl TerrainHeightSnapshot {
    fn new(layer: &scriptbots_core::TerrainLayer) -> Self {
        let dims = UVec2::new(layer.width(), layer.height());
        let total = (dims.x as usize) * (dims.y as usize);
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
        Self {
            dims,
            cell_size: layer.cell_size(),
            elevation,
            moisture,
            accent,
            fertility,
            temperature,
            kinds,
        }
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
}

#[derive(Clone, Copy)]
struct TerrainChunkBounds {
    origin: UVec2,
    size: UVec2,
}

#[derive(Clone, Copy, Debug)]
struct TerrainChunkSignature {
    sum_height: f64,
    sum_moisture: f64,
    sum_accent: f64,
    max_height: f32,
}

impl TerrainChunkSignature {
    fn new(sum_height: f64, sum_moisture: f64, sum_accent: f64, max_height: f32) -> Self {
        Self {
            sum_height,
            sum_moisture,
            sum_accent,
            max_height,
        }
    }

    fn is_close(&self, other: &Self) -> bool {
        (self.sum_height - other.sum_height).abs() < 1e-3
            && (self.sum_moisture - other.sum_moisture).abs() < 1e-3
            && (self.sum_accent - other.sum_accent).abs() < 1e-3
            && (self.max_height - other.max_height).abs() < 0.5
    }
}

#[derive(Clone)]
struct WorldSnapshot {
    tick: u64,
    world_size: Vec2,
    agent_radius: f32,
    terrain_color: TerrainColorMap,
    terrain_height: TerrainHeightSnapshot,
    agents: Vec<AgentVisual>,
}

#[derive(Clone)]
struct AgentVisual {
    id: AgentId,
    position: Vec2,
    color: [f32; 3],
    selection: SelectionState,
    health: f32,
    age: u32,
}

impl WorldSnapshot {
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
        let terrain_height = TerrainHeightSnapshot::new(terrain_layer);

        let arena = world.agents();
        let columns = arena.columns();
        let positions = columns.positions();
        let colors = columns.colors();
        let healths = columns.health();
        let ages = columns.ages();
        let runtime = world.runtime();

        let mut agents = Vec::with_capacity(arena.len());
        for (idx, agent_id) in arena.iter_handles().enumerate() {
            let runtime_entry = runtime.get(agent_id);
            let selection = runtime_entry.map(|rt| rt.selection).unwrap_or_default();
            agents.push(AgentVisual {
                id: agent_id,
                position: Vec2::new(positions[idx].x, positions[idx].y),
                color: colors[idx],
                selection,
                health: healths[idx],
                age: ages[idx],
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
        })
    }
}

fn setup_scene(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(0.03, 0.05, 0.09)),
            ..default()
        },
        Transform::from_xyz(0.0, 1800.0, 1400.0).looking_at(Vec3::ZERO, Vec3::Y),
        PrimaryCamera,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 9000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(-1200.0, 1800.0, 900.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let agent_mesh = meshes.add(Mesh::from(Sphere::new(1.0)));
    commands.insert_resource(AgentMeshAsset {
        mesh: agent_mesh,
        base_radius: 1.0,
    });
    commands.insert_resource(TerrainChunkRegistry {
        chunk_size: TERRAIN_CHUNK_SIZE,
        height_scale: TERRAIN_HEIGHT_SCALE,
        ..default()
    });
    commands.insert_resource(CameraRig::default());

    commands.spawn((Camera2d,));

    let primary_text_color = Color::WHITE;
    let secondary_text_color = Color::srgb(0.74, 0.82, 0.94);
    let primary_font = TextFont::from_font_size(18.0);
    let secondary_font = TextFont::from_font_size(15.0);

    let button_node = Node {
        padding: UiRect::axes(px(12.0), px(8.0)),
        border: UiRect::all(px(1.0)),
        align_items: AlignItems::Center,
        justify_content: JustifyContent::Center,
        min_width: px(120.0),
        ..default()
    };
    let button_row_node = Node {
        flex_direction: FlexDirection::Row,
        column_gap: px(8.0),
        row_gap: px(8.0),
        margin: UiRect::axes(px(0.0), px(8.0)),
        ..default()
    };
    let button_border_color = Color::srgba(0.32, 0.38, 0.58, 1.0);

    let hud_root = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: px(12.0),
                left: px(12.0),
                padding: UiRect::all(px(10.0)),
                flex_direction: FlexDirection::Column,
                row_gap: px(6.0),
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
                        BorderRadius::all(px(6.0)),
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
                        BorderRadius::all(px(6.0)),
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
                    BorderRadius::all(px(6.0)),
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
    });
}

fn poll_snapshots(inbox: NonSendMut<SnapshotInbox>, mut state: ResMut<SnapshotState>) {
    let receiver = &inbox.receiver;
    while let Ok(snapshot) = receiver.try_recv() {
        state.latest = Some(snapshot);
    }
}

fn sync_world(
    mut commands: Commands,
    mut state: ResMut<SnapshotState>,
    mut registry: ResMut<AgentRegistry>,
    mut terrain_registry: ResMut<TerrainChunkRegistry>,
    assets: Res<AgentMeshAsset>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(snapshot) = state.latest.as_ref() else {
        return;
    };

    if state.last_applied_tick == snapshot.tick {
        return;
    }

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
    );
    sync_agents(
        snapshot,
        &mut commands,
        &mut registry,
        &assets,
        materials.as_mut(),
    );

    state.last_applied_tick = snapshot_tick;
    state.focus_point = focus_point;
    state.world_size = world_size;
    state.world_center = world_center;
    state.selection_bounds = selection_bounds;
    state.selection_center = selection_center;
    state.oldest_position = oldest.map(|(pos, _)| pos);
    state.first_agent_position = first_agent;
}

fn update_hud(
    mut state: ResMut<SnapshotState>,
    rig: Res<CameraRig>,
    hud: Option<Res<HudElements>>,
    time: Res<Time>,
    controls: Res<SimulationControl>,
    mut texts: Query<&mut Text>,
) {
    let Some(_) = state.latest.as_ref() else {
        return;
    };

    let (tick, agent_count, world_size, agent_radius, selected_count, primary_selection) = {
        let snapshot = state.latest.as_ref().expect("snapshot available");
        let tick = snapshot.tick;
        let agent_count = snapshot.agents.len();
        let world_size = snapshot.world_size;
        let agent_radius = snapshot.agent_radius;
        let mut selected_count = 0usize;
        let mut primary: Option<(AgentId, u32, f32)> = None;
        for agent in &snapshot.agents {
            if matches!(agent.selection, SelectionState::Selected) {
                selected_count += 1;
                if primary.is_none() {
                    primary = Some((agent.id, agent.age, agent.health));
                }
            }
        }
        (
            tick,
            agent_count,
            world_size,
            agent_radius,
            selected_count,
            primary,
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
                world_size.x as i32,
                world_size.y as i32,
                agent_radius
            );
        }
    }
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
    mut query: Query<(&PlaybackButton, &Interaction), (Changed<Interaction>, With<Button>)>,
) {
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
                    state.step_requested = false;
                    state.auto_pause_reason = None;
                    command.paused = Some(false);
                    command.speed_multiplier = Some(state.speed_multiplier);
                    info!("Bevy playback: resume");
                }
                PlaybackAction::Pause => {
                    state.paused = true;
                    state.step_requested = false;
                    command.paused = Some(true);
                    info!("Bevy playback: pause");
                }
                PlaybackAction::Step => {
                    state.step_requested = true;
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
            submit_simulation_command(submitter, command);
        }
    }
}

fn handle_playback_shortcuts(
    keys: Res<ButtonInput<KeyCode>>,
    controls: Res<SimulationControl>,
    submitter: Option<Res<CommandSubmitter>>,
) {
    if keys.just_pressed(KeyCode::Space) {
        let mut command = SimulationCommand::default();
        controls.update(|state| {
            state.paused = !state.paused;
            if !state.paused && state.speed_multiplier <= MIN_SPEED {
                state.speed_multiplier = 1.0;
            }
            state.step_requested = false;
            state.auto_pause_reason = None;
            info!(paused = state.paused, "Bevy playback toggled via Space");
            command.paused = Some(state.paused);
            command.speed_multiplier = Some(state.speed_multiplier);
        });
        if let (Some(submitter), Some(command)) = (submitter.as_ref(), Some(command)) {
            submit_simulation_command(submitter, command);
        }
    }

    if keys.just_pressed(KeyCode::KeyN) {
        let mut command = SimulationCommand::default();
        controls.update(|state| {
            state.step_requested = true;
            state.paused = true;
            state.auto_pause_reason = None;
            info!("Bevy playback: step requested via keyboard");
            command.paused = Some(true);
            command.step_once = true;
        });
        if let (Some(submitter), Some(command)) = (submitter.as_ref(), Some(command)) {
            submit_simulation_command(submitter, command);
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
            submit_simulation_command(submitter, command);
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
            submit_simulation_command(submitter, command);
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
    mut query: Query<(&FollowButton, &Interaction), (Changed<Interaction>, With<Button>)>,
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
    if state.latest.is_some() {
        if let Some(command) = rig.pending_fit {
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
                        let distance =
                            fit_distance_for_extent(state.world_size, FIT_SELECTION_FACTOR);
                        rig.distance = distance;
                        rig.distance_smoothed = distance;
                    }
                }
            }
            rig.pending_fit = None;
        }
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
) {
    let dims = snapshot.terrain_height.dims;
    if dims.x == 0 || dims.y == 0 {
        return;
    }

    let chunk_size = registry.chunk_size.max(1);
    let chunks_x = (dims.x + chunk_size - 1) / chunk_size;
    let chunks_y = (dims.y + chunk_size - 1) / chunk_size;

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

            let built = build_chunk_mesh(snapshot, bounds, registry.height_scale);

            match registry.chunks.get_mut(&key) {
                Some(record) => {
                    if !record.signature.is_close(&built.stats.signature) {
                        if let Some(existing) = meshes.get_mut(&record.mesh) {
                            *existing = built.mesh;
                        } else {
                            let mesh_handle = meshes.add(built.mesh);
                            record.mesh = mesh_handle.clone();
                            commands
                                .entity(record.entity)
                                .insert((Mesh3d(mesh_handle.clone()),));
                        }
                        update_chunk_material(materials, &record.material, &built.stats);
                        record.signature = built.stats.signature;
                        record.bounds = bounds;
                    }
                    record.last_tick = snapshot.tick;
                }
                None => {
                    let mesh_handle = meshes.add(built.mesh);
                    let material_handle = materials.add(create_chunk_material(&built.stats));
                    let entity = commands
                        .spawn((
                            Mesh3d(mesh_handle.clone()),
                            MeshMaterial3d(material_handle.clone()),
                        ))
                        .id();
                    registry.chunks.insert(
                        key,
                        TerrainChunkRecord {
                            entity,
                            mesh: mesh_handle,
                            material: material_handle,
                            bounds,
                            signature: built.stats.signature,
                            last_tick: snapshot.tick,
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
        if let Some(record) = registry.chunks.remove(&key) {
            commands.entity(record.entity).despawn();
            meshes.remove(&record.mesh);
            materials.remove(&record.material);
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

struct BuiltChunk {
    mesh: Mesh,
    stats: TerrainChunkStats,
}

struct TerrainChunkStats {
    mean_moisture: f32,
    mean_slope: f32,
    height_factor: f32,
    signature: TerrainChunkSignature,
}

fn build_chunk_mesh(
    snapshot: &WorldSnapshot,
    bounds: TerrainChunkBounds,
    height_scale: f32,
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
    let mut sum_height = 0.0f64;
    let mut sum_accent = 0.0f64;
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
            sum_height += height as f64;
            max_height = max_height.max(height);

            let uv_x = global_x as f32 / terrain.dims.x.max(1) as f32;
            let uv_z = global_z as f32 / terrain.dims.y.max(1) as f32;
            uvs.push([uv_x, uv_z]);

            let color = terrain_vertex_color(terrain, global_x, global_z);
            colors.push(color);

            let sample = terrain.sample_tile(global_x, global_z);
            sum_moisture += sample.moisture as f64;
            let slope = compute_tile_slope(terrain, global_x, global_z);
            sum_slope += slope as f64;
            sum_accent += sample.accent as f64;
        }
    }

    let mut indices = Vec::with_capacity((bounds.size.x * bounds.size.y * 6) as usize);
    let stride = verts_x;
    for z in 0..bounds.size.y {
        for x in 0..bounds.size.x {
            let i0 = (z * stride + x) as u32;
            let i1 = i0 + 1;
            let i2 = i0 + stride as u32;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    for tri in indices.chunks_exact(3) {
        let ia = tri[0] as usize;
        let ib = tri[1] as usize;
        let ic = tri[2] as usize;
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

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));

    let vertex_total = vertex_count as f64;
    let stats = TerrainChunkStats {
        mean_moisture: (sum_moisture / vertex_total) as f32,
        mean_slope: (sum_slope / vertex_total) as f32,
        height_factor: (max_height / height_scale).clamp(0.0, 1.0),
        signature: TerrainChunkSignature::new(sum_height, sum_moisture, sum_accent, max_height),
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

fn terrain_vertex_color(terrain: &TerrainHeightSnapshot, x: u32, z: u32) -> [f32; 4] {
    let sample = terrain.sample_tile(x, z);
    let slope = compute_tile_slope(terrain, x, z);
    let daylight = 0.65;
    let mut rgb = terrain_kind_color(sample.kind);

    let brightness = match sample.kind {
        TerrainKind::DeepWater => {
            (0.42 + daylight * 0.25 + sample.moisture * 0.2).clamp(0.25, 1.05)
        }
        TerrainKind::ShallowWater => {
            (0.55 + daylight * 0.35 + sample.moisture * 0.3).clamp(0.4, 1.25)
        }
        TerrainKind::Sand => (0.72 + daylight * 0.18 + sample.elevation * 0.35).clamp(0.45, 1.35),
        TerrainKind::Grass => (0.62 + daylight * 0.28 + sample.moisture * 0.4).clamp(0.4, 1.35),
        TerrainKind::Bloom => (0.68 + daylight * 0.35 + sample.moisture * 0.5).clamp(0.45, 1.45),
        TerrainKind::Rock => (0.60 + daylight * 0.22 + slope * 0.45).clamp(0.35, 1.25),
    };

    rgb[0] *= brightness;
    rgb[1] *= brightness;
    rgb[2] *= brightness;

    match sample.kind {
        TerrainKind::Bloom | TerrainKind::Grass => {
            let factor = (0.9 + sample.moisture * 0.3 + sample.accent * 0.05).clamp(0.6, 1.4);
            rgb[0] *= factor;
            rgb[1] *= factor;
            rgb[2] *= factor;
        }
        TerrainKind::Sand => {
            let factor = (0.9 + sample.accent * 0.08).clamp(0.6, 1.3);
            rgb[0] *= factor;
            rgb[1] *= factor;
            rgb[2] *= factor;
        }
        TerrainKind::Rock => {
            let factor = (0.85 + slope * 0.3).clamp(0.6, 1.2);
            rgb[0] *= factor;
            rgb[1] *= factor;
            rgb[2] *= factor;
        }
        _ => {}
    }

    let clamped = [
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
    ];
    let linear = srgb_to_linear_rgb(clamped);
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
    use bevy::render::mesh::VertexAttributeValues;
    use scriptbots_core::{TerrainKind, TerrainLayer, TerrainTile};
    use slotmap::KeyData;

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
        let height = TerrainHeightSnapshot::new(&layer);
        let dims = height.dims;
        let cell = layer.cell_size() as f32;
        let world_size = Vec2::new(dims.x as f32 * cell, dims.y as f32 * cell);
        let color = TerrainColorMap {
            width: dims.x,
            height: dims.y,
            pixels: vec![255; (dims.x * dims.y * 4) as usize],
        };
        WorldSnapshot {
            tick: 42,
            world_size,
            agent_radius: 12.0,
            terrain_color: color,
            terrain_height: height,
            agents: Vec::new(),
        }
    }

    #[test]
    fn chunk_mesh_positions_match_heightfield() {
        let snapshot = sample_world_snapshot();
        let bounds = TerrainChunkBounds {
            origin: UVec2::ZERO,
            size: snapshot.terrain_height.dims,
        };
        let built = build_chunk_mesh(&snapshot, bounds, 100.0);

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
        assert!(built.stats.signature.max_height > 0.0);
    }

    #[test]
    fn agent_translation_respects_terrain_height() {
        let mut snapshot = sample_world_snapshot();
        snapshot.agents.push(AgentVisual {
            id: AgentId::from(KeyData::from_ffi(1)),
            position: Vec2::new(50.0, 50.0),
            color: [0.5, 0.5, 0.5],
            selection: SelectionState::Selected,
            health: 80.0,
            age: 10,
        });

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

fn sync_agents(
    snapshot: &WorldSnapshot,
    commands: &mut Commands,
    registry: &mut AgentRegistry,
    assets: &AgentMeshAsset,
    materials: &mut Assets<StandardMaterial>,
) {
    let mut seen: HashSet<AgentId> = HashSet::with_capacity(snapshot.agents.len());
    for agent in &snapshot.agents {
        seen.insert(agent.id);
        if let Some(record) = registry.records.get(&agent.id) {
            update_agent_entity(
                record,
                agent,
                snapshot,
                commands,
                materials,
                assets.base_radius,
            );
        } else {
            let record = spawn_agent_entity(agent, snapshot, commands, assets, materials);
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
            commands.entity(record.entity).despawn();
            materials.remove(&record.material);
        }
    }
}

fn spawn_agent_entity(
    agent: &AgentVisual,
    snapshot: &WorldSnapshot,
    commands: &mut Commands,
    assets: &AgentMeshAsset,
    materials: &mut Assets<StandardMaterial>,
) -> AgentRecord {
    let (base_color, emissive) = agent_colors(agent);
    let material = materials.add(StandardMaterial {
        base_color,
        emissive: emissive.into(),
        metallic: 0.04,
        perceptual_roughness: 0.58,
        reflectance: 0.08,
        ..Default::default()
    });

    let mut transform = Transform::from_translation(agent_translation(snapshot, agent));
    transform.scale = agent_scale(snapshot.agent_radius, assets.base_radius);

    let entity = commands
        .spawn((
            Mesh3d(assets.mesh.clone()),
            MeshMaterial3d(material.clone()),
            transform,
        ))
        .id();

    AgentRecord { entity, material }
}

fn update_agent_entity(
    record: &AgentRecord,
    agent: &AgentVisual,
    snapshot: &WorldSnapshot,
    commands: &mut Commands,
    materials: &mut Assets<StandardMaterial>,
    base_radius: f32,
) {
    let mut transform = Transform::from_translation(agent_translation(snapshot, agent));
    transform.scale = agent_scale(snapshot.agent_radius, base_radius);
    commands.entity(record.entity).insert((transform,));

    if let Some(material) = materials.get_mut(&record.material) {
        let (base_color, emissive) = agent_colors(agent);
        material.base_color = base_color;
        material.emissive = emissive.into();
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

fn agent_scale(agent_radius: f32, base_radius: f32) -> Vec3 {
    let scale = (agent_radius / base_radius).max(0.25);
    Vec3::new(scale, scale * 0.35, scale)
}

const TERRAIN_BASE_COLORS: [[f32; 3]; 6] = [
    [0.117_647, 0.247_059, 0.400_000], // Deep water
    [0.184_314, 0.450_980, 0.701_961], // Shallow water
    [0.694_118, 0.305_882, 0.027_451], // Sand
    [0.313_725, 0.662_745, 0.074_510], // Grass
    [0.474_510, 0.831_373, 0.427_451], // Bloom
    [0.662_745, 0.694_118, 0.729_412], // Rock
];

fn terrain_kind_color(kind: TerrainKind) -> [f32; 3] {
    TERRAIN_BASE_COLORS[terrain_kind_index(kind)]
}

fn terrain_kind_index(kind: TerrainKind) -> usize {
    match kind {
        TerrainKind::DeepWater => 0,
        TerrainKind::ShallowWater => 1,
        TerrainKind::Sand => 2,
        TerrainKind::Grass => 3,
        TerrainKind::Bloom => 4,
        TerrainKind::Rock => 5,
    }
}

fn agent_colors(agent: &AgentVisual) -> (Color, Color) {
    let mut rgb = agent.color;
    for c in &mut rgb {
        *c = c.clamp(0.0, 1.0);
    }
    let health_factor = (agent.health / 100.0).clamp(0.45, 1.0);
    let base = Color::srgb(
        rgb[0] * health_factor,
        rgb[1] * health_factor,
        rgb[2] * health_factor,
    );
    let highlight = match agent.selection {
        SelectionState::None => 0.12,
        SelectionState::Hovered => 0.28,
        SelectionState::Selected => 0.48,
    };
    let emissive = Color::linear_rgb(
        (rgb[0] + highlight * 0.8).min(1.0),
        (rgb[1] + highlight * 0.6).min(1.0),
        (rgb[2] + highlight).min(1.0),
    );
    (base, emissive)
}

fn close_on_esc(mut exit_events: MessageWriter<AppExit>, keyboard: Res<ButtonInput<KeyCode>>) {
    if keyboard.just_pressed(KeyCode::Escape) {
        exit_events.write(AppExit::Success);
    }
}

pub fn render_png_offscreen(world: &WorldState, width: u32, height: u32) -> Result<Vec<u8>> {
    let snapshot = WorldSnapshot::from_world(world)
        .ok_or_else(|| anyhow!("unable to build world snapshot for Bevy render"))?;
    let width = width.max(1);
    let height = height.max(1);

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
        let (base_color, _) = agent_colors(agent);
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
    command_drain: CommandDrainFn,
    controls: SimulationControl,
    running: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut last = Instant::now();
        let mut accumulator = 0.0f32;

        while running.load(Ordering::Relaxed) {
            let now = Instant::now();
            let mut dt = (now - last).as_secs_f32();
            last = now;
            if !dt.is_finite() || dt > 0.25 {
                dt = 0.25;
            }

            if let Ok(mut world_guard) = world.lock() {
                (command_drain.as_ref())(&mut world_guard);
                let pending = world_guard.drain_simulation_commands();
                if !pending.is_empty() {
                    for command in pending {
                        controls.update(|state| apply_simulation_command_to_state(state, &command));
                    }
                }
            }

            let (paused, speed, step_once) = {
                let mut paused = false;
                let mut speed = 1.0;
                let mut step_once = false;
                controls.update(|state| {
                    paused = state.paused;
                    speed = state.speed_multiplier.clamp(MIN_SPEED, MAX_SPEED);
                    if state.step_requested {
                        step_once = true;
                        state.step_requested = false;
                        state.paused = true;
                        state.auto_pause_reason = None;
                    }
                });
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

            if let Ok(mut world_guard) = world.lock() {
                if steps == 0 && !step_once && speed <= MIN_SPEED {
                    drop(world_guard);
                    thread::sleep(Duration::from_millis(4));
                    continue;
                }

                if steps == 0 && step_once {
                    steps = 1;
                }

                for _ in 0..steps {
                    world_guard.step();
                }

                let control = world_guard.config().control.clone();
                let agent_count = world_guard.agent_count();
                let max_age = world_guard.last_max_age();
                let spike_hits = world_guard.last_spike_hits();

                let mut reason: Option<String> = None;
                if control.auto_pause_on_spike_hit && spike_hits > 0 {
                    reason = Some(format!("Spike hits detected ({spike_hits})"));
                } else if let Some(age_limit) = control.auto_pause_age_above {
                    if max_age >= age_limit {
                        reason = Some(format!("Max age {max_age} ≥ {age_limit}"));
                    }
                } else if let Some(limit) = control.auto_pause_population_below {
                    if agent_count as u32 <= limit {
                        reason = Some(format!("Population {agent_count} ≤ {limit}"));
                    }
                }

                if let Some(reason) = reason {
                    controls.update(|state| {
                        state.paused = true;
                        state.auto_pause_reason = Some(reason.clone());
                        state.step_requested = false;
                    });
                    world_guard.enqueue_simulation_command(SimulationCommand {
                        paused: Some(true),
                        speed_multiplier: Some(0.0),
                        step_once: false,
                    });
                    info!(%reason, "Bevy simulation auto-paused");
                } else if steps > 0 {
                    controls.update(|state| {
                        state.auto_pause_reason = None;
                    });
                }

                drop(world_guard);
            }

            if steps == 0 {
                thread::sleep(Duration::from_millis(2));
            }
        }
    })
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
    use scriptbots_core::ScriptBotsConfig;
    use std::sync::{Arc, Mutex};

    #[test]
    fn bevy_offscreen_renderer_produces_png() -> Result<()> {
        let config = ScriptBotsConfig::default();
        let mut world = WorldState::new(config).expect("world initialization");
        for _ in 0..32 {
            world.step();
        }
        let png = render_png_offscreen(&world, 640, 360)?;
        assert!(png.len() > 4096, "expected non-trivial PNG output");
        assert_eq!(&png[0..8], b"\x89PNG\r\n\x1a\n", "invalid PNG header");
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
}
