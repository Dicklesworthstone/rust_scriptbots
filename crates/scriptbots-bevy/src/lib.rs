//! Bevy renderer integration for ScriptBots.

use anyhow::{Result, anyhow};
use bevy::render::{
    render_asset::RenderAssetUsages,
    render_resource::{Extent3d, TextureDimension, TextureFormat},
    texture::ImageSampler,
};
use bevy::{
    app::AppExit,
    ecs::schedule::IntoSystemConfigs,
    ecs::system::NonSendMut,
    input::mouse::{MouseMotion, MouseWheel},
    math::primitives::{Plane3d, Sphere},
    prelude::*,
    window::{PresentMode, WindowPlugin},
};
use image::{ImageBuffer, Rgba as ImgRgba};
use scriptbots_core::{AgentId, SelectionState, TerrainKind, WorldState};
use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};
use tracing::info;

/// Launch context supplied by the ScriptBots application shell.
pub struct BevyRendererContext {
    pub world: Arc<Mutex<WorldState>>,
}

/// Entry point for the Bevy renderer; blocks until the window closes.
pub fn run_renderer(ctx: BevyRendererContext) -> Result<()> {
    info!("Launching Bevy renderer (Phase 1: static world visuals)");

    let (tx, rx) = mpsc::channel::<WorldSnapshot>();
    let running = Arc::new(AtomicBool::new(true));
    let worker_flag = Arc::clone(&running);
    let world = Arc::clone(&ctx.world);

    let worker = thread::spawn(move || {
        let mut last_tick = 0u64;
        while worker_flag.load(Ordering::Relaxed) {
            let snapshot = {
                let guard = world.lock().expect("world mutex poisoned");
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

    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.03, 0.05, 0.09)))
        .insert_resource(AmbientLight {
            color: Color::srgb(0.45, 0.52, 0.65),
            brightness: 800.0,
        })
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
            (poll_snapshots, sync_world, control_camera, update_hud).chain(),
        )
        .add_systems(Update, close_on_esc);

    app.run();

    running.store(false, Ordering::Relaxed);
    let _ = worker.join();
    Ok(())
}

#[derive(Component)]
struct GroundPlane;

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

const CAMERA_MIN_DISTANCE: f32 = 300.0;
const CAMERA_MAX_DISTANCE: f32 = 6000.0;
const CAMERA_SMOOTHING_LERP: f32 = 8.0;
const FIT_WORLD_FACTOR: f32 = 0.38;
const FIT_SELECTION_FACTOR: f32 = 0.55;

fn bounds_extent(bounds: (Vec2, Vec2)) -> Vec2 {
    let size = bounds.1 - bounds.0;
    Vec2::new(size.x.abs().max(1.0), size.y.abs().max(1.0))
}

fn fit_distance_for_extent(extent: Vec2, factor: f32) -> f32 {
    let max_extent = extent.max_element().max(200.0);
    (max_extent * factor).clamp(CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE)
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
struct TerrainResources {
    texture: Handle<Image>,
    dims: (u32, u32),
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

#[derive(Clone)]
struct TerrainImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

#[derive(Clone)]
struct WorldSnapshot {
    tick: u64,
    world_size: Vec2,
    agent_radius: f32,
    terrain: TerrainImage,
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
            terrain: TerrainImage {
                width: terrain_w,
                height: terrain_h,
                pixels: terrain_pixels,
            },
            agents,
        })
    }
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 1800.0, 1400.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        },
        PrimaryCamera,
    ));

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 9000.0,
            shadows_enabled: true,
            ..Default::default()
        },
        transform: Transform::from_xyz(-1200.0, 1800.0, 900.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });

    let mut placeholder_image = Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[24, 32, 44, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    placeholder_image.sampler = ImageSampler::nearest();
    let terrain_texture = images.add(placeholder_image);

    let plane_mesh = meshes.add(Mesh::from(Plane3d::default()));
    let plane_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(terrain_texture.clone()),
        perceptual_roughness: 0.92,
        metallic: 0.02,
        reflectance: 0.06,
        ..Default::default()
    });

    commands.spawn((
        PbrBundle {
            mesh: plane_mesh,
            material: plane_material.clone(),
            transform: Transform::default(),
            ..Default::default()
        },
        GroundPlane,
    ));

    let agent_mesh = meshes.add(Mesh::from(Sphere::new(1.0)));
    commands.insert_resource(AgentMeshAsset {
        mesh: agent_mesh,
        base_radius: 1.0,
    });
    commands.insert_resource(TerrainResources {
        texture: terrain_texture,
        dims: (1, 1),
    });
    commands.insert_resource(CameraRig::default());

    commands.spawn(Camera2dBundle::default());

    let text_style = TextStyle {
        font_size: 18.0,
        color: Color::WHITE,
        ..Default::default()
    };
    let secondary_style = TextStyle {
        font_size: 15.0,
        color: Color::srgb(0.74, 0.82, 0.94),
        ..Default::default()
    };

    let hud_root = commands
        .spawn(NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                padding: UiRect::all(Val::Px(10.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                ..Default::default()
            },
            background_color: Color::srgba(0.07, 0.11, 0.18, 0.72).into(),
            ..Default::default()
        })
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
            .spawn(TextBundle::from_section("Tick: --", text_style.clone()))
            .id();
        agents = parent
            .spawn(TextBundle::from_section("Agents: --", text_style.clone()))
            .id();
        selection = parent
            .spawn(TextBundle::from_section(
                "Selection: --",
                secondary_style.clone(),
            ))
            .id();
        follow = parent
            .spawn(TextBundle::from_section("Follow: --", text_style.clone()))
            .id();
        camera = parent
            .spawn(TextBundle::from_section("Camera: --", text_style.clone()))
            .id();
        playback = parent
            .spawn(TextBundle::from_section(
                "Playback: --",
                secondary_style.clone(),
            ))
            .id();
        fps = parent
            .spawn(TextBundle::from_section(
                "FPS: --",
                secondary_style.clone(),
            ))
            .id();
        world = parent
            .spawn(TextBundle::from_section(
                "World: --",
                secondary_style.clone(),
            ))
            .id();
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
    mut terrain: ResMut<TerrainResources>,
    mut images: ResMut<Assets<Image>>,
    assets: Res<AgentMeshAsset>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut plane_query: Query<&mut Transform, With<GroundPlane>>,
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
    let focus_point = selection_center
        .or(first_agent)
        .unwrap_or(world_center);

    align_ground_plane(snapshot, &mut plane_query);
    update_terrain_texture(snapshot, &mut terrain, &mut images);
    sync_agents(
        snapshot,
        &mut commands,
        &mut registry,
        &assets,
        &mut materials,
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
        (tick, agent_count, world_size, agent_radius, selected_count, primary)
    };

    if tick != state.last_reported_tick && tick % 120 == 0 {
        info!(
            tick,
            agents = agent_count,
            "Bevy world snapshot applied"
        );
    }
    state.last_reported_tick = tick;

    let now = time.elapsed_seconds_f64();
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
    let playback_status = if now - state.hud_prev_time > 0.75 {
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
            text.sections[0].value = format!("Tick: {}", tick);
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.agents) {
            text.sections[0].value = format!(
                "Agents: {:>4} (selected {:>2})",
                agent_count, selected_count
            );
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.selection) {
            text.sections[0].value = selection_text;
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.follow) {
            text.sections[0].value = format!(
                "Follow: {} • F cycle • Ctrl+S sel • Ctrl+O oldest",
                rig.follow_mode.label()
            );
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.camera) {
            let yaw_deg = rig.yaw.to_degrees();
            let pitch_deg = rig.pitch.to_degrees();
            text.sections[0].value = format!(
                    "Camera: dist {:>5.0} yaw {:>6.1}° pitch {:>5.1}° • Ctrl+F fit selection • Ctrl+W fit world",
                    rig.distance,
                    yaw_deg,
                    pitch_deg
                );
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.playback) {
            if state.sim_rate.is_finite() && state.sim_rate > 0.0 {
                text.sections[0].value = format!(
                    "Playback: {} • {:>4.1} t/s",
                    playback_status,
                    state.sim_rate.clamp(0.0, 999.9)
                );
            } else {
                text.sections[0].value = format!("Playback: {}", playback_status);
            }
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.fps) {
            let delta_seconds = time.delta_seconds();
            let fps = if delta_seconds > f32::EPSILON {
                1.0 / delta_seconds
            } else {
                0.0
            };
            text.sections[0].value = format!("FPS: {:>5.1}", fps);
        }
        if let Ok(mut text) = texts.get_mut(hud_elements.world) {
            text.sections[0].value = format!(
                "World: {:>4.0}×{:>4.0} • r {:>4.0}",
                world_size.x, world_size.y, agent_radius
            );
        }
    }
}

fn control_camera(
    time: Res<Time>,
    mut rig: ResMut<CameraRig>,
    state: Res<SnapshotState>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    mut camera_query: Query<&mut Transform, With<PrimaryCamera>>,
) {
    let Ok(mut transform) = camera_query.get_single_mut() else {
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
    rig.distance = rig
        .distance
        .clamp(CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE);

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
        rig.yaw += time.delta_seconds() * 1.2;
    }
    if keys.pressed(KeyCode::KeyE) {
        rig.yaw -= time.delta_seconds() * 1.2;
    }
    if keys.pressed(KeyCode::PageUp) {
        rig.pitch += time.delta_seconds() * 0.8;
    }
    if keys.pressed(KeyCode::PageDown) {
        rig.pitch -= time.delta_seconds() * 0.8;
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
        let delta = (right * pan_input.x + forward * pan_input.y) * 600.0 * time.delta_seconds();
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
                    let distance =
                        fit_distance_for_extent(state.world_size, FIT_WORLD_FACTOR);
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
                        let distance =
                            fit_distance_for_extent(extent, FIT_SELECTION_FACTOR);
                        rig.distance = distance;
                        rig.distance_smoothed = distance;
                    } else if let Some(selected) = state.first_agent_position {
                        focus_override = Some(selected);
                        let distance = fit_distance_for_extent(
                            Vec2::splat(400.0),
                            FIT_SELECTION_FACTOR,
                        );
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

    let smoothing = 1.0 - (-time.delta_seconds() * CAMERA_SMOOTHING_LERP).exp();
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

fn align_ground_plane(
    snapshot: &WorldSnapshot,
    plane_query: &mut Query<&mut Transform, With<GroundPlane>>,
) {
    if let Ok(mut transform) = plane_query.get_single_mut() {
        let width = snapshot.world_size.x.max(1.0);
        let height = snapshot.world_size.y.max(1.0);
        transform.scale = Vec3::new(width, 1.0, height);
    }
}

fn update_terrain_texture(
    snapshot: &WorldSnapshot,
    terrain: &mut TerrainResources,
    images: &mut Assets<Image>,
) {
    if let Some(image) = images.get_mut(&terrain.texture) {
        let new_image = Image::new_fill(
            Extent3d {
                width: snapshot.terrain.width.max(1),
                height: snapshot.terrain.height.max(1),
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &snapshot.terrain.pixels,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
        );
        *image = new_image;
        image.sampler = ImageSampler::nearest();
    }
    terrain.dims = (snapshot.terrain.width, snapshot.terrain.height);
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
            commands.entity(record.entity).despawn_recursive();
            materials.remove(record.material.id());
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
        .spawn(PbrBundle {
            mesh: assets.mesh.clone(),
            material: material.clone(),
            transform,
            ..Default::default()
        })
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
    commands.entity(record.entity).insert(transform);

    if let Some(material) = materials.get_mut(&record.material) {
        let (base_color, emissive) = agent_colors(agent);
        material.base_color = base_color;
        material.emissive = emissive.into();
    }
}

fn agent_translation(snapshot: &WorldSnapshot, agent: &AgentVisual) -> Vec3 {
    let half = snapshot.world_size * 0.5;
    let x = agent.position.x - half.x;
    let z = half.y - agent.position.y;
    Vec3::new(x, snapshot.agent_radius * 0.35, z)
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

fn close_on_esc(mut exit_events: EventWriter<AppExit>, keyboard: Res<ButtonInput<KeyCode>>) {
    if keyboard.just_pressed(KeyCode::Escape) {
        exit_events.send(AppExit::Success);
    }
}

pub fn render_png_offscreen(world: &WorldState, width: u32, height: u32) -> Result<Vec<u8>> {
    let snapshot = WorldSnapshot::from_world(world)
        .ok_or_else(|| anyhow!("unable to build world snapshot for Bevy render"))?;
    let width = width.max(1);
    let height = height.max(1);

    let mut image = ImageBuffer::<ImgRgba<u8>, Vec<u8>>::new(width, height);

    let terrain_w = snapshot.terrain.width.max(1);
    let terrain_h = snapshot.terrain.height.max(1);

    for y in 0..height {
        let tile_y = (terrain_h as u64 - 1)
            .saturating_sub(((y as u64) * terrain_h as u64) / height as u64)
            as u32;
        for x in 0..width {
            let tile_x =
                ((x as u64) * terrain_w as u64 / width as u64).min((terrain_w - 1) as u64) as u32;
            let idx = ((tile_y * terrain_w) + tile_x) as usize * 4;
            let px = ImgRgba([
                snapshot.terrain.pixels[idx],
                snapshot.terrain.pixels[idx + 1],
                snapshot.terrain.pixels[idx + 2],
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
}
