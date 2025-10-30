//! Bevy renderer scaffold for ScriptBots.

use anyhow::Result;
use bevy::{
    app::AppExit,
    prelude::*,
};
use tracing::info;

/// Launch the Bevy renderer stub and block until the window closes.
pub fn run_stub_renderer() -> Result<()> {
    info!("Launching Bevy renderer stub (Phase 0)");

    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.05, 0.07, 0.10)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "ScriptBots • Bevy (stub)".to_string(),
                present_mode: bevy::window::PresentMode::AutoVsync,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_systems(Startup, setup_scene)
        .add_systems(Update, close_on_esc);
    app.run();
    Ok(())
}

fn setup_scene(mut commands: Commands) {
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 5.0, 12.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });

    commands.spawn(DirectionalLightBundle {
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
            0.0,
        )),
        ..Default::default()
    });

    commands.spawn(PbrBundle {
        mesh: Mesh::from(shape::Plane::from_size(10.0)),
        material: StandardMaterial {
            base_color: Color::srgb(0.10, 0.16, 0.22),
            ..Default::default()
        },
        ..Default::default()
    });
}

fn close_on_esc(mut exit_events: EventWriter<AppExit>, keyboard: Res<Input<KeyCode>>) {
    if keyboard.just_pressed(KeyCode::Escape) {
        exit_events.send(AppExit);
    }
}
