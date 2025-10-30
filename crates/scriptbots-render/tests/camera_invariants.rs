use gpui::{px, Point};
use scriptbots_core::ScriptBotsConfig;
use scriptbots_render::CameraState;

const VIEWPORT_WIDTH: f32 = 1600.0;
const VIEWPORT_HEIGHT: f32 = 900.0;
const WORLD_WIDTH: f32 = 6000.0;
const WORLD_HEIGHT: f32 = 3000.0;

fn base_scale() -> f32 {
    (VIEWPORT_WIDTH / WORLD_WIDTH).min(VIEWPORT_HEIGHT / WORLD_HEIGHT)
}

fn configured_camera() -> CameraState {
    let mut camera = CameraState::default();
    let base = base_scale();
    camera.ensure_default_zoom(base);
    camera.record_render_metrics(
        (0.0, 0.0),
        (VIEWPORT_WIDTH, VIEWPORT_HEIGHT),
        (WORLD_WIDTH, WORLD_HEIGHT),
        base,
    );
    camera
}

fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() <= epsilon
}

#[test]
fn screen_to_world_maps_corners_within_world_bounds() {
    let camera = configured_camera();

    let top_left = camera
        .screen_to_world(Point {
            x: px(0.0),
            y: px(0.0),
        })
        .expect("top-left maps");
    assert!(
        approx_eq(top_left.0, 0.0, 1e-3),
        "expected x≈0.0, got {}",
        top_left.0
    );
    assert!(
        approx_eq(top_left.1, 0.0, 1e-3),
        "expected y≈0.0, got {}",
        top_left.1
    );

    let bottom_right = camera
        .screen_to_world(Point {
            x: px(VIEWPORT_WIDTH),
            y: px(VIEWPORT_HEIGHT),
        })
        .expect("bottom-right maps");
    assert!(
        approx_eq(bottom_right.0, WORLD_WIDTH, 1e-3),
        "expected x≈{}, got {}",
        WORLD_WIDTH,
        bottom_right.0
    );
    assert!(
        approx_eq(bottom_right.1, WORLD_HEIGHT, 1e-3),
        "expected y≈{}, got {}",
        WORLD_HEIGHT,
        bottom_right.1
    );
}

#[test]
fn default_zoom_keeps_agents_visible() {
    let camera = configured_camera();
    let mid_point = Point {
        x: px(VIEWPORT_WIDTH * 0.5),
        y: px(VIEWPORT_HEIGHT * 0.5),
    };

    let world_mid = camera
        .screen_to_world(mid_point)
        .expect("midpoint maps to world");
    let world_next_px = camera
        .screen_to_world(Point {
            x: px(VIEWPORT_WIDTH * 0.5 + 1.0),
            y: px(VIEWPORT_HEIGHT * 0.5),
        })
        .expect("offset maps to world");
    let world_units_per_pixel = (world_next_px.0 - world_mid.0).abs();
    assert!(
        world_units_per_pixel > 0.0,
        "world units per pixel should be positive"
    );

    let pixels_per_world = 1.0 / world_units_per_pixel;
    let bot_radius = ScriptBotsConfig::default().bot_radius;
    let pixel_radius = pixels_per_world * bot_radius;
    assert!(
        pixel_radius >= 2.0,
        "expected pixel radius ≥ 2.0, got {}",
        pixel_radius
    );
}
