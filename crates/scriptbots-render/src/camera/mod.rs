use gpui::{Pixels, Point, ScrollDelta, ScrollWheelEvent};
use scriptbots_core::Position;

#[derive(Clone, Copy, Debug)]
pub struct CameraConfig {
    pub min_zoom: f32,
    pub max_zoom: f32,
    pub legacy_scale: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            min_zoom: 0.4,
            max_zoom: 2.5,
            legacy_scale: 0.2,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct CameraState {
    offset_px: (f32, f32),
    zoom: f32,
    zoom_initialized: bool,
    last_canvas_origin: (f32, f32),
    last_canvas_size: (f32, f32),
    last_world_size: (f32, f32),
    last_scale: f32,
    last_base_scale: f32,
    centered_once: bool,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub struct CameraSnapshot {
    pub offset_px: (f32, f32),
    pub zoom: f32,
    pub last_canvas_origin: (f32, f32),
    pub last_canvas_size: (f32, f32),
    pub last_world_size: (f32, f32),
    pub last_scale: f32,
    pub last_base_scale: f32,
    pub centered_once: bool,
    pub zoom_initialized: bool,
    pub panning: bool,
}

pub struct Camera {
    config: CameraConfig,
    state: CameraState,
    panning: bool,
    pan_anchor: Option<Point<Pixels>>,
}

#[derive(Clone, Copy, Debug)]
pub struct ViewLayout {
    pub base_scale: f32,
    pub scale: f32,
    pub pad: (f32, f32),
    pub offset: (f32, f32),
    pub render_size: (f32, f32),
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(CameraConfig::default())
    }
}

impl Camera {
    pub fn new(config: CameraConfig) -> Self {
        Self {
            config,
            state: CameraState::default(),
            panning: false,
            pan_anchor: None,
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub fn zoom(&self) -> f32 {
        self.state.zoom
    }

    #[allow(dead_code)]
    #[inline]
    pub fn offset(&self) -> (f32, f32) {
        self.state.offset_px
    }

    #[inline]
    pub fn is_panning(&self) -> bool {
        self.panning
    }

    #[allow(dead_code)]
    #[inline]
    pub fn is_centered(&self) -> bool {
        self.state.centered_once
    }

    #[allow(dead_code)]
    #[inline]
    pub fn mark_centered(&mut self) {
        self.state.centered_once = true;
    }

    #[inline]
    pub fn ensure_default_zoom(&mut self, base_scale: f32) {
        if self.state.zoom_initialized || base_scale <= 0.0 {
            return;
        }
        let desired = (self.config.legacy_scale / base_scale)
            .clamp(self.config.min_zoom, self.config.max_zoom);
        self.state.zoom = desired;
        self.state.zoom_initialized = true;
    }

    pub fn start_pan(&mut self, cursor: Point<Pixels>) {
        self.panning = true;
        self.pan_anchor = Some(cursor);
    }

    pub fn update_pan(&mut self, cursor: Point<Pixels>) -> bool {
        if !self.panning {
            return false;
        }
        if let Some(anchor) = self.pan_anchor {
            let dx = f32::from(cursor.x) - f32::from(anchor.x);
            let dy = f32::from(cursor.y) - f32::from(anchor.y);
            if dx.abs() > f32::EPSILON || dy.abs() > f32::EPSILON {
                self.state.offset_px.0 += dx;
                self.state.offset_px.1 += dy;
                self.pan_anchor = Some(cursor);
                return true;
            }
        }
        false
    }

    pub fn end_pan(&mut self) {
        self.panning = false;
        self.pan_anchor = None;
    }

    pub fn apply_scroll(&mut self, event: &ScrollWheelEvent) -> bool {
        let scroll_y = match event.delta {
            ScrollDelta::Pixels(delta) => -f32::from(delta.y) / 120.0,
            ScrollDelta::Lines(lines) => -lines.y,
        };
        if scroll_y.abs() < 0.01 {
            return false;
        }

        let old_zoom = self.state.zoom;
        let new_zoom =
            (old_zoom * (1.0 + scroll_y * 0.1)).clamp(self.config.min_zoom, self.config.max_zoom);
        if (new_zoom - old_zoom).abs() <= f32::EPSILON {
            return false;
        }

        let canvas_x = f32::from(event.position.x);
        let canvas_y = f32::from(event.position.y);
        let origin_x = self.state.last_canvas_origin.0;
        let origin_y = self.state.last_canvas_origin.1;
        let pad_x = (self.state.last_canvas_size.0
            - self.state.last_world_size.0 * self.state.last_scale)
            * 0.5;
        let pad_y = (self.state.last_canvas_size.1
            - self.state.last_world_size.1 * self.state.last_scale)
            * 0.5;

        let cursor_x = canvas_x;
        let cursor_y = canvas_y;
        let old_scale = self.state.last_scale;
        let new_scale = self.state.last_base_scale * new_zoom;

        let world_x = (cursor_x - origin_x - pad_x - self.state.offset_px.0) / old_scale;
        let world_y = (cursor_y - origin_y - pad_y - self.state.offset_px.1) / old_scale;

        self.state.zoom = new_zoom;
        self.state.offset_px.0 = cursor_x - origin_x - pad_x - world_x * new_scale;
        self.state.offset_px.1 = cursor_y - origin_y - pad_y - world_y * new_scale;
        self.state.zoom_initialized = true;
        true
    }

    pub fn record_render_metrics(
        &mut self,
        canvas_origin: (f32, f32),
        canvas_size: (f32, f32),
        world_size: (f32, f32),
        base_scale: f32,
    ) {
        self.state.last_canvas_origin = canvas_origin;
        self.state.last_canvas_size = canvas_size;
        self.state.last_world_size = world_size;
        self.state.last_base_scale = base_scale;
        self.state.last_scale = base_scale * self.state.zoom;
    }

    pub fn center_on(&mut self, position: Position) {
        let scale = self.state.last_base_scale * self.state.zoom;
        if !scale.is_finite() || scale <= f32::EPSILON {
            return;
        }

        let center_x = self.state.last_canvas_origin.0 + self.state.last_canvas_size.0 * 0.5;
        let center_y = self.state.last_canvas_origin.1 + self.state.last_canvas_size.1 * 0.5;
        let pad_x = (self.state.last_canvas_size.0 - self.state.last_world_size.0 * scale) * 0.5;
        let pad_y = (self.state.last_canvas_size.1 - self.state.last_world_size.1 * scale) * 0.5;

        let world_screen_x =
            self.state.last_canvas_origin.0 + pad_x + self.state.offset_px.0 + position.x * scale;
        let world_screen_y =
            self.state.last_canvas_origin.1 + pad_y + self.state.offset_px.1 + position.y * scale;

        self.state.offset_px.0 += center_x - world_screen_x;
        self.state.offset_px.1 += center_y - world_screen_y;
    }

    pub fn screen_to_world(&self, point: Point<Pixels>) -> Option<(f32, f32)> {
        let scale = self.state.last_scale;
        if scale <= f32::EPSILON {
            return None;
        }
        let canvas_x = f32::from(point.x);
        let canvas_y = f32::from(point.y);
        let origin_x = self.state.last_canvas_origin.0;
        let origin_y = self.state.last_canvas_origin.1;
        let canvas_width = self.state.last_canvas_size.0;
        let canvas_height = self.state.last_canvas_size.1;
        let world_w = self.state.last_world_size.0;
        let world_h = self.state.last_world_size.1;

        let render_w = world_w * scale;
        let render_h = world_h * scale;
        let pad_x = (canvas_width - render_w) * 0.5;
        let pad_y = (canvas_height - render_h) * 0.5;

        let world_x = (canvas_x - origin_x - pad_x - self.state.offset_px.0) / scale;
        let world_y = (canvas_y - origin_y - pad_y - self.state.offset_px.1) / scale;

        if !world_x.is_finite() || !world_y.is_finite() {
            return None;
        }

        if world_x < 0.0 || world_y < 0.0 || world_x > world_w || world_y > world_h {
            return None;
        }

        Some((world_x, world_y))
    }

    pub fn snapshot(&self) -> CameraSnapshot {
        CameraSnapshot {
            offset_px: self.state.offset_px,
            zoom: self.state.zoom,
            last_canvas_origin: self.state.last_canvas_origin,
            last_canvas_size: self.state.last_canvas_size,
            last_world_size: self.state.last_world_size,
            last_scale: self.state.last_scale,
            last_base_scale: self.state.last_base_scale,
            centered_once: self.state.centered_once,
            zoom_initialized: self.state.zoom_initialized,
            panning: self.panning,
        }
    }

    #[allow(dead_code)]
    pub fn world_to_screen(&self, point: (f32, f32)) -> Option<(f32, f32)> {
        let scale = self.state.last_scale;
        if scale <= f32::EPSILON {
            return None;
        }

        let world_w = self.state.last_world_size.0;
        let world_h = self.state.last_world_size.1;
        if point.0 < 0.0 || point.0 > world_w || point.1 < 0.0 || point.1 > world_h {
            return None;
        }

        let pad_x = (self.state.last_canvas_size.0 - world_w * scale) * 0.5;
        let pad_y = (self.state.last_canvas_size.1 - world_h * scale) * 0.5;
        let x = self.state.last_canvas_origin.0 + pad_x + self.state.offset_px.0 + point.0 * scale;
        let y = self.state.last_canvas_origin.1 + pad_y + self.state.offset_px.1 + point.1 * scale;
        Some((x, y))
    }

    pub fn layout(
        &mut self,
        canvas_origin: (f32, f32),
        canvas_size: (f32, f32),
        world_size: (f32, f32),
    ) -> ViewLayout {
        let width_px = canvas_size.0.max(1.0);
        let height_px = canvas_size.1.max(1.0);
        let world_w = world_size.0.max(1.0);
        let world_h = world_size.1.max(1.0);
        let base_scale = (width_px / world_w).min(height_px / world_h).max(0.0001);

        self.ensure_default_zoom(base_scale);

        let mut layout = self.compute_layout(canvas_origin, canvas_size, world_size, base_scale);

        self.record_render_metrics(canvas_origin, canvas_size, world_size, base_scale);

        let world_center = Position {
            x: world_size.0 * 0.5,
            y: world_size.1 * 0.5,
        };

        if layout.fully_offscreen {
            self.center_on(world_center);
            layout = self.compute_layout(canvas_origin, canvas_size, world_size, base_scale);
            self.record_render_metrics(canvas_origin, canvas_size, world_size, base_scale);
        }

        if !self.state.centered_once {
            self.center_on(world_center);
            layout = self.compute_layout(canvas_origin, canvas_size, world_size, base_scale);
            self.state.centered_once = true;
            self.record_render_metrics(canvas_origin, canvas_size, world_size, base_scale);

            if layout.fully_offscreen {
                self.center_on(world_center);
                layout = self.compute_layout(canvas_origin, canvas_size, world_size, base_scale);
                self.record_render_metrics(canvas_origin, canvas_size, world_size, base_scale);
            }
        }

        ViewLayout {
            base_scale,
            scale: layout.scale,
            pad: (layout.pad_x, layout.pad_y),
            offset: (layout.offset_x, layout.offset_y),
            render_size: (layout.render_w, layout.render_h),
        }
    }

    fn compute_layout(
        &self,
        canvas_origin: (f32, f32),
        canvas_size: (f32, f32),
        world_size: (f32, f32),
        base_scale: f32,
    ) -> LayoutComputation {
        let width_px = canvas_size.0.max(1.0);
        let height_px = canvas_size.1.max(1.0);
        let world_w = world_size.0.max(1.0);
        let world_h = world_size.1.max(1.0);

        let scale = base_scale * self.state.zoom;
        let render_w = world_w * scale;
        let render_h = world_h * scale;
        let pad_x = (width_px - render_w) * 0.5;
        let pad_y = (height_px - render_h) * 0.5;
        let offset_x = canvas_origin.0 + pad_x + self.state.offset_px.0;
        let offset_y = canvas_origin.1 + pad_y + self.state.offset_px.1;
        let fully_offscreen = (offset_x + render_w) < canvas_origin.0
            || offset_x > (canvas_origin.0 + width_px)
            || (offset_y + render_h) < canvas_origin.1
            || offset_y > (canvas_origin.1 + height_px);

        LayoutComputation {
            scale,
            render_w,
            render_h,
            pad_x,
            pad_y,
            offset_x,
            offset_y,
            fully_offscreen,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::px;
    use scriptbots_core::ScriptBotsConfig;

    const VIEWPORT: (f32, f32) = (1600.0, 900.0);
    const WORLD: (f32, f32) = (6000.0, 3000.0);

    fn configured_camera() -> Camera {
        let mut camera = Camera::default();
        let base = (VIEWPORT.0 / WORLD.0).min(VIEWPORT.1 / WORLD.1);
        camera.ensure_default_zoom(base);
        camera.record_render_metrics((0.0, 0.0), VIEWPORT, WORLD, base);
        camera
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn screen_to_world_maps_visible_bounds() {
        let camera = configured_camera();
        let mid_point = Point {
            x: px(VIEWPORT.0 * 0.5),
            y: px(VIEWPORT.1 * 0.5),
        };
        let world_mid = camera
            .screen_to_world(mid_point)
            .expect("midpoint maps to world");
        let world_next_x = camera
            .screen_to_world(Point {
                x: px(VIEWPORT.0 * 0.5 + 1.0),
                y: px(VIEWPORT.1 * 0.5),
            })
            .expect("adjacent pixel maps to world");
        let world_next_y = camera
            .screen_to_world(Point {
                x: px(VIEWPORT.0 * 0.5),
                y: px(VIEWPORT.1 * 0.5 + 1.0),
            })
            .expect("adjacent pixel maps to world");

        let scale_x = 1.0 / (world_next_x.0 - world_mid.0).abs();
        let scale_y = 1.0 / (world_next_y.1 - world_mid.1).abs();
        let pad_x = (VIEWPORT.0 - WORLD.0 * scale_x) * 0.5;
        let pad_y = (VIEWPORT.1 - WORLD.1 * scale_y) * 0.5;

        let top_left = camera
            .screen_to_world(Point {
                x: px(pad_x),
                y: px(pad_y),
            })
            .expect("top-left visible bounds map");
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
                x: px(VIEWPORT.0 - pad_x),
                y: px(VIEWPORT.1 - pad_y),
            })
            .expect("bottom-right visible bounds map");
        assert!(
            approx_eq(bottom_right.0, WORLD.0, 1e-3),
            "expected x≈{}, got {}",
            WORLD.0,
            bottom_right.0
        );
        assert!(
            approx_eq(bottom_right.1, WORLD.1, 1e-3),
            "expected y≈{}, got {}",
            WORLD.1,
            bottom_right.1
        );
    }

    #[test]
    fn default_zoom_keeps_agents_visible() {
        let camera = configured_camera();
        let mid_point = Point {
            x: px(VIEWPORT.0 * 0.5),
            y: px(VIEWPORT.1 * 0.5),
        };
        let world_mid = camera
            .screen_to_world(mid_point)
            .expect("midpoint maps to world");
        let world_next = camera
            .screen_to_world(Point {
                x: px(VIEWPORT.0 * 0.5 + 1.0),
                y: px(VIEWPORT.1 * 0.5),
            })
            .expect("adjacent pixel maps to world");

        let world_units_per_px = (world_next.0 - world_mid.0).abs();
        assert!(
            world_units_per_px > 0.0,
            "world units per pixel should be positive"
        );

        let pixels_per_world = 1.0 / world_units_per_px;
        let bot_radius = ScriptBotsConfig::default().bot_radius;
        let pixel_radius = pixels_per_world * bot_radius;

        assert!(
            pixel_radius >= 2.0,
            "expected pixel radius ≥ 2.0, got {}",
            pixel_radius
        );
    }

    #[test]
    fn world_to_screen_round_trip() {
        let camera = configured_camera();
        let world_point = (WORLD.0 * 0.25, WORLD.1 * 0.75);
        let screen = camera
            .world_to_screen(world_point)
            .expect("world point converts to screen");
        let recovered = camera
            .screen_to_world(Point {
                x: px(screen.0),
                y: px(screen.1),
            })
            .expect("screen point converts back to world");
        assert!(
            approx_eq(world_point.0, recovered.0, 1e-3)
                && approx_eq(world_point.1, recovered.1, 1e-3),
            "round-trip mismatch: {:?} vs {:?}",
            world_point,
            recovered
        );
    }
}

struct LayoutComputation {
    scale: f32,
    render_w: f32,
    render_h: f32,
    pad_x: f32,
    pad_y: f32,
    offset_x: f32,
    offset_y: f32,
    fully_offscreen: bool,
}
