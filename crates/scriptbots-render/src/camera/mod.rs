use gpui::{Pixels, Point, ScrollWheelEvent, px};
use scriptbots_core::Position;

const SCROLL_LINE_HEIGHT_PX: f32 = 20.0;
const ZOOM_PER_SCROLL_LINE: f32 = 1.1;
const INITIAL_AGENT_DIAMETERS_ACROSS: f32 = 120.0;

#[derive(Clone, Copy, Debug)]
pub struct CameraConfig {
    pub min_zoom: f32,
    pub max_zoom: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            min_zoom: 0.4,
            max_zoom: 2.5,
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
    initial_population_view_resolved: bool,
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

impl CameraSnapshot {
    pub fn world_to_screen(&self, point: (f32, f32)) -> Option<(f32, f32)> {
        let scale = self.last_scale;
        if scale <= f32::EPSILON {
            return None;
        }

        let world_w = self.last_world_size.0;
        let world_h = self.last_world_size.1;
        if point.0 < 0.0 || point.0 > world_w || point.1 < 0.0 || point.1 > world_h {
            return None;
        }

        let pad_x = (self.last_canvas_size.0 - world_w * scale) * 0.5;
        let pad_y = (self.last_canvas_size.1 - world_h * scale) * 0.5;
        let x = self.last_canvas_origin.0 + pad_x + self.offset_px.0 + point.0 * scale;
        let y = self.last_canvas_origin.1 + pad_y + self.offset_px.1 + point.1 * scale;
        Some((x, y))
    }
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
    pub fn cancel_initial_population_view(&mut self) {
        self.state.initial_population_view_resolved = true;
    }

    #[inline]
    pub fn ensure_default_zoom(&mut self, base_scale: f32) {
        if self.state.zoom_initialized || base_scale <= 0.0 {
            return;
        }
        self.state.zoom = 1.0_f32.clamp(self.config.min_zoom, self.config.max_zoom);
        self.state.zoom_initialized = true;
    }

    pub fn fit_world(&mut self) {
        if self.state.last_base_scale <= f32::EPSILON {
            return;
        }
        // `last_base_scale` is already the exact aspect-preserving scale that fits the
        // whole world inside the current canvas. Applying the historical 0.2 world-unit
        // scale a second time turned "Fit World" into "shrink world", commonly hitting
        // the 0.4× clamp and surrounding the simulation with dead space.
        self.state.zoom = 1.0_f32.clamp(self.config.min_zoom, self.config.max_zoom);
        self.state.offset_px = (0.0, 0.0);
        self.state.centered_once = true;
        self.state.zoom_initialized = true;
    }

    pub fn fit_bounds(&mut self, bounds_min: Position, bounds_max: Position, padding_px: f32) {
        if self.state.last_base_scale <= f32::EPSILON {
            return;
        }
        let canvas = self.state.last_canvas_size;
        if canvas.0 <= f32::EPSILON || canvas.1 <= f32::EPSILON {
            return;
        }

        let min_x = bounds_min.x.min(bounds_max.x);
        let max_x = bounds_min.x.max(bounds_max.x);
        let min_y = bounds_min.y.min(bounds_max.y);
        let max_y = bounds_min.y.max(bounds_max.y);

        let min_extent = 200.0;
        let width_world = (max_x - min_x).max(min_extent);
        let height_world = (max_y - min_y).max(min_extent);

        let available_w = (canvas.0 - padding_px * 2.0).max(32.0);
        let available_h = (canvas.1 - padding_px * 2.0).max(32.0);

        let target_scale_x = available_w / width_world;
        let target_scale_y = available_h / height_world;
        let target_scale = target_scale_x.min(target_scale_y).max(1e-6);

        let base_scale = self.state.last_base_scale;
        let mut target_zoom = target_scale / base_scale;
        target_zoom = target_zoom.clamp(self.config.min_zoom, self.config.max_zoom);
        self.state.zoom = target_zoom;
        self.state.zoom_initialized = true;

        let center = Position {
            x: (min_x + max_x) * 0.5,
            y: (min_y + max_y) * 0.5,
        };
        self.center_on(center);
        self.state.centered_once = true;
    }

    pub fn start_pan(&mut self, cursor: Point<Pixels>) {
        self.cancel_initial_population_view();
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
        // GPUI normalizes platform wheel direction: positive Y is ScrollUp on every backend.
        // It does not expose a second "natural scrolling" flag, so applying another inversion
        // here reverses the user's platform setting.
        let scroll_lines =
            f32::from(event.delta.pixel_delta(px(SCROLL_LINE_HEIGHT_PX)).y) / SCROLL_LINE_HEIGHT_PX;
        if !scroll_lines.is_finite() || scroll_lines.abs() < 0.01 {
            return false;
        }
        self.cancel_initial_population_view();

        let old_zoom = self.state.zoom;
        let base_scale = self.state.last_base_scale;
        if !old_zoom.is_finite()
            || old_zoom <= 0.0
            || !base_scale.is_finite()
            || base_scale <= f32::EPSILON
        {
            return false;
        }

        let new_zoom = (old_zoom * ZOOM_PER_SCROLL_LINE.powf(scroll_lines))
            .clamp(self.config.min_zoom, self.config.max_zoom);
        if (new_zoom - old_zoom).abs() <= f32::EPSILON {
            return false;
        }

        let canvas_x = f32::from(event.position.x);
        let canvas_y = f32::from(event.position.y);
        let origin_x = self.state.last_canvas_origin.0;
        let origin_y = self.state.last_canvas_origin.1;
        let canvas_size = self.state.last_canvas_size;
        let world_size = self.state.last_world_size;
        let old_scale = base_scale * old_zoom;
        let new_scale = base_scale * new_zoom;
        let old_pad_x = (canvas_size.0 - world_size.0 * old_scale) * 0.5;
        let old_pad_y = (canvas_size.1 - world_size.1 * old_scale) * 0.5;
        let new_pad_x = (canvas_size.0 - world_size.0 * new_scale) * 0.5;
        let new_pad_y = (canvas_size.1 - world_size.1 * new_scale) * 0.5;

        let world_x = (canvas_x - origin_x - old_pad_x - self.state.offset_px.0) / old_scale;
        let world_y = (canvas_y - origin_y - old_pad_y - self.state.offset_px.1) / old_scale;
        if !world_x.is_finite() || !world_y.is_finite() {
            return false;
        }

        self.state.zoom = new_zoom;
        self.state.offset_px.0 = canvas_x - origin_x - new_pad_x - world_x * new_scale;
        self.state.offset_px.1 = canvas_y - origin_y - new_pad_y - world_y * new_scale;
        self.state.last_scale = new_scale;
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

    pub fn layout_with_initial_population<I>(
        &mut self,
        canvas_origin: (f32, f32),
        canvas_size: (f32, f32),
        world_size: (f32, f32),
        agent_radius: f32,
        positions: I,
    ) -> ViewLayout
    where
        I: IntoIterator<Item = Position>,
    {
        self.try_initialize_population_view(
            canvas_origin,
            canvas_size,
            world_size,
            agent_radius,
            positions,
        );
        self.layout(canvas_origin, canvas_size, world_size)
    }

    fn try_initialize_population_view<I>(
        &mut self,
        canvas_origin: (f32, f32),
        canvas_size: (f32, f32),
        world_size: (f32, f32),
        agent_radius: f32,
        positions: I,
    ) -> bool
    where
        I: IntoIterator<Item = Position>,
    {
        if self.state.initial_population_view_resolved
            || !canvas_size.0.is_finite()
            || !canvas_size.1.is_finite()
            || canvas_size.0 <= f32::EPSILON
            || canvas_size.1 <= f32::EPSILON
            || !world_size.0.is_finite()
            || !world_size.1.is_finite()
            || world_size.0 <= f32::EPSILON
            || world_size.1 <= f32::EPSILON
            || !agent_radius.is_finite()
            || agent_radius <= f32::EPSILON
        {
            return false;
        }

        let Some(center) = population_density_center(positions, world_size) else {
            return false;
        };

        let base_scale = (canvas_size.0 / world_size.0)
            .min(canvas_size.1 / world_size.1)
            .max(0.0001);
        let target_world_span = INITIAL_AGENT_DIAMETERS_ACROSS * agent_radius * 2.0;
        let target_scale = canvas_size.0 / target_world_span;
        let target_zoom =
            (target_scale / base_scale).clamp(self.config.min_zoom, self.config.max_zoom);
        if !target_zoom.is_finite() || target_zoom <= f32::EPSILON {
            return false;
        }

        self.state.zoom = target_zoom;
        self.state.zoom_initialized = true;
        self.state.offset_px = (0.0, 0.0);
        self.record_render_metrics(canvas_origin, canvas_size, world_size, base_scale);
        self.center_on(center);
        self.state.centered_once = true;
        self.state.initial_population_view_resolved = true;
        true
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

fn population_density_center<I>(positions: I, world_size: (f32, f32)) -> Option<Position>
where
    I: IntoIterator<Item = Position>,
{
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for position in positions {
        if position.x.is_finite() && position.y.is_finite() {
            xs.push(position.x.rem_euclid(world_size.0));
            ys.push(position.y.rem_euclid(world_size.1));
        }
    }

    Some(Position::new(
        toroidal_median(&mut xs, world_size.0)?,
        toroidal_median(&mut ys, world_size.1)?,
    ))
}

fn toroidal_median(values: &mut [f32], extent: f32) -> Option<f32> {
    if values.is_empty() || !extent.is_finite() || extent <= f32::EPSILON {
        return None;
    }
    values.sort_by(f32::total_cmp);

    let mut largest_gap = f32::NEG_INFINITY;
    let mut unwrap_origin = values[0];
    for index in 0..values.len() {
        let current = values[index];
        let next = if index + 1 < values.len() {
            values[index + 1]
        } else {
            values[0] + extent
        };
        let gap = next - current;
        if gap > largest_gap {
            largest_gap = gap;
            unwrap_origin = if index + 1 < values.len() {
                values[index + 1]
            } else {
                values[0]
            };
        }
    }

    for value in values.iter_mut() {
        *value = (*value - unwrap_origin).rem_euclid(extent);
    }
    values.sort_by(f32::total_cmp);
    let midpoint = values.len() / 2;
    let median = if values.len() % 2 == 0 {
        let lower = values[midpoint - 1];
        lower + (values[midpoint] - lower) * 0.5
    } else {
        values[midpoint]
    };
    Some((unwrap_origin + median).rem_euclid(extent))
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

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{ScrollDelta, px};
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

    fn scroll_event(position: (f32, f32), delta_y: f32) -> ScrollWheelEvent {
        ScrollWheelEvent {
            position: Point {
                x: px(position.0),
                y: px(position.1),
            },
            delta: ScrollDelta::Lines(Point { x: 0.0, y: delta_y }),
            ..ScrollWheelEvent::default()
        }
    }

    #[test]
    fn positive_platform_scroll_zooms_in_and_inverse_scroll_restores_zoom() {
        let mut camera = configured_camera();
        let original_zoom = camera.zoom();
        let cursor = (VIEWPORT.0 * 0.5, VIEWPORT.1 * 0.5);

        assert!(camera.apply_scroll(&scroll_event(cursor, 1.0)));
        assert!(
            camera.zoom() > original_zoom,
            "positive GPUI scroll delta must zoom in without a second platform-direction inversion"
        );

        assert!(camera.apply_scroll(&scroll_event(cursor, -1.0)));
        assert!(
            approx_eq(camera.zoom(), original_zoom, 1e-6),
            "equal opposite notches must be multiplicative inverses: {} versus {original_zoom}",
            camera.zoom()
        );
    }

    #[test]
    fn precise_pixels_and_lines_produce_the_same_exponential_zoom_step() {
        let mut line_camera = configured_camera();
        let mut pixel_camera = configured_camera();
        let cursor = Point {
            x: px(VIEWPORT.0 * 0.5),
            y: px(VIEWPORT.1 * 0.5),
        };

        assert!(line_camera.apply_scroll(&ScrollWheelEvent {
            position: cursor,
            delta: ScrollDelta::Lines(Point { x: 0.0, y: 1.0 }),
            ..ScrollWheelEvent::default()
        }));
        assert!(pixel_camera.apply_scroll(&ScrollWheelEvent {
            position: cursor,
            delta: ScrollDelta::Pixels(Point {
                x: px(0.0),
                y: px(SCROLL_LINE_HEIGHT_PX),
            }),
            ..ScrollWheelEvent::default()
        }));

        assert!(
            approx_eq(line_camera.zoom(), pixel_camera.zoom(), 1e-6),
            "one precise pixel line and one wheel line must have the same zoom factor"
        );
        assert!(
            approx_eq(line_camera.zoom(), ZOOM_PER_SCROLL_LINE, 1e-6,),
            "one positive line must multiply zoom by the documented exponential step"
        );
    }

    #[test]
    fn scroll_zoom_keeps_the_world_point_under_an_off_center_cursor() {
        let mut camera = Camera::default();
        let canvas_origin = (137.0, 83.0);
        let base = (VIEWPORT.0 / WORLD.0).min(VIEWPORT.1 / WORLD.1);
        camera.ensure_default_zoom(base);
        camera.record_render_metrics(canvas_origin, VIEWPORT, WORLD, base);
        let cursor = Point {
            x: px(canvas_origin.0 + VIEWPORT.0 * 0.75),
            y: px(canvas_origin.1 + VIEWPORT.1 / 3.0),
        };
        let world_before = camera
            .screen_to_world(cursor)
            .expect("off-center cursor starts over the world");

        assert!(camera.apply_scroll(&ScrollWheelEvent {
            position: cursor,
            delta: ScrollDelta::Lines(Point { x: 0.0, y: 1.0 }),
            ..ScrollWheelEvent::default()
        }));

        let world_after = camera
            .screen_to_world(cursor)
            .expect("off-center cursor remains over the world");
        assert!(
            approx_eq(world_before.0, world_after.0, 1e-3)
                && approx_eq(world_before.1, world_after.1, 1e-3),
            "cursor anchor drifted from {world_before:?} to {world_after:?}"
        );
        let snapshot = camera.snapshot();
        assert!(
            approx_eq(
                snapshot.last_scale,
                snapshot.last_base_scale * snapshot.zoom,
                f32::EPSILON,
            ),
            "scroll must update the cached scale before the next repaint"
        );
    }

    #[test]
    fn horizontal_only_scroll_is_not_misread_as_zoom() {
        let mut camera = configured_camera();
        let before = camera.snapshot();

        assert!(!camera.apply_scroll(&ScrollWheelEvent {
            position: Point {
                x: px(VIEWPORT.0 * 0.5),
                y: px(VIEWPORT.1 * 0.5),
            },
            delta: ScrollDelta::Lines(Point { x: 1.0, y: 0.0 }),
            ..ScrollWheelEvent::default()
        }));

        let after = camera.snapshot();
        assert_eq!(after.zoom.to_bits(), before.zoom.to_bits());
        assert_eq!(after.offset_px.0.to_bits(), before.offset_px.0.to_bits());
        assert_eq!(after.offset_px.1.to_bits(), before.offset_px.1.to_bits());
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

    fn clustered_population_with_one_outlier() -> Vec<Position> {
        let mut positions = Vec::new();
        for x in [120.0, 240.0, 360.0] {
            for y in [120.0, 240.0, 360.0] {
                positions.push(Position::new(x, y));
            }
        }
        positions.push(Position::new(5_900.0, 2_900.0));
        positions
    }

    #[test]
    fn initial_population_frame_is_density_centered_and_viewport_independent() {
        let agent_radius = ScriptBotsConfig::default().bot_radius;
        let positions = clustered_population_with_one_outlier();

        for viewport in [(1280.0, 720.0), (1600.0, 900.0)] {
            let mut camera = Camera::default();
            let layout = camera.layout_with_initial_population(
                (0.0, 0.0),
                viewport,
                WORLD,
                agent_radius,
                positions.iter().copied(),
            );

            let visible_agent_diameters = viewport.0 / (layout.scale * agent_radius * 2.0);
            assert!(
                approx_eq(visible_agent_diameters, 120.0, 1e-3),
                "{viewport:?} must show 120 agent diameters across, got {visible_agent_diameters}"
            );
            assert!(
                layout.scale * agent_radius >= 5.0,
                "{viewport:?} must make agents legible, got radius {} px",
                layout.scale * agent_radius
            );

            let population_center = camera
                .world_to_screen((240.0, 240.0))
                .expect("population median is inside the world");
            assert!(
                approx_eq(population_center.0, viewport.0 * 0.5, 1e-3)
                    && approx_eq(population_center.1, viewport.1 * 0.5, 1e-3),
                "coordinate median must be centered despite the distant outlier: {population_center:?}"
            );
        }
    }

    #[test]
    fn population_center_respects_the_toroidal_seam() {
        let center = population_density_center(
            [
                Position::new(5_980.0, 1_490.0),
                Position::new(10.0, 1_500.0),
                Position::new(30.0, 1_510.0),
            ],
            WORLD,
        )
        .expect("non-empty finite population has a center");

        assert!(
            center.x < 50.0 || center.x > WORLD.0 - 50.0,
            "a seam cluster must not be centered in empty mid-world space: {}",
            center.x
        );
        assert!(approx_eq(center.y, 1_500.0, 1e-3));
    }

    #[test]
    fn successful_population_frame_is_one_shot() {
        let mut camera = Camera::default();
        let agent_radius = ScriptBotsConfig::default().bot_radius;
        camera.layout_with_initial_population(
            (0.0, 0.0),
            VIEWPORT,
            WORLD,
            agent_radius,
            clustered_population_with_one_outlier(),
        );
        let initialized = camera.snapshot();

        camera.layout_with_initial_population(
            (0.0, 0.0),
            VIEWPORT,
            WORLD,
            agent_radius,
            [Position::new(5_500.0, 2_500.0)],
        );
        let second_frame = camera.snapshot();

        assert_eq!(second_frame.zoom.to_bits(), initialized.zoom.to_bits());
        assert_eq!(
            second_frame.offset_px.0.to_bits(),
            initialized.offset_px.0.to_bits()
        );
        assert_eq!(
            second_frame.offset_px.1.to_bits(),
            initialized.offset_px.1.to_bits()
        );
    }

    #[test]
    fn empty_frame_stays_eligible_for_population_framing() {
        let mut camera = Camera::default();
        let agent_radius = ScriptBotsConfig::default().bot_radius;

        camera.layout_with_initial_population(
            (0.0, 0.0),
            VIEWPORT,
            WORLD,
            agent_radius,
            std::iter::empty(),
        );
        camera.layout_with_initial_population(
            (0.0, 0.0),
            VIEWPORT,
            WORLD,
            agent_radius,
            clustered_population_with_one_outlier(),
        );

        let population_center = camera
            .world_to_screen((240.0, 240.0))
            .expect("population median is inside the world");
        assert!(
            approx_eq(population_center.0, VIEWPORT.0 * 0.5, 1e-3)
                && approx_eq(population_center.1, VIEWPORT.1 * 0.5, 1e-3),
            "an empty first render must not consume the one-shot population frame"
        );
    }

    #[test]
    fn camera_input_before_population_cancels_pending_auto_frame() {
        let mut camera = Camera::default();
        let agent_radius = ScriptBotsConfig::default().bot_radius;

        camera.layout_with_initial_population(
            (0.0, 0.0),
            VIEWPORT,
            WORLD,
            agent_radius,
            std::iter::empty(),
        );
        camera.start_pan(Point {
            x: px(400.0),
            y: px(300.0),
        });
        assert!(camera.update_pan(Point {
            x: px(475.0),
            y: px(340.0),
        }));
        camera.end_pan();
        let after_input = camera.snapshot();

        let layout = camera.layout_with_initial_population(
            (0.0, 0.0),
            VIEWPORT,
            WORLD,
            agent_radius,
            clustered_population_with_one_outlier(),
        );
        let after_population = camera.snapshot();

        assert_eq!(
            after_population.offset_px.0.to_bits(),
            after_input.offset_px.0.to_bits()
        );
        assert_eq!(
            after_population.offset_px.1.to_bits(),
            after_input.offset_px.1.to_bits()
        );
        assert!(
            approx_eq(
                layout.scale,
                (VIEWPORT.0 / WORLD.0).min(VIEWPORT.1 / WORLD.1),
                1e-6
            ),
            "user camera input must preserve the fitted scale instead of being overwritten"
        );
    }

    #[test]
    fn fit_world_restores_the_geometric_fit_after_population_frame() {
        let agent_radius = ScriptBotsConfig::default().bot_radius;

        for viewport in [(1280.0, 720.0), (1600.0, 900.0)] {
            let mut camera = Camera::default();
            camera.layout_with_initial_population(
                (0.0, 0.0),
                viewport,
                WORLD,
                agent_radius,
                clustered_population_with_one_outlier(),
            );

            camera.fit_world();
            let layout = camera.layout((0.0, 0.0), viewport, WORLD);
            let snapshot = camera.snapshot();
            let world_center = camera
                .world_to_screen((WORLD.0 * 0.5, WORLD.1 * 0.5))
                .expect("geometric world center is visible");

            assert!(approx_eq(
                layout.scale,
                (viewport.0 / WORLD.0).min(viewport.1 / WORLD.1),
                1e-6
            ));
            assert_eq!(snapshot.offset_px.0.to_bits(), 0.0_f32.to_bits());
            assert_eq!(snapshot.offset_px.1.to_bits(), 0.0_f32.to_bits());
            assert!(approx_eq(world_center.0, viewport.0 * 0.5, 1e-3));
            assert!(approx_eq(world_center.1, viewport.1 * 0.5, 1e-3));
        }
    }
}
