//! Pointer gestures, hit regions, cursor-centered transforms, and shared selection (bd-2z0.14.2.5).
//!
//! Provides the complete interaction contract:
//! - Cached layout hit regions (`HitRegionMap`, `HitRegion`) for O(1) hit testing.
//! - Explicit pointer gesture state machine (`PointerGestureState`): Down, Dragging, Cancelled, Idle.
//! - Pan by drag and cursor-centered zoom invariance via fractional canvas coordinates.
//! - Deterministic stacked-agent cycling and empty-click selection clearing.
//! - Stabilized hover probe (150 ms threshold) exposing: uid, diet, energy, health, age, brain_key.
//! - Clamped splitter resizing between map canvas and sidebar columns.
//! - Status toggle targets in header for pause, speed, theme, and palette.

use crossterm::event::MouseButton;
use ratatui::layout::Rect;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Minimum and maximum allowable percentage split for the map canvas column.
pub const MIN_MAP_SPLIT_PCT: u16 = 25;
pub const MAX_MAP_SPLIT_PCT: u16 = 85;
pub const DEFAULT_MAP_SPLIT_PCT: u16 = 60;

/// Required hover stabilization threshold before activating hover tooltips (150 ms).
pub const HOVER_STABILIZATION_DURATION: Duration = Duration::from_millis(150);

/// Drag slop distance in cells before a Down gesture transitions to Dragging.
pub const DRAG_THRESHOLD_CELLS: u16 = 1;

// ---------------------------------------------------------------------------
// Hit Regions
// ---------------------------------------------------------------------------

/// Sub-targets in the top status header for direct click toggles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeaderHitTarget {
    PauseToggle,
    SpeedCycle,
    PaletteCycle,
    ThemeToggle,
    HelpToggle,
}

/// Identifies each distinct sidebar panel in the TUI layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SidebarPanelKind {
    Stats,
    Trends,
    Leaderboard,
    Oldest,
    Insights,
    Brains,
    Mortality,
    Events,
}

/// Category of splitter in the interactive layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitterKind {
    MainVertical,
}

/// Semantic hit region under a terminal cell coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HitRegion {
    /// World canvas map area with fractional coordinates (fx, fy) in [0.0, 1.0].
    Map { fx: f32, fy: f32 },
    /// Status header or one of its clickable controls.
    Header(Option<HeaderHitTarget>),
    /// Narrative timeline rail, with optional event index clicked.
    Rail(Option<usize>),
    /// Resizable splitter between map column and sidebar.
    Splitter(SplitterKind),
    /// One of the sidebar panels.
    SidebarPanel(SidebarPanelKind),
    /// Sensor probe panel below map.
    Probe,
    /// Command palette modal overlay.
    Palette,
    /// Help overlay.
    Help,
    /// Outside interactive application bounds.
    Outside,
}

/// Cached hit-test map holding bounding boxes for the active frame.
#[derive(Debug, Clone, Default)]
pub struct HitRegionMap {
    pub map_rect: Option<Rect>,
    pub header_rect: Option<Rect>,
    pub header_targets: Vec<(HeaderHitTarget, Rect)>,
    pub rail_rect: Option<Rect>,
    pub splitter_rect: Option<Rect>,
    pub probe_rect: Option<Rect>,
    pub sidebar_panels: Vec<(SidebarPanelKind, Rect)>,
    pub palette_rect: Option<Rect>,
    pub help_rect: Option<Rect>,
}

impl HitRegionMap {
    /// Test which semantic region contains the given column and row.
    #[must_use]
    pub fn hit_test(&self, col: u16, row: u16) -> HitRegion {
        // 1. Modals take priority
        if let Some(help) = self.help_rect
            && rect_contains(help, col, row)
        {
            return HitRegion::Help;
        }
        if let Some(palette) = self.palette_rect
            && rect_contains(palette, col, row)
        {
            return HitRegion::Palette;
        }

        // 2. Header and its sub-targets
        if let Some(header) = self.header_rect
            && rect_contains(header, col, row)
        {
            for (target, rect) in &self.header_targets {
                if rect_contains(*rect, col, row) {
                    return HitRegion::Header(Some(*target));
                }
            }
            return HitRegion::Header(None);
        }

        // 3. Splitter between map and sidebar
        if let Some(splitter) = self.splitter_rect
            && rect_contains(splitter, col, row)
        {
            return HitRegion::Splitter(SplitterKind::MainVertical);
        }

        // 4. World map canvas
        if let Some(map) = self.map_rect
            && rect_contains(map, col, row)
            && map.width > 0
            && map.height > 0
        {
            let fx = ((f32::from(col - map.x) + 0.5) / f32::from(map.width)).clamp(0.0, 1.0);
            let fy = ((f32::from(row - map.y) + 0.5) / f32::from(map.height)).clamp(0.0, 1.0);
            return HitRegion::Map { fx, fy };
        }

        // 5. Probe
        if let Some(probe) = self.probe_rect
            && rect_contains(probe, col, row)
        {
            return HitRegion::Probe;
        }

        // 6. Rail
        if let Some(rail) = self.rail_rect
            && rect_contains(rail, col, row)
        {
            return HitRegion::Rail(None);
        }

        // 7. Sidebar panels
        for (panel, rect) in &self.sidebar_panels {
            if rect_contains(*rect, col, row) {
                return HitRegion::SidebarPanel(*panel);
            }
        }

        HitRegion::Outside
    }
}

/// Test whether a Ratatui Rect contains a point.
#[must_use]
pub const fn rect_contains(rect: Rect, col: u16, row: u16) -> bool {
    col >= rect.x
        && col < rect.x.saturating_add(rect.width)
        && row >= rect.y
        && row < rect.y.saturating_add(rect.height)
}

// ---------------------------------------------------------------------------
// Pointer Gesture State Machine
// ---------------------------------------------------------------------------

/// The target of an active mouse drag operation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DragTarget {
    /// Panning the world canvas.
    PanMap {
        initial_pan: (f32, f32),
        initial_zoom: f32,
    },
    /// Resizing the splitter between map and sidebar.
    ResizeSplitter { initial_split_pct: u16 },
    /// Drag-scrolling a sidebar panel.
    ScrollPanel {
        panel: SidebarPanelKind,
        initial_scroll: usize,
    },
}

/// Explicit pointer gesture lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PointerGestureState {
    /// No pointer interaction in progress.
    #[default]
    Idle,
    /// Button pressed down at a specific position, waiting for drag threshold or release.
    Down {
        start_col: u16,
        start_row: u16,
        button: MouseButton,
        region: HitRegion,
        start_time: Instant,
    },
    /// Drag threshold exceeded; actively dragging a target.
    Dragging {
        start_col: u16,
        start_row: u16,
        current_col: u16,
        current_row: u16,
        button: MouseButton,
        target: DragTarget,
    },
    /// Gesture cancelled explicitly (e.g. via Escape key or loss of window focus).
    Cancelled { reason: &'static str },
}

impl PointerGestureState {
    /// Whether a drag gesture is currently active.
    #[must_use]
    pub const fn is_dragging(&self) -> bool {
        matches!(self, Self::Dragging { .. })
    }

    /// Whether a mouse button is currently down (either Down or Dragging).
    #[must_use]
    pub const fn is_down(&self) -> bool {
        matches!(self, Self::Down { .. } | Self::Dragging { .. })
    }

    /// Begin a pointer gesture on mouse down.
    pub fn on_down(
        &mut self,
        col: u16,
        row: u16,
        button: MouseButton,
        region: HitRegion,
        now: Instant,
    ) {
        *self = Self::Down {
            start_col: col,
            start_row: row,
            button,
            region,
            start_time: now,
        };
    }

    /// Update pointer position during movement.
    ///
    /// Transitions from `Down` to `Dragging` if the movement exceeds [`DRAG_THRESHOLD_CELLS`].
    pub fn on_move(
        &mut self,
        col: u16,
        row: u16,
        default_target_resolver: impl FnOnce(HitRegion) -> Option<DragTarget>,
    ) {
        match *self {
            Self::Down {
                start_col,
                start_row,
                button,
                region,
                ..
            } => {
                let dx = (col as i32 - start_col as i32).abs();
                let dy = (row as i32 - start_row as i32).abs();
                if (dx >= DRAG_THRESHOLD_CELLS as i32 || dy >= DRAG_THRESHOLD_CELLS as i32)
                    && let Some(target) = default_target_resolver(region)
                {
                    *self = Self::Dragging {
                        start_col,
                        start_row,
                        current_col: col,
                        current_row: row,
                        button,
                        target,
                    };
                }
            }
            Self::Dragging {
                start_col,
                start_row,
                button,
                target,
                ..
            } => {
                *self = Self::Dragging {
                    start_col,
                    start_row,
                    current_col: col,
                    current_row: row,
                    button,
                    target,
                };
            }
            _ => {}
        }
    }

    /// Finalize gesture on mouse up.
    ///
    /// Returns `Some(ClickAction)` if this was a click rather than a drag.
    pub fn on_up(&mut self) -> Option<PointerClickEvent> {
        let outcome = match *self {
            Self::Down {
                start_col,
                start_row,
                button,
                region,
                ..
            } => Some(PointerClickEvent {
                col: start_col,
                row: start_row,
                button,
                region,
            }),
            _ => None,
        };
        *self = Self::Idle;
        outcome
    }

    /// Explicitly cancel the current gesture, returning the cancelled drag target if any.
    pub fn cancel(&mut self, reason: &'static str) -> Option<DragTarget> {
        let prev_target = match *self {
            Self::Dragging { target, .. } => Some(target),
            _ => None,
        };
        *self = Self::Cancelled { reason };
        prev_target
    }
}

/// A synthesized click event produced when a mouse press releases without dragging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointerClickEvent {
    pub col: u16,
    pub row: u16,
    pub button: MouseButton,
    pub region: HitRegion,
}

// ---------------------------------------------------------------------------
// Hover Probe Stabilizer
// ---------------------------------------------------------------------------

/// Hover probe tooltip contents with all 6 promised fields (bd-2z0.14.2.5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RichHoverTooltip {
    pub cell_x: u16,
    pub cell_y: u16,
    pub agent_uid: Option<u64>,
    pub diet: f32,
    pub energy: f32,
    pub health: f32,
    pub age: u32,
    pub brain_key: Option<u64>,
}

/// Stabilizes hover probe tooltips: cursor must remain stationary for 150 ms before
/// presenting rich agent telemetry.
#[derive(Debug, Clone)]
pub struct HoverStabilizer {
    candidate: Option<(u16, u16, Instant)>,
    active_tooltip: Option<RichHoverTooltip>,
}

impl Default for HoverStabilizer {
    fn default() -> Self {
        Self::new()
    }
}

impl HoverStabilizer {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            candidate: None,
            active_tooltip: None,
        }
    }

    /// Current active tooltip, if stabilized.
    #[must_use]
    pub const fn active(&self) -> Option<&RichHoverTooltip> {
        self.active_tooltip.as_ref()
    }

    /// Clear both active tooltip and candidate.
    pub fn clear(&mut self) {
        self.candidate = None;
        self.active_tooltip = None;
    }

    /// Record a mouse move event.
    ///
    /// If the position changed, resets the 150 ms stabilization timer and clears
    /// any previously active tooltip.
    pub fn on_mouse_move(&mut self, col: u16, row: u16, now: Instant) {
        if let Some((prev_col, prev_row, _)) = self.candidate
            && prev_col == col
            && prev_row == row
        {
            // Stationary: keep current candidate timer running
            return;
        }
        self.candidate = Some((col, row, now));
        self.active_tooltip = None;
    }

    /// Poll whether the hover candidate has stabilized (>= 150 ms elapsed).
    ///
    /// If stabilized, calls `resolver` to generate the rich tooltip.
    pub fn poll<F>(&mut self, now: Instant, resolver: F) -> Option<&RichHoverTooltip>
    where
        F: FnOnce(u16, u16) -> Option<RichHoverTooltip>,
    {
        if self.active_tooltip.is_some() {
            return self.active_tooltip.as_ref();
        }

        if let Some((col, row, start_time)) = self.candidate
            && now.duration_since(start_time) >= HOVER_STABILIZATION_DURATION
        {
            self.active_tooltip = resolver(col, row);
            return self.active_tooltip.as_ref();
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Coordinate Transforms & Math
// ---------------------------------------------------------------------------

/// Computes cursor-centered zoom parameters.
///
/// Preserves the exact world coordinate under the cursor `(fx, fy)` across zoom changes:
/// `W_x = C_x - S/2 + fx * S`
/// `C_new_x = W_x + S_new * (0.5 - fx)`
///
/// Returns `(new_zoom, (new_center_x, new_center_y))`.
#[must_use]
pub fn cursor_centered_zoom(
    current_zoom: f32,
    current_center: (f32, f32),
    zoom_factor: f32,
    fx: f32,
    fy: f32,
    min_span: f32,
    max_zoom: f32,
) -> (f32, (f32, f32)) {
    let current_zoom = if current_zoom.is_finite() && current_zoom >= 1.0 {
        current_zoom
    } else {
        1.0
    };
    let current_span = (1.0 / current_zoom).clamp(min_span, 1.0);

    // 1. World point currently under cursor
    let half = current_span / 2.0;
    let wx = current_center.0 - half + fx * current_span;
    let wy = current_center.1 - half + fy * current_span;

    // 2. New zoom and span
    let new_zoom = (current_zoom * zoom_factor).clamp(1.0, max_zoom);
    let new_span = (1.0 / new_zoom).clamp(min_span, 1.0);

    // 3. New center that keeps (wx, wy) under (fx, fy)
    let new_half = new_span / 2.0;
    let new_cx = wx + new_span * (0.5 - fx);
    let new_cy = wy + new_span * (0.5 - fy);

    // Clamp center so the window does not escape the [0.0, 1.0] world
    let clamped_cx = new_cx.clamp(new_half, 1.0 - new_half);
    let clamped_cy = new_cy.clamp(new_half, 1.0 - new_half);

    (new_zoom, (clamped_cx, clamped_cy))
}

/// Compute new pan offset from mouse drag delta.
///
/// Dragging the mouse right (+dx) moves the view window left (-world_dx),
/// keeping the terrain under the cursor aligned.
#[must_use]
pub fn drag_pan(
    current_pan: (f32, f32),
    delta_col: i32,
    delta_row: i32,
    area_w: u16,
    area_h: u16,
    span: f32,
) -> (f32, f32) {
    if area_w == 0 || area_h == 0 {
        return current_pan;
    }

    let world_dx = (delta_col as f32 / area_w as f32) * span;
    let world_dy = (delta_row as f32 / area_h as f32) * span;

    let half = span / 2.0;
    let new_cx = (current_pan.0 - world_dx).clamp(half, 1.0 - half);
    let new_cy = (current_pan.1 - world_dy).clamp(half, 1.0 - half);

    (new_cx, new_cy)
}

/// Clamp a splitter percentage to within safe layout bounds [25, 85].
#[must_use]
pub const fn clamp_splitter_pct(pct: u16) -> u16 {
    if pct < MIN_MAP_SPLIT_PCT {
        MIN_MAP_SPLIT_PCT
    } else if pct > MAX_MAP_SPLIT_PCT {
        MAX_MAP_SPLIT_PCT
    } else {
        pct
    }
}

/// Compute new splitter percentage given mouse column and total body width.
#[must_use]
pub fn calculate_splitter_pct(mouse_col: u16, body_x: u16, body_width: u16) -> u16 {
    if body_width == 0 || mouse_col < body_x {
        return MIN_MAP_SPLIT_PCT;
    }
    let relative_col = mouse_col.saturating_sub(body_x);
    let raw_pct = ((f32::from(relative_col) / f32::from(body_width)) * 100.0).round() as u16;
    clamp_splitter_pct(raw_pct)
}

/// Deterministically cycles through candidate stacked agents.
///
/// If `current_uid` is in the candidates list, returns the next candidate (wrapping around).
/// Otherwise returns the first candidate.
#[must_use]
pub fn cycle_stacked_agents(candidates: &[u64], current_uid: Option<u64>) -> Option<u64> {
    if candidates.is_empty() {
        return None;
    }
    if let Some(uid) = current_uid
        && let Some(pos) = candidates.iter().position(|&id| id == uid)
    {
        return Some(candidates[(pos + 1) % candidates.len()]);
    }
    Some(candidates[0])
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_centered_zoom_invariance() {
        let center = (0.5, 0.5);
        let zoom = 2.0;
        let fx = 0.75;
        let fy = 0.25;

        // Current world point under cursor
        let span = 1.0 / zoom;
        let half = span / 2.0;
        let wx_before = center.0 - half + fx * span;
        let wy_before = center.1 - half + fy * span;

        // Zoom in by 1.5x around (fx, fy)
        let (new_zoom, new_center) = cursor_centered_zoom(zoom, center, 1.5, fx, fy, 0.05, 32.0);

        // Compute world point under same cursor position with new zoom
        let new_span = 1.0 / new_zoom;
        let new_half = new_span / 2.0;
        let wx_after = new_center.0 - new_half + fx * new_span;
        let wy_after = new_center.1 - new_half + fy * new_span;

        assert!(
            (wx_before - wx_after).abs() < 1e-4,
            "X invariance violated: before {wx_before} vs after {wx_after}"
        );
        assert!(
            (wy_before - wy_after).abs() < 1e-4,
            "Y invariance violated: before {wy_before} vs after {wy_after}"
        );
    }

    #[test]
    fn test_drag_pan_translation() {
        let initial_pan = (0.5, 0.5);
        let span = 0.5; // 2x zoom
        let area_w = 100;
        let area_h = 50;

        // Drag right 20 cells, down 10 cells
        let new_pan = drag_pan(initial_pan, 20, 10, area_w, area_h, span);
        // world_dx = (20 / 100) * 0.5 = 0.1
        // world_dy = (10 / 50) * 0.5 = 0.1
        // new_cx = 0.5 - 0.1 = 0.4
        // new_cy = 0.5 - 0.1 = 0.4
        assert!((new_pan.0 - 0.4).abs() < 1e-5);
        assert!((new_pan.1 - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_splitter_clamping() {
        assert_eq!(clamp_splitter_pct(10), MIN_MAP_SPLIT_PCT);
        assert_eq!(clamp_splitter_pct(95), MAX_MAP_SPLIT_PCT);
        assert_eq!(clamp_splitter_pct(60), 60);

        let pct = calculate_splitter_pct(50, 0, 100);
        assert_eq!(pct, 50);

        let pct_low = calculate_splitter_pct(5, 0, 100);
        assert_eq!(pct_low, MIN_MAP_SPLIT_PCT);

        let pct_high = calculate_splitter_pct(95, 0, 100);
        assert_eq!(pct_high, MAX_MAP_SPLIT_PCT);
    }

    #[test]
    fn test_stacked_agent_cycling_and_empty_click() {
        let candidates = vec![101, 202, 303];

        // First click selects first candidate
        assert_eq!(cycle_stacked_agents(&candidates, None), Some(101));

        // Subsequent clicks cycle through candidates deterministically
        assert_eq!(cycle_stacked_agents(&candidates, Some(101)), Some(202));
        assert_eq!(cycle_stacked_agents(&candidates, Some(202)), Some(303));
        assert_eq!(cycle_stacked_agents(&candidates, Some(303)), Some(101));

        // Empty candidates list returns None (empty click clearing)
        assert_eq!(cycle_stacked_agents(&[], Some(101)), None);
    }

    #[test]
    fn test_hover_stabilization_timing() {
        let mut stabilizer = HoverStabilizer::new();
        let t0 = Instant::now();

        stabilizer.on_mouse_move(10, 5, t0);
        assert!(stabilizer.active().is_none());

        // At 100 ms (< 150 ms threshold), poll should NOT activate
        let t_early = t0 + Duration::from_millis(100);
        let resolved = stabilizer.poll(t_early, |col, row| {
            Some(RichHoverTooltip {
                cell_x: col,
                cell_y: row,
                agent_uid: Some(42),
                diet: 0.8,
                energy: 50.0,
                health: 100.0,
                age: 200,
                brain_key: Some(7),
            })
        });
        assert!(resolved.is_none());
        assert!(stabilizer.active().is_none());

        // Moving mouse before 150 ms resets timer
        let t_moved = t0 + Duration::from_millis(120);
        stabilizer.on_mouse_move(11, 5, t_moved);

        // At 120 + 100 = 220 ms (only 100 ms from last move), still not activated
        let t_check = t_moved + Duration::from_millis(100);
        assert!(
            stabilizer
                .poll(t_check, |_, _| panic!("should not be called"))
                .is_none()
        );

        // At 120 + 160 = 280 ms (> 150 ms from last move), resolves and activates
        let t_ready = t_moved + Duration::from_millis(160);
        let resolved = stabilizer.poll(t_ready, |col, row| {
            Some(RichHoverTooltip {
                cell_x: col,
                cell_y: row,
                agent_uid: Some(42),
                diet: 0.8,
                energy: 50.0,
                health: 100.0,
                age: 200,
                brain_key: Some(7),
            })
        });

        assert!(resolved.is_some());
        let tip = resolved.unwrap();
        assert_eq!(tip.agent_uid, Some(42));
        assert_eq!(tip.diet, 0.8);
        assert_eq!(tip.energy, 50.0);
        assert_eq!(tip.health, 100.0);
        assert_eq!(tip.age, 200);
        assert_eq!(tip.brain_key, Some(7));
    }

    #[test]
    fn test_gesture_state_machine_drag_and_cancel() {
        let mut gesture = PointerGestureState::default();
        let t0 = Instant::now();

        // 1. Mouse down at (50, 10) on Map
        gesture.on_down(
            50,
            10,
            MouseButton::Left,
            HitRegion::Map { fx: 0.5, fy: 0.5 },
            t0,
        );
        assert!(gesture.is_down());
        assert!(!gesture.is_dragging());

        // 2. Small motion under threshold stays Down
        gesture.on_move(50, 10, |_| None);
        assert!(!gesture.is_dragging());

        // 3. Motion beyond threshold transitions to Dragging
        gesture.on_move(55, 12, |region| match region {
            HitRegion::Map { .. } => Some(DragTarget::PanMap {
                initial_pan: (0.5, 0.5),
                initial_zoom: 1.0,
            }),
            _ => None,
        });
        assert!(gesture.is_dragging());

        // 4. Escape / cancel restores previous target
        let cancelled_target = gesture.cancel("user escape");
        assert_eq!(
            cancelled_target,
            Some(DragTarget::PanMap {
                initial_pan: (0.5, 0.5),
                initial_zoom: 1.0,
            })
        );
        assert!(!gesture.is_dragging());
        assert_eq!(
            gesture,
            PointerGestureState::Cancelled {
                reason: "user escape"
            }
        );
    }

    #[test]
    fn test_hit_region_map_comprehensive() {
        let mut map = HitRegionMap {
            header_rect: Some(Rect::new(0, 0, 100, 3)),
            header_targets: vec![
                (HeaderHitTarget::PauseToggle, Rect::new(10, 1, 9, 1)),
                (HeaderHitTarget::SpeedCycle, Rect::new(20, 1, 6, 1)),
                (HeaderHitTarget::PaletteCycle, Rect::new(30, 1, 12, 1)),
                (HeaderHitTarget::ThemeToggle, Rect::new(45, 1, 10, 1)),
                (HeaderHitTarget::HelpToggle, Rect::new(60, 1, 8, 1)),
            ],
            map_rect: Some(Rect::new(0, 3, 60, 20)),
            splitter_rect: Some(Rect::new(60, 3, 1, 20)),
            sidebar_panels: vec![
                (SidebarPanelKind::Stats, Rect::new(61, 3, 39, 10)),
                (SidebarPanelKind::Trends, Rect::new(61, 13, 39, 10)),
            ],
            ..Default::default()
        };

        // 1. Header hit tests
        assert_eq!(
            map.hit_test(12, 1),
            HitRegion::Header(Some(HeaderHitTarget::PauseToggle))
        );
        assert_eq!(
            map.hit_test(22, 1),
            HitRegion::Header(Some(HeaderHitTarget::SpeedCycle))
        );
        assert_eq!(
            map.hit_test(32, 1),
            HitRegion::Header(Some(HeaderHitTarget::PaletteCycle))
        );
        assert_eq!(
            map.hit_test(48, 1),
            HitRegion::Header(Some(HeaderHitTarget::ThemeToggle))
        );
        assert_eq!(
            map.hit_test(62, 1),
            HitRegion::Header(Some(HeaderHitTarget::HelpToggle))
        );
        assert_eq!(map.hit_test(5, 1), HitRegion::Header(None));

        // 2. Splitter hit test
        assert_eq!(
            map.hit_test(60, 10),
            HitRegion::Splitter(SplitterKind::MainVertical)
        );

        // 3. Map hit test with fractional coords
        match map.hit_test(30, 13) {
            HitRegion::Map { fx, fy } => {
                // (30 - 0 + 0.5) / 60 = 30.5 / 60 = ~0.5083
                // (13 - 3 + 0.5) / 20 = 10.5 / 20 = 0.525
                assert!((fx - 0.5083).abs() < 0.01);
                assert!((fy - 0.525).abs() < 0.01);
            }
            other => panic!("expected HitRegion::Map, got {other:?}"),
        }

        // 4. Sidebar panels hit test
        assert_eq!(
            map.hit_test(70, 5),
            HitRegion::SidebarPanel(SidebarPanelKind::Stats)
        );
        assert_eq!(
            map.hit_test(70, 15),
            HitRegion::SidebarPanel(SidebarPanelKind::Trends)
        );

        // 5. Outside
        assert_eq!(map.hit_test(200, 200), HitRegion::Outside);

        // 6. Modal priority
        map.palette_rect = Some(Rect::new(20, 5, 40, 15));
        assert_eq!(map.hit_test(30, 10), HitRegion::Palette);

        map.help_rect = Some(Rect::new(0, 0, 100, 30));
        assert_eq!(map.hit_test(30, 10), HitRegion::Help);
    }

    #[test]
    fn test_cursor_centered_zoom_property_invariance() {
        let center = (0.5, 0.5);
        let zooms = [2.0, 4.0, 8.0];
        let multipliers = [1.2, 1.5, 2.0];
        let f_points = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.3, 0.7)];

        for &zoom in &zooms {
            for &mult in &multipliers {
                for &(fx, fy) in &f_points {
                    let span = 1.0 / zoom;
                    let half = span / 2.0;
                    let wx_before = center.0 - half + fx * span;
                    let wy_before = center.1 - half + fy * span;

                    let (new_zoom, new_center) =
                        cursor_centered_zoom(zoom, center, mult, fx, fy, 0.001, 64.0);

                    let new_span = 1.0 / new_zoom;
                    let new_half = new_span / 2.0;
                    let wx_after = new_center.0 - new_half + fx * new_span;
                    let wy_after = new_center.1 - new_half + fy * new_span;

                    assert!(
                        (wx_before - wx_after).abs() < 1e-4,
                        "zoom={zoom}, mult={mult}, fx={fx}: wx_before={wx_before}, wx_after={wx_after}"
                    );
                    assert!(
                        (wy_before - wy_after).abs() < 1e-4,
                        "zoom={zoom}, mult={mult}, fy={fy}: wy_before={wy_before}, wy_after={wy_after}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_stacked_cycling_full_permutation() {
        let uids = vec![10, 20, 30, 40, 50];
        let mut current = None;
        let mut visited = Vec::new();

        for _ in 0..uids.len() {
            current = cycle_stacked_agents(&uids, current);
            visited.push(current.unwrap());
        }

        assert_eq!(visited, uids);

        // Next cycle wraps around to the first element
        let wrapped = cycle_stacked_agents(&uids, current);
        assert_eq!(wrapped, Some(uids[0]));
    }
}
