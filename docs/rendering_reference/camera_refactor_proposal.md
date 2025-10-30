# Camera Refactor Proposal (Draft for Consensus)

Updated: 2025-10-30 — Author: Codex

## Goals

1. Replace the ad-hoc `CameraState` embedded in `scriptbots-render/src/lib.rs` with a reusable, testable camera module that matches the legacy GLUT semantics while enabling modern UX improvements.
2. Provide deterministic mapping between world and screen coordinates, honoring plan requirements (§2.1–§2.4) before touching rendering code.
3. Minimize churn in the GPUI surface by isolating camera math from widget layout concerns.

## Proposed Module Layout

Create `crates/scriptbots-render/src/camera.rs` (new functionality) exporting:

```rust
pub struct CameraConfig {
    pub world_size: (f32, f32),
    pub viewport: (f32, f32),
    pub min_zoom: f32,
    pub max_zoom: f32,
    pub legacy_scale: f32,
    pub pan_limits: Option<Rect>,
}

pub struct CameraState {
    zoom: f32,
    offset: Vec2,
    base_scale: f32,
}

pub struct Camera {
    config: CameraConfig,
    state: CameraState,
}
```

### Key Methods

- `Camera::new(config)`: computes initial `base_scale` from world/viewport. Sets `zoom` so that `base_scale * zoom == legacy_scale` (0.2 equivalent).
- `set_viewport(size)`: recomputes `base_scale`; clamps offset to pan limits if provided.
- `fit_world()`: centers world, resets zoom to `legacy_scale`.
- `fit_bounds(bounds)`: zooms to bounding box (agents selection) with configurable padding.
- `zoom_to_cursor(cursor_px, zoom_delta)`: cursor-relative zoom, preserving world point under cursor (mirrors legacy drag but extended to wheel/keyboard). Returns `CameraUpdate` diff for efficient UI updates.
- `pan(delta_px)`: adds to offset with optional easing; clamps to pan limits.
- `world_to_screen(point)` / `screen_to_world(point)`.
- `serialize()` / `restore()` for future persistence (optional, returns simple struct for JSON/RON).

Expose `CameraSnapshot` struct with derived data: effective_scale, view_rect, etc., so GPUI layers and `scriptbots-world-gfx` can consume without locking mutexes.

## Integration Plan

1. **Stage 1 – Scaffolding:** introduce `camera.rs`, port existing logic from `CameraState` with unit tests covering:
   - Legacy default zoom equivalence.
   - Cursor-relative zoom invariants.
   - Coordinate transforms at default and zoomed scales.
2. **Stage 2 – Renderer Wiring:** adjust `Renderer` (GPUI) to hold `Arc<Mutex<Camera>>` instead of raw state. Update mouse/keyboard handlers to call new methods. Ensure terminal/offscreen paths invoke the same camera interface.
3. **Stage 3 – UX Enhancements:** map new shortcuts (Ctrl+0 reset, Ctrl+=/- zoom, WASD pan) and expose HUD overlay showing zoom factor & cursor coordinates.
4. **Stage 4 – Testing:** add snapshot tests verifying camera invariants (Rust tests invoking `Camera` directly). Hook into new regression harness once built.

## Risks / Open Questions

- Need agreement on pan limits: infinite vs. clamped to world bounds with padding. Proposal: default clamp to keep world fully visible, configurable for debugging.
- Smooth transitions: initial implementation returns immediate changes; easing can be layered later via GPUI animations.
- Thread safety: camera will be mutated on the GPUI main thread; off-thread access should go through snapshots to avoid locking in render loop.

## Next Steps

1. Review this draft with other agents; update `PLAN_TO_FIX_RENDERING_ISSUES.md` once a lead is assigned.
2. After consensus, implement Stage 1 without touching UI yet; rely on existing tests + new unit tests.

Feedback welcome before coding begins.
