# Camera Refactor Task Breakdown

_Updated: 2025-10-30 – drafted by PurpleBear for Stage 1 deep dive._

## Current Touch Points (pre-refactor)

| Function | Call Sites | Responsibility |
| --- | --- | --- |
| `CameraState::ensure_default_zoom` | `paint_world_with_wgpu` (~line 949), `configured_camera` test helper, `render_png_offscreen` (~line 11067) | Initialize camera zoom based on computed base scale. |
| `CameraState::start_pan` | Mouse down handler (`handle_mouse_down` block around line 2490) | Capture initial cursor position when right button pressed. |
| `CameraState::update_pan` | Mouse move handler (`handle_mouse_move` around line 2505) | Apply pan delta during drag; trigger repaint if offset changed. |
| `CameraState::end_pan` | Mouse up handler (`handle_mouse_up` around line 2572) | Reset state when button released. |
| `CameraState::apply_scroll` | Mouse wheel handler (`handle_mouse_wheel` around line 2607); keyboard zoom shortcuts reuse via synthetic `ScrollWheelEvent` | Perform cursor-centric zoom calculations, update offset to keep focus under cursor. |
| `CameraState::record_render_metrics` | `paint_world_with_wgpu` (~line 950) and `render_png_offscreen` (~line 11070) | Persist viewport/world dimensions + base scale for later coordinate transforms. |
| `CameraState::center_on` | Selection/follow logic (`follow_selected_agent` etc. around line 2879) | Recenters camera on agent-of-interest. |
| `CameraState::screen_to_world` | Picker (`pick_world_point` around line 1435), hover overlays, tests | Map screen coordinates to world coordinates; returns `None` when outside view bounds. |
| `CameraState::world_to_screen` *(missing)* | Various layers perform manual math instead; will become a direct method post-refactor. |

## Data Flow Summary

1. **Viewport Change:** `paint_world_with_wgpu` computes base scale from `CanvasState` bounds → `ensure_default_zoom` (first frame) → `record_render_metrics` for subsequent transforms.
2. **Input:** GPUI events mutate camera via mutex (`start_pan`, `update_pan`, `apply_scroll`, `end_pan`). After each mutation the renderer schedules a repaint (see `self.repaint()` calls inside event handlers).
3. **Rendering:** `paint_world_with_wgpu` locks camera, derives `zoom`, `offset`, and feeds them into terrain/agent paint routines. Offscreen renderer replicates this in `render_png_offscreen` (CPU path).
4. **Picking:** `pick_world_point` grabs a snapshot, calls `screen_to_world`, and uses toroidal math to locate nearest agent.

## Extraction Checklist (Stage 1)

- [x] Create `crates/scriptbots-render/src/camera/mod.rs` exporting `CameraConfig`, `Camera`, `CameraSnapshot` (Pan state handled internally).
- [x] Move state fields (`offset_px`, `zoom`, `last_*`) into the new struct, replacing direct field access with getters/setters.
- [x] Implement `Camera::record_render_metrics` successor (now `Camera::record_render_metrics`) capturing canvas/world metrics.
- [x] Add `Camera::world_to_screen` util; expose via tests (runtime wiring TBD in Stage 2).
- [x] Cover cursor-centric zoom, pan, and fit-to-world invariants with unit tests (`camera/tests.rs`).
- [x] Provide `Camera::snapshot()` returning a lightweight copy for rendering threads.
- [x] Maintain backwards compatibility with mutex usage during Stage 2 (replaced `Arc<Mutex<CameraState>>` with `Arc<Mutex<Camera>>`).

## Open Questions for Stage 1 Owner

1. How should pan limits work? Legacy C++ allowed infinite wrap; plan currently prefers clamped view. Should we expose `PanBoundary` enum? (Default proposal: clamp to world rect with optional padding.)
2. Do we persist `last_canvas_*` inside camera or inside snapshot? (Recommendation: keep in camera so `screen_to_world` remains a pure method post-update.)
3. Should `Camera` expose change notifications (e.g., returning a `CameraUpdate` struct) or should GPUI poll after mutations? (Leaning toward update struct to minimize redundant repaints.)

### Stage 2 TODOs (ongoing)

- [x] Update `render_png_offscreen` to leverage `Camera::layout`/`world_to_screen` so offscreen snapshots match interactive view.
- [ ] Replace remaining manual pad/offset math in HUD overlays with helpers from `Camera` (e.g., `ViewLayout`).

Please add notes inline or update the checklist once ownership is confirmed.
