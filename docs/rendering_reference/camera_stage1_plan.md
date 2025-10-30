# Camera Refactor Stage 1 – Work Breakdown

Owner: PinkMountain (Codex)
Updated: 2025-10-30

## Objectives

1. Introduce a reusable `Camera` module that encapsulates transform math without modifying GPUI rendering behaviour yet.
2. Preserve existing interactions (pan, zoom, follow) while routing them through the new API.
3. Add deterministic unit tests covering coordinate mapping to share logic with the viewport invariants harness.

## Deliverables

- `crates/scriptbots-render/src/camera.rs`
  - `CameraConfig`, `CameraState`, `CameraSnapshot`, `CameraUpdate` structs.
  - Methods: `new`, `set_world`, `set_viewport`, `fit_world`, `set_zoom`, `zoom_to_cursor`, `pan_pixels`, `screen_to_world`, `world_to_screen`, `snapshot`.
  - Legacy constants (`legacy_scale = 0.2`, min/max clamp) configurable via `CameraConfig`.
- Refactored usage in `lib.rs`
  - Replace inline `CameraState` with `camera::Camera`.
  - Update mouse/scroll handlers to call `zoom_to_cursor`/`pan_pixels`.
  - Ensure follow logic uses `Camera::recenter(world_point)`.
- Tests in `crates/scriptbots-render/src/camera.rs` (or dedicated `tests/camera.rs`)
  - `default_zoom_matches_legacy`
  - `cursor_zoom_preserves_world_point`
  - `screen_to_world_roundtrip`
  - `fit_world_centers_scene`
- Integration glue for viewport invariants
  - Expose `CameraSnapshot` (contains scale, offset, viewport rect) for tests.
  - Add helper to compute minimum agent radius in pixels given world radius.

## Non-Goals for Stage 1

- No new UI controls/icons yet (handled in Stage 2+).
- No smoothing/easing; behaviour should remain step-based to avoid regression risk.
- No persistence; focus on in-memory state.

## Dependencies

- Align with PurpleBear’s snapshot harness to reuse deterministic seed + viewport sizing.
- Confirm with visuals lead (once assigned) about default zoom expectations before merging.

## Next Steps

1. Draft API skeleton & tests (WIP branch).
2. Verify unit tests pass (`cargo test -p scriptbots-render`) and snapshot harness still green.
3. Share diff for review with PurpleBear/RedCastle before Stage 2 (UX enhancements).
