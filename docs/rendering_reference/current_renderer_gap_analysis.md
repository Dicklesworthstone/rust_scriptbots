# Current Renderer Gap Analysis (Rust GPUI vs Legacy GLUT)

This note summarises observed differences between the existing `scriptbots-render` implementation and the legacy behaviour captured in `legacy_renderer_spec.md`. It is intended to guide ownership assignments for Sections 2 and 3 of `PLAN_TO_FIX_RENDERING_ISSUES.md`.

## Camera System

- `CameraState` (`crates/scriptbots-render/src/lib.rs:9938`) is tightly coupled to the HUD view model, stores mutable state per render pass, and lacks pure transformation helpers. The plan calls for a dedicated `Camera` struct with deterministic methods (`fit_world`, `fit_selection`, cursor-relative zoom). Current code provides `screen_to_world` but not `world_to_screen`, nor does it expose state in a testable form.
- Zoom defaults to `1.0` until `ensure_default_zoom` is triggered during render, which infers base scale opportunistically. Legacy behaviour expects an initial zoom equivalent to `scalemult=0.2`. Need explicit calculation from viewport/world dimensions before first frame.
- Cursor-centered zoom is partially implemented via `apply_scroll`, but the library only handles wheel scroll delta, not keyboard shortcuts or smooth transitions. No constraints on pan boundaries; agents can scroll into empty space indefinitely.
- Follow mechanics (center on agent) exist (`center_on`), yet they operate on last render data; the plan wants fit-to-selection and UI buttons. No persistence hooks for camera state beyond run (no serialization).

## Rendering Fidelity

- Terrain colours are derived from `terrain_surface_color` (inlined hex values). Need validation against legacy palette, plus alternate palettes (color-blind/high contrast) demanded in §3.4.
- Agent depiction uses GPUI paint paths with halos, spikes, and overlays, but design review is required to ensure radius, outline contrast, and selection cues match the legacy look and remain legible at default zoom. No drop shadows or AO.
- HUD layout is declarative (Div/Stack). Styling is mostly flat backgrounds; lacks the “rounded cards / gradients” improvements listed in §3.3. Responsive behaviour exists but needs audit.

## Snapshot & Regression Tooling

- Headless PNG rendering entry point `render_png_offscreen` exists, yet there is no automated command wiring it to the app binary or CI. No golden assets stored.
- No viewport/unit tests verifying camera coordinate mapping. QA needs integration with `cargo test` plus future GitHub Action (plan §1.3, §2.4).

## Dependencies & Risks

- Refactoring `CameraState` will touch GPUI layout code and the WGPU bridge (`scriptbots-world-gfx`). Coordination is required to avoid regressions in the offscreen renderer, terminal fallback, and control overlays.
- Large `lib.rs` (~10k lines) complicates incremental changes; consider extracting submodules (camera.rs, hud.rs) as part of cleanup, ensuring new files reflect genuinely new functionality per repo rules.

## Next Steps

1. Assign leads for each plan section (camera, visuals, QA). Update `PLAN_TO_FIX_RENDERING_ISSUES.md` with bracketed tags once confirmed.
2. Capture legacy screenshots to complete spec baseline and validate palette values.
3. Draft API sketch for new camera module (traits, events) and circulate for consensus before code changes.

Last updated: 2025-10-30 by Codex.
