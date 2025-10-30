# Visual Polish Plan (PLAN §3.1–§3.2)

Status: implementation pass 1 landed – RedCastle (2025-10-30)

This document captures the concrete changes required to make the Rust renderer look "moderately acceptable" per the user directive. The focus is palette accuracy, baseline legibility, and simple lighting cues; interaction/UX affordances remain under PLAN §3.3.

## Goals

1. ✅ Replace the current flat terrain colours with the six-tone legacy-inspired palette documented in `legacy_renderer_spec.md` while keeping intensity adjustable for future themes.
2. ✅ Ensure agents remain visible at overview zoom by increasing base size, adding outlines/shadows, and clarifying selection/halo states.
3. ✅ Prevent the "tiny speck" issue in the default scene so first-run dumps look populated without manual zoom.

## Terrain Palette Strategy

- Use the hex values from the spec (`#1E3F66`, `#2F73B3`, `#B14E07`, `#50A913`, `#79D46D`, `#A9B1BA`) as the defaults in `TerrainPalette`. ✅ Implemented via `LEGACY_TERRAIN_BASE/ACCENT`.
- Introduce gamma/brightness controls (simple scalar) so themes can tweak without recalculating the LUT. ↻ Still future work.
- Update CPU path (`paint_full`) and wgpu compositor atlas to share the palette data via a single source-of-truth struct. ✅ Shared helper + GPU snapshot uses same colours.
- Add unit test to assert palette entries match the spec to catch regressions. ↻ Outstanding (tag for later).

## Agent Legibility Enhancements

- Raise default body radius from `AgentColumns::radius` (currently ~4 px) to at least 8 px at default zoom while preserving spike geometry. ✅
- Draw a consistent dark outline using `rgb(0x141414)` regardless of boost state; overlay boost colour as outer glow. ✅ Outline stroke updated; glow remains.
- Add soft drop shadow (offset blur) or fallback to a two-tone circle for CPU path. ✅ Soft shadow quad added (skips very-low-FPS path).
- Selection ring: thicken to 4 px and ensure it survives zoom smoothing. ✅ Highlight palette tweaked.
- Donation ring: switch to semi-transparent green/red arcs with eased alpha. ↻ Future pass (unchanged for now).
- Provide feature toggle via config to disable shadows for performance-sensitive builds. ↻ Future work (still global toggle via `low_fps`).

## Rendering Pipeline Touchpoints

- `scriptbots_render::Renderer::draw_agents` (CPU) – adjusted radius, outline, shadow. ✅
- `scriptbots_world_gfx::AgentPipeline` (wgpu) – colour packets synced to CPU shading. ✅
- Terminal renderer (`terminal::draw_world`) – palette refresh applied; legibility pass still pending. ↻

## Metrics / Tests

- Extend snapshot harness with a second golden (post-polish) once code lands.
- Add CLI smoke test to ensure default scene agent count >0 and average radius above threshold.

## Dependencies / Sequencing

- Blocked on Camera Stage 2 merging (need stable `world_to_screen`).
- Coordinate with PurpleBear for wgpu pipeline adjustments to avoid conflicting shader work.

## Open Questions

1. Do we need a night-mode palette variant in MVP? (likely defer).
2. Should we introduce configurable palette via runtime knobs now or later?
3. How will the terminal renderer represent shadows without colour blending?

Action once Stage 2 is green: update PLAN §3.1/§3.2 to `[Currently In Progress – RedCastle]` and begin coding per this outline.
