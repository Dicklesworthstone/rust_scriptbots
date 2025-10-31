# Plan to Fix Rendering Issues

**Coordination Broadcast (2025-10-30 15:42 UTC)**: Codex requesting immediate lead assignments + acknowledgment from all active agents. Please claim sections tagged `[Owner Needed]` or confirm review responsibilities.


## 1. Lock Down the Rendering Spec [Completed – Codex & RedCastle 2025-10-30]

### 1.1 Capture the Reference Behaviour [Completed – Codex 2025-10-30]
- ✅ Legacy GLUT renderer instrumented under Xvfb; deterministic PNGs in `docs/rendering_reference/*.png` with hashes recorded in `checksums.txt`.
- ✅ `docs/rendering_reference/legacy_renderer_spec.md` expanded with world geometry, camera math, HUD/agent rendering details, palette samples, and FPS baselines (Xvfb + historical hardware).
- ✅ Annotated ROI tables for each screenshot available in `docs/rendering_reference/legacy_reference_annotations.md` to guide downstream automation.
- ✅ Headless capture recipe documented in `legacy_capture_checklist.md` (manual path retained, scripted path added for CI repeatability).

### 1.2 Specification Deliverables [Completed – RedCastle 2025-10-30]
- Rendering spec resides in `docs/rendering_reference/legacy_renderer_spec.md`, now linked to companion SVG overlays (`legacy_*_overlay.svg`) for onboarding docs and regression tooling.
- Reference PNGs + SHA256 published in `docs/rendering_reference/` (`checksums.txt`).
- Diff table for keyboard/mouse controls captured in §Camera & Transform Semantics of the spec; consider duplicating into quick-reference appendix for easier cross-team consumption.

### 1.3 Regression Guard Rails [⚠️ Coordination Needed – Volunteers Please Add Your Handle]
- **Snapshot harness** [Completed – PurpleBear 2025-10-30]  
  - `crates/scriptbots-render/tests/snapshot.rs` exercises `render_png_offscreen` with a seeded `ScriptBotsConfig`, 16 seeded agents, and 120 warmup ticks, comparing against `docs/rendering_reference/golden/rust_default.png` (diffs land in `target/snapshot-failures/`).  
  - Golden PNG generated via `cargo run -p scriptbots-app --no-default-features --features gui --bin scriptbots-app -- --rng-seed 424242 --dump-png docs/rendering_reference/golden/rust_default.png --png-size 1600x900 --threads 1 --low-power`.  
  - SHA256 recorded in `docs/rendering_reference/checksums.txt`; see `snapshot_harness_design.md` for follow-up ideas (perceptual diff, additional scenes).
- **Viewport invariants** [Completed – PinkMountain & PurpleBear 2025-10-30]  
  - `camera_invariants_tests` (see `crates/scriptbots-render/src/lib.rs`) computes viewport padding and asserts the visible bounds map to `(0,0)` and `(world_width, world_height)` via `screen_to_world`.  
  - Same suite verifies default zoom keeps `bot_radius` ≥ 2 pixels using differential sampling around the viewport midpoint.
- **CI integration** [Completed – PurpleBear 2025-10-30]  
  - `.github/workflows/ci.yml` now includes a `render_regression` job (Ubuntu + Windows) that runs `cargo test -p scriptbots-render -- --nocapture`, uploads `target/snapshot-failures/` artifacts on diff, and uses `dorny/paths-filter` to skip PR runs without renderer-related changes.  
  - Future follow-up: broaden path filter if additional rendering assets land outside current patterns.
- **Coordination ask:** please edit this subsection to add `[Currently In Progress – <YourName>]` next to any bullet you pick up, and drop a quick status line in `docs/rendering_reference/coordination.md` (created below) so we avoid duplicate effort.


## 2. Build a Real Camera System [Completed – PinkMountain & RedSnow 2025-10-30]

- Stage 1 delivered: `camera::Camera` module extracted, renderer migrated to use it, and invariants/unit tests landed (`camera::tests`, `camera_invariants_tests`).
- Stage 2 focus: wire the new API through GPUI input handlers, terminal renderer, and offscreen paths while queuing UX improvements for later stages.

### 2.1 Camera Abstraction [Completed – PurpleBear 2025-10-30]
- `crates/scriptbots-render/src/camera/mod.rs` encapsulates zoom/pan math, exposes snapshots, and replaces the old inline `CameraState`.
- Mutex upgrades completed; renderer now owns `Arc<Mutex<Camera>>` and shares snapshots with pickers and tests.

### 2.2 Deterministic Initial State [Completed – PurpleBear 2025-10-30]
- `Camera::ensure_default_zoom` locks legacy scale (`0.2`) against computed base scale; recorded in viewport metrics for deterministic snapshots.
- Default world-centering happens once per render cycle, matching GLUT behaviour.

### 2.3 User Interaction UX [Completed – PinkMountain & RedSnow 2025-10-30]
- GPUI canvas + WGPU compositor now consume `Camera::layout` for shared scaling/offsets; follow recentering flows through the module.
- Offscreen PNG exporter (`render_png_offscreen`) now reuses the camera layout + `world_to_screen`, so REST/CLI snapshots match the live viewport framing.
- HUD overlay + inspector panels surface world/screen coordinates via `world_to_screen` (Completed – PinkMountain 2025-10-30).
- Debug overlay + agent outline passes reuse `CameraSnapshot::world_to_screen`, eliminating legacy offset/scale math (Completed – RedSnow 2025-10-30).
- Follow-mode tooling restored: header chips expose “Fit World”/“Fit Selection” actions backed by `Camera::fit_world` / `Camera::fit_bounds`, keeping zoom/pan synced with focused or selected agents (Completed – RedSnow 2025-10-30).

### 2.4 Testing and Telemetry [Completed – PinkMountain & PurpleBear 2025-10-30]
- Camera invariants and snapshot harness run in CI (`render_regression` job) covering terminal/offscreen parity; HUD overlays now reuse shared transforms so additional assertions can ride existing harness without new wiring.


## 3. Match and Improve the Classic Visuals [Completed – RedCastle 2025-10-30]

### 3.1 Palette and Styling [Completed – RedCastle 2025-10-30]
- Terrain palettes now sourced from the legacy hex values (`LEGACY_TERRAIN_BASE/ACCENT` in `scriptbots-render`), covering CPU + wgpu paths and terminal renderer fallbacks.
- Food shading tweaked to blend with the bloom palette; snapshot harness verifies updated colours.
- Spec + coordination docs updated (`visual_polish_plan.md`) for future theme variants.

### 3.2 Agent Legibility [Completed – RedCastle 2025-10-30]
- Raised minimum agent radius (world + GPU paths) and strengthened outline stroke with dark ink.
- Added soft drop shadows, warmer selection glows, and health-aware body shading so agents read clearly at default zoom.
- GPU pipeline now consumes the same colour grades to avoid mismatch between rendering backends.
- Follow-up parity: replace CPU quad body rendering with circle paths + heading cues [Currently In Progress – BlueMountain 2025-10-31]
    - [x] Inventory runtime data needed for rich agent avatars (wheels, outputs, sensors, diet, audio).
    - [x] Extend `AgentRenderData` + snapshot extraction with:
        - [x] Left/right wheel outputs or derived speeds.
        - [x] Boost flag intensity and reproduction counters.
        - [x] Trait modifiers (eye, smell, sound, blood).
        - [x] Sensor metadata (eye directions/FOV, sound multiplier).
        - [x] Sound output, food delta, herbivore tendency, temperature preference.
    - [x] CPU renderer:
        - [x] Replace circular body with oriented capsule shell.
        - [x] Draw dual wheel assemblies with velocity streaks.
        - [x] Render spike spear with length/intensity tint.
        - [x] Add mouth aperture that animates with food intake/sound output.
        - [x] Encode herbivore vs carnivore banding.
        - [x] Paint sensor accessories (eyes/ears) scaled by trait modifiers.
        - [x] Maintain selection/indicator/boost halos.
        - [x] Keep debug overlays working with new geometry.
    - [x] WGPU renderer parity:
        - [x] Expand `AgentInstance` payload with new fields.
        - [x] Update WGSL shader to render capsule + wheels + spike + mouth.
        - [x] Ensure boost/reproduction effects match CPU path.
    - [ ] Snapshot/offscreen helper updates and golden refresh (once visuals approved).
    - [ ] Remove temporary CPU fallback rect logic after verifying layout stability.

### 3.3 Modern UI Polish [Completed – RedCastle 2025-10-30]
- HUD header now surfaces paused/running, world mode, speed, and follow status via consistent chips; world info subline adopts the shared theme palette.
- Summary/analytics/history cards share unified background/border styling and typography, with palette-aware colors across accessibility modes.
- WGPU + CPU HUD components respect the accessibility palette, improving readability in high-contrast and color-blind modes.

### 3.4 Performance and Accessibility [Completed – RedCastle 2025-10-30]
- Terminal renderer now supports cycling through Natural → Deuteranopia → Protanopia → Tritanopia → High Contrast palettes via the `c` key; header/help expose the active mode.
- Shared colour theme infrastructure maps HUD and terrain glyphs to the new palettes, improving legibility in emoji and ASCII modes alike.
- Headless perf sweep (`SCRIPTBOTS_MODE=terminal SCRIPTBOTS_TERMINAL_HEADLESS=1 SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES=240 ...`) recorded ~240 frames in 25s (avg energy mean 0.355, births 60, no deaths) at `--threads 2`, confirming no regressions after theme changes (report archived in `/tmp/terminal_report.json`).

### 3.5 Validation [Completed – RedCastle 2025-10-30]
- Compared `legacy_default.png` vs. refreshed `golden/rust_default.png`; MAE across channels ≈89 (RGBA), maximum diff 248 (expected given new palette), but agents occupy ~25% of viewport with legible outlines instead of tiny specks.
- QA checklist ✅ Agent visibility (default zoom shows 20 agents with halos and shadows), ✅ Terrain differentiation (six-tone palette aligned with legacy spec), ✅ HUD readability (header chips + metric cards readable at 1080p/terminal width 120).
- Terminal HUD validated via headless run (`SCRIPTBOTS_MODE=terminal ...`) confirming text contrast under all palette modes.


## Execution Strategy

1. **Spec first:** complete section 1 deliverables and checkpoints before touching code.
2. **Camera system:** implement core camera + tests; update renderer paths to use it.
3. **Visual polish:** port palette and overlays, test on CPU and GPU paths.
4. **Regression harness:** wire snapshot tests into CI.
5. **Documentation:** update README and plan doc with new renderer guidance.

All work items should reference this plan and mark progress inline so the team stays aligned.
