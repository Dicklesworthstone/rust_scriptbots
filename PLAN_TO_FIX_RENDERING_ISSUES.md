# Plan to Fix Rendering Issues

**Coordination Broadcast (2025-10-30 15:42 UTC)**: Codex requesting immediate lead assignments + acknowledgment from all active agents. Please claim sections tagged `[Owner Needed]` or confirm review responsibilities.


## 1. Lock Down the Rendering Spec [Currently In Progress]

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


## 2. Build a Real Camera System [Stage 2 In Progress – PinkMountain]

- Stage 1 delivered: `camera::Camera` module extracted, renderer migrated to use it, and invariants/unit tests landed (`camera::tests`, `camera_invariants_tests`).
- Stage 2 focus: wire the new API through GPUI input handlers, terminal renderer, and offscreen paths while queuing UX improvements for later stages.

### 2.1 Camera Abstraction [Completed – PurpleBear 2025-10-30]
- `crates/scriptbots-render/src/camera/mod.rs` encapsulates zoom/pan math, exposes snapshots, and replaces the old inline `CameraState`.
- Mutex upgrades completed; renderer now owns `Arc<Mutex<Camera>>` and shares snapshots with pickers and tests.

### 2.2 Deterministic Initial State [Completed – PurpleBear 2025-10-30]
- `Camera::ensure_default_zoom` locks legacy scale (`0.2`) against computed base scale; recorded in viewport metrics for deterministic snapshots.
- Default world-centering happens once per render cycle, matching GLUT behaviour.

### 2.3 User Interaction UX [Stage 2 In Progress – PinkMountain]
- TODO: route GPUI mouse/keyboard events through `Camera::start_pan/update_pan/apply_scroll` without syncing legacy helpers.
- TODO: add `Camera::world_to_screen` consumers for HUD overlays and inspector panels.
- TODO: schedule follow-mode refactor (fit-selection buttons) after Stage 2 baseline is stable.

### 2.4 Testing and Telemetry [Ongoing – PinkMountain]
- Stage 1 unit + invariant tests complete; extend coverage in Stage 2 to include terminal/offscreen camera parity and HUD coordinate readouts.
- Coordinate with PurpleBear to hook new assertions into the snapshot harness once Stage 2 lands.


## 3. Match and Improve the Classic Visuals [Needs Lead]

### 3.1 Palette and Styling [Currently In Progress – RedCastle]
- Recreate the six terrain colours from the GLUT shader.
- Define agent colour rules (herbivore vs carnivore tinting, health indicators).
- Implement halo, spike, selection rings, and indicator pulses (matching timing).

### 3.2 Agent Legibility [Currently In Progress – RedCastle]
- Increase default render radius or draw scaled sprites so agents remain visible at overview zooms.
- Add drop shadows or outlines for contrast against terrain.
- Provide optional AO/lighting/elevation shading.

### 3.3 Modern UI Polish [Owner Needed]
- Port HUD layout faithfully, then upgrade with:
  - Rounded cards, subtle gradients, consistent typography.
  - Responsive layout for different window sizes.
  - Clear state indicators (paused, closed world, debug overlays).
- Ensure theme works in both light/dark modes if implemented.

### 3.4 Performance and Accessibility [Owner Needed]
- Validate new styling doesn’t regress FPS.
- Support colour-blind palettes (toggle in settings).
- Provide high-contrast mode.

### 3.5 Validation [Owner Needed]
- Compare new renderer output side-by-side with legacy screenshot.
- Include QA checklist: agent visibility, terrain differentiation, HUD readability.


## Execution Strategy

1. **Spec first:** complete section 1 deliverables and checkpoints before touching code.
2. **Camera system:** implement core camera + tests; update renderer paths to use it.
3. **Visual polish:** port palette and overlays, test on CPU and GPU paths.
4. **Regression harness:** wire snapshot tests into CI.
5. **Documentation:** update README and plan doc with new renderer guidance.

All work items should reference this plan and mark progress inline so the team stays aligned.
