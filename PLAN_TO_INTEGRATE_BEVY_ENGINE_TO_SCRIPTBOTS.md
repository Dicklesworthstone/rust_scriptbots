# Plan to Integrate Bevy Engine into ScriptBots

_Prepared by RedSnow — 2025-10-30_

> Goal: Introduce an opt-in Bevy-powered 3D/2.5D renderer (camera + world + HUD) that can coexist with the current GPUI renderer, controlled by feature flags and runtime switches, without disrupting existing workflows.

---

### Phase 4 Working TODO — OrangeLake (2025-10-31)

- [x] Add `ControlCommand::UpdateSimulation` + `SimulationCommand` plumbing in `scriptbots-core`.
    - [x] Extend `WorldState` with pending simulation control requests + helpers.
    - [x] Update `apply_control_command` and add regression test.
- [x] Wire GPUI renderer (`scriptbots-render`) to emit simulation commands for play/pause/step/speed.
- [x] Teach terminal renderer to honour queued simulation commands before stepping.
- [x] Update Bevy playback handlers to submit `SimulationCommand` and sync with queued requests.
- [x] Enhance Bevy simulation driver to merge queued commands into live control and respect auto-pause reason.
- [x] Refresh Bevy HUD styling to reflect terrain relief palette (drop shadow, accent borders).
- [x] [Completed – BrownLake 2025-10-31] Document command contract + styling adjustments in phase plan & coordination log and notify RedSnow.
    - [x] Update Phase 4 section with SimulationCommand contract summary.
    - [x] Append coordination log entry reflecting the new contract and styling work.
    - [x] Send Agent Mail to RedSnow confirming updates and awaiting feedback.

#### Simulation Command Contract Notes (2025-10-31 – BrownLake)

- All control surfaces now emit `ControlCommand::UpdateSimulation(SimulationCommand)`; the struct carries `paused: Option<bool>`, `speed_multiplier: Option<f32>`, and `step_once: bool`.
- `WorldState::enqueue_simulation_command` clamps `speed_multiplier` into `[0.0, 32.0]` and stores the request until render drivers call `drain_simulation_commands`.
- Bevy’s driver (`apply_simulation_command_to_state`) mirrors GPUI/terminal handling: toggling `paused` clears `auto_pause_reason`, `speed_multiplier` ≤ `0.0` forces a pause, and `step_once` both pauses and asks for a single tick.
- Auto-pause sources (population/follow/spike guards) push a synthetic `SimulationCommand { paused: Some(true), speed_multiplier: Some(0.0), step_once: false }`, so every surface converges on the same state and HUD messaging.
- GPUI (`crates/scriptbots-render/src/lib.rs`), terminal (`crates/scriptbots-app/src/terminal/mod.rs`), and Bevy (`crates/scriptbots-bevy/src/lib.rs`) all funnel through the same submission helper to ensure parity in logging, throttling, and error handling.

#### HUD Styling Refresh (2025-10-31 – BrownLake)

- Playback and follow control rows now share the relief palette: idle backgrounds `Color::srgba(0.16, 0.22, 0.33, 0.92)`, hover `Color::srgba(0.22, 0.30, 0.46, 0.95)`, active `Color::srgba(0.34, 0.26, 0.64, 0.95)`, and a border accent `Color::srgba(0.32, 0.38, 0.58, 1.0)`.
- Buttons incorporate iconography plus shortcut hints (`▶ Run (Space)`, `🎯 Follow selected (Ctrl+S)`, etc.) so parity checks can confirm both affordance and key bindings.
- HUD copy now exposes playback multiplier and optional `auto_pause_reason`, which is kept in sync across renderers via the shared `SimulationControl` snapshot.
- Root HUD card uses a translucent navy backdrop (`Color::srgba(0.07, 0.11, 0.18, 0.72)`) with 12 px offsets and 10 px padding, matching the terrain relief theme introduced earlier in Phase 3.
- [x] Run `cargo check` for workspace & targeted tests (`scriptbots-bevy`, `scriptbots-core`).
    - [x] 2025-10-31 – BrownCreek: `cargo check -p scriptbots-bevy` clean under Bevy 0.17.2; full workspace check currently blocked by `crates/scriptbots-world-gfx` missing an explicit `scriptbots-core` dependency (pre-existing TODO).

---

## 0. Guiding Principles

1. **Feature-flag everything**  
   - Workspace-level feature `bevy_render` gates all new crates/dependencies.  
   - Binary `--renderer=bevy|gpui` CLI switch selects implementation at runtime when the feature is compiled in.
2. **Rust-native pipeline**  
   - Prefer Bevy & Rust crates (bevy_ecs, bevy_asset, bevy_panorbit_camera) before exploring FFI or script layers.  
   - Maintain compliance with internal best practices (`RUST_SYSTEM_PROGRAMMING_BEST_PRACTICES.md`).
3. **Zero-regression fallback**  
   - GPUI renderer remains default; snapshot harness must run against both backends until parity proven.  
   - No destructive edits; old renderer paths remain untouched unless explicitly refactored for reuse.
4. **Composable architecture**  
   - Reuse existing `RenderFrame`/`HudSnapshot` data as the stable interface.  
   - Encapsulate Bevy-specific logic inside a dedicated crate to avoid leaking dependencies into core simulation.

---

## 1. Dependency & Workspace Setup

1.1 **Create Bevy integration crate** `crates/scriptbots-bevy`
   - `Cargo.toml` lists Bevy minimal set (`bevy_app`, `bevy_ecs`, `bevy_render`, `bevy_pbr`, `bevy_winit`, `bevy_sprite`, `bevy_diagnostic`) using `default-features = false` plus required feature flags (`bevy_wgpu`, `bevy_gltf`).  
   - Add `bevy_panorbit_camera` (or `bevy_map_camera`) dependency behind optional feature `bevy_camera_rts`.

1.2 **Workspace features**
   - Root `Cargo.toml`: define `[features] bevy_render = ["scriptbots-bevy"]`.  
   - Condition existing binaries (`scriptbots-app`, CLI tools) to compile Bevy crate only when feature set.  
   - Ensure CI matrix includes `--no-default-features --features bevy_render` build job.

1.3 **Engine configuration**
   - Extend `rust-toolchain.toml` if Bevy requires nightly components (e.g., `rustfmt`, `clippy` compatibility).  
   - Record any OS-level dependencies (Vulkan/Metal, X11) in README + onboarding docs.

1.4 **Bevy 0.17 migration — BrownCreek (2025-10-31)**  
   - Bump Bevy crates to `0.17.2` and add explicit `bevy_mesh` dependency for typed mesh/indices access.  
   - Replace legacy bundles with typed components: `Mesh3d`/`MeshMaterial3d`, `Camera3d` + `Camera`, `Button`, `Node`, and `Text` stacks now drive the renderer and HUD.  
   - Swap deprecated `EventReader`/`EventWriter` usage for `MessageReader`/`MessageWriter`, and migrate timing helpers to `delta_secs()`/`elapsed_secs_f64()`.  
   - Configure clear color per camera via `Camera::clear_color`; global resource insertion is no longer required.  
   - Rebuild the HUD UI using typed nodes/text, adjust integration tests, and confirm `cargo check -p scriptbots-bevy` passes on 2025-10-31.

---

## 2. Data Model & Interop Contract

2.1 **Define Snapshot Interface**
   - Use existing `RenderFrame` (world state, agent draws, terrain cells) as handoff structure.  
   - Add conversion trait `IntoBevyScene` (in new crate) that maps `RenderFrame` + `HudSnapshot` into ECS resources/components.

2.2 **Coordinate Systems**
   - Document world units vs. Bevy units (meters).  
   - Provide helper `WorldScale` constant so agents, terrain, and camera share uniform conversions.  
   - Support Z-up vs. Y-up decision; default to Bevy’s Y-up, adapt existing data accordingly.

2.3 **Asset & Material Mapping**
   - Terrain: translate heightmap/tiles into meshes or instanced quads with materials.  
   - Agents: use instanced meshes (billboarded quads or low-poly models) matching existing palette.  
   - HUD: plan for Bevy UI overlay (2D camera) or egui integration for textual metrics.

2.4 **Wave Function Collapse Terrain Heightfield**
   - Snapshot payload must expose WFC-derived fields from `TerrainLayer` (width, height, `cell_size`, per-tile `elevation`, `moisture`, `accent`, `TerrainKind`, and fertility/temperature biases).  
   - Encode topography as a contiguous f32 grid in world units; apply `WorldScale` so 1 terrain cell aligns with Bevy meters.  
   - Define `TerrainSnapshot` struct (`heights: Vec<f32>`, `palette: Vec<TerrainKind>`, `biome_metrics: Vec<BiomeSample>`) to keep renderer-agnostic semantics.  
   - Introduce chunk metadata (`TerrainChunkId`, extent, dirty flag) so Bevy can rebuild meshes incrementally when simulation mutates tiles.  
   - Persist palette-to-material lookup table that maps `TerrainKind` + moisture bands to albedo/tint/roughness/emissive parameters; reuse existing GPUI palette constants for parity.  
   - Document how WFC seed / deterministic replay guarantees identical terrain between renderers; ensure snapshot hash includes the heightfield so CI catches divergence.

---

## 3. Architectural Components (Bevy Side)

3.1 **Entry Point**
   - `scriptbots-bevy::launch(RenderRuntimeConfig)` spins up `App` with minimal plugins (`DefaultPlugins` subset).  
   - Provide `App::add_plugins` wrapper toggling diagnostics, FPS counters when `SB_DIAGNOSTICS=1`.

3.2 **Systems & Resources**
   - `SimulationSnapshot` resource: receives latest `RenderFrame` copy.  
   - `CameraRig` resource: stores orbit/RTS parameters; updates Bevy `Transform` every frame.
   - `AgentMaterialCache`: caches GPU materials per agent palette/selection state.
   - `TerrainChunkMap`: manages terrain/food meshes, updated incrementally (dirty-rectangle updates).

3.3 **ECS Schedule**
   - Stage `BeforeRender`: ingest new snapshots from channel (see §4).  
   - Stage `Update`:  
     - `sync_agents_system`: create/update/despawn agent entities.  
     - `sync_terrain_system`: update terrain meshes or textures.  
     - `sync_hud_system`: push HUD metrics to UI components.  
   - Stage `PostUpdate`: camera controllers, debug overlays.

3.4 **Camera Controllers**
   - Provide `OrbitCamera` plugin (via `bevy_panorbit_camera`) for default view.  
   - Implement custom `FollowCameraSystem` to match GPUI follow modes (Off/Selected/Oldest).  
   - Map inputs: WASD pan, QE rotate, scroll zoom, middle-mouse drag.

3.5 **Rendering Pipeline**
   - Use Bevy’s PBR for 3D path; fallback to 2D if we keep flat world.  
   - Define `EnvironmentSettings` resource controlling lighting, skybox, bloom toggles.  
   - Hook up screenshot capture (use `bevy::render::texture::Image` exports) for parity with snapshot harness.

3.6 **Terrain Mesh & Material Pipeline**
   - `TerrainChunkMap` owns a pool of `TerrainChunk` components (configurable chunk size, default 64×64 tiles) with attached `Handle<Mesh>` + `Handle<StandardMaterial>`.  
   - Mesh generation: sample height grid per chunk, emit vertices in XZ plane with Y = height, compute normals via Sobel kernel or cross-product of adjacent triangles, and populate tangents for future normal maps.  
   - UV strategy: `u = x / terrain_width`, `v = z / terrain_height` to support procedural texture sampling; add secondary UV set for tri-planar shaders if needed.  
   - Material authoring: use Bevy `StandardMaterial` with palette-derived base color + roughness/metallic values; pack biome accents into emissive/clear coat channels for stylised highlights.  
   - Procedural texturing hooks: expose `TerrainMaterialParams` resource (height thresholds, moisture tints, shoreline foam intensity) so shader graph or WGSL material can blend snow/sand/moss overlays without new asset pipelines.  
   - LOD considerations: provide optional decimation pass (skip every other vertex beyond configurable distance) and frustum-aligned chunk culling; defer to future perf work but keep interfaces ready.

---

## 4. Runtime Integration

4.1 **Cross-thread Communication**
   - Spawn Bevy app on dedicated thread (Bevy manages its own event loop).  
   - Create `std::sync::mpsc` or `crossbeam` channel bridging simulation (GPUI) thread → Bevy renderer.  
   - Snapshot producer sends `Arc<RenderFrame>` at fixed cadence (every simulation tick or when state changes).

4.2 **Renderer Selection Layer**
   - In `scriptbots-app/src/main.rs`, wrap renderer launch in strategy enum:
     ```rust
     match renderer_kind {
         RendererKind::Gpui => launch_gpui_renderer(...),
         RendererKind::Bevy => #[cfg(feature = "bevy_render")] launch_bevy_renderer(...),
     }
     ```
   - Parse CLI/ENV: `SCRIPTBOTS_RENDERER=bevy` or `--renderer=bevy`.

4.3 **Input Bridging**
   - Simulation events (pause, speed, selection) emitted via existing bus.  
   - Mirror interactions from Bevy UI back to core through same channel (e.g., selecting agent in Bevy triggers `ControlCommand::FocusAgent`).  
   - Provide trait `RendererControlSink` for parity across GPUI/Bevy.

4.4 **HUD/UI**
   - Phase 1: replicate textual overlay using Bevy UI (stacked flex layout).  
   - Phase 2: optional egui integration for inspector panel parity.  
   - Consider reusing `HudSnapshot` to avoid diverging logic.

---

## 5. Feature Rollout Phases

| Phase | Objective | Deliverables | Exit Criteria |
| --- | --- | --- | --- |
| 0 [Completed – RedSnow 2025-10-30] | Scaffolding | New crate, feature flag, minimal Bevy app that opens window and clears background. | `cargo run --features bevy_render --renderer=bevy` opens blank window. |
| 1 [Currently In Progress – GPT-5 Codex 2025-10-30] | Static world visuals | Stream WFC terrain snapshot into chunked heightfield meshes with palette-driven PBR materials, render agents as instanced meshes under static camera. | Snapshot harness captures Bevy terrain/agent frame that matches GPUI reference histogram + feature checks within tolerance. |
| 2 [Ready for Review – OrangeLake 2025-10-30 (prev: RedSnow 2025-10-30)] | Camera controls | Orbit + follow modes mapped, input parity with GPUI (mouse, keyboard). | QA sign-off that camera UX matches spec. |
| 3 [Ready for Review – OrangeLake 2025-10-30 (prev: RedSnow 2025-10-30)] | HUD parity | Overlay tick stats, controls, selection info. | HUD shows same metrics as GPUI reference screenshot. |
| 4 [Currently In Progress – OrangeLake 2025-10-30] | Interactivity | Agent selection, follow toggles, command buttons. | Round-trip commands (select agent) confirmed via simulation logs. |
| 5 | Polish + QA | Performance tuning, lighting, debug overlays, CI integration. | Bevy path passes `render_regression` job + manual smoke checklist. |

- Phase 2 & 3 Review Punch-List [In Progress – OrangeCreek 2025-10-31]
  - [x] Validate camera orbit/pan/zoom parity against GPUI shortcuts (`F`, `Ctrl+F`, `Ctrl+W`, `Q/E`, `PageUp/PageDown`, WASD + middle-mouse). _[Completed – OrangeCreek 2025-10-31]_ Reviewed `control_camera` key handling, confirmed bindings match GPUI spec, and exercised via `cargo test -p scriptbots-bevy` (covers camera shortcut + playback unit tests).
  - [ ] Confirm follow-mode cycle (Off/Selected/Oldest) maintains target in frame within ±3 % distance/yaw tolerance using shared cursor logs. _Blocked pending replay cursor logs; need capture from interactive session to compute deltas._
  - [x] Exercise HUD overlays for selection details, playback multiplier, FPS, and world stats; verify copy & formatting match GPUI snapshots. _[Completed – OrangeCreek 2025-10-31]_ Added `hud_overlay_populates_metrics` system test to assert shortcut hints, selection copy, and auto-pause messaging align with GPUI strings.
  - [ ] Capture readability screenshots at 1080p, 1440p, 4K; log findings in `docs/rendering_reference/coordination.md`. _Blocked – automated offscreen renders now pass at target resolutions; awaiting visual QA capture once GPU workstation is free._
  - [ ] Document review outcomes & sign-off in plan once two reviewers approve.

- Progress (2025-10-30 – RedSnow): Scaffolded `scriptbots-bevy` crate, workspace feature flag, CLI `--renderer=bevy`, and stub window launcher [Phase 0 ✅].
- Progress (2025-10-30 – RedSnow): Phase 1 baseline in place — Bevy renderer streams live `WorldState` snapshots, displays placeholder ground plane + agent spheres, and logs tick cadence every 120 frames; establishes plumbing for terrain/material upgrades.
- Progress (2025-10-30 – RedSnow): Minted `docs/rendering_reference/golden/bevy_default.png` via new `--dump-bevy-png` flag; checksum recorded in `docs/rendering_reference/checksums.txt`.
- Progress (2025-10-30 – RedSnow): Added `crates/scriptbots-bevy/tests/snapshot.rs` comparing `render_png_offscreen` output against `golden/bevy_default.png`; diff tooling now fails tests on byte mismatches.
- Progress (2025-10-30 – RedSnow): Camera controls underway — mouse orbit/scroll zoom/WASD pan implemented via `CameraRig`; `F` toggles follow mode, while Q/E yaw and PageUp/PageDown pitch mirror GPUI shortcuts.
- Progress (2025-10-30 – RedSnow): HUD parity underway — Bevy UI overlay replicates tick, agent counts, follow mode, and camera state.
- Progress (2025-10-30 – OrangeLake): Continuing Phase 2 camera polish (fit selection shortcut, follow parity smoothing, easing) and Phase 3 HUD parity expansion toward GPUI completeness.
- Progress (2025-10-30 – OrangeLake): Delivered camera follow-mode cycle (`F`), targeted toggles (`Ctrl+S`/`Ctrl+O`), fit selection/world shortcuts (`Ctrl+F`/`Ctrl+W`), recenter smoothing, and HUD upgrades (selection details, playback rate, FPS, world stats) ready for review.
- Progress (2025-10-31 – OrangeLake): Phase 4 interactivity in progress — Bevy renderer now submits GPUI-parity selection commands on left-click (shift extends/toggles, empty click clears) via the shared control channel.
- Coordination (2025-10-31 – OrangeLake → RedSnow): Sent Agent Mail contact request to sync on Phase 4 scope (selection radius + command semantics); awaiting acknowledgement before extending to follow toggles/command buttons.
- Progress (2025-10-31 – OrangeLake): Added HUD action row with follow-mode buttons + clear selection wired through `ControlCommand::UpdateSelection`, plus keyboard hints aligned with GPUI shortcuts.
- Note (2025-10-31 – OrangeLake): Pause/resume buttons deferred pending new `ControlCommand`; will align with RedSnow once contact request is accepted.
- TODO (2025-10-31 – OrangeLake): Build SimulationCommand pipeline + UI styling refresh per phase checklist.
- Progress (2025-10-31 – GPT-5 Codex): WFC terrain snapshot export, chunked heightfield meshing, and agent elevation alignment landed; snapshot harness updated for deterministic regeneration.
- Progress (2025-10-31 – BrownLake): Documented SimulationCommand handshake + HUD palette updates, updated coordination log, and notified RedSnow/OrangeLake for review.
- Progress (2025-10-31 – BrownLake): Added cross-platform Bevy launch scripts (Windows/Linux/macOS), updated README quickstart instructions, and closed out plan §7.2 scripting checklist pending QA follow-up.
- Coordination (2025-10-31 – OrangeCreek): Aligned with WhiteCastle & BrownSnow on remaining Bevy integration lanes — OrangeCreek owning automation/parity coverage, BrownSnow handling benchmarks + stability sweeps, WhiteCastle driving render polish backlog; BrownSnow will primary-review the Phase 4 SimulationCommand/UI wiring with an OrangeCreek secondary pass. Initial rolling check-ins pencilled for 2025-10-31 20:30 UTC; superseded by the 19:45 UTC huddle noted below, plus the standing daily sync on 2025-11-01 16:00 UTC.
- Coordination (2025-10-31 – WhiteCastle): Confirmed immediate execution plan with BrownSnow & OrangeCreek — same lane ownership, 2025-10-31 19:45 UTC huddle to unblock hardware access, automation PRs, and render polish sequencing. Review pairings: BrownSnow primary on automation/parity PRs, OrangeCreek primary on render polish branches, WhiteCastle primary on benchmark/stability artifacts. Target is to lock first deliverables within the next 6 hours and publish status back here.

#### Phase 5 QA & Performance Checklist [Currently In Progress – BrownLake 2025-10-31]

- [ ] [Currently In Progress – BlueMountain 2025-10-31] Establish benchmark scenarios
  - [x] Drafted measurement procedure + data template (`docs/perf/bevy_vs_gpui.md`) and recorded outstanding config TODOs.
  - [x] Added scenario configs (`docs/rendering_reference/configs/{dense_agents,storm_event}.toml`) aligned with measurement plan.
  - [x] Prepare turnkey benchmarking script for GPU-capable hosts and circulate instructions.
  - [ ] Capture CPU/GPU timings for three canonical seeds across GPUI vs Bevy.
    - [ ] Linux • GPUI • `default`  — _Blocked (requires Vulkan-capable workstation)._  \
          Attempted 2025-10-31 on CI host; run stalled awaiting presentation surface (see `logs/perf/20251031_default_gui.log`).
    - [ ] Linux • Bevy  • `default`  — _Blocked (requires Vulkan-capable workstation)._  \
          Current build fails (`TerrainChunkStats` field mismatches, borrow move) when compiling `scriptbots-bevy` with `--features bevy_render`; captured in `logs/perf/20251031_default_bevy.log`.
    - [ ] Linux • GPUI • `dense_agents`  — _Blocked (requires Vulkan-capable workstation)._
    - [ ] Linux • Bevy  • `dense_agents`  — _Blocked (requires Vulkan-capable workstation)._
    - [ ] Linux • GPUI • `storm_event`  — _Blocked (requires Vulkan-capable workstation)._
    - [ ] Linux • Bevy  • `storm_event`  — _Blocked (requires Vulkan-capable workstation)._
    - [ ] Windows • GPUI • all scenarios  — _Pending hardware scheduling._
    - [ ] Windows • Bevy • all scenarios  — _Pending hardware scheduling._
  - [ ] Record baseline FPS, frame time percentiles, and simulation ticks/sec; log results to `docs/perf/bevy_vs_gpui.md` once vetted.
    - [x] Added `scripts/parse_perf_logs.py` to summarise SB_DIAGNOSTICS output (CSV/CLI).
- [x] Derived initial golden diff metrics (MAE/RMSE) between `rust_default.png` and `bevy_default.png` as a reference point.
- [x] [Completed – BrownLake 2025-10-31] Instrument diagnostics
  - [x] Enable Bevy `FrameTimeDiagnosticsPlugin` + custom tracing spans gated by `SB_DIAGNOSTICS`.
  - [x] Surface summarized stats in console output (colorized) every 300 frames without flooding logs.
- [ ] Regression automation
  - [x] Extend `render_regression` workflow to run Bevy snapshot/camera tests on Linux headless GPU (reuse existing path filters).
  - [x] Add opt-in `cargo test -p scriptbots-bevy -- --include-ignored` job for exhaustive runs before releases.
  - [In Progress – OrangeCreek 2025-10-31] Exercised `render_png_offscreen` at 1080p/1440p/4K via expanded unit test; HUD metrics coverage added (`hud_overlay_populates_metrics`) to guard shortcut copy and auto-pause messaging.
  - [In Progress – OrangeCreek 2025-10-31] Golden refresh playbook drafted (`BEVY_REGEN_GOLDEN=1 cargo test -p scriptbots-bevy --test snapshot`); awaiting lighting/material updates before executing.
- [ ] Visual parity spot-checks
  - [ ] Rebuild Bevy golden PNG after lighting polish; diff against GPUI using histogram/feature checks documented in §6.1.
  - [In Progress – OrangeCreek 2025-10-31] Automated multi-resolution render smoke tests green; awaiting manual readability review + screenshots for 1080p/1440p/4K capture.
    - [In Progress – OrangeCreek 2025-10-31] Use `cargo run -p scriptbots-app --features bevy_render -- --mode bevy --dump-bevy-png <out.png> --png-size <WIDTHxHEIGHT>` (1920x1080, 2560x1440, 3840x2160) to capture HUD snapshots; document readability results in the coordination log and attach PNG paths.
- [x] Snapshot refresh status: No lighting/material changes landed in this pass; defer golden regeneration until next visual update. Documented in coordination log.
- [ ] Stability sweeps
  - [x] Documented soak-test procedure in `docs/perf/bevy_vs_gpui.md` (diagnostics-on, 30-minute runtime, per-platform targets).
  - [ ] Run 30-minute soak tests on Windows (D3D12 + Vulkan) and Linux (Vulkan) ensuring no panics or runaway memory growth.
  - [ ] Track auto-pause reasons and ensure SimulationCommand feedback loop remains consistent after long runs.
- [ ] [Currently In Progress – BrownSnow 2025-10-31] Coordination
  - [x] Confirm responsibilities with OrangeLake/RedSnow via Agent Mail before executing benchmarks.
  - [x] [Completed – OrangeCreek 2025-10-31 19:15 UTC] Capture lane split + review plan in coordination log after automation punch-list publication and reference BrownSnow’s Vulkan blocker notes in `docs/perf/bevy_vs_gpui.md`.
  - [In Progress – WhiteCastle 2025-10-31 19:22 UTC] Ping BrownLake + BlueMountain for immediate Vulkan hardware slot; contact requests filed, awaiting approval before logging confirmed slot in `docs/perf/bevy_vs_gpui.md`.
  - [x] [Completed – BrownSnow 2025-10-31 19:20 UTC] Draft soak-test execution commands/env vars in `docs/rendering_reference/coordination.md` while hardware access is pending.
  - [x] [Completed – WhiteCastle 2025-10-31 19:27 UTC] Re-ran `cargo test -p scriptbots-bevy -- --nocapture`; suite is green, confirming the previously reported AgentVisual/VertexAttributeValues regression is resolved upstream.
  - [x] [Completed – WhiteCastle 2025-10-31 19:31 UTC] Logged results + open follow-ups in `docs/rendering_reference/coordination.md` (see “Open follow-ups” list covering hardware booking, automation status, render polish kickoff).
  - [x] [Completed – BrownSnow 2025-10-31 19:30 UTC] Circulated agenda + attendance check for 20:30 UTC coordination huddle via Agent Mail (“20:30 UTC Bevy check-in – agenda & roll call”).
  - [x] [Completed – BrownSnow 2025-10-31 19:33 UTC] Pinged OrangeCreek privately to confirm availability for the 20:30 UTC huddle and offer prep support on automation/parity items.
  - [x] [Completed – BrownSnow 2025-10-31 19:31 UTC] Verified `scripts/parse_perf_logs.py` output path (`python3 scripts/parse_perf_logs.py logs/perf/20251031_default_bevy.log`) so benchmark logs can be summarised immediately once GPU runs succeed.
  - [x] [Completed – BrownSnow 2025-10-31 19:36 UTC] Ran `cargo check` (default features) to confirm workspace builds cleanly ahead of coordination huddle; only pre-existing unused helper warnings observed.
  - [In Progress – OrangeCreek 2025-10-31 19:35 UTC] Requested WhiteCastle to get OrangeLake registered on Agent Mail; Phase 4 review coordination remains blocked until OrangeLake appears in `resource://agents/data-projects-rust-scriptbots`.

#### Bevy 3D Agent Avatars [Currently In Progress – BlueMountain 2025-10-31]

- [x] Establish rich Bevy agent data plumbing [Completed – BlueMountain 2025-10-31 @ 06:05 UTC]
  - [x] Reconcile `TerrainChunkStats` changes blocking current build (add missing fields / adjust mesh pipeline). [Completed – BlueMountain 2025-10-31 @ 06:05 UTC]
  - [x] Extend Bevy snapshot conversion to include:
    - [x] Wheel velocities / boost / reproduction metrics.
    - [x] Trait modifiers (eye, smell, sound, blood) and sensor metadata (eye dirs/FOV, sound multiplier).
    - [x] Mouth/food delta, sound level/output, herbivore tendency, temperature preference.
  - [x] Ensure new fields are feature-flagged to avoid impacting GPUI/terminal builds.
- [x] 3D geometry + materials [Completed – BlueMountain 2025-10-31 @ 06:05 UTC]
  - [x] Model capsule body + dual wheels with instanced meshes (capsule + torus/lathed wheels).
  - [x] Add spike mesh with length/intensity tint driven from runtime data.
  - [x] Implement mouth slot, diet accent band, and boost exhaust using emissive/alpha quads.
  - [x] Attach sensor meshes (eyes/ears) scaled by trait modifiers; orient via eye directions.
- [x] Visual effects & state cues [Completed – BlueMountain 2025-10-31 @ 06:34 UTC]
  - [x] Encode herbivore vs carnivore tinting via material uniforms.
  - [x] Animate mouth aperture with food_delta/sound_output.
  - [x] [Completed – BlueMountain 2025-10-31 @ 06:32 UTC] Add vocalization arcs and directional audio decals.
    - [x] Spawn dedicated additive quads/meshes for inner/outer sonic arcs aligned to heading.
    - [x] Drive scale/opacity from `sound_output`, `sound_multiplier`, and `sound_level`.
    - [x] Cull/hide arcs when amplitude falls below threshold.
  - [x] Fold selection halos and boost flames into the visibility logic so inactive agents skip those draw calls.
- [ ] Shader & pipeline updates
  - [x] [Completed – BlueMountain 2025-10-31 @ 06:34 UTC] Introduce Bevy-side accessibility palette support.
    - [x] Mirror `ColorPaletteMode` enum + transforms.
    - [x] Add renderer resource storing current palette (keyboard + HUD toggle).
    - [x] Apply palette adjustments to agent materials and HUD swatches.
  - [ ] Optimize agent draw calls / batching.
    - [x] Add visibility toggles for zero-intensity overlays (boost, indicator, vocalization, selection).
    - [ ] Evaluate grouping of add-on quads into shared gizmo/mesh batches and document follow-up for full instancing path.
- [ ] Validation
  - [ ] Add snapshot capture of Bevy 3D avatars (headless `--dump-bevy-png`) and compare against CPU reference.
  - [ ] Document new assets/material workflow and update coordination log.

---

## 6. Testing & Regression Strategy

6.1 **Snapshot Harness Extension**
   - Add test target `cargo test -p scriptbots-render --features bevy_render -- --nocapture` capturing Bevy screenshot via headless mode (use `WGPU_BACKEND=gl` for CI).  
   - Compare output PNG to new golden `docs/rendering_reference/golden/bevy_default.png`.
   - CI now runs `cargo test -p scriptbots-bevy` alongside render harness to enforce parity; `BEVY_REGEN_GOLDEN=1 cargo test -p scriptbots-bevy --test snapshot` regenerates the deterministic golden (seed fixed via `ScriptBotsConfig::rng_seed`).
   - Extend diff tooling to compute terrain height histograms and SIFT keypoints on the rendered frame; fail builds when variance exceeds GPUI thresholds to catch heightfield regressions.

6.2 **Unit Tests**
   - `scriptbots-bevy` crate: test conversion helpers (`RenderFrame` → `BevyAgentBundle`).  
   - Camera controller: property tests ensuring orbit/follow maintains agent in frame.
   - Terrain mesh builder: deterministic fixture asserting chunk vertex positions, normals, material selection, and dirty-rectangle rebuild logic.

6.3 **CI Matrix**
   - Linux GPU (WGPU) + Windows builds.  
   - Feature combos:  
     - `default` (GPUI)  
     - `--features bevy_render` (Bevy)  
     - Optional: `--features bevy_render,ci_headless`

6.4 **Performance Baselines**
   - Capture FPS metrics via Bevy diagnostics plugin; log to console with color-coded output abiding by project logging style.  
   - Compare GPU timings vs. existing renderer; document results in `docs/perf/bevy_vs_gpui.md`.

---

## 7. Documentation & Tooling

7.1 **Developer Onboarding**
   - Update README with Bevy prerequisites, feature flag usage, CLI examples.  
   - Create `docs/rendering_reference/bevy_integration.md` describing architecture, camera controls, debugging tips.

7.2 **Scripts**
   - [In Progress – BrownLake 2025-10-31] Extend `run_*` scripts with Bevy-aware launch options.  
       - [x] Added `run_windows_version_with_bevy.bat` to invoke `--features bevy_render --mode bevy` with Vulkan/high-performance defaults.  
       - [x] Add `run_linux_with_bevy.sh` (export Linux-friendly defaults, ensure executable bit, invoke `--features bevy_render -- --mode bevy`).  
       - [x] Add `run_macos_version_with_bevy.sh` (Metal backend, retina scaling guidance, same cargo invocation).  
       - [x] Verify scripts respect existing env overrides (threads, renderer flags) and mention where to tweak.  
       - [x] Update onboarding docs/README once cross-platform scripts land.  
       - [x] Announce script availability via Agent Mail + coordination log entry.

7.3 **Logging & Telemetry**
   - Use `tracing` subscribers bridging Bevy logs into existing `SB_LOG_LEVEL`.  
   - Ensure colorized console output matches project expectations.

---

## 8. Risk Mitigation

- **Bevy version churn**: Pin to specific release (e.g., 0.15) and monitor migration guides; plan quarterly upgrades.  
- **Resource contention**: Bevy runtime has its own event loop; ensure we cleanly shut it down on exit (send `AppExit`).  
- **Asset bloat**: Avoid bundling large assets; derive terrain/agents procedurally or reuse existing textures.  
- **CI GPU availability**: WGPU/CI can be flaky; provide `--headless` flag to render offscreen using Bevy’s headless plugin.  
- **Input divergence**: Maintain a shared command enum so UI actions are consistent across renderers.

---

## 9. Open Questions (to resolve before Phase 1 coding)

1. Target visual style: full 3D terrain or stylized 2.5D (billboards)?  
2. Rendering determinism: do we require Bevy frames to match GPUI pixel-perfect, or is perceptual similarity sufficient?  
3. Audio integration: should Bevy handle spatial audio or defer to existing audio backend?  
4. Multiplayer/hosted mode: will Bevy runtime need to run headless on servers (affects plugin selection)?

---

## 10. Next Actions Checklist

- [ ] Review plan with PinkMountain, PurpleBear, RedCastle; capture feedback in coordination log.  _[In Progress – BrownCreek 2025-10-31: coordination mail sent, awaiting responses]_  
- [x] Approve Bevy version & dependency policy.  _[Completed – BrownCreek 2025-10-31: locked to Bevy 0.17.2 + explicit `bevy_mesh` dependency]_  
- [x] Kick off Phase 0 scaffold (branch `feature/bevy-integration-phase0`).  _[Completed – RedSnow 2025-10-30 (confirmed in review pass)]_  
- [x] Prepare CI job definitions (`ci/bevy_render.yml`).  _[Completed – BrownLake 2025-10-31: see CI workflow updates logged in coordination doc]_  
- [x] Establish visual parity acceptance criteria with design/QA.  _[Completed – BrownCreek 2025-10-31: criteria documented below]_

#### Visual Parity Acceptance Criteria (2025-10-31 – BrownCreek)

1. **Snapshot delta thresholds**  
   - Run `render_regression` suite against GPUI and Bevy backends; PNG pixel MAE must remain ≤ 12 and SSIM ≥ 0.96 for reference frames (default terrain, dense agents, storm event).  
   - Heightfield-derived terrain meshes compared via vertex histogram; variance ≤ 0.5% relative to GPUI baseline.
2. **HUD/overlay consistency**  
   - Tick, agent counts, selection summaries, and playback state text must match exactly across renderers for the same snapshot (string compare).  
   - Follow/playback buttons present identical shortcut hints and color state transitions (idle/hover/active).
3. **Interaction equivalence**  
   - Selection raycasts: picking identical agent IDs given shared cursor logs across 25 random seeds (no mismatches allowed).  
   - Camera fit/follow commands bring targets within ±3% of GPUI camera distance and yaw/pitch.
4. **Performance sanity**  
   - Bevy renderer achieves ≥ 55 FPS on replay benchmark `dense_agents` at 1080p on baseline Linux GPU runner; gap ≤ 10 FPS vs GPUI reference.
5. **Automation hooks**  
   - Add parity suite invocation (`cargo test -p scriptbots-bevy -- --include-ignored parity`) to CI docs before enabling blocking status; design sign-off requires one successful run logged in coordination doc.

Once phases progress, update this document inline with `[In Progress – <Name>]` markers to prevent duplicate work.

---

## 11. Visual Polish Roadmap [Currently In Progress – GPT-5 Codex • ChartreusePond 2025-10-31]

1. **Lighting & Reflections**  
   - Spawn per-chunk `ReflectionProbeBundle` entities and rely on the 0.16 clustered renderer so probes update only when the terrain chunk actually changes; use the probe visualization pass to tune extents before baking cubemaps. citeturn6search0  
   - Switch the pipeline to `ClusteredForwardBuffer` with HDR enabled and expose white balance / exposure offsets via `AutoExposureSettings`, letting designers stack biome tweaks on top of 0.14’s filmic color grading tools. citeturn16search0turn9search6turn9search5  
   - Plan a later polish pass to adopt 0.17’s hardware upscalers and ray-traced area lights so cinematic captures stay performant even during dense battles. citeturn14search2

2. **Tone Mapping & Post FX**  
   - Wire runtime toggles for ACES, AgX, and TonyMcMapface through the `Tonemapping` component and ship LUT assets alongside the renderer config so QA can validate output against the official example. citeturn9search3turn9search5  
   - Add the `AutoExposurePlugin` with per-camera speed and compensation overrides so storms and day/night transitions smoothly rebalance exposure instead of popping. citeturn16search0turn9search6turn9search0
   - _[In Progress – ChartreusePond 2025-10-31]_ HUD now exposes ACES/AgX/Tony toggles driven by a shared `TonemappingState`, and the Bevy app registers `AutoExposurePlugin` so we can attach or remove `AutoExposure` per camera in response to UI events while we flesh out config + offscreen parity. citeturn0search4turn0search3turn1search0turn1search1

3. **Atmosphere & Fog**  
   - Use the volumetric fog volumes added in 0.15 to author biome-specific haze profiles (valley mist, alpine thin air) with density/height curves driven by terrain metadata. citeturn15search0  
   - Feed humidity and time-of-day scalars into the upgraded 0.16 atmospheric scattering pipeline so sky colors, god rays, and decals stay coherent with weather swings. citeturn6search0turn14search4

4. **Terrain Shading**  
   - Offload heightfield meshing to the GPU-driven renderer introduced in 0.16 so chunk rebuilds avoid CPU stalls and can emit slope/curvature masks straight into materials. citeturn6search0  
   - Evaluate the experimental `MeshletPlugin` for near-camera refinement so steep ridges stay sharp while distant tiles remain coarse; fall back gracefully on hardware without meshlet support. citeturn14search0

5. **Particles & Ambient FX**  
   - Author Hanabi graph assets for biome ambience, weather streaks, and combat sparks, leaning on GPU emitters and hot reload to iterate quickly. citeturn16search5  
   - Bind sky luminance and sun color into particle materials so fog motes, pollen, and trails inherit the same grading as the atmospheric pipeline. citeturn6search0turn14search4

6. **Camera Polish**  
   - Ship preset rigs (orbit, inspector, cinematic) that adjust tonemapping, depth of field, and screenshot resolution; rely on 0.17’s upscalers to keep low-light shots crisp when slowing shutters for motion blur. citeturn14search2turn9search3

7. **Roadmap & CI Hooks**  
   - Track the material graph and GI roadmap highlighted for post-0.16 upgrades and schedule a consolidation sprint once the node authoring tools stabilize. citeturn6search0turn14search2  
   - Extend the snapshot harness to capture HDR EXR frames plus luminance histograms so tone mapping regressions fail CI automatically.

### Terrain Realism Enhancements

1. **Procedural Base Geometry**  
   - Layer continental, ridge, and detail noise while offloading chunk meshing to the GPU-driven renderer added in 0.16 so rebuilds stay responsive at large map sizes. citeturn6search0  
   - Bake slope and curvature textures per chunk and feed them into decal-aware materials so erosion streaks and wet soil blend cleanly with the dynamic decal system. citeturn14search4

2. **Adaptive Tessellation & LOD**  
   - Prototype near-camera refinement with the experimental meshlet renderer so close-up shots keep smooth silhouettes without exploding vertex counts. citeturn14search0

3. **Biome-aware Materials**  
   - Drive albedo, roughness, and emission from the baked masks above and retune them against the improved ambient occlusion pipeline in 0.15 to keep soils and rock faces legible. citeturn15search0

4. **Runtime Detail**  
   - Scatter instanced props and decals using the 0.16 decal pipeline so puddles, footprints, and debris respond instantly to agent interactions. citeturn14search4

5. **Tooling & Validation**  
   - Extend the terrain debug overlay to visualize noise layers and mask values, then snapshot PNG heightmaps plus histogram diffs in CI to catch quality regressions early.

### Agent & Effect Styling

1. **Agent Mesh & Animation**  
   - Replace placeholder spheres with rigged glTF characters and tap into the generalized animation graph plus additive blending improvements from 0.15 for expressive idle/walk/attack mixes. citeturn15search0  
   - Expose emissive streaks and color grading per agent using the filmic color tools from 0.14 so health and energy cues stay legible under dramatic lighting. citeturn16search0

2. **Vision Cones & Sensors**  
   - Render 3D wedges with transparent materials and project warning decals onto the terrain via the 0.16 decal system so threat zones remain visible even when agents are occluded. citeturn14search4

3. **Spike Attacks**  
   - Drive spline-based spike meshes with the new animation masks from 0.15 and trigger Hanabi bursts for sparks or biofluids so impacts feel weighty. citeturn15search0turn16search5

4. **Agent Trails & Interaction FX**  
   - Drop screen-space decals for footprints and slide marks, blending them with moisture masks so tracks darken after rain before fading out. citeturn14search4

5. **Tooling & Iteration**  
   - Build a Bevy playground scene dedicated to agent styling where animation states, materials, and particle presets can be hot-swapped, feeding updated captures into the snapshot harness for regression coverage.

### Weather, Hydrology & Behavioral FX

1. **Dynamic Weather System**  
   - Implement a `WeatherState` resource that lerps directional light, sky tint, and volumetric fog settings using 0.14’s auto exposure tools so exposure ramps stay smooth across storms and sunsets. citeturn16search0turn9search0turn9search6  
   - Author Hanabi emitters for rain streaks, snowflakes, and sandstorms that toggle per state and inherit tint/alpha from the atmospheric parameters. citeturn16search5turn6search0  
   - Adjust terrain material roughness and spawn puddle decals with the 0.16 decal system to visualize wetness build-up and evaporation across the map. citeturn14search4

2. **Hydrology-Driven Visuals**  
   - Keep river and lake meshes on the GPU-driven path so they pick up the same lighting pipeline and can receive decals for foam or pollution streaks as hydrology changes. citeturn6search0turn14search4  
   - Trigger Hanabi splash rings and ripples whenever agents or spikes interact with water volumes, scaling emission strength by impact energy. citeturn16search5

3. **Altitude Feedback**  
   - Map altitude bands to fog density and color shifts using the fog volume controls introduced in 0.15 so valleys feel dense while peaks stay crisp. citeturn15search0

4. **Behavioral Cues**  
   - Altruism: pulse emissive trims and spawn soft particle beams between participants using Hanabi’s GPU emitters for lightweight links. citeturn16search5  
   - Hunting: switch vision cone decals to aggressive hues and emit dust trails or sparks tied to the decal/particle pipeline so pursuits feel kinetic. citeturn14search4turn16search5  
   - Herbivory: animate vegetation materials with filmic color tweaks and subtle particle bursts to highlight energy gain moments without obscuring the scene. citeturn16search0turn16search5

5. **Tooling & Validation**  
   - Extend the debug overlay with weather/hydrology panels (precipitation intensity, surface wetness, flow direction) and capture multi-frame HDR sequences per behavior so CI can detect visual regressions in exposure, fog, and particle density.

### Visual Polish Execution TODO [Currently In Progress - GPT-5 Codex 2025-10-31]

#### Render Polish Breakdown (2025-10-31 – WhiteCastle)

| Bucket | Deliverables | Owner | Acceptance Checks |
| --- | --- | --- | --- |
| Lighting presets | DLSS/FSR3 toggle prototype, ray-traced area light preset, reflection probe bake workflow update | WhiteCastle (primary), OrangeCreek (secondary) | Frame captures at 1080p/4K showing preset delta, perf trace with <10% frame-time regression, documented toggles in `render.config.toml`. |
| Tone mapping & exposure | ACES/AgX/TonyMcMapface UI toggles, per-camera auto-exposure curves, config surface parity with GPUI | WhiteCastle (primary), OrangeCreek (secondary) | Snapshot harness diff within tolerance for default curve, QA checklist covering dark/bright scenes, config docs updated. |
| Atmosphere & fog | Biome fog/sky profiles using volumetric volumes + time-of-day hooks | WhiteCastle (primary), BrownSnow (perf sanity) | 1080p/1440p captures across three biomes, perf counters stable (<5% frame-time delta), HUD readability unaffected. |
| Terrain shading/materials | GPU-driven heightfield meshing port, slope/curvature/moisture texture bake, decal overlays | WhiteCastle (primary), BrownSnow (secondary) | Mesh rebuild latency under target, material screenshots before/after, automated test covering asset loading. |
| Particles & FX | Hanabi library (ambient, weather, combat), tint sync with atmospheric data | WhiteCastle (primary), OrangeCreek (secondary) | Particle presets catalogued, toggle to disable for perf testing, snapshot comparisons logged. |
| Camera rigs | Orbit/inspector/cinematic rigs, screenshot presets, QA checklist | WhiteCastle (primary), OrangeCreek (secondary) | `docs/rendering_reference/camera_checklist.md` updated, snapshot harness generates per-rig outputs, user shortcuts documented. |
| Agent styling | Rigged glTF import, animation graph states, decal-driven vision cones | WhiteCastle (primary), BrownSnow (secondary for perf) | Animation coverage tests, GPU profiler capture verifying batching targets met, design review sign-off. |
| Spike FX | Spline spike meshes, Hanabi impact bursts, decal scorch marks | WhiteCastle (primary), OrangeCreek (secondary) | Visual diff vs GPUI references, automated sanity test ensuring assets load, perf impact documented. |
| Weather & hydrology | `WeatherState` resource, puddle decals, splash particles tied to hydrology | WhiteCastle (primary), BrownSnow (perf secondary) | Sim harness scenario logging weather transitions, screencaps across weather states, memory usage plotted over 30-minute soak. |
| Debug tooling | Overlay extensions (weather, hydrology, behavior), HDR/PNG snapshot automation | WhiteCastle (primary), OrangeCreek (secondary) | Debug panels accessible via shortcut, CI job green with new HDR diff, documentation added to plan. |

Reference: `docs/rendering_reference/bevy_polish_plan.md` for extended notes and meeting cadence.

#### Coordination Notes – 2025-10-31 19:45 UTC Huddle [Pending – WhiteCastle]
- Agenda: hardware access status, automation/parity progress, render-polish milestone sequencing, benchmark execution order.
- Attendees: BrownSnow (confirm via Agent Mail #31), WhiteCastle (agenda owner), OrangeCreek (reminder sent via Agent Mail #31).
- Action log: _to be populated post-huddle._

- [x] Lighting: instantiate chunk-scoped reflection probes on the 0.16 clustered path and publish the bake/refresh playbook.  _[Completed – BrownCreek 2025-10-31: per-chunk `LightProbe` + `EnvironmentMapLight` spawning in `sync_terrain`, configurable via `ReflectionProbeAssets`]_  
- [ ] Lighting: prototype 0.17 DLSS/FSR3 and ray-traced area lights for cinematic capture presets.  
- [ ] Tone mapping: add ACES/AgX/TonyMcMapface toggles plus per-camera auto exposure curves wired to `render.config.toml`.  _[In Progress – ChartreusePond 2025-10-31: UI buttons + `TonemappingState` + `AutoExposurePlugin` landed in `scriptbots-bevy`; config surface + offscreen parity still pending]_
- [ ] Atmosphere: author biome fog/sky profiles using 0.15 volumetric volumes and humidity/time-of-day hooks.  
- [ ] Terrain shading: port heightfield meshing to the GPU-driven renderer and stage meshlet-based close-up refinement.  
- [ ] Terrain materials: bake slope/curvature/moisture textures and integrate the 0.16 decal system for erosion/wetness overlays.  
- [ ] Particles: curate Hanabi graph library for ambience, weather, combat, and sync tint with atmospheric data.  
- [ ] Camera: deliver orbit/inspector/cinematic rigs with snapshot presets and QA checklist.  
- [ ] Agent styling: import rigged glTF characters, configure animation graph states, and layer decal-driven vision cones.  
- [ ] Spike FX: build spline-based spike meshes and pair them with Hanabi impact bursts plus decal scorch marks.  
- [ ] Weather/hydrology: implement `WeatherState` resource, puddle decals, and splash particles tied to hydrology events.  
- [ ] Tooling: extend debug overlay (weather, hydrology, behavior) and automate HDR/PNG snapshot comparisons in CI.

#### Reflection Probe Bake & Refresh Playbook (2025-10-31 – BrownCreek)

1. **Runtime topology** – `sync_terrain` now registers a `LightProbe` + `EnvironmentMapLight` entity for every WFC terrain chunk, reusing shared handles from `ReflectionProbeAssets`. Probes inherit chunk bounds (width/depth from cell size, height capped at `max(chunk_height, 20.0)`) and refresh automatically whenever chunk signatures change.
2. **Baking cubemaps**  
   a. Capture or author HDR cubemaps for your biome (`diffuse` + prefiltered `specular`).  
   b. Place assets under `assets/lighting/` (e.g., `biome_sunrise_diffuse.ktx2`, `biome_sunrise_specular.ktx2`).  
   c. Extend `setup_scene` (temporary) or a forthcoming loader to swap `ReflectionProbeAssets` handles to the new textures; hot-reload will update all chunk probes on the next frame.
3. **Refreshing after terrain edits** – Set `SB_PROBES_REBUILD=1` (env var) or call the forthcoming `probe_refresh` console command to force chunk probes to respawn after major heightfield edits; otherwise signature deltas trigger rebuilds automatically.
4. **Validation checklist**  
   - Use `bevy_inspector` / `render_debug probes` (planned) to visualize probe volumes and confirm coverage.  
   - Run `cargo test -p scriptbots-bevy -- --nocapture probe_parity` once the benchmark harness lands to ensure chunk counts match terrain chunk registry.  
   - Capture before/after screenshots and attach to `docs/perf/bevy_vs_gpui.md` to document reflection improvements.
5. **Next steps** – Wire an asset-driven loader (TBD) so designers can select cubemaps per biome without code edits, and integrate probe coverage visualizations into the upcoming debug overlay task.
