# Plan to Integrate Bevy Engine into ScriptBots

_Prepared by RedSnow — 2025-10-30_

> Goal: Introduce an opt-in Bevy-powered 3D/2.5D renderer (camera + world + HUD) that can coexist with the current GPUI renderer, controlled by feature flags and runtime switches, without disrupting existing workflows.

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
| 0 [Currently In Progress – RedSnow 2025-10-30] | Scaffolding | New crate, feature flag, minimal Bevy app that opens window and clears background. | `cargo run --features bevy_render --renderer=bevy` opens blank window. |
| 1 | Static world visuals | Render terrain, agents as instanced meshes/quads, static camera. | Snapshot harness for Bevy path produces comparable frame to GPUI. |
| 2 | Camera controls | Orbit + follow modes mapped, input parity with GPUI (mouse, keyboard). | QA sign-off that camera UX matches spec. |
| 3 | HUD parity | Overlay tick stats, controls, selection info. | HUD shows same metrics as GPUI reference screenshot. |
| 4 | Interactivity | Agent selection, follow toggles, command buttons. | Round-trip commands (select agent) confirmed via simulation logs. |
| 5 | Polish + QA | Performance tuning, lighting, debug overlays, CI integration. | Bevy path passes `render_regression` job + manual smoke checklist. |

- Progress (2025-10-30 – RedSnow): Scaffolded `scriptbots-bevy` crate, workspace feature flag, CLI `--renderer=bevy`, and stub window launcher. Pending manual runtime verification before marking Phase 0 complete.

---

## 6. Testing & Regression Strategy

6.1 **Snapshot Harness Extension**
   - Add test target `cargo test -p scriptbots-render --features bevy_render -- --nocapture` capturing Bevy screenshot via headless mode (use `WGPU_BACKEND=gl` for CI).  
   - Compare output PNG to new golden `docs/rendering_reference/golden/bevy_default.png`.

6.2 **Unit Tests**
   - `scriptbots-bevy` crate: test conversion helpers (`RenderFrame` → `BevyAgentBundle`).  
   - Camera controller: property tests ensuring orbit/follow maintains agent in frame.

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
   - Extend `run_*` scripts with `--renderer` parameter.  
   - Provide `run_linux_with_bevy.sh` helper enabling relevant environment flags.

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

- [ ] Review plan with PinkMountain, PurpleBear, RedCastle; capture feedback in coordination log.  
- [ ] Approve Bevy version & dependency policy.  
- [ ] Kick off Phase 0 scaffold (branch `feature/bevy-integration-phase0`).  
- [ ] Prepare CI job definitions (`ci/bevy_render.yml`).  
- [ ] Establish visual parity acceptance criteria with design/QA.

Once phases progress, update this document inline with `[In Progress – <Name>]` markers to prevent duplicate work.
