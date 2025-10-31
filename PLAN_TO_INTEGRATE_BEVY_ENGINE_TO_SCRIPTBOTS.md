# Plan to Integrate Bevy Engine into ScriptBots

_Prepared by RedSnow — 2025-10-30_

> Goal: Introduce an opt-in Bevy-powered 3D/2.5D renderer (camera + world + HUD) that can coexist with the current GPUI renderer, controlled by feature flags and runtime switches, without disrupting existing workflows.

---

### Phase 4 Working TODO — OrangeLake (2025-10-31)

- [ ] Add `ControlCommand::UpdateSimulation` + `SimulationCommand` plumbing in `scriptbots-core`.
    - [ ] Extend `WorldState` with pending simulation control requests + helpers.
    - [ ] Update `apply_control_command` and add regression test.
- [ ] Wire GPUI renderer (`scriptbots-render`) to emit simulation commands for play/pause/step/speed.
- [ ] Teach terminal renderer to honour queued simulation commands before stepping.
- [ ] Update Bevy playback handlers to submit `SimulationCommand` and sync with queued requests.
- [ ] Enhance Bevy simulation driver to merge queued commands into live control and respect auto-pause reason.
- [ ] Refresh Bevy HUD styling to reflect terrain relief palette (drop shadow, accent borders).
- [ ] Document command contract + styling adjustments in phase plan & coordination log and notify RedSnow.
- [ ] Run `cargo check` for workspace & targeted tests (`scriptbots-bevy`, `scriptbots-core`).

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

---

## 11. Visual Polish Roadmap

1. **Lighting & Reflections**  
   - Add reflection probes per zone (spawned from terrain chunks), bake environment maps, and iterate with probe debug draws before locking values. citeturn0search1  
   - Adopt the forward+ pipeline and HDR backdrop improvements landing in Bevy 0.15, keeping exposure and white-point tunable in config. citeturn0search0

2. **Tone Mapping & Post FX**  
   - Enable the new auto-exposure + ACES tone mapper; surface toggles for bloom, chromatic aberration, motion blur, and film grain in `render.config.toml`. citeturn0search4turn0search8  
   - Document HDR workflow (fall back to SDR paths for CI) and add color grading LUT slots for cinematic presets.

3. **Atmosphere & Fog**  
   - Integrate Bevy’s physical sky plugin for sun/sky scattering, hook into biome metadata (humidity, time-of-day), and layer height fog to reinforce depth. citeturn0search2turn0search9

4. **Terrain Shading**  
   - Extend chunk materials with triplanar detail maps, parallax occlusion, and layered noise LUTs to differentiate rock, sand, and moss; drive accents via GPU-computed slope/curvature masks. citeturn0search0  
   - Pre-warm sampler states so terrain responds to dynamic daylight/sky color changes without shimmer.

5. **Particles & Ambient FX**  
   - Use Hanabi GPU particles for biome ambience (pollen, dust motes, fireflies) with emission budgets tied to chunk visibility. citeturn0search6  
   - Feed particle color/alpha from the sky’s sun disk for cohesive lighting.

6. **Camera Polish**  
   - Add per-mode camera LUTs, depth-of-field targets, and screenshot presets (wider FOV, eased motion).  
   - Track TAA/FXAA upstream and plan integration once Bevy exposes official graph nodes (reduces terrain shimmer in distant shots). citeturn0search8

7. **Roadmap & CI Hooks**  
   - Monitor upcoming material nodegraph & global illumination work; schedule an upgrade pass post 0.15 once tools stabilize. citeturn0search8  
   - Extend snapshot harness: capture HDR EXR outputs, compute histogram/contrast deltas, and gate merges on acceptable drift.

### Terrain Realism Enhancements

1. **Procedural Base Geometry**  
   - Layer multi-octave Perlin/FastNoise height fields (continental, ridge, fine detail) with domain warping to break repetition; expose noise seeds per biome for WFC compatibility.  
   - Run GPU-friendly hydraulic erosion passes (thermal creep + river carving) on height buffers, following techniques from erosion-focused terrain articles, then bake results into chunk signatures. citeturn0search2turn0search3

2. **Adaptive Tessellation & LOD**  
   - Evaluate tessellated terrain (compute-driven or meshlet LOD) to smooth silhouettes when the camera dives low; keep chunk meshes coarse for distance, dense near focus. citeturn0search4

3. **Biome-aware Materials**  
   - Drive diffuse/roughness/normal blends from procedural masks: slope (rock), concavity (soil accumulation), moisture/erosion (mud), altitude (snow).  
   - Use precomputed moisture and erosion maps to feed moss and shoreline foam shaders; optionally generate AO/curvature maps offline for static chunks. citeturn0search2

4. **Runtime Detail**  
   - Scatter procedural props (stones, shrubs) via noise-based placement tied to chunk signatures; leverage instancing to stay GPU-bound.  
   - Integrate surface-aware decals (wet patches, trails) anchored to agent interactions or scripted events. citeturn0search7

5. **Tooling & Validation**  
   - Build a terrain debug overlay showing noise layers, erosion strength, and biome weights to iterate quickly.  
   - Capture before/after snapshots (PNG + heightmap diff) in CI to prevent accidental regressions in terrain quality.

### Agent & Effect Styling

1. **Agent Mesh & Animation**  
   - Replace sphere proxies with rigged glTF characters and drive idle/walk/attack blends through Bevy’s skinned mesh animation graph for expressive motion. citeturn0search9  
   - Layer toon/outline passes (`bevy_toon_shader`, `bevy_outline`) so selected agents pop with stylized rims and team hues. citeturn0search12  
   - Add emissive pulses or armor streaks keyed to energy/selection via `LinearRgba` color transitions (Bevy 0.14+). citeturn0search13

2. **Vision Cones & Sensors**  
   - Render vision arcs as translucent volumetric wedges with gradient falloff and animated scan lines; fade to wireframe on occlusion. citeturn0search10  
   - Use clustered decals to project warning textures where cones touch terrain and trigger ripple particles on detection events. citeturn0search10

3. **Spike Attacks**  
   - Model spikes as skinned meshes or spline ribbons that extend/retract with anticipation/recoil clips; add motion blur trails via post-processing. citeturn0search11  
   - Trigger Hanabi sparks, biofluid particles, and scorch/blood decals on impact to leave readable footprints. citeturn0search6turn0search11

4. **Agent Trails & Interaction FX**  
   - Lay down decal footprints/tracks via `bevy_decal` or custom instanced quads, fading over time.  
   - Spawn soft-ground deformation by blending parallax detail textures where agents linger.  
   - Hook up audio-reactive VFX (waveform-driven shader parameters) for spikes or vision lock-ons.

5. **Tooling & Iteration**  
   - Create an agent styling playground scene to tweak materials, animations, and FX live; integrate with the snapshot harness for regression coverage.  
   - Document the import pipeline (glTF authoring, animation naming, shader param conventions) so new meshes drop in without code changes.

### Weather, Hydrology & Behavioral FX

1. **Dynamic Weather System**  
   - Implement a `WeatherState` resource with variants (Clear, Overcast, Rain, Snow, Sandstorm). Each tick, lerp ambient light, directional-light color/intensity, skybox, and fog using `DistanceFog`/`VolumetricFog` components on the active cameras (supports exponential/linear falloffs and volumetric shafts). citeturn0search0turn0search1  
   - Attach Hanabi particle effect graphs per state (rain streaks via cylinder spawn + velocity, snowflakes via plane spawn drift, sand via cone spawn) and toggle them by activating/deactivating effect assets. Track precipitation intensity with a scalar that also influences fog density and cloud opacity. citeturn0search5turn1search7  
   - Maintain `SurfaceWetness` per terrain chunk: when precipitation starts, increment toward 1; when dry, decay toward 0. Use this to modify material roughness/metallic and to enable clustered decals for puddles or snow accumulation textures. citeturn0search0turn1search1  
   - Capture HDR frame sets for weather regression (storm vs clear) and compare histograms, ensuring tone mapping still lands in the intended range.

2. **Hydrology-Driven Visuals**  
   - Integrate the `bevy_water` material for large bodies of water; feed it the simulation wave spectrum and call `get_wave_point` so boats/agents ride the surface height. Extend the shader with shoreline foam, depth-based coloration, and projected caustics textures sampled through clustered decals. citeturn1search0turn1search3turn1search1  
   - When the hydrology system raises/lowers water tables, adjust terrain chunk materials (darken, add specular highlights) and spawn edge decals for waterlines. Use `bevy_water` mask support to remove water tiles in drained regions. citeturn1search0  
   - Trigger ripple Hanabi effects at contact points (agent feet, spike hits) using plane emitters with radial velocity; synchronize with splash audio and temporary refractive screen-space shaders for impact moments. citeturn0search5  
   - Color vegetation shaders with flow/erosion metrics so plants near fast water appear saturated and lowlands pick up silt tones.

3. **Altitude Feedback**  
   - Blend snowcaps, moss, or desaturated palettes above configurable altitude bands; modulate atmospheric scattering and fog height for valleys vs peaks. citeturn0search2turn0search7  
   - Expose altitude thresholds in config so biome designers can retune visuals without code.

4. **Behavioral Cues**  
   - Altruism: emit soft bilateral beams using Hanabi cone emitters between participants, pulse emissive armor rims via `LinearRgba` tweening, and project icon decals on the ground for readability. citeturn1search2turn0search5turn1search1  
   - Hunting: swap vision cones to red volumetric wedges with time-based scan-line shaders; when lock-on triggers, spawn dust Hanabi trails and, on impact, place scorch/blood decals plus spike spark effects. citeturn0search10turn0search6turn1search1  
   - Herbivory: animate vegetation bending through morph targets, spawn pollen clouds (Hanabi sphere spawn with slow upward drift), and run a green energy ring shader around the agent; tie regrowth animation to the simulation timer. citeturn0search5turn1search1  
   - Expose hooks for audio-reactive shader parameters so dramatic cues (spikes, focus) can sync with soundtrack amplitude.

5. **Tooling & Validation**  
   - Extend the debug overlay with weather/hydrology layers (humidity, rainfall timers, flow vectors) plus live behavior markers.  
   - Add snapshot sequences for each weather state and major behavior (altruism, hunting, eating) to CI, comparing histogram/contrast drift across episodes.
