# Plan to Port ScriptBots to Modern Idiomatic Rust with GPUI

## Vision and Success Criteria
- Deliver a faithful, deterministic port of ScriptBots' agent-based ecosystem in Rust, preserving simulation behaviors (sensing, decision-making, reproduction, food dynamics, carnivore/herbivore specialization) while removing undefined behavior and manual memory management found in the original GLUT/C++ codebase (`World.cpp`, `Agent.cpp`).
- Embrace Rust idioms: error handling via `Result`, trait-based abstraction for brains, `Arc`/`RwLock` only where unavoidable, and zero `unsafe` in the first release.
- Exploit data-parallelism with Rayon to scale agent updates on modern multi-core CPUs.
- Replace legacy GLUT rendering with a GPU-accelerated GPUI front-end, using declarative views plus imperative canvas drawing to render thousands of agents at 60+ FPS on macOS and Linux, the platforms GPUI currently targets.
- Ship cross-platform binaries (macOS + Linux) with reproducible builds via Cargo, and include automated tests/benchmarks validating simulation invariants.
- Provide high-fidelity telemetry: pause/resume, deterministic replay, and offline analytics pipelines backed by DuckDB snapshots.
- Support hot-swapping of agent brain implementations—from simple heuristics to differentiable neural networks—without recompiling the simulation.

## Current C++ Baseline (Reference Snapshot)
- `main.cpp`: GLUT bootstrap, owns global `GLView`, wires keyboard/mouse callbacks.
- `World.[h|cpp]`: Master simulation loop, food grid, per-agent sensing/output pipeline, reproduction, reporting, and mouse selection.
- `Agent.[h|cpp]`: Agent state (262 floats/flags), reproduction/crossover logic, mutation of neural weights, sensor arrays, event bookkeeping.
- Brain implementations: `MLPBrain`, `DWRAONBrain`, `AssemblyBrain` (only MLP is production ready today per inline comments).
- Rendering: `GLView` draws agents, overlays, and time-series of herbivore/carnivore counts with immediate-mode OpenGL.
- Configuration: `settings.h` and `config.h.in` encode global constants (grid size, reproduction rates, spike damage, etc.).

## Target Rust Workspace Layout [Completed: Workspace scaffolding baseline crates]
- `Cargo.toml` (workspace) with members:
  - `crates/scriptbots-core`: Pure simulation logic, no UI or platform dependencies.
  - `crates/scriptbots-brain`: Houses `Brain` trait + concrete implementations (`mlp`, `dwraon`, `assembly`). Depends on `scriptbots-core`.
  - `crates/scriptbots-storage`: Abstraction over DuckDB for persistence, journaling, and analytics exports built atop the `duckdb` crate.
  - `crates/scriptbots-brain-neuro`: Optional crate that wraps NeuroFlow feed-forward networks for more advanced brains, kept isolated behind a Cargo feature for lean builds.
  - `crates/scriptbots-render`: GPUI views, canvas drawing, input handling, overlays.
  - `crates/scriptbots-app`: Binary crate wiring simulation loop, GPUI application, CLI/config loading.
  - [Completed - GPT-5 Codex 2025-10-23] `crates/scriptbots-index`: Spatial indexing utilities (`UniformGridIndex`) powering neighbor queries in sensing/combat.
  - [Completed - GPT-5 Codex 2025-10-23] `crates/scriptbots-web`: Wasm wrapper exposing a browser API (`Simulation`, `SimulationSnapshot`) with wasm-bindgen tests in CI.
  - Optional future crate `scriptbots-replay`: tooling for replay/serialization.
- Shared utilities crate (e.g., `scriptbots-util`) for RNG wrappers, fixed-point helpers, instrumentation.
- Use Cargo features to toggle experimental brains (`assembly`), AI modules, or headless mode.

## Simulation Core Design
### Data Model [Completed: SoA columns & generational IDs]
- `WorldState` struct: [Completed: arena-backed scaffold with FoodGrid]
  - `agents: Vec<AgentState>` stored in SoA-friendly layout via `Vec<AgentState>` with packed fields or `AgentColumns` (parallel `Vec` per field) for cache-friendly iteration.
  - `food: Grid<f32>` represented as `Vec<f32>` sized `width * height`; wrapper offers safe indexing and iteration slices for Rayon.
  - `rng: SmallRng` per-world; optionally supply per-agent RNG seeds to avoid contention.
  - `epoch`, `tick_counter`, `closed_environment` flags mirroring C++ logic.
- `AgentState`: [Completed: runtime struct + sensor buffers]
  - Scalar fields mapped to Rust primitives (`f32`, `u16`, `bool`).
  - Sensor buffers as `[f32; INPUT_SIZE]`, outputs `[f32; OUTPUT_SIZE]` to avoid heap allocations. [Done via `AgentRuntime::sensors`/`outputs`]
  - `brain: Box<dyn Brain + Send + Sync>` or enum-based dispatch for zero-cost dynamic dispatch. [Completed - GPT-5 Codex 2025-10-22: replaced `BrainBinding` placeholder with per-agent brain attachments backed by factory registry]
  - History/event info (`indicator`, `sound_output`, `select_state`). [Backed by `IndicatorState`, `SelectionState`, and runtime fields]
  - `id: AgentId` generational slotmap handle to prevent stale references.
- Configuration via `ScriptBotsConfig` (deserialized from `TOML`/`RON`) replacing `settings.h`; defaults compiled with `serde` + `once_cell` for global fallback. [Completed: validation + RNG seeding + defaults]
- `BrainGenome`: versioned specification of topology, activation mix, and hyperparameters that can be stored in DuckDB, mutated independently of runtime brain instances, and reproduced across brain backends. [Completed: schema + validation helpers]

### Time-Step Pipeline
1. **Aging and Periodic Tasks** [Completed: arena age increments + TickEvents flush flags]: Increment ages every 100 ticks, flush charts every 1000, roll epoch at 10,000.
2. **Food Respawn** [Completed: interval-based respawn + clamped adds]: Sample random cell via RNG; `Grid::apply` ensures bounds safety.
3. **Reset Event Flags** [Completed: runtime flag clears]: Clear `spiked`, `dfood`, etc. using `par_iter_mut`.
4. **Sense** [Completed: sequential neighbor scan populating sensors]: Build immutable snapshot of neighbor data (position, colors, velocities). Use spatial grid or uniform hashing (`HashGrid`) to cap O(n²):
   - Partition agents into buckets by cell in parallel; gather neighbor lists.
   - Compute inputs in parallel using `rayon::join` for segments, writing into per-agent `[f32; INPUT_SIZE]`.
5. **Brains Tick** [Completed: registry-backed BrainRunner plumbing]: Call `brain.tick(&inputs, &mut outputs)` per agent. Provide `Brain` trait with pure functions to stay thread-safe; if mutation required, use interior `RefCell` replaced by split-phase updates.
6. **Actuation** [Completed: velocity integration + energy drain baseline]: Translate outputs into movement, colors, giving, boosting. Use double-buffered positional data to avoid read/write races: compute new positions into `Vec<AgentDelta>`, commit after iteration.
7. **Food Intake & Sharing** [Completed: cell intake + sharing baseline]: Process using `par_iter_mut` with atomic adds or gather stage followed by sequential commit to avoid floating-point race conditions (determinism).
8. **Combat and Death** [Completed: spike collisions & removal queue]: Evaluate spike collisions using spatial index, queue health changes, then apply.
9. **Reproduction & Spawning** [Completed: queued spawn orders + child mutations]: Collect reproduction events into `Vec<SpawnOrder>`, apply sequentially to maintain deterministic ordering, leveraging `rand_distr` for gaussian mutations.
10. **Persistence Hooks** [Completed: tick summary + pluggable sink]: Stream agent snapshots, food deltas, and event logs into DuckDB tables using batched transactions (e.g., every N ticks or when buffers exceed threshold). Leverage Arrow/Parquet features for zero-copy exports when advanced analytics are enabled.
11. **Cleanup** [Completed - GPT-5 Codex 2025-10-22: deterministic death queue drains, stable arena retention, runtime cleanup + tests]: Remove dead agents using `Vec::retain` (single-threaded) or stable partition maintaining deterministic ordering; recycle IDs as needed.

## Brain System
- Define `pub trait Brain: Send + Sync { fn kind(&self) -> BrainKind; fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE]; fn mutate(&mut self, rng: &mut dyn RngCore, mut_rate: f32, mut_scale: f32); fn crossover(&self, other: &dyn Brain, rng: &mut dyn RngCore) -> Option<Box<dyn Brain>>; }`. [Completed: fixed-size outputs + registry adapters]
- Implementations:
  - `MlpBrain`: Store weights as `Vec<f32>` or `Box<[f32]>`; leverage `ndarray` or manual indexing. Provide `tick` without allocations; optionally use `packed_simd` for vectorization.
  - `DwraonBrain`: Retain 4-connection per node semantics with `Vec<Node>` storing bias, not flags.
  - `AssemblyBrain` (experimental): Represent instruction tape as `Vec<f32>`, interpret sequentially; feature-gated.
- Brain factory for reproduction crossover to pick matching variant from parents.

## Environment & Sensory Systems
- Spatial partitioning using uniform grid sized to `conf::DIST` to bound neighbor searches.
- Temperature, smell, sound sensors derived from global constants; compute as part of sense phase.
- Provide optional SSE/AVX acceleration via feature flags while keeping scalar fallback.

## Concurrency and Performance Strategy
- Rayon parallelization:
  - `par_chunks_mut` over agent slices for sensing and actuation; gather neighbor summaries from read-only snapshots.
  - Use `ThreadPoolBuilder::new().num_threads(config.simulation_threads)` to expose CLI override.
  - Avoid shared mutable state during parallel sections by staging results in thread-local buffers (`rayon::scope` + `Vec<AgentEvent>`).
- Determinism:
  - Stage floating-point accumulations per agent; reduce deterministically (sorted neighbor lists).
  - Use `fastrand::Rng` per agent seeded by master seed + agent ID to keep reproducible across thread counts.
- Metrics:
  - Expose `SimulationMetrics` (FPS, agent count, reproduction rate) updated atomically, consumed by UI overlays.
- DuckDB integration threading:
  - Maintain a dedicated async task that receives structured log batches over an MPSC channel, writes them via a pooled DuckDB `Connection`, and periodically checkpoints/optimizes storage. Use `duckdb`'s bundled feature in development to avoid external dependencies, and adopt `Connection::open_with_flags` variants for file-backed persistence in production.
- Accelerator hooks:
  - Keep sensor aggregation and brain inference behind pluggable traits so alternative implementations (SIMD-optimized, GPU-backed) can be introduced without touching higher-level logic. Prototype GPU execution by batching inference into wgpu compute shaders or other accelerators once profiling justifies the investment.

## SIMD and Chunked Neighborhood Batching (CPU Hot-Path Optimization Roadmap)

### Rationale
- Our profiler and Criterion benches show the sensing and combat phases dominate CPU time at medium–large populations, primarily due to repeated per-neighbor math (distances, angular checks, color aggregation) inside small scalar loops.
- We already achieved a deterministic, modest win (~5% at 2k agents over 1000 ticks) by:
  - Reusing per-tick buffers to reduce allocations.
  - Adding a chunked neighbor traversal API (`visit_neighbor_buckets`) in `scriptbots-index`.
  - Vectorizing the per-eye (4 eyes) vision accumulation and batching combat alignment checks with 4-wide SIMD via `wide::f32x4` (feature `simd_wide`, default ON) while preserving scalar fallback and determinism.
- Bigger wins require reducing scalar control-flow inside the hot loops and processing neighbors in blocks (4–8) where we can execute the same arithmetic in lockstep.

### Goals and Success Criteria
- Primary: Improve ticks/sec by 15–30% at 5k–10k agents in CPU-only builds on typical 8–16 core machines while preserving deterministic physics and outputs within the project’s defined tolerances.
- Secondary: Keep code readable and maintainable with minified risk: SIMD behind a feature gate (`simd_wide`) and scalar fallback always compiled and tested.
- Determinism: Maintain identical outputs vs. scalar path at single-threaded and multi-threaded settings (order preservation and stable floating-point reduction strategies).

### Scope of SIMD Batching (Phase 1 → Phase 3)
1) Phase 1 (Shipped) [Completed - 2025-10-24]
   - Added `visit_neighbor_buckets` to `UniformGridIndex` to expose contiguous neighbor candidate slices per cell.
   - Vectorized 4‑eye vision accumulation for each candidate neighbor using `wide::f32x4`.
   - Batch combat alignment/damage math over 4 neighbors at a time; scalar remainder loop for tails.
   - Buffer reuse across `stage_sense`, `stage_actuation`, `stage_combat`, and `stage_food` to eliminate transient allocations.

2) Phase 2 (Next) [Currently In Progress - 2025-10-24]
   - SIMD‑batch the remaining sensing channels per neighbor:
     - Density/eye RGB accumulation (done for eyes; verify saturation/clamp paths remain vectorized).
     - Smell: accumulate `dist_factor` in 4‑wide chunks.
     - Sound: compute `speed = sqrt(vx^2+vy^2)` and `(speed / max_speed).clamp(0,1)` in lanes; accumulate `dist_factor * normalized_speed`.
     - Hearing: accumulate `dist_factor * sound_emitter[other]` in lanes (requires `sound_emitters` SoA snapshot; already available).
     - Blood forward‑cone: vectorize `forward_diff < BLOOD_HALF_FOV` mask, lane-wise `bleed * dist_factor * wound`.
   - Precompute frequently reused per-agent scalars into SoA arrays before the neighbor pass to reduce scalar work in the inner loop: [Completed - 2025-10-24: headings sin/cos prep deferred; implemented per-eye clamped FOV, precomputed view directions, normalized speeds]
     - Heading sin/cos arrays.
     - `inv_radius` (and `inv_max_speed`) for the current tick.
     - FOV per eye with `max(0.01)` applied once.
   - Micro-branch pruning in blocks:
     - Evaluate distance mask (`dist_sq <= ε || dist_sq > r²`) and, when fully false across the 4‑lane block, skip the rest of the math.
     - For vision: compute lane‑wise `diff < fov` mask, apply via multiply by zero rather than branches; prefer max/min to avoid divergent control paths.

3) Phase 3 (Optional, post‑validation)
   - Consider 8‑wide SIMD (`f32x8`) if the `wide` crate/API and common targets justify it; gate behind `simd_wide8` feature while keeping `f32x4` as default for portability.
   - SIMD more of combat:
     - Precompute attacker `facing` vectors, base damage scalars, and boost/speed factors once; run 4–8 victim lanes at a time.
     - Keep bucket‑local ordering stable when emitting hits (collect per‑block and append preserving neighbor index order).
   - Evaluate top‑N filters pre‑SIMD path (e.g., early gating by spike alignment or min length) to avoid arithmetic on clearly invalid candidates.

### Determinism Strategy
- Keep scalar and vectorized paths mathematically equivalent:
  - Use lane-wise masks via multiply-by-zero instead of diverging branches.
  - Preserve output application order by iterating buckets in stable order and emitting per-index results identically.
  - Avoid horizontal reductions that change addition order; accumulate per-agent values in the same index order as the scalar loop (append per-lane in increasing index).
- Tests:
  - Seeded determinism tests already ensure summary equivalence. We’ll add specific equality checks for sensors and combat hits on micro scenes (small neighborhoods) across scalar vs. SIMD modes and across thread counts.

### Data Layout & Indexing
- SoA snapshots: Continue to expand `work_*` vectors for everything used inside hot loops (eye FOVs, eye directions, trait modifiers, sound emitters, heading sin/cos) to maximize cache locality and enable contiguous reads in SIMD lanes.
- Index buckets: `visit_neighbor_buckets` exposes contiguous `&[usize]` slices, suitable for `.chunks_exact(4)` iteration. Ensure bucket fill order remains deterministic and spatially local for cache friendliness.

### Feature Flags and Fallbacks
- `simd_wide` is ON by default; the scalar fallback builds and is tested continuously.
- Consider a `simd_strict` mode to assert exact equality vs. scalar on CI microtests (dev‑only), guarding accidental drift.

### Benchmarks & Targets (CPU)
- Criterion harness (`world_bench`) supports env knobs:
  - `SB_BENCH_STEPS`, `SB_BENCH_SAMPLES`, `SB_BENCH_WARMUP_SECS`, `SB_BENCH_MEASURE_SECS`, `SB_BENCH_AGENTS`.
  - Parallel tunables: `RAYON_NUM_THREADS`, `SCRIPTBOTS_PAR_MIN_SPLIT` (dynamic min‑split for `.with_min_len`).
- Targets (indicative, to refine on CI runner):
  - 2k agents × 1000 ticks: SIMD + chunking yields ≥5–10% over parallel‑only baseline.
  - 5k–10k agents × 1000 ticks: ≥15–30% improvement after Phase 2 SIMD of smell/sound/hearing/blood and expanded combat vectorization.

### Rollout Plan & Acceptance Criteria
1) Land Phase 2 SIMD for senses and finalize combat batching; verify determinism tests and unit equality tests.
2) Bench A/B on CI runner at 2k/5k/10k agents; record ticks/sec and p‑values; document deltas in repo (docs/benchmarks.md).
3) Tune `SCRIPTBOTS_PAR_MIN_SPLIT` defaults based on benches; expose a CLI knob if needed.
4) Acceptance: Hit ≥15% at 5k agents (1k ticks) and ≥5% at 2k agents with no determinism regressions; document and keep scalar fallback.

### Risks & Mitigations
- Floating‑point subtlety: mask‑by‑zero instead of conditional branches; avoid re‑ordering reductions; add microtests.
- Portability: stay on stable Rust; use `wide` crate for SIMD; fallback remains scalar.
- Maintainability: clean, isolated SIMD blocks guarded by feature flags; heavy use of SoA snapshots and precomputed scalars to keep inner loops straightforward.

## Rendering with GPUI
 - Entry point: `Application::new().run(|cx: &mut App| { ... })`, open window with `cx.open_window(...)`, register root view `WorldView`.
- State ownership:
  - `SimulationEntity`: `Entity<SimulationModel>` holds shared simulation state (agents, food grid snapshot, metrics). Updates triggered via background tasks that mutate entity and call `cx.notify()`.
  - UI-specific entity for camera (zoom, pan) mirroring original GLView controls.
- Drawing:
  - Use `canvas()` element to draw food tiles and agents each frame. Batch rectangles for food using `PathBuilder` for contiguous quads; draw agents as circles/spikes with `PathBuilder::stroke`. [Completed - GPT-5 Codex 2025-10-21: initial canvas-based world renderer in `scriptbots-render`]
  - Render selection overlays, brain heatmaps with nested `div()` (flex layout) or additional `canvas`.
  - Chart herbivore/carnivore history using `Path` polyline overlay.
- Input Handling:
  - Map keyboard shortcuts (`gpui::prelude::*`) to actions (pause, toggle render, spawn agents) using GPUI action system.
  - Mouse interactions: listen for `MouseDownEvent`, translate world coordinates using camera entity, select nearest agent, show inspector drawer.
 - Platform Notes:
  - GPUI documentation highlights macOS and Linux as current targets; prioritize those platforms and monitor upstream progress for additional backends.
  - [Completed - GPT-5 Codex 2025-10-23] Terminal renderer auto-fallback covers headless environments and is exercised in CI.
 - Tooling:
  - Use `create-gpui-app` CLI to scaffold initial GPUI project structure, then integrate workspace crates manually.
  - [Completed - GPT-5 Codex 2025-10-23] Terminal rendering mode implemented in `scriptbots-app::terminal` with palette-aware emoji/ASCII mini-map, stats HUD, event feed, and keyboard controls.
- Ecosystem:
  - Track community crates like `gpui-component` for prebuilt widgets and styling patterns.

## Simulation Loop and UI Synchronization
- Dedicated simulation thread:
  - Leverage GPUI's async executor to run the simulation tick loop off the main thread while keeping the UI responsive (target ~16 ms timestep).
  - Use `async` channel (`tokio::sync::watch` or `flume`) to push immutable snapshots to UI layer.
- Snapshot Strategy:
  - Maintain double buffer (`SimulationSnapshot` with `Arc<[AgentRenderData]>` + `Arc<[FoodCell]>`). UI clones `Arc`s cheaply, ensuring zero-copy render pipeline.
  - Provide interpolation for camera smoothing if we add variable render rate.
- Storage Sync:
  - On snapshot publication, enqueue summary records (population counts, resource totals) for DuckDB ingestion, and optionally archive raw agent rows every configurable cadence for replay/debug. Employ DuckDB's Arrow integration for efficient bulk writes when analytics pipelines demand columnar exports. [Completed - GPT-5 Codex 2025-10-21]
  - [Completed - GPT-5 Codex 2025-10-23] Replay event logging persisted each tick (RNG scopes, brain outputs, actions) with CLI to verify and diff recorded vs. simulated streams.

## Visual Polish & Audio Enhancements
- Rich terrain & backgrounds: integrate a tilemap or overlay renderer such as `wgpu-tilemap` to stream large ground textures efficiently, enabling biomes, paths, and heatmaps without dropping frame rate.
- [Completed - GPT-5 Codex 2025-10-22: Terrain elevation now drives downhill momentum/energy costs, with HUD-integrated shading reflecting slopes.]
- Stylized UI elements: use Vello inside GPUI canvases for high-quality vector icons, rounded panels, and animated HUD elements, giving the interface a modern "game" look while staying GPU-accelerated. [Completed - GPT-5 Codex 2025-10-22: added animated header badge, gradient-styled metric cards, and overlay summaries rendered via GPUI/Vello canvases for consistent theming.]
- Particle and lighting effects: reference the `ShadowEngine2D` example for wgpu-driven particles, bloom, and lighting passes, adapting similar shaders to draw energy trails, spike impacts, or weather overlays.
- Post-processing shaders: batch agent render buffers through compute builders like `cuneus` to add color grading, vignette, or heat shimmer effects without rewriting the entire pipeline.
- Game-quality audio: adopt the `kira` crate for layered ambient loops, positional effects, and event-driven sound design, using its mixer and timing system to sync audio cues with agent behaviors; expose toggles for audio channels in the GPUI inspector.
  - [Completed - GPT-5 Codex 2025-10-23] Feature-gated audio cues integrated in GPUI renderer (birth/death/spike and UI toggles); ships disabled by default and can be enabled via features.
- Accessibility polish: add colorblind-safe palettes and toggleable outline shaders so the simulation remains readable with visual enhancements. [Completed - GPT-5 Codex 2025-10-22: added HUD inspector overlay toggle (Shift) plus agent outlines and existing palette modes to improve readability.]
- Procedural sandbox terrain: add an interactive map builder leveraging Wave Function Collapse to synthesize believable topography/biomes with configurable resource strata, oxygen levels, and hazard tuning.

## Input Mapping and Feature Parity
- Keyboard:
  - `P` pause, `D` toggle drawing, `F` show food, `Ctrl+Shift+O` toggle agent outlines, `+/-` adjust speed, `A` add crossover agents, `Q/H` spawn carnivore/herbivore, `C` toggle closed environment, `S/O` follow selected/oldest.
  - Implement via GPUI action bindings to mutate simulation config entity.
  - [Completed - GPT-5 Codex 2025-10-22: Holding `Shift` now enables a hover inspector overlay in the HUD for quick agent stats.]
- Mouse:
  - Right-drag pan, middle-drag zoom, left-click select agent; hold `Shift` to inspect stats overlay.
- Agent Inspector:
  - Modal or side panel displaying sensor/brain heatmaps, trait values, mutation history with scrollable `div()`.

## Testing, Tooling, and CI
- Unit tests in `scriptbots-core` covering reproduction math, spike damage, food sharing formulas.
- Property tests (QuickCheck/Proptest) verifying invariants (health clamped [0, 2], reproduction resets counter).
- Determinism tests running simulation for fixed ticks with seeded RNG and asserting stable snapshots. [Completed - GPT-5 Codex 2025-10-21]
- Bench harness (`criterion`) measuring ticks/sec at various agent counts.
- GPUI view tests using `#[gpui::test]` macro to ensure layout compiles and actions dispatch.
- Storage tests:
  - Integration suite writing simulated batches into DuckDB in-memory databases, asserting schema evolution, transaction durability, and query latency. [Completed - GPT-5 Codex 2025-10-21]
  - Snapshot-based golden tests verifying historical queries (population trends, kill counts) match expected outputs when replayed from DuckDB logs. [Completed - GPT-5 Codex 2025-10-22]
- Continuous integration: GitHub Actions with matrix (macOS 14, Ubuntu 24.04), caching `cargo` artifacts, running tests + release build. [Completed - GPT-5 Codex 2025-10-22]
 - [Completed - GPT-5 Codex 2025-10-23] Replay determinism pipeline in CI: generate baseline/candidate DuckDB runs in headless terminal mode and diff event streams via CLI (`--replay-db`, `--compare-db`).
 - [Completed - GPT-5 Codex 2025-10-23] Wasm CI job builds `scriptbots-web` via `wasm-pack` and runs headless Chrome tests (Playwright-provisioned Chromium).

## Advanced Brain Architecture Strategy
- Trait hierarchy: extend the `Brain` interface with marker traits (`EvolvableBrain`, `TrainableBrain`, `BatchBrain`) so the world loop can branch on capabilities (e.g., some brains may not support online learning).
- NeuroFlow integration:
  - Implement `NeuroflowBrain` around `neuroflow::FeedForward`, leveraging builder helpers for declarative layer construction and `io::to_file`/`from_file` for serialization.
  - Support asynchronous fine-tuning by accumulating recent experience tuples and training in background tasks, swapping weights atomically on completion.
- Genome-centric evolution:
  - `BrainGenome` persists topology, activation mix, optimizer hyperparameters, and provenance metadata (parent IDs, mutation notes). Sexual reproduction performs uniform crossover on layer specs and blends hyperparameters; asexual reproduction perturbs and, occasionally, inserts/removes layers.
  - Maintain compatibility metrics and speciation buckets (NEAT-inspired) to avoid catastrophic mixing of divergent architectures.
- Pluggable factories:
  - Register brain constructors through a `BrainRegistry` so additional engines (ONNXRuntime, Torch via tch-rs, GPU kernels) can be added behind feature flags without core refactors.
- [Completed - GPT-5 Codex 2025-10-22] Expose runtime control surfaces for selecting brain families and tuning evolution knobs (mutation rates, speciation thresholds, environmental parameters) without rebuilds.
  - Introduced `ControlHandle`/`KnobEntry` scaffolding shared by REST, CLI, and MCP front-ends for atomic JSON patching of `ScriptBotsConfig` and live snapshots.
  - [Completed - GPT-5 Codex 2025-10-23] Shipped an axum-based REST API (`/api/knobs`, `/api/config`, `/api/knobs/apply`) with generated OpenAPI docs and embedded Swagger UI for operator discovery.
  - [Completed - GPT-5 Codex 2025-10-23] Added an HTTP MCP server (tools: `list_knobs`, `get_config`, `apply_updates`, `apply_patch`) so external LLM tooling can orchestrate simulations safely; stdio/SSE transports deprecated in favor of HTTP-only.
  - [Completed - 2025-10-24] REST surface expanded: tick summaries (`/api/ticks/latest`, `/api/ticks/stream`, `/api/ticks/ndjson`), screenshots (`/api/screenshot/ascii|png`), hydrology (`/api/hydrology`), event tail (`/api/events/tail`), scoreboards (`/api/scoreboard`), agent debug (`/api/agents/debug`), selection updates (`POST /api/selection`), presets (`/api/presets`, `/api/presets/apply`), and config audit (`/api/config/audit`).
  - Delivered the `scriptbots-control` CLI (clap + reqwest + ratatui) offering scripted updates (`set`, `patch`) and a live dashboard (`watch`) that highlights config deltas while the GPUI shell runs.
- [Completed - GPT-5 Codex 2025-10-22] Migrated configuration writes onto a crossfire-backed command bus drained inside the simulation loop and exposed an HTTP MCP server (default `127.0.0.1:8090`) so control surfaces enqueue consistent `ControlCommand`s rather than mutating `WorldState` directly.
- Diagnostics:
  - Persist brain metrics (loss curves, weight norms, training tick) alongside genomes in DuckDB for analytics and UI visualization.
  - Provide debug tooling to render network topologies (layer shapes, activations) inside the inspector panel.

## Modularity & Extensibility Enhancements
- Optional ECS: evaluate `hecs`/`legion` for certain subsystems (sensing, effects) after benchmarking; keep default architecture straightforward.
- Scripting sandbox: offer a Wasm or Lua plug-in layer for experimental sensors/reward shaping, gated by capability lists to preserve determinism.
- Deterministic replay & branching: treat every RNG draw and brain choice as an event stored in DuckDB, enabling branch-and-replay workflows and regression reproduction.
- Scenario layering: allow configs to be composed (base + biome + experiment) controlling constants, active sensors, brain registries, and UI themes.

## Parity Gaps vs. Original ScriptBots (Non-Rendering)
- **World Mechanics**
  - [Completed - GPT-5 Codex 2025-10-22: added configurable regrowth/decay/diffusion with tests] Reproduce food diffusion/decay curves from `World.cpp` (current respawn is basic pulse).
  - [Completed - GPT-5 Codex 2025-10-22: Terrain-aware food fertility profiles with capacity/growth modulation plus reproduction bonuses tied to geography] Improve ecological fidelity by deriving per-cell fertility from moisture/elevation/slope, applying nutrient density to grazing energy, and adding regression tests for fertile vs. barren biomes.
  - [Completed - GPT-5 Codex 2025-10-22: spike damage scales with spike length/speed, herbivore vs. carnivore hits emit analytics flags, and death cleanup consumes the pending queue in stable world order] Flesh out combat resolution: spike damage scaling, carnivore/herbivore event flags, queued death cleanup ordering.
  - [Completed - GPT-5 Codex 2025-10-22: species-tuned reproduction rates, hybrid crossover pipeline, spawn jitter parity, gene logging, and lineage tracking with tests] Implement reproduction nuance: crossover/mutation weighting, spawn jitter, gene logging, lineage tracking.
  - [Completed - GPT-5 Codex 2025-10-22: multi-eye vision cones, smell/sound/blood channels, temperature discomfort, clock sensors wired with deterministic tests] Complete sensory modeling: angular vision cones, smell/sound attenuation, change-sensitive synapses.
  - [Completed - GPT-5 Codex 2025-10-22: metabolism ramps & age-based decay wired into actuation/aging with tests] Add energy/aging modifiers (metabolism ramps, age-based decay) to match C++ behavior.
  - [Completed - GPT-5 Codex 2025-10-22: implemented configurable temperature gradients, discomfort drains, and parity tests] Restore environmental temperature mechanics: apply discomfort-based health drain tied to agents' `temperature_preference`, expose configurable gradients, and extend tests covering equator/edge scenarios.
  - [Completed - GPT-5 Codex 2025-10-22: carcass sharing restored with age scaling, energy/reproduction rewards, and persistence metrics] Reintroduce carcass distribution: when an agent dies (especially from spikes), deterministically share meat resources with nearby carnivores/herbivores using the original age-based scaling and ensure persistence metrics capture these events.
  - [Completed - GPT-5 Codex 2025-10-22: constant-rate food sharing restored with deterministic ordering and analytics parity] Match food sharing semantics: implement the constant-rate `FOODTRANSFER` giving behavior gated by output 8, include distance checks, and keep deterministic ordering so altruistic strategies mirror the C++ dynamics.
  - [Completed - GPT-5 Codex 2025-10-22: world population seeding restored with closed-flag enforcement and scheduled spawns] Reinstate world population seeding: honor the `closed` flag, maintain minimum agent counts, and periodically inject random newcomers or crossover spawns to mirror `addRandomBots`/`addNewByCrossover` scheduling.
  - [Completed - GPT-5 Codex 2025-10-22: output-channel side effects aligned with legacy spike easing, sound multiplier persistence, and indicator pulses] Map output-channel side effects: ease spikes toward requested length, persist `sound_multiplier`/`give_intent`, update indicator pulses on key events, and surface matching config hooks for downstream rendering/audio layers.
  - [Completed - GPT-5 Codex 2025-10-22: differential drive locomotion aligned with legacy wheel outputs and boost scaling] Align locomotion with differential drive physics: interpret outputs[0]/[1] as wheel velocities, derive pose updates/boost scaling from the original formulas, and validate wrapping math against `processOutputs`.
  - [Completed - GPT-5 Codex 2025-10-23: herbivore attack gating in combat, diet-weighted carcass/food handling, and reproduction rates split via `reproduction_rate_herbivore/carnivore`; tests cover herbivore gains and carnivore waste] Port herbivore vs carnivore behaviors: enforce attack restrictions for herbivores, replicate reproduction timers (`REPRATEH/REPRATEC`), and apply diet-based modifiers in food intake and carcass sharing.
  - [Completed - GPT-5 Codex 2025-10-23: ground food intake/waste modeled with speed scaling and herbivore tendency; integration tests added for herbivore gain and carnivore waste] Match food consumption math: reuse `FOODINTAKE`, `FOODWASTE`, and speed-dependent gains tied to wheel speeds and herbivore tendency, updating tests to cover stationary vs. fast agents.
  - [Completed - 2025-10-23 Codex] Restore modcounter cadence: introduce configurable tick scheduler for aging every 100 ticks, periodic charts/reporting, reproduction gating randomness, and guard against regressions with deterministic seeds.
  - [Completed - GPT-5 Codex 2025-10-23: per-agent `mutation_rates` (primary/secondary), `temperature_preference`, and trait modifiers are inherited/mixed and mutated during reproduction with lineage logging] Carry mutation parameter genetics: track per-agent `mutrate1/2`, `temperature_preference`, and trait modifiers, mutating them during reproduction following the C++ meta-mutation rules.
  - [ ] Build regression tests comparing Rust tick traces against captured C++ snapshots.
- **Brain Implementations**
  - [Completed - GPT-5 Codex 2025-10-23: DWRAON brain ported under `dwraon` feature with runner tests; C++ parity audit pending] Port DWRAON brain with parity tests and feature gating.
  - [Completed - GPT-5 Codex 2025-10-23: Assembly brain implemented behind `experimental` feature with determinism guardrails] Port Assembly brain (even if experimental) with determinism guardrails.
  - [ ] Implement mutation/crossover suites validated against C++ reference data.
  - [ ] Develop brain registry benchmarking (per-brain tick cost, cache hit rates).
- **Analytics & Replay**
- [Completed - GPT-5 Codex 2025-10-23: Storage exposes `load_replay_events`/`max_tick`/`replay_event_counts`; `scriptbots-app` ships headless replay CLI (`--replay-db`, `--compare-db`, `--tick-limit`) with colored divergence diagnostics] Extend persistence schema to store replay events (per-agent RNG draws, brain outputs, actions).
  - [Completed - GPT-5 Codex 2025-10-23: Deterministic replay runner records per-tick events via `ReplayCollector` and verifies against DuckDB logs.]
  - [Currently In Progress - GPT-5 Codex 2025-10-23: Wire deterministic replay CLI into CI pipelines for baseline vs. candidate runs]
  - [ ] Add DuckDB parity queries (population charts, kill ratios, energy histograms) vs. C++ scripts.
  - [ ] Provide CLI tooling to diff runs (Rust vs. C++ baseline) and highlight divergences.
- **Feature Toggles & UX Integration (Non-rendering)**
  - [Completed - GPT-5 Codex 2025-10-23: Added layered scenario configs (`--config` / `SCRIPTBOTS_CONFIG`) merging TOML/RON files ahead of env overrides] Surfacing runtime toggles: CLI/ENV for enabling brains, selecting indices, adjusting mechanics.
  - [Completed - GPT-5 Codex 2025-10-23: Added `--print-config`, `--write-config`, `--config-format`, and `--config-only` flags so scenarios can be audited or exported without launching the sim] Provide config inspection tooling (print/write/dry-run) so layered scenarios are easy to audit.
  - [Currently In Progress - 2025-10-23 Codex] Selection and debug hooks: expose APIs to query agent state, highlight subsets (without GPUI coupling).
  - [ ] Audio hooks: structure event bus for future `kira` integration (without touching render crate yet).
  - [ ] Accessibility/logging: structured tracing spans, machine-readable summaries for external dashboards.

## Rendering Roadmap Snapshot (Avoid Editing Without Render Owner)
- **Shipped in Rust GPUI Layer**
  - ✅ Window bootstrap with configured title/titlebar.
  - ✅ HUD cards (tick/agents, births/deaths, energy/health) plus quick overlay panel.
  - ✅ Tick-history table with growth coloring and energy averages.
  - ✅ Canvas renderer for food tiles and agents (health-adjusted color, radius scaling).
  - ✅ Camera controls: middle-click pan, scroll zoom with anchor, HUD readouts for zoom/pan.
- **Outstanding / Upcoming**
- [Completed - GPT-5 Codex 2025-10-22: Canvas selection/hover highlights, brush tooling UI stubs, and debug probe toggles (follow-up: hook brush/probe ops to world).]
  - [Completed - GPT-5 Codex 2025-10-22: Event overlays (combat flashes, reproduction halos, food diffusion glow) with brush/probe telemetry surfaced; follow-up: connect overlays to simulation history buffers.]
  - [Completed - GPT-5 Codex 2025-10-22: Performance diagnostics panel (frame time/fps tracking, live stats overlay); follow-up: wire GPU backend counters once exposed.]
  - [Completed - GPT-5 Codex 2025-10-22: Terrain/lighting polish (dynamic sky palette, food tile shading, reproduction/spike VFX); follow-up: hook palettes to configurable themes.]
  - [Completed - GPT-5 Codex 2025-10-22: Input rebinding & accessibility (keyboard remaps, colorblind-safe palettes, narration hooks); follow-up: stream narration events into audio/log subsystem.]

## Migration Roadmap
1. **Project Bootstrap (Week 1)**
   - Initialize workspace via `create-gpui-app --workspace`.
   - Set up linting (`rustfmt`, `clippy`), choose MSRV (Rust 1.81+) to match GPUI requirement.
2. **Core Data Structures (Weeks 1-2)**
   - Port `settings.h` constants into `ScriptBotsConfig`. [Completed - GPT-5 Codex 2025-10-22: aligned default food/reproduction constants with legacy values and added validation tests]
   - Implement `Vector2` replacement via `glam::Vec2`.
   - Port agent struct, reproduction, mutation logic with unit tests.
3. **World Mechanics (Weeks 2-4)**
   - Implement food grid, sensing pipeline (sequential first), reproduction queue, death cleanup.
   - Ensure parity with original via scenario tests (e.g., spike kill distribution).
   - [Completed - GPT-5 Codex 2025-10-22: Terrain-aware food ecology + fertility-weighted reproduction with regression tests]
4. **Introduce Concurrency (Weeks 4-5)** [Completed: stage_sense/actuation/combat parallelized with Rayon]
   - Integrate Rayon, add spatial partition acceleration, verify determinism under multi-thread.
5. **Brain Ports (Weeks 5-7)**
   - MLP (baseline) complete with mutate/crossover. [Completed - GPT-5 Codex 2025-10-21]
   - DWRAON feature gate; assembly brain behind `--features experimental`. [Completed - GPT-5 Codex 2025-10-21]
   - Implement NeuroFlow-backed brain module and wire through the brain registry for opt-in builds. [Completed - GPT-5 Codex 2025-10-22]
   - Seed NeuroFlow weights using the world RNG for deterministic runs. [Completed - GPT-5 Codex 2025-10-21]
   - Add runtime configuration toggle to enable NeuroFlow brains without compile-time features. [Completed - GPT-5 Codex 2025-10-21]
6. **Persistence Layer (Weeks 7-8)**
   - Stand up `scriptbots-storage`, define DuckDB schema (agents, ticks, events, metrics). [Completed - GPT-5 Codex 2025-10-22]
   - Implement buffered writers, compaction routines, and analytics helpers (e.g., top predators query). [Completed - GPT-5 Codex 2025-10-22]
7. **Rendering Layer (Weeks 8-10)** [Completed - GPT-5 Codex 2025-10-22: GPUI stats overlay and controls polished; Terminal renderer MVP implemented with auto-fallback + headless CI]
- Build GPUI window, canvas renderer, agent inspector UI. [Completed - GPT-5 Codex 2025-10-22: window shell, HUD, canvas renderer, and inspector panel shipped]
   - Implement camera controls, overlays, history chart. [Completed - GPT-5 Codex 2025-10-22: middle-click pan, scroll zoom, overlay HUD, tick-history chart]
   - Prototype tile-based terrain, vector HUD, and post-processing shader pipeline for polished visuals. [Completed - GPT-5 Codex 2025-10-22: terrain driven by core layer, velocity-aware vector HUD, palette-aware post FX; follow-up: experiment with GPU shader hooks once GPUI exposes them.]
8. **Integration & UX Polish (Weeks 10-11)**
  - Hook actions to simulation, selection workflows, debug overlays. [Completed - GPT-5 Codex 2025-10-22: left-click/shift multi-select integrated with world runtime, hover highlighting feeds the inspector, selection controls get shortcuts/logging, and the debug overlay now exposes velocity/sense visualisation toggles from the HUD.]
  - Add metrics HUD, performance counters. [Completed - GPT-5 Codex 2025-10-22: summary grid now includes frame-time/FPS cards backed by rolling PerfStats, plus live simulation state metrics mirrored in the inspector controls.]
  - [Completed - GPT-5 Codex 2025-10-22: Inspector exposes brain info, mutation rate tuning, and persistence toggles; follow-up: wire actions into shared world event bus.]
  - [Completed - GPT-5 Codex 2025-10-22: Layered `kira` audio cues wired to births/deaths/spikes with accessibility toggles; follow-up: move event capture to shared bus and expand particle sync.]
9. **Testing, Benchmarks, Packaging (Weeks 11-12)**
   - Determinism/regression suite, `cargo bench`. [Completed - GPT-5 Codex 2025-10-22]
   - Release pipeline (`cargo dist` or `cargo bundle`), signed macOS binaries. [Completed - GPT-5 Codex 2025-10-22: `cargo dist` workflow now emits platform archives with optional macOS signing and README docs cover the release process; follow-up: add notarization once certificates are in place.]

## Risks and Mitigations
- **GPU backend availability**: GPUI is still evolving; focus initial support on macOS/Linux per official guidance, while monitoring upstream platform work.
- **Determinism under parallelism**: Floating-point reductions can diverge; solve via deterministic orderings and staged accumulations.
- **Performance regressions**: Large agent counts may stress GPUI canvas; prototype rendering with 10k agents early, profile using Tracy/`wgpu-profiler`.
- **Brain extensibility**: Trait object overhead; consider `enum BrainImpl` with static dispatch once stable.
- **DuckDB durability/throughput**: Excessive per-tick writes could bottleneck; mitigate with batched transactions, asynchronous writers, and optional toggles for high-frequency logging. Evaluate Parquet exports for heavy analytics workloads.
- **NeuroFlow maturity**: Crate targets CPU feed-forward networks; keep abstractions loose so alternative engines can slot in if requirements outgrow NeuroFlow.
- **Team familiarity with GPUI**: Documentation is evolving; allocate ramp-up time exploring official docs, tutorials, and community component libraries such as `gpui-component`.

## Open Questions
- Should we adopt serde-based save/load for simulation snapshots at launch?
- Do we need remote observer mode (headless simulation + streaming state to GPUI frontend)?
- What telemetry is acceptable for release builds (opt-in metrics)?
- Long term: evaluate advanced vision/AI brain integration once core port is stable.

---

## High-Impact Incremental Enhancements (Backlog) [Currently In Progress - 2025-10-23]

These scoped additions improve usability, insight, and experiment velocity without large refactors. Each item lists Purpose, MVP, Control Surfaces, Data/Perf notes, Testing, and Complexity.

### 1) Config presets + quick toggle
- Purpose: instant scenario swapping without editing files; great for demos and teaching.
- MVP: ship a small set of curated layered configs (e.g., `arctic.toml`, `boom_bust.toml`, `closed_world.toml`) and a preset switcher.
- Surfaces: CLI `scriptbots-control presets` / `scriptbots-control apply-preset <name>`; GPUI dropdown in HUD; TUI picker; REST `GET /api/presets`, `POST /api/presets/apply { name }`.
- Data/Perf: uses existing layered-config loader; no extra state; deterministic.
- Testing: verify merged config equals golden snapshots; run same seed before/after swap.
- Complexity: S.

[Currently In Progress - 2025-10-23] REST endpoints added: `GET /api/presets` (lists names) and `POST /api/presets/apply` (applies JSON patch for preset); OpenAPI updated.
Follow-up [Currently In Progress - 2025-10-23]: GPUI preset picker upgraded to a compact micro-menu with keyboard focus/hover and a confirmation toast on apply.
Keyboard/UX details: the preset micro-menu mirrors button styling, supports focus rings, hover color shifts, and click feedback; a toast appears under the header with the applied preset name and fades out automatically.

### 2) Metrics baseline compare (Δ vs. baseline) [Currently In Progress - 2025-10-23]
- Purpose: quick A/B within a run (population, births/deaths, avg energy).
- MVP: "Set Baseline" button stores current summary; HUD/TUI shows Δ and %Δ.
- Surfaces: HUD button + toggle (GPUI baseline button added in Simulation Controls); TUI key `b` to set/reset (implemented in Terminal HUD); CLI `control_cli baseline set/reset`.
- Data/Perf: store one struct in memory; optional DuckDB event for audit.
- Testing: unit-test delta math; snapshot HUD/TUI lines.
- Complexity: S.

### 3) Auto-pause on conditions [Completed - 2025-10-23 Codex]
- Purpose: stop at interesting moments automatically.
- MVP: triggers: population < X, first spike kill, age > Y.
- Surfaces: HUD panel with checkboxes/thresholds; TUI toggles; REST patch `control.auto_pause.*` keys; CLI `--auto-pause-below <COUNT>`, `--auto-pause-age-above <AGE>`, `--auto-pause-on-spike`.
- Data/Perf: O(1) checks driven by cached per-tick stats; deterministic.
- Testing: seeded scenarios that cross thresholds → assert paused.
- Complexity: S.

Implementation notes [2025-10-23]:
- Config: `ControlSettings` now exposes `auto_pause_age_above` and `auto_pause_on_spike_hit` alongside the existing population threshold.
- CLI: new flags/env vars configure population, age, and spike-hit pausing; reflected in emitted config.
- Terminal & headless: snapshot evaluation auto-pauses and logs the trigger reason, halting the loop and zeroing speed.
- GPUI renderer: simulation pump honours the same three triggers, pausing and logging via tracing.
- Terminal & GPUI: main render loops set paused when `agent_count <= limit`.
- CLI: `--auto-pause-below` (env `SCRIPTBOTS_AUTO_PAUSE_BELOW`) wires into config at startup.

### 4) Event feed (birth/kill/combat) [Completed - 2025-10-23 Codex; [Currently In Progress] polishing filters]
- Purpose: narrative understanding; lightweight teaching log.
- MVP: bounded ring buffer sourced from tick summaries (births/deaths/spike_hits).
- Surfaces: HUD side panel; TUI panel; REST `GET /api/events/tail?limit=N`.
- Implementation:
  - Terminal HUD now renders a colorized Recent Events panel; event ingestion hooks births/deaths deltas and population changes.
  - REST: `/api/events/tail` returns latest events (tick, kind, count).
- Data/Perf: reads from in-memory history; zero extra allocations on hot paths.
- Follow-ups: add filter chips (All|Birth|Death|Combat) and expose in Swagger.
- Complexity: S.

### 5) Selection tags (cohorts)
- Purpose: track cohorts across ticks for focus and stats.
- MVP: add/remove string tags to selected agents; highlight + cohort counts.
- Surfaces: HUD tag input; TUI command `:tag add foo`/`remove`.
- Data/Perf: transient runtime map `AgentId -> SmallVec<Tag>`; optional DuckDB write on explicit export.
- Testing: determinism (tags do not affect physics); UI smoke.
- Complexity: S.

### 6) Spawn brushes
- Purpose: faster experiment setup; inject small clusters.
- MVP: brush kinds (herbivore/carnivore/mixed) with size/density sliders.
- Surfaces: HUD brush toolbar; TUI `A/Q/H` variants with size modifier.
- Data/Perf: uses existing spawn queue; deterministic order preserved.
- Testing: seeded brush at coordinates → expected agent count/types.
- Complexity: S-M.

### 7) Sensor rays overlay (scoped)
- Purpose: visualize perception without heavy cost.
- MVP: draw limited rays for focused agent (and optional nearest 4).
- Surfaces: HUD toggle; TUI `v` toggle.
- Data/Perf: compute from existing sense snapshot; throttle to every N frames.
- Testing: visual snapshot; ensure disabled has zero overhead.
- Complexity: S.

### 8) Lineage mini-tree
- Purpose: quick glance at heredity and mutation flow.
- MVP: 2-level ancestry (parents→child) with ages/traits; link to inspector.
- Surfaces: HUD inspector subpanel; TUI detail pane.
- Data/Perf: store parent ids at birth; optional DuckDB write.
- Testing: reproduction unit test ensures parent linkage; UI snapshot.
- Complexity: S-M.

### 9) Rolling heatmap overlays
- Purpose: spatial intuition for births, deaths, intake, collisions.
- MVP: rolling grids (last N ticks) for 2–3 metrics; simple color ramp.
- Surfaces: HUD overlay selector; TUI cycle key.
- Data/Perf: maintain small `Grid<u16>` counters; decay each tick; bounded cost.
- Testing: seeded events increment expected cells; determinism across threads.
- Complexity: M.

### 10) Determinism self-check (threads parity) [Currently In Progress - 2025-10-23]
- Purpose: confidence guard for contributors/CI.
- MVP: CLI `--det-check [ticks]` runs 1-thread vs N-threads and compares summaries/events. [Completed - 2025-10-24: implemented in the app binary; CI job executes det runs and compares outputs.]
- Surfaces: CLI; CI job gate (non-blocking initially).
- Data/Perf: temporary run in memory; no DB write unless `--save`.
- Testing: fixture worlds pass; inject a known race → diff surfaces red.
- Complexity: M.

### 11) NDJSON tick tail endpoint [Currently In Progress - 2025-10-23]
- Purpose: easy streaming to dashboards/scripts without websockets.
- MVP: `GET /api/ticks/latest` (JSON), `GET /api/ticks/stream` (SSE; emits JSON objects periodically). [Completed - 2025-10-24: `GET /api/ticks/ndjson` streams newline-delimited JSON]
- Surfaces: REST only; docs example with `curl`.
- Data/Perf: send from in-memory summaries; backpressure via chunked responses.
- Testing: integration test reading N events; cancellation.
- Complexity: S.

### 12) Screenshot/export (GUI + TUI) [Currently In Progress - 2025-10-23]
- Purpose: reporting and bug repro.
- MVP: GPUI PNG capture; TUI saves ASCII frame to `.txt`. [Completed - 2025-10-24: Terminal HUD adds 'S' to save ASCII under `screenshots/frame_<tick>.txt`; CLI `scriptbots-control screenshot --out FILE [--png]`; REST: `GET /api/screenshot/ascii|png`. PNG requires GUI feature; offscreen renderer guards large sizes.]
- Surfaces: HUD button + hotkey; TUI `S` key; CLI `control_cli screenshot`.
- Data/Perf: file I/O off main tick; enqueue to worker.
- Testing: files exist with non-zero bytes; deterministic filenames with seed/tick.
- Complexity: S.

### 13) Quick CSV exports [Completed - 2025-10-23 Codex: added `scriptbots-control export` for ticks/metrics CSV dumps]
- Purpose: spreadsheet-friendly metrics without opening DuckDB.
- MVP: CLI `export metrics --last 1000 --out metrics.csv`; similar for `ticks`.
- Surfaces: CLI; REST `GET /api/export/metrics.csv` (optional).
- Data/Perf: simple SELECT + CSV writer; small temp buffer.
- Testing: header/row counts; types; delimiter escaping.
- Complexity: S.

### 14) Scenario runner (N seeds)
- Purpose: aggregate behavior across seeds quickly.
- MVP: CLI `run-seeds --preset arctic --seeds 20 --ticks 500` → summary table (mean/p95 of key metrics).
- Surfaces: CLI only; writes markdown/CSV summary.
- Data/Perf: headless runs; optional DuckDB per run path.
- Testing: deterministic results per seed list; performance guard.
- Complexity: M.

### 15) HUD perf knobs (speed/threads)
- Purpose: direct feedback loop while tuning perf.
- MVP: live sliders for speed multiplier and Rayon threads with current FPS/ticks/sec.
- Surfaces: HUD; TUI shortcuts (`+/-` already) plus `T` cycles threads.
- Data/Perf: applies via command bus; safe at frame boundaries.
- Testing: assert thread count propagates; FPS trend observable.
- Complexity: S.

### 16) Config change audit + revert [Currently In Progress - 2025-10-23]
- Purpose: transparency and quick undo during experiments.
- MVP: bounded list of last K patches with timestamp; revert re-applies inverse patch.
- Surfaces: HUD panel; TUI list; REST `GET /api/config/audit` (implemented: returns recent in-process config patches with tick).
- Data/Perf: store patches in ring buffer; optional DuckDB audit table.
- Testing: apply→revert round-trip yields identical config; determinism preserved.
- Complexity: M.

### 17) World annotations (pins/regions)
- Purpose: contextualize terrain spots and study areas.
- MVP: add named pin or rectangular region with color; overlay toggle.
- Surfaces: HUD placement tool; TUI `:pin x y name`.
- Data/Perf: lightweight list in memory; optional persist on export.
- Testing: serialization round-trip; overlay render smoke.
- Complexity: S-M.

### 18) Agent scoreboard [Completed - 2025-10-23 Codex]
- Purpose: at-a-glance "Top predators" and "Oldest living".
- MVP: two small tables sourced from current snapshot (not heavy analytics).
- Surfaces: Terminal HUD shows "Top Predators" (carnivores by energy) and "Oldest Agents"; REST `GET /api/scoreboard?limit=K`.
- Implementation:
  - Control layer computes top carnivores (energy/health tie-break) and oldest agents.
  - OpenAPI schemas exposed; endpoints documented in Swagger UI.
- Data/Perf: computed from live world state; bounded sort/truncate (`K=10` default).
- Complexity: S.

### 19) Brush-based food edits
- Purpose: controlled perturbations for experiments.
- MVP: increase/decrease local food density using brush with radius/strength.
- Surfaces: HUD brush; TUI command `:food +|- r=.. s=..`.
- Data/Perf: bounded cell updates via command bus; deterministic application order.
- Testing: grid diffs equal expected kernel; no agent physics side-effects until next tick.
- Complexity: M.

### 20) One-agent replay snippet (last 50 ticks)
- Purpose: micro-replay to understand recent fate of an agent.
- MVP: when selecting a recent death, show 50-tick path/health mini-timeline.
- Surfaces: HUD inspector mini-chart; TUI detail pane.
- Data/Perf: reuse `ReplayCollector` buffer in memory; no new schema required. [Currently In Progress]
- Testing: fixed seed reproduces identical snippet; UI snapshot.
- Complexity: M.

### 21) Selection APIs (cohorts) [Completed - 2025-10-24]
- REST: `POST /api/selection` accepts `mode: set|add|remove`, `agent_ids: []`, and optional `state` to update selection cohorts deterministically (applied via command bus inside the tick loop).
- Debug: `GET /api/agents/debug` provides filtered agent rows (ids/diet/selection/brain) with deterministic sorting (energy/health/age/id tie-breakers) for lightweight queries.

---

## World Renderer Evolution — Offscreen wgpu Readback Composited in GPUI (Chosen Path)

### Rationale
- The GPUI canvas path is excellent for HUD/UI, vector shapes, and small custom drawings, but our world view requires high-throughput 2D rendering of thousands of agents and tiles every frame with post‑FX. A dedicated GPU pipeline using wgpu gives us instancing, batching, and predictable performance.
- We deliberately avoid forking GPUI. Instead, we render the world into an offscreen wgpu texture and composite it inside GPUI using a per‑frame updated image. This works with unmodified upstream GPUI on all platforms we target.
- The bandwidth cost of GPU→CPU→GPU at 1080p/1440p, 60–120 Hz, is well within modern hardware headroom. We will engineer the pipeline to minimize latency and CPU cost via triple‑buffered staging and persistent allocations.

### Goals
- Smooth 60–120 FPS world rendering at 1080p/1440p with 10k+ agents under typical settings.
- Deterministic visuals (no racey CPU/GPU feedback loops). GPUI retains input, overlays, inspector panels, and charts.
- Clean, testable crate boundaries: world renderer is a small, well‑documented wgpu crate with no GPUI dependency; GPUI owns composition only.

### Architecture Overview
1) New crate: `crates/scriptbots-world-gfx`
   - Public API (stable):
     - `WorldRenderer::new(device, queue, size, options) -> Self`
     - `WorldRenderer::resize(&mut self, new_size)`
     - `WorldRenderer::render(&mut self, snapshot: &WorldSnapshot) -> RenderFrame`
     - `RenderFrame { color: wgpu::Texture, extent: (u32, u32) }`
     - `WorldRenderer::readback_rgba8(&mut self, &RenderFrame) -> &[u8]` (triple‑buffered; returns pointer to a persistently mapped staging region valid until next rotation)
   - Internals:
     - Terrain pipeline: instanced quads using a tileset atlas; per‑cell variation (moisture/elevation) via SSBO or texture; one draw per layer.
     - Agents pipeline: instanced quads with per‑instance buffer (position, size, color, spike, flags); one draw per species/material.
     - Selection rings/highlights: either a second instanced pipeline or SDF in the main shader.
     - Culling: CPU frustum culling on agent instances; optional compute‑binning path later.
     - Post‑FX: color grade/vignette/bloom implemented in a lightweight full‑screen pass.

2) Readback/Upload Strategy (polished, low‑latency)
   - Render target: RGBA8UnormSrgb.
   - CopyTextureToBuffer into a row‑padded staging buffer (wgpu requires 256‑byte alignment for `bytes_per_row`). We compute stride = align(width * 4, 256).
   - Triple‑buffered staging:
     - Allocate 3 persistent `wgpu::Buffer`s in MAP_READ | COPY_DST usage; rotate each frame.
     - Issue copy → `buffer.slice(..).map_async(Read)` without awaiting; poll via a lightweight fence API and read only when `IsReady`. The previous frame’s buffer is typically ready by the time we need to upload to GPUI, avoiding stalls.
   - GPUI image persistence:
     - Maintain persistent GPUI image objects/buffers sized to the current viewport; reuse them every frame (no reallocations).
     - CPU‑side memcpy from mapped staging into GPUI’s image backing memory; then invalidate/mark dirty for GPUI to present.
   - Ring‑buffer the uploads: when a frame’s staging map isn’t ready, skip re‑upload and reuse the last image (prevents blocking the UI thread).

3) Color/format discipline
   - Always RGBA8 sRGB for the render target and GPUI image. No per‑pixel swizzles or conversions. No tonemapping on CPU.

4) Resize & scale
   - Renderer tracks `world_view_size` separately from the window; when GPUI’s layout changes, call `resize` to recreate the color target and the three staging buffers with correctly aligned rows.
   - Snapshots include world size; camera maps world→view consistently; culling uses updated frustum.

5) Integration in `scriptbots-render`
   - Add a small compositor struct:
     - Owns `WorldRenderer` and a triple of `StagingSlot { mapped_ptr, bytes_per_row, fence }` plus a persistent `GpuImage` wrapper for GPUI.
     - Each UI frame:
       1. Acquire latest `WorldSnapshot` (read‑only snapshot already exists).
       2. Call `render(snapshot)` to produce a wgpu texture.
       3. Enqueue `copy_to_staging` into the current ring slot; request map; poll the previous slot; if ready, memcpy into GPUI image.
       4. Present the GPUI image inside the existing world viewport node.
   - No changes to input handling, inspector, or HUD—only the world drawing path changes.

6) Performance techniques
   - Instanced draws with tightly packed per‑instance buffers; avoid per‑agent draw calls.
   - Persistent bind groups/pipelines; update only dynamic buffers.
   - Avoid transient allocations each frame (reserve vecs; reuse `Vec::clear`).
   - CPU frustum culling reduces instance count when zoomed in.
   - Optional compute pass for large worlds to bin agents into screen tiles for more aggressive culling (backlog).

7) Scheduling & latency
   - UI thread never blocks on GPU readback. We always upload last‑ready frame.
   - Typical added latency: ~1 frame at 60–120 FPS. Acceptable for the simulation; HUD remains interactive.

8) 4K/High‑refresh guidance
   - 4K@60 (~33.2 MB/frame) is fine on PCIe 4.0 and Apple Silicon unified memory. 4K@120 approaches ~8 GB/s round‑trip; still feasible on high‑end systems but we’ll expose a toggle to halve the world buffer rate or resolution when needed.
   - If we outgrow readback at ultra‑high refresh rates, we can add a zero‑copy external‑texture path later (requires GPUI API support).

### Deliverables
- New crate `scriptbots-world-gfx` with:
  - Instanced terrain/agents pipelines and post‑FX.
  - Triple‑buffered readback manager.
  - Safe, documented API and unit tests for alignment/stride math.
- `scriptbots-render` integration behind a feature flag (`world_wgpu`).
- Fallback path: keep the current GPUI canvas renderer behind a toggle for regression comparison.

### Testing & Benchmarks
- Determinism: snapshots driven by seeded `WorldSnapshot` → capture images and hash; compare across platforms.
- Throughput microbench: measure `render+readback` time at 1080p/1440p with 1k/5k/10k agents.
- Resize thrash: random resize patterns; assert no realloc leaks; staging ring rotates cleanly.
- CI: headless `wgpu` test that renders synthetic scenes, validates map readiness cadence, and bounds alignment math.

### Implementation Steps
1) Create `scriptbots-world-gfx` crate; add `wgpu`, `wgpu-types`, `bytemuck`, `glam`, `glyphon` (optional for labels), feature‑gate compute binning.
2) Build pipelines (terrain, agents, post‑FX); validate with a synthetic snapshot generator.
3) Implement readback ring (3 slots): allocate persistent MAP_READ buffers with aligned `bytes_per_row`.
4) Integrate in `scriptbots-render`: add compositor node that owns the renderer and GPUI image; replace world canvas draw with image presentation.
5) Bench/optimize; document knobs: render size, max FPS, culling on/off.
6) Keep canvas renderer as a fallback toggle (`--renderer=canvas|wgpu`).

### [Currently In Progress] Execution Log
- 2025-10-24: Created new crate `scriptbots-world-gfx` (workspace member) with initial `WorldRenderer` skeleton, RGBA8 sRGB color target, and triple‑buffered readback ring API (non‑blocking poll). This establishes the offscreen render + readback contract for GPUI composition. Next: wire minimal terrain/agent pipelines and the GPUI compositor stub.
- 2025-10-24: Implemented two instanced pipelines (terrain + agents). Terrain uses a texture‑atlas, linear sampling, and alpha blending; agents render as SDF circles with rim highlights for premium visuals. Added viewport uniform (bind groups) and correct NDC mapping, plus per‑frame updates and resize propagation. Next: biome-aware atlas UVs, water shimmer/slope accents, CPU frustum culling, and GPUI compositor upload path.

### Maintenance & Risk Notes
- No GPUI fork. Unmodified upstream stays in `Cargo.toml`.
- The renderer crate is self‑contained; future upgrades (e.g., zero‑copy textures) won’t impact core/hud code.
- Platform backends come from wgpu; we test Metal (macOS) and Vulkan (Linux/Windows) in CI where available.


## Sandbox Map Creator (Wave Function Collapse) — Comprehensive Roadmap [Currently In Progress - GPT-5 Codex 2025-10-23]

### Purpose & Outcomes
- Provide an interactive, deterministic sandbox for generating and editing world maps using Wave Function Collapse (WFC) and complementary procedural techniques.
- Output coherent terrain/biome tiles and a fertility/food seed map that plugs directly into `scriptbots-core` (terrain, food grid, population seeding).
- Expose REST/MCP/CLI knobs for automated sweeps; ship example tilesets and recipes for visually appealing results.

### Scope (MVP → Advanced)
1) MVP (rule-based WFC) [Currently In Progress - GPT-5 Codex 2025-10-23]
   - Tileset-driven WFC (adjacency matrices, rotations/reflections, weights), CPU implementation.
   - Deterministic generation (seeded RNG), resolution presets (e.g., 64×64, 128×128, 256×256).
   - Export to core layers: `TerrainLayer` (enum per tile) and `fertility: Grid<f32>` (0..1); optional temperature mask.
   - GPUI editor: generate/regen with seed, visualize collisions/contradictions, paint override brush, save/load tileset and map.
2) Advanced
   - Sample-based WFC (pattern extraction from example images/tilesheets), pattern size k=2..3 with rotations.
   - Mixed model: pre-pass WFC for macro-structure, noise-based post for micro-variation (Perlin/Simplex/fBm ridges, rivers).
   - Constraints: rivers/roads/lakes pinned with soft/hard constraints; keep-outs for spawn sites; guaranteed traversability corridors.
   - Tiling-at-scale: chunked generation (e.g., 512×512) with seamless borders; streaming view in GPUI.
   - Optional compute path: batch entropy/propagation on worker threads; evaluate GPU compute later.

### Data Model & Formats
- Tileset spec (TOML/RON):
  - `tile { id, label, weight, terrain_kind, fertility_bias, temperature_bias, palette_index }`
  - `adjacency { tile_a, side, tile_b, side, allowed }` (4-way + optional diagonals)
  - `symmetry { rotate=true|false, reflect=true|false }`
- Map artifact:
  - `terrain: Grid<TerrainKind>` — one tile per cell.
  - `fertility: Grid<f32>` — derived via per-tile bias + smoothing kernels.
  - `temperature: Grid<f32>` (optional) — gradient + per-tile offsets.
- Version & provenance metadata: seed, generator version, model (rule|sample), pattern size, constraints.

### Library Survey (Rust, 2025)
- WFC implementations (CPU):
  - `wfc` (crates.io) — general-purpose WFC in Rust; good starting point for rule-based model.
  - Alternative lightweight WFC ports exist; select one with permissive license (MIT/Apache) and active maintenance.
- General utilities:
  - `image` for reading tilesheets/examples in sample-based mode; `png`/`qoi` support.
  - Noise: `noise` crate (Perlin/Simplex) to blend fertility/temperature or carve rivers.
  - Parallelism: Rayon for propagation hot loops (feature-gated; keep single-threaded for wasm).
- Rendering:
  - GPUI canvas already available; draw tile quads with a palette; optional mini-map thumbnails.

### Determinism & Performance
- Deterministic orderings for: cell selection (lowest entropy with stable tie-breaks), tile choice (seeded RNG), propagation queue.
- Seeds: `seed = master_seed ⊕ world_id ⊕ tileset_hash`; record in artifact metadata.
- Performance targets (CPU): 128×128 in < 75 ms (rule-based) on CI baseline; 256×256 in < 300 ms; pattern-extraction in sample mode amortized and cached.
- Memory: limit states per cell by precomputing allowed sets from adjacency; reuse buffers; avoid per-iteration allocs.

### Integration with Core Layers
- TerrainKind mapping: tileset enumerates `TerrainKind` (e.g., Water, Sand, Grass, Rock, Snow).
- Fertility derivation: per-tile `fertility_bias` blended with Gaussian kernel; optional erosion/diffusion post.
- Food seeding: initialize `food` grid based on fertility (e.g., `food = fertility * food_max * biome_factor`).
- Spawn rules: mark legal spawn tiles; bias herbivore/carnivore spawn toward biomes; prevent water spawns.
- Temperature: combine world gradient with per-tile bias for discomfort modeling.

### UI/UX (GPUI Editor)
- Panels: Tileset (list, weights, symmetry), Constraints (pin, forbid, keep-out), Generation (seed, size, retries), Post (fertility smoothing, river carve), Export.
- Tools: Brush (paint tile/constraint), Picker, Flood fill; Undo stack (N steps), regen with same seed.
- Overlays: show contradictions in red; toggles for fertility/temperature heatmaps; mini-map preview; palette themes.

### Control Surfaces (REST/MCP/CLI)
- REST endpoints (proposed):
  - `POST /api/wfc/generate` { model: "rule"|"sample", seed, size, tileset_id, constraints? } → map artifact id
  - `GET /api/wfc/artifacts/:id` → metadata + preview
  - `POST /api/wfc/apply/:id` → replace world terrain/fertility/temperature; re-seed food/spawns deterministically
  - `POST /api/wfc/tilesets` (CRUD) → upload/download tilesets (TOML/ZIP with PNGs)
- MCP tools mirror: `wfc_generate`, `wfc_apply`, `wfc_list_tilesets`, enabling LLM-driven experimentation.
- CLI: `scriptbots-control wfc generate --tileset arctic --size 128 --seed 42`.

### Wasm Viability
- Rule-based WFC compiles to wasm fine (single-thread); sample-based pattern extraction uses `image` and is CPU-bound but acceptable for moderate sizes.
- Hosting: no special headers required for single-thread; if we parallelize in wasm later, COOP/COEP and `wasm-bindgen-rayon` apply.

### Testing & QA
- Property tests: adjacency closure (no orphan rules), symmetry correctness, reversibility of rotations/reflections.
- Determinism: fixed seed produces identical terrain/fertility outputs across platforms and thread counts.
- Regression suite: golden artifacts (terrain/fertility PNGs) for canonical tilesets; checksum in CI.
- Failure handling: contradiction detection and repair strategies (backtracking cap, retries, constraints hints) with clear UX.

### Deliverables & Milestones
- M1 (Rule-based MVP): tileset spec + CPU WFC + REST `generate/apply` + basic GPUI editor.
- M2 (Sample-based mode): pattern extraction from example tilesheets; caching; preview thumbnails.
- M3 (Constraints & rivers): pin/forbid/keep-out + noise/ridge-based river carving + fertility derivation.
- M4 (UX polish & artifacts): undo/redo, export/import tilesets and maps, screenshot/export previews.
- M5 (Automation): CLI + MCP tools; demo notebooks or scripts that sweep seeds/tilesets and log population outcomes.

### Risks & Mitigations
- Contradictions in WFC: provide constraint hints, backtracking with cap, and guided regeneration (`low-entropy patch` mode) instead of full resets.
- Tileset authoring difficulty: ship curated example tilesets (islands, archipelago, arctic, desert steppe) with clear adjacency diagrams.
- Performance at larger sizes: chunked WFC with seam constraints; cache per-chunk entropy; optional Rayon for native builds (feature-gated).

### References (curated)
- Wave Function Collapse, original repo — tileset and sample models (MIT).

### Hydrology & Dynamic Water Systems (Rainstorms + Floodable Basins) — Proposed Blueprint [Currently In Progress - GPT-5 Codex 2025-10-23]
- **Motivation**: Elevate sandbox maps beyond static biomes by layering deterministic hydrology that can react to scripted rainstorms. Agents must negotiate seasonal floods, learn to detour, or evolve swimming. The system needs to dovetail cleanly with WFC generation so basins, rivers, and floodplains emerge coherently and remain reproducible.

- **Tileset Schema Extensions**
  - Add optional hydrology fields to `TileSpec`: `permeability`, `runoff_bias`, `basin_rank`, `channel_priority`, `swim_cost`. Defaults keep legacy tilesets valid.
  - Permit adjacency rules tagged with hydrology intents (e.g., `requires_inflow`, `cannot_border_deep_basin_without_channel`). Compilation maps these to per-pattern metadata stored alongside `CompiledTile` so runtime cost lookups stay O(1).

- **Hydrologic Graph Construction (Post-WFC)**
  - Derive a lattice graph from `TerrainLayer`: each cell stores elevation and permeability. Run a priority-flood or modified D8 flow solver to compute:
    - `flow_direction` (best downhill neighbor respecting WFC-imposed channels).
    - `flow_accumulation` (catchment area proxy) and `basin_outlet` per cell.
    - `spill_threshold` (height at which basin overflows into downstream cell) using partial union-find of depressions.
  - Persist results in a new `HydrologyField` struct stored inside `MapArtifact` so deterministic replays can reconstruct the graph without recomputation.

- **Rainstorm Events & Water Depth Field**
  - Initialize `water_depth` scalar field to zero; attach to `WorldState` alongside fertility/temperature.
  - Weather controller API:
    - `start_rain(seed, intensity, duration, focus_region?)`
    - `tick_rain(dt)`: increments `water_depth` = rainfall * accumulation factor, subtracts evaporation, and routes overflow using `flow_direction` + `spill_threshold`.
    - `stop_rain()` gracefully winds down precipitation while runoff continues until all basins stabilize.
  - Expose commands via control CLI/REST/MCP: e.g., `wfc.weather.rainburst --seed 88 --intensity 0.7 --duration 600 --focus basin:delta`.

- **Terrain / Agent Interactions**
  - Define three water regimes per tile: `Dry`, `Shallow` (foraging hindered), `Deep` (requires swimming). Thresholds combine base elevation, water depth, and tile permeability.
  - Update `TerrainTile` at runtime with dynamic `water_state`, `movement_penalty`, `swim_required` flags.
  - Extend locomotion: If `topography_enabled`, add `water_depth` penalty. Introduce optional `Swimming` trait on agents; non-swimmers reroute using A* pathing informed by `movement_penalty`.
  - Fertility feedback: flooded grass converts to `Wetlands` tile variant (higher nutrient density but slower regrowth post-drought), while prolonged drought reverts wetlands to plains.

- **Integration with WFC Pipeline**
  - During `RuleBasedMapGenerator::generate`, enforce hydrology-aware adjacency: channels must connect basins; plains near coastlines must offer spillways.
  - Provide deterministic "seed rain" preview: run a shallow rain simulation (few ticks) inside generation, capturing expected max water depth. Feed results into tile selection weights (e.g., discourage high-density settlements in perennial floodplains unless tileset explicitly opts in).
  - Allow tilesets to tag "aquifer" tiles; WFC ensures they appear beneath rivers to justify perennial flow.

- **Renderer / UX Hooks**
  - GPUI: animate water surfaces with shader palettes driven by `water_depth`. Add hydrology inspector showing basin outlines, flow vectors, rainfall timeline graphs.
  - Terminal renderer: extend glyph palette with dynamic water emojis or colorized ASCII shading (e.g., `~`, `≈` for shallow/deep), plus sidebar sparkline tracking flood extent.
  - Provide "rain storyboard" overlay: timeline scrubber shows predicted inundation when planning rainfall experiments.

- **Persistence & Replay**
  - Extend `MapArtifactMetadata` with `hydrology_digest` (hash of hydrology field + rainfall seeds) and store rain events in the persistence log for deterministic replays.
  - Add DuckDB tables `hydrology_basins`, `rain_events`, `flood_snapshots` for analytics.

- **Testing & Validation**
  - Unit tests for hydrology solver (e.g., synthetic bowl map with known outlet) verifying overflow order.
  - Property tests ensuring water never becomes negative, total volume conserved during routing (minus evaporation).
  - Golden scenes: fixed tileset + scripted storm produces identical `water_depth` snapshots on native + wasm.

- **Future Enhancements**
  - Noise-driven microstreams feeding major rivers for visual richness.
  - Coupling with agent evolution: track swimming trait prevalence vs. flood frequency.
  - Multiplayer experiments: expose hydrology knobs via MCP so external agents can orchestrate storms during evolutionary runs.
- Articles & guides on WFC pattern extraction and constraint propagation.
- Rust crates: `wfc` (crates.io), `image`, `noise`.
## Terminal Rendering Mode (Emoji TUI) [Currently In Progress - Terminal Text Renderer]

### Goal
- Deliver an opt-in, emoji-rich terminal interface that mirrors core simulation insights while preserving GPUI as the primary shell. The terminal renderer must auto-activate when GPUI fails to launch (e.g., headless SSH sessions) and must never regress desktop UX.

### Architectural Tenets
- Introduce a `Renderer` trait inside `scriptbots-app` that abstracts lifecycle hooks (`init`, `run`, `shutdown`) and surface telemetry contracts shared by both GPUI and terminal backends.
- Ensure renderers operate on read-only world views via `Arc<Mutex<WorldState>>` and shared control channels so simulation determinism is untouched.
- Keep dependencies isolated: GPUI stays in `scriptbots-render`; the terminal implementation lives in a new module/crate (`scriptbots-terminal`) compiled behind a `terminal` feature flag.
- Runtime selection is driven by CLI flag (`--mode {auto|gui|terminal}`) with environment override (`SCRIPTBOTS_MODE`); `auto` attempts GPUI first, falling back to terminal on failure or lack of DISPLAY/Wayland environment.

### Dependency & Capability Survey
- Adopt `ratatui` (successor to `tui-rs`) for styled widgets, Unicode/emoji support, and terminal layout primitives.
- Use `crossterm` for cross-platform terminal control (alternate screen, input, color) and `supports-color` to detect 24-bit vs. limited palettes.
- Add `unicode-width` or `ratatui`'s width utilities to keep emoji-aligned HUD panels.
- Explore `ratatui-animations` for smooth gauge updates and `ratatui-extras` for mini-charts and tabular dashboards.

### Implementation Plan
1. **Trait Abstraction & CLI Wiring** [Completed - CLI/Renderer refactor 2025-10-22]
   - Define `Renderer` trait in `scriptbots-app` with `fn run(&self, world: SharedWorld, controls: &ControlRuntime) -> Result<()>` and optional teardown hooks.
   - Refactor existing GPUI entry to implement the trait without altering rendering internals.
   - Extend CLI parsing (using `clap` or current argument parser) to accept `--mode` and environment overrides; update `ControlServerConfig` to propagate mode choice.
2. **Terminal Renderer Scaffolding**
   - Create `scriptbots-terminal` module/crate with `ratatui`, `crossterm`, `supports-color`, and `unicode-width` dependencies; gate behind `terminal` feature in workspace.
   - Implement initialization (alternate screen, input polling) and a rendering loop targeting ~20 FPS with frame timing derived from the simulation tick summaries.
3. **HUD & Visualization Design**
   - Build panels for global stats (tick, population, births/deaths, energy averages) using color-coded gauges and sparkline charts.
   - Render a coarse world mini-map using emoji glyphs for agents/food/terrain; leverage palette downgrades when terminals lack truecolor.
   - Surface selection/inspector data via tabbed panes, mirroring GPUI inspector fields where practical.
4. **Event & Input Handling**
   - Map keyboard shortcuts (pause, speed multiplier, selection cycling) to TUI equivalents; ensure commands feed into existing `ControlRuntime` APIs.
   - Provide in-terminal help overlay with emoji legends and keybindings.
5. **Fallback & Detection Logic**
   - On startup, detect headless environments (missing DISPLAY/Wayland) and log an intent to use terminal mode.
   - Wrap GPUI launch in error handling; on failure, emit structured tracing and re-run using terminal renderer if enabled.
   - Ensure failures cascade gracefully with clear messaging when terminal mode is disabled or unsupported.
6. **Testing & QA**
   - Add CI job running `cargo test --features terminal` plus a smoke test that boots the terminal renderer in headless mode (using `TERM=xterm-256color`).
   - Write integration test harness using `ratatui`'s backend recorder (or golden snapshots) to validate layout rendering for canonical world states.
   - Conduct manual QA across macOS Terminal, iTerm2, Windows Terminal, and Linux SSH to verify emoji palettes and fallback color schemes.
7. **Documentation & Rollout**
   - Update `README.md` with usage instructions, feature flags, and screenshots/gifs of the terminal HUD.
   - Add observability notes (structured logs when mode switches) and highlight that terminal mode is experimental but safe to enable in CI/headless setups.
   - Coordinate with rendering owners to ensure future GPUI changes keep the shared telemetry surface stable.

### Library survey & selections (2025-10)
- Terminal UI framework: adopt `ratatui` (active fork of `tui`) for composable widgets (Block, Paragraph, Table, Chart, Sparkline) and a stable API surface.
- Terminal backend & input: `crossterm` for cross-platform input/colors/alternate screen; headless testing via `ratatui::backend::TestBackend`.
- Color capability detection: `supports-color` to differentiate truecolor/256/basic and downgrade palettes deterministically.
- Unicode handling: `unicode-width` and `unicode-segmentation` to ensure aligned glyphs (including emoji) across terminals.
- Logging in-UI: integrate a simple log viewer panel backed by `tracing` ring buffer; prefer custom widget over `tui-logger` to avoid hard ties to the legacy `tui` crate.
- Charts: start with `ratatui::widgets::{Chart,Sparkline}`; if we need advanced plots, evaluate a `plotters` adapter rendered into a `ratatui` Canvas.
- Optional inline images (behind feature flags, non-default):
  - `viuer` (kitty/iTerm2 graphics protocol) for preview screenshots when supported.
  - `sixel-rs` for DEC SIXEL-capable terminals (xterm, mlterm); both gated behind detection + explicit opt-in.

### Visual design spec
- Layout (80×36 minimum target):
  - Header (3 rows): tick, epoch, agents, births/deaths, avg energy; status flags (RUNNING/PAUSED), speed multiplier.
  - History panel (7 rows): rolling last N ticks with Δbirths/Δdeaths sparkline and average energy trend.
  - World mini-map (remaining rows): emoji-aware heatmap using density-based symbols; palette-aware fallback to ASCII (`*`, `+`, `#`).
  - Help overlay: modal panel with keybindings and palette legend.
- Glyph strategy:
  - Truecolor: use colored emojis (🟢 🟠 🔴) or shaded blocks (▁ ▂ ▃ ▄ ▅ ▆ ▇ █) based on density.
  - 256/basic color: ASCII or shaded blocks with limited palette; ensure contrast ≥ WCAG AA equivalent.
- Color system:
  - Palettes: Natural, Deuteranopia, Protanopia, Tritanopia, HighContrast; apply consistent transforms to header, history, and map symbols.
  - Deterministic palette downgrade based on `supports-color` result; never rely on ad-hoc terminal heuristics.

### Performance targets & scheduling
- Target 60 Hz simulation stepping with an interactive UI draw budget of ~10 Hz (configurable). Cap per-frame simulation steps to maintain responsiveness (already implemented via accumulator and MAX_STEPS_PER_FRAME).
- Avoid allocations in the draw path; reuse buffers for map rows and sparkline points.
- Use `event::poll` with small timeouts; batch redraw when multiple events arrive in a single scheduling quantum.

### Input & accessibility
- Keys (already mapped): pause (`space`), speed ± (`+`/`-`), single-step (`s`), help (`?`/`h`), quit (`q`/`Esc`).
- Planned: cycle palettes (`p`), toggle density mode (emoji/block/ascii), toggle map overlays (biome/food/terrain), selection cycling.
- Headless mode: `SCRIPTBOTS_TERMINAL_HEADLESS=1` renders to `TestBackend` for CI snapshots.

### Data plumbing & control
- The renderer consumes an immutable `Snapshot` built from `WorldState`; do not lock the world longer than necessary.
- Control commands (pause, speed changes, selection) are submitted via the `CommandBus`; the world drains in-tick to preserve determinism.

### Compatibility & fallbacks
- Capability matrix (macOS Terminal, iTerm2, Windows Terminal, Linux SSH):
  - Emoji availability varies; always provide ASCII/block fallbacks.
  - Truecolor vs 256 colors: detect and downgrade palettes; never assume 24-bit.
  - Inline image protocols (kitty/iTerm2/SIXEL) are opt-in only and not required for MVP.

### Testing & QA (expanded)
- Golden snapshot tests: compare textual mini-map and header lines against fixtures for a seeded world; allow small diffs when palette modes change.
- Input loop tests: simulate key events and assert state transitions (paused, speed multiplier, help overlay) without requiring a real TTY.
- Performance guard: micro-benchmark drawing functions to assert ≤ 2 ms per draw on CI hardware baseline.

### Deliverables & milestones (revised)
- M2 (Renderer MVP):
  - Header/history/map renderers implemented with palette support and deterministic downgrades.
  - CI smoke test using `SCRIPTBOTS_TERMINAL_HEADLESS=1` producing stable snapshots.
- M3 (Parity & overlays):
  - Overlays for food/biome/terrain density; selection cycling; palette cycling; density mode switch.
  - Basic log viewer panel fed by `tracing` ring buffer.
- M4 (Polish & optional graphics):
  - Advanced charts (Chart widget or plotters-backed canvas), configurable themes, optional inline images (kitty/SIXEL) behind feature flags.
  - Documentation with animated asciicasts and keybinding tables.

### Risks & mitigations (terminal-specific)
- Terminal heterogeneity → mitigate with strict capability detection and exhaustive fallbacks.
- Emoji rendering width inconsistencies → rely on `unicode-width`/`unicode-segmentation`; prefer block glyphs when misalignment detected.
- CPU hotspots in large worlds → cap draw frequency; cache map bins; precompute row strings.

### Milestone Targets
- **M1**: Renderer trait + CLI flag merged; GPUI unchanged but conditional path ready.
- **M2**: Terminal renderer MVP displaying stats and responding to pause/resume.
- **M3**: Mini-map, emoji styling, and input parity complete; fallback pathway validated.
- **M4**: Testing matrix, docs, and release notes published; terminal mode enabled in nightly builds/CI.

### Risks & Mitigations
- *Terminal capability variance*: mitigate via `supports-color` checks and palette fallback tables.
- *Input conflicts*: centralize keybinding definitions shared with GPUI to avoid drift.
- *Performance drift*: throttle terminal redraws and reuse diffing buffers to minimize CPU usage during long runs.
- *Feature parity expectations*: document intentionally omitted GPUI features (e.g., advanced camera) and ensure Control Server APIs remain the extension point for deep inspection.

### Execution TODOs [Currently In Progress]
- [Currently In Progress - 2025-10-23] Terminal renderer v2 visual overhaul (denser map glyphs, biome layers, agent inspectors, sparkline telemetry, narrative event log, configurable palettes).
- [Currently In Progress - 2025-10-23] Global parallelism guard to cap rayon thread pool (default ≤4, env override), preventing ScriptBots processes from saturating host CPUs.
- [Completed - 2025-10-22] Dependency alignment: restored `thiserror`, added terminal dependencies (`supports-color`, ratatui already present), and confirmed no redundant `utoipa-axum` entries remain.
- [Completed - 2025-10-22] CLI mode integration: `resolve_renderer` now detects headless environments, honors override env vars, and logs terminal fallback activation.
- [Completed - 2025-10-22] Terminal renderer scaffolding: new `terminal` module implements the shared `Renderer` trait with the crossterm/ratatui event loop.
- [Completed - 2025-10-22] HUD implementation pass 1: terminal view renders status header, rolling history, and an emoji mini-map with palette fallbacks.
- [Completed - 2025-10-22] Input handling parity: pause/resume, speed scaling, single-step, help overlay, and quit shortcuts are wired into the control runtime.
- [Completed - 2025-10-22] Automated testing: headless smoke test runs the `scriptbots-app` binary in terminal mode (`SCRIPTBOTS_TERMINAL_HEADLESS=1`) to guard against regressions.
- [Completed - 2025-10-22] Documentation updates: README now covers terminal mode flags, fallback behavior, and headless usage tips.

#### [Completed - 2025-10-24] Emoji mode v1.1 (default-on + curated glyphs)
- Purpose: make the terminal renderer visually appealing by default while remaining robust on headless/limited terminals.
- Behavior:
  - Emoji mode defaults to ON when stdout is a modern terminal and the locale is UTF‑8; auto-disabled on obviously minimal terminals or CI.
  - Hotkey: `e` toggles emoji mode at runtime and logs an event ("Emoji mode ON/OFF").
  - Env override: `SCRIPTBOTS_TERMINAL_EMOJI=1|true|yes|on` forces ON; `0|false|off|no` forces OFF.
- Detection heuristic:
  - `TERM` not in {"", "dumb", "linux", "vt100"}
  - locale from `LC_ALL`/`LC_CTYPE`/`LANG` contains `utf-8` or `utf8`
  - `CI` unset
- Emoji mappings:
  - Terrain: DeepWater=🌊, ShallowWater=💧 (lush → 🐟), Sand=🏜 (lush → 🌴), Grass=🌿 / Bloom=🌺 (lush → 🌾, barren → 🥀), Rock=🪨.
  - Agents: single Herbivore=🐇, Omnivore=🦝, Carnivore=🦊; small groups 🐑/🐻/🐺; large cluster 👥; boosted 🚀; spike peak ⚔ with underline; heading arrows preserved when available.
- Files: `crates/scriptbots-app/src/terminal/mod.rs` (`Palette::detect/terrain_symbol/agent_symbol`, key handler, help overlay).
- Risks & mitigations: width variance across terminals (press `e` to revert); README documents installing emoji-capable fonts if glyphs appear as tofu.
- Narrow mode: key `n` toggles a width-1 symbol set (`emoji_narrow`) that substitutes compact glyphs where emojis may misalign; emoji backgrounds are suppressed for visual clarity.

#### [Completed - 2025-10-24] Headless reporting & CI knobs
- `SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES` limits frames during headless terminal runs (default 12; max 360).
- `SCRIPTBOTS_TERMINAL_HEADLESS_REPORT` writes a JSON report (ticks, births/deaths, energy stats) consumed by CI.

### [Completed - 2025-10-24] Brain families default-on and mixed-species evolution
- Enabled all brain families by default: MLP (baseline), DWRAON, Assembly (experimental), NeuroFlow (optional crate), and register them at app startup. Mixed populations are now the default.
- Random new spawns receive a brain binding sampled from the BrainRegistry.
- Sexual reproduction enforces a species barrier: crossover only occurs when both parents share the same brain kind; otherwise, the spawn falls back to random seeding. This makes differing brain kinds act as distinct species while fairly competing in the same environment.
- NeuroFlow is enabled in default `ScriptBotsConfig` and still configurable at runtime (layers/activation).
- Control/CLI/REST remain unchanged; analytics/inspector now implicitly reflect multiple brain kinds (brain_kind, brain_key already surfaced).

Implementation notes:
- `BrainRegistry::random_key(&mut rng) -> Option<u64>` added for unbiased brain sampling.
- `WorldState::spawn_random_agent` binds a sampled brain key to the newly spawned agent.
- `WorldState::spawn_crossover_agent` now checks parent brain kinds via registry keys; when mismatched, it returns `false` so the scheduled spawner falls back to a random newcomer.
- Default config toggles `neuroflow.enabled = true`.
- App features default to including optional brain crates so binaries ship with all families available by default.

Benchmarks/Determinism:
- The selection of brain family for random spawns is driven by the world RNG, so runs remain seed-stable.
- Crossover gating (species barrier) removes undefined mixing across brain kinds and reduces variance in hybrid viability analysis.
