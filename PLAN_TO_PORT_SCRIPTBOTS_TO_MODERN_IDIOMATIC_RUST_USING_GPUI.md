# Plan to Port ScriptBots to Modern Idiomatic Rust with GPUI

## Vision and Success Criteria
- Deliver a faithful, deterministic port of ScriptBots’ agent-based ecosystem in Rust, preserving simulation behaviors (sensing, decision-making, reproduction, food dynamics, carnivore/herbivore specialization) while removing undefined behavior and manual memory management found in the original GLUT/C++ codebase (`World.cpp`, `Agent.cpp`).
- Embrace Rust idioms: error handling via `Result`, trait-based abstraction for brains, `Arc`/`RwLock` only where unavoidable, and zero `unsafe` in the first release.
- Exploit data-parallelism with Rayon to scale agent updates on modern multi-core CPUs.
- Replace legacy GLUT rendering with a GPU-accelerated GPUI front-end, using declarative views plus imperative canvas drawing to render thousands of agents at 60+ FPS on macOS and Linux, the platforms GPUI currently targets.citeturn0search0
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
  - `crates/scriptbots-storage`: Abstraction over DuckDB for persistence, journaling, and analytics exports built atop the `duckdb` crate.citeturn1search0
  - `crates/scriptbots-brain-neuro`: Optional crate that wraps NeuroFlow feed-forward networks for more advanced brains, kept isolated behind a Cargo feature for lean builds.citeturn2search1
  - `crates/scriptbots-render`: GPUI views, canvas drawing, input handling, overlays.
  - `crates/scriptbots-app`: Binary crate wiring simulation loop, GPUI application, CLI/config loading.
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
10. **Persistence Hooks** [Completed: tick summary + pluggable sink]: Stream agent snapshots, food deltas, and event logs into DuckDB tables using batched transactions (e.g., every N ticks or when buffers exceed threshold). Leverage Arrow/Parquet features for zero-copy exports when advanced analytics are enabled.citeturn1search0turn1search5
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
  - Maintain a dedicated async task that receives structured log batches over an MPSC channel, writes them via a pooled DuckDB `Connection`, and periodically checkpoints/optimizes storage. Use `duckdb`'s bundled feature in development to avoid external dependencies, and adopt `Connection::open_with_flags` variants for file-backed persistence in production.citeturn1search0
- Accelerator hooks:
  - Keep sensor aggregation and brain inference behind pluggable traits so alternative implementations (SIMD-optimized, GPU-backed) can be introduced without touching higher-level logic. Prototype GPU execution by batching inference into wgpu compute shaders or other accelerators once profiling justifies the investment.

## Rendering with GPUI
- Entry point: `Application::new().run(|cx: &mut App| { ... })`, open window with `cx.open_window(...)`, register root view `WorldView`.citeturn0search0 [Completed - GPT-5 Codex 2025-10-21: window shell + metrics HUD scaffolding]
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
  - GPUI documentation highlights macOS and Linux as current targets; prioritize those platforms and monitor upstream progress for additional backends.citeturn0search0
- Tooling:
  - Use `create-gpui-app` CLI to scaffold initial GPUI project structure, then integrate workspace crates manually.citeturn0search1
- Ecosystem:
  - Track community crates like `gpui-component` for prebuilt widgets and styling patterns.citeturn0search8

## Simulation Loop and UI Synchronization
- Dedicated simulation thread:
  - Leverage GPUI’s async executor to run the simulation tick loop off the main thread while keeping the UI responsive (target ~16 ms timestep).citeturn0search0
  - Use `async` channel (`tokio::sync::watch` or `flume`) to push immutable snapshots to UI layer.
- Snapshot Strategy:
  - Maintain double buffer (`SimulationSnapshot` with `Arc<[AgentRenderData]>` + `Arc<[FoodCell]>`). UI clones `Arc`s cheaply, ensuring zero-copy render pipeline.
  - Provide interpolation for camera smoothing if we add variable render rate.
- Storage Sync:
  - On snapshot publication, enqueue summary records (population counts, resource totals) for DuckDB ingestion, and optionally archive raw agent rows every configurable cadence for replay/debug. Employ DuckDB's Arrow integration for efficient bulk writes when analytics pipelines demand columnar exports. [Completed - GPT-5 Codex 2025-10-21]citeturn1search0turn1search5

## Visual Polish & Audio Enhancements
- Rich terrain & backgrounds: integrate a tilemap or overlay renderer such as `wgpu-tilemap` to stream large ground textures efficiently, enabling biomes, paths, and heatmaps without dropping frame rate.citeturn0search0
- [Completed - GPT-5 Codex 2025-10-22: Terrain elevation now drives downhill momentum/energy costs, with HUD-integrated shading reflecting slopes.]
- Stylized UI elements: use Vello inside GPUI canvases for high-quality vector icons, rounded panels, and animated HUD elements, giving the interface a modern “game” look while staying GPU-accelerated. [Completed - GPT-5 Codex 2025-10-22: added animated header badge, gradient-styled metric cards, and overlay summaries rendered via GPUI/Vello canvases for consistent theming.]citeturn0search4
- Particle and lighting effects: reference the `ShadowEngine2D` example for wgpu-driven particles, bloom, and lighting passes, adapting similar shaders to draw energy trails, spike impacts, or weather overlays.citeturn0reddit15
- Post-processing shaders: batch agent render buffers through compute builders like `cuneus` to add color grading, vignette, or heat shimmer effects without rewriting the entire pipeline.citeturn0reddit18
- Game-quality audio: adopt the `kira` crate for layered ambient loops, positional effects, and event-driven sound design, using its mixer and timing system to sync audio cues with agent behaviors; expose toggles for audio channels in the GPUI inspector.citeturn2search1turn2search2
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
- GPUI view tests using `#[gpui::test]` macro to ensure layout compiles and actions dispatch.citeturn0search0
- Storage tests:
  - Integration suite writing simulated batches into DuckDB in-memory databases, asserting schema evolution, transaction durability, and query latency. [Completed - GPT-5 Codex 2025-10-21]citeturn1search0
  - Snapshot-based golden tests verifying historical queries (population trends, kill counts) match expected outputs when replayed from DuckDB logs. [Completed - GPT-5 Codex 2025-10-22]citeturn1search0turn1search5
- Continuous integration: GitHub Actions with matrix (macOS 14, Ubuntu 24.04), caching `cargo` artifacts, running tests + release build. [Completed - GPT-5 Codex 2025-10-22]

## Advanced Brain Architecture Strategy
- Trait hierarchy: extend the `Brain` interface with marker traits (`EvolvableBrain`, `TrainableBrain`, `BatchBrain`) so the world loop can branch on capabilities (e.g., some brains may not support online learning).
- NeuroFlow integration:
  - Implement `NeuroflowBrain` around `neuroflow::FeedForward`, leveraging builder helpers for declarative layer construction and `io::to_file`/`from_file` for serialization.citeturn2search1turn2search2
  - Support asynchronous fine-tuning by accumulating recent experience tuples and training in background tasks, swapping weights atomically on completion.
- Genome-centric evolution:
  - `BrainGenome` persists topology, activation mix, optimizer hyperparameters, and provenance metadata (parent IDs, mutation notes). Sexual reproduction performs uniform crossover on layer specs and blends hyperparameters; asexual reproduction perturbs and, occasionally, inserts/removes layers.
  - Maintain compatibility metrics and speciation buckets (NEAT-inspired) to avoid catastrophic mixing of divergent architectures.
- Pluggable factories:
  - Register brain constructors through a `BrainRegistry` so additional engines (ONNXRuntime, Torch via tch-rs, GPU kernels) can be added behind feature flags without core refactors.
- [Completed - GPT-5 Codex 2025-10-22] Expose runtime control surfaces for selecting brain families and tuning evolution knobs (mutation rates, speciation thresholds, environmental parameters) without rebuilds.
  - Introduced `ControlHandle`/`KnobEntry` scaffolding shared by REST, CLI, and MCP front-ends for atomic JSON patching of `ScriptBotsConfig` and live snapshots.
  - Shipped an axum-based REST API (`/api/knobs`, `/api/config`, `/api/knobs/apply`) with generated OpenAPI docs and embedded Swagger UI for operator discovery.
  - Added a stdio MCP server (`list_knobs`, `get_config`, `apply_updates`, `apply_patch`) so external LLM tooling can orchestrate simulations safely.
  - Delivered the `scriptbots-control` CLI (clap + reqwest + ratatui) offering scripted updates (`set`, `patch`) and a live dashboard (`watch`) that highlights config deltas while the GPUI shell runs.
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
  - [ ] Port herbivore vs carnivore behaviors: enforce attack restrictions for herbivores, replicate reproduction timers (`REPRATEH/REPRATEC`), and apply diet-based modifiers in food intake and carcass sharing.
  - [ ] Match food consumption math: reuse `FOODINTAKE`, `FOODWASTE`, and speed-dependent gains tied to wheel speeds and herbivore tendency, updating tests to cover stationary vs. fast agents.
  - [ ] Restore modcounter cadence: introduce configurable tick scheduler for aging every 100 ticks, periodic charts/reporting, reproduction gating randomness, and guard against regressions with deterministic seeds.
  - [ ] Carry mutation parameter genetics: track per-agent `mutrate1/2`, `temperature_preference`, and trait modifiers, mutating them during reproduction following the C++ meta-mutation rules.
  - [ ] Build regression tests comparing Rust tick traces against captured C++ snapshots.
- **Brain Implementations**
  - [ ] Port DWRAON brain with parity tests and feature gating.
  - [ ] Port Assembly brain (even if experimental) with determinism guardrails.
  - [ ] Implement mutation/crossover suites validated against C++ reference data.
  - [ ] Develop brain registry benchmarking (per-brain tick cost, cache hit rates).
- **Analytics & Replay**
  - [ ] Extend persistence schema to store replay events (per-agent RNG draws, brain outputs, actions).
  - [ ] Implement deterministic replay runner (headless) driven by stored events.
  - [ ] Add DuckDB parity queries (population charts, kill ratios, energy histograms) vs. C++ scripts.
  - [ ] Provide CLI tooling to diff runs (Rust vs. C++ baseline) and highlight divergences.
- **Feature Toggles & UX Integration (Non-rendering)**
  - [ ] Surfacing runtime toggles: CLI/ENV for enabling brains, selecting indices, adjusting mechanics.
  - [ ] Selection and debug hooks: expose APIs to query agent state, highlight subsets (without GPUI coupling).
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
   - Implement NeuroFlow-backed brain module and wire through the brain registry for opt-in builds. [Completed - GPT-5 Codex 2025-10-22]citeturn2search1turn2search2
   - Seed NeuroFlow weights using the world RNG for deterministic runs. [Completed - GPT-5 Codex 2025-10-21]
   - Add runtime configuration toggle to enable NeuroFlow brains without compile-time features. [Completed - GPT-5 Codex 2025-10-21]
6. **Persistence Layer (Weeks 7-8)**
   - Stand up `scriptbots-storage`, define DuckDB schema (agents, ticks, events, metrics). [Completed - GPT-5 Codex 2025-10-22]
   - Implement buffered writers, compaction routines, and analytics helpers (e.g., top predators query). [Completed - GPT-5 Codex 2025-10-22]
7. **Rendering Layer (Weeks 8-10)** [Completed - GPT-5 Codex 2025-10-22: GPUI stats overlay and controls polished]
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
- **GPU backend availability**: GPUI is still evolving; focus initial support on macOS/Linux per official guidance, while monitoring upstream platform work.citeturn0search0
- **Determinism under parallelism**: Floating-point reductions can diverge; solve via deterministic orderings and staged accumulations.
- **Performance regressions**: Large agent counts may stress GPUI canvas; prototype rendering with 10k agents early, profile using Tracy/`wgpu-profiler`.
- **Brain extensibility**: Trait object overhead; consider `enum BrainImpl` with static dispatch once stable.
- **DuckDB durability/throughput**: Excessive per-tick writes could bottleneck; mitigate with batched transactions, asynchronous writers, and optional toggles for high-frequency logging. Evaluate Parquet exports for heavy analytics workloads.citeturn1search0turn1search5
- **NeuroFlow maturity**: Crate targets CPU feed-forward networks; keep abstractions loose so alternative engines can slot in if requirements outgrow NeuroFlow.citeturn2search1
- **Team familiarity with GPUI**: Documentation is evolving; allocate ramp-up time exploring official docs, tutorials, and community component libraries such as `gpui-component`.citeturn0search8

## Open Questions
- Should we adopt serde-based save/load for simulation snapshots at launch?
- Do we need remote observer mode (headless simulation + streaming state to GPUI frontend)?
- What telemetry is acceptable for release builds (opt-in metrics)?
- Long term: evaluate advanced vision/AI brain integration once core port is stable.

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
- [ ] Dependency alignment: ensure `thiserror` is restored, drop redundant `utoipa-axum`, and add terminal-mode crates (`supports-color`, `unicode-width`) gated appropriately.
- [ ] CLI mode integration: update `resolve_renderer` to prefer terminal mode when headless, and surface structured logs when fallback happens.
- [ ] Terminal renderer scaffolding: add `terminal` module implementing the `Renderer` trait with screen setup, event loop, and world stepping.
- [ ] HUD implementation pass 1: render tick/epoch, population metrics, recent summaries, and a coarse emoji world map with palette fallbacks.
- [ ] Input handling parity: wire pause/resume, speed adjustments, help overlay, and graceful quit to existing control runtime APIs.
- [ ] Automated testing: add headless smoke test (CI) invoking terminal renderer under `TERM=xterm-256color` to guard regressions.
- [ ] Documentation updates: README usage section, environment variables, and example captures; note feature flag expectations.
