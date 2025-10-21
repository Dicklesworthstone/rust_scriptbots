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
  - `brain: Box<dyn Brain + Send + Sync>` or enum-based dispatch for zero-cost dynamic dispatch. [Pending: replace temporary `BrainBinding` once registry is wired]
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
5. **Brains Tick**: Call `brain.tick(&inputs, &mut outputs)` per agent. Provide `Brain` trait with pure functions to stay thread-safe; if mutation required, use interior `RefCell` replaced by split-phase updates.
6. **Actuation**: Translate outputs into movement, colors, giving, boosting. Use double-buffered positional data to avoid read/write races: compute new positions into `Vec<AgentDelta>`, commit after iteration.
7. **Food Intake & Sharing**: Process using `par_iter_mut` with atomic adds or gather stage followed by sequential commit to avoid floating-point race conditions (determinism).
8. **Combat and Death**: Evaluate spike collisions using spatial index, queue health changes, then apply.
9. **Reproduction & Spawning**: Collect reproduction events into `Vec<SpawnOrder>`, apply sequentially to maintain deterministic ordering, leveraging `rand_distr` for gaussian mutations.
10. **Persistence Hooks**: Stream agent snapshots, food deltas, and event logs into DuckDB tables using batched transactions (e.g., every N ticks or when buffers exceed threshold). Leverage Arrow/Parquet features for zero-copy exports when advanced analytics are enabled.citeturn1search0turn1search5
11. **Cleanup**: Remove dead agents using `Vec::retain` (single-threaded) or stable partition maintaining deterministic ordering; recycle IDs as needed.

## Brain System
- Define `pub trait Brain: Send + Sync { fn kind(&self) -> BrainKind; fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE]; fn mutate(&mut self, rng: &mut impl Rng, mut_rate: f32, mut_scale: f32); fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Box<dyn Brain>; }`.
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
- Entry point: `Application::new().run(|cx: &mut App| { ... })`, open window with `cx.open_window(...)`, register root view `WorldView`.citeturn0search0
- State ownership:
  - `SimulationEntity`: `Entity<SimulationModel>` holds shared simulation state (agents, food grid snapshot, metrics). Updates triggered via background tasks that mutate entity and call `cx.notify()`.
  - UI-specific entity for camera (zoom, pan) mirroring original GLView controls.
- Drawing:
  - Use `canvas()` element to draw food tiles and agents each frame. Batch rectangles for food using `PathBuilder` for contiguous quads; draw agents as circles/spikes with `PathBuilder::stroke`.
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
  - On snapshot publication, enqueue summary records (population counts, resource totals) for DuckDB ingestion, and optionally archive raw agent rows every configurable cadence for replay/debug. Employ DuckDB's Arrow integration for efficient bulk writes when analytics pipelines demand columnar exports.citeturn1search0turn1search5

## Visual Polish & Audio Enhancements
- Rich terrain & backgrounds: integrate a tilemap or overlay renderer such as `wgpu-tilemap` to stream large ground textures efficiently, enabling biomes, paths, and heatmaps without dropping frame rate.citeturn0search0
- Stylized UI elements: use Vello inside GPUI canvases for high-quality vector icons, rounded panels, and animated HUD elements, giving the interface a modern “game” look while staying GPU-accelerated.citeturn0search4
- Particle and lighting effects: reference the `ShadowEngine2D` example for wgpu-driven particles, bloom, and lighting passes, adapting similar shaders to draw energy trails, spike impacts, or weather overlays.citeturn0reddit15
- Post-processing shaders: batch agent render buffers through compute builders like `cuneus` to add color grading, vignette, or heat shimmer effects without rewriting the entire pipeline.citeturn0reddit18
- Game-quality audio: adopt the `kira` crate for layered ambient loops, positional effects, and event-driven sound design, using its mixer and timing system to sync audio cues with agent behaviors; expose toggles for audio channels in the GPUI inspector.citeturn2search1turn2search2
- Accessibility polish: add colorblind-safe palettes and toggleable outline shaders so the simulation remains readable with visual enhancements.

## Input Mapping and Feature Parity
- Keyboard:
  - `P` pause, `D` toggle drawing, `F` show food, `+/-` adjust speed, `A` add crossover agents, `Q/H` spawn carnivore/herbivore, `C` toggle closed environment, `S/O` follow selected/oldest.
  - Implement via GPUI action bindings to mutate simulation config entity.
- Mouse:
  - Right-drag pan, middle-drag zoom, left-click select agent; hold `Shift` to inspect stats overlay.
- Agent Inspector:
  - Modal or side panel displaying sensor/brain heatmaps, trait values, mutation history with scrollable `div()`.

## Testing, Tooling, and CI
- Unit tests in `scriptbots-core` covering reproduction math, spike damage, food sharing formulas.
- Property tests (QuickCheck/Proptest) verifying invariants (health clamped [0, 2], reproduction resets counter).
- Determinism tests running simulation for fixed ticks with seeded RNG and asserting stable snapshots.
- Bench harness (`criterion`) measuring ticks/sec at various agent counts.
- GPUI view tests using `#[gpui::test]` macro to ensure layout compiles and actions dispatch.citeturn0search0
- Storage tests:
  - Integration suite writing simulated batches into DuckDB in-memory databases, asserting schema evolution, transaction durability, and query latency.citeturn1search0
  - Snapshot-based golden tests verifying historical queries (population trends, kill counts) match expected outputs when replayed from DuckDB logs.citeturn1search0turn1search5
- Continuous integration: GitHub Actions with matrix (macOS 14, Ubuntu 24.04), caching `cargo` artifacts, running tests + release build.

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
  - Expose CLI/GUI toggles allowing per-run selection of active brain families and evolution knobs (mutation rates, speciation thresholds).
- Diagnostics:
  - Persist brain metrics (loss curves, weight norms, training tick) alongside genomes in DuckDB for analytics and UI visualization.
  - Provide debug tooling to render network topologies (layer shapes, activations) inside the inspector panel.

## Modularity & Extensibility Enhancements
- Optional ECS: evaluate `hecs`/`legion` for certain subsystems (sensing, effects) after benchmarking; keep default architecture straightforward.
- Scripting sandbox: offer a Wasm or Lua plug-in layer for experimental sensors/reward shaping, gated by capability lists to preserve determinism.
- Deterministic replay & branching: treat every RNG draw and brain choice as an event stored in DuckDB, enabling branch-and-replay workflows and regression reproduction.
- Scenario layering: allow configs to be composed (base + biome + experiment) controlling constants, active sensors, brain registries, and UI themes.

## Migration Roadmap
1. **Project Bootstrap (Week 1)**
   - Initialize workspace via `create-gpui-app --workspace`.
   - Set up linting (`rustfmt`, `clippy`), choose MSRV (Rust 1.81+) to match GPUI requirement.
2. **Core Data Structures (Weeks 1-2)**
   - Port `settings.h` constants into `ScriptBotsConfig`.
   - Implement `Vector2` replacement via `glam::Vec2`.
   - Port agent struct, reproduction, mutation logic with unit tests.
3. **World Mechanics (Weeks 2-4)**
   - Implement food grid, sensing pipeline (sequential first), reproduction queue, death cleanup.
   - Ensure parity with original via scenario tests (e.g., spike kill distribution).
4. **Introduce Concurrency (Weeks 4-5)**
   - Integrate Rayon, add spatial partition acceleration, verify determinism under multi-thread.
5. **Brain Ports (Weeks 5-7)**
   - MLP (baseline) complete with mutate/crossover.
   - DWRAON feature gate; assembly brain behind `--features experimental`.
   - Implement NeuroFlow-backed brain module and wire through the brain registry for opt-in builds.citeturn2search1turn2search2
6. **Persistence Layer (Weeks 7-8)**
   - Stand up `scriptbots-storage`, define DuckDB schema (agents, ticks, events, metrics).
   - Implement buffered writers, compaction routines, and analytics helpers (e.g., top predators query).
7. **Rendering Layer (Weeks 8-10)**
   - Build GPUI window, canvas renderer, agent inspector UI.
   - Implement camera controls, overlays, history chart.
   - Prototype tile-based terrain, vector HUD, and post-processing shader pipeline for polished visuals.
8. **Integration & UX Polish (Weeks 10-11)**
   - Hook actions to simulation, selection workflows, debug overlays.
   - Add metrics HUD, performance counters.
   - Surface brain controls (selection, evolution rates) and storage toggles in the inspector.
   - Layer in audio cues with `kira`, tie particle/lighting effects to simulation events, and add accessibility options.
9. **Testing, Benchmarks, Packaging (Weeks 11-12)**
   - Determinism/regression suite, `cargo bench`.
   - Release pipeline (`cargo dist` or `cargo bundle`), signed macOS binaries.

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
